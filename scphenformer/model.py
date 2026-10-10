from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(
    scores: torch.Tensor, groups: torch.Tensor, number_of_groups: int
) -> torch.Tensor:
    """Softmax independently over sparse edges sharing the same group."""
    maxima = torch.full(
        (number_of_groups,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    values = torch.exp(scores - maxima[groups])
    totals = torch.zeros_like(maxima)
    totals.scatter_add_(0, groups, values)
    return values / totals[groups].clamp_min(1e-12)


class ScPhenFormerBackbone(nn.Module):
    """Gene-aware pathway Transformer used for pretraining and fine-tuning."""

    def __init__(
        self,
        vocab_size: int,
        membership_indptr: torch.Tensor,
        membership_indices: torch.Tensor,
        number_of_pathways: int = 384,
        number_of_complement_tokens: int = 32,
        dimension: int = 256,
        depth: int = 12,
        number_of_heads: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        if not 0 < mask_ratio <= 1 or not 0 <= dropout < 1:
            raise ValueError("Require 0 < mask_ratio <= 1 and 0 <= dropout < 1")
        if number_of_complement_tokens < 1 or number_of_pathways < 1:
            raise ValueError("Pathway and complement token counts must be positive")
        self.vocab_size = int(vocab_size)
        self.number_of_pathways = int(number_of_pathways)
        self.number_of_complement_tokens = int(number_of_complement_tokens)
        self.dimension = int(dimension)
        self.mask_ratio = float(mask_ratio)
        self.register_buffer("membership_indptr", membership_indptr.long())
        self.register_buffer("membership_indices", membership_indices.long())

        gene_ids = torch.arange(vocab_size, dtype=torch.long)
        complement = torch.remainder(
            gene_ids * 2654435761 + 1013904223, number_of_complement_tokens
        )
        complement[0] = 0
        self.register_buffer("complement_by_gene", complement)

        self.gene_embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.expression_encoder = nn.Sequential(
            nn.Linear(1, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )
        self.expression_gate = nn.Sequential(nn.Linear(1, dimension), nn.Sigmoid())
        self.gene_norm = nn.LayerNorm(dimension)
        self.pool_key = nn.Linear(dimension, dimension, bias=False)
        self.pool_value = nn.Linear(dimension, dimension, bias=False)
        self.pathway_query = nn.Parameter(torch.empty(number_of_pathways, dimension))
        self.pathway_identity = nn.Parameter(torch.empty(number_of_pathways, dimension))
        self.complement_query = nn.Parameter(
            torch.empty(number_of_complement_tokens, dimension)
        )
        self.complement_identity = nn.Parameter(
            torch.empty(number_of_complement_tokens, dimension)
        )
        self.inactive_embedding = nn.Embedding(2, dimension)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dimension))

        block = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=number_of_heads,
            dim_feedforward=dimension * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, num_layers=depth, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(dimension)

        self.decoder_query = nn.Linear(dimension, dimension, bias=False)
        self.decoder_key = nn.Linear(dimension, dimension, bias=False)
        self.decoder_value = nn.Linear(dimension, dimension, bias=False)
        self.decoder_context_projection = nn.Linear(dimension, dimension)
        self.decoder_gate = nn.Linear(dimension, dimension)
        self.decoder_fusion_norm = nn.LayerNorm(dimension)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        with torch.no_grad():
            self.gene_embedding.weight[0].zero_()
        for parameter in (
            self.pathway_query,
            self.pathway_identity,
            self.complement_query,
            self.complement_identity,
            self.cls_token,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            with torch.no_grad():
                block.self_attn.out_proj.weight.mul_((2.0 * len(self.transformer.layers)) ** -0.5)

    def _expand_memberships(
        self, gene_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.membership_indptr[gene_ids]
        degrees = self.membership_indptr[gene_ids + 1] - starts
        edge_count = int(degrees.sum().item())
        if edge_count == 0:
            empty = gene_ids.new_empty(0)
            return empty, empty
        sources = torch.repeat_interleave(
            torch.arange(gene_ids.numel(), device=gene_ids.device), degrees
        )
        group_starts = torch.repeat_interleave(
            torch.cumsum(degrees, 0) - degrees, degrees
        )
        offsets = torch.arange(edge_count, device=gene_ids.device) - group_starts
        pathways = self.membership_indices[torch.repeat_interleave(starts, degrees) + offsets]
        return sources, pathways

    def _pool(
        self,
        gene_hidden: torch.Tensor,
        sources: torch.Tensor,
        cell_ids: torch.Tensor,
        token_ids: torch.Tensor,
        queries: torch.Tensor,
        number_of_cells: int,
        number_of_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = gene_hidden.new_zeros(
            number_of_cells * number_of_tokens, self.dimension
        )
        active = torch.zeros(
            number_of_cells * number_of_tokens,
            dtype=torch.bool,
            device=gene_hidden.device,
        )
        if sources.numel() == 0:
            return (
                pooled.view(number_of_cells, number_of_tokens, -1),
                active.view(number_of_cells, number_of_tokens),
            )
        groups = cell_ids[sources] * number_of_tokens + token_ids
        keys = F.normalize(self.pool_key(gene_hidden[sources]).float(), dim=-1)
        query = F.normalize(queries[token_ids].float(), dim=-1)
        weights = segment_softmax((keys * query).sum(-1), groups, pooled.shape[0])
        pooled.index_add_(
            0, groups, self.pool_value(gene_hidden[sources]).float() * weights[:, None]
        )
        active[torch.unique(groups)] = True
        return (
            pooled.view(number_of_cells, number_of_tokens, -1),
            active.view(number_of_cells, number_of_tokens),
        )

    def encode(
        self,
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        valid_genes: torch.Tensor,
        masked_genes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode padded non-zero genes from a batch of cells."""
        if masked_genes is None:
            masked_genes = torch.zeros_like(valid_genes)
        visible = valid_genes & (expression > 0) & (gene_ids > 0) & ~masked_genes
        number_of_cells = gene_ids.shape[0]
        grid = torch.arange(number_of_cells, device=gene_ids.device)[:, None]
        cell_ids = grid.expand_as(gene_ids)[visible]
        ids = gene_ids[visible]
        values = expression[visible]
        identity = self.gene_embedding(ids)
        value = values.float()[:, None]
        hidden = self.gene_norm(
            identity + self.expression_encoder(value) + identity * self.expression_gate(value)
        )

        degrees = self.membership_indptr[ids + 1] - self.membership_indptr[ids]
        covered_positions = torch.nonzero(degrees > 0, as_tuple=False).flatten()
        sources, pathway_ids = self._expand_memberships(ids[degrees > 0])
        if sources.numel():
            sources = covered_positions[sources]
        pathways, pathway_active = self._pool(
            hidden,
            sources,
            cell_ids,
            pathway_ids,
            self.pathway_query,
            number_of_cells,
            self.number_of_pathways,
        )

        uncovered = torch.nonzero(degrees == 0, as_tuple=False).flatten()
        complement_ids = self.complement_by_gene[ids[uncovered]]
        complements, complement_active = self._pool(
            hidden,
            uncovered,
            cell_ids,
            complement_ids,
            self.complement_query,
            number_of_cells,
            self.number_of_complement_tokens,
        )
        pathways = (
            pathways
            + self.pathway_identity[None]
            + self.inactive_embedding(pathway_active.long())
        )
        complements = (
            complements
            + self.complement_identity[None]
            + self.inactive_embedding(complement_active.long())
        )
        cls = self.cls_token.expand(number_of_cells, -1, -1)
        encoded = self.output_norm(
            self.transformer(torch.cat([cls, pathways, complements], dim=1))
        )
        stop = 1 + self.number_of_pathways
        return {
            "cls": encoded[:, 0],
            "pathways": encoded[:, 1:stop],
            "complements": encoded[:, stop:],
            "pathway_active": pathway_active,
        }

    def sample_mask(self, valid_genes: torch.Tensor) -> torch.Tensor:
        counts = valid_genes.sum(1)
        number_to_mask = (counts.float() * self.mask_ratio).round().long().clamp_min(1)
        scores = torch.rand(valid_genes.shape, device=valid_genes.device)
        scores.masked_fill_(~valid_genes, float("inf"))
        order = scores.argsort(1)
        rank = torch.empty_like(order)
        positions = torch.arange(valid_genes.shape[1], device=valid_genes.device)
        rank.scatter_(1, order, positions[None].expand_as(order))
        return valid_genes & (rank < number_to_mask[:, None])

    def reconstruct(
        self,
        encoded: dict[str, torch.Tensor],
        target_gene_ids: torch.Tensor,
        target_cell_ids: torch.Tensor,
    ) -> torch.Tensor:
        identity = self.gene_embedding(target_gene_ids)
        context = identity.new_zeros(identity.shape)
        starts = self.membership_indptr[target_gene_ids]
        degrees = self.membership_indptr[target_gene_ids + 1] - starts
        covered_positions = torch.nonzero(degrees > 0, as_tuple=False).flatten()
        sources, pathways = self._expand_memberships(target_gene_ids[degrees > 0])
        if sources.numel():
            sources = covered_positions[sources]
            pathway_hidden = encoded["pathways"][target_cell_ids[sources], pathways]
            query = F.normalize(self.decoder_query(identity[sources]).float(), dim=-1)
            key = F.normalize(self.decoder_key(pathway_hidden).float(), dim=-1)
            weights = segment_softmax(
                (query * key).sum(-1), sources, target_gene_ids.numel()
            )
            context.index_add_(
                0, sources, self.decoder_value(pathway_hidden).float() * weights[:, None]
            )
        uncovered = torch.nonzero(degrees == 0, as_tuple=False).flatten()
        if uncovered.numel():
            token_ids = self.complement_by_gene[target_gene_ids[uncovered]]
            context[uncovered] = self.decoder_value(
                encoded["complements"][target_cell_ids[uncovered], token_ids]
            )
        fused = self.decoder_fusion_norm(
            identity + self.decoder_context_projection(context)
            + identity * torch.sigmoid(self.decoder_gate(context))
        )
        return F.softplus(self.decoder(fused).squeeze(-1))

    def pretraining_loss(
        self, gene_ids: torch.Tensor, expression: torch.Tensor, valid_genes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        valid_genes = valid_genes & (expression > 0) & (gene_ids > 0)
        masked = self.sample_mask(valid_genes)
        encoded = self.encode(gene_ids, expression, valid_genes, masked)
        cells = torch.arange(gene_ids.shape[0], device=gene_ids.device)[:, None]
        target_cells = cells.expand_as(gene_ids)[masked]
        prediction = self.reconstruct(encoded, gene_ids[masked], target_cells)
        point_loss = F.smooth_l1_loss(
            prediction.float(), expression[masked].float(), reduction="none"
        )
        cell_loss = point_loss.new_zeros(gene_ids.shape[0])
        cell_loss.scatter_add_(0, target_cells, point_loss)
        loss = (cell_loss / masked.sum(1).clamp_min(1)).mean()
        return {"loss": loss, "prediction": prediction, "masked_genes": masked}

    def forward(
        self, gene_ids: torch.Tensor, expression: torch.Tensor, valid_genes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.pretraining_loss(gene_ids, expression, valid_genes)


class PatientPathwayAggregation(nn.Module):
    """Align the same pathway across cells to form patient pathway instances."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.combine = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proportion = nn.Sequential(
            nn.Linear(1, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, dimension)
        )
        self.identity_projection = nn.Linear(dimension, dimension, bias=False)
        self.output_norm = nn.LayerNorm(dimension)
        self.context = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dimension),
        )

    def forward(
        self,
        pathway_tokens: torch.Tensor,
        pathway_active: torch.Tensor,
        cell_mask: torch.Tensor,
        pathway_identity: torch.Tensor,
        cls_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = pathway_active & cell_mask[:, :, None]
        weights = valid[:, :, :, None].to(pathway_tokens.dtype)
        mean = (pathway_tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        proportions = valid.sum(1).float() / cell_mask.sum(1, keepdim=True).clamp_min(1)
        instances = (
            self.combine(mean)
            + self.proportion(proportions[:, :, None])
            + self.identity_projection(pathway_identity)[None]
        )
        cell_weights = cell_mask[:, :, None].to(cls_tokens.dtype)
        count = cell_weights.sum(1).clamp_min(1)
        cls_mean = (cls_tokens * cell_weights).sum(1) / count
        centered = (cls_tokens - cls_mean[:, None]) * cell_weights
        spread = (centered.float().square().sum(1) / count).clamp_min(1e-8).sqrt()
        context = self.context(torch.cat([cls_mean, spread.to(cls_mean.dtype)], dim=-1))
        return self.output_norm(instances), valid.any(1), context


class PhenotypePathwayMIL(nn.Module):
    """Learn one pathway-attention distribution for each phenotype class."""

    def __init__(
        self, dimension: int, number_of_classes: int, attention_dimension: int, dropout: float
    ) -> None:
        super().__init__()
        self.number_of_classes = number_of_classes
        width = number_of_classes * attention_dimension
        self.norm = nn.LayerNorm(dimension)
        self.v = nn.Linear(dimension, width)
        self.u = nn.Linear(dimension, width)
        self.v_context = nn.Linear(dimension, width, bias=False)
        self.u_context = nn.Linear(dimension, width, bias=False)
        self.query = nn.Parameter(torch.empty(number_of_classes, attention_dimension))
        self.value = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dimension),
        )
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self, instances: torch.Tensor, instance_mask: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, pathways, _ = instances.shape
        x = self.norm(instances)
        shape = batch, pathways, self.number_of_classes, -1
        v = self.v(x).view(shape) + self.v_context(context).view(batch, 1, self.number_of_classes, -1)
        u = self.u(x).view(shape) + self.u_context(context).view(batch, 1, self.number_of_classes, -1)
        gated = torch.tanh(v) * torch.sigmoid(u)
        scores = torch.einsum("bpca,ca->bcp", gated.float(), self.query.float())
        scores.masked_fill_(~instance_mask[:, None], -1e4)
        attention = torch.softmax(scores, dim=-1) * instance_mask[:, None].float()
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        representation = torch.einsum(
            "bcp,bpd->bcd", attention, self.value(x).float()
        )
        expanded_context = context[:, None].expand(-1, self.number_of_classes, -1)
        representation = self.output(torch.cat([representation.to(instances.dtype), expanded_context], dim=-1))
        return representation, attention


class GeneContextBranch(nn.Module):
    """Small complementary branch for transcriptomic signals outside pathways."""

    def __init__(self, number_of_genes: int, dimension: int, dropout: float) -> None:
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.LayerNorm(number_of_genes),
            nn.Linear(number_of_genes, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, expression: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        cells = self.cell_encoder(expression)
        weights = cell_mask[:, :, None].to(cells.dtype)
        return self.output_norm((cells * weights).sum(1) / weights.sum(1).clamp_min(1))


class ScPhenFormer(nn.Module):
    """Patient phenotype model shown in Figure 1."""

    def __init__(
        self,
        backbone: ScPhenFormerBackbone,
        number_of_context_genes: int,
        number_of_classes: int,
        attention_dimension: int = 128,
        dropout: float = 0.1,
        cell_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for name in (
            "decoder_query", "decoder_key", "decoder_value", "decoder_context_projection",
            "decoder_gate", "decoder_fusion_norm", "decoder",
        ):
            delattr(self.backbone, name)
        self.cell_chunk_size = cell_chunk_size
        dimension = backbone.dimension
        self.cell_aggregation = PatientPathwayAggregation(dimension, dropout)
        self.pathway_mil = PhenotypePathwayMIL(
            dimension, number_of_classes, attention_dimension, dropout
        )
        self.gene_branch = GeneContextBranch(
            number_of_context_genes, dimension, dropout
        )
        self.pathway_norm = nn.LayerNorm(dimension)
        self.gene_norm = nn.LayerNorm(dimension)
        self.fusion = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier_weight = nn.Parameter(torch.empty(number_of_classes, dimension))
        self.classifier_bias = nn.Parameter(torch.zeros(number_of_classes))
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        valid_genes: torch.Tensor,
        context_expression: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, cells, padded_genes = gene_ids.shape
        flat_valid_cells = cell_mask.reshape(-1)
        flat_ids = gene_ids.reshape(-1, padded_genes)[flat_valid_cells]
        flat_expression = expression.reshape(-1, padded_genes)[flat_valid_cells]
        flat_valid_genes = valid_genes.reshape(-1, padded_genes)[flat_valid_cells]

        encoded_parts = []
        for start in range(0, flat_ids.shape[0], self.cell_chunk_size):
            stop = start + self.cell_chunk_size
            encoded_parts.append(
                self.backbone.encode(
                    flat_ids[start:stop],
                    flat_expression[start:stop],
                    flat_valid_genes[start:stop],
                )
            )
        pathways = torch.cat([part["pathways"] for part in encoded_parts])
        cls_tokens = torch.cat([part["cls"] for part in encoded_parts])
        active = torch.cat([part["pathway_active"] for part in encoded_parts])
        full_pathways = pathways.new_zeros(
            batch * cells, self.backbone.number_of_pathways, self.backbone.dimension
        )
        full_active = torch.zeros(
            batch * cells,
            self.backbone.number_of_pathways,
            dtype=torch.bool,
            device=gene_ids.device,
        )
        full_pathways[flat_valid_cells] = pathways
        full_active[flat_valid_cells] = active
        full_cls = cls_tokens.new_zeros(batch * cells, self.backbone.dimension)
        full_cls[flat_valid_cells] = cls_tokens
        instances, instance_mask, context = self.cell_aggregation(
            full_pathways.view(batch, cells, *full_pathways.shape[1:]),
            full_active.view(batch, cells, -1),
            cell_mask,
            self.backbone.pathway_identity,
            full_cls.view(batch, cells, -1),
        )
        pathway_representation, pathway_attention = self.pathway_mil(
            instances, instance_mask, context
        )
        gene_representation = self.gene_branch(context_expression, cell_mask)
        gene_by_class = self.gene_norm(gene_representation)[:, None].expand(
            -1, pathway_representation.shape[1], -1
        )
        fused = self.fusion(torch.cat([self.pathway_norm(pathway_representation), gene_by_class], dim=-1))
        logits = torch.einsum("bcd,cd->bc", fused, self.classifier_weight)
        logits = logits + self.classifier_bias
        return {
            "logits": logits,
            "pathway_attention": pathway_attention,
            "pathway_instances": instances,
            "pathway_representation": pathway_representation,
            "gene_representation": gene_representation,
        }


def load_pretrained_backbone(
    backbone: ScPhenFormerBackbone, checkpoint: dict
) -> dict[str, int]:
    source = checkpoint.get("model", checkpoint)
    target = backbone.state_dict()
    exact, expanded = 0, 0
    generated_buffers = {
        "membership_indptr",
        "membership_indices",
        "complement_by_gene",
        "residual_bucket_by_gene",
    }
    aliases = {
        "value_encoder": "expression_encoder", "value_gate": "expression_gate",
        "gene_fusion_norm": "gene_norm", "pathway_pool_query": "pathway_query",
        "residual_pool_query": "complement_query", "residual_identity": "complement_identity",
        "activity_embedding": "inactive_embedding", "encoder_norm": "output_norm",
        "decoder_gene_query": "decoder_query", "decoder_context_key": "decoder_key",
        "decoder_context_value": "decoder_value", "expression_decoder": "decoder",
    }
    loaded = set()
    for name, value in source.items():
        name = name.removeprefix("module.").removeprefix("backbone.")
        prefix, separator, suffix = name.partition(".")
        name = aliases.get(prefix, prefix) + separator + suffix
        if name in generated_buffers or name not in target:
            continue
        if value.shape == target[name].shape:
            target[name] = value
            exact += 1
        elif name == "gene_embedding.weight" and value.ndim == 2:
            rows = min(value.shape[0], target[name].shape[0])
            if value.shape[1] != target[name].shape[1]:
                raise ValueError("Pretrained and downstream embedding dimensions differ")
            target[name][:rows] = value[:rows]
            expanded += 1
        else:
            raise ValueError(
                f"Incompatible pretrained tensor {name}: "
                f"{tuple(value.shape)} versus {tuple(target[name].shape)}"
            )
        loaded.add(name)
    missing = set(target) - generated_buffers - loaded
    if missing:
        raise ValueError(f"Pretrained checkpoint is missing weights: {sorted(missing)}")
    backbone.load_state_dict(target, strict=True)
    return {"exact_tensors": exact, "expanded_tensors": expanded}
