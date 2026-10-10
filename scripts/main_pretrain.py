from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from scphenformer.data import CellDataset, collate_cells, load_membership, load_vocab
from scphenformer.model import ScPhenFormerBackbone
from scphenformer.train_utils import cosine_learning_rate, distributed_device, move, require_path, seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("data", "vocab", "membership", "output"):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-genes", type=int, default=0)
    parser.add_argument("--maximum-lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--complement-tokens", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-project", default="")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    for name in ("data", "vocab", "membership", "output"):
        require_path(getattr(args, name), name)
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("Batch size and epochs must be positive")
    cosine_learning_rate(0, args.epochs, args.maximum_lr, args.minimum_lr, args.warmup_epochs)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new checkpoint path to preserve the existing experiment")
    device, rank, world_size = distributed_device()
    seed_everything(args.seed + rank)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    vocab = load_vocab(args.vocab)
    indptr, indices, pathways = load_membership(args.membership, len(vocab))
    backbone = ScPhenFormerBackbone(
        len(vocab), indptr, indices, number_of_pathways=pathways,
        number_of_complement_tokens=args.complement_tokens, dimension=args.dimension,
        depth=args.depth, number_of_heads=args.heads, dropout=args.dropout,
        mask_ratio=args.mask_ratio,
    ).to(device)
    model = backbone
    if world_size > 1:
        model = DistributedDataParallel(
            backbone, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    dataset = CellDataset(args.data, args.vocab, args.maximum_genes)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=args.workers, collate_fn=collate_cells,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.maximum_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    tracker = None
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.wandb_project:
            import wandb

            tracker = wandb.init(project=args.wandb_project, config=vars(args))
    history = []
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        for step, batch in enumerate(loader):
            progress = epoch + (step + 1) / len(loader)
            lr = cosine_learning_rate(progress, args.epochs, args.maximum_lr, args.minimum_lr, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = model(**batch)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = batch["gene_ids"].shape[0]
            totals[0] += result["loss"].detach().double() * count
            totals[1] += count
        if world_size > 1:
            dist.all_reduce(totals)
        if rank == 0:
            record = {"epoch": epoch + 1, "reconstruction_loss": float(totals[0] / totals[1]), "learning_rate": lr}
            history.append(record)
            pd.DataFrame(history).to_csv(output.with_suffix(".csv"), index=False)
            print(record, flush=True)
            if tracker:
                tracker.log(record, step=epoch + 1)
    if rank == 0:
        torch.save(
            {
                "model": backbone.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs,
                "vocab_symbols": [gene for gene, index in sorted(vocab.items(), key=lambda item: item[1])],
                "model_config": {
                    "vocab_size": len(vocab), "number_of_pathways": pathways,
                    "number_of_complement_tokens": args.complement_tokens, "dimension": args.dimension,
                    "depth": args.depth, "number_of_heads": args.heads, "mask_ratio": backbone.mask_ratio,
                    "dropout": args.dropout,
                },
                "arguments": vars(args),
            },
            output,
        )
        if tracker:
            tracker.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
