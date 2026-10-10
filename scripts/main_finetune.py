from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scphenformer.data import PatientDataset, collate_patients, load_membership, load_vocab
from scphenformer.model import ScPhenFormer, ScPhenFormerBackbone, load_pretrained_backbone
from scphenformer.train_utils import classification_metrics, cosine_learning_rate, move, require_path, seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("data", "vocab", "membership", "pretrained", "output"):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cells-per-patient", type=int, choices=(300, 400, 500), default=300)
    parser.add_argument("--maximum-genes", type=int, default=2000)
    parser.add_argument("--context-genes", type=int, default=4000)
    parser.add_argument("--reuse-hvg", action="store_true")
    parser.add_argument("--test-cohort", default="")
    parser.add_argument("--cohort-column", default="cohort")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--maximum-lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-bags", type=int, default=2)
    parser.add_argument("--test-bags", type=int, default=5)
    parser.add_argument("--cell-chunk-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb-project", default="")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: ScPhenFormer, loader: DataLoader, device: torch.device, bags: int, sampling_seed: int
) -> tuple[dict, pd.DataFrame]:
    if bags < 1:
        raise ValueError("Evaluation requires at least one sampled cell set")
    model.eval()
    probabilities = {}
    labels = {}
    order = []
    for bag in range(bags):
        loader.dataset.sampling_seed = sampling_seed + bag
        for batch in loader:
            batch = move(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["gene_ids"], batch["expression"], batch["valid_genes"],
                    batch["context_expression"], batch["cell_mask"],
                )
            values = output["logits"].float().softmax(-1).cpu().numpy()
            for patient, label, vector in zip(batch["patient_ids"], batch["labels"].cpu().tolist(), values):
                if patient not in probabilities:
                    probabilities[patient] = vector.copy()
                    labels[patient] = label
                    order.append(patient)
                else:
                    probabilities[patient] += vector
    loader.dataset.sampling_seed = None
    mean_probabilities = np.stack([probabilities[patient] / bags for patient in order])
    predictions = mean_probabilities.argmax(-1).tolist()
    truth = [labels[patient] for patient in order]
    rows = pd.DataFrame({"GroundTruth": truth, "prediction": predictions, "patient_id": order})
    for index, name in enumerate(loader.dataset.label_names):
        rows[f"probability_{name}"] = mean_probabilities[:, index]
    return classification_metrics(truth, predictions, len(loader.dataset.label_names)), rows


def main() -> None:
    args = arguments()
    for name in ("data", "vocab", "membership", "pretrained", "output"):
        require_path(getattr(args, name), name)
    if args.patience < 1 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("Patience, batch size and epochs must be positive")
    cosine_learning_rate(0, args.epochs, args.maximum_lr, args.minimum_lr, args.warmup_epochs)
    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatientDataset(
        args.data, args.vocab, args.maximum_genes, args.cells_per_patient,
        args.context_genes, args.reuse_hvg, args.test_cohort, args.cohort_column, args.split_seed,
    )
    splits = {name: dataset.subset(name) for name in ("train", "validation", "test")}
    if min(len(rows) for rows in dataset.rows.values()) <= args.cells_per_patient:
        print("Some patients have no larger cell pool than the requested sample; use complete cell pools for fresh subsets.")
    loaders = {
        name: DataLoader(
            subset, batch_size=args.batch_size, shuffle=name == "train",
            num_workers=args.workers, collate_fn=collate_patients, pin_memory=device.type == "cuda",
        )
        for name, subset in splits.items()
    }
    vocab = load_vocab(args.vocab)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pretrained_symbols = checkpoint.get("vocab_symbols")
    symbols = [gene for gene, index in sorted(vocab.items(), key=lambda item: item[1])]
    if pretrained_symbols is not None and symbols[:len(pretrained_symbols)] != pretrained_symbols:
        raise ValueError("Downstream vocabulary must retain pretrained token IDs and append new genes")
    indptr, indices, pathways = load_membership(args.membership, len(vocab))
    model_config = checkpoint.get("model_config", {})
    backbone = ScPhenFormerBackbone(
        len(vocab), indptr, indices, number_of_pathways=pathways,
        number_of_complement_tokens=model_config.get("number_of_complement_tokens", model_config.get("num_residual_tokens", 32)),
        dimension=model_config.get("dimension", model_config.get("embed_dim", 256)), depth=model_config.get("depth", 12),
        number_of_heads=model_config.get("number_of_heads", model_config.get("num_heads", 8)),
        dropout=model_config.get("dropout", 0.1),
    )
    print(f"pretrained_weights={load_pretrained_backbone(backbone, checkpoint)}")
    model = ScPhenFormer(
        backbone, number_of_context_genes=len(dataset.context_columns),
        number_of_classes=len(dataset.label_names), cell_chunk_size=args.cell_chunk_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.maximum_lr, weight_decay=args.weight_decay
    )
    output = Path(args.output)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a new output directory to preserve the existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    patients = dataset.obs[["patient_id", "label", "split"]].drop_duplicates()
    patients.to_csv(output / "partitions.csv", index=False)
    context_symbols = dataset.var.iloc[dataset.context_columns].get(
        "gene_symbol", dataset.var.iloc[dataset.context_columns].index.to_series()
    ).astype(str).tolist()
    pd.Series(context_symbols, name="gene").to_csv(output / "context_genes.csv", index=False)
    tracker = None
    if args.wandb_project:
        import wandb

        tracker = wandb.init(project=args.wandb_project, config=vars(args))
    best_score, stale_epochs = -1.0, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_patients = 0.0, 0
        for step, batch in enumerate(loaders["train"]):
            progress = epoch + (step + 1) / len(loaders["train"])
            lr = cosine_learning_rate(
                progress, args.epochs, args.maximum_lr, args.minimum_lr, args.warmup_epochs
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = model(
                    batch["gene_ids"], batch["expression"], batch["valid_genes"],
                    batch["context_expression"], batch["cell_mask"],
                )
                loss = F.cross_entropy(result["logits"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch["patient_ids"])
            total_loss += float(loss.detach()) * count
            total_patients += count
        validation, _ = evaluate(model, loaders["validation"], device, args.validation_bags, args.seed * 10000)
        record = {
            "epoch": epoch + 1, "train_loss": total_loss / total_patients, "learning_rate": lr,
            "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(record, flush=True)
        if tracker:
            tracker.log(record, step=epoch + 1)
        if validation["macro_f1"] > best_score:
            best_score, stale_epochs = validation["macro_f1"], 0
            torch.save(
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "labels": dataset.label_names, "epoch": epoch + 1,
                    "selection_split": "validation", "selection_metric": "macro_f1",
                    "validation_metrics": validation, "context_genes": context_symbols,
                    "vocab_symbols": symbols, "arguments": vars(args),
                    "model_config": {
                        "vocab_size": len(vocab), "number_of_pathways": pathways,
                        "number_of_complement_tokens": backbone.number_of_complement_tokens,
                        "dimension": backbone.dimension, "depth": len(backbone.transformer.layers),
                        "number_of_heads": backbone.transformer.layers[0].self_attn.num_heads,
                        "backbone_dropout": backbone.transformer.layers[0].dropout.p,
                        "dropout": args.dropout, "number_of_context_genes": len(dataset.context_columns),
                        "number_of_classes": len(dataset.label_names),
                    },
                },
                output / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, predictions = evaluate(model, loaders["test"], device, args.test_bags, args.seed * 10000 + 5000)
    predictions.to_csv(output / "predictions.csv", index=False)
    report = {
        "best_epoch": best["epoch"], "selection_split": "validation", "selection_metric": "macro_f1",
        "validation": best["validation_metrics"], "test": test_metrics, "labels": dataset.label_names,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    if tracker:
        tracker.log({f"test_{key}": value for key, value in test_metrics.items()})
        tracker.finish()


if __name__ == "__main__":
    main()
