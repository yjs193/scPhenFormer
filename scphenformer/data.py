from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def load_vocab(path: str | Path) -> dict[str, int]:
    vocab = json.loads(Path(path).read_text())
    if not isinstance(vocab, dict) or vocab.get("<pad>") != 0:
        raise ValueError("Vocabulary must reserve token zero for <pad>")
    if any(type(index) is not int for index in vocab.values()):
        raise ValueError("Vocabulary IDs must be integers")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be unique and contiguous")
    return {str(gene): index for gene, index in vocab.items()}


def load_membership(path: str | Path, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    matrix = sp.load_npz(path).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape[1] != vocab_size - 1 or matrix.shape[0] == 0:
        raise ValueError("Membership columns must match vocabulary IDs 1..N")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data != 1):
        raise ValueError("Membership must be a binary gene-pathway matrix")
    by_gene = matrix.transpose().tocsr()
    indptr = np.empty(vocab_size + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = by_gene.indptr
    return (
        torch.from_numpy(indptr),
        torch.from_numpy(by_gene.indices.astype(np.int64, copy=True)),
        int(matrix.shape[0]),
    )


def _row_to_tokens(
    row: sp.csr_matrix, column_to_token: np.ndarray, maximum_genes: int
) -> tuple[np.ndarray, np.ndarray]:
    row = row.copy()
    row.sum_duplicates()
    values = row.data.astype(np.float32, copy=False)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expression must be finite, non-negative and normalized with log1p")
    keep = values > 0
    tokens, values = column_to_token[row.indices[keep]], values[keep]
    unique, inverse = np.unique(tokens, return_inverse=True)
    if len(unique) != len(tokens):
        summed = np.zeros(len(unique), dtype=np.float64)
        np.add.at(summed, inverse, np.expm1(values.astype(np.float64)))
        tokens = unique
        values = np.log1p(summed).astype(np.float32)
    if maximum_genes > 0 and len(values) > maximum_genes:
        selected = np.argpartition(values, -maximum_genes)[-maximum_genes:]
        selected.sort()
        tokens, values = tokens[selected], values[selected]
    return tokens.astype(np.int64, copy=False), values


class CellDataset(Dataset):
    def __init__(self, h5ad_path: str | Path, vocab_path: str | Path, maximum_genes: int = 0):
        self.path = str(Path(h5ad_path).resolve())
        metadata = ad.read_h5ad(self.path, backed="r")
        self.obs = metadata.obs.copy()
        self.var = metadata.var.copy()
        self.number_of_cells = metadata.n_obs
        metadata.file.close()
        vocab = load_vocab(vocab_path)
        symbols = self.var.get("gene_symbol", self.var.index.to_series()).astype(str)
        missing = sorted(set(symbols) - set(vocab))
        if missing:
            raise ValueError(f"Genes absent from vocabulary: {missing[:10]}; extend the vocabulary first")
        self.column_to_token = np.asarray([vocab[gene] for gene in symbols], dtype=np.int64)
        if np.any(self.column_to_token == 0):
            raise ValueError("Padding cannot appear on the expression gene axis")
        self.maximum_genes = maximum_genes
        self._adata = None
        self._pid = None

    def _matrix(self, rows: np.ndarray) -> sp.csr_matrix:
        if self._adata is None or self._pid != os.getpid():
            if self._adata is not None:
                self._adata.file.close()
            self._adata = ad.read_h5ad(self.path, backed="r")
            self._pid = os.getpid()
        unique, inverse = np.unique(rows, return_inverse=True)
        return sp.csr_matrix(self._adata.X[unique])[inverse]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_adata"] = None
        state["_pid"] = None
        return state

    def __len__(self) -> int:
        return self.number_of_cells

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return _row_to_tokens(self._matrix(np.asarray([index])), self.column_to_token, self.maximum_genes)


def collate_cells(batch: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, torch.Tensor]:
    width = max(1, max(len(item[0]) for item in batch))
    ids = torch.zeros(len(batch), width, dtype=torch.long)
    values = torch.zeros(len(batch), width, dtype=torch.float32)
    mask = torch.zeros(len(batch), width, dtype=torch.bool)
    for row, (gene_ids, expression) in enumerate(batch):
        length = len(gene_ids)
        ids[row, :length] = torch.from_numpy(gene_ids)
        values[row, :length] = torch.from_numpy(expression)
        mask[row, :length] = True
    return {"gene_ids": ids, "expression": values, "valid_genes": mask}


def validate_patient_partitions(obs: pd.DataFrame) -> pd.DataFrame:
    columns = ["patient_id", "label", "split"]
    if not set(columns).issubset(obs):
        raise ValueError("H5AD obs must contain patient_id, label and a predefined split")
    if obs[columns].isna().any().any():
        raise ValueError("Patient identifiers, phenotypes and splits cannot be missing")
    obs = obs.copy()
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs["label"] = obs["label"].astype(str)
    if obs[["patient_id", "label"]].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Patient identifiers and phenotypes cannot be empty")
    obs["split"] = obs["split"].astype(str).replace({"val": "validation"})
    if not set(obs["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Split values must be train, validation or test")
    patients = obs[columns].drop_duplicates()
    if patients["patient_id"].duplicated().any():
        raise ValueError("A patient cannot cross labels or data partitions")
    if set(patients["split"]) != {"train", "validation", "test"}:
        raise ValueError("Predefined splits must include training, validation and test patients")
    return obs


class PatientDataset(CellDataset):
    def __init__(
        self, h5ad_path: str | Path, vocab_path: str | Path, maximum_genes: int = 2000,
        cells_per_patient: int = 300, context_genes: int = 4000, reuse_hvg: bool = False,
    ):
        super().__init__(h5ad_path, vocab_path, maximum_genes)
        self.obs = validate_patient_partitions(self.obs)
        patients = self.obs[["patient_id", "label", "split"]].drop_duplicates()
        self.patient_ids = patients["patient_id"].tolist()
        self.label_names = sorted(patients["label"].unique())
        self.label_to_index = {name: index for index, name in enumerate(self.label_names)}
        self.patient_labels = patients.set_index("patient_id")["label"].to_dict()
        self.rows = {str(patient): np.asarray(rows, dtype=np.int64) for patient, rows in self.obs.groupby("patient_id", sort=False).indices.items()}
        self.cells_per_patient = int(cells_per_patient)
        if self.cells_per_patient < 1:
            raise ValueError("Cells per patient must be positive")
        if not 1 <= context_genes <= len(self.var):
            raise ValueError("Requested context genes must fit the expression gene axis")
        if reuse_hvg:
            if "highly_variable" not in self.var:
                raise ValueError("Reusing HVGs requires a training-derived highly_variable annotation")
            self.context_columns = np.flatnonzero(self.var["highly_variable"].to_numpy(dtype=bool))
        else:
            import scanpy as sc

            train_rows = np.flatnonzero(self.obs["split"].eq("train").to_numpy())
            if not len(train_rows):
                raise ValueError("HVG selection requires training patients")
            training = ad.AnnData(self._matrix(train_rows), var=self.var.copy())
            sc.pp.highly_variable_genes(training, n_top_genes=min(context_genes, len(self.var)), flavor="seurat")
            dispersion = training.var["dispersions_norm"].to_numpy(dtype=float)
            dispersion = np.nan_to_num(dispersion, nan=-np.inf)
            self.context_columns = np.sort(np.argsort(-dispersion, kind="stable")[:context_genes])
            del training
        if len(self.context_columns) != context_genes:
            raise ValueError("Stored highly_variable annotation must match the requested context-gene count")
        if self._adata is not None:
            self._adata.file.close()
            self._adata = None
        self.sampling_seed = None

    def subset(self, split: str) -> "PatientDataset":
        patients = self.obs.loc[self.obs["split"].eq(split), "patient_id"].unique().tolist()
        if not patients:
            raise ValueError(f"No patients in {split}")
        clone = object.__new__(PatientDataset)
        clone.__dict__ = self.__dict__.copy()
        clone.patient_ids = patients
        clone.rows = {patient: self.rows[patient] for patient in patients}
        clone._adata = None
        return clone

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        patient = self.patient_ids[index]
        available = self.rows[patient]
        if self.sampling_seed is None:
            rows = np.random.choice(available, self.cells_per_patient, replace=len(available) < self.cells_per_patient)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.sampling_seed, index]))
            rows = rng.choice(available, self.cells_per_patient, replace=len(available) < self.cells_per_patient)
        matrix = self._matrix(rows)
        cells = [_row_to_tokens(matrix[row], self.column_to_token, self.maximum_genes) for row in range(len(rows))]
        context = matrix[:, self.context_columns].toarray().astype(np.float32)
        return {"patient_id": patient, "cells": cells, "context": context, "label": self.label_to_index[self.patient_labels[patient]]}


def collate_patients(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    maximum_cells = max(len(item["cells"]) for item in batch)
    maximum_genes = max(1, max(len(cell[0]) for item in batch for cell in item["cells"]))
    context_genes = batch[0]["context"].shape[1]
    shape = len(batch), maximum_cells, maximum_genes
    ids = torch.zeros(shape, dtype=torch.long)
    expression = torch.zeros(shape, dtype=torch.float32)
    valid_genes = torch.zeros(shape, dtype=torch.bool)
    context = torch.zeros(len(batch), maximum_cells, context_genes)
    cell_mask = torch.zeros(len(batch), maximum_cells, dtype=torch.bool)
    for patient_index, item in enumerate(batch):
        number_of_cells = len(item["cells"])
        cell_mask[patient_index, :number_of_cells] = True
        context[patient_index, :number_of_cells] = torch.from_numpy(item["context"])
        for cell_index, (gene_ids, values) in enumerate(item["cells"]):
            length = len(gene_ids)
            ids[patient_index, cell_index, :length] = torch.from_numpy(gene_ids)
            expression[patient_index, cell_index, :length] = torch.from_numpy(values)
            valid_genes[patient_index, cell_index, :length] = True
    return {
        "patient_ids": [item["patient_id"] for item in batch], "gene_ids": ids,
        "expression": expression, "valid_genes": valid_genes, "context_expression": context,
        "cell_mask": cell_mask, "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
