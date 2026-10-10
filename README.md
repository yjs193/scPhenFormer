# scPhenFormer

scPhenFormer learns gene-aware pathway representations from single-cell
transcriptomes and aggregates pathway-aligned instances for patient phenotype
prediction.

## Repository structure

```text
scPhenFormer/
├── scphenformer/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train_utils.py
├── scripts/
│   ├── main_pretrain.py
│   └── main_finetune.py
├── requirements.txt
├── pyproject.toml
└── README.md
```


## Installation

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Install a PyTorch build compatible with your GPU before training.


## Patient phenotype prediction

```bash
python scripts/main_finetune.py \
  --data "/path/data/..." 
```

