# 🌟scPhenFormer

scPhenFormer learns pathway representations from single-cell
transcriptomes and aggregates pathway-aligned instances for patient phenotype
prediction.

## 📂 Repository structure

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
## 📊 Data Preparation

The required datasets in this paper are hosted on Google Drive. You can download the complete data package from the link below:

> 🔗 **Data Download Link:** [link](https://drive.google.com/drive/folders/1pn0MjqsHwBp-BIJQY3hkUAgjoAu23Ft7?usp=drive_link)

Fine-tuning requires `patient_id`, `label` and `split` in the H5AD observation
metadata. The fixed `split` values must be `train`, `validation` and `test`;
`val` is also accepted as an alias for `validation`. The code preserves these
patient partitions and never generates new splits. Missing or inconsistent
partitions raise an error.

For COVID-19 leave-one-cohort-out evaluation, provide a separately prepared
H5AD with the fixed patient partitions for each held-out cohort and run each
fold independently. A COMBAT-held-out file represents only that fold.

## 🚀 Installation

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Install a PyTorch build compatible with your GPU before training.


## 🚀 Patient phenotype prediction

```bash
python scripts/main_finetune.py \
  --data "/path/data/..." 
```

