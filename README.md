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

