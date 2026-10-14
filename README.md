# scPhenFormer：Pathway-Informed Patient Phenotype Prediction from Single-Cell Transcriptomes

The official code for "Adaptive Signature Learning for Niche-aware Spatial Transcriptomics Deconvolution".

**scPhenFormer** is a deep learning framework that learns pathway representations from single-cell transcriptomes and aggregates pathway-aligned instances for patient phenotype prediction.

------

## 🌟 Key Innovations

- **Pathway-Informed Representation**
- **Patient-Level Pathway Aggregation**
- **Pathway and Gene-Context Integration**

The workflow of scPhenFormer includes:
1. data.py: preprocess scRNA-seq data, construct gene–pathway mappings, and prepare patient-level cell samples.
2. model.py: implement pathway-informed representation learning, patient-level pathway aggregation, and phenotype prediction.
3. main_pretrain.py: pretrain the pathway encoder through masked gene-expression reconstruction.
4. main_finetune.py: fine-tune the pretrained model for patient phenotype prediction and evaluate the performance.

------

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

------

## 📊 Data Preparation

The required datasets in this paper are hosted on Google Drive. You can download the complete data package from the link below:

> 🔗 **Data Download Link:** [link](https://drive.google.com/drive/folders/1pn0MjqsHwBp-BIJQY3hkUAgjoAu23Ft7?usp=drive_link)

------

## 🚀 Installation

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

------

Install a PyTorch build compatible with your GPU before training.

## 🚀 Pretraining
```bash
python scripts/main_pretrain.py \
  --data "/path/data/..." \
```

## 🚀 Patient phenotype prediction

```bash
python scripts/main_finetune.py \
  --data "/path/data/..." 
```

