# Multimodal Spatial GNN

## A Capstone Project for Spatial Transcriptomics-to-Proteomics Prediction

This repository contains the 2026 capstone project of [Yixuan Li](https://www.linkedin.com/in/yixuanli1105/) at the **Southern University of Science and Technology (SUSTech)**. The project investigates whether spatial transcriptomics, histology, and cell-composition information can be integrated with a graph neural network to predict spatial protein expression.

The resulting solution ranked **1st in the development phase** and **2nd in the testing phase** of the [STP Open Challenge on Codabench](https://www.codabench.org/competitions/10696/) under the participant name `liyx2022`. The corresponding results are available on the [competition leaderboard](https://www.codabench.org/competitions/10696/#/results-tab).

## Project Overview

Spatial transcriptomics measures gene expression while preserving tissue coordinates, but matched protein measurements are often more difficult or expensive to obtain. This capstone frames protein prediction as a spatial, multimodal learning problem: each measurement spot is represented as a graph node, nearby spots are connected by edges, and complementary biological and image-derived features are fused before graph message passing.

### Research question

Can spatial protein abundance be predicted more accurately by jointly modeling:

- RNA expression at each spatial spot;
- local morphology from H&E image patches;
- estimated cell-type abundance; and
- spatial relationships and protein co-expression priors?

### Model at a glance

The implemented pipeline consists of:

1. **RNA preprocessing and reduction** — normalized transcriptomic features are compressed with a supervised MLP reducer.
2. **Histology feature extraction** — multi-view H&E patches are encoded with an ImageNet-pretrained ResNet50 in the best submitted configuration.
3. **Biological context integration** — cell2location abundance estimates are added as cell-composition features.
4. **Multimodal fusion** — modality-specific representations are transformed and concatenated.
5. **Spatial graph learning** — a two-layer GATv2 model propagates information between neighboring tissue spots.
6. **Biologically informed optimization** — the training objective combines protein regression with spatial smoothness and protein co-expression regularization.

Modality masking, input noise, feature dropout, and edge dropout are used to improve robustness across tissue slices.

## Repository Structure

```text
multimodal-spatial-gnn/
├── cfgs/                         # Experiment configuration
├── common/                       # Losses, metrics, optimization, and utilities
├── data/                         # Local datasets, priors, and extracted features
├── datasets/                     # Graph datasets and preprocessing pipeline
├── models/                       # GATv2, fusion, and feature-reduction modules
├── scripts/                      # Data processing, training, and inference commands
├── trainers/                     # Training and prediction logic
├── train.py                      # Training entry point
├── test_final.py                 # Final inference entry point
└── environment.yml               # Reference Conda environment
```

## Environment Setup

The reference experiments used Ubuntu 22.04.5 LTS, Python 3.10, CUDA 12.1, and an NVIDIA RTX 4090 with 24 GB VRAM.

### 1. Clone the repository

```bash
git clone https://github.com/ShellyLeee/multimodal-spatial-gnn.git
cd multimodal-spatial-gnn
```

### 2. Create the environment

For an environment close to the original experiment, use the supplied specification:

```bash
conda env create -f environment.yml
conda activate stp_gnn
```

Alternatively, create a lightweight environment manually:

```bash
conda create -n stp_gnn python=3.10 -y
conda activate stp_gnn

pip install scanpy anndata h5py numpy pandas matplotlib seaborn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install scikit-learn tqdm scipy pyyaml joblib timm huggingface_hub
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Verify that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Dataset Preparation

Download the challenge dataset from [Google Drive](https://drive.google.com/drive/folders/1eq6sbTUaWCCOKcnkei6B65rozx-VX70K), then arrange the core files as follows:

```text
data/
├── train_rna.h5ad
├── train_pro.h5ad
├── valid_rna.h5ad
├── test_rna.h5ad
├── HE_image_full_resolution.tif
├── test_HE_image_full_resolution.tif
└── bio_info/
    ├── cell2location_predicted_cell_abundance_mean_train.csv
    ├── cell2location_predicted_cell_abundance_mean_valid.csv
    ├── cell2location_predicted_cell_abundance_mean_test.csv
    └── protein_coexpression_matrix.txt
```

The large challenge data, extracted image features, and checkpoints are not stored directly in this repository. See [`data/readme.md`](data/readme.md) for further details.

## Reproducing the Best Submission

Pretrained weights and preprocessed artifacts provide the shortest path to reproducing the testing-phase submission. Model hyperparameters and random seeds are stored in `logs/best_run/config.yaml`.

### 1. Download the pretrained run

Download the `best_run` package from [Google Drive](https://drive.google.com/drive/folders/1ww_mRSO0mRrSCqaWFgvZ9XO7xyPBlUeQ?usp=sharing) and place it under `logs/`:

```text
logs/best_run/
├── checkpoints/
│   ├── gnn_best.pt
│   └── mlp_reducer_best.pt
├── config.yaml
├── he_scaler.joblib
└── cell_mean_scaler.joblib
```

### 2. Prepare test-set H&E features

Recommended: download `test_he_features_robust_all.npy` from the same Google Drive folder and place it at:

```text
data/extracted_features/test_he_features_robust_all.npy
```

To regenerate the features instead, place the [ImageNet-pretrained ResNet50 weights](https://download.pytorch.org/models/resnet50-0676ba61.pth) in `data/pretrained_models/`, then run:

```bash
export CUDA_VISIBLE_DEVICES=0
python datasets/processing/extract_image_features.py \
  --config logs/best_run/config.yaml \
  --mode test
```

Feature extraction typically takes 40–60 minutes on a GPU and longer on a CPU.

### 3. Run inference

```bash
export CUDA_VISIBLE_DEVICES=0
python test_final.py \
  --config logs/best_run/config.yaml \
  --checkpoint logs/best_run/checkpoints/gnn_best.pt \
  --output logs/best_run/predictions_best.csv \
  --exp_name best_run
```

Predictions are written to `logs/best_run/predictions_best.csv`. Inference takes approximately one minute on the reference GPU. Minor numerical variation may occur because of floating-point and GPU-level nondeterminism.

## Training from Scratch

### 1. Extract H&E features

Place `resnet50-0676ba61.pth` under `data/pretrained_models/`, confirm the paths in `cfgs/gnn.yaml`, and run:

```bash
bash scripts/data_processing.sh
```

Generated features are saved in `data/extracted_features/`.

### 2. Train the multimodal GNN

```bash
bash scripts/train.sh
```

By default, experiment artifacts are written to `logs/run_01/`, including checkpoints, the resolved configuration, metrics, fitted scalers, and training-history plots.

### 3. Generate final predictions

```bash
bash scripts/test_final.sh
```

The default output is `logs/run_01/predictions_run_01_final.csv`.

## Configuration and Experimentation

The main experiment settings are defined in [`cfgs/gnn.yaml`](cfgs/gnn.yaml). This file controls input paths, preprocessing, graph construction, multimodal fusion, GATv2 architecture, regularization, optimization, and reproducibility settings.

Useful ablations for extending the capstone include disabling H&E or cell-abundance features, changing the graph radius and neighborhood size, removing co-expression regularization, and comparing modality masking settings. Give each experiment a distinct `EXP_NAME` in the scripts so its artifacts remain isolated.

## Results and Interpretation

The Codabench placements demonstrate that the proposed multimodal spatial approach was competitive on both development and held-out testing data. The gap between the two phases also highlights an important capstone finding: generalization across tissue slices remains harder than fitting an internal split. The included robustness mechanisms are intended to reduce reliance on any single modality and improve transfer to an unseen slice.

Leaderboard rankings should be interpreted in the context of the STP Open Challenge dataset and evaluation protocol; they are not evidence of clinical validity.

## Limitations and Future Work

- Evaluation is based on the challenge data and may not generalize to other tissues, platforms, or staining protocols.
- Cell-abundance estimates and pretrained image representations may introduce upstream bias.
- GPU memory and H&E feature-extraction time are substantial.
- Future work could evaluate pathology-specific foundation models, stronger cross-modal fusion, uncertainty estimation, and external biological validation.

## Author and Acknowledgements

**Yixuan Li** — [LinkedIn](https://www.linkedin.com/in/yixuanli1105/)<br>
Capstone Project, Southern University of Science and Technology (SUSTech), 2026

This project was developed using data and evaluation infrastructure provided by the STP Open Challenge. The repository is intended for academic research and reproducibility.
