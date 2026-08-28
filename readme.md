# Multimodal Spatial GNN

## A Capstone Project for Spatial Transcriptomics-to-Proteomics Prediction

This repository contains the 2026 capstone project of [Yixuan Li](https://www.linkedin.com/in/yixuanli1105/) at the **Southern University of Science and Technology (SUSTech)**. The project investigates whether spatial transcriptomics, histology, and cell-composition information can be integrated with a graph neural network to predict spatial protein expression.

The resulting solution ranked **1st in the development phase** and **2nd in the testing phase** of the [STP Open Challenge on Codabench](https://www.codabench.org/competitions/10696/) under the participant name `liyx2022`. The corresponding results are available on the [competition leaderboard](https://www.codabench.org/competitions/10696/#/results-tab).

## Project Overview

Spatial transcriptomics measures gene expression while preserving tissue coordinates, but matched protein measurements are often more difficult or expensive to obtain. This capstone frames protein prediction as a spatial, multimodal learning problem: each measurement spot is represented as a graph node, nearby spots are connected by edges, and complementary biological and image-derived features are fused before graph message passing.

### Research question

This capstone asks two connected questions:

1. Can spatial protein abundance be predicted more accurately by jointly modeling RNA expression, tissue morphology, cell composition, and local spatial dependencies?
2. How well does such a model generalize from an observed tissue region or slice to a spatially shifted region or an unseen slice?

The framework therefore combines:

- RNA expression at each spatial spot;
- local morphology from H&E image patches;
- estimated cell-type abundance; and
- spatial relationships and protein co-expression priors.

### Model at a glance

The implemented pipeline consists of:

1. **RNA preprocessing and reduction** — normalized transcriptomic features are compressed with a supervised MLP reducer.
2. **Histology feature extraction** — multi-view H&E patches are encoded with an ImageNet-pretrained ResNet50 in the best submitted configuration.
3. **Biological context integration** — cell2location abundance estimates are added as cell-composition features.
4. **Multimodal fusion** — modality-specific representations are transformed and concatenated.
5. **Spatial graph learning** — a two-layer GATv2 model propagates information between neighboring tissue spots.
6. **Biologically informed optimization** — the training objective combines protein regression with spatial smoothness and protein co-expression regularization.

The task loss is Huber regression; Spearman correlation is used for validation and final evaluation. Modality masking, input noise, feature dropout, edge dropout, self-loops, normalization, and neighbor sampling are used to improve training stability and robustness across tissue slices.

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

The main STP Challenge dataset contains adjacent glioma tissue sections measured with Visium HD spatial transcriptomics, CODEX spatial proteomics, and matched H&E imaging. Its feature space includes approximately 18,085 genes and 44 protein markers at 16 μm spatial resolution. The internal training/validation split is approximately 19:1, while the held-out challenge test data come from a different tissue slice and therefore measure cross-slice generalization.

The thesis also evaluates the framework on an independent renal cell carcinoma (RCC) Xenium dataset containing approximately 405 genes, 27 protein markers, and H&E images. Left and right regions of the same section form a weaker spatial distribution-shift setting. This supplementary dataset is discussed in the thesis but is not included in the default repository workflow.

Download the STP Challenge data from [Google Drive](https://drive.google.com/drive/folders/1eq6sbTUaWCCOKcnkei6B65rozx-VX70K), then arrange the core files as follows:

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

The thesis compares a U-Net baseline, an RNA-only GAT, and the multimodal GAT, and also investigates alternative fusion strategies, graph operators, image encoders, and training schemes. The final design uses nonlinear MLP-based fusion and GATv2Conv: GATv2 was approximately 2% better than the compared GraphSAGE and GCN variants, while MLP fusion was more stable than direct concatenation or learned linear weighting. UNI-2h was slightly better than ResNet50, but ResNet50 was retained for accessibility and reproducibility. Neighbor sampling reduced memory use without degrading performance and appeared to provide useful stochastic regularization.

Useful follow-up ablations include disabling H&E or cell-abundance features, changing graph radius and neighborhood size, removing biological regularizers, varying H&E patch resolution and receptive field, and comparing modality masking settings. Give each experiment a distinct `EXP_NAME` in the scripts so its artifacts remain isolated.

## Results and Interpretation

Spearman correlation is the primary metric because it measures agreement in expression ranking while remaining insensitive to scale. The challenge score, **Top-10 SCC Mean**, averages the ten best per-protein Spearman correlations.

| Evaluation setting | Top-10 SCC Mean | Mean Spearman | Interpretation |
|---|---:|---:|---|
| STP validation, U-Net | 0.660 | 0.400 | Image-style convolutional baseline |
| STP validation, GAT (RNA only) | 0.700 | 0.630 | Spatial graph structure provides a clear gain |
| STP validation, GAT (RNA + H&E) | **0.747** | **0.6437** | Best in-distribution result; ranked 1st |
| STP unseen-slice OOD test | **0.559** | 0.324 | Cross-slice distribution shift; ranked 2nd |
| RCC spatial holdout | 0.740 | 0.560 | Stable transfer between regions of the same section |

The multimodal model's mean training Spearman was 0.6884 on the best STP run, close enough to the validation value of 0.6437 to show no severe in-distribution overfitting. Adding H&E features improved the validation Top-10 SCC Mean from 0.700 to 0.747, supporting the thesis claim that morphology complements noisy or incomplete RNA measurements.

The central result is nevertheless the generalization gap: performance fell from 0.747 on the in-distribution validation split to 0.559 on an unseen glioma slice. Per-protein behavior was also heterogeneous. Structure- or microenvironment-associated markers such as MAP2, IDH1, and CD14 remained comparatively predictable on the OOD test, whereas markers including Ki67 and SMA were more difficult. Across the reported analyses, individual protein correlations spanned approximately 0.42–0.83, reflecting differences in spatial organization and biological regulation.

The RCC experiment provides an informative contrast. Its training, validation, and spatial-holdout mean Spearman values were approximately 0.53, 0.52, and 0.56, respectively. Because the regions came from the same section, morphology and spatial structure were more consistent than in the cross-slice STP test. Taken together, the experiments indicate that the model handles mild within-section shift well, while stain, tissue-structure, and experimental shifts between sections remain the main challenge.

The biological priors also improve interpretability: cell-composition-aware smoothness encourages locally consistent predictions only where cellular microenvironments are similar, while the STRING-derived co-expression constraint helps preserve coordinated relationships between proteins. These findings are empirical within the thesis experiments and should not be interpreted as causal or clinical validation.

## Limitations and Future Work

- **Cross-slice distribution shift remains the main limitation.** Stain variation, tissue architecture, and experimental noise substantially reduce performance on an unseen section. Future work should prioritize domain adaptation and pretrained domain-invariant representations for cross-sample and cross-tissue transfer.
- **Morphology depends on image preprocessing choices.** Patch resolution, receptive-field size, and augmentation can materially affect H&E feature quality and should be evaluated systematically.
- **External validation is still limited.** The RCC experiment measures a left-to-right split within one section, which is weaker than transfer across patients, tissues, laboratories, or measurement platforms. Broader external cohorts are needed.
- **Upstream priors may propagate bias.** Cell2location estimates and pretrained image encoders are learned representations rather than direct ground truth; their uncertainty should be modeled explicitly.
- **Performance varies considerably by protein.** Future work should investigate protein-specific uncertainty, regulatory complexity, and whether adaptive or pathway-aware objectives help difficult markers.
- **Reproducibility and performance trade off.** Pathology foundation models such as UNI-2h showed a small gain but require controlled access and further tuning; systematic benchmarking against ResNet50 remains useful.
- **Compute remains non-trivial.** Neighbor sampling controls graph memory usage, but whole-slide feature extraction is still expensive. More efficient encoders, caching, and scalable graph construction would improve accessibility.

## Author and Acknowledgements

**Yixuan Li** — [LinkedIn](https://www.linkedin.com/in/yixuanli1105/)<br>
Capstone Project, Southern University of Science and Technology (SUSTech), 2026

This project was developed using data and evaluation infrastructure provided by the STP Open Challenge. The repository is intended for academic research and reproducibility.
