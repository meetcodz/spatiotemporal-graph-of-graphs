# Spatiotemporal Graph-of-Graphs (ST-GoG) for Multivariate Time Series Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

A modular, production-ready implementation of the **ST-GoG Framework** for **Multivariate Time Series Classification (MTSC)**, originating from the M.Tech dissertation by Tushar Sharma under the supervision of Dr. Anjali at ABV-IIITM Gwalior.

---

## 📌 Executive Summary

Multivariate time series recordings (such as 12-lead ECGs, high-density EEGs, multi-axis IMU sensors, and traffic networks) simultaneously exhibit three distinct dependency structures:
1. **Intra-Channel Temporal Dynamics:** Temporal phase alignments, local motifs, and long-range dependencies within individual channels.
2. **Inter-Channel Spatial Dynamics:** Coordinated relational interactions across sensor dimensions.
3. **Temporal Non-Stationarity:** Relational coupling that dynamically evolves across temporal segments.

Prior state-of-the-art models like **ROCKET** treat channels independently or via simple linear projections, while **SAGoG** constructs dense instance-level similarity graphs via Dynamic Time Warping (DTW) with crippling $\mathcal{O}(N^2)$ memory complexity.

**ST-GoG** solves this challenge by cleanly decoupling temporal feature extraction and multi-scale graph reasoning, and unifying them via hierarchical segment-level graph aggregation.

```
                          ┌──────────────────────────┐
                          │ Multivariate Time Series │
                          │     X in R^(C x T)       │
                          └─────────────┬────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │        Temporal Stream       │              │        Spatial Stream        │
  │     ResTCN (d = 1..16)       │              │       Multi-Scale GCN        │
  │   + Squeeze-and-Excitation   │              │   (Chain Radii r = 1, 3, 5)  │
  └──────────────┬───────────────┘              └──────────────┬───────────────┘
                 │                                             │
         ┌───────┴───────┐                             ┌───────┴───────┐
         ▼               ▼                             ▼               ▼
  [Global Pool]   [Hierarchical GoG 1]          [Global Pool]   [Hierarchical GoG 2]
     (t_pool)        (Inner GCN r=3,               (g_pool)        (Inner GCN r=3,
                     Outer GCN r=1)                                Outer GCN r=1)
         │               │                             │               │
         └───────────────┼──────────────┬──────────────┘               │
                         ▼              ▼                              ▼
                 ┌─────────────────────────────────────────────────────────────┐
                 │          4-Pathway Softmax Fusion [w1, w2, w3, w4]          │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │   2-Block MLP Classifier    │
                                 └─────────────────────────────┘
```

---

## 📂 Repository Structure

```
.
├── configs/                     # YAML experiment configurations
│   ├── default.yaml             # Standard hyperparameter configuration
│   ├── natops.yaml              # Optimized settings for NATOPS Aviation dataset
│   ├── pemssf.yaml              # Optimized settings for PEMS-SF Traffic dataset
│   └── synthetic.yaml           # Fast offline configuration for smoke testing
├── stgog/                       # Main Python Package
│   ├── __init__.py              # Package API export
│   ├── models/                  # Neural network modules
│   │   ├── restcn.py            # Residual Dilated TCN + Squeeze-and-Excitation
│   │   ├── graph_conv.py        # GraphConv, MultiScaleGCN, Chain Adjacency
│   │   ├── gog.py               # Hierarchical Graph-of-Graphs & Attention Pooling
│   │   ├── fusion.py            # 4-Pathway Softmax Fusion & Classification Head
│   │   └── stgog_model.py       # Unified end-to-end STGoGModel
│   ├── data/                    # Data ingestion & processing
│   │   ├── loader.py            # UEA/UCR loader, standard scaling, caching, synthetic data
│   │   └── augmentations.py     # MixUp, Gaussian jitter, Test-Time Augmentation (TTA)
│   ├── training/                # Training & optimization engine
│   │   ├── trainer.py           # Hardware-aware Trainer (CPU/CUDA, AMP, DDP support)
│   │   ├── snapshot.py          # Snapshot Ensembling manager
│   │   └── scheduler.py         # CosineAnnealingWarmRestarts & stochastic depth
│   └── utils/                   # Metrics & visualization
│       ├── config.py            # Config parser & default settings
│       ├── metrics.py           # Accuracy, Macro-F1, Calibration ECE, Confusion Matrix
│       └── visualization.py     # Plotting training curves & confusion matrices
├── scripts/                     # Executable command-line interfaces
│   ├── train.py                 # Single-dataset training CLI
│   ├── evaluate.py              # Checkpoint evaluation & inference CLI
│   └── run_batch_benchmark.py   # Batch benchmarking across 22 UEA datasets
├── tests/                       # Automated unit and integration test suite
│   ├── test_models.py           # Shape and forward pass verification
│   ├── test_data.py             # Data loader, MixUp, and augmentations tests
│   └── test_training.py         # End-to-end trainer smoke tests
├── pyproject.toml               # Modern build configuration (PEP 517/518)
├── setup.py                     # Setuptools packaging script
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Testing and development dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install it in editable mode:

```bash
# Clone the repository
git clone https://github.com/your-username/stgog.git
cd stgog

# Install dependencies and the package
pip install -e .
```

To include benchmarking dependencies (`sktime`) and development tools (`pytest`):

```bash
pip install -e ".[benchmarks,dev]"
```

### 2. Verify Installation with Unit Tests

Run the complete test suite:

```bash
pytest tests -v
```

All 13 unit tests should pass.

### 3. Fast Smoke Test (Synthetic Data)

Train an ST-GoG model on synthetic multivariate time series without downloading external datasets:

```bash
python scripts/train.py --config configs/synthetic.yaml
```

Outputs will be saved in `outputs/synthetic/` (including `metrics.json`, `training_curves.png`, and `confusion_matrix.png`).

### 4. Training on Benchmark Datasets (e.g. NATOPS)

```bash
python scripts/train.py --config configs/natops.yaml
```

Or override settings directly from the command line:

```bash
python scripts/train.py --dataset FingerMovements --epochs 100 --batch-size 16 --lr 1e-3
```

### 5. Evaluating Checkpoints

Evaluate a saved model checkpoint with 7-pass Test-Time Augmentation (TTA):

```bash
python scripts/evaluate.py --checkpoint checkpoints/synthetic/best_model.pt --config configs/synthetic.yaml --synthetic
```

---

## 📊 Benchmark Results (22 UEA Datasets)

ST-GoG was evaluated across 22 standardized UEA benchmark datasets:

| Metric | ST-GoG (Proposed) | SAGoG | ROCKET | Simple Parallel |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Classification Accuracy** | **74.3%** | 74.0% | 65.7% | 69.7% |
| **Average Rank (Lower is better)** | **1.73** | 2.43 | 2.86 | 2.98 |
| **Total Benchmark Wins** | **10 Datasets** | 6 Datasets | 6 Datasets | 1 Dataset |
| **Head-to-Head vs ST-GoG** | — | 14 Wins / 8 Losses | 16 Wins / 6 Losses | 20 Wins / 0 Losses |

### Performance Highlights:
* **Motion & Kinematics (Rank #1):** NATOPS (**99.4%**), Libras (**95.6%**), Handwriting (**66.0%**), ArticularyWordRecognition (**99.7%**).
* **Extreme Data Scarcity (Medical):** On *AtrialFibrillation* ($N=15$ training samples), ST-GoG achieves **53.33%** vs. SAGoG's 43.00% (+10.33%), demonstrating the effectiveness of the MixUp + TTA + GoG regularization pipeline.

---

## 🛠️ Configuration Guide

Parameters can be specified in YAML files under `configs/`:

```yaml
# Data settings
dataset: "NATOPS"          # Name of UEA dataset
batch_size: 16             # Batch size
cache_dir: "data/cache"    # Directory for caching preprocessed datasets

# Model architecture
tcn_hidden: 128            # Number of feature maps in ResTCN
gcn_hidden: 128            # GCN hidden dimension
gog_hidden: 128            # GoG hidden dimension
num_segments: null         # null = adaptively determined by sequence length
dropout: 0.42              # Dropout rate
use_se: true               # Enable Squeeze-and-Excitation channel attention
gcn_radii: [1, 3, 5]       # Multi-scale chain adjacency radii
pooling: "mean"            # GoG pooling: 'mean' or 'attention'
strided_downsample: null   # Pre-pooling stride for ultra-long sequences (T > 2000)

# Optimization
epochs: 150                # Total training epochs
lr: 0.001                  # Maximum learning rate (Cosine Annealing)
weight_decay: 0.0001       # Weight decay (AdamW)
mixup_alpha: 0.2           # MixUp Beta distribution parameter
tta_passes: 7              # Passes of Test-Time Augmentation
use_amp: true              # Automatic Mixed Precision
```

---

## 📖 Citation

If you use ST-GoG in your research, please cite:

```bibtex
@mastersthesis{sharma2026stgog,
  author       = {Tushar Sharma},
  title        = {A Spatiotemporal Graph-of-Graphs Framework for Multivariate Time Series Classification},
  school       = {Atal Bihari Vajpayee Indian Institute of Information Technology and Management (ABV-IIITM), Gwalior},
  year         = {2026},
  month        = {May},
  note         = {Supervised by Dr. Anjali}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
