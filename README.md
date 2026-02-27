# ECFR: Entropy-Consistency Flow Rectification in Test-Time Adaptation for Medical Foundation Models

This repository contains the official PyTorch implementation of the **Entropy-Consistency Flow Rectification (ECFR)** strategy, alongside our classification-adapted version of the Gradient Alignment (GraTa) baseline, submitted to MICCAI 2026.

---

## 📌 Overview

Deploying medical Foundation Models (FMs) is often severely compromised by continuous clinical distribution shifts. Existing Test-Time Adaptation (TTA) methods typically rely on uni-directional optimization, which inevitably precipitates error accumulation and catastrophic model collapse when dealing with flawed feature representations.

ECFR fundamentally redefines TTA as a **spatial routing problem**. 

By introducing the **Entropy-Consistency Quadrants** and a **Quadrant-wise Flow Rectification** mechanism, ECFR dynamically disentangles streaming data into distinct diagnostic states. It safely consolidates high-fidelity predictions while actively penalizing and isolating high-risk samples. Coupled with a dynamic memory bank for real-time threshold calibration, this theoretically rigorous approach effectively halts the propagation of erroneous gradients and ensures safe clinical adaptation.

---

## 📂 Repository Structure

```text
ECFR/
├── datasets/                 
│   ├── augmentations.py      # Asymmetric augmentation pipelines (Clean/Weak/Strong)
│   └── build_dataset.py      # Standardized streaming dataloader mechanics
├── methods/                  
│   ├── ecfr.py               # [Core] Quadrant-wise Flow Rectification and spatial routing logic
│   └── grata_adapted.py      # [Baseline] GraTa dynamically adapted for classification
├── models/                   
│   └── memory_bank.py        # Dynamic threshold queue (Capacity=128)
├── utils/                    
│   ├── metrics.py            # Brier Score, ECE, Accuracy, entropy and JS Divergence
│   └── visualization.py      # Advanced hybrid-scale quadrant flow visualization
├── main_adaptation.py        # Central execution and online streaming loop
└── run_grid_search.sh        # Oracle quasi-logarithmic grid search protocol
```

## 🚀 Quick Start

### 1. Environment Setup
Dependencies required for execution and visualization:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm
```

### 2. Single Run: ECFR
To execute the ECFR streaming adaptation protocol on a specific test dataset with a fixed learning rate:
```bash
python main_adaptation.py --dataset_csv path/to/dataset.csv --method ecfr --lr 1e-4 --capacity 128 --quantile 0.5
```

### 3. Single Run: GraTa (Classification Adaptation)
To execute the gradient alignment baseline:
```bash
python main_adaptation.py --dataset_csv path/to/dataset.csv --method grata --lr 1e-4
```

### 4. Oracle Grid Search (Fair Comparison Protocol)
To ensure a rigorous and fair comparison of the upper-bound potentials across all methods, we provide the automated quasi-logarithmic grid search script utilized in our experiments.
The search space spans {1, 3.3, 6.6} × {1e-5, 1e-4} ∪ {1e-3}.

Run the fully automated search via:

```bash
bash run_grid_search.sh
```
Note: This shell script sequentially iterates through the predefined learning rates, passing them via the --lr argument to main_adaptation.py. The best-performing hyperparameter (Oracle selection) determines the final reported metrics.

### 🔍 Core Mechanism Reference
Reviewers looking to verify the mathematical implementation of our Quadrant-wise Flow Regulation should refer directly to methods/ecfr.py, specifically the forward_and_adapt function where the flow rectification strategy is executed based on dynamic spatial partitioning. The dynamic thresholding logic can be strictly verified in models/memory_bank.py.