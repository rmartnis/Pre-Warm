# Pre-Warm: Input-Conditioned Weight Initialization for Convolutional Neural Networks

Code for the paper **Pre-Warm: Input-Conditioned Weight Initialization for Convolutional Neural Networks** (Martinshn, 2026), submitted to *Pattern Recognition Letters*.

Pre-Warm is a zero-training-cost method that conditions the first convolutional layer's weights on a single batch of training images before any gradient step is taken. It extracts mean-centered local patches, clusters them with MiniBatchKMeans, applies inverse Manhattan spatial weighting, and seeds half of the first-layer filters with the resulting centroids. The remaining filters retain Kaiming initialization.

ArXiv: https://arxiv.org/abs/2606.25256

---

## Repository contents

| Notebook | Description |
|---|---|
| `PreWarm-Fashion.ipynb` | Grid search on Fashion-MNIST establishing the grayscale hyperparameter rules; proof-of-concept multi-seed experiment |
| `PreWarmExp_Acc_MNIST.ipynb` | Paired accuracy experiment on MNIST and Fashion-MNIST (8 seeds, 5000 steps, one-sided paired t-test) |
| `PreWarmExp_Acc_CIFAR10.ipynb` | Paired accuracy experiment on CIFAR-10 (8 seeds, LR 3e-3→3e-5) |
| `PreWarmExp_Acc_SVNH.ipynb` | Paired accuracy experiment on SVHN (8 seeds, LR 5e-4→3e-6) |
| `PreWarmExp_Acc_CIFAR100.ipynb` | Paired accuracy experiment on CIFAR-100 (8 seeds, LR 2e-3→3e-5) |
| `PreWarm_ColorFormula.ipynb` | Derivation of the color dataset n_patches rule from mean patch L2 norm; validates predictions against grid-search optima |

---

## Requirements

```
torch
torchvision
scikit-learn
scipy
numpy
datasets        # for CIFAR-100 and SVHN via HuggingFace
tensorflow      # used in CIFAR-10 and SVHN notebooks for data loading
matplotlib
```

Install with:

```bash
pip install torch torchvision scikit-learn scipy numpy datasets tensorflow matplotlib
```

All experiments were run on a single NVIDIA T4 GPU (Google Colab).

---

## Reproducing the results

Each notebook is self-contained. Run cells top to bottom. The 8 random seeds used throughout are `{42, 7, 13, 99, 2025, 1, 8, 21}`. Pre-Warm and Kaiming runs are exactly paired at each seed.

Key hyperparameters per dataset:

| Dataset | n_patches | scale (σ) | LR start → end |
|---|---|---|---|
| MNIST | 58 | 0.20 | 3e-3 → 3e-5 |
| Fashion-MNIST | 22 | 0.20 | 3e-3 → 3e-5 |
| CIFAR-10 | 28 | 0.25 | 3e-3 → 3e-5 |
| SVHN | 16 | 0.25 | 5e-4 → 3e-6 |
| CIFAR-100 | 16 | 0.25 | 2e-3 → 3e-5 |

All datasets share `F=32`, `K=3`, `n_clusters=16`, batch size 256, 5000 training steps.
