# 🌌 Relational Calculus: The Core Module

Welcome to the mathematical engine of the Relational Calculus framework. 

This directory contains the primary implementation of the dimensionless loss functions used across the entire repository. By anchoring neural learning to a system's **"North Star"** (its intrinsic capacity), we mathematically delete environmental entropy, stabilize gradients, and enable zero-shot scale transfer.

## 📖 The Foundation

At the heart of this module is the **[WHITE_PAPER.md](./WHITE_PAPER.md)**. 
This document formalizes the shift from "Continuous Calculus" (exploring landscapes point-by-point) to **"Relational Calculus"** (revealing the landscape's underlying blueprint). It proves that the true content of a physical or logical law resides entirely in its dimensionless form.

### The Contrast
*   **The Absolute Trap**: Traditional MSE and Cross-Entropy loss functions force models to memorize arbitrary human units (dollars, Newtons, 255 RGB). This creates ill-conditioned, deformed loss landscapes that require heavy optimizers (Adam) and fail during data drift.
*   **The Relational Fix**: We express the target as a ratio $r = \text{Actual} / \text{Capacity}$. This makes the loss landscape perfectly spherical, allowing models to converge exponentially faster with pure, un-tuned SGD.

## 🗂️ Core Implementation

### `losses.py`
This file contains the drop-in PyTorch replacements for standard loss functions:

1.  **`RelationalMSELoss`**: 
    *   **The Problem**: Exploding gradients when targets have massive absolute ranges.
    *   **The Fix**: Normalizes the target by the dynamic `capacity` before computing the mean squared error.
    *   **Usage**: Ideal for physics simulations, finance, and continuous control.

2.  **`RelationalCrossEntropyLoss`**:
    *   **The Problem**: Scale-locking in classification and probability estimation.
    *   **The Fix**: Uses Binary Cross Entropy with Logits applied to relational targets bounded in `[0, 1]`.

## 🚀 How to Use

To integrate Relational Calculus into your own PyTorch models, simply import the losses:

```python
import torch
from relational_calculus.losses import RelationalMSELoss

# 1. Initialize the loss
criterion = RelationalMSELoss()

# 2. Inside your training loop
# pred_ratio: model output (should be bound to [0,1] e.g. via Sigmoid)
# target_absolute: raw ground truth (e.g. 500 Newtons)
# capacity: the local North Star (e.g. 1000 Newtons max capacity)
loss = criterion(pred_ratio, target_absolute, capacity)

loss.backward()
```

### Dependencies
*   `torch >= 2.0`
*   `numpy`

---
*For the complete mathematical derivation of these loss functions, refer to the [WHITE_PAPER.md](./WHITE_PAPER.md) included in this directory.*
