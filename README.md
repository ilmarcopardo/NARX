# narxpy: PyTorch NARX Implementation

A simple, reusable PyTorch implementation of a Nonlinear Autoregressive Network with Exogenous Inputs (NARX).

## Features

*   Supports configurable input/output delays (`d_i`, `d_o`).
*   Handles multi-dimensional exogenous (`x`) and endogenous (`y`) time series.
*   Configurable hidden layer size and output activation function.
*   Supports different simulation modes:
    *   **Close Loop (Parallel):** Uses own predictions for feedback (standard simulation/forecasting).
    *   **Open Loop (Series-Parallel):** Uses provided true values for feedback (teacher forcing, common for training).
*   Includes optional bootstrapping for initializing close-loop simulations.

## Installation

```pip install narxpy```

## Usage
```
import torch
from narxpy import NARX

model = NARX(d_i=d_i, d_o=d_o, d_x=d_x, d_y=d_y, d_hl=d_hl, act_func=act_func)

# --- Run Modes ---

# 1. Close Loop (Simulation - Default)
y_pred_close = model(x_data, mode="close")
print(f"Close loop output shape: {y_pred_close.shape}")

# 2. Open Loop (Teacher Forcing)
y_pred_open = model(x_data, mode="open", y=y_true)
print(f"Open loop output shape: {y_pred_open.shape}")

# 3. Close Loop with Bootstrap
y_pred_bootstrap = model(x_data, mode="close", y=y_true, bootstrap=bootstrap_steps)
print(f"Bootstrap output shape: {y_pred_bootstrap.shape}")
```
