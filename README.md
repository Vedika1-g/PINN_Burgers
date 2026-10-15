# Physics-Informed Neural Networks for the Viscous Burgers' Equation

A from-scratch PyTorch reproduction of **Raissi, Perdikaris & Karniadakis (2019)**, *"Physics-Informed Neural Networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs"* (J. Computational Physics), applied to the 1-D viscous Burgers' equation.

**Roll No.** 24CS3060 · **Paper DOI:** [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

---

## Overview

We train a neural network to satisfy a PDE by embedding the equation's residual in the loss, then use the same framework for two tasks:

| Task | Goal | Result |
|---|---|---|
| **Forward** | given ν, find u(x,t) | **0.56%** rel. L₂ vs exact |
| **Inverse** | given sparse data, recover ν | **2.30%** error (ν = 0.003256 vs 0.003183) |

The equation solved:

```
u_t + u·u_x = ν·u_xx ,   x ∈ [-1, 1],  t ∈ [0, 1]
u(x,0) = -sin(πx) ,   u(-1,t) = u(1,t) = 0 ,   ν = 0.01/π ≈ 0.003183
```

The small viscosity produces a steep interior front (near-shock) at x = 0, which makes the benchmark a meaningful accuracy test.

![Forward solution](figures/forward_solution.png)

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── PINN_Burgers.ipynb        # runnable notebook (Colab T4 / CPU)
├── figures/                  # result plots (PNG)
│   ├── forward_training_loss.png
│   ├── forward_solution.png
│   ├── pinn_vs_exact_vs_fd.png
│   ├── inverse_training.png
│   └── ablation_studies.png
└── report/
    └── PINN_Burgers_Report.pdf   # 4-page IEEE two-column report
```

---

## Setup

### Option A — Google Colab (recommended)

1. Upload `PINN_Burgers.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all.** PyTorch is pre-installed; no setup needed.

Full run on a T4 takes **~25–30 minutes** (forward + inverse + both ablations).

### Option B — Local (VS Code / Jupyter)

Requires Python ≥ 3.9.

```bash
pip install -r requirements.txt
jupyter notebook PINN_Burgers.ipynb      # or open in VS Code
```

On CPU the notebook auto-detects the absence of a GPU and runs the **ablation** section with reduced settings so it finishes in reasonable time; the **forward and inverse** problems still run at full 15,000 epochs. Colab is strongly preferred.

---

## How to run

Run the cells top to bottom. The notebook is organized into nine sections:

1. **Setup & config** — imports, seeds, domain, output folders.
2. **Exact solution** — Cole–Hopf via Gauss–Hermite quadrature (the ground truth).
3. **Network** — the `PINN` class and the `burgers_residual` core module.
4. **Forward training** — Adam (15k epochs) → L-BFGS (100 steps).
5. **Evaluation & FD comparison** — contour plots, time slices, error metrics.
6. **Inverse problem** — trainable ν, data from the exact solution.
7. **Ablations** — collocation density and optimizer choice.
8. **Paper comparison** — settings table.
9. **Results summary** — prints the table and writes `pinn_outputs/results/results.json`.

### Reproducibility (seeds)

All randomness is seeded:

- Global seed **42** for Python `random`, NumPy, and PyTorch (CPU + CUDA) in the setup cell.
- Collocation points are **resampled every 2,000 epochs** (deterministically, under the fixed seed).
- Inverse-problem observations are sampled with `np.random.default_rng(42)`.
- Ablations run over seeds **{42, 123, 2026}** (GPU) or **{42}** (CPU) and report mean ± std.

Running all cells reproduces every number in the tables below.

---

## Results

### Forward problem

Training loss (Adam → L-BFGS):

![Training loss](figures/forward_training_loss.png)

Solution field vs finite-difference reference and absolute error:

![Forward solution comparison](figures/pinn_vs_exact_vs_fd.png)

| Metric | Value |
|---|---|
| PINN vs Exact, rel L₂ | **0.56%** |
| FD(800) vs Exact, rel L₂ | 1.17% |
| PINN vs FD(800), rel L₂ | 1.28% |
| MAE vs Exact | 6.5 × 10⁻⁴ |
| RMSE vs Exact | 3.4 × 10⁻³ |

> The PINN is closer to the exact solution than the finite-difference reference is; the larger PINN-vs-FD number reflects the reference's own 1.17% error.

### Inverse problem

![Inverse training](figures/inverse_training.png)

| Quantity | Value |
|---|---|
| True ν | 0.003183 |
| Estimated ν | 0.003256 |
| **ν error** | **2.30%** |
| Field rel. L₂ vs Exact | 0.14% |
| Effective ν of FD data | 0.004838 |

> Observations are drawn from the **exact** solution. Using finite-difference data instead biases the recovered ν to ≈0.0048 (a ~50% error) because upwind differencing adds numerical diffusion.

### Ablation studies

![Ablations](figures/ablation_studies.png)

**A — collocation density** (mean ± std, 3 seeds):

| N_f | Relative L₂ |
|---|---|
| 1,000 | 13.24% ± 6.32% |
| 3,000 | 1.54% ± 1.56% |
| 10,000 | **0.15% ± 0.02%** |

**B — optimizer** (mean ± std, 3 seeds):

| Configuration | Relative L₂ |
|---|---|
| Adam only | 2.43% ± 1.10% |
| Adam + L-BFGS | **0.15% ± 0.02%** |

### Comparison with the paper

| Setting | Paper | Ours |
|---|---|---|
| Architecture | 8 × 20 | 4 × 64 |
| IC/BC weights | 1 (uniform) | 20 |
| IC/BC points | 100 each | 500 each |
| Collocation | 10,000 | 10,000 |
| Optimizer | L-BFGS only | Adam + L-BFGS |
| Reference / inverse data | Exact (Cole–Hopf) | Exact (Cole–Hopf) |
| Forward rel. L₂ | 6.7 × 10⁻⁴ | 5.6 × 10⁻³ |

Our forward error is about an order of magnitude above the paper's, attributable to a shallower/wider network and fewer L-BFGS iterations. The method is faithfully reproduced across the forward, inverse, and ablation axes.

---

## Method summary

- **Network:** 4 hidden layers × 64 units, `tanh` activation, Xavier init (~12.7k parameters). `tanh` gives a continuous second derivative, required by the residual.
- **Residual:** `f = u_t + u·u_x − ν·u_xx`, with all derivatives from **automatic differentiation** (exact, not finite differences).
- **Loss:** `w_pde·MSE(f) + w_ic·MSE(u − u₀) + w_bc·MSE(u)` with `w_ic = w_bc = 20` to anchor the known conditions first.
- **Optimizer:** Adam (15k epochs, cosine LR 1e-3 → 1e-5) then L-BFGS (strong-Wolfe) to polish.
- **Inverse:** ν stored as `log ν` (keeps it positive) and trained jointly with the weights, with an added data loss over observations from the exact solution.
- **Exact reference:** Cole–Hopf transform → heat equation → Gauss–Hermite quadrature, with log-sum-exp stabilization to avoid overflow at small ν.

A detailed report is in [`report/PINN_Burgers_Report.pdf`](report/PINN_Burgers_Report.pdf).

---

## Known limitations

- Forward accuracy (0.56%) is below the paper's 6.7 × 10⁻⁴ due to the shallower network and fewer L-BFGS steps.
- The finite-difference reference itself carries ~1.17% error; it is used only for the required FD comparison, not as inverse-problem data.

---

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks.* J. Comput. Phys. 378 (2019) 686–707.
2. J. D. Cole. *On a quasi-linear parabolic equation occurring in aerodynamics.* Quart. Appl. Math. 9 (1951) 225–236.
3. K. Hornik, M. Stinchcombe, H. White. *Universal approximation of an unknown mapping and its derivatives.* Neural Networks 3 (1990) 551–560.
4. D. P. Kingma, J. Ba. *Adam: A method for stochastic optimization.* ICLR 2015.
5. D. C. Liu, J. Nocedal. *On the limited memory BFGS method for large scale optimization.* Math. Program. 45 (1989) 503–528.

---

## License

Academic coursework (Roll No. 24CS3060). Reproduction of the referenced paper for educational purposes.
