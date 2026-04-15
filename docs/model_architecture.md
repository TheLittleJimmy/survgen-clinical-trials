# Multi-Version HI-VAE: Model Architecture Documentation

## Index

This document serves as the landing page for the Multi-Version HI-VAE architecture documentation. The model is documented across the following files:

| Document | Description |
|----------|-------------|
| [model_architecture_summary.md](model_architecture_summary.md) | Concise overview of all version groups (V0–V4) |
| [model_architecture_v0.md](model_architecture_v0.md) | V0: Static baseline HI-VAE |
| [model_architecture_v1.md](model_architecture_v1.md) | V1: Baseline + continuous endpoint |
| [model_architecture_v2a.md](model_architecture_v2a.md) | V2A: Lightweight longitudinal HI-VAE |
| [model_architecture_v3.md](model_architecture_v3.md) | V3: Survival HI-VAE (Weibull + piecewise-constant) |
| [model_architecture_v4.md](model_architecture_v4.md) | V4: Joint and sequential multi-modal models |

## Shared Components

All versions share the same latent backbone:

- **Discrete latent:** $s \in \{1, \ldots, K\}$, with uniform prior $p(s) = 1/K$
- **Continuous latent:** $z \in \mathbb{R}^{d_z}$, with Gaussian mixture prior $p(z \mid s) = \mathcal{N}(\mu_s, I)$
- **Deterministic transform:** $y = W_y z$, partitioned into per-feature slices
- **Feature-wise decoders:** $\theta_j = f_j([y^{(j)}; s])$, mapping to type-specific distributional parameters

## Version Lineage

```
V0 (baseline)
├── V1 (baseline + endpoint)
├── V2A (baseline + longitudinal)
│   └── V4_joint (baseline + longitudinal + survival)
│       └── V4_seq (sequential conditional variant)
└── V3 (baseline + survival)
    ├── V3_weibull
    └── V3_piecewise
```

## Quick Start

Select a model version via `--model_version`:
- `v0`, `v1`, `v2a`, `v3_weibull`, `v3_piecewise` (existing)
- `v4_joint`, `v4_seq` (new)

See `execute/run_unified.py` for CLI usage and `utils/src.py` for the model class.

---

*Last updated: 2026-04-15.*
