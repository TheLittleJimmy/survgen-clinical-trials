# Multi-Version HI-VAE: Summary of All Model Versions

## Shared Latent Backbone

All versions share:

- **Encoder:** $x^{obs} \to q(s \mid x^{obs}),\; q(z \mid s, x^{obs})$
- **Prior:** $p(s) = 1/K$ (uniform), $\; p(z \mid s) = \mathcal{N}(\mu_s, I)$
- **Decoder:** $z \to y = W_y z \to \theta_j(y_j, s) \to p(x_j \mid \theta_j)$

Two encoder variants exist: `HIVAE_inputDropout` (default) and `HIVAE_factorized`. Both share the same decoder and loss.

## Version Comparison

| Version | Target Distribution | Key Addition | Primary Use Case |
|---------|-------------------|--------------|-----------------|
| **V0** | $P(X_{\text{baseline}})$ | None | Static baseline cohort generation |
| **V1** | $P(X_{\text{baseline}}, U)$ | Endpoint as additional `real` feature | Joint baseline-endpoint synthesis |
| **V2A** | $P(X_{\text{baseline}}, Y_{1:T})$ | Longitudinal summary encoder + time-conditioned Gaussian decoder | Longitudinal trajectory generation |
| **V3** | $P(X, t, \delta)$ | Survival likelihood heads (Weibull or piecewise-constant) | Joint baseline-survival synthesis |
| **V4_joint** | $P(X, Y_{1:T}, t, \delta)$ | Combines V2A longitudinal + V3 survival; all share $(s,z)$ | Joint three-modality generation |
| **V4_seq** | $P(X, Y_{1:T}, t, \delta)$ | Sequential conditional structure with summary conditioning | Leakage-free sequential generation |

## V0: Static Baseline HI-VAE

**Purpose:** Generate synthetic baseline/pre-randomization cohorts.

**Distribution:** $P(X) = \sum_s \int p(s)\, p(z|s)\, \prod_j p(x_j | \theta_j(y(z), s))\, dz$

**Decoder:** Standard feature-wise decoders (real, pos, count, cat, ordinal).

**Limitation:** No temporal, longitudinal, or survival modelling.

See [model_architecture_v0.md](model_architecture_v0.md) for full details.

## V1: Baseline + Continuous Endpoint

**Purpose:** Jointly model baseline variables and a single continuous post-randomization endpoint.

**Distribution:** $P(X, U) = \sum_s \int p(s)\, p(z|s)\, \prod_j p(x_j | \theta_j) \cdot p(U | \theta_U(y, s))\, dz$

**Decoder:** Endpoint $U$ added as an additional `real` feature. No architectural changes needed.

**Limitation:** Single scalar endpoint only; no trajectory or survival modelling.

See [model_architecture_v1.md](model_architecture_v1.md) for full details.

## V2A: Longitudinal HI-VAE

**Purpose:** Model baseline variables plus repeated continuous measurements over time.

**Distribution:** $P(X, Y_{1:n_i}) = \sum_s \int p(s)\, p(z|s)\, \prod_j p(x_j | \theta_j) \cdot \prod_v p(y_{iv} | z, s, t_{iv})\, dz$

**Key additions:**
- Time embedding network $e(t)$ (2-layer MLP with SiLU)
- Time-conditioned Gaussian decoder: $\mu_{iv} = W_\mu [z; s; e(t_{iv})]$
- Longitudinal summary encoder for posterior augmentation
- Supports baseline-conditioned and history-conditioned generation

**Limitation:** No survival modelling. Trajectory variance depends only on $(s, t)$, not $z$.

See [model_architecture_v2a.md](model_architecture_v2a.md) for full details.

## V3: Survival HI-VAE

**Purpose:** Jointly model baseline covariates and a survival outcome $(t, \delta)$.

**Distribution:** $P(X, t, \delta) = \sum_s \int p(s)\, p(z|s)\, \prod_j p(x_j | \theta_j) \cdot p(t, \delta | \theta_{\text{surv}}(y, s))\, dz$

**Primary survival families:**
- **Weibull** (`v3_weibull`): shape/scale parameterization via `torchsurv`
- **Piecewise-constant** (`v3_piecewise`): softmax probability masses over intervals

**Secondary/experimental heads:** `surv` (log-normal), `surv_loglog` (log-logistic) — available in code but not via `--model_version`. Known limitations documented in the full V3 doc.

**Limitation:** No longitudinal modelling.

See [model_architecture_v3.md](model_architecture_v3.md) for full details.

## V4: Joint and Sequential Multi-Modal Models

**Purpose:** Jointly model baseline, longitudinal, and survival data.

Two sub-versions:

### V4_joint
Minimal joint extension of V2A + V3. All three modalities share the same $(s, z)$.

$$P(X, Y_{1:T}, t, \delta) = \sum_s \int p(s)\, p(z|s)\, p(X|z,s)\, p(Y_{1:T}|z,s)\, p(t,\delta|z,s)\, dz$$

### V4_seq
Sequential conditional model with summary-conditioned generation.

$$P(X, Y_{1:T}, t, \delta) = \sum_s \int p(s)\, p(z|s)\, p(X|z,s)\, p(Y_{1:T}|c_X,z,s)\, p(t,\delta|c_X,r_Y,z,s)\, dz$$

Generation order: $s,z \to X \to \text{planned } Y \text{ grid} \to r_Y \to (t,\delta) \to \text{truncate } Y$

See [model_architecture_v4.md](model_architecture_v4.md) for full details including the information-leakage avoidance design.

## Known Implementation Limitations (All Versions)

1. **Per-batch normalization:** Baseline feature normalization is computed per-batch from observed values only. Statistics vary across batches and between training/generation.
2. **Missing-data no_grad:** The theta layers use `torch.no_grad()` for missing-data forward passes, preventing gradient flow through those entries. This is a legacy design inherited from the original HI-VAE codebase.
3. **V2A longitudinal normalization** is computed once globally (not per-batch), which is more principled.

---

*This document provides a concise overview. For full mathematical formulations, see the version-specific documents. Last updated: 2026-04-15.*
