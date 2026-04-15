# V2A: Lightweight Longitudinal HI-VAE

## Purpose

V2A extends the HI-VAE to repeated continuous measurements over time per patient. It maintains a single patient-level latent $(s, z)$ shared across all visits, with trajectory shape captured through a time-conditioned decoder rather than autoregressive recurrence.

## Target Distribution

$$P(X_{\text{pre}}, Y_{1:n_i}) = \sum_s \int p(s)\, p(z \mid s)\, \left[\prod_j p(x_j \mid \theta_j)\right] \cdot \left[\prod_{v=1}^{n_i} p(y_{iv} \mid z, s, t_{iv})\right]\, dz$$

where $s$ is summed over $\{1, \ldots, K\}$ and $z$ is integrated over $\mathbb{R}^{d_z}$.

## Key Design Choices

1. **One patient-level latent $(s, z)$** shared across all visits
2. **Time-conditioned decoder** — trajectory shape from time embedding, not RNN/Transformer
3. **Longitudinal summary encoder** — augments the posterior with observed visit history
4. **Variance depends on $(s, t)$ only** — intentional simplification for stability on small datasets

## Additional Modules

### Time Embedding

$$e(t) = W_2 \cdot \text{SiLU}(W_1 \cdot t + b_1) + b_2 \qquad (W_1 \in \mathbb{R}^{16 \times 1},\; W_2 \in \mathbb{R}^{16 \times 16})$$

Shared between the summary encoder and the longitudinal decoder.

### Longitudinal Summary Encoder

Each observed visit $(y_{iv}, t_{iv})$ is encoded and mean-pooled:

$$h_{iv} = \text{MLP}_{\text{summary}}([y_{iv}; e(t_{iv})]) \qquad (\text{output dim} = 16)$$

$$r_i = \frac{1}{n_i} \sum_{v: m_{iv}=1} h_{iv} \qquad \text{(masked mean pooling)}$$

The encoder operates on the augmented input $[X^{obs}; r_i]$:

$$q(s \mid X^{obs}, r_i), \qquad q(z \mid s, X^{obs}, r_i)$$

When no longitudinal data is available, $r_i = \mathbf{0}$.

**Summary network architecture:**

| Layer | Input dim | Output dim | Activation |
|-------|-----------|------------|------------|
| `long_summary_net` layer 1 | $D_{\text{out}} + 16$ | 16 | ReLU |
| `long_summary_net` layer 2 | 16 | 16 | none |

### Longitudinal Decoder (Time-Conditioned Gaussian)

$$\mu_{iv} = W_\mu [z_i; s_i; e(t_{iv})] \qquad (W_\mu \in \mathbb{R}^{D_{\text{out}} \times (d_z + K + 16)})$$

$$\sigma^2_{iv} = \text{softplus}(W_\sigma [s_i; e(t_{iv})]), \quad \text{clamped to } [10^{-3}, 10^3]$$

$$y_{iv} \mid z_i, s_i, t_{iv} \sim \mathcal{N}(\mu_{iv}, \sigma^2_{iv})$$

**Variance parameterization:** The mean $\mu$ depends on $(z, s, \text{time})$ while the variance depends on $(s, \text{time})$ only. This mirrors the baseline HI-VAE pattern. Patient-specific trajectory *level* is captured by $z$ through $\mu$; *noise structure* is determined by the mixture component $s$ and time.

### Augmented Encoder Layers

When `model_version='v2a'`, encoder layers are re-created with augmented input:

| Layer | Input dim | Output dim |
|-------|-----------|------------|
| `s_layer` | $D + 16$ | $K$ |
| `z_layer` | $D + 16 + K$ | $2 d_z$ |

## Longitudinal Log-Likelihood

$$\mathcal{L}_{\text{long},i} = \sum_{v=1}^{n_i} m_{iv} \left[-\frac{1}{2}\log(\sigma^2_{iv}) - \frac{(y_{iv} - \mu_{iv})^2}{2\sigma^2_{iv}} - \frac{1}{2}\log(2\pi)\right]$$

## ELBO

$$\text{ELBO}_{v2a} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \mathcal{L}_{\text{long},i}\right] - D_{\text{KL},s} - D_{\text{KL},z}$$

## Data Format

Longitudinal data stored as padded tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `times` | $(N, V_{\max})$ | Normalised to $[0, 1]$ |
| `values` | $(N, V_{\max})$ or $(N, V_{\max}, D)$ | Zero-mean, unit-variance |
| `masks` | $(N, V_{\max})$ | Binary ($1$ = observed visit) |

Normalisation is computed **globally** from all observed values (not per-batch).

## Generation Modes

### Baseline-Conditioned Generation

$$r_i = \mathbf{0}, \qquad q(s, z \mid X^{obs})$$

For each time $t_k$ in grid: sample $y_k \sim \mathcal{N}(\mu_k, \sigma^2_k)$.

### History-Conditioned Completion

$$r_i = \text{mean-pool}(\text{MLP}_{\text{summary}}([y_{iv}; e(t_{iv})]) \text{ for observed } v)$$

Encoder conditions on $[X^{obs}; r_i]$, then generates on future time grid.

### Prior Generation

Sample $s, z$ from prior, then generate trajectories on the time grid.

## Implementation

- **Model class:** `HIVAE` base with V2A-specific init block
- **Version argument:** `--model_version v2a`
- **CLI args:** `--longitudinal_file`, `--patient_id_col`, `--time_col`, `--longitudinal_value_col`

## Limitations

- No survival modelling
- Trajectory variance does not depend on $z$ (intentional simplification)
- All V0 limitations apply for baseline features
- No autoregressive structure — cannot model visit-to-visit dependencies

---

*Last updated: 2026-04-15.*
