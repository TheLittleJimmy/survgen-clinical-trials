# V1: Baseline + Continuous Endpoint HI-VAE

## Purpose

V1 jointly models baseline/pre-randomization variables $X_{\text{pre}}$ and a single continuous post-randomization endpoint $U$. Because both are generated from the same latent $(s, z)$, the model implicitly captures $P(U \mid X_{\text{pre}})$ through the shared latent structure.

## Target Distribution

$$P(X_{\text{pre}}, U) = \sum_s \int p(s)\, p(z \mid s)\, \left[\prod_j p(x_j \mid \theta_j)\right] \cdot p(U \mid \theta_U(y, s))\, dz$$

where $s$ is summed over $\{1, \ldots, K\}$ and $z$ is integrated over $\mathbb{R}^{d_z}$.

## Key Design

The endpoint $U$ is modelled as an additional `real`-type feature in the data specification. **No architectural changes** are needed beyond data preparation—the endpoint is simply another column in the data CSV with type `real` in `data_types.csv`.

$$U \mid z, s \sim \mathcal{N}\!\left(\mu_U(y^{(U)}, s),\; \sigma^2_U(s)\right)$$

where $\mu_U$ and $\sigma^2_U$ follow the same theta-layer pattern as other real features:

$$\mu_U = W_\mu [y^{(U)}; s], \qquad \sigma^2_U = \text{softplus}(W_\sigma s)$$

## ELBO

$$\text{ELBO}_{v1} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \log p(U \mid \theta_U) \cdot m_U\right] - D_{\text{KL},s} - D_{\text{KL},z}$$

## Encoder, Prior, Decoder

Identical to V0. The encoder sees the full input (baseline + endpoint), and the decoder has one additional `real`-type theta layer for $U$.

## Sampling

Same as V0. The endpoint $U$ is sampled alongside baseline features from the same $\theta$ structure.

## Implementation

- **Model class:** Same as V0 (`HIVAE_inputDropout`)
- **Version argument:** `--model_version v1`
- **Endpoint column:** Specified via `--endpoint_column` for documentation only; the actual feature type is set in `data_types.csv`

## Use Case

- Joint baseline-endpoint synthesis
- Conditional endpoint imputation given baseline structure via the shared latent
- Counterfactual-style generation (sample baseline, then sample endpoint from the same latent)

## Limitations

- Single scalar endpoint only
- No trajectory or survival modelling
- All V0 limitations apply (per-batch normalization, missing-data no_grad)

---

*Last updated: 2026-04-15.*
