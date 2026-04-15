# V3: Survival HI-VAE

## Purpose

V3 jointly models baseline covariates $X$ and a survival outcome $(t, \delta)$. The survival outcome represents the observed pair from independent competing event and censoring processes.

## Survival Data Notation

| Symbol | Meaning |
|--------|---------|
| $T^*_i$ | Latent event time for patient $i$ |
| $C_i$ | Latent censoring time for patient $i$ |
| $t_i$ | Observed time: $t_i = \min(T^*_i, C_i)$ |
| $\delta_i$ | Event indicator: $\delta_i = \mathbf{1}\{T^*_i \le C_i\}$ |

## Target Distribution

$$P(X, t, \delta) = \sum_s \int p(s)\, p(z \mid s)\, \left[\prod_j p(x_j \mid \theta_j)\right] \cdot p(t, \delta \mid \theta_{\text{surv}}(y, s))\, dz$$

where $s$ is summed over $\{1, \ldots, K\}$ and $z$ is integrated over $\mathbb{R}^{d_z}$.

## Observed-Data Survival Likelihood

Under the independent-competing-risks assumption:

$$p(t, \delta) = [f_T(t) \cdot S_C(t)]^\delta \cdot [f_C(t) \cdot S_T(t)]^{1-\delta}$$

Using the hazard decomposition $f(t) = h(t) S(t)$ and $S(t) = \exp(-H(t))$:

$$\log p(t, \delta) = \delta \cdot \log h_T(t) + (1-\delta) \cdot \log h_C(t) - H_T(t) - H_C(t)$$

## Primary Survival Families

### Weibull (`v3_weibull`)

**Normalization:** Min-max scaling to $[0, 1]$: $t_{\text{scaled}} = (t - t_{\min})/(t_{\max} - t_{\min})$

**Parameters:** Shape $k$ and scale $\lambda$ for event and censoring, each via softplus with clamps $[10^{-3}, 10^3]$:

$$k_T, \lambda_T, k_C, \lambda_C = \text{softplus}(\theta_{\text{raw}})$$

**Theta layer:** Single `Linear(d_y + K, 4, bias=False)` producing all four parameters.

**Hazard and cumulative hazard:**

$$\log h(t; k, \lambda) = \log\!\left(\frac{k}{\lambda}\right) + (k-1)\log\!\left(\frac{t}{\lambda}\right)$$

$$H(t; k, \lambda) = \left(\frac{t}{\lambda}\right)^k$$

Uses `torchsurv.loss.weibull` with log-scale/log-shape parameterization.

**Sampling:**

$$U \sim \text{Uniform}(0,1), \quad T^*_{\text{scaled}} = \lambda_T (-\log U)^{1/k_T}$$

$$V \sim \text{Uniform}(0,1), \quad C_{\text{scaled}} = \lambda_C (-\log V)^{1/k_C}$$

$$t = \min(T^*, C) \cdot (t_{\max} - t_{\min}) + t_{\min}, \qquad \delta = \mathbf{1}\{T^* \le C\}$$

### Piecewise-Constant (`v3_piecewise`)

**Normalization:** Min-max scaling to $[0, 1]$.

**Intervals:** $K$ equal-width intervals via `linspace`, plus one boundary interval beyond the data range.

**Parameters:** Per-interval probability masses via softmax:

$$\pi_T = \text{softmax}(\theta_T), \qquad \pi_C = \text{softmax}(\theta_C)$$

**Theta layers:** Either 1-layer `Linear(d_y + K, n_intervals)` or 2-layer MLP (`d_y + K → 20 → n_intervals` with ReLU).

**Density and survival:**

$$f_T(t) = \frac{\pi_{T,k}}{a_k - a_{k-1}}, \qquad S_T(t) = \text{linear interpolation of } 1 - \text{CDF}_T$$

**Log-likelihood:** Uses linear interpolation of survival function within bins.

**Sampling:**

1. $k \sim \text{Categorical}(\pi)$ — select interval by mass
2. $T^*_{\text{scaled}} = a_k + U \cdot (a_{k+1} - a_k)$ — uniform within interval

## Secondary / Experimental Survival Heads

### `surv` (Log-Normal) — *not part of mainline*

Available via `data_types.csv` but **not** via `--model_version`.

**Known limitation:** Log-likelihood operates on log1p-transformed scale without proper Jacobian correction. Training objective is not a correct log-density on the original time scale.

### `surv_loglog` (Log-Logistic) — *not part of mainline*

Available via `data_types.csv` but **not** via `--model_version`.

**Known inconsistencies:**
1. Censoring likelihood uses log-logistic but sampling uses Weibull inverse CDF
2. Censoring shape parameter offset mismatch between likelihood and sampling

## ELBO

$$\text{ELBO}_{v3} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \log p(t, \delta \mid \theta_{\text{surv}})\right] - D_{\text{KL},s} - D_{\text{KL},z}$$

## Implementation

- **Model class:** Same `HIVAE` base; survival is handled entirely through feature-type dispatch
- **Version argument:** `--model_version v3_weibull` or `--model_version v3_piecewise`
- The survival feature is specified in `data_types.csv` with type `surv_weibull` or `surv_piecewise`
- `run_unified.py` maps `v3_weibull` → `surv_type='surv_weibull'` which adjusts `data_processing.read_data`

## Conditional Generation

Filter generated samples by condition (e.g., treatment = 0) and repeat until the desired count is reached. Implemented in `generate_from_condition_HIVAE()`.

## Limitations

- No longitudinal modelling
- Survival head shares the same theta-layer pattern as other features (observed/missing split with no_grad)
- Per-batch min-max normalization for survival times
- Secondary heads have known mathematical issues

---

*Last updated: 2026-04-15.*
