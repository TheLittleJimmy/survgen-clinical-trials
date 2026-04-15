# Multi-Version HI-VAE: Model Architecture and Mathematical Formulation

## Table of Contents

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [Shared Latent Backbone](#3-shared-latent-backbone)
4. [Feature-Wise Decoder (Theta Layers)](#4-feature-wise-decoder-theta-layers)
5. [Observation Likelihoods](#5-observation-likelihoods)
6. [Loss Function (ELBO)](#6-loss-function-elbo)
7. [Version-Specific Formulations](#7-version-specific-formulations)
8. [Data Normalization](#8-data-normalization)
9. [Sampling Procedures](#9-sampling-procedures)
10. [Training Details](#10-training-details)
11. [Network Dimensions Summary](#11-network-dimensions-summary)

---

## 1. Overview

The Multi-Version Heterogeneous Incomplete VAE (HI-VAE) is a unified generative framework for clinical trial data synthesis. It learns a shared latent representation of patients and reconstructs heterogeneous observed variables through type-specific decoders.

The codebase provides **two encoder variants**: `HIVAE_inputDropout` (default, used in production) and `HIVAE_factorized`. Both share the same decoder and loss logic from the base `HIVAE` class; they differ only in how the categorical posterior $q(s \mid x^o)$ is parameterised (see Section 3.1). The `run()` entry point hardcodes `HIVAE_inputDropout`.

All versions share the same latent backbone:

$$\text{Encoder:} \quad x^{obs} \;\longrightarrow\; q(s \mid x^{obs}),\; q(z \mid s, x^{obs})$$

$$\text{Prior:} \quad p(s) = \mathrm{Uniform}(1/K),\quad p(z \mid s) = \mathcal{N}(\mu_s,\, I)$$

$$\text{Decoder:} \quad z \;\longrightarrow\; y = W_y z \;\longrightarrow\; \theta_j(y_j, s) \;\longrightarrow\; p(x_j \mid \theta_j)$$

The versions differ in **which observation likelihoods are active** and, for V2A, in **additional encoder and decoder structure** for longitudinal data:

| Version | Target Distribution | Additional Structure |
|---|---|---|
| V0 | $P(X_{\text{baseline}})$ | None |
| V1 | $P(X_{\text{baseline}}, U)$ | Endpoint as additional `real` feature |
| V2A | $P(X_{\text{baseline}}, Y_{1:T})$ | Longitudinal summary encoder + time-conditioned decoder |
| V3_weibull | $P(X, t, \delta)$ | Weibull survival likelihood |
| V3_piecewise | $P(X, t, \delta)$ | Piecewise-constant survival likelihood |

> **Implementation note on V2A:** In the current codebase (`src.py`), the V2A-specific layer initialization block (time embedding, longitudinal decoder, summary encoder, and augmented encoder layers) is placed at the end of `HIVAE.__init__`, **after** the theta-layer construction and **before** `get_theta_view()` is ever called. When `model_version='v2a'`, the V2A layers are properly instantiated and the encoder layers (`s_layer`, `z_layer`) are re-created with the augmented input dimension (`input_dim + long_summary_dim`). All V2A methods (`_encode_longitudinal_summary`, `compute_longitudinal_log_lik`, `generate_longitudinal`) are fully functional. The forward pass in `HIVAE.forward()` and the training/generation code in `surv_hivae.py` have correct V2A branching logic.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of patients |
| $D$ | Total input dimension (after encoding) |
| $J$ | Number of features |
| $K$ | Dimension of categorical latent $s$ (`s_dim`) |
| $d_z$ | Dimension of continuous latent $z$ (`z_dim`) |
| $d_y$ | Per-feature partition size of deterministic layer $y$ (`y_dim` hyperparameter) |
| $x_j$ | Observation for feature $j$ |
| $s_i$ | Categorical latent assignment for patient $i$, $s_i \in \{1,\ldots,K\}$ |
| $z_i$ | Continuous latent vector for patient $i$, $z_i \in \mathbb{R}^{d_z}$ |
| $y_i$ | Deterministic transform, $y_i = W_y z_i$, $y_i \in \mathbb{R}^{J \cdot d_y}$ |
| $y_i^{(j)}$ | Partition of $y$ assigned to feature $j$, $y_i^{(j)} \in \mathbb{R}^{d_y}$ |
| $m_j$ | Binary missingness indicator ($1$ = observed) |
| $\tau$ | Gumbel-softmax temperature |
| $\theta_j$ | Distributional parameters for feature $j$ |
| $T^*_i$ | Latent event time for patient $i$ (V3) |
| $C_i$ | Latent censoring time for patient $i$ (V3) |
| $t_i$ | Observed time, $t_i = \min(T^*_i, C_i)$ (V3) |
| $\delta_i$ | Event indicator, $\delta_i = \mathbf{1}\{T^*_i \le C_i\}$ (V3) |
| $t_{iv}$ | Visit time $v$ for patient $i$ (V2A) |
| $y_{iv}$ | Outcome at visit $v$ for patient $i$ (V2A) |
| $n_i$ | Number of observed visits for patient $i$ (V2A) |

---

## 3. Shared Latent Backbone

### 3.1 Encoder

The codebase provides two encoder variants. The `run()` entry point hardcodes `HIVAE_inputDropout`.

#### 3.1.1 Input-Dropout Encoder (`HIVAE_inputDropout`, default)

Given normalized observed data $X^{obs} \in \mathbb{R}^D$:

**Categorical latent $s$** — approximate posterior $q(s \mid X^{obs})$:

The input-dropout encoder delegates to `s_proposal_multinomial` in `statistic.py`, which uses log-softmax:

$$\text{logits}_s = W_s X^{obs} + b_s \qquad (W_s \in \mathbb{R}^{K \times D})$$

$$\log \pi = \log\mathrm{softmax}(\text{logits}_s)$$

Sampling via Gumbel-softmax with temperature $\tau$:

$$g_k \sim \mathrm{Gumbel}(0,1) \qquad \bigl(g_k = -\log(-\log(u_k)),\; u_k \sim \mathrm{Uniform}(0,1)\bigr)$$

$$s = \mathrm{softmax}\!\left(\frac{\log \pi + g}{\tau}\right) \qquad \text{(relaxed one-hot, $K$-dimensional)}$$

**Continuous latent $z$** — approximate posterior $q(z \mid s, X^{obs})$:

$$h = W_z \bigl[X^{obs}\,;\, s\bigr] + b_z \qquad (W_z \in \mathbb{R}^{2d_z \times (D+K)})$$

$$[\mu_q,\, \log \sigma^2_q] = \mathrm{split}(h) \qquad \text{(each} \in \mathbb{R}^{d_z}\text{)}$$

$$\log \sigma^2_q = \mathrm{clamp}(\log \sigma^2_q,\, -15,\, 15)$$

Sampling via reparameterization:

$$\epsilon \sim \mathcal{N}(0, I)$$

$$z = \mu_q + \exp\!\bigl(\tfrac{1}{2}\log \sigma^2_q\bigr) \cdot \epsilon$$

#### 3.1.2 Factorized Encoder (`HIVAE_factorized`)

The factorized encoder computes $q(s \mid X^{obs})$ directly via softmax (not log-softmax):

$$p_s = \mathrm{softmax}(W_s X^{obs} + b_s)$$

$$s = \mathrm{softmax}\!\left(\frac{\log\bigl(\mathrm{clamp}(p_s, 10^{-6}, 1)\bigr) + g}{\tau}\right)$$

The $z$ encoder is identical to the input-dropout variant.

#### 3.1.3 Temperature Annealing

Temperature is annealed during training:

$$\tau(\text{epoch}) = \max(1.0 - 0.01 \cdot \text{epoch},\; 10^{-3})$$

### 3.2 Prior

**Categorical prior** $p(s)$:

$$p(s = k) = \frac{1}{K} \qquad \text{for } k = 1, \ldots, K$$

**Gaussian mixture prior** $p(z \mid s)$:

$$p(z \mid s) = \mathcal{N}\bigl(\mu_p(s),\, I\bigr)$$

$$\mu_p(s) = W_{pz}\, s \qquad (W_{pz} \in \mathbb{R}^{d_z \times K})$$

$$\log \sigma^2_p = 0 \qquad \text{(fixed unit variance)}$$

### 3.3 Deterministic Transform $z \to y$

$$y = W_y\, z \qquad (W_y \in \mathbb{R}^{(J \cdot d_y) \times d_z},\; \text{with bias})$$

The $y$ vector is **partitioned** into $J$ non-overlapping slices, one per feature. By default (when no custom partition is provided), each feature $j$ receives its own $d_y$-dimensional slice:

$$y = \bigl[y^{(1)},\, y^{(2)},\, \ldots,\, y^{(J)}\bigr]$$

$$y^{(j)} \in \mathbb{R}^{d_y} \qquad \text{for each } j = 1, \ldots, J$$

$$\dim(y) = J \cdot d_y \qquad \text{(total $y$ dimension)}$$

The partition sizes are configurable via `y_dim_partition`; the default sets each entry to $d_y$.

---

## 4. Feature-Wise Decoder (Theta Layers)

Each feature $j$ has dedicated parameter networks (theta layers) that map the latent representation to distributional parameters. These operate on the concatenation of the feature's $y$-partition and the $s$ sample.

The general pattern:

$$\theta_j = f_j\bigl(\bigl[y^{(j)}\,;\, s\bigr]\bigr)$$

where $f_j$ is a feature-type-specific neural network.

**Observed / missing split (inherited implementation detail):** During theta estimation, the data is partitioned into observed ($m_j = 1$) and missing ($m_j = 0$) subsets. The theta layers are applied to both subsets, but the missing-data forward pass uses `torch.no_grad()` so that parameter gradients are computed only from observed entries. This is a legacy design choice inherited from the original HI-VAE codebase. An alternative approach — running a normal forward pass on all entries and masking the loss — would be mathematically equivalent for the loss gradient but differs in that the current approach also prevents any indirect gradient flow through the theta-layer outputs for missing entries. The current behavior is retained for compatibility with the original implementation.

| Feature Type | Parameters | Network | Status |
|---|---|---|---|
| real | $\mu,\, \sigma^2$ | $\mu$: `Linear(d_y^{(j)} + K, 1)`, $\sigma$: `Linear(K, 1)` | primary |
| pos | $\mu,\, \sigma^2$ | $\mu$: `Linear(d_y^{(j)} + K, 1)`, $\sigma$: `Linear(K, 1)` | primary |
| count | $\lambda$ | `Linear(d_y^{(j)} + K, 1)` | primary |
| cat | logits $\pi_{1..C-1}$ | `Linear(d_y^{(j)} + K, C-1)` | primary |
| ordinal | thresholds $\theta_{1..C-1}$, $\mu$ | $\theta$: `Linear(K, C-1)`, $\mu$: `Linear(d_y^{(j)} + K, 1)` | primary |
| **surv_weibull** | $k_T, \lambda_T, k_C, \lambda_C$ | Single `Linear(d_y^{(j)} + K, 4)` | **primary** |
| **surv_piecewise** | $\text{mass}_T$, $\text{mass}_C$ (per interval) | 1-layer or 2-layer MLP per component | **primary** |
| surv | $\mu_T, \sigma_T, \mu_C, \sigma_C$ | $\mu_{T/C}$: `Linear(d_y^{(j)} + K, 1)`, $\sigma_{T/C}$: `Linear(K, 1)` | *secondary* |
| surv_loglog | $\beta_T{-}1, \alpha_T, \beta_C{-}1, \alpha_C$ | Single `Linear(d_y^{(j)} + K, 4)` | *secondary* |

All theta-layer linear layers are **bias-free** (`bias=False`).

**Survival head tiers:**
- **Primary (mainline model family):** `surv_weibull` and `surv_piecewise` — mathematically clean, exposed via `--model_version v3_weibull` / `v3_piecewise`, validated.
- **Secondary (experimental):** `surv` (log-normal) and `surv_loglog` (log-logistic) — available by setting the type in `data_types.csv`, but **not** exposed through `--model_version`. These have known mathematical limitations documented in Sections 5.6 and 5.8.

---

## 5. Observation Likelihoods

All log-likelihoods are masked by the observation indicator:

$$\mathcal{L}_j^{obs} = \log p(x_j \mid \theta_j) \cdot m_j$$

$$\mathcal{L}_j^{miss} = \log p(x_j \mid \theta_j) \cdot (1 - m_j)$$

### 5.1 Real-Valued (Gaussian)

**Normalization:** z-score on observed data within the current batch.

$$\mu_{\text{norm}},\, \sigma^2_{\text{norm}} = \mathrm{mean}(x_j^{obs}),\, \mathrm{var}(x_j^{obs})$$

$$x_j^{\text{normalized}} = \frac{x_j - \mu_{\text{norm}}}{\sqrt{\sigma^2_{\text{norm}}}}$$

**Parameters from theta layers:**

$$\mu_{\text{raw}},\, \sigma^2_{\text{raw}} = \theta_j$$

$$\sigma^2_{\text{raw}} = \mathrm{softplus}(\sigma^2_{\text{raw}}), \quad \text{clamped to } [10^{-3},\, 10^{20}]$$

**Affine transformation back to data space:**

$$\mu = \sqrt{\sigma^2_{\text{norm}}} \cdot \mu_{\text{raw}} + \mu_{\text{norm}}$$

$$\sigma^2 = \sigma^2_{\text{norm}} \cdot \sigma^2_{\text{raw}}$$

**Log-likelihood (Gaussian):**

$$\log p(x_j \mid \theta_j) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\sum_d \log(\sigma^2_d) - \frac{1}{2}\sum_d \frac{(x_{j,d} - \mu_d)^2}{\sigma^2_d}$$

**Sampling:**

$$x_j \sim \mathcal{N}(\mu,\, \sigma^2)$$

### 5.2 Positive Real-Valued (Shifted Log-Normal)

The model uses a **shifted log-normal** via the `log1p` transform: if $Y = \log(1 + X) \sim \mathcal{N}(\mu, \sigma^2)$, then $X = \exp(Y) - 1$.

**Normalization:** log1p-transform then z-score on observed data within the current batch.

$$x_{\log} = \log(1 + x_j)$$

$$\mu_{\log},\, \sigma^2_{\log} = \mathrm{mean}(x_{\log}^{obs}),\, \mathrm{var}(x_{\log}^{obs})$$

$$x_j^{\text{normalized}} = \frac{x_{\log} - \mu_{\log}}{\sqrt{\sigma^2_{\log}}}$$

**Parameters:**

$$\mu_{\text{raw}},\, \sigma^2_{\text{raw}} = \theta_j$$

$$\sigma^2_{\text{raw}} = \mathrm{softplus}(\sigma^2_{\text{raw}}), \quad \text{clamped to } [10^{-3},\, 1.0]$$

$$\mu = \sqrt{\sigma^2_{\log}} \cdot \mu_{\text{raw}} + \mu_{\log}$$

$$\sigma^2 = \sigma^2_{\log} \cdot \sigma^2_{\text{raw}}$$

**Log-likelihood:** The change-of-variable Jacobian for $x \mapsto \log(1+x)$ is $1/(1+x)$, giving:

$$\log p(x_j) = -\frac{1}{2}\sum_d \frac{(\log(1 + x_{j,d}) - \mu_d)^2}{\sigma^2_d} - \frac{1}{2}\sum_d \log(2\pi\,\sigma^2_d) - \sum_d \log(1 + x_{j,d})$$

**Sampling:**

$$Y \sim \mathcal{N}(\mu,\, \sigma^2)$$

$$x_j = \exp(Y) - 1, \quad \text{clamped to } [0,\, 2 \cdot \max(x_j)]$$

### 5.3 Count Data (Poisson)

**Normalization:** log1p-transform (no z-score).

**Parameters:**

$$\lambda = \mathrm{softplus}(\theta_j), \quad \text{clamped to } [10^{-6},\, 10^{20}]$$

**Log-likelihood:**

$$\log p(x_j) = \sum_d \bigl[ x_{j,d} \log(\lambda_d) - \lambda_d - \log\Gamma(x_{j,d} + 1) \bigr]$$

**Sampling:**

$$x_j \sim \mathrm{Poisson}(\lambda)$$

### 5.4 Categorical

**Encoding:** One-hot representation with $C$ classes.

**Parameters:**

$$\text{logits} = \bigl[ 0,\, \theta_1,\, \ldots,\, \theta_{C-1} \bigr] \qquad \text{(zero-padded for identifiability)}$$

**Log-likelihood:**

$$\log p(x_j = c) = \log \mathrm{softmax}(\text{logits})_c = -\mathrm{CrossEntropy}(\text{logits},\, c)$$

**Sampling:**

$$x_j \sim \mathrm{Categorical}(\text{logits}) \qquad \text{then one-hot encoded}$$

### 5.5 Ordinal

**Encoding:** Thermometer encoding (cumulative binary vector).

**Parameters:** Threshold parameters $\theta_{1..C-1}$ and mean $\mu$:

$$\theta_{\text{cumulative}} = \mathrm{cumsum}\bigl(\mathrm{softplus}(\theta_{\text{raw}})\bigr) \qquad \text{(ensures increasing thresholds)}$$

$$P(x_j \le k) = \sigma(\theta_{\text{cumulative},k} - \mu)$$

$$P(x_j = k) = P(x_j \le k) - P(x_j \le k-1)$$

with $P(x_j \le 0) = 0$ and $P(x_j \le C) = 1$.

**Log-likelihood:**

$$\log p(x_j = k) = \log\bigl(P(x_j = k)\bigr), \quad \text{clamped to } [10^{-6},\, 1.0]$$

**Sampling:**

$$x_j \sim \mathrm{Categorical}\bigl(P(x_j = k)\bigr) \qquad \text{then thermometer encoded}$$

### 5.6 Survival: Log-Normal (`surv`) — *secondary / experimental*

> **Status:** This head is **not** part of the validated mainline model family. It is available in code via `data_types.csv` but is **not** exposed through `--model_version`. Use with caution.

The `surv` type models independent latent event time $T^*$ and censoring time $C$ using **log-normal** distributions (via `Normal` hazard functions on the log1p-transformed time). Only the observed pair $(t, \delta)$ is used.

**Normalization:** log1p-transform then z-score on observed data within the current batch (same as `pos` type).

**Parameters:** Mean and variance for event and censoring, with affine transformation:

$$\mu_{T,\text{raw}},\, \sigma^2_{T,\text{raw}},\, \mu_{C,\text{raw}},\, \sigma^2_{C,\text{raw}} = \theta$$

$$\sigma^2_{\text{raw}} = \mathrm{softplus}(\sigma^2_{\text{raw}}), \quad \text{clamped to } [10^{-5},\, 1.0]$$

$$\mu_T = \sqrt{\sigma^2_{\log}} \cdot \mu_{T,\text{raw}} + \mu_{\log}, \qquad \sigma^2_T = \sigma^2_{\log} \cdot \sigma^2_{T,\text{raw}}$$

(Same affine transform for $\mu_C, \sigma^2_C$.)

**Log-likelihood:** Uses the hazard formulation with the Normal distribution on the log1p-transformed time:

$$t_{\log} = \log(1 + t)$$

$$\log h(t_{\log};\, \mu,\, \sigma) = \log\!\left(\frac{\phi\!\left(\frac{t_{\log} - \mu}{\sigma}\right)}{1 - \Phi\!\left(\frac{t_{\log} - \mu}{\sigma}\right)}\right)$$

$$H(t_{\log};\, \mu,\, \sigma) = -\log\!\left(1 - \Phi\!\left(\frac{t_{\log} - \mu}{\sigma}\right)\right)$$

$$\log p(t, \delta) = \delta \cdot \log h_T(t_{\log}) + (1-\delta) \cdot \log h_C(t_{\log}) - H_T(t_{\log}) - H_C(t_{\log})$$

> **Known limitation:** The log-likelihood above operates on the transformed scale $t_{\log} = \log(1+t)$ but does **not** include the change-of-variable Jacobian $-\log(1+t)$ that would be needed for a proper density on the original time scale. As a result, the training objective is not a mathematically correct log-density of the observed time. The sampling procedure (below) is internally consistent: it draws from Normal and inverts via `exp()-1`. This scale mismatch between likelihood and density is one reason this head remains secondary/experimental.

**Sampling:**

$$Y_T \sim \mathcal{N}(\mu_T,\, \sigma^2_T), \quad T^* = \exp(Y_T) - 1$$

$$Y_C \sim \mathcal{N}(\mu_C,\, \sigma^2_C), \quad C = \exp(Y_C) - 1$$

$$t = \min(T^*, C), \quad \delta = \mathbf{1}\{T^* \le C\}$$

### 5.7 Survival: Weibull (`surv_weibull`)

The model assumes independent latent event time $T^*$ and censoring time $C$, each with its own Weibull distribution. Only the observed pair $(t, \delta) = (\min(T^*, C),\, \mathbf{1}\{T^* \le C\})$ is used for training.

**Normalization:** Min-max scaling of observed times to $[0, 1]$.

$$t_{\text{scaled}} = \frac{t - t_{\min}}{t_{\max} - t_{\min}}$$

**Parameters:** Shape ($k$) and scale ($\lambda$) for event and censoring, each passed through softplus with clamps $[10^{-3}, 10^3]$:

$$k_T,\, \lambda_T,\, k_C,\, \lambda_C = \mathrm{softplus}(\theta_{\text{raw}})$$

**Observed-data likelihood.** Under the independent-competing-risks assumption:

$$p(t, \delta) = \bigl[f_T(t) \cdot S_C(t)\bigr]^\delta \cdot \bigl[f_C(t) \cdot S_T(t)\bigr]^{1-\delta}$$

Using the hazard decomposition $f(t) = h(t)\,S(t)$ and $S(t) = \exp(-H(t))$:

$$\log p(t, \delta) = \delta \cdot \log h_T(t) + (1-\delta) \cdot \log h_C(t) - H_T(t) - H_C(t)$$

For the Weibull distribution:

$$\log h(t;\, k,\, \lambda) = \log\!\left(\frac{k}{\lambda}\right) + (k-1) \log\!\left(\frac{t}{\lambda}\right)$$

$$H(t;\, k,\, \lambda) = \left(\frac{t}{\lambda}\right)^k$$

(Uses `torchsurv.loss.weibull` with log-scale / log-shape parameterization and `respective_times=True`.)

**Sampling:** Both latent times are sampled independently, then the observed pair is formed:

$$U, V \sim \mathrm{Uniform}(0,1)$$

$$T^*_{\text{scaled}} = \lambda_T \cdot (-\log U)^{1/k_T}$$

$$C_{\text{scaled}} = \lambda_C \cdot (-\log V)^{1/k_C}$$

$$t = \min(T^*_{\text{scaled}},\, C_{\text{scaled}}) \cdot (t_{\max} - t_{\min}) + t_{\min}$$

$$\delta = \mathbf{1}\{T^*_{\text{scaled}} \le C_{\text{scaled}}\}$$

### 5.8 Survival: Log-Logistic (`surv_loglog`) — *secondary / experimental*

> **Status:** This head is **not** part of the validated mainline model family. It is available in code via `data_types.csv` but is **not** exposed through `--model_version`. Use with caution.

The `surv_loglog` type models both event time $T^*$ and censoring time $C$ with **log-logistic** distributions for the likelihood. This head uses the same `Linear(d_y^{(j)} + K, 4)` architecture as Weibull.

**Normalization:** Min-max scaling of observed times to $[0, 1]$.

**Parameters:** Shape-minus-one ($\beta_{-1}$), scale ($\alpha$) for event and censoring:

$$\beta_{T,-1},\, \alpha_T,\, \beta_{C,-1},\, \alpha_C = \mathrm{softplus}(\theta_{\text{raw}}), \quad \text{clamped to } [10^{-3},\, 10^3]$$

$$\beta_T = \beta_{T,-1} + 1, \qquad \beta_C = \beta_{C,-1} + 1$$

**Log-logistic hazard and cumulative hazard:**

$$h(t;\, \alpha,\, \beta) = \frac{(\beta/\alpha)\,(t/\alpha)^{\beta - 1}}{1 + (t/\alpha)^\beta}$$

$$H(t;\, \alpha,\, \beta) = \log\!\bigl(1 + (t/\alpha)^\beta\bigr)$$

**Log-likelihood:**

$$\log p(t, \delta) = \delta \cdot \log h_T(t_{\text{scaled}}) + (1-\delta) \cdot \log h_C(t_{\text{scaled}}) - H_T(t_{\text{scaled}}) - H_C(t_{\text{scaled}})$$

**Sampling:**

$$U \sim \mathrm{Uniform}(0,1), \qquad T^*_{\text{scaled}} = \alpha_T \cdot \left(\frac{1-U}{U}\right)^{1/\beta_T} \quad \text{(log-logistic inverse CDF)}$$

$$V \sim \mathrm{Uniform}(0,1), \qquad C_{\text{scaled}} = \alpha_C \cdot (-\log V)^{1/\beta_{C,-1}} \quad \text{(Weibull inverse CDF, using raw parameter without $+1$)}$$

> **Known inconsistencies:** The censoring component has two compounding mismatches:
> 1. **Distribution mismatch:** The censoring **likelihood** uses log-logistic $h_C / H_C$, but the censoring **sampling** uses Weibull inverse CDF.
> 2. **Parameter offset mismatch:** In the likelihood, the raw network output for censoring shape is passed through `shapem1 + 1` to obtain $\beta_C$ (matching the event-time convention). In the sampling code, the same raw output is used directly as the Weibull shape exponent **without** the $+1$ offset. So the sampling shape is $\beta_{C,-1}$, not $\beta_C$.
>
> The event time likelihood and sampling are both consistently log-logistic with the $+1$ offset applied in both paths. These mismatches are implementation artifacts and are reasons this head remains secondary/experimental.

$$t = \min(T^*_{\text{scaled}},\, C_{\text{scaled}}) \cdot (t_{\max} - t_{\min}) + t_{\min}$$

$$\delta = \mathbf{1}\{T^*_{\text{scaled}} \le C_{\text{scaled}}\}$$

### 5.9 Survival: Piecewise-Constant (`surv_piecewise`)

As with Weibull, latent event time $T^*$ and censoring time $C$ are modelled independently, each with a piecewise-constant distribution. Only the observed pair $(t, \delta)$ is used.

**Normalization:** Min-max scaling of observed times to $[0, 1]$.

**Intervals:** $K$ equal-width intervals $[a_0, a_1),\, [a_1, a_2),\, \ldots,\, [a_{K-1}, a_K)$ computed by `linspace` over the normalized time range, with an additional boundary extending beyond the data range.

**Parameters:** Each patient gets per-interval **probability masses** via softmax:

$$\pi_T = \mathrm{softmax}(\theta_T) \qquad (K\text{-dim vector, sums to 1;}\; \pi_{T,k} = P(T^* \in [a_{k-1}, a_k)))$$

$$\pi_C = \mathrm{softmax}(\theta_C)$$

$$\mathrm{CDF}_T = \mathrm{cumsum}(\pi_T), \qquad S_T = 1 - \mathrm{CDF}_T$$

The **density** at a point within interval $k$ is the mass divided by the bin width:

$$f_T(t) = \frac{\pi_{T,k}}{a_k - a_{k-1}}$$

For 2-layer MLP variant:

$$\theta_T = W_2 \cdot \mathrm{ReLU}(W_1 \cdot [y^{(j)}\,;\, s]) \qquad (\text{hidden dim} = 20)$$

**Log-likelihood with linear interpolation of survival:**

$$\text{bin} = \mathrm{bucketize}(t_{\text{scaled}})$$

$$w = \frac{t_{\text{scaled}} - a_{\text{bin}}}{a_{\text{bin}+1} - a_{\text{bin}}}$$

$$S_T(t) = (1 - w) \cdot S_T(a_{\text{bin}-1}) + w \cdot S_T(a_{\text{bin}})$$

$$\log f_T(t) = \log\!\left(\frac{\pi_T[\text{bin}]}{\text{bin\_width}} + \epsilon\right)$$

$$\log S_T(t) = \log\bigl(S_T(t) + \epsilon\bigr)$$

$$\log p(t, \delta) = \delta \cdot \log f_T(t) + (1-\delta) \cdot \log S_T(t) + (1-\delta) \cdot \log f_C(t) + \delta \cdot \log S_C(t)$$

This is equivalent to the standard hazard form: $\delta \cdot \log h_T + (1-\delta) \cdot \log h_C - H_T - H_C$.

**Sampling:** Both latent times are sampled independently, then the observed pair is formed:

For $T^*$:
1. $k \sim \mathrm{Categorical}(\pi_T)$ — select interval by mass
2. $T^*_{\text{scaled}} = a_k + U \cdot (a_{k+1} - a_k)$ — uniform within interval

For $C$: same procedure with $\pi_C$.

$$t = \min(T^*_{\text{scaled}},\, C_{\text{scaled}}) \cdot (t_{\max} - t_{\min}) + t_{\min}$$

$$\delta = \mathbf{1}\{T^*_{\text{scaled}} \le C_{\text{scaled}}\}$$

---

## 6. Loss Function (ELBO)

The training objective maximises the Evidence Lower Bound:

$$\mathrm{ELBO} = \mathbb{E}_q\!\left[\sum_j \log p(x_j \mid \theta_j) \cdot m_j\right] - D_{\mathrm{KL}}\bigl(q(s \mid X) \,\|\, p(s)\bigr) - D_{\mathrm{KL}}\bigl(q(z \mid s, X) \,\|\, p(z \mid s)\bigr)$$

### KL divergence for $s$ (categorical)

With uniform prior $p(s = k) = 1/K$:

$$D_{\mathrm{KL}}\bigl(q(s) \,\|\, p(s)\bigr) = \sum_k \pi_k \log \pi_k - \sum_k \pi_k \log\!\left(\frac{1}{K}\right) = \sum_k \pi_k \log \pi_k + \log K$$

where $\pi_k = \mathrm{softmax}(\text{logits}_s)_k$.

In code, $\sum_k \pi_k \log \pi_k$ is computed as the negative of $\mathrm{CrossEntropy}(\text{logits}_s, \mathrm{softmax}(\text{logits}_s))$, yielding:

$$D_{\mathrm{KL},s} = -\mathrm{CrossEntropy}(\text{logits}_s,\, \pi) + \log(K)$$

### KL divergence for $z$ (Gaussian)

$$D_{\mathrm{KL},z} = -\frac{d_z}{2} + \frac{1}{2}\sum_d \left[\exp(\log\sigma^2_{q,d} - \log\sigma^2_{p,d}) + \frac{(\mu_{p,d} - \mu_{q,d})^2}{\exp(\log\sigma^2_{p,d})} - \log\sigma^2_{q,d} + \log\sigma^2_{p,d}\right]$$

Since $\log\sigma^2_p = 0$ (unit variance prior conditioned on $s$), this simplifies to:

$$D_{\mathrm{KL},z} = -\frac{d_z}{2} + \frac{1}{2}\sum_d \left[\exp(\log\sigma^2_{q,d}) + (\mu_{p,d} - \mu_{q,d})^2 - \log\sigma^2_{q,d}\right]$$

### Complete objective

$$\mathrm{Loss} = -\mathrm{ELBO} = -\mathrm{mean\_over\_batch}\!\left(\sum_j \mathcal{L}_j^{obs} - D_{\mathrm{KL},z} - D_{\mathrm{KL},s}\right)$$

The model minimises `neg_ELBO_loss` $= -\mathrm{ELBO}$.

---

## 7. Version-Specific Formulations

### 7.1 V0: Static Baseline HI-VAE

**Target distribution:**

$$P(X_{\text{pre}}) = \sum_s \int_z \left[\prod_j p(x_j \mid \theta_j(y(z), s))\right] p(z \mid s)\, p(s)\, dz$$

where $j$ ranges over **baseline/pre-randomization variables only** (real, pos, count, cat, ordinal types).

**ELBO:**

$$\mathrm{ELBO}_{\text{v0}} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j\right] - D_{\mathrm{KL},s} - D_{\mathrm{KL},z}$$

**Use case:** Synthetic baseline cohort generation; static benchmark.

### 7.2 V1: Baseline + Continuous Endpoint

V1 jointly models baseline variables $X_{\text{pre}}$ and a single continuous post-randomization endpoint $U$. The joint model is:

$$P(X_{\text{pre}}, U) = \sum_s \int_z \left[\prod_j p(x_j \mid \theta_j) \cdot p(U \mid \theta_U(y, s))\right] p(z \mid s)\, p(s)\, dz$$

Because both $X_{\text{pre}}$ and $U$ are generated from the same latent $(s, z)$, the model implicitly captures $P(U \mid X_{\text{pre}})$ through the shared latent structure.

The endpoint $U$ is modelled with a Gaussian likelihood (type `real`):

$$U \mid z, s \sim \mathcal{N}\!\left(\mu_U(y^{(U)}, s),\; \sigma^2_U(s)\right)$$

where $\mu_U$ and $\sigma^2_U$ follow the same theta-layer pattern as other real features (Section 5.1).

**ELBO:**

$$\mathrm{ELBO}_{\text{v1}} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \log p(U \mid \theta_U) \cdot m_U\right] - D_{\mathrm{KL},s} - D_{\mathrm{KL},z}$$

**Implementation:** The endpoint is included as an additional `real` feature in the `data_types.csv` specification. No architectural changes are needed beyond data preparation.

**Use case:** Joint baseline-endpoint synthesis; conditional endpoint imputation given baseline structure via the shared latent.

### 7.3 V2A: Longitudinal HI-VAE

V2A extends the model to repeated continuous measurements over time per patient. The target distribution factorises the longitudinal outcomes conditionally on the shared patient latent:

$$P(X_{\text{pre}}, Y_{1:n_i}) = \sum_s \int_z \left[\prod_j p(x_j \mid \theta_j) \cdot \prod_{v=1}^{n_i} p(y_{iv} \mid z, s, t_{iv})\right] p(z \mid s)\, p(s)\, dz$$

**Key design:** One patient-level latent $(s, z)$ shared across all visits. Trajectory shape is captured through a time-conditioned decoder, **not** autoregressive recurrence.

#### 7.3.1 Longitudinal Summary Encoder

To enable the posterior $q(s \mid \cdot)$, $q(z \mid \cdot)$ to use observed longitudinal history, V2A augments the encoder input with a **longitudinal summary vector** computed by masked mean-pooling of per-visit encodings.

Each observed visit (value $y_{iv}$, time $t_{iv}$) is encoded as:

$$h_{iv} = \mathrm{MLP}_{\text{summary}}\bigl([y_{iv}\,;\, e(t_{iv})]\bigr) \qquad (\text{output dim} = 16)$$

where $e(t)$ is the shared time embedding (see below). The summary aggregates over observed visits:

$$r_i = \frac{1}{n_i} \sum_{v:\, m_{iv}=1} h_{iv} \qquad \text{(masked mean pooling)}$$

The encoder then operates on the augmented input $[X^{obs}\,;\, r_i]$:

$$q(s \mid X^{obs}, r_i), \qquad q(z \mid s, X^{obs}, r_i)$$

When no longitudinal data is available, $r_i = \mathbf{0}$.

This design enables two posterior-based generation modes:

- **Baseline-conditioned generation:** $r_i = \mathbf{0}$ → encoder sees only baseline → full trajectories generated from the baseline-conditioned posterior.
- **History-conditioned completion:** $r_i$ encodes observed prefix visits → encoder conditions on baseline + history → future visits generated from the informed posterior.

(True unconditional generation — sampling from the prior with no observed data — is described in Section 9.)

#### 7.3.2 Time Embedding

$$e(t) = W_2 \cdot \mathrm{SiLU}(W_1 \cdot t + b_1) + b_2 \qquad (W_1 \in \mathbb{R}^{16 \times 1},\; W_2 \in \mathbb{R}^{16 \times 16})$$

where $\mathrm{SiLU}(x) = x \cdot \sigma(x)$. The time embedding is shared between the summary encoder and the longitudinal decoder.

#### 7.3.3 Longitudinal Decoder (Time-Conditioned Gaussian)

$$\mu_{iv} = W_\mu \cdot [z_i\,;\, s_i\,;\, e(t_{iv})] \qquad (W_\mu \in \mathbb{R}^{D_{\text{out}} \times (d_z + K + 16)})$$

$$\sigma^2_{iv} = \mathrm{softplus}\bigl(W_\sigma \cdot [s_i\,;\, e(t_{iv})]\bigr), \quad \text{clamped to } [10^{-3},\, 10^3]$$

$$y_{iv} \mid z_i, s_i, t_{iv} \sim \mathcal{N}(\mu_{iv},\, \sigma^2_{iv})$$

where $D_{\text{out}}$ is `n_long_outcomes` (number of longitudinal outcome variables, default 1).

**Variance parameterization (intentional simplification):** The mean $\mu$ depends on $(z, s, \text{time})$ while the variance depends on $(s, \text{time})$ only. This mirrors the baseline HI-VAE pattern where the mean theta layer receives $[y; s]$ but the sigma layer receives $[s]$ only. The rationale is that patient-specific trajectory *level* is captured by $z$ through $\mu$, while the *noise structure* across timepoints is determined by the mixture component $s$ and time. This is a deliberate design choice for stability on small datasets.

#### 7.3.4 Longitudinal Log-Likelihood

Masked over observed visits:

$$\mathcal{L}_{\text{long},i} = \sum_{v=1}^{n_i} m_{iv} \cdot \left[-\frac{1}{2}\log(\sigma^2_{iv}) - \frac{(y_{iv} - \mu_{iv})^2}{2\,\sigma^2_{iv}} - \frac{1}{2}\log(2\pi)\right]$$

#### 7.3.5 ELBO

$$\mathrm{ELBO}_{\text{v2a}} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \mathcal{L}_{\text{long},i}\right] - D_{\mathrm{KL},s} - D_{\mathrm{KL},z}$$

#### 7.3.6 Data Format

Longitudinal data is stored as padded tensors:

| Tensor | Shape | Description |
|---|---|---|
| `times` | $(N, V_{\max})$ | Normalised to $[0, 1]$ |
| `values` | $(N, V_{\max})$ or $(N, V_{\max}, D)$ | Normalised to zero-mean unit-variance. 3D for multi-outcome. |
| `masks` | $(N, V_{\max})$ | Binary ($1$ = observed visit) |

where $V_{\max}$ = max number of visits across all patients. Patients with fewer visits have trailing zeros and mask $= 0$.

Longitudinal normalisation is computed **globally** from all observed values (not per-batch), and stored for denormalisation at generation time.

#### 7.3.7 Generation

**Baseline-conditioned generation.** No longitudinal history provided:

$$r_i = \mathbf{0} \qquad \text{(zero summary)}$$

Encode baseline $X^{obs}$ with zero summary → $q(s, z \mid X^{obs})$. For each time $t_k$ in grid:

$$\mu_k = W_\mu \cdot [z\,;\, s\,;\, e(t_k)]$$

$$\sigma^2_k = \mathrm{softplus}\bigl(W_\sigma \cdot [s\,;\, e(t_k)]\bigr)$$

$$y_k \sim \mathcal{N}(\mu_k,\, \sigma^2_k)$$

Denormalise: $y_{\text{original}} = y_{\text{normalised}} \cdot \text{std} + \text{mean}$.

**History-conditioned completion.** Observed prefix visits provided:

$$r_i = \text{mean-pool}\bigl(\mathrm{MLP}_{\text{summary}}([y_{iv}\,;\, e(t_{iv})]) \text{ for observed } v\bigr)$$

Encode baseline with summary → $q(s, z \mid X^{obs}, r_i)$. Generate on future time grid using the conditioned $(s, z)$.

### 7.4 V3: Survival HI-VAE

V3 jointly models baseline covariates $X$ and a survival outcome. The observed survival data is the pair $(t_i, \delta_i)$ where $t_i = \min(T^*_i, C_i)$ and $\delta_i = \mathbf{1}\{T^*_i \le C_i\}$.

**Target distribution:**

$$P(X, t, \delta) = \sum_s \int_z \left[\prod_j p(x_j \mid \theta_j) \cdot p(t, \delta \mid \theta_{\text{surv}}(y, s))\right] p(z \mid s)\, p(s)\, dz$$

The survival likelihood family is selected by the feature type in `data_types.csv`:

- **Primary (accessible via `--model_version`):**
  - `surv_weibull` → Section 5.7
  - `surv_piecewise` → Section 5.9
- **Secondary (available via `data_types.csv` but not via `--model_version`):**
  - `surv` (log-normal) → Section 5.6
  - `surv_loglog` (log-logistic) → Section 5.8

**ELBO:**

$$\mathrm{ELBO}_{\text{v3}} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j + \log p(t, \delta \mid \theta_{\text{surv}})\right] - D_{\mathrm{KL},s} - D_{\mathrm{KL},z}$$

---

## 8. Data Normalization

**Inherited implementation detail:** Normalization for baseline features is computed **per batch** from observed values only. Normalization parameters are computed inside each forward pass and passed to the likelihood functions; they are not stored globally. This is a legacy design inherited from the original HI-VAE codebase, not an ideal modeling choice.

**Known limitation:** Per-batch normalization means statistics vary across batches and between training and generation. For features with high variance or small batch sizes, this can introduce inconsistencies. A more principled approach would compute normalization statistics once on the training set and reuse them throughout; this is left as a future improvement. The V2A longitudinal normalisation, by contrast, is already computed once globally.

| Feature Type | Transform | Normalization | Stored Parameters |
|---|---|---|---|
| real | identity | z-score: $(x - \mu) / \sigma$ | $(\mu,\, \sigma^2)$ per batch |
| pos | $\log(1+x)$ | z-score on log1p: $(\log(1+x) - \mu_{\log}) / \sigma_{\log}$ | $(\mu_{\log},\, \sigma^2_{\log})$ per batch |
| count | $\log(1+x)$ | log transform only | $(0, 1)$ |
| cat | one-hot | none | $(0, 1)$ |
| ordinal | thermometer | none | $(0, 1)$ |
| surv | $\log(1+x)$ | z-score on log1p (same as pos) | $(\mu_{\log},\, \sigma^2_{\log})$ per batch |
| surv_weibull | identity | min-max: $(x - \min)/(\max - \min)$ | $(\min,\, \max)$ per batch |
| surv_loglog | identity | min-max | $(\min,\, \max)$ per batch |
| surv_piecewise | identity | min-max | $(\min,\, \max)$ per batch |

**V2A longitudinal normalization** (computed once globally from training data):

| Component | Transform |
|---|---|
| Visit times | $(t - t_{\min}) / (t_{\max} - t_{\min}) \to [0, 1]$ |
| Outcome values | $(y - \mu_y) / \sigma_y \to$ zero-mean, unit-variance |

---

## 9. Sampling Procedures

### From posterior (data-conditioned)

1. Encode $X^{obs} \to q(s \mid X^{obs}),\, q(z \mid s, X^{obs})$
   (V2A: $X^{obs}$ is augmented with longitudinal summary $r_i$)
2. Sample $s \sim \mathrm{Gumbel\text{-}softmax}(q(s))$, $\;z \sim \mathcal{N}(\mu_q, \mathrm{diag}(\exp(\log\sigma^2_q)))$
3. Compute $y = W_y z$
4. Compute $\theta_j$ for each feature
5. Sample $x_j \sim p(x_j \mid \theta_j)$ for each feature
6. (V2A) Sample $y_{iv} \sim \mathcal{N}(\mu(z, s, t),\, \sigma^2(s, t))$ on time grid

### From prior (unconditional)

1. Sample $s \sim \mathrm{Categorical}(1/K)$
2. Sample $z \sim \mathcal{N}(W_{pz}\, s,\, I)$
3. Compute $y = W_y z$
4. Compute $\theta_j$ for each feature
5. Sample $x_j \sim p(x_j \mid \theta_j)$ for each feature
6. (V2A) Sample $y_{iv}$ on time grid

### Survival sampling (V3)

Both latent event and censoring times are sampled independently from their respective models, then combined into the observed pair:

$$T^* \sim \text{event distribution}, \qquad C \sim \text{censoring distribution}$$

$$t = \min(T^*, C), \qquad \delta = \mathbf{1}\{T^* \le C\}$$

For the **primary** survival heads (Weibull, piecewise-constant), the sampling distributions match the training likelihoods exactly. For the **secondary** heads (`surv`, `surv_loglog`), see the caveats noted in Sections 5.6 and 5.8.

### Conditional generation (existing, for survival versions)

Filter generated samples by condition (e.g., treatment $= 0$) and repeat until the desired number of samples is collected.

---

## 10. Training Details

**Optimizer:** Adam

**Learning rate:** Configurable (default $10^{-3}$).

**Temperature annealing:**

$$\tau(\text{epoch}) = \max(1.0 - 0.01 \cdot \text{epoch},\; 10^{-3})$$

Reaches minimum ($10^{-3}$) at epoch 100.

**Early stopping:**
- Train/validation split: 90% / 10%
- Validation evaluated every 50 epochs
- Patience: 10 consecutive non-improving validation checks
- Minimum training: 100 epochs before early stopping activates

**Batch processing:**
- Data is shuffled each epoch (using `numpy.random.default_rng` with seed 42)
- Batches are formed by sequential slicing
- Missing data mask is element-wise multiplied with true-missing mask
- Batch size is clamped to $\min(\text{batch\_size},\, \lfloor 0.9 \cdot N \rfloor)$

**V2A-specific training:**
- Longitudinal tensors are split, shuffled, and batched in parallel with baseline data
- Longitudinal NLL is logged separately from baseline reconstruction loss
- Longitudinal loss is added to ELBO in the forward pass before backpropagation

**Differential privacy training (`train_HIVAE_DP`):**
- Uses `opacus.PrivacyEngine` with `noise_multiplier=2.0`, `max_grad_norm=1.0`
- Same early stopping logic as standard training

---

## 11. Network Dimensions Summary

**Encoder layers:**

| Layer | Input dim | Output dim |
|---|---|---|
| `s_layer` | $D$ (or $D + 16$ for V2A) | $K$ |
| `z_layer` | $D + K$ (or $D + 16 + K$ for V2A) | $2 \cdot d_z$ |

**Decoder layers:**

| Layer | Input dim | Output dim |
|---|---|---|
| `z_distribution_layer` | $K$ | $d_z$ |
| `y_layer` | $d_z$ | $J \cdot d_y$ |

**Theta layers (per feature $j$):**

| Feature Type | Layer | Input dim | Output dim |
|---|---|---|---|
| real / pos | mean | $d_y^{(j)} + K$ | 1 |
| real / pos | sigma | $K$ | 1 |
| count | lambda | $d_y^{(j)} + K$ | 1 |
| cat | logits | $d_y^{(j)} + K$ | $C - 1$ |
| ordinal | theta | $K$ | $C - 1$ |
| ordinal | mean | $d_y^{(j)} + K$ | 1 |
| surv | mean_T, mean_C | $d_y^{(j)} + K$ | 1 each |
| surv | sigma_T, sigma_C | $K$ | 1 each |
| surv_weibull | theta | $d_y^{(j)} + K$ | 4 |
| surv_loglog | theta | $d_y^{(j)} + K$ | 4 |
| surv_piecewise (1-layer) | theta_T, theta_C | $d_y^{(j)} + K$ | $n_{\text{intervals}}$ each |
| surv_piecewise (2-layer) | MLP | $d_y^{(j)} + K \to 20 \to n_{\text{intervals}}$ | $n_{\text{intervals}}$ each |

**V2A longitudinal layers:**

| Layer | Input dim | Output dim | Activation |
|---|---|---|---|
| `time_embed` layer 1 | 1 | 16 | SiLU |
| `time_embed` layer 2 | 16 | 16 | none |
| `long_summary_net` layer 1 | $D_{\text{out}} + 16$ | 16 | ReLU |
| `long_summary_net` layer 2 | 16 | 16 | none |
| `longitudinal_mu` | $d_z + K + 16$ | $D_{\text{out}}$ | none |
| `longitudinal_log_var` | $K + 16$ | $D_{\text{out}}$ | softplus (applied post-hoc) |

where $D_{\text{out}} = $ `n_long_outcomes` (default 1).

**Default hyperparameters:**

| Parameter | Default | Search range (Optuna) |
|---|---|---|
| $d_z$ (`z_dim`) | 20 | $[10, 200]$ step 10 |
| $d_y$ (`y_dim`) | 15 | $[10, 200]$ step 5 |
| $K$ (`s_dim`) | 20 | $[10, 200]$ step 10 |
| learning rate | $10^{-3}$ | $\{10^{-3},\, 2 \times 10^{-4},\, 10^{-4}\}$ |
| batch size | 100 | Adaptive to dataset |
| epochs | 1000 | — |
| `n_intervals` (piecewise) | 10 | $\{5, 10, 15, 20\}$ |
| `n_layers` (piecewise) | 1 | $\{1, 2\}$ |
| `time_embed_dim` (V2A) | 16 | Fixed |
| `long_summary_dim` (V2A) | 16 | Fixed |
| `n_long_outcomes` (V2A) | 1 | Inferred from data |

---

*This document describes the model as implemented in the `survgen-clinical-trials` repository with the multi-version extension (v0/v1/v2a/v3). The validated mainline model family is: V0, V1, V2A, V3\_weibull, V3\_piecewise. Last updated: 2026-04-15.*
