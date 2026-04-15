# V0: Static Baseline HI-VAE

## Purpose

V0 is the foundational HI-VAE model that generates synthetic baseline/pre-randomization cohorts. It models only static tabular features with heterogeneous types.

## Target Distribution

$$P(X_{\text{baseline}}) = \sum_s \int p(s)\, p(z \mid s)\, \prod_{j=1}^{J} p(x_j \mid \theta_j(y(z), s))\, dz$$

where $s$ is summed over $\{1, \ldots, K\}$ and $z$ is integrated over $\mathbb{R}^{d_z}$.

## Notation

| Symbol | Meaning |
|--------|---------|
| $K$ | Dimension of categorical latent $s$ (`s_dim`) |
| $d_z$ | Dimension of continuous latent $z$ (`z_dim`) |
| $d_y$ | Per-feature partition size of deterministic layer $y$ (`y_dim`) |
| $J$ | Number of features |
| $x_j$ | Observation for feature $j$ |
| $m_j$ | Binary missingness indicator ($1$ = observed) |
| $\tau$ | Gumbel-softmax temperature |

## Encoder

Given normalized observed data $X^{obs} \in \mathbb{R}^D$:

**Categorical posterior** $q(s \mid X^{obs})$:

$$\text{logits}_s = W_s X^{obs} + b_s \qquad (W_s \in \mathbb{R}^{K \times D})$$

Sampling via Gumbel-softmax:

$$g_k \sim \text{Gumbel}(0,1), \qquad s = \text{softmax}\!\left(\frac{\log \pi + g}{\tau}\right)$$

**Continuous posterior** $q(z \mid s, X^{obs})$:

$$[\mu_q, \log \sigma^2_q] = \text{split}\!\left(W_z [X^{obs}; s] + b_z\right)$$

$$z = \mu_q + \exp(\tfrac{1}{2}\log \sigma^2_q) \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

## Prior

$$p(s = k) = \frac{1}{K}, \qquad p(z \mid s) = \mathcal{N}(W_{pz}\, s,\, I)$$

## Decoder

$$y = W_y z \qquad \text{(deterministic transform)}$$

$$y = [y^{(1)}, y^{(2)}, \ldots, y^{(J)}] \qquad \text{(partitioned per feature)}$$

Each feature $j$ has type-specific theta layers $\theta_j = f_j([y^{(j)}; s])$:

| Feature Type | Parameters | Network |
|---|---|---|
| real | $\mu, \sigma^2$ | $\mu$: `Linear(d_y + K, 1)`, $\sigma$: `Linear(K, 1)` |
| pos | $\mu, \sigma^2$ | Same as real (on log1p-transformed scale) |
| count | $\lambda$ | `Linear(d_y + K, 1)` |
| cat | logits $\pi_{1..C-1}$ | `Linear(d_y + K, C-1)` |
| ordinal | thresholds, $\mu$ | $\theta$: `Linear(K, C-1)`, $\mu$: `Linear(d_y + K, 1)` |

All theta-layer linear layers use `bias=False`.

## Observation Likelihoods

All log-likelihoods are masked: $\mathcal{L}_j^{obs} = \log p(x_j \mid \theta_j) \cdot m_j$.

- **Real:** Gaussian on z-scored data
- **Pos:** Shifted log-normal via log1p transform
- **Count:** Poisson on log1p-transformed data
- **Cat:** Categorical cross-entropy
- **Ordinal:** Cumulative distribution with increasing thresholds

See the shared observation-likelihood reference for full formulas.

## ELBO

$$\text{ELBO}_{v0} = \mathbb{E}_q\!\left[\sum_{j \in \text{baseline}} \log p(x_j \mid \theta_j) \cdot m_j\right] - D_{\text{KL}}(q(s) \| p(s)) - D_{\text{KL}}(q(z|s) \| p(z|s))$$

**KL for $s$:**

$$D_{\text{KL},s} = -\text{CrossEntropy}(\text{logits}_s, \pi) + \log(K)$$

**KL for $z$:**

$$D_{\text{KL},z} = -\frac{d_z}{2} + \frac{1}{2}\sum_d \left[\exp(\log\sigma^2_{q,d} - \log\sigma^2_{p,d}) + \frac{(\mu_{p,d} - \mu_{q,d})^2}{\exp(\log\sigma^2_{p,d})} - \log\sigma^2_{q,d} + \log\sigma^2_{p,d}\right]$$

## Sampling

**From posterior:** Encode $X^{obs}$, sample $s, z$, compute $y = W_y z$, compute $\theta_j$, sample $x_j$.

**From prior:** Sample $s \sim \text{Categorical}(1/K)$, $z \sim \mathcal{N}(W_{pz} s, I)$, then decode.

## Data Normalization

Computed **per batch** from observed values only (inherited limitation):

| Type | Transform | Normalization |
|------|-----------|---------------|
| real | identity | z-score: $(x - \mu)/\sigma$ |
| pos | $\log(1+x)$ | z-score on log1p |
| count | $\log(1+x)$ | log transform only |
| cat | one-hot | none |
| ordinal | thermometer | none |

## Implementation

- **Model class:** `HIVAE` base with `HIVAE_inputDropout` encoder (default)
- **Version argument:** `--model_version v0`
- **No additional modules** beyond the shared backbone

## Limitations

- Per-batch normalization means statistics vary across batches
- Missing-data `no_grad` behavior prevents gradient flow through unobserved theta outputs
- No temporal, longitudinal, or survival modeling capability

---

*Last updated: 2026-04-15.*
