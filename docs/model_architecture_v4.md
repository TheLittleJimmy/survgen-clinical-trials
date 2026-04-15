# V4: Joint and Sequential Multi-Modal HI-VAE

## Design Motivation

V0–V3 each handle a subset of clinical trial data modalities:
- V0/V1: baseline covariates (± endpoint)
- V2A: baseline + longitudinal trajectories
- V3: baseline + survival outcome

In practice, clinical trial datasets contain **all three modalities**: baseline covariates, repeated longitudinal measurements, and a survival endpoint. V4 provides two architectures that jointly model all three within the shared HI-VAE latent backbone.

### Why Two Sub-Versions?

- **V4_joint** is the minimal joint extension: all modalities share the same patient latent $(s, z)$ with no additional conditioning between modalities. This is the simplest possible combination of V2A + V3.

- **V4_seq** introduces sequential conditional structure, where later modalities condition on summaries of earlier ones. This is motivated by the clinical data-generating process where longitudinal observations influence survival, and allows the survival head to leverage trajectory information beyond what the shared latent alone captures.

## Notation

All notation from V0–V3 applies. Additional symbols:

| Symbol | Meaning |
|--------|---------|
| $c_X$ | Baseline summary vector: $c_X = \phi_X(X)$ |
| $r_Y$ | Longitudinal summary vector: $r_Y = \phi_Y(Y_{1:T}, \text{times}, \text{masks})$ |
| $\phi_X$ | Baseline summary network (V4_seq only) |
| $\phi_Y$ | Longitudinal summary network (reuses V2A summary encoder logic) |
| $T^*_i$ | Latent event time |
| $C_i$ | Latent censoring time |
| $t_i = \min(T^*_i, C_i)$ | Observed time |
| $\delta_i = \mathbf{1}\{T^*_i \le C_i\}$ | Event indicator |

## Survival Likelihood (Both V4 Models)

Both V4 sub-versions use the same observed-data survival likelihood as V3:

$$p(t, \delta) = [f_T(t) \cdot S_C(t)]^\delta \cdot [f_C(t) \cdot S_T(t)]^{1-\delta}$$

$$\log p(t, \delta) = \delta \cdot \log h_T(t) + (1-\delta) \cdot \log h_C(t) - H_T(t) - H_C(t)$$

Primary supported families: **Weibull** and **piecewise-constant** (same as V3). The survival family is selected via the `surv_type` argument (e.g., `--surv_type weibull` or `--surv_type piecewise`).

---

## A. V4_joint: Joint Shared-Latent Model

### Target Distribution

$$P(X, Y_{1:T}, t, \delta) = \sum_s \int p(s)\, p(z \mid s)\, p(X \mid z, s)\, p(Y_{1:T} \mid z, s)\, p(t, \delta \mid z, s)\, dz$$

All three modalities share the same patient latent $(s, z)$ with **no inter-modality conditioning** in the decoder.

### Interpretation

- **Baseline:** reuses V0/V1-style feature-wise decoders
- **Longitudinal:** reuses V2A time-conditioned Gaussian decoder
- **Survival:** reuses V3 survival head (Weibull or piecewise)

### Encoder

The encoder is the V2A augmented encoder: baseline input is concatenated with the longitudinal summary $r_i$ (from `_encode_longitudinal_summary`). When no longitudinal data is available at test time, $r_i = \mathbf{0}$.

$$q(s \mid X^{obs}, r_i), \qquad q(z \mid s, X^{obs}, r_i)$$

### ELBO

$$\text{ELBO}_{v4\_joint} = \mathbb{E}_q\!\left[\underbrace{\sum_j \log p(x_j \mid \theta_j) \cdot m_j}_{\text{baseline reconstruction}} + \underbrace{\mathcal{L}_{\text{long}}}_{\text{longitudinal log-lik}} + \underbrace{\log p(t, \delta \mid \theta_{\text{surv}})}_{\text{survival log-lik}}\right] - D_{\text{KL},s} - D_{\text{KL},z}$$

### Implementation Notes

- **Almost no new modules.** V4_joint reuses V2A encoder augmentation, V2A longitudinal decoder, and V3 survival head layers.
- The `HIVAE.__init__` initializes both V2A layers (time embedding, longitudinal decoder, summary encoder) **and** survival theta layers when `model_version` starts with `v4`.
- In `forward()`, the ELBO is computed as the sum of baseline reconstruction, longitudinal NLL, and survival NLL, minus the KL terms.
- The survival log-likelihood is computed via the existing feature-type dispatch in `loglik_evaluation` — the survival feature is simply included in `feat_types_dict`.

### Sampling Procedure

**Posterior-based generation:**
1. Encode $[X^{obs}; r_i]$ → sample $s, z$
2. Decode baseline: compute $y = W_y z$, compute $\theta_j$, sample $x_j$
3. Generate longitudinal: for each $t_k$ in time grid, sample $y_k \sim \mathcal{N}(\mu_k, \sigma^2_k)$
4. Survival is sampled as part of step 2 via the survival feature decoder

**Prior generation:**
1. Sample $s \sim \text{Categorical}(1/K)$, $z \sim \mathcal{N}(W_{pz} s, I)$
2. Decode baseline + survival as in V3
3. Generate longitudinal trajectories on time grid

---

## B. V4_seq: Sequential Conditional Model

### Target Distribution

$$P(X, Y_{1:T}, t, \delta) = \sum_s \int p(s)\, p(z \mid s)\, p(X \mid z, s)\, p(Y_{1:T} \mid c_X, z, s)\, p(t, \delta \mid c_X, r_Y, z, s)\, dz$$

The key difference from V4_joint is the **conditional structure**:
- Longitudinal decoder conditions on a baseline summary $c_X$
- Survival head conditions on both baseline summary $c_X$ and longitudinal summary $r_Y$

### Why Summary-Conditioned Rather Than Autoregressive?

A full autoregressive model $p(Y_v \mid Y_{<v}, X, z, s)$ or a Transformer-based sequence model would be substantially more complex, harder to train on small clinical datasets, and would require a fundamentally different decoder architecture. The summary-conditioned approach:

1. **Keeps the lightweight V2A decoder** — only the conditioning input changes
2. **Requires only small MLP summary networks** — no RNN/Transformer
3. **Maintains the shared latent backbone** — $(s, z)$ still captures the primary patient-level structure
4. **Is compatible with the existing training infrastructure** — batch processing, early stopping, etc.

### Summary Networks

**Baseline summary** $\phi_X$:

$$c_X = \phi_X(X_{\text{decoded}}) \qquad \text{(MLP: } D_{\text{baseline}} \to 32 \to 16\text{, ReLU activation)}$$

where $X_{\text{decoded}}$ is the concatenated decoded baseline features (from the theta layers).

At training time, $X_{\text{decoded}}$ is the reconstructed baseline; at generation time, it is the generated baseline sample.

**Longitudinal summary** $\phi_Y$:

Reuses the V2A `_encode_longitudinal_summary` architecture, which produces a 16-dimensional vector via masked mean-pooling of per-visit encodings.

At generation time, $r_Y$ is computed from the **generated planned trajectory** on the time grid, not from observed data. This is critical for avoiding information leakage (see below).

### Longitudinal Decoder Conditioning

The V4_seq longitudinal decoder augments the V2A decoder with $c_X$:

$$\mu_{iv} = W_\mu^{\text{seq}} [z_i; s_i; e(t_{iv}); c_X] \qquad (\text{input dim} = d_z + K + 16 + 16)$$

$$\sigma^2_{iv} = \text{softplus}(W_\sigma^{\text{seq}} [s_i; e(t_{iv})]), \quad \text{clamped to } [10^{-3}, 10^3]$$

Note: variance does not depend on $c_X$ (same design rationale as V2A).

### Survival Head Conditioning

The V4_seq survival head augments the V3 survival theta layers with $c_X$ and $r_Y$:

For Weibull: `Linear(d_y + K + 16 + 16, 4, bias=False)` — input is $[y^{(j)}; s; c_X; r_Y]$.

For piecewise: same augmented input dimension for theta_T and theta_C networks.

### ELBO

$$\text{ELBO}_{v4\_seq} = \mathbb{E}_q\!\left[\sum_j \log p(x_j \mid \theta_j) \cdot m_j + \mathcal{L}_{\text{long}}(c_X) + \log p(t, \delta \mid c_X, r_Y, \theta_{\text{surv}})\right] - D_{\text{KL},s} - D_{\text{KL},z}$$

### Training-Time Handling

At training time:
1. **Baseline reconstruction** proceeds as in V0 (standard forward pass)
2. **$c_X$ is computed** from the reconstructed (decoded) baseline features, detached from the reconstruction path to prevent the longitudinal/survival gradients from distorting baseline reconstruction
3. **Longitudinal NLL** uses the V2A decoder conditioned on $c_X$
4. **$r_Y$ for the survival head** is computed from the **observed longitudinal history** available up to each patient's observed time. Specifically, visits with $t_{iv} \le t_i$ are used; visits after $t_i$ are masked. This ensures no look-ahead information leaks into the survival head during training
5. **Survival NLL** uses the augmented survival head conditioned on $c_X$ and $r_Y$

---

## Information Leakage / Look-Ahead Bias in V4_seq

### The Risk

In a naïve implementation, the survival head might condition on a longitudinal summary $r_Y$ computed from **all** observed visits, including those occurring after the patient's event/censoring time. This creates a logical inconsistency: the model uses information that would only be available *after* the outcome is known to *predict* the outcome.

### How V4_seq Avoids It

**Training time:**
- $r_Y$ is computed using only pre-event visits: visits where $t_{iv} \le t_i$ (the observed survival time)
- A visit-level mask `pre_event_mask = (times <= t_i.unsqueeze(1))` is applied before the mean-pooling step
- This ensures the survival head only sees longitudinal information that would be logically available at the time of the event/censoring

**Generation time:**
The generation follows a coherent sequential story:

1. **Sample** $s, z$ from prior or posterior
2. **Generate baseline** $X$ from the standard feature-wise decoders
3. **Compute** $c_X = \phi_X(X)$ from the generated baseline
4. **Generate planned longitudinal trajectory** on a predefined time grid covering the full study horizon
5. **Compute** $r_Y = \phi_Y(\text{generated trajectory, time grid, all-ones mask})$ from the **planned** (generated) trajectory
6. **Generate** $(t, \delta)$ from the survival head conditioned on $c_X$, $r_Y$, $z$, $s$
7. **Post-process:** truncate/mask the generated longitudinal trajectory at visits after the sampled observed time $t$

### Why This Is Coherent

The generative story is: a patient enters the trial with baseline characteristics $X$. Given their latent type $(s, z)$ and baseline, a planned trajectory of measurements is implicitly defined. The survival outcome depends on the patient's type, their baseline, and the trajectory they would have followed. After the event/censoring time $t$, no further measurements are observed.

This is analogous to a "potential outcomes" framework: the planned trajectory exists conceptually for the full horizon; the survival event determines how much of it is actually observed.

---

## Generation Order Summary

### V4_joint

```
s, z  →  [X, (t,δ)]  (baseline + survival, decoded together)
      →  Y on time grid  (longitudinal, independent of survival)
```

### V4_seq

```
s, z  →  X                          (1. generate baseline)
      →  c_X = φ_X(X)               (2. baseline summary)
      →  Y_planned on full grid      (3. generate planned trajectory, conditioned on c_X)
      →  r_Y = φ_Y(Y_planned)       (4. longitudinal summary from planned trajectory)
      →  (t, δ)                      (5. survival conditioned on c_X, r_Y)
      →  truncate Y at t             (6. post-process: mask visits after t)
```

---

## Relationship to V0/V1/V2A/V3

| Component | V4_joint source | V4_seq source |
|-----------|----------------|---------------|
| Encoder backbone | V2A (augmented with $r_i$) | V2A (augmented with $r_i$) |
| Baseline decoder | V0/V1 feature-wise decoders | V0/V1 feature-wise decoders |
| Longitudinal decoder | V2A time-conditioned Gaussian | V2A + $c_X$ conditioning |
| Survival head | V3 Weibull / piecewise | V3 + $c_X$, $r_Y$ conditioning |
| Summary networks | V2A long_summary_net (encoder) | V2A long_summary_net + new $\phi_X$ |

## Posterior-Based vs Prior-Based Generation

**Posterior-based:** Given observed baseline $X^{obs}$ (and optionally longitudinal history), encode to get $(s, z)$, then generate. This is the default for data augmentation tasks.

**Prior-based:** Sample $s, z$ from prior. Requires normalization parameters from a reference batch. Used for unconditional synthetic data generation.

**Baseline-conditioned:** Encode baseline only (no longitudinal summary), then generate longitudinal + survival. Available for both V4_joint and V4_seq.

**History-conditioned completion:** Encode baseline + observed longitudinal prefix, then generate future trajectory + survival. Primarily useful for V4_joint; for V4_seq the generation order enforces planned trajectory → survival.

## Implementation Summary

### V4_joint
- Adds to `HIVAE.__init__`: V2A layers + survival theta layers
- `forward()`: baseline + longitudinal + survival loss terms summed into ELBO
- Generation: baseline + survival via standard decode, longitudinal via `generate_longitudinal`
- **New modules:** none (reuses V2A + V3 components)

### V4_seq
- Adds to `HIVAE.__init__`: V2A layers + survival theta layers + baseline summary $\phi_X$ + adapted longitudinal decoder + adapted survival theta layers
- `forward()`: sequential computation of $c_X$ → longitudinal → $r_Y$ → survival
- Generation: follows the explicit order above
- **New modules:**
  - `baseline_summary_net`: MLP $(D_{\text{baseline}} \to 32 \to 16)$
  - `longitudinal_mu_seq`: augmented longitudinal mean layer (input includes $c_X$)
  - Augmented survival theta layers (input includes $c_X + r_Y$)

---

*Last updated: 2026-04-15.*
