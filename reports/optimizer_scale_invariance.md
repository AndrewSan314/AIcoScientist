# Mathematical Analysis of Target-Scale Invariance in Bayesian Optimization

## Abstract & Rationale

In domain-agnostic Bayesian optimization (BO), the chosen acquisition function and trust region controller must exhibit **target-scale invariance** (unit invariance). Specifically, for any positive affine transformation of the objective target:

$$y' = a \cdot y + b \quad \text{where } a > 0, \; b \in \mathbb{R}$$

the underlying mathematical optimization problem is identical. The optimizer's candidate ranking, proposal trajectory, and trust region expansion/contraction dynamics should remain invariant under arbitrary physical measurement units (e.g. converting electrochemical rate constants $k^0$ from $\text{m/s}$ to $\text{cm/s}$, or energy density from $\text{Wh/kg}$ to $\text{J/g}$).

---

## Mathematical Proofs of Scale Invariance

### 1. Gaussian Process Posterior Transformations

Let $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$ denote the GP surrogate fitted on observations $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$. Under target transformation $y'_i = a \cdot y_i + b$:

$$\mu'(x) = a \cdot \mu(x) + b$$
$$\sigma'(x) = a \cdot \sigma(x)$$
$$\Sigma'(X_1, X_2) = a^2 \cdot \Sigma(X_1, X_2)$$

When using standardized target normalization (`normalize_y=True`), the normalized residuals $\tilde{y} = (y - \bar{y}) / s_y$ satisfy:

$$\tilde{y}' = \frac{(a \cdot y + b) - (a \cdot \bar{y} + b)}{a \cdot s_y} = \frac{a(y - \bar{y})}{a \cdot s_y} = \tilde{y}$$

Hence, the normalized latent GP posterior distribution is strictly invariant.

---

### 2. Greedy Exploitation

For maximization:
$$\alpha_{\text{Greedy}}'(x) = \mu'(x) = a \cdot \mu(x) + b$$

Since $a > 0$, for any two candidates $x_1, x_2$:
$$\mu'(x_1) > \mu'(x_2) \iff \mu(x_1) > \mu(x_2)$$

**Conclusion**: Ranking is strictly preserved.

---

### 3. Upper Confidence Bound (GP-UCB)

For maximization with exploration parameter $\beta > 0$:
$$\alpha_{\text{UCB}}'(x) = \mu'(x) + \beta \cdot \sigma'(x) = (a \cdot \mu(x) + b) + \beta \cdot (a \cdot \sigma(x)) = a \cdot (\mu(x) + \beta \cdot \sigma(x)) + b$$

Since $a > 0$ and $b$ is a constant additive shift:
$$\alpha_{\text{UCB}}'(x_1) > \alpha_{\text{UCB}}'(x_2) \iff \alpha_{\text{UCB}}(x_1) > \alpha_{\text{UCB}}(x_2)$$

**Conclusion**: Candidate ranking is strictly invariant under positive affine scaling.

---

### 4. Expected Improvement (EI) with Canonical $\xi = 0.0$

Analytic Expected Improvement evaluates:
$$\text{EI}(x; \xi) = (\mu(x) - y^* - \xi) \Phi(\gamma) + \sigma(x) \phi(\gamma)$$
where $\gamma = \frac{\mu(x) - y^* - \xi}{\sigma(x)}$.

#### Case A: Canonical Default $\xi = 0.0$
Under $y' = a \cdot y + b$:
$$y^{*\prime} = a \cdot y^* + b$$
$$\gamma' = \frac{\mu'(x) - y^{*\prime}}{\sigma'(x)} = \frac{(a \mu(x) + b) - (a y^* + b)}{a \sigma(x)} = \frac{a(\mu(x) - y^*)}{a \sigma(x)} = \gamma$$

Therefore:
$$\Phi(\gamma') = \Phi(\gamma) \quad \text{and} \quad \phi(\gamma') = \phi(\gamma)$$
$$\text{EI}'(x; 0) = a(\mu(x) - y^*) \Phi(\gamma) + a \sigma(x) \phi(\gamma) = a \cdot \text{EI}(x; 0)$$

Since $\text{EI}'(x) = a \cdot \text{EI}(x)$ with $a > 0$, candidate rankings satisfy:
$$\text{EI}'(x_1) > \text{EI}'(x_2) \iff \text{EI}(x_1) > \text{EI}(x_2)$$

#### Case B: Non-Zero Absolute $\xi > 0$ (e.g. $\xi = 0.01$)
When $\xi > 0$, the standardized improvement ratio becomes:
$$\gamma' = \frac{a(\mu(x) - y^*) - \xi}{a \sigma(x)} = \gamma - \frac{\xi}{a \sigma(x)}$$

The second term $\frac{\xi}{a \sigma(x)}$ depends inversely on $a$. If the target scale is changed (e.g. $k^0 \in [0, 0.014]$ vs $[0, 14.0]$), an absolute margin of $\xi = 0.01$ changes from $70\%$ of the entire dynamic range down to $0.07\%$, drastically altering the optimizer's exploration-exploitation trade-off.

**Correctness Decision**: Set default $\xi = 0.0$ across all acquisition functions (`expected_improvement`, `compute_true_mc_nei`, `probability_of_improvement`, `ClosedLoopOptimizer`). Support $\xi > 0$ strictly as an optional domain-specific absolute threshold.

---

### 5. True Monte Carlo Noisy Expected Improvement (True NEI) with $\xi = 0.0$

True NEI draws $K$ joint latent posterior fantasies $\tilde{f} \sim \mathcal{N}(\mu_{\text{joint}}, \Sigma_{\text{joint}})$. Under $y' = a \cdot y + b$:

$$\tilde{f}' = a \cdot \tilde{f} + b$$
$$\tilde{f}^{*\prime} = \max_{i \in \text{obs}} \tilde{f}'(x_i) = a \left( \max_{i \in \text{obs}} \tilde{f}(x_i) \right) + b = a \cdot \tilde{f}^* + b$$

With $\xi = 0.0$, the per-fantasy improvement is:
$$I^{(k)\prime}(x) = \max(0, \, \tilde{f}^{(k)\prime}(x) - \tilde{f}^{*(k)\prime}) = \max(0, \, a(\tilde{f}^{(k)}(x) - \tilde{f}^{*(k)})) = a \cdot I^{(k)}(x)$$

Averaging over $K$ Monte Carlo fantasies with identical pseudo-random seed:
$$\alpha_{\text{NEI}}'(x) = \frac{1}{K} \sum_{k=1}^K I^{(k)\prime}(x) = a \cdot \alpha_{\text{NEI}}(x)$$

**Conclusion**: True NEI ranking is strictly invariant under positive affine scaling.

---

### 6. TuRBO Trust Region Posterior Evidence with $\text{success\_delta} = 0.0$

TuRBO evaluates whether a newly measured candidate represents a statistically significant posterior improvement over the previous incumbent:

$$P(\Delta f > \delta_{\text{succ}} \mid \mathcal{D}_{t+1}) \ge p_{\text{threshold}}$$

where $\Delta f = f_{\text{cand}} - f_{\text{inc}} \sim \mathcal{N}(\Delta \mu, \, \Delta \sigma^2)$ with:
$$\Delta \mu = \mu_{\text{cand}} - \mu_{\text{inc}}$$
$$\Delta \sigma^2 = \text{Var}(f_{\text{cand}}) + \text{Var}(f_{\text{inc}}) - 2\,\text{Cov}(f_{\text{cand}}, f_{\text{inc}})$$
$$z = \frac{\Delta \mu - \delta_{\text{succ}}}{\Delta \sigma}$$

#### With Canonical $\delta_{\text{succ}} = 0.0$:
Under $y' = a \cdot y + b$:
$$\Delta \mu' = (a \mu_{\text{cand}} + b) - (a \mu_{\text{inc}} + b) = a \cdot \Delta \mu$$
$$\Delta \sigma^{\prime 2} = a^2 \text{Var}(f_{\text{cand}}) + a^2 \text{Var}(f_{\text{inc}}) - 2 a^2 \text{Cov}(f_{\text{cand}}, f_{\text{inc}}) = a^2 \cdot \Delta \sigma^2 \implies \Delta \sigma' = a \cdot \Delta \sigma$$
$$z' = \frac{a \cdot \Delta \mu - 0}{a \cdot \Delta \sigma} = \frac{\Delta \mu}{\Delta \sigma} = z$$

Since $z' = z$:
$$p_{\text{succ}}' = \Phi(z') = \Phi(z) = p_{\text{succ}}$$

**Conclusion**: The success probability $p_{\text{succ}}$, expansion counter, contraction counter, trust region length trajectory, and center coordinates are 100% invariant under target scaling.
