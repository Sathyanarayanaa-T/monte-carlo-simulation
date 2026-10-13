# Project Notes & Milestone 1 Summary

## Milestone 1 Summary: Data-Generating Process & Simulator Validation

Milestone 1 implemented the complete Data-Generating Process (DGP) and validation test suite according to SPEC.md Sections 3, 8, and 9, in compliance with all project rules in `agents.md`.

### Deliverables Completed:
1. **uv Project (Python 3.12.13)**: Initialized with pinned dependency lockfile `uv.lock`. Directory structure strictly adheres to SPEC.md Section 8 (`config/`, `src/`, `src/methods/`, `tests/`, `results/`, `report/`).
2. **`config/scenarios.yaml`**: Contains every DGP parameter from Section 3 (latent correlations, individual factor weights, linear outcome weights, non-linear forms, psi variance shares, target R², item counts, loading ranges, reverse-keying indices, acquiescence factor gamma, ceiling shifts, discretisation cut points, cross-loading, outcome contamination kappa, and covariate settings). Placeholder item counts from the spec are marked `# TODO`. `config/methods.yaml` created as a placeholder for Milestone 2+.
3. **`src/dgp.py`**:
   - `generate(scenario, seed)`: generates synthetic survey data returning a `pd.DataFrame` containing Likert item responses (with reverse-keyed items re-coded as `6 - x`), latent factor scores, scored sub-construct means (`score_*`), block indices (`index_individual`, `index_contextual`), and covariates.
   - `draw_scenario_loadings(scenario)`: factor loadings drawn once per scenario and fixed across replications.
   - `calibrate_scenario(scenario)`: performs exact analytical/numerical calibration of variance scaling ($k_{lin}$, $k_{nl}$) and latent G parameters ($w_C$, $\sigma_{uG}$).
   - `spawn_seeds(master_seed, n_scenarios, n_replications)`: uses `np.random.SeedSequence.spawn` for independent child seeds per (scenario, replication) without ever touching the global numpy RNG.
4. **Validation Test Suite (`tests/`)**:
   - `tests/test_moments.py`: validates latent moments at $n = 200,000$ (contextual correlations, unit variances, individual factor residual correlation 0.30, sensitivity correlations 0.10 and 0.50, latent G target correlation ~0.50 with C, and covariate distributions).
   - `tests/test_reliability.py`: validates empirical McDonald's omega and Cronbach's alpha within the target band (0.75 - 0.85 in base; 0.55 - 0.70 in low-reliability sensitivity).
   - `tests/test_variance_shares.py`: validates that linear and non-linear parts explain target R² (0.35 and 0.20) and non-linear part carries variance share psi (0.0, 0.03, 0.08) for all forms (none, interaction, threshold, and null settings).
   - `tests/test_item_distributions.py`: validates no empty categories (1 to 5 all non-empty) and ceiling shift behavior (+0.3 vs +0.6).
   - `tests/test_htmt.py`: validates added HTMT check (false-flag rate HTMT > 0.85 stays <= 5% in base case at $n=150$, and elevated HTMT under cross-loading 0.3 and kappa 0.4).
   - `tests/test_marker_and_loadings.py`: validates marker items load only on acquiescence factor, loadings persist across replications, and index scoring calculations match Section 3.4.
   - `tests/test_visual_check.py`: produces and verifies diagnostic plots saved to `report/diagnostic_plots.png`.

---

## Assumptions Made

1. **Variance Scaling Algebra**:
   - Latent outcome $C = k_{lin} L_0 + k_{nl} N_0 + \epsilon$, where $Var(C) = 1.0$ and $\epsilon \sim N(0, 1 - R^2)$.
   - $k_{nl} = \sqrt{\psi / Var(N_0)}$ ensures $Var(k_{nl} N_0) = \psi$.
   - $k_{lin}$ is solved via the quadratic equation $a k_{lin}^2 + b k_{lin} + c = 0$ where $a = Var(L_0) > 0$, $b = 2 k_{nl} Cov(L_0, N_0)$, and $c = \psi - R^2 < 0$. Because $a > 0$ and $c < 0$, the discriminant $\Delta = b^2 - 4ac > 0$ is strictly positive and yields a unique strictly positive root $k_{lin} = (-b + \sqrt{b^2 - 4ac}) / (2a)$.
   - Added configuration switch `orthogonalise_nonlinear: true` (default true). When true, the planted non-linear term (interaction or threshold) is residualised on $A_{tech}, A_{mkt}, B\text{-}i, B\text{-}ii, B\text{-}iii$ using the calibration sample, so that $\psi$ represents the pure orthogonal non-linear variance share with $R^2 = 0.0$ linear contamination. When false, the unorthogonalised centered threshold term is scaled using the exact quadratic formula.
2. **Latent G Calibration**:
   - $G = 0.35 B_{ii} + 0.15 B_i + w_C C + u_G$.
   - With $Var(C) = 1.0$, $Cov(G, C) = Cov(G_{sub}, C) + w_C$. Setting $Corr(G, C) = 0.50$ sets $w_C = 0.50 - Cov(G_{sub}, C)$.
   - Independent residual variance is set to $\sigma_{uG}^2 = 1.0 - Var(G_{sub} + w_C C)$, ensuring $Var(G) = 1.0$ and $Corr(G, C) = 0.50$ exactly.
3. **Marker Variable M**:
   - Marker items have substantive loading $\lambda_j = 0$ and load only on the acquiescence factor: $x^*_{M, j} = \gamma a_i + e_{M, j}$, where $e_{M, j} \sim N(0, 1)$, then standardized and discretized with the same thresholds.
   - When $\gamma = 0$, marker items are pure independent noise.
4. **Covariates**:
   - Semester: Categorical over $\{6, 7, 8\}$ with uniform probability $1/3$ each.
   - Prior-exposure flags: Binary variables generated as $\mathbf{1}[u < \sigma(0.8 \cdot \theta + \alpha)]$. The intercept $\alpha$ is dynamically solved at scenario build from `slope` and `target_prevalence` via root-finding (`scipy.optimize.root_scalar` convolved with $N(0, 1)$), producing $\alpha \approx -0.462218756$ for exactly 40.0% prevalence without hardcoding numbers in YAML.
5. **Reverse-Keyed Item Keying and Selection**:
   - For 20% reverse-keyed items, rounded to the nearest integer count per construct (1 item per construct for 4, 5, and 6-item constructs).
   - In accordance with SPEC.md Section 3.4, the loading sign is flipped on generation ($\lambda_j \to -\lambda_j$), and the resulting raw response is re-coded as $6 - x_{raw}$.

---

## Open Questions & Ambiguities in SPEC.md

1. **Exact FEDRS Survey Item Counts & Instrument Items**:
   - Section 3.4 lists placeholder item counts (A-i: 5, A-ii: 4, B-i: 5, B-ii: 4, B-iii: 6, C: 4).
   - *Question*: What are the finalized item counts and which specific items in the FEDRS survey correspond to reverse-keyed questions?
2. **Semester Distribution in Cohort**:
   - Section 3.5 states Semester (6, 7, 8) has no true effect on C.
   - *Question*: Is the student cohort evenly split across semesters 6, 7, and 8, or does the institution have an uneven semester distribution?
3. **Ceiling Shift Interaction with Reverse-Keyed Items**:
   - Section 3.4 specifies: "cut x* + shift at -1.5, -0.5, 0.5 and 1.5 to give categories 1 to 5. Base shift +0.3 (mild ceiling, typical of self-rated resilience)."
   - If $+0.3$ is applied to the raw propensity $x^*$, respondents endorse the raw statement more, meaning they endorse negative statements more on reverse-keyed items (a response-scale acquiescence-like shift). Alternatively, if the ceiling reflects higher true self-rated resilience, respondents should endorse the positive pole (disagreeing with negative statements).
   - *Current implementation*: Follows the literal equation `x* + shift` on raw propensity, and then re-codes $6 - x_{raw}$ during scoring.
4. **Pipeline-Level Negative & Positive Controls (Section 9 Checks 5 & 6)**:
   - Section 9 checks 5 and 6 describe running the full analysis pipeline with noise C (Type I rate ~5%) and strong signal at $n=300$ (power -> 1). These checks evaluate models M1–M6 and nested CV, which belong to Milestone 2 & 3.
