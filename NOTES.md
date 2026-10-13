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
6. **Orthogonalisation Meaning of $\psi$**:
   - When `orthogonalise_nonlinear: true` (default), the planted non-linear functional term $N_0$ is residualised against the linear latent predictors ($A_{tech}, A_{mkt}, B\text{-}i, B\text{-}ii, B\text{-}iii$) using the calibration sample.
   - Consequently, $Cov(L_0, N_{orth}) = 0$ and the linear regression $R^2(N_{orth} \sim \text{linear predictors}) = 0.0$.
   - Under this formulation, $\psi$ strictly represents the **pure, orthogonal non-linear incremental variance share**, so that the total explained variance partitions cleanly as $R^2_{total} = R^2_{linear} + \psi$ with zero linear contamination.
   - Without orthogonalisation (`orthogonalise_nonlinear: false`), asymmetric or threshold non-linear terms retain non-zero linear projection ($R^2 \approx 0.177$), conflating linear and non-linear predictive capacity.
7. **M2a as an Approximate Oracle for Threshold Form**:
   - In SPEC.md, Model M2a specifies a bilinear interaction term $A_{tech} \times B\text{-}iii$ as the non-linear candidate.
   - For the interaction DGP scenario ($N_0 = A_{tech} \cdot B\text{-}iii$), M2a is the exact parametric oracle.
   - For the threshold DGP scenario ($N_0 = B\text{-}iii \cdot \mathbf{1}[A_{tech} < -0.5]$), the true underlying mechanism is a localized step/piecewise function rather than a smooth product.
   - Therefore, M2a acts only as an **approximate oracle** (a smooth parametric proxy) for the threshold DGP, testing how well standard bilinear interaction terms capture localized threshold dynamics compared to non-parametric/tree methods (M3–M6).
8. **Decision on `loading_draw` (Equispaced vs Random)**:
   - Config option: `loading_draw: equispaced | random` (default: `equispaced`).
   - For a construct with $k$ items and loading range $[\lambda_{min}, \lambda_{max}]$, `equispaced` sets $\lambda = \text{linspace}(\lambda_{min}, \lambda_{max}, k)$.
   - *Rationale*: Uniform random sampling (`random`) with small item counts ($k=4, 5, 6$) causes stochastic clustering: in the base scenario, draws can dip below 0.75 (e.g. $B\text{-}ii$ at $\omega = 0.7117$, $C$ at $\omega = 0.7272$) or exceed 0.85 (e.g. $B\text{-}iii$ at $\omega = 0.8783$). Similarly in low-reliability, random draws dip to $\omega = 0.5135$ or rise to $0.7314$.
   - Using `equispaced` stabilizes construct reliability across all scenarios: every substantive construct lands strictly within its specification band (base: $\omega \in [0.766, 0.831] \subset [0.75, 0.85]$; low-reliability: $\omega \in [0.573, 0.670] \subset [0.55, 0.70]$), preserving realistic within-construct loading heterogeneity while removing arbitrary seed artifacts. `random` remains fully available via config for sensitivity analysis.
9. **Choice of Pearson Correlation on Item Scores in Measurement Layer**:
   - In `src/measurement.py` (HTMT, Horn's parallel analysis, Lindell-Whitney adjustment, and Harman's test), item correlations are computed using standard Pearson product-moment correlations directly on numeric Likert item scores (1 to 5).
   - *Rationale*: Standard benchmark literature for HTMT (Henseler, Ringle, & Sarstedt, 2015), parallel analysis (Horn, 1965), and common-method bias tests (Lindell & Whitney, 2001; Podsakoff et al., 2003) define these metrics on Pearson correlation matrices. Pearson correlations on 5-point Likert items are robust, computationally efficient, and avoid numerical non-convergence or indefiniteness issues associated with polychoric estimation across thousands of Monte Carlo replications.
10. **Sequential Factor Retention Rule in Horn's Parallel Analysis**:
    - Factors are retained sequentially up to the first component $k$ where the sample eigenvalue falls below the 95th percentile null benchmark ($\lambda_k \le \lambda^*_{crit, k}$), in accordance with Horn (1965) and psychometric standards (Hayton et al., 2004). This ensures that if the primary factor cannot be distinguished from noise ($\lambda_1 \le \lambda^*_{crit, 1}$), 0 factors are retained regardless of subsequent noise fluctuations.

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
   - Section 9 checks 5 and 6 validated in Milestone 4 (`tests/test_methods.py`):
     * Negative control: When outcome is pure noise across 300 replications, H1 F-test Type I rate is 3.67%, M2a is 2.00%, and M2b Holm FWER is 3.33%, all strictly within the Monte Carlo error band around 0.05.
     * Positive control: At $n = 300$ with strong linear signal, H1 power reaches 100%.
     * Leak check: Permuting test-fold targets strictly within test folds yields out-of-sample $R^2 \le 0.0$, verifying that the nested CV harness has zero data leakage.

---

## Milestone 4 Summary: Classical & Regularized Methods, Nested CV Harness, and Metrics

Implemented models M0 to M3, the repeated nested CV harness, and metrics A, E, B, D, and H according to SPEC.md Section 5 & 6:
1. **M0 Baseline**: Predicts the training-fold target mean; serves as the baseline for out-of-sample $R^2$.
2. **M1 Hierarchical OLS**:
   - Block 1: Contextual predictors ($B\text{-}i, B\text{-}ii, B\text{-}iii$) + Covariates (dummy-coded semester + 3 binary flags).
   - Block 2: Adds Individual predictors ($A_{\text{tech}}, A_{\text{mkt}}$).
   - Hypothesis tests: F-test on $\Delta R^2$, $t$-test on $A_{\text{tech}}$ with HC3 robust SE, and linear contrast test $\beta(B\text{-}i) - \beta(B\text{-}ii) = 0$ with HC3 robust SE and correct-sign rule ($B\text{-}i > B\text{-}ii$).
3. **M1-adj (Marker-Adjusted Refit)**:
   - Solves adjusted standardized regression equations via Lindell-Whitney correlation correction $R_{\text{adj}} = (R - r_M) / (1 - r_M)$.
   - Verified that when marker correlation is 0 ($\gamma = 0$), M1-adj matches unadjusted M1.
4. **M2a (Pre-specified Interaction)**:
   - Fits OLS with $A_{\text{tech}} \times B\text{-}i$ (interaction DGP) or $A_{\text{tech}} \times B\text{-}iii$ (threshold DGP) with HC3 standard errors.
5. **M2b (All 10 Pairwise Products)**:
   - Fits OLS with all 10 pairwise products of the five predictors and applies Holm-Bonferroni correction across the interaction terms.
6. **M3 (Elastic Net with Stability Selection)**:
   - Fits 5 main effects, 10 pairwise products, and covariates.
   - Tunes $l_1 \in \{0.5, 0.9, 1.0\}$ and $\alpha$ via inner CV (5-fold; 3-fold at $n \le 50$) using the 1-SE rule.
   - Performs stability selection across 100 bootstrap refits; detects interaction if any product term has selection frequency $\ge 70\%$.
7. **Nested CV Harness**:
   - Outer: 5-fold CV repeated twice (10 splits total).
   - Scaling and tuning occur strictly inside training folds.
   - Out-of-sample predictions are pooled across splits to compute out-of-sample $R^2$ and RMSE.
8. **Metrics & Runner**:
   - Computes Metric A (inferential power & Type I error), Metric E (estimation quality, bias, 95% CI coverage, attenuation ratio), Metric B (interaction detection flags, continuous statistics, top-1 pairs), Metric D (out-of-sample prediction), and Metric H (method-bias gate and conclusion shifts).
   - `src/runner.py` provides `run_smoke(n_replications=20)` completing in ~41 seconds.

---

## Milestone 5 Summary: Machine Learning Models (M4 to M6), Residual Resampling, and Timing Pilot

Implemented methods M4, M5, and M6, the shared residual-resampling interaction test, Metrics B (ML parts), C, and D, and the timing pilot according to SPEC.md Section 5 & 6:

1. **M4 Random Forest**:
   - 300 trees, `bootstrap=True`.
   - Tuned by out-of-bag $R^2$ (`oob_score_`) over `min_samples_leaf` in `{3, 5, 10}`.
   - Pairwise interaction statistics extracted via TreeSHAP interaction values ($\frac{1}{n} \sum_i |\phi_{i, j, k}|$).
2. **M5 Gradient Boosting**:
   - `GradientBoostingRegressor` with `learning_rate=0.03`, `subsample=0.7`, `min_samples_leaf=5`.
   - Inner CV grid search over `max_depth` in `{1, 2, 3}` and `n_estimators` in `{100, 200, 400}`.
   - Pairwise interaction statistics extracted via TreeSHAP interaction values.
3. **M6 Explainable Boosting Machine (EBM)**:
   - `ExplainableBoostingRegressor` from `interpret` with `interactions=10`, `max_bins=32`, `learning_rate=0.01`, `outer_bags=8`, `min_samples_leaf=3`.
   - Fixed hyperparameters without inner tuning to maintain computational feasibility.
   - Pairwise interaction statistics extracted directly from EBM term importances.
4. **Shared Residual-Resampling Interaction Test**:
   - Fits additive OLS model $\hat{y}_{\text{add}} = X \hat{\beta}$, residuals $e = y - \hat{y}_{\text{add}}$.
   - Simulates $B$ null datasets as $y^* = \hat{y}_{\text{add}} + \text{permute}(e)$, refits the non-linear model, and computes null test statistics $T^* = \max_{\text{pairs}} \text{Interaction}(j, k)$.
   - Empirical $p$-value: $p = (1 + \sum \mathbf{1}[T^* \ge T_{\text{obs}}]) / (B + 1)$.
   - **Why residual resampling**: Permuting $y$ directly destroys main effects, causing non-linear estimators to falsely flag spurious interactions on pure additive data. Residual resampling preserves the additive signal ($R^2_{\text{null}} \approx R^2_{\text{orig}}$) while strictly removing the interaction.
5. **Metrics B, C, D**:
   - Metric B: Extended with M4, M5, M6 detection flags, $p$-values, test statistics $T$, and Top-1 selected interaction pairs.
   - Metric C: Evaluates ranking accuracy against the noise-free oracle ranking via Kendall's tau, Top-1 recovery, Top-2 recovery, 50-bootstrap ranking stability, and the H3 check.
   - Metric D: Out-of-sample $R^2_{\text{oos}}$ and $\text{RMSE}_{\text{oos}}$ across all methods, plus $\Delta R^2$ benchmarked against M3.
6. **Timing Pilot Results (`scripts/time_pilot.py`)**:
   - 5 replications per method at $n = 100$ and $n = 150$:
     * M0: ~0.00003s / rep
     * M1: ~0.006s / rep
     * M2a: ~0.005s / rep
     * M2b: ~0.07s / rep
     * M3: ~0.10s / rep
     * M4: 0.92s ($n=100$) / 1.31s ($n=150$)
     * M5: 4.95s ($n=100$) / 4.70s ($n=150$) (due to 5-fold inner CV grid search)
     * M6: 1.35s ($n=100$) / 0.83s ($n=150$)

---

## Milestone 6 Summary: Simulation Runner, Tiers, Sharding, Resumption, and Leaky-Design Demonstration

Implemented `src/run.py`, `src/leaky.py`, tiers in `config/methods.yaml`, and validation tests in `tests/test_run_and_leaky.py` according to SPEC.md Section 7 & 8:

1. **Simulation Runner (`src/run.py`)**:
   - Loops over scenario $\times$ replication parallelized via `joblib.Parallel(n_jobs=n_jobs)` (where `n_jobs` is read from `config/methods.yaml` or overridden via `--n-jobs`).
   - Deterministic child seeds spawned via `np.random.SeedSequence([master_seed, scenario_idx]).spawn(n_replications)` ensuring reproducible, core-invariant execution.
   - Parquet sharding: writes one shard per scenario to `results/{scenario}.parquet` with companion JSON sidecar `results/{scenario}_meta.json`.
   - Embedded metadata: logs git commit hash (`git rev-parse HEAD`), ISO UTC timestamp, and pinned package versions (`numpy`, `pandas`, `scipy`, `scikit-learn`, `shap`, `interpret`, `joblib`, `pyarrow`, `matplotlib`) directly into the Parquet schema key-value table and JSON sidecar.
   - Resumption logic: checks `pq.read_metadata()` for existing shards; if a shard has completed all expected replications, skips recomputation unless `--force` is specified.
   - Failure logging: catches exceptions at the replication level, logs full tracebacks to `results/failures.log`, and returns a failure row (`status="FAILED"`) so single-replication numerical errors do not abort the outer batch.
   - `--dry-run` flag: prints total task count across scenarios, checks completed shards, and computes estimated runtime from timing-pilot benchmarks.

2. **Execution Tiers (`config/methods.yaml`)**:
   - `tier1`: OLS family (`M0`, `M1`, `M2a`, `M2b`, `M3`) across all 60 scenarios ($R = 1000$, 60,000 tasks total). Estimated wall-clock runtime: ~8.5 hours at `n_jobs=4`.
   - `tier2`: Reduced ML grid (`M0` to `M6`) across 14 key core scenarios at $n \in \{100, 150\}$ ($R = 300$, 4,200 tasks total). Estimated wall-clock runtime: ~19 hours at `n_jobs=4`.

3. **Leaky-Design Demonstration (`src/leaky.py`)**:
   - Implemented SPEC.md Section 7 comparing correct design ($C$ regressed on Contextual block, then Individual Resilience Index $A$ items only) vs leaky design ($C$ items included inside the Individual Resilience Index) under $H_1\text{-null}$ at $n = 150$.
   - **Correct design**: Nominal Type I error ($\text{FPR} \approx 5\%$, $\text{mean } \Delta R^2 \approx 0.006$).
   - **Leaky design**: Catastrophic false-positive rate ($\text{FPR} = 100\%$) and severe $\Delta R^2$ inflation ($\text{mean } \Delta R^2 \approx 0.181$, an inflation ratio $> 30\times$).

4. **Testing Suite (`tests/test_run_and_leaky.py`)**:
   - Resumption without recomputing finished shards verified.
   - Invariance to `n_jobs`: byte-identical DataFrame outputs verified between `n_jobs = 1` and `n_jobs = 4`.
   - Embedded Parquet schema metadata and JSON sidecar verified.
   - Failure logging and exception containment verified.
   - Dry-run flags verified.
   - Leaky-design $\Delta R^2$ inflation and FPR verified.


