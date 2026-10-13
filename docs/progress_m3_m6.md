# Comprehensive Progression Report: Milestones 3 through 6

**Project:** Monte Carlo Simulation Study — Non-linear & Machine Learning Methods in Small-Sample Management/Entrepreneurship Surveys  
**Specification:** `SPEC.md`  
**Working Environment:** Python 3.12 (managed via `uv`)  
**Git Base Commit:** `a50fb46` (Milestones 1–5), updated in Milestone 6  
**Document Generated:** October 13, 2026  

---

## Executive Summary

From Milestone 3 through Milestone 6, the simulation platform progressed from an isolated Data-Generating Process (DGP) into an end-to-end, reproducible, high-performance Monte Carlo simulation suite. The suite includes:

1. **Measurement-layer validity diagnostics and screening tools** (HTMT, Horn's parallel analysis, Lindell-Whitney marker correction, Harman's single-factor test) and an exact noise-free **oracle permutation importance** benchmark.
2. **Classical and regularized linear estimators (M0 to M3)** embedded in a strict **repeated nested cross-validation harness** (5-fold, 2 repeats; 10 outer splits), capturing inferential hypotheses ($H_1, H_2$), parameter recovery (bias, RMSE, coverage, attenuation), and exploratory interaction discovery.
3. **Machine learning models (M4 to M6)**—Random Forest, Gradient Boosting, and Explainable Boosting Machine—equipped with **TreeSHAP interaction matrices**, term importances, and a novel **residual-resampling null test** ($B=50$ to $100$ refits) that strictly preserves additive signal while testing non-linear effects.
4. **Ranking accuracy and stability metrics** against the noise-free oracle, out-of-sample $R^2$ differences benchmarked against Elastic Net, and an empirical timing pilot.
5. **Parallel simulation runner (`src/run.py`)** with `joblib` multi-processing, core-invariant `SeedSequence.spawn` seeding, Parquet sharding (`results/{scenario}.parquet`), automatic resumption, Git/version provenance tracking, replication failure logging, execution tiers, and `--dry-run` task accounting.
6. **Leaky-design demonstration (`src/leaky.py`)** proving why placing outcome items inside an index inflates $\Delta R^2$ by $>30\times$ and drives $H_1$ false-positive rates to $100\%$ under $H_1$-null.

Across all milestones, the test suite contains **48 fast unit tests** (plus 1 marked slow null-calibration test), all passing with $100\%$ success rate.

---

## Milestone 3: Measurement Layer & Oracle Feature Importance

### 1. Scope & Objectives
Implement the measurement-validity screening gates specified in SPEC.md Sections 3.4, 5, 9, and 10, and construct the large-sample noise-free ground truth oracle for feature importance.

### 2. Implementation Details

#### A. Measurement Diagnostics (`src/measurement.py`)
- **Heterotrait-Monotrait Ratio of Correlations (HTMT)**:
  Measures discriminant validity between pairs of constructs $(A, B)$ using Pearson correlations among items:
  $$\text{HTMT}_{AB} = \frac{\frac{1}{K_A K_B} \sum_{i=1}^{K_A} \sum_{j=1}^{K_B} |r(A_i, B_j)|}{\sqrt{\left(\frac{2}{K_A(K_A-1)} \sum_{i=1}^{K_A-1} \sum_{j=i+1}^{K_A} |r(A_i, A_j)|\right) \left(\frac{2}{K_B(K_B-1)} \sum_{k=1}^{K_B-1} \sum_{l=k+1}^{K_B} |r(B_k, B_l)|\right)}}$$
  A cutoff threshold of $0.85$ flags discriminant validity issues (e.g. cross-loading between $B\text{-}i$ and $A\text{-}i$ or outcome contamination).
- **Horn's Parallel Analysis (`parallel_analysis`)**:
  Determines the number of empirical factors to retain by comparing sample eigenvalues from the correlation matrix against the 95th percentile of eigenvalues from $N_{\text{iter}} = 500$ random uncorrelated datasets of identical sample size and item count.
- **Lindell-Whitney Marker Technique (`lindell_whitney`)**:
  Adjusts the correlation matrix for Common Method Variance (CMV) induced by acquiescence factor $a_i$ using an uninformative marker variable $M$:
  $$r_{ij,\text{adj}} = \frac{r_{ij} - r_M}{1 - r_M}$$
  where $r_M$ is the minimum or mean absolute correlation between marker items and substantive items.
- **Harman's Single-Factor Test (`harman_first_factor_share`)**:
  Extracts the first unrotated principal factor via eigendecomposition:
  $$\text{Share} = \frac{\lambda_1}{\sum_{j} \lambda_j}$$
  Flags CMV if $\text{Share} > 0.50$ (used in the simulation to demonstrate the known insensitivity/weakness of Harman's test).

#### B. Oracle Feature Importance (`src/oracle.py`)
- Simulates an infinite-sample population ($N_{\text{pop}} = 200,000$) directly through the noise-free latent function:
  $$f^*(X) = \beta^T X_{\text{linear}} + g_{\text{nonlinear}}(X)$$
- Performs permutation feature importance on the noise-free signal.
- Produces the ground-truth feature ranking:
  $$\text{Oracle Ranking} = [A_{\text{tech}}, B\text{-}i, A_{\text{mkt}}, B\text{-}iii, B\text{-}ii]$$
  matching the absolute standardized theoretical weights ($0.30, 0.20, 0.15, 0.15, 0.10$).

#### C. Validation Results
- At $r_M = 0$, Lindell-Whitney returns the input matrix unchanged ($0.0$ max absolute difference).
- Single-factor synthetic data yields Harman first-factor share $> 0.65$ ($> 0.50$ threshold).
- Parallel analysis recovers exactly 2 factors on 2-factor latent data and 0 factors on pure Gaussian noise.
- HTMT flag rates evaluated at $n=120$ and $n=150$ under $0.3$ cross-loading confirm high screening power ($>85\%$ flag rate), satisfying SPEC.md line 192.

---

## Milestone 4: Classical Models (M0–M3), Nested CV Harness & Inference Metrics

### 1. Scope & Objectives
Implement estimators M0 through M3, the 10-split repeated nested cross-validation harness, hypothesis testing metrics ($H_1, H_2$), estimation quality metrics, and exploratory interaction selection.

### 2. Implementation Details

#### A. Estimators (`src/methods/`)
- **M0 Baseline (`m0_baseline.py`)**:
  Predicts the training-fold target mean $\bar{y}_{\text{train}}$ for all test observations. Establishes the floor benchmark for out-of-sample $R^2$.
- **M1 Hierarchical OLS (`m1_hierarchical.py`)**:
  * **Block 1 (Contextual)**: Regresses $C$ on contextual sub-constructs ($B\text{-}i, B\text{-}ii, B\text{-}iii$) plus covariates (`semester`, `covar_prototype`, `covar_hackathon`, `covar_internship`).
  * **Block 2 (Individual)**: Adds individual sub-constructs ($A_{\text{tech}}, A_{\text{mkt}}$).
  * **$H_1$ Tests**:
    1. $F$-test on block $\Delta R^2$: $F = \frac{(R^2_2 - R^2_1)/2}{(1 - R^2_2)/(n - p_2)}$ with $\alpha = 0.05$.
    2. $t$-test on $\beta(A_{\text{tech}})$ with HC3 robust standard errors.
  * **$H_2$ Linear Contrast**:
    Tests $H_0: \beta(B\text{-}i) - \beta(B\text{-}ii) \le 0$ using HC3 robust covariance:
    $$t = \frac{\hat{\beta}_{B\text{-}i} - \hat{\beta}_{B\text{-}ii}}{\sqrt{\text{Var}(\hat{\beta}_{B\text{-}i}) + \text{Var}(\hat{\beta}_{B\text{-}ii}) - 2\text{Cov}(\hat{\beta}_{B\text{-}i}, \hat{\beta}_{B\text{-}ii})}}$$
    Rejection requires $p < 0.05$ **and** correct sign ($\hat{\beta}_{B\text{-}i} > \hat{\beta}_{B\text{-}ii}$).
  * **M1-adj (Marker-Adjusted OLS)**:
    Recomputes survey correlations using the Lindell-Whitney marker correction and solves $\hat{\beta}_{\text{adj}} = R_{XX,\text{adj}}^{-1} r_{Xy,\text{adj}}$, assessing shifts in inferential conclusions.
- **M2a Pre-Specified Interaction (`m2_interaction.py`)**:
  Adds the single planted interaction term matching the DGP:
  * Interaction DGP: $A_{\text{tech}} \times B\text{-}i$
  * Threshold DGP: $A_{\text{tech}} \times B\text{-}iii$
  Evaluated with HC3 standard errors, bounding what classical OLS can achieve under perfect oracle guidance.
- **M2b All Pairwise Products (`m2_interaction.py`)**:
  Adds all 10 pairwise product terms among the 5 substantive predictors ($A_{\text{tech}}, A_{\text{mkt}}, B\text{-}i, B\text{-}ii, B\text{-}iii$). Applies **Holm-Bonferroni step-down correction** across the 10 $p$-values. Detects an interaction if $\min p_{\text{corrected}} < 0.05$.
- **M3 Elastic Net with Stability Selection (`m3_elasticnet.py`)**:
  * Features: 5 main effects, 10 pairwise products, 4 covariates.
  * Hyperparameter tuning: Inner CV grid across $l_1 \in \{0.5, 0.9, 1.0\}$ and path $\alpha$ chosen via the **1-SE rule** (most parsimonious model within 1 SE of minimum MSE).
  * Stability Selection: 100 bootstrap refits; an interaction is detected if any product term is retained with selection frequency $\ge 70\%$.

#### B. Repeated Nested Cross-Validation Harness (`src/harness.py`)
- **Structure**: Outer 5-fold cross-validation repeated twice (10 splits total) using `RepeatedKFold(n_splits=5, n_repeats=2)`.
- **Leakage Prevention**: All feature standardization ($z$-scoring) and inner CV tuning occur strictly inside the training folds. Outer test folds are held completely out.
- **Prediction Aggregation**: Out-of-sample predictions $\hat{y}_i$ are pooled across folds to calculate out-of-sample $R^2_{\text{oos}}$ and $\text{RMSE}_{\text{oos}}$:
  $$R^2_{\text{oos}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  (which can honestly be negative for overfit models).
- **Full Model Fits**: Executed on the entire sample for parameter inference, hypothesis testing, and variable selection.

#### C. Metrics Collected (`src/metrics.py`)
- **Metric A (Inference)**: $H_1$ block power, $H_1$ $t$-test power, $H_2$ contrast power, $\Delta R^2$.
- **Metric E (Estimation Quality)**: Parameter bias ($\hat{\beta} - \beta^*$), RMSE, 95% CI coverage, and attenuation ratio ($\hat{\beta}/\beta^*$).
- **Metric B (OLS Detection)**: Detection flags, test statistics, $p$-values, Top-1 interaction pair.
- **Metric D (Prediction)**: Out-of-sample $R^2_{\text{oos}}$, $\text{RMSE}_{\text{oos}}$, difference vs M3 benchmark.
- **Metric H (Method Bias)**: Attenuation in $\beta(A_{\text{tech}})$ and change in $H_1/H_2$ conclusions between M1 and M1-adj.

#### D. Validation & Control Tests (`tests/test_methods.py`)
1. **Negative Control**: With outcome $C$ replaced by pure Gaussian noise, Type I error rate across 300 replications lands at $0.05 \pm 0.013$, within Monte Carlo error.
2. **Positive Control**: With a strong linear signal at $n = 300$, power for $H_1$ and $H_2$ approaches $1.00$.
3. **Leak Check**: Shuffling test-fold outcomes yields $R^2_{\text{oos}} \le 0$, proving zero label leakage across outer splits.

---

## Milestone 5: Machine Learning Models (M4–M6), TreeSHAP & Residual Resampling

### 1. Scope & Objectives
Implement non-linear models M4 (Random Forest), M5 (Gradient Boosting), and M6 (Explainable Boosting Machine), TreeSHAP interaction metrics, the shared residual-resampling interaction test, ranking stability metrics, and the empirical timing pilot.

### 2. Implementation Details

#### A. Machine Learning Estimators
- **M4 Random Forest (`src/methods/m4_random_forest.py`)**:
  * 300 trees, `bootstrap=True`.
  * Tuned by out-of-bag $R^2$ (`oob_score_`) over `min_samples_leaf` $\in \{3, 5, 10\}$.
  * Respects low-capacity constraint for small sample sizes ($n \le 150$).
  * Interaction matrices computed via TreeSHAP interaction values ($\frac{1}{n} \sum_i |\phi_{i, j, k}|$).
- **M5 Gradient Boosting (`src/methods/m5_gradient_boosting.py`)**:
  * `learning_rate=0.03`, `subsample=0.7`, `min_samples_leaf=5`.
  * Inner CV tuning over `max_depth` $\in \{1, 2, 3\}$ and `n_estimators` $\in \{100, 200, 400\}$.
  * Interaction matrices computed via TreeSHAP interaction values.
- **M6 Explainable Boosting Machine (`src/methods/m6_ebm.py`)**:
  * Uses `interpret.glassbox.ExplainableBoostingRegressor`.
  * Hyperparameters: `interactions=10`, `max_bins=32`, `learning_rate=0.01`, `outer_bags=8`, `min_samples_leaf=3`.
  * Hyperparameters fixed to maintain small-$n$ computational feasibility.
  * Term importances extracted directly from EBM interaction components.

#### B. Shared Residual-Resampling Interaction Test (`src/methods/residual_resampling.py`)
- **Motivation (SPEC.md line 101)**:
  Permuting $y$ destroys both main effects and interaction effects. A non-linear tree-based estimator trained on pure noise would discover spurious "interactions" simply because the baseline additive signal was eliminated. The null distribution must preserve the linear additive signal and eliminate only the non-linear interaction.
- **Algorithm**:
  1. Fit additive OLS model: $\hat{y}_{\text{add}} = X \hat{\beta}$, residuals $e = y - \hat{y}_{\text{add}}$.
  2. Compute observed interaction statistic:
     $$T_{\text{obs}} = \max_{j < k} \text{Interaction}(j, k)$$
  3. Generate $B$ null datasets ($B=50$ or $100$):
     $$y^* = \hat{y}_{\text{add}} + \pi_b(e)$$
     where $\pi_b(e)$ is a random permutation of the residuals.
  4. Refit the non-linear estimator on $(X, y^*)$ and compute null statistic $T^*_b$.
  5. Compute exact empirical $p$-value:
     $$p = \frac{1 + \sum_{b=1}^B \mathbf{1}[T^*_b \ge T_{\text{obs}}]}{B + 1}$$
  6. Flag interaction detected if $p < 0.05$.
- **Validation**:
  Fast unit tests verified that $R^2(X, y^*) \approx R^2(X, y)$ (additive signal preserved), while interaction curvature is eliminated.

#### C. Metrics C & D (`src/metrics.py`)
- **Metric C (Ranking & Stability)**:
  * Kendall's rank correlation $\tau$ between estimated feature importance and the noise-free oracle ranking.
  * Top-1 and Top-2 feature recovery rates.
  * 50-bootstrap ranking stability: mean Kendall's $\tau$ between original fit ranking and 50 bootstrap resample rankings.
  * $H_3$ check rate: records how often naive exploratory rules flag an interaction under $\psi = 0$.
- **Metric D (Prediction Differences)**:
  * Out-of-sample $R^2$ difference: $\Delta R^2 = R^2_{\text{oos}}(M_k) - R^2_{\text{oos}}(M_3)$.

#### D. Timing Pilot (`scripts/time_pilot.py`)
Executed across 5 replications per method at $n = 100$ and $n = 150$:
- M0: $< 0.0001\text{s}$ / rep
- M1: $\approx 0.006\text{s}$ / rep
- M2a: $\approx 0.005\text{s}$ / rep
- M2b: $\approx 0.070\text{s}$ / rep
- M3: $\approx 0.100\text{s}$ / rep
- M4: $\approx 1.115\text{s}$ / rep
- M5: $\approx 4.825\text{s}$ / rep (dominated by inner 5-fold CV grid search)
- M6: $\approx 1.090\text{s}$ / rep

---

## Milestone 6: Runner, Tiers, Sharding, Resumption & Leaky-Design Demonstration

### 1. Scope & Objectives
Implement `src/run.py` to drive the large-scale simulation loop with `joblib` multi-processing, Parquet sharding, deterministic seeding, error isolation, resume capabilities, configuration tiers, and `--dry-run` task accounting. Implement `src/leaky.py` to empirically demonstrate the consequences of endogenous index construction under SPEC.md Section 7.

### 2. Implementation Details

#### A. Scenario Catalog (`src/dgp.py`)
Expanded `src/dgp.py` with `get_scenario_catalog()` building the complete 60-scenario grid from SPEC.md Section 4:
- **Core Grid (28 scenarios)**:
  * Sample sizes: $n \in \{50, 100, 150, 300\}$.
  * 5 non-linearity forms: `none` ($\psi = 0$), `inter_small` ($\psi = 0.03$), `inter_med` ($\psi = 0.08$), `thresh_small` ($\psi = 0.03$), `thresh_med` ($\psi = 0.08$).
  * 2 null hypotheses: `h1_null` ($\beta_A = 0$), `h2_null` ($\beta_{B\text{-}i} = \beta_{B\text{-}ii} = 0.15$).
  * $4 \times (5 + 2) = 28$ scenarios.
- **Sensitivity Grid (32 scenarios)**:
  * Sample sizes: $n \in \{100, 150\}$.
  * 2 non-linearity forms: `none` ($\psi = 0$), `inter_med` ($\psi = 0.08$).
  * 8 sensitivity variations: `corr_high` (0.50), `corr_low` (0.10), `low_reliability` ($\lambda \in [0.40, 0.65]$), `r2_low` ($R^2 = 0.20$), `acquiescence` ($\gamma = 0.3$), `ceiling_shift` (0.6), `cross_loading` (0.3), `contaminated` ($\kappa = 0.4$).
  * $2 \times 2 \times 8 = 32$ scenarios.
- **Total**: $28 + 32 = 60$ distinct scenarios.

#### B. Experiment Runner (`src/run.py`)
- **Parallel Dispatch**:
  Outer loop parallelized across replications with `joblib.Parallel(n_jobs=n_jobs)`.
- **Deterministic Child Seeds**:
  ```python
  scenario_seq = np.random.SeedSequence([master_seed, sc_idx])
  child_seeds = scenario_seq.spawn(n_replications)
  ```
  Guarantees that replication $r$ for scenario $s$ receives an identical seed whether run on 1 core or 32 cores.
- **Parquet Sharding & Resumption**:
  * Saves to `results/{scenario_name}.parquet`.
  * Resumption: `check_shard_completed` inspects Parquet table row count; skips existing completed shards automatically unless `--force` is provided.
- **Embedded & Sidecar Metadata**:
  * Pinned package versions (`numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`, `shap`, `interpret`, `joblib`, `pyarrow`, `matplotlib`), Git commit hash, and scenario configuration serialized safely (converting all NumPy ndarrays to Python primitives).
  * Stored in Parquet schema metadata (`table.schema.metadata`) and JSON companion file (`results/{scenario}_meta.json`).
- **Failure Logging**:
  * Traps any replication exception.
  * Formats full stack trace with timestamp, scenario name, and replication number into `results/failures.log`.
  * Returns a structured failure row (`status="FAILED"`, `error_type=...`).
- **Dry-Run Engine**:
  * Command: `uv run python src/run.py --dry-run [--tier tier1 | --tier tier2]`
  * Queries existing shards, counts pending replications, loads timing-pilot benchmarks, and prints a formatted summary table without running tasks.

#### C. Execution Tiers (`config/methods.yaml`)
```yaml
tiers:
  tier1:
    name: "tier1"
    description: "OLS family on all scenarios"
    methods: ["M0", "M1", "M2a", "M2b", "M3"]
    replications: 1000
    scenarios: "all"
    run_ml_interaction_tests: false
    run_stability_selection: true
  tier2:
    name: "tier2"
    description: "Reduced ML grid"
    methods: ["M0", "M1", "M2a", "M2b", "M3", "M4", "M5", "M6"]
    replications: 300
    scenarios: "reduced_ml"
    run_ml_interaction_tests: true
    b_null_refits: 50
    run_stability_selection: true
```

#### D. Leaky-Design Demonstration (`src/leaky.py`)
Implements SPEC.md Section 7 to quantify the methodological bias in the paper's original design:
- **Design Comparison**:
  * Block 1: Contextual predictors ($B\text{-}i, B\text{-}ii, B\text{-}iii$) + covariates regressed on $C$.
  * Block 2 (Correct): Adds Individual Resilience Index built strictly from $A$ items:
    $$\text{Index}_{\text{correct}} = \text{mean}(A\text{-items})$$
  * Block 2 (Leaky): Adds Individual Resilience Index that also includes $C$ items:
    $$\text{Index}_{\text{leaky}} = \text{mean}(A\text{-items} + C\text{-items})$$
- **Simulation Results ($n = 150$, $H_1$-null)**:
  $$\begin{array}{lcccc}
  \hline
  \textbf{Design} & \textbf{Mean } \Delta R^2 & \textbf{SE } (\Delta R^2) & \textbf{False-Positive Rate (FPR)} & \textbf{MCSE} \\
  \hline
  \text{Correct Design} & 0.0059 & 0.0011 & 0.080 & 0.038 \\
  \text{Leaky Design}   & 0.1809 & 0.0071 & 1.000 & 0.000 \\
  \hline
  \end{array}$$
  * **$\Delta R^2$ Inflation Ratio**: $\mathbf{30.46\times}$
  * **Inferential Impact**: The leaky design turns a nominal $5\%$ Type I error rate into a certain ($100\%$) false rejection of the null, demonstrating why keeping $C$ items outside the index is vital.

---

## Complete Verification & Test Suite Summary

The test suite consists of **48 active unit tests** across 9 test modules:

```bash
$ uv run pytest -m "not slow"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/envee/Projects/fyp/monte-carlo-simulation
configfile: pyproject.toml
testpaths: tests
plugins: dash-4.4.1
collected 49 items / 1 deselected / 48 selected

tests/test_htmt.py ..                                                    [  4%]
tests/test_item_distributions.py ..                                      [  8%]
tests/test_marker_and_loadings.py ....                                   [ 16%]
tests/test_measurement.py ....                                           [ 25%]
tests/test_methods.py .....                                              [ 35%]
tests/test_ml_methods.py ....                                            [ 43%]
tests/test_moments.py ......                                             [ 56%]
tests/test_oracle.py .                                                   [ 58%]
tests/test_reliability.py ..                                             [ 62%]
tests/test_run_and_leaky.py ........                                     [ 79%]
tests/test_variance_shares.py .........                                  [ 97%]
tests/test_visual_check.py .                                             [100%]

================= 48 passed, 1 deselected in 101.18s (0:01:41) =================
```

### Key Validation Checks Established
1. **Parallel Invariance**: Output DataFrames are bit-for-bit identical when running with `n_jobs = 1` vs `n_jobs = 4`.
2. **Resumption**: Re-running a completed scenario shard returns immediately without recomputing or altering file timestamps; `--force` successfully recomputes and overwrites.
3. **Provenance Integrity**: Parquet schema metadata and JSON sidecars reliably log the exact git commit hash and all 10 dependency package versions.
4. **Fault Tolerance**: Raising exceptions inside individual replications are cleanly trapped, logged to `results/failures.log`, and marked `FAILED` without crashing the parallel batch.
5. **Statistical Correctness**:
   - Negative controls yield nominal $\alpha \approx 0.05$.
   - Positive controls achieve power $\approx 1.00$.
   - Residual resampling preserves additive signal while removing non-linear curvature.
   - Leaky design exhibits $>30\times$ $\Delta R^2$ inflation and $100\%$ false-positive rate.

---

## Repository Architecture

```
monte-carlo-simulation/
├── config/
│   ├── scenarios.yaml           # DGP parameters, loadings, grid presets
│   └── methods.yaml             # CV rules, hyperparameters, tiers, timing benchmarks
├── scripts/
│   └── time_pilot.py            # Empirical runtime pilot for n in {100, 150}
├── src/
│   ├── dgp.py                   # Latent factors, outcome, measurement model, 60-scenario catalog
│   ├── harness.py               # 10-split repeated nested CV harness
│   ├── leaky.py                 # SPEC Section 7 leaky vs correct design comparison
│   ├── measurement.py           # HTMT, Horn's parallel analysis, Lindell-Whitney, Harman
│   ├── metrics.py               # Metrics A, B, C, D, E, H computation
│   ├── oracle.py                # Infinite-sample permutation feature importance
│   ├── run.py                   # Parallel simulation runner, sharding, resumption, dry-run
│   └── methods/
│       ├── m0_baseline.py       # Training-fold mean benchmark
│       ├── m1_hierarchical.py   # Hierarchical OLS (Block F-test, t-test, contrast, M1-adj)
│       ├── m2_interaction.py    # Pre-specified OLS (M2a) & Holm-corrected all-pairs (M2b)
│       ├── m3_elasticnet.py     # ElasticNetCV (1-SE rule) & 100-bootstrap stability selection
│       ├── m4_random_forest.py  # Random Forest with OOB tuning & TreeSHAP
│       ├── m5_gradient_boosting.py # Gradient Boosting with inner-CV grid & TreeSHAP
│       ├── m6_ebm.py            # Explainable Boosting Machine with term importances
│       └── residual_resampling.py # Residual-resampling null test for non-linear detection
├── tests/
│   ├── test_htmt.py             # HTMT false-flag rates and sensitivity
│   ├── test_item_distributions.py # Category proportions and ceiling shift checks
│   ├── test_marker_and_loadings.py # Equispaced/random loadings and Lindell-Whitney checks
│   ├── test_measurement.py      # Unit tests for HTMT, parallel analysis, marker, Harman
│   ├── test_methods.py          # Classical estimators, negative/positive controls, leakage
│   ├── test_ml_methods.py       # ML estimators, TreeSHAP, residual resampling
│   ├── test_moments.py          # Moment matching on N=200,000 population
│   ├── test_oracle.py           # Oracle importance ranking vs standardized weights
│   ├── test_reliability.py      # Empirical Cronbach's alpha and McDonald's omega bands
│   ├── test_run_and_leaky.py    # Runner resumption, n_jobs invariance, metadata, leaky demo
│   └── test_variance_shares.py  # Psi variance shares and target R^2 calibration
├── NOTES.md                     # Technical decisions, audit log, mathematical justifications
└── progression done.md          # Complete narrative summary of Milestones 3 through 6
```
