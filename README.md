# Monte Carlo Simulation Study: Statistical & Machine Learning Methods for Survey Research

A rigorous, high-performance Monte Carlo simulation suite designed to evaluate the statistical power, Type I error control, and feature recovery of classical econometrics versus modern machine learning in the presence of complex survey measurement structures (Likert discretization, cross-loadings, common method variance, and indicator contamination).

---

## 📌 Research Overview & Study Design

In survey-based management, educational, and social science research, researchers frequently rely on linear additive models (e.g., OLS) while hypothesizing complex moderating or threshold dynamics. This simulation study benchmarks classical econometric estimators against regularized and non-linear machine learning architectures across finite sample regimes ($n \in \{50, 100, 150, 300\}$).

### Evaluated Estimators
* **M0 (Null Benchmark)**: Intercept-only baseline.
* **M1 (Hierarchical OLS)**: Two-block linear model with block-gain $F$-test, $A_{\text{tech}}$ $t$-test, and $B\text{-}i - B\text{-}ii$ HC3-robust contrast test.
* **M1-adj (Marker-Adjusted OLS)**: Lindell-Whitney single-marker adjustment technique to mitigate Common Method Variance (CMV).
* **M2a (Pre-Specified Interaction)**: OLS augmented with one hypothesized product term ($A_{\text{tech}} \times B\text{-}i$ or threshold proxy).
* **M2b (Exploratory Interactions)**: All 10 pairwise product terms with Holm-Bonferroni family-wise error rate control.
* **M3 (Elastic Net + Stability Selection)**: `ElasticNetCV` with 1-SE rule and 100-resample bootstrap stability selection ($\tau = 0.70$).
* **M4 (Random Forest)**: OOB $R^2$-tuned random forest + TreeSHAP interaction strengths + residual-resampling null interaction test ($B=50$).
* **M5 (Gradient Boosting)**: Inner-CV tuned boosted trees + TreeSHAP interaction values + residual-resampling null interaction test ($B=50$).
* **M6 (Explainable Boosting Machine - EBM)**: Interpretable Generalized Additive Model with pairwise interaction terms + residual-resampling null test ($B=50$).

### Methodological Guarantees
1. **Strict Nested Cross-Validation**: Outer 5-fold cross-validation repeated twice ($5 \times 2 = 10$ test evaluations per replication). All transformations (z-scoring, imputation, tuning) execute strictly inside training folds with zero data leakage.
2. **Exact Seed Determinism**: Managed via `np.random.SeedSequence.spawn` yielding unique, uncoupled child seeds per (scenario, replication).
3. **Data Provenance**: Every output Parquet shard records the exact Git commit hash, UTC timestamp, and pinned dependency versions in its schema metadata table.

---

## 📁 Repository Structure

```
├── config/
│   ├── scenarios.yaml       # Scenario grid definitions, loading intervals, structural weights
│   └── methods.yaml         # Hyperparameter grids, CV parameters, parallel workers
├── src/
│   ├── dgp.py               # Data-Generating Process (7 latent constructs, Likert categorisation)
│   ├── measurement.py       # Psychometrics: HTMT, Horn's Parallel Analysis, Lindell-Whitney, Harman
│   ├── oracle.py            # Permutation-based oracle feature importances
│   ├── harness.py           # Nested CV harness and outer evaluation loop
│   ├── metrics.py           # Metrics: OOS R2, Type I error, Top-k recovery (D13), Kendall tau
│   ├── run.py               # Production batch runner with joblib, Parquet sharding & resume
│   ├── leaky.py             # Leaky vs. correct psychometric design demonstration
│   └── methods/             # Implementations for M0 through M6
├── scripts/
│   ├── run_tier.sh          # Production runner with nice -n 10 and systemd-inhibit
│   ├── time_e2e.sh          # End-to-end timing benchmark script
│   ├── compare_timing.py    # Timing calibration and config updater
│   ├── null_calibration.py  # Residual-resampling interaction test Type I calibration check
│   ├── sanity_report.py     # Live diagnostic reporter for finished Parquet shards
│   ├── generate_paper_plots.py # Publication plot (Figs 1–7) & table generator
│   ├── gate_check.py        # Structural-validity gate audit (HTMT, Parallel Analysis)
│   ├── reliability_check.py # Scripted Cronbach's alpha and McDonald's omega audit
│   └── backup_results.sh    # Parquet shard archive utility
├── report/                  # Publication figures (report/figures) and tables (report/tables)
├── reports/                 # Audited empirical verification reports (tracked in Git)
├── tests/                   # Comprehensive unit and integration test suite
├── spec.md                  # Study specification and source of truth
├── NOTES.md                 # Architecture decisions (D1–D25) and milestone log
└── pyproject.toml           # Project configuration & pinned dependencies
```

---

## 🚀 Setup & Installation

The project uses Python 3.12 managed with [`uv`](https://docs.astral.sh/uv/).

### 1. Install `uv` & Dependencies
```bash
# Clone the repository
git clone https://github.com/Sathyanarayanaa-T/monte-carlo-simulation.git
cd monte-carlo-simulation

# Synchronize exact dependencies from uv.lock
uv sync --frozen
```

### 2. Run the Test Suite
Run the fast unit and integration tests (58 tests):
```bash
uv run pytest -m "not slow"
```

---

## 🔬 Simulation Workload & Tiers

The study is partitioned into two execution tiers:

* **Tier 1 (OLS Family — M0, M1, M1-adj, M2a, M2b, M3)**:
  * **Core Group (28 Scenarios)**: Base, $h_1\text{-null}$, and $h_2\text{-null}$ scenarios across $n \in \{50, 100, 150, 300\}$ at 1,000 replications (28,000 tasks).
  * **Sensitivity Group (40 Scenarios)**: Cross-loadings ($\lambda_{\text{cross}} \in \{0.3, 0.5\}$), indicator contamination ($\kappa = 0.4$), common method variance ($\gamma = 0.3$), low reliability, and high latent correlation ($r = 0.60$) at 1,000 replications (40,000 tasks).
* **Tier 2 (Full ML Grid — M0 through M6)**:
  * **14 Scenarios**: Core signal, interaction, and threshold scenarios at 300 replications ($4,200$ tasks). Staged in steps of $R=100$.

---

## 💻 How to Run the Simulation

### Step 1: Real Hardware Timing Calibration (~5 min)
Before running extensive simulations, calibrate execution speeds on your local hardware:
```bash
# 1. Measure real end-to-end task durations
./scripts/time_e2e.sh

# 2. Update timing benchmarks in config
uv run python scripts/compare_timing.py

# 3. View calibrated wall-clock projections
uv run python -m src.run --dry-run --tier tier1 --group core
```

### Step 2: Run Tier 1 Core Scenarios
Run inside a persistent `tmux` session to ensure uninterrupted execution:
```bash
tmux new -s sim

# Set single-threading for sub-libraries to prevent CPU oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run core scenarios staged at 300 replications
nice -n 10 uv run python -m src.run --tier tier1 --group core --max-reps 300
```

#### Monitor Progress & Check Sanity
In a separate terminal, inspect live results as shards complete:
```bash
uv run python scripts/sanity_report.py --group core
```
* **Go / No-Go Checks**:
  * `FAILED` count must be `0`.
  * Type I error rates on null scenarios must fall in $[0.02, 0.08]$ (nominal $\alpha = 0.05$).
  * $H_1$ detection power must increase with sample size $n$.

#### Extend to Full 1,000 Replications
Once validated, extend to the full 1,000 replications. The runner will **automatically resume**, evaluating only the remaining 700 replications:
```bash
nice -n 10 uv run python -m src.run --tier tier1 --group core --max-reps 1000
```

### Step 3: Run Tier 1 Sensitivity Scenarios
```bash
nice -n 10 uv run python -m src.run --tier tier1 --group sens --max-reps 1000
```

### Step 4: Run Tier 2 Machine Learning Grid
1. **Calibration Check** (validates residual-resampling Type I error on additive data):
   ```bash
   uv run python scripts/null_calibration.py --reps 100 --reps-m6 50 --n-jobs 6 --out calibration/
   ```
2. **Execute Tier 2 in Stages**:
   ```bash
   # Stage A (R = 100 reps)
   nice -n 10 uv run python -m src.run --tier tier2 --max-reps 100

   # Stage B (R = 200 reps) & Stage C (R = 300 reps)
   nice -n 10 uv run python -m src.run --tier tier2 --max-reps 200
   nice -n 10 uv run python -m src.run --tier tier2 --max-reps 300
   ```

### Step 5: Archive & Back Up Results
```bash
./scripts/backup_results.sh /path/to/backup_directory
```

### Step 6: Generate Publication Figures & Appendix Tables
Once the simulation shards are written to `results/`, generate all 8 publication figures and appendix LaTeX/CSV tables:
```bash
uv run python scripts/generate_paper_plots.py --results-dir results/ --out-dir report/
```
Output files will be saved directly into `report/figures/` (Figures 1–7 at 300 DPI) and `report/tables/` (Tables 1–2 in `.csv` and `.tex`).

---

## 📊 Key Architectural Decisions

All key project decisions and mathematical justifications are tracked in [`NOTES.md`](NOTES.md):
* **D3 (Ceiling-Shift Direction)**: Applied in the scored direction to prevent floor effects on reverse-coded items from mimicking acquiescence bias.
* **D5 (Parallel Analysis)**: Horn's 95th percentile eigenvalue threshold criterion with in-memory caching by $(n_{\text{obs}}, n_{\text{items}})$.
* **D7 (Nonlinear Orthogonalisation)**: Interaction and threshold functional terms are orthogonalised against linear predictors on population data to isolate non-linear variance share $\psi$.
* **D13 (Top-$k$ Feature Recovery Rule)**: Relative gap criterion $(v_k - v_{k+1}) / v_k \ge 0.10$ required for a well-defined top-$k$ target; returns `NaN` if oracle margin is insufficient.
* **D18 (Replication-Level Resume)**: Shards are committed to Parquet in 25-replication batches with sidecar ID manifests, guaranteeing deterministic resumption without redundant compute.
* **D25 (Report Provenance Rule)**: Every quantitative number in published documentation originates from an executable script output in `reports/<name>.txt`.
* **D26 (Publication Plots & Tables)**: Standardized plot and table generator in `scripts/generate_paper_plots.py` automating Figures 1–7 and Tables 1–2 in `report/figures/` and `report/tables/` with the mandatory `[Simulated Data]` provenance label.

---

## 📜 License

Academic Research Project — FYP Monte Carlo Simulation Study.
