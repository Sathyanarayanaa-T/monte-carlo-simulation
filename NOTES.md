# Project Notes & Architecture Decisions

## 1. Decisions (Resolved)

- **D1 (Item Counts & Keying)**: Item counts and reverse-keyed items stay at the placeholder values from SPEC.md Section 3.4 ($A\text{-}i: 5, A\text{-}ii: 4, B\text{-}i: 5, B\text{-}ii: 4, B\text{-}iii: 6, C: 4$, with 1 reverse-keyed item per construct). Simulation results are conditional on them; the key scenarios are re-run with real counts after the survey pilot.
- **D2 (Semester Distribution)**: Semester is categorical and uniform over $\{6, 7, 8\}$ with equal probability $1/3$, and has no true effect on $C$.
- **D3 (Ceiling-Shift Direction)**: Ceiling shift applies in the scored direction (`ceiling_shift_direction: scored | raw`, default `scored`). In scored mode, the shift pushes every item toward category 5 of its scored scale; for reverse-keyed items, the shift is subtracted from the raw propensity before discretisation ($x^* - \text{shift}$), so post-scoring ($6 - x$) yields elevated endorsement at category 5 without inducing spurious acquiescence when $\gamma = 0$. Marker items are unaffected.
- **D4 (Pearson Item Correlations)**: Item correlations are computed using standard Pearson product-moment correlations on item scores. The real-data empirical analysis uses the exact same functions; polychoric correlation is treated as a real-data sensitivity analysis and is not simulated.
- **D5 (Parallel Analysis Factor Retention)**: Horn's parallel analysis retains factors sequentially (Horn rule) up to the first component where the sample eigenvalue falls below the 95th percentile null benchmark ($\lambda_k \le \lambda^*_{crit, k}$). If the first eigenvalue fails, 0 factors are retained.
- **D6 (Factor Loadings Sampling)**: `loading_draw` defaults to `equispaced` across the specified loading interval (e.g., $[0.55, 0.85]$ for base; $[0.40, 0.65]$ for low-reliability). This ensures empirical reliabilities ($\alpha, \omega$) reliably land within their target bands ($[0.75, 0.85]$ and $[0.55, 0.70]$) without stochastic sample clustering. `random` remains available for sensitivity analysis.
- **D7 (Nonlinear Orthogonalisation)**: `orthogonalise_nonlinear` defaults to `true`. When true, the planted non-linear functional term (bilinear product or threshold term) is residualised against the linear latent predictors ($A_{\text{tech}}, A_{\text{mkt}}, B\text{-}i, B\text{-}ii, B\text{-}iii$) on the calibration population, ensuring that $\psi$ is strictly the pure orthogonal non-linear variance share with $R^2(\text{linear}) = 0.0$. When false, the unorthogonalised threshold $R^2$ on the linear predictors is approximately 0.333 (measured), conflating linear and non-linear predictive power.
- **D8 (Structural-Validity Gate)**: The structural-validity gate (cross-loading $\lambda = 0.30$ under $H_1$, HTMT false-flag rate $\le 5\%$) is evaluated at $n = 100$ and $n = 150$ (the simulation grid has no $n = 120$), reporting exact empirical rates with Monte Carlo Standard Error (MCSE) and no interpolation.
- **D9 (Tier 2 Scenario Selection & Null Refits)**: Tier 2 (`reduced_ml`) comprises exactly the 14 scenarios specified in Part D: $\{\text{none}, \text{inter\_small}, \text{inter\_med}, \text{thresh\_small}, \text{thresh\_med}\}$ at $n \in \{100, 150\}$ (10 scenarios), plus $\{\text{none}, \text{inter\_med}\}$ at $n \in \{50, 300\}$ (4 scenarios). Replications are set to 300 with $B = 50$ null refits for residual-resampling interaction tests, unless the slow calibration test shows the interaction test is miscalibrated, in which case $B$ increases to 100 and Tier 2 is re-timed.
- **D10 (Compute Parallelism)**: Execution parallelism is fixed to `n_jobs = 6` in `config/methods.yaml`. Threadpools (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) are pinned to 1 to avoid thread thrashing and CPU oversubscription.
- **D11 (Scale Differences as Stated Limitation)**: The simulation uses 5-point Likert items for $C$, $B\text{-}i$, and $B\text{-}ii$. The real-world FEDRS survey uses a 7-point scale for $C$ and behavioral count formats for $B\text{-}i$ and $B\text{-}ii$. This is accepted as a stated methodological limitation in the dissertation, with no re-run required unless the pilot data suggests substantial distortion.
- **D12 (Parquet Storage & Provenance Tracking)**: Output parquet shards (`results/*.parquet`) are excluded from git tracking (added to `.gitignore`). Data provenance is guaranteed because each shard embeds the git commit hash, UTC timestamp, and pinned package versions directly into its Parquet schema metadata table and JSON sidecar. Raw results are backed up independently.
- **M2a Interaction Term Clarification**: In Model M2a, the pre-specified bilinear interaction term for the interaction DGP is $A_{\text{tech}} \times B\text{-}i$ (matching the true planted form $A_{\text{tech}} \cdot B\text{-}i$). For the threshold DGP, M2a uses $A_{\text{tech}} \times B\text{-}iii$, which acts as an approximate parametric proxy for the localized threshold mechanism $B\text{-}iii \cdot \mathbf{1}[A_{\text{tech}} < -0.5]$.

---

## 2. Open Questions (Still Open)

1. **Survey Pilot Real-Data Item Properties**:
   - What are the observed distributions, category counts, and reverse-keying alignments in the real FEDRS student cohort once pilot data collection completes?
2. **Interaction Calibration Check**:
   - Will the standalone slow calibration test (`pytest -m slow`) confirm nominal Type I error rates ($\le 5\%$) for the residual-resampling interaction test with $B = 50$ across ML methods, or will calibration require elevating $B$ to 100?

---

## 3. Milestone Log (Short)

- **Milestone 1 (DGP & Simulator)**: Implemented complete Data-Generating Process (`src/dgp.py`), calibration of variance scaling, root-solving of logistic intercepts, discrete Likert categorisation, and moment validation test suite (`tests/`).
- **Milestone 2 (Audits & Stabilisation)**: Refined root-solving of logistic intercepts to reach exactly 40.0% prevalence, added `orthogonalise_nonlinear: true`, equispaced factor loadings (`loading_draw: equispaced`), and full tolerance audits.
- **Milestone 3 (Measurement & Oracle)**: Implemented psychometric measurement checks (`src/measurement.py`: HTMT, Horn's parallel analysis, Lindell-Whitney correction, Harman's single-factor share) and permutation oracle importances (`src/oracle.py`).
- **Milestone 4 (Classical Methods & Nested CV Harness)**: Implemented nested CV harness (outer 5-fold $\times$ 2 repeats), models M0, M1, M1-adj, M2a, M2b, and M3 (Elastic Net with 100-bootstrap stability selection), along with metrics A, E, B, D, and H.
- **Milestone 5 (Machine Learning & TreeSHAP)**: Implemented M4 (Random Forest), M5 (Gradient Boosting), M6 (EBM), residual-resampling interaction null tests ($B=50$), TreeSHAP interaction metrics, Metric C ranking accuracy, and empirical timing pilot.
- **Milestone 6 (Runner, Tiers & Leaky Design)**: Implemented production batch runner (`src/run.py`) with `joblib` parallelism, seed isolation, parquet sharding, resume capabilities, `--dry-run`, and the empirical leaky-design demonstration (`src/leaky.py`).
- **Milestone 6.5 (Pre-Run Fixes & Readiness)**: Implemented `ceiling_shift_direction: scored`, added 8 per-replication measurement check columns to shards, resolved oracle ties with Kendall $\tau_b$, updated Tier 2 to 14 core scenarios, ran 1000-rep leaky demonstration, added `--max-reps`/`--out-dir`, created timing benchmark and execution helper scripts, and finalized documentation.
