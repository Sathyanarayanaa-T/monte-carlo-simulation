<!-- Change Log: Updated Section 3.3 (psi pure non-linear share after residualising; loadings equispaced by default), Section 3.4 (ceiling-shift in scored direction), Section 4 (Tier 2 14-scenario grid), Section 10 (structural-validity gate evaluated at n=100 and n=150). -->
<!-- Milestone 6.7: Added sens_cross_loading_0p5_n{100,150}_{none,inter_med} sensitivity scenarios (cross_loading 0.50); Tier 1 expanded from 60 to 64 scenarios (Tier 2 unchanged at 14). -->
<!-- Milestone 6.8: Updated Section 3.4 for symmetric cross-loading pairs [[B-i, A-i], [B-ii, A-ii]]; added sens_high_latent_corr_n{100,150}_{none,inter_med} (Tier 1 68 scenarios). -->
Monte Carlo Simulation Plan: Validating the FEDRS Analysis Pipeline
Prepared for Sathyanarayanaa T | Draft v2 (reviewer-criticism revision) | 13 October 2026
1. Purpose and scope
The paper proposes hierarchical regression plus an ML layer (random forest, gradient boosting, SHAP, clustering) for about 100 to 150 Likert survey responses. Nothing yet shows that this pipeline can recover a real effect at that size, or that it stays quiet when no effect exists. This simulation tests both before any real data comes in.
The simulation answers six questions.
•	S1 Power. At what sample size does the hierarchical regression detect H1 (the individual block adds variance) and H2 (B-i is a stronger predictor than B-ii)?
•	S2 Type I error. When the true effect is zero, how often does each test claim one?
•	S3 Non-linearity. Which method finds a planted interaction or threshold effect, at which n, and with what false-positive rate?
•	S4 Stability. How stable are SHAP importance rankings across resamples at this n?
•	S5 Prediction. Do RF, boosting and EBM predict held-out commitment scores better than regularised linear models?
•	S6 Measurement. How much do coarse Likert scales, unreliable items and shared-method bias distort the estimates?
Two further questions came from a reviewer-style critique of the paper.
•	S7 Structural validity. If B-i items partly load on A-i, or the C items partly measure general startup intent, how much are H1 and H2 biased, and do the planned checks (HTMT, parallel-analysis EFA) catch it at n = 100 to 150?
•	S8 Common-method bias. How well do marker-variable adjustment and Harman's single-factor test detect and correct a shared acquiescence factor?
What the results can and cannot support. Simulated data supports claims about how the methods behave. It never supports a claim about real students. Every table and figure must carry the label 'simulated'.
2. Workflow overview
1.	Specify scenarios (parameters fixed in a config file before any run).
2.	Generate one dataset per scenario and replication, from latent variables down to Likert item responses.
3.	Validate the simulator itself (Section 9) before fitting any model.
4.	Fit all methods to each dataset with nested cross-validation.
5.	Compute metrics per replication and store raw results.
6.	Aggregate across replications, with Monte Carlo standard errors, and produce figures and tables.
3. Data-generating process (DGP)
3.1 Latent contextual factors
Draw (B-i, B-ii, B-iii) from a multivariate normal with mean 0, variance 1 and correlations 0.35 (B-i with B-ii), 0.30 (B-i with B-iii) and 0.25 (B-ii with B-iii). Sensitivity runs use all three correlations at 0.10 and at 0.50.
3.2 Latent individual factors
The individual factors depend partly on the contextual ones. This builds in the overlap between exposure and disposition that the paper itself worries about.
•	A_tech = 0.35 · B-i + 0.15 · B-iii + u1
•	A_mkt = 0.35 · B-ii + 0.15 · B-iii + u2
The residuals u1 and u2 are bivariate normal with correlation 0.30 (a shared general-resilience component). Their variances are computed numerically so that each A has variance 1.
3.3 Outcome (Commitment Intention, C)
C_latent = linear part + non-linear part + noise.
Linear part. Relative weights on the five standardised predictors:
Predictor	Base weight	H2-null setting	H1-null setting
A_tech	0.30	0.30	0
A_mkt	0.15	0.15	0
B-i	0.20	0.15	0.20
B-ii	0.10	0.15	0.10
B-iii	0.15	0.15	0.15
Non-linear part. Three forms:
•	Interaction: the z-scored product A_tech × B-i (technical resilience matters more when technical exposure is high, the paper's own example).
•	Threshold: B-iii × 1[A_tech < −0.5] (the safety net matters only for low-resilience students, the paper's other example).
•	None: the null case.
Scaling. The planted non-linear part is residualised against the linear predictors so that ψ is the pure non-linear share after residualising: 0 (none), 0.03 (small), 0.08 (medium). The linear and non-linear parts are rescaled so that together they explain a target R² (base 0.35, sensitivity 0.20). The noise term takes the rest.
These effect sizes are assumptions, not literature estimates. Before finalising, pull plausible ranges from the studies the paper cites and record the source or reasoning for each value.
Outcome specificity (added after review). C is the deeptech commitment latent. Its items (14 to 17) must be worded around multi-year R&D horizons, capital intensity and technical feasibility risk. A separate latent G, general startup intent (4 items on low-capital, short-cycle ventures), is generated from B-ii (weight 0.35), B-i (0.15) and C so that corr(G, C) is about 0.5. In the contaminated-outcome sensitivity, each C item loads on a mix, x* = λ(√(1−κ²)·C + κ·G) + e, with κ = 0.4. This tests whether H2 survives when respondents read C partly as generic startup intent. G enters every model as a covariate, as planned for the real analysis.
3.4 Measurement layer
Do not simulate the sub-construct scores directly. Simulate items, then score them as the real analysis will.
•	Items per construct (placeholders; replace with real FEDRS counts): A-i 5, A-ii 4, B-i 5, B-ii 4, B-iii 6, C 4 (items 14 to 17).
•	Item model: x*_ij = λ_j · θ_i + γ · a_i + e_ij, then standardised. θ_i is the construct's latent score and a_i ~ N(0,1) is a respondent-level acquiescence factor shared by all items (common-method bias). Base γ = 0, sensitivity γ = 0.3.
•	Loadings: λ_j are equispaced across the loading range by default (0.55 to 0.85 in base, giving omega of roughly 0.75 to 0.85; 0.40 to 0.65 in low-reliability, giving roughly 0.55 to 0.70), fixed across replications. Random uniform draws remain available for sensitivity.
•	Reverse-keyed items: 20 percent of items per construct have the loading sign flipped, and are re-coded as 6 − x in scoring, matching the paper's protocol.
•	Discretisation: cut at −1.5, −0.5, 0.5 and 1.5 to give categories 1 to 5. Base ceiling shift +0.3 (mild ceiling, typical of self-rated resilience), sensitivity shift +0.6. The ceiling shift applies in the scored direction: it pushes every item toward category 5 of its scored scale (for reverse-keyed items, subtracting the shift from raw propensity before discretisation). Marker items are unaffected.
•	Scoring: sub-construct score = mean of its items after reverse-scoring. Individual Resilience Index = mean of A-i and A-ii items. Contextual Exposure Index = mean of Part B items.
•	General startup intent G (control, added after review): 4 items, loadings Uniform(0.55, 0.85), same discretisation.
•	Marker variable M (added after review): 4 items that load only on the acquiescence factor a_i (same weight as the other items) plus noise, with no substantive loading. The marker is only informative when γ > 0.
•	Cross-loading (sensitivity, added after review): symmetric cross-loading specified via config pairs [[B-i, A-i], [B-ii, A-ii]]. Each B-i item also loads on the A-i (A_tech) latent, and each B-ii item also loads on the A-ii (A_mkt) latent, mimicking domain exposure reported through confidence. Base value 0, sensitivity values 0.3 and 0.5.
•	Contaminated outcome (sensitivity, added after review): κ = 0.4, as in Section 3.3.
3.5 Covariates
•	Semester (6, 7, 8) as a categorical variable with no true effect on C.
•	Three binary prior-exposure flags (working prototype, pitch or hackathon, research-lab internship). Generate each as a logistic function of a related latent factor (prototype and lab internship from B-i, hackathon from B-ii; slope 0.8, intercept set for about 40 percent prevalence). No direct effect on C.
•	Branch is dropped. The sample is single-branch CSE, so it is constant. The paper currently lists it as a covariate and should remove it.
4. Experiment grid
Core grid under the base DGP (28 scenarios).
•	n ∈ {50, 100, 150, 300}.
•	Non-linearity cells: none; interaction at ψ 0.03; interaction at ψ 0.08; threshold at ψ 0.03; threshold at ψ 0.08. That is 4 × 5 = 20 scenarios.
•	H1-null and H2-null scenarios (no non-linearity) at each n: 2 × 4 = 8 scenarios.
Sensitivity grid (32 scenarios). One factor changed at a time from base, at n ∈ {100, 150}, each with non-linearity none and interaction medium:
•	Contextual correlations high (0.50) or low (0.10)
•	Low reliability
•	R² target 0.20
•	Acquiescence γ = 0.3
•	Ceiling shift 0.6
•	B-i items cross-loading on A-i (loading 0.3)
•	Contaminated outcome (κ = 0.4)
Correction: the correlation factor has two settings (high and low), so the original list had 6 settings, not 5. With the two settings added after review, that is 8 settings × 2 sample sizes × 2 non-linearity cells = 32 scenarios, for 60 in total. Acquiescence at γ = 0.3 is also analysed with and without marker adjustment, which adds analysis arms, not scenarios.
Replications. R = 1000 for OLS-family methods (Tier 1: all 60 scenarios) and R = 300 for ML-family methods (Tier 2: reduced ML grid of exactly 14 scenarios: {none, inter_small, inter_med, thresh_small, thresh_med} at n=100 and n=150, plus {none, inter_med} at n=50 and n=300), which cost far more. The Monte Carlo standard error of a rate p is sqrt(p(1−p)/R). For p = 0.80 it is 0.013 at R = 1000 and 0.023 at R = 300. For p = 0.05 it is 0.007 and 0.013. Report these with every rate.
5. Methods under test
All predictors are z-scored within each training fold. Covariates are included in every model.
ID	Method	Purpose
M0	Training-fold mean	Baseline for out-of-sample R²
M1	Hierarchical OLS	The paper's classical analysis
M2a	OLS + one pre-specified interaction	Upper bound for OLS (oracle-specified)
M2b	OLS + all 10 pairwise products	Realistic exploratory OLS
M3	Elastic net with pairwise products	Regularised linear alternative
M4	Random forest + SHAP	The paper's proposed ML method
M5	Gradient boosting + SHAP	The paper's second proposed method
M6	Explainable Boosting Machine	Honest non-linear alternative at small n
M1 Hierarchical OLS. Block 1: contextual predictors (B-i, B-ii, B-iii) plus covariates. Block 2: add A_tech and A_mkt. Tests: F-test on ΔR² for the block (α = 0.05) and the t-test on the A_tech coefficient in the full model (H1). For H2, a linear contrast test of β(B-i) − β(B-ii) = 0 in the full model with HC3 robust standard errors, using standardised predictors. A rejection counts as a hit only if the sign is correct.
M2a. M1 full model plus one product term matching the true DGP (A_tech × B-i for the interaction DGP, A_tech × B-iii for the threshold DGP). This is an oracle that knows where to look, so it bounds what OLS can do. Report it as such.
M2b. M1 full model plus all 10 pairwise products of the five predictors. Holm correction across the 10 product terms. Detection means any corrected p < 0.05.
M3 Elastic net. Five main effects, 10 pairwise products, covariates. ElasticNetCV with l1_ratio in {0.5, 0.9, 1.0}, 5-fold inner CV (3-fold at n = 50), 1-SE rule. Detection by stability selection: 100 bootstrap refits; an interaction is detected if any product term is selected in at least 70 percent of them.
M4 Random forest. 300 trees, bootstrap on. Inner CV grid: min_samples_leaf in {3, 5, 10}, max_features in {0.33, 0.5, 1.0}. This respects the paper's own statement that Step 2 stays low-capacity.
M5 Gradient boosting. Learning rate 0.03, subsample 0.7, min_samples_leaf 5. Inner CV grid: max_depth in {1, 2, 3}, n_estimators in {100, 200, 400}.
M6 EBM. interpret's ExplainableBoostingRegressor with interactions = 10, max_bins = 32, learning_rate = 0.01, outer_bags = 8, min_samples_leaf = 3. Use these settings without inner tuning to keep compute manageable.
Nested cross-validation. Outer: 5-fold, repeated twice, for out-of-sample performance. Inner: 3 to 5 folds for hyperparameters. SHAP, importance rankings and interaction tests are computed from a fit on the full dataset, as the real analysis would do. Out-of-sample R² is computed from pooled outer-fold predictions and can be negative.
Shared interaction-detection rule for M4, M5, M6.
•	Statistic T = largest mean absolute SHAP interaction value over predictor pairs (TreeSHAP for M4 and M5; largest pairwise term importance for M6).
•	Null distribution by residual resampling, not label permutation. Fit the additive model (OLS main effects plus covariates), then build 100 null datasets as y* = fitted values + randomly permuted residuals, refit the model and recompute T.
•	p = (1 + number of null T ≥ observed T) / 101. Detect if p < 0.05.
•	Why not permute y directly: that destroys the main effects too. A forest on main-effect-only data would then look as if it had found an interaction compared with pure noise. The null must keep the additive signal and remove only the interaction.
•	Run a timing pilot of about 5 replications before the full run. If compute is too high, cut the null refits to 50 and report that choice.
Measurement-validity and bias checks, run on every dataset (added after review). These test the procedures planned for the real data.
•	HTMT. Compute HTMT from the simulated items for A-i with B-i, A-ii with B-ii, and C with G. Record the share of replications above 0.85 in the base, cross-loading and contaminated-outcome settings.
•	Parallel-analysis EFA. For each of the same three pairs, record how often parallel analysis recovers two factors.
•	Marker adjustment (M1-adj). Refit M1 after the Lindell-Whitney correction, r_adj = (r − r_M) / (1 − r_M), with r_M the mean absolute correlation of the marker with the study variables. Compare bias and power with unadjusted M1 at γ = 0.3.
•	Harman's test. Record how often the first unrotated factor explains more than 50 percent of variance. This shows how weak the test is as a method-bias detector.
6. Metrics
A. Inferential performance (M1)
Metric	Definition	Scenarios
H1 power	Share of replications with ΔR² F-test p < 0.05	Base DGP
H1 Type I error	Same, when A weights are 0	H1-null
H1 coefficient power	Share with A_tech t-test p < 0.05	Base DGP
H2 power	Share with significant contrast and correct sign	Base DGP (B-i > B-ii)
H2 Type I error	Share with significant contrast	H2-null
Mean ΔR²	Average block-2 gain, against the true gain	All
B. Interaction detection (M2a, M2b, M3, M4, M5, M6)
Metric	Definition
True positive rate	Share of replications where the method detects an interaction when ψ > 0
False positive rate	Share where it detects one when ψ = 0
Top-1 accuracy	Share where the true pair ranks first among the 10 pairs (threshold-free)
AUC	Area under the ROC curve, pooling null and non-null replications, using the method's continuous statistic (p-value or T)
The top-1 and AUC metrics let methods with different detection rules be compared fairly.
C. Importance and stability (M1, M3, M4, M5, M6)
•	Oracle importance. Compute the true feature importance from the noise-free DGP function on a very large population sample (n = 200,000) using permutation importance. This is the ground truth ranking.
•	Ranking accuracy. Kendall's tau between each method's estimated ranking and the oracle ranking, plus top-1 and top-2 recovery rates.
•	Stability. 50 bootstrap resamples per dataset. Mean Kendall's tau between the original ranking and each bootstrap ranking. For OLS, rank by absolute standardised beta.
•	H3 check. Record how often the paper's current H3 rule ('at least one interaction surfaces') is satisfied under ψ = 0. I expect a very high rate, which is the argument for rewriting H3 with the null-calibrated rule above.
D. Prediction (all methods)
Out-of-sample R² and RMSE from the outer folds. Report the difference against M3 (the regularised linear benchmark) with a paired bootstrap interval across replications.
E. Estimation quality (M1)
Bias, RMSE and 95 percent CI coverage of each β compared with the true latent weights. Report the attenuation ratio (estimated β divided by true β) to show how measurement error shrinks effects.
F. Simulator diagnostics
Empirical Cronbach's alpha and McDonald's omega per construct, item-category proportions (especially at ceiling), and the observed correlation matrix of sub-construct scores, each against its target.
G. Housekeeping
Monte Carlo SE for every rate, convergence failures per method and scenario, runtime per method.
H. Measurement validity and method bias (added after review)
•	HTMT flag rate (share above 0.85) by pair and scenario: false-flag rate in the base case, detection rate under cross-loading and contamination.
•	Parallel-analysis recovery rate of the correct factor count.
•	Bias in β(A-i) and in the B-i minus B-ii contrast, and H1 and H2 power, under cross-loading and contamination, against the base case.
•	Marker adjustment: bias and power of M1-adj versus M1 at γ = 0.3, and the share of replications where adjustment changes the H1 or H2 conclusion.
•	Harman detection rate at γ = 0 and γ = 0.3.
7. Leaky-design demonstration
The original draft put Commitment Intention inside the Internal index. Quantify what that did, as a small extra experiment at n = 150 with 1000 replications.
•	Correct design: C regressed on the contextual block, then the Individual Resilience Index (A items only).
•	Leaky design: the index also includes the C items.
•	Report: mean ΔR² in each design, and the false-positive rate of the H1 test under the H1-null scenario. The leaky design should show a high false-positive rate and a large ΔR² inflation. This gives one clear figure for why the redesign matters.
8. Implementation
Stack. Python 3.11 with numpy, scipy, pandas, statsmodels, scikit-learn, shap, interpret, joblib and matplotlib. Pin exact versions in a lockfile at the start and log them with every run.
Repository layout.
sim/
  config/        scenarios.yaml, methods.yaml
  src/
    dgp.py       latent factors, outcome, items, scoring
    methods/     m1_ols.py, m2_interactions.py, m3_enet.py, m4_rf.py, m5_gb.py, m6_ebm.py
    detect.py    residual-resampling null, stability selection
    metrics.py   all metrics in Section 6
    run.py       scenario x replication loop
  tests/         simulator validation (Section 9)
  results/       one parquet shard per scenario
  report/        figures, tables
Reproducibility.
•	One master seed per project. Use numpy SeedSequence.spawn to create one independent child seed per (scenario, replication), so results do not depend on run order or core count.
•	Store the scenario config, package versions and git commit hash alongside each results shard.
•	Parallelise over replications with joblib. Set n_jobs = 1 inside each model to avoid oversubscription.
•	Checkpoint after each scenario so a crash resumes instead of restarting.
Core loop in outline.
for scenario in scenarios:
    for rep in range(R):
        data = generate(scenario, seed=child_seed(scenario, rep))
        for method in methods:
            fit = nested_cv_fit(method, data)
            row = metrics(fit, data, scenario)
            save(row)
    aggregate(scenario)
9. Validating the simulator before any model runs
A wrong DGP invalidates everything after it, so check it first.
1.	Moments. At n = 200,000, the latent correlations match the specification within Monte Carlo error.
2.	Reliability. Empirical alpha and omega per construct land in the target band.
3.	Variance shares. The non-linear part carries variance share ψ, and the true function reaches the target R².
4.	Item distributions. Category proportions look plausible: no empty categories in base settings and a modest ceiling effect.
5.	Negative control. Run the whole pipeline with C replaced by pure noise. Every Type I rate should land near 5 percent, and no ML method should detect interactions more often than its nominal rate.
6.	Positive control. Run with a strong linear signal at n = 300. Power should approach 1.
7.	Visual check. Plot histograms of item responses, a scatter matrix of sub-construct scores, and the true function surfaces for the interaction and threshold terms.
Added check: with κ = 0 and cross-loading = 0, the HTMT false-flag rate should stay low (about 5 percent or less), and the marker-adjusted and unadjusted estimates should agree at γ = 0.
If any check fails, fix the simulator or the method code before running the grid.
10. Decision rules and outputs
Fix these before the main run and register the plan with a timestamp (for example on OSF).
•	Sample size. The minimum n is the smallest value at which M1 reaches at least 80 percent power for H1 and H2 under the base DGP. If no tested n reaches it, say so and revise the hypotheses.
•	Does ML add value? The ML methods are an exploratory benchmark. Confirmatory tests stay with moderated regression (M1, M2a). M4, M5 or M6 is reported as adding value only if its out-of-sample R² exceeds that of M3 (elastic net) by at least δ, with the paired-bootstrap interval across repeated nested CV excluding zero. δ starts at 0.02 and is recalibrated from the simulation so that it exceeds the noise in the R² difference at n = 100 to 150. Otherwise the paper reports no detectable added value at this sample size, which does not show that non-linear effects are absent. This rule is fixed before real data are seen.
•	H3 rewrite. Replace the current wording with a specific, falsifiable claim tested by the null-calibrated rule, naming the interaction in advance.
•	Structural-validity gate (added after review). If HTMT for A-i with B-i, or for C with G, exceeds 0.85 in the real data, the affected hypothesis is revised or dropped before it is tested. The simulation sets how reliable this rule is. The structural-validity gate is evaluated at n = 100 and n = 150 (the simulation grid has no n = 120), reporting exact empirical flag rates with Monte Carlo Standard Error (MCSE) and no interpolation. If the flag rate under 0.3 cross-loading is below 70 percent, the paper says the check is weak.
•	Method-bias gate (added after review). Report H1 and H2 both unadjusted and marker-adjusted. If adjustment changes the conclusion in the simulation at the assumed γ, treat the marker-adjusted real-data result as primary.
Figures and tables.
1.	Power curves by n for H1 and H2.
2.	Interaction detection (TPR and FPR) by method and n, interaction and threshold DGPs separately.
3.	ROC or AUC comparison at n = 100 and 150.
4.	Stability and ranking accuracy by method.
5.	Out-of-sample R² by method and n.
6.	Leaky-design bias figure.
7.	Sensitivity heatmap: change in power or AUC under each sensitivity factor.
8.	Appendix tables: all scenario parameters, simulator diagnostics, Monte Carlo SEs.
11. Limitations
•	Results hold only under the assumed DGPs. The sensitivity grid reduces this problem but does not remove it.
•	Latent variables are normal and items come from a threshold model. Real Likert data may have stronger clustering and item-specific quirks.
•	Students in one institution are nested in class sections and teams. The DGP assumes independent respondents. If the real design is clustered, add an intraclass correlation scenario later.
•	Missing data and non-response bias are not simulated. Add an item-level missingness scenario (about 5 percent, at random) as an extension.
•	The k-means profile clustering and the embedding-based thematic coding are not covered. A separate small study can test cluster recovery with the adjusted Rand index on planted profiles.
•	Effect sizes are assumptions until they are tied to sources and later updated from the n = 20 to 30 pilot.
•	The cross-loading size (0.3) and the contamination size (κ = 0.4) are assumptions. Real values should come from the pilot and the cognitive interviews.
•	HTMT, parallel analysis and marker adjustment at n of about 120 are screening tools, not proof of structural validity. A full CFA stays exploratory at this sample size.
12. Work plan
1.	DGP and simulator checks (Sections 3 and 9). Output: validated generator and diagnostic plots.
2.	Methods M0 to M3 and metrics A, B, E. Output: first power and Type I results, fast to run.
3.	Methods M4 to M6 with the null-calibrated detection rule. Output: timing pilot, then the ML results.
4.	Leaky-design demo and sensitivity grid.
5.	Aggregation, figures, decision rules applied. Output: revised pipeline and H3 wording for the paper.
6.	Recalibration after the pilot. Update correlations and reliabilities from the real pilot data and rerun the key scenarios.
Added after review: build the measurement layer (G, M, cross-loading, κ) and the HTMT, parallel-analysis, marker-adjustment and Harman checks before step 2, since they change the DGP and the validation in Section 9.
