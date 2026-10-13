Project rules
This repo implements the Monte Carlo study in SPEC.md. SPEC.md is the source of truth.

Do not change any parameter, method, metric or decision rule from SPEC.md without asking me first.
If SPEC.md leaves something open, do not invent a value. Put it in config/*.yaml with a TODO comment and list it in NOTES.md under Open Questions.
Python 3.12 managed with uv. Pin exact versions in the lockfile.
All parameters live in config/scenarios.yaml and config/methods.yaml. No magic numbers in code.
Randomness: one master seed, np.random.SeedSequence.spawn for one child seed per (scenario, replication). Never use the global numpy RNG.
Models run with n_jobs=1. Parallelism is only over replications, via joblib.
Anything fitted (z-scoring, imputation, tuning) happens inside each CV training fold, never before the split.
Never weaken or delete a test to make it pass. If a test fails, report it and explain why.
Mark any test that takes more than 1 minute with @pytest.mark.slow and do not run slow tests without asking me.
Do not start any run longer than about 2 minutes. Give me the command and the expected time instead.
Keep NOTES.md: assumptions made, spec ambiguities, anything you were unsure about.
Work on one milestone at a time. Post your plan first, then build. Stop at the end of each milestone and summarise what you did, what passed and what you are unsure about.