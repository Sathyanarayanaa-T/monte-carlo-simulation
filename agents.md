# Project rules

This repo implements the Monte Carlo study in SPEC.md. SPEC.md is the source of truth.

- Do not change any parameter, method, metric or decision rule from SPEC.md without asking me first.
- If SPEC.md leaves something open, do not invent a value. Put it in config/*.yaml with a TODO comment and list it in NOTES.md under Open Questions.
- Python 3.12 managed with uv. Pin exact versions in the lockfile.
- All parameters live in config/scenarios.yaml and config/methods.yaml. No magic numbers in code.
- Randomness: one master seed, np.random.SeedSequence.spawn for one child seed per (scenario, replication). Never use the global numpy RNG.
- Models run with n_jobs=1. Parallelism is only over replications, via joblib.
- Preprocessing (z-scoring, imputation, anything fitted) happens inside each CV training fold, never before the split.
- Tests are not to be weakened or deleted to make them pass. If a test fails, report it and explain why.
- Do not start any long run (more than about 2 minutes). Tell me the command and expected time instead.
- Keep a NOTES.md: assumptions made, spec ambiguities, anything you were unsure about.
- Work on one milestone at a time. Stop at the end of each one and summarise what you did.