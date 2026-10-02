# Next-stage analysis freeze — 3 October 2026

Protocol ID: `cfpb-neural-v1`. Full specification: [full_protocol.md](full_protocol.md). SHA256 inventory: [protocol_freeze.json](protocol_freeze.json).

This is a local, dated pre-neural analysis plan. It has not been submitted to OSF, a registry or a journal. Debt Collection future CPU outcomes are already known; Credit Card predictive outcomes are not known. Do not describe this as a fully untouched confirmatory Debt Collection study or statistically independent data collection.

## Frozen commitments

- Authentic original Issue labels, narrative-only states, seven Debt Collection classes and 12 Credit Card classes; no teacher soft labels, future-performance label filtering or label merging.
- Primary Laya hard-label CE; exact Laya encoder/input conventional control; official ModernBERT-large external baseline; one Debt Collection-only hard-label RLCD+CE ablation. Existing TF-IDF results are retained.
- Three fixed seeds 17/43/101; fixed optimizer/budget/512-token setting; checkpoint selection uses Jul–Sep 2024 macro-F1, NLL tie-break, then earliest epoch. No future tuning or best-seed reporting.
- Train Sep 2023–Jun 2024. October 2024 temperature; November gate; December independent verification; 2025Q1–2026Q1 main replay; 2026Q2 exploratory. Families are disjoint across the three historical calibration/gate roles.
- Historical gate target 90%, >=10% coverage and >=200 accepted family representatives. Fixed 51-threshold grid, Bonferroni one-sided historical bound, maximum qualifying coverage. No qualifying gate or failed December verification is a reported result; neither causes threshold reselection.
- Frozen T versus previous-quarter-only rolling T, unchanged numerical gate and model weights. Uniform prior-stream label budgets 100/500/2,000, 20 nested fixed draws; no oracle label stratification or accepted-only feedback.
- Primary paired model contrast: three-seed mean of equal-quarter frozen NLL differences, Laya versus exact matched control. Primary gate outcome: accepted risk above/below .10 with coverage and uncertainty, not accuracy alone.
- Global-family cluster resampling; paired ID alignment; 2,000 replicate effect intervals; 10,000 replicate aggregate gate bands across five quarters; descriptive true-Issue selective outcomes, with explicit sparse-count treatment.
- Publish adverse/null/stable outcomes, all seeds, unavailable gates, minority failures and recalibration harms. No guarantee of novelty or calibration is inferred from model branding or a feasibility GO.

## Allowed before training

Source/dependency/model revision verification; deterministic exports and input-budget checks; Credit Card counts/taxonomy/family audit; synthetic unit fixtures for mathematical/code contracts. Unit fixtures are not synthetic training labels or a new performance benchmark.

## Not yet executed

No pretrained neural weights downloaded, no neural optimization, no CFPB neural inference, no Credit Card classifier. GPU timing estimates are estimates, not measurements. Exact Kaggle package/import/weight-loading/memory verification occurs at the beginning of the explicitly authorized GPU phase before a full run.

## Amendment policy

Unexpected environment failure, unusable checkpoint or resource shortfall must be logged with timestamp, reason, outcomes already viewed and affected comparisons. Change only what fixes the failure, preserve authentic labels/data chronology, and never call amended results prospectively untouched. After future neural outcomes, any changed hyperparameter, gate, prompt or label set is exploratory. Updating dependencies or a Git SHA requires a new freeze hash and a documented reason; silent `main`/`latest` updates are forbidden.
