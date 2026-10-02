# Decisions log

## Frozen before acquisition/model results, 2 October 2026

- Scope: official CFPB acquisition, audit, chronological preprocessing, CPU TF-IDF/logistic regression pilot, decision report. No neural model or GPU use.
- The four user conditions and >=3/4 rule are retained. The config records conservative numerical interpretations of qualitative terms before model fitting. Special NO-GO warnings are reported separately and honored; they are not silently absorbed into altered scores.
- Existing knowledge: the preceding review inspected July/August archive availability. Those months therefore are not pristine holdouts. No Debt Collection classifier results have been seen.
- All nonempty original Issue labels with >=200 training narratives are included. No label merging. Future counts are audit annotations only.
- Word unigram/bigram TF-IDF, three C values, select on Jul-Sep 2024 macro-F1 only. No random headline split, no character model unless necessary (not planned).
- Exact-repeat training texts retain the earliest historical example. Conflicting labels remain documented. Future operational cohorts retain repetitions; novel-family sensitivity removes families in training/validation and keeps one future family representative per period.
- MinHash candidate retrieval plus exact Jaccard confirmation is approximate: it may miss paraphrases and does not prove person-level independence. Report family-level counts as an operational independence proxy.
- Keyword phrases will be derived only from training label names before any model results; ablated model refits using the same chosen C, and frozen-normal-model masking is also assessed. No phrases selected from future errors.
- Main future cohorts: 2025Q1 through 2026Q1. Q4 2024 is an earlier pilot test. Later availability is assessed separately without performance-based inclusion.
- A GO is permission to propose further work, not authorization to run it. Stop after the pilot.

## Pilot completed

- Audit complete before fitting; seven unchanged labels included. Q2 2026 exploratory; July/August excluded from fitting/evaluation under the frozen availability rule.
- C=3.0 selected on historical validation only. No future tuning.
- GO: 4/4 conditions; warnings: []. No scope expansion or neural/GPU work.
- Plot checks and independent probability/NLL/accuracy recomputation completed. Added per-class selective diagnostics to expose minority-class failures; these did not change any model or threshold.

## Full neural protocol preparation — 3 October 2026

- User explicitly approved moving beyond CPU feasibility to protocol/preparation only. No neural/GPU training or neural CFPB inference is authorized in this step.
- Reviewed repository/prior decisions, inspected pinned official Laya source/notebook/docs and checkpoint/tokenizer metadata. Hard one-hot outcomes are valid inputs; CE-only requires our labeled custom loop. Primary Laya-CE, exact encoder/input conventional control, generic ModernBERT-large and one Debt Collection RLCD+CE ablation are fixed.
- Debt Collection is retrospective and pilot-exposed. Credit Card predictive outcomes remain uninspected. Its 12 labels follow the same >=200 training-count rule; it qualifies for aggregate secondary replication, with explicit sparse subgroup limits (minimum 56 future families).
- Q4 roles fixed to October temperature, November gate, December verification, with disjoint families after historical exclusions. No gate is manufactured if 90% target/10% coverage/200 accepted families cannot be supported.
- Fresh-label maintenance is previous-quarter-only, 100/500/2,000 uniform-record labels with 20 nested draws. Weights and numerical threshold remain fixed; actual publication/feedback delay is unknown.
- Three fixed training seeds, identical historical rows/shared state slices, validation-only epoch selection. Global-family paired bootstrap, seed reporting and minority accepted-count rules fixed in YAML/full protocol.
- New preparation datasets/tables have separate paths under outputs/neural_preparation. Original pilot models/predictions/metrics/figures/configs are preserved. Source snapshot/revisions and freeze hashes record the next-stage plan; it is not an external registry submission.
