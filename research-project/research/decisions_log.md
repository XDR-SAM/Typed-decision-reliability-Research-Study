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
