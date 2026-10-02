# Official data and transformations

Source: CFPB's official historical Consumer Complaint Database Narratives Archive. `source_manifest.json` records exact URLs, download timestamps (UTC), byte sizes and SHA256 for all 17 ZIP parts spanning September 2023–August 2026, the archive landing page and official August 2023 taxonomy PDF.

Raw ZIPs are unchanged. `audit_cfpb.py` streams all rows, validates dates, checks Complaint ID uniqueness, counts availability across all products and retains Debt Collection/Credit Card narratives for local audit. Credit Card is not modeled unless a later authorization permits it. Only the narrative enters TF-IDF. No record metadata is concatenated into model input.

`preprocess.py` sorts by receipt date and Complaint ID; includes original Issue labels using training counts only; saves a deterministic YAML policy. Normal narrative text is not stripped of financial terms. Raw exact hashes use trimmed text. Normalized hashes collapse case/whitespace. Template detection additionally normalizes numbers, punctuation and redaction runs, but these transformations do not enter the normal classifier.

`deduplicate.py`: 128 MinHash permutations, seed 20261002, LSH candidate threshold .70; verify word-5-shingle Jaccard >=.85; connected components define families. Exact/canonical repetitions are linked. Texts under 20 words do not use LSH, though exact/canonical checks apply. Retrieval recall is probabilistic and semantic paraphrases can be missed. A family is an independence proxy, not proof of a distinct consumer. Connected-component endpoints need not have .85 pairwise similarity.

Training retains one earliest exact-text example. Conflicting labels are counted, not silently relabeled. The operational evaluation retains all narrative rows. The novel-family sensitivity removes families seen during training/validation and retains one chronological representative per family and evaluation period. It is a different, harder population, not an unbiased correction of the full stream.

Dates refer to complaint receipt, not narrative publication. Public narrative selection varies over time. August is audit-only. Q2 2026 is exploratory if the frozen availability rule passes; July requires both availability and class-family support. No causal population-level claim follows from these descriptive samples.

Raw sources and derived narrative files are excluded from Git by default because of size. Public availability does not imply that narratives are expert-adjudicated or representative. Do not publish unnecessary verbatim narratives in report examples.
