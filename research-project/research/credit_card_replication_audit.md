# Credit Card → Issue replication audit

Audit/freeze: **3 October 2026**. Only official archived counts, taxonomy, publication availability and text-family support were examined. No TF-IDF, neural classifier, predictive score or future error analysis was produced for Credit Card.

**Decision: qualifies as an independently specified secondary aggregate-reliability replication, with limited minority-class precision.** It does not qualify for a precise 90%-gate guarantee in every Issue/quarter. This distinction must remain visible in the paper; do not call it an independent consumer population or a uniformly well-powered subgroup study.

Source: the same 17 official CFPB archives and August 2023 taxonomy already downloaded for the pilot, with original URLs/SHA256 in `data/source_manifest.json`. Original Product equals `Credit card`; original Issue strings are targets. Inclusion rule is exactly the pilot's training-only rule: >=200 nonempty Sep 2023–Jun 2024 narratives. No labels are merged or dropped based on later counts/performance. Fifteen original label strings occur; twelve qualify, with no new later label strings. Excluded classes are credit monitoring/identity protection (86 training narratives), fraud alerts/security freezes (31), and inability to get credit report/score (10).

## Label support

| Original Issue | Raw training narratives | Minimum records / distinct families in a main future quarter |
|---|---:|---:|
| Advertising and marketing, including promotional offers | 1,453 | 462 / 458 |
| Closing your account | 1,890 | 641 / 639 |
| Fees or interest | 2,764 | 1,124 / 1,123 |
| Getting a credit card | 3,300 | 864 / 801 |
| Improper use of your report | 341 | 57 / 56 |
| Incorrect information on your report | 2,669 | 371 / 357 |
| Other features, terms, or problems | 3,204 | 1,148 / 1,146 |
| Problem when making payments | 1,664 | 493 / 493 |
| Problem with a company's investigation into an existing problem | 5,848 | 207 / 201 |
| Problem with a purchase shown on your statement | 6,622 | 2,417 / 2,403 |
| Struggling to pay your bill | 296 | 113 / 113 |
| Trouble using your card | 1,088 | 391 / 390 |

The two small training classes remain included, despite limited future support. “Improper use of your report” has only 56 novel families in its smallest main quarter; “Struggling to pay your bill” has 113. The earlier Debt Collection pilot's >=200-per-class criterion is therefore **not** satisfied for every replication class. We do not change that historical pilot decision or claim the replication passes the same uniform-support rule. Large aggregate cohorts support NLL/Brier/risk–coverage replication; small subgroup selective accuracy will be sparse and reported as such.

## Cohorts and dependence

| Period | Included narratives | Distinct families | Fraction in train/validation families |
|---|---:|---:|---:|
| Training | 31,139 | 24,799 | Historical reference |
| Validation | 7,972 | 7,520 | Historical reference |
| 2024Q4 | 8,261 | 7,809 | 1.985% |
| 2025Q1 | 9,881 | 9,131 | 1.356% |
| 2025Q2 | 9,709 | 9,422 | 1.318% |
| 2025Q3 | 11,570 | 10,349 | 1.167% |
| 2025Q4 | 10,181 | 9,614 | .796% |
| 2026Q1 | 9,090 | 8,993 | .473% |
| 2026Q2, exploratory | 9,437 | 9,259 | .657% |

There are **109,519** Credit Card narratives across the full audited range; **25,182** exact-unique included training examples and **50,431** included main-future examples. Excluded-label mass is below .5% in each listed quarter. October/November/December decontaminated one-family historical calibration/gate sets contain **2,620 / 2,446 / 2,724** examples respectively. Those sizes support scalar temperature fitting and a possible aggregate gate; they do not ensure that a threshold meets the fixed target.

The unchanged MinHash/Jaccard method gives 100,633 raw exact unique texts and 98,968 families across all Credit Card labels. There are 121 cross-period families comprising 2,496 rows, and 203 conflicting-label families. Cross-product audit finds **119 exact narrative texts** shared with Debt Collection, affecting **486 Credit Card rows** across the entire range (353 in training; only 7–22 in each main future quarter). These are administrative complaints from one system; different products do not establish separate authors. Cross-domain semantic near-duplicate linkage was not performed, so exact-overlap counts are a lower-bound diagnostic. Each replication model starts from the pinned pretrained checkpoint, never a Debt Collection fine-tune. Report an exact-cross-product-text exclusion sensitivity.

## Taxonomy and publication caveats

The [August 2023 taxonomy](https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf), pp. 6–8, describes core Credit Card issues and routes “Problem with credit report or credit score” to the reporting product. Nevertheless the authentic export retains reporting-style Issue values under Product=Credit card, including investigation, improper-use and incorrect-report labels. We retain the actual fields rather than repairing them to fit the form: this is administrative Issue prediction, not an expert-verified credit-card-only ontology. Stable strings do not establish stable routing/consumer interpretation.

The reporting-investigation class changes strongly in prevalence (5,848 training rows but 207 in 2026Q1). That is a count/taxonomy warning, not a model result or grounds to remove it. Per-class prior changes and excluded mass must be reported alongside calibration outcomes.

Counts show ample aggregate published narratives through 2026Q1. Publication fractions change; 2026Q2 stays exploratory, and July/August are excluded by the already frozen neural horizon. August has zero Credit Card narratives. No claim of missing-at-random publication or population representativeness follows from these sizes. We have not silently substituted another dataset or manufactured labels. An alternative dataset is unnecessary for this limited secondary estimand; if a uniformly precise subgroup-gate replication is required, this domain is inadequate without more data or a separately justified task.

Machine-readable evidence: `configs/credit_card_issues.yaml`; `outputs/neural_preparation/credit_card/{audit_summary.json,class_support.csv,period_support.csv,role_support.csv,template_families.csv,verified_near_duplicate_edges.csv,split_manifest.csv.gz}`. Preparation preserves the original pilot artifacts.
