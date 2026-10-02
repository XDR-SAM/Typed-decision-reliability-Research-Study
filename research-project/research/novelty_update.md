# Focused novelty update

Search date: 2 October 2026. This is a focused update to the prior skeptical review, not an exhaustive new systematic review. A successful feasibility pilot does not establish paper novelty.

## Queries used

- CFPB Issue classification; CFPB Issue classification calibration
- CFPB temporal classification; CFPB temporal calibration
- CFPB selective prediction
- Laya temporal calibration; Laya calibration drift
- typed decision model deployment shift; typed decision models temporal shift
- confidence threshold transfer temporal calibration maintenance

## Closest verified evidence

| Source | Status | Implication |
|---|---|---|
| [TDM audit, arXiv:2609.32160](https://arxiv.org/abs/2609.32160) | September 2026 preprint; revisited | Deployment-shift reliability remains a stated evidence gap. It includes hosted-version/workload issues broader than this study. |
| [Laya assessor study, arXiv:2609.33843](https://arxiv.org/html/2609.33843v1) | September preprint, inspected in preceding review | Calibration/selective escalation already tested; authored OOD documents are not calendar-time CFPB cohorts. |
| [Jev crash study, arXiv:2609.24052](https://arxiv.org/html/2609.24052v1) | September preprint, inspected previously | Held-out random-half recalibration of a real workload; not evidence of future-period transfer. |
| [OpenJev-RLCD, arXiv:2609.38850](https://arxiv.org/html/2609.38850v1) | September preprint, inspected previously | CE/RLCD and calibration comparisons are already available; no matched CFPB temporal experiment found. |
| [Human Factors Framework](https://www.mdpi.com/2078-2489/17/9/824) | Published journal article; indexed primary text | Temporal CFPB classification is established; target is a response-derived severity proxy. |
| [Think Before You Classify](https://www.mdpi.com/2079-9292/14/6/1070) | Published journal article, inspected previously | Complaint classification with LLMs is established. |
| [Remedy-Aware Financial Innovation](https://www.americaspg.com/article/4448/pdf/stream) | Journal article; methods revisited | Platt fitting on September–October and evaluation on November–December 2024 already establishes chronological CFPB calibration evaluation. |
| [State Conditional Boosting](https://link.springer.com/article/10.1007/s43069-026-00695-2) | September journal article; abstract-only access in preceding review | Prospective CFPB workload, probability calibration, drift and periodic refitting further constrain broad novelty. Full transfer details remain unresolved. |
| [Second Thought](https://github.com/KNambiarDJsc/second-thought) | Newly inspected repository, not peer-reviewed evidence | Already implements typed-model calibration monitoring over time and review/correction infrastructure. README evidence is small/toy demonstrations, not this natural longitudinal controlled experiment. Do not claim a new monitoring concept. |
| [Laya financial sentiment adaptation](https://huggingface.co/shanaka95/laya-fintiment) | Newly inspected author model card | Reports a stratified financial sentiment test and warns about domain drift; no empirical calendar-shift study demonstrated there. Financial Laya adaptation alone is not novel. |
| [PLaND](https://saspublishers.com/article/24800/) | September 2026 journal abstract; direct page access failed | Includes CFPB in code/model selective execution, with failure at selection and no final CFPB test. This is not frozen posterior calibration transfer. Treat findings as abstract-level, not independently replicated. |
| [Online Platt Scaling with Calibeating](https://proceedings.mlr.press/v202/gupta23c.html) | ICML 2023 | Frozen versus windowed/online probability maintenance is established. |
| [Selective QA under Domain Shift](https://aclanthology.org/2020.acl-main.503/) | ACL 2020 | Confidence-based abstention under shift is established. |
| [kotoba typed-decisions](https://github.com/kotoba-lang/typed-decisions), [Sev](https://github.com/LakoreAI/sev) | Repositories, reviewed previously | Matched backbone/loss controls are required controls, not new research questions. |

Search also surfaced a confidence-gate/temporal MovieLens project page, but direct access failed. It is an unresolved lead, not a verified basis for claiming either novelty or redundancy. Search results and model-card warnings are not substitutes for complete empirical methods.

## Assessment

I did not find a verified paper performing the complete combination of Laya, authentic calendar-time CFPB Issue cohorts, same-initialization conventional controls, frozen accepted-error gates, and recent-label maintenance budgets. This qualified negative search result does not justify “first ever.” The contribution would need to be the controlled evidence and useful explanation of reliability behavior, not a new dataset/task, calibration metric, monitoring interface or demonstration that drift can happen.

Even if the four feasibility conditions pass, publication potential remains uncertain. The full paper must also address published-narrative selection, template dependence, administrative-label ambiguity, and model pretraining chronology. This pilot evaluates all listed future cohorts, so their aggregate performance is now exploratory knowledge; a later paper must not describe them as wholly unseen preregistered confirmatory data.

No Laya repository code was installed or executed. No neural weights were downloaded for this pilot. [Laya repository](https://github.com/NandhaKishorM/laya), [fine-tuning documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md), and [Wild-Time](https://github.com/huaxiuyao/Wild-Time) remain references only.

## Pre-neural update — 3 October 2026

The preceding section describes the completed CPU pilot. In this subsequent preparation stage we downloaded and inspected the official Laya source snapshot, notebook, configuration/tokenizer files and model metadata; tiny CPU contract tests import mathematical/formatting helpers. No pretrained weights, neural optimization or CFPB neural predictions are involved.

Fresh focused searches used: `Laya temporal calibration drift confidence gates CFPB`; `CFPB Issue classification selective prediction calibration temporal 2026`; `Laya frozen calibration temporal shift`; `typed decision model deployment shift calibration`; `typed decision class-conditional calibration`; `CFPB confidence threshold Issue`; `Laya same encoder calibration CE RLCD`; `typed decision rolling recalibration`; and `confidence rolling recalibration temporal classifier`. Searches include the current client date, 3 October 2026, but indexing delays prevent an exhaustive same-day census. Unrelated uses of “Laya” (EEG/music/software) were excluded.

### Additional primary sources checked

| Source | Evidence inspected | Consequence for novelty/protocol |
|---|---|---|
| [Official Laya source, pinned SHA](https://github.com/NandhaKishorM/laya/tree/fa9a2a7070b1789912a49ae24603bbfb1a78b001) | Source/notebook/docs/reward tests; no model performance reproduced | Hard one-hot reward targets are supported. CE-only is a custom loop, not a shipped trainer switch. Source already implements confidence gates and slice monitoring; neither is a new concept. |
| [CE-only Laya reproduction model card](https://huggingface.co/minhleduc/laya-typed-decisions-ce-1024) | Author's benchmark/objective description | CE-only adaptation and CE/RLCD comparisons are established. Its teacher-derived synthetic benchmark does not demonstrate genuine calendar-time CFPB reliability. No novelty claim for the loss ablation. |
| [COGNIT-Guard, arXiv:2609.33671](https://arxiv.org/abs/2609.33671) | Primary preprint abstract/method overview | Calibrated Laya/direct-decision confidence escalation under latency and false-positive constraints is already studied. Not verified evidence of the same frozen calendar-time Issue/maintenance experiment. |
| [Jev versus open decision models](https://github.com/elcronos/jev-vs-open-decision-models) | Repository README/protocol/output inventory | Frozen zero-shot comparisons, calibration, selective prediction and conventional supervised context already exist. Calendar-time CFPB Issue calibration/maintenance with exact Laya-encoder controls was not found there. |
| [Laya key-value fine-tuning project](https://github.com/sothi-em/laya-finetuning) | README/training and release-gate descriptions | Hard-label/domain adaptation, strict held-out gates and failure-driven revisions are existing engineering practice. The controlled retrospective temporal study remains a different empirical setting. |
| [ModernBERT announcement](https://www.answer.ai/posts/2024-12-19-modernbert.html), [official model card](https://huggingface.co/answerdotai/ModernBERT-large) | Primary release/training description | December 2024 release predates Laya; initialization chronology must be disclosed and generic ModernBERT cannot substitute for exact Laya-encoder attribution control. |

Previously verified overlap remains material: the [Laya assessor preprint](https://arxiv.org/abs/2609.33843) already reports failed historical selective-risk targets and OOD probes; the [TDM audit](https://arxiv.org/abs/2609.32160) identifies deployment-shift evidence gaps but does not confer novelty. CFPB classification/chronological evaluation are already established by the Human Factors, Think Before You Classify, Remedy-Aware and workload studies listed above. Online/windowed calibration maintenance and selective prediction under shift are older concepts.

**Updated conclusion:** I did not locate a verified primary work covering the full conjunction of authentic CFPB Issue labels, Laya, calendar-time replay, frozen historically assessed confidence gates, true-Issue selective reliability, exact encoder/input conventional control, and fixed recent-label maintenance budgets. This is a qualified search result, not “first ever,” an exhaustive citation review or a guarantee of publication. The defensible contribution is comparative evidence, including failures/nulls and mechanisms/limits, not a new loss, gate, dataset or discovery that calibration can drift.

The required exact encoder control and independently specified Credit Card domain improve interpretability. They do not eliminate head-prior/capacity, administrative-label, publication-selection, source-contamination or prior-pilot-inspection limitations. The selected 2026 Laya initialization is later than the nominal evaluation dates; a literal prospective-deployment claim would be invalid.
