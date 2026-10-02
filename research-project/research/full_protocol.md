# Frozen full neural experiment protocol — cfpb-neural-v1

Freeze: **3 October 2026, Asia/Dhaka**, before any neural training or neural predictions. Scope authorized: protocol, preparation and tests. This file is a prospective analysis plan for the next neural stage; it is **not an externally registered preregistration** and does not make already inspected data pristine.

Working title: **Do Frozen Confidence Gates Survive Temporal Shift? A Controlled Retrospective Audit of Laya on Consumer Complaint Issues.**

## Evidence and prior decisions

Read the existing acquisition/preprocessing/model/evaluation scripts, configs, tests, report, decisions log, GO/NO-GO decision, literature review and artifact inventory. The CPU pilot remains intact. Its 4/4 GO is a feasibility decision, not a Laya result. Debt Collection 2024Q4–2026Q2 predictive outcomes were inspected during that pilot. Debt Collection is therefore a retrospective temporal replay/development study. Credit Card was previously counted, but no classifier was trained or predictive result inspected; its labels and neural analysis are now independently specified. The two products come from the same administrative system, so replication is not a statistically independent population or literal prospective deployment.

Official acquisition URLs/download dates/SHA256 remain in `data/source_manifest.json`. Labels are original administrative/consumer-selected CFPB Issue values, not expert-adjudicated semantic truth. Results concern the published narrative subset. Complaint receipt dates do not establish narrative publication or label availability at that date. Archive snapshots were downloaded after the nominal deployment periods; publication selection and delay prevent a literal historical deployment reconstruction.

## Resolved Laya hard-label formulation

The selected source is official [Laya Git SHA fa9a2a7](https://github.com/NandhaKishorM/laya/tree/fa9a2a7070b1789912a49ae24603bbfb1a78b001), package version 0.3.24. We inspected its fine-tuning documentation, full Kaggle notebook, single-device trainer, DecisionModel, proper_reward, collator, loading code and training tests. Sources/configs/metadata are archived under `research/sources/neural/`; no pretrained weights were downloaded during preparation.

1. **Hard labels work technically.** `DecisionModel.forward` returns raw option logits. Hard-label cross-entropy is exactly the one-hot special case of the official soft CE. The reward's documented target type explicitly permits one-hot or soft distributions, and upstream tests evaluate rewards against one-hot observed outcomes. [Pinned common.py](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/laya/common.py), [upstream training tests](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/tests/test_training.py).
2. **CE-only is not a shipped CLI mode.** The inspected official trainer/notebook implement RLCD plus CE; no documented CE-only switch was found. Our CE-only implementation is a clearly labeled custom training loop using the official architecture/checkpoint, not an asserted official recipe. [Trainer](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/research/scripts/finetune_single_device.py), [fine-tuning documentation](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/docs/finetune.md).
3. **RLCD does not require genuine soft targets.** The worked benchmark uses teacher distributions; that is its data recipe, not a restriction of the scoring function. For repeated draws of an administrative label Y conditional on narrative X, the expected proper score motivates estimation of P(Y|X). One observed one-hot label is a realized outcome; it is not evidence that the underlying conditional distribution is degenerate. Finite-data fitting, ambiguity, dependence, optimization and shift can still cause miscalibration. Clipped log rewards and noisy policy-gradient optimization further prevent claiming a theorem of calibration for the fitted model. This interpretation is our methodological inference from the source, not an empirical result.
4. **Primary: Laya-CE.** Minimize unweighted hard-label multiclass CE, without label smoothing, auxiliary act losses or invented teacher probabilities. Keep the typed `choice` readout. This gives every primary neural model the same authentic supervision and loss. It evaluates reliability of an adapted typed model, not the unique causal benefit of RLCD.
5. **One prespecified ablation: Laya-RLCD+CE on Debt Collection.** Same checkpoint/rows/seeds/optimizer/epochs; one-hot recorded labels in the official noisy-logit policy-gradient objective plus unit-weight CE. Four samples, sigma .4→.1 across four epochs, spherical weight .75, no ordinal reward for unordered Issue classes, log floor -9.21. This is valid supervised proper-score training under the recorded-label estimand, not teacher-distribution reproduction. No Credit Card RLCD run is added. CE versus RLCD is an attribution check, not novelty.
6. **No synthetic soft labels.** LLM teacher probabilities would change the supervision, add teacher bias/cost/contamination, and obstruct the authentic-label comparison. No distillation, soft-label invention or manual Issue merging is permitted. The source's Apache-2.0 license permits use/modification subject to its terms; one-hot targets are supported inputs. [License](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/LICENSE).

## Fixed models and attribution limits

| Model | Pinned initialization | Input/readout | Role |
|---|---|---|---|
| Laya-CE | `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, English root, full checkpoint | Fixed typed choice; original options at marker positions | Primary TDM |
| Matched conventional | Exact encoder tensor objects extracted from that same Laya revision | Identical encoder token IDs; mean over narrative hidden states; LayerNorm, dropout .1, new K-class linear head | Required initialization/input control |
| ModernBERT conventional | `answerdotai/ModernBERT-large@45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | Same narrative token slice, conventional narrative-only sequence; same fixed head as matched control | Practical generic pretrained baseline |
| Laya-RLCD+CE | Same Laya revision | Same typed readout, authentic one-hot outcomes | Debt Collection ablation |
| Existing TF-IDF | Existing saved CPU pilot checkpoint/results | Existing word TF-IDF/logistic regression | Contextual baseline; no refit |

The matched control's encoder digest must equal Laya's before training. A generic ModernBERT checkpoint is not this control: Laya's encoder has already been adapted. The matched control includes the same constant question/options in the encoder input, so prompt tokens and state budget are controlled; its output remains an ordinary fixed K-class head. The external baseline sees narrative-only input. No true Issue, Product, Company, response, Sub-issue, tags, geography or consent field enters state; every option is the same constant vocabulary for every example.

**Interpretation limit:** the typed head is pretrained, larger and structurally different; the new conventional head is randomly initialized. Thus the matched comparison isolates encoder initialization and input, but compares **readout packages including head prior/capacity**, not a pure causal effect of typed training. All primary models use CE. A claim of purely architectural superiority would require additional head-initialization/capacity controls; these are not silently introduced after results. The static fixed label set also does not test arbitrary unseen decision schemas.

## Data, labels and family handling

Debt Collection retains all seven pilot labels unchanged, with 34,343 earliest-exact-deduplicated training examples. Credit Card retains 12 original labels with >=200 training narratives, selected without predictive outcomes or future-performance filtering, yielding 25,182 exact-deduplicated training examples. See `configs/*_issues.yaml` and the replication audit. Excluded labels stay excluded and their population mass is reported; no unknown-label row is silently recoded.

Use existing lexical family definitions: exact/canonical matching plus 128-permutation MinHash, normalized word 5-shingles, .70 LSH candidates, verified Jaccard >=.85, transitive connected components. Credit Card applies the identical method. Families are dependence proxies, not consumer identities; paraphrases may be missed and transitive endpoints need not exceed .85 similarity. Full-horizon unsupervised text is used only to audit/group dependence; future labels or predictions do not define inclusion, model features or hyperparameters.

Training: retain the earliest exact narrative; keep historical ambiguous labels documented rather than relabeling. Validation: all included historical narratives, matching the pilot; report its historical-template exposure as a limitation. Training pairs use identical Complaint IDs, labels, ordering and distributed sampler schedule across neural models. The standard DDP sampler pads at most one training row per epoch to split evenly between two GPUs; this small, documented padding is identical across models, not additional unique training evidence.

Operational evaluation retains every included narrative and its repetition. Primary uncertainty clusters entire families. Sensitivities: (i) one representative per quarter excluding any family observed in **any earlier period**, including calibration and prior evaluation; (ii) retain the original pilot's train/validation-only exclusion for comparability; (iii) Credit Card exclude exact texts also observed in Debt Collection, without claiming semantic cross-product decontamination. Strict novel-family results concern a different population, not a corrected estimate of the operational stream.

## Chronology, inputs and model selection

| Role | Dates | Permitted use |
|---|---|---|
| Train | 2023-09-01–2024-06-30 | Weight updates only |
| Validation | 2024-07-01–2024-09-30 | Epoch/checkpoint selection only |
| Temperature fit | October 2024 | Scalar NLL temperature fit |
| Gate selection | November 2024 | Historical threshold selection |
| Independent gate verification | December 2024 | One-shot verification; no reselection |
| Main evaluation | 2025Q1–2026Q1, separately | Frozen model/calibration/gate outcomes |
| Exploratory | 2026Q2 | Explicitly exploratory only |
| Excluded | July/August 2026 | Availability audit only |

Q4 is a three-role development period, not a single set reused for fitting and verification. October/November/December role datasets exclude families in train/validation and all earlier Q4 roles, and keep one chronological representative per surviving family. Eligible counts: Debt Collection **4,191 / 3,753 / 4,373**; Credit Card **2,620 / 2,446 / 2,724**. The full monthly streams remain available for descriptive operational diagnostics, but are not the independent gate-development samples. Decontamination changes the historical reference population; never advertise its confidence bound as a guarantee for the future full stream.

One fixed question: “Which official CFPB Issue best matches this complaint?” Original labels, alphabetical order, no handcrafted class descriptions or prompt search. No ensembling or post-result option rearrangement. Literal `[MASK]` is neutralized identically in state to protect Laya's marker format; legitimate financial vocabulary stays intact.

Pinned Laya tokenizer; 512 total tokens, head limit 256. All options remain distinct and untruncated. The measured fixed head uses 62 tokens for Debt Collection and 103 for Credit Card, leaving **449 and 408 narrative tokens** respectively. Keep the first state tokens; no sliding windows. Matched input is token-identical to Laya. ModernBERT receives `[CLS] same_state_slice [SEP]`, with the same token vocabulary/merge rules; native long-context capacity is deliberately not used. Log truncation frequency by domain/cohort and class. Long-context follow-up would be a separately amended experiment.

All models: seeds **17, 43, 101**; full encoder fine-tuning; four epochs; AdamW, encoder LR 2.5e-5, head LR 1e-4, weight decay .01, betas .9/.999, epsilon 1e-8, 6% warmup then cosine schedule, gradient clip 1.0, fp16 autocast/GradScaler and non-reentrant gradient checkpointing. Two T4 GPUs, microbatch 4/GPU × 8 accumulation = global 64. OOM-only prespecified fallback: microbatch 2 with accumulation 16, then 1 with 32, preserving tokens, rows, effective batch, loss and epochs. Record the fallback before result inspection and use matching settings for the paired control. GPU memory is not pooled across cards.

No learning-rate/prompt/loss/length search. For each seed/model choose the epoch with highest Jul–Sep macro-F1; exact ties use lowest validation NLL, then earliest epoch. Calibration/test metrics cannot enter selection. Archive every epoch's validation record and exact final selection. Finish all 12 Debt Collection runs and all 9 replication runs before future inference in the corresponding domain. If resource/time problems require fewer seeds/models, amend the protocol **before future neural results**, report incompleteness and never select the best seed.

## Calibration and historical gate rule

Cache raw float32 logits and raw probabilities before any post-hoc adjustment, with IDs/date/Issue/family and model revision/seed. Infer from the raw forward pass, bypassing inherited Laya temperature buckets, entropy confidence and act head. Confidence is max softmax probability. Laya's runtime temperature clamp and inherited per-option buckets must not silently alter externally fitted temperatures. [Pinned calibration source](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/laya/common.py).

Fit one scalar positive temperature by October NLL minimization, log-space bounded to [.05,20]. Require >=100 rows and >=2 represented classes; boundary fits are flagged. A failed initial fit reports failure and raw T=1 fallback, not successful calibration. Serialize the exact float64 scalar separately from Laya's service config. Freeze it and apply unchanged to later quarters. Positive temperature cannot repair full-coverage prediction errors; multiclass confidence rankings may change.

November threshold grid is fixed: .50,.51,…,.99,.995 (**51 candidates**), accepted if confidence >= threshold; confidence ties stay together. On independent family representatives, require accepted n>=200 and coverage >=10%. Compute a one-sided Clopper–Pearson lower bound using alpha=.05/51; a candidate qualifies only if its lower bound >=90% recorded-label accuracy. Among qualifying candidates choose greatest coverage, breaking ties by smallest threshold. These bounds assume independent representative outcomes; the family proxy does not establish that assumption in real consumers. They are a conservative historical selection rule, not a prospective guarantee.

If no candidate qualifies, save a null gate and report **no certifiable historical gate under the frozen rule**. Do not reduce the target, loosen the rule, invent a threshold from later quarters or claim 0% coverage satisfies the target. Probability and classification endpoints remain evaluable. Fixed diagnostic thresholds .50/.60/.70/.80/.90/.95 and full risk–coverage curves are descriptive and do not rescue an unavailable primary gate.

December tests the exact chosen T/threshold once: >=200 accepted families, >=10% coverage and one-sided 95% lower bound >=90%. A failed verification is reported and the unchanged candidate is still followed diagnostically into the future, explicitly as a historically unverified gate. No November reselection or pooled Q4 refit follows December outcomes. Future target survival is an empirical claim only when simultaneous quarter bounds support it; intervals crossing the target are inconclusive, not proof of survival.

Report per quarter/model/seed: accuracy, macro-F1, balanced accuracy, all class precision/recall/F1/confusion; NLL, summed multiclass Brier (0–2), 15-bin equal-width ECE, 15-quantile approximately equal-mass ECE with ties intact, mean confidence, confidence minus accuracy and reliability diagrams; coverage, accepted error/accuracy and risk–coverage curves. ECE alone is never the headline.

Prespecified secondary gate endpoint conditions on **true recorded Issue**, not predicted Issue: support, accepted count, distinct accepted families, within-class coverage, accuracy/risk and dependence-aware 95% intervals. Aggregate correctness can hide zero minority coverage or catastrophic minority error. No class-specific thresholds are fitted. Empty sets have undefined/null accuracy and intervals. Under 30 accepted families, show counts and a conservative [0,1] interval rather than a headline percentage; 30–99 families are explicitly imprecise. At least 100 families does not by itself certify a class's 90% target.

## Maintenance and label budgets

Weights never change. Main arms: raw; frozen October T; full preceding-quarter temperature recalibration. For 2025Q1, preceding-quarter maintenance uses 2024Q4 (including its full available stream), while the frozen arm retains October only. This initial comparison changes both recency and calibration sample size and is disclosed; all later rolling fits use exactly the immediately previous quarter. Never use the current quarter or future quarter to fit its temperature.

Budget arms: 100/500/2,000 fresh labeled records per previous quarter, uniform sampling without replacement, 20 prespecified repeated draws, nested budgets within a draw. Seeds depend only on fixed sampling seed, period and draw; identical IDs across models/training seeds. Sample the entire prior stream, not accepted predictions and not strata defined by unseen true labels. Record repeated-family exposure and distinct families per budget. If fewer than the requested records exist, mark that budget unavailable; do not call a smaller sample “2,000 labels.” No query/adaptive selection, gate reselection, bias vector or retraining is added.

Keep the exact original numerical threshold for all recalibration arms. Changes in coverage therefore reflect probability maintenance. Report restoration failures as well as improvements. The full prior-quarter arm is an upper-information comparator; fresh-label budget curves estimate statistical sample demand, **not measured labeling cost**. Official Issue may already exist in an internal intake system. Public archive publication/label delay is unknown, so this is idealized immediate-feedback replay, not a proven deployable feedback schedule. A future delay study requires an explicit amendment/data source.

In strict novel-family sensitivities, exclude every previously seen family, including prior labeled maintenance periods; repetitions remain in the operational arm. A temperature fitted on representatives versus a full record stream targets different prevalences; our fixed historical T uses representatives and the practical maintenance stream uses records. Report this population distinction and repeat the no-prior-family sensitivity, rather than attributing any benefit solely to semantic drift.

## Endpoints and statistical rules

Primary model contrast: **equal-quarter mean frozen-temperature future NLL difference, Laya-CE minus matched-head**, averaged across the three fixed seeds after computing each seed's losses separately. Never average probabilities/logits into an unplanned ensemble. Primary gate endpoint: per-quarter accepted-risk excess above .10 together with coverage, December status and availability of a qualifying gate. A missing gate cannot be assigned zero risk. Accuracy is contextual, not the sole headline.

Unit of resampling: **global template family**, with shared resampling weights across quarters, models and seeds. Retain every row/multiplicity inside a sampled family; this preserves cross-quarter family links. For comparisons, align exact Complaint IDs and original label order. Use 2,000 ordinary cluster-bootstrap replicates, seed 20261002, percentile 95% intervals. The principal NLL contrast uses a centered two-sided paired cluster-bootstrap test at .05. Intervals are conditional on the observed calendar cohorts and these three seeds; report each seed's estimate/range separately. Three seeds do not support strong claims about the full training-randomness distribution. Lexical clusters miss other consumer/company/time dependence; bootstrap intervals are approximate, not deployment certificates.

Gate aggregate intervals use 10,000 family-bootstrap replicates, with Bonferroni-adjusted 95% simultaneous coverage over the five main quarters (alpha .01 per quarter). For a particular model/seed, label its historical gate “supported across observed quarters” only if all five lower accuracy bounds >=.90, minimum coverage >=.10 and sufficient accepted families hold. Any upper bound <.90 indicates supported failure in that quarter; crossing intervals are inconclusive. Across models/seeds these are separate descriptive audits, not a global familywise discovery claim. True-class intervals are pointwise 95% descriptive; no subgroup-wide certification is asserted.

Prespecified secondary paired contrasts: matched versus Laya macro-F1; ModernBERT versus Laya frozen NLL; rolling-full versus frozen Laya NLL; Debt Collection RLCD versus CE frozen NLL. Holm adjustment across four contrasts in Debt Collection and the three applicable contrasts in Credit Card; no fabricated replication RLCD contrast. Report unadjusted effects/intervals as well. Macro-F1/balanced accuracy differences use the same global-family weights to recompute confusion matrices, not rowwise t-tests. Different model gates accept different subsets: compare risk **and coverage**, and add common-coverage descriptive curves, without claiming that conditional accepted accuracy alone is a paired competence contrast.

Exploratory 2026Q2, diagnostic thresholds, budget draw curves, prior drift, text-length changes and class/publication/template associations do not enter primary significance. With only five quarters, do not claim a causal drift mechanism or a robust correlation test. Label-prior standardization or classwise threshold learning would require a separately labeled exploratory amendment. Data/performance stability, no Laya advantage, harmful recalibration and no qualifying gate are valid outcomes.

## Pretraining chronology and contamination limits

| Initialization | Verifiable date | Documented pretraining cutoff |
|---|---|---|
| Selected Laya English root | Hub repository created 2026-09-18; pinned revision last modified 2026-09-24 | Not disclosed for complete Laya adaptation corpus; exact latest-document date unknown |
| Matched encoder | Same 2026 Laya checkpoint | Same unknown cutoff/exposure as Laya |
| Generic ModernBERT-large | Official public announcement 2024-12-19; Hub created 2024-12-11; pinned repo last modified 2025-01-15 | Reviewed card/paper do not specify a verifiable maximum-document date for all pretraining data |
| RLCD ablation | Same selected Laya initialization | Same Laya limitation |

[Laya card/revision](https://huggingface.co/convaiinnovations/laya/tree/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), [ModernBERT revision](https://huggingface.co/answerdotai/ModernBERT-large/tree/45bb4654a4d5aaff24dd11d4781fa46d39bf8c13), [official ModernBERT announcement](https://www.answer.ai/posts/2024-12-19-modernbert.html). Hub creation/commit dates are publication evidence, not training-data cutoffs. Laya was released after every main evaluation quarter and may have encountered language or data from those dates. Unreported CFPB exposure cannot be ruled out. The encoder-matched control shares this exposure; generic ModernBERT does not control it. Never claim literal 2024 prospective deployment or contamination-free holdouts.

## Frozen research questions and stopping boundary

RQ1: After historical adaptation, how do unchanged calibration and historically assessed gates behave across later published complaint cohorts, including class-conditional failure and coverage?

RQ2: Under identical encoder initialization, inputs, authentic supervision and training budget, does the Laya readout package differ from a fixed conventional head in temporal probability/gate reliability? How does a generic ModernBERT baseline compare?

RQ3: Can preceding-quarter scalar recalibration maintain reliability with frozen weights/gate, and what happens at 100/500/2,000 recent labels?

RQ4: Do the same conclusions replicate in independently specified Credit Card Issue prediction, and how sensitive are they to template novelty and published-narrative composition?

All configs/source revisions/label policies/preparation manifests and code are hashed in `protocol_freeze.json`; changes require an amendment logged before affected neural outcomes. No Laya/ModernBERT weights, predictive runs, GPU training or Credit Card pilot are authorized by this preparation step. Stop after documentation, export and CPU tests.
