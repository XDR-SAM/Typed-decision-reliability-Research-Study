# Neural runbook and resource plan

Status: preparation only. Do not execute training cells until the GPU phase is explicitly authorized. Protocol: [full_protocol.md](full_protocol.md); exact environment: `requirements-neural.txt`; frozen inputs: `protocol_freeze.json`.

## What is ready

The three primary adapters, hard-label CE, Debt Collection RLCD ablation, historical-only trainer, raw-logit exporter, temperature/gate/maintenance evaluator and dependence-aware bootstrap helpers are implemented under `src/neural/`. Preparation scripts write separate train/validation files, so training never parses future rows. CPU tests cover authentic one-hot supervision, exact shared input slices, chronological guards, calibration/gate family separation, future-selection rejection, tie/empty handling and paired statistics. GPU import/weight-loading/memory/throughput validation is intentionally deferred to the beginning of the next authorized phase.

Official source SHA: `fa9a2a7070b1789912a49ae24603bbfb1a78b001` (Laya 0.3.24). Laya checkpoint: `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. ModernBERT-large: `answerdotai/ModernBERT-large@45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. Never use a moving `main`, installed unpinned PyPI Laya, or a different checkpoint subfolder.

## Environment and data

Use a Linux Python 3.11 environment on the user-specified Kaggle 2×T4. T4 uses fp16, not bf16. `requirements-neural.txt` pins the complete set of direct experiment dependencies, including PyTorch 2.5.1, Transformers 4.48.3 and a Git-SHA-pinned Laya. This is a direct-dependency lock, not a claim that every transitive dependency or Kaggle image is immutable; capture `pip freeze`, Python, CUDA, driver and GPU details at run time. Preparation tests currently run in the local CPU environment; the exact Linux CUDA stack must pass the next-phase smoke checks before full training. A pin mismatch requires a logged environment amendment before neural outcomes.

Example next-phase setup, **not executed now**:

```bash
cd research-project
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-neural.txt
python -m pytest -q tests/test_neural_protocol.py tests/test_neural_statistics.py tests/test_metrics.py
```

Use the existing official-source release assets; do not reacquire a newer CFPB population. The newly prepared datasets are local at `outputs/neural_preparation/{debt_collection,credit_card}/`: `rows.jsonl.gz`, `train.jsonl.gz`, `validation.jsonl.gz`. These are large local artifacts, not automatically included in the earlier GitHub release. Copy them and their manifests to the GPU workspace, preserving paths, and verify `protocol_freeze.json`. No notebook credentials are hardcoded and no automatic Hub upload is enabled.

To recreate these preparation artifacts without refitting the CPU baseline, use the existing official `debt_collection.jsonl.gz`, `credit_card.jsonl.gz` and immutable pilot split manifest:

```bash
python data/audit_credit_replication.py  # counts/families only; no classifier
python data/prepare_neural_rows.py
```

Both preparation scripts use portable JSONL/CSV; neural training uses JSONL only. The original pilot's pickle-dependent tests remain in its original environment; the GPU setup runs the new portable contract tests and metric tests. Re-exported compressed files may differ bytewise by compression header/environment, and probabilistic candidate matching must retain the frozen parameters/software provenance; copy the completed frozen families/exports where possible. Record a preparation-only freeze amendment if restoring rather than copying the exact frozen files. Never amend labels or family definitions to improve model results.

## Ordered next-phase execution

1. Confirm package pins, two separate 16GB T4s and free disk. Verify the frozen files and source/model revisions. Load each selected checkpoint, record encoder tensor digests and tokenizer/input budgets, and check that all classes retain distinct option tokens. No actual CFPB future inference occurs in this step.
2. On **training rows only**, perform the memory/throughput smoke check. The first few updates may belong to the first actual run; do not use their outcomes to tune learning rates or losses. If OOM, use the frozen microbatch/accumulation fallback. Estimate remaining wall time from training throughput and record it. Check nonfinite loss, GradScaler behavior and encoder/head gradients.
3. Train Debt Collection: Laya-CE, matched-head, ModernBERT and RLCD+CE, each seeds 17/43/101. Train replication: Laya-CE, matched-head and ModernBERT, each seeds 17/43/101, freshly initialized from the pinned pretrained sources. No Debt Collection-to-Credit Card warm start. Four epochs per run; only validation selects the best epoch.
4. Once every domain run is complete, compare encoder-initialization digests and exact training IDs. Export raw logits/probabilities for validation/Q4/future quarters with the unchanged selected checkpoints. Future inference is blocked if any planned training completion or matched digest is missing.
5. Run post-hoc analysis on cached logits: October T, November threshold, December verification; raw/frozen/rolling and fixed-budget maintenance. Export all seeds and minority-class outcomes before interpretation. The CPU TF-IDF outputs stay unchanged.
6. Summarize the three fixed seeds; run `src.neural.compare` for ID-aligned paired NLL/macro-F1 cluster-bootstrap contrasts and Holm adjustments; produce figures from saved tables. `statistics.py` supplies the paired NLL, macro-F1 and family-ratio implementation. Final publication plotting and descriptive balanced-accuracy intervals are a reporting integration step after predictions exist, following the already frozen method; they cannot introduce model or threshold tuning.

Example one-run commands, **not executed during preparation**:

```bash
torchrun --standalone --nproc_per_node=2 -m src.neural.train \
  --domain debt_collection --model laya --seed 17 --execute-training
# Other --model values: matched_head, modernbert, laya_rlcd.
# For Credit Card: laya, matched_head, modernbert only.

python -m src.neural.infer --domain debt_collection --model laya \
  --seed 17 --execute-inference
python -m src.neural.posthoc --domain debt_collection --model-key laya_ce --seed 17
```

Trainer checkpoints: `best.pt` (selected weights) and `latest.pt` (weights/optimizer/scheduler/scaler/per-rank RNG at the end of an epoch). Use `--resume` only for an interrupted run with the same frozen config/microbatch. Save files out of the ephemeral session after each epoch; interruptions during a checkpoint write can corrupt that latest file, so retain an external prior-epoch copy. Unfinished runs never count as a complete seed and must not be replaced by a favorable partial epoch.

## Planning estimates, not benchmark measurements

The [official fine-tuning guide](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/docs/finetune.md) gives an approximate 4–5 hours for four epochs over ~30k questions with 1,024-token settings. Our 512-token state budgets, validation costs, hardware/load variability and implementation differ, so extrapolation is uncertain. These are conservative **wall-clock estimates for one seed using both T4s**, including training/epoch validation but excluding package installation/download and long bootstrap reporting:

| Run | Debt Collection, 34,343 rows | Credit Card, 25,182 rows | Persistent run storage |
|---|---:|---:|---:|
| Laya-CE (~421M) | 3–6 h | 2–4.5 h | ~7–9 GB with latest optimizer state |
| Same-Laya-encoder fixed head (~395M + small head) | 2.5–5 h | 2–4 h | ~6.5–8 GB |
| Generic ModernBERT-large fixed head | 2.5–5 h | 2–4 h | ~6.5–8 GB |
| Laya-RLCD+CE | 3–6.5 h | Not planned | ~7–9 GB |

RLCD samples only noisy logits after one encoder forward, so four reward samples do not imply four full encoder passes. Estimates may still be exceeded. Plan **~51–105 wall-clock training hours for all 21 runs**, plus ~0.25–1 hour/domain-model-seed for raw inference and CPU reporting overhead. Two GPUs mean roughly twice those GPU-hours, not one 32GB card. Free-session/quota interruptions may stretch work across weeks.

Each model's FP32 selected weights are roughly 1.6–1.7GB. Adam moments make a resumable latest checkpoint about 4.7–5.1GB, in addition to selected weights. Keep >=20–25GB free for a running session, model caches and data; retaining every optimizer checkpoint for 21 runs would need roughly 140–180GB. After safely archiving selected weights/predictions and deciding that resume is unnecessary, retain only required final optimizer provenance; a compact all-selected-weights archive is roughly 35GB plus predictions/tables. Never silently delete checkpoints during this preparation step.

Inference logits are small relative to weights; metadata/token cache can be larger. Bootstrap and fresh-label temperature fitting use cached logits on CPU. No GPU training should be attempted on the local 6GB RTX 2060 under this full-fine-tuning recipe.

## Readiness boundary

No conceptual or data-access blocker requires changing the authentic-label study. Gates may be unavailable and minority intervals wide; those are prespecified results, not reasons to retune. The practical checks still required in the GPU phase are hardware allocation, pinned-stack imports, exact weight loading and memory/throughput smoke verification. No full GPU or neural experiment has run during preparation.
