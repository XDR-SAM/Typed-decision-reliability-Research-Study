# GPU execution log — 3 October 2026

Protocol: `cfpb-neural-v1`. Approved frozen base: `3d3b711c41259eea4433d6714fd2002bc68b99b0`.

The user explicitly authorized the GPU phase. The logged-in Kaggle notebook is `sami787/cfpb-frozen-protocol-laya-ce-seed-17`. GPU T4 x2 was selected and Internet was already enabled. The account reported 30 accelerator hours remaining at launch.

The first executed environment cell detected two Tesla T4 devices, each 15,360 MiB; NVIDIA driver 580.178.04; driver-reported maximum CUDA 13.0; and Kaggle kernel Python 3.13.15. The frozen training environment is therefore installed in a separate Python 3.11 environment under `/kaggle/temp/cfpb-neural-env`. This keeps the Python/dependency protocol unchanged; the notebook kernel only launches subprocesses. Driver-reported CUDA is not the PyTorch runtime version.

The approved commit was cloned. All six prepared exports were restored from the checksum-verified release archive. No future predictive outcomes were loaded or inspected during setup.

New `gpu_smoke.py` is an execution/instrumentation entrypoint, not a change to frozen methods. It checks full tokenizer token-to-ID semantics and preprocessing/merge definitions, pinned environments/files, two-T4 NCCL, each model's checkpoint loading, fp16 forward/backward, gradients, optimizer/scaler/scheduler and checkpoint/RNG round-trip using training rows only. Disposable smoke updates are discarded and the real run freshly initializes from the pinned checkpoint with seed 17. Only frozen OOM fallback microbatches 4, 2, 1 are permitted.

New `execute_gpu.py` adds optimizer timing/VRAM logs while executing the unchanged frozen trainer. It does not alter loss, batch size, selection, model, prompts, labels, data, or optimization. Setup, smoke, and real-run results are recorded separately. No amendment is implied by adding instrumentation. Any actual integration failure must be recorded before continuing the affected run.

## First integration attempt

Pinned installation and all frozen hashes passed; 17 neural contract tests passed in 8.68 seconds on Kaggle. Laya's initial smoke step passed at microbatch 4, accumulation 8, global batch 64 and 512 tokens: 6.47494 seconds and 11,812,768,256 allocated bytes on each GPU. Its original encoder digest was `7ac385179d24b87045da78b8d6590cc408ecd9439ebb19f143292b1e97466cb0`; checkpoint/model/optimizer/scaler/scheduler/RNG restoration passed.

The matched-head smoke then raised `AssertionError: Nonfinite gradient classifier.2.weight` at the initial fp16 scale 65,536. The original diagnostic assertion ran before GradScaler could perform its normal overflow backoff. This is a smoke-harness defect: the frozen trainer already delegates overflow handling to GradScaler. The diagnostic now permits skipped overflow steps with verified backoff, requires two successful optimizer steps, and fails on persistent nonfinite gradients. No frozen implementation or hyperparameter changed. No actual neural training run, validation evaluation, calibration outcome, or future neural prediction had been observed at this point; only training-row smoke forward/backward computations occurred. Smoke weights are discarded before actual initialization.
