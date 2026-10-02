# GPU execution log — 3 October 2026

Protocol: `cfpb-neural-v1`. Approved frozen base: `3d3b711c41259eea4433d6714fd2002bc68b99b0`.

The user explicitly authorized the GPU phase. The logged-in Kaggle notebook is `sami787/cfpb-frozen-protocol-laya-ce-seed-17`. GPU T4 x2 was selected and Internet was already enabled. The account reported 30 accelerator hours remaining at launch.

The first executed environment cell detected two Tesla T4 devices, each 15,360 MiB; NVIDIA driver 580.178.04; driver-reported maximum CUDA 13.0; and Kaggle kernel Python 3.13.15. The frozen training environment is therefore installed in a separate Python 3.11 environment under `/kaggle/temp/cfpb-neural-env`. This keeps the Python/dependency protocol unchanged; the notebook kernel only launches subprocesses. Driver-reported CUDA is not the PyTorch runtime version.

The approved commit was cloned. All six prepared exports were restored from the checksum-verified release archive. No future predictive outcomes were loaded or inspected during setup.

New `gpu_smoke.py` is an execution/instrumentation entrypoint, not a change to frozen methods. It checks full tokenizer token-to-ID semantics and preprocessing/merge definitions, pinned environments/files, two-T4 NCCL, each model's checkpoint loading, fp16 forward/backward, gradients, optimizer/scaler/scheduler and checkpoint/RNG round-trip using training rows only. Disposable smoke updates are discarded and the real run freshly initializes from the pinned checkpoint with seed 17. Only frozen OOM fallback microbatches 4, 2, 1 are permitted.

New `execute_gpu.py` adds optimizer timing/VRAM logs while executing the unchanged frozen trainer. It does not alter loss, batch size, selection, model, prompts, labels, data, or optimization. Setup, smoke, and real-run results are recorded separately. No amendment is implied by adding instrumentation. Any actual integration failure must be recorded before continuing the affected run.
