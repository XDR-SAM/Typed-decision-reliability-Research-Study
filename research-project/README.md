# CFPB Debt Collection Issue feasibility pilot

CPU-only TF-IDF + Logistic Regression. No Laya/ModernBERT training and no GPU use. The project stops at a GO/NO-GO report.

Start with `research/pilot_report.md`, `research/go_no_go.md`, `research/novelty_update.md` and `research/decisions_log.md`. Raw source provenance is in `data/source_manifest.json`. CSV audit/metric tables and PNG plots are under `outputs/`.

## Reproduce in PowerShell

Use the project virtual environment (Python 3.14), or create an environment with the versions in `requirements-lock.txt`.

```powershell
.\.venv\Scripts\python.exe data\download_cfpb.py
.\.venv\Scripts\python.exe data\audit_cfpb.py
.\.venv\Scripts\python.exe data\preprocess.py
.\.venv\Scripts\python.exe data\deduplicate.py
.\.venv\Scripts\python.exe data\make_splits.py
.\.venv\Scripts\python.exe data\complete_audit.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe src\tfidf_baseline.py
.\.venv\Scripts\python.exe research\verify_outputs.py
.\.venv\Scripts\python.exe research\build_report.py
```

Downloading needs network permission. Training uses CPU and caps BLAS threads at four. All classes and model choices are historical-only. Three C values are compared on Jul–Sep 2024, with one diagnostic ablation refit using the selected C. No random-split headline, no post-hoc calibration, no frozen 90% gate certification, and no full paper experiment are performed.

## Interpretation

The requested >=3/4 rule is implemented in the report generator; prespecified operational definitions and special warnings are in `configs/data.yaml`. This is feasibility, not proof of Laya benefit or paper novelty. Since future performance is examined here, later work must disclose this pilot rather than presenting those aggregate cohorts as untouched confirmatory holdouts.
