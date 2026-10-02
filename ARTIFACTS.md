# Complete study artifacts

Large files are distributed in the [pilot-2026-10-02 GitHub release](https://github.com/XDR-SAM/Typed-decision-reliability-Research-Study/releases/tag/pilot-2026-10-02), because processed data exceed GitHub's ordinary Git file-size limit.

Download both ZIP files and extract them into the repository root, preserving their directory structure:

- `official-cfpb-sources.zip`: all original official CFPB downloads, including the earlier literature-review archive copies.
- `pilot-data-models-predictions.zip`: processed datasets, fitted CPU pilot models, environment provenance, and prediction files.

All other study sources, reports, audit tables, metrics, figures, and code are tracked in Git. Only installed dependency environments and disposable caches are omitted.

`artifact_manifest.json` records the SHA256 and byte size of every bundled file and both release archives. The original acquisition URLs and download timestamps are in [source_manifest.json](research-project/data/source_manifest.json).

Pickle and joblib files can execute code when loaded; use these files only if you trust this repository and verify their checksums. Third-party documents and data retain their original rights and terms.
