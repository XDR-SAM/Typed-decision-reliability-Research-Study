# Complete study artifacts

Large files are distributed in the [pilot-2026-10-02 GitHub release](https://github.com/XDR-SAM/Typed-decision-reliability-Research-Study/releases/tag/pilot-2026-10-02), because processed data exceed GitHub's ordinary Git file-size limit.

Download both ZIP files and extract them into the repository root, preserving their directory structure:

- `official-cfpb-sources.zip`: all original official CFPB downloads, including the earlier literature-review archive copies.
- `pilot-data-models-predictions.zip`: processed datasets, fitted CPU pilot models, environment provenance, and prediction files.

All other study sources, reports, audit tables, metrics, figures, and code are tracked in Git. Only installed dependency environments and disposable caches are omitted.

## Frozen neural preparation — 3 October 2026

The [neural-protocol-2026-10-03 release](https://github.com/XDR-SAM/Typed-decision-reliability-Research-Study/releases/tag/neural-protocol-2026-10-03) adds `neural-preparation-data.zip`: the six frozen Debt Collection/Credit Card JSONL exports (complete rows, training rows and validation rows). Extract it into the repository root, preserving paths. Download the earlier pilot assets as well when reproducing the full study.

`neural_artifact_manifest.json` records archive and member SHA256 checksums and sizes. The protocol's independent file/data hashes remain in `research-project/research/protocol_freeze.json`. No pretrained neural weights, neural training checkpoints or future neural predictions have been produced.

The pinned official Laya source archive is tracked at `research-project/research/sources/neural/laya_source.zip`; its extracted duplicate directory is omitted. Local virtual environments and caches are not uploaded.

`artifact_manifest.json` records the SHA256 and byte size of every bundled file and both release archives. The original acquisition URLs and download timestamps are in [source_manifest.json](research-project/data/source_manifest.json).

Pickle and joblib files can execute code when loaded; use these files only if you trust this repository and verify their checksums. Third-party documents and data retain their original rights and terms.
