"""Package the complete large study artifacts for a GitHub release."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '.release-assets'
OUT.mkdir(exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

groups = {
    'official-cfpb-sources.zip': sorted(
        list((ROOT / 'research-project/data/raw').rglob('*'))
        + list((ROOT / 'tdm-literature-review/sources').glob('*.zip'))
    ),
    'pilot-data-models-predictions.zip': sorted(
        p for directory in ['data/processed', 'outputs/models', 'outputs/predictions']
        for p in (ROOT / 'research-project' / directory).rglob('*')
    ),
}
manifest = {'release_tag': 'pilot-2026-10-02', 'assets': []}
for name, paths in groups.items():
    target = OUT / name
    entries = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in paths:
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, relative, compress_type=zipfile.ZIP_STORED if path.suffix in {'.zip', '.gz'} else zipfile.ZIP_DEFLATED)
            entries.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest['assets'].append({'name': name, 'bytes': target.stat().st_size, 'sha256': sha(target), 'files': entries})
    print(name, target.stat().st_size, len(entries), flush=True)
(ROOT / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
