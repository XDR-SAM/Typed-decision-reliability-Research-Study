"""Refuse silent dependency/source drift at the start of the GPU phase."""
import sys,json
from importlib import metadata
from src.utils import ROOT

def check_environment():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Frozen GPU environment requires Python 3.11; log an amendment before changing it')
    versions={}
    for line in (ROOT/'requirements-neural.txt').read_text().splitlines():
        if '==' not in line or line.lstrip().startswith('#'):continue
        package,pin=line.strip().split('==',1)
        actual=metadata.version(package);versions[package]=actual
        if actual.split('+')[0]!=pin:raise RuntimeError(f'Pinned dependency mismatch: {package} {actual} != {pin}')
    installed=metadata.distribution('laya')
    direct=json.loads(installed.read_text('direct_url.json') or '{}')
    commit=direct.get('vcs_info',{}).get('commit_id')
    expected='fa9a2a7070b1789912a49ae24603bbfb1a78b001'
    if commit!=expected:raise RuntimeError('Laya must be installed from the exact frozen Git SHA')
    return versions
