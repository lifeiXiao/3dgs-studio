import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATASETS = DATA / 'datasets'
RUNS = DATA / 'runs'
for p in (DATASETS, RUNS): p.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path = Path(path)
    payload=json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        temp=Path(f.name)
        f.write(payload)
    try: os.replace(temp,path)
    finally: temp.unlink(missing_ok=True)



def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default


def project_path(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT) or '.venv' in p.relative_to(ROOT).parts or '.cache' in p.relative_to(ROOT).parts:
        raise ValueError('请选择当前项目内的数据目录')
    return p
