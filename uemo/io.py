"""Small atomic receipts and immutable-input fingerprints."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def model_fingerprint(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {'.safetensors', '.json', '.py', '.jinja', '.txt'})
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'No model/adapter weights found in {path}')
    return {p.name: sha256(p) for p in files}
