"""Download one public split and restore the media paths in its metadata."""
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .data import load_records
from .io import sha256, write_json

DATA_REVISION = '2d0c348eee48d44e14c0c0872b1740f5e8e7b619'


def extract_archive(archive):
    archive = Path(archive)
    root = archive.parent.resolve()
    records = []
    with tarfile.open(archive, 'r:') as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 2 or parts[0] not in {'audio', 'image', 'video'}
                    or '..' in parts or PurePosixPath(member.name).is_absolute()):
                raise ValueError(f'Unsafe or unexpected archive entry: {member.name}')
            destination = root/member.name
            if not destination.resolve().is_relative_to(root) or destination.is_symlink():
                raise ValueError(f'Unsafe extraction destination: {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f'Unreadable archive entry: {member.name}')
            digest = hashlib.sha256()
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != member.size:
                    raise ValueError(f'Existing media differs from archive: {destination}')
                for block in iter(lambda: source.read(8 << 20), b''):
                    digest.update(block)
                if sha256(destination) != digest.hexdigest():
                    raise ValueError(f'Existing media hash differs: {destination}')
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.'+destination.name+'.', delete=False) as out:
                        temporary = Path(out.name)
                        for block in iter(lambda: source.read(8 << 20), b''):
                            digest.update(block); out.write(block)
                    if temporary.stat().st_size != member.size:
                        raise ValueError('Incomplete archive member')
                    temporary.replace(destination)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
            records.append({'path': member.name, 'bytes': member.size})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', required=True, choices=('train', 'test'))
    p.add_argument('--data-root', default='data/UEmo-Data-v1')
    p.add_argument('--revision', default=DATA_REVISION)
    p.add_argument('--local-only', action='store_true', help='Use already downloaded TAR files')
    p.add_argument('--check-only', action='store_true', help='Only validate JSON and extracted media paths')
    args = p.parse_args()
    root = Path(args.data_root).resolve()
    if not args.local_only and not args.check_only:
        from huggingface_hub import snapshot_download
        snapshot_download('wudq/UEmo-Data-v1', repo_type='dataset', revision=args.revision,
                          local_dir=root, allow_patterns=[f'{args.split}/*'], max_workers=4)
    if not args.check_only:
        archives = sorted((root/args.split).glob('*/*.tar'))
        if not archives:
            raise FileNotFoundError('No TAR shards found for the selected split')
        for archive in archives:
            relative = str(archive.relative_to(root))
            receipt = root/'.extracted'/(relative+'.json')
            digest = sha256(archive)
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if (previous['archive_sha256'] == digest and all(
                        (archive.parent/f['path']).is_file()
                        and not (archive.parent/f['path']).is_symlink()
                        and (archive.parent/f['path']).stat().st_size == f['bytes']
                        for f in previous['files'])):
                    print('READY', relative, flush=True)
                    continue
            files = extract_archive(archive)
            write_json(receipt, {'archive_sha256': digest, 'files': files})
            print('EXTRACTED', relative, len(files), flush=True)
    metadata = root/args.split/f'{args.split}_metadata.json'
    rows = load_records(metadata, check_media=True)
    print(json.dumps({'split': args.split, 'samples': len(rows), 'metadata': str(metadata),
                      'all_referenced_media_present': True}, indent=2))


if __name__ == '__main__':
    main()
