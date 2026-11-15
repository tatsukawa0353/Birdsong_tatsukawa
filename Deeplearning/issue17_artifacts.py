"""Lossless storage of complete candidate/learner records, without thinning."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .issue15_analysis import write_json


def read_record(path):
    path=Path(path)
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt',encoding='utf-8') as stream:return json.load(stream)


def refresh_checksums(output):
    output=Path(output)
    entries={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'}
    write_json(output/'checksums.json',entries)
    return entries


def compress_records(output):
    output=Path(output);started=time.perf_counter();entries=[]
    registry=output/'compression.json'
    if registry.exists():return json.loads(registry.read_text())
    for path in sorted((output/'runs').glob('*/*/*/seed_*/*.json')):
        if path.name not in ('curve.json','training_trace.json'):continue
        original=path.read_bytes();packed=gzip.compress(original,compresslevel=6,mtime=0)
        if gzip.decompress(packed)!=original:raise RuntimeError('Lossless verification failed')
        destination=path.with_suffix(path.suffix+'.gz')
        if destination.exists():raise FileExistsError(destination)
        destination.write_bytes(packed)
        if gzip.decompress(destination.read_bytes())!=original:
            raise RuntimeError('Compressed file failed round-trip; original retained')
        entries.append(dict(original_path=str(path.relative_to(output)),compressed_path=str(destination.relative_to(output)),
            original_sha256=hashlib.sha256(original).hexdigest(),compressed_sha256=hashlib.sha256(packed).hexdigest(),
            original_bytes=len(original),compressed_bytes=len(packed)))
        path.unlink()
    data=dict(lossless=True,no_thinning=True,entries=entries,
        original_bytes=sum(e['original_bytes'] for e in entries),compressed_bytes=sum(e['compressed_bytes'] for e in entries),
        wall_s=time.perf_counter()-started)
    write_json(registry,data)
    return data
