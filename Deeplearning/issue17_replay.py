"""Replay frozen numerical conditions into a fresh nested output, without replacement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
import multiprocessing
import re
import shutil

from .issue15_analysis import write_json
from .issue16 import digest,protected_snapshot,verify_protected
from .issue17 import DEST,load_ready,run_one,run_specs
from .issue17_artifacts import compress_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--condition',choices=('constant_1d','constant_2d','constant_3d'),default='constant_1d')
    parser.add_argument('--method',choices=('TD3','SAC','DE','random_search'),default='TD3')
    parser.add_argument('--weight',type=float,choices=(1.,.5),default=1.)
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--all',action='store_true',help='Replay all 54 runs and independently audit them')
    parser.add_argument('--workers',type=int,default=27)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):parser.error('Use a fresh alphanumeric/underscore/hyphen name')
    if args.method in ('DE','random_search') and args.weight!=1:parser.error('DE/Random use lambda=1 only')
    manifest,_,_=load_ready(DEST)
    parent=DEST/'reproductions'
    if parent.resolve()!=parent.absolute():parser.error('Reproduction parent must not traverse symlinks')
    output=parent/args.name
    if output.exists():parser.error(f'Output already exists: {output}')
    protected=protected_snapshot()
    output.mkdir(parents=True)
    (output/'diagnostics').mkdir()
    for name in ('target.npz','review.json'):shutil.copy2(DEST/'diagnostics'/name,output/'diagnostics'/name)
    manifest=dict(manifest,replay_of_manifest_sha256=digest(DEST/'manifest.json'),
                  replay_created_utc=datetime.now(timezone.utc).isoformat(),canonical_replacement=False)
    write_json(output/'manifest.json',manifest)
    write_json(output/'config.json',manifest['settings'])
    write_json(output/'protected_before.json',protected)
    if args.all:
        specs=run_specs(manifest['settings'])
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_one,output,c,m,s,w) for c,m,s,w in specs]
            summaries=[future.result() for future in futures]
        write_json(output/'summary.json',summaries)
        compress_records(output)
        from .issue17_audit import audit
        audit(output)
    else:
        run_one(output,args.condition,args.method,args.seed,args.weight)
        compress_records(output)
    verify_protected(output)
    print(output)


if __name__=='__main__':main()
