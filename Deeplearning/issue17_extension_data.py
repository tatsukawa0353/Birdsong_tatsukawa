"""Read immutable baseline runs plus the direct-weight extension, without relearning."""
import csv
import gzip
import json
from pathlib import Path

from .issue15_analysis import write_json
from .issue16 import digest
from .issue17 import DEST,weight_label
from .issue17_artifacts import read_record
from .issue17_selection import selection_curve

METHODS=('TD3','SAC','DE','random_search')
CONDITIONS=('constant_1d','constant_2d','constant_3d')


def specs():
    return [(c,m,s,w) for c in CONDITIONS for m in METHODS for w in (1.,.5) for s in (0,1,2)]


def run_folder(stage,c,m,s,w):
    relative=Path('runs')/c/m/weight_label(w)/f'seed_{s}'
    return stage/relative if (stage/relative/'summary.json').exists() else DEST/relative


def load_runs(stage):
    result={}
    for c,m,s,w in specs():
        folder=run_folder(stage,c,m,s,w)
        summary=json.loads((folder/'summary.json').read_text())
        curve=selection_curve(read_record(folder/'curve.json'),w)
        result[(c,m,s,w)]=dict(summary=summary,curve=curve,folder=folder,
            final=curve[-1],objective_selected=curve[curve[-1]['objective_selected_index']],
            common_posthoc=curve[curve[-1]['common_posthoc_index']])
    return result


def clarify_random_metadata(stage):
    """A reused pool has source returns, not newly collected rewards."""
    changes=[]
    for c in CONDITIONS:
        for seed in (0,1,2):
            folder=stage/'runs'/c/'random_search/lambda0p5'/f'seed_{seed}'
            records=read_record(folder/'curve.json')
            if 'negative_return' not in records[0]:continue
            for row in records:row['source_lambda1_negative_return']=row.pop('negative_return')
            original=digest(folder/'curve.json.gz')
            payload=json.dumps(records,ensure_ascii=False,allow_nan=False).encode()
            (folder/'curve.json.gz').write_bytes(gzip.compress(payload,mtime=0))
            summary=json.loads((folder/'summary.json').read_text())
            for key in ('initial','final','objective_selected','common_posthoc_best'):
                row=summary[key]
                if 'negative_return' in row:row['source_lambda1_negative_return']=row.pop('negative_return')
            summary['curve_sha256']=digest(folder/'curve.json.gz')
            summary['rescoring_semantics']='weighted_loss is reconstructed J0.5; source_lambda1_negative_return is historical, no new training reward collected'
            write_json(folder/'summary.json',summary)
            changes.append(dict(path=str(folder.relative_to(stage)),original_curve_sha256=original,
                current_curve_sha256=summary['curve_sha256'],change='rename inherited lambda1 return; all errors/actions unchanged'))
    if changes:write_json(stage/'random_record_semantics.json',changes)


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
