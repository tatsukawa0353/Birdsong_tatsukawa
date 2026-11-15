"""Stage, audit, review and publish the direct-weight/combined-figure revision."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from .issue15_analysis import write_json
from .issue16 import digest
from .issue17 import DEST,ROOT
from .issue17_artifacts import refresh_checksums
from .issue17_cleanup import safe_unlink
from .issue17_direct_weights import WORK,EXTENSION,prepare,run
from .issue17_extension_audit import audit,verify_preserved
from .issue17_extension_data import clarify_random_metadata
from .issue17_extension_report import report

DERIVED=('README.md','REPORT.md','comparison_per_seed.csv','comparison_aggregate.csv','comparison_curves.csv',
         'checkpoint_comparison.csv','parameter_errors.csv','lambda_effects.csv','figure_sources.json',
         'figure_audit.json','visual_review.json','report_timing.json')


def verify_existing_csv_rows():
    identities={'comparison_per_seed.csv':('condition','method','envelope_weight','seed'),
        'comparison_aggregate.csv':('condition','method','envelope_weight'),
        'comparison_curves.csv':('condition','method','envelope_weight','seed','episodes'),
        'checkpoint_comparison.csv':('condition','method','envelope_weight','seed','episodes'),
        'lambda_effects.csv':('condition','method')}
    counts={}
    for name,fields in identities.items():
        with (DEST/name).open() as stream:old=list(csv.DictReader(stream))
        with (WORK/name).open() as stream:new={tuple(r[k] for k in fields):r for r in csv.DictReader(stream)}
        for row in old:
            key=tuple(row[k] for k in fields)
            if key not in new:raise RuntimeError(f'Existing CSV row missing:{name}:{key}')
            for field,value in row.items():
                if value!=new[key].get(field):
                    try:equal=float(value)==float(new[key].get(field))
                    except (ValueError,TypeError):equal=False
                    if not equal:raise RuntimeError(f'Existing CSV value changed:{name}:{key}:{field}: {value} vs {new[key].get(field)}')
        counts[name]=len(old)
    write_json(WORK/'csv_preservation.json',dict(passed=True,existing_rows_verified=counts))
    return counts


def publish():
    if EXTENSION.exists():raise FileExistsError('Published extension already exists')
    for name in ('audit.json','figure_audit.json','visual_review.json'):
        if not json.loads((WORK/name).read_text())['passed']:raise RuntimeError(f'Gate not passed:{name}')
    verify_preserved(WORK);verify_existing_csv_rows()
    old=json.loads((WORK/'old_single_curve_images.json').read_text())
    for row in old:
        path=DEST/row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(DEST/'figures') or digest(path)!=row['sha256']:
            raise RuntimeError(f'Old image changed; preserving it:{path}')
    for source in (WORK/'figures').rglob('*.png'):
        target=DEST/source.relative_to(WORK)
        if target.exists():raise FileExistsError(f'Unexpected image collision:{target}')
    for c in ('constant_1d','constant_2d','constant_3d'):
        for m in ('DE','random_search'):
            source=WORK/'runs'/c/m/'lambda0p5';target=DEST/'runs'/c/m/'lambda0p5'
            if target.exists():raise FileExistsError(target)
            source.rename(target)
    for source in (WORK/'figures').rglob('*.png'):
        target=DEST/source.relative_to(WORK);target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    # Only the explicitly listed, old single-method error plots are unlinked.
    for row in old:safe_unlink(DEST/row['path'],DEST,row['sha256'])
    for name in DERIVED:
        source=WORK/name
        if source.exists():source.replace(DEST/name)
    expected={r['path'] for r in json.loads((DEST/'figure_sources.json').read_text())['images'] if 'field' in r}
    actual={str(p.relative_to(DEST)) for sub in ('learning_curves','best_observed') for p in (DEST/'figures'/sub).iterdir() if p.is_file()}
    if expected!=actual or len(actual)!=24:raise RuntimeError('Comparison directories contain unexpected files')
    (WORK/'deleted_single_curve_images.txt').write_text('\n'.join(r['path'] for r in old)+'\n')
    write_json(WORK/'publication.json',dict(passed=True,published_utc=datetime.now(timezone.utc).isoformat(),
        new_DE_runs=9,derived_Random_runs=9,profiles_total=72,RL_retrained=False,
        deleted_single_method_images=len(old),deleted_image_bytes=sum(r['bytes'] for r in old),
        deleted_image_files=[r['path'] for r in old],moved_old_images_to_archive=False,
        comparison_images=24,protected_files_verified=verify_preserved(WORK),
        no_existing_spectrograms_controls_models_audio_or_raw_data_deleted=True))
    # Historical/extension metadata lives in one canonical subtree; no old PNG
    # copy is created. Empty staging run/figure directories are omitted.
    EXTENSION.mkdir()
    for path in list(WORK.iterdir()):
        if path.is_file():path.rename(EXTENSION/path.name)
    update_root_index()
    refresh_checksums(DEST)
    print('Published all-four-method comparisons; deleted72 single-method plots',flush=True)


def update_root_index():
    p=ROOT/'README.md';text=p.read_text()
    marker='Issue #17の'
    position=text.find(marker)
    if position>=0:
        end=text.find('\n\n',position)
        text=text[:position]+('Issue #17は [4手法×2λの正本索引](results/phase1_constant_020s/README.md) を最初に確認する。\n'
            '0.20秒・一定1D/2D/3D・4096候補。TD3/SACを再学習せずDE λ=.5を追加し、Random λ=.5は全保存候補を再評価した。\n'
            '誤差図は同一次元・同一λの4手法をまとめた24枚へ統一。旧単独手法図は削除済み。')+text[end:]
        p.write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','audit','render','publish'))
    parser.add_argument('--workers',type=int,default=9);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.workers)
    elif args.action=='audit':clarify_random_metadata(WORK);audit(WORK)
    elif args.action=='render':report(WORK)
    else:publish()
