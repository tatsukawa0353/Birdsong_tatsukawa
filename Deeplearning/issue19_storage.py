"""Explicit retention review, lossless temporary-log consolidation and size audit."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import subprocess

from .issue19_data import ROOT, OUT, BASE
from .issue19_protection import workspace_inventory
from .issue15_analysis import write_json
from .issue16 import digest
from .issue17_extension_data import write_csv


def review():
    with (OUT/'protected_before.csv').open() as stream:rows=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for row in rows:
        if row['path'].endswith('.png'):groups[row['sha256']].append(row)
    duplicates=[]
    for group in groups.values():
        if len(group)<2:continue
        for r in group:
            duplicates.append(dict(**r, duplicate_group_size=len(group),
                decision='retain',reason='Referenced historical figures/checksums; preserve live links and historical audits',
                same_content_paths=json.dumps([x['path'] for x in group if x!=r])))
    write_csv(OUT/'duplicate_png_review.csv',duplicates)
    plan=[]
    for prefix in ('results/phase1_constant_020s','results/phase1_3d_domain_diagnostics'):
        for r in rows:
            if r['path'].startswith(prefix) and r['path'].endswith('.png'):
                plan.append(dict(**r,decision='retain',
                    regeneration='python -m Deeplearning.issue17_visualization render (historical staged workflow; fresh output required)' if 'constant_020s' in prefix else 'python -m Deeplearning.issue18_report (historical source/build required)',
                    reason='Raw arrays and renderer exist, but report/figure registry/audit retain exact paths and hashes'))
    for prefix in ('.issue16-work','.issue17-work','results/phase1_3d_domain_diagnostics/interrupted_attempt_1'):
        for p in sorted((ROOT/prefix).rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            plan.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p),decision='retain',
                regeneration='',reason='Existing trained weights, history, or past working script; no demonstrated redundancy'))
    # Progress-only old temporary logs are preserved byte-for-byte in one archive.
    logs=[ROOT/'.issue17-work'/n for n in ('direct_weights_run.log','direct_weights_audit.log','direct_weights_render.log')]
    archive=OUT/'consolidated_temporary_logs.json.gz';changes=[]
    if not archive.exists():
        payload={str(p.relative_to(ROOT)):p.read_text() for p in logs if p.is_file() and not p.is_symlink()}
        archive.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False).encode(),mtime=0))
        decoded=json.loads(gzip.decompress(archive.read_bytes()))
        # No current tracked source or research document references these paths.
        tracked=subprocess.check_output(['git','ls-files','--','*.py','*.md'],cwd=ROOT,text=True).splitlines()
        documents=[(ROOT/n).read_text(errors='replace') for n in tracked if (ROOT/n).is_file()]
        for p in logs:
            rel=str(p.relative_to(ROOT))
            if rel not in payload:continue
            assert decoded[rel].encode()==p.read_bytes()
            assert not any(rel in text for text in documents), f'Live reference to {rel}'
            changes.append(dict(path=rel,sha256=digest(p),bytes=p.stat().st_size,
                action='losslessly_archived_then_removed',replacement=str(archive.relative_to(ROOT)),
                reason='Ignored temporary progress/audit stdout; research histories and audit.json retained'))
            p.unlink()
        write_json(OUT/'cleanup_changes.json',changes)
    else:
        changes=json.loads((OUT/'cleanup_changes.json').read_text())
    for r in plan:
        if any(c['path']==r['path'] for c in changes):r['decision']='losslessly_archived_then_removed'
    for c in changes:
        if not any(r['path']==c['path'] for r in plan):
            plan.append(dict(path=c['path'],bytes=c['bytes'],sha256=c['sha256'],decision=c['action'],
                regeneration='gzip JSON archive: '+c['replacement'],reason=c['reason']))
    write_csv(OUT/'artifact_retention_review.csv',plan)
    summary=dict(duplicate_png_groups=sum(len(g)>1 for g in groups.values()),
        duplicate_png_extra_files=sum(max(0,len(g)-1) for g in groups.values()),
        duplicate_png_extra_bytes=sum(sum(int(r['bytes']) for r in g[1:]) for g in groups.values()),
        original_research_files_removed=0, temporary_logs_consolidated=len(changes),
        temporary_log_bytes=sum(r['bytes'] for r in changes),archive_bytes=archive.stat().st_size,
        issue17_pngs=sum(r['path'].startswith('results/phase1_constant_020s/') and r['path'].endswith('.png') for r in rows),
        issue18_pngs=sum(r['path'].startswith('results/phase1_3d_domain_diagnostics/') and r['path'].endswith('.png') for r in rows),
        policy='Keep referenced historical PNG and all scientific files; future outputs use eight summary images, no per-seed plots or checkpoint dumps. No symlinks or history rewriting.')
    write_json(OUT/'cleanup_summary.json',summary)
    return summary


def after():
    (OUT/'storage_after.json').touch(exist_ok=True)
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    # git ls-files is relative to cwd; status is relative to the repository root.
    new=[ROOT/p for p in untracked if (ROOT/p).is_file()]
    result=dict(workspace=workspace_inventory(),new_git_candidate_files=len(new),
        new_git_candidate_bytes=sum(p.stat().st_size for p in new),
        modified_tracked_files=sum(r[:2].strip() not in ('??','D') for r in status),
        deleted_tracked_files=sum('D' in r[:2] for r in status),
        status=status,measurement='Untracked nonignored files are candidates, not staged or committed; logical bytes; .git excluded')
    # Include this measurement file itself; integer digit lengths converge.
    for _ in range(5):
        report=OUT/'REPORT.md'
        if report.exists():
            before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total']
            current=result['workspace']['total']
            block=f'''<!-- ISSUE19_STORAGE_BEGIN -->
## ファイル数と容量の実測

| 対象 | 変更前 | 変更後 |
| --- | --- | --- |
| 作業ツリーの通常ファイル数（ignored含む） | {before['files']} | {current['files']} |
| 作業ツリーの論理容量 | {before['bytes']/1e6:.2f} MB | {current['bytes']/1e6:.2f} MB |
| Git新規追加候補 | 3個（初期スクリプトのみ） | {result['new_git_candidate_files']}個 / {result['new_git_candidate_bytes']/1e6:.2f} MB |
| 既存追跡ファイルの変更 / 削除 | 0 / 0 | {result['modified_tracked_files']} / {result['deleted_tracked_files']} |

変更前は初期3スクリプト作成後・実験前の棚卸し時点。着手時のGitツリー自体はクリーンで未pushコミットなし。通常ファイルの数・容量は.gitとsymlinkを除外し、仮想環境やignoredログを含む。新規追加候補は未追跡・非ignoreファイルで、commit予定を確定したものではない。旧研究ファイル削除0、ignoredログ3本を1つの圧縮記録へ統合した。
<!-- ISSUE19_STORAGE_END -->'''
            body=report.read_text()
            body=re.sub(r'<!-- ISSUE19_STORAGE_BEGIN -->.*?<!-- ISSUE19_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE19_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n'
            report.write_text(body)
        write_json(OUT/'storage_after.json',result)
        measured=workspace_inventory()
        new_bytes=sum(p.stat().st_size for p in new)
        if measured['total']==result['workspace']['total'] and new_bytes==result['new_git_candidate_bytes']:break
        result['workspace']=measured
        result['new_git_candidate_bytes']=new_bytes
    return result


def freeze_checksums():
    """Freeze the rendered report and artifacts after their integrity checks."""
    path=OUT/'artifact_checksums.json'
    path.touch(exist_ok=True)
    from .issue19_audit import finalize
    finalize()
    after()
    result=dict(excludes=['artifact_checksums.json','storage_after.json'],
        implementation_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.glob('issue19*.py'))},
        artifacts={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*'))
                   if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    write_json(path,result)
    after()
    result['artifacts']['REPORT.md']=digest(OUT/'REPORT.md')
    write_json(path,result)
    return result


if __name__=='__main__':
    review();print(json.dumps(after(),ensure_ascii=False))
