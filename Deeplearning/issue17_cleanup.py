"""Reviewed, hash-checked Phase 1 retirement; never traverse symbolic links.

All new runs/audits/figures must pass before publish() replaces the pilot or
unlinks old generated files. Original Git history and historical audit records
retain their original meaning; no Phase 0 output is edited.
"""
from .result_paths import resolve as resolve_result_path
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

from .issue15_analysis import write_json
from .issue16 import digest,verify_protected
from .issue17 import ROOT,DEST,OLD_PHASE1,WORK

GITHUB='https://github.com/tatsukawa0353/Birdsong_tatsukawa'


def checked_file(path,base):
    """Validate lexical/real boundaries and every ancestor without following links."""
    path,base=Path(path).absolute(),Path(base).absolute()
    if (not path.is_relative_to(base) or path==base or base.is_symlink() or
            base.resolve()!=base or path.resolve()!=path):
        raise RuntimeError(f'Unsafe cleanup boundary: {path}')
    for ancestor in [path,*path.parents]:
        if ancestor==base.parent:break
        if ancestor.is_symlink():raise RuntimeError(f'Symbolic link is outside cleanup scope: {ancestor}')
    if not stat.S_ISREG(path.lstat().st_mode):raise RuntimeError(f'Not a regular file: {path}')
    return path


def safe_unlink(path,base,expected_sha):
    path=checked_file(path,base)
    if digest(path)!=expected_sha:raise RuntimeError(f'Changed after inventory, preserving: {path}')
    parent_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.unlink(path.name,dir_fd=parent_fd)
    finally:os.close(parent_fd)


def empty_directories(base):
    for p in sorted(Path(base).rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_symlink():raise RuntimeError(f'Unexpected symlink; preserving: {p}')
        if p.is_dir():p.rmdir()
    Path(base).rmdir()


def review_references(output):
    files=subprocess.check_output(['rg','--files','-g','*.md','-g','*.py','-g','*.json',
        '-g','!results/**','-g','!reports/server_smoke_cleanup_20261113/**'],cwd=ROOT,text=True).splitlines()
    references=[]
    for name in files:
        text=(resolve_result_path(ROOT/name)).read_text(errors='replace')
        pattern=r'(?:Deeplearning/)?results/(phase1(?:_extended|_terminal_baselines|_validation_20261031|_issue13_20261107|_constant_020s)?)/([^\s\]\)\}\"\x27`<>;,]+)'
        for match in re.finditer(pattern,text):
            relative='results/'+match[1]+'/'+match[2].rstrip('.')
            target=resolve_result_path(ROOT/relative)
            references.append(dict(source=name,line=text.count('\n',0,match.start())+1,
                path=relative,exists=target.exists(),is_file=target.is_file()))
    write_json(Path(output)/'reference_review.json',dict(references=references))


def plan_cleanup(output,*,refresh=False):
    output=Path(output)
    if (output/'cleanup_plan.json').exists() and not refresh:return json.loads((output/'cleanup_plan.json').read_text())
    if not (output/'reference_review.json').exists():review_references(output)
    inventory=json.loads((output/'phase1_inventory_before.json').read_text())
    review=json.loads((output/'reference_review.json').read_text())
    git_head=json.loads((output/'manifest.json').read_text())['git_head']
    references={r['path'] for r in review['references'] if r['is_file']}
    keep={};rows=[]
    metadata={'manifest.json','config.json','summary.json','execution.json','execution_summary.json'}
    for name,item in inventory.items():
        path=resolve_result_path(ROOT/name);relative=Path(name);experiment=relative.parts[1]
        within=relative.relative_to(Path('results')/experiment)
        is_pilot=experiment=='phase1_constant_020s'
        if is_pilot:
            reason='pilot conditions/aggregate retained; detailed pilot outputs superseded'
            retain=within.as_posix() in ('config.json','manifest.json','comparison_aggregate.csv','comparison_per_seed.csv')
            destination=Path('history/pilot_256')/within
        else:
            is_top=len(within.parts)==1
            retain=((is_top and (path.name in metadata or path.suffix in ('.md','.csv'))) or
                    (name in references and path.suffix not in ('.png','.svg','.wav')) or
                    within.as_posix()=='diagnostics/target.npz' and experiment=='phase1_validation_20261031' or
                    experiment=='phase1_validation_20261031' and within.match('runs/A/*/seed_*/curve.json'))
            reason=('retained historical metadata, cited numeric evidence, or live legacy-code dependency' if retain else
                    'retired generated plots/audio/models/detailed logs; cited images remain at immutable Git commit')
            destination=Path('history')/experiment/within
        row=dict(path=name,absolute_path=str(path.absolute()),sha256=item['sha256'],bytes=item['bytes'],
                 action='retain_in_history' if retain else 'delete_generated',reason=reason,
                 destination=str(destination) if retain else None)
        rows.append(row)
        if retain:keep[name]=destination
    history=output/'history';history.mkdir(exist_ok=True)
    for name,destination in keep.items():
        source=checked_file(resolve_result_path(ROOT/name),ROOT/'results'/Path(name).parts[1])
        if digest(source)!=inventory[name]['sha256']:raise RuntimeError(f'Old data changed: {source}')
        target=output/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

    def old_link(path):
        return GITHUB+'/blob/'+git_head+'/Deeplearning/'+path

    def rewrite_markdown(text,old_source,new_source):
        def substitute(match):
            bang,label,url=match.groups()
            if '://' in url or url.startswith(('#','mailto:')):return match.group(0)
            part,sep,anchor=url.partition('#')
            target=(resolve_result_path(ROOT/old_source)).parent/part
            target=Path(os.path.normpath(target))
            if not target.is_relative_to(ROOT):return match.group(0)
            relative=str(target.relative_to(ROOT))
            if not old_source.startswith('results/') and relative=='results/Phase1/CaseA/parameter_dimension/REPORT.md':
                link=os.path.relpath(DEST/'REPORT.md',(resolve_result_path(ROOT/new_source)).parent)
            elif relative in inventory:
                if relative in keep:
                    final=DEST/keep[relative]
                    link=os.path.relpath(final,(resolve_result_path(ROOT/new_source)).parent)
                else:
                    link=old_link(relative)
                    if bang:
                        link=f'https://raw.githubusercontent.com/tatsukawa0353/Birdsong_tatsukawa/{git_head}/Deeplearning/{relative}'
            elif target.is_dir() and any(name.startswith(relative+'/') for name in keep):
                experiment=Path(relative).parts[1]
                final=DEST/'history'/experiment/Path(relative).relative_to(Path('results')/experiment)
                link=os.path.relpath(final,(resolve_result_path(ROOT/new_source)).parent)
            elif target.is_dir() and any(relative.startswith('results/'+name) for name in OLD_PHASE1):
                link=GITHUB+'/tree/'+git_head+'/Deeplearning/'+relative
            elif target.exists():
                link=os.path.relpath(target,(resolve_result_path(ROOT/new_source)).parent)
            else:return match.group(0)
            return f'{bang}[{label}]({link}{sep}{anchor})'
        return re.sub(r'(!?)\[([^\]]*)\]\(([^)]+)\)',substitute,text)

    # Historical report numbers/logs are not rewritten. Only their copied local
    # links are migrated, with a visible archival notice and original checksum.
    for name,destination in keep.items():
        if destination.suffix=='.md':
            original=(resolve_result_path(ROOT/name)).read_text()
            new_source=str((DEST/destination).relative_to(ROOT))
            text=rewrite_markdown(original,name,new_source)
            notice=('> 歴史資料（4096回の正本ではない）。元の実行条件・数値を保持。'
                    '整理した画像等への参照は実行時Git commitへ固定した。新しい検証は正本のaudit.jsonを参照。\n\n')
            (output/destination).write_text(notice+text)
    updates=[]
    source_files=set(r['source'] for r in review['references'])
    source_files.update(('phase1_validation.py','issue13.py','configs/issue13_reproduction.json',
                         'phase1_validation_audit.py','run.py'))
    for name in sorted(source_files):
        if name.startswith('issue17') or name.startswith('results/'):continue
        path=resolve_result_path(ROOT/name)
        old=path.read_text();new=old
        if path.suffix=='.md':new=rewrite_markdown(new,name,name)
        else:
            # Legacy report generators embed image links in Python strings.
            # Keep those references usable without retaining obsolete plots.
            for old_name in references:
                if Path(old_name).suffix in ('.png','.svg') and old_name not in keep:
                    new=new.replace(old_name,f'https://raw.githubusercontent.com/tatsukawa0353/Birdsong_tatsukawa/{git_head}/Deeplearning/{old_name}')
        # Do not rewrite immutable Git URLs created above into nonexistent paths
        # inside the historical commit. Migrate only local path text/code.
        pieces=re.split(r'(https?://[^\s)<>`]+)',new)
        for index in range(0,len(pieces),2):
            for experiment in OLD_PHASE1:
                if experiment=='phase1_constant_020s':continue
                pieces[index]=pieces[index].replace('results/'+experiment+'/', 'results/Phase1/CaseA/parameter_dimension/history/'+experiment+'/')
                pieces[index]=pieces[index].replace('results/'+experiment+'"','results/Phase1/CaseA/parameter_dimension/history/'+experiment+'"')
        new=''.join(pieces)
        if name in ('issue13_audit.py','phase1_validation_audit.py'):
            new=new.replace('    output = args.output.resolve()\n',
                '    output = args.output.resolve()\n    if (output / "ARCHIVED.md").exists():\n'
                '        parser.error("Archived partial historical results; use python -m Deeplearning.issue17 audit for the current canonical experiment")\n')
        if name=='run.py':
            new=new.replace('    config = json.loads(config_path.read_text())',
                '    if args.phase == "phase1":\n'
                '        canonical = root / "results/Phase1/CaseA/parameter_dimension"\n'
                '        if args.output is None:\n'
                '            parser.error("Use python -m Deeplearning.issue17 for current Phase 1; historical tasks require an explicit new --output below phase1_constant_020s/reproductions")\n'
                '        if not output.resolve().is_relative_to((canonical / "reproductions").resolve()) or output.exists():\n'
                '            parser.error("Historical Phase 1 requires a new output below phase1_constant_020s/reproductions")\n'
                '    config = json.loads(config_path.read_text())')
        if name=='README.md':
            new=new.replace('4手法各3 seed×256候補','TD3/SACはλ=1/.5、DE/Randomはλ=1、各3 seed×4096候補（計54 run）')
            new=new.replace('γ=1・全区間報酬を使う。','γ=1・全区間報酬を使う。最初は [正本の索引](results/Phase1/CaseA/parameter_dimension/README.md) を読む。')
            new=new.replace('Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 audit\n','')
            new+='\n旧Phase1は正本内の [history](results/Phase1/CaseA/parameter_dimension/history/README.md) に必要な歴史情報を集約した。旧監査のprotected_beforeは当時の記録で、現行の保存検証はIssue17の新監査に切り替える。Phase0の保存済み監査ログは改変しない。\n'
        if new!=old:
            updates.append(dict(path=name,original_sha256=digest(path),text=new))
    write_json(output/'documentation_updates.json',updates)
    note=['# 歴史資料（正本ではない）','',
        '現在の正本は [4096回レポート](../REPORT.md)。旧256回の詳細モデル・音声・図は置換済み。',
        'pilot_256/ は当時の設定と小さい数表のみ。旧0.35秒は条件の異なる歴史情報。',
        '旧監査ログの条件・チェックサムは当時の状態を示し、新4096回の検証を主張しない。',
        '新しい保存検査は ../audit.json と ../publication_audit.json を使用する。','',
        f'整理前のファイルはGit commit [{git_head}]({GITHUB}/tree/{git_head}/Deeplearning/results) で参照できる。',
        '公開済みcommitを固定しているため、新しい正本と混同しない。ローカルでもgit show <commit>:<path>で復元可能。','',
        '保存した各歴史資料の理由・元のhash・削除ファイルは [整理一覧](../cleanup_plan.json) を参照。']
    (history/'README.md').write_text('\n'.join(note)+'\n')
    for experiment in OLD_PHASE1:
        if experiment=='phase1_constant_020s':continue
        p=history/experiment
        if p.exists():
            (p/'ARCHIVED.md').write_text('歴史資料の部分保存。旧フル監査の再実行対象ではない。現行正本は ../../REPORT.md、現行監査は ../../audit.json。元の完全な成果はGitの整理前commitを参照。\n')
    for row in rows:
        if row['destination']:
            row['retained_sha256']=digest(output/row['destination'])
    plan=dict(scope=list(OLD_PHASE1),original_git_head=git_head,entries=rows,
        before_bytes=sum(r['bytes'] for r in rows),before_files=len(rows),
        retained_original_bytes=sum(r['bytes'] for r in rows if r['action']=='retain_in_history'),
        deleted_generated_bytes=sum(r['bytes'] for r in rows if r['action']=='delete_generated'),
        retained_files=sum(r['action']=='retain_in_history' for r in rows),
        deleted_files=sum(r['action']=='delete_generated' for r in rows),
        historical_images='No obsolete PNG/SVG/WAV copied; cited images use immutable Git URLs',
        protected_scope='All preexisting results outside the six explicitly authorized Phase1 directories')
    write_json(output/'cleanup_plan.json',plan)
    text=f'''# Phase 1 整理記録

新4096回の全54 run・独立監査・単一条件の図を検証してから置換/削除する。
対象はIssue17で指定された6ディレクトリのみ。Phase0・実音声・Python環境は対象外。

旧生成物 {plan['before_files']}ファイル / {plan['before_bytes']:,} bytesを調査。
必要な歴史資料 {plan['retained_files']}ファイル / 元サイズ{plan['retained_original_bytes']:,} bytesはhistoryへ集約。
不要生成物 {plan['deleted_files']}ファイル / {plan['deleted_generated_bytes']:,} bytesを削除予定。
旧画像・SVG・音声はコピーせず、外部資料の画像参照は整理前の公開Git commitへ固定する。
残す理由は引用された数値、旧コードのtarget/model依存、条件/集計の再現記録。
新4096回の保存容量は別に計上するため、最終全体容量が増える場合も削減量と区別する。

全パス・hash・サイズ・削除/保存理由は [cleanup_plan.json](cleanup_plan.json)。
全削除前に絶対パス、対象境界、ファイルhash、symlink不在を検証する。symlink先は辿らない。
旧監査ログは当時の証拠として保持/ Git参照し、新正本の監査へ切り替える。過去ログを新条件として書き換えない。
現在の公開状態と実際の削除件数・前後容量は [publication_audit.json](publication_audit.json) に記録する。
'''
    (output/'CLEANUP.md').write_text(text)
    return plan


def publish(output,destination):
    output,destination=Path(output),Path(destination)
    if output!=WORK or destination!=DEST:raise RuntimeError('Unexpected publication paths')
    for name in ('audit.json','figure_audit.json'):
        if not json.loads((output/name).read_text())['passed']:raise RuntimeError(f'Failed gate: {name}')
    plan=json.loads((output/'cleanup_plan.json').read_text())
    updates=json.loads((output/'documentation_updates.json').read_text())
    verify_protected(output)
    for update in updates:
        if digest(resolve_result_path(ROOT/update['path']))!=update['original_sha256']:
            raise RuntimeError(f'Documentation changed concurrently: {update["path"]}')
    inventory={r['path']:r for r in plan['entries']}
    for experiment in OLD_PHASE1:
        base=ROOT/'results'/experiment
        if base.is_symlink() or any(p.is_symlink() for p in base.rglob('*')):
            raise RuntimeError(f'Symlinks in cleanup directory; preserving it: {base}')
        expected={name for name in inventory if name.startswith('results/'+experiment+'/')}
        actual={str(p.relative_to(ROOT)) for p in base.rglob('*') if p.is_file()}
        if actual!=expected:raise RuntimeError(f'Old directory inventory changed: {base}')
    # Validate every deletion before unlinking anything, including retained files
    # whose exact bytes have already been copied into the audited new hierarchy.
    for name,row in inventory.items():
        path=checked_file(resolve_result_path(ROOT/name),ROOT/'results'/Path(name).parts[1])
        if digest(path)!=row['sha256']:raise RuntimeError(f'Changed after inventory: {path}')
        if row['destination'] and digest(output/row['destination'])!=row['retained_sha256']:
            raise RuntimeError(f'Historical copy failed verification: {row["destination"]}')
    retired=WORK.parent/'retired256'
    if retired.exists():raise FileExistsError(retired)
    destination.rename(retired)
    try:output.rename(destination)
    except BaseException:
        retired.rename(destination)
        raise
    for update in updates:(resolve_result_path(ROOT/update['path'])).write_text(update['text'])
    removed=0
    for name,row in inventory.items():
        relative=Path(name);experiment=relative.parts[1]
        base=retired if experiment=='phase1_constant_020s' else ROOT/'results'/experiment
        path=base/relative.relative_to(Path('results')/experiment)
        safe_unlink(path,base,row['sha256']);removed+=1
    for experiment in OLD_PHASE1:
        empty_directories(retired if experiment=='phase1_constant_020s' else ROOT/'results'/experiment)
    cleanup_note=destination/'CLEANUP.md'
    cleanup_note.write_text(cleanup_note.read_text().replace('を削除予定','を削除済み').replace(
        '検証してから置換/削除する','検証後に置換・削除を完了した'))
    verified=verify_protected(destination)
    write_json(destination/'reference_updates.json',[dict(path=u['path'],original_sha256=u['original_sha256'],
        current_sha256=digest(resolve_result_path(ROOT/u['path']))) for u in updates])
    (destination/'documentation_updates.json').unlink()
    (destination/'phase1_inventory_before.json').unlink()
    after_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    history_bytes=sum(p.stat().st_size for p in (destination/'history').rglob('*') if p.is_file())
    write_json(destination/'publication_audit.json',dict(passed=True,removed_original_files=removed,
        old_phase1_before_bytes=plan['before_bytes'],new_canonical_bytes_at_publication=after_bytes,
        retained_history_bytes=history_bytes,old_generated_bytes_removed=plan['deleted_generated_bytes'],
        total_phase1_net_change_bytes=after_bytes-plan['before_bytes'],protected_files_verified=verified,
        absolute_paths_validated=True,symbolic_links_followed=False,documentation_files_updated=[u['path'] for u in updates],
        old_audits_not_rewritten=True,git_commit_created=False,git_push_performed=False))
    # Full integrity ledger for future read-only verification; no old hashes are
    # fabricated to make the retired pilot appear equivalent to this revision.
    checksums={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file() and p.name!='checksums.json'}
    write_json(destination/'checksums.json',checksums)
