import ast,csv,hashlib,importlib.util,json,os,re,shutil
from pathlib import Path
ROOT=Path.cwd(); M=ROOT/'results/maintenance/20261126'
spec=importlib.util.spec_from_file_location('paths',ROOT/'result_paths.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
FOLDERS=mod.FOLDERS
before=json.load(open('/tmp/results-cleanup-before.json')); bypath={f['path']:f for f in before['files']}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def writecsv(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def mapped(p): return str(mod.resolve(p))
shutil.copyfile('/tmp/results-cleanup-before.json',M/'inventory_before.json')
shutil.copyfile('/tmp/results-cleanup-ready-before.json',M/'ready_before.json')
# Freeze sources independently of the artifact inventory before any edits.
src={str(p.relative_to(ROOT)):dict(before_sha256=digest(p),before_bytes=p.stat().st_size) for base in (ROOT,ROOT/'tests') for p in base.glob('*.py')}
dump(M/'source_before.json',src)
# Actual deletion plan: unique raw waves and checkpoints are NEVER discarded.
deleted={}
for f in before['files']:
 p=f['path'];reason=None
 if f['image']:
  if any('/'+n+'/' in p for n in ('phase1_constant_020s','phase1_3d_domain_diagnostics')) and re.search(r'/figures/(spectrograms|amplitude_envelopes)/(final|selected_best)/.*seed[12]\.png$',p):
   reason='再生成可能なseed1/2個別図。seed0代表図・集約学習/制御図・全seedのNPZ/履歴/描画コードを保持'
  elif '/issue10/' in p and re.search(r'/(SAC|TD3)_[12]/[^/]+\.png$',p): reason='旧報酬診断のseed別個別図。集約図・seed0・全学習データを保持'
  elif '/phase1_caseB_self_target_020s_methods_f0lr_scan/figures/' in p and Path(p).name in {'Target_spectrogram.png','Target_waveform.png','training_best_spectrograms.png','training_best_waveforms.png','posthoc_spectrograms.png','posthoc_waveforms.png','selected_internal_seed1.png','final_internal_seed1.png','group2_spectrograms.png','group3_spectrograms.png','three_groups_previous_color.png'}:
   reason='予備b3_c00実験の描画バリエーション。27条件共通色軸・物理比較・学習曲線・Final/Selected集約図と生データを保持'
 if '/server_smoke_' in p and (f['image'] or Path(p).suffix in {'.wav','.npz','.pt','.zip'}): reason='完了済み環境smokeの生成物。環境/設定/監査/ログを保持。研究Target/学習結果ではない'
 if reason: deleted[p]=dict(f,reason=reason,action='delete_regenerable',replacement='')
# Exact duplicate artifacts retain every historical path through a relative symlink.
# Documents/configuration/ledgers/hash records remain physical originals.
links={}
for h,group in before['duplicate_groups'].items():
 eligible=[p for p in group if p not in deleted and Path(p).suffix.lower() in {'.wav','.npz','.pt','.zip','.png','.jpg','.jpeg','.pdf','.svg'} and bypath[p]['bytes']>0]
 if len(eligible)<2:continue
 def priority(p):
  return (0 if '/phase1_caseB_self_target_020s_3targets_3d4d/preflight/T' in p else 1,
          0 if '/diagnostics/target.' in p or '/preflight/target.' in p else 1,
          0 if '/interrupted_attempt/' not in p and '/smoke/' not in p else 1,len(p),p)
 keep=min(eligible,key=priority)
 for p in eligible:
  if p==keep:continue
  links[p]=keep
  deleted[p]=dict(bypath[p],reason='SHA256・長さ一致の重複バイナリ。相対symlinkで旧ファイル参照とバイト列を保持',action='deduplicate_symlink',replacement=mapped(keep))
writecsv(M/'deletions.csv',[{**{k:v for k,v in r.items() if k not in ('mtime_ns','image')},'new_path':mapped(p)} for p,r in sorted(deleted.items())])
writecsv(M/'folder_mapping.csv',[dict(old_path='results/'+o,new_path='results/'+n) for o,n in FOLDERS.items()])
dump(M/'plan.json',dict(date='2026-11-26',head=before['head'],baseline_git_status=before['git_status'],running_calculation_processes=[],process_check='Host ps -C python,python3,python3.10,python3.12,birdsong,g++ returned header only; no jobs at inspection',operations_planned=len(deleted),policy='Retain all unique scientific raw data/models/configuration/numeric logs; delete regenerable individual drawings and retired smoke payloads; byte-exact duplicates become relative symlinks; original manifests stay unchanged. No training or physical simulation.'))
# Validate every planned input before first mutation.
for p,r in deleted.items():assert digest(ROOT/p)==r['sha256'],p
for p,r in deleted.items():
 if r['action']=='delete_regenerable':(ROOT/p).unlink()
for p,keep in links.items():
 dest=ROOT/p;dest.unlink();dest.symlink_to(os.path.relpath(ROOT/keep,dest.parent))
# Moving symlink-rich trees requires rebasing each link after all moves.
for old,new in FOLDERS.items():
 a=ROOT/'results'/old;b=ROOT/'results'/new
 assert a.is_dir() and not b.exists(),(a,b)
 b.parent.mkdir(parents=True,exist_ok=True);a.rename(b)
for old,keep in links.items():
 dest=ROOT/mapped(old);dest.unlink();dest.symlink_to(os.path.relpath(ROOT/mapped(keep),dest.parent))
# Empty obsolete work directories only.
for p in sorted((ROOT/'results').rglob('*'),key=lambda p:-len(p.parts)):
 if p.is_dir() and not p.is_symlink() and not any(p.iterdir()):p.rmdir()
# Source edits are solely path resolution and explicit original/current hash checking.
changes={};reference_changes=[]
for rel,info in src.items():
 p=ROOT/rel;text=p.read_text();old=text
 for a,b in sorted(FOLDERS.items(),key=lambda kv:-len(kv[0])):text=text.replace('results/'+a,'results/'+b)
 tree=ast.parse(text);lines=text.splitlines(keepends=True);offsets=[0]
 for l in lines:offsets.append(offsets[-1]+len(l))
 def span(n):return offsets[n.lineno-1]+n.col_offset,offsets[n.end_lineno-1]+n.end_col_offset
 edits=[];use_resolve=False;use_match=False
 for node in ast.walk(tree):
  if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id=='ROOT' and not isinstance(node.right,(ast.Constant,ast.JoinedStr)):
   a,b=span(node);edits.append((a,b,'resolve_result_path('+text[a:b]+')'));use_resolve=True
  if isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],(ast.Eq,ast.NotEq)):
   left,right=node.left,node.comparators[0]
   ishash=lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('source_hashes','hashes','current_sources','baseline_hashes')
   if (ishash(left) or ishash(right)) and not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='digest' for n in (left,right) for x in ast.walk(n)):
    a,b=span(node);la,lb=span(left);ra,rb=span(right)
    edits.append((a,b,('not ' if isinstance(node.ops[0],ast.NotEq) else '')+'source_hashes_match('+text[la:lb]+', '+text[ra:rb]+')'));use_match=True
 # Comparisons contain no ROOT binop in this code; reject any overlapping edits.
 edits.sort()
 for e1,e2 in zip(edits,edits[1:]): assert e1[1]<=e2[0],(rel,e1,e2)
 for a,b,s in reversed(edits):text=text[:a]+s+text[b:]
 if use_resolve or use_match:
  imports=[]
  if use_resolve:imports.append('resolve as resolve_result_path')
  if use_match:imports.append('source_hashes_match')
  tree=ast.parse(text);pos=0
  for node in tree.body:
   if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and node is tree.body[0] or isinstance(node,ast.ImportFrom) and node.module=='__future__':pos=node.end_lineno
   else:break
  lines=text.splitlines(keepends=True);lines.insert(pos,'from '+('..' if rel.startswith('tests/') else '.')+'result_paths import '+', '.join(imports)+'\n');text=''.join(lines)
 if text!=old:
  p.write_text(text);reference_changes.append(dict(path=rel,kind='python',old_sha256=info['before_sha256'],new_sha256=digest(p)))
  changes[rel]=dict(info,after_sha256=digest(p),reason='Directory migration; resolve stored original paths; explicitly validate recorded source hash transitions (no algorithm or physical changes)')
dump(M/'source_changes.json',changes)
dump(M/'reference_changes.json',reference_changes)
print(json.dumps(dict(deleted_unique=sum(r['action']=='delete_regenerable' for r in deleted.values()),duplicate_payloads=len(links),payload_bytes_removed=sum(r['bytes'] for r in deleted.values()),source_files_updated=len(changes)),indent=2))
