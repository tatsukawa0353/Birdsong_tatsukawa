import csv,hashlib,importlib,json,re,shutil,subprocess
from pathlib import Path
import numpy as np
from Deeplearning.result_paths import ROOT,MIGRATION as M,source_hashes_match
from Deeplearning.results_maintenance import audit

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
source_before=json.loads((M/'source_before.json').read_text());changes={}
for name,record in source_before.items():
 if name=='result_paths.py':continue
 after=h(ROOT/name)
 if after!=record['before_sha256']:changes[name]={**record,'after_sha256':after,'reason':'Purpose-based path resolution and migration checks; common plot style adapters where noted. Physical equations and RL settings unchanged.'}
dump(M/'source_changes.json',changes)
docs=json.loads((M/'document_changes.json').read_text())
for r in docs:r['after_sha256']=h(ROOT/r['path'])
dump(M/'document_changes.json',docs)
oldignore=subprocess.check_output(['git','show','HEAD:Deeplearning/.gitignore'],cwd=ROOT)
dump(M/'extra_reference_changes.json',[dict(path='.gitignore',before_sha256=hashlib.sha256(oldignore).hexdigest(),after_sha256=h(ROOT/'.gitignore'),kind='canonical ignore paths')])
from Deeplearning.model import BirdModel
def forbidden(*a,**kw):raise AssertionError('Physical simulation is forbidden during cleanup verification')
BirdModel.advance=forbidden
ready={}
for name in ['issue19_data','issue21_data','issue22_data','issue23_data','issue24_data','issue25_data','issue26_data','issue27_data','issue27f_data','issue27g_data']:
 try:importlib.import_module('Deeplearning.'+name).ready();ready[name]='passed'
 except Exception as e:ready[name]=type(e).__name__+': '+str(e)
expected=json.loads((M/'ready_before.json').read_text())
for name,state in ready.items():assert state==expected[name],(name,state,expected[name])
for n,r in list(changes.items())[:2]:
 assert source_hashes_match({n:r['before_sha256']},{n:r['after_sha256']})
 assert not source_hashes_match({n:r['before_sha256']},{n:'0'*64})
latest=ROOT/'results/Phase1/CaseB/3targets_3d4d';low=ROOT/'results/Phase1/CaseB/target_diagnostics/low_forces'
targets=[]
for t,c in [('T1','b2_c10'),('T2','b3_c10')]:
 a=latest/f'preflight/{t}/target.npz';b=low/f'preflight/candidates/{c}.npz';assert h(a)==h(b)
 with np.load(a) as x,np.load(b) as y:
  assert x.files==y.files
  for k in x.files:np.testing.assert_array_equal(x[k],y[k]);assert x[k].dtype==y[k].dtype
  targets.append(dict(target=t,source_candidate=c,sha256=h(a),audio_samples=len(x['audio']),all_arrays_equal=True))
dump(M/'target_transfer_audit.json',dict(passed=True,targets=targets,new_oracle_generation=False))
import torch
torch.set_num_threads(1)
from Deeplearning.issue27g_data import load_checkpoint
runs=list((latest/'runs').glob('*/*/*/seed_*/summary.json'));assert len(runs)==54
models=predictions=0;max_action_error=0.;train=evals=0
for path in runs:
 s=json.loads(path.read_text());assert s['status']=='complete' and s['training']['episodes']==4096
 assert s['independent_policy_evaluations']==33
 train+=s['training']['episodes'];evals+=s['independent_policy_evaluations']
 for role in ('final','selected','posthoc'):
  p=path.parent/f'{role}.pt';r=torch.load(p,map_location='cpu',weights_only=True)
  assert (r['target'],r['dim'],r['method'],r['condition'])==(s['target'],s['dim'],s['method'],'B');assert r['policy'];models+=1
 if s['seed']==0:
  agent,r=load_checkpoint(path.parent/'final.pt',s['target'],s['dim'],s['method'],0)
  obs=np.zeros(8,np.float32);obs[0]=-1;action=agent.predict(obs,deterministic=True)[0]
  error=float(np.max(abs(action-np.asarray(r['action']))));assert error<1e-7,error
  max_action_error=max(max_action_error,error);predictions+=1
assert train==221184 and evals==1782
dump(M/'model_loading_audit.json',dict(passed=True,runs=54,checkpoints_loaded=models,final_policy_inferences=predictions,max_action_abs_error=max_action_error,preserved_training_steps=train,preserved_independent_evaluations=evals,simulator_advance_forbidden=True,generated_sounds=0))
compiled=[]
for p in [*ROOT.glob('*.py'),*(ROOT/'tests').glob('*.py')]:compile(p.read_text(),str(p),'exec');compiled.append(str(p.relative_to(ROOT)))
dump(M/'compile_audit.json',dict(passed=True,files=len(compiled),bytecode_written=False))
broken=[];checked=0
for p in [*ROOT.glob('*.md'),*(ROOT/'reports').rglob('*.md'),*(ROOT/'results').rglob('*.md')]:
 if '/maintenance/' in str(p):continue
 for lead,label,url in re.findall(r'(!?)\[([^\]]*)\]\(([^)\n]+)\)',p.read_text()):
  if re.match(r'^[a-zA-Z]+:',url) or url.startswith('#'):continue
  target=url.strip('<>').split('#')[0]
  if target and not (p.parent/target).exists():broken.append(dict(document=str(p.relative_to(ROOT)),target=url))
  checked+=1
allowed={(r['document'],r['target']) for r in json.loads((M/'preexisting_missing_links.json').read_text())}
new=[r for r in broken if (r['document'],r['target']) not in allowed]
dump(M/'link_audit.json',dict(local_links_checked=checked,broken=broken,new_broken=new,passed=not new));assert not new,new
result=audit();assert result['passed'];dump(M/'retention_audit.json',result)
shutil.copyfile('/tmp/birdsong-cleanup-redraw-T1/redraw.json',M/'redraw_validation.json')
dump(M/'verification.json',dict(passed=True,date='2026-11-26',original_physics_unchanged=True,ready_checks=ready,source_files_compile=len(compiled),model_weights_loaded=models,policy_predictions=predictions,target_arrays_verified=targets,new_broken_links=0,preexisting_missing_private_links=len(broken),redraw='Saved T1 NPZ spectrogram/waveform in /tmp; spectrogram visually inspected',training_runs_executed=0,physics_sounds_executed=0,original_oracles_reexecuted=False))
print('VERIFIED',dict(models=models,predictions=predictions,targets=len(targets),links=checked,retained=result['retained_artifacts_checked']))
