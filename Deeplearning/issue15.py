"""Issue 15: analyze saved A, train B/C, audit, then integrate Phase 0."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from .issue15_analysis import analyze, write_json

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"results/phase0_fixed_target_20261108"
LEGACY = ROOT/"results/phase0"
CONDITIONS = ("spectral_only", "envelope_only")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory):
    if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError(f"Refuse symlink source: {directory}")
    return {str(p.relative_to(directory)): dict(sha256=sha(p), size=p.stat().st_size)
            for p in directory.rglob("*") if p.is_file()}


def worker(condition, name, seed, config, output):
    from .issue14 import train_one
    return dict(condition=condition, **train_one(name, seed, config, output))


def execute(workers=6):
    if not 1 <= workers <= 6:
        raise ValueError("Use 1..6 single-thread CPU workers")
    if (PARENT/"reward_ablation").exists():
        raise FileExistsError("Ablation results already exist; use the report command to read them, do not overwrite")
    branch = subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    if branch != "main_deeplearning":
        raise RuntimeError("Issue 15 uses main_deeplearning, not the older main results")
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    control = inventory(PARENT)
    legacy = inventory(LEGACY)
    tracked = set(subprocess.check_output(["git","ls-files","--",str(LEGACY.relative_to(ROOT))],cwd=ROOT,text=True).splitlines())
    if {str((LEGACY/p).relative_to(ROOT)) for p in legacy} != tracked:
        raise RuntimeError("Legacy directory has untracked or missing files; preserve and inspect")
    for filename, item in legacy.items():
        path = str((LEGACY/filename).relative_to(ROOT.parent))
        data = subprocess.check_output(["git","show",f"{commit}:{path}"],cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Legacy content differs from recoverable commit: {filename}")
    work = ROOT/"results/.issue15-work"/(datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
    work.mkdir(parents=True)
    output = work/"artifacts"
    output.mkdir()
    protected = {}
    for filename in subprocess.check_output(["git","ls-files","-z"],cwd=ROOT,text=True).split("\0"):
        if not filename: continue
        p = ROOT/filename
        if filename.startswith("results/") and not p.is_relative_to(PARENT) and not p.is_relative_to(LEGACY):
            if p.is_file(): protected[filename] = sha(p)
    write_json(output/"control_before.json",control)
    write_json(output/"legacy_inventory.json",dict(commit=commit,old_path=str(LEGACY.relative_to(ROOT.parent)),
               new_path=str((PARENT/"legacy_random_target").relative_to(ROOT.parent)),files=legacy,git_contents_verified=True))
    write_json(output/"protected_before.json",protected)
    shutil.copyfile(PARENT/"REPORT.md",output/"control_report_original.md")
    analysis = analyze(PARENT,output/"reanalysis")
    print("JSON-only reanalysis complete; no training or policy evaluation added",flush=True)
    tests=[sys.executable,"-m","unittest","Deeplearning.tests.test_issue15","Deeplearning.tests.test_issue14",
           "Deeplearning.tests.test_core.SineTests","Deeplearning.tests.test_core.AcousticTests","Deeplearning.tests.test_plotting"]
    with (output/"test_results.txt").open("w") as stream:
        result=subprocess.run(tests,cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Tests failed; see {output/'test_results.txt'}")
    import re
    count=re.search(r"Ran (\d+) tests?",(output/"test_results.txt").read_text())
    write_json(output/"test_results.json",dict(passed=True,tests=int(count.group(1)),command=tests))
    from .issue14 import diagnostics
    base=json.loads((PARENT/"config.json").read_text())
    packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","torch","stable-baselines3","gymnasium","matplotlib","SciencePlots")}
    recorded=json.loads((PARENT/"manifest.json").read_text())["package_versions"]
    differences={k:dict(control=recorded.get(k),new=v) for k,v in packages.items() if recorded.get(k)!=v}
    sources=("issue14.py","sine_env.py","experiments.py","acoustics.py","plotting.py","configs/issue14_fixed_target.json")
    provenance=dict(workflow="issue15_reward_ablation",branch=branch,control_commit=commit,
        started_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),workers=workers,
        base_config=base,package_versions=packages,package_differences_from_control=differences,
        training_source_sha256={p:sha(ROOT/p) for p in sources},control_retrained=False,
        common_evaluation="combined full 0.35 s distance and f/A MAE; deterministic separate env",
        source_code_difference="SineConfig reward selector and info fields; baseline branch retains original total exactly",
        weight_tuning_preregistered=False,reproducible_command="PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python -m Deeplearning.issue15 --workers 6")
    write_json(output/"manifest.json",provenance)
    configs={}
    for condition in CONDITIONS:
        folder=output/condition
        folder.mkdir()
        config=dict(base,reward_terms=condition)
        configs[condition]=config
        write_json(folder/"config.json",config)
        diagnostics(folder,config)
    print(f"Tests PASS. Training 18 fresh B/C runs in {output}",flush=True)
    started=time.perf_counter(); summaries=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs=[pool.submit(worker,c,name,seed,configs[c],output/c/f"{name}_{seed}")
              for seed in (0,1,2) for c in CONDITIONS for name in ("SAC","TD3","PPO")]
        for future in as_completed(jobs):
            summaries.append(future.result())
            write_json(output/"summary.json",summaries)
    write_json(output/"execution.json",dict(completed_runs=len(summaries),training_steps=sum(r["training_steps"] for r in summaries),
        evaluation_steps=sum(r["evaluation_steps"] for r in summaries),execution_wall_s=time.perf_counter()-started))
    for condition in CONDITIONS:
        selected=[{k:v for k,v in r.items() if k!="condition"} for r in summaries if r["condition"]==condition]
        write_json(output/condition/"summary.json",selected)
        for p in (output/condition).glob("*_[012]/progress.json"): p.unlink()
    finish(output)


def finish(output):
    """Resume completed learning in a fresh process; never learn again here."""
    output=Path(output).resolve()
    if output.parent.parent!=ROOT/"results/.issue15-work" or output.name!="artifacts":
        raise ValueError("Finish only a complete isolated Issue 15 staging area")
    execution=json.loads((output/"execution.json").read_text())
    if execution["completed_runs"]!=18 or execution["training_steps"]!=1843200 or execution["evaluation_steps"]!=32760:
        raise RuntimeError("Learning is incomplete; finish cannot substitute for missing runs")
    control=json.loads((output/"control_before.json").read_text())
    legacy=json.loads((output/"legacy_inventory.json").read_text())["files"]
    if inventory(PARENT)!=control or inventory(LEGACY)!=legacy:
        raise RuntimeError("A or legacy changed during B/C learning; nothing will be published")
    from .issue15_audit import audit_ablation
    from .issue15_report import generate_report
    audit_ablation(output,PARENT)
    generate_report(output,PARENT)
    from .issue15_integration import integrate
    integrate(output,PARENT,LEGACY)
    shutil.rmtree(output.parent)
    print(f"Issue 15 complete: {PARENT}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--finish",type=Path,help="Audit and publish completed staged runs without additional learning")
    args=parser.parse_args()
    if args.finish: finish(args.finish)
    else: execute(args.workers)
