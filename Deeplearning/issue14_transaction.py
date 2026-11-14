"""Stage Issue 14, validate old Git recovery, and publish only audited results."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .issue14 import ROOT, DEFAULT_OUTPUT, run, write_json, digest


def old_inventory(commit, target):
    repo = ROOT.parent
    relative = str(target.relative_to(repo))
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", relative], cwd=repo, text=True).splitlines()
    actual = {str(p.relative_to(repo)) for p in target.rglob("*") if p.is_file()}
    if not files or actual != set(files) or any(p.is_symlink() for p in target.rglob("*")):
        raise RuntimeError("Canonical results contain untracked/missing/symlink files; preserve them and inspect")
    hashes = {}
    for name in files:
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        value = hashlib.sha256(data).hexdigest()
        if digest(repo/name) != value:
            raise RuntimeError(f"Old tracked result differs from recoverable Git version: {name}")
        hashes[name] = value
    return hashes


def tree_inventory(target):
    if target.is_symlink() or any(p.is_symlink() for p in target.rglob("*")):
        raise RuntimeError("Result symlinks are not eligible for replacement")
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in target.rglob("*") if p.is_file()}


def replace_results(settings, workers):
    if not 1 <= workers <= 6:
        raise ValueError("Use 1..6 independent CPU workers")
    target = DEFAULT_OUTPUT.resolve()
    if target != ROOT/"results/phase0_fixed_target_20261108" or target.is_symlink():
        raise ValueError("Only the canonical Issue 14 directory may be replaced")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    completed_path = target/"completion_verification.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
    if "files_sha256" in completed:
        # An exact, self-recorded prior publication may be rerun even before a
        # user commits it. Unknown/untracked additions or edits still abort.
        inventory = tree_inventory(target)
        expected = {str(target.relative_to(ROOT.parent)/Path(name)): value for name, value in completed["files_sha256"].items()}
        actual = {k: v for k, v in inventory.items() if k != str(completed_path.relative_to(ROOT.parent))}
        if actual != expected:
            raise RuntimeError("Completed results have edits or unknown files; preserve and inspect before replacing")
        recovery = json.loads((target/"legacy_recovery.json").read_text())
        for name, value in recovery["tracked_files"].items():
            data = subprocess.check_output(["git", "show", f"{recovery['commit_sha']}:{name}"], cwd=ROOT.parent)
            if hashlib.sha256(data).hexdigest() != value:
                raise RuntimeError("Original 4096-step Git recovery cannot be verified")
        legacy = json.loads((target/"legacy_fixed_reference.json").read_text())
    else:
        inventory = old_inventory(commit, target)
        recovery = dict(commit_sha=commit, path=str(target.relative_to(ROOT.parent)), tracked_files=inventory,
                        untracked_files=[], verified_git_contents=True,
                        restore_command=f"git restore --source={commit} --worktree -- Deeplearning/results/phase0_fixed_target_20261108",
                        note="Restores tracked old files; remove only extra new files if fully reverting. No Git history is rewritten.")
        legacy = []
        for name in ("PPO", "TD3", "SAC"):
            for seed in (0, 1, 2):
                value = json.loads((target/f"{name}_{seed}/summary.json").read_text())
                legacy.append({key: value[key] for key in ("algorithm", "seed", "training_steps", "initial_error", "error", "tracking", "metrics")})
    work = ROOT/"results/.issue14-work"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
    work.mkdir(parents=True)
    stage = work/"artifacts"
    write_json(work/"legacy_recovery.json", recovery)
    tests = [sys.executable, "-m", "unittest", "Deeplearning.tests.test_issue14",
             "Deeplearning.tests.test_core.SineTests", "Deeplearning.tests.test_core.AcousticTests", "Deeplearning.tests.test_plotting"]
    started = time.perf_counter()
    with (work/"tests.log").open("w") as stream:
        checked = subprocess.run(tests, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
    text = (work/"tests.log").read_text()
    import re
    count = re.search(r"Ran (\d+) tests?", text)
    tests_record = dict(passed=checked.returncode == 0, tests_passed=int(count.group(1)) if count else None,
                        command=tests, wall_s=time.perf_counter()-started)
    write_json(work/"test_results.json", tests_record)
    if checked.returncode:
        raise RuntimeError(f"Pre-training tests failed; see {work/'tests.log'}")
    print(f"Git recovery and tests PASS. Staging 9 fresh runs in {stage}", flush=True)
    run(stage, settings, workers)
    for path in stage.glob("*_[012]/progress.json"):
        path.unlink()  # Transient monitoring is redundant once summaries exist.
    write_json(stage/"legacy_recovery.json", recovery)
    write_json(stage/"legacy_fixed_reference.json", legacy)
    write_json(stage/"test_results.json", tests_record)
    shutil.copyfile(work/"tests.log", stage/"test_results.txt")
    from .issue14_audit import audit
    from .issue14_report import generate_report
    manifest = json.loads((stage/"manifest.json").read_text())
    for filename in ("issue14_transaction.py", "issue14_audit.py", "issue14_report.py", "tests/test_issue14.py"):
        manifest["source_hashes"][filename] = digest(ROOT/filename)
    manifest["final_output"] = str(target.relative_to(ROOT.parent))
    manifest["previous_results_commit"] = recovery["commit_sha"]
    write_json(stage/"manifest.json", manifest)
    (stage/"source_changes.patch").write_text(subprocess.check_output(["git", "diff", commit, "--", *manifest["source_hashes"]], cwd=ROOT, text=True))
    audit(stage)
    generate_report(stage)
    if tree_inventory(target) != inventory:
        raise RuntimeError("Old results changed while training; will not publish")
    backup = work/"old_results"
    os.replace(target, backup)
    try:
        os.replace(stage, target)
        audit(target)
        generate_report(target)
    except BaseException:
        if target.exists():
            os.replace(target, work/"failed_publication")
        os.replace(backup, target)
        raise
    # Every old file was verified against Git before the transaction. No other
    # experiment, untracked user file or symlink target is removed here.
    assert work.parent == ROOT/"results/.issue14-work" and not work.is_symlink()
    shutil.rmtree(work)
    print(f"Published audited results: {target}", flush=True)
