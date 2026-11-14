"""Publish audited B/C and relocate every legacy byte before removing old path."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from .issue15 import ROOT, inventory, sha
from .issue15_analysis import write_json


def integrate(output,parent,legacy):
    output,parent,legacy=map(lambda p:Path(p).resolve(),(output,parent,legacy))
    expected_parent=ROOT/"results/phase0_fixed_target_20261108"
    if parent!=expected_parent or legacy!=ROOT/"results/phase0":
        raise ValueError("Integration is limited to the two declared Phase 0 directories")
    if output.parent.parent!=ROOT/"results/.issue15-work" or output.name!="artifacts":
        raise ValueError("Publish only the isolated Issue 15 work directory")
    if any(p.is_symlink() for p in (parent,legacy,output)):
        raise ValueError("Symlink roots are not eligible")
    control_before=json.loads((output/"control_before.json").read_text())
    migration=json.loads((output/"legacy_inventory.json").read_text())
    if inventory(parent)!=control_before or inventory(legacy)!=migration["files"]:
        raise RuntimeError("Source artifacts changed; no integration or deletion allowed")
    if not json.loads((output/"audit.json").read_text())["passed"]:
        raise RuntimeError("B/C audit gate failed")
    manifest=json.loads((output/"manifest.json").read_text())
    source_names=("issue15.py","issue15_analysis.py","issue15_audit.py","issue15_report.py","issue15_integration.py",
        "issue14.py","issue14_audit.py","issue14_report.py","issue14_transaction.py","sine_env.py","experiments.py",
        "legacy_paths.py","redraw.py","run.py","make_report.py","tests/test_issue15.py","tests/test_phase0_paths.py")
    manifest["published_source_sha256"]={name:sha(ROOT/name) for name in source_names}
    manifest["diagnostic_logging_difference"]="B/C record per-step reward components; no new RNG operations or learner hyperparameters"
    manifest["path_maintenance_difference"]="experiments.py CLI default relocated; imported learning-source prefix exactly unchanged"
    patch=subprocess.check_output(["git","diff",manifest["control_commit"],"--",*source_names],cwd=ROOT,text=True)
    for name in source_names:
        tracked=subprocess.run(["git","ls-files","--error-unmatch","--",name],cwd=ROOT,capture_output=True)
        if tracked.returncode:
            delta=subprocess.run(["git","diff","--no-index","--","/dev/null",str(Path(ROOT.name)/name)],cwd=ROOT.parent,capture_output=True,text=True)
            if delta.returncode not in (0,1): raise RuntimeError(delta.stderr)
            patch+=delta.stdout
    write_json(output/"manifest.json",manifest)
    (output/"source_changes.patch").write_text(patch)
    for name,value in json.loads((output/"protected_before.json").read_text()).items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=value:
            raise RuntimeError(f"An unrelated result changed: {name}")
    destination=parent/"legacy_random_target"
    ablation=parent/"reward_ablation"
    archived_report=parent/"CONTROL_REPORT.md"
    if destination.exists() or ablation.exists() or archived_report.exists():
        raise FileExistsError("Will not replace integrated or unknown data")
    copied=output.parent/"legacy_copy"
    shutil.copytree(legacy,copied,symlinks=True)
    if inventory(copied)!=migration["files"]:
        raise RuntimeError("Legacy copy hash or size differs")
    os.replace(copied,destination)
    os.replace(output,ablation)
    original=(ablation/"control_report_original.md").read_bytes()
    assert hashlib.sha256(original).hexdigest()==control_before["REPORT.md"]["sha256"]
    archived_report.write_bytes(original)
    (ablation/"control_report_original.md").unlink()
    try:
        # The source path remains present throughout these gates. The resolver
        # must already read the new copy, including every unique legacy NPZ.
        from .legacy_paths import legacy_phase0_root, protected_path
        if legacy_phase0_root()!=destination:
            raise RuntimeError("Legacy resolver did not select the relocated directory")
        for relative,value in migration["files"].items():
            old_relative=Path(migration["old_path"])/relative
            if sha(protected_path(old_relative))!=value["sha256"]:
                raise RuntimeError(f"Historical reference mismatch: {relative}")
            saved=subprocess.check_output(["git","show",f"{migration['commit']}:{old_relative}"],cwd=ROOT)
            if hashlib.sha256(saved).hexdigest()!=value["sha256"]:
                raise RuntimeError(f"Git restoration mismatch: {relative}")
        for name in ("PPO","TD3","SAC"):
            for seed in (0,1,2):
                curve=json.loads((destination/f"{name}_{seed}/curve.json").read_text())
                if [p["steps"] for p in curve]!=list(range(0,4097,512)):
                    raise RuntimeError("Historical curve loader no longer sees original inputs")
        from .issue14_audit import Auditor
        auditor=Auditor(parent)
        checked=auditor.run()
        write_json(ablation/"control_audit.json",checked)
        if not checked["passed"]:
            raise RuntimeError(f"Frozen A audit failed after relocation: {checked['errors'][:3]}")
        for name,value in control_before.items():
            path=archived_report if name=="REPORT.md" else parent/name
            if sha(path)!=value["sha256"]:
                raise RuntimeError(f"A changed during integration: {name}")
        for name,value in json.loads((ablation/"protected_before.json").read_text()).items():
            if sha(ROOT/name)!=value:
                raise RuntimeError(f"Unrelated result changed: {name}")
        if inventory(legacy)!=migration["files"]:
            raise RuntimeError("Old directory gained unknown files; preserve it")
        # All checks above passed while both copies existed. Rename the exact
        # old tree into our private work area; only that verified copy is later
        # removed. No symlink target or other results directory is traversed.
        removed=output.parent/"verified_legacy_removed"
        os.replace(legacy,removed)
        from .issue15_report import update_index
        import csv
        with (ablation/"comparison_aggregate.csv").open() as stream:
            aggregates=[{k:float(v) if k not in ("condition","algorithm") else v for k,v in r.items()} for r in csv.DictReader(stream)]
        update_index(parent,aggregates)
        result=dict(passed=True,completed_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            branch="main_deeplearning",legacy_files=len(migration["files"]),
            legacy_hashes_and_sizes_equal=inventory(destination)==migration["files"],
            old_path_removed=not legacy.exists(),git_restoration_verified=True,
            control_original_files=len(control_before),control_bytes_unchanged=True,
            original_control_report="CONTROL_REPORT.md",control_audit_passed=True,
            protected_results=len(json.loads((ablation/"protected_before.json").read_text())),
            protected_results_unchanged=True,symlink_targets_followed=False,
            historical_metadata="A manifest/CSV/protected records retain original paths and hashes; resolver maps moved legacy bytes",
            historical_completion="Parent completion_verification.json describes original A, not the extended integrated tree",
            old_legacy_restore_to_empty_directory=f"git archive {migration['commit']} {migration['old_path']} | tar -x -C /path/to/empty/recovery-directory")
        write_json(ablation/"integration.json",result)
        if not result["legacy_hashes_and_sizes_equal"] or not result["old_path_removed"]:
            raise RuntimeError("Post-integration invariant failed")
    except BaseException:
        # Preserve failed work and the exact old source; do not destroy an
        # unexpected file merely to force a rollback.
        removed=output.parent/"verified_legacy_removed"
        if removed.exists() and not legacy.exists(): os.replace(removed,legacy)
        (parent/"REPORT.md").write_bytes(original)
        if archived_report.exists() and archived_report.read_bytes()==original: archived_report.unlink()
        if destination.exists(): os.replace(destination,output.parent/"legacy_failed")
        if ablation.exists(): os.replace(ablation,output)
        raise
