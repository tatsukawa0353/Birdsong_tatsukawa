"""Audit common evaluation, reward decomposition and saved B/C artifacts."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .issue14_audit import Auditor
from .issue15_analysis import write_json


def audit_ablation(output, control):
    output, control = Path(output), Path(control)
    start=time.perf_counter(); results={}; distributions=[]
    base=json.loads((control/"config.json").read_text())
    for condition in ("spectral_only","envelope_only"):
        folder=output/condition
        config=json.loads((folder/"config.json").read_text())
        if {k:v for k,v in config.items() if k!="reward_terms"} != base or config["reward_terms"]!=condition:
            raise RuntimeError("Conditions must differ from A only in selected reward terms")
        auditor=Auditor(folder)
        expected_files={folder/f"{name}_{seed}/summary.json" for name in ("PPO","TD3","SAC") for seed in (0,1,2)}
        found_files=set(folder.glob("*_[012]/summary.json"))
        if found_files!=expected_files:
            raise RuntimeError("Ablation cohort is incomplete or has unknown runs")
        auditor.counts["runs_found"]=len(found_files)
        for name in ("PPO","TD3","SAC"):
            for seed in (0,1,2):
                directory=folder/f"{name}_{seed}"
                auditor.audit_run(directory/"summary.json",name,seed)
                summary=json.loads((directory/"summary.json").read_text())
                original=json.loads((control/f"{name}_{seed}/summary.json").read_text())
                if summary["effective_hyperparameters"]!=original["effective_hyperparameters"]:
                    raise RuntimeError("Effective learner settings differ from A")
                if summary["initial_record"]["parameter_sha256"]!=original["initial_record"]["parameter_sha256"] or summary["initial_metrics"]!=original["initial_metrics"]:
                    raise RuntimeError("Same-seed initial policy or evaluation differs from A")
                with np.load(directory/"training_trace.npz",allow_pickle=False) as trace:
                    rewards=trace["rewards"].copy()
                    spectral=trace["log_spectral"].copy(); envelope=trace["envelope"].copy()
                    expected=-spectral if condition=="spectral_only" else -envelope
                    if rewards.shape!=(102400,) or not np.isfinite([rewards,spectral,envelope]).all():
                        raise RuntimeError("Non-finite or incomplete reward decomposition")
                    np.testing.assert_allclose(rewards,expected,rtol=1e-6,atol=1e-8)
                    with np.load(control/f"{name}_{seed}/training_trace.npz") as initial_trace:
                        np.testing.assert_array_equal(trace["actions"][:256],initial_trace["actions"][:256])
                if summary["training_reward_terms"]!=condition or summary["evaluation_reward_terms"]!="combined":
                    raise RuntimeError("Learning reward leaked into common evaluation")
                for field,data in (("reward",rewards),("local_spectral",spectral),("local_envelope",envelope)):
                    distributions.append(dict(condition=condition,algorithm=name,seed=seed,quantity=field,
                        count=len(data),minimum=float(data.min()),mean=float(data.mean()),sd=float(data.std()),
                        q05=float(np.quantile(data,.05)),median=float(np.median(data)),q95=float(np.quantile(data,.95)),maximum=float(data.max())))
        auditor.oracle()
        result=dict(passed=not auditor.errors,checks=auditor.checks,errors=auditor.errors,counts=auditor.counts)
        write_json(folder/"audit.json",result)
        if auditor.errors:
            raise RuntimeError(f"{condition} audit failed: {auditor.errors[:3]}")
        results[condition]=result
    # A's scalar reward trace already exists. No training or policy evaluation
    # is needed to recover its actual distribution; per-step components were
    # not logged in A and are deliberately not invented here.
    for name in ("PPO","TD3","SAC"):
        for seed in (0,1,2):
            with np.load(control/f"{name}_{seed}/training_trace.npz") as trace:
                data=trace["rewards"]
                distributions.append(dict(condition="combined",algorithm=name,seed=seed,quantity="reward",count=len(data),
                    minimum=float(data.min()),mean=float(data.mean()),sd=float(data.std()),q05=float(np.quantile(data,.05)),
                    median=float(np.median(data)),q95=float(np.quantile(data,.95)),maximum=float(data.max())))
    from .issue15_analysis import write_csv
    write_csv(output/"training_reward_distributions.csv",distributions)
    manifest=json.loads((output/"manifest.json").read_text())
    for filename,expected in manifest["training_source_sha256"].items():
        actual=hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()
        if actual!=expected:
            prefix=(Path(__file__).parent/filename).read_text().split('if __name__ == "__main__":')[0]
            cli_only=(filename=="experiments.py" and hashlib.sha256(prefix.encode()).hexdigest()==manifest.get("experiments_import_prefix_sha256"))
            if not cli_only:
                raise RuntimeError(f"A computational source changed during learning: {filename}")
    result=dict(passed=True,conditions={k:dict(checks=v["checks"],counts=v["counts"]) for k,v in results.items()},
                wall_s=time.perf_counter()-start,source_version_diff=manifest["source_code_difference"],
                added_training_steps=0,registered_training_steps=1843200,registered_evaluation_steps=32760,
                audit_policy_evaluation_steps=18*35,local_reward_mapping_verified=True,
                common_final_evaluation=True)
    result["initial_policies_and_first_256_actions_match_A"]=True
    write_json(output/"audit.json",result)
    return result
