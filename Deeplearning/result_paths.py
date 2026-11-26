"""Resolve historical result paths without rewriting frozen experiment records.

Only directory names change. No numerical data or historical digests are altered.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIGRATION = ROOT / 'results/maintenance/20261126'
FOLDERS = {
    'phase0_fixed_target_20261108': 'Phase0/synthetic',
    'phase1_constant_020s': 'Phase1/CaseA/parameter_dimension',
    'phase1_3d_domain_diagnostics': 'Phase1/CaseA/domain_diagnostics',
    'phase1_restricted3d_weights': 'Phase1/CaseA/reward_weights/restricted_domain',
    'phase1_caseA_3d_network_width': 'Phase1/CaseA/network_width',
    'phase1_caseA_sac128_reward_weights': 'Phase1/CaseA/reward_weights/sac128',
    'phase1_caseA_sac128_actor_lr': 'Phase1/CaseA/learning_rate/actor',
    'phase1_caseA_sac128_critic_lr': 'Phase1/CaseA/learning_rate/critic',
    'phase1_caseA_sac128_stability_diagnostics': 'Phase1/CaseA/stability_diagnostics',
    'phase1_caseA_wide_local_exploration': 'Phase1/CaseA/exploration_strategy',
    'phase1_caseB_self_target_020s_sac128': 'Phase1/CaseB/target_diagnostics/initial_stop',
    'phase1_caseB_self_target_020s_methods': 'Phase1/CaseB/target_diagnostics/high_left_force',
    'phase1_caseB_self_target_020s_methods_f0lr_scan': 'Phase1/CaseB/target_diagnostics/low_forces',
    'phase1_caseB_self_target_020s_3targets_3d4d': 'Phase1/CaseB/3targets_3d4d',
    'issue9': 'reference/phonation_domain',
    'issue10': 'Phase1/CaseA/reward_diagnostics',
    'issue12_synthetic': 'reference/activity_mask',
    'diagnostics': 'reference/implementation_diagnostics',
    'server_smoke_20261113T190418+0900': 'reference/environment_checks/cpu',
    'server_smoke_gpu_20261113T194403+0900': 'reference/environment_checks/gpu',
}


def resolve(path):
    """Map a stored old relative/absolute path; keep its input path convention."""
    path = Path(path)
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part == 'results' and parts[i + 1] in FOLDERS:
            return Path(*parts[:i + 1], FOLDERS[parts[i + 1]], *parts[i + 2:])
    return path


def source_hashes_match(expected, actual):
    """Accept exact hashes or explicitly recorded, hash-checked migration edits.

    This is not a digest replacement. Both original and current values remain
    visible in source_changes.json. An unrecorded source change still fails.
    """
    import hashlib
    import json
    if expected == actual:
        return True
    if not isinstance(expected, dict) or not isinstance(actual, dict) or expected.keys() != actual.keys():
        return False
    record = MIGRATION / 'source_changes.json'
    changes = json.loads(record.read_text()) if record.is_file() else {}
    for name, old in expected.items():
        if old == actual[name]:
            continue
        item = changes.get(name, {})
        if (item.get('before_sha256'), item.get('after_sha256')) != (old, actual[name]):
            return False
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != actual[name]:
            return False
    return True
