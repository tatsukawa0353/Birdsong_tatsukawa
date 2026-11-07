"""One-decision constant-control adapters for synthetic Issue #13.

The physical model is Case A in both conditions. One condition selects only
epsilon_l; the other selects epsilon_l and pressure. Every successful decision
holds its controls over the original 35 native 10 ms blocks (0.35 seconds).
Existing integration, initial conditions, filtering, action decoding and reward
calculations are inherited unchanged from ValidationEnv.
"""
from .phase1_validation_env import ValidationEnv


CONTROL_CONDITIONS = ((0,), (0, 2))


class ConstantControlEnv(ValidationEnv):
    """Hold one or two selected controls for one complete 0.35-second episode.

    ``control_indices`` accepts only ``(0,)`` or ``(0, 2)``. The parent first
    constructs the matching action/observation spaces and ModelConfig; only its
    decision timing is adapted. Internally ``task='A'`` selects the parent's
    constant-hold path in both conditions. It does not denote a new model Case.

    All native advances use target-matched nonlearned values. As in the original
    environment, reset itself uses DEFAULT_CONTROLS and zero state/history; the
    decoded action is applied before any numerical integration takes place.
    Setup target-generation cost and cumulative simulation counters retain the
    parent's definitions. A supplied target avoids target generation entirely.
    """
    def __init__(self, control_indices=(0,), target_audio=None):
        try:
            indices = tuple(control_indices)
        except TypeError as exc:
            raise ValueError("Select control_indices=(0,) or (0, 2)") from exc
        if indices not in CONTROL_CONDITIONS:
            raise ValueError("Select control_indices=(0,) or (0, 2)")
        super().__init__(task="A" if indices == (0,) else "C", target_audio=target_audio)
        self.task = "A"
        self.horizon = self.decision_horizon = 1


Issue13Env = ConstantControlEnv
