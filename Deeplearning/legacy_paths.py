"""Physical model locations and stable historical protection-record paths."""
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
LEGACY_SRC = REPO / "legacy" / "src"
PHASE0_ROOT = REPO / "Deeplearning/results/phase0_fixed_target_20261108"


def legacy_phase0_root():
    """Read the relocated random-target experiment, with pre-migration fallback."""
    relocated = PHASE0_ROOT / "legacy_random_target"
    previous = REPO / "Deeplearning/results/phase0"
    return relocated if relocated.exists() or not previous.exists() else previous


def model_files():
    """Keep historical basename keys so existing protection records stay valid."""
    return list(REPO.glob("BirdsongModel*")) + list(LEGACY_SRC.glob("BirdsongModel*.cpp")) + list(LEGACY_SRC.glob("BirdsongModel*.h"))


def protected_path(relative):
    """Resolve historical model and Phase 0 artifact paths to their current home.

    Hash values are never altered; all other repository paths retain their meaning.
    """
    path = Path(relative)
    previous = Path("Deeplearning/results/phase0")
    if path.is_relative_to(previous):
        return legacy_phase0_root() / path.relative_to(previous)
    if len(path.parts) == 1 and path.name.startswith("BirdsongModel") and path.suffix in (".cpp", ".h"):
        return LEGACY_SRC / path.name
    return REPO / path
