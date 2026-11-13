"""Physical model locations and stable historical protection-record paths."""
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
LEGACY_SRC = REPO / "legacy" / "src"


def model_files():
    """Keep historical basename keys so existing protection records stay valid."""
    return list(REPO.glob("BirdsongModel*")) + list(LEGACY_SRC.glob("BirdsongModel*.cpp")) + list(LEGACY_SRC.glob("BirdsongModel*.h"))


def protected_path(relative):
    """Resolve only historical root model source/header names to their new home.

    Hash values are never altered; all other repository paths retain their meaning.
    """
    path = Path(relative)
    if len(path.parts) == 1 and path.name.startswith("BirdsongModel") and path.suffix in (".cpp", ".h"):
        return LEGACY_SRC / path.name
    return REPO / path
