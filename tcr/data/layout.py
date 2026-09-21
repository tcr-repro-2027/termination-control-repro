"""Dataset locations and separation of format previews from experiment inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

FULL_DATA_URL = "https://huggingface.co/datasets/tcr-repro-2027/termination-control-data"
PREVIEW_MARKER = "PREVIEW_ONLY.json"


def stage_directory(dataset_root: Path, stage: str) -> Path:
    """The final cleaned data are at the root; optional upstream stages are nested."""
    root = Path(dataset_root)
    return root / "cleanv2" if stage == "cleanv2" else root / "stages" / stage


def require_full_data(path: str | Path) -> None:
    """Reject a bundled preview directory or a file within it."""
    candidate = Path(path).resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / PREVIEW_MARKER).is_file():
            raise ValueError(
                "The bundled datasets contain only 10 records per file for format "
                "preview. Download the full data from " + FULL_DATA_URL +
                " into full_data/datasets and set DATA_ROOT to that directory."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        for path in args.paths:
            require_full_data(path)
    except ValueError as exc:
        parser.exit(1, f"[data] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
