from __future__ import annotations

import shutil
from pathlib import Path


def collect_outputs(run_dir: Path, destination: Path | None = None) -> tuple[Path, int]:
    destination = destination or run_dir / "collected"
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for tif_path in run_dir.glob("patches/*/outputs/output_SR_image_*.tif"):
        shutil.copy2(tif_path, destination / tif_path.name)
        copied += 1
    return destination, copied
