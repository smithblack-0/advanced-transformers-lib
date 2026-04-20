"""Produce a Hub-compatible staging directory from the llama3 model source directory.

The llama3 model directory is already flat — no subdirectories contain Python files,
so no renaming or import rewriting is required. This module provides the same
stage(source_dir, dest_dir) interface as the other models so the CI orchestrator
can treat every model identically.
"""

import shutil
from pathlib import Path


def stage(source_dir: Path, dest_dir: Path) -> None:
    """Copy all model files from source_dir into dest_dir, excluding build artifacts.

    The llama3 source tree is flat; every file is copied directly to dest_dir root
    with its original name. dest_dir must already exist and should be empty.

    Files excluded from staging:
    - __pycache__ directories and .pyc files

    Args:
        source_dir: Source model directory (e.g. src/llama3/model/).
        dest_dir: Destination staging directory to write files into.
    """
    for src_path in sorted(source_dir.rglob("*")):
        if not src_path.is_file():
            continue

        rel = src_path.relative_to(source_dir)

        if any(part == "__pycache__" for part in rel.parts):
            continue
        if src_path.suffix == ".pyc":
            continue

        shutil.copy2(src_path, dest_dir / rel.name)
