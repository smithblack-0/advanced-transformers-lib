"""Install the shram model's dependencies into the current Python environment.

Run from the repository root:

    python src/shram/setup_dev_environment.py

Installs all runtime dependencies declared in src/shram/pyproject.toml via
pip install -e. After running, the shram package is importable as 'shram'
in addition to the standard 'src.shram' path used when running from the
repository root.

The primary distribution channel for this architecture is HuggingFace Hub.
For those who want more control — running experiments directly from source,
modifying the architecture locally, or contributing changes — local install
via this script is the alternative path.
"""

import shutil
import subprocess
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).parent


def setup() -> None:
    """Install shram's dependencies via pip install -e and requirements-dev.txt.

    Installs the dev requirements first (which pins torch to the CPU-only build
    via PyTorch's wheel index), then installs the package in editable mode.
    Resolves paths relative to this file so the script is runnable from any
    working directory.
    """
    requirements = MODEL_DIR / "requirements-dev.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(MODEL_DIR)],
        check=True,
    )
    build_dir = MODEL_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    setup()
