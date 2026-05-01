"""Install the llama3 model's dependencies into the current Python environment.

Run from the repository root:

    python src/llama3/setup_dev_environment.py

Installs all runtime dependencies declared in src/llama3/pyproject.toml via
pip install -e. After running, the llama3 package is importable as 'llama3'
in addition to the standard 'src.llama3' path used when running from the
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
    """Install llama3's dependencies via pip install -e.

    Resolves the model directory relative to this file so the script is
    runnable from any working directory.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(MODEL_DIR)],
        check=True,
    )
    build_dir = MODEL_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    setup()
