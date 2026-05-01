"""Tests for setup_dev_environment.py.

Verifies the dev environment setup script exposes the correct interface, and
that pip install --target correctly installs the llama3 package such that
Llama3Config is importable under the short package name.
"""

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.llama3.setup_dev_environment import setup

LLAMA3_DIR = Path(__file__).parent.parent.parent / "src" / "llama3"


class TestSetupDevEnvironment:
    def test_setup_is_callable(self):
        """setup() must be a callable."""
        assert callable(setup)


class TestPackageInstalled:
    def test_llama3_importable_after_install(self):
        """Llama3Config must be importable under the short package name after install.

        Installs into a temporary directory to avoid mutating the active
        environment. The temporary directory is added to sys.path for the
        duration of the test and removed in cleanup.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = str(tmp)

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--target", tmp_path, str(LLAMA3_DIR)],
                check=True,
                capture_output=True,
            )

            build_dir = LLAMA3_DIR / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)

            sys.path.insert(0, tmp_path)
            try:
                stale = [k for k in sys.modules if k == "llama3" or k.startswith("llama3.")]
                for k in stale:
                    del sys.modules[k]

                mod = importlib.import_module("llama3.model.configuration")
                assert hasattr(mod, "Llama3Config")
            finally:
                sys.path.remove(tmp_path)
                stale = [k for k in sys.modules if k == "llama3" or k.startswith("llama3.")]
                for k in stale:
                    del sys.modules[k]
