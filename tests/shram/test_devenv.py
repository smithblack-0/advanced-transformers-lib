"""Tests for setup_dev_environment.py.

Verifies the dev environment setup script exposes the correct interface, and
that pip install --target correctly installs the shram package such that
ShramConfig is importable under the short package name.
"""

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.shram.setup_dev_environment import setup

SHRAM_DIR = Path(__file__).parent.parent.parent / "src" / "shram"


class TestSetupDevEnvironment:
    def test_setup_is_callable(self):
        """setup() must be a callable."""
        assert callable(setup)


class TestPackageInstalled:
    def test_shram_importable_after_install(self):
        """ShramConfig must be importable under the short package name after install.

        Installs into a temporary directory to avoid mutating the active
        environment. The temporary directory is added to sys.path for the
        duration of the test and removed in cleanup.
        """
        # ignore_cleanup_errors=True is required on Windows: compiled .pyd extensions
        # (pulled in via libcst -> markupsafe) are locked by the OS while loaded and
        # cannot be deleted. The import succeeds; only cleanup is affected.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_path = str(tmp)

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--target", tmp_path, str(SHRAM_DIR)],
                check=True,
                capture_output=True,
            )

            build_dir = SHRAM_DIR / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)

            sys.path.insert(0, tmp_path)
            try:
                # Remove any cached module to ensure a fresh import from tmp_path.
                stale = [k for k in sys.modules if k == "shram" or k.startswith("shram.")]
                for k in stale:
                    del sys.modules[k]

                mod = importlib.import_module("shram.model.configuration")
                assert hasattr(mod, "ShramConfig")
            finally:
                sys.path.remove(tmp_path)
                stale = [k for k in sys.modules if k == "shram" or k.startswith("shram.")]
                for k in stale:
                    del sys.modules[k]
