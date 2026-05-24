"""Tests for stage_for_hub.py.

Covers the sentinel pipeline normalisation helpers, import resolution, and the
stage_model() inlining orchestrator. A synthetic two-file source tree is used
for integration tests so they are fast, deterministic, and independent of model
evolution. The real-model smoke test catches import patterns present in the
model but absent from the synthetic tree.
"""

import tempfile
from pathlib import Path

import pytest

from src.shram.stage_for_hub import (
    _is_module_in_directory,
    comment_out_type_checking,
    inline_imports,
    resolve_comments_to_sentinels,
    resolve_import_blocks_to_sentinels,
    restore_sentinels,
    resolve_key,
    resolve_import,
    stage_model,
    standardize_import_blocks,
    validate_source,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "src" / "shram" / "model"

ENTRY_SOURCE = '''\
"""Entry point docstring."""
import torch
from .helper import HelperClass


class MainClass:
    """Main class docstring."""
    pass
'''

HELPER_SOURCE = '''\
"""Helper module docstring."""
import torch


class HelperClass:
    """Helper class."""
    pass
'''


@pytest.fixture
def synthetic_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "huggingface.py").write_text(ENTRY_SOURCE, encoding="utf-8")
    (src / "helper.py").write_text(HELPER_SOURCE, encoding="utf-8")
    (src / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    return src, dest


# ---------------------------------------------------------------------------
# Normalisation pipeline — unit tests
#
# The inliner operates through a sentinel pipeline to avoid false-triggering
# on the word "import" in docstrings, comments, and TYPE_CHECKING blocks.
# Each stage below has contracts that the downstream stage depends on:
#
#   comment_out_type_checking  — TYPE_CHECKING imports become comments so they
#       are not mistaken for live imports by the next stage.
#   resolve_comments_to_sentinels  — docstrings and comments are extracted
#       before import detection so their content (which often contains the word
#       "import") cannot confuse the import extractor.
#   resolve_import_blocks_to_sentinels  — top-level import blocks, including
#       parenthesized multiline forms, are replaced with sentinels so the
#       standardizer sees clean, bounded units.
#   standardize_import_blocks  — each sentinel's value is expanded into
#       canonical single-import lines; this is the form resolve_key and
#       resolve_import consume.
#   validate_source  — after all imports are sentineled, any remaining
#       "import" keyword indicates an unsupported construct (e.g. indented
#       import outside TYPE_CHECKING) that would silently produce broken output
#       if allowed through.
#
# A failure in any of these contracts propagates silently through the pipeline
# and produces a merged file that either crashes on import or inlines the wrong
# code — neither of which is detectable without running the result.
# ---------------------------------------------------------------------------

class TestCommentOutTypeChecking:
    def test_type_checking_block_commented(self):
        """Lines inside a TYPE_CHECKING block must be commented out."""
        source = "if TYPE_CHECKING:\n    import foo\n"
        result = comment_out_type_checking(source)
        assert "# if TYPE_CHECKING:" in result
        assert "#     import foo" in result

    def test_code_after_block_unchanged(self):
        """Code at a lower indent level after the block must pass through unchanged."""
        source = "if TYPE_CHECKING:\n    import foo\nx = 1\n"
        result = comment_out_type_checking(source)
        assert "x = 1" in result
        assert "# x = 1" not in result

    def test_code_before_block_unchanged(self):
        """Code before the TYPE_CHECKING block must pass through unchanged."""
        source = "x = 1\nif TYPE_CHECKING:\n    import foo\n"
        result = comment_out_type_checking(source)
        assert result.startswith("x = 1")


class TestResolveCommentsToSentinels:
    def test_docstring_becomes_sentinel(self):
        """Triple-quoted docstrings must be replaced by a sentinel key."""
        source = '"""A docstring."""\nx = 1\n'
        result, sentinels = resolve_comments_to_sentinels(source)
        assert '"""' not in result
        assert len(sentinels) == 1

    def test_line_comment_becomes_sentinel(self):
        """Single-line comments must be replaced by a sentinel key."""
        source = "# a comment\nx = 1\n"
        result, sentinels = resolve_comments_to_sentinels(source)
        assert "# a comment" not in result
        assert len(sentinels) == 1

    def test_restore_recovers_original(self):
        """restore_sentinels must recover the original source exactly."""
        source = '"""Docstring."""\n# comment\nx = 1\n'
        sentineled, sentinels = resolve_comments_to_sentinels(source)
        restored = restore_sentinels(sentineled, sentinels)
        assert restored == source

    def test_non_comment_code_unchanged(self):
        """Non-comment lines must pass through unchanged."""
        source = "x = 1\ny = 2\n"
        result, sentinels = resolve_comments_to_sentinels(source)
        assert result == source
        assert len(sentinels) == 0


class TestResolveImportBlocksToSentinels:
    def test_simple_import_becomes_sentinel(self):
        """A bare import statement must be replaced by a sentinel."""
        source = "import os\nx = 1\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        assert "import os" not in result
        assert len(sentinels) == 1
        assert "import os" in list(sentinels.values())[0]

    def test_from_import_becomes_sentinel(self):
        """A from-import statement must be replaced by a sentinel."""
        source = "from pathlib import Path\nx = 1\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        assert "from pathlib import Path" not in result
        assert len(sentinels) == 1

    def test_parenthesized_multiline_is_single_sentinel(self):
        """A parenthesized multiline import must produce exactly one sentinel."""
        source = "from os import (\n    path,\n    getcwd,\n)\nx = 1\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        assert len(sentinels) == 1

    def test_non_import_lines_unchanged(self):
        """Non-import lines must pass through unchanged."""
        source = "x = 1\ny = 2\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        assert result == source
        assert len(sentinels) == 0

    def test_restore_recovers_import_content(self):
        """restore_sentinels on the import sentinel dict must recover the original imports."""
        source = "import os\nfrom pathlib import Path\nx = 1\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        restored = restore_sentinels(result, sentinels)
        assert "import os" in restored
        assert "from pathlib import Path" in restored
        assert "x = 1" in restored

    def test_inline_comments_promoted(self):
        """Comment sentinels within import lines must appear before the import sentinel."""
        source = "import os  __COMMENT_0__\nx = 1\n"
        result, sentinels = resolve_import_blocks_to_sentinels(source)
        lines = result.split("\n")
        comment_idx = next(i for i, l in enumerate(lines) if "__COMMENT_0__" in l)
        import_idx = next(i for i, l in enumerate(lines) if "__IMPORT_" in l)
        assert comment_idx < import_idx


class TestStandardizeImportBlocks:
    def test_multi_name_expands(self):
        """A comma-separated from-import must expand to one line per name."""
        sentinels = {"__IMPORT_0__": "from os import path, getcwd"}
        result = standardize_import_blocks(sentinels)
        lines = result["__IMPORT_0__"].splitlines()
        assert len(lines) == 2
        assert "from os import path" in lines
        assert "from os import getcwd" in lines

    def test_parenthesized_expands(self):
        """A parenthesized import must expand to one line per name."""
        sentinels = {"__IMPORT_0__": "from os import (path, getcwd)"}
        result = standardize_import_blocks(sentinels)
        lines = result["__IMPORT_0__"].splitlines()
        assert len(lines) == 2

    def test_semicolon_splits(self):
        """A semicolon-separated import must split into separate lines."""
        sentinels = {"__IMPORT_0__": "import os; import sys"}
        result = standardize_import_blocks(sentinels)
        lines = result["__IMPORT_0__"].splitlines()
        assert len(lines) == 2
        assert "import os" in lines
        assert "import sys" in lines

    def test_star_import_raises(self):
        """A star import must raise ValueError."""
        sentinels = {"__IMPORT_0__": "from os import *"}
        with pytest.raises(ValueError):
            standardize_import_blocks(sentinels)

    def test_single_import_unchanged(self):
        """A single canonical import must pass through unchanged."""
        sentinels = {"__IMPORT_0__": "import os"}
        result = standardize_import_blocks(sentinels)
        assert result["__IMPORT_0__"] == "import os"


class TestValidateSource:
    def test_clean_source_passes(self):
        """Source with no import keywords must pass silently."""
        validate_source("x = 1\ny = 2\n")  # must not raise

    def test_import_in_source_raises(self):
        """Source containing an import keyword must raise ValueError."""
        with pytest.raises(ValueError):
            validate_source("    import os\n")


# ---------------------------------------------------------------------------
# Local import detection
# ---------------------------------------------------------------------------

class TestIsModuleInDirectory:
    """_is_module_in_directory resolves a module name to its on-disk origin and
    checks whether that origin falls under a given directory.

    This is the factual predicate used by resolve_import to detect absolute
    imports that reference local source files rather than installed packages.
    It crashes rather than returning a safe default on unresolvable inputs,
    because staging cannot produce correct output in those cases.
    """

    def test_external_package_not_in_temp_dir(self, tmp_path):
        """An installed external package must not appear to be under a temp directory."""
        assert not _is_module_in_directory("torch", tmp_path)

    def test_module_detected_in_its_own_parent(self):
        """A module must be detected as present when checked against its own parent directory."""
        import importlib.util
        spec = importlib.util.find_spec("torch")
        torch_parent = Path(spec.origin).parent
        assert _is_module_in_directory("torch", torch_parent)

    def test_missing_module_raises(self, tmp_path):
        """A module absent from the environment must raise ValueError."""
        with pytest.raises(ValueError):
            _is_module_in_directory("nonexistent_module_xyz_abc_123", tmp_path)

    def test_namespace_package_raises(self, tmp_path):
        """A namespace package with no concrete origin must raise ValueError."""
        import sys
        ns_dir = tmp_path / "_shram_test_ns_pkg"
        ns_dir.mkdir()
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(ValueError):
                _is_module_in_directory("_shram_test_ns_pkg", tmp_path)
        finally:
            sys.path.pop(0)
            sys.modules.pop("_shram_test_ns_pkg", None)


# ---------------------------------------------------------------------------
# Import resolution — unit tests (external imports only)
# ---------------------------------------------------------------------------

class TestResolveKeyExternal:
    """resolve_key derives the canonical deduplication key for an import statement.

    The key encodes whether an import is relative (rel:<posix-path>) or external
    (abs:<statement>). This distinction drives all downstream deduplication and
    inlining decisions — a wrong key means the wrong file gets inlined, or an
    external import gets silently dropped or duplicated.

    Relative import resolution requires real files on disk and is covered by the
    integration tests. These tests cover the external path only.
    """

    def test_absolute_import_gets_abs_key(self):
        """An absolute import must produce an abs: key."""
        key = resolve_key("import torch", Path("/some/dir"))
        assert key == "abs:import torch"

    def test_from_external_import_gets_abs_key(self):
        """A from-external import must produce an abs: key."""
        key = resolve_key("from transformers import PreTrainedModel", Path("/some/dir"))
        assert key.startswith("abs:")


class TestResolveImportExternal:
    """resolve_import converts a canonical key into the substitution string for a sentinel.

    External imports are collected into _external on first encounter and return
    empty string so they are emitted at the top of the merged file rather than
    inline. Duplicate encounters also return empty string — deduplication is
    handled by the seen set, so _external receives each statement exactly once.
    Relative import duplicates return empty string unchanged.

    Relative import recursion (rel: keys) is covered by the integration tests.
    """

    def test_first_encounter_adds_to_external_and_returns_empty(self):
        """First encounter of an external import must add to _external and return empty string."""
        seen = set()
        external = []
        result = resolve_import("abs:import torch", seen, external)
        assert result == ""
        assert "import torch" in external
        assert "abs:import torch" in seen

    def test_second_encounter_returns_empty(self):
        """Second encounter of an external import must return empty string without re-adding."""
        seen = {"abs:import torch"}
        external = ["import torch"]
        result = resolve_import("abs:import torch", seen, external)
        assert result == ""
        assert external.count("import torch") == 1

    def test_local_absolute_import_raises(self, tmp_path):
        """An absolute import resolving to a file under source_dir must raise ValueError."""
        import importlib.util
        spec = importlib.util.find_spec("torch")
        torch_source_dir = Path(spec.origin).parent
        seen = set()
        external = []
        with pytest.raises(ValueError, match="relative import"):
            resolve_import("abs:import torch", seen, external, source_dir=torch_source_dir)

    def test_relative_duplicate_returns_empty(self):
        """Second encounter of a relative import key must return empty string."""
        seen = {"rel:/some/path.py"}
        result = resolve_import("rel:/some/path.py", seen)
        assert result == ""


# ---------------------------------------------------------------------------
# Integration — synthetic source tree
# ---------------------------------------------------------------------------

class TestStageModel:
    """Integration tests for stage_model() against a synthetic source tree.

    The synthetic tree consists of an entry file with a relative import and a
    shared external import, a helper file also sharing that external import, and
    a non-Python file. This exercises the full pipeline: normalisation, sentinel
    resolution, recursion, deduplication, and file copying. A synthetic tree is
    used rather than the real model so tests are fast, deterministic, and
    independent of model evolution.
    """

    def test_single_python_file(self, synthetic_source):
        """Output must contain exactly one .py file named huggingface.py."""
        src, dest = synthetic_source
        stage_model(src, dest)
        py_files = list(dest.glob("*.py"))
        assert len(py_files) == 1
        assert py_files[0].name == "huggingface.py"

    def test_no_relative_imports(self, synthetic_source):
        """Merged huggingface.py must contain no relative import statements."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        for line in merged.splitlines():
            stripped = line.strip()
            assert not (stripped.startswith("from .") or stripped.startswith("from ..")), (
                f"Relative import found: {line!r}"
            )

    def test_external_import_deduplicated(self, synthetic_source):
        """External import present in both files must appear exactly once uncommented."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        uncommented = [l for l in merged.splitlines() if l.strip() == "import torch"]
        assert len(uncommented) == 1

    def test_docstrings_preserved(self, synthetic_source):
        """Docstrings from all source files must appear in the merged output."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        assert "Entry point docstring." in merged
        assert "Helper module docstring." in merged

    def test_generated_header_present(self, synthetic_source):
        """The generated-file header must appear at the very top of the merged output."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        assert merged.startswith("# This file is auto-generated")

    def test_section_headers_present(self, synthetic_source):
        """A section header must appear before each inlined relative import."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        assert "# Inlined from: helper.py" in merged

    def test_absolute_imports_precede_code(self, synthetic_source):
        """External imports must appear before any class or function definitions."""
        src, dest = synthetic_source
        stage_model(src, dest)
        merged = (dest / "huggingface.py").read_text(encoding="utf-8")
        import_pos = merged.index("import torch")
        class_pos = merged.index("class ")
        assert import_pos < class_pos


# ---------------------------------------------------------------------------
# Real model smoke test
# ---------------------------------------------------------------------------

class TestStageModelReal:
    """Smoke test: stage_model() against the actual model source.

    Verifies that the real model source inlines without error and produces a
    single merged huggingface.py. Does not verify loadability — that is covered
    by the full stage() roundtrip test in test_upload_to_hub.py.
    """

    def test_produces_single_python_file(self):
        """stage_model on the real model must produce exactly one .py file."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            stage_model(MODEL_DIR, dest)
            py_files = list(dest.glob("*.py"))
            assert len(py_files) == 1
            assert py_files[0].name == "huggingface.py"

    def test_no_relative_imports_in_output(self):
        """Merged real model file must contain no relative import statements."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            stage_model(MODEL_DIR, dest)
            merged = (dest / "huggingface.py").read_text(encoding="utf-8")
            for line in merged.splitlines():
                stripped = line.strip()
                assert not (stripped.startswith("from .") or stripped.startswith("from ..")), (
                    f"Relative import found: {line!r}"
                )
