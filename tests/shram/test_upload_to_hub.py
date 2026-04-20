"""Tests for upload_to_hub.py and stage_for_hub.py.

Hub interaction (upload_folder) cannot be unit tested and is verified manually.
These tests cover the independently verifiable functions: config table rendering,
card rendering, and the staging system.
"""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest

from src.shram.model.configuration import ShramConfig
from src.shram.stage_for_hub import _rewrite_imports, compute_flat_name, stage
from src.shram.upload_to_hub import _render_card, _render_config_table, upload

MODEL_DIR = Path(__file__).parent.parent.parent / "src" / "shram" / "model"


def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        embedding_width=64,
        intermediate_size=128,
        num_decoder_layers=2,
        num_sliding_window_heads=4,
        num_mosrah_heads=4,
        num_selected_heads=4,
        head_dim=16,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


# ---------------------------------------------------------------------------
# None guard
# ---------------------------------------------------------------------------

class TestNoneGuard:
    def test_upload_none_repo_id_exits_cleanly(self):
        """upload(repo_id=None) must return without error and without attempting any upload."""
        upload(repo_id=None)  # must not raise


# ---------------------------------------------------------------------------
# _render_config_table
# ---------------------------------------------------------------------------

class TestRenderConfigTable:
    def test_contains_header_row(self):
        """Table must include a markdown header row."""
        table = _render_config_table(small_config())
        assert "| Parameter | Default |" in table

    def test_contains_separator_row(self):
        """Table must include a markdown separator row."""
        table = _render_config_table(small_config())
        assert "|-----------|---------|" in table

    def test_config_values_present(self):
        """Each config parameter value must appear in the rendered table."""
        config = small_config()
        table = _render_config_table(config)
        assert str(config.hidden_size) in table
        assert str(config.num_hidden_layers) in table
        assert str(config.vocab_size) in table


# ---------------------------------------------------------------------------
# _render_card
# ---------------------------------------------------------------------------

class TestRenderCard:
    def test_repo_id_substituted(self):
        """The repo_id placeholder must be replaced throughout the card."""
        card = _render_card(small_config(), "test-user/test-repo")
        assert "test-user/test-repo" in card
        assert "{repo_id}" not in card

    def test_config_table_substituted(self):
        """The config_table placeholder must be replaced with actual content."""
        card = _render_card(small_config(), "test-user/test-repo")
        assert "{config_table}" not in card
        assert "| Parameter | Default |" in card

    def test_trust_remote_code_warning_present(self):
        """The trust_remote_code warning must be prominent in the card."""
        card = _render_card(small_config(), "test-user/test-repo")
        assert "trust_remote_code" in card

    def test_returns_nonempty_string(self):
        """Rendered card must be a non-empty string."""
        card = _render_card(small_config(), "test-user/test-repo")
        assert isinstance(card, str) and len(card) > 0


# ---------------------------------------------------------------------------
# compute_flat_name
# ---------------------------------------------------------------------------

class TestComputeFlatName:
    def test_root_file_unchanged(self):
        """Root-level files keep their stem."""
        assert compute_flat_name(Path("huggingface.py")) == "huggingface"

    def test_subfolder_file_prefixed(self):
        """Subdirectory files get double-underscore prefix."""
        assert compute_flat_name(Path("cache/shram_cache.py")) == "__cache__shram_cache"
        assert compute_flat_name(Path("attention/mosrah.py")) == "__attention__mosrah"

    def test_subfolder_init_dropped(self):
        """__init__.py inside a subdirectory is dropped (returns None)."""
        assert compute_flat_name(Path("cache/__init__.py")) is None
        assert compute_flat_name(Path("attention/__init__.py")) is None

    def test_root_init_kept(self):
        """__init__.py at the root is kept."""
        assert compute_flat_name(Path("__init__.py")) == "__init__"


# ---------------------------------------------------------------------------
# _rewrite_imports
# ---------------------------------------------------------------------------

class TestRewriteImports:
    def test_cache_import_rewritten(self):
        """Relative imports from .cache.x must be rewritten to .__cache__x."""
        source = "from .cache.shram_cache import ShramCache\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("huggingface.py"))
        assert "from .__cache__shram_cache import ShramCache" in result

    def test_attention_import_rewritten(self):
        """Relative imports from .attention.x must be rewritten to .__attention__x."""
        source = "from .attention.mosrah import MoSRAHLayer\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("huggingface.py"))
        assert "from .__attention__mosrah import MoSRAHLayer" in result

    def test_non_subfolder_import_unchanged(self):
        """Relative imports not crossing a subfolder are left unchanged."""
        source = "from .rope import RotaryEmbedding\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("huggingface.py"))
        assert "from .rope import RotaryEmbedding" in result

    def test_absolute_external_import_unchanged(self):
        """Absolute imports from external packages are never modified."""
        source = "from transformers import PreTrainedModel\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("huggingface.py"))
        assert "from transformers import PreTrainedModel" in result

    def test_source_otherwise_preserved(self):
        """Rewriting must not alter lines that do not contain matched imports."""
        source = "x = 1\nfrom .cache.shram_cache import ShramCache\ny = 2\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("huggingface.py"))
        assert "x = 1" in result
        assert "y = 2" in result

    def test_absolute_local_import_rewritten(self):
        """Absolute imports starting with src.shram.model must be rewritten."""
        source = "from src.shram.model.attention.router import MoSRAHRouter\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("attention/mosrah.py"))
        assert "from .__attention__router import MoSRAHRouter" in result

    def test_absolute_cross_subfolder_import_rewritten(self):
        """Absolute imports crossing subfolders must be rewritten correctly."""
        source = "from src.shram.model.cache.mosrah_cache import MoSRAHCache\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("attention/mosrah.py"))
        assert "from .__cache__mosrah_cache import MoSRAHCache" in result

    def test_absolute_root_import_rewritten(self):
        """Absolute imports of root-level files must be rewritten without prefix."""
        source = "from src.shram.model.configuration import ShramConfig\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("attention/router.py"))
        assert "from .configuration import ShramConfig" in result

    def test_sibling_import_in_subfolder_rewritten(self):
        """Relative sibling imports within a subfolder must gain the subfolder prefix."""
        source = "from .mosrah_cache import MoSRAHCache\n"
        result = _rewrite_imports(source, MODEL_DIR, Path("cache/shram_layer_cache.py"))
        assert "from .__cache__mosrah_cache import MoSRAHCache" in result


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------

class TestStage:
    def test_root_files_present(self):
        """Root-level Python files must appear unchanged in the staging directory."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            assert (Path(tmp) / "huggingface.py").exists()
            assert (Path(tmp) / "configuration.py").exists()

    def test_subfolder_files_flattened(self):
        """Subdirectory files must appear with double-underscore prefix at root."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            assert (Path(tmp) / "__cache__shram_cache.py").exists()
            assert (Path(tmp) / "__attention__mosrah.py").exists()

    def test_no_subdirectories_in_staging(self):
        """Staging directory must contain no subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            subdirs = [p for p in Path(tmp).iterdir() if p.is_dir()]
            assert subdirs == []

    def test_subdirectory_init_files_dropped(self):
        """__init__.py files from subdirectories must not appear in staging."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            staged_files = [p.name for p in Path(tmp).iterdir()]
            init_files = [f for f in staged_files if f == "__init__.py"]
            assert len(init_files) <= 1

    def test_staged_files_importable(self):
        """Staged huggingface module must be importable and expose ShramForCausalLM.

        Staged files use relative imports, so they must be loaded as a package.
        We create a temporary parent directory, stage into a named subpackage
        inside it, and import from there so relative imports resolve correctly.
        """
        pkg_name = "_shram_staging_test"
        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp)
            pkg_dir = parent_dir / pkg_name
            pkg_dir.mkdir()
            stage(MODEL_DIR, pkg_dir)
            sys.path.insert(0, str(parent_dir))
            try:
                stale = [k for k in sys.modules if k == pkg_name or k.startswith(pkg_name + ".")]
                for k in stale:
                    del sys.modules[k]
                mod = importlib.import_module(f"{pkg_name}.huggingface")
                assert hasattr(mod, "ShramForCausalLM")
            finally:
                sys.path.remove(str(parent_dir))
