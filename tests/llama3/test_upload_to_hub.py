"""Tests for upload_to_hub.py and stage_for_hub.py.

Hub interaction (upload_folder) cannot be unit tested and is verified manually.
These tests cover the independently verifiable functions: config table rendering,
card rendering, the None guard, and the staging interface.
"""

import tempfile
from pathlib import Path

from src.llama3.model.configuration import Llama3Config
from src.llama3.stage_for_hub import stage
from src.llama3.upload_to_hub import _render_card, _render_config_table, upload

MODEL_DIR = Path(__file__).parent.parent.parent / "src" / "llama3" / "model"


def small_config(**kwargs) -> Llama3Config:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=256,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


# ---------------------------------------------------------------------------
# None guard
# ---------------------------------------------------------------------------

class TestNoneGuard:
    def test_upload_none_repo_id_exits_cleanly(self):
        """upload(repo_id=None) must return without error and without attempting any upload."""
        upload(repo_id=None)  # must not raise


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------

class TestStage:
    def test_callable_with_source_and_dest(self):
        """stage must accept (source_dir, dest_dir) and complete without error."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))

    def test_python_files_present(self):
        """Root-level Python files must appear in the staging directory."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            assert (Path(tmp) / "huggingface.py").exists()
            assert (Path(tmp) / "configuration.py").exists()

    def test_no_pycache_in_staging(self):
        """__pycache__ files must not appear in the staging directory."""
        with tempfile.TemporaryDirectory() as tmp:
            stage(MODEL_DIR, Path(tmp))
            staged = list(Path(tmp).iterdir())
            assert not any(p.suffix == ".pyc" for p in staged)
            assert not any(p.name == "__pycache__" for p in staged)


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
