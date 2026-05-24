"""Tests for upload_to_hub.py.

Hub interaction (upload_folder) cannot be unit tested and is verified manually.
These tests cover the independently verifiable functions: config table rendering,
card rendering, and the full stage() orchestrator including the save/load
roundtrip that verifies a staged directory is loadable end-to-end.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.shram.model.configuration import ShramConfig
from src.shram.upload_to_hub import _render_card, _render_config_table, stage, upload

MODEL_DIR = Path(__file__).parent.parent.parent / "src" / "shram" / "model"

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        embedding_width=64,
        mlp_width=128,
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
        assert str(config.embedding_width) in table
        assert str(config.num_decoder_layers) in table
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
# Real model integration
# ---------------------------------------------------------------------------

class TestStageReal:
    """Integration test: stage() against the actual model, verify save/load roundtrip.

    Exercises the full stage() orchestrator — config write, model inlining, file
    copy — then verifies the staged directory is loadable end-to-end via
    HuggingFace's trust_remote_code path. Saves and reloads to confirm parameter
    identity. This catches cases where staging appears to succeed but the output
    is not actually usable.
    """

    def test_staged_model_save_load_roundtrip(self):
        """A model loaded from the staged directory must survive a save/load roundtrip."""
        with tempfile.TemporaryDirectory() as staging_tmp, \
             tempfile.TemporaryDirectory() as save_tmp:
            staging_dir = Path(staging_tmp)
            save_dir = Path(save_tmp)

            stage(MODEL_DIR, staging_dir, repo_id="smithblack-0/SHRAM")

            config = AutoConfig.from_pretrained(staging_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.save_pretrained(save_dir)

            loaded = AutoModelForCausalLM.from_pretrained(
                save_dir,
                trust_remote_code=True,
                local_files_only=True,
            )

            for (name, p1), (_, p2) in zip(
                model.named_parameters(),
                loaded.named_parameters(),
            ):
                torch.testing.assert_close(p1, p2, msg=f"Mismatch in {name}")
