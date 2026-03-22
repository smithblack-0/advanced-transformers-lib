"""Tests for upload_to_hub.py.

Hub interaction (create_repo, upload_folder) cannot be unit tested and is
verified manually per the strategy in plan.md. These tests cover the
independently verifiable functions: config table rendering and card rendering.
"""

from src.llama3.model.configuration import Llama3Config
from src.llama3.upload_to_hub import _render_card, _render_config_table


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
# _render_config_table
# ---------------------------------------------------------------------------

class TestRenderConfigTable:
    def test_contains_header_row(self):
        """Table must include a markdown header row."""
        table = _render_config_table(small_config(), "1.0M")
        assert "| Parameter | Default |" in table

    def test_contains_separator_row(self):
        """Table must include a markdown separator row."""
        table = _render_config_table(small_config(), "1.0M")
        assert "|-----------|---------|" in table

    def test_config_values_present(self):
        """Each config parameter value must appear in the rendered table."""
        config = small_config()
        table = _render_config_table(config, "1.0M")
        assert str(config.hidden_size) in table
        assert str(config.num_hidden_layers) in table
        assert str(config.vocab_size) in table

    def test_param_str_present(self):
        """The pre-formatted parameter count string must appear in the table."""
        table = _render_config_table(small_config(), "42.7M")
        assert "42.7M" in table


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
