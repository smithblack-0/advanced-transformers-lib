"""Tests for tokenizer.py.

Three test categories:

1. _ensure_fast_tokenizer_class — fully offline. Uses a tmp_path with a
   synthetic tokenizer_config.json. No network, no tokenizer files required.

2. prepare_tokenizer — requires network. Marked @pytest.mark.network.
   Calls prepare_tokenizer into a tmp_path and verifies the result.

3. Model directory state — requires prepare_tokenizer to have been run at
   least once against the real model directory. Skipped automatically if
   tokenizer files are not present; run src/shram/tokenizer.py to populate.

4. Config alignment — verifies ShramConfig default vocab_size matches the
   GPT-NeoX tokenizer. Offline; no tokenizer files required.
"""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.shram.model.configuration import ShramConfig
from src.shram.tokenizer import _ensure_fast_tokenizer_class, prepare_tokenizer

MODEL_DIR = Path("src/shram/model")
EXPECTED_VOCAB_SIZE = 50277


def _tokenizer_files_present() -> bool:
    """True if prepare_tokenizer has been run against the real model directory."""
    return (MODEL_DIR / "tokenizer.json").exists()


# ---------------------------------------------------------------------------
# _ensure_fast_tokenizer_class (offline)
# ---------------------------------------------------------------------------

class TestEnsureFastTokenizerClass:
    def test_slow_class_is_corrected(self, tmp_path):
        """A slow tokenizer_class must be upgraded to the Fast variant."""
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GPT2Tokenizer"})
        )
        _ensure_fast_tokenizer_class(tmp_path)
        result = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert result["tokenizer_class"] == "GPT2TokenizerFast"

    def test_already_fast_class_unchanged(self, tmp_path):
        """A tokenizer_class already ending in Fast must not be modified."""
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GPT2TokenizerFast"})
        )
        _ensure_fast_tokenizer_class(tmp_path)
        result = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert result["tokenizer_class"] == "GPT2TokenizerFast"

    def test_absent_tokenizer_class_left_unchanged(self, tmp_path):
        """A config without tokenizer_class must not be modified."""
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"other_key": "value"})
        )
        _ensure_fast_tokenizer_class(tmp_path)
        result = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert "tokenizer_class" not in result


# ---------------------------------------------------------------------------
# prepare_tokenizer (network required)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestPrepareTokenizer:
    def test_writes_tokenizer_json(self, tmp_path):
        """prepare_tokenizer must write tokenizer.json — the fast tokenizer data file."""
        prepare_tokenizer(tmp_path)
        assert (tmp_path / "tokenizer.json").exists()

    def test_writes_tokenizer_config(self, tmp_path):
        """prepare_tokenizer must write tokenizer_config.json."""
        prepare_tokenizer(tmp_path)
        assert (tmp_path / "tokenizer_config.json").exists()

    def test_tokenizer_config_has_fast_class(self, tmp_path):
        """tokenizer_config.json must name a fast tokenizer class after prepare."""
        prepare_tokenizer(tmp_path)
        config = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert config.get("tokenizer_class", "").endswith("Fast")

    def test_vocab_size(self, tmp_path):
        """The prepared tokenizer must have exactly 50,277 tokens (base vocab + added tokens)."""
        prepare_tokenizer(tmp_path)
        tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
        assert len(tokenizer) == EXPECTED_VOCAB_SIZE

    def test_encode_decode_roundtrip(self, tmp_path):
        """Encode followed by decode must be lossless for standard text."""
        prepare_tokenizer(tmp_path)
        tokenizer = AutoTokenizer.from_pretrained(str(tmp_path))
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text

    def test_idempotent(self, tmp_path):
        """Running prepare_tokenizer twice must produce identical files."""
        prepare_tokenizer(tmp_path)
        first = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
        prepare_tokenizer(tmp_path)
        second = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
        assert first == second


# ---------------------------------------------------------------------------
# Tokenizer in model directory (requires prepare_tokenizer to have been run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _tokenizer_files_present(),
    reason="tokenizer not prepared — run: python src/shram/tokenizer.py",
)
class TestTokenizerInModelDir:
    def test_tokenizer_config_has_fast_class(self):
        """tokenizer_config.json in model dir must name a fast tokenizer class."""
        config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())
        assert config.get("tokenizer_class", "").endswith("Fast")

    def test_vocab_size(self):
        """Tokenizer in model dir must have exactly 50,277 tokens (base vocab + added tokens)."""
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        assert len(tokenizer) == EXPECTED_VOCAB_SIZE

    def test_encode_decode_roundtrip(self):
        """Encode followed by decode must be lossless for standard text."""
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text


# ---------------------------------------------------------------------------
# Config alignment (offline)
# ---------------------------------------------------------------------------

class TestConfigAlignment:
    def test_default_vocab_size_matches_tokenizer(self):
        """ShramConfig default vocab_size must match GPT-NeoX's 50,280.

        A mismatch means the embedding table and logit layer would be sized
        wrong for the tokenizer that ships with the model.
        """
        assert ShramConfig().vocab_size == EXPECTED_VOCAB_SIZE
