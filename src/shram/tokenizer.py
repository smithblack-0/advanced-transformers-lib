"""Prepare the GPT-NeoX tokenizer for Hub distribution.

Run this script before upload_to_hub.py to ensure the tokenizer files are
present and correct in src/shram/model/. upload_to_hub.py will then include
them in the Hub upload so that AutoTokenizer.from_pretrained works for
researchers pulling the repository.

    python src/shram/tokenizer.py

Known issue handled here: AutoTokenizer.save_pretrained can write the slow
tokenizer class name to tokenizer_config.json even when the fast tokenizer
files are present. AutoTokenizer uses that field to decide which implementation
to load, so this script corrects it after saving.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

# --- Configuration -----------------------------------------------------------

SOURCE_REPO = "EleutherAI/gpt-neox-20b"
MODEL_DIR = Path(__file__).parent / "model"

# -----------------------------------------------------------------------------


def prepare_tokenizer(dest: Path = MODEL_DIR) -> None:
    """Download GPT-NeoX tokenizer and write files to dest.

    Fetches the tokenizer from HuggingFace Hub, saves all tokenizer files
    to dest, and corrects tokenizer_config.json to ensure AutoTokenizer
    loads the fast implementation.

    Idempotent: safe to run multiple times. Existing files are overwritten
    with the current upstream version.

    Args:
        dest: Directory to write tokenizer files into. Defaults to
            src/shram/model/ — the Hub distribution folder.
    """
    print(f"Fetching tokenizer from {SOURCE_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_REPO)

    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(dest)
    print(f"Tokenizer files written to {dest}")

    _ensure_fast_tokenizer_class(dest)
    print("Done.")


def _ensure_fast_tokenizer_class(model_dir: Path) -> None:
    """Correct tokenizer_class in tokenizer_config.json to the fast variant.

    save_pretrained can write the slow class name (e.g. "GPT2Tokenizer") even
    when the fast tokenizer files are present. AutoTokenizer uses tokenizer_class
    to decide which implementation to load, so it must name the fast class
    (e.g. "GPT2TokenizerFast"). If the field is already fast or absent, this
    function does nothing.
    """
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    tokenizer_class = config.get("tokenizer_class", "")
    if tokenizer_class and not tokenizer_class.endswith("Fast"):
        fast_class = tokenizer_class + "Fast"
        config["tokenizer_class"] = fast_class
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Corrected tokenizer_class: {tokenizer_class} -> {fast_class}")


if __name__ == "__main__":
    prepare_tokenizer()
