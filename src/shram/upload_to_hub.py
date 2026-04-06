"""Publish the Shram architecture and tokenizer to HuggingFace Hub.

Run as a module from the repository root:

    python -m src.shram.upload_to_hub

The script prompts for a HuggingFace write-access token at runtime. The token
is passed directly to the API and never stored anywhere.

What is uploaded: every file in src/shram/model/ -- Python source, config.json,
tokenizer files, and README.md (architecture card). This folder is the exact Hub root.

What is never uploaded: weights. No weight files exist in src/shram/model/,
making accidental upload structurally impossible.
"""

from pathlib import Path

from huggingface_hub import upload_folder

from src.shram.model.configuration import ShramConfig
from src.shram.tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPO_ID = "smithblack-0/shram"
MODEL_DIR = Path(__file__).parent / "model"
_CARD_TEMPLATE = Path(__file__).parent / "model_card.md"

# -----------------------------------------------------------------------------


def _render_config_table(config: ShramConfig) -> str:
    """Render a markdown table of default configuration parameters.

    Args:
        config: ShramConfig providing the values to tabulate.

    Returns:
        Markdown table string ready for insertion into the architecture card.
    """
    rows = [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("rope_theta", config.rope_theta),
    ]
    lines = ["| Parameter | Default |", "|-----------|---------|"]
    for name, value in rows:
        lines.append(f"| `{name}` | {value} |")
    return "\n".join(lines)


def _render_card(config: ShramConfig, repo_id: str) -> str:
    """Render the architecture card by filling placeholders in model_card.md.

    Reads the static template, substitutes the repo_id and config defaults table.
    Parameter count is deliberately omitted — this is a configurable architecture,
    not a fixed pretrained model, so no single count is meaningful.

    Args:
        config: ShramConfig providing default parameter values.
        repo_id: Hub repository identifier inserted wherever {repo_id} appears.

    Returns:
        Rendered markdown string ready to write as README.md.
    """
    config_table = _render_config_table(config)
    template = _CARD_TEMPLATE.read_text(encoding="utf-8")
    return template.format(repo_id=repo_id, config_table=config_table)


def upload(repo_id: str = REPO_ID) -> None:
    """Prepare and publish the architecture and tokenizer to the Hub.

    Prompts for a HuggingFace write-access token scoped to this repository,
    then runs four steps:
    1. Refresh tokenizer files in model/ via prepare_tokenizer()
    2. Write config.json to model/ from ShramConfig defaults
    3. Render and write README.md (architecture card) to model/
    4. Upload model/ contents to the Hub repository root atomically

    The repository must already exist on HuggingFace Hub before running.
    See src/shram/documentation.md for setup instructions.

    Args:
        repo_id: Target Hub repository in 'namespace/name' format.
    """
    token = input("HuggingFace write token: ").strip()

    print("Step 1/4 -- Refreshing tokenizer...")
    prepare_tokenizer()

    config = ShramConfig()

    print("Step 2/4 -- Writing config.json...")
    config.save_pretrained(MODEL_DIR)

    print("Step 3/4 -- Rendering architecture card...")
    card = _render_card(config, repo_id)
    (MODEL_DIR / "README.md").write_text(card, encoding="utf-8")

    print(f"Step 4/4 -- Uploading {MODEL_DIR} to {repo_id}...")
    upload_folder(
        repo_id=repo_id,
        folder_path=MODEL_DIR,
        repo_type="model",
        ignore_patterns=["__pycache__", "*.pyc"],
        commit_message="Update architecture and tokenizer",
        token=token,
    )

    print("\nDone. Verify from a fresh environment:")
    print(f"  AutoConfig.from_pretrained('{repo_id}', trust_remote_code=True)")
    print(f"  AutoModelForCausalLM.from_config(config)")
    print(f"  AutoTokenizer.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    upload()
