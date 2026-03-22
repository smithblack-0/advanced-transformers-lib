"""Publish the Llama 3 architecture and tokenizer to HuggingFace Hub.

Run once per release:

    python -m src.llama3.upload_to_hub

Authentication options are documented in src/llama3/documentation.md.

What is uploaded: every file in src/llama3/model/ -- Python source, config.json,
tokenizer files, and README.md (architecture card). This folder is the exact Hub root.

What is never uploaded: weights. No weight files exist in src/llama3/model/,
making accidental upload structurally impossible.
"""

from pathlib import Path

from huggingface_hub import create_repo, upload_folder

from .model.configuration import Llama3Config
from .model.huggingface import Llama3ForCausalLM
from .tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPO_ID = "your-namespace/advanced-transformers-lib"
MODEL_DIR = Path(__file__).parent / "model"
_CARD_TEMPLATE = Path(__file__).parent / "model_card.md"

# -----------------------------------------------------------------------------


def _render_config_table(config: Llama3Config, param_str: str) -> str:
    """Render a markdown table of default configuration parameters.

    Args:
        config: Llama3Config providing the values to tabulate.
        param_str: Pre-formatted parameter count string (e.g. "190.4M").

    Returns:
        Markdown table string, ready for direct insertion into the card.
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
        ("Parameters (default config)", param_str),
    ]
    lines = ["| Parameter | Default |", "|-----------|---------|"]
    for name, value in rows:
        lines.append(f"| `{name}` | {value} |")
    return "\n".join(lines)


def _render_card(config: Llama3Config, repo_id: str) -> str:
    """Render the architecture card by filling placeholders in model_card.md.

    Reads the static template from model_card.md, computes the values that are
    only resolvable at upload time (parameter count, config defaults), and
    substitutes the named placeholders.

    Args:
        config: Llama3Config providing default parameter values.
        repo_id: Hub repository identifier, inserted wherever {repo_id} appears.

    Returns:
        Rendered markdown string ready to write as README.md.
    """
    param_count = sum(p.numel() for p in Llama3ForCausalLM(config).parameters())
    param_str = (
        f"{param_count / 1e6:.1f}M"
        if param_count < 1e9
        else f"{param_count / 1e9:.2f}B"
    )
    config_table = _render_config_table(config, param_str)
    template = _CARD_TEMPLATE.read_text(encoding="utf-8")
    return template.format(repo_id=repo_id, config_table=config_table)


def upload(repo_id: str = REPO_ID) -> None:
    """Prepare and publish the architecture and tokenizer to the Hub.

    Sequence:
    1. Refresh tokenizer files in model/ via prepare_tokenizer()
    2. Write config.json to model/ from Llama3Config defaults
    3. Render and write README.md (architecture card) to model/
    4. Create the Hub repository if it does not already exist
    5. Upload model/ contents to the Hub repository root atomically

    Args:
        repo_id: Target Hub repository in 'namespace/name' format.
            Defaults to the REPO_ID constant at the top of this file.
    """
    print("Step 1/5 -- Refreshing tokenizer...")
    prepare_tokenizer()

    config = Llama3Config()

    print("Step 2/5 -- Writing config.json...")
    config.save_pretrained(MODEL_DIR)

    print("Step 3/5 -- Rendering architecture card...")
    card = _render_card(config, repo_id)
    (MODEL_DIR / "README.md").write_text(card, encoding="utf-8")

    print(f"Step 4/5 -- Creating repository '{repo_id}' (if needed)...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Step 5/5 -- Uploading {MODEL_DIR} to {repo_id}...")
    upload_folder(
        repo_id=repo_id,
        folder_path=MODEL_DIR,
        repo_type="model",
        ignore_patterns=["__pycache__", "*.pyc"],
        commit_message="Update architecture and tokenizer",
    )

    print("\nDone. Verify from a fresh environment:")
    print(f"  AutoConfig.from_pretrained('{repo_id}', trust_remote_code=True)")
    print(f"  AutoModelForCausalLM.from_config(config)")
    print(f"  AutoTokenizer.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    upload()
