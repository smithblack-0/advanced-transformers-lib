"""Publish the Llama 3 architecture and tokenizer to HuggingFace Hub.

Run as a module from the repository root:

    python -m src.llama3.upload_to_hub

The script prompts for a HuggingFace write-access token at runtime. The token
is passed directly to the API and never stored anywhere.

What is uploaded: every file in src/llama3/model/ -- Python source, config.json,
tokenizer files, and README.md (architecture card). This folder is the exact Hub root.

What is never uploaded: weights. No weight files exist in src/llama3/model/,
making accidental upload structurally impossible.
"""

import os
import tempfile
from pathlib import Path

from huggingface_hub import upload_folder

from .model.configuration import Llama3Config
from .stage_for_hub import stage
from .tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPO_ID = "smithblack-0/llama3_baseline"
MODEL_DIR = Path(__file__).parent / "model"
_CARD_TEMPLATE = Path(__file__).parent / "model_card.md"

# -----------------------------------------------------------------------------


def _render_config_table(config: Llama3Config) -> str:
    """Render a markdown table of default configuration parameters.

    Args:
        config: Llama3Config providing the values to tabulate.

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


def _render_card(config: Llama3Config, repo_id: str) -> str:
    """Render the architecture card by filling placeholders in model_card.md.

    Reads the static template, substitutes the repo_id and config defaults table.
    Parameter count is deliberately omitted — this is a configurable architecture,
    not a fixed pretrained model, so no single count is meaningful.

    Args:
        config: Llama3Config providing default parameter values.
        repo_id: Hub repository identifier inserted wherever {repo_id} appears.

    Returns:
        Rendered markdown string ready to write as README.md.
    """
    config_table = _render_config_table(config)
    template = _CARD_TEMPLATE.read_text(encoding="utf-8")
    return template.format(repo_id=repo_id, config_table=config_table)


def upload(repo_id: str = REPO_ID) -> None:
    """Prepare and publish the architecture and tokenizer to the Hub.

    Reads the HuggingFace write token from the LLAMA3_HF_TOKEN environment
    variable, then runs five steps:
    1. Refresh tokenizer files in model/ via prepare_tokenizer()
    2. Write config.json to model/ from Llama3Config defaults
    3. Render and write README.md (architecture card) to model/
    4. Stage model files into a temporary flat directory
    5. Upload the staging directory to the Hub repository root

    If REPO_ID is None, exits immediately with an informative message.
    The repository must already exist on HuggingFace Hub before running.
    See src/llama3/documentation.md for setup instructions.

    Args:
        repo_id: Target Hub repository in 'namespace/name' format, or None to skip.
    """
    if repo_id is None:
        print("REPO_ID is not set. Skipping upload.")
        return

    token = os.environ.get("LLAMA3_HF_TOKEN")
    if token is None:
        token = input("LLAMA3_HF_TOKEN not set. Enter HuggingFace write token: ").strip()
    if not token:
        raise EnvironmentError("No token provided. Upload aborted.")

    print("Step 1/5 -- Refreshing tokenizer...")
    prepare_tokenizer()

    config = Llama3Config()

    print("Step 2/5 -- Writing config.json...")
    config.save_pretrained(MODEL_DIR)

    print("Step 3/5 -- Rendering architecture card...")
    card = _render_card(config, repo_id)
    (MODEL_DIR / "README.md").write_text(card, encoding="utf-8")

    print("Step 4/5 -- Staging model files...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir)
        stage(MODEL_DIR, staging_dir)

        print(f"Step 5/5 -- Uploading to {repo_id}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=staging_dir,
            repo_type="model",
            commit_message="Update architecture and tokenizer",
            token=token,
        )

    print("\nDone. Verify from a fresh environment:")
    print(f"  AutoConfig.from_pretrained('{repo_id}', trust_remote_code=True)")
    print(f"  AutoModelForCausalLM.from_config(config)")
    print(f"  AutoTokenizer.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    upload()
