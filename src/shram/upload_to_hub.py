"""Publish the Shram architecture and tokenizer to HuggingFace Hub.

Run as a module from the repository root:

    python -m src.shram.upload_to_hub

The script prompts for a HuggingFace write-access token at runtime. The token
is passed directly to the API and never stored anywhere.

What is uploaded: every file in src/shram/model/ -- Python source, config.json,
tokenizer files, and README.md (architecture card). Files are placed at the root
of the Hub repository's default branch.

What is never uploaded: weights. No weight files exist in src/shram/model/,
making accidental upload structurally impossible.
"""

import os
import tempfile
from pathlib import Path

from huggingface_hub import upload_folder

from .model.configuration import ShramConfig
from .stage_for_hub import stage
from .tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPO_ID = "smithblack-0/SHRAM"
MODEL_DIR = Path(__file__).parent / "model"
_CARD_TEMPLATE = Path(__file__).parent / "model_card.md"

# Keys produced by HuggingFace's PretrainedConfig base that do not belong in
# the researcher-facing config parameter table.
_HF_INTERNAL_KEYS = frozenset({
    "transformers_version", "model_type", "auto_map",
    "is_encoder_decoder", "is_decoder", "add_cross_attention",
    "cross_attention_hidden_size", "tie_encoder_to_decoder",
    "pruned_heads", "chunk_size_feed_forward",
    "output_attentions", "return_dict",
    "architectures", "task_specific_params", "tokenizer_class",
    "prefix", "finetuning_task", "problem_type",
    "id2label", "label2id", "torch_dtype",
})

# -----------------------------------------------------------------------------


def _render_config_table(config: ShramConfig) -> str:
    """Render a markdown table of default configuration parameters.

    Derives rows dynamically from the config's serialised state, excluding
    HuggingFace-internal bookkeeping keys that are not meaningful to researchers.

    Args:
        config: ShramConfig providing the values to tabulate.

    Returns:
        Markdown table string ready for insertion into the architecture card.
    """
    rows = sorted(
        (k, v)
        for k, v in config.to_dict().items()
        if k not in _HF_INTERNAL_KEYS and not k.startswith("_")
    )
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

    Reads the HuggingFace write token from the SHRAM_HF_TOKEN environment
    variable, then runs five steps:
    1. Refresh tokenizer files in model/ via prepare_tokenizer()
    2. Write config.json to model/ from ShramConfig defaults
    3. Render and write README.md (architecture card) to model/
    4. Stage model files into a temporary flat directory
    5. Upload the staging directory to the Hub repository root

    If REPO_ID is None, exits immediately with an informative message.
    The repository must already exist on HuggingFace Hub before running.
    See src/shram/documentation.md for setup instructions.

    Load from Hub after uploading:
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)

    Args:
        repo_id: Target Hub repository in 'namespace/name' format, or None to skip.
    """
    if repo_id is None:
        print("REPO_ID is not set. Skipping upload.")
        return

    token = os.environ.get("SHRAM_HF_TOKEN")
    if token is None:
        raise EnvironmentError("SHRAM_HF_TOKEN environment variable is not set.")

    print("Step 1/4 -- Refreshing tokenizer...")
    prepare_tokenizer()

    config = ShramConfig()

    print("Step 2/4 -- Writing config.json...")
    config.save_pretrained(MODEL_DIR)

    print("Step 3/4 -- Rendering architecture card...")
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

    print("\nDone. Load from a fresh environment:")
    print(f"  config = AutoConfig.from_pretrained('{repo_id}', trust_remote_code=True)")
    print(f"  model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    upload()
