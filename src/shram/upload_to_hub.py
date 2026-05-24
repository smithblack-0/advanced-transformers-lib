"""Publish the Shram architecture and tokenizer to HuggingFace Hub.

Invoked by the GitHub Actions release pipeline. Requires a target environment
argument: "dev" uploads to the staging repository, "main" uploads to production.

    python -m shram.upload_to_hub dev
    python -m shram.upload_to_hub main

The script reads a HuggingFace write-access token from the SHRAM_HF_TOKEN
environment variable. Tokens are stored as GitHub secrets and are only available
within the release pipeline — manual invocation is not supported.

What is uploaded: every file in src/shram/model/ -- Python source, config.json,
tokenizer files, and README.md (architecture card). Files are placed at the root
of the Hub repository's default branch.

What is never uploaded: weights. No weight files exist in src/shram/model/,
making accidental upload structurally impossible.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import upload_folder

from .model.configuration import ShramConfig
from .stage_for_hub import stage_model
from .tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPOS = {
    "dev": "smithblack-0/SHRAM-dev",
    "main": "smithblack-0/SHRAM",
}
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


def stage(
    source_dir: Path,
    dest_dir: Path,
    repo_id: str,
    config: ShramConfig = ShramConfig(),
) -> None:
    """Prepare a complete Hub-ready staging directory from a model source directory.

    Refreshes tokenizer files, writes config.json and README.md into dest_dir,
    then inlines all model Python code into a single dest_dir/huggingface.py
    and copies all non-Python files alongside it. The result is a directory
    from which AutoModelForCausalLM.from_pretrained with local_files_only=True
    succeeds.

    Args:
        source_dir: Source model directory (e.g. src/shram/model/).
        dest_dir: Destination staging directory; must already exist.
        repo_id: Hub repository identifier, used to render the architecture card.
        config: ShramConfig providing defaults; overridable for testing.
    """
    print("Step 1/3 -- Refreshing tokenizer...")
    prepare_tokenizer(dest_dir)

    print("Step 2/3 -- Writing config.json and README.md...")
    config.save_pretrained(dest_dir)
    card = _render_card(config, repo_id)
    (dest_dir / "README.md").write_text(card, encoding="utf-8")

    print("Step 3/3 -- Inlining model source and copying configuration...")
    shutil.copy2(source_dir / "configuration.py", dest_dir / "configuration.py")
    stage_model(source_dir, dest_dir)


def upload(repo_id: str) -> None:
    """Prepare and publish the architecture and tokenizer to the Hub.

    Reads the HuggingFace write token from the SHRAM_HF_TOKEN environment
    variable, stages the model into a temporary directory, then uploads it
    to the Hub repository root.

    The repository must already exist on HuggingFace Hub before running.
    See src/shram/documentation.md for the release pipeline instructions.

    Args:
        repo_id: Target Hub repository in 'namespace/name' format.
    """
    if repo_id is None:
        print("REPO_ID is not set. Skipping upload.")
        return

    token = os.environ.get("SHRAM_HF_TOKEN")
    if token is None:
        raise EnvironmentError("SHRAM_HF_TOKEN environment variable is not set.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir)
        stage(MODEL_DIR, staging_dir, repo_id)

        print(f"Uploading to {repo_id}...")
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
    if len(sys.argv) != 2 or sys.argv[1] not in REPOS:
        print(f"Usage: python -m shram.upload_to_hub [dev|main]", file=sys.stderr)
        sys.exit(1)
    upload(REPOS[sys.argv[1]])
