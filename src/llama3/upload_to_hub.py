"""Publish the Llama 3 baseline architecture and tokenizer to HuggingFace Hub.

Run once per release to make the architecture available to researchers:

    python src/llama3/upload_to_hub.py

Authentication — two options (see src/llama3/documentation.md for details):
  - HF_TOKEN environment variable (impermanent, preferred for one-off uploads)
  - huggingface-cli login (stores a token persistently in ~/.cache/huggingface/)

What is uploaded: every file in src/llama3/model/ — Python source, config.json,
tokenizer files, and README.md (model card). This folder is the exact Hub root.

What is never uploaded: model weights. No weight files exist in src/llama3/model/,
making accidental upload structurally impossible.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path when this script is run directly.
# resolve() handles invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from huggingface_hub import create_repo, upload_folder

from src.llama3.model.configuration import Llama3Config
from src.llama3.model.huggingface import Llama3ForCausalLM
from src.llama3.tokenizer import prepare_tokenizer

# --- Configuration -----------------------------------------------------------

REPO_ID = "your-namespace/advanced-transformers-lib"
MODEL_DIR = Path(__file__).parent / "model"

# --- Model card data ---------------------------------------------------------
# Text blocks are stored here as data so that _generate_model_card remains a
# renderer and not a content source. This keeps large descriptions editable
# without touching rendering logic.

_USAGE_SNIPPET = """\
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull config from Hub and override parameters as needed
config = AutoConfig.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
    num_hidden_layers=16,  # example override
)

# Instantiate with fresh random weights — no checkpoint required
model = AutoModelForCausalLM.from_config(config)

# Load tokenizer from Hub
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Save and reload a checkpoint
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```"""

_ARCHITECTURE_NOTES = """\
- **Positional encoding:** Rotary Position Embeddings (RoPE), configurable theta (default 500,000)
- **Attention:** Grouped Query Attention (GQA) — configurable KV heads
- **Activation:** SwiGLU (SiLU gate x up-projection, no bias)
- **Normalisation:** RMSNorm pre-norm throughout, including final backbone norm
- **Tokenizer:** GPT-NeoX (`EleutherAI/gpt-neox-20b`), 50,277-token vocabulary, Apache 2.0"""

# -----------------------------------------------------------------------------


def _generate_model_card(config: Llama3Config, repo_id: str) -> str:
    """Render a model card populated with live architectural details.

    Computes parameter count by instantiating the model so the card reflects
    whatever defaults are current at upload time. Text descriptions come from
    module-level data so this function remains a renderer only.

    Args:
        config: Llama3Config providing the parameter values to document.
        repo_id: Hub repository identifier, inserted into the usage snippet.

    Returns:
        Markdown string ready to write as README.md.
    """
    usage = _USAGE_SNIPPET.format(repo_id=repo_id)

    # Instantiate once at upload time to count parameters accurately.
    param_count = sum(p.numel() for p in Llama3ForCausalLM(config).parameters())
    param_str = (
        f"{param_count / 1e6:.1f}M"
        if param_count < 1e9
        else f"{param_count / 1e9:.2f}B"
    )

    return f"""\
# advanced-transformers-lib -- Llama 3 Baseline

A Llama 3-style decoder-only transformer for research. Pull the architecture
from the Hub and instantiate a freshly initialised model from config -- no
pretrained weights involved. Override any parameter at instantiation time.

> **Important:** `trust_remote_code=True` is required. This downloads the model
> source files from the Hub and executes them in your Python process. Review
> the source at [{repo_id}](https://huggingface.co/{repo_id}) before use.

## Usage

{usage}

## Architecture

{_ARCHITECTURE_NOTES}

### Default configuration

| Parameter | Default |
|-----------|---------|
| `vocab_size` | {config.vocab_size} |
| `hidden_size` | {config.hidden_size} |
| `intermediate_size` | {config.intermediate_size} |
| `num_hidden_layers` | {config.num_hidden_layers} |
| `num_attention_heads` | {config.num_attention_heads} |
| `num_key_value_heads` | {config.num_key_value_heads} |
| `head_dim` | {config.head_dim} |
| `max_position_embeddings` | {config.max_position_embeddings} |
| `rope_theta` | {config.rope_theta} |
| Parameters at default config | {param_str} |

## License

MIT. The implementation is a clean-room synthesis: the human author has not
read the Llama source code. All architectural decisions derive from the
published paper. The tokenizer is GPT-NeoX (Apache 2.0).
"""


def upload(repo_id: str = REPO_ID) -> None:
    """Prepare and publish the architecture and tokenizer to the Hub.

    Sequence:
    1. Refresh tokenizer files in model/ via prepare_tokenizer()
    2. Write config.json to model/ from Llama3Config defaults
    3. Generate and write README.md (model card) to model/
    4. Create the Hub repository if it does not already exist
    5. Upload model/ contents to the Hub repository root atomically

    Args:
        repo_id: Target Hub repository in 'namespace/name' format.
            Defaults to the REPO_ID constant at the top of this file.
    """
    print("Step 1/5 -- Refreshing tokenizer files...")
    prepare_tokenizer()

    config = Llama3Config()

    print("Step 2/5 -- Writing config.json...")
    config.save_pretrained(MODEL_DIR)

    print("Step 3/5 -- Generating model card...")
    readme = _generate_model_card(config, repo_id)
    (MODEL_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Step 4/5 -- Creating repository '{repo_id}' (if needed)...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Step 5/5 -- Uploading {MODEL_DIR} to {repo_id}...")
    upload_folder(
        repo_id=repo_id,
        folder_path=MODEL_DIR,
        repo_type="model",
        ignore_patterns=["__pycache__", "*.pyc"],
        commit_message="Update architecture, tokenizer, and model card",
    )

    print("\nUpload complete. Verify the invariants from a fresh environment:")
    print(f"  config = AutoConfig.from_pretrained('{repo_id}', trust_remote_code=True)")
    print(f"  model  = AutoModelForCausalLM.from_config(config)")
    print(f"  tok    = AutoTokenizer.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    upload()
