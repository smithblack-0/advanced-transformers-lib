# advanced-transformers-lib — Llama 3 Baseline: Developer Guide

## Using the Published Model

Once uploaded to the Hub, researchers instantiate the model with no checkpoint:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull config from Hub — override any parameter at instantiation time
config = AutoConfig.from_pretrained(
    "your-namespace/advanced-transformers-lib",
    trust_remote_code=True,
    num_hidden_layers=16,  # example override
)

# Fresh random weights — no checkpoint required
model = AutoModelForCausalLM.from_config(config)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-namespace/advanced-transformers-lib")

# Save and reload a checkpoint after training
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

`trust_remote_code=True` is required — it downloads the model source files from
the Hub and imports them locally. This is how HuggingFace loads custom
architectures that are not built into the `transformers` library.

---

## Uploading to HuggingFace Hub

### Step 1 — Set the target repository

Edit `REPO_ID` at the top of `src/llama3/upload_to_hub.py`:

```python
REPO_ID = "your-namespace/advanced-transformers-lib"
```

The repository is created automatically on first upload if it does not exist.

### Step 2 — Authenticate

The upload script requires a HuggingFace account with write access. Two options:

**Option A — Environment variable (impermanent)**

Pass your token for this invocation only. The token is never written to disk or
stored anywhere:

```bash
# Inline (token visible in shell history — use Option B or export if that is a concern)
HF_TOKEN=hf_your_token_here python src/llama3/upload_to_hub.py

# Or export it for the session
export HF_TOKEN=hf_your_token_here
python src/llama3/upload_to_hub.py
```

Get your token at: https://huggingface.co/settings/tokens (use a write-access token).

**Option B — CLI login (persistent)**

Run once to store a token in `~/.cache/huggingface/token`. Subsequent uploads
use it automatically with no further setup:

```bash
huggingface-cli login
# Paste your token when prompted. It is stored locally, not in this repository.
```

Then run uploads normally:

```bash
python src/llama3/upload_to_hub.py
```

To remove the stored token: `huggingface-cli logout`

### Step 3 — Run the upload

```bash
python src/llama3/upload_to_hub.py
```

The script runs five steps:
1. Refreshes tokenizer files in `src/llama3/model/`
2. Writes `config.json` to `src/llama3/model/`
3. Generates and writes `README.md` (model card) to `src/llama3/model/`
4. Creates the Hub repository if it does not exist
5. Uploads the entire `src/llama3/model/` directory to the Hub repository root

### Step 4 — Verify

From a fresh Python environment with no local HuggingFace cache:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained(
    "your-namespace/advanced-transformers-lib",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("your-namespace/advanced-transformers-lib")
```

All three must succeed with no errors. The Hub repository must contain no
weight files (`.bin`, `.safetensors`).

---

## Running the Tests

```bash
python -m pytest tests/llama3/
```

Tests that require network access are marked `@pytest.mark.network` and will
run if a connection is available. Tests that require the tokenizer to have been
prepared locally are skipped automatically if the files are absent — run
`python src/llama3/tokenizer.py` first to populate them.

---

## Development Notes

This section will be expanded in the documentation unit. See `plan.md` for the
full implementation history and decision log.
