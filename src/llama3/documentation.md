# advanced-transformers-lib -- Llama 3 Baseline: Developer Guide

## Using the Published Architecture

Once uploaded, researchers instantiate the architecture with no checkpoint:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull architecture config -- override any parameter at instantiation time
config = AutoConfig.from_pretrained(
    "your-namespace/advanced-transformers-lib",
    trust_remote_code=True,
    num_hidden_layers=16,  # example override
)

# Fresh random weights -- no checkpoint required
model = AutoModelForCausalLM.from_config(config)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-namespace/advanced-transformers-lib")

# Save and reload after training
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

`trust_remote_code=True` is required -- it downloads the architecture source files
from the Hub and imports them locally. This is how HuggingFace loads custom
architectures not built into the `transformers` library.

---

## Uploading to HuggingFace Hub

### Step 1 -- Set the target repository

Edit `REPO_ID` at the top of `src/llama3/upload_to_hub.py`:

```python
REPO_ID = "your-namespace/advanced-transformers-lib"
```

The repository is created automatically on first upload if it does not exist.

### Step 2 -- Authenticate

The upload script requires a HuggingFace account with write access. Two options:

**Option A -- Environment variable (impermanent, preferred for one-off uploads)**

The token is used for this invocation only and is never written to disk:

```bash
# Set for the session
export HF_TOKEN=hf_your_token_here
python -m src.llama3.upload_to_hub
```

Get a write-access token at: https://huggingface.co/settings/tokens

**Option B -- CLI login (persistent)**

Run once to store a token in `~/.cache/huggingface/token`. All subsequent uploads
use it automatically:

```bash
huggingface-cli login
# Paste your token when prompted. Stored locally, never in this repository.
```

Then run the upload normally:

```bash
python -m src.llama3.upload_to_hub
```

To remove the stored token: `huggingface-cli logout`

### Step 3 -- Run the upload

```bash
python -m src.llama3.upload_to_hub
```

The script runs five steps:
1. Refreshes tokenizer files in `src/llama3/model/`
2. Writes `config.json` to `src/llama3/model/`
3. Renders the architecture card (`README.md`) from `src/llama3/model_card.md`
4. Creates the Hub repository if it does not exist
5. Uploads the entire `src/llama3/model/` directory to the Hub repository root

### Step 4 -- Verify

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

All three must succeed. The Hub repository must contain no weight files
(`.bin`, `.safetensors`).

---

## Running the Tests

```bash
python -m pytest tests/
```

Tests marked `@pytest.mark.network` require a network connection to HuggingFace Hub.
Tests in `TestTokenizerInModelDir` are skipped automatically if the tokenizer has
not been prepared locally -- run `python src/llama3/tokenizer.py` first.

---

## Development Notes

This section will be expanded in Unit 9. See `plan.md` for the full implementation
history and decision log.
