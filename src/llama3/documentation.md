# advanced-transformers-lib -- Llama 3 Baseline: Developer Guide

## Uploading the Architecture to HuggingFace Hub

### Steps

**1. Edit the repository name if needed.**
Open `src/llama3/upload_to_hub.py` and check `REPO_ID` at the top. It is set to
`smithblack-0/llama3_baseline` by default. Change it to match your intended
HuggingFace repository before proceeding.

**2. Create the repository on HuggingFace if it does not already exist.**
1. Log in at https://huggingface.co
2. Click New Model
3. Name it to match the `REPO_ID` you set above
4. Set visibility as desired and click Create

**3. Create a write-access token scoped to that repository only.**
1. Go to Settings > Access Tokens
2. Click New Token, choose Fine-grained
3. Under Repository permissions, add your repository and grant Write access
4. Copy the token

**4. Run the upload.**

```bash
python -m src.llama3.upload_to_hub
```

Paste the token when prompted. The script will:
1. Refresh tokenizer files
2. Write `config.json`
3. Render the architecture card (`README.md`) from `model_card.md`
4. Upload `src/llama3/model/` to the Hub repository root

**5. Delete the token (recommended).**
Go back to Settings > Access Tokens and delete the token you just used.

### Verifying the upload

From a fresh Python environment with no local HuggingFace cache:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained(
    "smithblack-0/llama3_baseline",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("smithblack-0/llama3_baseline")
```

All three must succeed. The Hub repository must contain no weight files.

---

## Using the Published Architecture

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained(
    "smithblack-0/llama3_baseline",
    trust_remote_code=True,
    num_hidden_layers=16,  # override any parameter
)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("smithblack-0/llama3_baseline")
```

`trust_remote_code=True` is required -- it downloads and imports the architecture
source files from the Hub.

---

## Running the Tests

```bash
python -m pytest tests/
```

Tests marked `@pytest.mark.network` require a network connection. Tests in
`TestTokenizerInModelDir` are skipped if the tokenizer has not been prepared
locally -- run `python src/llama3/tokenizer.py` first.

---

## Development Notes

See `plan.md` for the full implementation history and decision log.
