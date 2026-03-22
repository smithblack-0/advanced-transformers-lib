# advanced-transformers-lib -- Llama 3 Baseline: Developer Guide

## Uploading the Architecture to HuggingFace Hub

### What you need

**Step 1 -- Create the repository on HuggingFace:**
1. Log in at https://huggingface.co
2. Click New Model
3. Set the name to `llama3_baseline` and owner to `smithblack-0`
4. Set visibility as desired and click Create

**Step 2 -- Create a write token scoped to that repository only:**
1. Go to Settings > Access Tokens
2. Click New Token, choose Fine-grained
3. Under Repository permissions, select `smithblack-0/llama3_baseline` only
4. Grant Write access on that repository
5. Copy the token -- you will paste it when the script prompts you

### Running the upload

From the repository root:

```bash
python -m src.llama3.upload_to_hub
```

The script will prompt for your token, then run five steps:
1. Refresh tokenizer files
2. Write `config.json`
3. Render the architecture card (`README.md`) from `model_card.md`
4. Create the Hub repository if it does not exist
5. Upload `src/llama3/model/` to the Hub repository root

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
