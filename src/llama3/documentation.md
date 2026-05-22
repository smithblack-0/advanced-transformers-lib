# advanced-transformers-lib -- Llama 3 Baseline: Developer Guide

## Automated Release Pipeline

The preferred way to publish is through the GitHub Actions release pipeline. It enforces the invariant that the production Hub repository is never written unless E2E tests have passed against a staging repository first.

**How to trigger a release:**

1. On GitHub, go to Releases and create a new release against `master`.
2. Set it as a **Pre-release** (check the "This is a pre-release" checkbox).
3. Publish it immediately — do not save as draft.

Publishing a prerelease triggers the pipeline:
1. Integration tests run against the current code.
2. If tests pass, the architecture is uploaded to the staging Hub repository (`smithblack-0/llama3_baseline_dev`).
3. E2E network tests run against the staging repository.
4. If E2E tests pass, the architecture is uploaded to the production Hub repository (`smithblack-0/llama3_baseline`).
5. The prerelease is automatically promoted to a full release.

The prerelease state is intentional and honest — it signals that the pipeline is in progress. Promotion to a full release is the signal that everything passed.

**If the pipeline fails:** the prerelease remains in prerelease state. Inspect the failing job in the Actions tab, fix the issue, delete the failed release, and create a new prerelease to re-run.

---

## Uploading the Architecture to HuggingFace Hub (manual)

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

### Repository structure and the WYSIWYG design

`src/llama3/model/` is the Hub distribution unit. Its contents are uploaded directly to the Hub
repository root by `upload_to_hub.py` using `upload_folder` — no manifest, no copying, no
transformation. What is in that directory is exactly what researchers receive. This means:

- Source files that belong on the Hub must live inside `model/`.
- Files outside `model/` (`tokenizer.py`, `upload_to_hub.py`, etc.) are local operational tools
  and are never uploaded.

### Relative imports inside model/

All Python files inside `model/` must use relative imports (e.g. `from .configuration import
Llama3Config`), not absolute imports (e.g. `from src.llama3.model.configuration import ...`).

When a researcher calls `AutoConfig.from_pretrained(..., trust_remote_code=True)`, HuggingFace
downloads the `model/` contents into a local cache directory and adds that directory to `sys.path`.
In that context there is no `src` or `llama3` package — absolute imports break immediately.
Relative imports work because `model/` contains `__init__.py` and Python resolves them within
the package.

Files outside `model/` are never uploaded and may use absolute imports.

### Implementation history

See `plan.md` in this directory for the full implementation history, architectural decisions,
and the human-supervised process record.

---

## Design Decisions and Deviations

### Tokenizer

The specification requires the Llama 3 tokenizer. The GPT-NeoX tokenizer
(`EleutherAI/gpt-neox-20b`, Apache 2.0) was substituted because the Llama 3 tokenizer
is gated behind a license agreement, making it unsuitable for a freely distributable
research baseline. Researchers who require exact Llama 3 tokenisation must substitute
it themselves.

### Padding and sequence packing

Right-padding is the only supported training configuration. This model does not accept
an `attention_mask` parameter — passing one raises `ValueError`. Left-padded batches
or packed sequences without explicit position management will produce silently wrong
representations because position encoding is applied from position 0 regardless of
padding layout. Use right-padding with `-100` labels on pad positions.
