# SHRAM

The **SRAM effect** is the hypothesis that certain routed sparse-attention systems can trade parameters for long-sequence performance at a roughly linear asymptotic rate. The main paper for that claim in this repository is `main.tex` at the repository root.

This `src/shram/` subtree is the concrete implementation used to probe that effect. The paper is the source of truth for the theorem, architectural argument, and experiment. This document is the source of truth for how to use and maintain the SHRAM code.

## Overview

SHRAM is a decoder-only transformer. Its distinctive feature is that each standard attention layer is replaced by a hybrid SHRAM layer composed of two paths:

- a **local sliding-window path**, which preserves nearby-token behavior
- a **sparse routed MoSRAH path**, which provides the long-range routed-attention behavior discussed in the paper

Outside that substitution, the surrounding transformer remains fairly conventional: decoder blocks are pre-norm, the feedforward path is SwiGLU, positional handling is RoPE/YaRN, and the HuggingFace-facing language-model surface is owned by a wrapper around the pure backbone.

## User Guide

### Basic usage

The two normal ways to work with SHRAM are:

1. instantiate it locally from `ShramConfig`
2. instantiate it through the HuggingFace AutoClass surface

Local construction:

```python
from src.shram.model.configuration import ShramConfig
from src.shram.model.huggingface import ShramForCausalLM

config = ShramConfig(
    num_mosrah_heads=32,
    num_selected_heads=16,
    rope_mode="main_sequence",
)
model = ShramForCausalLM(config)
```

HuggingFace-style construction:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    "smithblack-0/SHRAM",
    trust_remote_code=True,
    num_mosrah_heads=32,
    num_selected_heads=16,
    rope_mode="main_sequence",
)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

`trust_remote_code=True` is required because SHRAM is a custom architecture rather than a built-in `transformers` model class.

To extend the inference context window beyond the training length using YaRN, call
`set_inference_context()` on the config before constructing the model:

```python
config = AutoConfig.from_pretrained(
    "smithblack-0/SHRAM",
    trust_remote_code=True,
)
config.set_inference_context(131072)  # extend to 128k tokens
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
```

`training_sequence_length` is the base context the model was trained at. `set_inference_context()`
sets the target inference length; the YaRN scale factor `s = inference / training` is derived
automatically. When `inference == training`, YaRN reduces exactly to standard RoPE.

The current Hub surface is architecture-oriented. The normal Hub workflow is to load the config and instantiate fresh random weights. A normal weight-loading `from_pretrained()` path only applies once actual weights have been uploaded in the usual HuggingFace way.

### Important Notes

A few SHRAM settings are easier to understand in terms of their relationships with each other, especially when contextualized by the paper.

- The ratio `num_selected_heads / num_mosrah_heads` is directly correlated with the sparsity ratio discussed in the paper.
- The YaRN dilation rate is directly determined by `inference_sequence_length / training_sequence_length`.
- `head_dim` sets the per-head width in both the sliding-window and MoSRAH pathways.
- RoPE has two modes of operation. In `"main_sequence"` mode it uses original token positions after packing. In `"semantic_sequence"` mode, MoSRAH instead uses local positions inside each expert bucket.
- The default configuration has `num_selected_heads == num_mosrah_heads`, meaning every token selects all available expert heads — this is dense routing, not sparse. To engage the sparse regime described in the paper, set `num_selected_heads < num_mosrah_heads`.

Ordinary generation modes are intended to work almost completely flawlessly. Greedy, beam, and contrastive paths are explicitly tested. The main caveat worth documenting at the user level is that speculative rollback through `crop()` is not supported due to the use of a sliding-window cache structure.

### Config reference

This is the control surface. It is written so you do not need to open `configuration.py` just to understand what the fields do.

#### Core model size

- `vocab_size` (default `50277`) Controls tokenizer vocabulary size, input embedding width, and output logits width. Change this only if you are changing the tokenizer surface.
- `hidden_size` (default `512`) Controls residual-stream width for the whole model.
- `intermediate_size` (default `1366`) Controls the hidden width of the SwiGLU MLP.
- `num_hidden_layers` (default `12`) Controls decoder depth; each additional layer adds another SHRAM+MLP block.
- `rms_norm_eps` (default `1e-5`) Epsilon for RMSNorm layers. Usually left alone unless doing numerical stability work.

#### Local sliding-window path

- `num_sliding_window_heads` (default `16`) Controls the number of heads in the local sliding-window path.
- `head_dim` (default `16`) Controls per-head width in both the local and sparse paths. Must remain even because RoPE rotates dimension pairs.
- `window_size` (default `128`) Controls how far the local path can see into recent context.
- `local_rope_theta` (default `10000.0`) Controls the RoPE base frequency used by the local path.

#### Sparse routed MoSRAH path

- `num_mosrah_heads` (default `16`) Controls total routed head capacity `L`.
- `num_selected_heads` (default `16`) Controls how many routed heads each token actually selects `K`.
- `rope_mode` (default `"main_sequence"`) Controls how the sparse path interprets position: `"main_sequence"` uses original token positions after packing; `"semantic_sequence"` uses local per-expert packed positions.
- `mosrah_rope_theta` (default `10000.0`) Controls the RoPE base frequency used by the sparse routed path.
- `training_sequence_length` (default `8192`) Base training context length used by sparse-path YaRN scaling.
- `inference_sequence_length` — Target inference context length used by sparse-path YaRN scaling. Not a constructor parameter; defaults to `training_sequence_length` at construction (scale=1.0, standard RoPE). Set it via `config.set_inference_context(length)`.
- `alpha` (default `1.0`) Lower YaRN ramp boundary for the sparse path. Leave alone unless changing sparse-path YaRN behavior intentionally.
- `beta` (default `32.0`) Upper YaRN ramp boundary for the sparse path. Leave alone unless changing sparse-path YaRN behavior intentionally.

#### Wrapper and integration controls

- `attention_dropout` (default `0.0`) Controls attention-weight dropout probability. Defaults to off.
- `use_cache` (default `True`) Controls whether the wrapper returns and uses cache objects. Leave this on for normal generation workflows.
- `output_hidden_states` (default `False`) Controls whether intermediate hidden-state outputs are returned.
- `tie_word_embeddings` (default `False`) Controls whether input embeddings and the LM head share weights.

## Running the Tests

```bash
python -m pytest tests/shram/
```

Tests in `TestTokenizerInModelDir` are skipped if the tokenizer has not been prepared
locally — run `python -m src.shram.tokenizer` first.

---

## Maintainer Guide

### Uploading the Architecture to HuggingFace Hub

**1. Edit the repository name if needed.**
Open `src/shram/upload_to_hub.py` and check `REPO_ID` at the top. It is set to
`smithblack-0/SHRAM` by default. Change it to match your intended HuggingFace
repository before proceeding.

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
python -m src.shram.upload_to_hub
```

Paste the token when prompted. The script will:
1. Refresh tokenizer files
2. Write `config.json`
3. Render the architecture card (`README.md`) from `model_card.md`
4. Upload `src/shram/model/` to the repository root

**5. Delete the token (recommended).**
Go back to Settings > Access Tokens and delete the token you just used.

### Verifying the upload

From a fresh Python environment with no local HuggingFace cache:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained("smithblack-0/SHRAM", trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("smithblack-0/SHRAM")
```

All three must succeed. The Hub repository must contain no weight files.

---

### Repository structure and the Hub distribution

`src/shram/model/` is the Hub distribution unit. Its contents are uploaded to the root
of the Hub repository by `upload_to_hub.py`. What is in
that directory is exactly what researchers receive via `trust_remote_code=True`. This means:

- Source files that belong on the Hub must live inside `model/`.
- Files outside `model/` (`tokenizer.py`, `upload_to_hub.py`, etc.) are local operational
  tools and are never uploaded.

### Relative imports inside model/

All Python files inside `model/` must use relative imports (e.g. `from .configuration import
ShramConfig`), not absolute imports (e.g. `from src.shram.model.configuration import ...`).

When a researcher calls `AutoConfig.from_pretrained(..., trust_remote_code=True)`, HuggingFace
downloads the `model/` contents into a local cache directory and adds that directory to
`sys.path`. In that context there is no `src` or `shram` package — absolute imports break
immediately. Relative imports work because `model/` contains `__init__.py` and Python
resolves them within the package.

Files outside `model/` are never uploaded and may use absolute imports.

### Hub upload and the staging system

HuggingFace cannot handle subdirectories in remote custom-model uploads. Its internal module loader resolves relative imports by appending `.py` to the dotted import path directly, without converting dots to path separators. A file at `cache/shram_cache.py` is therefore never found.

The upload script works around this via a staging step (`src/shram/stage_for_hub.py`). Before uploading, it produces a temporary flat directory where subdirectory files are renamed using double-underscore prefixes:

```
model/cache/shram_cache.py      ->  __cache__shram_cache.py
model/attention/mosrah.py       ->  __attention__mosrah.py
```

Relative imports in all Python files are rewritten to match using libcst. The staging directory is what gets uploaded; the source tree is never modified.

**Maintenance caveat:** `__init__.py` files inside `model/attention/` and `model/cache/` are dropped during staging and do not exist on the Hub. Do not place any meaningful logic in these files — it will not be available to Hub users.

---

### Major files and what they own

- `model/configuration.py` Owns the SHRAM config surface. Change this file when adding, removing, renaming, validating, or reinterpreting user-facing architectural controls.
- `model/huggingface.py` Owns the HuggingFace-facing wrapper. This is where token embedding lookup, LM-head projection, wrapper-side loss, generation/cache orchestration, and wrapper output behavior live.
- `model/model.py` Owns the pure backbone assembly.
- `model/decoder_layer.py` Owns one transformer block. This is where SHRAM attention and the MLP are threaded together inside a pre-norm residual block.
- `model/attention/shram.py` Owns the SHRAM hybrid relation `H(x) = h_l(x) + h_s(x)`.
- `model/attention/mosrah.py` Owns the MoSRAH sparse routed path orchestration.
- `model/attention/router.py` Owns token-choice routing, routing probabilities, load-balance loss, and MaxVio.
- `model/attention/expert_packing.py` Owns the token-choice ↔ expert-choice conversion boundary.
- `model/attention/positions_converter.py` Owns packed position construction for the sparse routed path.
- `model/attention/sliding_window_attention.py` Owns the local sliding-window attention path.
- `model/attention/bottlenecked_ensemble_attention.py` Owns BEA itself.
- `model/attention/load_balance_loss.py` Owns the custom autograd load-balance operator.
- `model/cache/shram_cache.py` Owns the top-level cache boundary used by the wrapper and HuggingFace.
- `model/cache/shram_layer_cache.py` Owns the per-layer cache composition: local cache plus sparse cache.
- `model/cache/sliding_window_cache.py` Owns the local sliding-window cache behavior.
- `model/cache/mosrah_cache.py` Owns the sparse expert-choice cache behavior.
- `model/cache/slow_mosrah_cache.py` Owns the slow correctness oracle for the sparse cache.
- `model/rope.py` Owns RoPE and YaRN mechanics.
- `model/mlp.py` Owns the SwiGLU feedforward layer.
- `upload_to_hub.py` Owns the SHRAM publishing path.

### Supporting files worth knowing about

- `documentation.md` Older developer-guide surface. Do not assume it is the current primary SHRAM documentation.
- `model/README.md` Hub/model-card-facing surface, not the primary SHRAM documentation file.
- `model_card.md` Model-card template input for the publish path.
- `plan.md` Implementation/process history and blocker trail. Use it when design rationale matters.
- `Legal.md` Licensing and clean-room synthesis notes.
- `tokenizer.py` Local tokenizer preparation/update script for the SHRAM subtree.

### Where to go when changing specific behavior

- Changing config knobs, defaults, or validation: start in `model/configuration.py`.
- Changing instantiation, wrapper outputs, generation behavior, or wrapper-side loss: start in `model/huggingface.py`.
- Changing backbone threading or model outputs: start in `model/model.py` and `model/decoder_layer.py`.
- Changing hybrid attention composition: start in `model/attention/shram.py`.
- Changing routed sparse behavior: start in `model/attention/mosrah.py`, then inspect `router.py`, `expert_packing.py`, `positions_converter.py`, and `bottlenecked_ensemble_attention.py` as needed.
- Changing local attention behavior: start in `model/attention/sliding_window_attention.py`, then inspect `model/cache/sliding_window_cache.py`.
- Changing cache behavior: start in `model/cache/shram_cache.py`, then inspect `shram_layer_cache.py` and the relevant sub-cache file.
- Changing publish/Hub behavior: start in `upload_to_hub.py`, then inspect `model_card.md` and `model/README.md`.

### Maintenance caveats

- The SHRAM code assumes the paper exists and owns the deeper theory. If you change theory or the meaning of the SRAM-effect-facing controls, the paper may also need an update.
- `configuration.py` is the single source of truth for user-facing architectural controls. If you add a new user-facing architectural knob and do not document it here, the user guide is stale.
- The wrapper and cache layers are tightly coupled to the HuggingFace boundary. Changes there should be treated as interface changes, not just internal refactors.
- `model/README.md` and `model_card.md` are publish surfaces. Do not confuse them with this documentation file.

### When to update this document

Update this document when any of the following change:

- the practical way a user instantiates or configures SHRAM
- the meaning of a config field
- the generation/cache caveats a user needs to know
- the major file ownership boundaries
- the upload/publish path
- the relationship between the SHRAM code and the paper

---

## Development History and Bumps

Implementation started from the Llama 3 baseline in this repository. The early units were
relatively clean — config, router, rope, cache scaffolding — with no major surprises.

Expert packing was where things got interesting. The scheme as implemented turned out to be
incompatible with inference. Tracing the problem back revealed the paper had the same gap.
The paper was patched; this is the one place the implementation corrected the source material.

A bug appeared at the MoSRAH/RoPE/masking boundary (Unit 11.A) that had to be isolated across
three interacting subsystems before the hybrid layer could be completed.

The biggest event was late and unexpected: masked continuation had never been designed for.
The MaxVio metric had also been missed entirely. Adding MaxVio forced a look at masking
behaviour, which revealed the whole attention stack was wrong under masked inputs. Eight
consecutive blocker units followed — local cache, cache wiring, sliding-window attention,
expert packing, router, MoSRAH orchestration, and mask plumbing end-to-end. The completion
record's long blocker run (14.B–14.I) is the scar from that.

The final notable bump was HuggingFace's `get_seq_length()` protocol, which surfaced as a
hard requirement only during wrapper integration.

What 