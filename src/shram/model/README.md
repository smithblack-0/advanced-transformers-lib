---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- research
- sparse-attention
- mixture-of-experts
---

# SHRAM — Sparse Hybrid Token Routed Attention Mixture

A research baseline implementing the SHRAM architecture from "An Examination of Sparse
Attention for Long Context Purposes." No pretrained weights — pull the architecture from
the Hub and instantiate a freshly initialised model from config. Every parameter is
overridable at instantiation time via kwargs.

> **Important:** `trust_remote_code=True` is required. It downloads the architecture
> source files from the Hub and imports them into your Python process. Review the
> source at [smithblack-0/SHRAM](https://huggingface.co/smithblack-0/SHRAM) before use.

## Architecture

SHRAM replaces every standard attention layer with a hybrid layer `H(x) = h_l(x) + h_s(x)`:

- **h_l** — local sliding-window causal attention path.
- **h_s** — MoSRAH sparse routed path. Each token selects K of L available expert heads
  via token-choice routing. Bottlenecked Ensemble Attention (BEA) is applied per head.

All other components follow the Llama 3 baseline (RMSNorm, SwiGLU FFN, RoPE).

## Usage

This repository contains no pretrained weights. The intended workflow is: pull the
architecture config from the Hub, instantiate a model with fresh random weights, then
train it yourself.

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Step 1: pull the architecture config from the Hub.
# AutoConfig.from_pretrained downloads config.json only — no weights are loaded.
# Override any parameter via kwargs.
config = AutoConfig.from_pretrained(
    "smithblack-0/SHRAM",
    trust_remote_code=True,
    num_hidden_layers=16,       # example override
    num_mosrah_heads=32,        # example override
)

# Step 2: instantiate with fresh random weights.
# from_config never loads a checkpoint — it always produces a randomly initialised model.
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Step 3: load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained("smithblack-0/SHRAM")
```

After training your own checkpoint, save and reload it in the standard way:

```python
model.save_pretrained("./my-checkpoint")
model = AutoModelForCausalLM.from_pretrained("./my-checkpoint", trust_remote_code=True)
```

## Constructor Defaults

The values below are the defaults you get if you call `AutoConfig.from_pretrained` with
no overrides. They are not the parameters of a pretrained model — this repository
contains no weights. All values are overridable via kwargs.

| Parameter | Default |
|-----------|---------|
| `vocab_size` | 50277 |
| `hidden_size` | 512 |
| `intermediate_size` | 1366 |
| `num_hidden_layers` | 12 |
| `num_sliding_window_heads` | 16 |
| `num_mosrah_heads` | 16 |
| `num_selected_heads` | 16 |
| `head_dim` | 16 |
| `window_size` | 128 |
| `rope_mode` | main_sequence |
| `local_rope_theta` | 10000.0 |
| `mosrah_rope_theta` | 10000.0 |
| `training_sequence_length` | 8192 |
| `inference_sequence_length` | 8192 |

## License

MIT. Clean-room synthesis informed by the reference paper. Tokenizer is GPT-NeoX
(`EleutherAI/gpt-neox-20b`, Apache 2.0).
