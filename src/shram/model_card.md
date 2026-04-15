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
> source at [{repo_id}](https://huggingface.co/{repo_id}) before use.

## Architecture

SHRAM replaces every standard attention layer with a hybrid layer `H(x) = h_l(x) + h_s(x)`:

- **h_l** — local sliding-window causal attention path.
- **h_s** — MoSRAH sparse routed path. Each token selects K of L available expert heads
  via token-choice routing. Bottlenecked Ensemble Attention (BEA) is applied per head.

All other components follow the Llama 3 baseline (RMSNorm, SwiGLU FFN, RoPE).

## Usage

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull architecture config — override any parameter at instantiation time
config = AutoConfig.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
    num_hidden_layers=16,       # example override
    num_mosrah_heads=32,        # example override
)

# Instantiate with fresh random weights — no checkpoint required
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Save and reload after training
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

## Constructor Defaults

The values below are the defaults you get if you call `AutoConfig.from_pretrained` with
no overrides. They are not the parameters of a pretrained model — this repository
contains no weights. All values are overridable via kwargs.

{config_table}

## License

MIT. Clean-room synthesis informed by the reference paper. Tokenizer is GPT-NeoX
(`EleutherAI/gpt-neox-20b`, Apache 2.0).
