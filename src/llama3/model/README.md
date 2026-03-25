---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- research
- llama
---

# advanced-transformers-lib -- Llama 3 Baseline

A Llama 3-style decoder-only transformer architecture for research. No pretrained
weights -- pull the architecture from the Hub and instantiate a freshly initialised
model from config. Override any parameter at instantiation time.

> **Important:** `trust_remote_code=True` is required. It downloads the architecture
> source files from the Hub and imports them into your Python process. Review the
> source at [smithblack-0/llama3_baseline](https://huggingface.co/smithblack-0/llama3_baseline) before use.

## Usage

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Pull architecture config -- override any parameter at instantiation time
config = AutoConfig.from_pretrained(
    "smithblack-0/llama3_baseline",
    trust_remote_code=True,
    num_hidden_layers=16,  # example override
)

# Instantiate with fresh random weights -- no checkpoint required
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("smithblack-0/llama3_baseline")

# Save and reload after training
model.save_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

## Default Configuration

| Parameter | Default |
|-----------|---------|
| `vocab_size` | 50277 |
| `hidden_size` | 768 |
| `intermediate_size` | 1568 |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `head_dim` | 48 |
| `max_position_embeddings` | 8192 |
| `rope_theta` | 500000.0 |

## License

MIT. Clean-room synthesis: the human author has not read the Llama source code.
Architectural decisions derive from the published paper. Tokenizer is GPT-NeoX
(`EleutherAI/gpt-neox-20b`, Apache 2.0).
