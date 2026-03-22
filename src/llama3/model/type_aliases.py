"""Shared type aliases for the Llama 3 model.

Centralised here to avoid repeating complex type expressions across modules and
to give the KV cache structure a readable name at each layer of the model.
"""

from torch import Tensor

# KV cache for a single attention layer.
# A pair of chunk lists — (key_chunks, value_chunks) — where each chunk has shape
# (batch, num_kv_heads, chunk_seq_len, head_dim). Chunks are appended rather than
# concatenated at each generation step; concatenation happens once per forward pass
# at attention-compute time.
KVCache = tuple[list[Tensor], list[Tensor]]

# Full model KV cache: one KVCache per decoder layer, indexed by layer position.
ModelKVCache = list[KVCache]
