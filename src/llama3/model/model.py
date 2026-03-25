"""Transformer backbone for the Llama 3 baseline.

Llama3Model is a pure PyTorch module: a sequence of DecoderLayer blocks followed
by a final RMSNorm. It accepts pre-embedded hidden states and returns contextual
representations. It has no knowledge of tokens, vocabulary, generation, or the
HuggingFace contract — those concerns belong on Llama3ForCausalLM.

Keeping the embedding out of the backbone is the correct HF convention and makes
the backbone genuinely modality-agnostic. The token interface — embedding lookup,
LM head, weight tying — belongs on the task wrapper (Llama3ForCausalLM), which is
the only class that knows this backbone is being used for language modelling.

The final RMSNorm is necessary because the decoder stack uses pre-norm throughout:
each sublayer normalises its own input, leaving the residual stream itself
unnormalised. After many layers of accumulated residuals, that stream arrives at the
top with uncontrolled magnitude. The final norm brings it to a well-scaled state
before any projection. Without it, the LM head would receive signals of arbitrary
scale.

KV caching is caller-managed. If a Cache object is provided as past_key_values, it
is threaded through every decoder layer (each layer writes to its own slot via
layer_idx) and returned in the output dict. If None is provided, no caching occurs.
The decision of whether to create a cache and when belongs to the caller.

Returns a plain dict with keys:
- "last_hidden_state": normed backbone output, shape (batch, seq_len, hidden_size)
- "past_key_values": the Cache object passed in (updated in place), or None
- "hidden_states": tuple of per-layer activations if output_hidden_states=True, else None
"""

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from .configuration import Llama3Config
from .decoder_layer import DecoderLayer


class Llama3Model(nn.Module):
    """Pure transformer backbone: decoder stack and final normalisation.

    Accepts pre-embedded hidden states of shape (batch, seq_len, hidden_size)
    and returns contextual representations of the same shape. No token embedding,
    vocabulary projection, or HuggingFace lifecycle concerns.

    RoPE is applied inside each attention layer. Positional information is
    encoded in the relationship between Q and K, not added to the residual
    stream, so the backbone is agnostic to how positions are represented.

    Args:
        config: Model configuration. Must be a ``Llama3Config`` instance.
    """

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # RMSNorm over LayerNorm: omits mean subtraction, faster, and proved more
        # stable at scale. This is the final norm that stabilises the accumulated
        # residual stream — distinct from the per-layer pre-norms inside each block.
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_hidden_states: bool = False,
    ) -> dict:
        """Run the transformer stack over a batch of pre-embedded sequences.

        Args:
            inputs_embeds: Pre-embedded input of shape (batch, seq_len, hidden_size).
            position_ids: Absolute positions of shape (batch, seq_len). When None,
                positions are generated automatically starting from the end of any
                cached sequence. This offset is required for correctness during
                cached generation: RoPE frequencies are position-dependent, so each
                new token must carry the position it actually occupies in the full
                sequence, not position 0.
            past_key_values: A Cache object carrying the accumulated K/V history from
                prior forward passes, or None. When provided, each decoder layer writes
                new K/V into its slot and reads back the full accumulated history. The
                cache is updated in place and returned as-is. When None, no caching
                occurs and None is returned for past_key_values.
            output_hidden_states: When True, the output dict includes a tuple of
                per-layer hidden states: (inputs_embeds, layer_0_out, ..., layer_N_out),
                collected before the final norm.

        Returns:
            Plain dict with keys:
            - ``"last_hidden_state"``: normed backbone output,
              shape (batch, seq_len, hidden_size).
            - ``"past_key_values"``: the Cache object (updated in place), or None.
            - ``"hidden_states"``: tuple of per-layer activations (including
              inputs_embeds as position 0) if ``output_hidden_states`` is True,
              else None. Collected before the final norm so each entry reflects the
              unnormalised residual stream at that depth.
        """
        batch, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            # During cached generation, new tokens must be positioned after all
            # previously cached tokens. get_seq_length(0) returns the number of
            # tokens already stored for layer 0 — all layers hold the same count
            # because they were all updated together on previous steps.
            past_seq_len = past_key_values.get_seq_length(0) if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seq_len, past_seq_len + seq_len, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch, -1)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, cache=past_key_values, layer_idx=i)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values,
            "hidden_states": all_hidden_states,
        }
