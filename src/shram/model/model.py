"""Transformer backbone for Shram.

ShramModel is a pure PyTorch module: a sequence of DecoderLayer blocks followed
by a final RMSNorm. It accepts pre-embedded hidden states and returns contextual
representations. It has no knowledge of tokens, vocabulary, generation, or the
HuggingFace causal-LM wrapper contract.

Keeping the embedding out of the backbone is the correct convention and makes
the backbone genuinely modality-agnostic. The token interface — embedding lookup,
LM head, weight tying, and generation-facing naming conventions — belongs on the
task wrapper (ShramForCausalLM), which is the only class that knows this
backbone is being used for language modelling.

The final RMSNorm is necessary because the decoder stack uses pre-norm throughout:
each sublayer normalises its own input, leaving the residual stream itself
unnormalised. After many layers of accumulated residuals, that stream arrives at
the top with uncontrolled magnitude. The final norm brings it to a well-scaled
state before any projection. Without it, the LM head would receive signals of
arbitrary scale.

Caching is caller-managed. If a ShramCache is provided, ShramModel threads the
corresponding per-layer ShramLayerCache into each DecoderLayer and returns the
same top-level ShramCache object in the output dict. If None is provided, no
caching occurs.

Returns a plain dict with keys:
- "last_hidden_state": normed backbone output, shape (batch, seq_len, hidden_size)
- "past_key_values": the ShramCache object passed in, or None
- "hidden_states": tuple of per-layer activations if output_hidden_states=True, else None
- "load_balance_loss": scalar sum of per-layer SHRAM load-balance losses
- "max_vio": detached scalar maximum routing-imbalance across all decoder layers
"""

import torch
import torch.nn as nn

from .cache.shram_cache import ShramCache
from .configuration import ShramConfig
from .decoder_layer import DecoderLayer


class ShramModel(nn.Module):
    """Pure transformer backbone: decoder stack and final normalisation.

    Accepts pre-embedded hidden states of shape (batch, seq_len, hidden_size)
    and returns contextual representations of the same shape. No token embedding,
    vocabulary projection, or causal-LM lifecycle concerns.

    RoPE is applied inside each attention layer. Positional information is
    encoded in the relationship between Q and K, not added to the residual
    stream, so the backbone is agnostic to how positions are represented.

    Args:
        config: Model configuration. Must be a ``ShramConfig`` instance.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache: ShramCache | None = None,
        output_hidden_states: bool = False,
    ) -> dict:
        """Run the transformer stack over a batch of pre-embedded sequences.

        Args:
            inputs_embeds: Pre-embedded input of shape (batch, seq_len, hidden_size).
            position_ids: Absolute positions of shape (batch, seq_len). Required.
                Must be provided explicitly by the caller — this module does not
                infer positions from cache state.
            cache: Optional top-level ShramCache. When provided, each DecoderLayer
                receives its own layer-local cache via ``cache.layers[layer_idx]``.
                The top-level cache object is updated in place and returned unchanged.
            output_hidden_states: When True, the output dict includes a tuple of
                per-layer hidden states: (inputs_embeds, layer_0_out, ..., layer_N_out),
                collected before the final norm.

        Returns:
            Plain dict with keys:
            - ``"last_hidden_state"``: normed backbone output,
              shape (batch, seq_len, hidden_size).
            - ``"past_key_values"``: the cache object passed in, or None.
            - ``"hidden_states"``: tuple of per-layer activations (including
              inputs_embeds as position 0) if ``output_hidden_states`` is True,
              else None. Collected before the final norm so each entry reflects the
              unnormalised residual stream at that depth.
            - ``"load_balance_loss"``: scalar sum of per-layer SHRAM
              load-balance losses.
            - ``"max_vio"``: detached scalar maximum routing-imbalance across
              all decoder layers. Zero means perfectly balanced routing across
              every layer; higher values identify the worst-case head imbalance.
        """
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        total_load_balance_loss = inputs_embeds.new_zeros(())
        max_vio = inputs_embeds.new_zeros(())

        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache.layers[layer_idx]
            hidden_states, layer_load_balance_loss, layer_max_vio = layer(
                hidden_states,
                position_ids,
                cache=layer_cache,
            )
            total_load_balance_loss = total_load_balance_loss + layer_load_balance_loss
            max_vio = torch.maximum(max_vio, layer_max_vio)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": cache,
            "hidden_states": all_hidden_states,
            "load_balance_loss": total_load_balance_loss,
            "max_vio": max_vio,
        }