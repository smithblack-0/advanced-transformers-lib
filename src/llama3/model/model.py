"""Transformer backbone for the Llama 3 baseline.

Llama3Model is a pure transformer stack: a sequence of DecoderLayer blocks
followed by a final RMSNorm. It accepts pre-embedded hidden states and returns
contextual representations. It has no knowledge of tokens, vocabulary, or how
the input was produced.

Keeping the embedding out of the backbone is the correct HF convention and
makes the backbone genuinely modality-agnostic. The token interface — embedding
lookup, LM head, weight tying — belongs on the task wrapper (Llama3ForCausalLM),
which is the only class that knows this backbone is being used for language
modelling.

The final RMSNorm is necessary because the decoder stack uses pre-norm
throughout: each sublayer normalises its own input, leaving the residual stream
itself unnormalised. After many layers of accumulated residuals, that stream
arrives at the top with uncontrolled magnitude. The final norm brings it to a
well-scaled state before any projection. Without it, the LM head would receive
signals of arbitrary scale.

Returns a plain dict. The keys are explicit and the contract is visible without
importing HF output dataclass machinery.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration import Llama3Config
from .decoder_layer import DecoderLayer
from .type_aliases import ModelKVCache


class Llama3Model(PreTrainedModel):
    """Pure transformer backbone: decoder stack and final normalisation.

    Accepts pre-embedded hidden states of shape (batch, seq_len, hidden_size)
    and returns contextual representations of the same shape. No token embedding
    or vocabulary projection — those live on Llama3ForCausalLM.

    RoPE is applied inside each attention layer. Positional information is
    encoded in the relationship between Q and K, not added to the residual
    stream, so the backbone is agnostic to how positions are represented.

    Args:
        config: Model configuration. Must be a ``Llama3Config`` instance.
    """

    config_class = Llama3Config
    base_model_prefix = "model"

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # RMSNorm over LayerNorm: omits mean subtraction, faster, and proved more
        # stable at scale. This is the final norm that stabilises the accumulated
        # residual stream — distinct from the per-layer pre-norms inside each block.
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # Suppress HF's default reinitialisation pass. HF's _init_weights overwrites
        # all Linear and Embedding weights with normal(0, 0.02) after construction,
        # silently replacing PyTorch's own defaults (kaiming_uniform_ for Linear,
        # normal(0,1) for Embedding). PyTorch's reset_parameters() already ran at
        # construction time and those initialisations should stand.
        pass

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
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
            past_key_values: Full-model KV cache from a prior forward pass, one
                ``KVCache`` entry per decoder layer. None during prefill.
            use_cache: Whether to return an updated ``past_key_values``. Overrides
                ``config.use_cache`` when provided.
            output_hidden_states: Whether to return per-layer hidden states.
                Overrides ``config.output_hidden_states`` when provided.

        Returns:
            Dict with keys:
            - ``"last_hidden_state"``: normed backbone output,
              shape (batch, seq_len, hidden_size).
            - ``"past_key_values"``: updated ``ModelKVCache``, or None if
              ``use_cache`` is False.
            - ``"hidden_states"``: tuple of (inputs_embeds, layer_0 output, ...,
              layer_N output) — all before the final norm — or None if
              ``output_hidden_states`` is False. Collected before the final norm
              so each entry reflects the unnormalised residual stream at that depth,
              which is what probing and representation research typically requires.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        batch, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            # During cached generation, new tokens must be positioned after all
            # previously cached tokens. The past sequence length is the sum of all
            # chunk sizes stored in the first layer's key cache.
            past_seq_len = 0
            if past_key_values is not None:
                past_seq_len = sum(
                    chunk.shape[2] for chunk in past_key_values[0][0]
                )
            position_ids = torch.arange(
                past_seq_len, past_seq_len + seq_len, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch, -1)

        hidden_states = inputs_embeds

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        present_key_values: ModelKVCache = []

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(hidden_states, position_ids, layer_past)

            if use_cache:
                present_key_values.append(present_kv)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": present_key_values if use_cache else None,
            "hidden_states": all_hidden_states,
        }
