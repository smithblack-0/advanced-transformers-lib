"""Llama 3 transformer backbone and causal LM wrapper.

This file contains two classes:

  Llama3Model           — pure backbone: embedding + decoder stack + final norm.
                          No LM head, no loss. Independently verifiable.

  Llama3ForCausalLM     — HF wrapper around Llama3Model. Adds the LM head,
                          weight tying, loss computation, and the full HF
                          AutoClass/GenerationMixin contract.

Keeping them separate means the backbone can be verified in isolation before
the HF machinery is layered on top.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from src.llama3.configuration import Llama3Config
from src.llama3.decoder_layer import DecoderLayer
from src.llama3.type_aliases import ModelKVCache


class Llama3Model(PreTrainedModel):
    """Pure transformer backbone: token embedding, decoder stack, final norm.

    No LM head, no loss, no generation machinery. Returns a dict so the
    output contract is explicit and does not depend on HF dataclass imports.

    Args:
        config: Model configuration. Must be a ``Llama3Config`` instance.
    """

    config_class = Llama3Config
    base_model_prefix = "model"

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # Final norm applied to the decoder stack output before any projection.
        # RMSNorm chosen over LayerNorm: omits mean subtraction, faster, and
        # proved more stable at scale. Independent from the per-layer norms —
        # each occupies a different position in the signal flow.
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> dict:
        """Run the transformer backbone.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            position_ids: Absolute positions of shape (batch, seq_len). Generated
                automatically when None, correctly offset by past sequence length
                so that cached generation continues from the right positions.
            past_key_values: Full-model KV cache from a prior forward pass, or None
                during prefill. One ``KVCache`` entry per decoder layer.
            use_cache: Whether to return ``past_key_values``. Overrides
                ``config.use_cache`` when provided.
            output_hidden_states: Whether to return per-layer hidden states.
                Overrides ``config.output_hidden_states`` when provided.

        Returns:
            Dict with keys:
            - ``"last_hidden_state"``: normed output of shape (batch, seq_len, hidden_size).
            - ``"past_key_values"``: updated ``ModelKVCache``, or None if ``use_cache``
              is False.
            - ``"hidden_states"``: tuple of tensors — embedding output followed by each
              decoder layer output (all before the final norm) — or None if
              ``output_hidden_states`` is False.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        batch, seq_len = input_ids.shape

        # Build position_ids when not provided. During cached generation the new
        # tokens must be positioned after all cached tokens, so the starting offset
        # equals the total cached sequence length.
        if position_ids is None:
            past_seq_len = 0
            if past_key_values is not None:
                past_seq_len = sum(
                    chunk.shape[2] for chunk in past_key_values[0][0]
                )
            position_ids = torch.arange(
                past_seq_len, past_seq_len + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch, -1)

        hidden_states = self.embed_tokens(input_ids)

        # Collect the embedding output as the first hidden state when requested.
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
