"""HuggingFace wrapper for the Llama 3 baseline.

Llama3ForCausalLM wraps Llama3Model with everything a researcher needs to
train, evaluate, and generate from it through the HuggingFace ecosystem:
token embedding, vocabulary projection, next-token loss, weight tying, and
the full AutoClass and GenerationMixin contracts.

The token embedding lives here, not on the backbone. Llama3Model is a pure
transformer stack that accepts pre-embedded hidden states — it has no knowledge
of tokens or vocabulary. This is the correct HF convention: the backbone is
modality-agnostic; the token interface belongs on the task wrapper.

The LM head projects the backbone's (batch, seq, hidden_size) output to
(batch, seq, vocab_size) logits. When labels are provided, cross-entropy loss
is computed with a one-position shift: token i predicts token i+1. The shift
is applied here rather than expected from the caller — a causal LM always
trains this way and there is no use case for an unshifted loss.

Weight tying: when config.tie_word_embeddings is True, lm_head.weight is
directly assigned to embed_tokens.weight after post_init(). Both matrices are
shape (vocab_size, hidden_size) — same shape, no transpose needed.

Returns a CausalLMOutputWithPast. ModelOutput subclasses support both attribute
access (output.logits) and dict-style access (output["logits"]), satisfying
GenerationMixin's attribute access requirements while keeping existing code unchanged.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import Llama3Config
from .model import Llama3Model
from .type_aliases import ModelKVCache


class Llama3ForCausalLM(PreTrainedModel, GenerationMixin):
    """Llama 3 causal language model: token embedding, backbone, LM head, HF contract.

    Owns the token embedding and LM head. Delegates all transformer computation
    to Llama3Model. Adds loss computation for training, weight tying between the
    LM head and the input embedding, and the full HuggingFace AutoClass and
    GenerationMixin contracts.

    Args:
        config: Model configuration. Must be a ``Llama3Config`` instance.
    """

    config_class = Llama3Config
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Llama3Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model = Llama3Model(config)

        # No bias — consistent with all other projections in this architecture.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # Direct weight tying: both matrices are (vocab_size, hidden_size) — same shape,
        # no transpose. Explicit here for visibility; post_init() → tie_weights() also
        # performs this via get_input/output_embeddings(), but that is less readable.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        # Suppress HF's default reinitialisation pass. HF's _init_weights overwrites
        # all Linear and Embedding weights with normal(0, 0.02) after construction,
        # silently replacing PyTorch's own defaults (kaiming_uniform_ for Linear,
        # normal(0,1) for Embedding). PyTorch's reset_parameters() already ran at
        # construction time and those initialisations should stand.
        pass

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: ModelKVCache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Run the causal language model.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            position_ids: Absolute positions of shape (batch, seq_len). Passed
                through to the backbone; generated automatically if None.
            past_key_values: Full-model KV cache from a prior step, or None
                during prefill. Passed through to the backbone.
            use_cache: Whether to return updated past_key_values. Passed
                through to the backbone.
            output_hidden_states: Whether to return per-layer hidden states.
                Passed through to the backbone.
            labels: Target token indices of shape (batch, seq_len) for computing
                next-token prediction loss. The loss is computed over positions
                1..seq_len predicting from positions 0..seq_len-1 — the shift
                is applied internally. Positions with label value -100 are
                ignored by cross-entropy, following the HuggingFace convention
                for padding and masked positions.

        Returns:
            CausalLMOutputWithPast with fields:
            - ``logits``: vocabulary scores of shape (batch, seq_len, vocab_size).
            - ``loss``: scalar cross-entropy loss, or None if labels not provided.
            - ``past_key_values``: updated ``ModelKVCache``, or None.
            - ``hidden_states``: per-layer hidden states, or None.
        """
        inputs_embeds = self.embed_tokens(input_ids)

        backbone_out = self.model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = self.lm_head(backbone_out["last_hidden_state"])

        loss = None
        if labels is not None:
            # Shift so that each position predicts the next token. The final
            # logit has no target; the first label has no corresponding input.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            past_key_values=backbone_out.past_key_values,
            hidden_states=backbone_out.hidden_states,
        )
