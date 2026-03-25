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

KV caching uses HuggingFace's Cache protocol. GenerationMixin creates and
manages the DynamicCache for generate() calls, passing it as past_key_values
on every forward call. The backbone updates the cache in place and returns the
same object. _reorder_cache delegates to DynamicCache.reorder_cache() for beam
search, keeping all beam-reordering logic inside the cache implementation.

Returns a CausalLMOutputWithPast. ModelOutput subclasses support both attribute
access (output.logits) and dict-style access (output["logits"]), satisfying
GenerationMixin's attribute access requirements while keeping existing code unchanged.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import Llama3Config
from .model import Llama3Model


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

    def _reorder_cache(
        self, past_key_values: Cache, beam_idx: torch.Tensor
    ) -> Cache:
        """Reorder the KV cache to match beam reordering during beam search.

        GenerationMixin calls this after pruning and reordering beams at each
        step. beam_idx[i] is the old batch position whose cache should move to
        position i. DynamicCache.reorder_cache() handles the index-select on
        every stored tensor's batch dimension, keeping the cache consistent with
        the reordered beam hypotheses.

        Args:
            past_key_values: The active Cache object.
            beam_idx: 1-D tensor of shape (batch * num_beams,) mapping new batch
                positions to old ones.

        Returns:
            The same Cache object, reordered in place.
        """
        past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Run the causal language model.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            position_ids: Absolute positions of shape (batch, seq_len). Passed
                through to the backbone; generated automatically if None.
            past_key_values: A HuggingFace Cache object from a prior step, or
                None. When use_cache=True and this is None, a fresh DynamicCache
                is created here before calling the backbone.
            use_cache: Whether to accumulate and return a KV cache. When True
                and no cache is provided, a DynamicCache is created. When False,
                None is passed to the backbone regardless of what was provided.
                Defaults to config.use_cache when None.
            output_hidden_states: Whether to return per-layer hidden states.
                Passed through to the backbone.
            labels: Target token indices of shape (batch, seq_len) for computing
                next-token prediction loss. The loss is computed over positions
                1..seq_len predicting from positions 0..seq_len-1 — the shift
                is applied internally. Positions with label value -100 are
                ignored by cross-entropy, following the HuggingFace convention
                for padding and masked positions.
            cache_position: Accepted for GenerationMixin compatibility in newer
                HuggingFace versions. Not used — position tracking is handled
                internally via position_ids.
            **kwargs: Additional keyword arguments passed by GenerationMixin
                (e.g. return_dict, attention_mask). Accepted and ignored for
                forward compatibility. We always return CausalLMOutputWithPast
                regardless of return_dict.

        Returns:
            CausalLMOutputWithPast with fields:
            - ``logits``: vocabulary scores of shape (batch, seq_len, vocab_size).
            - ``loss``: scalar cross-entropy loss, or None if labels not provided.
            - ``past_key_values``: the updated Cache object, or None.
            - ``hidden_states``: per-layer hidden states, or None.
        """
        # Resolve both flags against config defaults. Config sets the default;
        # per-call arguments override it. Both fields in Llama3Config remain live.
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Cache lifecycle is owned here — the backbone only receives a cache or None
        # and never decides whether to create one.
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
        else:
            past_key_values = None

        inputs_embeds = self.embed_tokens(input_ids)

        backbone_out = self.model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
            past_key_values=backbone_out["past_key_values"],
            hidden_states=backbone_out["hidden_states"],
        )
