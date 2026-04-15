"""HuggingFace causal-LM wrapper for SHRAM.

ShramForCausalLM is the HuggingFace-facing language-model boundary for SHRAM.
It owns token embedding lookup, LM-head projection, wrapper-level next-token
cross-entropy loss, config-controlled tied embeddings, and generation/cache
orchestration at the wrapper boundary.

The backbone remains a pure transformer stack. ShramModel accepts pre-embedded
hidden states together with current position IDs, a current active mask, and an
optional ShramCache. It has no knowledge of token IDs, vocabulary projection,
or causal-LM loss.

HuggingFace generation reaches this wrapper with two different tensor
conventions:

- ``position_ids`` is a current-step tensor. GenerationMixin updates the total
  sequence state between steps, then slices position-bearing tensors back down
  before calling ``forward()``.
- ``attention_mask`` is a full 2D mask over the total sequence so far. This
  wrapper slices its recent chunk to produce the current semantic liveness mask
  expected by the backbone.

Generation-created caches are handled in ``_prepare_cache_for_generation``.
That hook ensures HuggingFace generation uses ShramCache rather than a generic
dynamic cache. The direct ``forward()`` path does not silently create caches;
when ``use_cache=True`` it expects a truthful ShramCache to have been supplied.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationMode
from transformers.modeling_outputs import CausalLMOutputWithPast

from .cache.shram_cache import ShramCache
from .configuration import ShramConfig
from .model import ShramModel


@dataclass
class ShramCausalLMOutput(CausalLMOutputWithPast):
    """SHRAM causal-LM wrapper output.

    This subclasses HuggingFace's standard ``CausalLMOutputWithPast``.
    Dataclass inheritance is sufficient here: all standard causal-LM fields and
    ModelOutput behavior are inherited from the parent, and this subclass adds
    only the SHRAM-specific wrapper outputs.
    """

    load_balance_loss: torch.FloatTensor | None = None
    max_vio: torch.FloatTensor | None = None


class ShramForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-facing causal language model wrapper for SHRAM.

    Owns token embeddings, LM-head projection, wrapper-level shifted CE loss,
    tied embedding configuration, and generation/cache boundary behavior.
    Delegates all transformer computation to ``ShramModel``.

    Args:
        config: SHRAM model configuration.
    """

    config_class = ShramConfig
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ShramConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model = ShramModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._configure_tied_embeddings()
        self.post_init()

    def _configure_tied_embeddings(self) -> None:
        """Apply config-controlled tied embedding behavior on this instance."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            self._tied_weights_keys = {
                "lm_head.weight": "embed_tokens.weight",
            }
        else:
            self._tied_weights_keys = {}

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding matrix."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the token embedding matrix."""
        self.embed_tokens = value
        self._configure_tied_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        """Return the LM head."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Replace the LM head."""
        self.lm_head = value
        self._configure_tied_embeddings()

    def _build_shram_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> ShramCache:
        """Construct a fresh top-level SHRAM cache."""
        return ShramCache(
            num_hidden_layers=self.config.num_hidden_layers,
            sliding_window=self.config.window_size,
            num_local_heads=self.config.num_sliding_window_heads,
            local_head_dim=self.config.head_dim,
            num_mosrah_heads=self.config.num_mosrah_heads,
            mosrah_head_dim=self.config.hidden_size // self.config.num_selected_heads,
            batch_size=batch_size,
            device=device,
        )

    def _validate_generation_cache_request(
        self,
        generation_config: Any,
        model_kwargs: dict[str, Any],
        generation_mode: GenerationMode,
    ) -> None:
        """Validate SHRAM's generation-side cache policy."""
        if generation_mode in {
            GenerationMode.ASSISTED_GENERATION,
            GenerationMode.CONTRASTIVE_SEARCH,
        }:
            raise NotImplementedError(
                "ShramForCausalLM does not currently support assisted generation "
                "or contrastive search because ShramCache does not support crop()."
            )

        user_defined_cache = model_kwargs.get("past_key_values")
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    "Passing both `cache_implementation` and `past_key_values` "
                    "is unsupported. Please use only one."
                )
            if isinstance(user_defined_cache, tuple):
                raise ValueError(
                    "Passing a tuple of `past_key_values` is not supported. "
                    "Please use a `ShramCache` instance."
                )
            if not isinstance(user_defined_cache, ShramCache):
                raise TypeError(
                    "ShramForCausalLM requires `past_key_values` to be a "
                    "`ShramCache` instance."
                )

        if (
            user_defined_cache is None
            and generation_config.use_cache
            and generation_config.cache_implementation is not None
        ):
            raise ValueError(
                "ShramForCausalLM does not support `cache_implementation`. "
                "Generation-created caches must be `ShramCache` objects."
            )

    def _prepare_cache_for_generation(
        self,
        generation_config: Any,
        model_kwargs: dict[str, Any],
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
    ) -> None:
        """Ensure HuggingFace generation uses ShramCache.

        This is the SHRAM-specific generation hook. The rest of the default
        generation plumbing is kept intact as much as possible.

        Args:
            generation_config: Active generation configuration.
            model_kwargs: Generation kwargs, updated in place.
            generation_mode: HuggingFace generation mode.
            batch_size: Effective generation batch size.
            max_cache_length: Requested cache length. Accepted but unused here.
        """
        self._validate_generation_cache_request(
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            generation_mode=generation_mode,
        )

        if model_kwargs.get("past_key_values") is not None:
            return

        if not generation_config.use_cache:
            return

        model_kwargs["past_key_values"] = self._build_shram_cache(
            batch_size=batch_size,
            device=self.embed_tokens.weight.device,
        )

    def _reorder_cache(
        self,
        past_key_values: Cache,
        beam_idx: torch.Tensor,
    ) -> Cache:
        """Reorder the cache in place for beam search."""
        past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        """Validate token IDs at the wrapper boundary."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len).")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids sequence length must be nonzero.")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be an long int tensor.")

    def _validate_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        """Validate the full-sequence attention mask."""
        if attention_mask is None:
            return
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (batch, total_seq_len).")
        if attention_mask.shape[0] != input_ids.shape[0]:
            raise ValueError("attention_mask batch dimension must match input_ids.")
        if attention_mask.shape[1] < input_ids.shape[1]:
            raise ValueError(
                "attention_mask must be at least as long as the current input_ids chunk."
            )

    def _validate_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> None:
        """Validate current-step position IDs."""
        if position_ids is None:
            return
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape (batch, seq_len).")
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must match the current input_ids shape exactly."
            )
        if input_ids.dtype != torch.long:
            raise TypeError("position_ids must be an long tensor.")

    def _validate_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> None:
        """Validate label shape at the wrapper boundary."""
        if labels is None:
            return
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, seq_len).")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids.")
        if input_ids.dtype != torch.long:
            raise TypeError("labels must be a long tensor.")

    def _validate_cache_inputs(
        self,
        use_cache: bool,
        past_key_values: Cache | None,
    ) -> None:
        """Validate cache policy for direct wrapper calls."""
        if use_cache:
            if past_key_values is None:
                raise ValueError(
                    "use_cache=True requires an explicit ShramCache. During "
                    "generate(), HuggingFace should supply this through "
                    "_prepare_cache_for_generation()."
                )
            if not isinstance(past_key_values, ShramCache):
                raise TypeError(
                    "past_key_values must be a ShramCache when use_cache=True."
                )
            return

        if past_key_values is not None:
            raise ValueError("past_key_values was provided while use_cache=False.")

    def _validate_position_sources(
        self,
        use_cache: bool,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> None:
        """Validate that cached forward has a truthful source of positions."""
        if use_cache and attention_mask is None and position_ids is None:
            raise ValueError(
                "Cached forward requires either position_ids or attention_mask."
            )

    def _validate_hf_boundary(
        self,
        output_attentions: bool | None,
        return_dict: bool | None,
        inputs_embeds: torch.Tensor | None,
        cache_position: torch.Tensor | None,
        extra_kwargs: dict[str, Any],
    ) -> None:
        """Validate unsupported HuggingFace-facing wrapper inputs."""
        if output_attentions:
            raise NotImplementedError(
                "ShramForCausalLM does not expose output_attentions."
            )
        if return_dict is False:
            raise ValueError(
                "return_dict=False is not supported. "
                "ShramForCausalLM always returns ShramCausalLMOutput."
            )
        if inputs_embeds is not None:
            raise ValueError(
                "inputs_embeds is not supported at the SHRAM wrapper boundary. "
                "Pass input_ids instead."
            )
        if extra_kwargs:
            unsupported = ", ".join(sorted(extra_kwargs))
            raise TypeError(
                f"Unsupported forward kwargs for ShramForCausalLM: {unsupported}"
            )

    def _standardize_full_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.BoolTensor:
        """Return a concrete full-sequence boolean attention mask."""
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return attention_mask.to(dtype=torch.bool)

    def _resolve_current_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        full_attention_mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """Resolve concrete current-step position IDs for the backbone."""
        if position_ids is not None:
            return position_ids.to(dtype=torch.long)

        full_position_ids = full_attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        full_position_ids = full_position_ids.masked_fill(~full_attention_mask, 0)
        current_length = input_ids.shape[1]
        return full_position_ids[:, -current_length:]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> ShramCausalLMOutput:
        """Run the SHRAM causal language model wrapper.

        Args:
            input_ids: Current token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional full 2D mask of shape
                ``(batch, total_seq_len)``. The wrapper slices its recent chunk
                to produce the current semantic liveness mask expected by the
                backbone.
            position_ids: Optional current-step position IDs of shape
                ``(batch, seq_len)``. In ordinary HuggingFace generation this is
                already the current-step tensor when it reaches ``forward()``.
            past_key_values: Optional SHRAM cache. Required when
                ``use_cache=True``.
            use_cache: Whether to use and return a cache. Defaults to
                ``config.use_cache``.
            output_hidden_states: Whether to return backbone hidden states.
                Defaults to ``config.output_hidden_states``.
            labels: Optional target token IDs of shape ``(batch, seq_len)``.
            return_dict: Must be ``True`` or ``None``.
            **kwargs: Unsupported HuggingFace kwargs fail explicitly.

        Returns:
            ``ShramCausalLMOutput`` with:
            - ``logits`` of shape ``(batch, seq_len, vocab_size)``,
            - ``loss`` when labels are provided,
            - ``past_key_values`` as the active ``ShramCache`` or ``None``,
            - ``hidden_states`` when requested,
            - ``load_balance_loss`` from the backbone,
            - detached ``max_vio`` from the backbone.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        inputs_embeds = kwargs.pop("inputs_embeds", None)
        output_attentions = kwargs.pop("output_attentions", None)
        cache_position = kwargs.pop("cache_position", None)

        # ------------------------------------------------------------------
        # Validation zone.
        #
        # The wrapper boundary is where HuggingFace-facing inputs are judged
        # for truthfulness before any internal work begins. These checks are
        # intentionally front-loaded so the core logic below can assume one
        # coherent interpretation of the call rather than defensively checking
        # shapes, cache policy, or unsupported HF knobs at the point of use.
        # This keeps the main sequence readable while ensuring invalid states
        # fail before they can silently contaminate backbone execution.
        # ------------------------------------------------------------------
        self._validate_input_ids(input_ids)
        self._validate_attention_mask(input_ids, attention_mask)
        self._validate_position_ids(input_ids, position_ids)
        self._validate_labels(input_ids, labels)
        self._validate_cache_inputs(use_cache, past_key_values)
        self._validate_position_sources(use_cache, attention_mask, position_ids)
        self._validate_hf_boundary(
            output_attentions=output_attentions,
            return_dict=return_dict,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            extra_kwargs=kwargs,
        )

        # ------------------------------------------------------------------
        # Standardization zone.
        #
        # HuggingFace and SHRAM use different boundary conventions: generation
        # carries a full-sequence 2D attention mask, while the SHRAM backbone
        # wants a current-step active mask and concrete current position IDs.
        # This zone collapses those wrapper-facing conventions into one valid
        # backbone-facing state. After this point the core no longer reasons
        # about optional or ambiguous input forms; it works only with concrete
        # tensors whose semantics are already fixed.
        # ------------------------------------------------------------------
        full_attention_mask: torch.BoolTensor = self._standardize_full_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        current_length: int = input_ids.shape[1]
        current_active_mask: torch.BoolTensor = full_attention_mask[:, -current_length:]
        current_position_ids: torch.LongTensor = self._resolve_current_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            full_attention_mask=full_attention_mask,
        )
        shram_cache: ShramCache | None = past_key_values if use_cache else None

        # ------------------------------------------------------------------
        # Core wrapper responsibilities.
        #
        # The wrapper's primary job is kept visible here: convert token IDs to
        # embeddings, delegate transformer computation to ShramModel, project
        # hidden states back to vocabulary logits, optionally compute the
        # wrapper-level shifted next-token loss, and return the HuggingFace-
        # facing output object. The backbone remains responsible only for
        # transformer semantics; token/vocabulary/loss concerns stay here.
        # ------------------------------------------------------------------
        token_embeddings: torch.FloatTensor = self.embed_tokens(input_ids)
        backbone_outputs = self.model(
            inputs_embeds=token_embeddings,
            position_ids=current_position_ids,
            active_mask=current_active_mask,
            cache=shram_cache,
            output_hidden_states=output_hidden_states,
        )

        logits: torch.FloatTensor = self.lm_head(backbone_outputs["last_hidden_state"])

        loss: torch.FloatTensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return ShramCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=backbone_outputs["past_key_values"],
            hidden_states=backbone_outputs["hidden_states"],
            load_balance_loss=backbone_outputs["load_balance_loss"],
            max_vio=backbone_outputs["max_vio"],
        )