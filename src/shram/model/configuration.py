"""Configuration for the SHRAM transformer.

All architectural parameters that vary across model scales or are meaningful research
variables are expressed here. Architectural constants (no bias in linear layers,
SwiGLU activation with SiLU gate) are implemented in the relevant modules and
documented at the point of use — they are not config parameters because they do not
vary and changing them produces a different architecture, not a different scale.

RoPE configuration is owned entirely by this config. Each attention path reads its
parameters directly and constructs its own RotaryEmbedding instance explicitly — no
HuggingFace rope infrastructure is used. See Unit 5.A design decisions in plan.md.
"""

import math

from transformers import PretrainedConfig


class ShramConfig(PretrainedConfig):
    """Configuration class for the SHRAM decoder-only transformer.

    SHRAM (Sparse Hybrid Token Routed Attention Mixture) replaces every standard
    attention layer with a hybrid layer H(x) = h_l(x) + h_s(x), where h_l is a
    local sliding-window causal attention path and h_s is the MoSRAH sparse routed
    path. All other components follow the Llama 3 baseline.

    This config is the single source of truth for every architectural dimension of the
    model. Nothing in the architecture may use a literal number that belongs here.

    Two independent RoPE configurations exist — one per attention path:

    - h_l always uses standard RoPE with ``local_rope_theta``.
    - BEA always uses YaRN with ``mosrah_rope_theta``, ``training_sequence_length``,
      ``inference_sequence_length``, ``alpha``, and ``beta``. When
      ``inference_sequence_length == training_sequence_length`` the YaRN scale factor
      ``s = 1`` and YaRN reduces exactly to standard RoPE — this is the default state
      and the correct setting for experiments that do not require context extension.

    Registered with HuggingFace AutoClass via ``auto_map``. Instantiate from the Hub::

        config = AutoConfig.from_pretrained(
            "your-namespace/advanced-transformers-lib",
            trust_remote_code=True,
            num_decoder_layers=12,
        )
        model = AutoModelForCausalLM.from_config(config)

    Args:
        vocab_size: Vocabulary size. Controls the embedding table and output logits
            dimension. Must match the tokenizer.
        embedding_width: Model width ``d``. The dimension of the residual stream.
        mlp_width: FFN hidden dimension.
        num_decoder_layers: Number of transformer blocks stacked in sequence.
        num_sliding_window_heads: Number of heads in the local sliding-window path h_l.
        num_mosrah_heads: Total MoSRAH expert heads available ``L``.
        num_selected_heads: MoSRAH heads each token selects ``K``.
        head_dim: Per-head dimension, shared by both attention paths. Must be even
            (RoPE rotates dimensions in pairs). Paper uses 16.
        window_size: Sliding window size for h_l. Paper uses 128.
        rope_mode: RoPE position encoding mode for BEA. ``"main_sequence"`` supplies
            original sequence positions; ``"semantic_sequence"`` supplies local slot
            indices. Both are required; experimentally correct mode is undetermined
            (paper §4). Default ``"main_sequence"``.
        rms_norm_eps: Epsilon for RMSNorm layers.
        local_rope_theta: RoPE base frequency ``b`` for the local attention path h_l.
            Paper uses b=10000.
        mosrah_rope_theta: RoPE base frequency ``b`` for the BEA path. Paper uses
            b=10000.
        training_sequence_length: Context length ``C_train`` the model was or will be
            trained at. Used to compute the YaRN scale factor for BEA.
        inference_sequence_length: Context length ``C_target`` the model must support
            at inference. Optional; defaults to ``training_sequence_length`` so that
            ``scale=1`` and YaRN reduces to standard RoPE unless explicitly extended.
        alpha: YaRN ramp lower boundary α (paper §A.2). Frequency dimensions with
            ``r(d) < alpha`` are fully interpolated by scale s. Paper value: 1.0.
        beta: YaRN ramp upper boundary β (paper §A.2). Frequency dimensions with
            ``r(d) > beta`` are left unscaled. Paper value: 32.0.
        attention_dropout: Dropout probability on attention weights. Default 0.0.
        use_cache: Whether to return past_key_values for KV caching.
        output_hidden_states: Whether to return hidden states after each layer.
        tie_word_embeddings: Whether input embedding and LM head share weights.
        use_residual_gate: When True, each DecoderLayer gates its residual contributions
            with a learnable scalar parameter (init: zero). When False, uses a fixed
            ``1/√num_decoder_layers`` scale instead, which preserves O(1) residual
            variance at depth with no learnable gate. Default True.
    """

    model_type = "shram"

    auto_map = {
        "AutoConfig": "configuration.ShramConfig",
        "AutoModelForCausalLM": "huggingface.ShramForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 50277,
        embedding_width: int = 512,
        mlp_width: int = 1366,
        num_decoder_layers: int = 12,
        num_sliding_window_heads: int = 16,
        num_mosrah_heads: int = 16,
        num_selected_heads: int = 16,
        head_dim: int = 16,
        window_size: int = 128,
        rope_mode: str = "main_sequence",
        rms_norm_eps: float = 1e-5,
        local_rope_theta: float = 10000.0,
        mosrah_rope_theta: float = 10000.0,
        training_sequence_length: int = 1024,
        inference_sequence_length: int | None = None,
        alpha: float = 1.0,
        beta: float = 32.0,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        tie_word_embeddings: bool = False,
        use_residual_gate: bool = True,
        **kwargs
    ):
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even (RoPE rotates dimensions in pairs). "
                f"Got head_dim={head_dim}."
            )

        if rope_mode not in {"main_sequence", "semantic_sequence"}:
            raise ValueError(
                f"rope_mode must be 'main_sequence' or 'semantic_sequence', "
                f"got '{rope_mode}'."
            )

        if training_sequence_length <= 0:
            raise ValueError(
                f"training_sequence_length must be positive, "
                f"got {training_sequence_length}."
            )

        if inference_sequence_length is None:
            inference_sequence_length = training_sequence_length
        if inference_sequence_length <= 0:
            raise ValueError(
                f"inference_sequence_length must be positive, "
                f"got {inference_sequence_length}."
            )

        if num_mosrah_heads % num_selected_heads != 0:
            raise ValueError(
                f"num_mosrah_heads must be exactly divisible by num_selected_heads. "
                f"Mechanical load balancing partitions the sequence into blocks of "
                f"W = num_mosrah_heads // num_selected_heads tokens; each block covers "
                f"every expert exactly once, which requires an integer W. "
                f"Got num_mosrah_heads={num_mosrah_heads}, num_selected_heads={num_selected_heads}."
            )

        self.vocab_size = vocab_size
        self.embedding_width = embedding_width
        self.mlp_width = mlp_width
        self.num_decoder_layers = num_decoder_layers
        self.num_sliding_window_heads = num_sliding_window_heads
        self.num_mosrah_heads = num_mosrah_heads
        self.num_selected_heads = num_selected_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.rope_mode = rope_mode
        self.rms_norm_eps = rms_norm_eps
        self.local_rope_theta = local_rope_theta
        self.mosrah_rope_theta = mosrah_rope_theta
        self.training_sequence_length = training_sequence_length
        self.inference_sequence_length = inference_sequence_length
        self.alpha = alpha
        self.beta = beta
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.use_residual_gate = use_residual_gate

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        # Promote auto_map to an instance attribute so PretrainedConfig.to_dict()
        # serialises it into config.json.
        self.auto_map = type(self).auto_map

    @property
    def scale(self) -> float:
        """YaRN context extension scale factor s = inference_sequence_length / training_sequence_length.

        When scale == 1.0, YaRN reduces exactly to standard RoPE — all frequency
        adjustments cancel and A_rope = 1. This is the default state.
        """
        return self.inference_sequence_length / self.training_sequence_length

    @property
    def mosrah_packed_length(self) -> int:
        """Static packed time dimension T for expert packing.

        Mechanical load balancing guarantees exactly
        ``training_sequence_length * num_selected_heads / num_mosrah_heads``
        tokens per expert. The ceiling handles non-integer results when
        training_sequence_length is not divisible by the block length W.

        All consumers of the packed buffer size must read this property rather
        than deriving T independently.
        """
        return math.ceil(
            self.training_sequence_length
            * self.num_selected_heads
            / self.num_mosrah_heads
        ) + self.block_length

    @property
    def mosrah_cache_length(self) -> int:
        """Static per-(batch, head) slot capacity for the MoSRAH inference cache.

        Mechanical load balancing guarantees exactly
        ``inference_sequence_length * num_selected_heads / num_mosrah_heads``
        tokens per expert over the full inference context. The ceiling handles
        non-integer results when inference_sequence_length is not divisible by
        the block length W.

        Distinct from ``mosrah_packed_length``, which sizes the training packing
        buffer using ``training_sequence_length``. This property uses
        ``inference_sequence_length`` because the cache must hold the full
        accumulated token history across the entire inference run.

        All consumers of the MoSRAH cache buffer size must read this property
        rather than deriving the capacity independently.
        """
        return math.ceil(
            self.inference_sequence_length
            * self.num_selected_heads
            / self.num_mosrah_heads
        ) + self.block_length

    @property
    def block_length(self) -> int:
        """Routing block length W = num_mosrah_heads // num_selected_heads.

        Within each block of W consecutive tokens every expert is used exactly once,
        giving perfect load balance by construction. The E % K == 0 constraint
        enforced at construction guarantees W is an exact integer.

        All consumers of the routing block length must read this property rather
        than deriving W independently.
        """
        return self.num_mosrah_heads // self.num_selected_heads
