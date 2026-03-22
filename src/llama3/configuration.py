"""Configuration for the Llama 3 baseline transformer.

All architectural parameters that vary across model scales or are meaningful research
variables are expressed here. Architectural constants of Llama 3 (no bias in linear
layers, SwiGLU activation with SiLU gate) are implemented in the relevant modules and
documented at the point of use — they are not config parameters because they do not
vary across Llama 3 scales and changing them produces a different architecture, not a
different scale of this one.
"""

from transformers import PretrainedConfig


class Llama3Config(PretrainedConfig):
    """Configuration class for the Llama 3 baseline decoder-only transformer.

    This config is the single source of truth for every architectural dimension of the
    model. Nothing in the architecture may use a literal number that belongs here —
    doing so breaks the library's ability to express different model scales without
    code changes.

    Defaults correspond to the Llama 3.1 8B scale and are provided as a concrete
    reference point, not as an implicit assumption. Every parameter must be set
    explicitly when targeting a different scale.

    Registered with HuggingFace AutoClass via ``auto_map``. Instantiate from the Hub::

        config = AutoConfig.from_pretrained(
            "your-namespace/advanced-transformers-lib",
            trust_remote_code=True,
            num_hidden_layers=16,  # override any parameter at instantiation time
        )
        model = AutoModelForCausalLM.from_config(config)

    Args:
        vocab_size: Vocabulary size. Controls the embedding table and output logits
            dimension. Must match the tokenizer.
        hidden_size: Model width. The central dimension from which all others are
            derived or to which they project.
        intermediate_size: FFN hidden dimension. Expressed directly rather than derived
            from a formula because Llama 3 ratios vary by scale (~3.5x at 8B/70B,
            ~3.25x at 405B). A formula would be wrong for at least some scales.
        num_hidden_layers: Number of transformer blocks stacked in sequence.
        num_attention_heads: Number of query heads. Determines how hidden_size is
            partitioned per head.
        num_key_value_heads: Number of KV heads for Grouped Query Attention. Must
            evenly divide num_attention_heads. Setting equal to num_attention_heads
            gives standard MHA; setting to 1 gives MQA; values between give GQA.
            Llama 3 uses 8 at all scales, motivated by KV cache memory at 128K context.
        head_dim: Dimension per attention head. Normally hidden_size //
            num_attention_heads, but exposed as a parameter for architectures that
            decouple head count from head size. Computed automatically if None.
        rms_norm_eps: Epsilon passed to torch.nn.RMSNorm. Prevents division by zero
            when layer activations are near zero.
        rope_theta: Base rotation frequency for RoPE. Controls how fast position angles
            rotate per dimension — higher values mean slower rotation, preventing
            positional aliasing at long sequence distances. Llama 3 uses 500,000
            (vs ~10,000 typical) as a prerequisite for 128K context support. This
            value has physical meaning tied to the target context length and must
            never be hardcoded in the architecture.
        rope_scaling: Optional configuration for RoPE frequency scaling, enabling
            context extension beyond the training length without retraining. See
            ``_validate_rope_scaling`` for the expected dict structure. None means
            no scaling is applied.
        attention_dropout: Dropout probability applied to attention weights. Default
            0.0 for deterministic behaviour.
        use_cache: Whether the model returns past_key_values for KV caching. Set True
            for inference, may be set False during training to reduce memory pressure.
        tie_word_embeddings: Whether the input embedding table and the LM head share
            weights. False for Llama 3.
    """

    model_type = "llama3_baseline"

    # auto_map tells HuggingFace which classes to instantiate when loading this config
    # with trust_remote_code=True. Paths are relative to the Hub repository root, not
    # the local src/ layout — these are the paths used after HF downloads the files.
    auto_map = {
        "AutoConfig": "configuration.Llama3Config",
        "AutoModelForCausalLM": "model.Llama3ForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Validate structural constraints before storing anything, so that an invalid
        # config fails loudly at construction rather than silently producing wrong
        # shapes at forward-pass time.
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})."
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads}). GQA requires query "
                f"heads to divide evenly across KV head groups."
            )
        if rope_scaling is not None:
            _validate_rope_scaling(rope_scaling)

        # head_dim is normally hidden_size // num_attention_heads but is exposed as a
        # parameter for architectures that decouple head count from head size.
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _validate_rope_scaling(rope_scaling: dict) -> None:
    """Validate the rope_scaling configuration dict.

    rope_scaling must contain 'type' and 'factor'. 'type' selects the scaling
    algorithm; 'factor' is the context extension multiplier and must be > 1.0
    (a factor <= 1.0 does not extend context and is not a valid use of scaling).

    Raises:
        ValueError: If required keys are missing, the type is unsupported, or the
            factor is out of range.
    """
    required_keys = {"type", "factor"}
    missing = required_keys - rope_scaling.keys()
    if missing:
        raise ValueError(
            f"rope_scaling is missing required keys: {missing}. "
            f"Expected at minimum: {required_keys}."
        )

    supported_types = {"linear", "yarn"}
    if rope_scaling["type"] not in supported_types:
        raise ValueError(
            f"rope_scaling 'type' must be one of {supported_types}, "
            f"got '{rope_scaling['type']}'."
        )

    factor = rope_scaling["factor"]
    if not isinstance(factor, (int, float)) or factor <= 1.0:
        raise ValueError(
            f"rope_scaling 'factor' must be a number > 1.0 (values <= 1.0 do not "
            f"extend context), got {factor}."
        )
