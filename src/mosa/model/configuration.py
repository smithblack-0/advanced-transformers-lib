"""Configuration for the Llama 3 baseline transformer.

All architectural parameters that vary across model scales or are meaningful research
variables are expressed here. Architectural constants of Llama 3 (no bias in linear
layers, SwiGLU activation with SiLU gate) are implemented in the relevant modules and
documented at the point of use — they are not config parameters because they do not
vary across Llama 3 scales and changing them produces a different architecture, not a
different scale of this one.

RoPE configuration is handled by HuggingFace's RotaryEmbeddingConfigMixin (mixed into
PretrainedConfig in transformers 5.x). rope_theta and rope_scaling are passed through
to the base class, which validates and standardises them into config.rope_parameters.
Do not bypass or duplicate this system.
"""

from transformers import PretrainedConfig


class MosaConfig(PretrainedConfig):
    """Configuration class for the Llama 3 baseline decoder-only transformer.

    This config is the single source of truth for every architectural dimension of the
    model. Nothing in the architecture may use a literal number that belongs here —
    doing so breaks the library's ability to express different model scales without
    code changes.

    RoPE scaling is handled by HuggingFace's rope system. Pass rope_scaling as a dict
    using HF's format (key is ``rope_type``, not ``type``). Supported types:
    ``"linear"``, ``"dynamic"``, ``"yarn"``, ``"longrope"``, ``"mosa"``. HF validates
    the dict and standardises it into ``config.rope_parameters``.

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
        max_position_embeddings: The context length the model was trained at. Used by
            HF's rope system as original_max_position_embeddings for scaling types that
            need it (yarn, longrope, mosa). This is the training context length, not
            an inference ceiling — the rope module handles longer sequences at runtime
            via lazy cache extension. Llama 3 base training context: 8192.
        rope_scaling: Optional RoPE scaling configuration for extending context beyond
            max_position_embeddings without retraining. Pass as a dict in HF's format
            with ``rope_type`` as the key. HF's RotaryEmbeddingConfigMixin validates
            and stores this. None means no scaling (default RoPE behaviour).
        attention_dropout: Dropout probability applied to attention weights. Default
            0.0 for deterministic behaviour.
        use_cache: Whether the model returns past_key_values for KV caching. Set True
            for inference, may be set False during training to reduce memory pressure.
        output_hidden_states: Whether the model returns the hidden state tensor after
            each decoder layer. Useful for probing or intermediate representation
            extraction. Default False.
        tie_word_embeddings: Whether the input embedding table and the LM head share
            weights. False for Llama 3.
    """

    model_type = "mosa_baseline"

    # auto_map tells HuggingFace which classes to instantiate when loading this config
    # with trust_remote_code=True. Paths are relative to the Hub repository root, not
    # the local src/ layout — these are the paths used after HF downloads the files.
    auto_map = {
        "AutoConfig": "configuration.MosaConfig",
        "AutoModelForCausalLM": "huggingface.MosaForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 50277,
        hidden_size: int = 768,
        intermediate_size: int = 1568,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 8192,
        rope_scaling: dict | None = None,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        output_hidden_states: bool = False,
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

        # RoPE rotates dimensions in pairs. An odd head_dim has no valid pairing and
        # produces a cos/sin cache of size head_dim+1 (torch.arange(0, odd, 2) rounds
        # up), causing a shape mismatch at runtime. Catch it here rather than at
        # forward-pass time.
        resolved_head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        if resolved_head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even (RoPE rotates dimensions in pairs). "
                f"Got head_dim={resolved_head_dim} from hidden_size={hidden_size} "
                f"and num_attention_heads={num_attention_heads}."
            )

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
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        # rope_theta, max_position_embeddings, and rope_scaling are passed to HF's
        # base class, which owns rope configuration via RotaryEmbeddingConfigMixin.
        # HF validates rope_scaling and standardises everything into rope_parameters.
        # Do not store or validate these ourselves.
        super().__init__(
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            rope_scaling=rope_scaling,
            tie_word_embeddings=tie_word_embeddings,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        # Promote auto_map to an instance attribute so PretrainedConfig.to_dict()
        # serialises it into config.json. Class-level attributes are not picked up
        # by to_dict() — only self.__dict__ is serialised. model_type is the sole
        # exception handled specially by HF; auto_map is not.
        self.auto_map = type(self).auto_map
