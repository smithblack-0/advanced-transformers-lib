"""Configuration for the SHRAM transformer.

All architectural parameters that vary across model scales or are meaningful research
variables are expressed here. Architectural constants (no bias in linear layers,
SwiGLU activation with SiLU gate) are implemented in the relevant modules and
documented at the point of use — they are not config parameters because they do not
vary and changing them produces a different architecture, not a different scale.

RoPE configuration is handled by HuggingFace's RotaryEmbeddingConfigMixin (mixed into
PretrainedConfig in transformers 5.x). rope_theta and rope_scaling are passed through
to the base class, which validates and standardises them into config.rope_parameters.
Do not bypass or duplicate this system.
"""

from transformers import PretrainedConfig


class ShramConfig(PretrainedConfig):
    """Configuration class for the SHRAM decoder-only transformer.

    SHRAM (Sparse Hybrid Token Routed Attention Mixture) replaces every standard
    attention layer with a hybrid layer H(x) = h_l(x) + h_s(x), where h_l is a
    local sliding-window causal attention path and h_s is the MoSRAH sparse routed
    path. All other components follow the Llama 3 baseline.

    This config is the single source of truth for every architectural dimension of the
    model. Nothing in the architecture may use a literal number that belongs here —
    doing so breaks the library's ability to express different model scales without
    code changes.

    RoPE scaling is handled by HuggingFace's rope system. Pass rope_scaling as a dict
    using HF's format (key is ``rope_type``, not ``type``). Supported types:
    ``"linear"``, ``"dynamic"``, ``"yarn"``, ``"longrope"``, ``"llama3"``. HF validates
    the dict and standardises it into ``config.rope_parameters``.

    Registered with HuggingFace AutoClass via ``auto_map``. Instantiate from the Hub::

        config = AutoConfig.from_pretrained(
            "your-namespace/advanced-transformers-lib",
            trust_remote_code=True,
            num_hidden_layers=12,  # override any parameter at instantiation time
        )
        model = AutoModelForCausalLM.from_config(config)

    Args:
        vocab_size: Vocabulary size. Controls the embedding table and output logits
            dimension. Must match the tokenizer.
        hidden_size: Model width ``d``. The dimension of the residual stream; all
            attention paths project into and out of this dimension.
        intermediate_size: FFN hidden dimension. Expressed directly rather than derived
            from a formula because optimal ratios vary by scale.
        num_hidden_layers: Number of transformer blocks stacked in sequence.
        num_sliding_window_heads: Number of heads in the local sliding-window attention
            path h_l. Independent of the MoSRAH path head counts. Paper uses 16.
        num_mosrah_heads: Total number of MoSRAH expert heads available ``L``. Each
            token selects ``num_selected_heads`` of these. Controls the routed
            attention capacity and directly governs the long-sequence scaling behaviour
            described by the design law (paper §3 Theory). Paper tests L=16/32/64.
        num_selected_heads: Number of MoSRAH heads each token selects ``K``. Together
            with ``num_mosrah_heads``, determines the sparsity ratio K/L.
            Paper uses K=16.
        head_dim: Per-head dimension, shared by both the local sliding-window path and
            the MoSRAH/BEA path. Specified directly — not derived from other parameters.
            Must be even (RoPE rotates dimensions in pairs). Paper uses 16.
        window_size: Sliding window size for the local attention path h_l. Each token
            attends to the previous ``window_size`` tokens. Must be passed to the
            kernel natively — not enforced by an external boolean mask (job.md
            Architecture §). Paper uses window_size=128.
        rope_mode: RoPE position encoding mode for BEA. ``"main_sequence"`` encodes
            original sequence positions (0..N-1); ``"semantic_sequence"`` encodes local
            slot indices (0, 1, 2, ...) within each head's packed representation. Both
            modes are required; the experimentally correct mode is undetermined (paper
            §4 Hyperparameter Tuning). Default ``"main_sequence"``.
        rms_norm_eps: Epsilon passed to torch.nn.RMSNorm. Prevents division by zero
            when layer activations are near zero.
        rope_theta: Base rotation frequency for RoPE. Controls how fast position angles
            rotate per dimension. Paper uses 500,000.
        max_position_embeddings: The context length the model was trained at. Used by
            HF's rope system as original_max_position_embeddings for scaling types that
            need it (yarn, longrope, llama3).
        rope_scaling: Optional RoPE scaling configuration for extending context beyond
            max_position_embeddings without retraining. Pass as a dict in HF's format
            with ``rope_type`` as the key. Paper uses YaRN extrapolation mode.
        yarn_alpha: YaRN ramp lower boundary α (paper §A.2 RoPE Treatment). Frequency
            dimensions with ``r(d) < yarn_alpha`` are fully interpolated by scale s.
            Paper value (LLaMA family): 1.0.
        yarn_beta: YaRN ramp upper boundary β (paper §A.2 RoPE Treatment). Frequency
            dimensions with ``r(d) > yarn_beta`` are left unscaled. Dimensions between
            α and β are linearly blended. Paper value (LLaMA family): 32.0.
        attention_dropout: Dropout probability applied to attention weights. Default
            0.0 for deterministic behaviour.
        use_cache: Whether the model returns past_key_values for KV caching. Set True
            for inference, may be set False during training to reduce memory pressure.
        output_hidden_states: Whether the model returns the hidden state tensor after
            each decoder layer. Default False.
        tie_word_embeddings: Whether the input embedding table and the LM head share
            weights. False for the paper's SHRAM models.
    """

    model_type = "shram"

    # auto_map tells HuggingFace which classes to instantiate when loading this config
    # with trust_remote_code=True. Paths are relative to the Hub repository root, not
    # the local src/ layout — these are the paths used after HF downloads the files.
    auto_map = {
        "AutoConfig": "configuration.ShramConfig",
        "AutoModelForCausalLM": "huggingface.ShramForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 50277,
        hidden_size: int = 512,
        intermediate_size: int = 1366,
        num_hidden_layers: int = 12,
        num_sliding_window_heads: int = 16,
        num_mosrah_heads: int = 16,
        num_selected_heads: int = 16,
        head_dim: int = 16,
        window_size: int = 128,
        rope_mode: str = "main_sequence",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 8192,
        rope_scaling: dict | None = None,
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Validate structural constraints before storing anything, so that an invalid
        # config fails loudly at construction rather than silently producing wrong
        # shapes at forward-pass time.

        # RoPE rotates dimensions in pairs. An odd head_dim has no valid pairing and
        # produces a cos/sin cache of the wrong size, causing a shape mismatch at
        # runtime. Both attention paths share head_dim so this single check covers both.
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

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_sliding_window_heads = num_sliding_window_heads
        self.num_mosrah_heads = num_mosrah_heads
        self.num_selected_heads = num_selected_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.rope_mode = rope_mode
        self.rms_norm_eps = rms_norm_eps
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache

        # For YaRN, inject yarn_alpha/yarn_beta as beta_slow/beta_fast so that HF's
        # _compute_yarn_parameters picks them up. yarn_alpha and yarn_beta are the
        # single source of truth — the dict values are derived from them.
        if rope_scaling is not None and rope_scaling.get("rope_type") == "yarn":
            rope_scaling = {**rope_scaling, "beta_slow": yarn_alpha, "beta_fast": yarn_beta}

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
