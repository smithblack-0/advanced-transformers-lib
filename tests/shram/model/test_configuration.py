"""Tests for ShramConfig.

Each test verifies a specific invariant documented in the plan. The grouping mirrors
the structure of the invariants: defaults, parameter overrides, structural validation,
rope configuration, and serialisation.

RoPE scaling validation is owned by HF's RotaryEmbeddingConfigMixin and is not tested
here — we test that our config correctly passes parameters through to HF's system.
"""

import pytest

from src.shram.model.configuration import ShramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    """Return a config with small dimensions suitable for testing.

    Using full-scale defaults in tests that don't care about scale adds noise.
    This helper applies a consistent small baseline that satisfies all structural
    constraints. Defaults match paper §4.3 proportions.
    """
    defaults = dict(
        hidden_size=512,
        num_sliding_window_heads=16,
        num_mosrah_heads=16,
        num_selected_heads=16,
        head_dim=16,
        window_size=128,
        intermediate_size=1024,
        num_hidden_layers=4,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_default_instantiation_succeeds(self):
        """Config must instantiate without arguments — all parameters have defaults."""
        config = ShramConfig()
        assert config is not None

    def test_model_type(self):
        """model_type must be unique to avoid colliding with HF's built-in 'llama'."""
        assert ShramConfig.model_type == "shram"

    def test_auto_map_present(self):
        """auto_map must be set so HuggingFace trust_remote_code can find the classes."""
        assert "AutoConfig" in ShramConfig.auto_map
        assert "AutoModelForCausalLM" in ShramConfig.auto_map

    def test_auto_map_points_to_correct_files(self):
        """auto_map paths must match the actual file layout used on the Hub."""
        assert ShramConfig.auto_map["AutoConfig"] == "configuration.ShramConfig"
        assert ShramConfig.auto_map["AutoModelForCausalLM"] == "huggingface.ShramForCausalLM"


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------

class TestParameterStorage:
    def test_vocab_size_stored(self):
        config = small_config(vocab_size=50000)
        assert config.vocab_size == 50000

    def test_num_hidden_layers_stored(self):
        config = small_config(num_hidden_layers=16)
        assert config.num_hidden_layers == 16

    def test_rope_theta_stored(self):
        config = small_config(rope_theta=10000.0)
        assert config.rope_theta == 10000.0

    def test_max_position_embeddings_stored(self):
        config = small_config(max_position_embeddings=4096)
        assert config.max_position_embeddings == 4096

    def test_num_sliding_window_heads_stored(self):
        config = small_config(num_sliding_window_heads=8)
        assert config.num_sliding_window_heads == 8

    def test_num_mosrah_heads_stored(self):
        config = small_config(num_mosrah_heads=32)
        assert config.num_mosrah_heads == 32

    def test_num_selected_heads_stored(self):
        config = small_config(num_selected_heads=8)
        assert config.num_selected_heads == 8

    def test_head_dim_stored(self):
        """head_dim is explicitly specified and stored as-is."""
        config = small_config(head_dim=32)
        assert config.head_dim == 32

    def test_window_size_stored(self):
        config = small_config(window_size=256)
        assert config.window_size == 256

    def test_rope_mode_stored(self):
        config = small_config(rope_mode="semantic_sequence")
        assert config.rope_mode == "semantic_sequence"

    def test_use_cache_stored(self):
        config = small_config(use_cache=False)
        assert config.use_cache is False

    def test_output_hidden_states_defaults_false(self):
        """output_hidden_states must default to False — opt-in, not opt-out."""
        config = small_config()
        assert config.output_hidden_states is False

    def test_output_hidden_states_stored(self):
        config = small_config(output_hidden_states=True)
        assert config.output_hidden_states is True

    def test_tie_word_embeddings_stored(self):
        config = small_config(tie_word_embeddings=True)
        assert config.tie_word_embeddings is True

    def test_yarn_alpha_stored(self):
        config = small_config(yarn_alpha=2.0)
        assert config.yarn_alpha == 2.0

    def test_yarn_beta_stored(self):
        config = small_config(yarn_beta=16.0)
        assert config.yarn_beta == 16.0


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

class TestStructuralValidation:
    def test_odd_head_dim_raises(self):
        """head_dim must be even — RoPE rotates dimensions in pairs for both paths."""
        with pytest.raises(ValueError, match="head_dim must be even"):
            small_config(head_dim=15)

    def test_invalid_rope_mode_raises(self):
        """rope_mode must be one of the two supported values."""
        with pytest.raises(ValueError, match="rope_mode"):
            small_config(rope_mode="invalid_mode")

    def test_main_sequence_rope_mode_valid(self):
        config = small_config(rope_mode="main_sequence")
        assert config.rope_mode == "main_sequence"

    def test_semantic_sequence_rope_mode_valid(self):
        config = small_config(rope_mode="semantic_sequence")
        assert config.rope_mode == "semantic_sequence"

    def test_even_head_dim_valid(self):
        config = small_config(head_dim=16)
        assert config.head_dim == 16


# ---------------------------------------------------------------------------
# RoPE configuration
# ---------------------------------------------------------------------------

class TestRopeConfiguration:
    def test_no_rope_scaling_by_default(self):
        """Without rope_scaling, HF leaves rope_parameters as None — rope_theta is used directly."""
        config = small_config()
        assert config.rope_parameters is None

    def test_linear_rope_scaling_accepted(self):
        """Linear scaling is the simplest extension method — divides all frequencies by factor."""
        config = small_config(
            rope_scaling={"rope_type": "linear", "factor": 4.0}
        )
        assert config.rope_parameters["rope_type"] == "linear"

    def test_yarn_rope_scaling_accepted(self):
        """YaRN applies frequency-aware scaling — used in the paper (§4.3 Pretraining)."""
        config = small_config(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
            }
        )
        assert config.rope_parameters["rope_type"] == "yarn"


# ---------------------------------------------------------------------------
# YaRN α/β parameters
# ---------------------------------------------------------------------------

class TestYarnParameters:
    """yarn_alpha (α) and yarn_beta (β) are the ramp boundaries from paper §A.2.

    They must be first-class config fields so every hyperparameter the architecture
    uses is explicitly tunable. They are injected into rope_parameters as beta_slow
    and beta_fast so HF's _compute_yarn_parameters sees the correct values.
    """

    def test_yarn_alpha_default(self):
        """Default yarn_alpha must be 1.0 — paper's LLaMA-family recommendation."""
        config = small_config()
        assert config.yarn_alpha == 1.0

    def test_yarn_beta_default(self):
        """Default yarn_beta must be 32.0 — paper's LLaMA-family recommendation."""
        config = small_config()
        assert config.yarn_beta == 32.0

    def test_yarn_alpha_beta_injected_into_rope_parameters(self):
        """Custom yarn_alpha/yarn_beta must appear in rope_parameters as beta_slow/beta_fast.

        HF's _compute_yarn_parameters reads beta_slow and beta_fast from rope_parameters.
        Without injection, those parameters would silently fall back to HF's own defaults
        regardless of what ShramConfig was given.
        """
        config = small_config(
            yarn_alpha=2.0,
            yarn_beta=16.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        assert config.rope_parameters["beta_slow"] == 2.0
        assert config.rope_parameters["beta_fast"] == 16.0

    def test_yarn_params_not_injected_for_non_yarn(self):
        """yarn_alpha/yarn_beta must not pollute non-YaRN rope_parameters.

        Injecting beta_slow/beta_fast into a linear or default config would be
        incorrect — those keys are meaningless outside YaRN and could confuse HF.
        """
        config = small_config(
            yarn_alpha=5.0,
            yarn_beta=64.0,
            rope_scaling={"rope_type": "linear", "factor": 4.0},
        )
        assert config.rope_parameters["rope_type"] == "linear"
        assert "beta_slow" not in config.rope_parameters
        assert "beta_fast" not in config.rope_parameters

    def test_yarn_roundtrip_preserves_alpha_beta(self):
        """yarn_alpha and yarn_beta must survive a to_dict/from_dict roundtrip."""
        original = small_config(
            yarn_alpha=2.0,
            yarn_beta=16.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        restored = ShramConfig.from_dict(original.to_dict())
        assert restored.yarn_alpha == original.yarn_alpha
        assert restored.yarn_beta == original.yarn_beta


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_roundtrip_preserves_all_fields(self):
        """A config serialised to dict and restored must be identical to the original.

        This is the HF contract for config persistence. If it breaks, save/load of
        model checkpoints will silently use wrong architectural parameters.
        """
        original = small_config(
            rope_theta=10000.0,
            max_position_embeddings=4096,
            attention_dropout=0.1,
            use_cache=False,
            num_mosrah_heads=32,
            num_selected_heads=8,
            window_size=64,
            rope_mode="semantic_sequence",
            head_dim=16,
        )
        restored = ShramConfig.from_dict(original.to_dict())

        assert restored.vocab_size == original.vocab_size
        assert restored.hidden_size == original.hidden_size
        assert restored.intermediate_size == original.intermediate_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_sliding_window_heads == original.num_sliding_window_heads
        assert restored.num_mosrah_heads == original.num_mosrah_heads
        assert restored.num_selected_heads == original.num_selected_heads
        assert restored.head_dim == original.head_dim
        assert restored.window_size == original.window_size
        assert restored.rope_mode == original.rope_mode
        assert restored.rms_norm_eps == original.rms_norm_eps
        assert restored.rope_theta == original.rope_theta
        assert restored.max_position_embeddings == original.max_position_embeddings
        assert restored.attention_dropout == original.attention_dropout
        assert restored.use_cache == original.use_cache
        assert restored.output_hidden_states == original.output_hidden_states
        assert restored.tie_word_embeddings == original.tie_word_embeddings
