
"""Tests for RotaryEmbedding.

Each test verifies a specific invariant documented in the plan. The grouping mirrors
the invariant categories: shape, mathematical correctness, configuration sensitivity,
lazy cache extension, and rope type compatibility.

The relative position property is the core correctness guarantee of RoPE: the inner
product of a rotated query at position i with a rotated key at position j depends only
on (i - j), not on i and j individually. This is what makes RoPE encode relative rather
than absolute position.
"""

import pytest
import torch

from src.llama3.model.configuration import Llama3Config
from src.llama3.model.rope import RotaryEmbedding, _rotate_half


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> Llama3Config:
    """Config with small dimensions sufficient to exercise rope without scale noise."""
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
    )
    defaults.update(kwargs)
    return Llama3Config(**defaults)


def random_qk(
    config: Llama3Config,
    batch: int = 2,
    seq: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (q, k, position_ids) matching the config's head layout."""
    head_dim = config.head_dim
    q = torch.randn(batch, config.num_attention_heads, seq, head_dim)
    k = torch.randn(batch, config.num_key_value_heads, seq, head_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    return q, k, position_ids


# ---------------------------------------------------------------------------
# Shape invariants
# ---------------------------------------------------------------------------

class TestShapePreservation:
    def test_query_shape_preserved(self):
        """Rotation must not change the shape of the query tensor."""
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config)
        q_rot, _, _ = rope(q, k, position_ids)
        assert q_rot.shape == q.shape

    def test_key_shape_preserved(self):
        """Rotation must not change the shape of the key tensor."""
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config)
        _, k_rot, _ = rope(q, k, position_ids)
        assert k_rot.shape == k.shape

    def test_gqa_different_head_counts(self):
        """GQA layout (num_kv_heads < num_heads) produces correct output shapes."""
        config = small_config(hidden_size=128, num_attention_heads=8, num_key_value_heads=2)
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config, seq=16)
        q_rot, k_rot, _ = rope(q, k, position_ids)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:
    def test_identity_at_position_zero(self):
        """At position 0 the rotation is the identity: output equals input.

        cos(0) = 1 and sin(0) = 0 for all frequency components, so
        x * cos + rotate_half(x) * sin = x * 1 + rotate_half(x) * 0 = x.
        """
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                               rope_theta=10000.0)
        rope = RotaryEmbedding(config)
        batch, seq = 1, 4
        q = torch.randn(batch, config.num_attention_heads, seq, config.head_dim)
        k = torch.randn(batch, config.num_key_value_heads, seq, config.head_dim)
        position_ids = torch.zeros(batch, seq, dtype=torch.long)

        q_rot, k_rot, _ = rope(q, k, position_ids)

        torch.testing.assert_close(q_rot, q)
        torch.testing.assert_close(k_rot, k)

    def test_relative_position_property(self):
        """Inner product of rotated q and k depends only on relative offset, not absolute positions.

        <RoPE(q, i), RoPE(k, j)> == <RoPE(q, i'), RoPE(k, j')> whenever i - j == i' - j'.
        Tested with two pairs sharing the same relative offset of 2.
        """
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
                               rope_theta=10000.0)
        rope = RotaryEmbedding(config)
        torch.manual_seed(42)
        # Single head, single batch for a clean dot-product test.
        q = torch.randn(1, 1, 1, config.head_dim)
        k = torch.randn(1, 1, 1, config.head_dim)

        def dot_at(pos_q: int, pos_k: int) -> float:
            pid_q = torch.tensor([[pos_q]])
            pid_k = torch.tensor([[pos_k]])
            q_rot, _, _ = rope(q, q, pid_q)
            k_rot, _, _ = rope(k, k, pid_k)
            return (q_rot * k_rot).sum().item()

        # Pairs (3, 1) and (5, 3) both have relative offset 2.
        assert abs(dot_at(3, 1) - dot_at(5, 3)) < 1e-5

    def test_rotate_half_shape_preserved(self):
        """_rotate_half must not change the tensor shape."""
        x = torch.randn(2, 4, 8, 16)
        assert _rotate_half(x).shape == x.shape

    def test_rotate_half_twice_is_negation(self):
        """Applying _rotate_half twice negates the input.

        rotate_half([x1, x2]) = [-x2, x1]
        rotate_half([-x2, x1]) = [-x1, -x2] = -[x1, x2]
        """
        x = torch.randn(2, 4, 8, 16)
        torch.testing.assert_close(_rotate_half(_rotate_half(x)), -x)


# ---------------------------------------------------------------------------
# Configuration sensitivity
# ---------------------------------------------------------------------------

class TestConfigurationSensitivity:
    def test_rope_theta_affects_output(self):
        """Different rope_theta values must produce different rotated outputs."""
        config_a = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                                rope_theta=10000.0)
        config_b = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                                rope_theta=500000.0)
        torch.manual_seed(0)
        q, k, position_ids = random_qk(config_a, seq=4)

        q_rot_a, _, _ = RotaryEmbedding(config_a)(q, k, position_ids)
        q_rot_b, _, _ = RotaryEmbedding(config_b)(q, k, position_ids)

        assert not torch.allclose(q_rot_a, q_rot_b)

    def test_default_attention_scaling_is_one(self):
        """Standard RoPE (no scaling) must return attention_scaling == 1.0."""
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config)
        _, _, scaling = rope(q, k, position_ids)
        assert scaling == 1.0

    def test_unsupported_rope_type_raises(self):
        """Unsupported rope types must raise NotImplementedError at construction time."""
        config = small_config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_scaling={"rope_type": "longrope", "factor": 4.0,
                          "original_max_position_embeddings": 512},
        )
        with pytest.raises(NotImplementedError, match="longrope"):
            RotaryEmbedding(config)


# ---------------------------------------------------------------------------
# Lazy cache extension
# ---------------------------------------------------------------------------

class TestLazyCacheExtension:
    def test_short_then_long_sequence_consistent(self):
        """Positions 0..3 must produce identical output whether in a short or extended pass."""
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                               rope_theta=10000.0)
        rope = RotaryEmbedding(config)
        torch.manual_seed(7)
        q_short = torch.randn(1, config.num_attention_heads, 4, config.head_dim)
        k_short = torch.randn(1, config.num_key_value_heads, 4, config.head_dim)
        pos_short = torch.arange(4).unsqueeze(0)

        q_long = torch.randn(1, config.num_attention_heads, 16, config.head_dim)
        k_long = torch.randn(1, config.num_key_value_heads, 16, config.head_dim)
        q_long[:, :, :4, :] = q_short
        k_long[:, :, :4, :] = k_short
        pos_long = torch.arange(16).unsqueeze(0)

        q_rot_short, k_rot_short, _ = rope(q_short, k_short, pos_short)
        q_rot_long, k_rot_long, _ = rope(q_long, k_long, pos_long)

        torch.testing.assert_close(q_rot_short, q_rot_long[:, :, :4, :])
        torch.testing.assert_close(k_rot_short, k_rot_long[:, :, :4, :])

    def test_cache_length_grows(self):
        """After processing a longer sequence the cos/sin cache must cover all positions."""
        config = small_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config, seq=32)
        rope(q, k, position_ids)
        assert rope._cos_cached is not None
        assert rope._cos_cached.shape[0] >= 32


# ---------------------------------------------------------------------------
# Rope type compatibility
# ---------------------------------------------------------------------------

class TestRopeTypeCompatibility:
    def test_linear_scaling_runs(self):
        """Linear RoPE scaling must run without error and preserve output shapes."""
        config = small_config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_scaling={"rope_type": "linear", "factor": 4.0},
        )
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config, seq=8)
        q_rot, k_rot, _ = rope(q, k, position_ids)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_yarn_scaling_runs(self):
        """YaRN scaling must run without error and return attention_scaling != 1.0."""
        config = small_config(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 512,
            },
        )
        rope = RotaryEmbedding(config)
        q, k, position_ids = random_qk(config, seq=8)
        q_rot, k_rot, scaling = rope(q, k, position_ids)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert scaling != 1.0
