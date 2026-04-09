"""Tests for RotaryEmbedding.

Invariant groups:

1. Shape — output dimensions are correct regardless of q/k layout.
2. Mathematical correctness — identity at position 0, relative position property,
   rotate_half properties.
3. Mode behaviour — default mode produces A_rope=1; yarn mode produces the paper's
   frequency schedule and A_rope formula; s=1 makes yarn identical to default.
4. Parameter validation — unsupported mode raises NotImplementedError; missing yarn
   parameters raise ValueError.
5. Cache extension — the cache grows lazily and covers the maximum position seen.
6. Cache sharing — instances with identical parameters reference the same cached tensors.
7. Arbitrary position shape — forward accepts 2D and 3D position_ids correctly.
"""

import math

import pytest
import torch

from src.shram.model.rope import RotaryEmbedding, _rotate_half


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEAD_DIM = 16
THETA = 10000.0


def default_rope(**kwargs) -> RotaryEmbedding:
    """Construct a default-mode RotaryEmbedding with small test dimensions."""
    params = dict(mode="default", head_dim=HEAD_DIM, theta=THETA)
    params.update(kwargs)
    return RotaryEmbedding(**params)


def yarn_rope(s: float = 2.0, **kwargs) -> RotaryEmbedding:
    """Construct a yarn-mode RotaryEmbedding with scale factor s."""
    params = dict(
        mode="yarn",
        head_dim=HEAD_DIM,
        theta=THETA,
        initial_seq_length=512,
        dilation=s,
        alpha=1.0,
        beta=32.0,
    )
    params.update(kwargs)
    return RotaryEmbedding(**params)


def random_qk(
    batch: int = 2,
    num_heads: int = 4,
    seq: int = 8,
    head_dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (q, k, position_ids) for standard 2D layout."""
    q = torch.randn(batch, num_heads, seq, head_dim)
    k = torch.randn(batch, num_heads, seq, head_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    return q, k, position_ids


# ---------------------------------------------------------------------------
# 1. Shape invariants
# ---------------------------------------------------------------------------

class TestShapePreservation:
    def test_query_shape_preserved(self):
        """Rotation must not change the shape of the query tensor."""
        rope = default_rope()
        q, k, pos = random_qk()
        q_rot, _, _ = rope(q, k, pos)
        assert q_rot.shape == q.shape

    def test_key_shape_preserved(self):
        """Rotation must not change the shape of the key tensor."""
        rope = default_rope()
        q, k, pos = random_qk()
        _, k_rot, _ = rope(q, k, pos)
        assert k_rot.shape == k.shape

    def test_different_head_counts(self):
        """Different q and k head counts produce correct output shapes."""
        rope = default_rope()
        q = torch.randn(2, 8, 6, HEAD_DIM)
        k = torch.randn(2, 2, 6, HEAD_DIM)
        pos = torch.arange(6).unsqueeze(0).expand(2, -1)
        q_rot, k_rot, _ = rope(q, k, pos)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# 2. Mathematical correctness
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:
    def test_identity_at_position_zero(self):
        """At position 0 the rotation is the identity: output equals input.

        cos(0) = 1 and sin(0) = 0 for all frequency components, so
        x * cos + rotate_half(x) * sin = x.
        """
        rope = default_rope()
        batch, seq = 1, 4
        q = torch.randn(batch, 4, seq, HEAD_DIM)
        k = torch.randn(batch, 4, seq, HEAD_DIM)
        position_ids = torch.zeros(batch, seq, dtype=torch.long)

        q_rot, k_rot, _ = rope(q, k, position_ids)

        torch.testing.assert_close(q_rot, q)
        torch.testing.assert_close(k_rot, k)

    def test_relative_position_property(self):
        """Inner product of rotated q and k depends only on relative offset, not absolute positions.

        <RoPE(q, i), RoPE(k, j)> == <RoPE(q, i'), RoPE(k, j')> whenever i - j == i' - j'.
        """
        rope = default_rope()
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, HEAD_DIM)
        k = torch.randn(1, 1, 1, HEAD_DIM)

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
        x = torch.randn(2, 4, 8, HEAD_DIM)
        assert _rotate_half(x).shape == x.shape

    def test_rotate_half_twice_is_negation(self):
        """Applying _rotate_half twice negates the input.

        rotate_half([x1, x2]) = [-x2, x1]
        rotate_half([-x2, x1]) = [-x1, -x2] = -[x1, x2]
        """
        x = torch.randn(2, 4, 8, HEAD_DIM)
        torch.testing.assert_close(_rotate_half(_rotate_half(x)), -x)


# ---------------------------------------------------------------------------
# 3. Mode behaviour
# ---------------------------------------------------------------------------

class TestModeBehaviour:
    def test_default_attention_scaling_is_one(self):
        """Default mode must return attention_scaling == 1.0."""
        rope = default_rope()
        q, k, pos = random_qk()
        _, _, scaling = rope(q, k, pos)
        assert scaling == 1.0

    def test_yarn_attention_scaling_formula(self):
        """YaRN mode must return A_rope = (0.1·ln(s)+1)² per paper §A.2."""
        s = 4.0
        rope = yarn_rope(s=s)
        q, k, pos = random_qk()
        _, _, scaling = rope(q, k, pos)
        expected = (0.1 * math.log(s) + 1.0) ** 2
        assert abs(scaling - expected) < 1e-6

    def test_yarn_scale_one_matches_default(self):
        """When s=1, YaRN must produce identical output to default mode.

        s=1 means inference_seq_len == training_seq_len. The ramp collapses:
        θ_d' = (1-γ)·θ_d/1 + γ·θ_d = θ_d, and A_rope = (0.1·ln(1)+1)² = 1.
        """
        rope_default = default_rope()
        rope_yarn_s1 = yarn_rope(s=1.0)

        torch.manual_seed(0)
        q, k, pos = random_qk(seq=6)

        q_def, k_def, scale_def = rope_default(q, k, pos)
        q_yarn, k_yarn, scale_yarn = rope_yarn_s1(q, k, pos)

        torch.testing.assert_close(q_def, q_yarn, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_def, k_yarn, atol=1e-5, rtol=1e-5)
        assert abs(scale_def - scale_yarn) < 1e-6

    def test_yarn_different_from_default_when_s_gt_one(self):
        """YaRN with s > 1 must produce different output to default mode."""
        rope_default = default_rope()
        rope_yarn = yarn_rope(s=4.0)

        torch.manual_seed(1)
        q, k, pos = random_qk(seq=6)

        q_def, _, _ = rope_default(q, k, pos)
        q_yarn, _, _ = rope_yarn(q, k, pos)

        assert not torch.allclose(q_def, q_yarn)

    def test_theta_affects_output(self):
        """Different theta values must produce different rotated outputs."""
        rope_a = default_rope(theta=10000.0)
        rope_b = default_rope(theta=500000.0)

        torch.manual_seed(0)
        q, k, pos = random_qk(seq=4)

        q_a, _, _ = rope_a(q, k, pos)
        q_b, _, _ = rope_b(q, k, pos)

        assert not torch.allclose(q_a, q_b)

    def test_yarn_frequency_formula(self):
        """YaRN rotation_freqs must match paper §A.2 equations exactly.

        Verifies the frequency tensor against a reference implementation of
        θ_d' = (1−γ)·θ_d/s + γ·θ_d where γ = clamp((r−α)/(β−α), 0, 1)
        and r = C_train·θ_d / (2π).
        """
        training, s, alpha, beta = 512, 4.0, 1.0, 32.0
        rope = RotaryEmbedding(
            mode="yarn", head_dim=HEAD_DIM, theta=THETA,
            initial_seq_length=training, dilation=s,
            alpha=alpha, beta=beta,
        )

        d_index = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
        base_freqs = 1.0 / (THETA ** (d_index / HEAD_DIM))
        r = training * base_freqs / (2.0 * math.pi)
        gamma = ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)
        expected_freqs = (1.0 - gamma) * (base_freqs / s) + gamma * base_freqs

        torch.testing.assert_close(rope.rotation_freqs, expected_freqs)


# ---------------------------------------------------------------------------
# 4. Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_unknown_mode_raises(self):
        """Unsupported mode must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not supported"):
            RotaryEmbedding(mode="longrope", head_dim=HEAD_DIM, theta=THETA)

    def test_yarn_missing_initial_seq_length_raises(self):
        with pytest.raises(ValueError, match="initial_seq_length"):
            RotaryEmbedding(
                mode="yarn", head_dim=HEAD_DIM, theta=THETA,
                dilation=4.0, alpha=1.0, beta=32.0,
            )

    def test_yarn_missing_dilation_raises(self):
        with pytest.raises(ValueError, match="dilation"):
            RotaryEmbedding(
                mode="yarn", head_dim=HEAD_DIM, theta=THETA,
                initial_seq_length=512, alpha=1.0, beta=32.0,
            )

    def test_yarn_missing_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            RotaryEmbedding(
                mode="yarn", head_dim=HEAD_DIM, theta=THETA,
                initial_seq_length=512, dilation=4.0, beta=32.0,
            )

    def test_yarn_missing_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            RotaryEmbedding(
                mode="yarn", head_dim=HEAD_DIM, theta=THETA,
                initial_seq_length=512, dilation=4.0, alpha=1.0,
            )


# ---------------------------------------------------------------------------
# 5. Cache extension
# ---------------------------------------------------------------------------

class TestCacheExtension:
    def test_short_then_long_sequence_consistent(self):
        """Positions 0..3 must produce identical output whether computed in a short or long pass."""
        rope = default_rope()
        torch.manual_seed(7)
        q_short = torch.randn(1, 4, 4, HEAD_DIM)
        k_short = torch.randn(1, 4, 4, HEAD_DIM)
        pos_short = torch.arange(4).unsqueeze(0)

        q_long = torch.randn(1, 4, 16, HEAD_DIM)
        k_long = torch.randn(1, 4, 16, HEAD_DIM)
        q_long[:, :, :4, :] = q_short
        k_long[:, :, :4, :] = k_short
        pos_long = torch.arange(16).unsqueeze(0)

        q_rot_short, k_rot_short, _ = rope(q_short, k_short, pos_short)
        q_rot_long, k_rot_long, _ = rope(q_long, k_long, pos_long)

        torch.testing.assert_close(q_rot_short, q_rot_long[:, :, :4, :])
        torch.testing.assert_close(k_rot_short, k_rot_long[:, :, :4, :])

    def test_cache_length_covers_max_position(self):
        """After processing a sequence the cache must cover all positions seen."""
        rope = default_rope()
        q, k, pos = random_qk(seq=32)
        rope(q, k, pos)
        assert rope._cos_cached is not None
        assert rope._cos_cached.shape[0] >= 32


# ---------------------------------------------------------------------------
# 6. Cache sharing
# ---------------------------------------------------------------------------

class TestCacheSharing:
    def test_identical_instances_share_cache_tensor(self):
        """Two instances with identical parameters must reference the same cached tensor.

        This verifies the class-level registry works: only one cos/sin table is
        built regardless of how many layers share the same parametrisation.
        """
        rope_a = default_rope()
        rope_b = default_rope()

        q, k, pos = random_qk(seq=8)
        rope_a(q, k, pos)
        rope_b(q, k, pos)

        # Both instances must point to the same tensor object in memory.
        assert rope_a._cos_cached is rope_b._cos_cached
        assert rope_a._sin_cached is rope_b._sin_cached

    def test_different_theta_uses_separate_cache(self):
        """Instances with different theta must not share a cache."""
        rope_a = default_rope(theta=10000.0)
        rope_b = default_rope(theta=500000.0)

        q, k, pos = random_qk(seq=8)
        rope_a(q, k, pos)
        rope_b(q, k, pos)

        assert rope_a._cos_cached is not rope_b._cos_cached


# ---------------------------------------------------------------------------
# 7. Arbitrary position shape
# ---------------------------------------------------------------------------

class TestArbitraryPositionShape:
    """forward must accept position_ids of any shape, not only 2D (B, N).

    BEA uses 3D position_ids (B, L, T) paired with q/k of shape (B, L, T, head_dim).
    """

    def test_3d_position_ids_output_shapes(self):
        """forward must accept 3D position_ids (B, L, T) and preserve q/k shapes."""
        rope = default_rope()
        B, L, T = 2, 4, 6
        q = torch.randn(B, L, T, HEAD_DIM)
        k = torch.randn(B, L, T, HEAD_DIM)
        position_ids = torch.randint(0, 16, (B, L, T))
        q_rot, k_rot, _ = rope(q, k, position_ids)
        assert q_rot.shape == (B, L, T, HEAD_DIM)
        assert k_rot.shape == (B, L, T, HEAD_DIM)

    def test_3d_rotation_matches_direct_gather(self):
        """Rotation values for 3D position_ids must match manually gathered cos/sin."""
        rope = default_rope()
        B, L, T = 2, 3, 4
        q = torch.randn(B, L, T, HEAD_DIM)
        k = torch.randn(B, L, T, HEAD_DIM)
        position_ids = torch.randint(0, 10, (B, L, T))

        q_rot, k_rot, _ = rope(q, k, position_ids)

        cos = rope._cos_cached[position_ids]
        sin = rope._sin_cached[position_ids]
        q_expected = q * cos + _rotate_half(q) * sin
        k_expected = k * cos + _rotate_half(k) * sin

        torch.testing.assert_close(q_rot, q_expected)
        torch.testing.assert_close(k_rot, k_expected)

    def test_3d_single_L_matches_2d(self):
        """3D position_ids (B, 1, N) must produce the same rotation as 2D (B, N)."""
        rope = default_rope()
        B, N = 1, 6
        q = torch.randn(B, 1, N, HEAD_DIM)
        k = torch.randn(B, 1, N, HEAD_DIM)
        positions_3d = torch.arange(N).reshape(1, 1, N).expand(B, 1, N)
        positions_2d = positions_3d.squeeze(1)

        q_rot_3d, k_rot_3d, _ = rope(q, k, positions_3d)
        q_rot_2d, k_rot_2d, _ = rope(q, k, positions_2d)

        torch.testing.assert_close(q_rot_3d, q_rot_2d)
        torch.testing.assert_close(k_rot_3d, k_rot_2d)

    def test_attention_scaling_unaffected_by_position_ids_shape(self):
        """attention_scaling must be identical regardless of position_ids dimensionality."""
        rope = yarn_rope(s=4.0)
        D = HEAD_DIM
        q2 = torch.randn(1, 1, 4, D)
        k2 = torch.randn(1, 1, 4, D)
        q3 = torch.randn(1, 1, 4, D)
        k3 = torch.randn(1, 1, 4, D)
        _, _, scaling_2d = rope(q2, k2, torch.arange(4).unsqueeze(0))
        _, _, scaling_3d = rope(q3, k3, torch.arange(4).reshape(1, 1, 4))
        assert scaling_2d == scaling_3d
