"""Tests for BottleneckedEnsembleAttention.

Invariants verified: output shape, packed expert-choice input/output contract,
per-head parameter independence, supplied-position RoPE behavior, YaRN rescale
response, causal masking over packed positions, inactive-row non-influence on
active outputs, cached post-RoPE/raw-V wiring, use of accumulated cached state,
cached/uncached contract equivalence under ragged cached prefixes, and use of
the fused FlexAttention path.
"""

import math

import pytest
import torch

import src.shram.model.attention.bottlenecked_ensemble_attention as bea_module
from src.shram.model.attention.bottlenecked_ensemble_attention import BottleneckedEnsembleAttention
from src.shram.model.configuration import ShramConfig
from src.shram.model.cache.mosrah_cache import MoSRAHCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ShramConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_sliding_window_heads=2,
        num_mosrah_heads=2,
        num_selected_heads=4,
        head_dim=4,
        window_size=4,
        rope_mode="main_sequence",
        local_rope_theta=10000.0,
        mosrah_rope_theta=10000.0,
        training_sequence_length=16,
        inference_sequence_length=16,
        alpha=1.0,
        beta=32.0,
        attention_dropout=0.0,
    )
    defaults.update(kwargs)
    return ShramConfig(**defaults)


def make_inputs(
    config: ShramConfig,
    batch: int = 1,
    packed_length: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_embeddings = torch.randn(
        batch,
        config.num_mosrah_heads,
        packed_length,
        config.hidden_size,
    )
    position_ids = torch.arange(packed_length).view(1, 1, packed_length).expand(
        batch,
        config.num_mosrah_heads,
        -1,
    )
    active_mask = torch.ones(
        batch,
        config.num_mosrah_heads,
        packed_length,
        dtype=torch.bool,
    )
    return packed_embeddings, position_ids, active_mask


def gather_current_rows_from_full(
    full_outputs: torch.Tensor,
    num_tokens_processed: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    """Gather the current-step rows from a full accumulated packed output tensor."""
    query_rows = torch.arange(query_length, device=full_outputs.device).view(1, 1, query_length)
    full_rows = num_tokens_processed.unsqueeze(-1) + query_rows
    return full_outputs.gather(
        dim=2,
        index=full_rows.unsqueeze(-1).expand(-1, -1, -1, full_outputs.shape[-1]),
    )


class SpyMoSRAHCache:
    """Minimal cache spy for BEA boundary tests.

    This is a legitimate dependency-boundary spy, not a surrogate BEA implementation.
    It records what BEA passes into cache.update() and lets the test control what
    accumulated state BEA receives back from the cache.
    """

    def __init__(
        self,
        num_tokens_processed: torch.Tensor,
        returned_keys: torch.Tensor,
        returned_values: torch.Tensor,
        returned_active_mask: torch.Tensor,
    ) -> None:
        self.num_tokens_processed = num_tokens_processed
        self.returned_keys = returned_keys
        self.returned_values = returned_values
        self.returned_active_mask = returned_active_mask

        self.seen_keys: torch.Tensor | None = None
        self.seen_values: torch.Tensor | None = None
        self.seen_active_mask: torch.Tensor | None = None

    def get_heads_lengths(self) -> torch.Tensor:
        return self.num_tokens_processed

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.seen_keys = key_states.clone()
        self.seen_values = value_states.clone()
        self.seen_active_mask = active_mask.clone()
        return self.returned_keys, self.returned_values, self.returned_active_mask


# ---------------------------------------------------------------------------
# Shape / packed-space contract
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape(self):
        """(B, L, T, d) -> (B, L, T, d)."""
        config = small_config()
        bea = BottleneckedEnsembleAttention(config)
        packed_embeddings, position_ids, active_mask = make_inputs(config)

        out = bea(packed_embeddings, position_ids, active_mask)

        assert out.shape == packed_embeddings.shape

    def test_accepts_packed_inputs_and_returns_packed_outputs(self):
        """BEA should preserve the packed expert-choice tensor space."""
        config = small_config(num_mosrah_heads=3)
        bea = BottleneckedEnsembleAttention(config)
        packed_embeddings, position_ids, active_mask = make_inputs(
            config,
            batch=2,
            packed_length=5,
        )

        out = bea(packed_embeddings, position_ids, active_mask)

        assert out.shape[:3] == packed_embeddings.shape[:3]
        assert out.shape[-1] == config.hidden_size


# ---------------------------------------------------------------------------
# Independent paper-formula anchor
# ---------------------------------------------------------------------------

class TestPaperFormulaAnchor:
    def test_tiny_single_head_case_matches_manual_formula(self):
        """A tiny hand-solvable BEA case should match the paper formula directly.

        This test exercises the *actual* BEA layer and actual fused path. The only
        simplification is choosing all-zero positions so RoPE becomes the identity
        for this case. The expected output is then computed manually from the paper
        equations for QK^T / sqrt(u), softmax, V, and O.
        """
        config = small_config(
            hidden_size=2,
            num_mosrah_heads=1,
            head_dim=2,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.q_proj.zero_()
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.o_proj.zero_()

            bea.q_proj[0] = torch.eye(2)
            bea.k_proj[0] = torch.eye(2)
            bea.v_proj[0] = torch.eye(2)
            bea.o_proj[0] = torch.eye(2)

        packed_embeddings = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        position_ids = torch.zeros(1, 1, 2, dtype=torch.long)
        active_mask = torch.tensor([[[True, True]]])

        out = bea(packed_embeddings, position_ids, active_mask)

        scale = 1.0 / math.sqrt(2.0)
        weight_10 = math.exp(0.0)
        weight_11 = math.exp(scale)
        denom = weight_10 + weight_11
        expected_second = torch.tensor([weight_10 / denom, weight_11 / denom])

        expected = torch.tensor([[[[1.0, 0.0], expected_second.tolist()]]]).reshape(1, 1, 2, 2)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Projection independence
# ---------------------------------------------------------------------------

class TestHeadIndependence:
    def test_modifying_one_head_parameters_only_changes_that_head_output(self):
        """Per-head parameters should be independent."""
        config = small_config(num_mosrah_heads=2)
        torch.manual_seed(0)
        bea_a = BottleneckedEnsembleAttention(config)
        torch.manual_seed(0)
        bea_b = BottleneckedEnsembleAttention(config)

        packed_embeddings, position_ids, active_mask = make_inputs(config, batch=1, packed_length=4)

        with torch.no_grad():
            bea_b.q_proj[0].add_(1.0)
            bea_b.k_proj[0].add_(1.0)
            bea_b.v_proj[0].add_(1.0)
            bea_b.o_proj[0].add_(1.0)

        out_a = bea_a(packed_embeddings, position_ids, active_mask)
        out_b = bea_b(packed_embeddings, position_ids, active_mask)

        assert not torch.allclose(out_a[:, 0], out_b[:, 0])
        torch.testing.assert_close(out_a[:, 1], out_b[:, 1])


# ---------------------------------------------------------------------------
# RoPE / YaRN behavior
# ---------------------------------------------------------------------------

class TestRopeBehavior:
    def test_supplied_position_tensors_change_attention_behavior(self):
        """Different supplied position tensors should change BEA outputs."""
        config = small_config()
        torch.manual_seed(1)
        bea = BottleneckedEnsembleAttention(config)
        packed_embeddings, _, active_mask = make_inputs(config, batch=1, packed_length=4)

        position_ids_a = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
        position_ids_b = torch.tensor([[[0, 2, 4, 6], [0, 2, 4, 6]]])

        out_a = bea(packed_embeddings, position_ids_a, active_mask)
        out_b = bea(packed_embeddings, position_ids_b, active_mask)

        assert not torch.allclose(out_a, out_b)

    def test_bea_passes_supplied_positions_through_to_rope_and_uses_yarn_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """BEA should not internally compute positions or override the RoPE mode."""
        config = small_config(inference_sequence_length=64)
        bea = BottleneckedEnsembleAttention(config)
        packed_embeddings, position_ids, active_mask = make_inputs(config, batch=1, packed_length=3)

        seen_position_ids: list[torch.Tensor] = []

        def spy_rope(q, k, supplied_position_ids):
            seen_position_ids.append(supplied_position_ids.clone())
            return q, k, 1.0

        monkeypatch.setattr(bea.rope, "forward", spy_rope)
        bea(packed_embeddings, position_ids, active_mask)

        assert bea.rope.mode == "yarn"
        torch.testing.assert_close(seen_position_ids[0], position_ids)

    def test_yarn_dilation_stretches_rotation_rate_in_fully_interpolated_regime(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ):
        """With full YaRN interpolation, doubling dilation doubles the distance to the same phase.

        This tests the real BEA layer, not a surrogate backend. We force a tiny one-head identity
        setup, capture the query tensor BEA hands to the fused backend, and compare the actual
        rotated vectors. With full interpolation, scale=2 should make position 4 match the phase
        that scale=1 reaches at position 2.
        """
        config_a = small_config(
            hidden_size=2,
            num_mosrah_heads=1,
            head_dim=2,
            training_sequence_length=16,
            inference_sequence_length=16,
            alpha=10 ** 9,
            beta=10 ** 9 + 1,
        )
        config_b = small_config(
            hidden_size=2,
            num_mosrah_heads=1,
            head_dim=2,
            training_sequence_length=16,
            inference_sequence_length=32,
            alpha=10 ** 9,
            beta=10 ** 9 + 1,
        )

        bea_a = BottleneckedEnsembleAttention(config_a)
        bea_b = BottleneckedEnsembleAttention(config_b)

        with torch.no_grad():
            for bea in (bea_a, bea_b):
                bea.q_proj.zero_()
                bea.k_proj.zero_()
                bea.v_proj.zero_()
                bea.o_proj.zero_()
                bea.q_proj[0] = torch.eye(2)
                bea.k_proj[0] = torch.eye(2)

        captured = {}

        def fake_flex_attention_a(q, k, v, block_mask, scale):
            del k, v, block_mask, scale
            captured["q_a"] = q.clone()
            return torch.zeros_like(q)

        def fake_flex_attention_b(q, k, v, block_mask, scale):
            del k, v, block_mask, scale
            captured["q_b"] = q.clone()
            return torch.zeros_like(q)

        packed_embeddings = torch.tensor([[[[1.0, 0.0]]]])
        active_mask = torch.tensor([[[True]]])

        monkeypatch.setattr(bea_module, "flex_attention", fake_flex_attention_a)
        bea_a(packed_embeddings, torch.tensor([[[2]]]), active_mask)

        monkeypatch.setattr(bea_module, "flex_attention", fake_flex_attention_b)
        bea_b(packed_embeddings, torch.tensor([[[4]]]), active_mask)

        torch.testing.assert_close(captured["q_a"], captured["q_b"], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

class TestCausality:
    def test_future_active_tokens_do_not_affect_past_active_outputs(self):
        """Changing future packed tokens must not change earlier active outputs."""
        config = small_config(hidden_size=8, num_mosrah_heads=2, head_dim=4)
        bea = BottleneckedEnsembleAttention(config)
        bea.eval()
        torch.manual_seed(3)

        packed_embeddings, position_ids, active_mask = make_inputs(config, batch=1, packed_length=4)
        out_a = bea(packed_embeddings, position_ids, active_mask)

        packed_embeddings_modified = packed_embeddings.clone()
        packed_embeddings_modified[:, :, 3, :] = torch.randn_like(packed_embeddings_modified[:, :, 3, :])
        out_b = bea(packed_embeddings_modified, position_ids, active_mask)

        torch.testing.assert_close(out_a[:, :, :3, :], out_b[:, :, :3, :])


# ---------------------------------------------------------------------------
# Padding / inactivity behavior
# ---------------------------------------------------------------------------

class TestPaddingBehavior:
    def test_inactive_query_rows_do_not_influence_active_outputs(self):
        """Changing inactive rows should not affect active outputs when only Q depends on them.

        BEA consumes one packed tensor for Q, K, and V. To isolate query-side inactivity,
        K and V projections are zeroed so perturbing inactive packed rows can only change
        Q for those inactive rows. Active outputs must remain unchanged.
        """
        config = small_config(hidden_size=8, num_mosrah_heads=2, head_dim=4)
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.k_proj.zero_()
            bea.v_proj.zero_()

        packed_embeddings, position_ids, _ = make_inputs(config, batch=1, packed_length=4)
        active_mask = torch.tensor(
            [[[True, True, False, False], [True, False, False, False]]]
        )

        out_a = bea(packed_embeddings, position_ids, active_mask)

        packed_embeddings_modified = packed_embeddings.clone()
        inactive_mask = (~active_mask).unsqueeze(-1).expand_as(packed_embeddings_modified)
        packed_embeddings_modified[inactive_mask] = torch.randn_like(
            packed_embeddings_modified[inactive_mask]
        )

        out_b = bea(packed_embeddings_modified, position_ids, active_mask)

        torch.testing.assert_close(out_a[active_mask], out_b[active_mask])

    def test_inactive_key_value_rows_do_not_influence_active_outputs(self):
        """Changing inactive rows should not affect active outputs when only K/V depend on them.

        To isolate key/value-side inactivity, Q projection is zeroed so perturbing inactive
        packed rows cannot change the active queries. The only remaining possible effect is
        through inactive K/V rows. Active outputs must stay unchanged.
        """
        config = small_config(hidden_size=8, num_mosrah_heads=2, head_dim=4)
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.q_proj.zero_()

        packed_embeddings, position_ids, _ = make_inputs(config, batch=1, packed_length=4)
        active_mask = torch.tensor(
            [[[True, True, False, False], [True, False, False, False]]]
        )

        out_a = bea(packed_embeddings, position_ids, active_mask)

        packed_embeddings_modified = packed_embeddings.clone()
        inactive_mask = (~active_mask).unsqueeze(-1).expand_as(packed_embeddings_modified)
        packed_embeddings_modified[inactive_mask] = torch.randn_like(
            packed_embeddings_modified[inactive_mask]
        )

        out_b = bea(packed_embeddings_modified, position_ids, active_mask)

        torch.testing.assert_close(out_a[active_mask], out_b[active_mask])

    def test_inactive_packed_rows_do_not_influence_active_outputs(self):
        """Changing inactive packed rows should not affect active outputs in the full path.

        This is the combined end-to-end version of the two more targeted tests above.
        We rewrite only inactive packed rows, rerun BEA, and compare only active output rows.
        Inactive output rows are allowed to contain junk because unpacking removes them later.
        """
        config = small_config(hidden_size=8, num_mosrah_heads=2, head_dim=4)
        bea = BottleneckedEnsembleAttention(config)

        packed_embeddings, position_ids, _ = make_inputs(config, batch=1, packed_length=4)
        active_mask = torch.tensor(
            [[[True, True, False, False], [True, False, False, False]]]
        )

        out_a = bea(packed_embeddings, position_ids, active_mask)

        packed_embeddings_modified = packed_embeddings.clone()
        inactive_mask = (~active_mask).unsqueeze(-1).expand_as(packed_embeddings_modified)
        packed_embeddings_modified[inactive_mask] = torch.randn_like(
            packed_embeddings_modified[inactive_mask]
        )

        out_b = bea(packed_embeddings_modified, position_ids, active_mask)

        torch.testing.assert_close(out_a[active_mask], out_b[active_mask])
    def test_junk_cached_key_value_slots_do_not_influence_active_outputs(self):
        """Changing inactive cached tail slots should not affect active outputs.

        This is the cache-side version of the inactive-row tests above. The spy cache returns the
        same active cached prefix in both runs but different junk data in inactive tail slots.
        Active outputs must remain unchanged if BEA is correctly ignoring inactive cached slots.
        """
        config = small_config(
            hidden_size=2,
            num_mosrah_heads=1,
            head_dim=2,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.q_proj.zero_()
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.o_proj.zero_()
            bea.q_proj[0] = torch.eye(2)
            bea.k_proj[0] = torch.eye(2)
            bea.v_proj[0] = torch.eye(2)
            bea.o_proj[0] = torch.eye(2)

        packed_embeddings = torch.tensor([[[[1.0, 0.0]]]])
        position_ids = torch.zeros(1, 1, 1, dtype=torch.long)
        active_mask = torch.tensor([[[True]]])

        active_prefix_keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]])
        active_prefix_values_a = torch.tensor([[[[10.0, 0.0], [0.0, 20.0], [999.0, 999.0]]]])
        active_prefix_values_b = torch.tensor([[[[10.0, 0.0], [0.0, 20.0], [-999.0, -999.0]]]])
        returned_active_mask = torch.tensor([[[True, True, False]]])

        cache_a = SpyMoSRAHCache(
            num_tokens_processed=torch.tensor([[2]]),
            returned_keys=active_prefix_keys,
            returned_values=active_prefix_values_a,
            returned_active_mask=returned_active_mask,
        )
        cache_b = SpyMoSRAHCache(
            num_tokens_processed=torch.tensor([[2]]),
            returned_keys=active_prefix_keys,
            returned_values=active_prefix_values_b,
            returned_active_mask=returned_active_mask,
        )

        out_a = bea(packed_embeddings, position_ids, active_mask, cache=cache_a)
        out_b = bea(packed_embeddings, position_ids, active_mask, cache=cache_b)

        torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)

# ---------------------------------------------------------------------------
# Cache boundary behavior
# ---------------------------------------------------------------------------

class TestCacheBoundary:
    def test_cached_execution_stores_post_rope_keys_and_raw_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The cache boundary should receive K̃ and raw V, not raw K."""
        config = small_config(
            hidden_size=4,
            num_mosrah_heads=1,
            head_dim=4,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.k_proj[0] = torch.eye(4)
            bea.v_proj[0] = torch.eye(4)

        def fake_rope(q, k, position_ids):
            del q, position_ids
            return torch.zeros_like(k), k + 7.0, 1.0

        monkeypatch.setattr(bea.rope, "forward", fake_rope)

        packed_embeddings = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]])
        position_ids = torch.tensor([[[0, 1]]])
        active_mask = torch.tensor([[[True, True]]])

        spy_cache = SpyMoSRAHCache(
            num_tokens_processed=torch.zeros(1, 1, dtype=torch.long),
            returned_keys=torch.zeros(1, 1, 2, 4),
            returned_values=torch.zeros(1, 1, 2, 4),
            returned_active_mask=active_mask,
        )

        bea(packed_embeddings, position_ids, active_mask, cache=spy_cache)

        torch.testing.assert_close(spy_cache.seen_keys, packed_embeddings + 7.0)
        torch.testing.assert_close(spy_cache.seen_values, packed_embeddings)
        torch.testing.assert_close(spy_cache.seen_active_mask, active_mask)

    def test_cached_execution_uses_accumulated_state_returned_by_cache(self):
        """BEA should attend against the accumulated state returned by cache.update()."""
        config = small_config(
            hidden_size=2,
            num_mosrah_heads=1,
            head_dim=2,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.q_proj.zero_()
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.o_proj.zero_()

            bea.q_proj[0] = torch.eye(2)
            bea.k_proj[0] = torch.eye(2)
            bea.v_proj[0] = torch.eye(2)
            bea.o_proj[0] = torch.eye(2)

        # Zero positions keep RoPE as identity in this small behavioral case.
        packed_embeddings = torch.tensor([[[[1.0, 0.0]]]])
        position_ids = torch.zeros(1, 1, 1, dtype=torch.long)
        active_mask = torch.tensor([[[True]]])

        returned_keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        returned_values = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
        returned_active_mask = torch.tensor([[[True, True]]])

        spy_cache = SpyMoSRAHCache(
            num_tokens_processed=torch.tensor([[1]]),
            returned_keys=returned_keys,
            returned_values=returned_values,
            returned_active_mask=returned_active_mask,
        )

        out = bea(packed_embeddings, position_ids, active_mask, cache=spy_cache)

        scale = 1.0 / math.sqrt(2.0)
        weight_0 = math.exp(scale)
        weight_1 = math.exp(0.0)
        denom = weight_0 + weight_1
        expected = torch.tensor(
            [[[[10.0 * weight_0 / denom, 20.0 * weight_1 / denom]]]]
        )

        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    def test_uncached_execution_uses_current_step_rotated_keys_and_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Without a cache, BEA should pass current-step K̃ and raw V directly to the backend."""
        config = small_config(
            hidden_size=4,
            num_mosrah_heads=1,
            head_dim=4,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.k_proj[0] = torch.eye(4)
            bea.v_proj[0] = torch.eye(4)

        def fake_rope(q, k, position_ids):
            del q, position_ids
            return torch.zeros_like(k), k + 7.0, 1.0

        seen = {}

        def fake_flex_attention(q, k, v, block_mask, scale):
            del q, block_mask, scale
            seen["k"] = k.clone()
            seen["v"] = v.clone()
            return torch.zeros_like(v)

        monkeypatch.setattr(bea.rope, "forward", fake_rope)
        monkeypatch.setattr(bea_module, "flex_attention", fake_flex_attention)

        packed_embeddings = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]])
        position_ids = torch.tensor([[[0, 1]]])
        active_mask = torch.tensor([[[True, True]]])

        bea(packed_embeddings, position_ids, active_mask, cache=None)

        torch.testing.assert_close(seen["k"], packed_embeddings + 7.0)
        torch.testing.assert_close(seen["v"], packed_embeddings)

# ---------------------------------------------------------------------------
# Cached / uncached equivalence under ragged prefixes
# ---------------------------------------------------------------------------

class TestCachedCausality:
    def test_cached_and_uncached_match_under_ragged_cached_prefixes(self):
        """Cached BEA should match one-shot BEA when heads have different prefix lengths."""
        config = small_config(
            hidden_size=2,
            num_mosrah_heads=2,
            head_dim=2,
            training_sequence_length=8,
            inference_sequence_length=8,
        )
        bea = BottleneckedEnsembleAttention(config)

        with torch.no_grad():
            bea.q_proj.zero_()
            bea.k_proj.zero_()
            bea.v_proj.zero_()
            bea.o_proj.zero_()
            for head_idx in range(config.num_mosrah_heads):
                bea.q_proj[head_idx] = torch.eye(2)
                bea.k_proj[head_idx] = torch.eye(2)
                bea.v_proj[head_idx] = torch.eye(2)
                bea.o_proj[head_idx] = torch.eye(2)

        prefix_embeddings = torch.tensor(
            [[
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [9.0, 9.0]],
            ]]
        )
        prefix_positions = torch.zeros(1, 2, 2, dtype=torch.long)
        prefix_active_mask = torch.tensor(
            [[[True, True], [True, False]]]
        )

        current_embeddings = torch.tensor(
            [[
                [[2.0, 0.0], [0.0, 2.0]],
                [[2.0, 2.0], [3.0, 3.0]],
            ]]
        )
        current_positions = torch.zeros(1, 2, 2, dtype=torch.long)
        current_active_mask = torch.tensor(
            [[[True, True], [True, True]]]
        )

        cache = MoSRAHCache(
            num_mosrah_heads=config.num_mosrah_heads,
            head_dim=config.head_dim,
            batch_size=1,
            device=torch.device("cpu"),
            initial_buffer_size=4,
        )
        _ = bea(prefix_embeddings, prefix_positions, prefix_active_mask, cache=cache)
        num_tokens_processed = cache.get_heads_lengths().clone()

        out_cached = bea(
            current_embeddings,
            current_positions,
            current_active_mask,
            cache=cache,
        )

        full_embeddings = torch.tensor(
            [[
                [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]],
            ]]
        )
        full_positions = torch.zeros(1, 2, 4, dtype=torch.long)
        full_active_mask = torch.tensor(
            [[[True, True, True, True], [True, True, True, False]]]
        )

        out_full = bea(full_embeddings, full_positions, full_active_mask)
        expected_current = gather_current_rows_from_full(
            full_outputs=out_full,
            num_tokens_processed=num_tokens_processed,
            query_length=current_embeddings.shape[2],
        )

        torch.testing.assert_close(out_cached, expected_current, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Fused-path usage
# ---------------------------------------------------------------------------

class TestBackendPath:
    def test_bea_routes_through_flex_attention(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """BEA should call the fused FlexAttention path."""
        config = small_config()
        bea = BottleneckedEnsembleAttention(config)
        packed_embeddings, position_ids, active_mask = make_inputs(config, batch=1, packed_length=3)

        seen = {"called": False}

        def fake_create_block_mask(*args, **kwargs):
            return "mask"

        def fake_flex_attention(q, k, v, block_mask, scale):
            del q, k, v, scale
            seen["called"] = True
            assert block_mask == "mask"
            return torch.zeros(1, config.num_mosrah_heads, 3, config.head_dim)

        monkeypatch.setattr(bea_module, "create_block_mask", fake_create_block_mask)
        monkeypatch.setattr(bea_module, "flex_attention", fake_flex_attention)

        bea(packed_embeddings, position_ids, active_mask)

        assert seen["called"]
