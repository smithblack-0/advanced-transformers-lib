"""Tests for ShramModel.

Unit 13 coverage strategy:
- real runtime smoke tests at the backbone public boundary
- preserved output-dict and hidden-state contract checks
- no fake decoder layers
- no handwritten alternate forward path
- no brittle internal seam instrumentation
"""

import torch

from src.shram.model.cache.shram_cache import ShramCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.model import ShramModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_config(**overrides) -> ShramConfig:
    """Construct a small SHRAM config for fast backbone tests."""
    config_kwargs = dict(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_sliding_window_heads=2,
        num_mosrah_heads=5,
        num_selected_heads=2,
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
        rms_norm_eps=1e-5,
        use_cache=True,
    )
    config_kwargs.update(overrides)
    return ShramConfig(**config_kwargs)


def make_model(
    config: ShramConfig,
    seed: int = 0,
) -> ShramModel:
    """Construct a deterministically initialized ShramModel."""
    torch.manual_seed(seed)
    return ShramModel(config).eval()


def random_embeds(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Construct deterministic random pre-embedded inputs."""
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(seed)
    return torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        generator=random_generator,
    )


def position_ids(
    batch_size: int,
    sequence_length: int,
    offset: int = 0,
) -> torch.Tensor:
    """Construct authoritative absolute position ids."""
    return torch.arange(
        offset,
        offset + sequence_length,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)


def make_cache(
    config: ShramConfig,
    batch_size: int,
    initial_buffer_size: int = 8,
) -> ShramCache:
    """Construct a real top-level ShramCache."""
    return ShramCache(
        num_hidden_layers=config.num_hidden_layers,
        sliding_window=config.window_size,
        num_local_heads=config.num_sliding_window_heads,
        local_head_dim=config.head_dim,
        num_mosrah_heads=config.num_mosrah_heads,
        mosrah_head_dim=config.head_dim,
        batch_size=batch_size,
        device=torch.device("cpu"),
        initial_buffer_size=initial_buffer_size,
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_forward_returns_expected_output_dict_keys(self):
        """ShramModel should expose the preserved output keys plus load_balance_loss and max_vio."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(1, 4, config.hidden_size, seed=1)
        out = model(inputs_embeds, position_ids(1, 4), torch.ones(1, 4, dtype=torch.bool))

        assert type(out) is dict
        assert set(out.keys()) == {
            "last_hidden_state",
            "past_key_values",
            "hidden_states",
            "load_balance_loss",
            "max_vio",
        }

    def test_last_hidden_state_shape_and_load_balance_loss_scalar_are_valid(self):
        """Real forward should preserve shape and return finite scalar auxiliary losses."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(2, 5, config.hidden_size, seed=2)
        out = model(inputs_embeds, position_ids(2, 5), torch.ones(2, 5, dtype=torch.bool))

        assert out["last_hidden_state"].shape == (2, 5, config.hidden_size)
        assert out["load_balance_loss"].ndim == 0
        assert out["max_vio"].ndim == 0
        assert torch.isfinite(out["last_hidden_state"]).all()
        assert torch.isfinite(out["load_balance_loss"])
        assert torch.isfinite(out["max_vio"])
        assert not out["max_vio"].requires_grad


# ---------------------------------------------------------------------------
# Hidden states
# ---------------------------------------------------------------------------

class TestHiddenStates:
    def test_output_hidden_states_false_returns_none(self):
        """The hidden_states field should remain None unless explicitly requested."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(1, 4, config.hidden_size, seed=3)
        out = model(
            inputs_embeds,
            position_ids(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            output_hidden_states=False,
        )

        assert out["hidden_states"] is None

    def test_output_hidden_states_true_returns_inputs_plus_one_entry_per_layer(self):
        """The hidden_states tuple should preserve the backbone's existing count semantics."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(1, 4, config.hidden_size, seed=4)
        out = model(
            inputs_embeds,
            position_ids(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            output_hidden_states=True,
        )

        hidden_states = out["hidden_states"]

        assert len(hidden_states) == config.num_hidden_layers + 1
        torch.testing.assert_close(hidden_states[0], inputs_embeds)

        for hidden_state in hidden_states:
            assert hidden_state.shape == (1, 4, config.hidden_size)


# ---------------------------------------------------------------------------
# Cache boundary
# ---------------------------------------------------------------------------

class TestCacheBoundary:
    def test_without_cache_returns_none_for_past_key_values(self):
        """The no-cache backbone contract should remain unchanged."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(1, 4, config.hidden_size, seed=5)
        out = model(
            inputs_embeds,
            position_ids(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            cache=None,
        )

        assert out["past_key_values"] is None

    def test_with_cache_returns_same_top_level_shram_cache_object(self):
        """ShramModel should return the same top-level cache object it was given."""
        config = small_config()
        model = make_model(config, seed=0)

        inputs_embeds = random_embeds(1, 4, config.hidden_size, seed=6)
        cache = make_cache(config, batch_size=1)

        out = model(
            inputs_embeds,
            position_ids(1, 4),
            torch.ones(1, 4, dtype=torch.bool),
            cache=cache,
        )

        assert out["past_key_values"] is cache