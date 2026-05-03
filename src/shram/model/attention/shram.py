"""SHRAM hybrid attention layer.

This module implements the hybrid attention construction H(x) = h_l(x) + h_s(x)
used at one decoder attention slot in SHRAM.

The local sliding-window path and the MoSRAH sparse path are already verified
independently. The responsibility here is therefore not to introduce new
attention logic, but to preserve the bridge contracts between them: both paths
must consume the same input hidden state, each path must receive the sub-cache
it actually owns, the two model-space outputs must be summed directly, and the
sparse-path load-balance loss must remain visible to the caller.
"""

import torch
from torch import nn

from src.shram.model.cache.shram_layer_cache import ShramLayerCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.sliding_window_attention import SlidingWindowAttention
from src.shram.model.attention.mosrah import MoSRAHLayer


class SHRAMHybridLayer(nn.Module):
    """Hybrid attention layer H(x) = h_l(x) + h_s(x) for one decoder slot.

    The local path preserves nearby-token behavior through sliding-window causal
    attention. The sparse path is the theorem-facing MoSRAH routed attention
    path. Both operate over the same model-space hidden state and return
    model-space outputs, so the hybrid composition is a direct sum in model
    space.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.local_attention = SlidingWindowAttention(config)
        self.sparse_attention = MoSRAHLayer(config)

    def num_mosrah_parameters(self) -> int:
        """Return the total number of trainable parameters in the MoSRAH sparse path."""
        return self.sparse_attention.num_mosrah_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: ShramLayerCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the SHRAM hybrid attention layer.

        Args:
            hidden_states: Input hidden states of shape (B, N, d).
            position_ids: Authoritative token positions of shape (B, N).
            active_mask: Current-chunk active mask of shape (B, N), where True
                means the token is semantically live. Forwarded unchanged to
                both the local path and the sparse path.
            cache: Optional per-layer SHRAM cache. When provided, the owned
                sliding-window and MoSRAH sub-caches are dispatched directly to
                their corresponding attention paths.

        Returns:
            hybrid_output: Model-space hybrid attention output of shape (B, N, d).
            load_balance_loss: Scalar sparse-path load-balance loss.
            max_vio: Detached scalar routing-imbalance summary. Passed through
                unchanged from MoSRAHLayer; see MoSRAHRouter for semantics.
        """
        # ------------------------------------------------
        # It is not possible, due to how bea constructs its block mask,
        # for the model to process a sequence that does not start at zero
        # without a cache to track the per-head offsets
        # ------------------------------------------------

        if cache is None and torch.any(position_ids[:, 0] != 0):
            raise ValueError(
                "Uncached SHRAMHybridLayer does not support nonzero starting positions. "
                "Either provide a matching ShramLayerCache populated by the prefix for "
                "continued decoding, or rebase the uncached sequence to start at 0."
            )

        # -------------------------------------------------------------------
        # The hybrid layer's first responsibility is cache dispatch. The layer
        # cache already owns the concrete sub-cache objects required by each
        # path, so this unit should forward those exact references rather than
        # reinterpret cache ownership or invent a composite update protocol here.
        # -------------------------------------------------------------------
        if cache is None:
            sliding_window_cache = None
            mosrah_cache = None
        else:
            sliding_window_cache = cache.sliding_window_cache
            mosrah_cache = cache.mosrah_cache

        # -------------------------------------------------------------------
        # Both attention paths must see the same model-space hidden state for
        # the current decoder layer. The local path preserves short-range
        # structure, while the sparse path provides the routed long-range
        # contribution and emits the load-balance signal used by training.
        # -------------------------------------------------------------------
        local_output = self.local_attention(
            x=hidden_states,
            position_ids=position_ids,
            active_mask=active_mask,
            cache=sliding_window_cache,
        )
        sparse_output, load_balance_loss, max_vio = self.sparse_attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            active_mask=active_mask,
            cache=mosrah_cache,
        )

        # -------------------------------------------------------------------
        # The composition rule is intentionally simple at this boundary. Both
        # sublayers already return model-space tensors of matching shape, so the
        # correct hybrid behavior is their direct sum with no additional mixing
        # logic introduced here.
        # -------------------------------------------------------------------
        hybrid_output = local_output + sparse_output

        return hybrid_output, load_balance_loss, max_vio