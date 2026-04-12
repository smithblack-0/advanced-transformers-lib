"""Full MoSRAH sparse path for SHRAM.

This module coordinates the routed sparse attention path used inside the SHRAM
hybrid attention layer. The underlying mechanics already live in verified
subunits. The responsibility here is to connect those subunits without
corrupting their bridge contracts.

In particular, this path must preserve three architectural distinctions:

- selected head indices are not routing probabilities
- packed position semantics are chosen before BEA, not inside it
- weighted reduction must consume the router's unbiased renormalized
  probabilities after token-choice order has been restored
"""

import torch
from torch import nn

from src.shram.model.cache.mosrah_cache import MoSRAHCache
from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.bottlenecked_ensemble_attention import BottleneckedEnsembleAttention
from src.shram.model.attention.expert_packing import (
    pack_experts,
    setup_packing,
    unpack_experts,
)
from src.shram.model.attention.router import MoSRAHRouter
from src.shram.model.attention.positions_converter import SparseMoSRAHPositions


class MoSRAHLayer(nn.Module):
    """Full routed sparse attention path for SHRAM.

    The MoSRAH path consumes model-space hidden states together with
    authoritative per-token positions and returns the model-space sparse-path
    contribution, the router's load-balance loss, and the router's MaxVio
    routing-imbalance scalar.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.num_experts = config.num_mosrah_heads

        self.router = MoSRAHRouter(config)
        self.positions = SparseMoSRAHPositions(config)
        self.bea = BottleneckedEnsembleAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: MoSRAHCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full MoSRAH sparse path.

        Args:
            hidden_states: Model-space hidden states x of shape (B, N, d).
            position_ids: Authoritative per-token positions of shape (B, N).
            cache: Optional layer-local MoSRAH cache. Pass None for uncached
                execution and the layer-local cache instance for cached execution.

        Returns:
            sparse_output: Model-space sparse-path output of shape (B, N, d).
            load_balance_loss: Scalar router load-balance loss.
            max_vio: Detached scalar routing-imbalance summary. Passed through
                unchanged from the router; see MoSRAHRouter for semantics.
        """

        # -------------------------------------------------------------------
        # The first transition moves from model-space token-choice input into
        # the packed expert-choice sparse-attention state. Routing decides both
        # which experts each token uses and which unbiased probabilities must be
        # reserved for the final reduction. Packing then realizes that routing
        # decision in the expert-major frame BEA consumes, while carrying the
        # authoritative upstream positions through the same rearrangement so
        # packed hidden states and packed positions remain aligned.
        # -------------------------------------------------------------------
        selected_heads, routing_probs, load_balance_loss, max_vio = self.router(hidden_states)

        flattened_selected_heads, permutation, inverse_permutation = setup_packing(
            selected_heads
        )
        packed_hidden_states, packed_positions, active_mask = pack_experts(
            hidden_states=hidden_states,
            position_ids=position_ids,
            selected_heads=selected_heads,
            num_experts=self.num_experts,
            flattened_selected_heads=flattened_selected_heads,
            permutation=permutation,
        )

        # -------------------------------------------------------------------
        # Sparse attention runs entirely in the packed expert-choice frame, so
        # the RoPE position semantics must also be chosen in that frame. The
        # position layer therefore decides whether BEA should see packed
        # original-token positions or packed local-slot positions. BEA then
        # consumes that packed position tensor together with the packed hidden
        # states and the layer-local sparse cache, which it owns directly.
        # -------------------------------------------------------------------
        bea_positions = self.positions(
            packed_positions=packed_positions,
            cache=cache,
        )
        packed_outputs = self.bea(
            packed_embeddings=packed_hidden_states,
            position_ids=bea_positions,
            active_mask=active_mask,
            cache=cache,
        )

        # -------------------------------------------------------------------
        # The final transition restores token-choice meaning and only then
        # collapses the K routed copies back into model space. This ordering is
        # required because routing_probs live in token-choice space, whereas BEA
        # returns expert-choice packed outputs. The reduction must therefore
        # happen after unpacking, and it must use the router's unbiased
        # renormalized probabilities rather than any biased selection scores.
        # -------------------------------------------------------------------
        token_choice_outputs = unpack_experts(
            expert_outputs=packed_outputs,
            selected_heads=selected_heads,
            active_mask=active_mask,
            inverse_permutation=inverse_permutation,
        )
        final_output = (
            token_choice_outputs * routing_probs.unsqueeze(-1)
        ).sum(dim=2)

        return final_output, load_balance_loss, max_vio
