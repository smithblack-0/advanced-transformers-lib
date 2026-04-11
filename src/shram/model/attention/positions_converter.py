"""Position computation for the MoSRAH sparse path.

This layer computes the packed position tensor P consumed by BEA.

- In main-sequence mode, P is the packed original-token position tensor from the
  packing path.
- In semantic-sequence mode, P is a per-expert local sequence over the packed
  expert-choice layout, optionally offset by the current sparse-cache occupancies
  during cached inference.
"""

import torch
from torch import nn

from src.shram.model.configuration import ShramConfig
from src.shram.model.cache.mosrah_cache import MoSRAHCache


class SparseMoSRAHPositions(nn.Module):
    """Compute the packed RoPE position tensor for the MoSRAH sparse path.

    This layer operates in the packed expert-choice frame used by BEA. The input
    packed_positions tensor is always the packed original-token position tensor
    produced by the packing path. The configured rope_mode determines whether that
    tensor is forwarded directly or replaced by a semantic local-slot sequence.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.rope_mode = config.rope_mode

    def forward(
        self,
        packed_positions: torch.Tensor,
        cache: MoSRAHCache | None,
    ) -> torch.Tensor:
        """Compute the packed position tensor P consumed by BEA.

        Args:
            packed_positions: Packed original-token positions J' of shape (B, L, T).
            cache: Optional layer-local MoSRAH cache. When present in semantic-sequence
                mode, the current per-head occupancies offset the local packed sequence.

        Returns:
            Packed position tensor P of shape (B, L, T).
        """
        if self.rope_mode == "main_sequence":
            return self._main_sequence_positions(packed_positions)

        if self.rope_mode == "semantic_sequence":
            return self._semantic_sequence_positions(packed_positions, cache)

        raise NotImplementedError(
            f"Unsupported MoSRAH rope_mode '{self.rope_mode}'."
        )

    def _main_sequence_positions(
        self,
        packed_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward packed original-token positions unchanged."""
        return packed_positions

    def _semantic_sequence_positions(
        self,
        packed_positions: torch.Tensor,
        cache: MoSRAHCache | None,
    ) -> torch.Tensor:
        """Compute semantic-sequence packed positions in expert-choice space.

        Without a sparse cache, semantic positions are the local packed sequence
        0, 1, 2, ... over the expert-local T dimension. With a sparse cache, that
        same local sequence is offset by the current per-(batch, expert) occupancies
        returned by get_heads_lengths().
        """
        batch_size, num_experts, packed_length = packed_positions.shape

        # -------------------------------------------------------------------
        # Construct the local packed sequence 0, 1, 2, ... over the expert-local
        # sequence dimension T. This is then broadcast across batch and experts.
        # -------------------------------------------------------------------
        local_positions = torch.arange(
            packed_length,
            device=packed_positions.device,
            dtype=packed_positions.dtype,
        ).view(1, 1, packed_length).expand(
            batch_size,
            num_experts,
            packed_length,
        )

        # -------------------------------------------------------------------
        # In cached semantic-sequence mode, positions continue from the current
        # sparse-cache occupancies rather than restarting at zero for the local
        # chunk.
        # -------------------------------------------------------------------
        if cache is None:
            return local_positions

        cached_lengths = cache.get_heads_lengths().to(
            device=packed_positions.device,
            dtype=packed_positions.dtype,
        ).unsqueeze(-1)

        return local_positions + cached_lengths