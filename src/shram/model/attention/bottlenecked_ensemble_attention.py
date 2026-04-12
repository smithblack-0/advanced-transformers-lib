"""Bottlenecked Ensemble Attention (BEA) for the MoSRAH sparse path.

BEA is the packed expert-choice attention operator over the MoSRAH sparse path.
It consumes packed expert-choice tensors, a supplied position tensor, an active
token mask, and an optional layer-local MoSRAH cache. It returns outputs in the
same packed expert-choice space expected by later unpacking.

BEA does not compute positions and does not choose packed-position semantics.
Those are supplied by the caller. If caching is used, BEA stores post-RoPE keys
(K̃) and raw values (V) into the sparse cache and attends against the
accumulated cached state returned by that cache.
"""

import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ..configuration import ShramConfig
from ..cache.mosrah_cache import MoSRAHCache
from ..rope import RotaryEmbedding


class BottleneckedEnsembleAttention(nn.Module):
    """
    Packed expert-choice attention operator for the MoSRAH sparse path.
    Operates per-head independently on an ensemble of tokens.
    FlexAttention saves flops on dead tokens.

    Architectural properties:
      - consumes packed expert-choice tensors of shape (B, L, T, d)
      - uses independent per-head Q/K/V/O projection parameters
      - applies YaRN-capable RoPE using supplied position_ids
      - stores post-RoPE K̃ and raw V in MoSRAHCache when caching is enabled
      - uses a fast fused attention path
      - returns outputs in the same packed expert-choice space (B, L, T, d)

    Args:
        config: SHRAM config. Must expose `hidden_size`, `num_mosrah_heads`,
            `head_dim`, `mosrah_rope_theta`, `training_sequence_length`,
            `inference_sequence_length`, `alpha`, and `beta`.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_mosrah_heads
        self.head_dim = config.head_dim

        # Independent per-head projections. No cross-head parameter sharing.
        self.q_proj = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_size, self.head_dim)
        )
        self.k_proj = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_size, self.head_dim)
        )
        self.v_proj = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_size, self.head_dim)
        )
        self.o_proj = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.hidden_size)
        )

        self._reset_parameters()

        # BEA uses the YaRN-capable RoPE path. The caller supplies the position tensor;
        # this unit only consumes it. In training modes, dilation will be 1.0 and so
        # no yarn dilation occurs.
        self.rope = RotaryEmbedding(
            mode="yarn",
            head_dim=self.head_dim,
            theta=config.mosrah_rope_theta,
            initial_seq_length=config.training_sequence_length,
            dilation=config.scale,
            alpha=config.alpha,
            beta=config.beta,
        )

    def forward(
        self,
        packed_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: MoSRAHCache | None = None,
    ) -> torch.Tensor:
        """Apply BEA to packed expert-choice tensors.

        Args:
            packed_embeddings: Packed expert-choice hidden states of shape (B, L, T, d).
            position_ids: Supplied packed positions of shape (B, L, T).
            active_mask: Boolean active-token mask of shape (B, L, T).
            cache: Optional layer-local MoSRAH cache.

        Returns:
            Packed expert-choice output tensor of shape (B, L, T, d).
        """
        batch_size, _, query_length, _ = packed_embeddings.shape
        self._validate_tensor_shape(packed_embeddings)
        self._validate_position_shape(packed_embeddings, position_ids)
        self._validate_active_mask_shape(packed_embeddings, active_mask)

        # Independent per-head projections:
        # (B, L, T, d) x (L, d, u) -> (B, L, T, u)
        query_states = torch.einsum("bltd,ldu->bltu", packed_embeddings, self.q_proj)
        key_states = torch.einsum("bltd,ldu->bltu", packed_embeddings, self.k_proj)
        value_states = torch.einsum("bltd,ldu->bltu", packed_embeddings, self.v_proj)

        rotated_query_states, rotated_key_states, attention_scaling = self.rope(
            query_states,
            key_states,
            position_ids,
        )

        if cache is not None:
            # In cached execution, the current query tensor uses local tensor rows
            # 0..Q-1, but the key tensor returned by the cache is the full accumulated
            # packed sequence for each (batch, head) slot. The only additional data
            # needed to align those two views is the pre-update cached prefix length.
            # which will indicate how many queries were processed before now.
            num_tokens_processed = cache.get_heads_lengths().clone()
            key_states, value_states, key_active_mask = cache.update(
                rotated_key_states,
                value_states,
                active_mask,
            )
        else:
            num_tokens_processed = torch.zeros(
                batch_size,
                self.num_heads,
                dtype=torch.long,
                device=packed_embeddings.device,
            )
            key_states = rotated_key_states
            key_active_mask = active_mask

        block_mask = self._make_block_mask(
             query_active_mask=active_mask,
             key_active_mask=key_active_mask,
             num_tokens_processed=num_tokens_processed,
             query_length=query_length,
             key_length=key_states.shape[2],
             device=packed_embeddings.device,
         )
        attended_states = flex_attention(
             rotated_query_states,
             key_states,
             value_states,
             block_mask=block_mask,
             scale=attention_scaling / math.sqrt(self.head_dim),
        )

        # Project back to model width:
        # (B, L, T, u) x (L, u, d) -> (B, L, T, d)
        return torch.einsum("bltu,lud->bltd", attended_states, self.o_proj)

    def _reset_parameters(self) -> None:
        """Initialize per-head projection weights."""
        for weight in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(weight)

    def _validate_tensor_shape(self, packed_embeddings: torch.Tensor) -> None:
        """Validate the local packed-embedding shape contract required by BEA."""
        if packed_embeddings.shape[1] != self.num_heads:
            raise ValueError(
                f"Expected packed_embeddings.shape[1] == num_mosrah_heads={self.num_heads}, "
                f"got {packed_embeddings.shape[1]}."
            )

        if packed_embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected packed_embeddings last dim == hidden_size={self.hidden_size}, "
                f"got {packed_embeddings.shape[-1]}."
            )

    def _validate_position_shape(
        self,
        packed_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Validate the supplied packed-position tensor shape."""
        if position_ids.shape != packed_embeddings.shape[:3]:
            raise ValueError(
                f"position_ids must have shape {tuple(packed_embeddings.shape[:3])}, "
                f"got {tuple(position_ids.shape)}."
            )

    def _validate_active_mask_shape(
        self,
        packed_embeddings: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Validate the supplied active-token mask shape."""
        if active_mask.shape != packed_embeddings.shape[:3]:
            raise ValueError(
                f"active_mask must have shape {tuple(packed_embeddings.shape[:3])}, "
                f"got {tuple(active_mask.shape)}."
            )

    def _make_block_mask(
        self,
        query_active_mask: torch.Tensor,
        key_active_mask: torch.Tensor,
        num_tokens_processed: torch.Tensor,
        query_length: int,
        key_length: int,
        device: torch.device,
    ):
        """Create the packed-sequence causal mask for FlexAttention.

        At the root, causality is still triangular. The only nuance is cached
        execution: query rows are indexed locally as 0..Q-1 inside the current
        query tensor, but the key tensor may already contain a cached prefix for
        that (batch, head) slot. The causal horizon for query tensor row q is
        therefore:

            cached_prefix_lengths[b, h] + q

        Query and key activity masks are then composed with that triangular rule
        so FlexAttention can skip padded query rows and ignore inactive key slots.
        """
        batch_size, num_heads, _ = query_active_mask.shape

        # Build the per-(batch, head, query_row) triangular horizon from a simple
        # arange over query rows plus the cached prefix lengths for each slot.
        relative_query_positions = torch.arange(
            query_length,
            device=device,
            dtype=torch.long,
        ).view(1, 1, query_length)
        causal_query_positions = num_tokens_processed.unsqueeze(-1) + relative_query_positions

        def packed_causal_mask(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            query_idx: torch.Tensor,
            key_idx: torch.Tensor,
        ) -> torch.Tensor:
            query_is_active = query_active_mask[batch_idx, head_idx, query_idx]
            key_is_active = key_active_mask[batch_idx, head_idx, key_idx]
            is_causal = key_idx <= causal_query_positions[batch_idx, head_idx, query_idx]
            return query_is_active & key_is_active & is_causal

        return create_block_mask(
            packed_causal_mask,
            B=batch_size,
            H=num_heads,
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=device,
        )