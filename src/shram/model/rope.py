"""Rotary Position Embeddings (RoPE).

RoPE encodes position in the *relationship* between query and key vectors. When the
attention dot product Q·Kᵀ is computed, the per-position rotations cancel to produce
a score that depends only on the relative distance — not on absolute positions.

Two modes are supported:

  default  Standard RoPE with base frequency b. Each dimension pair d is assigned
           frequency θ_d = b^{-2d/u} where u is the head dimension. The attention
           scaling A_rope = 1.

  yarn     YaRN frequency interpolation for long-context extrapolation (Peng et al.,
           "YaRN: Efficient Context Window Extension of Large Language Models", 2023,
           §A.2). Three frequency regimes:
             - Low-frequency dimensions (r < α): fully interpolated by scale s.
               These dimensions have long wavelengths relative to the training window
               and must be compressed to avoid out-of-distribution positions.
             - High-frequency dimensions (r > β): left unchanged. Short-wavelength
               dimensions already encode relative position accurately at any scale.
             - Intermediate dimensions (α ≤ r ≤ β): linearly blended via ramp γ(r).
           Returns A_rope = (0.1·ln(s)+1)². When s = 1, YaRN reduces exactly to
           standard RoPE.

Each attention path (h_l and BEA) constructs its own RotaryEmbedding with explicit
parameters — no shared instance, no config reading. See Unit 5.A design decisions.

Cache sharing: all instances with identical parameters share one cos/sin table via a
class-level registry. The first instance that needs a particular (parameters, seq_len,
device, dtype) combination builds the table; all subsequent instances reference it
directly. This avoids redundant builds across the num_hidden_layers instances that
share the same parametrisation.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the 90° rotation used in the RoPE update formula.

    Splits the last dimension into two halves [x1, x2] and returns [-x2, x1].
    Combined with ``x * cos + rotate_half(x) * sin``, this implements a 2D rotation
    on each consecutive pair of dimensions, matching the block-diagonal operator
    R^u_{Θ,p} in the paper.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings with explicit mode and parameter control.

    Each caller constructs its own instance with the exact parameters it needs.
    h_l always uses ``mode="default"``; BEA always uses ``mode="yarn"``. No
    config object is read inside this module.

    The cos/sin cache is built lazily on the first forward call and extended
    automatically when a longer sequence is encountered. Instances with identical
    parameters share one cache via the class-level ``_cache`` registry,
    avoiding redundant computation across decoder layers.

    Args:
        mode: ``"default"`` for standard RoPE; ``"yarn"`` for YaRN extrapolation.
        head_dim: Per-head embedding dimension ``u``. Must be even.
        theta: Base frequency ``b`` in θ_d = b^{-2d/u}.
        initial_seq_length: ``C_train`` — context length the model was trained at.
            Required for ``mode="yarn"``.
        dilation: Scale factor ``s = C_target / C_train`` — how much the context
            window is extended beyond training length. Required for ``mode="yarn"``.
            When ``dilation=1.0``, YaRN reduces to standard RoPE.
        alpha: YaRN ramp lower boundary α. Dimensions with r(d) < α are fully
            interpolated. Required for ``mode="yarn"``.
        beta: YaRN ramp upper boundary β. Dimensions with r(d) > β are left
            unchanged. Required for ``mode="yarn"``.
        device: Optional device for initial buffer placement.

    Raises:
        NotImplementedError: If ``mode`` is not ``"default"`` or ``"yarn"``.
        ValueError: If ``mode="yarn"`` and any of ``initial_seq_length``,
            ``dilation``, ``alpha``, ``beta`` are absent.
    """

    # Maps (freq_key, seq_len, device_str, dtype_str) → (cos_table, sin_table).
    # Shared across all RotaryEmbedding instances in the process. Keys include device
    # and dtype so that tables built on different devices or in different precisions
    # are stored independently.
    _cache: dict = {}

    def __init__(
        self,
        mode: str,
        head_dim: int,
        theta: float,
        initial_seq_length: int | None = None,
        dilation: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self._validate_mode(mode)
        self._validate_yarn_params(mode, initial_seq_length, dilation, alpha, beta)
        self.mode = mode

        # Compute per-dimension rotation frequencies θ_d (default) or θ_d' (yarn).
        # d_index ranges over 0, 2, 4, ..., head_dim-2 — one index per dimension pair,
        # so rotation_freqs has head_dim/2 entries.
        d_index = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        base_freqs = 1.0 / (theta ** (d_index / head_dim))  # θ_d = b^{-2d/u}

        if mode == "default":
            rotation_freqs = base_freqs
            self.attention_scaling: float = 1.0

        else:  # yarn
            s = dilation

            # r(d) = C_train · θ_d / (2π) — normalized frequency used by the ramp
            # function to classify each dimension into one of three regimes.
            normalized_freqs = initial_seq_length * base_freqs / (2.0 * math.pi)

            # γ(r) ramp: 0 for r < α (fully interpolate), 1 for r > β (unchanged),
            # linear blend between α and β.
            blend_weights = ((normalized_freqs - alpha) / (beta - alpha)).clamp(0.0, 1.0)

            # θ_d' = (1 − γ) · θ_d / s + γ · θ_d
            rotation_freqs = (1.0 - blend_weights) * (base_freqs / s) + blend_weights * base_freqs

            # A_rope = (0.1 · ln(s) + 1)² — attention logit scaling returned to caller.
            self.attention_scaling = (0.1 * math.log(s) + 1.0) ** 2

        # freq_key uniquely identifies the parameter set that produced rotation_freqs.
        # Used as the primary component of the cache registry key.
        if mode == "default":
            self._freq_key: tuple = ("default", head_dim, float(theta))
        else:
            self._freq_key = (
                "yarn", head_dim, float(theta),
                int(initial_seq_length), float(dilation),
                float(alpha), float(beta),
            )

        # rotation_freqs is a non-persistent buffer so it moves with the model across
        # devices via .to() / .cuda() without appearing in saved checkpoints.
        # It is stored per-instance rather than in the shared cache because it is
        # small (head_dim/2 floats) — negligible cost compared to the cos/sin tables
        # it is used to build. The meaningful sharing win is on those tables.
        self.register_buffer("rotation_freqs", rotation_freqs, persistent=False)

        # Cache tensors are plain instance attributes (not registered buffers) so that
        # sharing across identically-parametrised instances survives .to() calls.
        # Registered buffers are copied on device move; plain attributes are aliased,
        # preserving the shared-tensor identity that the cache design depends on.
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    # ---------------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _validate_mode(mode: str) -> None:
        """Raise NotImplementedError if mode is not a supported value."""
        if mode not in {"default", "yarn"}:
            raise NotImplementedError(
                f"RoPE mode '{mode}' is not supported. Supported modes: 'default', 'yarn'."
            )

    @staticmethod
    def _validate_yarn_params(
        mode: str,
        initial_seq_length: int | None,
        dilation: float | None,
        alpha: float | None,
        beta: float | None,
    ) -> None:
        """Raise ValueError if mode='yarn' and any required parameter is absent."""
        if mode != "yarn":
            return
        missing = [
            name for name, val in [
                ("initial_seq_length", initial_seq_length),
                ("dilation", dilation),
                ("alpha", alpha),
                ("beta", beta),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(f"mode='yarn' requires {missing}.")

    # ---------------------------------------------------------------------------
    # Cache management
    # ---------------------------------------------------------------------------

    def _extend_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build the cos/sin table to cover positions [0, seq_len).

        Checks the class-level registry first. If a table already exists for this
        exact (parameters, seq_len, device, dtype) combination it is reused directly;
        otherwise it is computed and stored. The instance attributes are pointed at
        the registry entry so that all layers sharing the same parametrisation
        reference the same tensor.
        """
        cache_key = (self._freq_key, seq_len, str(device), str(dtype))

        if cache_key not in RotaryEmbedding._cache:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            # outer product → (seq_len, head_dim // 2); duplicate to (seq_len, head_dim)
            freqs = torch.outer(
                positions,
                self.rotation_freqs.to(device=device, dtype=torch.float32),
            )
            angle_embedding = torch.cat((freqs, freqs), dim=-1)
            RotaryEmbedding._cache[cache_key] = (
                angle_embedding.cos().to(dtype),
                angle_embedding.sin().to(dtype),
            )

        self._cos_cached, self._sin_cached = RotaryEmbedding._cache[cache_key]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Apply rotary embeddings to query and key tensors.

        The cos/sin cache is extended lazily when position_ids reference positions
        beyond its current length, or when the device or dtype has changed.

        ``position_ids`` may be any integer tensor shape. Its values are valid
        position indices into the cos/sin cache:

        - h_l (standard causal): position_ids (B, N), q/k (B, H, N, head_dim).
        - BEA (packed):          position_ids (B, L, T), q/k (B, L, T, head_dim).

        When q/k have head dimensions absent from position_ids, broadcast dimensions
        are inserted automatically at dim 1.

        Args:
            q: Query tensor of shape (batch, [heads,] *pos_dims, head_dim).
            k: Key tensor of shape (batch, [heads,] *pos_dims, head_dim).
            position_ids: Integer positions of shape (batch, *pos_dims).

        Returns:
            Tuple of (q_rotated, k_rotated, attention_scaling). attention_scaling is
            1.0 for default mode; YaRN returns (0.1·ln(s)+1)² which the caller must
            apply to attention logits before softmax.
        """
        seq_len = int(position_ids.max().item()) + 1

        # The cache is valid when it exists, covers all positions referenced by
        # position_ids, and matches q's dtype and device. Each condition is named
        # separately so the rebuild trigger is readable rather than a compound predicate.
        cache_missing = self._cos_cached is None
        cache_too_short = not cache_missing and seq_len > self._cos_cached.shape[0]
        wrong_dtype = not cache_missing and self._cos_cached.dtype != q.dtype
        wrong_device = not cache_missing and self._cos_cached.device != q.device

        if cache_missing or cache_too_short or wrong_dtype or wrong_device:
            self._extend_cache(seq_len, device=q.device, dtype=q.dtype)

        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]

        # Insert broadcast dimensions for any head axes present in q/k but absent
        # from position_ids. Standard: pos (B,N) → cos (B,N,D), q (B,H,N,D) → unsqueeze once.
        # BEA: pos (B,L,T) → cos (B,L,T,D), q (B,L,T,D) → no unsqueeze needed.
        while cos.ndim < q.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_rotated = q * cos + _rotate_half(q) * sin
        k_rotated = k * cos + _rotate_half(k) * sin

        return q_rotated, k_rotated, self.attention_scaling
