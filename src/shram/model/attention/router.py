"""Token-choice router for the MoSRAH sparse attention path.

This module implements mechanically load-balanced routing for MoSRAH. Given an
input hidden state x, the router produces two outputs used downstream:

  - selected_heads (I): which K of the L available expert heads each token
    routes to, determined by a block-balanced causal solver.
  - routing_probs (P): the weights used for the weighted output reduction,
    gathered from the softmax routing scores at the selected indices and
    renormalized to sum to 1 per token.

Routing uses a single learnable projection:

  - routing_weight: shape (L, embedding_width). Maps input to per-head routing
    scores. Task loss trains this parameter through routing_probs; regret_loss
    trains it to prefer expert assignments at positions of peak preference.

Block-balanced routing partitions the sequence into non-overlapping blocks of
W = L/K tokens. Within each block every expert is assigned to exactly one token,
guaranteeing perfect load balance by construction. The L % K == 0 compatibility
constraint (enforced in ShramConfig) makes W an exact integer.

Selection is causal within each block: at each of the W steps the current
token chooses its K experts from those not yet claimed by earlier tokens in
the same block. All W steps execute in parallel across blocks and batch via
a fully-unrolled Python for loop, keeping the compiled graph flat.

Paper ref: Appendix A.Routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache.router_cache import RouterCache
from ..configuration import ShramConfig


class MoSRAHRouter(nn.Module):
    """Token-choice router for MoSRAH sparse attention.

    Each input token independently selects K of the L available expert heads
    through a block-balanced causal solver. Within each block of W = L/K
    consecutive tokens every expert is used exactly once, giving perfect load
    balance by construction.

    routing_weight is nn.Parameter rather than nn.Linear so that HuggingFace
    _init_weights does not override its kaiming initialization at construction.

    Attributes:
        routing_weight: Shape (L, embedding_width). Maps input hidden states to
            per-head routing scores.
        block_length: Tokens per routing block W = L / K. Within each block
            every expert is used exactly once.

    Args:
        config: Model configuration. Must expose ``embedding_width``,
            ``num_mosrah_heads`` (L), ``num_selected_heads`` (K), and
            ``block_length`` (W).
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.num_mosrah_heads   = config.num_mosrah_heads
        self.num_selected_heads = config.num_selected_heads
        self.block_length       = config.block_length

        # Routing projection: maps input (B, N, d) to per-head routing scores (B, N, L).
        # nn.Parameter ensures HuggingFace _init_weights does not override kaiming init.
        self.routing_weight = nn.Parameter(
            torch.empty(config.num_mosrah_heads, config.embedding_width)
        )
        nn.init.kaiming_normal_(self.routing_weight)

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        router_cache: RouterCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Route input tokens to K expert heads each and compute routing probabilities.

        Args:
            x: Input hidden states of shape (batch, seq_len, embedding_width).
            active_mask: Current-chunk active mask of shape (batch, seq_len), where
                True marks a semantically live token. Dead tokens do not contribute
                to regret_loss or logit_regret.

        Returns:
            selected_heads: Head indices I of shape (batch, seq_len, num_selected_heads).
                Each token's K selected head indices from the block-balanced solver.
            routing_probs: Routing probabilities P of shape (batch, seq_len,
                num_selected_heads). Gathered from the pre-balance softmax at
                selected_heads and renormalized to sum to 1 per token.
            router_diagnostics: Dict of routing scalars:
                - ``regret_loss``: gradient-carrying mean regret, mean of
                  max(p_max_active − p_chosen, 0) over live (B, num_blocks, L)
                  entries. In [0, 1]. Zero when every expert is assigned at its
                  peak-preference token within the block.
                - ``logit_regret``: detached logit-space regret; same formula
                  applied to routing logits rather than softmax probabilities.
                  In [0, ∞). Monitoring only.
                - ``logit_std``: detached mean per-token std of routing logits.
        """
        # ── Algorithm overview ──────────────────────────────────────────────────────
        #
        # Problem: each token independently selects its top-K heads with no knowledge
        # of what other tokens in the same sequence will choose. Independent selection
        # means a single popular head can be chosen by every token while another is
        # never used — statistics-based corrections (auxiliary losses, bias vectors)
        # can only push routing probabilistically and have proven unstable when tuned
        # strongly enough to prevent degeneracy.
        #
        # Approach: the compatibility constraint E % K == 0 (enforced in ShramConfig)
        # makes W = E / K an exact integer. A block of W consecutive tokens contains
        # exactly W × K = E selection slots — one per expert. Enforcing that each
        # expert is used exactly once per block makes the block perfectly balanced by
        # construction, eliminating any need for auxiliary losses or correction steps.
        # Enforcement is causal: at each of the W steps the current position picks its
        # K experts from those not yet claimed earlier in the same block, by masking
        # claimed experts with -inf before top-K. All W steps run simultaneously across
        # blocks and batch via a Python for loop that is fully unrolled at compile time.

        B, N, _ = x.shape
        L = self.num_mosrah_heads
        K = self.num_selected_heads
        W = self.block_length

        # ── Phase: pre-balance scoring ─────────────────────────────────────────
        #
        # Establishes the clean routing distribution before any -inf masking.
        # logit_std is captured here because the block solver's masking would
        # corrupt the standard deviation. routing_scores is used both for
        # regret_loss and for the final routing_probs.
        routing_logits = self._compute_routing_logits(x)                       # (B, N, L)
        logit_std      = routing_logits.std(dim=-1).mean().detach()
        routing_scores = F.softmax(routing_logits, dim=-1)                     # (B, N, L)

        # ── Phase: block-balanced causal selection ─────────────────────────────
        #
        # Three execution modes, distinguished by router_cache and sequence length:
        #
        # Training (router_cache is None): the full sequence is available. All W
        # steps of the block solver run simultaneously across every block in the
        # sequence. No cache interaction.
        #
        # Prefill (router_cache is not None, N > 1): identical to training, but
        # the partial last-block state is written to the cache so decode steps can
        # continue within the same block without a gap.
        #
        # Decode (router_cache is not None, N == 1): one token arrives at a known
        # position within the current block. The cached used_in_block mask is
        # applied before TopK to enforce the one-usage-per-block contract, then
        # the cache is updated in-place with this step's selections.

        if router_cache is not None and N == 1:
            # ── Decode mode ───────────────────────────────────────────────────
            #
            # Single token; block position and claimed-expert state come from the
            # cache. Treating this as a one-token, one-step block means the regret
            # computation downstream sees a (B, 1, 1, K) assignment tensor and
            # produces exactly zero regret, which is correct: with only one active
            # token per "block" there is no alternative assignment with higher
            # preference.
            used_in_block = router_cache.get_used_in_block()                   # (B, L)
            step_logits   = routing_logits[:, 0, :]                            # (B, L)
            available     = step_logits.masked_fill(used_in_block, float('-inf'))
            step_heads    = available.topk(K, dim=-1).indices                  # (B, K)

            router_cache.update_decode(step_heads)

            selected_heads = step_heads.unsqueeze(1)                           # (B, 1, K)
        else:
            # ── Training / prefill mode ───────────────────────────────────────
            #
            # The full N-token sequence is available. Padding extends it to a
            # multiple of W; padded tokens occupy the tail of the last block and
            # never consume experts needed by real tokens because the real tokens
            # preceding them have already had their pick each step. The pad is
            # discarded after the solver.
            num_blocks = (N + W - 1) // W
            N_pad      = num_blocks * W
            pad_len    = N_pad - N

            if pad_len > 0:
                padded_logits = torch.cat(
                    [routing_logits, routing_logits.new_zeros(B, pad_len, L)], dim=1
                )                                                               # (B, N_pad, L)
            else:
                padded_logits = routing_logits

            blocked_logits = padded_logits.view(B, num_blocks, W, L)           # (B, blk, W, L)

            # used_in_block tracks which experts have been claimed within each block.
            # No gradient here — expert availability is a hard structural constraint,
            # not a differentiable quantity. Gradient flows through routing_probs.
            used_in_block   = torch.zeros(B, num_blocks, L, dtype=torch.bool, device=x.device)
            step_heads_list = []

            for step in range(W):
                step_logits = blocked_logits[:, :, step, :]                    # (B, blk, L)

                # Claimed experts receive -inf so top-K never selects them.
                available  = step_logits.masked_fill(used_in_block, float('-inf'))
                step_heads = available.topk(K, dim=-1).indices                 # (B, blk, K)
                step_heads_list.append(step_heads)

                # Mark the K chosen experts as unavailable for the rest of this block.
                used_in_block = used_in_block.scatter(-1, step_heads, True)

            # Stack W steps and reshape to (B, N_pad, K), then unpad.
            selected_heads_blocked = torch.stack(step_heads_list, dim=2)       # (B, blk, W, K)
            selected_heads         = selected_heads_blocked.view(B, N_pad, K)[:, :N, :]  # (B, N, K)

            if router_cache is not None:
                # Prefill: persist the partial last-block state so decode steps
                # that follow can continue within the same block.
                router_cache.update_prefill(selected_heads_blocked, N)

        # ── Phase: regret loss ─────────────────────────────────────────────────
        #
        # Regret measures how much routing preference was sacrificed at each expert
        # assignment relative to the peak active preference within the same block.
        # A non-zero regret at expert l in block bl means some other active token
        # in that block would have preferred expert l more than the one assigned.
        # Minimising regret trains the router to save experts for the tokens that
        # want them most.
        #
        # Decode mode returns zeros: regret is only defined over complete W-token
        # blocks, and a single decode step is not a complete block. Backward is
        # never called during inference so the zero is a correct no-op.
        if router_cache is not None and N == 1:
            regret_loss  = routing_logits.new_zeros(())
            logit_regret = routing_logits.new_zeros(()).detach()
        else:
            regret_loss, logit_regret = self._compute_regret(
                routing_scores,
                routing_logits,
                selected_heads_blocked,
                active_mask,
            )

        # ── Phase: routing probabilities ────────────────────────────────────────
        #
        # Gathered from the pre-balance routing_scores to reflect genuine routing
        # preference; renormalized so they sum to 1 per token.
        gathered      = routing_scores.gather(dim=-1, index=selected_heads)    # (B, N, K)
        routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)          # (B, N, K)

        router_diagnostics = {
            "regret_loss":  regret_loss,
            "logit_regret": logit_regret,
            "logit_std":    logit_std,
        }
        return selected_heads, routing_probs, router_diagnostics

    def _compute_routing_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-head routing logits from input hidden states.

        Args:
            x: Input hidden states, shape (batch, seq_len, embedding_width).

        Returns:
            Routing logits, shape (batch, seq_len, num_mosrah_heads).
        """
        return F.linear(x, self.routing_weight)                                # (B, N, L)

    @staticmethod
    def _compute_regret(
        routing_scores: torch.Tensor,
        routing_logits: torch.Tensor,
        selected_heads_blocked: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute regret_loss and logit_regret from a completed block assignment.

        Regret at expert l in block bl = max(p_max_active − p_chosen, 0), where
        p_max_active is the highest routing probability any active token holds for
        expert l within the block, and p_chosen is the routing probability of the
        token actually assigned to expert l (0 if that token is dead).

        regret_loss is the mean over live (batch, block, expert) triples. A block is
        live iff it contains at least one active token; all L experts in a live block
        contribute. Result is in [0, 1].

        logit_regret applies the same formula to routing_logits and is returned
        detached — it is a monitoring scalar only, in [0, ∞).

        Args:
            routing_scores:         Softmax routing probabilities, shape (B, N, L).
                                    Gradient flows through this tensor into regret_loss.
            routing_logits:         Pre-softmax routing logits, shape (B, N, L).
                                    Used only for the detached logit_regret.
            selected_heads_blocked: Expert assignments from the block solver,
                                    shape (B, num_blocks, W, K). Block geometry
                                    (num_blocks, W) is derived from this shape.
            active_mask:            Boolean live-token mask, shape (B, N).

        Returns:
            regret_loss:   Gradient-carrying scalar in [0, 1].
            logit_regret:  Detached scalar in [0, ∞).
        """
        B, num_blocks, W, _K = selected_heads_blocked.shape
        L   = routing_scores.shape[-1]
        N   = routing_scores.shape[1]
        N_pad = num_blocks * W

        # ── Reshape into block form ─────────────────────────────────────────
        #
        # Block geometry is read from selected_heads_blocked — no recomputation
        # needed here. Padded tail positions receive zero scores and False
        # activity; they do not contribute to any block metric.
        if N_pad > N:
            pad_len = N_pad - N
            scores_blocked = torch.cat(
                [routing_scores, routing_scores.new_zeros(B, pad_len, L)], dim=1
            ).view(B, num_blocks, W, L)                                        # (B, nb, W, L)
            logits_blocked = torch.cat(
                [routing_logits, routing_logits.new_zeros(B, pad_len, L)], dim=1
            ).view(B, num_blocks, W, L)                                        # (B, nb, W, L)
            active_blocked = torch.cat(
                [active_mask, active_mask.new_zeros(B, pad_len)], dim=1
            ).view(B, num_blocks, W)                                           # (B, nb, W)
        else:
            scores_blocked = routing_scores.view(B, num_blocks, W, L)
            logits_blocked = routing_logits.view(B, num_blocks, W, L)
            active_blocked = active_mask.view(B, num_blocks, W)

        active_float = active_blocked.float()                                  # (B, nb, W)
        block_active = active_blocked.any(dim=-1)                              # (B, nb)

        # ── Assignment mask ─────────────────────────────────────────────────
        #
        # One-hot indicator of which token was assigned to each expert. Block
        # balance guarantees exactly one entry per (b, bl, l) triple, so
        # summing over W recovers exactly one score value per expert.
        assignment_mask = scores_blocked.new_zeros(B, num_blocks, W, L)
        assignment_mask.scatter_(dim=-1, index=selected_heads_blocked, value=1.0)
                                                                               # (B, nb, W, L)

        # ── Prob regret (gradient flows through routing_scores) ─────────────
        #
        # p_chosen: routing score at the assigned token, gated by active_float
        # so dead assignments contribute 0 — the expert accrues full regret
        # against the active maximum rather than no penalty.
        # p_max: peak routing score over active tokens; dead tokens zeroed before
        # max (safe because softmax outputs are non-negative).
        p_chosen = (assignment_mask * active_float.unsqueeze(-1) * scores_blocked).sum(dim=2)
                                                                               # (B, nb, L)
        p_max    = (active_float.unsqueeze(-1) * scores_blocked).max(dim=2).values
                                                                               # (B, nb, L)

        regret = (p_max - p_chosen).clamp(min=0.0)                            # (B, nb, L)

        # Mean over live (B, num_blocks, L) entries. Clamped to 1 for the
        # all-dead edge case where the numerator is already 0.
        num_live    = block_active.float().sum()                               # scalar
        regret_loss = (
            block_active.float().unsqueeze(-1) * regret
        ).sum() / num_live.mul(L).clamp(min=1.0)

        # ── Logit regret (detached monitoring) ──────────────────────────────
        #
        # Same formula applied to routing_logits. Dead tokens cannot be zeroed
        # before max (logits may be negative), so they are masked to -inf;
        # dead blocks are replaced with 0 before subtraction. Detached so it
        # never influences any parameter during backward.
        logit_chosen = (
            assignment_mask * active_float.unsqueeze(-1) * logits_blocked
        ).sum(dim=2)                                                           # (B, nb, L)

        logit_max = logits_blocked.masked_fill(
            ~active_blocked.unsqueeze(-1), float('-inf')
        ).max(dim=2).values                                                    # (B, nb, L)
        logit_max = logit_max.masked_fill(~block_active.unsqueeze(-1), 0.0)

        logit_regret = (
            block_active.float().unsqueeze(-1) * (logit_max - logit_chosen).clamp(min=0.0)
        ).sum() / num_live.mul(L).clamp(min=1.0)
        logit_regret = logit_regret.detach()

        return regret_loss, logit_regret
