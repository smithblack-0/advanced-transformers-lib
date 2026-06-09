"""Token-choice router for the MoSRAH sparse attention path.

This module implements the routing mechanism described in Appendix A.Routing of the
paper. Given an input hidden state x, the router produces two outputs used downstream:

  - selected_heads (I): which K of the L available expert heads each token routes to,
    determined by TopK over capacity-balanced routing scores.
  - routing_probs (P): the weights used for the weighted output reduction, gathered from
    the routing scores at the selected indices and renormalized to sum to 1 per token.

Routing uses a single learnable projection:

  - routing_weight: shape (L, embedding_width). Maps input to per-head routing scores.
    Both task loss and load_balance_loss train this parameter directly — there is no
    gradient isolation between the two signals.

This coupled design is intentional. SHRAM has an unusually strong task-level incentive
to concentrate tokens into the same expert bucket (sparse attention only occurs among
tokens routed to the same expert), so any indirect balancing pathway will be outlearned.
Coupling the gradients allows the load balance loss to act with full strength directly
on the parameter that determines routing.

routing_weight is nn.Parameter so that HuggingFace _init_weights does not override
its kaiming initialization at construction.

routing_probs are computed before balance_capacity applies -1e8 sentinels. Post-capacity
softmax would corrupt routing_probs for over-capacity experts (near-zero probability
after masking does not reflect genuine routing preference).

The router computes and returns:
  - load_balance_loss: scalar auxiliary loss (see load_balance_loss.py); gradient flows
    to routing_weight.
  - max_vio: detached scalar summarising routing imbalance:
      MaxVio = mean_b( L · max_l(f_bl − 1/L) )
    where f_bl is the per-batch-item realised routing frequency of head l. Zero means
    perfect balance; 1.0 means the most loaded head received double its fair share.
  - logit_std: detached scalar; mean per-token standard deviation of routing logits.
    Monitoring metric for routing sharpness.

Paper ref: Appendix A.Routing, Appendix A.Load Balancing, §MaxVio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration import ShramConfig
from .load_balance_loss import make_load_balance_loss, reduce_frequency_tokens

class MoSRAHRouter(nn.Module):
    """Token-choice router for MoSRAH sparse attention.

    Each input token independently selects K of the L available expert heads.
    A single routing projection maps input hidden states to per-head scores; both
    task loss and load_balance_loss train this projection directly.

    routing_weight is nn.Parameter rather than nn.Linear so that HuggingFace
    _init_weights does not override its kaiming initialization at construction.

    Attributes:
        routing_weight: Shape (L, embedding_width). Maps input hidden states to
            per-head routing scores. Receives gradients from both task loss and
            load_balance_loss.

    Args:
        config: Model configuration. Must expose ``embedding_width``,
            ``num_mosrah_heads`` (L), ``num_selected_heads`` (K),
            ``load_balance_loss_type``, ``maximum_expert_overclaim``, ``max_bid_rounds``,
            ``use_cache``, ``mosrah_cache_length``, and ``mosrah_packed_length``.
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.num_mosrah_heads = config.num_mosrah_heads
        self.num_selected_heads = config.num_selected_heads
        if config.use_cache:
            self.capacity = config.mosrah_cache_length
        else:
            self.capacity = config.mosrah_packed_length

        self.max_bid_rounds = config.max_bid_rounds
        self._load_balance_loss = make_load_balance_loss(
            config.load_balance_loss_type,
            num_selected_heads=config.num_selected_heads,
            num_total_heads=config.num_mosrah_heads,
            maximum_expert_overclaim=config.maximum_expert_overclaim,
        )

        # Routing projection: maps input (B, N, d) to per-head routing scores (B, N, L).
        # nn.Parameter ensures HuggingFace _init_weights does not override kaiming init.
        self.routing_weight = nn.Parameter(
            torch.empty(config.num_mosrah_heads, config.embedding_width)
        )
        nn.init.kaiming_normal_(self.routing_weight)

    @staticmethod
    def get_best_proposals(
            tensor: torch.Tensor,
            dim: int,
            n: int | torch.Tensor,
            capacity_scalar: int,
    ) -> torch.Tensor:
        """Return a boolean mask selecting the top-n entries along dim.

        Uses topk to select exactly min(n_per_slice, dim_length) True entries
        per slice along dim. Unlike a threshold comparison, this never
        over-selects under tied logit values, which occurs when padding tokens
        contribute identical scores to multiple expert slots.

        Args:
            tensor: Input tensor. Higher values rank first.
            dim: Dimension to select along.
            n: Per-slice selection count. Scalar int or tensor broadcastable
               to tensor with dim removed. Slices where n=0 produce all-False
               outputs.
            capacity_scalar: Static upper bound on n; used to derive topk k as
               min(tensor.shape[dim], capacity_scalar). Must be a Python int
           for compile compatibility.

        Returns:
            Boolean mask of the same shape as tensor.
        """
        positive_dim = dim % tensor.ndim
        dim_length = tensor.shape[positive_dim]
        k = min(dim_length, capacity_scalar)

        topk_indices = tensor.topk(k, dim=dim).indices

        # Rank tensor broadcast-compatible with topk_indices: rank r along dim
        # corresponds to the (r+1)-th highest value in that slice.
        rank_shape = [1] * tensor.ndim
        rank_shape[positive_dim] = k
        ranks = torch.arange(k, device=tensor.device, dtype=torch.long).view(rank_shape)

        # element_included: True where this rank falls within the per-slice budget.
        # For scalar n all k ranks satisfy rank < n (since k = min(dim_length, n)).
        # For tensor n per-slice budgets differ; rank >= n[slice] yields False,
        # correctly excluding excess slots including those with n=0.
        if isinstance(n, int):
            element_included = ranks < n
        else:
            element_included = ranks < n.unsqueeze(positive_dim)

        # Allocate from explicit logical shape rather than using zeros_like. This keeps
        # the output mask tied to tensor.shape, not to any stride/layout metadata carried
        # by tensor from earlier view operations or compiler lowering.
        mask = torch.zeros(
            tuple(tensor.shape),
            device=tensor.device,
            dtype=torch.bool,
        )

        # Materialize the scatter source shape explicitly. This avoids passing a
        # broadcast-view source into scatter while preserving the same logical rule:
        # every selected top-k index receives True iff its rank is within budget.
        scatter_values = torch.broadcast_to(element_included, topk_indices.shape)
        mask = mask.scatter(dim, topk_indices, scatter_values)
        return mask

    @staticmethod
    def _check_bidding_converged(acceptances: torch.Tensor,
                                 min_choices: int,
                                 max_rounds: int) -> None:
        """Raise if the bidding loop exhausted max_rounds without satisfying all tokens.

        Args:
            acceptances: bool tensor of shape (B, N, L) indicating what experts L accepted
                what tokens.
            min_choices: Convergence has been reached if acceptances are such that a sum along
                N always has at least min_choices choices.
            max_rounds: The iteration ceiling that was applied, for the error message. Used
                for reporting
        """
        msg = (
            f"balance_capacity bidding did not converge within {max_rounds} rounds. "
            f"Increase mosrah_overallocation_factor or max_bid_rounds."
        )
        converged = (acceptances.sum(dim=-1) >= min_choices).all()
        torch._assert_async(converged, msg)

    @classmethod
    def _run_bidding(
            cls,
            logits: torch.Tensor,
            remaining_capacity: int | torch.Tensor,
            min_choices: int,
            max_rounds: int,
            capacity_scalar: int,
    ) -> torch.Tensor:
        """Deferred-acceptance (Gale-Shapley) bidding solver for joint capacity enforcement.

        Tokens propose experts in descending preference order; experts provisionally
        accept their top-``remaining_capacity`` proposed tokens each round. Proposals
        are monotone (never retracted), so once all tokens are satisfied, subsequent
        iterations are no-ops. Runs unconditionally for exactly ``max_rounds`` iterations
        to keep the compiled graph flat and free of data-dependent control flow.

        Both the column bound (per-expert token count ≤ remaining_capacity) and the
        row bound (per-token expert count ≥ min_choices) are satisfied simultaneously
        on the returned mask by construction.

        Args:
            logits: Routing scores of shape (B, N, L).
            remaining_capacity: Per-expert token budget. Scalar int for training;
                (B, L) tensor for inference.
            min_choices: Minimum experts each token must have accepted (K).
            max_rounds: Number of iterations to run. Convergence is checked after
                all rounds via ``_check_bidding_converged``; raises if not met.
            capacity_scalar: Static upper bound on remaining_capacity, passed to
                ``get_mask`` as the topk k bound for the acceptance step.

        Returns:
            accepted: (B, N, L) bool — True at positions accepted by the solver.
        """
        proposals   = torch.zeros_like(logits, dtype=torch.bool)
        acceptances = torch.zeros_like(logits, dtype=torch.bool)

        for _ in range(max_rounds):
            # ── token proposal step ───────────────────────────────────────────
            #
            # Tokens with fewer than min_choices accepted experts propose their
            # next-best unproposed expert(s). The deficit determines how many new
            # proposals each token makes; satisfied tokens propose nothing
            # (deficit = 0 → get_mask returns all-False). Proposals are monotone:
            # once all tokens are satisfied, subsequent iterations are no-ops.
            accepted_per_token = acceptances.sum(dim=-1)           # (B, N)
            choices_deficit = (min_choices - accepted_per_token).clamp_min(0)

            unproposed_logits = logits.masked_fill(proposals, float('-inf'))
            new_proposals = cls.get_best_proposals(
                unproposed_logits, dim=-1, n=choices_deficit, capacity_scalar=min_choices,
            )
            proposals = proposals | new_proposals

            # ── expert acceptance step ────────────────────────────────────────
            #
            # Each expert accepts its top-remaining_capacity proposed tokens.
            # Acceptances are recomputed from scratch each round so that a
            # stronger new proposal can displace a weaker prior one.
            proposed_logits = logits.masked_fill(~proposals, float('-inf'))
            acceptances = cls.get_best_proposals(
                proposed_logits, dim=-2, n=remaining_capacity, capacity_scalar=capacity_scalar,
            )

        return acceptances

    @classmethod
    def balance_capacity(
            cls,
            logits: torch.Tensor,
            used_capacity: torch.Tensor | None,
            capacity: int,
            min_choices: int,
            max_rounds: int,
            mask_value: float = -1e8,
    ) -> torch.Tensor:
        """Mask logits so both capacity constraints hold simultaneously on the output.

        Two constraints must hold:
          - Column bound: per-expert unmasked token count ≤ remaining_capacity.
          - Row bound:    per-token unmasked expert count ≥ min_choices.

        A training fast path is attempted before the bidding solver:

        1. Training with N ≤ capacity: return logits unchanged.
        2. Bidding: deferred-acceptance solver guaranteeing both bounds simultaneously.

        Args:
            logits: Routing scores of shape (B, N, L).
            used_capacity: Tokens already accumulated per expert, shape (B, L).
                ``None`` during training (full capacity available).
            capacity: Maximum tokens per expert (from config).
            min_choices: Minimum experts each token must retain (K).
            max_rounds: Bidding iteration ceiling (from config.max_bid_rounds).
            mask_value: Value written to masked positions. Default -1e8.

        Returns:
            Logits with unavailable positions set to ``mask_value``, shape (B, N, L).
        """
        # ── Algorithm overview ────────────────────────────────────────────────
        #
        # Problem: mask (B, N, L) logits so that both the column bound (each
        # expert receives at most remaining_capacity tokens) and the row bound
        # (each token retains at least min_choices expert choices) hold
        # simultaneously. Satisfying either constraint greedily can violate the
        # other, requiring a joint solver for the hard case.
        #
        # Approach: deferred-acceptance (Gale-Shapley) bidding. Each round,
        # tokens that still lack min_choices accepted experts propose their
        # next-best unproposed expert. Each expert then provisionally accepts its
        # top-remaining_capacity proposed tokens, potentially displacing weaker
        # prior acceptances. Proposals are monotone (never retracted). The loop
        # terminates when every token has min_choices accepted experts or
        # max_bid_rounds is exhausted (RuntimeError in the latter case).
        #
        # Training fast path — when N ≤ capacity and all experts start empty,
        # no expert can overflow regardless of routing. No masking is needed.

        # Training fast path: N ≤ capacity with empty experts → no overflow possible.
        if used_capacity is None and logits.shape[-2] <= capacity:
            return logits

        # Compute per-expert remaining budget.
        # Training (N > capacity path): scalar — all experts start with full capacity.
        # Inference: subtract already-accumulated tokens; clamp prevents negatives
        #            when rounding causes used_capacity to slightly exceed capacity.
        if used_capacity is None:
            remaining_capacity = capacity
        else:
            remaining_capacity = (capacity - used_capacity).clamp(min=0)  # (B, L)

        # Bidding solver: jointly satisfies column and row bounds. Runs under
        # no_grad because the boolean mask is a hard routing decision and must
        # not accumulate gradient memory.
        with torch.no_grad():
            final_mask = cls._run_bidding(logits, remaining_capacity,
                                          min_choices, max_rounds, capacity)
            cls._check_bidding_converged(final_mask, min_choices, max_rounds)
        return logits.masked_fill(~final_mask, mask_value)

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        used_capacity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Route input tokens to K expert heads each and compute routing probabilities.

        Args:
            x: Input hidden states of shape (batch, seq_len, embedding_width).
            active_mask: Current-chunk active mask of shape (batch, seq_len), where
                True means the token is semantically live. Dead tokens do not
                contribute to routing frequencies, load_balance_loss, or max_vio.
            used_capacity: Used for capacity management during inference, missing during training.

        Returns:
            selected_heads: Head indices I of shape (batch, seq_len, num_selected_heads).
                Each token's K selected head indices, determined by TopK on
                capacity-balanced routing scores.
            routing_probs: Routing probabilities P of shape (batch, seq_len,
                num_selected_heads). Gathered from pre-capacity routing softmax at
                selected_heads indices and renormalized to sum to 1 per token.
            router_diagnostics: Dict of routing feedback scalars. Keys:
                - ``load_balance_loss``: scalar load-balance loss with gradient.
                - ``max_vio``: detached scalar routing-imbalance summary.
                - ``logit_std``: detached mean per-token std of routing logits;
                  monitoring metric for routing sharpness.
        """
        B, N, _ = x.shape
        L = self.num_mosrah_heads
        K = self.num_selected_heads

        # ── Phase: pre-capacity scoring ───────────────────────────────────────
        #
        # Establishes the clean pre-sentinel distribution that all downstream
        # consumers draw from. logit_std must be captured here — balance_capacity
        # injects -1e8 sentinels that would corrupt the standard deviation.
        # routing_scores is the pre-capacity probability distribution; both the
        # load balance signal and the final routing_probs gather from it.
        routing_logits = self._compute_routing_logits(x)                       # (B, N, L)
        logit_std      = routing_logits.std(dim=-1).mean().detach()
        routing_scores = F.softmax(routing_logits, dim=-1)                     # (B, N, L)

        # ── Phase: load balance signal ────────────────────────────────────────
        #
        # The loss must observe the unconstrained routing decision — the genuine
        # routing pressure before capacity enforcement masks any imbalance.
        # pre_cap_heads and assignment_mask exist solely to give the loss this
        # honest view; nothing downstream uses them.
        pre_cap_heads   = routing_scores.topk(K, dim=-1).indices               # (B, N, K)
        assignment_mask = torch.zeros(B, N, L, device=x.device, dtype=x.dtype)
        assignment_mask.scatter_(-1, pre_cap_heads, 1.0)

        load_balance_loss = self._load_balance_loss(
            routing_logits, assignment_mask, active_mask
        )

        # ── Phase: capacity enforcement and final selection ───────────────────
        #
        # Produces the capacity-enforced routing that all downstream consumers
        # depend on. max_vio is computed here because it measures realized routing
        # imbalance — the actual post-capacity assignment, not the unconstrained
        # preference. routing_probs are gathered from the pre-capacity routing_scores
        # (not the balanced distribution) to avoid sentinel corruption — overloaded
        # experts would otherwise receive near-zero probability regardless of genuine
        # routing preference.
        balanced_logits = self.balance_capacity(
            routing_logits,
            used_capacity,
            self.capacity,
            self.num_selected_heads,
            self.max_bid_rounds,
        )
        selected_heads = F.softmax(balanced_logits, dim=-1).topk(K, dim=-1).indices  # (B, N, K)

        realized_mask = torch.zeros(B, N, L, device=x.device, dtype=x.dtype)
        realized_mask.scatter_(-1, selected_heads, 1.0)
        max_vio = self._compute_max_vio(realized_mask, active_mask, L)

        gathered      = routing_scores.gather(dim=-1, index=selected_heads)    # (B, N, K)
        routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)          # P, (B, N, K)

        router_diagnostics = {
            "load_balance_loss": load_balance_loss,
            "max_vio":           max_vio,
            "logit_std":         logit_std,
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
    def _compute_max_vio(
        assignment_mask: torch.Tensor,
        active_mask: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        """Compute the MaxVio routing-imbalance scalar.

        MaxVio = mean_b( L · max_l(f_bl − 1/L) ), where f_bl is the per-batch-item
        realised routing frequency of head l. Uses reduce_frequency_tokens for consistent
        per-batch-item frequency computation with dead tokens excluded, matching how the
        load balance loss computes frequencies. A value of zero indicates perfect balance;
        a value of 0.5 means the most overloaded head in the average batch item received
        50% more routed tokens than ideal.

        The result is detached — MaxVio is a monitoring scalar and must not contribute
        gradients to any parameter.

        Args:
            assignment_mask: Per-token head-assignment indicators, shape (B, N, L).
            active_mask:     Boolean active-token mask, shape (B, N).
            num_heads:       Total number of MoSRAH heads L.

        Returns:
            Detached scalar MaxVio tensor.
        """
        f_bl = reduce_frequency_tokens(assignment_mask, active_mask)                   # (B, L)
        per_item_max_vio = num_heads * (f_bl - 1.0 / num_heads).max(dim=-1).values    # (B,)
        return per_item_max_vio.mean().detach()
