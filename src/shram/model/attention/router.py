"""Token-choice router for the MoSRAH sparse attention path.

This module implements the routing mechanism described in Appendix A.Routing of the
paper. Given an input hidden state x, the router produces two outputs used downstream:

  - selected_heads (I): which K of the L available expert heads each token routes to,
    determined by TopK over capacity-balanced semantic routing scores.
  - routing_probs (P): the weights used for the weighted output reduction, gathered from
    the semantic routing scores at the selected indices and renormalized to sum to 1
    per token.

Base routing uses two learnable projection matrices and two gradient-isolated pathways:

  - routing_weight (A): shape (L, embedding_width). Maps input to per-head routing
    scores. Receives gradients from task loss; balance_weight is isolated.
  - balance_weight (B): shape (L, embedding_width). Maps input to per-head load-balance
    correction scores. Receives gradients from load_balance_loss; routing_weight is
    isolated.

The two gradient-isolated base pathways over numerically identical values:

  - semantic_logits = A·x + (B·x).detach(): task gradients reach routing_weight;
    balance_weight is isolated from task loss.
  - load_balancing_logits = (A·x).detach() + B·(x.detach()): load balance gradients
    reach balance_weight; routing_weight and x are isolated from load balance loss.

Integral routing extension (routing_mode == "integral"):

Standard routing is parallel — each token routes based on its own hidden state alone,
with no direct read on what earlier tokens in the sequence have already selected.
Integral routing adds a cumulative-sum signal that gives each token a view of the
prior routing history within the sequence.

Two additional (L, L) parameter matrices are introduced:

  - routing_integral_weight (A'): shape (L, L). Maps the cumulative logit history to
    per-head semantic corrections. Receives gradients from task loss.
  - balance_integral_weight (B'): shape (L, L). Maps the cumulative logit history to
    per-head load-balance corrections. Receives gradients from load_balance_loss.

The cumulative history signal u is the exclusive cumsum of the base logits along the
sequence dimension: u[n] = sum(logits[0..n-1]), shape (B, N, L). Position 0 receives
zeros (no prior history). The same gradient isolation pattern as A/B applies:

  - semantic_logits   += A'·u_semantic + (B'·u_semantic).detach()
  - lb_logits         += (A'·u_load).detach() + B'·u_load

Detaching the full B'·u_semantic result (rather than just B') mirrors the
(B·x).detach() pattern in the base pathway and prevents double-counting the
cumsum gradient path back to routing_weight.

Both base matrices and both integral matrices are nn.Parameter so that HuggingFace
_init_weights does not override their kaiming initialization at construction.

Assignment probabilities are computed before balance_capacity applies -1e8 sentinels.
Post-capacity softmax would invert the load balance gradient for over-capacity experts
(near-zero probability after masking signals "increase corrections" for an already-
overloaded expert).

The router also computes and returns the load balance loss via a log-probability auxiliary
loss (see load_balance_loss.py). The loss formulation is selected by config; the default
is cross-entropy.

The router additionally computes and returns MaxVio, a detached scalar summarising
routing imbalance for the current forward pass:

    MaxVio = mean_b( L · max_l(f_bl − 1/L) )

where f_bl is the per-batch-item realised routing frequency of head l and 1/L is the
perfectly balanced target. MaxVio is averaged over batch items and is a monitoring
quantity only; it never contributes gradients.

Paper ref: Appendix A.Routing, Appendix A.Load Balancing, §MaxVio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration import ShramConfig
from .load_balance_loss import make_load_balance_loss, reduce_frequency_tokens

class MoSRAHRouter(nn.Module):
    """Token-choice router for MoSRAH sparse attention.

    Each input token independently selects K of the L available expert heads. Both
    selection and routing_probs incorporate balance_weight via two gradient-isolated
    pathways over numerically identical values. See module docstring for the
    two-pathway architecture and the integral routing extension.

    All four learnable matrices are nn.Parameter rather than nn.Linear so that
    HuggingFace _init_weights does not override their kaiming initialization at
    construction.

    Attributes:
        routing_weight: A, shape (L, embedding_width). Task-loss pathway.
        balance_weight: B, shape (L, embedding_width). Load-balance pathway.
        routing_integral_weight: A', shape (L, L). Integral task-loss pathway.
            Present only when ``routing_mode == "integral"``.
        balance_integral_weight: B', shape (L, L). Integral load-balance pathway.
            Present only when ``routing_mode == "integral"``.
        routing_mode: ``"integral"`` or ``"default"``, from config.

    Args:
        config: Model configuration. Must expose ``embedding_width``,
            ``num_mosrah_heads`` (L), ``num_selected_heads`` (K), and
            ``routing_mode``.
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
        self.routing_mode = config.routing_mode
        self._load_balance_loss = make_load_balance_loss(config.load_balance_loss_type)

        # W_r (A): semantic routing matrix. Maps input (B, N, d) to per-head routing
        # scores (B, N, L) for selection and routing_probs. nn.Parameter ensures
        # HuggingFace _init_weights does not override kaiming initialization.
        self.routing_weight = nn.Parameter(
            torch.empty(config.num_mosrah_heads, config.embedding_width)
        )
        nn.init.kaiming_uniform_(self.routing_weight)

        # W_b (B): load-balancing projection matrix. Maps input (B, N, d) to per-head
        # correction scores (B, N, L). Receives gradients only from load_balance_loss.
        # nn.Parameter ensures HuggingFace _init_weights does not override kaiming init.
        self.balance_weight = nn.Parameter(
            torch.empty(config.num_mosrah_heads, config.embedding_width)
        )
        nn.init.kaiming_uniform_(self.balance_weight)

        if self.routing_mode == "integral":
            L = config.num_mosrah_heads
            # A': integral semantic matrix. Maps cumulative logit history (B, N, L) to
            # per-head semantic corrections (B, N, L). Shape (L, L). Receives gradients
            # from task loss; balance_integral_weight is isolated from task loss.
            # Zero-initialized so that corrections start at zero and grow from gradient
            # updates — kaiming init produces corrections that immediately overwhelm the
            # base routing signal via the cumsum feedback path.
            self.routing_integral_weight = nn.Parameter(torch.zeros(L, L))

            # B': integral load-balance matrix. Maps cumulative logit history (B, N, L)
            # to per-head load-balance corrections (B, N, L). Shape (L, L). Receives
            # gradients from load_balance_loss; routing_integral_weight is isolated.
            # Zero-initialized for the same reason as routing_integral_weight.
            self.balance_integral_weight = nn.Parameter(torch.zeros(L, L))

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
                capacity-balanced semantic scores.
            routing_probs: Routing probabilities P of shape (batch, seq_len,
                num_selected_heads). Gathered from pre-capacity semantic softmax at
                selected_heads indices and renormalized to sum to 1 per token.
            router_diagnostics: Dict of routing feedback scalars. Keys:
                - ``load_balance_loss``: scalar load-balance loss with gradient.
                - ``max_vio``: detached scalar routing-imbalance summary.
                - ``raw_logit_std``: mean per-token std of routing_logits; natural
                  routing preference scale and baseline for interpreting bias_std.
                - ``bias_std``: mean per-token std of balance_logits; near-zero
                  means balance corrections have not built up relative to routing scale.
                - ``logit_std``: mean per-token std of semantic_logits; lower than
                  raw_logit_std means balance is flattening preferences (healthy correction).
                - ``bias_alignment``: mean cosine similarity of routing_logits vs
                  balance_logits per token. Negative means balance opposes routing direction
                  (healthy correction); positive means runaway reinforcement.
        """
        B, N, _ = x.shape
        L = self.num_mosrah_heads
        K = self.num_selected_heads

        logits = self._compute_routing_logits(x, active_mask)

        # Diagnostic scalars characterising the two routing pathways. Must be computed
        # before balance_capacity injects -1e8 sentinels that would corrupt std and
        # cosine similarity. Extracted to _compute_bias_diagnostics to keep the forward
        # body free of non-(B,N,L) reduction logic.
        bias_diagnostics = self._compute_bias_diagnostics(
            logits["routing_logits"], logits["balance_logits"], logits["semantic_logits"]
        )

        # Pre-capacity semantic softmax for gathering routing_probs. Computed before
        # balance_capacity so that gathered probabilities reflect genuine preference
        # magnitudes rather than hard-masked sentinel values.
        routing_scores = F.softmax(logits["semantic_logits"], dim=-1)          # (B, N, L)

        # Capacity-balanced semantic logits for selection. Injects -1e8 into positions
        # that would exceed per-expert token budget, enforcing the packing constraint.
        balanced_semantic_logits = self.balance_capacity(
            logits["semantic_logits"],
            used_capacity,
            self.capacity,
            self.num_selected_heads,
            self.max_bid_rounds,
        )
        selection_scores = F.softmax(balanced_semantic_logits, dim=-1)    # (B, N, L)

        # selected_heads I = TopK over capacity-balanced semantic scores.
        selected_heads = selection_scores.topk(K, dim=-1).indices          # (B, N, K)

        # Routing probabilities P: gathered from pre-capacity semantic softmax at
        # selected_heads positions, renormalized so they sum to 1 per token.
        gathered      = routing_scores.gather(dim=-1, index=selected_heads)    # (B, N, K)
        routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)          # P, (B, N, K)

        # assignment_mask: (B, N, L) float — 1.0 at each token's K selected heads, 0 elsewhere.
        # The discrete routing decision; no gradient flows through it. Passed alongside
        # load_balancing_logits and active_mask to the loss and max_vio methods, which
        # own all frequency aggregation and reduction internally.
        assignment_mask = torch.zeros(B, N, L, device=x.device, dtype=x.dtype)
        assignment_mask.scatter_(-1, selected_heads, 1.0)

        load_balance_loss = self._load_balance_loss(
            logits["load_balancing_logits"], assignment_mask, active_mask
        )

        # MaxVio: detached monitoring scalar averaged over batch items. Computed from
        # the same (B, N, L) assignment_mask so frequencies are consistent with the loss.
        max_vio = self._compute_max_vio(assignment_mask, active_mask, L)

        router_diagnostics = {
            "load_balance_loss": load_balance_loss,
            "max_vio": max_vio,
            **bias_diagnostics,
        }
        return selected_heads, routing_probs, router_diagnostics

    @staticmethod
    def exclusive_cumsum(logits: torch.Tensor) -> torch.Tensor:
        """Compute the exclusive cumulative sum along the sequence dimension.

        u[n] = sum(logits[0..n-1]): position n receives the accumulated sum of all
        prior positions, giving it a read on the routing preferences expressed by
        earlier tokens in the sequence. Position 0 always receives zeros — no prior
        history exists at the first position.

        Args:
            logits: Shape (B, N, L). Any per-head score tensor along a sequence.

        Returns:
            Exclusive cumsum, shape (B, N, L). Same dtype and device as input.
        """
        shifted = torch.cat(
            [torch.zeros_like(logits[:, :1, :]), logits[:, :-1, :]], dim=1
        )
        return shifted.cumsum(dim=1)

    def _compute_routing_logits(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the gradient-isolated logit pathways from input hidden states.

        Base pathways (both modes):

          Two gradient-isolated pathways over numerically identical values:
          - semantic_logits = A·x + (B·x).detach(): task gradients reach routing_weight;
            balance_weight is isolated from task loss.
          - load_balancing_logits = (A·x).detach() + B·(x.detach()): load balance
            gradients reach balance_weight; routing_weight and x are isolated.

        Integral extension (routing_mode == "integral"):

          Dead tokens are zeroed out of the logits before computing the cumsum, so
          inactive positions do not contribute to the routing history of downstream
          live tokens. u_semantic and u_load therefore represent history from live
          tokens only.

          u_semantic = exclusive_cumsum(semantic_logits * active_mask)    — (B, N, L)
          u_load     = exclusive_cumsum(load_balancing_logits * active_mask) — (B, N, L)

          semantic_logits       += A'·u_semantic + (B'·u_semantic).detach()
          load_balancing_logits += (A'·u_load).detach() + B'·u_load

          Detaching the full (B'·u_semantic) result mirrors the (B·x).detach() base
          pattern: it isolates balance_integral_weight from task loss AND prevents
          double-counting the cumsum gradient path back to routing_weight.
          The same reasoning applies to (A'·u_load).detach() in the load-balance
          pathway — u_load already has no path to routing_weight (routing_logits is
          detached in load_balancing_logits), and the detach additionally blocks
          routing_integral_weight.

        Args:
            x: Input hidden states, shape (batch, seq_len, embedding_width).
            active_mask: Boolean active-token mask, shape (batch, seq_len). Dead tokens
                are excluded from the cumsum history in integral mode.

        Returns:
            Dict with keys:
            - ``routing_logits``:        A·x, shape (B, N, L).
            - ``balance_logits``:        B·x, shape (B, N, L).
            - ``semantic_logits``:       combined task-loss pathway, shape (B, N, L).
            - ``load_balancing_logits``: combined load-balance pathway, shape (B, N, L).
        """
        routing_logits = F.linear(x, self.routing_weight)                     # (B, N, L)
        balance_logits = F.linear(x, self.balance_weight)                     # (B, N, L)
        semantic_logits       = routing_logits + balance_logits.detach()
        load_balancing_logits = routing_logits.detach() + F.linear(x.detach(), self.balance_weight)

        if self.routing_mode == "integral":
            # Zero out dead token positions before cumsum so inactive tokens do not
            # contaminate the routing history of subsequent live tokens.
            live = active_mask.unsqueeze(-1)                                   # (B, N, 1)
            u_semantic = self.exclusive_cumsum(semantic_logits * live)         # (B, N, L)
            u_load     = self.exclusive_cumsum(load_balancing_logits * live)   # (B, N, L)

            # Semantic pathway: A' trains on task loss; B' term is fully detached to
            # isolate balance_integral_weight from task loss and prevent double-counting
            # the cumsum gradient path back to routing_weight.
            semantic_logits = (
                semantic_logits
                + F.linear(u_semantic, self.routing_integral_weight)
                + F.linear(u_semantic, self.balance_integral_weight).detach()
            )

            # Load-balance pathway: B' trains on load_balance_loss; A' term is fully
            # detached to isolate routing_integral_weight from load_balance_loss.
            load_balancing_logits = (
                load_balancing_logits
                + F.linear(u_load, self.routing_integral_weight).detach()
                + F.linear(u_load, self.balance_integral_weight)
            )

        return {
            "routing_logits":        routing_logits,
            "balance_logits":        balance_logits,
            "semantic_logits":       semantic_logits,
            "load_balancing_logits": load_balancing_logits,
        }

    @staticmethod
    def _compute_bias_diagnostics(
        routing_logits: torch.Tensor,
        balance_logits: torch.Tensor,
        semantic_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute detached diagnostic scalars characterising the two routing pathways.

        All scalars must be computed from pre-capacity logits; balance_capacity
        applies -1e8 sentinels that would corrupt std and cosine similarity.
        Extracted from forward to keep the main body free of reduction logic.

        Args:
            routing_logits:  A·x, routing pathway output, shape (B, N, L).
            balance_logits:  B·x, balance pathway output, shape (B, N, L).
            semantic_logits: A·x + (B·x).detach(), combined signal, shape (B, N, L).

        Returns:
            Dict with keys:
            - ``raw_logit_std``:  Mean per-token std of routing_logits. Natural
                                   routing preference scale; reference baseline for
                                   interpreting bias_std.
            - ``bias_std``:       Mean per-token std of balance_logits. Near-zero
                                   means balance corrections have not built up
                                   relative to the routing scale.
            - ``logit_std``:      Mean per-token std of semantic_logits. Lower than
                                   raw_logit_std indicates balance is flattening
                                   preferences (healthy correction signal).
            - ``bias_alignment``: Mean cosine similarity of routing_logits vs
                                   balance_logits per token. Range [-1, 1]. Negative
                                   means balance opposes routing direction (healthy
                                   correction); positive means runaway reinforcement.
        """
        return {
            "raw_logit_std":  routing_logits.std(dim=-1).mean().detach(),
            "bias_std":       balance_logits.std(dim=-1).mean().detach(),
            "logit_std":      semantic_logits.std(dim=-1).mean().detach(),
            "bias_alignment": F.cosine_similarity(
                routing_logits, balance_logits, dim=-1
            ).mean().detach(),
        }

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
