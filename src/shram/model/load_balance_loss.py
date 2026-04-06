"""Auxiliary-loss-free load balancing operator for MoSRAH routing.

This module implements the custom autograd Function H(b, f) described in the paper's
Implementation Concerns section. The operator bridges two requirements that are in
tension: it must behave like a standard auxiliary loss (scalar output, scalable via
multiplication) so that existing training loops remain compatible, while simultaneously
implementing DeepSeek-style bias correction rather than the usual auxiliary-loss gradient
path through the router weights.

The resolution is a custom backward pass. The forward emits the load balance imbalance
as a scalar loss. The backward, instead of differentiating that scalar with respect to
its inputs, writes a bias-correction gradient directly to expert_bias. This gradient is
then consumed by the main AdamW optimizer in the normal way, achieving DeepSeek-style
correction without a standalone SGD update step.

Paper ref: Appendix A.Implementation Concerns.
"""

import torch


class LoadBalanceLoss(torch.autograd.Function):
    """Custom autograd operator for DeepSeek-style auxiliary-loss-free load balancing.

    Forward computes the load balance imbalance:

        L_load_balance = H(b, f) = sum_l | f_l - 1/L |

    Backward emits a bias-correction gradient to expert_bias:

        grad_b = L_grad * sign(f_l - 1/L)

    expert_bias (b) is included as a forward input so PyTorch registers it as a node
    in the computation graph and routes gradients through it. routing_freqs (f) receives
    no gradient — its origin is the discrete TopK operation which has no gradient, so
    defining a gradient for f here would be mathematically incorrect.

    Paper ref: Appendix A.Implementation Concerns.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        expert_bias: torch.Tensor,
        routing_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the load balance loss.

        Args:
            ctx: Autograd context for saving state needed in backward.
            expert_bias: Learned per-head bias b, shape (L,). Included as an input so
                PyTorch tracks it as a computation graph node needing a gradient.
            routing_freqs: Realized routing frequency f_l per head, shape (L,). Computed
                from the discrete TopK selection — not differentiable.

        Returns:
            Scalar loss equal to sum_l |f_l - 1/L|.
        """
        L = expert_bias.shape[0]
        # imbalance = f_l - 1/L for each head: positive means overloaded, negative means
        # underloaded. Saved for backward where sign(imbalance) determines the direction
        # of the bias-correction update.
        imbalance = routing_freqs - 1.0 / L
        ctx.save_for_backward(imbalance)
        return imbalance.abs().sum()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Emit the DeepSeek-style bias-correction gradient.

        Args:
            ctx: Autograd context carrying imbalance saved in forward.
            grad_output: Incoming gradient L_grad (scalar). Any rescaling of the loss
                by the training loop arrives here and is propagated to grad_b, so the
                correction magnitude is proportional to the loss weight chosen by the
                consumer.

        Returns:
            Gradient for expert_bias: L_grad * sign(f_l - 1/L), shape (L,).
            None for routing_freqs: no gradient is defined for the discrete routing
            frequency.
        """
        (imbalance,) = ctx.saved_tensors
        grad_expert_bias = grad_output * imbalance.sign()
        return grad_expert_bias, None
