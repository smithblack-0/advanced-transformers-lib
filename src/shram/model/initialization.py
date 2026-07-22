"""Initialization policy for SHRAM raw projection parameters.

Hugging Face initializes ordinary ``nn.Linear`` and ``nn.Embedding`` modules.
SHRAM also stores two custom raw-parameter forms that Hugging Face cannot infer:

- a three-dimensional bank of independent expert linear maps
- a two-dimensional router projection

The expert-bank leading dimension is ownership/storage, not fan geometry. Each
expert matrix must therefore receive Xavier initialization independently. The
router follows the proven BALANCE initialization scale instead. Tensor rank is
used here as the explicit representation contract for these two owners, not as
a general initialization heuristic. Adding another raw projection shape must
therefore extend this contract deliberately rather than inheriting a fallback.
"""

import torch
import torch.nn as nn


ROUTER_INIT_STD = 0.02


@torch.no_grad()
def initialize_projection_parameter(parameter: nn.Parameter) -> None:
    """Initialize one of SHRAM's supported raw projection representations.

    A rank-three tensor is an independent bank of linear matrices and is
    initialized matrix-by-matrix with Xavier uniform. A rank-two tensor is the
    router projection and receives the BALANCE-normal initialization.
    """
    if parameter.ndim == 3:
        for matrix in parameter.unbind(dim=0):
            nn.init.xavier_uniform_(matrix)
        return

    if parameter.ndim == 2:
        nn.init.normal_(parameter, mean=0.0, std=ROUTER_INIT_STD)
        return

    raise ValueError(
        "SHRAM raw projection parameters must be rank two (router) or rank "
        f"three (independent expert bank), got shape {tuple(parameter.shape)}."
    )
