"""Shared initialization policy for SHRAM raw projection parameters.

Ordinary ``nn.Linear`` and ``nn.Embedding`` modules are initialized by the
Hugging Face ``PreTrainedModel`` pass. Some SHRAM components instead store
banks of independent linear maps directly as higher-rank ``nn.Parameter``
tensors. Fan-based initializers interpret those storage dimensions as
convolutional geometry, so storage rank must not determine their scale.
"""

import torch
import torch.nn as nn


PROJECTION_INIT_STD = 0.02


@torch.no_grad()
def initialize_projection_parameter(parameter: nn.Parameter) -> None:
    """Initialize a projection-like raw parameter independently of storage rank."""
    nn.init.normal_(
        parameter,
        mean=0.0,
        std=PROJECTION_INIT_STD,
    )
