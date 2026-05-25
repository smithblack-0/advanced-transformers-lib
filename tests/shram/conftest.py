"""Shared pytest fixtures for the SHRAM test suite."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate device for tests that require flex attention.

    Flex attention requires CUDA in torch 2.12+. This fixture resolves the
    correct device at the single point of authority for the entire test suite:
    - CUDA if available (preferred; required for flex attention in torch 2.12+)
    - CPU if this is a CPU-only torch build (supports flex attention)
    - Skip otherwise (no suitable device present)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if "+cpu" in torch.__version__:
        return torch.device("cpu")
    pytest.skip("No suitable device for flex attention tests (need CUDA or CPU-only torch build)")
