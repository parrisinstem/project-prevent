"""Shared fixtures: small images and tiny stand-ins for the SD encoder and editor,
so the whole pipeline can be tested on a laptop CPU in seconds."""

import pytest
import torch


@pytest.fixture
def images() -> torch.Tensor:
    """A batch of 2 smooth-ish random images, (2, 3, 32, 32) in [-1, 1]."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 8, 8, generator=g) * 2 - 1
    return torch.nn.functional.interpolate(x, size=32, mode="bilinear", align_corners=False)


class TinyEncoder(torch.nn.Module):
    """Stand-in for the SD VAE encoder: (B, 3, H, W) -> (B, 4, H/8, W/8)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=8, stride=8)
        self.requires_grad_(False)

    def forward(self, x):
        return torch.tanh(self.conv(x))


@pytest.fixture
def encoder():
    return TinyEncoder()


@pytest.fixture
def edit_fn():
    """Stand-in editor: a fixed blur plus brightness shift, never safety-flagged."""
    kernel = torch.ones(3, 1, 3, 3) / 9.0

    def _edit(x: torch.Tensor):
        y = torch.nn.functional.conv2d(x[None].cpu(), kernel, padding=1, groups=3)[0]
        return (y * 0.8 + 0.1).clamp(-1, 1), False

    return _edit
