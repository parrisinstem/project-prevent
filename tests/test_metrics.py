"""Metric tests. xfail-marked tests are the spec for task 3."""

import math

import pytest
import torch

from pgremoval.metrics import lpips_distance, psnr, ssim

todo = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="your task: not implemented yet")


# ---- worked example: PSNR -----------------------------------------------------

def test_psnr_identical_is_inf(images):
    assert torch.isinf(psnr(images, images)).all()


def test_psnr_known_value():
    a = torch.zeros(1, 3, 4, 4)
    b = torch.full((1, 3, 4, 4), 0.2)  # mse = 0.04 -> 10*log10(4/0.04) = 20 dB
    assert math.isclose(psnr(a, b).item(), 20.0, rel_tol=1e-5)


def test_psnr_per_image_shape(images):
    assert psnr(images, images.flip(0)).shape == (2,)


# ---- task 3a: SSIM ------------------------------------------------------------

@todo
def test_ssim_identical_is_one(images):
    assert torch.allclose(ssim(images, images), torch.ones(2), atol=1e-4)


@todo
def test_ssim_drops_with_noise(images):
    noisy = (images + 0.3 * torch.randn_like(images)).clamp(-1, 1)
    assert (ssim(images, noisy) < 0.9).all()


# ---- task 3b: LPIPS -----------------------------------------------------------

@todo
def test_lpips_identical_is_zero(images):
    assert torch.allclose(lpips_distance(images, images), torch.zeros(2), atol=1e-4)


@todo
def test_lpips_grows_with_noise(images):
    small = (images + 0.05 * torch.randn_like(images)).clamp(-1, 1)
    large = (images + 0.4 * torch.randn_like(images)).clamp(-1, 1)
    assert (lpips_distance(images, large) > lpips_distance(images, small)).all()
