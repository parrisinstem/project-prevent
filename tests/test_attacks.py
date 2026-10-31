"""Attack tests.

The tests marked xfail are the spec for YOUR TASKS. When you implement an
attack, delete its `@todo` marker and make every test for it pass.
"""

import pytest
import torch

from pgremoval.attacks import build_attacks, gaussian_noise, identity, jpeg, resize
from pgremoval.data import quantise

todo = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="your task: not implemented yet")


def _valid(out, x):
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.min() >= -1 and out.max() <= 1


# ---- worked examples ------------------------------------------------------

def test_identity(images):
    out = identity(images)
    assert torch.equal(out, images) and out is not images


def test_jpeg_valid_and_changes_image(images):
    x = quantise(images)
    out = jpeg(x, quality=50)
    _valid(out, x)
    assert not torch.equal(out, x)


def test_jpeg_lower_quality_changes_more(images):
    x = quantise(images)
    d_hi = (jpeg(x, 95) - x).abs().mean()
    d_lo = (jpeg(x, 20) - x).abs().mean()
    assert d_lo > d_hi


def test_rejects_unbatched_input(images):
    with pytest.raises(ValueError):
        jpeg(images[0])


# ---- task 1: resize --------------------------------------------------------

@todo
def test_resize_valid(images):
    _valid(resize(images, 0.5), images)


@todo
def test_resize_scale_one_is_identity(images):
    assert torch.allclose(resize(images, 1.0), images, atol=1e-6)


@todo
def test_resize_smaller_scale_loses_more_detail(images):
    d_75 = (resize(images, 0.75) - images).abs().mean()
    d_25 = (resize(images, 0.25) - images).abs().mean()
    assert d_25 > d_75


# ---- task 2: gaussian noise -------------------------------------------------

@todo
def test_noise_valid(images):
    _valid(gaussian_noise(images, 0.05), images)


@todo
def test_noise_reproducible_with_seed(images):
    assert torch.equal(gaussian_noise(images, 0.05, seed=1), gaussian_noise(images, 0.05, seed=1))
    assert not torch.equal(gaussian_noise(images, 0.05, seed=1), gaussian_noise(images, 0.05, seed=2))


@todo
def test_noise_magnitude_roughly_sigma():
    x = torch.zeros(1, 3, 64, 64)  # far from the clamp boundary
    out = gaussian_noise(x, sigma=0.05)
    assert abs(out.std().item() - 0.05) < 0.01


# ---- registry ----------------------------------------------------------------

def test_registry_without_pipe_has_no_purification():
    attacks = build_attacks()
    assert "none" in attacks and "jpeg_q75" in attacks
    assert not any(k.startswith("purify") for k in attacks)
