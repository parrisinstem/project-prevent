"""Removal attacks: cheap transforms an adversary applies to a protected image
before editing it, hoping to strip the perturbation.

Contract for every attack:
  * input  (B, 3, H, W) tensor in [-1, 1]
  * output (B, 3, H, W) tensor in [-1, 1], same device and dtype
  * deterministic given its arguments (seeded where randomness is involved)

`jpeg` is implemented as a worked example. The others are YOUR TASKS: each has a
spec in its docstring and tests in tests/test_attacks.py marked xfail. When you
implement one, delete its xfail marker and make the tests pass.
"""

from __future__ import annotations

import io
from functools import partial
from typing import Callable

import torch
from PIL import Image

from .data import from_pil, to_pil

Attack = Callable[[torch.Tensor], torch.Tensor]


def _check(x: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")


def identity(x: torch.Tensor) -> torch.Tensor:
    """No attack. The baseline: how much protection survives untouched."""
    _check(x)
    return x.clone()


def jpeg(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Re-encode each image as a JPEG at `quality` (1-100) and decode it again.

    JPEG throws away high-frequency detail, which is where much of an adversarial
    perturbation lives. Lower quality = stronger attack, but also more damage
    to the image itself.
    """
    _check(x)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be in 1..100")
    out = []
    for img in x:
        buf = io.BytesIO()
        to_pil(img).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(from_pil(Image.open(buf)))
    return torch.stack(out).to(device=x.device, dtype=x.dtype)


# --------------------------------------------------------------------------
# YOUR TASKS
# --------------------------------------------------------------------------

def resize(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """TODO (task 1). Downscale then upscale back to the original size.

    Spec:
      * scale is in (0, 1]; scale=1.0 must return the input unchanged (within 1e-6).
      * Downsample to (round(H*scale), round(W*scale)) with bicubic interpolation
        and antialias=True, then upsample back to (H, W) with bicubic.
      * Clamp the result to [-1, 1] (bicubic can overshoot).
    Hint: torch.nn.functional.interpolate.
    Think about: why does this remove adversarial noise, and what does it cost?
    """
    raise NotImplementedError


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05, seed: int = 0) -> torch.Tensor:
    """TODO (task 2). Add i.i.d. Gaussian noise with standard deviation `sigma`.

    Spec:
      * sigma is in [-1, 1] pixel units (0.05 is about 6/255).
      * Use a torch.Generator seeded with `seed` so results are reproducible;
        the same seed must give the same output.
      * Clamp the result to [-1, 1].
    Think about: noise alone rarely removes a perturbation. Why might it still
    help when the edit model then denoises the image?
    """
    raise NotImplementedError


def diffusion_purify(
    x: torch.Tensor,
    pipe,
    strength: float = 0.2,
    seed: int = 0,
) -> torch.Tensor:
    """TODO (task 4, GPU). Purify with a diffusion model: noise, then denoise.

    This is the strongest attack in Hönig et al. (2024, arXiv:2406.12027) and
    the one most worth getting right.

    Spec:
      * Run `pipe` (a StableDiffusionImg2ImgPipeline) on each image with
        prompt="" (empty), strength=`strength`, guidance_scale=1.0 and a
        torch.Generator(pipe.device) seeded with `seed`.
      * Low strength (0.1-0.3) adds a little noise and denoises it away,
        keeping the image content while washing out the perturbation.
      * Convert in and out with data.to_pil / data.from_pil; return the same
        shape, device and dtype as x.
    Think about: the attacker here uses the same model family as the editor.
    Is that realistic? What changes if they used a different model?
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Registry used by the experiment
# --------------------------------------------------------------------------

def build_attacks(pipe=None) -> dict[str, Attack]:
    """Named attack settings to evaluate. Add or change settings here."""
    attacks: dict[str, Attack] = {
        "none": identity,
        "jpeg_q90": partial(jpeg, quality=90),
        "jpeg_q75": partial(jpeg, quality=75),
        "jpeg_q50": partial(jpeg, quality=50),
        "resize_0.75": partial(resize, scale=0.75),
        "resize_0.5": partial(resize, scale=0.5),
        "noise_0.03": partial(gaussian_noise, sigma=0.03),
        "noise_0.06": partial(gaussian_noise, sigma=0.06),
    }
    if pipe is not None:
        attacks["purify_0.1"] = partial(diffusion_purify, pipe=pipe, strength=0.1)
        attacks["purify_0.2"] = partial(diffusion_purify, pipe=pipe, strength=0.2)
        attacks["purify_0.3"] = partial(diffusion_purify, pipe=pipe, strength=0.3)
    return attacks
