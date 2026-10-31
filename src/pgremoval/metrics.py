"""Image similarity metrics.

Every metric takes two (B, 3, H, W) tensors in [-1, 1] and returns a (B,)
tensor, one value per image.

`psnr` is implemented as a worked example. `ssim` and `lpips_distance` are
YOUR TASKS (task 3), with tests in tests/test_metrics.py marked xfail.

How the experiment uses them:
  * edit similarity: compare the edit of an attacked, protected image with the
    edit of the clean image. High similarity means the removal attack worked.
  * image cost: compare the attacked image with the clean image. Shows how much
    the attack itself damaged the photo.
"""

from __future__ import annotations

from typing import Callable

import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _check(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape or a.dim() != 4:
        raise ValueError(f"expected two equal (B, 3, H, W) tensors, got {tuple(a.shape)} and {tuple(b.shape)}")


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB. Higher = more similar; identical images give inf.

    Data range is 2.0 because pixels are in [-1, 1].
    """
    _check(a, b)
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10(4.0 / mse)  # 4.0 = data_range ** 2


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """TODO (task 3a). Structural similarity, per image. 1.0 = identical.

    Spec:
      * Either use skimage.metrics.structural_similarity (channel_axis, data_range=2.0)
        on each image converted to numpy (H, W, 3), or implement the Gaussian-window
        SSIM yourself in torch. The second teaches you more.
      * Return a (B,) float tensor on CPU.
    """
    raise NotImplementedError


def lpips_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """TODO (task 3b). LPIPS perceptual distance, per image. 0.0 = identical; lower = more similar.

    Spec:
      * Use the `lpips` package with net="alex". It expects inputs in [-1, 1],
        which is already our convention.
      * Create the LPIPS model once and cache it (module-level or functools.cache),
        not on every call.
      * Return a (B,) float tensor on CPU.
    Note the direction: unlike PSNR and SSIM, lower is more similar. Keep that
    clear in your write-up and plots.
    """
    raise NotImplementedError


METRICS: dict[str, Metric] = {
    "psnr": psnr,
    "ssim": ssim,
    "lpips": lpips_distance,
}
