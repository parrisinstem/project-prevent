"""PhotoGuard encoder attack (Salman et al., ICML 2023, arXiv:2302.06588).

Idea: Stable Diffusion edits an image by first encoding it into a latent with
its VAE encoder E. PhotoGuard finds a small perturbation delta, with
||delta||_inf <= eps, so that E(x + delta) lands close to the latent of a
target image (plain grey here). Any edit then starts from the wrong latent and
comes out degraded, while x + delta still looks like x to a person.

This is projected gradient descent (PGD):
    repeat:
        loss  = || E(x_adv) - z_target ||_2
        x_adv = x_adv - step * sign(grad loss)       # move toward the target latent
        x_adv = x + clip(x_adv - x, -eps, eps)       # stay inside the eps-ball
        x_adv = clip(x_adv, -1, 1)                   # stay a valid image
The step size decays linearly, as in the PhotoGuard reference code.

`encode` is passed in as a function, so the attack can be unit-tested on CPU
with a tiny stand-in encoder and run for real with the SD VAE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

Encoder = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class PGDConfig:
    eps: float = 0.06          # L-inf budget in [-1, 1] units (about 8/255)
    step_size: float = 0.01    # starting step; decays to step_size / 100
    iters: int = 200
    seed: int = 0


@dataclass
class PGDResult:
    x_adv: torch.Tensor
    losses: list[float] = field(default_factory=list)  # one per iteration, for convergence checks


def grey_target(x: torch.Tensor) -> torch.Tensor:
    """A mid-grey image with the same shape as x (0.0 is mid-grey in [-1, 1])."""
    return torch.zeros_like(x)


def encoder_attack(
    x: torch.Tensor,
    encode: Encoder,
    target_latent: torch.Tensor,
    cfg: PGDConfig = PGDConfig(),
) -> PGDResult:
    """Run the PhotoGuard encoder attack on a batch x of shape (B, 3, H, W) in [-1, 1]."""
    if x.dim() != 4:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")

    x = x.detach()
    gen = torch.Generator(device=x.device).manual_seed(cfg.seed)
    noise = torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    x_adv = (x + (noise * 2 - 1) * cfg.eps).clamp(-1, 1)

    losses: list[float] = []
    for i in range(cfg.iters):
        step = cfg.step_size - (cfg.step_size - cfg.step_size / 100) / cfg.iters * i
        x_adv.requires_grad_(True)
        loss = (encode(x_adv) - target_latent).norm(p=2)
        (grad,) = torch.autograd.grad(loss, x_adv)
        losses.append(loss.item())
        with torch.no_grad():
            x_adv = x_adv - step * grad.sign()
            x_adv = x + (x_adv - x).clamp(-cfg.eps, cfg.eps)
            x_adv = x_adv.clamp(-1, 1)
        x_adv = x_adv.detach()

    return PGDResult(x_adv=x_adv, losses=losses)


def sd_encoder(vae) -> Encoder:
    """Wrap a diffusers AutoencoderKL as an encoder function (deterministic: uses the mean)."""
    return lambda x: vae.encode(x).latent_dist.mean
