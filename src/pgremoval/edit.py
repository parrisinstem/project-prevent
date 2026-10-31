"""Stable Diffusion models used by the experiment (GPU only).

Two models are loaded from the same checkpoint:
  * the img2img pipeline in fp16, which does the edits (and diffusion purification)
  * a separate fp32 copy of the VAE, which the PhotoGuard attack differentiates
    through (fp16 gradients are too noisy for PGD)

The pipeline's built-in safety checker is left ON. Edit prompts are benign;
if the checker flags an output, the run records it rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import from_pil, to_pil

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@dataclass
class EditConfig:
    prompt: str = "a watercolour painting"
    strength: float = 0.5
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: int = 1234


def load_models(device: str = "cuda", model_id: str = MODEL_ID):
    """Return (img2img_pipe, vae_fp32). Downloads about 2.5 GB the first time."""
    from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device).eval().requires_grad_(False)
    return pipe, vae


@dataclass
class EditOutput:
    image: torch.Tensor   # (3, H, W) in [-1, 1]
    flagged: bool         # safety checker flagged the output


def edit(pipe, x: torch.Tensor, cfg: EditConfig = EditConfig()) -> EditOutput:
    """Edit one (3, H, W) image with a fixed prompt and seed."""
    generator = torch.Generator(device=pipe.device).manual_seed(cfg.seed)
    out = pipe(
        prompt=cfg.prompt,
        image=to_pil(x),
        strength=cfg.strength,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
    )
    flags = getattr(out, "nsfw_content_detected", None) or [False]
    return EditOutput(image=from_pil(out.images[0]), flagged=bool(flags[0]))
