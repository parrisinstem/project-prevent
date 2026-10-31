"""The experiment for one image, independent of any particular model.

For an image x:
  1. Protect it:        x_p = quantise(PhotoGuard(x))
  2. Edit the clean one: E_clean = edit(x)
  3. For every removal attack a:
        x_ap = quantise(a(x_p))     attacked protected image
        x_ac = quantise(a(x))       attacked clean image (control)
        compare edit(x_ap) with edit(x)    -> did the attacker get the edit they wanted?
        compare edit(x_ap) with edit(x_ac) -> how much protection survives this attack,
                                              with the attack's own damage factored out
        compare x_ap with x                -> what the attack cost in image quality
        latent_rel_dist = ||enc(x_ap) - enc(x_ac)|| / ||enc(x_ac)||
                                           -> did the attack move the latent back?
                                              (cheap, needs no edit at all)

The encoder and editor are passed in, so this whole function runs in the CPU
tests with tiny stand-ins and on Colab with Stable Diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .attacks import Attack
from .data import Sample, quantise, to_pil
from .metrics import Metric
from .protect import Encoder, PGDConfig, encoder_attack, grey_target

# Takes a (3, H, W) image, returns (edited image (3, H, W) on CPU, safety-flagged?)
EditFn = Callable[[torch.Tensor], tuple[torch.Tensor, bool]]


@dataclass
class ImageResult:
    rows: list[dict] = field(default_factory=list)
    skipped_attacks: set[str] = field(default_factory=set)
    skipped_metrics: set[str] = field(default_factory=set)


def _save(img: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(img).save(path)


def evaluate_image(
    sample: Sample,
    encode: Encoder,
    edit_fn: EditFn,
    attacks: dict[str, Attack],
    metrics: dict[str, Metric],
    pgd_cfg: PGDConfig,
    device: str | torch.device = "cpu",
    save_dir: Path | None = None,
) -> ImageResult:
    result = ImageResult()
    x = sample.image[None].to(device=device, dtype=torch.float32)

    # 1. Protect
    with torch.no_grad():
        target_latent = encode(grey_target(x))
    pgd = encoder_attack(x, encode, target_latent, pgd_cfg)
    x_p = quantise(pgd.x_adv)

    # 2. Edit the clean image
    e_clean, flag_clean = edit_fn(x[0])
    clean_edits: dict[str, tuple[torch.Tensor, bool]] = {}

    if save_dir is not None:
        d = save_dir / sample.name
        _save(x[0], d / "clean.png")
        _save(x_p[0], d / "protected.png")
        _save(e_clean, d / "edit_clean.png")

    x_cpu = x.cpu()

    # 3. Attacks
    for name, attack in attacks.items():
        try:
            x_ap = quantise(attack(x_p))
            x_ac = quantise(attack(x))
        except NotImplementedError:
            result.skipped_attacks.add(name)
            continue

        e_ap, flag_ap = edit_fn(x_ap[0])
        if name == "none":
            e_ac, flag_ac = e_clean, flag_clean  # identical by construction
        else:
            e_ac, flag_ac = edit_fn(x_ac[0])
        clean_edits[name] = (e_ac, flag_ac)

        with torch.no_grad():
            z_ap, z_ac = encode(x_ap), encode(x_ac)
            latent_rel = ((z_ap - z_ac).norm() / z_ac.norm()).item()

        row = {
            "image": sample.name,
            "attack": name,
            "pgd_final_loss": pgd.losses[-1] if pgd.losses else float("nan"),
            "protection_linf": (x_p - x).abs().max().item(),
            "latent_rel_dist": latent_rel,
            "safety_flagged": flag_clean or flag_ap or flag_ac,
        }

        e_ap_b, e_clean_b, e_ac_b = e_ap[None], e_clean[None], e_ac[None]
        for m_name, metric in metrics.items():
            try:
                row[f"edit_vs_clean_{m_name}"] = metric(e_ap_b, e_clean_b).item()
                row[f"edit_vs_attacked_clean_{m_name}"] = metric(e_ap_b, e_ac_b).item()
                row[f"image_cost_{m_name}"] = metric(x_ap.cpu(), x_cpu).item()
            except NotImplementedError:
                result.skipped_metrics.add(m_name)
                for prefix in ("edit_vs_clean_", "edit_vs_attacked_clean_", "image_cost_"):
                    row.pop(prefix + m_name, None)

        result.rows.append(row)

        if save_dir is not None:
            d = save_dir / sample.name
            _save(x_ap[0], d / f"{name}_input.png")
            _save(e_ap, d / f"{name}_edit.png")

    return result
