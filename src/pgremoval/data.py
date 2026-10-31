"""Image loading, conversion and a provenance check.

Every image in `<data_dir>/images/` must be listed in `<data_dir>/SOURCES.csv`
with where it came from and its licence. The loader refuses to run otherwise.
This keeps the dataset auditable, which matters for a project in this area.

SOURCES.csv columns:
    filename         exact file name inside images/
    source_url       where you got it
    licence          e.g. CC0, CC BY 4.0, Unsplash Licence
    contains_people  "no" or "adults"
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
REQUIRED_COLUMNS = {"filename", "source_url", "licence", "contains_people"}
ALLOWED_PEOPLE_VALUES = {"no", "adults"}


@dataclass
class Sample:
    name: str
    image: torch.Tensor  # (3, H, W) in [-1, 1]


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def from_pil(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> (3, H, W) float tensor in [-1, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_pil(x: torch.Tensor) -> Image.Image:
    """(3, H, W) tensor in [-1, 1] -> PIL RGB image (rounds to 8 bits)."""
    if x.dim() != 3:
        raise ValueError(f"expected (3, H, W), got {tuple(x.shape)}")
    arr = ((x.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round()
    return Image.fromarray(arr.permute(1, 2, 0).numpy().astype(np.uint8))


def quantise(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest 8-bit value, as happens when an image is saved.

    Uploaded photos are 8-bit files, so the protected image is quantised before
    anything else touches it. Works on (3, H, W) or (B, 3, H, W).
    """
    return (((x.clamp(-1, 1) + 1.0) * 127.5).round() / 127.5 - 1.0).to(x.dtype)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_sources(data_dir: Path) -> dict[str, dict[str, str]]:
    path = Path(data_dir) / "SOURCES.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. List every image with its source and licence before running."
        )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"SOURCES.csv is missing columns: {sorted(missing)}")
        rows = {row["filename"].strip(): row for row in reader}

    for name, row in rows.items():
        value = row["contains_people"].strip().lower()
        if value not in ALLOWED_PEOPLE_VALUES:
            raise ValueError(
                f"{name}: contains_people must be one of {sorted(ALLOWED_PEOPLE_VALUES)}, got {value!r}"
            )
        if not row["licence"].strip():
            raise ValueError(f"{name}: licence is empty")
    return rows


def load_image(path: Path, size: int = 512) -> torch.Tensor:
    """Centre-crop to a square, resize to `size`, return (3, size, size) in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
    return from_pil(img)


def load_dataset(data_dir: str | Path, size: int = 512, limit: int | None = None) -> list[Sample]:
    data_dir = Path(data_dir)
    sources = read_sources(data_dir)
    image_dir = data_dir / "images"
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found in {image_dir}")

    unlisted = [p.name for p in files if p.name not in sources]
    if unlisted:
        raise ValueError(f"images not listed in SOURCES.csv: {unlisted}")

    if limit is not None:
        files = files[:limit]
    return [Sample(name=p.stem, image=load_image(p, size)) for p in files]
