import numpy as np
import pytest
import torch
from PIL import Image

from pgremoval.data import from_pil, load_dataset, quantise, to_pil

HEADER = "filename,source_url,licence,contains_people\n"


def _make_dataset(tmp_path, rows, files):
    (tmp_path / "images").mkdir()
    for name in files:
        Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(tmp_path / "images" / name)
    (tmp_path / "SOURCES.csv").write_text(HEADER + "".join(rows))
    return tmp_path


def test_loads_listed_images(tmp_path):
    d = _make_dataset(tmp_path, ["a.png,https://x,CC0,no\n"], ["a.png"])
    samples = load_dataset(d, size=16)
    assert len(samples) == 1
    assert samples[0].name == "a"
    assert samples[0].image.shape == (3, 16, 16)
    assert samples[0].image.min() >= -1 and samples[0].image.max() <= 1


def test_missing_sources_file(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_unlisted_image_rejected(tmp_path):
    d = _make_dataset(tmp_path, ["a.png,https://x,CC0,no\n"], ["a.png", "b.png"])
    with pytest.raises(ValueError, match="b.png"):
        load_dataset(d)


def test_invalid_people_value_rejected(tmp_path):
    d = _make_dataset(tmp_path, ["a.png,https://x,CC0,unsure\n"], ["a.png"])
    with pytest.raises(ValueError, match="contains_people"):
        load_dataset(d)


def test_empty_licence_rejected(tmp_path):
    d = _make_dataset(tmp_path, ["a.png,https://x,,no\n"], ["a.png"])
    with pytest.raises(ValueError, match="licence"):
        load_dataset(d)


def test_pil_round_trip_is_exact_after_quantise(images):
    q = quantise(images[0])
    assert torch.allclose(from_pil(to_pil(q)), q, atol=1e-6)


def test_quantise_is_idempotent(images):
    q = quantise(images)
    assert torch.equal(quantise(q), q)
