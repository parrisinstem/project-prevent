"""End-to-end smoke test of one image through the pipeline, with stand-in models."""

from pgremoval.attacks import build_attacks
from pgremoval.data import Sample
from pgremoval.experiment import evaluate_image
from pgremoval.metrics import METRICS
from pgremoval.protect import PGDConfig
from pgremoval.summarise import summarise, to_markdown

import pandas as pd


def test_pipeline_runs_and_skips_unimplemented(images, encoder, edit_fn, tmp_path):
    sample = Sample(name="img0", image=images[0])
    res = evaluate_image(
        sample, encoder, edit_fn, build_attacks(), METRICS,
        PGDConfig(iters=10), save_dir=tmp_path,
    )
    attacks_run = {r["attack"] for r in res.rows}
    assert {"none", "jpeg_q75"} <= attacks_run
    # Unimplemented pieces are skipped, not crashed on.
    assert attacks_run.isdisjoint(res.skipped_attacks)
    for row in res.rows:
        assert "edit_vs_attacked_clean_psnr" in row
        assert 0 <= row["latent_rel_dist"]
    assert (tmp_path / "img0" / "protected.png").exists()

    table = summarise(pd.DataFrame(res.rows))
    assert table.index[0] == "none"
    assert "| none |" in to_markdown(table)
