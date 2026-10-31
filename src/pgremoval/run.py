"""Command-line entry point for the full experiment (GPU).

    python -m pgremoval.run --data data --out results --limit 10

Results are appended to <out>/metrics.csv after every image, so if Colab
disconnects you can rerun the same command and it carries on where it stopped.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from . import __version__
from .attacks import build_attacks
from .data import load_dataset
from .edit import MODEL_ID, EditConfig, edit, load_models
from .experiment import evaluate_image
from .metrics import METRICS
from .protect import PGDConfig, sd_encoder


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="folder with images/ and SOURCES.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--limit", type=int, default=None, help="only the first N images")
    p.add_argument("--attacks", default=None, help="comma-separated subset of attack names")
    p.add_argument("--save-images", type=int, default=3, help="save image grids for the first N images")
    p.add_argument("--eps", type=float, default=PGDConfig.eps)
    p.add_argument("--step-size", type=float, default=PGDConfig.step_size)
    p.add_argument("--iters", type=int, default=PGDConfig.iters)
    p.add_argument("--prompt", default=EditConfig.prompt)
    p.add_argument("--strength", type=float, default=EditConfig.strength)
    p.add_argument("--steps", type=int, default=EditConfig.num_inference_steps)
    p.add_argument("--seed", type=int, default=EditConfig.seed)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No GPU found. On Colab: Runtime > Change runtime type > T4 GPU.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "metrics.csv"

    pgd_cfg = PGDConfig(eps=args.eps, step_size=args.step_size, iters=args.iters)
    edit_cfg = EditConfig(prompt=args.prompt, strength=args.strength,
                          num_inference_steps=args.steps, seed=args.seed)

    (out / "config.json").write_text(json.dumps({
        "pgremoval_version": __version__,
        "model_id": args.model_id,
        "pgd": asdict(pgd_cfg),
        "edit": asdict(edit_cfg),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))

    samples = load_dataset(args.data, limit=args.limit)
    done: set[str] = set()
    if csv_path.exists():
        done = set(pd.read_csv(csv_path)["image"].astype(str))
        print(f"Resuming: {len(done)} images already in {csv_path}")

    print(f"Loading {args.model_id} ...")
    pipe, vae = load_models(args.device, args.model_id)
    encode = sd_encoder(vae)

    def edit_fn(x):
        r = edit(pipe, x, edit_cfg)
        return r.image, r.flagged

    attacks = build_attacks(pipe)
    if args.attacks:
        wanted = [a.strip() for a in args.attacks.split(",")]
        unknown = set(wanted) - set(attacks)
        if unknown:
            raise SystemExit(f"Unknown attacks: {sorted(unknown)}. Options: {sorted(attacks)}")
        attacks = {k: attacks[k] for k in wanted}

    skipped_a: set[str] = set()
    skipped_m: set[str] = set()
    for i, sample in enumerate(samples):
        if sample.name in done:
            continue
        t0 = time.time()
        save_dir = out / "images" if i < args.save_images else None
        res = evaluate_image(sample, encode, edit_fn, attacks, METRICS, pgd_cfg, args.device, save_dir)
        pd.DataFrame(res.rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        skipped_a |= res.skipped_attacks
        skipped_m |= res.skipped_metrics
        print(f"[{i + 1}/{len(samples)}] {sample.name}: {time.time() - t0:.0f}s")

    if skipped_a:
        print(f"Skipped (not implemented yet): attacks {sorted(skipped_a)}")
    if skipped_m:
        print(f"Skipped (not implemented yet): metrics {sorted(skipped_m)}")
    print(f"Done. Results in {csv_path}. Summarise with: python -m pgremoval.summarise {csv_path}")


if __name__ == "__main__":
    main()
