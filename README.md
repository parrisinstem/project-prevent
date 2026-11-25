# photoguard-removal

**Question:** PhotoGuard adds an invisible perturbation to a photo so that diffusion-model edits of it come out broken. How much of that protection survives cheap transforms an adversary can apply first: JPEG compression, resizing, added noise and diffusion purification?

This is a small empirical follow-up to a proposal I designed for Thorn (Analyst Impact Fund 2026): protecting photos at upload against manipulation by generative models. A prevention-only defence is only as good as its robustness to removal, so this project measures that directly.

## Background

- **PhotoGuard** (Salman et al., ICML 2023) runs projected gradient descent so that Stable Diffusion's VAE encoder maps the protected image onto the latent of a plain grey image. Edits then start from the wrong latent.
- **Hönig, Rando, Carlini & Tramèr** (2024) showed that low-effort methods such as noisy upscaling defeat several perturbation-based protections, Glaze included. This project reproduces the spirit of that result on a small scale for PhotoGuard's encoder attack.

## What is measured

For each image `x`, protected image `x_p` and removal attack `a`:

| Column | Compares | Reads as |
|---|---|---|
| `latent_rel_dist` | `enc(a(x_p))` vs `enc(a(x))` | Did the attack pull the latent back? 0 = fully removed. Needs no edit. |
| `edit_vs_attacked_clean_*` | `edit(a(x_p))` vs `edit(a(x))` | **Main result.** How much protection survives, with the attack's own damage factored out. |
| `edit_vs_clean_*` | `edit(a(x_p))` vs `edit(x)` | Did the attacker get the edit they wanted? |
| `image_cost_*` | `a(x_p)` vs `x` | What the attack cost in image quality. |

`*` is one of `psnr`, `ssim` and `lpips`. Higher PSNR and SSIM mean more similar; for LPIPS, lower means more similar. The `none` attack is the baseline: protection with nothing done to it.

The interesting output is the **trade-off curve**: for each attack family, how much protection is removed against how much the image is damaged.

## Data and ethics rules

This project is about protecting photos of children. **It never uses photos of children, and it does not need to**: whether an edit is disrupted can be measured on any image.

- Every image in `data/images/` must be listed in `data/SOURCES.csv` with its source, licence and `contains_people` (`no` or `adults`). The loader refuses to run otherwise.
- Start with images containing **no people at all** (landscapes, objects, animals, buildings) from CC0 or Unsplash-licensed sources. Only add images of adults from datasets released for research with consent.
- Images are not committed to git (see `.gitignore`); `SOURCES.csv` is, so anyone can rebuild the dataset.
- Edit prompts are benign ("a watercolour painting"). The model's safety checker stays on, and any flagged output is recorded in `safety_flagged`.
- Efficacy is measured by disruption, never by producing harmful content.

## Quickstart

**On your laptop (CPU, no GPU needed).** The tests use tiny stand-in models, so the whole pipeline is checked in seconds:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

At the start you'll see passes plus `xfailed` tests. The xfailed tests are your tasks.

**On Google Colab (free T4 GPU).** Open `notebooks/colab.ipynb` in Colab, set *Runtime → Change runtime type → T4 GPU*, and run the cells. Results are written to Google Drive, and the run resumes if Colab disconnects.

```bash
python -m pgremoval.run --data data --out results --limit 5 --iters 50   # quick smoke run
python -m pgremoval.run --data data --out results                        # full run
python -m pgremoval.summarise results/metrics.csv
```

Time the smoke run first (it prints seconds per image), then pick `--limit` and `--iters` to fit a Colab session.

## Layout

```
src/pgremoval/
  data.py        loading, provenance check, 8-bit quantisation
  protect.py     PhotoGuard encoder attack (PGD)            ✅ done
  attacks.py     removal attacks: JPEG ✅, resize / noise / purification  ✏️ yours
  metrics.py     PSNR ✅, SSIM / LPIPS                       ✏️ yours
  edit.py        Stable Diffusion img2img + fp32 VAE
  experiment.py  the per-image experiment (model-agnostic, so it's testable)
  run.py         CLI for the full run
  summarise.py   per-attack mean ± std table
tests/           CPU tests; xfail tests are the spec for your tasks
notebooks/       Colab notebook
docs/WRITEUP.md  write-up template
```

## Your tasks

Each task has a spec in its docstring and tests marked `@todo`. To finish a task, implement it, delete the `@todo` markers and get `pytest` green. The markers are strict, so a test that starts passing while still marked will fail and remind you.

1. **`resize` attack.** Downscale and upscale with `F.interpolate`.
2. **`gaussian_noise` attack.** Seeded noise and clamping.
3. **Metrics.** (a) SSIM, ideally written yourself in torch; (b) LPIPS via the `lpips` package.
4. **`diffusion_purify` attack (GPU).** The strongest attack, implemented with the SD pipeline. There's no CPU test, so verify it on Colab by eye, then with numbers.
5. **Run it and plot it.** Run on 20–50 images, then plot protection remaining (`edit_vs_attacked_clean_lpips`) against image cost (`image_cost_psnr`), one line per attack family.
6. **Write it up** in `docs/WRITEUP.md`: question, setup, results, limitations. Report only numbers you measured.

Stretch, once the basics work: vary PhotoGuard's `--eps` and see whether a bigger budget survives purification; test an *adaptive* defender that runs PGD through a JPEG approximation; repeat with a second edit prompt.

## Publishing to GitHub

```bash
git init && git add . && git commit -m "Scaffold: PhotoGuard protection, JPEG removal attack, CPU tests"
gh repo create photoguard-removal --public --source . --push   # or create it on github.com and push
```

The included GitHub Actions workflow runs the CPU tests on every push.

## References

- H. Salman, A. Khaddaj, G. Leclerc, A. Ilyas, A. Madry. *Raising the Cost of Malicious AI-Powered Image Editing.* ICML 2023. [arXiv:2302.06588](https://arxiv.org/abs/2302.06588) · [code](https://github.com/madrylab/photoguard)
- R. Hönig, J. Rando, N. Carlini, F. Tramèr. *Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI.* 2024. [arXiv:2406.12027](https://arxiv.org/abs/2406.12027)
- Model: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (CreativeML OpenRAIL-M licence)
