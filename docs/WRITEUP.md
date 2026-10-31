# Does PhotoGuard survive simple removal attacks?

*Draft. Fill each section from your own runs; report only numbers you measured.*

## Question

<!-- One paragraph: the threat (manipulation of uploaded photos), the defence
(PhotoGuard encoder attack), and the question (how much protection survives
cheap transforms?). Mention the Thorn proposal in one line. -->

## Setup

- Model: stable-diffusion-v1-5, img2img, prompt "...", strength ..., steps ..., seed ...
- Protection: PhotoGuard encoder attack, eps = ..., step = ..., iters = ..., grey target
- Images: N images, no people / adults only, sources in `data/SOURCES.csv`
- Attacks: none, JPEG (q90, q75, q50), resize (0.75, 0.5), noise (0.03, 0.06), purification (0.1, 0.2, 0.3)
- Hardware: Colab T4 (`results/config.json` records exact versions)

## Results

<!-- Paste the table from `python -m pgremoval.summarise`. -->

<!-- Figure: protection remaining vs image cost, one line per attack family. -->

<!-- Figure: one example image: clean, protected, edit of each, and edits after
the two or three most interesting attacks. -->

## What I found

<!-- 3-5 bullets. Which attacks removed the protection, at what image cost?
Does latent_rel_dist predict edit similarity? Anything surprising? -->

## Limitations

<!-- Be specific. For example: small N; one model, one prompt; images without
people while the real threat model is photos of people; the purifier shares the
editor's model family; a non-adaptive defender; metrics as proxies for
"the edit worked". -->

## What this means for a prevention-only defence

<!-- Link back to the Thorn proposal: what this suggests about relying on
upload-time perturbation alone versus pairing it with detection. -->

## Next steps

<!-- E.g. adaptive defence, larger eps, other editors, a perceptual study. -->
