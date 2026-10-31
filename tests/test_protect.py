import torch

from pgremoval.protect import PGDConfig, encoder_attack, grey_target

CFG = PGDConfig(eps=0.06, step_size=0.01, iters=40)


def _run(images, encoder):
    with torch.no_grad():
        target = encoder(grey_target(images))
    return encoder_attack(images, encoder, target, CFG), target


def test_perturbation_stays_in_budget(images, encoder):
    res, _ = _run(images, encoder)
    assert (res.x_adv - images).abs().max() <= CFG.eps + 1e-6


def test_output_is_valid_image(images, encoder):
    res, _ = _run(images, encoder)
    assert res.x_adv.shape == images.shape
    assert res.x_adv.min() >= -1 and res.x_adv.max() <= 1


def test_latent_moves_toward_target(images, encoder):
    res, target = _run(images, encoder)
    with torch.no_grad():
        before = (encoder(images) - target).norm()
        after = (encoder(res.x_adv) - target).norm()
    assert after < before
    assert res.losses[-1] < res.losses[0]


def test_deterministic_given_seed(images, encoder):
    a, _ = _run(images, encoder)
    b, _ = _run(images, encoder)
    assert torch.equal(a.x_adv, b.x_adv)
