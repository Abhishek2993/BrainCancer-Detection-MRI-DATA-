import numpy as np

from bcdc.gradcam import (border_energy, centre_prior, energy_in_mask, iou_area_matched,
                          localisation_metrics, pointing_game)


def square_mask(n=64, y0=10, x0=40, s=8):
    m = np.zeros((n, n), bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_perfect_map():
    mask = square_mask()
    cam = mask.astype(float)
    m = localisation_metrics(cam, mask)
    assert m["pointing_game"] == 1 and m["energy_in_mask"] == 1 and m["iou_area_matched"] == 1


def test_pointing_game_tolerance():
    mask = square_mask()
    cam = np.zeros(mask.shape)
    cam[10, 36] = 1                                 # 4 px left of the tumour
    assert pointing_game(cam, mask, tolerance_frac=5 / 64)
    assert not pointing_game(cam, mask, tolerance_frac=2 / 64)


def test_centre_prior_misses_corner_tumour():
    mask = square_mask(y0=2, x0=2)
    prior = centre_prior(mask.shape)
    assert not pointing_game(prior, mask)
    assert energy_in_mask(prior, mask) < mask.mean() * 1.5
    assert iou_area_matched(prior, mask) == 0


def test_border_energy():
    cam = np.zeros((100, 100))
    cam[0, :] = 1
    assert border_energy(cam) == 1
    assert border_energy(centre_prior((100, 100), 0.05)) < 0.01
