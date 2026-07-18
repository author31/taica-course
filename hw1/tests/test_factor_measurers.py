import sys
from pathlib import Path

import numpy as np
from PIL import Image


HW1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW1_DIR))

import api  # noqa: E402


def _depth(tmp_path, name, metres):
    path = tmp_path / name
    raw = np.rint(np.asarray(metres) * 1000.0).clip(0, 65535).astype(np.uint16)
    Image.fromarray(raw).save(path)
    return str(path)


def test_high_frequency_residual_uses_consumer_validity_and_exports_mask(tmp_path):
    rng = np.random.default_rng(7)
    clean = np.full((96, 96), 12.0)
    noisy = clean + rng.normal(0.0, 0.01, clean.shape)
    path = _depth(tmp_path, "noise.png", noisy)

    value = api.frame_high_frequency_depth_residual(path, 5.0)
    mask = api.frame_high_frequency_depth_residual_mask(path, 5.0)

    assert np.isfinite(value)
    assert 0.006 <= value <= 0.014
    assert mask.shape == clean.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_high_frequency_residual_fails_closed_without_valid_window(tmp_path):
    path = _depth(tmp_path, "dropout.png", np.zeros((4, 4)))
    assert np.isinf(api.frame_high_frequency_depth_residual(path))
    assert not api.frame_high_frequency_depth_residual_mask(path).any()


def test_flying_pixel_plane_discriminator(tmp_path):
    depth = np.ones((9, 9))
    depth[:, 5:] = 3.0
    clean = _depth(tmp_path, "clean_step.png", depth)
    assert api.frame_flying_pixel_ratio(clean, 5, 0.05) == 0.0

    depth[4, 4] = 2.0
    mixed = _depth(tmp_path, "mixed.png", depth)
    mask = api.frame_flying_pixel_ratio_mask(mixed, 5, 0.05)
    assert mask[4, 4] == 255
    assert api.frame_flying_pixel_ratio(mixed, 5, 0.05) > 0.0


def test_valid_tile_coverage_flags_whole_under_supported_tile(tmp_path):
    depth = np.ones((8, 8))
    depth[:4, :4] = 0.0
    path = _depth(tmp_path, "tiles.png", depth)
    value = api.frame_valid_tile_coverage(path, 4, 0.5)
    mask = api.frame_valid_tile_coverage_mask(path, 4, 0.5)
    assert value == 0.75
    assert np.all(mask[:4, :4] == 255)
    assert not mask[4:, 4:].any()


def test_identity_change_uses_joint_pixels_and_flags_outlier(tmp_path):
    d0 = np.full((6, 6), 2.0)
    d1 = np.full((6, 6), 2.1)
    d0[0, 0] = 0.0
    d1[1, 1] = 3.0
    p0 = _depth(tmp_path, "d0.png", d0)
    p1 = _depth(tmp_path, "d1.png", d1)
    value, mask, count = api._identity_median_depth_change(p0, p1, 3.0)
    assert np.isclose(value, 0.1)
    assert count == 35
    assert mask[1, 1] == 255
    assert mask[0, 0] == 0


def test_joint_valid_ratio_and_mask_are_exact(tmp_path):
    d0 = np.ones((4, 4))
    d1 = np.ones((4, 4))
    d0[0, :] = 0.0
    d1[:, 0] = 0.0
    p0 = _depth(tmp_path, "j0.png", d0)
    p1 = _depth(tmp_path, "j1.png", d1)
    value, mask, count = api._joint_valid_depth_ratio(p0, p1)
    assert count == 9
    assert value == 9 / 16
    assert np.count_nonzero(mask) == 7


def test_prior_warp_residual_identity_and_bias(tmp_path):
    d0 = np.full((8, 8), 2.0)
    p0 = _depth(tmp_path, "p0.png", d0)
    p1 = _depth(tmp_path, "p1.png", d0)
    intrinsics = {"width": 8, "height": 8, "hfov": 90.0}
    value, mask, count = api._prior_warp_depth_residual(
        p0, p1, np.eye(4), intrinsics, 0.1)
    assert value == 0.0
    assert count == 64
    assert not mask.any()

    p2 = _depth(tmp_path, "p2.png", d0 + 0.2)
    value, mask, count = api._prior_warp_depth_residual(
        p0, p2, np.eye(4), intrinsics, 0.1)
    assert np.isclose(value, 0.2)
    assert count == 64
    assert np.all(mask == 255)
