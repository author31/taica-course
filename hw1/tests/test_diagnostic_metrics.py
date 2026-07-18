"""Focused regression tests for reconstruction-aware diagnostic evidence."""

import os
import sys

import numpy as np

HW1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HW1_DIR)

import utils  # noqa: E402


def test_gt_subset_maps_frame_stems_to_capture_order(tmp_path):
    """Image identifiers 1..N must not be used as zero-based GT row indices."""
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    for stem in (1, 2, 3):
        (tmp_path / "rgb" / f"{stem}.png").touch()
        (tmp_path / "depth" / f"{stem}.png").touch()
    gt = np.arange(21, dtype=float).reshape(3, 7)
    np.save(tmp_path / "GT_pose.npy", gt)

    selected = utils._load_gt(str(tmp_path), frames=[1, 3])

    np.testing.assert_array_equal(selected, gt[[0, 2]])


def test_gt_subset_preserves_nonconsecutive_requested_order(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    for stem in (10, 20, 40):
        (tmp_path / "rgb" / f"{stem}.png").touch()
        (tmp_path / "depth" / f"{stem}.png").touch()
    gt = np.arange(21, dtype=float).reshape(3, 7)
    np.save(tmp_path / "GT_pose.npy", gt)

    selected = utils._load_gt(str(tmp_path), frames=[40, 10])

    np.testing.assert_array_equal(selected, gt[[2, 0]])
