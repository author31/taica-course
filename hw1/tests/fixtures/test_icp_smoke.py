"""
Self-check for the HW1 geometry smoke fixture.

This does NOT test your code. It tests the fixture itself, using only the parts of
`utils.py` that ship working — `local_icp_algorithm` (the Open3D reference
baseline) and `mean_l2` (the score). If these pass, the numbers in
`smoke/expected.json` are trustworthy, and any disagreement you see when running
your own `depth_image_to_point_cloud` / `reconstruct` against the fixture is in
your code.

    pixi run -e habitat python -m pytest hw1/tests/fixtures/test_icp_smoke.py -v

Everything here reads from the committed fixture — the capture, `expected.json`
and `clouds.npz`. There is no generator in the student tree: the fixture is
consumed, not regenerated.

Deliberately absent: nothing here builds a point cloud from a depth image, and
nothing here chains per-step transforms into a trajectory. Those are the two
`#TODO`s, and this file must not do them for you.
"""

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_HW1 = os.path.dirname(os.path.dirname(_HERE))       # .../hw1

sys.path.insert(0, _HW1)

import utils                                         # noqa: E402  (hw1/utils.py)

cv2 = pytest.importorskip("cv2")
o3d = pytest.importorskip("open3d")

SMOKE = os.path.join(_HERE, "smoke")


@pytest.fixture(scope="module")
def oracle():
    with open(os.path.join(SMOKE, "expected.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def clouds(oracle):
    """The reference clouds shipped in clouds.npz, as Open3D point clouds.

    These are the fixture's own geometry, not the output of anything under test,
    which is what keeps this file independent of the `depth_image_to_point_cloud`
    TODO.
    """
    data = np.load(os.path.join(SMOKE, oracle["reference_clouds"]["file"]))
    out = []
    for i in range(oracle["n_frames"]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            data[f"frame_{i}"].astype(np.float64))
        out.append(pcd)
    return out


def test_fixture_is_present(oracle):
    for sub in ("rgb", "depth"):
        for i in range(oracle["n_frames"]):
            assert os.path.exists(os.path.join(SMOKE, sub, f"{i}.png"))
    for name in ("GT_pose.npy", "intrinsics.json", "expected.json", "clouds.npz"):
        assert os.path.exists(os.path.join(SMOKE, name))


def test_capture_layout_matches_the_real_dataset(oracle):
    """The fixture must be a drop-in --data_root, intrinsics.json included."""
    with open(os.path.join(SMOKE, "intrinsics.json")) as fh:
        intr = json.load(fh)
    assert set(intr) == {"width", "height", "hfov"}
    assert intr == oracle["intrinsics"]

    gt = np.load(os.path.join(SMOKE, "GT_pose.npy"))
    assert gt.shape == (oracle["n_frames"], 7)
    np.testing.assert_allclose(np.linalg.norm(gt[:, 3:], axis=1), 1.0, atol=1e-12)


def test_depth_is_uint16_millimetres_and_fully_valid(oracle):
    """No dropout in the fixture — the scene is closed, so every ray hits
    something. Depth validity is not what this fixture is testing."""
    w, h = oracle["intrinsics"]["width"], oracle["intrinsics"]["height"]
    for i in range(oracle["n_frames"]):
        depth = cv2.imread(os.path.join(SMOKE, "depth", f"{i}.png"),
                           cv2.IMREAD_UNCHANGED)
        assert depth.dtype == np.uint16, "depth must ship as uint16 millimetres"
        assert depth.shape == (h, w)
        assert depth.min() > 0


def test_rgb_frames_load_and_match_the_intrinsics(oracle):
    w, h = oracle["intrinsics"]["width"], oracle["intrinsics"]["height"]
    for i in range(oracle["n_frames"]):
        rgb = cv2.imread(os.path.join(SMOKE, "rgb", f"{i}.png"))
        assert rgb is not None
        assert rgb.shape == (h, w, 3)
        assert rgb.dtype == np.uint8


def test_reference_clouds_agree_with_cloud_stats(oracle, clouds):
    """clouds.npz and cloud_stats describe the same geometry.

    cloud_stats covers the FULL cloud (what depth_image_to_point_cloud returns);
    clouds.npz is its voxel downsample. So they must occupy the same extent, to
    within one voxel on each side — if the two ever disagreed, one of the two
    oracles would be lying.

    Note what is NOT asserted: the two centroids. Voxel downsampling is roughly
    area-uniform while the full cloud is pixel-density weighted (near surfaces get
    far more pixels per unit area), so the centroid legitimately moves by over a
    metre. Same geometry, different sampling measure.
    """
    voxel = oracle["reference_clouds"]["voxel_size"]
    for i, (stats, pcd) in enumerate(zip(oracle["cloud_stats"], clouds)):
        pts = np.asarray(pcd.points)
        lo, hi = np.array(stats["aabb_min"]), np.array(stats["aabb_max"])
        assert len(pts) == oracle["reference_clouds"]["n_points"][i]
        assert len(pts) < stats["n_points"]
        # inside the full cloud's box ...
        assert np.all(pts.min(axis=0) >= lo - voxel)
        assert np.all(pts.max(axis=0) <= hi + voxel)
        # ... and spanning essentially all of it
        assert np.all(pts.min(axis=0) <= lo + voxel)
        assert np.all(pts.max(axis=0) >= hi - voxel)


def test_perfect_trajectory_scores_exactly_zero(oracle):
    """The headline property: on this fixture a correct pipeline scores 0.0.

    The ground truth is derived from the same poses the depth was rendered from,
    so there is no residual to explain away — any nonzero mean_l2 you get is
    yours. Uses the shipped `mean_l2` and the camera centres already tabulated in
    expected.json; it does not re-derive them.
    """
    pred = np.array([p["camera_centre_cam0"] for p in oracle["poses"]])
    gt = np.load(os.path.join(SMOKE, "GT_pose.npy"))
    assert utils.mean_l2(pred, gt) == pytest.approx(oracle["expected_mean_l2_m"],
                                                    abs=1e-12)


def test_open3d_icp_recovers_every_step(oracle, clouds):
    """The fixture's per-step transforms must be recoverable by the SHIPPED ICP.

    Registration runs from an identity initial guess — no motion prior, no
    feature matching — so this isolates one thing: is `steps[i].T_rel` the
    transform that actually aligns frame i onto frame i-1? If your own ICP cannot
    reach these numbers on this data, the problem is your ICP.
    """
    tol = oracle["tolerances"]
    threshold = oracle["reference_clouds"]["icp_threshold_m"]
    for step in oracle["steps"]:
        i = step["source_frame"]
        result = utils.local_icp_algorithm(
            clouds[i], clouds[i - 1], np.eye(4), threshold)

        recovered = np.asarray(result.transformation)
        residual = np.linalg.inv(np.asarray(step["T_rel"])) @ recovered

        trans_err = float(np.linalg.norm(residual[:3, 3]))
        rot_err = utils._rot_angle_deg(residual)
        assert trans_err < tol["step_translation_m"], (
            f"step {i}: translation off by {trans_err:.4f} m")
        assert rot_err < tol["step_rotation_deg"], (
            f"step {i}: rotation off by {rot_err:.4f} deg")


def test_steps_are_within_the_default_gate(oracle):
    """The fixture must not be gated away by reconstruct's default plausibility
    bounds, or it would exercise the coast-on-prior path instead of ICP."""
    defaults = {"gate_trans": 0.5, "gate_rot": 30.0}
    for step in oracle["steps"]:
        assert step["translation_m"] < defaults["gate_trans"]
        assert step["rotation_deg"] < defaults["gate_rot"]


def test_oracle_is_internally_consistent(oracle):
    """steps[] and poses[] must describe the same trajectory.

    Guards the oracle against a hand-edit that changes one and not the other.
    Checks a scalar consequence — the distance between consecutive camera
    centres — rather than recomposing the poses, which is the student's job.
    """
    centres = [np.array(p["camera_centre_cam0"]) for p in oracle["poses"]]
    assert len(centres) == oracle["n_frames"]
    np.testing.assert_allclose(centres[0], np.zeros(3), atol=1e-12)
    for step in oracle["steps"]:
        i = step["source_frame"]
        gap = float(np.linalg.norm(centres[i] - centres[i - 1]))
        # Rotation between frames is small, so the world-frame gap and the
        # step's own translation magnitude must agree to within a few percent.
        assert gap == pytest.approx(step["translation_m"], rel=0.05)
