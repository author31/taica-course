"""
Coverage-aware reconstruction score for HW1 — accuracy / completeness / F-score.

WHY (beyond mean L2)
    Trajectory error alone rewards doing less: a student who captures 5 frames of
    one corner can post a tiny error, while one who covers the whole apartment in
    ~380 frames scores worse. Correctness must couple *accuracy* with *coverage*.
    This module compares the reconstructed cloud against a whole-floor GT map:

        accuracy(tau)     = frac of PRED points within tau of the GT map
        completeness(tau) = frac of GT-map points within tau of PRED
        F(tau)            = 2 * A * C / (A + C)

    accuracy alone is gameable by a well-placed sliver; completeness collapses for
    a sliver (most of the floor is uncovered); F folds both into one number.

FRAME MISMATCH — handled by ONE known transform (no fitting)
    The reconstruction R is expressed relative to the first camera. Habitat gives
    the world pose of that first camera, so a single anchor matrix lifts every
    reconstructed point into the world frame:

        T_anchor = Twc0 @ F
          Twc0 = [ R(quat0) | t0 ]     world  <- habitat camera 0   (from GT_pose[0])
          F    = diag(1, -1, -1)       habitat camera <- optical frame

    No Umeyama, no ICP, no RANSAC — nothing to overfit or diverge. The residual
    pred->GT distance therefore stays equal to the real reconstruction drift
    (an alignment fit would hide that drift; anchoring keeps the score honest).

GT REFERENCE
    The whole-floor map is built from the *baseline* (clean, scheduler-off)
    capture of evaluate.py's two-run flow — eval/_data/second_floor/baseline/ —
    NEVER from the uncertainty-corrupted mixed/ capture (build_gt_reference
    rejects a mixed/ dir outright). Every frame is unprojected and placed with
    its GROUND-TRUTH pose (same anchor math, per frame). It is fixed and
    independent of what a student collected, so a 5-frame submission is scored
    against the entire apartment.

THIS MODULE IS THE GRADER — IT IMPORTS NOTHING FROM utils.py
    utils.py is student-editable, so anything the grader takes from it is a
    channel through which a submission can move its own score. Two ways that
    bites, and the second is the serious one:
      * utils.py's unprojection ships blank (`#TODO`) — an unfinished utils.py
        would make the grader itself uncallable;
      * build_gt_reference builds the REFERENCE the score is measured against,
        so a creative or simply buggy student unprojection would corrupt the
        yardstick rather than the thing being measured.
    Hence the frozen private helpers below (_read_intrinsics, _sorted_frames,
    _load_depth_meters, _unproject). They deliberately duplicate a little of
    utils.py; see the comment on _unproject before "cleaning that up".
"""

import os
import json
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

F_OPTICAL = np.diag([1.0, -1.0, -1.0])   # optical frame -> habitat camera frame
DEFAULT_TAUS = (0.05, 0.10, 0.20)
PRIMARY_TAU = 0.10
_DEPTH_SCALE = 1000.0                    # uint16 depth PNGs store millimetres


def quat_to_R(q):
    """q = [qw, qx, qy, qz] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def anchor_transform(gt0):
    """World <- frame-0-optical transform from the first camera's GT pose only.
    gt0 = [x, y, z, qw, qx, qy, qz]."""
    Twc0 = np.eye(4)
    Twc0[:3, :3] = quat_to_R(gt0[3:7])
    Twc0[:3, 3] = gt0[:3]
    F = np.eye(4)
    F[:3, :3] = F_OPTICAL
    return Twc0 @ F


def _apply(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# ══════════════════════════════════════════════════════════════════════════════
#  Frozen grader-private I/O + geometry
#
#  DELIBERATE DUPLICATION — DO NOT DE-DUPLICATE.
#  The helpers below restate utils.py's frame listing, depth loading and
#  unprojection. That is not an oversight and not a leftover: this module is the
#  grader, and utils.py is part of what students hand in. Importing the
#  student-facing versions would let a submission decide how its own ground-truth
#  reference map is built — and, since the unprojection there ships blank
#  (`#TODO`), would stop the grader from running at all against an unfinished
#  submission. A few dozen lines of repeated pinhole math is the cheap side of
#  that trade. Keep them frozen: they must not drift with utils.py, and a change
#  here changes everyone's score.
# ══════════════════════════════════════════════════════════════════════════════

def _read_intrinsics(capture_root):
    """Return (width, height, hfov_deg) from <capture_root>/intrinsics.json.

    Every capture ships its own camera parameters next to GT_pose.npy (plan D5),
    so the GT map is always unprojected through the camera that actually recorded
    it. There is deliberately NO fallback default: a silently wrong camera makes a
    silently wrong reference map, which is worse than a crash because every score
    downstream is then quietly wrong."""
    path = os.path.join(capture_root, "intrinsics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing camera intrinsics: {path!r}\n"
            "Every capture directory must ship intrinsics.json "
            '({"width", "height", "hfov"}) alongside GT_pose.npy, and the GT '
            "reference map is unprojected through it. Re-collect the capture "
            "with hw1/load.py or scripts/evaluate.py (both emit it), or copy the "
            "file from the capture it was derived from. It is NOT defaulted — a "
            "guessed camera yields a wrong reference and therefore wrong scores.")
    with open(path) as f:
        data = json.load(f)
    missing = [k for k in ("width", "height", "hfov") if k not in data]
    if missing:
        raise ValueError(
            f"{path!r} is missing required key(s) {missing}; it must hold exactly "
            '{"width": int, "height": int, "hfov": float (DEGREES)}')
    return int(data["width"]), int(data["height"]), float(data["hfov"])


def _sorted_frames(directory):
    """List `directory`'s .png files sorted by integer stem (numeric frame order)."""
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))


def _load_depth_meters(depth_path):
    """Read a depth PNG preserving bit depth and return float64 METRES, or None.

    uint16 -> millimetres / 1000; anything else is a Habitat 8-bit visualisation
    and maps 0..255 onto 0..10 m."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.dtype == np.uint16:
        return d.astype(np.float64) / _DEPTH_SCALE
    return d.astype(np.float64) / 255.0 * 10.0


def _unproject(rgb, depth_m, width, height, hfov):
    """Back-project one RGB-D frame into a colored cloud in the CAMERA frame.

    Pinhole, square pixels, no skew/distortion, principal point at the image
    centre:  fx = fy = (width/2) / tan(hfov/2),  cx = width/2,  cy = height/2.
    Pixels with depth_m <= 0 are dropped (no return), so nothing lands at the
    origin. Points are metres, +X right, +Y down, +Z forward; colors are RGB in
    [0,1] (rgb comes in BGR, as cv2.imread returns it). Row-major pixel order.

    A shape mismatch is raised rather than papered over: it means intrinsics.json
    does not describe these images, i.e. the wrong camera for this capture."""
    h, w = depth_m.shape
    if (height, width) != (h, w):
        raise ValueError(
            f"intrinsics ({width}x{height}) do not match the frame ({w}x{h}) — "
            "intrinsics.json belongs to a different capture")

    valid = depth_m > 0
    Z = depth_m[valid].astype(np.float64)
    rgb_v = rgb[valid]

    fx = fy = (width / 2.0) / np.tan(np.radians(hfov / 2.0))
    cx, cy = width / 2.0, height / 2.0

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    X = (u_grid[valid] - cx) * Z / fx
    Y = (v_grid[valid] - cy) * Z / fy

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([X, Y, Z]))
    pcd.colors = o3d.utility.Vector3dVector(
        rgb_v.astype(np.float64)[:, ::-1] / 255.0)          # BGR -> RGB
    return pcd


def build_gt_reference(baseline_root, stride=4, voxel=0.03):
    """Whole-floor GT map: unproject every `stride`-th baseline frame and place it
    with its ground-truth pose. Returns an o3d.geometry.PointCloud (world frame).

    `baseline_root` must be a BASELINE (clean, scheduler-off) capture dir from
    evaluate.py's two-run flow. Pointing it at the uncertainty-corrupted mixed/
    capture would poison the reference every score is measured against, so a
    dir whose basename is "mixed" is rejected.

    Camera parameters come from `baseline_root/intrinsics.json` — the capture's
    own camera, never a config and never a default (see _read_intrinsics).
    Unprojection uses this module's frozen _unproject, NOT utils.py: the grader
    does not run student code."""
    base = os.path.basename(os.path.normpath(baseline_root))
    if base == "mixed":
        raise ValueError(
            f"build_gt_reference must consume a baseline/ capture, got {baseline_root!r} "
            "— the mixed/ (uncertainty-corrupted) run must never feed the GT reference")
    width, height, hfov = _read_intrinsics(baseline_root)
    gt = np.load(os.path.join(baseline_root, "GT_pose.npy"))
    rgb_dir = os.path.join(baseline_root, "rgb")
    dep_dir = os.path.join(baseline_root, "depth")
    rf, df = _sorted_frames(rgb_dir), _sorted_frames(dep_dir)
    n = min(len(rf), len(df), len(gt))
    acc = o3d.geometry.PointCloud()
    for i in range(0, n, stride):
        rgb = cv2.imread(os.path.join(rgb_dir, rf[i]))
        d = _load_depth_meters(os.path.join(dep_dir, df[i]))
        if rgb is None or d is None:
            continue
        p = _unproject(rgb, d, width, height, hfov).voxel_down_sample(voxel)
        w = _apply(anchor_transform(gt[i]), np.asarray(p.points))
        q = o3d.geometry.PointCloud()
        q.points = o3d.utility.Vector3dVector(w)
        acc += q
    return acc.voxel_down_sample(voxel)


def anchor_pred_cloud(pred_cam0_cloud, gt_poses):
    """Lift a reconstruction (in the frame-0 camera frame) into the world frame
    using only the first GT pose. Returns a new world-frame PointCloud."""
    T = anchor_transform(gt_poses[0])
    w = _apply(T, np.asarray(pred_cam0_cloud.points))
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(w)
    return out


def score(pred_world, gt_ref, taus=DEFAULT_TAUS):
    """Return {tau: {'accuracy', 'completeness', 'f'}} plus 'mean_pred_to_gt'."""
    pw = np.asarray(pred_world.points)
    gw = np.asarray(gt_ref.points)
    if len(pw) == 0 or len(gw) == 0:
        return {"mean_pred_to_gt": float("inf"),
                **{t: {"accuracy": 0.0, "completeness": 0.0, "f": 0.0}
                   for t in taus}}
    d_pred = cKDTree(gw).query(pw, workers=-1)[0]      # pred -> GT (accuracy)
    d_gt = cKDTree(pw).query(gw, workers=-1)[0]        # GT -> pred (completeness)
    out = {"mean_pred_to_gt": float(d_pred.mean())}
    for tau in taus:
        acc = float(np.mean(d_pred < tau))
        comp = float(np.mean(d_gt < tau))
        f = 0.0 if acc + comp == 0 else 2 * acc * comp / (acc + comp)
        out[tau] = {"accuracy": acc, "completeness": comp, "f": f}
    return out


def fscore_for_capture(pred_cam0_cloud, gt_poses, gt_ref, taus=DEFAULT_TAUS):
    """Convenience: anchor a reconstruction and score it against the GT map."""
    pred_world = anchor_pred_cloud(pred_cam0_cloud, gt_poses)
    return score(pred_world, gt_ref, taus)
