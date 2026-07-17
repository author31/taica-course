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
    The whole-floor map is built from the floor's *baseline* (clean) capture:
    every frame is unprojected and placed with its GROUND-TRUTH pose (same anchor
    math, per frame). It is fixed and independent of what a student collected, so
    a 5-frame submission is scored against the entire apartment.
"""

import os
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

import utils as U   # same directory (hw1/) — added to sys.path by the caller

F_OPTICAL = np.diag([1.0, -1.0, -1.0])   # optical frame -> habitat camera frame
DEFAULT_TAUS = (0.05, 0.10, 0.20)
PRIMARY_TAU = 0.10


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


def build_gt_reference(baseline_root, stride=4, voxel=0.03):
    """Whole-floor GT map: unproject every `stride`-th baseline frame and place it
    with its ground-truth pose. Returns an o3d.geometry.PointCloud (world frame)."""
    gt = np.load(os.path.join(baseline_root, "GT_pose.npy"))
    rgb_dir = os.path.join(baseline_root, "rgb")
    dep_dir = os.path.join(baseline_root, "depth")
    rf, df = U._sorted_frames(rgb_dir), U._sorted_frames(dep_dir)
    n = min(len(rf), len(df), len(gt))
    acc = o3d.geometry.PointCloud()
    for i in range(0, n, stride):
        rgb = cv2.imread(os.path.join(rgb_dir, rf[i]))
        d = U.load_depth_meters(os.path.join(dep_dir, df[i]))
        if rgb is None or d is None:
            continue
        p = U.depth_image_to_point_cloud(rgb, d).voxel_down_sample(voxel)
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
