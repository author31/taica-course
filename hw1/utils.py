"""
Geometry-only ICP SLAM utilities for the HW1 robustness/generalization eval.

Split out of the original scripts/reconstruct.py so the reconstruction pipeline
can be driven headless (no Open3D window) from the evaluator, while the thin
hw1/reconstruct.py CLI still imports these for interactive visualisation.

PIPELINE (geometry-only, by design)
    Per consecutive frame pair: FPFH RANSAC global registration -> point-to-plane
    ICP refinement. NO colour is used in registration. Lighting perturbation
    therefore reaches the geometry ONLY through the depth sensor's ambient-light
    coupling (see load.apply_depth_sensor): brighter/darker exposure raises depth
    noise / dropout / range loss, which moves the reconstruction metric.

DEPTH FORMAT
    load_depth_meters auto-detects the on-disk depth encoding:
      * uint16 PNG  -> millimetres      (value / 1000 = metres)   [eval path]
      * uint8  PNG  -> Habitat 8-bit vis (value / 255 * 10 = metres)
    The eval collector (load.save_frame) writes 16-bit mm so the injected
    coupling noise survives to the reconstructor instead of being swamped by
    8-bit quantisation.

KEY EXPORTS
    load_depth_meters, depth_image_to_point_cloud, preprocess_point_cloud,
    global_registration, local_icp_algorithm, my_local_icp_algorithm,
    remove_ceiling, make_trajectory,
    reconstruct(data_root, version="open3d") -> (pcd, pred_cam_pos, gt_poses),
    mean_l2(pred_cam_pos, gt_poses) -> float
"""

import numpy as np
import open3d as o3d
import os
import time
import cv2
import copy
from scipy.spatial import cKDTree

# ──────────────────────────────────────────────────────────────────────────────
# Camera Intrinsics
#   Pinhole camera · Resolution 512*512 · FOV 90° (H and V)
#   fx = fy = (W/2) / tan(FOV/2) = 256 / tan(45°) = 256
#   cx = cy = 256
#   depth_scale = 1000  →  Z_meters = raw_depth / 1000
# ──────────────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 512, 512
FOV_DEG      = 90.0
fx = fy      = (IMG_W / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))   # 256.0
cx, cy       = IMG_W / 2.0, IMG_H / 2.0                              # 256.0
DEPTH_SCALE  = 1000.0


def load_depth_meters(depth_path):
    """Read a depth PNG and return it as float64 metres.

    uint16 → millimetres (/1000); anything else → Habitat 8-bit vis (/255*10).
    Returns None if the file cannot be read."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.dtype == np.uint16:
        return d.astype(np.float64) / DEPTH_SCALE          # mm → m
    return d.astype(np.float64) / 255.0 * 10.0             # 8-bit vis → m


def depth_image_to_point_cloud(rgb, depth_m):
    """
    Convert an RGB image and a depth map (metres, float) into a colored 3-D point
    cloud using the pinhole camera model. No Open3D projection utilities are used.

    Args:
        rgb     : H*W*3 uint8 BGR image (as loaded by cv2).
        depth_m : H*W float array, depth in METRES (see load_depth_meters).

    Returns:
        o3d.geometry.PointCloud with XYZ positions and RGB colors.
    """
    h, w = depth_m.shape

    # ── Validity mask ────────────────────────────────────────────────────────
    valid = depth_m > 0
    Z = depth_m[valid].astype(np.float64)
    rgb_v = rgb[valid]

    # ── Pixel grids ──────────────────────────────────────────────────────────
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_v = u_grid[valid]
    v_v = v_grid[valid]

    # ── Back-projection  (pinhole inverse) ──────────────────────────────────
    X = (u_v - cx) * Z / fx
    Y = (v_v - cy) * Z / fy

    points = np.column_stack([X, Y, Z])                     # N * 3

    # ── Colors (BGR → RGB, normalised) ─────────────────────────────────────
    colors = rgb_v.astype(np.float64) / 255.0
    colors = colors[:, ::-1]                                # BGR → RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Voxel-downsample a point cloud and compute FPFH descriptors for
    feature-based global registration.

    Returns:
        pcd_down : downsampled PointCloud with normals
        fpfh     : o3d.pipelines.registration.Feature
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normal estimation for ICP (Point-to-Plane) and FPFH
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0, max_nn=30))

    # FPFH feature (Fast Point Feature Histogram)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5.0, max_nn=100))

    return pcd_down, fpfh


def global_registration(source_down, target_down, source_fpfh,
                        target_fpfh, voxel_size):
    """
    Estimate an initial rigid transform between two downsampled point clouds
    using RANSAC with FPFH feature matching.

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    dist_thr = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thr,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        criteria=o3d.pipelines.registration
            .RANSACConvergenceCriteria(100000, 0.999))

    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    Refine alignment with Open3D's Point-to-Plane ICP.

    Args:
        source_down : source PointCloud (with normals)
        target_down : target PointCloud (with normals)
        trans_init  : 4*4 initial transform (from RANSAC)
        threshold   : max correspondence distance

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    # Guarantee normals exist on both clouds
    for pcd in (source_down, target_down):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=threshold * 2, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100))

    return result


def multiscale_icp(source_down, target_down, trans_init,
                   thresholds=(0.4, 0.2, 0.1, 0.05), max_iter=60):
    """Coarse-to-fine point-to-plane ICP: refine `trans_init` through decreasing
    correspondence thresholds. The coarse passes give ICP a wide capture range so
    it reaches the true alignment from a constant-velocity init (a single tight
    threshold under-converges at turns and drifts). Geometry-only."""
    for pcd in (source_down, target_down):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    T = trans_init
    for thr in thresholds:
        T = o3d.pipelines.registration.registration_icp(
            source_down, target_down, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter)).transformation
    class _Result:
        def __init__(self, transformation):
            self.transformation = transformation
    return _Result(T)


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom point-to-point ICP (SVD per iteration, cKDTree correspondences).

    Returns a duck-typed result with a `.transformation` (4*4 np.ndarray).
    """
    threshold    = voxel_size * 1.5
    max_iter     = 60
    tolerance    = 1e-6

    src = np.asarray(source_down.points, dtype=np.float64)   # N*3
    tgt = np.asarray(target_down.points, dtype=np.float64)   # M*3

    T = trans_init.copy().astype(np.float64)
    tree = cKDTree(tgt)
    prev_err = np.inf

    for _it in range(max_iter):
        R_cur  = T[:3, :3]
        t_cur  = T[:3, 3]
        src_t  = (R_cur @ src.T).T + t_cur          # N*3

        dists, idx = tree.query(src_t, k=1, workers=1)

        mask = dists < threshold
        if mask.sum() < 10:
            break

        P = src_t[mask]
        Q = tgt[idx[mask]]

        p_bar = P.mean(axis=0)
        q_bar = Q.mean(axis=0)
        Pc = P - p_bar
        Qc = Q - q_bar

        H        = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        R_delta  = Vt.T @ U.T
        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T
        t_delta = q_bar - R_delta @ p_bar

        T_delta          = np.eye(4)
        T_delta[:3, :3]  = R_delta
        T_delta[:3,  3]  = t_delta
        T                = T_delta @ T

        mean_err = dists[mask].mean()
        if abs(prev_err - mean_err) < tolerance:
            break
        prev_err = mean_err

    class _Result:
        def __init__(self, transformation):
            self.transformation = transformation

    return _Result(T)


# ══════════════════════════════════════════════════════════════════════════════
#  Reconstruction (headless) + metric
# ══════════════════════════════════════════════════════════════════════════════
def _sorted_frames(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))


def _rot_angle_deg(R):
    """Geodesic rotation angle (deg) of a 3x3 rotation matrix."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def reconstruct(data_root, version="open3d", voxel_size=0.05, verbose=True,
                build_cloud=True, robust=True, gate_trans=0.5, gate_rot=30.0):
    """
    Geometry-only ICP SLAM over the frames under `data_root`.

    Steps:
      1. Load RGB + depth frames (depth via load_depth_meters).
      2. Unproject each depth frame → local point cloud.
      3. Pairwise registration (RANSAC + point-to-plane ICP) between consecutive
         frames; chain the relative transforms into a global trajectory.
      4. Accumulate all clouds into one global map.
      5. Load ground-truth poses from data_root/GT_pose.npy.

    Args:
        data_root : dir containing rgb/, depth/, GT_pose.npy.
        version   : "open3d" (point-to-plane ICP) or "my_icp" (custom SVD ICP).
        build_cloud : accumulate the full global map (default). Pass False in the
                      eval path — only the trajectory is scored, and the map runs
                      to ~10^8 points, which is slow and memory-heavy to build.
        robust    : DEFAULT. Consecutive frames are registered by point-to-plane
                    ICP initialised from a CONSTANT-VELOCITY prior (previous
                    relative transform), with NO FPFH RANSAC. A physical gate
                    rejects any relative transform whose translation/rotation is
                    implausible for one step (> gate_trans m / gate_rot deg) and
                    coasts on the prior instead. This is what makes the pipeline
                    stable AND reproducible: FPFH RANSAC on Replica's rotationally
                    symmetric geometry returns spurious ~10^1 m alignments (one bad
                    pair derails the whole trajectory), and its RNG is not covered
                    by seed → non-deterministic scores. Still geometry-only, so
                    lighting reaches the metric ONLY through the depth sensor.
                    robust=False restores the raw FPFH-RANSAC + ICP chain (the
                    fragile student-template pipeline) for comparison.
        gate_trans/gate_rot : per-step plausibility gate (m / deg). GT steps here
                    are ~0.09 m / ~6 deg, so 0.5 m / 30 deg is generous headroom.

    Returns:
        (global_pcd, pred_cam_pos, gt_poses)
          global_pcd   : o3d.geometry.PointCloud — accumulated global map.
          pred_cam_pos : (N,3) float64 — estimated camera centres in the frame-0
                         camera frame (RAW; mean_l2 handles axis reconciliation).
          gt_poses     : (M,7) float — GT [x,y,z,qw,qx,qy,qz], or None if absent.
    """
    rgb_dir   = os.path.join(data_root, 'rgb')
    depth_dir = os.path.join(data_root, 'depth')

    rgb_files   = _sorted_frames(rgb_dir)
    depth_files = _sorted_frames(depth_dir)
    n_frames    = min(len(rgb_files), len(depth_files))
    if verbose:
        print(f"[reconstruct] {data_root}: {n_frames} frames | version={version}")

    # ── Load all point clouds ───────────────────────────────────────────────
    pcds = []
    for i in range(n_frames):
        rgb   = cv2.imread(os.path.join(rgb_dir, rgb_files[i]))
        depth_m = load_depth_meters(os.path.join(depth_dir, depth_files[i]))
        if rgb is None or depth_m is None:
            if verbose:
                print(f"  Warning: could not load frame {i}, skipping.")
            continue
        pcds.append(depth_image_to_point_cloud(rgb, depth_m))

    n = len(pcds)
    if n == 0:
        return o3d.geometry.PointCloud(), np.zeros((0, 3)), _load_gt(data_root)

    # ── Sequential pairwise registration ────────────────────────────────────
    T_global     = [np.eye(4)]
    all_pcds     = [pcds[0]] if build_cloud else None
    pred_cam_pos = [np.zeros(3)]
    T_rel_prev   = np.eye(4)          # constant-velocity prior (robust path)
    icp_thr      = voxel_size * 1.5
    n_gated      = 0

    for i in range(1, n):
        t0 = time.time()
        source = pcds[i]
        target = pcds[i - 1]

        src_down, _ = preprocess_point_cloud(source, voxel_size)
        tgt_down, _ = preprocess_point_cloud(target, voxel_size)

        if robust:
            trans_init = T_rel_prev                      # constant-velocity init
        else:
            src_down2, src_fpfh = preprocess_point_cloud(source, voxel_size)
            tgt_down2, tgt_fpfh = preprocess_point_cloud(target, voxel_size)
            trans_init = global_registration(
                src_down2, tgt_down2, src_fpfh, tgt_fpfh, voxel_size).transformation

        if version == 'open3d':
            # Coarse-to-fine in the robust path (wide capture from the CV prior);
            # single tight-threshold ICP in the raw path for template fidelity.
            if robust:
                result_icp = multiscale_icp(src_down, tgt_down, trans_init)
            else:
                result_icp = local_icp_algorithm(
                    src_down, tgt_down, trans_init, icp_thr)
        else:   # my_icp
            result_icp = my_local_icp_algorithm(
                src_down, tgt_down, trans_init, voxel_size)
        T_rel = result_icp.transformation

        # Physical gate: reject an implausible one-step jump and coast on the
        # constant-velocity prior. This is what kills the RANSAC-symmetry
        # derailment (spurious ~10^1 m / large-angle pairs).
        if robust and (np.linalg.norm(T_rel[:3, 3]) > gate_trans
                       or _rot_angle_deg(T_rel) > gate_rot):
            T_rel = T_rel_prev
            n_gated += 1

        T_rel_prev = T_rel
        T_i = T_global[-1] @ T_rel
        T_global.append(T_i)
        pred_cam_pos.append(T_i[:3, 3])

        if build_cloud:
            pcd_i_world = copy.deepcopy(pcds[i])
            pcd_i_world.transform(T_i)
            all_pcds.append(pcd_i_world)

        if verbose and (i % 25 == 0 or i == n - 1):
            print(f"  frame {i:>4d}/{n-1}  dt={time.time()-t0:.2f}s  gated={n_gated}")

    global_pcd = o3d.geometry.PointCloud()
    if build_cloud:
        for pcd in all_pcds:
            global_pcd += pcd
    pred_cam_pos = np.array(pred_cam_pos, dtype=np.float64)

    return global_pcd, pred_cam_pos, _load_gt(data_root)


def _load_gt(data_root):
    gt_path = os.path.join(data_root, 'GT_pose.npy')
    if not os.path.exists(gt_path):
        return None
    return np.load(gt_path)


def mean_l2(pred_cam_pos, gt_poses):
    """
    Mean L2 distance between predicted and GT camera centres (metres).

    Reconciles the two coordinate frames exactly as the original reconstruct.py
    __main__ did before scoring:
      * pred is in frame-0 CAMERA space  → flip Y to reach the Habitat world frame.
      * GT is Habitat world [x,y,z,...]  → flip Z to match pred's display frame.
      * origins are aligned (ICP starts at the frame-0 camera, not world origin).
    Returns +inf if either trajectory is missing/empty.
    """
    if gt_poses is None or len(gt_poses) == 0 or pred_cam_pos is None \
            or len(pred_cam_pos) == 0:
        return float("inf")

    pred = np.asarray(pred_cam_pos, dtype=np.float64).copy()
    pred[:, 1] *= -1                                   # cam → habitat world (Y flip)

    gt_c = np.asarray(gt_poses, dtype=np.float64)[:, :3].copy()
    gt_c[:, 2] *= -1                                   # match pred display frame (Z flip)

    n = min(len(pred), len(gt_c))
    offset = gt_c[0] - pred[0]                          # align origins
    pred_al = pred[:n] + offset
    return float(np.mean(np.linalg.norm(pred_al - gt_c[:n], axis=1)))


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers (used by the thin reconstruct.py CLI)
# ══════════════════════════════════════════════════════════════════════════════
def make_trajectory(positions, color):
    """Create an Open3D LineSet from a sequence of XYZ camera positions."""
    positions = np.asarray(positions)
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def remove_ceiling(pcd, margin=0.3):
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    y_min = pts[:, 1].min()                    # ceiling is at min in camera frame
    mask  = pts[:, 1] > (y_min + margin)
    out   = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    out.colors = o3d.utility.Vector3dVector(cols[mask])
    return out
