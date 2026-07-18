"""
Geometry-only ICP SLAM utilities for the HW1 robustness/generalization eval.

Split out of the original reconstruction script so the reconstruction pipeline
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
    """Read a depth PNG from disk and return it as a float64 depth map in metres.

    SPEC:
        Read `depth_path` preserving the on-disk bit depth (cv2.IMREAD_UNCHANGED).
        - If the read fails, return None.
        - If the image has 3 channels, collapse to the first channel.
        - Auto-detect the encoding by dtype and convert to METRES:
            * uint16  -> millimetres:      value / DEPTH_SCALE (1000.0).
            * anything else (uint8 vis) -> Habitat 8-bit vis: value / 255.0 * 10.0.
        Return a float64 H*W array (or None on read failure).

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
    """
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

    SPEC:
        Back-project every valid pixel through the module-level pinhole intrinsics
        (fx, fy, cx, cy) with NO Open3D projection helpers.
        - Validity: keep only pixels with depth_m > 0.
        - For each kept pixel (u, v) with depth Z = depth_m[v, u]:
              X = (u - cx) * Z / fx
              Y = (v - cy) * Z / fy
              Z = Z
          Camera frame: +Z forward (into scene), +X right, +Y down (image order).
        - Colors: BGR uint8 -> RGB float in [0, 1] (divide by 255, reverse channels).
        - Return an o3d.geometry.PointCloud carrying both points (N*3) and colors,
          one point per valid pixel, in row-major pixel order.
        WIKI: https://en.wikipedia.org/wiki/Pinhole_camera_model

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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

    SPEC:
        1. Voxel-downsample `pcd` at `voxel_size`.
        2. Estimate normals on the downsample with a hybrid KD-tree search of
           radius = voxel_size * 2.0, max_nn = 30 (needed for point-to-plane ICP
           and for FPFH).
        3. Compute the FPFH feature on the downsample with a hybrid KD-tree search
           of radius = voxel_size * 5.0, max_nn = 100.
        Return (pcd_down, fpfh). The feature radius must exceed the normal radius.
        WIKI: https://en.wikipedia.org/wiki/Point_Feature_Histograms

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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

    SPEC:
        Run feature-based RANSAC (registration_ransac_based_on_feature_matching)
        aligning `source_down` -> `target_down` from their FPFH features.
        - Correspondence distance threshold: dist_thr = voxel_size * 1.5.
        - mutual_filter = True; estimation = point-to-point (no scaling); ransac_n = 3.
        - Pruning checkers: edge-length ratio 0.9 and distance <= dist_thr.
        - Convergence: RANSACConvergenceCriteria(max_iteration=100000,
          confidence=0.999).
        Return the RegistrationResult; `.transformation` is the coarse 4*4 init
        handed to ICP. (This is the fragile, non-deterministic path — see the
        reconstruct() `robust` note.)
        WIKI: https://en.wikipedia.org/wiki/Random_sample_consensus

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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

    SPEC:
        Refine `trans_init` with a single-threshold point-to-plane ICP.
        - Ensure both clouds have normals; if missing, estimate with a hybrid
          KD-tree of radius = threshold * 2, max_nn = 30.
        - Run registration_icp(source_down, target_down, threshold, trans_init,
          estimation = TransformationEstimationPointToPlane,
          criteria = ICPConvergenceCriteria(max_iteration=100)).
        Return the RegistrationResult (`.transformation` is the refined 4*4).
        WIKI: https://en.wikipedia.org/wiki/Iterative_closest_point

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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
    threshold under-converges at turns and drifts). Geometry-only.

    SPEC:
        Ensure both clouds have normals (hybrid KD-tree radius=0.1, max_nn=30 if
        missing). Start T = trans_init. For each threshold in `thresholds` (coarse
        to fine, default 0.4, 0.2, 0.1, 0.05), run point-to-plane registration_icp
        with ICPConvergenceCriteria(max_iteration=max_iter) and feed the resulting
        transform forward as the init for the next (finer) threshold. Return a
        duck-typed object exposing `.transformation` (final 4*4 np.ndarray) — the
        same attribute an Open3D RegistrationResult exposes, so callers are uniform.
        WIKI: https://en.wikipedia.org/wiki/Iterative_closest_point

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
    """
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

    SPEC:
        Implement point-to-point ICP from scratch (no Open3D registration calls).
        Setup: threshold = voxel_size * 1.5, max_iter = 60, tolerance = 1e-6.
        Take source/target XYZ as float64 arrays; build a cKDTree on the target;
        start T = trans_init (copied). Each iteration:
          1. Transform source by the current T.
          2. Nearest-neighbour correspondences (k=1) into the target via the tree.
          3. Keep pairs with distance < threshold; if fewer than 10 survive, stop.
          4. Solve the optimal rigid transform via the Kabsch/Umeyama SVD:
             centre both point sets, H = Pc^T Qc, U S V^T = svd(H),
             R = V U^T with a det(R) < 0 reflection fix (negate V's last row),
             t = q_bar - R p_bar.
          5. Compose the delta onto T (T = T_delta @ T).
          6. Converge when |prev_mean_err - mean_err| < tolerance.
        Return a duck-typed object with `.transformation` = final 4*4 T.
        WIKI: https://en.wikipedia.org/wiki/Kabsch_algorithm

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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
    """List `directory`'s .png files sorted by integer stem (numeric frame order)."""
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))


def _rot_angle_deg(R):
    """Geodesic rotation angle (deg) of a 3x3 rotation matrix."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def reconstruct(data_root, version="open3d", voxel_size=0.05, verbose=True,
                build_cloud=True, robust=True, gate_trans=0.5, gate_rot=30.0,
                down_voxel=None, frames=None):
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
        down_voxel : if set (and build_cloud), voxel-downsample each frame's world
                    cloud before accumulating so the global map stays small (the
                    coverage/F-score eval needs the cloud, not the full ~10^8 pts).
        frames    : optional list of integer frame stems (ascending) selected
                    upstream (e.g. an ontology/SPARQL query written to a CSV). When
                    None (default), behaviour is EXACTLY as before: glob every .png
                    under rgb/ and depth/ via _sorted_frames. When given, ONLY those
                    stems are used, in the given order: rgb/depth path lists are
                    built as data_root/rgb/<stem>.png and data_root/depth/<stem>.png,
                    and the whole pipeline (pairwise registration, constant-velocity
                    prior, accumulation) runs over this reduced ordered sequence. GT
                    rows are subset by the SAME stems (gt_all[frames]) so pred/GT
                    stay index-aligned.
                    NOTE: dropping interior frames WIDENS per-step motion. The
                    constant-velocity prior and the gate_trans/gate_rot plausibility
                    gate both assume CONSECUTIVE frames (~0.09 m / ~6 deg/step), so
                    large gaps between selected stems degrade accuracy — this is
                    expected behaviour of a subset, not a bug.

    Returns:
        (global_pcd, pred_cam_pos, gt_poses)
          global_pcd   : o3d.geometry.PointCloud — accumulated global map.
          pred_cam_pos : (N,3) float64 — estimated camera centres in the frame-0
                         camera frame (RAW; mean_l2 handles axis reconciliation).
          gt_poses     : (M,7) float — GT [x,y,z,qw,qx,qy,qz], or None if absent.

    SPEC:
        Build a per-frame trajectory (and optionally a global map) by chaining
        pairwise rigid registrations, GEOMETRY ONLY (colour never enters registration).
        - Frames: numeric-sorted .png pairs from data_root/rgb and data_root/depth;
          n_frames = min(len(rgb), len(depth)); depth via load_depth_meters; skip a
          frame whose RGB or depth fails to load. If no cloud loads, return an empty
          map, zeros((0,3)), and the GT.
        - Unproject each frame to a local cloud (depth_image_to_point_cloud).
        - Frame 0 anchors the world: T_global[0] = identity, pred_cam_pos[0] = 0.
        - For each i>=1, register source=frame i onto target=frame i-1 (both
          voxel-downsampled at `voxel_size`) to get a relative transform T_rel:
            * robust=True (default): init T_rel from the constant-velocity prior
              (previous T_rel), NO FPFH RANSAC; refine with multiscale_icp when
              version=="open3d" (else my_local_icp_algorithm). Then a PHYSICAL GATE:
              if ||T_rel translation|| > gate_trans OR geodesic rotation angle >
              gate_rot, discard T_rel and coast on the prior (count as gated).
            * robust=False: init from global_registration (FPFH RANSAC) then a
              single-threshold local_icp_algorithm (icp_thr = voxel_size*1.5) —
              the fragile, non-deterministic student-template chain.
          version=="my_icp" uses my_local_icp_algorithm in both paths.
        - Accumulate: T_i = T_global[i-1] @ T_rel; append T_i and its translation
          T_i[:3,3] as the camera centre. If build_cloud, deep-copy frame i, apply
          T_i, optionally voxel-downsample at down_voxel, and add into the map.
        - GT: load data_root/GT_pose.npy (None if absent).
        Determinism: the robust path has no uncovered RNG, so scores reproduce.
        Return (global_pcd, pred_cam_pos as (N,3) float64, gt_poses).
        WIKI: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
    """
    rgb_dir   = os.path.join(data_root, 'rgb')
    depth_dir = os.path.join(data_root, 'depth')

    if frames is None:
        # Default: every .png under rgb/ and depth/, numeric-sorted (unchanged).
        rgb_files   = _sorted_frames(rgb_dir)
        depth_files = _sorted_frames(depth_dir)
        n_frames    = min(len(rgb_files), len(depth_files))
    else:
        # Subset: use ONLY the given stems, in the given order.
        frames      = [int(s) for s in frames]
        rgb_files   = [f"{s}.png" for s in frames]
        depth_files = [f"{s}.png" for s in frames]
        n_frames    = len(frames)
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
        return o3d.geometry.PointCloud(), np.zeros((0, 3)), _load_gt(data_root, frames)

    # ── Sequential pairwise registration ────────────────────────────────────
    def _maybe_down(p):
        return p.voxel_down_sample(down_voxel) if down_voxel else p

    T_global     = [np.eye(4)]
    all_pcds     = [_maybe_down(pcds[0])] if build_cloud else None
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
            all_pcds.append(_maybe_down(pcd_i_world))

        if verbose and (i % 25 == 0 or i == n - 1):
            print(f"  frame {i:>4d}/{n-1}  dt={time.time()-t0:.2f}s  gated={n_gated}")

    global_pcd = o3d.geometry.PointCloud()
    if build_cloud:
        for pcd in all_pcds:
            global_pcd += pcd
    pred_cam_pos = np.array(pred_cam_pos, dtype=np.float64)

    return global_pcd, pred_cam_pos, _load_gt(data_root, frames)


def _load_gt(data_root, frames=None):
    """Load data_root/GT_pose.npy (the (M,7) GT pose array), or None if absent.

    When `frames` (a list of integer stems) is given, subset the GT rows by those
    SAME stems (row i <-> frame i in the original capture order) so the returned GT
    stays index-aligned with a reconstruction run over the frame subset. Stems out
    of range for the GT array are skipped safely, preserving the given order.
    """
    gt_path = os.path.join(data_root, 'GT_pose.npy')
    if not os.path.exists(gt_path):
        return None
    gt = np.load(gt_path)
    if frames is None:
        return gt
    stems = [int(s) for s in frames if 0 <= int(s) < len(gt)]
    return gt[stems]


def mean_l2(pred_cam_pos, gt_poses):
    """
    Mean L2 distance between predicted and GT camera centres (metres).

    Reconciles the two coordinate frames exactly as the original reconstruct.py
    __main__ did before scoring:
      * pred is in frame-0 CAMERA space  → flip Y to reach the Habitat world frame.
      * GT is Habitat world [x,y,z,...]  → flip Z to match pred's display frame.
      * origins are aligned (ICP starts at the frame-0 camera, not world origin).
    Returns +inf if either trajectory is missing/empty.

    SPEC:
        Compute the mean per-frame Euclidean distance (metres) between predicted and
        GT camera centres over the first n = min(len(pred), len(gt)) frames.
        - Guard: return float("inf") if either input is None or empty.
        - Reconcile frames before comparing:
            pred (frame-0 camera space): negate the Y column (camera -> Habitat world).
            gt   (Habitat world x,y,z):  negate the Z column (match pred display frame).
        - Align origins by adding offset = gt_c[0] - pred[0] to every pred point
          (ICP starts at the frame-0 camera, not the world origin — no scale/rotation
          fit, translation only).
        - Return mean over i of ||pred_aligned[i] - gt_c[i]||_2.

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
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
    """Create an Open3D LineSet from a sequence of XYZ camera positions.

    SPEC:
        Given positions (N*3) and an RGB `color` triple, build an o3d LineSet whose
        vertices are the positions and whose edges connect each consecutive pair
        [i, i+1] for i in 0..N-2 (a polyline through the trajectory), with every
        line painted `color`. Return the LineSet.

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
    """
    positions = np.asarray(positions)
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def remove_ceiling(pcd, margin=0.3):
    """Crop the ceiling out of a point cloud for a cleaner top-down view.

    SPEC:
        In the camera frame the ceiling sits at the MINIMUM y (y points down).
        Let y_min = min of the y coordinates. Keep only points with
        y > y_min + margin, carrying their matching colors across. Return a new
        PointCloud of the kept points/colors (input left unmodified).

    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.
    """
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    y_min = pts[:, 1].min()                    # ceiling is at min in camera frame
    mask  = pts[:, 1] > (y_min + margin)
    out   = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    out.colors = o3d.utility.Vector3dVector(cols[mask])
    return out
