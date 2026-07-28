"""
Geometry-only ICP SLAM utilities for the HW1 robustness/generalization eval.

Split out of the original reconstruction script so the reconstruction pipeline
can be driven headless (no Open3D window) from the evaluator, while the thin
hw1/reconstruct.py CLI still imports these for interactive visualisation.

GEOMETRY ONLY, BY DESIGN
    NO colour is used in registration, anywhere. Lighting perturbation therefore
    reaches the geometry ONLY through the depth sensor's ambient-light coupling
    (see load.apply_depth_sensor): brighter/darker exposure raises depth noise /
    dropout / range loss, which moves the reconstruction metric.

    Keep it that way. RGB is a declared *proxy* in this assignment, not a cause —
    if colour leaked into registration, the causal chain being measured would stop
    being the one being claimed, and every conclusion drawn downstream would be
    unsupported.

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
# Camera intrinsics — PER CAPTURE, never hardcoded here.
#
#   Every capture directory ships its own `intrinsics.json` next to rgb/, depth/
#   and GT_pose.npy, carrying exactly three keys:
#
#       {"width": <px>, "height": <px>, "hfov": <degrees>}
#
#   Read them from the capture you are reconstructing and pass them in. Two
#   different floors, resolutions or fields of view must not be able to end up
#   unprojected through the same baked-in constants — that failure is silent and
#   produces a plausible-looking, wrong reconstruction.
#
#   Depth ENCODING is a separate thing and is not per capture: it is fixed by the
#   on-disk format and handled by load_depth_meters.
# ──────────────────────────────────────────────────────────────────────────────
DEPTH_SCALE  = 1000.0     # uint16 depth PNGs store millimetres


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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.dtype == np.uint16:
        return d.astype(np.float64) / DEPTH_SCALE          # mm → m
    return d.astype(np.float64) / 255.0 * 10.0             # 8-bit vis → m


def depth_image_to_point_cloud(rgb, depth_m, width, height, hfov):
    """
    Back-project one RGB-D frame into a colored 3-D point cloud.

    CONTRACT
        Inputs
            rgb     : H*W*3 uint8, channels in **BGR** order (cv2.imread order).
            depth_m : H*W float array, depth in **METRES** (see load_depth_meters).
                      Both arrays describe the same frame and share H and W.
            width   : int, PIXELS   — sensor width  from the capture's intrinsics.
            height  : int, PIXELS   — sensor height from the capture's intrinsics.
            hfov    : float, DEGREES — horizontal field of view, ditto.
                      These three are the capture's camera parameters and they are
                      ORDINARY ARGUMENTS: the caller reads them from the
                      `intrinsics.json` sitting in the capture directory being
                      reconstructed and passes them down. This function opens no
                      files, reads no config, and assumes no resolution or FOV.
                      The provided phase-1 capture and anything collected in phase 2
                      each carry their own, so the same code path serves both floors
                      and cannot unproject one floor's depth through another
                      floor's camera.

        Output
            o3d.geometry.PointCloud carrying BOTH `.points` and `.colors`:
              points : (N,3) float64, **metres**, in this frame's CAMERA frame —
                       +X right, +Y down, +Z forward into the scene (image axes).
              colors : (N,3) float in [0,1], **RGB** order (i.e. the channel order
                       is reversed relative to the `rgb` argument).

        Validity
            A pixel contributes a point iff `depth_m > 0`. Zero (and any negative)
            depth means "no return" and is dropped — depth dropout is one of the
            failure modes this assignment measures, so it must not become a point
            at the origin.

        Invariants
            * (height, width) == depth_m.shape == rgb.shape[:2]. The intrinsics
              describe THIS image; a mismatch is a caller bug, not something to
              paper over by falling back on the array shape.
            * len(points) == len(colors) == number of valid pixels.
            * Every returned point has z > 0 (a point with z <= 0 means the
              validity mask was not applied).
            * Points appear in row-major pixel order (row 0 left-to-right first).
            * N == 0 is legal: an all-invalid depth map yields an EMPTY cloud, not
              an exception.
            * Pure: `rgb` and `depth_m` must not be modified.

        Geometry
            A pinhole camera with square pixels, no skew and no distortion; the
            principal point is the image centre. The focal length in pixels follows
            from `width` and `hfov` alone, and the vertical focal length equals the
            horizontal one — so (width, height, hfov) fully determines the model.
            Do the projection yourself: Open3D's projection helpers
            (create_from_depth_image / create_from_rgbd_image,
            PinholeCameraIntrinsic, ...) are OFF-LIMITS here — the mapping from
            (u, v, depth) to (X, Y, Z) is the thing being learned.
            Reference: https://en.wikipedia.org/wiki/Pinhole_camera_model

        Smoke fixture: hw1/tests/fixtures/ ships five synthetic frames whose exact
        clouds are known. `expected.json -> cloud_stats` gives the point count,
        AABB and centroid of each full cloud, and `clouds.npz` holds the clouds
        themselves for a point-for-point comparison — so you can check this
        function on its own, before touching reconstruct(). See the README there.
    """
    #TODO


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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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
    Geometry-only ICP SLAM over the frames under `data_root`: estimate the camera
    trajectory (and optionally a global map) by chaining pairwise registrations of
    consecutive frames.

    CONTRACT

    Dataset layout
        data_root/rgb/<stem>.png     8-bit colour, one per frame
        data_root/depth/<stem>.png   depth, encoding handled by load_depth_meters
        data_root/GT_pose.npy        optional (M,7) ground-truth pose array
        data_root/intrinsics.json    {"width", "height", "hfov"} for THIS capture
        `<stem>` is an integer. An rgb file and a depth file with the same stem are
        the same frame. Default frame ORDER is ascending integer stem.

        The intrinsics belong to the capture, not to this file: read them from
        `data_root/intrinsics.json` here and pass them into
        depth_image_to_point_cloud. Never hardcode them, and never take them from a
        config — a capture must be reconstructable from its own directory alone.

    Inputs
        data_root  : str — the directory above.
        version    : str — which pairwise-registration backend to use. "open3d"
                     (default) selects the Open3D point-to-plane wrapper that ships
                     working in this file; it is the reference baseline any other
                     backend is compared against. The choice of backend does not
                     change anything else in this contract.
        voxel_size : float, METRES — the resolution clouds are reduced to before
                     registration. Affects accuracy and runtime, not the contract.
        verbose    : bool — progress printing only. Must NOT change the return value.
        build_cloud: bool — when True (default) the accumulated global map is built
                     and returned; when False the map is skipped and an EMPTY
                     PointCloud is returned instead. `pred_cam_pos` and `gt_poses`
                     are identical either way. The evaluator passes False because
                     only the trajectory is scored and the full map reaches ~1e8
                     points.
        robust     : bool, default True. Two guarantees are attached to True, and
                     they are the reason it is the default:
                       (1) DETERMINISM — two runs over the same data return the
                           same `pred_cam_pos`. No uncovered RNG anywhere in the
                           path. A score that moves between runs is not a score.
                       (2) The per-step gate below is enforced.
                     robust=False selects the plain feature-matching-then-ICP chain
                     and carries neither guarantee; it exists so the two can be
                     compared.
        gate_trans : float, METRES  — per-step plausibility bound on translation.
        gate_rot   : float, DEGREES — per-step plausibility bound on the geodesic
                     rotation angle. Both apply only when robust=True, where the
                     invariant is: EVERY relative transform actually applied to the
                     trajectory satisfies ||t|| <= gate_trans and angle <= gate_rot.
                     A single implausible pair must not be allowed to derail the
                     whole trajectory. Defaults (0.5 m / 30 deg) are generous
                     headroom over the ~0.09 m / ~6 deg per-step motion of the
                     provided captures.
        down_voxel : float or None, METRES — when set AND build_cloud, each frame's
                     cloud is reduced to this resolution before being merged into
                     the global map, so the map stays a workable size. Affects the
                     returned map only, never `pred_cam_pos`.
        frames     : optional list of integer frame stems — see SUBSETTING below.

    Returns
        (global_pcd, pred_cam_pos, gt_poses)
          global_pcd   : o3d.geometry.PointCloud — every frame's cloud expressed in
                         the FRAME-0 CAMERA frame and merged. Empty when
                         build_cloud=False.
          pred_cam_pos : (N,3) float64, METRES — estimated camera centres, also in
                         the FRAME-0 CAMERA frame (+X right, +Y down, +Z forward).
                         RAW: do not rotate, scale or align to the GT here —
                         mean_l2 owns the frame reconciliation, and doing it twice
                         is the classic way to produce a wrong score that looks
                         plausible.
          gt_poses     : (M,7) float — [x, y, z, qw, qx, qy, qz] as stored on disk,
                         or None when GT_pose.npy is absent (a capture with no GT
                         must still reconstruct).

    Invariants
        * Frame 0 anchors the world: pred_cam_pos[0] == (0, 0, 0) exactly.
        * len(pred_cam_pos) == the number of frames actually used. A frame whose
          rgb or depth fails to load is skipped, not faked.
        * No frames loadable  -> (empty PointCloud, zeros((0,3)), gt_poses). Not an
          exception.
        * GEOMETRY ONLY (D1). Colour must never enter registration — it is carried
          on the clouds for visualisation and nothing else. This is what makes the
          whole assignment legible: lighting reaches the score ONLY through the
          depth sensor's light coupling, so if colour leaked into registration the
          causal chain being measured would no longer be the one being claimed.
        * `data_root` is read-only; nothing is written.

    SUBSETTING (the `frames` argument / the --frames-csv path)
        hw1/reconstruct.py --frames-csv reads the integer `frame` column of a CSV
        emitted by `hw1/api.py retrieve` (an ontology/SPARQL selection of frames)
        and passes those stems here as `frames`.
          * frames=None (default): use every frame present under rgb/ and depth/,
            in ascending stem order — n = min(#rgb, #depth).
          * frames=[...]: use ONLY those stems, IN THE GIVEN ORDER, resolved as
            data_root/rgb/<stem>.png and data_root/depth/<stem>.png. The entire
            pipeline runs over that reduced sequence — "consecutive" means
            consecutive IN THE SUBSET, not in the original capture.
          * GT is subset by the SAME stems, so gt_poses[i] corresponds to
            pred_cam_pos[i] and mean_l2 stays meaningful. (_load_gt does this.)
          * Expected consequence, not a bug: dropping interior frames WIDENS the
            motion between the frames that remain, while the per-step gate above is
            sized for consecutive frames. A sparse selection therefore scores worse
            for reasons that have nothing to do with the quality of the frames it
            kept. This trade-off is the point of phase 1 — a filter that throws away
            too much is as wrong as one that keeps bad frames.

    Reference: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping

    Smoke fixture: hw1/tests/fixtures/ ships five synthetic frames with exactly
    known per-step transforms and a GT trajectory whose perfect mean_l2 is 0.0.
    Run this function on that directory before running it on real data — it tells
    "my ICP is wrong" apart from "my loop is wrong". See the README there.
    """
    #TODO


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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
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

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    y_min = pts[:, 1].min()                    # ceiling is at min in camera frame
    mask  = pts[:, 1] > (y_min + margin)
    out   = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    out.colors = o3d.utility.Vector3dVector(cols[mask])
    return out
