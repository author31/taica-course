import os
import re
import glob
import numpy as np
import cv2
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# Depth decoding. Two layouts are supported so this works regardless of how the
# collector wrote depth (see scripts/load.py):
#   * 16-bit PNG  -> raw millimeters,   depth_m = px / DEPTH_SCALE
#   * 8-bit PNG   -> normalized-to-range preview (depth_to_vis in load.py did
#                    px = clip(depth_m / max_range, 0, 1) * 255),
#                    so depth_m = px / 255 * DEPTH_MAX_RANGE
# DEPTH_MAX_RANGE must match depth.max_range in scripts/config.yaml.
DEPTH_SCALE = 1000.0
DEPTH_MAX_RANGE = 10.0


def _frame_index(path):
    """Numeric key so 2.png sorts before 10.png (load.py names frames 1..N)."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_depth_meters(depth_path):
    """Read a depth PNG and return a float32 (H,W) map in meters."""
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(depth_path)
    if raw.ndim == 3:                       # load.py saves a 3-channel gray preview
        raw = raw[:, :, 0]
    if raw.dtype == np.uint16:              # true metric depth in mm
        return raw.astype(np.float32) / DEPTH_SCALE
    # 8-bit normalized preview -> back to meters against the known max range.
    return raw.astype(np.float32) / 255.0 * DEPTH_MAX_RANGE


def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Inputs to numpy (rgb HxWx3 uint8/RGB, depth HxW float32 meters).
    rgb = np.asarray(rgb_image)
    depth = np.asarray(depth_image, dtype=np.float32)
    h, w = depth.shape[:2]

    # 2. Pixel coordinate grid.
    us, vs = np.meshgrid(np.arange(w), np.arange(h))

    # 3. Valid pixels only: drop 0 (no return) and out-of-range readings.
    valid = (depth > 0.0) & (depth < DEPTH_MAX_RANGE)
    z = depth[valid]
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)

    # Pinhole unprojection in the Habitat camera frame (+X right, +Y up, looking
    # down -Z). Image v grows downward, so Y is negated; the camera looks along
    # -Z, so Z is negated.
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    points_3d = np.stack([x, -y, -z], axis=1)

    colors_norm = rgb[valid][:, :3].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normals are required for Point-to-Plane ICP and for FPFH.
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Compute FPFH features for Global Registration [cite: 30]
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def global_registration(source_down, source_fpfh, target_down, target_fpfh, voxel_size):
    """RANSAC feature matching -> coarse initial alignment [cite: 30]."""
    distance_threshold = voxel_size * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )


def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform, threshold, max_iter=60):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%)
    Point-to-Plane ICP solved by linearizing the rotation each iteration.
    """
    src = deepcopy(source_pcd)
    T_global = initial_transform.copy()
    src.transform(T_global)

    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    t_pts = np.asarray(target_pcd.points)
    t_nrm = np.asarray(target_pcd.normals)
    if len(t_nrm) == 0:                     # need normals for point-to-plane
        target_pcd.estimate_normals()
        t_nrm = np.asarray(target_pcd.normals)

    prev_rmse = None
    for _ in range(max_iter):
        s_pts = np.asarray(src.points)

        # 1. Nearest-neighbor correspondences within the gating threshold.
        tgt_idx = np.full(len(s_pts), -1)
        for j, p in enumerate(s_pts):
            _, idx, d2 = target_tree.search_knn_vector_3d(p, 1)
            if d2[0] <= threshold * threshold:
                tgt_idx[j] = idx[0]
        keep = tgt_idx >= 0
        if keep.sum() < 6:
            break
        sc = s_pts[keep]
        tc = t_pts[tgt_idx[keep]]
        nc = t_nrm[tgt_idx[keep]]

        # 2. Linear system (A^T A) x = A^T b for x = [rx, ry, rz, tx, ty, tz].
        #    Point-to-plane residual: ((s - t) . n) minimized.
        A = np.zeros((len(sc), 6))
        A[:, 0] = nc[:, 2] * sc[:, 1] - nc[:, 1] * sc[:, 2]
        A[:, 1] = nc[:, 0] * sc[:, 2] - nc[:, 2] * sc[:, 0]
        A[:, 2] = nc[:, 1] * sc[:, 0] - nc[:, 0] * sc[:, 1]
        A[:, 3:6] = nc
        b = -np.sum((sc - tc) * nc, axis=1)

        # 3. Solve, build the incremental transform, update T_global.
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        dT = np.eye(4)
        dT[:3, :3] = R.from_euler("xyz", x[:3]).as_matrix()
        dT[:3, 3] = x[3:6]
        T_global = dT @ T_global
        src.transform(dT)

        rmse = np.sqrt(np.mean(b * b))
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    return o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def _trajectory_lineset(poses, color):
    """LineSet connecting consecutive camera centers for visualization."""
    pts = np.array([p[:3, 3] for p in poses])
    ls = o3d.geometry.LineSet()
    if len(pts) < 2:
        return ls
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    geoms = [reconstructed_pcd]

    # 1./2. Estimated trajectory (red) and ground-truth trajectory (black).
    pred = np.asarray(predicted_cam_poses)
    geoms.append(_trajectory_lineset(pred, [1.0, 0.0, 0.0]))
    if gt_poses is not None and len(gt_poses) > 0:
        geoms.append(_trajectory_lineset(gt_poses, [0.0, 0.0, 0.0]))

    # Mean L2 distance between predicted and GT camera centers [cite: 38].
    mean_l2_error = 0.0
    if gt_poses is not None and len(gt_poses) > 0:
        n = min(len(pred), len(gt_poses))
        pred_c = np.array([p[:3, 3] for p in pred[:n]])
        gt_c = np.array([g[:3, 3] for g in gt_poses[:n]])
        mean_l2_error = float(np.mean(np.linalg.norm(pred_c - gt_c, axis=1)))

    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # 3. Visualization
    o3d.visualization.draw_geometries(
        geoms, window_name=f"Floor {args.floor} Reconstruction")
    return mean_l2_error


def remove_ceiling(pcd, ratio=0.1):
    """Post-processing: drop the top `ratio` of points by height (Habitat +Y up)
    to remove the ceiling so the floor plan is visible [cite: 37]."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    cutoff = np.quantile(pts[:, 1], 1.0 - ratio)
    keep = pts[:, 1] < cutoff
    return pcd.select_by_index(np.where(keep)[0])


def reconstruct(args):
    voxel_size = 0.25
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    # Numeric sort so frame order matches capture order (and GT_pose rows).
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")), key=_frame_index)
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")), key=_frame_index)

    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    # Anchor the estimated trajectory to the GT start pose so both live in the
    # same world frame (reconstruction is only recoverable up to that pose).
    if len(gt_poses) > 0:
        camera_poses = [gt_poses[0].copy()]
    else:
        camera_poses = [np.eye(4)]

    # Per-frame full-resolution clouds (kept for accumulation into the map).
    frames = []
    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1]   # BGR -> RGB
        depth_m = load_depth_meters(depth_path)
        frames.append(depth_image_to_point_cloud(rgb, depth_m))

    if not frames:
        print(f"no frames under {rgb_dir}")
        return o3d.geometry.PointCloud(), camera_poses, gt_poses

    accumulated_pcd = deepcopy(frames[0]).transform(camera_poses[0])
    prev_down, prev_fpfh = preprocess_point_cloud(frames[0], voxel_size)

    # Reconstruction Loop [cite: 29-30]
    for i in range(1, len(frames)):
        print(f"Processing Frame {i}...")
        # 1. (already unprojected)  2. Preprocess (voxel / normals / FPFH).
        src_down, src_fpfh = preprocess_point_cloud(frames[i], voxel_size)

        # 3. Global registration (RANSAC) for a coarse init.
        ransac = global_registration(
            src_down, src_fpfh, prev_down, prev_fpfh, voxel_size)

        # 4. Local registration (ICP) refines source(i) -> target(i-1).
        icp_thresh = voxel_size * 0.4
        if args.version == "my_icp":
            icp = my_local_icp_algorithm(
                src_down, prev_down, ransac.transformation, icp_thresh)
        else:
            icp = local_icp_algorithm(
                src_down, prev_down, ransac.transformation, icp_thresh)

        # 5. Chain to a global pose, accumulate the transformed cloud.
        global_pose = camera_poses[i - 1] @ icp.transformation
        camera_poses.append(global_pose)
        accumulated_pcd += deepcopy(frames[i]).transform(global_pose)

        prev_down, prev_fpfh = src_down, src_fpfh

    # Keep the map manageable, then drop the ceiling [cite: 37].
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size * 0.5)
    accumulated_pcd = remove_ceiling(accumulated_pcd)

    return accumulated_pcd, camera_poses, gt_poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)

    print(f"Total execution time: {time.time() - start_time:.2f}s") #
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
