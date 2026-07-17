import numpy as np
import open3d as o3d
import argparse
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

def depth_image_to_point_cloud(rgb, depth):
    """
    Convert an RGB image and a raw depth image into a colored 3-D point cloud
    using the pinhole camera model.  No Open3D projection utilities are used.
 
    Args:
        rgb   : H*W*3  uint8  BGR image (as loaded by cv2).
        depth : H*W    numeric array.
             • uint16 → values are millimetres  (depth_scale = 1000)
             • uint8  → Habitat's visualisation encoding
                        (saved as (d_metres / 10 * 255).astype(uint8))
 
    Returns:
        o3d.geometry.PointCloud with XYZ positions and RGB colors.
    """
    # TODO: Get point cloud from rgb and depth image 
    h, w = depth.shape

    # ── Validity mask ────────────────────────────────────────────────────────
    valid = depth > 0
    depth_v = depth[valid]
    rgb_v = rgb[valid]

    # ── Convert raw depth to metres ──────────────────────────────────────────
    if depth_v.dtype == np.uint16:
        Z = depth_v.astype(np.float64) / DEPTH_SCALE          # mm → m
    else:
        Z = depth_v.astype(np.float64) / 255.0 * 10.0         # → m
 
    # ── Pixel grids ──────────────────────────────────────────────────────────
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_v = u_grid[valid]
    v_v = v_grid[valid]
 
    # ── Back-projection  (pinhole inverse) ──────────────────────────────────
    X = (u_v - cx) * Z / fx
    Y = (v_v - cy) * Z / fy
 
    points = np.column_stack([X, Y, Z]) # N * 3
 
    # ── Colors (BGR → RGB, normalised) ─────────────────────────────────────
    colors = rgb_v.astype(np.float64) / 255.0
    colors = colors[:, ::-1]                                 # BGR → RGB
 
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
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
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


def execute_global_registration(source_down, target_down, source_fpfh,
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
    # TODO: Use Open3D ICP function to implement
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


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom iterative closest-point algorithm.
 
    Algorithm:
      1. Transform source by current estimate T.
      2. Find nearest neighbours in target via cKDTree (fast).
      3. Filter correspondences by distance threshold.
      4. Compute optimal R, t via SVD on centred correspondences.
      5. Update T and repeat until convergence.
 
    Optimisations
    ─────────────
    • scipy cKDTree (C extension) for O(N log N) nearest-neighbour search.
    • Correspondence distance threshold to reject outliers.
    • Early stopping when mean-error improvement < tolerance.
    • Reflection check on SVD result (handles degenerate cases).
 
    Returns:
        A result object with attribute `.transformation` (4*4 np.ndarray).
    """
    # TODO: Write your own ICP function
    threshold    = voxel_size * 1.5
    max_iter     = 60
    tolerance    = 1e-6
 
    src = np.asarray(source_down.points, dtype=np.float64)   # N*3
    tgt = np.asarray(target_down.points, dtype=np.float64)   # M*3
 
    T = trans_init.copy().astype(np.float64)
 
    # 建立 Target 的 KDTree 以加速最近鄰搜尋
    tree = cKDTree(tgt)
 
    prev_err = np.inf
 
    for _it in range(max_iter):
        # ── 1. Transform source ───────────────────────────────────────────
        R_cur  = T[:3, :3]
        t_cur  = T[:3, 3]
        src_t  = (R_cur @ src.T).T + t_cur          # N*3
 
        # ── 2. Nearest-neighbour search ───────────────────────────────────
        dists, idx = tree.query(src_t, k=1, workers=1)
 
        # ── 3. Correspondence filtering ───────────────────────────────────
        mask    = dists < threshold
        if mask.sum() < 10:
            break
 
        P = src_t[mask]                              # matched source  N'*3
        Q = tgt[idx[mask]]                           # matched target  N'*3
 
        # ── 4. SVD-based optimal alignment ───────────────────────────────
        p_bar = P.mean(axis=0)
        q_bar = Q.mean(axis=0)
 
        Pc = P - p_bar
        Qc = Q - q_bar
 
        H        = Pc.T @ Qc                         # 3*3 covariance
        U, _, Vt = np.linalg.svd(H)
        R_delta  = Vt.T @ U.T
 
        # Handle reflection (det should be +1)
        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T
 
        t_delta = q_bar - R_delta @ p_bar
 
        # ── 5. Update cumulative transform ────────────────────────────────
        T_delta          = np.eye(4)
        T_delta[:3, :3]  = R_delta
        T_delta[:3,  3]  = t_delta
        T                = T_delta @ T
 
        # ── Convergence check ─────────────────────────────────────────────
        mean_err = dists[mask].mean()
        if abs(prev_err - mean_err) < tolerance:
            break
        prev_err = mean_err
 
    # Return a duck-typed result compatible with Open3D's result object
    class _Result:
        def __init__(self, transformation):
            self.transformation = transformation
 
    return _Result(T)


def reconstruct(args):
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    """
    Main pipeline:
      1. Load RGB + depth frames from data_root.
      2. Unproject each depth frame → local point cloud.
      3. Pairwise registration (RANSAC + ICP) between consecutive frames.
      4. Accumulate all clouds into a global map.
      5. Return (global_pcd, estimated_camera_positions).
 
    Returns:
        result_pcd   : o3d.geometry.PointCloud  - global map
        pred_cam_pos : np.ndarray (N*3)          - estimated camera XYZ
    """
    # TODO: Return results
    rgb_dir   = os.path.join(args.data_root, 'rgb')
    depth_dir = os.path.join(args.data_root, 'depth')
 
    # ── Collect sorted frame paths ──────────────────────────────────────────
    def sorted_frames(directory):
        files = [f for f in os.listdir(directory) if f.endswith('.png')]
        return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
 
    rgb_files   = sorted_frames(rgb_dir)
    depth_files = sorted_frames(depth_dir)
    n_frames    = min(len(rgb_files), len(depth_files))
    print(f"[reconstruct] Found {n_frames} frames  |  version = {args.version}")
 
    voxel_size  = 0.05   # 5 cm - good balance between speed and quality
 
    # ── Load all point clouds ───────────────────────────────────────────────
    print("[reconstruct] Unprojecting depth images …")
    pcds = []
    for i in range(n_frames):
        rgb   = cv2.imread(os.path.join(rgb_dir,   rgb_files[i]))
        depth = cv2.imread(os.path.join(depth_dir, depth_files[i]),
                           cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"  Warning: could not load frame {i}, skipping.")
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]          # pick one channel
        pcds.append(depth_image_to_point_cloud(rgb, depth))
 
    n = len(pcds)
    print(f"[reconstruct] Loaded {n} point clouds.")
 
    # ── Sequential pairwise registration ────────────────────────────────────
    # T_global[i] transforms frame-i camera coordinates → world frame (frame-0)
    T_global      = [np.eye(4)]
    global_pcd    = pcds[0]                  # frame-0 is the reference
    all_pcds       = [global_pcd]              # for optional visualisation of intermediate clouds
    pred_cam_pos  = [np.zeros(3)]
 
    for i in range(1, n):
        print(f"  Frame {i:>4d}/{n-1}  ", end='', flush=True)
        t0 = time.time()
 
        source = pcds[i]
        target = pcds[i - 1]
 
        # Downsample + FPFH
        src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
        tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)
 
        # Global registration (RANSAC)
        result_ransac = execute_global_registration(
            src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
 
        # Local refinement (ICP)
        icp_thr = voxel_size * 1.5
        if args.version == 'open3d':
            result_icp = local_icp_algorithm(
                src_down, tgt_down, result_ransac.transformation, icp_thr)
            T_rel = result_icp.transformation
        else:   # my_icp
            result_icp = my_local_icp_algorithm(
                src_down, tgt_down, result_ransac.transformation, voxel_size)
            T_rel = result_icp.transformation
 
        # Chain: world ← prev_world ← current
        T_i = T_global[-1] @ T_rel
        T_global.append(T_i)
 
        # Camera position = translation of T_i (origin of cam-i in world)
        pred_cam_pos.append(T_i[:3, 3])
 
        # Accumulate cloud
        pcd_i_world = copy.deepcopy(pcds[i])
        pcd_i_world.transform(T_i) 
        all_pcds.append(pcd_i_world)
 
        print(f"| dt = {time.time()-t0:.2f}s")
 
    # ── Return global point clouds and estimated camera poses ────────────────────────────────────────
    global_pcd = o3d.geometry.PointCloud()
    pred_cam_pos = np.array(pred_cam_pos, dtype=np.float64)

    for pcd in all_pcds:
        global_pcd += pcd

    return global_pcd, pred_cam_pos

# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════
 
def make_trajectory(positions: np.ndarray, color) -> o3d.geometry.LineSet:
    """Create an Open3D LineSet from a sequence of XYZ camera positions."""
    lines   = [[i, i + 1] for i in range(len(positions) - 1)]
    ls      = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls
 
 
def remove_ceiling(pcd, margin=0.3):
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    y_min = pts[:, 1].min()                    # ← 相機座標系的天花板在 min
    mask  = pts[:, 1] > (y_min + margin)       # ← 移除 y_min 附近的點
    out   = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    out.colors = o3d.utility.Vector3dVector(cols[mask])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "sample_data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "sample_data_collection/second_floor/"

    start_time = time.time()
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct(args)

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    # ── Load ground-truth poses ─────────────────────────────────────────────
    gt_path  = os.path.join(args.data_root, 'GT_pose.npy')
    gt_poses = np.load(gt_path)              # N*7: [x, y, z, qw, qx, qy, qz]
    gt_cam_pos = gt_poses[:, :3]             # N*3: XYZ positions

    # ── 把 GT 轉到和 pred 相同的顯示座標系 ──────────────────────────────
    # pred 已做 Y flip → 顯示空間是 [X, -Y_cam, Z_cam]
    #                              ≈ [X_world, Y_world, -Z_world]
    # 所以 GT 的 Z 也要翻轉才能對齊
    gt_cam_pos_display = gt_cam_pos.copy()
    gt_cam_pos_display[:, 2] *= -1          # ← flip Z

 
    # ── Align estimated trajectory origin to GT origin ──────────────────────
    # (ICP works in frame-0 camera space; shift so the two start at the same
    #  world position for a fair visual comparison.)
    # 翻轉 Y 軸，從相機座標系轉換到 Habitat 世界座標系
    pred_cam_pos_world = pred_cam_pos.copy()
    pred_cam_pos_world[:, 1] *= -1              # ← 加這行
    n_min = min(len(pred_cam_pos_world), len(gt_cam_pos_display))
    offset = gt_cam_pos_display[0] - pred_cam_pos_world[0]
    pred_cam_aligned = pred_cam_pos_world[:n_min] + offset
 
    # ── Mean L2 distance ────────────────────────────────────────────────────
    l2 = np.linalg.norm(pred_cam_aligned - gt_cam_pos_display[:n_min], axis=1)
    print("Mean L2 distance: ", f"{l2.mean():.4f} m  (over {n_min} frames)")
    print("Execution time: ", f"{time.time() - start_time:.2f} seconds")


    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # ── Remove ceiling ───────────────────────────────────────────────────────
    scene_no_ceil = remove_ceiling(result_pcd, margin=1.0)
    # ── Build trajectory line sets ──────────────────────────────────────────
    traj_pred = make_trajectory(pred_cam_aligned,       [1.0, 0.0, 0.0])  # red
    traj_gt   = make_trajectory(gt_cam_pos_display[:n_min], [0.0, 0.0, 0.0])  # black
 
    # Shift scene to align with GT coordinate frame
    scene_no_ceil.translate(offset)
 
    # ── Open3D visualisation ────────────────────────────────────────────────
    print("[main] Opening visualiser …  (press Q to quit)")
    o3d.visualization.draw_geometries(
        [scene_no_ceil, traj_pred, traj_gt],
        window_name=f'HW2 Reconstruction - Floor {args.floor} ({args.version})',
        zoom=0.5,
        front=[0, -1, 0],   # 相機朝 -Y 方向看（從上往下）
        lookat=[0, 0, 0],   # 看向原點
        up=[0, 0, -1]       # Z 軸朝上（畫面上方）
    )
