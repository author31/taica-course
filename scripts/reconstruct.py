#!/usr/bin/env python3
# Copyright (c) 2025. HW1 — Camera Pose Estimation in IsaacLab.
# SPDX-License-Identifier: BSD-3-Clause
"""HW1 — Offline reconstruction from a saved keyframe buffer.

This is the *geometric* half of the HW1 pipeline, split out of ``scripts/hw1.py``
so it runs on plain numpy + Open3D, with **no** IsaacLab / Omniverse dependency.

Input
-----
A ``keyframes.npz`` produced by ``scripts/hw1.py`` (press ``F`` while driving),
containing per-keyframe stacked arrays:

    rgb         (N, H, W, 3)   uint8
    depth       (N, H, W)      float32   metric z-buffer, camera frame
    intrinsics  (N, 3, 3)      float32
    pos         (N, 3)         float32   camera world position  [x, y, z]
    quat        (N, 4)         float32   camera world orientation  [qw, qx, qy, qz]

Pipeline (per captured keyframe)
--------------------------------
    depth            -> point-cloud unprojection via Open3D
                     (o3d.geometry.PointCloud.create_from_depth_image)
    -> voxel downsample / normals / FPFH          (Open3D)
    -> RANSAC global registration                  (Open3D)
    -> ICP local refinement                        (Open3D)
    -> accumulate into a single world map

The chained pairwise registration yields an *estimated* camera trajectory which
is compared against the ground-truth poses stored in the npz.

Outputs (written to <out>, default = the npz's directory)
---------------------------------------------------------
    reconstruction.ply    accumulated world point cloud
    trajectory.png        2D matplotlib plot: GT (black) vs estimated (red)
    trajectory_eval.npy   per-frame GT/est positions + translation error
    reconstruction.png    3D Open3D render: map + GT + estimated trajectories

Run
---
    python scripts/reconstruct.py outputs/hw1/keyframes.npz
    python scripts/reconstruct.py keyframes.npz --voxel 0.05 --max_depth 8.0 -o outputs/hw1
    python scripts/reconstruct.py outputs/hw1/keyframes.npz --show   # interactive 3D
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Pose loading
# =============================================================================
def load_gt_poses(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build per-keyframe 4x4 camera-to-world matrices from the saved poses.

    Each keyframe contributes a 7-vector ``[x, y, z, qw, qx, qy, qz]``. We form a
    4x4 homogeneous transform from the translation ``[x, y, z]`` and a rotation
    built with ``scipy.spatial.transform.Rotation.from_quat``. scipy expects the
    scalar-last ``[qx, qy, qz, qw]`` (xyzw) convention, so we re-order the stored
    ``[qw, qx, qy, qz]`` before passing it in.

    Args:
        pos:  (N, 3) world translations.
        quat: (N, 4) world orientations stored as ``[qw, qx, qy, qz]``.

    Returns:
        (N, 4, 4) homogeneous camera-to-world transforms.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    n = pos.shape[0]
    assert quat.shape[0] == n, f"pos/quat length mismatch: {n} vs {quat.shape[0]}"

    # Stored as [qw, qx, qy, qz]; scipy wants [qx, qy, qz, qw].
    scipy_quat = quat[:, [1, 2, 3, 0]]
    rots = R.from_quat(scipy_quat).as_matrix()  # (N, 3, 3)

    gt_poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    gt_poses[:, :3, :3] = rots
    gt_poses[:, :3, 3] = pos
    return gt_poses


# =============================================================================
# Per-frame pipeline (Open3D)
# =============================================================================
def frame_to_cloud(depth: np.ndarray, intrinsics: np.ndarray, voxel: float, max_depth: float):
    """depth -> Open3D unprojection -> voxel downsample -> normals -> FPFH.

    Points stay in the CAMERA frame (no GT pose is leaked in), which is what
    pairwise registration needs. Returns (downsampled PointCloud, FPFH feature).
    """
    import open3d as o3d

    h, w = depth.shape
    # Open3D expects a float32 depth image with the metric z-buffer in metres.
    depth_img = o3d.geometry.Image(depth.astype(np.float32))

    # Pinhole intrinsics from the saved 3x3 matrix.
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # Unproject. depth_scale=1.0 because `depth` is already metric (metres),
    # depth_trunc clips points beyond max_depth. project_valid_depth_only=True
    # drops the zero/invalid pixels Open3D treats as no-depth.
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img, intrinsic, depth_scale=1.0, depth_trunc=max_depth,
        project_valid_depth_only=True,
    )

    # Drop any remaining invalid / non-finite points, then downsample + feature.
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )
    return pcd, fpfh


def register_pair(src, src_f, dst, dst_f, voxel: float) -> np.ndarray:
    """RANSAC global registration (src -> dst) refined by point-to-plane ICP.

    Returns the 4x4 transform T mapping source points into the destination frame.
    """
    import open3d as o3d

    dist = voxel * 1.5
    # Global registration: FPFH feature matching with RANSAC for a coarse alignment.
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_f, dst_f,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    # Local refinement: point-to-plane ICP seeded with the RANSAC transform.
    icp = o3d.pipelines.registration.registration_icp(
        src, dst, dist, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return np.asarray(icp.transformation)


def reconstruct(keyframes: dict, voxel: float, max_depth: float, out_dir: str,
                show: bool = False) -> None:
    """Chain pairwise registration over keyframes, accumulate a world map, report vs GT.

    ``keyframes`` is the dict unpacked from the saved ``keyframes.npz``.
    ``show`` opens an interactive Open3D window overlaying the map + trajectories.
    """
    import open3d as o3d

    n = keyframes["depth"].shape[0]
    if n < 2:
        print(f"[reconstruct] need >= 2 keyframes to reconstruct (have {n}).")
        return

    # 1. Pose load: build ground-truth camera-to-world 4x4 matrices.
    gt_poses = load_gt_poses(keyframes["pos"], keyframes["quat"])

    print(f"[reconstruct] {n} keyframes, voxel={voxel} m, max_depth={max_depth} m.")

    est_T_world_cam: list[np.ndarray] = []
    global_map = o3d.geometry.PointCloud()
    clouds: list = []   # downsampled camera-frame Open3D clouds, kept for pairwise reg.
    feats: list = []    # matching FPFH features, aligned with `clouds`.

    # 2. Per-frame loop: unprojection -> voxel/FPFH/normals -> RANSAC -> ICP -> accumulate.
    for i in range(n):
        depth = keyframes["depth"][i]
        intr = keyframes["intrinsics"][i]

        # (a) depth -> point-cloud unprojection  +  (b) voxel downsample / normals / FPFH.
        pcd, fpfh = frame_to_cloud(depth, intr, voxel, max_depth)
        clouds.append(pcd)
        feats.append(fpfh)

        # (c) global + (d) local registration against the previous keyframe.
        if i == 0:
            # Frame 0 defines the world frame.
            est_T_world_cam.append(np.eye(4))
            global_map += pcd
        else:
            # Register frame i onto frame i-1, then compose into the world frame.
            T_prev_cur = register_pair(pcd, fpfh, clouds[i - 1], feats[i - 1], voxel)
            T_world_cur = est_T_world_cam[i - 1] @ T_prev_cur
            est_T_world_cam.append(T_world_cur)
            global_map += o3d.geometry.PointCloud(pcd).transform(T_world_cur)
            global_map = global_map.voxel_down_sample(voxel)

        print(f"[reconstruct]   frame {i}: {len(pcd.points)} downsampled pts  "
              f"gt=({gt_poses[i,0,3]:+.2f},{gt_poses[i,1,3]:+.2f},{gt_poses[i,2,3]:+.2f})")

    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, "reconstruction.ply")
    o3d.io.write_point_cloud(map_path, global_map)
    print(f"[reconstruct] saved world map: {map_path}  ({len(global_map.points)} pts)")

    # 3. Report estimated-vs-GT trajectory and plot it (2D matplotlib figure).
    gt_rel, est_rel = report_trajectory(gt_poses, np.stack(est_T_world_cam), out_dir)

    # 4. Visualize the reconstructed map together with both trajectories (Open3D).
    #    The map + estimated poses live in the frame-0 camera frame, so GT is shown
    #    relative to frame 0 as well, putting all three in one consistent frame.
    visualize_reconstruction(global_map, gt_rel, est_rel, out_dir, show=show)


# =============================================================================
# Reporting / plotting
# =============================================================================
def report_trajectory(gt_poses: np.ndarray, est_T: np.ndarray, out_dir: str):
    """Print + save estimated-vs-GT camera drift, then plot the two trajectories.

    Poses are expressed relative to frame 0 (so GT and estimated share an origin).
    Returns ``(gt_rel, est_rel)`` as (N, 4, 4) arrays for downstream 3D viz.
    """
    gt0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = gt0_inv @ gt_poses              # (N, 4, 4)
    est_rel = est_T                          # already relative to frame 0

    rows = []
    print(f"[reconstruct] {'idx':>3} {'gt_xyz':>26} {'est_xyz':>26} {'err[m]':>8}")
    for i, (gt, est) in enumerate(zip(gt_rel, est_rel)):
        gt_p, est_p = gt[:3, 3], est[:3, 3]
        err = float(np.linalg.norm(gt_p - est_p))
        rows.append([i, *gt_p, *est_p, err])
        print(
            f"[reconstruct] {i:>3} ({gt_p[0]:+.2f},{gt_p[1]:+.2f},{gt_p[2]:+.2f})"
            f"      ({est_p[0]:+.2f},{est_p[1]:+.2f},{est_p[2]:+.2f})   {err:>6.3f}"
        )
    arr = np.asarray(rows, dtype=np.float64)
    if len(arr):
        print(f"[reconstruct] mean translation error vs GT: {arr[:, -1].mean():.3f} m")
        np.save(os.path.join(out_dir, "trajectory_eval.npy"), arr)

    plot_trajectories(gt_rel, est_rel, out_dir)
    return gt_rel, est_rel


def plot_trajectories(gt_T: np.ndarray, est_T: np.ndarray, out_dir: str) -> None:
    """3D plot: ground-truth camera poses in black, estimated in red -> <out>/trajectory.png."""
    try:
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    except ImportError:
        print("[reconstruct] matplotlib not installed — skipping plot (`pip install matplotlib`).")
        return

    gt_p = gt_T[:, :3, 3]
    est_p = est_T[:, :3, 3]
    # Camera forward axis is +Z in the ROS optical frame; scale arrows to path extent.
    span = float(np.linalg.norm(gt_p.max(axis=0) - gt_p.min(axis=0)))
    alen = max(span * 0.08, 0.1)
    gt_fwd = gt_T[:, :3, 2] * alen
    est_fwd = est_T[:, :3, 2] * alen

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2], "-o", color="black", ms=4, lw=1.5, label="ground truth")
    ax.plot(est_p[:, 0], est_p[:, 1], est_p[:, 2], "-o", color="red", ms=4, lw=1.5, label="estimated")
    ax.quiver(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2], gt_fwd[:, 0], gt_fwd[:, 1], gt_fwd[:, 2],
              color="black", linewidth=1.0, arrow_length_ratio=0.3)
    ax.quiver(est_p[:, 0], est_p[:, 1], est_p[:, 2], est_fwd[:, 0], est_fwd[:, 1], est_fwd[:, 2],
              color="red", linewidth=1.0, arrow_length_ratio=0.3)
    ax.scatter(*gt_p[0], color="black", s=80, marker="*")
    for g, e in zip(gt_p, est_p):
        ax.plot([g[0], e[0]], [g[1], e[1]], [g[2], e[2]], color="0.6", lw=0.5)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Camera pose: ground truth (black) vs estimated (red)")
    ax.legend(loc="upper left")

    allp = np.concatenate([gt_p, est_p], axis=0)
    ctr = allp.mean(axis=0)
    rng = max(float((allp.max(axis=0) - allp.min(axis=0)).max()) * 0.5, 0.5)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "trajectory.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"[reconstruct] saved trajectory comparison plot: {png}")


# =============================================================================
# 3D visualization (Open3D): reconstructed map + both trajectories
# =============================================================================
def _trajectory_geometry(poses: np.ndarray, color: list[float], scale: float) -> list:
    """Build Open3D geometries for one trajectory: a polyline through the camera
    centers, a sphere at each pose, and a small coordinate frame showing heading."""
    import open3d as o3d

    geoms: list = []
    pts = poses[:, :3, 3]

    # Polyline connecting consecutive camera centers.
    if len(pts) >= 2:
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts.astype(np.float64)),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geoms.append(ls)

    # A marker sphere + a coordinate frame (orientation) at every keyframe pose.
    for T in poses:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.35)
        sphere.translate(T[:3, 3])
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geoms.append(sphere)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        geoms.append(frame.transform(T.copy()))

    return geoms


def visualize_reconstruction(global_map, gt_rel: np.ndarray, est_rel: np.ndarray,
                             out_dir: str, show: bool = False) -> None:
    """Overlay the reconstructed map with the GT (black) and estimated (red)
    trajectories in one Open3D scene. Always tries to save an offscreen PNG;
    opens an interactive window too when ``show`` is set.
    """
    import open3d as o3d

    # Reconstructed map (depth-only => no colour); paint a neutral grey.
    world = o3d.geometry.PointCloud(global_map)
    if not world.has_colors():
        world.paint_uniform_color([0.6, 0.6, 0.6])

    # Marker/frame size scaled to the trajectory extent.
    gt_p = gt_rel[:, :3, 3]
    span = float(np.linalg.norm(gt_p.max(axis=0) - gt_p.min(axis=0))) if len(gt_p) > 1 else 1.0
    scale = max(span * 0.04, 0.05)

    geoms = [world]
    geoms += _trajectory_geometry(gt_rel, [0.0, 0.0, 0.0], scale)    # ground truth: black
    geoms += _trajectory_geometry(est_rel, [1.0, 0.0, 0.0], scale)   # estimated:   red
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 3.0))

    png = os.path.join(out_dir, "reconstruction.png")
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="HW1 reconstruction", width=1280, height=960, visible=show)
        for g in geoms:
            vis.add_geometry(g)
        vis.get_render_option().point_size = 2.0
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(png, do_render=True)
        print(f"[reconstruct] saved 3D map+trajectory render: {png}")
        if show:
            print("[reconstruct] showing interactive Open3D window — close it to continue.")
            vis.run()
        vis.destroy_window()
    except Exception as e:  # headless without a GL context, etc.
        print(f"[reconstruct] Open3D rendering unavailable ({e}). "
              f"Map saved as reconstruction.ply — open it in any point-cloud viewer.")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="HW1 — offline reconstruction from keyframes.npz.")
    parser.add_argument("npz", type=str, help="Path to keyframes.npz saved by scripts/hw1.py.")
    parser.add_argument("--voxel", type=float, default=0.05, help="Reconstruction voxel size [m].")
    parser.add_argument("--max_depth", type=float, default=8.0, help="Drop depth beyond this [m].")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output directory. Defaults to the npz's directory.")
    parser.add_argument("--show", dest="show", action="store_true", default=None,
                        help="Open an interactive Open3D window (default: on iff $DISPLAY is set).")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Never open an interactive window (still saves the render PNG).")
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(args.npz)

    # Default: show an interactive window only when a display is available.
    show = bool(os.environ.get("DISPLAY")) if args.show is None else args.show

    out_dir = args.out if args.out is not None else os.path.dirname(os.path.abspath(args.npz))
    keyframes = np.load(args.npz)
    reconstruct(keyframes, args.voxel, args.max_depth, out_dir, show=show)


if __name__ == "__main__":
    main()
