"""
Thin CLI over hw1/utils.py: reconstruct ONE floor's captured run (geometry-only
ICP SLAM), print the mean L2 vs ground truth, and open an Open3D window with the
reconstructed cloud + estimated (red) and GT (black) trajectories.

The heavy lifting lives in utils.py so the evaluator can run headless. This file
is the interactive/visual entry point only.

    pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/baseline/
    pixi run -e habitat python hw1/reconstruct.py --floor 1 --version open3d
"""
import os
import sys
import time
import argparse

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d',
                        help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default=None,
                        help='explicit capture dir (overrides --floor default)')
    parser.add_argument('--no-vis', action='store_true',
                        help='skip the Open3D window (print metric only)')
    args = parser.parse_args()

    data_root = args.data_root
    if data_root is None:
        data_root = ("eval/_data/first_floor/baseline" if args.floor == 1
                     else "sample_data_collection/second_floor/")

    t0 = time.time()
    result_pcd, pred_cam_pos, gt_poses = utils.reconstruct(data_root, args.version)

    l2 = utils.mean_l2(pred_cam_pos, gt_poses)
    n = 0 if gt_poses is None else min(len(pred_cam_pos), len(gt_poses))
    print(f"Mean L2 distance: {l2:.4f} m  (over {n} frames)")
    print(f"Execution time: {time.time() - t0:.2f} seconds")

    if args.no_vis or gt_poses is None:
        return

    # ── Reproduce the coordinate reconciliation mean_l2 uses, for display ─────
    pred_world = np.asarray(pred_cam_pos, dtype=np.float64).copy()
    pred_world[:, 1] *= -1
    gt_disp = np.asarray(gt_poses, dtype=np.float64)[:, :3].copy()
    gt_disp[:, 2] *= -1
    n_min = min(len(pred_world), len(gt_disp))
    offset = gt_disp[0] - pred_world[0]
    pred_aligned = pred_world[:n_min] + offset

    scene_no_ceil = utils.remove_ceiling(result_pcd, margin=1.0)
    scene_no_ceil.translate(offset)
    traj_pred = utils.make_trajectory(pred_aligned,        [1.0, 0.0, 0.0])  # red
    traj_gt   = utils.make_trajectory(gt_disp[:n_min],     [0.0, 0.0, 0.0])  # black

    print("[main] Opening visualiser …  (press Q to quit)")
    o3d.visualization.draw_geometries(
        [scene_no_ceil, traj_pred, traj_gt],
        window_name=f'HW1 Reconstruction - {os.path.basename(os.path.normpath(data_root))} ({args.version})',
        zoom=0.5, front=[0, -1, 0], lookat=[0, 0, 0], up=[0, 0, -1])


if __name__ == '__main__':
    main()
