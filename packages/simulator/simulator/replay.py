"""Trajectory replay: .npy (N,7) sensor-pose teleport replay ONLY.

Action replay (.json) is deprecated — it reproduced a path only when the config's
actuation matched the values that generated it. load_trajectory raises on .json
with a pointer to .npy pose replay.

Contract highlights (plan.md):
- replay_poses returns the (N,7) captured poses; the CALLER saves GT_pose.npy.
- Missing/invalid trajectory is a hard error, never a silent skip.
"""

import os

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as Rot

import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs


def load_trajectory(path):
    """Return the (N,7) pose array [x,y,z, qw,qx,qy,qz] from a .npy trajectory.

    Action replay (.json sidecars) is deprecated: it only reproduced the path
    when the config's actuation matched the values that generated it. A .json
    path raises with a pointer to .npy pose replay; a missing path raises
    FileNotFoundError (hard error — the old evaluate silently skipped)."""
    if path.endswith(".json"):
        raise ValueError(
            f"action replay (.json) is no longer supported: {path} — "
            "use the .npy (N,7) sensor-pose trajectory instead (exact pose "
            "replay via replay_poses, independent of config actuation)")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory not found: {path}")
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError(f"unsupported trajectory file (want .npy): {path}")


def agent_state_from_sensor_pose(pose, config):
    """Build an AgentState that puts the COLOR sensor at world pose `pose`
    ([x,y,z, qw,qx,qy,qz]). Inverts the fixed camera extrinsic (config.camera
    position + orientation) so the teleported sensor matches the pose exactly."""
    cam = config["camera"]
    sensor_R = Rot.from_quat([pose[4], pose[5], pose[6], pose[3]])   # x,y,z,w
    cam_R = Rot.from_euler("xyz", [float(v) for v in cam["orientation"]])
    t_cam = np.asarray(cam["position"], dtype=np.float64)

    # world_sensor = agent ∘ extrinsic  =>  agent_R = sensor_R · cam_R⁻¹,
    # agent_pos = sensor_pos − agent_R · t_cam.
    agent_R = sensor_R * cam_R.inv()
    agent_pos = np.asarray(pose[:3], dtype=np.float64) - agent_R.apply(t_cam)

    st = habitat_sim.AgentState()
    st.position = agent_pos.astype(np.float32)
    st.rotation = quat_from_coeffs(agent_R.as_quat().astype(np.float32))   # [x,y,z,w]
    return st


def save_frame(frame, sensor_state, data_root, out, idx):
    """Write one capture's rgb/depth/semantic PNGs and return its GT pose row
    [x, y, z, qw, qx, qy, qz] from the color-sensor world pose."""
    if out["save_rgb"]:
        cv2.imwrite(os.path.join(data_root, "rgb", f"{idx}.png"),
                    frame["rgb"][:, :, ::-1])              # RGB -> BGR for cv2
    if out["save_depth"]:
        # 16-bit millimetres (NOT the 8-bit preview): preserves the injected
        # depth noise / coupling so the reconstructor sees it instead of it being
        # swamped by 8-bit quantisation. utils.load_depth_meters detects uint16.
        depth_mm = np.clip(frame["depth_m"] * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(data_root, "depth", f"{idx}.png"), depth_mm)
    if out["save_semantic"]:
        cv2.imwrite(os.path.join(data_root, "semantic", f"{idx}.png"),
                    frame["semantic"][:, :, ::-1])
    p, r = sensor_state.position, sensor_state.rotation
    return [p[0], p[1], p[2], r.w, r.x, r.y, r.z]


def replay_poses(engine, poses, out_cb):
    """Teleport the camera to each pose, capture a frame. Exact replay.

    Per frame i (1-based): the agent is teleported so the color sensor matches
    poses[i-1], the frame is processed via engine.observe(i / engine.fps_nominal)
    — frame index over nominal fps is replay's only deterministic clock, so
    flicker/windows/depth noise are reproducible across runs and machines —
    and `out_cb(frame, sensor_state, i)` is invoked. out_cb owns all output I/O
    (e.g. wrap save_frame, or preview); returning False aborts the replay early.

    Returns the (N,7) captured poses [x,y,z, qw,qx,qy,qz]; the CALLER saves
    GT_pose.npy."""
    cam_extr = []
    for i, pose in enumerate(poses, start=1):
        engine.agent.set_state(agent_state_from_sensor_pose(pose, engine.config))
        frame = engine.observe(i / engine.fps_nominal)
        sensor_state = engine.agent.get_state().sensor_states["color_sensor"]
        p, r = sensor_state.position, sensor_state.rotation
        cam_extr.append([p[0], p[1], p[2], r.w, r.x, r.y, r.z])
        if i % 25 == 0 or i == len(poses):
            print(f"replay: captured {i}/{len(poses)} frames")
        if out_cb(frame, sensor_state, i) is False:
            print(f"replay: aborted by user at frame {i}")
            break
    return np.asarray(cam_extr, dtype=np.float32)
