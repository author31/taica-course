# TAICA Course — HW1: Camera Pose Estimation in IsaacLab

A two-stage pipeline built on [IsaacLab](https://github.com/isaac-sim/IsaacLab):

1. **`scripts/hw1.py`** — drive a Jetbot around a warehouse scene inside Isaac Sim,
   capture RGB-D keyframes (with ground-truth camera poses), and save them to disk.
2. **`scripts/reconstruct.py`** — offline, simulator-free geometric reconstruction
   (depth unprojection → FPFH/RANSAC → ICP) that rebuilds a world map and compares
   the *estimated* camera trajectory against the ground truth.

---

## 1. Launch the IsaacLab workspace

The simulator runs inside a Docker image. The first invocation builds the image
(submodules must be initialized first); subsequent runs reuse it.

```bash
# One-time: pull the IsaacLab submodule
make submodules

# Build (if needed) and launch the container with GPU + X11 display
make launch-isaaclab
```

`make launch-isaaclab`:

- builds the `leisaac-isaaclab:latest` image (depends on `build-isaaclab`),
- enables local X11 access (`xhost +local:root`, reverted on exit),
- runs the container on GPU `device=0` with host networking, the repo
  bind-mounted at `/workspace/aicapstone`, and the display forwarded,
- selects the NVIDIA Vulkan ICD, verifies the required GL/X/Vulkan libs, then
  drops you into an interactive `bash` shell at `/workspace/aicapstone`.

Useful overrides:

```bash
make launch-isaaclab IMAGE=my-image:tag GPU=all CONTAINER_NAME=isaaclab
make check-isaaclab-gpu     # sanity-check GPU / Vulkan / torch CUDA inside the image
```

All commands below are run **inside** that container shell.

---

## 2. Collect keyframes — `scripts/hw1.py`

Opens the GUI, spawns a Jetbot with a forward-facing RGB-D camera in a warehouse,
and lets you teleoperate it while capturing keyframes.

```bash
python scripts/hw1.py
```

Common options:

```bash
python scripts/hw1.py \
    --warehouse_usd Environments/Simple_Warehouse/warehouse.usd \
    --robot_usd Isaac/Robots/Turtlebot/turtlebot3_burger.usd \
    --out outputs/hw1 \
    --width 640 --height 480 \
    --capture_every 0          # >0 = auto-capture every N control steps while moving
```

(`--assets_root` defaults to the Isaac nucleus assets dir. Standard IsaacLab
`AppLauncher` flags such as `--headless` and `--device` are also accepted.)

### Keybindings

| Key       | Action                                            |
|-----------|---------------------------------------------------|
| `W` / `S` | drive forward / backward                          |
| `A` / `D` | turn left / right                                 |
| `R`       | capture the current frame as a keyframe           |
| `C`       | clear the keyframe buffer                         |
| `F`       | save the keyframe buffer to `<out>/keyframes.npz` |
| `L`       | reset the robot to its spawn pose                 |
| `Q` / `ESC` | quit the simulation                             |

Driving is continuous (velocity is applied while the key is held); `R`/`C`/`F`/`L`
fire once per press.

### Output

Pressing `F` writes **`<out>/keyframes.npz`** (default `outputs/hw1/keyframes.npz`),
a compressed bundle of per-keyframe stacked arrays:

| Array        | Shape           | Dtype   | Meaning                                      |
|--------------|-----------------|---------|----------------------------------------------|
| `rgb`        | `(N, H, W, 3)`  | uint8   | RGB image                                    |
| `depth`      | `(N, H, W)`     | float32 | metric z-buffer depth (camera frame)         |
| `intrinsics` | `(N, 3, 3)`     | float32 | pinhole camera intrinsics                    |
| `pos`        | `(N, 3)`        | float32 | ground-truth camera world position `[x,y,z]` |
| `quat`       | `(N, 4)`        | float32 | ground-truth orientation `[qw, qx, qy, qz]`  |

Captures are also echoed to the console (`[HW1] captured keyframe #k ...`).

---

## 3. Reconstruct & evaluate — `scripts/reconstruct.py`

Runs entirely on numpy + Open3D + scipy + matplotlib — **no IsaacLab/Omniverse
dependency** — so it can run outside the simulator. It unprojects each keyframe's
depth into a point cloud, chains pairwise RANSAC + ICP registration to estimate
the camera trajectory, accumulates a world map, and scores the estimate against
the stored ground-truth poses.

```bash
python scripts/reconstruct.py outputs/hw1/keyframes.npz
```

Common options:

```bash
python scripts/reconstruct.py keyframes.npz \
    --voxel 0.05 \          # reconstruction voxel size [m]
    --max_depth 8.0 \       # drop depth beyond this [m]
    -o outputs/hw1          # output dir (defaults to the npz's directory)

python scripts/reconstruct.py outputs/hw1/keyframes.npz --show     # interactive 3D window
python scripts/reconstruct.py outputs/hw1/keyframes.npz --no-show  # render PNG only
```

(`--show` defaults to on when `$DISPLAY` is set, off otherwise.)

### Output

Written to `<out>` (default: the npz's directory):

| File                  | Description                                                          |
|-----------------------|---------------------------------------------------------------------|
| `reconstruction.ply`  | accumulated world point cloud                                       |
| `trajectory.png`      | 3D plot: ground-truth (black) vs estimated (red) camera poses       |
| `trajectory_eval.npy` | per-frame GT/estimated positions + per-frame translation error      |
| `reconstruction.png`  | 3D Open3D render of the map overlaid with both trajectories         |

The console also prints a per-frame GT-vs-estimated table and the **mean
translation error vs ground truth**. (At least 2 keyframes are required.)
