# TAICA Course — HW1: Camera Pose Estimation in Habitat-Sim

A two-stage pipeline built on [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
and the [Replica](https://github.com/facebookresearch/Replica-Dataset) `apartment_0` scene:

1. **`scripts/load.py`** — interactively drive an agent through the scene in a
   pygame window, previewing a first-person and a bird's-eye view, and capture
   RGB / depth / semantic keyframes (with ground-truth camera poses) to disk.
2. **`scripts/reconstruct.py`** — offline, simulator-free geometric
   reconstruction (depth unprojection → FPFH/RANSAC → ICP) that rebuilds a
   world map and scores the *estimated* camera trajectory against the ground truth.

The whole environment is managed by [pixi](https://pixi.sh); both scripts run in
the same `habitat` environment.

---

## 1. Environment preparation

### Install pixi (one-time, if you don't have it)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
# then restart the shell (or `source ~/.bashrc`) so `pixi` is on PATH
```

### Install the project environment

From the repo root — this solves and installs the `habitat` environment
(Python 3.9, habitat-sim 0.3.3 + bullet, habitat-lab, pygame, open3d, …) from
`pixi.lock`:

```bash
pixi install -e habitat
```

Sanity-check the install:

```bash
pixi run -e habitat smoke      # prints habitat-sim / habitat-lab versions + bullet build
```

### Download the Replica scene

`load.py` needs `apartment_0` at the path in `scripts/config.yaml`
(`replica_v1/apartment_0/…`). Fetch it with the provided task:

```bash
pixi run -e habitat fetch-replica   # downloads + unzips replica_v1/apartment_0
```

---

## 2. Collect keyframes — `scripts/load.py`

Opens a pygame window with a first-person RGB view and a top-down bird's-eye
view side by side, and lets you teleoperate the agent while capturing frames.

```bash
pixi run -e habitat python scripts/load.py
# optional explicit config:
pixi run -e habitat python scripts/load.py --config scripts/config.yaml
```

All environment conditions (spawn pose, camera intrinsics/extrinsics, bird's-eye
camera, lighting, depth-sensor faults, output paths) live in
**`scripts/config.yaml`** — retune there, no code edits.

### Keybindings (the pygame window must have focus)

| Key         | Action                          |
|-------------|---------------------------------|
| `W` / `S`   | move forward / backward         |
| `A` / `D`   | turn left / right               |
| `C` / `SPACE` | capture the current frame     |
| `Q` / `ESC` | finish and quit                 |

Movement only refreshes the live preview; **capture is decoupled from movement**,
so only the frames you explicitly capture are written to disk.

### Output

Written under `output.root` in `config.yaml` (default `data_collection/first_floor/`):

| Path              | Meaning                                                    |
|-------------------|------------------------------------------------------------|
| `rgb/<n>.png`     | first-person RGB (lighting applied)                        |
| `depth/<n>.png`   | depth preview (normalized to `depth.max_range`)            |
| `semantic/<n>.png`| colourised semantic labels                                 |
| `GT_pose.npy`     | `(N, 7)` captured poses `[x, y, z, qw, qx, qy, qz]`        |

> To collect a second-floor set, point `output.root` at
> `data_collection/second_floor/` in `config.yaml` and re-run.

---

## 3. Reconstruct & evaluate — `scripts/reconstruct.py`

Runs on numpy + Open3D + scipy — no simulator needed. It unprojects each
keyframe's depth into a point cloud, chains pairwise RANSAC + ICP registration to
estimate the trajectory, accumulates a world map, and scores the estimate against
the stored ground-truth poses.

```bash
pixi run -e habitat python scripts/reconstruct.py             # floor 1, Open3D ICP
```

Options:

```bash
pixi run -e habitat python scripts/reconstruct.py \
    -f 1 \            # floor: 1 -> data_collection/first_floor, 2 -> second_floor
    -v open3d         # ICP backend: open3d (required) or my_icp (custom, bonus)
```

`-f`/`--floor` selects the input directory; it must contain the `rgb/`, `depth/`,
and `GT_pose.npy` produced by step 2. At least 2 keyframes are required.

### Output

The console prints per-frame progress and the **mean L2 trajectory error vs
ground truth**, then opens an Open3D window showing the reconstructed point cloud
with the estimated (red) and ground-truth (black) camera trajectories.
