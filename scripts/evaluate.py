"""
HW1 robustness / generalization evaluator.

Orchestrates the whole metric sweep for the two floors and the three lighting
perturbation axes (+ a neutral baseline), then writes a results table and one
radar chart per floor.

FOR EACH (floor, axis):
  1. COLLECT  — subprocess `hw1/load.py` to REPLAY the floor's trajectory through
     that axis's config, writing rgb/ depth/ GT_pose.npy into
     eval/_data/<floor>/<axis>/. A subprocess isolates the habitat-sim lifecycle
     (one sim per config) and the GT pose is exact (teleport replay, identical
     across axes). `--output-root` overrides the config's output.root (B3);
     `--fps` fixes the flicker time base so flicker is reproducible (B1).
  2. SCORE    — utils.reconstruct(...) (geometry-only ICP, build_cloud=False) →
     predicted camera trajectory; utils.mean_l2 vs the GT poses.

OUTPUTS (eval/):
  results.csv            rows = floor, cols = axis, cells = mean L2 (m).
  radar_firstfloor.png   one radar per floor: 3 spokes (low_light, over_exposure,
  radar_secondfloor.png  flicker), single series = that floor's mean L2, with the
                         baseline drawn as a dashed reference ring.

RUN (pixi habitat env — habitat-sim + open3d both live there):
  pixi run -e habitat python scripts/evaluate.py                 # full sweep
  pixi run -e habitat python scripts/evaluate.py --no-collect    # rescore existing _data
  pixi run -e habitat python scripts/evaluate.py --floors first_floor
"""
import os
import sys
import csv
import argparse
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hw1"))
import utils   # noqa: E402  (geometry-only ICP reconstruct + mean_l2)

# Axis order used everywhere (CSV columns, radar spokes minus baseline).
AXES = ["baseline", "low_light", "over_exposure", "flicker"]
PERTURB_AXES = ["low_light", "over_exposure", "flicker"]   # radar spokes

# (floor -> trajectory + per-axis config). The floor is selected by the
# trajectory that gets replayed; the config supplies the lighting severity.
FLOORS = {
    "first_floor": {
        "traj": "trajectories/firstfloor.npy",
        "configs": {
            "baseline":      "configs/baseline.firstfloor.yaml",
            "low_light":     "configs/robustness.low_light.firstfloor.yaml",
            "over_exposure": "configs/robustness.over_exposure.firstfloor.yaml",
            "flicker":       "configs/robustness.flicker.firstfloor.yaml",
        },
    },
    "second_floor": {
        "traj": "trajectories/secondfloor.npy",
        "configs": {
            "baseline":      "configs/baseline.secondfloor.yaml",
            "low_light":     "configs/generalization.low_light.secondfloor.yaml",
            "over_exposure": "configs/generalization.over_exposure.secondfloor.yaml",
            "flicker":       "configs/generalization.flicker.secondfloor.yaml",
        },
    },
}


def collect(config, traj, out_root, fps):
    """Replay `traj` through `config` into out_root (rgb/ depth/ GT_pose.npy)."""
    os.makedirs(out_root, exist_ok=True)
    cmd = [sys.executable, os.path.join(REPO, "hw1", "load.py"),
           "--config", config, "--trajectory", traj,
           "--output-root", out_root, "--fps", str(fps)]
    print(f"  [collect] {os.path.basename(config)} -> {out_root}")
    subprocess.run(cmd, cwd=REPO, check=True)


def score(data_root, version):
    """Reconstruct the capture and return mean L2 (m)."""
    _, pred, gt = utils.reconstruct(data_root, version, verbose=False,
                                    build_cloud=False)
    return utils.mean_l2(pred, gt)


def write_csv(results, path):
    """results[floor][axis] = mean L2. Rows = floor, cols = axis."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["floor"] + AXES)
        for floor in FLOORS:
            if floor not in results:
                continue
            w.writerow([floor] + [f"{results[floor].get(a, float('nan')):.4f}"
                                  for a in AXES])
    print(f"[write] {path}")


def radar(floor, row, path):
    """One radar per floor: PERTURB_AXES spokes, series = mean L2, baseline as a
    dashed reference ring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [row.get(a, float("nan")) for a in PERTURB_AXES]
    baseline = row.get("baseline", float("nan"))

    angles = np.linspace(0, 2 * np.pi, len(PERTURB_AXES), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    vals_closed = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, vals_closed, color="tab:red", linewidth=2,
            label="mean L2 (perturbed)")
    ax.fill(angles_closed, vals_closed, color="tab:red", alpha=0.20)

    if np.isfinite(baseline):
        ring = [baseline] * len(angles_closed)
        ax.plot(angles_closed, ring, color="black", linewidth=1.5,
                linestyle="--", label=f"baseline = {baseline:.3f} m")

    ax.set_xticks(angles)
    ax.set_xticklabels([a.replace("_", " ") for a in PERTURB_AXES])
    # annotate each spoke with its value
    for ang, v in zip(angles, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.3f}", xy=(ang, v), fontsize=9,
                        ha="center", va="bottom")
    ax.set_title(f"{floor} — mean L2 vs lighting perturbation", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floors", nargs="*", default=list(FLOORS),
                    choices=list(FLOORS), help="which floors to run")
    ap.add_argument("--axes", nargs="*", default=AXES, choices=AXES,
                    help="which axes to run")
    ap.add_argument("--version", default="open3d", choices=("open3d", "my_icp"),
                    help="ICP backend for scoring")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="replay fps (flicker time base; keep fixed for repro)")
    ap.add_argument("--no-collect", action="store_true",
                    help="skip replay; rescore whatever is already under eval/_data")
    ap.add_argument("--data-root", default="eval/_data",
                    help="root for per-(floor,axis) captures")
    ap.add_argument("--out-dir", default="eval", help="where results.csv + radars go")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    for floor in args.floors:
        spec = FLOORS[floor]
        traj = os.path.join(REPO, spec["traj"])
        if not os.path.exists(traj):
            print(f"[skip] {floor}: missing trajectory {spec['traj']}")
            continue
        results[floor] = {}
        for axis in args.axes:
            out_root = os.path.join(REPO, args.data_root, floor, axis)
            if not args.no_collect:
                collect(spec["configs"][axis], traj, out_root, args.fps)
            if not os.path.isdir(os.path.join(out_root, "rgb")):
                print(f"  [skip] {floor}/{axis}: no capture at {out_root}")
                continue
            l2 = score(out_root, args.version)
            results[floor][axis] = l2
            print(f"  [score] {floor}/{axis}: mean L2 = {l2:.4f} m")

    write_csv(results, os.path.join(args.out_dir, "results.csv"))
    for floor in args.floors:
        if results.get(floor):
            radar(floor, results[floor],
                  os.path.join(args.out_dir, f"radar_{floor.replace('_floor','floor')}.png"))

    # console summary table
    print("\n=== mean L2 (m) ===")
    print("floor".ljust(14) + "".join(a.ljust(15) for a in AXES))
    for floor in args.floors:
        if results.get(floor):
            row = results[floor]
            print(floor.ljust(14) + "".join(
                (f"{row.get(a, float('nan')):.4f}").ljust(15) for a in AXES))


if __name__ == "__main__":
    main()
