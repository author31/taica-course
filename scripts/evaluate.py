"""
HW1 robustness evaluator — two-run (baseline vs mixed) spatial-uncertainty eval.

ONE config (hw1/configs/second_floor.yaml), TWO in-process runs of the same
trajectory replay (no subprocess, no pygame — simulator.viewer is never imported):

  1. COLLECT — simulator.Engine + simulator.replay_poses over config["trajectory"]:
       run 1: scheduler OFF (uncertainties.enabled=False) -> <output.root>/baseline/
       run 2: scheduler ON  (ZoneScheduler(cfg))          -> <output.root>/mixed/
     Each run writes rgb/ depth/ GT_pose.npy intrinsics.json and nothing else
     (plan.md D5 + D6 — the capture ships its own camera parameters and no
     effect labels; the spatial scheduler is stateless, so there is no window
     ground truth to persist any more). A missing trajectory file is a HARD
     ERROR (no silent [skip]).

  2. SCORE — utils.reconstruct (geometry-only ICP) per condition:
       * mean L2 (per-frame trajectory error vs GT poses; same reconciliation as
         utils.mean_l2).
       * F-score (hw1/completeness.py) — coverage-aware accuracy/completeness.
         The whole-floor GT reference is built from baseline/ ONLY.
     Reported for the WHOLE EPISODE (per condition) and PER ZONE: uncertainty is
     spatial (plan.md §3.1), so each captured pose is assigned to the first
     config zone whose circle contains its XZ (simulator.zone_frame_labels), and
     the per-zone rows report the mixed-run mean L2 inside that zone next to the
     baseline run over the SAME frames. A final "outside" row covers every frame
     in no zone at all — the within-capture clean control.

OUTPUTS (eval/):
  results.csv            one row per condition: mean L2 + accuracy/completeness/F.
  per_zone.csv           one row per uncertainty zone (+ the outside remainder).

RUN (pixi habitat env — habitat-sim + open3d both live there):
  pixi run -e habitat python scripts/evaluate.py                  # collect + score
  pixi run -e habitat python scripts/evaluate.py --no-collect     # rescore existing
  pixi run -e habitat python scripts/evaluate.py --no-collect --data-root DIR
                              # rescore an arbitrary <root>/{baseline,mixed} tree
                              # (sim-free: the simulator package is not imported,
                              #  so the zone breakdown is skipped — no config)
"""
import os
import sys
import csv
import copy
import argparse

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "hw1"))
import utils          # noqa: E402  (geometry-only ICP reconstruct + mean_l2)
import completeness   # noqa: E402  (anchor + accuracy/completeness/F-score)

DEFAULT_CONFIG = os.path.join("hw1", "configs", "second_floor.yaml")
CONDITIONS = ("baseline", "mixed")   # run order: baseline feeds the GT reference


def _abspath(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


# =============================================================================
# COLLECT — in-process two-run replay. The simulator package is imported lazily
# so the rescore path (--no-collect) never needs habitat; viewer is NEVER imported.
# =============================================================================
def collect(config, fps):
    """Replay config["trajectory"] twice from the ONE config: scheduler disabled
    -> <output.root>/baseline/, scheduler enabled -> <output.root>/mixed/.
    Returns the absolute output root. Missing trajectory raises."""
    from simulator import (Engine, ZoneScheduler, load_trajectory,
                           prepare_capture_dirs, replay_poses, save_frame)

    traj = _abspath(config["trajectory"])
    if not os.path.exists(traj):
        raise FileNotFoundError(
            f"trajectory not found: {traj} — generate it with "
            "scripts/search_traj.py (missing trajectory is a hard error)")
    poses = load_trajectory(traj)
    root = _abspath(config["output"]["root"])

    for cond in CONDITIONS:
        cfg = copy.deepcopy(config)
        cfg["uncertainties"]["enabled"] = (cond == "mixed")
        scheduler = (ZoneScheduler(cfg["uncertainties"])
                     if cfg["uncertainties"]["enabled"] else None)
        data_root = os.path.join(root, cond)
        # Shared capture path: dirs + the capture's own intrinsics.json (D5).
        prepare_capture_dirs(cfg, data_root)
        print(f"[collect] {cond}: replaying {len(poses)} poses -> {data_root}")
        if scheduler is not None:
            print(f"[collect] {cond}: {len(scheduler.zones)} uncertainty zone(s): "
                  + (", ".join(z["name"] for z in scheduler.zones) or "NONE"))

        engine = Engine(cfg, scheduler=scheduler, fps_nominal=fps)
        try:
            def out_cb(frame, sensor_state, idx,
                       _root=data_root, _out=cfg["output"]):
                save_frame(frame, sensor_state, _root, _out, idx)

            captured = replay_poses(engine, poses, out_cb)
            np.save(os.path.join(data_root, "GT_pose.npy"),
                    np.asarray(captured, dtype=np.float32))
        finally:
            engine.close()
    return root


# =============================================================================
# SCORE — sim-free metric path (reconstruction + per-frame L2 + F-score)
# =============================================================================
def per_frame_l2(pred_cam_pos, gt_poses):
    """Per-frame L2 errors (metres), same frame reconciliation as utils.mean_l2
    (whole-episode mean of this array == utils.mean_l2). None if either side is
    missing/empty. Index k <-> capture frame k+1 (frames are 1-indexed)."""
    if gt_poses is None or len(gt_poses) == 0 \
            or pred_cam_pos is None or len(pred_cam_pos) == 0:
        return None
    pred = np.asarray(pred_cam_pos, dtype=np.float64).copy()
    pred[:, 1] *= -1                                   # cam -> habitat world
    gt_c = np.asarray(gt_poses, dtype=np.float64)[:, :3].copy()
    gt_c[:, 2] *= -1                                   # match pred display frame
    n = min(len(pred), len(gt_c))
    pred_al = pred[:n] + (gt_c[0] - pred[0])           # align origins
    return np.linalg.norm(pred_al - gt_c[:n], axis=1)


def score_condition(data_root, version, gt_ref):
    """Reconstruct one condition's capture. Returns (per-frame L2 array|None,
    fscore dict|None). Missing capture is a hard error."""
    if not os.path.isdir(os.path.join(data_root, "rgb")):
        raise FileNotFoundError(
            f"no capture at {data_root} — run evaluate.py without --no-collect")
    pcd, pred, gt = utils.reconstruct(data_root, version, verbose=False,
                                      build_cloud=True, down_voxel=0.05)
    errs = per_frame_l2(pred, gt)
    fs = None
    if gt_ref is not None and gt is not None and len(gt) > 0:
        fs = completeness.fscore_for_capture(pcd, gt, gt_ref)
    return errs, fs


def load_capture_poses(data_root):
    """The capture's GT_pose.npy ((N,7) sensor poses), or None if absent."""
    path = os.path.join(data_root, "GT_pose.npy")
    if not os.path.exists(path):
        print(f"[warn] {path} missing — per-zone breakdown skipped")
        return None
    return np.load(path)


def per_zone_rows(config, poses, errs_mixed, errs_base):
    """One metric row per configured zone + a trailing 'outside' row for the
    frames in no zone at all (the within-capture clean control).

    Zone membership is a property of POSITION and both conditions replay the
    identical trajectory, so the very same frame indices are compared across
    conditions. `n_frames` counts captured poses (simulator.zone_frame_counts —
    the plan.md §3.1 step-3 check); the L2 means only average over the frames
    the reconstruction actually produced. The simulator import is lazy: the
    sim-free rescore path (--no-collect --data-root) has no config and skips
    this entirely."""
    from simulator import OUTSIDE, zone_frame_counts, zone_frame_labels

    counts = zone_frame_counts(config, poses)      # zone order, then "outside"
    labels = zone_frame_labels(config, poses)

    def _mean(errs, idx):
        sel = [i for i in idx if i < len(errs)] if errs is not None else []
        return float(np.mean(errs[sel])) if sel else float("nan")

    rows = []
    for name, n_frames in counts.items():
        want = None if name == OUTSIDE else name
        idx = [i for i, lab in enumerate(labels) if lab == want]
        rows.append({"zone": name,
                     "n_frames": n_frames,
                     "l2_mixed": _mean(errs_mixed, idx),
                     "l2_baseline": _mean(errs_base, idx)})
    return rows


def write_results_csv(results, path, tau=completeness.PRIMARY_TAU):
    """Whole-episode summary: one scored row per condition."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_frames", "mean_l2", "tau",
                    "accuracy", "completeness", "f_score"])
        for cond in CONDITIONS:
            r = results.get(cond)
            if r is None:
                continue
            n = 0 if r["errs"] is None else len(r["errs"])
            if r["fs"] is not None:
                s = r["fs"][tau]
                w.writerow([cond, n, f"{r['l2']:.4f}", f"{tau:.2f}",
                            f"{s['accuracy']:.4f}", f"{s['completeness']:.4f}",
                            f"{s['f']:.4f}"])
            else:
                w.writerow([cond, n, f"{r['l2']:.4f}", f"{tau:.2f}", "", "", ""])
    print(f"[write] {path}")


def write_per_zone_csv(rows, path):
    cols = ["zone", "n_frames", "l2_mixed", "l2_baseline"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["zone"], r["n_frames"],
                        "" if np.isnan(r["l2_mixed"]) else f"{r['l2_mixed']:.4f}",
                        "" if np.isnan(r["l2_baseline"]) else f"{r['l2_baseline']:.4f}"])
    print(f"[write] {path}")


def print_tables(results, rows, tau=completeness.PRIMARY_TAU):
    print("\n=== whole episode ===")
    hdr = ["condition", "frames", "mean L2 (m)", f"F@{tau}", "acc", "comp"]
    print("".join(h.ljust(14) for h in hdr))
    for cond in CONDITIONS:
        r = results.get(cond)
        if r is None:
            continue
        n = 0 if r["errs"] is None else len(r["errs"])
        if r["fs"] is not None:
            s = r["fs"][tau]
            cells = [cond, str(n), f"{r['l2']:.4f}", f"{s['f']:.3f}",
                     f"{s['accuracy']:.3f}", f"{s['completeness']:.3f}"]
        else:
            cells = [cond, str(n), f"{r['l2']:.4f}", "n/a", "n/a", "n/a"]
        print("".join(c.ljust(14) for c in cells))

    if rows:
        print("\n=== per-zone breakdown (mixed run; baseline over same frames) ===")
        hdr = ["zone", "n frames", "L2 mixed", "L2 base"]
        print("".join(h.ljust(15) for h in hdr))
        for r in rows:
            cells = [str(r["zone"]), str(r["n_frames"]),
                     "n/a" if np.isnan(r["l2_mixed"]) else f"{r['l2_mixed']:.4f}",
                     "n/a" if np.isnan(r["l2_baseline"]) else f"{r['l2_baseline']:.4f}"]
            print("".join(c.ljust(15) for c in cells))
        # per_zone_rows puts the "outside" remainder last; the rest are zones.
        empty = [r["zone"] for r in rows[:-1] if r["n_frames"] == 0]
        if empty:
            print(f"[warn] zones never entered by this trajectory: "
                  f"{', '.join(empty)} — they fired on nothing (plan.md §3.1 "
                  f"step 3: every zone must be entered)")
        if rows[-1]["n_frames"] == 0:
            print("[warn] no frame falls outside every zone — this capture has "
                  "no within-capture clean control (plan.md §3.1 step 3)")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="hw1 config yaml (loaded via simulator.load_config)")
    ap.add_argument("--version", default="open3d", choices=("open3d", "my_icp"),
                    help="ICP backend for scoring")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="nominal replay fps: the replay time base is t = i/fps")
    ap.add_argument("--no-collect", action="store_true",
                    help="skip the two sim runs; rescore existing captures")
    ap.add_argument("--data-root", default=None,
                    help="override <output.root> (dir holding baseline/ and "
                         "mixed/); with --no-collect the config is not read and "
                         "the simulator package is never imported")
    ap.add_argument("--out-dir", default="eval",
                    help="where results.csv / per_zone.csv go")
    args = ap.parse_args()

    config = None
    if args.no_collect and args.data_root:
        root = _abspath(args.data_root)          # fully sim-free rescore path
    else:
        from simulator import load_config
        config = load_config(_abspath(args.config))
        if args.data_root:
            config["output"]["root"] = args.data_root
        root = _abspath(config["output"]["root"])
        if not args.no_collect:
            collect(config, args.fps)

    out_dir = _abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_root = os.path.join(root, "baseline")
    mixed_root = os.path.join(root, "mixed")

    # Whole-floor GT reference for the coverage/F-score — from baseline/ ONLY
    # (completeness.build_gt_reference refuses a mixed/ dir).
    if not os.path.isdir(os.path.join(base_root, "rgb")):
        raise FileNotFoundError(
            f"no baseline capture at {base_root} — run evaluate.py without "
            "--no-collect first")
    print(f"[gt-ref] building whole-floor reference from {base_root} ...")
    gt_ref = completeness.build_gt_reference(base_root)
    print(f"[gt-ref] {len(gt_ref.points)} points")

    results = {}
    for cond, dr in (("baseline", base_root), ("mixed", mixed_root)):
        errs, fs = score_condition(dr, args.version, gt_ref)
        l2 = float("inf") if errs is None else float(np.mean(errs))
        results[cond] = {"errs": errs, "l2": l2, "fs": fs}
        print(f"[score] {cond}: mean L2 = {l2:.4f} m"
              + ("" if fs is None else
                 f" | F@{completeness.PRIMARY_TAU} = "
                 f"{fs[completeness.PRIMARY_TAU]['f']:.3f}"))

    # Per-zone breakdown: needs the config (zone geometry) and the mixed run's
    # captured poses. The sim-free rescore path has no config, so it is skipped.
    rows = []
    if config is None:
        print("[warn] no config (--no-collect --data-root) — per-zone "
              "breakdown skipped")
    else:
        poses = load_capture_poses(mixed_root)
        if poses is not None:
            rows = per_zone_rows(config, poses, results["mixed"]["errs"],
                                 results["baseline"]["errs"])

    write_results_csv(results, os.path.join(out_dir, "results.csv"))
    if rows:
        write_per_zone_csv(rows, os.path.join(out_dir, "per_zone.csv"))
    print_tables(results, rows)


if __name__ == "__main__":
    main()
