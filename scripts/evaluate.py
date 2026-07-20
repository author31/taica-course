"""
HW1 robustness evaluator — two-run (baseline vs mixed) temporal-uncertainty eval.

ONE config (hw1/configs/second_floor.yaml), TWO in-process runs of the same
trajectory replay (no subprocess, no pygame — simulator.viewer is never imported):

  1. COLLECT — simulator.Engine + simulator.replay_poses over config["trajectory"]:
       run 1: scheduler OFF (uncertainties.enabled=False) -> <output.root>/baseline/
       run 2: scheduler ON  (UncertaintyScheduler(cfg))   -> <output.root>/mixed/
     Each run writes rgb/ depth/ GT_pose.npy; the mixed run also persists the
     realized uncertainty windows to windows.json (seconds — window ground truth,
     kept outside the ontology store). A missing trajectory file is a HARD ERROR
     (no silent [skip]).

  2. SCORE — utils.reconstruct (geometry-only ICP) per condition:
       * mean L2 (per-frame trajectory error vs GT poses; same reconciliation as
         utils.mean_l2).
       * F-score (hw1/completeness.py) — coverage-aware accuracy/completeness.
         The whole-floor GT reference is built from baseline/ ONLY.
     Reported for the WHOLE EPISODE (per condition) and PER WINDOW: each window
     (start_s, end_s) from windows.json maps to 1-indexed frame indices via
       frame_start = max(1, round(start_s * fps)),  frame_end = round(end_s * fps)
     (replay time base t = i / fps, frames come 1-indexed from replay). Per-window
     rows report the mixed-run mean L2 inside the window next to the baseline run
     over the SAME frames; a final "clean" row covers all frames outside every
     window.

OUTPUTS (eval/):
  results.csv            one row per condition: mean L2 + accuracy/completeness/F.
  per_window.csv         one row per uncertainty window (+ the clean remainder).

RUN (pixi habitat env — habitat-sim + open3d both live there):
  pixi run -e habitat python scripts/evaluate.py                  # collect + score
  pixi run -e habitat python scripts/evaluate.py --no-collect     # rescore existing
  pixi run -e habitat python scripts/evaluate.py --no-collect --data-root DIR
                              # rescore an arbitrary <root>/{baseline,mixed} tree
                              # (sim-free: the simulator package is not imported)
"""
import os
import sys
import csv
import json
import copy
import shutil
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
def _prepare_capture_dirs(data_root, out_cfg):
    if out_cfg.get("clear_existing", False) and os.path.isdir(data_root):
        shutil.rmtree(data_root)
    subs = ["rgb", "depth"]
    if out_cfg.get("save_semantic", False):
        subs.append("semantic")
    for sub in subs:
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)


def collect(config, fps):
    """Replay config["trajectory"] twice from the ONE config: scheduler disabled
    -> <output.root>/baseline/, scheduler enabled -> <output.root>/mixed/.
    Returns the absolute output root. Missing trajectory raises."""
    from simulator import (Engine, UncertaintyScheduler, load_trajectory,
                           replay_poses, save_frame)

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
        scheduler = (UncertaintyScheduler(cfg["uncertainties"])
                     if cfg["uncertainties"]["enabled"] else None)
        data_root = os.path.join(root, cond)
        _prepare_capture_dirs(data_root, cfg["output"])
        print(f"[collect] {cond}: replaying {len(poses)} poses -> {data_root}")

        engine = Engine(cfg, scheduler=scheduler, fps_nominal=fps)
        try:
            def out_cb(frame, sensor_state, idx,
                       _root=data_root, _out=cfg["output"]):
                save_frame(frame, sensor_state, _root, _out, idx)

            captured = replay_poses(engine, poses, out_cb)
            np.save(os.path.join(data_root, "GT_pose.npy"),
                    np.asarray(captured, dtype=np.float32))
            if scheduler is not None:
                scheduler.save(os.path.join(data_root, "windows.json"))
                print(f"[collect] {cond}: {len(scheduler.windows)} uncertainty "
                      f"windows -> windows.json")
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


def load_windows(mixed_root):
    """Realized uncertainty windows ([{start_s, end_s, type, params}]) from the
    mixed run's windows.json; [] (with a warning) if absent."""
    path = os.path.join(mixed_root, "windows.json")
    if not os.path.exists(path):
        print(f"[warn] {path} missing — per-window breakdown skipped")
        return []
    with open(path) as f:
        return json.load(f)


def window_frame_range(win, fps, n_frames):
    """Map one (start_s, end_s) window to inclusive 1-indexed frame bounds under
    the replay time base t = i / fps. Empty window <-> f1 < f0."""
    f0 = max(1, int(round(float(win["start_s"]) * fps)))
    f1 = min(n_frames, int(round(float(win["end_s"]) * fps)))
    return f0, f1


def per_window_rows(windows, errs_mixed, errs_base, fps):
    """One metric row per window + a trailing 'clean' row for frames outside
    every window. Returns (rows, covered_mask)."""
    n = 0 if errs_mixed is None else len(errs_mixed)
    covered = np.zeros(n, dtype=bool)
    rows = []
    for k, w in enumerate(windows):
        f0, f1 = window_frame_range(w, fps, n)
        empty = not n or f1 < f0                   # window outside the capture
        row = {"window": k, "type": w.get("type", "?"),
               "start_s": float(w["start_s"]), "end_s": float(w["end_s"]),
               "frame_start": "" if empty else f0,
               "frame_end": "" if empty else f1,
               "n_frames": 0 if empty else f1 - f0 + 1,
               "l2_mixed": float("nan"), "l2_baseline": float("nan")}
        if not empty:
            sl = slice(f0 - 1, f1)                 # 1-indexed frames -> 0-based
            covered[sl] = True
            row["l2_mixed"] = float(np.mean(errs_mixed[sl]))
            if errs_base is not None and len(errs_base) >= f1:
                row["l2_baseline"] = float(np.mean(errs_base[sl]))
        rows.append(row)

    if n:
        clean = ~covered
        rows.append({
            "window": "clean", "type": "none",
            "start_s": float("nan"), "end_s": float("nan"),
            "frame_start": "", "frame_end": "",
            "n_frames": int(clean.sum()),
            "l2_mixed": float(np.mean(errs_mixed[clean])) if clean.any() else float("nan"),
            "l2_baseline": (float(np.mean(errs_base[clean[:len(errs_base)]]))
                            if errs_base is not None and clean[:len(errs_base)].any()
                            else float("nan")),
        })
    return rows, covered


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


def write_per_window_csv(rows, path):
    cols = ["window", "type", "start_s", "end_s", "frame_start", "frame_end",
            "n_frames", "l2_mixed", "l2_baseline"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["window"], r["type"],
                        "" if isinstance(r["start_s"], float) and np.isnan(r["start_s"]) else f"{r['start_s']:.3f}",
                        "" if isinstance(r["end_s"], float) and np.isnan(r["end_s"]) else f"{r['end_s']:.3f}",
                        r["frame_start"], r["frame_end"], r["n_frames"],
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
        print("\n=== per-window breakdown (mixed run; baseline over same frames) ===")
        hdr = ["window", "type", "t (s)", "frames", "n", "L2 mixed", "L2 base"]
        print("".join(h.ljust(15) for h in hdr))
        for r in rows:
            t_span = ("" if isinstance(r["start_s"], float) and np.isnan(r["start_s"])
                      else f"{r['start_s']:.2f}-{r['end_s']:.2f}")
            if r["frame_start"] != "":
                f_span = f"{r['frame_start']}-{r['frame_end']}"
            else:
                f_span = "outside" if r["window"] == "clean" else "none"
            cells = [str(r["window"]), r["type"], t_span, f_span,
                     str(r["n_frames"]),
                     "n/a" if np.isnan(r["l2_mixed"]) else f"{r['l2_mixed']:.4f}",
                     "n/a" if np.isnan(r["l2_baseline"]) else f"{r['l2_baseline']:.4f}"]
            print("".join(c.ljust(15) for c in cells))


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
                    help="nominal replay fps: time base t = i/fps for both the "
                         "replay and the seconds->frame window mapping")
    ap.add_argument("--no-collect", action="store_true",
                    help="skip the two sim runs; rescore existing captures")
    ap.add_argument("--data-root", default=None,
                    help="override <output.root> (dir holding baseline/ and "
                         "mixed/); with --no-collect the config is not read and "
                         "the simulator package is never imported")
    ap.add_argument("--out-dir", default="eval",
                    help="where results.csv / per_window.csv go")
    args = ap.parse_args()

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

    windows = load_windows(mixed_root)
    rows, _ = per_window_rows(windows, results["mixed"]["errs"],
                              results["baseline"]["errs"], args.fps)

    write_results_csv(results, os.path.join(out_dir, "results.csv"))
    write_per_window_csv(rows, os.path.join(out_dir, "per_window.csv"))
    print_tables(results, rows)


if __name__ == "__main__":
    main()
