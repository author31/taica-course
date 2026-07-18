"""
Thin CLI over hw1/utils.py: reconstruct ONE captured run (a dir holding
rgb/ depth/ GT_pose.npy) with geometry-only ICP SLAM, print trajectory mean L2 vs
ground truth, write the number back into the experiment that describes it, and
open an Open3D window with the reconstructed cloud + estimated (red) and GT
(black) trajectories.

The heavy lifting lives in utils.py so the evaluator can run headless. This file
is the interactive/visual entry point and the run orchestrator, nothing else.

    pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/baseline/
    pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/mixed/ --version open3d
    pixi run -e habitat python hw1/reconstruct.py \
        --data_root eval/_data/first_floor/baseline \
        --experiment hw1/experiments/strict_clip.ttl --no-vis

EXPERIMENT IN, TWO RUNS OUT  (`--experiment <path.ttl>`)
    One experiment Turtle is BOTH the selection input and the result sink, so a
    run is self-describing: which frames, under which settings, produced which
    error. Everything RDF-shaped is delegated to hw1/api.py — `read_experiment`
    on the way in, `write_run` on the way out (§8.2). This file
    never parses Turtle and never re-derives the IRI scheme; the frame-IRI tail
    parse has exactly one implementation, `api.frame_index_from_iri`, and
    `read_experiment` calls it for us.

    THE FILE IS STUDENT-AUTHORED ABOVE THE MARKER (§3.1). An
    experiment file is a student's declaration — batch, factor selection,
    thresholds — plus a machine section written once by `api.py experiment`.
    Runs are the experiment's OUTCOME, not its design, so `write_run` is the one
    mutation an assessed file accepts: it rewrites only below the marker and
    verifies the `hw1:declarationDigest` seal first, which is why this program
    refuses a file whose declaration was edited after assessment.

    Without `--experiment` this stays the plain visual entry point: whole batch,
    no selection, no write-back. (§7 lists `--experiment` in the
    command surface; it is kept OPTIONAL here so a fresh capture with no
    measurement pass yet is still reconstructable and viewable. `--selected-only`
    without an experiment is a hard error — there is nothing to select from.)

BASELINE AND SELECTED ARE TWO RUN NODES IN ONE EXPERIMENT
    This is the v2 change to read first. In v1 a selected run was a DERIVED
    EXPERIMENT with its own id and its own `hw1:includesFrame` list, and the two
    files had to be kept apart by a digest suffix. Both are deleted
    (§6/§11). Now:

        <exp> hw1:hasRun <exp>/run/baseline  (hw1:selectionMode hw1:FullBatch)
        <exp> hw1:hasRun <exp>/run/selected  (hw1:selectionMode hw1:GoodSegments)

    Two IRIs inside one experiment, so baseline and selected CANNOT overwrite
    each other by construction — which was the failure mode the derived-id
    machinery existed to prevent. BOTH RUNS EXECUTE BY DEFAULT, because the
    comparison between them is the deliverable and a default that produced only
    half of it invited reporting the wrong half. `--baseline-only` /
    `--selected-only` restrict; `--no-write` prints and touches no file.

    V3.2 INTERPRETATION: selected is a falsification probe, not a repair promise.
    It tests whether failed inputs are load-bearing while recording the splice,
    gap and gate mechanism by which deletion itself can be harmful. The
    full-batch baseline is the convergence outcome. Continuous values are the
    effect size; Pass/Fail is only the declared policy line.

SELECTION IS STATUS-DRIVEN, AND THE STATUSES ARE ALREADY IN THE FILE
    `api.py experiment` baked a `hw1:Pass`/`hw1:Fail` next to every observable
    when it measured it (§4.3), against the QualificationSettings
    recorded on the same experiment. So selection here is not a measurement and
    not a query: `read_experiment` hands over `usable_links` — every pair whose
    `hw1:qualificationStatus` is Pass (vacuously so when no pair factor was
    selected) AND whose two endpoint frames are usable, i.e. every annotation of
    theirs passed (vacuously so for a modality with no selected factor), §4.5 —
    and `api.cut_contiguous_segments(usable_links)` chains those links into
    maximal contiguous segments. EVERY maximal chain is kept, whatever its
    length: the `minSegmentLength` floor was deleted from the vocabulary on
    2026-07-31 because it encoded a property of the ICP backend, not of the data
    (api.cut_contiguous_segments).

    IF THERE IS NO USABLE LINK AT ALL, NO SELECTED RUN IS WRITTEN and the reason
    is printed (§7). Writing an `INF` run for "there was nothing to
    reconstruct" would put a measurement failure and a selection outcome in the
    same triple, and the attribution query cannot tell them apart.

    SUBSETTING — why segments and not a frame filter. Dropping interior frames
    widens the motion between the frames that remain, while the constant-velocity
    prior and the per-step gate in utils.reconstruct are sized for CONSECUTIVE
    frames. A sparse selection therefore scores worse for reasons that have
    nothing to do with the quality of the frames it kept — see utils.reconstruct's
    SUBSETTING note. That is exactly why the cut is contiguous: inside a segment
    every surviving pair is still consecutive, so the gate keeps the size it was
    designed for. The selected frame list handed to utils.reconstruct is the
    concatenation of the segments, so ONE spliced pair appears per segment
    boundary — a handful of splices instead of one per rejected frame, which is
    the whole point of cutting rather than filtering.

`--data_root` STAYS AN EXPLICIT ARGUMENT
    The experiment's `hw1:batchFile` names the capture directory, but this
    program keeps `--data_root` explicit: a capture must stay reconstructable
    from its own directory, which is utils.reconstruct's existing contract. The
    only thing standing between "scored capture A, wrote the number into
    capture B's experiment" and silent, unrecoverable nonsense is the guard
    below: if the experiment's `hw1:onBatch` name and the `--data_root`
    basename disagree, a loud warning naming both is printed. The batch name
    is floor-qualified (`floor1_baseline`) and the directory is not
    (`baseline`), so the comparison strips the `floor<N>_` prefix first.

`--version` / THE `icpBackend` SETTING
    `icpBackend` is an ordinary string-valued Parameter (§5, values
    `open3d` | `my_icp`), recorded on every experiment because it is one of the
    three run-factor parameters the completeness rule always requires (§4.2). So:
      * caller passes no `--version` and the experiment records the setting -> the
        backend comes FROM THE EXPERIMENT;
      * caller passes `--version` and it disagrees with the recorded setting ->
        hard error, not a silent override, because the file would otherwise
        assert a setting value that is not what produced the number;
      * neither -> `open3d`, the declared default.

WRITE-BACK
    `api.write_run(<experiment>, mode, {"mapMeanL2": l2}, …)` — the ONE writer of
    `hw1:ReconstructionRun` triples (§8.2). It rewrites the machine
    section in place (rdflib parse -> mutate -> serialize below the marker),
    which drops comments and reorders prefixes THERE; acceptable because that
    half of the file is machine generated, and the student's declaration above
    the marker is preserved byte for byte. It is idempotent PER KEY, so a later
    `completeness.py` adding `coverageF` to the same run does not erase
    `mapMeanL2`, and re-running this program replaces its own number instead of
    accumulating a second one. `write_run` also computes `hw1:mapMeanL2Status`
    from the experiment's own `maxMapMeanL2` threshold, so this file never
    compares a value to a threshold.

    THE NAME `mapMeanL2` IS LEGACY. `utils.mean_l2` compares predicted and GT
    camera centres, so it is a TRAJECTORY metric. `--reference-root` adds
    `coverageF`, the complementary point-cloud map metric. Every run also stores
    `gatedSteps`; selected stores `spliceCount` and `maxGapLength`.

    `mean_l2` returns `inf` when GT is missing and that `inf` is written as-is:
    `"INF"^^xsd:double` fails the finite threshold, so a scoreless run grades
    FAILED rather than excellent — the same fail-closed convention every
    measurer in `api.py` uses. `--no-write` prints the numbers and touches
    nothing.

    SELECTION PROVENANCE IS IN THE FILE. The selected run asserts one
    `hw1:usedFrame` per frame it consumed plus `hw1:runFrameCount`, so "which
    frames produced this number" is answerable from the experiment alone —
    `api.py explore <experiment>.ttl` prints it. There is no CSV sidecar to keep
    in sync, and none is needed.
"""
import os
import re
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                      # noqa: E402
import open3d as o3d                                    # noqa: E402

import api                                              # noqa: E402
import utils                                            # noqa: E402


# Batch names are floor-qualified (`floor1_baseline`); capture directories are not.
_FLOOR_PREFIX = re.compile(r"^floor\d+_")


def _capture_dir_name(batch_name):
    """The capture-dir basename a batch name ends with: floor1_baseline -> baseline."""
    return _FLOOR_PREFIX.sub("", batch_name)


def _plan_selected(exp):
    """The frames of the selected run, or None when there is nothing to select.

    Returns (frames, segments) with `frames` the ascending concatenation of the
    kept segments, or (None, []) when the experiment has no usable link at all —
    the §7 "do not write an INF run" case, which the caller reports and skips.

    There is no length floor to read: every maximal chain of usable links is a
    segment (`api.cut_contiguous_segments`, and the deletion note in the TBox).
    """
    segments = api.cut_contiguous_segments(exp["usable_links"])
    if not segments:
        print(f"[reconstruct] selection is EMPTY: {len(exp['usable_links'])} usable "
              f"links, so there is not one consecutive pair to chain.")
        print(f"[reconstruct]   -> NO selected run written (§7): an INF "
              f"run would report 'nothing to reconstruct' in the same triple a "
              f"measurement failure uses.")
        print(f"[reconstruct]   -> fix per the verdict: loosen a Qualification "
              f"threshold and re-measure, or regenerate the capture.")
        return None, segments

    frames = [f for seg in segments for f in seg]
    spans = ", ".join(f"{s[0]}..{s[-1]}({len(s)})" for s in segments[:8])
    if len(segments) > 8:
        spans += f", … +{len(segments) - 8} more"
    print(f"[reconstruct] selection: {len(segments)} segment(s), {len(frames)} frames "
          f"[{spans}]")
    return frames, segments


def _score(data_root, version, frames, build_cloud, mode, map_voxel=None,
           mask_root=None, prior_warp_depth_gate=0.10,
           collect_prior_warp=False):
    """Run and score once, preserving the gate count as experiment evidence."""
    print(f"[reconstruct] === {mode} run ===")
    t0 = time.time()
    pcd, pred_cam_pos, gt_poses, diagnostics = utils.reconstruct(
        data_root, version, frames=frames, build_cloud=build_cloud,
        down_voxel=map_voxel, return_diagnostics=True, mask_root=mask_root,
        prior_warp_depth_gate=prior_warp_depth_gate,
        collect_prior_warp=collect_prior_warp)
    l2 = utils.mean_l2(pred_cam_pos, gt_poses)
    n = 0 if gt_poses is None else min(len(pred_cam_pos), len(gt_poses))
    print(f"[reconstruct] {mode}: mean L2 = {l2:.4f} m  (over {n} frames, "
          f"{len(pred_cam_pos)} reconstructed)")
    print(f"[reconstruct] {mode}: execution time {time.time() - t0:.2f} s")
    return pcd, pred_cam_pos, gt_poses, l2, diagnostics


def _write_link_diagnostics(experiment_path, mode, diagnostics):
    """Persist the JSON-safe, versioned per-link evidence for one scored run."""
    root = os.path.splitext(os.path.abspath(experiment_path))[0]
    directory = os.path.join(root, "diagnostics")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{mode}.json")
    payload = {
        "schema_version": int(diagnostics.get("schema_version", 1)),
        "mode": mode,
        "gated_steps": int(diagnostics.get("gated_steps", 0)),
        "gated_frames": [int(v) for v in diagnostics.get("gated_frames", ())],
        "links": diagnostics.get("links", []),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    return os.path.relpath(
        path, os.path.dirname(os.path.abspath(experiment_path))).replace(os.sep, "/")


def _segment_metadata(segments):
    """Statusless evidence for the temporal discontinuities in a selected run."""
    if not segments:
        return {"spliceCount": 0, "maxGapLength": 0}
    gaps = [max(0, int(right[0]) - int(left[-1]) - 1)
            for left, right in zip(segments, segments[1:])]
    return {"spliceCount": max(0, len(segments) - 1),
            "maxGapLength": max(gaps, default=0)}


def _visualise(data_root, version, mode, result_pcd, pred_cam_pos, gt_poses):
    """The Open3D window: reconstructed cloud + estimated (red) and GT (black) path."""
    if gt_poses is None:
        return
    # ── Draw in the frame mean_l2 scores in, via the same two helpers, so the
    #    gap between the red and black paths IS the number printed above ───────
    pred_disp = utils.pred_positions_frame0(pred_cam_pos)
    gt_disp   = utils.gt_positions_frame0(gt_poses)
    n_min = min(len(pred_disp), len(gt_disp))
    if n_min == 0:
        return

    # remove_ceiling reads the camera frame (ceiling at min y), so crop BEFORE
    # rotating the cloud onto the scoring frame's axes.
    scene_no_ceil = utils.remove_ceiling(result_pcd, margin=1.0)
    to_gt_axes = np.eye(4)
    to_gt_axes[:3, :3] = utils.CAM_TO_GT_AXES
    scene_no_ceil.transform(to_gt_axes)
    traj_pred = utils.make_trajectory(pred_disp[:n_min],   [1.0, 0.0, 0.0])  # red
    traj_gt   = utils.make_trajectory(gt_disp[:n_min],     [0.0, 0.0, 0.0])  # black

    print(f"[reconstruct] opening visualiser for the {mode} run …  (press Q to close)")
    o3d.visualization.draw_geometries(
        [scene_no_ceil, traj_pred, traj_gt],
        window_name=f'HW1 Reconstruction - '
                    f'{os.path.basename(os.path.normpath(data_root))} '
                    f'({version}, {mode})',
        # +Z of the old display frame became -Z here, so the up vector flips with it.
        zoom=0.5, front=[0, -1, 0], lookat=[0, 0, 0], up=[0, 0, 1])


def main():
    parser = argparse.ArgumentParser()
    # dest default is None ON PURPOSE: it is the only way to tell "caller asked for
    # open3d" from "caller said nothing", and the icpBackend rule below needs that.
    parser.add_argument('-v', '--version', type=str, default=None,
                        help='open3d or my_icp (default: the experiment\'s icpBackend '
                             'setting, else open3d). Passing this AND an experiment '
                             'that records a different backend is an error.')
    parser.add_argument('--data_root', type=str,
                        default=os.path.join("eval", "_data", "second_floor",
                                             "baseline"),
                        help='capture dir to reconstruct (rgb/ depth/ GT_pose.npy). '
                             'Stays explicit even though the experiment names the '
                             'same directory in hw1:batchFile.')
    parser.add_argument('--no-vis', action='store_true',
                        help='skip the Open3D window (print metrics only)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='assessed experiment Turtle (hw1/experiments/<expname>.ttl): '
                             'frame selection in via the baked Pass/Fail statuses, run '
                             'values out, written below the machine marker. Omit for a '
                             'plain whole-batch visual run.')
    parser.add_argument('--no-write', action='store_true',
                        help='dry run: print the results but write nothing into the '
                             'experiment file')
    parser.add_argument('--reference-root', default=None,
                        help='optional CLEAN capture used to build the whole-scene GT '
                             'reference map. When supplied, write hw1:coverageF at '
                             '10 cm alongside mapMeanL2. This intentionally builds a '
                             'cloud even with --no-vis.')
    parser.add_argument('--mask-dir', default=None,
                        help='one exported factor-mask directory. PNG value 255 means '
                             'drop; frame masks are <stem>.png and pair masks are '
                             '<i>_<j>.png. Adds a hw1:MaskFiltered run.')
    parser.add_argument('--mask-factor', default=None,
                        help='factor local name for --mask-dir provenance (default: '
                             'the mask directory basename)')
    # Mutually exclusive because "both by default" is the contract (§7) and the two
    # flags are ways to ask for LESS than the default, never for a third thing.
    which = parser.add_mutually_exclusive_group()
    which.add_argument('--baseline-only', action='store_true',
                       help='run only the hw1:FullBatch (whole-batch) run')
    which.add_argument('--selected-only', action='store_true',
                       help='run only the hw1:GoodSegments (selected) run')
    args = parser.parse_args()

    data_root = args.data_root
    version = args.version
    if args.mask_factor is not None and args.mask_dir is None:
        parser.error("--mask-factor requires --mask-dir")

    if args.experiment is None and args.selected_only:
        parser.error("--selected-only needs --experiment: the selection comes from the "
                     "Pass/Fail statuses baked into an experiment file "
                     "(§4.5), and there is nothing to cut segments from without one.")

    mask_factor = None
    if args.mask_dir is not None:
        mask_factor = (args.mask_factor or
                       os.path.basename(os.path.normpath(args.mask_dir)))

    exp = None
    if args.experiment is not None:
        exp = api.read_experiment(args.experiment)
        print(f"[reconstruct] {args.experiment}: experiment {exp['exp_name']} on batch "
              f"{exp['batch_name']!r}, {len(exp['frame_status'])} frames, "
              f"{len(exp['pair_status'])} pairs, "
              f"{len(exp['usable_links'])} usable links")
        if mask_factor is not None and mask_factor not in exp["selected"]:
            parser.error(
                f"--mask-factor {mask_factor!r} is not selected by the experiment; "
                f"selected factors are {', '.join(exp['selected'])}")

        # ── Guard: is this experiment even about this capture? ────────────────
        # Scoring one capture and writing the number into another capture's
        # experiment is silent and unrecoverable, and nothing else catches it.
        exp_dir_name = _capture_dir_name(exp["batch_name"])
        arg_dir_name = os.path.basename(os.path.normpath(data_root))
        if exp_dir_name != arg_dir_name:
            print(f"[reconstruct] *** WARNING: BATCH MISMATCH ***\n"
                  f"[reconstruct]   experiment {args.experiment} is on batch "
                  f"{exp['batch_name']!r} (capture dir {exp_dir_name!r})\n"
                  f"[reconstruct]   --data_root is {data_root!r} "
                  f"(capture dir {arg_dir_name!r})\n"
                  f"[reconstruct]   the score of one capture would be written into "
                  f"another capture's experiment.", file=sys.stderr)

        # ── icpBackend: the RECORDED setting is authoritative ─────────────────
        set_backend = exp["settings"].get("icpBackend")
        if set_backend is not None:
            set_backend = str(set_backend)
            if version is None:
                version = set_backend
                print(f"[reconstruct] icpBackend from experiment: {version}")
            elif version != set_backend:
                # Not an override: running my_icp and storing the number under an
                # experiment that RECORDS open3d would make the file lie about the
                # treatment that produced it.
                parser.error(
                    f"--version {version!r} disagrees with the icpBackend setting "
                    f"{set_backend!r} recorded in {args.experiment}. That setting is "
                    f"part of the treatment this experiment declares; write a new "
                    f"declaration for the other backend (§3.1: every "
                    f"tuning is a brand-new experiment) instead of overriding it.")

    if version is None:
        version = 'open3d'                      # §5 declared default

    # ── What to run (§7: both by default) ────────────────────────
    # Without an experiment there is nothing to select from, so the single run is
    # the whole batch — the plain visual entry point, unchanged.
    plan = []
    if not args.selected_only:
        plan.append(("baseline", None, [], None, None))
    if exp is not None and not args.baseline_only:
        frames, segments = _plan_selected(exp)
        if frames is not None:
            plan.append(("selected", frames, segments, None, None))
    if args.mask_dir is not None:
        plan.append(("masked", None, [], args.mask_dir, mask_factor))

    # The map is only ever looked at by the visualiser; skipping it halves peak
    # memory for the two-run default and utils.reconstruct guarantees the
    # trajectory is identical either way.
    build_cloud = not args.no_vis or args.reference_root is not None

    gt_reference = None
    if args.reference_root is not None:
        import completeness
        print(f"[reconstruct] building fixed GT map from {args.reference_root} …")
        gt_reference = completeness.build_gt_reference(args.reference_root)
        print(f"[reconstruct] GT map: {len(gt_reference.points)} points")

    finished = []
    prior_selected = exp is not None and "PriorWarpDepthResidual" in exp["selected"]
    prior_gate = (float(exp["settings"].get("priorWarpDepthGate", 0.10))
                  if exp is not None else 0.10)
    for mode, frames, segments, mask_root, run_mask_factor in plan:
        pcd, pred_cam_pos, gt_poses, l2, diagnostics = _score(
            data_root, version, frames, build_cloud, mode,
            map_voxel=0.03 if gt_reference is not None else None,
            mask_root=mask_root, prior_warp_depth_gate=prior_gate,
            collect_prior_warp=bool(prior_selected and mode == "baseline"))
        values = {"mapMeanL2": l2}
        if gt_reference is not None and gt_poses is not None:
            coverage = completeness.fscore_for_capture(
                pcd, gt_poses, gt_reference)[completeness.PRIMARY_TAU]["f"]
            values["coverageF"] = coverage
            print(f"[reconstruct] {mode}: map coverage F@10cm = {coverage:.4f}")
        run_metadata = {"gatedSteps": diagnostics["gated_steps"]}
        if mode == "selected":
            run_metadata.update(_segment_metadata(segments))
        finished.append((mode, frames, pcd, pred_cam_pos, gt_poses, l2,
                         run_metadata, values))

        if exp is None:
            print(f"[reconstruct] no --experiment: mapMeanL2 = {l2!r} not written "
                  f"anywhere")
            continue
        if args.no_write:
            print(f"[reconstruct] --no-write: {mode} mapMeanL2 = {l2!r} NOT written "
                  f"to {args.experiment}")
            continue

        if prior_selected and mode == "baseline":
            api.write_pair_measurements(
                args.experiment, "PriorWarpDepthResidual",
                diagnostics.get("prior_warp_measurements", []))

        diagnostic_file = _write_link_diagnostics(
            args.experiment, mode, diagnostics)

        # ONE call per run, and the one writer of run triples (§8.2).
        # `used_frames` is the selected run's provenance (hw1:usedFrame); the
        # baseline asserts no frame list and states its size instead.
        frame_count = len(pred_cam_pos)
        if frames is not None and frame_count != len(frames):
            # utils.reconstruct skips a frame whose rgb/depth fails to load. Report
            # it: hw1:runFrameCount is the number actually consumed, so it would
            # otherwise silently disagree with COUNT(?usedFrame) (§4.4).
            print(f"[reconstruct] *** WARNING: {mode} run selected {len(frames)} "
                  f"frames but reconstructed {frame_count}; some frame failed to "
                  f"load. hw1:runFrameCount reports what was consumed.",
                  file=sys.stderr)
        api.write_run(
            args.experiment, mode, values,
            used_frames=frames if frames is not None else None,
            frame_count=frame_count, metadata=run_metadata,
            mask_factor=run_mask_factor, diagnostic_file=diagnostic_file)
        # `l2` is the computed double; `write_run` stores it at the file's precision
        # and grades THAT number (§4.5), so the literal in the file may
        # be a rounded version of what is printed here.
        print(f"[reconstruct] wrote {mode} run: mapMeanL2 = {l2!r} (stored at file "
              f"precision), runFrameCount = {frame_count} -> {args.experiment}")

    baseline_results = [row for row in finished if row[0] == "baseline"]
    if baseline_results:
        base_l2 = baseline_results[0][5]
        for row in finished:
            if row[0] == "baseline":
                continue
            other_l2 = row[5]
            print(f"[reconstruct] baseline {base_l2:.4f} m vs {row[0]} "
                  f"{other_l2:.4f} m "
                  f"({row[0]} {'helped' if other_l2 < base_l2 else 'HURT'})")

    if args.no_vis:
        return

    # One window per executed run, in order, each labelled with its mode: the
    # baseline-vs-selected comparison is the deliverable, and two labelled windows
    # are how you see it rather than infer it from two numbers.
    for mode, _frames, pcd, pred_cam_pos, gt_poses, _l2, _meta, _values in finished:
        _visualise(data_root, version, mode, pcd, pred_cam_pos, gt_poses)


if __name__ == '__main__':
    main()
