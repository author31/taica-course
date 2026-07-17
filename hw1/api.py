"""
Data-quality query CLI — per-frame brightness / valid-depth gating for the HW1 SLAM eval.

WHAT THIS FILE IS
    A small, dependency-light command-line tool that answers one question for a
    directory of captured frames: *is each frame's data quality inside the
    empirically-derived "Good" range for this pipeline?* It grounds that answer
    in the Class-B thresholds recorded in `definitions.md` (repo root), which were
    produced by `autoresearch.py` via OFAT sweeps against the geometry-only robust
    ICP reconstructor (`hw1/utils.reconstruct`).

    Unlike `autoresearch.py`'s aggregate measurers (`measure_luma`,
    `measure_valid_ratio`, which average over a whole directory), the functions
    here compute the SAME observables PER FRAME, so a query sample can be graded
    frame-by-frame and a PASS manifest carved out of it.

TWO API-SERVICEABLE AXES  (single-sample computable → checkable at inference)
    * Brightness  — observable `avg_luma_rec601`: the Rec.601 mean luma of the RGB
      frame, in [0, 255]. Gated against THRESHOLDS[floor]["luma"] = (lo, hi).
    * Depth       — observable `valid_depth_fraction`: fraction of depth pixels that
      are non-zero AND fall inside [min_range, max_range] metres, in [0, 1]. Gated
      against THRESHOLDS[floor]["valid_frac_min"].

WHY NO NOISE AXIS
    The third sweep axis, depth noise (`realized_sigma_z_m`), is deliberately
    EXCLUDED. Measuring it requires a clean-depth reference of the *same*
    trajectory (paired noisy−clean std). The OFAT sweep has that reference; an
    arbitrary API query sample does not. Un-computable at inference → un-checkable
    range → not API-serviceable. See "Why no depth-noise definition" in
    `definitions.md`.

FLOOR-2 CAVEAT
    On floor 2 the brightness band is DEGENERATE (full-range, non-gating): the
    brightness-axis neutral F is already at the noise floor, so every sweep point
    "passes" and the band collapses to the whole sweep. `--floor 2` therefore emits
    a stderr WARNING; the depth-ratio axis remains well-formed on both floors.

DEPENDENCIES
    Standard library + numpy + Pillow only. No OpenCV, no Open3D.

DEPTH FORMAT
    Depth PNGs are uint16 millimetres; metres = raw / 1000.0. A pixel is valid iff
    raw != 0 AND min_range <= metres <= max_range.

SEE ALSO
    definitions.md               — source of truth for the thresholds below
    autoresearch.py:145-206      — aggregate measurers this file mirrors per-frame
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

# =============================================================================
# Thresholds  (single source of truth — copied from definitions.md)
#   Class-B conventions: valid for the hw1 robust-ICP pipeline + coupling ONLY.
# =============================================================================
THRESHOLDS = {
    1: {"luma": (146.35, 230.87), "valid_frac_min": 0.570},
    2: {"luma": (18.04, 252.84),  "valid_frac_min": 0.493},  # floor-2 luma band is DEGENERATE (full-range, non-gating)
}

# Rec.601 luma weights, applied over the RGB channel axis.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# =============================================================================
# Per-frame observable measurers  (pure: numpy + Pillow)
#   Mirror autoresearch.measure_luma / measure_valid_ratio, but for ONE frame.
# =============================================================================
def frame_luma(rgb_path):
    """SPEC: mean Rec.601 luma (Y = 0.299R + 0.587G + 0.114B) of a single RGB frame, in [0,255].
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Load the PNG as RGB (PIL `Image.open(p).convert("RGB")`), cast to float64,
    take the per-pixel luma via the Rec.601 dot product, and return its mean over
    all pixels. This is the per-frame twin of autoresearch.measure_luma.
    """
    arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64)
    luma = arr @ _LUMA_WEIGHTS
    return float(luma.mean())


def frame_valid_fraction(depth_path, min_range=0.0, max_range=10.0):
    """SPEC: fraction of valid depth pixels in a single depth frame, in [0,1].
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Depth PNG is uint16 millimetres; metres = raw / 1000.0. A pixel is valid iff
    raw != 0 AND min_range <= metres <= max_range. Return valid_count / total_pixels.
    This is the per-frame twin of autoresearch.measure_valid_ratio.
    """
    raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    meters = raw.astype(np.float64) / 1000.0
    valid = (raw != 0) & (meters >= min_range) & (meters <= max_range)
    return float(valid.sum()) / float(valid.size)


# =============================================================================
# Threshold predicates  (grade an observable against THRESHOLDS[floor])
# =============================================================================
def good_brightness(luma, floor):
    """SPEC: True iff luma is inside the floor's GoodBrightnessRange band [lo, hi].
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Look up (lo, hi) = THRESHOLDS[floor]["luma"] and return lo <= luma <= hi.
    NOTE: on floor 2 this band is degenerate (full-range) and passes everything.
    """
    lo, hi = THRESHOLDS[floor]["luma"]
    return lo <= luma <= hi


def valid_depth_ratio(frac, floor):
    """SPEC: True iff the valid-depth fraction meets the floor's ValidDepthRatio minimum.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Return frac >= THRESHOLDS[floor]["valid_frac_min"].
    """
    return frac >= THRESHOLDS[floor]["valid_frac_min"]


def frame_verdict(rgb_path, depth_path, floor):
    """SPEC: grade one paired (rgb, depth) frame on both axes and return its verdict dict.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Compute luma and valid_fraction, apply good_brightness / valid_depth_ratio, and
    return {"luma", "valid_fraction", "brightness_ok", "valid_depth_ok", "verdict"}
    where verdict is "PASS" iff BOTH axes are ok, else "FAIL".
    """
    luma = frame_luma(rgb_path)
    frac = frame_valid_fraction(depth_path)
    brightness_ok = good_brightness(luma, floor)
    valid_depth_ok = valid_depth_ratio(frac, floor)
    return {
        "luma": luma,
        "valid_fraction": frac,
        "brightness_ok": brightness_ok,
        "valid_depth_ok": valid_depth_ok,
        "verdict": "PASS" if (brightness_ok and valid_depth_ok) else "FAIL",
    }


# =============================================================================
# Manifest CRUD  (operates on frame-name STEMS only — never touches image files)
#   A manifest is a JSON list of integer-stem strings, e.g. ["1", "2", "17"].
# =============================================================================
def manifest_create(verdicts):
    """SPEC: build a manifest = the sorted stems of every PASS frame.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    `verdicts` is a list of dicts each carrying a "frame" stem and a "verdict".
    Keep the stems whose verdict == "PASS", dedup, and return sorted by int value.
    """
    stems = {str(v["frame"]) for v in verdicts if v["verdict"] == "PASS"}
    return sorted(stems, key=int)


def manifest_read(path):
    """SPEC: load a manifest (JSON list of stems) from disk.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Read the JSON file at `path` and return the list of stem strings it holds.
    """
    with open(path, "r") as f:
        return json.load(f)


def manifest_update(manifest, add=None, remove=None):
    """SPEC: add and/or remove stems from a manifest, dedup, return sorted by int.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Start from `manifest`, union in `add` (iterable of stems), drop everything in
    `remove` (iterable of stems), dedup, and return the result sorted by int value.
    """
    stems = {str(s) for s in manifest}
    if add:
        stems |= {str(s) for s in add}
    if remove:
        stems -= {str(s) for s in remove}
    return sorted(stems, key=int)


def manifest_delete(manifest, stems):
    """SPEC: drop the given stems from a manifest, return the remainder sorted by int.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Remove every stem in `stems` from `manifest`, dedup, and return sorted by int.
    """
    drop = {str(s) for s in stems}
    return sorted({str(s) for s in manifest} - drop, key=int)


def manifest_write(manifest, path):
    """SPEC: persist a manifest to disk as a JSON list.
    REFERENCE IMPL (peer session) — carved to a TODO stub in the student pass.

    Write `manifest` (list of stems) to `path` as indented JSON.
    """
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# Frame pairing  (rgb/*.png <-> depth/*.png by integer stem, iterate sorted by int)
# =============================================================================
def _stem(path):
    """Integer filename stem of a frame path (e.g. '.../17.png' -> 17)."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _pair_frames(data_dir):
    """Return [(stem_str, rgb_path, depth_path), ...] paired by int stem, sorted by int.

    `data_dir` must contain `rgb/` and `depth/` subdirs of integer-stem .png frames.
    Only stems present in BOTH subdirs are yielded.
    """
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
        raise ValueError(f"data-dir must contain rgb/ and depth/ subdirs: {data_dir!r}")

    rgb = {_stem(p): p for p in glob.glob(os.path.join(rgb_dir, "*.png"))}
    depth = {_stem(p): p for p in glob.glob(os.path.join(depth_dir, "*.png"))}
    common = sorted(set(rgb) & set(depth))
    if not common:
        raise ValueError(f"No frames present in BOTH rgb/ and depth/ under: {data_dir!r}")
    return [(str(s), rgb[s], depth[s]) for s in common]


# =============================================================================
# CLI
# =============================================================================
def _build_parser():
    p = argparse.ArgumentParser(
        description="Per-frame data-quality query (brightness + valid-depth) grounded in definitions.md.",
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing rgb/ and depth/ subdirs of integer-stem .png frames.")
    p.add_argument("--floor", type=int, required=True, choices=(1, 2),
                   help="Which floor's empirical thresholds to grade against.")
    p.add_argument("--out", default=None,
                   help="Optional CSV report path.")
    p.add_argument("--manifest", default=None,
                   help="Optional JSON manifest path; written with the stems of PASS frames.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    floor = args.floor

    if floor == 2:
        print("WARNING: floor-2 brightness band is DEGENERATE (full-range, non-gating); "
              "only the valid-depth axis is meaningful on this floor.", file=sys.stderr)

    pairs = _pair_frames(args.data_dir)

    rows = []      # CSV / stdout rows
    verdicts = []  # for manifest_create
    header = ["frame", "luma", "valid_fraction", "brightness_ok", "valid_depth_ok", "verdict"]

    print("  ".join(f"{h:>14}" if h != "frame" else f"{h:>8}" for h in header))
    for stem, rgb_path, depth_path in pairs:
        v = frame_verdict(rgb_path, depth_path, floor)
        row = [
            stem,
            f"{v['luma']:.1f}",
            f"{v['valid_fraction']:.3f}",
            str(v["brightness_ok"]),
            str(v["valid_depth_ok"]),
            v["verdict"],
        ]
        rows.append(row)
        verdicts.append({"frame": stem, "verdict": v["verdict"]})
        print("  ".join(f"{c:>14}" if i else f"{c:>8}" for i, c in enumerate(row)))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"[csv] wrote {len(rows)} rows -> {args.out}")

    if args.manifest:
        manifest = manifest_create(verdicts)
        manifest_write(manifest, args.manifest)
        print(f"[manifest] wrote {len(manifest)} PASS stems -> {args.manifest}")

    n = len(rows)
    passed = sum(1 for r in rows if r[-1] == "PASS")
    failed = n - passed
    print(f"SUMMARY: {n} frames, {passed} pass, {failed} fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
