"""
HW1 data-quality triplestore CLI — turn a directory of captured SLAM frames into
RDF, measure the factors a student's DECLARATION selects, and bake the Pass/Fail
verdict of every number next to the number, in the student's own file.

WHAT THIS FILE IS
    A four-subcommand command-line tool that bridges a directory of captured
    SLAM frames to local RDF Turtle files, using `rdflib` and nothing else for
    the RDF. No server, no daemon, and — since §10 — no SPARQL:
    every artefact is a self-contained .ttl file on disk.

    * `declare`    — scaffold a declaration Turtle under `hw1/experiments/`:
                     prefixes, the Experiment node whose IRI tail equals the file
                     stem, the batch join (`hw1:batchFile` = the capture
                     directory, `hw1:onBatch` derived from `--floor` + basename),
                     a factor selection, and the PREDICTION block as TODO
                     comments. The declaration stays the student's to author —
                     this only spares them the blank page. Never assesses, never
                     overwrites.
    * `experiment` — take ONE argument, a student-authored declaration Turtle
                     (§4.2: which capture, which factors, under
                     which thresholds), measure the rasters in that capture
                     directory, and APPEND the machine section to that same
                     file: one FrameAnnotation per frame per modality that has a
                     selected factor, one FramePair per consecutive pair, every
                     selected observable with its baked status, the defaulted
                     FactorSettings, and the `hw1:declarationDigest` seal.
                     ONCE, ever — a file that already carries the marker is a
                     hard error (§3.1).
    * `explore`    — read-only terminal tables over a capture directory, a
                     declaration, an assessed experiment (values, statuses,
                     runs and the computed VERDICT section) or several
                     experiments side by side. Writes nothing, measures nothing
                     (§7.1).
    * `batch2ttl`  — DEPRECATED as a required step. `declare` / `experiment` /
                     `explore` face the capture directory directly. This command
                     remains only to write optional generation provenance
                     (`--gen`, `--derived-from`) into `<data_dir>/batch.ttl`.

    `reconstruct.py` is the fifth command of the suite and lives in its own
    file; it appends
    hw1:ReconstructionRun nodes below the marker of an experiment file through
    `write_run`.

    v2's `query` command, the `queries/*.rq` files and the pyoxigraph dependency
    are DELETED (§10). With statuses, settings, roles and runs baked
    into one self-contained file every shipped query degenerated into a
    projection over that file, and projections are `explore`'s job.

SIX DEPTH FACTORS + TWO RGB FACTORS + ONE BASELINE
    Depth frame: HighFrequencyDepthResidual, FlyingPixelRatio,
    ValidTileCoverage. Adjacent depth pair: IdentityMedianDepthChange,
    JointValidDepthRatio, PriorWarpDepthResidual. Every depth factor exports a
    scalar/status and a 0/255 drop mask. PriorWarpDepthResidual is measured from
    the real constant-velocity state before fitting the current pair and written
    back by `write_pair_measurements`.

    HighlightClipping and ShadowClipping are unchanged RGB intrinsic-health
    factors. Geometry-only ICP does not consume RGB, so they are negative controls
    for this consumer. A student selects 1..8 input factors; ReconstructionAccuracy
    and Coverage remain non-selectable, always-on run factors.

    plus the deliberately weak baseline `mean_value` in [0,255], carried into the
    file so the "did a clip factor beat the mean?" comparison is a lookup, not a
    hand calculation. It is NOT a quality factor: no threshold, no polarity, and
    deliberately NO status, so it takes no part in hw1:qualificationStatus.

    Both RGB factors are computed on the value channel V = max(R,G,B). There is no
    colorimetric weighting anywhere in this assignment: see `definitions.md`, and
    see `frame_mean_value` for why the mean ships only as a baseline to beat.

PARAMETERS, AND THE THREE VERDICTS THEIR ROLES CARRY
    Every settable number in this assignment — tau_hi, tau_lo, the depth range,
    the Sobel threshold, the edge dilation, the ICP backend, AND every Pass/Fail
    threshold — is one kind of thing: a `hw1:Parameter`, declared as an individual
    in `ontology/hw1.ttl` and recorded as a `hw1:FactorSetting` on the node it
    describes. An undeclared name is a hard error at parse time
    (`load_parameter_declarations` / `parse_setting_arg`), not a silent extra.

    A parameter's `hw1:paramRole` says WHERE its setting lives and WHAT TO FIX
    when the attribution query names it as the culprit:

    hw1:GenerationSetting     on the Batch, via the optional
                              `batch2ttl --gen NAME=VALUE` sidecar. Describes how
                              the PIXELS were produced; applied by nothing,
                              because the capture already embodies it.
                              Verdict: regenerate the data.
    hw1:MeasurementSetting    on the Experiment, as a FactorSetting in the
                              declaration. Changes only how pixels are SCORED.
                              Verdict: change the number and re-measure (which
                              means: write a NEW declaration).
    hw1:QualificationSetting  on the Experiment, likewise. A threshold: turns a
                              measured value into Pass or Fail.
                              Verdict: change the threshold and re-qualify (again:
                              a new declaration).

    Ask `explore` which ROLE the culprit setting has and it has told you what to
    go and fix. Three-way, and strictly more informative than the two-way
    MeasurementKnob/CorruptionKnob split it replaced.

    Nothing ever goes stale, because nothing is overwritten: an experiment file is
    WRITE-ONCE (§3.1). Measuring the same batch at a second tau means
    copying the declaration to a new name and assessing that, so both files stay
    side by side and `hw1/experiments/` is an append-only lab notebook.

STATUSES ARE BAKED AT MEASURE TIME
    Next to every observable value, `experiment` writes a `<value>Status` of
    hw1:Pass or hw1:Fail, computed by `status_for` against the
    QualificationSettings recorded on the SAME experiment, plus the aggregate
    hw1:qualificationStatus — on an annotation, Pass iff every selected factor of
    that modality passed; on a pair, Pass iff every selected pair factor did
    (vacuously Pass with none). The RAW VALUE is always stored too, so re-grading
    stays possible: a status is a denormalised cache whose truth is
    (value + settings). What it buys is that grading is a lookup instead of a
    join against threshold individuals — which is what v1 did, and where the
    tie-break between adjacent bands was a trap that produced plausible wrong
    answers.

    The per-FRAME verdict is NOT stored (v2's hw1:frameStatus is deleted): it is
    the conjunction of that frame's annotation statuses, computed in
    `read_experiment`, exactly as the usable-link conjunction already was.

IRI SCHEME  (frozen — §2; parsed in exactly one place, see below)
    STRUCTURE, shared by every experiment over one capture:
    batch            = <ns>batch/<name>
    frame            = <ns>batch/<name>/frame/<n>
    rgb              = <ns>batch/<name>/frame/<n>/rgb
    depth            = <ns>batch/<name>/frame/<n>/depth
    gen setting      = <ns>batch/<name>/setting/<param>

    EXPERIMENT-SCOPED, minted by the assessment of one declaration:
    experiment       = <ns>experiment/<expname>
    factor setting   = <ns>experiment/<expname>/setting/<factor>/<param>
    frame annotation = <ns>experiment/<expname>/annotation/<n>/<kind>  rgb | depth
    frame pair       = <ns>experiment/<expname>/pair/<i>_<j>
    run              = <ns>experiment/<expname>/run/<mode>     baseline | selected

    <name> = f"floor{floor}_{basename(data_dir)}" (see `batch_name`); it is ALSO
    the value of hw1:batchName and the last segment of the batch IRI. <n> is the
    integer file stem, decimal, unpadded; <i>_<j> are the stems of two consecutive
    PAIRED frames in ascending order; <expname> is the STEM OF THE DECLARATION
    FILE, which is also the tail of the Experiment IRI the student wrote — v2's
    8-hex digest identity is deleted (§6), and naming your
    experimental conditions is now part of designing the experiment.

    THE SPLIT IS THE ISOLATION MECHANISM. There are no named graphs and no
    persistent store: everything lands in one graph, and two experiments over one
    batch stay apart because everything they mint carries their name, while their
    annotations point at the SAME frame IRIs. That is the one rule never to relax.

    `reconstruct.py` has to go the other way — frame IRI back to an integer stem —
    because `hw1:frameIndex` lives in the batch file while reconstruct reads only
    an experiment file. It imports `frame_index_from_iri` / `batch_name_from_frame_iri`
    from here rather than re-deriving the scheme, so the tail parse exists ONCE.
    Do not write `str(iri).split("/")[-1]` in a second module.

DEPTH FORMAT
    Depth PNGs are uint16 millimetres; metres = raw / 1000.0. Every active depth
    factor uses the consumer rule raw != 0 with no range cap — exactly what
    `utils.depth_image_to_point_cloud` consumes.

STORAGE
    Local files only, and one of them is HAND-WRITTEN ABOVE THE MARKER. An
    experiment file is `<student declaration>` + `MACHINE_MARKER` +
    `<machine section>` (§3.1): `experiment` appends the second half
    once and never rewrites the first, `write_run` re-serializes only the second
    half, and both verify the `hw1:declarationDigest` seal before touching
    anything. `batch2ttl` is optional and still replaces
    `<data_dir>/batch.ttl` wholesale when used for generation provenance — that
    file holds no measurement and nobody hand-edits it. This module NEVER starts
    a server and does not require one to import.

DEPENDENCIES
    Standard library + numpy + Pillow + rdflib (graph build/serialize/parse).
    No pyoxigraph (§10), no scipy, no OpenCV, no Open3D. Every quality
    factor is computed DIRECTLY from the raw depth/RGB PNG — no unprojection, no
    point cloud, no normal estimation, and the pair factors keep that promise too:
    they compare two RASTERS, never two point clouds. That is what keeps this list
    short and what makes the factors grade the INPUT DATA rather than a
    reconstruction pipeline that may itself be unimplemented. Nothing here imports
    `utils.py` either, so an unfinished ICP cannot block this file and an
    unfinished measurer cannot block the ICP.

SEE ALSO
    ontology/hw1.ttl            — the TBox these triples must satisfy
    definitions.md              — the measurement contracts and their sources

STUDENT IMPLEMENTATION SURFACE (instruction.md §5.2)
    Students implement ONLY the eight qualification-factor measurers behind the
    factor menu — never the RDF machinery:

        frame_clip_hi_fraction                    HighlightClipping
        frame_clip_lo_fraction                    ShadowClipping
        frame_high_frequency_depth_residual (+_mask)  HighFrequencyDepthResidual
        frame_flying_pixel_ratio (+_mask)             FlyingPixelRatio
        frame_valid_tile_coverage (+_mask)            ValidTileCoverage
        pair_identity_median_depth_change (+_mask)    IdentityMedianDepthChange
        pair_joint_valid_depth_ratio (+_mask)         JointValidDepthRatio
        pair_prior_warp_depth_residual (+_mask)       PriorWarpDepthResidual

    Each currently carries the REFERENCE implementation so this file runs
    end-to-end and the contract tests pass; in the student-facing distribution
    each body is stripped to a `#TODO` stub and the docstring CONTRACT is the
    assignment. Everything else (CLI, declaration validation, setting
    resolution, status computation, ontology wiring, the marker/seal machinery)
    ships as-is and is off-limits.
"""

import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np
from PIL import Image

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

# =============================================================================
# Namespaces  (must match ontology/hw1.ttl exactly — do not rename)
#   §1 freezes this list. QUDT/UNIT/SKOS/PROV are not decoration:
#   qudt:unit is what keeps "0.41" from being a unitless number nobody can check,
#   skos: is what marks the four closed term vocabularies (Status, Polarity,
#   SettingRole, SelectionMode) as vocabularies rather than classes of measurable
#   things, and prov:wasDerivedFrom is the edge from a corrupted capture back to
#   the capture it was made from — BETWEEN BATCHES, the only place it survives
#   (§6: there are no derived experiments).
#
#   `dqv:` is deliberately absent (§1). It carried the v1 Result /
#   Metric shape, which is gone: a hw1:QualityFactor is defined by its own four
#   properties and a run outcome is an ordinary observable.
# =============================================================================
NS = "http://taica.course/hw1/ontology#"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Path to hw1/ontology/hw1.ttl relative to this file (this file lives in hw1/).
# It is the TBox, and it is also the AUTHORITY on which parameter and factor names
# exist: see load_parameter_declarations / load_quality_factors.
_ONTOLOGY_TTL = os.path.join(_HERE, "ontology", "hw1.ttl")

# Where `experiment` writes (§3).
_EXPERIMENT_DIR = os.path.join(_HERE, "experiments")

# Depth PNG encoding: uint16 millimetres.
_DEPTH_SCALE = 1000.0

# =============================================================================
# Immerkaer noise-estimation constants — the measurement contract of
# `frame_high_frequency_depth_residual`. FIXED: they are NOT yours to derive and
# they are NOT Parameters. Unlike the tau pair they need no per-experiment
# provenance, because they are not free parameters: they are the kernel and the
# two constants of one published estimator, and changing one does not measure the
# same quantity differently — it measures a different quantity. Read these names
# in your measurer; do not paste the numbers inline.
# =============================================================================
# Immerkaer 1996's 3x3 mask: the difference of two discrete Laplacians, so it
# annihilates any locally-linear depth surface and leaves the high-frequency
# residual behind.
_IMMERKAER_M = np.array([[1.0, -2.0, 1.0],
                         [-2.0, 4.0, -2.0],
                         [1.0, -2.0, 1.0]])
# ||M|| = sqrt(sum(M^2)) = sqrt(36) = 6: the factor by which the mask amplifies
# an i.i.d. noise standard deviation.
_IMMERKAER_NORM = 6.0
# Median of |N(0,1)| = Phi^-1(0.75). Converts a robust spread back to a sigma.
_MAD_TO_SIGMA = 0.6745


# =============================================================================
# Per-frame observable measurers
#   Every one of them is PURE and DETERMINISTIC: a path (plus parameters) in, one
#   documented scalar out. No I/O beyond reading the frame, no global state, no
#   randomness — that is what makes them autogradable and what makes a stored
#   observable reproducible from the file it was measured from.
# =============================================================================
def _value_channel(rgb_path):
    """V = max(R,G,B) per pixel, float64 HxW in [0,255].  HSV Value, no weighting.

    The single definition of the value channel for this assignment; both clip
    factors and the baseline are built on it.
    """
    arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64)
    return arr.max(axis=2)


def frame_mean_value(rgb_path):
    """The BASELINE, deliberately weak: mean of V = max(R,G,B) over one RGB frame.

    Returns a float in [0, 255].  Deterministic and pure (numpy + Pillow only).

    THIS IS NOT A QUALITY FACTOR AND HAS NO BAND.  It ships implemented for two
    reasons: it shows the shape every measurer in this file has (path in, one
    documented scalar out), and it is the baseline your two clip factors must
    OUTPERFORM.  It is a poor exposure metric on its own terms:

      * a first moment cannot represent two failure modes — under- and
        over-exposure — with one interval, so any threshold around it widens
        until it admits both;
      * it is confounded with scene content: pointing the camera at a bright
        window moves it as much as the effect you are trying to measure does;
      * it is blind to distribution shape.  A frame that is half crushed and half
        blown out has a perfectly ordinary mean of 127.5 while not one pixel in it
        is recoverable — see the half-black/half-white fixture in test_e2e.py;
      * the same trap is documented for sum-based gradient metrics (Zhang,
        Forster and Scaramuzza, ICRA 2017): they select over-exposed images, while
        percentile-based metrics won on 13 of 18 datasets.  Mean and sum
        statistics fail this way; tail and order statistics do not.

    That is the transferable lesson, and beating this baseline is how you meet it.
    """
    return float(_value_channel(rgb_path).mean())


def frame_clip_hi_fraction(rgb_path, tau_hi):
    """RGB quality factor 1 (HighlightClipping) — blown-out pixel fraction.

    CONTRACT
        In:     rgb_path — an RGB PNG.
                tau_hi — the highlight threshold on the 0-255 value scale.  It has
                NO DEFAULT and must never be hardcoded: it is a MEASUREMENT
                parameter you derive and justify, and a fraction measured at one
                tau is simply not the same quantity as a fraction measured at
                another.  That is why tau_hi is recorded on the Batch node.
        Out:    one float in [0, 1]:

                    clip_hi_fraction = |{ V >= tau_hi }| / N ,   V = max(R,G,B)

        Grade:  LowerIsBetter, one-sided: Pass iff <= maxClipHiFraction.
        Purity: deterministic; no randomness, no global state.

    THE TWO CLIP FACTORS USE THE SAME OPERATOR AND MEAN DIFFERENT QUANTIFIERS
    This is correct, deliberate, and the most likely thing for someone to "fix":

        V >= tau_hi   <=>   AT LEAST ONE channel is saturated      (exists)
        V <= tau_lo   <=>   EVERY channel is crushed               (for all)

    Both are right, because clipping is a per-channel phenomenon: one railed
    channel has already destroyed the pixel's colour, while a pixel is only truly
    black when nothing is left in any channel.

    SOURCE
        Shin, Kim, Kim, Lee and Kim, "Camera Exposure Control for Robust Robot
        Vision with Noise-Aware Image Quality Assessment", IROS 2019 — the
        unsaturated-region mask of their eq. 7, U(i) = 1 iff tau_l <= I(i) <= tau_h,
        which masks out exactly the pixels whose values carry no recoverable
        information.  Split into two observables rather than one union because the
        gate does not need the direction but the DIAGNOSIS does: a single number
        cannot tell a crushed frame from a blown-out one.

    STUDENT IMPLEMENTATION (instruction.md §5.2): implemented BY STUDENTS.
        The reference body below ships only so the pipeline can be tested end
        to end; the handout strips it to a `#TODO` stub, and this docstring's
        CONTRACT is the assignment.
    """
    V = _value_channel(rgb_path)
    return float(np.count_nonzero(V >= tau_hi)) / float(V.size)


def frame_clip_lo_fraction(rgb_path, tau_lo):
    """RGB quality factor 2 (ShadowClipping) — crushed pixel fraction.

    CONTRACT
        In:     rgb_path — an RGB PNG.
                tau_lo — the shadow threshold on the 0-255 value scale.  NO
                DEFAULT, never hardcoded, for the same reason as tau_hi: it is a
                measurement parameter, and it is recorded on the Batch node.
        Out:    one float in [0, 1]:

                    clip_lo_fraction = |{ V <= tau_lo }| / N ,   V = max(R,G,B)

        Grade:  LowerIsBetter, one-sided: Pass iff <= maxClipLoFraction.
        Purity: deterministic; no randomness, no global state.

    THE TWO CLIP FACTORS USE THE SAME OPERATOR AND MEAN DIFFERENT QUANTIFIERS
    Same operator as the highlight twin, different meaning — do not "fix" it:

        V >= tau_hi   <=>   AT LEAST ONE channel is saturated      (exists)
        V <= tau_lo   <=>   EVERY channel is crushed               (for all)

    SOURCE
        Shin et al., IROS 2019, eq. 7 — same source as the highlight factor.

    STUDENT IMPLEMENTATION (instruction.md §5.2): implemented BY STUDENTS.
        The reference body below ships only so the pipeline can be tested end
        to end; the handout strips it to a `#TODO` stub, and this docstring's
        CONTRACT is the assignment.
    """
    # WHY NOT min(R,G,B) <= tau_lo, the apparently symmetric alternative?  Because
    # it is wrong: pure red (255,0,0) has G = B = 0, so a fully saturated pixel
    # would be counted as crushed.  max(R,G,B) <= tau_lo is the test that means
    # "every channel is dark", which is what a crushed pixel actually is.
    V = _value_channel(rgb_path)
    return float(np.count_nonzero(V <= tau_lo)) / float(V.size)


# =============================================================================
# Depth-frame and pair observable measurers
#   Same purity contract as the RGB measurers: path(s) (plus parameters) in, one
#   documented scalar out — and, for every depth factor, a 0/255 drop mask too.
#   numpy + Pillow only, no scipy, no point cloud, no pose (the one exception:
#   PriorWarpDepthResidual receives the constant-velocity prior transform from
#   the consumer, but still compares two RASTERS).
#
#   The pair measurers exist because frame factors cannot see the ONE thing that
#   actually breaks frame-to-frame ICP: what happened BETWEEN two frames. A pair
#   of individually perfect frames taken a metre apart registers no better than
#   a pair of bad ones. Of the four literature drivers of pairwise ICP failure
#   (semantic_layer_design.md §2.1) — D1 overlap, D2 initial misalignment /
#   motion, D3 geometric degeneracy, D4 depth noise — the frame factors carry D4
#   (HighFrequencyDepthResidual, FlyingPixelRatio) and per-frame coverage
#   (ValidTileCoverage); the pair factors carry D1 (JointValidDepthRatio) and D2
#   (IdentityMedianDepthChange, PriorWarpDepthResidual).
# =============================================================================
def _consumer_depth_metres_valid(depth_path):
    """Read one uint16-mm depth raster under the ICP consumer's validity rule.

    Active depth factors use exactly the consumer rule: every non-zero return
    is valid, matching ``utils.depth_image_to_point_cloud``.
    """
    raw = np.asarray(Image.open(depth_path))
    if raw.ndim != 2:
        raise ValueError(f"depth raster must be two-dimensional, got {raw.shape}")
    metres = raw.astype(np.float64) / _DEPTH_SCALE
    return metres, raw != 0


def _flag_mask(flagged):
    """Boolean drop flags -> the on-disk mask convention (uint8 0/255)."""
    return np.where(flagged, np.uint8(255), np.uint8(0))


def _fully_valid_3x3(valid):
    if valid.shape[0] < 3 or valid.shape[1] < 3:
        return np.zeros((0, 0), dtype=bool)
    return (valid[:-2, :-2] & valid[:-2, 1:-1] & valid[:-2, 2:] &
            valid[1:-1, :-2] & valid[1:-1, 1:-1] & valid[1:-1, 2:] &
            valid[2:, :-2] & valid[2:, 1:-1] & valid[2:, 2:])


def _high_frequency_depth_residual(depth_path, residual_mask_k):
    """Shared scalar/mask computation for HighFrequencyDepthResidual.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    metres, valid = _consumer_depth_metres_valid(depth_path)
    flagged = np.zeros(valid.shape, dtype=bool)
    windows = _fully_valid_3x3(valid)
    if not windows.any():
        return float("inf"), _flag_mask(flagged)

    response = (metres[:-2, :-2] - 2.0 * metres[:-2, 1:-1] + metres[:-2, 2:] -
                2.0 * metres[1:-1, :-2] + 4.0 * metres[1:-1, 1:-1] -
                2.0 * metres[1:-1, 2:] + metres[2:, :-2] -
                2.0 * metres[2:, 1:-1] + metres[2:, 2:])
    absolute = np.abs(response)
    contributing = absolute[windows]
    median_abs = float(np.median(contributing))
    value = median_abs / (_IMMERKAER_NORM * _MAD_TO_SIGMA)

    k = float(residual_mask_k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("residualMaskK must be a finite non-negative number")
    # Quantised, perfectly planar input has median_abs == 0.  In that case a
    # literal zero threshold would turn every real step edge into noise, so use
    # one millimetre-response quantum as the minimum actionable residual.
    threshold = max(k * median_abs, 1.0 / _DEPTH_SCALE)
    centres = windows & (absolute > threshold)
    flagged[1:-1, 1:-1] = centres
    return value, _flag_mask(flagged)


def frame_high_frequency_depth_residual(depth_path, residual_mask_k=5.0):
    """Robust high-frequency depth residual in metres (LowerIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[0]


def frame_high_frequency_depth_residual_mask(depth_path, residual_mask_k=5.0):
    """255 where HighFrequencyDepthResidual recommends dropping a pixel.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[1]


def _local_extrema(values, valid, radius):
    """Window min/max without scipy; invalid samples never become extrema."""
    h, w = values.shape
    lo = np.full((h, w), np.inf, dtype=np.float64)
    hi = np.full((h, w), -np.inf, dtype=np.float64)
    padded_v = np.pad(values, radius, mode="edge")
    padded_ok = np.pad(valid, radius, mode="constant", constant_values=False)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            sample = padded_v[dy:dy + h, dx:dx + w]
            ok = padded_ok[dy:dy + h, dx:dx + w]
            lo = np.minimum(lo, np.where(ok, sample, np.inf))
            hi = np.maximum(hi, np.where(ok, sample, -np.inf))
    return lo, hi


def _flying_pixel_ratio(depth_path, window, planarity_tol):
    """Plane-discriminated mixed-boundary pixels and their drop mask.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    metres, valid = _consumer_depth_metres_valid(depth_path)
    flagged = np.zeros(valid.shape, dtype=bool)
    if not valid.any():
        return float("inf"), _flag_mask(flagged)

    size = int(round(float(window)))
    if size < 3 or size % 2 == 0:
        raise ValueError("flyingPixelWindow must be an odd integer >= 3")
    tol = float(planarity_tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("flyingPixelPlanarityTol must be finite and > 0 metres")

    radius = size // 2
    local_lo, local_hi = _local_extrema(metres, valid, radius)
    candidates = np.argwhere(valid & ((local_hi - local_lo) > 2.0 * tol))
    h, w = valid.shape
    for y, x in candidates:
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        ok = valid[y0:y1, x0:x1].copy()
        ok[y - y0, x - x0] = False
        yy, xx = np.nonzero(ok)
        if len(yy) < 6:
            continue
        z = metres[y0:y1, x0:x1][ok]

        # Split at the largest depth gap.  A real discontinuity supplies two
        # locally planar populations; a mixed/flying centre belongs to neither.
        order = np.argsort(z)
        sorted_z = z[order]
        gaps = np.diff(sorted_z)
        if gaps.size == 0:
            continue
        split_at = int(np.argmax(gaps)) + 1
        if gaps[split_at - 1] <= 2.0 * tol:
            continue
        low_idx, high_idx = order[:split_at], order[split_at:]
        if len(low_idx) < 3 or len(high_idx) < 3:
            continue

        coords = np.column_stack([xx + x0 - x, yy + y0 - y,
                                  np.ones(len(xx), dtype=np.float64)])
        try:
            low_plane = np.linalg.lstsq(coords[low_idx], z[low_idx], rcond=None)[0]
            high_plane = np.linalg.lstsq(coords[high_idx], z[high_idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        low_pred = float(low_plane[2])
        high_pred = float(high_plane[2])
        if abs(high_pred - low_pred) <= 2.0 * tol:
            continue
        centre = float(metres[y, x])
        if (min(low_pred, high_pred) - tol <= centre <=
                max(low_pred, high_pred) + tol and
                min(abs(centre - low_pred), abs(centre - high_pred)) > tol):
            flagged[y, x] = True

    return float(np.count_nonzero(flagged)) / float(flagged.size), _flag_mask(flagged)


def frame_flying_pixel_ratio(depth_path, flying_pixel_window=5,
                             flying_pixel_planarity_tol=0.03):
    """Fraction of planar-fit-discriminated flying pixels (LowerIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _flying_pixel_ratio(
        depth_path, flying_pixel_window, flying_pixel_planarity_tol)[0]


def frame_flying_pixel_ratio_mask(depth_path, flying_pixel_window=5,
                                  flying_pixel_planarity_tol=0.03):
    """255 where FlyingPixelRatio recommends dropping a pixel.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _flying_pixel_ratio(
        depth_path, flying_pixel_window, flying_pixel_planarity_tol)[1]


def _valid_tile_coverage(depth_path, tile_size, tile_valid_floor):
    """Tile-level valid-return coverage and its drop mask.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    metres, valid = _consumer_depth_metres_valid(depth_path)
    del metres
    size = int(round(float(tile_size)))
    if size <= 0:
        raise ValueError("tileSize must be a positive integer")
    floor = float(tile_valid_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("tileValidFloor must lie in [0, 1]")
    flagged = np.zeros(valid.shape, dtype=bool)
    h, w = valid.shape
    total = supported = 0
    for y0 in range(0, h, size):
        for x0 in range(0, w, size):
            tile = valid[y0:min(h, y0 + size), x0:min(w, x0 + size)]
            total += 1
            ok = bool(tile.size and float(np.mean(tile)) >= floor)
            supported += int(ok)
            if not ok:
                flagged[y0:min(h, y0 + size), x0:min(w, x0 + size)] = True
    value = 0.0 if total == 0 else float(supported) / float(total)
    return value, _flag_mask(flagged)


def frame_valid_tile_coverage(depth_path, tile_size=64, tile_valid_floor=0.5):
    """Fraction of depth tiles meeting the declared valid-return floor
    (HigherIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[0]


def frame_valid_tile_coverage_mask(depth_path, tile_size=64,
                                   tile_valid_floor=0.5):
    """255 over every tile below the declared valid-return floor.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[1]


def _identity_median_depth_change(d0_path, d1_path, change_mask_k):
    """Median |D0 - D1| at identity over jointly valid pixels, plus drop mask.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    d0, v0 = _consumer_depth_metres_valid(d0_path)
    d1, v1 = _consumer_depth_metres_valid(d1_path)
    flagged = np.zeros(d0.shape, dtype=bool)
    if d0.shape != d1.shape:
        return float("inf"), _flag_mask(flagged), 0
    joint = v0 & v1
    count = int(np.count_nonzero(joint))
    if count == 0:
        return float("inf"), _flag_mask(flagged), 0
    change = np.abs(d0 - d1)
    values = change[joint]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    k = float(change_mask_k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("changeMaskK must be a finite non-negative number")
    threshold = median + k * 1.4826 * mad
    # With a zero MAD, keep ordinary coherent motion and flag only values that
    # exceed the median by at least one quantisation step.
    threshold = max(threshold, median + 1.0 / _DEPTH_SCALE)
    flagged = joint & (change > threshold)
    return median, _flag_mask(flagged), count


def pair_identity_median_depth_change(d0_path, d1_path, change_mask_k=3.0):
    """Median absolute depth change at identity, in metres (LowerIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[0]


def pair_identity_median_depth_change_mask(d0_path, d1_path, change_mask_k=3.0):
    """255 where IdentityMedianDepthChange recommends dropping a pixel.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[1]


def _joint_valid_depth_ratio(d0_path, d1_path):
    """Fraction of pixels valid in BOTH frames, plus the not-joint drop mask.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    d0, v0 = _consumer_depth_metres_valid(d0_path)
    d1, v1 = _consumer_depth_metres_valid(d1_path)
    flagged = np.ones(d0.shape, dtype=bool)
    if d0.shape != d1.shape or d0.size == 0:
        return 0.0, _flag_mask(flagged), 0
    joint = v0 & v1
    count = int(np.count_nonzero(joint))
    return float(count) / float(joint.size), _flag_mask(~joint), count


def pair_joint_valid_depth_ratio(d0_path, d1_path):
    """Jointly-valid depth pixel fraction in [0, 1] (HigherIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _joint_valid_depth_ratio(d0_path, d1_path)[0]


def pair_joint_valid_depth_ratio_mask(d0_path, d1_path):
    """255 where a pixel is NOT valid in both frames.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _joint_valid_depth_ratio(d0_path, d1_path)[1]


def _camera_intrinsics(intrinsics, shape):
    """Accept the capture dict, a (width,height,hfov) tuple, or a 3x3 K."""
    h, w = shape
    arr = np.asarray(intrinsics) if not isinstance(intrinsics, dict) else None
    if arr is not None and arr.shape == (3, 3):
        return float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
    if isinstance(intrinsics, dict):
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        hfov = float(intrinsics["hfov"])
    else:
        width, height, hfov = intrinsics
        width, height, hfov = int(width), int(height), float(hfov)
    if (height, width) != (h, w):
        raise ValueError(
            f"intrinsics ({width}x{height}) do not match depth raster ({w}x{h})")
    fx = fy = (width / 2.0) / np.tan(np.radians(hfov / 2.0))
    return fx, fy, width / 2.0, height / 2.0


def _prior_warp(d0_m, d1_m, prior_T, intrinsics, depth_gate):
    """Warp depth-0 pixels into depth 1 and return residual/support rasters.

    ``prior_T`` maps camera-0 coordinates into camera-1 coordinates.  Residuals
    are stored at source-image coordinates, which makes the resulting drop mask
    directly applicable to frame 0's cloud.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    d0 = np.asarray(d0_m, dtype=np.float64)
    d1 = np.asarray(d1_m, dtype=np.float64)
    if d0.shape != d1.shape or d0.ndim != 2:
        raise ValueError("prior-warp depth rasters must be same-shape HxW arrays")
    gate = float(depth_gate)
    if not np.isfinite(gate) or gate <= 0:
        raise ValueError("priorWarpDepthGate must be finite and > 0 metres")
    fx, fy, cx, cy = _camera_intrinsics(intrinsics, d0.shape)
    transform = np.asarray(prior_T, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("prior_T must be a finite 4x4 transform")

    source_valid = d0 > 0
    y, x = np.nonzero(source_valid)
    residual = np.full(d0.shape, np.nan, dtype=np.float64)
    support = np.zeros(d0.shape, dtype=bool)
    projected = np.zeros(d0.shape, dtype=bool)
    if len(x) == 0:
        return residual, support, projected

    z = d0[y, x]
    xyz1 = np.vstack([(x - cx) * z / fx, (y - cy) * z / fy, z,
                      np.ones(len(z), dtype=np.float64)])
    warped = transform @ xyz1
    wz = warped[2]
    in_front = wz > 0
    u = np.rint(fx * warped[0] / np.where(in_front, wz, 1.0) + cx).astype(int)
    v = np.rint(fy * warped[1] / np.where(in_front, wz, 1.0) + cy).astype(int)
    h, w = d0.shape
    inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    src_y, src_x = y[inside], x[inside]
    dst_y, dst_x = v[inside], u[inside]
    warped_z = wz[inside]
    target_z = d1[dst_y, dst_x]
    target_valid = target_z > 0
    src_y, src_x = src_y[target_valid], src_x[target_valid]
    warped_z, target_z = warped_z[target_valid], target_z[target_valid]
    if len(src_x) == 0:
        return residual, support, projected

    r = np.abs(warped_z - target_z)
    projected[src_y, src_x] = True
    residual[src_y, src_x] = r
    # A point substantially behind the observed surface is occluded and is not
    # evidence about the prior residual distribution.  Points in front remain
    # contributing outliers and are exactly the ones the filter can remove.
    visible = warped_z <= target_z + gate
    support[src_y[visible], src_x[visible]] = True
    return residual, support, projected


def _prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics, depth_gate):
    """Median prior-warp depth residual in metres, plus drop mask and count.

    STUDENT IMPLEMENTATION target (instruction.md §5.2) — stripped to a
    `#TODO` stub in the student-facing distribution.
    """
    d0, _ = _consumer_depth_metres_valid(d0_path)
    d1, _ = _consumer_depth_metres_valid(d1_path)
    if d0.shape != d1.shape:
        return float("inf"), _flag_mask(np.zeros(d0.shape, dtype=bool)), 0
    residual, support, projected = _prior_warp(
        d0, d1, prior_T, intrinsics, depth_gate)
    count = int(np.count_nonzero(support))
    if count == 0:
        return float("inf"), _flag_mask(np.zeros(d0.shape, dtype=bool)), 0
    value = float(np.median(residual[support]))
    flagged = projected & np.isfinite(residual) & (residual > float(depth_gate))
    return value, _flag_mask(flagged), count


def pair_prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics,
                                   prior_warp_depth_gate=0.10):
    """Median depth residual under the constant-velocity prior (LowerIsBetter).

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[0]


def pair_prior_warp_depth_residual_mask(d0_path, d1_path, prior_T, intrinsics,
                                        prior_warp_depth_gate=0.10):
    """255 where the prior-warp residual exceeds the declared depth gate.

    Implemented BY STUDENTS (instruction.md §5.2): the reference body ships
    only so the pipeline can be tested end to end; the handout strips it to a
    `#TODO` stub. Exact contract: definitions.md and the factor tests.
    """
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[1]


# =============================================================================
# Frame pairing  (rgb/*.png <-> depth/*.png by integer stem, iterate sorted by int)
# =============================================================================
def _stem(path):
    """Integer filename stem of a frame path (e.g. '.../17.png' -> 17)."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _pair_frames(data_dir):
    """Return [(stem_str, rgb_path, depth_path), ...] paired by int stem, sorted by int.

    `data_dir` must contain `rgb/` and `depth/` subdirs of integer-stem .png frames.
    Only stems present in BOTH subdirs are yielded. A `semantic/` subdir, if
    present, is ignored.
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
# IRI helpers — the frozen scheme of §2, in ONE place
#   Every module that needs an IRI of this assignment calls a function from this
#   section. Nobody, in any file, writes an f-string with `batch/` in it, and
#   nobody writes `str(iri).split("/")[-1]`: the scheme is a contract between
#   `batch2ttl` (which mints the structural IRIs), `experiment` (which measures
#   those subjects) and `reconstruct.py` (which reads them back), and a contract
#   duplicated across three files is a contract that will disagree with itself the
#   first time anyone renames a segment.
#
#   TWO TIERS, AND THE RULE THAT MUST NEVER BE RELAXED (§2).
#   Batch, frame and image IRIs are SHARED STRUCTURE: they name pixels on disk, so
#   every experiment over one capture reaches the same nodes. Everything an
#   experiment mints — settings, annotations, pairs, runs — is EXPERIMENT-SCOPED,
#   i.e. hangs under `<ns>experiment/<expname>/`. That scoping is what replaced v1's
#   named graphs: two experiments over one batch produce two disjoint annotation /
#   pair / run sets over the SAME frame IRIs, so nothing collides and nothing is
#   overwritten. A loader can forget a `to_graph` argument; it cannot forget the
#   experiment name, because the name is inside the subject.
# =============================================================================

# The two modality segments of an annotation IRI and of `component_iri` — frozen
# (§2). They are the SAME two strings on purpose: an annotation's
# `<kind>` names the image node it describes, so `annotation_iri(name, n, kind)`
# and `component_iri(batch, n, kind)` line up segment for segment.
_ANNOTATION_KINDS = ("rgb", "depth")

# An experiment name is the declaration file's stem AND the IRI tail (§2/§6), so
# it has to survive both a filesystem and an IRI without quoting: letters, digits,
# underscore and hyphen. A name with a slash in it would silently restructure the
# IRI scheme; a name with a space in it would not survive a Turtle IRI at all.
_EXPNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def batch_name(data_dir, floor):
    """Floor-qualified batch name: f"floor{floor}_{basename(data_dir)}".

    Both floors ship captures with the same directory names, so a basename alone
    is not a unique key: floor 1's `mixed_dev` and floor 2's `mixed_dev` would map
    to ONE batch IRI and silently overwrite each other — and comparing the two
    floors is exactly what phase 2 asks you to do.  This name is what goes into
    hw1:batchName and into the batch IRI.
    """
    return f"floor{int(floor)}_{os.path.basename(os.path.normpath(data_dir))}"


def batch_iri(name):
    """IRI of a batch node: <ns>batch/<name>."""
    return URIRef(f"{NS}batch/{name}")


def frame_iri(name, idx):
    """IRI of a frame node: <ns>batch/<name>/frame/<n>, `n` decimal and unpadded.

    Unpadded is a decision, not an oversight: `frame/7` and `frame/007` are
    different IRIs, so a padded writer and an unpadded reader would build two
    disjoint graphs that look identical in a listing. `_stem` returns an int and
    everything downstream formats that int, so there is one spelling per frame.
    """
    return URIRef(f"{NS}batch/{name}/frame/{idx}")


def component_iri(name, idx, kind):
    """IRI of a frame's rgb or depth image node: <ns>batch/<name>/frame/<n>/<kind>.

    `kind` is "rgb" or "depth". The image node is a separate subject from the
    frame because the two rasters are two files with two paths, and a query that
    wants the picture wants one of them, not both.

    IT IS NOT THE OBSERVATION, WHICH IS A CHANGE FROM v1 (§4.3). In
    v1 the observables hung here: clipHiFraction on the RGBImage,
    validDepthFraction on the DepthImage, scoped by the experiment's named graph.
    With one graph that shape is broken — two experiments would write two
    contradictory hw1:clipHiFraction triples onto one image node. So an image node
    now carries `schema:contentUrl` and NOTHING ELSE, and the numbers live on the
    experiment-scoped FrameAnnotation (see `annotation_iri`).
    """
    return URIRef(f"{NS}batch/{name}/frame/{idx}/{kind}")


def generation_setting_iri(name, param):
    """IRI of a batch's generation setting: <ns>batch/<name>/setting/<param>.

    BATCH-scoped, not experiment-scoped, and that placement is the whole point of
    the role (§4.2): a GenerationSetting describes how the PIXELS were
    produced, so it survives every re-measurement of them and no experiment
    re-asserts it. `<param>` is the parameter's local name, which is globally
    unique across factors (§2), so no factor segment is needed here.
    """
    return URIRef(f"{NS}batch/{name}/setting/{param}")


def experiment_iri(expname):
    """IRI of an experiment: <ns>experiment/<expname>.

    Every node the experiment mints hangs under this prefix — settings,
    annotations, pairs, runs — and that is the entirety of the isolation
    mechanism. v1 ALSO loaded each experiment's triples into a named graph of this
    same IRI; there are no named graphs any more (§2), because Turtle
    cannot carry a graph name and one missed `to_graph` produced a silent zero-row
    query against two perfectly well-formed files.

    SIGNATURE CHANGED IN v3 (§6/§8): the argument is the experiment's
    NAME — the stem of the declaration file, matching [A-Za-z0-9_-]+ — not an
    8-hex digest. The digest is deleted: it identified a treatment, but the file
    was machine-named and therefore unreadable, and write-once (§3.1) now buys
    what the digest bought. The STUDENT writes this IRI in the declaration and
    `experiment` checks its tail against the file stem rather than picking one.
    """
    return URIRef(f"{NS}experiment/{expname}")


def setting_iri(expname, factor_local, param_local):
    """IRI of one experiment's setting of one parameter:
    <ns>experiment/<expname>/setting/<factor>/<param>.

    `factor_local` is the local name of the parameter's hw1:paramPrimaryFactor,
    `param_local` the parameter's own local name (§2).

    ONLY MACHINE-MINTED SETTINGS USE THIS. A student's FactorSettings may be blank
    nodes or any IRI at all (§2/§4.5) — a machine pass cannot re-open
    somebody else's blank node, and it does not need to: attribution traverses
    setting -> settingParameter -> paramPrimaryFactor / paramAffectsFactor in the
    TBox. What this function names are the DEFAULTS that `experiment` records
    below the marker to complete the required set (§4.2).

    WHY THE FACTOR SEGMENT, GIVEN THAT PARAMETER NAMES ARE ALREADY UNIQUE
        Not for disambiguation — `<expname>/setting/tauHi` would be unique on its
        own. It is there so the IRI reads as the sentence `explore` prints: "under
        HighlightClipping, tauHi was 250". Only the PRIMARY factor appears, and in
        v3 hw1:settingForFactor is single-valued for the same reason (§4.5).

    WHY SETTINGS ARE EXPERIMENT-SCOPED RATHER THAN TBox INDIVIDUALS
        A threshold is a per-experiment LEVEL, not a global Band. Measure one
        batch twice at two tauHi values and there must be two setting nodes; if the
        IRI did not carry the experiment name they would be one node with two
        contradictory hw1:settingValue triples — precisely the collision named
        graphs used to prevent, reappearing in the settings instead of the values.
    """
    return URIRef(f"{NS}experiment/{expname}/setting/{factor_local}/{param_local}")


def annotation_iri(expname, idx, kind):
    """IRI of one experiment's annotation of one frame's ONE MODALITY:
    <ns>experiment/<expname>/annotation/<n>/<kind>, `kind` in "rgb" | "depth".

    SIGNATURE CHANGED IN v3 (§2/§8): the `<kind>` segment is new.
    Annotations are PER MODALITY now, because selection is: an experiment that
    evaluates only depth factors has nothing to say about the rgb raster, and a
    single per-frame node would have had to carry either a hole or a fiction. A
    modality with no selected factor gets NO node at all (§4.3), which is what
    makes "a missing property is a bug, not a state" survive selection.

    `n` is the frame's integer stem, decimal and unpadded, exactly as in
    `frame_iri` — the annotation and the frame it annotates are numbered the same
    way so a human can read one off the other.

    The annotation, not the image node, is the OBSERVATION (§4.3): it
    carries the values and their statuses, and its IRI carries the experiment
    name. That is what satisfies RDF Data Cube IC-12 without named graphs — assess
    one batch under two declarations and you get two annotation sets whose
    `hw1:annotatesFrame` objects are the SAME frame IRIs, rather than two
    contradictory triples on one subject.
    """
    if kind not in _ANNOTATION_KINDS:
        raise ValueError(
            f"annotation kind {kind!r} is not one of {', '.join(_ANNOTATION_KINDS)}; "
            f"the <kind> segment of an annotation IRI is frozen (§2)")
    return URIRef(f"{NS}experiment/{expname}/annotation/{idx}/{kind}")


def pair_iri(expname, i, j):
    """IRI of one experiment's frame pair: <ns>experiment/<expname>/pair/<i>_<j>.

    SIGNATURE CHANGED IN v2 (§8). This used to be
    `pair_iri(batch_name, i, j)` over a batch-scoped node, because `batch2ttl` wrote
    FramePair skeletons that every experiment then decorated. It no longer does: a
    pair is two links and an index, so caching it in the batch file bought nothing
    and forced two experiments to write contradictory observables onto one node.

    `i` is the EARLIER stem. The pair is ORDERED — hw1:sourceFrame is the ICP
    source, hw1:targetFrame the target — so `<i>_<j>` and `<j>_<i>` are different
    nodes and only the ascending one is ever minted.

    STEMS, NOT ORDINALS, IN THE IRI. Pair `41_43` is a legitimate pair over a gap
    in the capture (see `build_batch_graph`, which warns about exactly that), and
    naming the node by position would hide that gap inside a tidy-looking index.
    The ordinal is carried separately by hw1:pairIndex, where a query can
    ORDER BY it as an integer instead of parsing an IRI (§2).
    """
    return URIRef(f"{NS}experiment/{expname}/pair/{i}_{j}")


def run_iri(expname, mode):
    """IRI of a reconstruction run: <ns>experiment/<expname>/run/<mode>.

    `mode` is "baseline" (hw1:FullBatch), "selected" (hw1:GoodSegments), or
    "masked" (hw1:MaskFiltered). The runs of one experiment differ only in this
    segment, which is what makes the
    baseline-vs-selected comparison — the deliverable — impossible to lose by
    collision. v1 minted an entire derived experiment (`<id>-sel-<6hex>`) for the
    selected run; §6 deletes that machinery, because all run modes
    share a batch and setting vector and therefore belong to one experiment.
    """
    return URIRef(f"{NS}experiment/{expname}/run/{mode}")


def frame_index_from_iri(iri):
    """Frame OR annotation IRI -> its integer stem. THE tail parse; there is no second.

    `reconstruct.py` needs integer stems and reads only an experiment file, but
    `hw1:frameIndex` lives in the BATCH graph — so the stem has to come out of the
    IRI. That is open item O8, resolved in §9 in favour of the tail
    parse WITH the parse confined to this function. Import it; do not re-derive
    it. If the scheme ever changes, this function and `batch_name_from_frame_iri`
    are the only two places that have to notice.

    TWO SHAPES, ONE IMPLEMENTATION (§8): the frame IRI
    `<ns>batch/<name>/frame/<n>` and — new in v3, because annotations are
    per modality — the annotation IRI `<ns>experiment/<expname>/annotation/<n>/<kind>`.
    Both carry the same `<n>`, and a second `split("/")` somewhere else for the
    second shape is exactly the duplication this function exists to prevent.

    Strict on purpose: `n` must be a decimal integer and the annotation's `<kind>`
    must be one of the two frozen modality segments. A component IRI
    (.../frame/7/depth) and anything outside the scheme raise ValueError rather
    than returning a plausible-looking number, because a frame list silently short
    by the frames it could not parse is the kind of bug that shows up as a
    slightly worse reconstruction score and never as an error.
    """
    text = str(iri)
    expected = (f"{NS}batch/<name>/frame/<n> or "
                f"{NS}experiment/<expname>/annotation/<n>/<kind>")
    if text.startswith(f"{NS}batch/") and "/frame/" in text:
        tail = text.split("/frame/", 1)[1]
    elif text.startswith(f"{NS}experiment/") and "/annotation/" in text:
        tail, _, kind = text.split("/annotation/", 1)[1].partition("/")
        if kind not in _ANNOTATION_KINDS:
            raise ValueError(
                f"annotation IRI {text!r} ends in modality segment {kind!r}; "
                f"expected one of {', '.join(_ANNOTATION_KINDS)} ({expected})")
    else:
        raise ValueError(
            f"not a frame or annotation IRI of this assignment: {text!r} "
            f"(expected {expected})")
    if not tail.isdigit():
        raise ValueError(
            f"IRI tail {tail!r} is not a decimal frame index in {text!r} "
            f"(expected {expected})")
    return int(tail)


def batch_name_from_frame_iri(iri):
    """Frame IRI -> the batch name embedded in it. The other half of the O8 parse.

    Used to check that the experiment you are about to score was measured on the
    capture you are about to reconstruct: scoring one batch and writing the number
    into another batch's experiment is silent, unrecoverable, and takes one
    mistyped `--data_root`.
    """
    text = str(iri)
    prefix = f"{NS}batch/"
    marker = "/frame/"
    if not text.startswith(prefix) or marker not in text:
        raise ValueError(
            f"not a frame IRI of this assignment: {text!r} "
            f"(expected {NS}batch/<name>/frame/<n>)")
    return text[len(prefix):].split(marker, 1)[0]


def _batch_name_from_batch_iri(iri):
    """Batch IRI -> its <name>. Private: batch IRIs are not parsed outside this file."""
    text = str(iri)
    prefix = f"{NS}batch/"
    if not text.startswith(prefix):
        raise ValueError(f"not a batch IRI of this assignment: {text!r} "
                         f"(expected {NS}batch/<name>)")
    return text[len(prefix):]


def _experiment_name_from_iri(iri, path=None):
    """Experiment IRI -> its <expname>. The v3 replacement for the 8-hex id parse.

    Everything the assessment mints hangs under this name (§2), so `write_run`
    mints its run IRI from it and `read_experiment` reports it. The name is
    checked against `_EXPNAME_RE` here rather than only at declaration time,
    because a hand-edited Experiment IRI with a slash in its tail would otherwise
    mint `…/experiment/a/b/run/baseline` — a well-formed IRI under a different
    scheme, joined to nothing, reported by nothing.
    """
    text = str(iri)
    prefix = f"{NS}experiment/"
    where = f"{path}: " if path else ""
    if not text.startswith(prefix):
        raise ValueError(f"{where}not an experiment IRI of this assignment: {text!r} "
                         f"(expected {NS}experiment/<expname>)")
    name = text[len(prefix):]
    if not _EXPNAME_RE.match(name):
        raise ValueError(
            f"{where}experiment IRI tail {name!r} is not a legal experiment name "
            f"(§2: [A-Za-z0-9_-]+, and it must equal the declaration "
            f"file's stem)")
    return name


# Backwards-compatible aliases. The two helpers were private (`_frame_iri`,
# `_component_iri`) while this file was the only caller; §9 makes
# them public because reconstruct.py and the tests import them. The old names
# stay bound so nothing that already imports them breaks on the rename.
_frame_iri = frame_iri
_component_iri = component_iri


# =============================================================================
# The TBox as the authority on names
#   Parameter names, factor names, polarities and thresholds are NOT string
#   constants in this file. They are read out of ontology/hw1.ttl at run time, so
#   the ontology is a thing the code OBEYS rather than a document that describes
#   it. That is the whole argument for having a TBox in a pipeline this small:
#   a declared `hw1:settingParameter hw1:tuaHi` is a hard error with the declared
#   list printed, not a seventeenth FactorSetting nobody ever reads recording a
#   treatment nobody ran.
#
#   Three loaders, one status rule: `load_parameter_declarations` (what may be
#   set, and how it parses), `load_quality_factors` (what is measured, which way
#   is better, and which parameter is its threshold) and `status_for` (the ONE
#   implementation of the §4.5 Pass rule — `experiment` calls it for
#   frame and pair observables, `write_run` for mapMeanL2 and coverageF, and
#   nobody anywhere re-derives `>=` versus `<=`).
# =============================================================================
_SETTING_ROLES = ("GenerationSetting", "MeasurementSetting", "QualificationSetting")
_VALUE_KINDS = ("double", "integer", "string")

# The two roles a setting may have ON AN EXPERIMENT, and therefore the two roles
# the selection-scoped completeness rule of §4.2 draws from. Generation
# is deliberately absent: those settings live on the Batch, they describe the pixels,
# and they are never defaulted — see `_resolve_generation_settings` and
# `read_declaration`, which rejects a Generation parameter in a declaration outright.
_EXPERIMENT_ROLES = ("MeasurementSetting", "QualificationSetting")


def _local(term):
    """Local name of a term in the hw1 namespace ('...#tauHi' -> 'tauHi')."""
    text = str(term)
    if text.startswith(NS):
        return text[len(NS):]
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _storable(value):
    """A double, rounded to the precision the .ttl file can actually hold.

    READ THIS BEFORE "SIMPLIFYING" IT AWAY. rdflib's Turtle serializer does not
    honour the lexical form of an xsd:double at all: it re-derives one from the
    value as `f"{float(v):e}"` — SEVEN significant digits — and then strips the
    mantissa's trailing zeros (rdflib 7.1.4, `term.Literal._literal_n3`). So
    `Literal("0.20000000000000018", datatype=XSD.double)` and `Literal(0.2, ...)`
    are the identical five bytes `2e-01` on disk, and a value read back out of the
    file is NOT the value that went in.

    That matters here and nowhere else in this project, because v2 bakes a verdict
    next to every value (§4.5) and calls the verdict "a denormalised
    cache whose truth is (value + settings)". Grade the in-memory 0.20000000000000018
    against a threshold of 0.2 and you write `Fail`; the file then reads
    `hw1:medianDepthDifference 2e-01 ; hw1:medianDepthDifferenceStatus hw1:Fail`,
    which is a self-contradicting record, and a student re-grading the stored value
    by hand gets the opposite answer. Depth
    rasters are quantised to the millimetre, so medians land exactly on round
    threshold values often — this is not a hypothetical.

    Rounding here makes the stored number and the graded number THE SAME NUMBER.
    The cost is 1e-7 relative on a measurement whose file only ever held 7 digits
    anyway; the gain is that the file is internally consistent and re-grading a
    stored value always reproduces the stored verdict. Non-finite values pass
    through untouched: `f"{inf:e}"` is `"inf"`, which floats back to `inf`.

    §5 asks for the `.17g` digest repr as the lexical form of a
    settingValue. That is not achievable through rdflib (see above) and the
    property §5 wanted — one value, one lexical form, so `STR(?value)` comparison
    is exact — holds anyway, because the written form is a pure function of the
    value. Report it if a future serializer makes the full form reachable.
    """
    return float(f"{float(value):e}")


def _double_literal(value):
    """One xsd:double literal, the one way (§4.6).

    Every double this project writes — observable values, thresholds, run results —
    goes through here, so `INF` / `-INF` / `NaN` are spelled the same way
    everywhere and every number is stored at the precision it was graded at (see
    `_storable`). `xsd:decimal` is not an option for any of them precisely because
    it cannot carry INF, and the measurers are fail-closed: they return INF or 0.0
    rather than raising (§9), so a non-finite value is a NORMAL
    outcome here, not an error path.

    THE NON-FINITE LEXICAL FORMS ARE WRITTEN OUT BY HAND, AND THAT IS NOT
    BELT-AND-BRACES. XSD 1.1 admits exactly `INF`, `-INF` and `NaN` in the lexical
    space of xsd:double; Python's `str(float('inf'))` is `'inf'`, which is NOT in
    it. Hand rdflib a float and it stores `'inf'` as the literal's lexical form —
    `str(Literal(float("inf"), datatype=XSD.double))` really is `'inf'`, and so is
    `str(Literal("INF", datatype=XSD.double))`, because rdflib normalises the
    lexical form it was given back through `str(float)`. The Turtle file comes out
    correct anyway, but ONLY because rdflib's serializer patches it at the last
    moment with `encoded.replace("inf", "INF")` (7.1.4, `term.Literal._literal_n3`).

    Depending on that fixup is not acceptable for a value whose whole job is to be
    seen. An ill-typed literal does not raise in SPARQL — it makes every comparison
    on it FALSE, so a strict engine reading `"inf"^^xsd:double` would silently drop
    exactly the fail-closed rows the convention exists to surface, and the symptom
    would be a short result set rather than an error. `normalize=False` keeps the
    XSD-valid spelling we passed in, so the literal is correct in memory as well as
    on disk and stays correct under any serializer.

    Finite values still go in as floats: for those, rdflib re-derives the lexical
    form from the value regardless of what it is handed (see `_storable`), so
    spelling one out here would buy nothing and could drift from what is written.
    """
    v = _storable(value)
    # NaN is the only value that is not equal to itself — the cheapest exact test,
    # and it avoids importing `math` for three comparisons.
    if v != v:
        return Literal("NaN", datatype=XSD.double, normalize=False)
    if v == float("inf"):
        return Literal("INF", datatype=XSD.double, normalize=False)
    if v == float("-inf"):
        return Literal("-INF", datatype=XSD.double, normalize=False)
    return Literal(v, datatype=XSD.double)


def _setting_value_literal(value, kind):
    """One `hw1:settingValue` literal, typed per the parameter's declared kind.

    §5: "double" -> xsd:double, "integer" -> xsd:integer, "string" ->
    xsd:string. The KIND comes from the TBox DECLARATION, never from the Python type
    of `value` — the same rule `parse_setting_arg` follows, and the reason
    `icpBackend` is an ordinary parameter rather than a special case. Guessing from
    the type would let the string "1" arrive as the number 1 somewhere between the
    command line and the file, and nothing downstream could tell.

    xsd:string is written EXPLICITLY rather than as a plain literal, because rdflib
    does not treat the two as equal — the same trap §5 warns about for
    `hw1:paramDefault`, which the TBox types as `"open3d"^^xsd:string`. Writing the
    datatype means an rdflib-side comparison of a setting against its default is a
    comparison of two terms of one type.
    """
    if kind == "string":
        return Literal(str(value), datatype=XSD.string)
    if kind == "integer":
        return Literal(int(value), datatype=XSD.integer)
    return _double_literal(value)


def load_parameter_declarations(path=_ONTOLOGY_TTL):
    """Read every declared `hw1:Parameter` out of the TBox. (Renamed in v2.)

    CONTRACT
        In:     path — the TBox Turtle (default hw1/ontology/hw1.ttl).
        Out:    {local_name: {"iri":         URIRef,
                              "role":        "GenerationSetting"
                                             | "MeasurementSetting"
                                             | "QualificationSetting",
                              "kind":        "double" | "integer" | "string",
                              "primary":     factor local name,
                              "primaryIri":  URIRef,
                              "affects":     (factor local name, ...),   # sorted
                              "affectsIris": (URIRef, ...),              # same order
                              "default":     float | int | str | None}}

        read from `hw1:paramRole`, `hw1:paramValueKind`, `hw1:paramPrimaryFactor`,
        `hw1:paramAffectsFactor` and `hw1:paramDefault` (§5).

    WHAT EACH FIELD DECIDES
        * `role` decides WHERE the setting lives and WHAT THE VERDICT IS. Generation
          settings hang off the Batch (`hw1:hasGenerationSetting`) and are applied by
          nothing — the pixels already embody them; Measurement and Qualification
          settings hang off the Experiment (`hw1:hasFactorSetting`). The same field
          is the fix instruction `explore`'s verdict section returns: Generation =>
          regenerate the data, Measurement => change the number and re-measure,
          Qualification => change the threshold and re-qualify (and the last two mean
          a NEW declaration, §3.1). v1 had two classes here; v2 has one class and
          three roles.
        * `kind` decides PARSING and the datatype of `hw1:settingValue` —
          xsd:double / xsd:integer / xsd:string. Declared per parameter rather than
          guessed from the text, which is what makes `icpBackend` an ordinary
          parameter instead of a special case: `hw1:settingValue "1"` for icpBackend is
          the STRING "1", not the number 1.
        * `primary` decides the SETTING IRI and the DIGEST LINE (§2/§6),
          which is why the TBox declares it single-valued and mandatory: an ambiguous
          primary factor would make one treatment hash two ways. `affects` never
          appears in an IRI or a digest; it exists so `hw1:settingForFactor` can be
          total and attribution can blame `brightnessGain` for a ShadowClipping
          failure whose primary factor is HighlightClipping.
        * `default` decides COMPLETENESS (§5). Every Measurement and
          Qualification parameter is recorded on every experiment — given on the
          command line, or filled from `hw1:paramDefault` and recorded EXPLICITLY —
          because an experiment that omits `tauHi` is otherwise indistinguishable
          from one that set it to 250, and then the digest identifies nothing. The
          two Generation parameters carry NO default on purpose and get the opposite
          treatment: absence means "not asserted", not "identity".

    READING A DEFAULT: `.toPython()`, NEVER AN EQUALITY AGAINST A BARE Literal
        §5 types every `hw1:paramDefault` explicitly, including the
        string-valued one (`"open3d"^^xsd:string`). rdflib does not treat a plain
        literal and an xsd:string-typed literal as equal, so
        `default_term == Literal("open3d")` is False for the very declaration it is
        checking. Cast through `.toPython()` and compare values, not terms.

    Raises ValueError on any declaration this code would otherwise have to guess
    at: an unknown role or value kind, a missing/multiple primary factor, or a
    default that will not parse as its declared kind. An ambiguous TBox is a TBox
    bug, and guessing which half to believe just moves the bug downstream into a
    file that looks fine.
    """
    g = Graph()
    g.parse(path, format="turtle")
    decls = {}
    for param in g.subjects(RDF.type, HW1.Parameter):
        local = _local(param)

        role_term = g.value(param, HW1.paramRole)
        role = _local(role_term) if role_term is not None else None
        if role not in _SETTING_ROLES:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramRole {role!r}; "
                f"expected one of {', '.join(_SETTING_ROLES)}")

        kind_term = g.value(param, HW1.paramValueKind)
        kind = str(kind_term) if kind_term is not None else None
        if kind not in _VALUE_KINDS:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramValueKind {kind!r}; "
                f"expected one of {', '.join(_VALUE_KINDS)}")

        # Single-valued and mandatory: it is a segment of the setting IRI and of
        # the digest line, so "several" and "none" are equally unusable.
        primaries = sorted(g.objects(param, HW1.paramPrimaryFactor), key=str)
        if len(primaries) != 1:
            raise ValueError(
                f"{path}: parameter {local!r} has {len(primaries)} "
                f"hw1:paramPrimaryFactor values; exactly one is required "
                f"(§5) because it names the setting IRI and the digest line")
        affects = sorted((URIRef(str(a)) for a in g.objects(param, HW1.paramAffectsFactor)),
                         key=str)

        default_term = g.value(param, HW1.paramDefault)
        if default_term is None:
            default = None
        else:
            raw = default_term.toPython() if hasattr(default_term, "toPython") \
                else str(default_term)
            try:
                default = {"double": float, "integer": int, "string": str}[kind](raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: parameter {local!r} declares hw1:paramDefault "
                    f"{str(default_term)!r}, which is not a {kind}") from None

        decls[local] = {
            "iri": URIRef(str(param)),
            "role": role,
            "kind": kind,
            "primary": _local(primaries[0]),
            "primaryIri": URIRef(str(primaries[0])),
            "affects": tuple(_local(a) for a in affects),
            "affectsIris": tuple(affects),
            "default": default,
        }
    return decls


def parse_setting_arg(arg, decls):
    """Parse one `--gen NAME=VALUE` string. -> (name, value).

    The value is typed by the DECLARATION, never by the text (§5): a
    "double" parameter yields a float, an "integer" parameter an int, a "string"
    parameter the text verbatim. Whitespace around the NAME is stripped; the value
    is not touched, because a string level is data.

    An undeclared name is FATAL, with the declared list printed. That is the point:
    a typo surviving as an extra FactorSetting gives the batch a provenance record
    that describes a treatment nobody ran, and every number measured over it then
    looks perfectly plausible. (`--set` is deleted with v2, §7; the
    declaration carries settings now and `read_declaration` applies the same rule
    to them. This function survives for `--gen`, which is still a command-line
    flag because a Generation setting describes pixels, not an experiment.)

    An "integer" parameter is parsed STRICTLY — `someCount=20.5` is an error, not
    a silent truncation: `int(20.5)` would record 20 while the shell history says
    20.5. No parameter currently declares that kind (`minSegmentLength`, the only
    one that ever did, was deleted on 2026-07-31), and the rule stays anyway —
    the kinds come from the TBox, so the next count-valued parameter inherits it.
    """
    if "=" not in arg:
        raise ValueError(f"a setting expects NAME=VALUE, got {arg!r}")
    name, _, text = arg.partition("=")
    name = name.strip()
    if name not in decls:
        raise ValueError(
            f"undeclared parameter {name!r}. Parameter names come from the TBox, not "
            f"from the command line; declared parameters are: "
            f"{', '.join(sorted(decls)) or '(none)'}")
    kind = decls[name]["kind"]
    if kind == "string":
        return name, text
    try:
        return name, (int(text) if kind == "integer" else float(text))
    except (TypeError, ValueError):
        raise ValueError(
            f"parameter {name!r} is declared hw1:paramValueKind {kind!r}; "
            f"{text!r} does not parse as one") from None


def load_quality_factors(path=_ONTOLOGY_TTL):
    """Read every declared `hw1:QualityFactor` out of the TBox (§8.0).

    Returns {factor_local: {"over":        value property local name,
                            "status":      status property local name,
                            "polarity":    "higher" | "lower",
                            "qualifiedBy": parameter local name}}

    from `hw1:overProperty` / `hw1:statusProperty` / `hw1:polarity` /
    `hw1:qualifiedBy`. Those four links are what make the whole grading path
    generic: `status_for` reads the polarity and the threshold parameter instead of
    branching per factor, and `explore`'s verdict section finds failing factors by
    following `hw1:statusProperty` to the predicate and looking for hw1:Fail, without
    naming one predicate.

    STATUS PREDICATES ARE READ FROM HERE, NEVER BUILT BY CONCATENATION. The local
    name of a status property is always the value property's plus "Status", and
    that is exactly why the rule must not be coded: a factor added to the TBox with
    a status property named any other way would silently stop being graded, and the
    failure would look like "that factor always passes".

    All four links are REQUIRED. A factor missing its polarity or its threshold
    parameter cannot be graded, and returning None there would push a `None`
    comparison into `status_for`, where the only available behaviours are "raise
    somewhere less informative" and "invent a verdict". v1 tolerated the absence
    because grading happened at query time against Bands that might simply not
    match; v2 bakes the verdict, so the TBox has to be complete.
    """
    g = Graph()
    g.parse(path, format="turtle")
    out = {}
    for f in g.subjects(RDF.type, HW1.QualityFactor):
        local = _local(f)
        over = g.value(f, HW1.overProperty)
        status = g.value(f, HW1.statusProperty)
        pol = g.value(f, HW1.polarity)
        qual = g.value(f, HW1.qualifiedBy)
        missing = [n for n, t in (("hw1:overProperty", over),
                                  ("hw1:statusProperty", status),
                                  ("hw1:polarity", pol),
                                  ("hw1:qualifiedBy", qual)) if t is None]
        if missing:
            raise ValueError(
                f"{path}: hw1:QualityFactor {local!r} is missing {', '.join(missing)}. "
                f"All four links are required (§4.5): without them the "
                f"factor cannot be graded, and a factor that cannot be graded reads as "
                f"a factor that always passes.")
        polarity = {"HigherIsBetter": "higher", "LowerIsBetter": "lower"}.get(_local(pol))
        if polarity is None:
            raise ValueError(
                f"{path}: factor {local!r} declares hw1:polarity {_local(pol)!r}; "
                f"expected hw1:HigherIsBetter or hw1:LowerIsBetter")
        out[local] = {"over": _local(over), "status": _local(status),
                      "polarity": polarity, "qualifiedBy": _local(qual)}
    return out


def status_for(value_property_local, value, settings, factors=None):
    """The ONE implementation of the §4.5 status rule. -> Pass | Fail.

    CONTRACT
        In:     value_property_local — the local name of the observable value
                property being graded ("clipHiFraction", "mapMeanL2", ...).
                value — the raw measured double, exactly as it will be STORED.
                settings — {param_local: value}, the setting vector recorded on the
                same experiment; the threshold is looked up in here and nowhere
                else.
                factors — `load_quality_factors()` output; loaded if omitted.
        Out:    HW1.Pass or HW1.Fail.

        With `t` the settingValue of the factor's `hw1:qualifiedBy` parameter:

            hw1:HigherIsBetter    Pass iff value >= t
            hw1:LowerIsBetter     Pass iff value <= t

    WHY THE NON-FINITE CASES NEED NO SPECIAL CASE
        IEEE comparison already does the right thing, and writing a branch for it
        would be a chance to get it wrong. Every comparison with NaN is False, so
        NaN falls through to Fail under either polarity — which is the frozen rule
        ("NaN is always Fail"). The fail-closed measurer sentinels grade themselves:
        INF fails a lower-is-better factor because `inf <= 0.20` is False, and 0.0
        fails a higher-is-better factor because `0.0 >= 0.10` is False. The
        BOUNDARY passes under both polarities, deliberately: the thresholds are
        course-calibrated round numbers, and a value sitting exactly on one is
        inside the interval the number was chosen to describe.

    WHY A MISSING FACTOR AND A MISSING THRESHOLD BOTH RAISE
        `hw1:meanValue` has no QualityFactor over it, on purpose (§4.3): no polarity, no threshold, nothing to compute a Pass from. Asking for
        its status is a bug in the caller, and answering would quietly promote the
        deliberately-weak baseline to a criterion. A missing THRESHOLD cannot happen
        under the §5 completeness rule at all — every Qualification parameter is
        recorded on every experiment — so if it happens the setting vector is not
        total, and grading against a guessed number would bake a verdict nothing
        can reproduce.

    WHY THE VERDICT IS BAKED INTO THE FILE AT ALL, GIVEN THAT IT IS DERIVED
        It is a denormalised cache whose truth is (value + settings), and the raw
        value is always stored beside it, so re-grading at query time stays
        possible — the stored value and the recorded threshold are both right there
        in the file. What it buys is that reading a verdict is a lookup instead of a
        four-way join against Band individuals — v1's shape, where the tie-break
        between adjacent bands was a correctness trap that produced plausible wrong
        answers.
    """
    factors = load_quality_factors() if factors is None else factors
    over = sorted(k for k, info in factors.items()
                  if info["over"] == value_property_local)
    if not over:
        raise ValueError(
            f"no hw1:QualityFactor is declared over hw1:{value_property_local} in "
            f"{_ONTOLOGY_TTL}, so it has no polarity, no threshold and no status. "
            f"hw1:meanValue is the deliberate case (§4.3); anything else "
            f"is a missing TBox declaration.")
    if len(over) > 1:
        raise ValueError(
            f"{len(over)} quality factors ({', '.join(over)}) are declared over "
            f"hw1:{value_property_local}. One observable has one verdict; two factors "
            f"over it would write two contradictory {value_property_local}Status "
            f"triples onto one node.")
    info = factors[over[0]]

    param = info["qualifiedBy"]
    if param not in settings:
        raise ValueError(
            f"factor {over[0]!r} is qualified by hw1:{param}, which is not in the "
            f"setting vector of this experiment. Every Qualification parameter is "
            f"recorded on every experiment (§5), so this means the vector "
            f"is not total — and guessing a threshold would bake a verdict nobody can "
            f"reproduce.")

    v = float(value)
    t = float(settings[param])
    passing = (v >= t) if info["polarity"] == "higher" else (v <= t)
    return HW1.Pass if passing else HW1.Fail


# =============================================================================
# Experiment files — two sections, one marker, one seal  (§3.1)
#   An experiment file is not machine-generated any more. It is a STUDENT
#   DECLARATION with a machine section appended to it:
#
#       <student turtle: Experiment node, factor selection, settings>
#       # ============ MACHINE SECTION (regenerated by api.py — do not edit) ===
#       <machine turtle: default-filled settings, annotations, pairs, runs>
#
#   Three writers/readers touch that structure and they have to agree to the
#   byte: `cmd_experiment` appends the marker and stamps the seal,
#   `read_experiment` verifies it, `write_run` verifies it and re-serializes the
#   half below the marker. That is why the split and the digest live HERE, once,
#   and why none of them does its own `text.find(...)`.
#
#   v2's DIGEST IDENTITY IS DELETED (§6/§11) — `experiment_digest`,
#   `hw1:experimentId`, `--exp-id` and the `<id>.ttl` filenames with it. An
#   experiment is identified by its NAME now, and what the digest used to buy —
#   "one treatment, one file, nothing overwritten" — is bought by write-once plus
#   the seal below, which is a stronger guarantee: it survives hand editing.
# =============================================================================

# The frozen marker line (§3.1), exported as part of the public
# surface (§8) because the tests and `explore` split files on it too. Matched as
# a WHOLE LINE, em dash and all: a "close enough" spelling would either cut a
# file at a line that is not the marker, or fail to find the marker and report a
# sealed experiment as an unassessed declaration.
MACHINE_MARKER = "# ============ MACHINE SECTION (regenerated by api.py — do not edit) ============"


def _marker_offset(text):
    """Character offset of the START of the marker LINE in `text`, or None.

    "Start of the line" is exactly what the seal is defined over (§3.1: the bytes
    from offset 0 up to the start of the marker line), so this one number is both
    the split point and the digest boundary — they cannot drift apart, because
    there is only one of them.

    The marker must occupy a WHOLE line: it has to begin at offset 0 or just after
    a newline, and end at a newline or at end of file. A declaration that quotes
    the marker inside a longer comment line therefore does not accidentally cut
    itself in half.
    """
    start = 0
    while True:
        idx = text.find(MACHINE_MARKER, start)
        if idx < 0:
            return None
        end = idx + len(MACHINE_MARKER)
        at_line_start = idx == 0 or text[idx - 1] == "\n"
        at_line_end = end == len(text) or text[end] == "\n"
        if at_line_start and at_line_end:
            return idx
        start = idx + 1


def _split_sections(text):
    """File text -> (student_text, machine_text or None). The ONE split.

    `student_text` is everything before the marker line, INCLUDING the newline
    that ends the last student line: it is the exact prefix every machine writer
    writes back unchanged, and the exact byte range the seal covers.
    `machine_text` is everything after the marker line's own newline, or None when
    the file carries no marker at all — i.e. an unassessed declaration, which is
    `explore` view 2 and the only input `experiment` accepts.
    """
    idx = _marker_offset(text)
    if idx is None:
        return text, None
    rest = text[idx + len(MACHINE_MARKER):]
    if rest.startswith("\n"):
        rest = rest[1:]
    return text[:idx], rest


def _declaration_digest(student_bytes):
    """sha256 hex of the student section's bytes. The tamper seal of §3.1.

    THE BYTE BOUNDARY, stated once so the three callers cannot disagree: the
    digest covers `file_bytes[0 : start_of_the_marker_line]` — every byte of the
    declaration INCLUDING the newline that terminates its last line, and NOT the
    '#' that begins the marker. `_split_sections` returns exactly that prefix, so
    the boundary is defined in one place and consumed everywhere.

    Turtle is UTF-8 by specification and this file writes UTF-8, so the character
    prefix `_split_sections` returns and the byte prefix on disk are the same
    thing; callers pass `student_text.encode("utf-8")`.

    WHY A SEAL AT ALL. Write-once (§3.1) is unenforceable without one: nothing
    else can tell an assessed file whose declaration was edited afterwards from
    one that was not, and such a file asserts verdicts for a treatment it no
    longer declares — the most misleading artefact this project could produce,
    because every number in it still looks perfectly plausible.
    """
    return hashlib.sha256(student_bytes).hexdigest()


def _read_text(path):
    """The whole file as text. Turtle is UTF-8 by specification (and so is this)."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _verify_seal(g, exp, student_text, path):
    """Raise unless `hw1:declarationDigest` still matches the student section.

    Called FIRST by `read_experiment` and by `write_run`, before either looks at a
    single value (§8.2). A file whose declaration changed after
    assessment describes one treatment above the marker and reports another below
    it, and no reader can tell which half to believe.
    """
    stored = g.value(exp, HW1.declarationDigest)
    if stored is None:
        raise ValueError(
            f"{path}: experiment {exp} carries no hw1:declarationDigest. Every "
            f"assessed file is sealed by `api.py experiment` (§3.1), "
            f"so a machine section without a seal was not written by this program.")
    actual = _declaration_digest(student_text.encode("utf-8"))
    if str(stored) != actual:
        raise ValueError(
            f"{path}: THE DECLARATION WAS EDITED AFTER ASSESSMENT. The recorded "
            f"hw1:declarationDigest ({str(stored)[:12]}…) does not match the bytes "
            f"above the marker ({actual[:12]}…), so the verdicts below the marker "
            f"were computed for a treatment this file no longer declares. An "
            f"edited declaration IS A NEW EXPERIMENT AND NEEDS A NEW FILE: copy "
            f"the declaration to a new name and run `api.py experiment "
            f"<newname>.ttl` (§3.1 — experiments are write-once).")


# =============================================================================
# Experiment files — the ONE reader and the ONE writer of run triples
#   `read_experiment` (§8.2) is the only code in the project that
#   turns an ASSESSED experiment Turtle into Python, and `write_run` (§8.2) is the
#   only code that emits a hw1:ReconstructionRun triple. reconstruct.py,
#   completeness.py and any other evaluator go through these two: none of them
#   parses Turtle, none of them re-derives the IRI scheme, none of them re-derives
#   the usable-link rule of §4.5, and none of them re-derives the Pass/Fail rule
#   (that one lives in `status_for`, and `write_run` calls it).
#
#   BOTH VERIFY THE SEAL FIRST (§3.1, new in v3), and `write_run`
#   rewrites ONLY below the marker. `read_declaration` — further down, with the
#   `experiment` command — is the reader for the other half of the file: the
#   student's declaration, before any of this exists.
#
#   v1's `write_result` / `hw1:Result` / `load_result_measures` shape is DELETED
#   (§11). A reconstruction outcome is now an ordinary observable —
#   `hw1:mapMeanL2` + `hw1:mapMeanL2Status` on a hw1:ReconstructionRun, graded by
#   the ordinary hw1:ReconstructionAccuracy factor — so nothing in this section
#   is special-cased for results any more.
# =============================================================================
def _sole_experiment(g, path):
    """The single hw1:Experiment subject of `g`. Zero or several is an error.

    Not a guess, not "the first one". An experiment file with two Experiment
    nodes means two different measurement passes were serialized into one graph,
    and every downstream answer — which settings, which statuses, which run a
    number belongs to — becomes a coin flip that nothing reports.
    """
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    if len(exps) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Experiment subject, found {len(exps)}"
            + (f": {', '.join(str(e) for e in exps)}" if exps else ""))
    return exps[0]


def _setting_value_to_python(lit, path, param_local):
    """One hw1:settingValue literal -> float | int | str, typed by ITS DATATYPE.

    The writer stamps the datatype declared by the parameter's `hw1:paramValueKind`
    (§5) onto the literal, so the datatype ON the literal is a
    faithful echo of the TBox declaration and reading it back needs no second trip
    into the ontology. Coercing per datatype rather than sniffing the text is the
    same rule `parse_setting_arg` obeys on the way in, and it matters in both
    directions: an "integer" parameter must come back as an int and never a float,
    and `icpBackend` must come back as the string "open3d" and never be guessed at.

    rdflib treats a plain literal and an xsd:string-typed literal as UNEQUAL
    (§5), which is why nothing here compares literals — every branch
    converts. `float()` on an rdflib Literal parses its lexical form, so
    "INF"^^xsd:double round-trips to `inf` and the fail-closed convention survives
    a serialize/parse cycle.
    """
    dt = lit.datatype
    if dt == XSD.integer:
        return int(lit)
    if dt in (XSD.double, XSD.decimal, XSD.float):
        return float(lit)
    if dt is None or dt == XSD.string:
        return str(lit)
    raise ValueError(
        f"{path}: hw1:settingValue of {param_local!r} carries datatype {dt}; "
        f"§5 declares exactly one of xsd:double / xsd:integer / "
        f"xsd:string, one per hw1:paramValueKind")


def _experiment_settings(g, exp, path):
    """{param_local: float|int|str} over every hw1:hasFactorSetting of `exp`.

    Reads BOTH sections when handed the whole file's graph, which is the point: a
    student's settings sit above the marker and the defaults `experiment` filled in
    sit below it, and the recorded vector is their union (§4.2). Which
    side a setting came from is recoverable from the split, not from this dict.

    Keyed by the PARAMETER's local name, not by the setting node: parameter local
    names are globally unique across factors (§2), which is why
    `status_for` can look a threshold up by name — and why a student's settings may
    be BLANK NODES (§2/§4.5) without anything downstream noticing. The factor a
    setting points at is not in the key either; `hw1:settingForFactor` names the
    primary factor only (§4.5) and readers traverse the TBox for the rest.

    Two settings of the SAME parameter at DIFFERENT values is a hard error, not a
    last-one-wins: `g.objects` has no defined order, so grading would silently
    depend on rdflib's iteration order, and the file would state two levels of one
    parameter while being, by construction, one treatment.
    """
    settings = {}
    for setting in g.objects(exp, HW1.hasFactorSetting):
        param = g.value(setting, HW1.settingParameter)
        if param is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting {setting} has no hw1:settingParameter")
        local = _local(param)
        lit = g.value(setting, HW1.settingValue)
        if lit is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting for {local!r} has no hw1:settingValue")
        value = _setting_value_to_python(lit, path, local)
        if local in settings and settings[local] != value:
            raise ValueError(
                f"{path}: parameter {local!r} is recorded twice with different values "
                f"({settings[local]!r} and {value!r}); one experiment is ONE treatment "
                f"(§3.1: two levels are two experiments, and the second "
                f"one needs its own declaration file)")
        settings[local] = value
    return settings


def _annotation_frame_index(g, ann, path):
    """The integer frame index one hw1:FrameAnnotation is about.

    `hw1:frameIndex` is carried redundantly by the annotation (§4.3)
    precisely so a reader need not join back into the batch file, which
    `read_experiment` does not open. When it is absent the frame IRI behind
    `hw1:annotatesFrame` still carries the stem, and `frame_index_from_iri` is THE
    tail parse (§8) — so the fallback re-uses that one implementation instead of
    adding a second `split("/")`.
    """
    idx = g.value(ann, HW1.frameIndex)
    if idx is not None:
        return int(idx)
    frame = g.value(ann, HW1.annotatesFrame)
    if frame is None:
        raise ValueError(
            f"{path}: hw1:FrameAnnotation {ann} carries neither hw1:frameIndex nor "
            f"hw1:annotatesFrame, so nothing says which frame it is about")
    return frame_index_from_iri(frame)


def _mask_factor_from_path(mask_file, path):
    """Extract the factor directory from ``.../masks/<factor>/<file>.png``."""
    parts = str(mask_file).replace("\\", "/").split("/")
    try:
        index = parts.index("masks")
        factor = parts[index + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"{path}: hw1:maskFile {str(mask_file)!r} does not follow "
            "<experiment>/masks/<factor>/<file>.png") from None
    if not factor:
        raise ValueError(f"{path}: hw1:maskFile has an empty factor directory")
    return factor


def read_experiment(path):
    """Read an ASSESSED experiment .ttl into the dict of §8.2. THE ONE READER.

    Returns

        {"exp_iri":      URIRef,
         "exp_name":     str,                     # the file stem == the IRI tail
         "batch_name":   str,                     # from hw1:onBatch, not the path
         "batch_iri":    URIRef,
         "selected":     [factor_local, …],       # hw1:evaluatesFactor, sorted
         "settings":     {param_local: float | int | str},   # the recorded vector
         "frame_status": {frame_index: bool},     # DERIVED, see below
         "pair_status":  {(i, j): bool},          # total: pairs are always minted
         "usable_links": [(i, j), …],             # §4.5, ascending by hw1:pairIndex
         "graph":        rdflib.Graph}            # the WHOLE file, both sections

    THE SEAL IS VERIFIED FIRST, BEFORE ANY VALUE IS READ (§3.1/§8.2).
    An experiment file is half hand-written; if its declaration changed after
    assessment, every verdict below the marker was computed for a treatment the
    file no longer declares, and there is no way to tell which half to believe.
    That is a hard error here and in `write_run`, and the fix is always the same:
    a new file (write-once).

    `frame_status` IS DERIVED IN v3 (§4.5), while keeping v2's key
    and shape so callers — `reconstruct.py` above all — need no change. A frame is
    usable iff EVERY annotation of it is Pass, and VACUOUSLY usable when it has no
    annotation at all: an experiment that selected only pair factors mints no
    annotations, and a modality with no selected factor mints no node (§4.3), so
    "absent" means "this experiment makes no claim", not "failed". v2's stored
    `hw1:frameStatus` is deleted; the conjunction lives here, exactly as the
    usable-link conjunction already did.

    `usable_links` IS DERIVED HERE, AND ONLY HERE (§4.5). A link is
    usable iff the pair's `hw1:qualificationStatus` is Pass — vacuously so when no
    pair factor was selected — AND both of its endpoint frames are usable. That is
    three nodes' worth of verdict, so leaving it to callers would mean every
    caller re-deriving it, and the first one to write `qualificationStatus == Pass`
    and forget the endpoints would select frames that failed their own factors.
    It is deliberately NOT a stored predicate either: it spans three nodes, so a
    cached copy goes stale the moment any of them is re-measured.

    ORDER IS `hw1:pairIndex`, NEVER THE IRI. Stems are unpadded (§2), so
    lexicographic order puts `pair/10_11` before `pair/9_10`. `pair_status` and
    `usable_links` are both built in pairIndex order and Python dicts preserve
    insertion order, so a caller may iterate either one and get capture order;
    `frame_status` is built in ascending frame-index order for the same reason.

    WHAT IS GONE. `exp_id` (§6: identity is the NAME now, and the key
    is `exp_name`), and v1's `frames` / `knobs` keys before it (§11). A caller that
    wants the selected frame set of a past run reads `hw1:usedFrame` off that run.

    Raises ValueError on a file with no machine section (an unassessed
    declaration), a broken seal, zero or several hw1:Experiment subjects, a
    missing `hw1:onBatch`, a FactorSetting without a parameter or a value, or an
    annotation or pair missing its `hw1:qualificationStatus` or its index. Every
    one of those is a malformed file rather than a state: §4.3 makes the aggregate
    total on every node it writes, so absence is a bug, and a bug that silently
    reads as `Fail` would show up only as a smaller selection and a slightly worse
    score.
    """
    text = _read_text(path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{path}: this file carries no MACHINE SECTION marker, so it is a "
            f"DECLARATION that has not been assessed yet — there are no values and "
            f"no verdicts in it to read. Run `api.py experiment {path}` first "
            f"(`api.py explore {path}` shows what it declares in the meantime).")

    # ONE GRAPH, BOTH SECTIONS (§3.1). The machine section re-opens
    # the student's Experiment subject, so only the whole file carries the whole
    # experiment: the selection and the student's own settings are above the
    # marker, the measurements and the defaulted settings below it.
    g = Graph()
    g.parse(data=text, format="turtle")
    exp = _sole_experiment(g, path)
    _verify_seal(g, exp, student_text, path)

    b = g.value(exp, HW1.onBatch)
    if b is None:
        raise ValueError(f"{path}: experiment {exp} has no hw1:onBatch")

    selected = sorted({_local(f) for f in g.objects(exp, HW1.evaluatesFactor)})
    settings = _experiment_settings(g, exp, path)

    # ── frame verdicts: the conjunction over that frame's annotations ─────────
    per_frame = {}
    mask_files = {}
    for ann in g.objects(exp, HW1.producesAnnotation):
        idx = _annotation_frame_index(g, ann, path)
        st = g.value(ann, HW1.qualificationStatus)
        if st is None:
            raise ValueError(
                f"{path}: hw1:FrameAnnotation {ann} has no hw1:qualificationStatus; "
                f"§4.3 makes the aggregate total on every annotation, so this file "
                f"was written by something that is not `api.py experiment`")
        per_frame[idx] = per_frame.get(idx, True) and (st == HW1.Pass)
        for mask_file in g.objects(ann, HW1.maskFile):
            factor = _mask_factor_from_path(mask_file, path)
            mask_files.setdefault(factor, {"frames": {}, "pairs": {}})[
                "frames"][idx] = str(mask_file)

    # ── pair verdicts, in capture order ───────────────────────────────────────
    ordered = []
    for pair in g.objects(exp, HW1.producesPair):
        src = g.value(pair, HW1.sourceFrame)
        tgt = g.value(pair, HW1.targetFrame)
        if src is None or tgt is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} is missing hw1:sourceFrame or "
                f"hw1:targetFrame")
        st = g.value(pair, HW1.qualificationStatus)
        if st is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} has no hw1:qualificationStatus; §4.3 "
                f"makes it total on every pair — vacuously hw1:Pass when no pair "
                f"factor was selected")
        pidx = g.value(pair, HW1.pairIndex)
        if pidx is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} has no hw1:pairIndex, so nothing puts "
                f"it in capture order (§2: order by pairIndex, never by IRI string)")
        pair_key = (frame_index_from_iri(src), frame_index_from_iri(tgt))
        for mask_file in g.objects(pair, HW1.maskFile):
            factor = _mask_factor_from_path(mask_file, path)
            mask_files.setdefault(factor, {"frames": {}, "pairs": {}})[
                "pairs"][pair_key] = str(mask_file)
        ordered.append((int(pidx),
                        pair_key,
                        st == HW1.Pass))
    ordered.sort(key=lambda row: row[0])

    # A frame that appears only as a pair endpoint is VACUOUSLY usable: with no
    # annotation there is no claim about it to fail (§4.5). v2 raised here instead,
    # because v2 annotated every frame; under selection that would reject every
    # pair-only experiment.
    endpoints = {end for _, ends, _ in ordered for end in ends}
    frame_status = {idx: per_frame.get(idx, True)
                    for idx in sorted(set(per_frame) | endpoints)}

    pair_status = {}
    usable_links = []
    for _, (i, j), ok in ordered:
        pair_status[(i, j)] = ok
        # §4.5, the whole rule, once: pair Pass AND both endpoints usable.
        if ok and frame_status[i] and frame_status[j]:
            usable_links.append((i, j))

    return {"exp_iri": URIRef(str(exp)),
            "exp_name": _experiment_name_from_iri(exp, path),
            "batch_name": _batch_name_from_batch_iri(b),
            "batch_iri": URIRef(str(b)),
            "selected": selected,
            "settings": settings,
            "frame_status": frame_status,
            "pair_status": pair_status,
            "usable_links": usable_links,
            "mask_files": mask_files,
            "graph": g}


# The IRI `<mode>` segment of §2 pinned to the hw1:SelectionMode
# individual of §4.4. One dict, so "baseline" cannot end up meaning GoodSegments in
# one caller and FullBatch in another.
_SELECTION_MODE = {"baseline": HW1.FullBatch,
                   "selected": HW1.GoodSegments,
                   "masked": HW1.MaskFiltered}

# Statusless reconstruction-mechanism evidence (§14).  These are
# diagnostics, not quality factors: there is no threshold to game and no baked
# Pass/Fail cache.  Keep the allow-list here so a misspelling cannot silently mint
# an RDF predicate that no reader knows about.
_RUN_METADATA = {
    "gatedSteps": XSD.integer,
    "spliceCount": XSD.integer,
    "maxGapLength": XSD.integer,
}


def _write_machine_section(path, student_text, machine_graph):
    """Rewrite `path` as: student bytes, the marker line, the serialized machine graph.

    THE STUDENT SECTION GOES BACK BYTE FOR BYTE (§3.1). `student_text`
    is what `_split_sections` returned, so the prefix the seal was computed over is
    the prefix written back, and the seal still verifies afterwards.

    ATOMIC, via a sibling temp file and `os.replace`, because the half of this file
    above the marker is HAND-WRITTEN and may be its author's only copy. Every other
    .ttl this project writes is reproducible from a command line; this one is not,
    so a crash between `open(..., "w")` and the last `write` must not be able to
    eat somebody's declaration.
    """
    body = machine_graph.serialize(format="turtle")
    if isinstance(body, bytes):                      # rdflib < 6 returned bytes
        body = body.decode("utf-8")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(student_text)
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")
    os.replace(tmp, path)


def write_run(exp_path, mode, values, used_frames=None, frame_count=None,
              metadata=None, mask_factor=None, diagnostic_file=None):
    """Write one hw1:ReconstructionRun into an experiment file. THE ONE WRITER.

    CONTRACT (§8.2)
        In:     exp_path    — an ASSESSED experiment file (student declaration +
                              marker + machine section). Only the machine section
                              is rewritten.
                mode        — "baseline" | "selected". Fixes the run IRI (§2) and the
                              hw1:selectionMode individual (§4.4).
                values      — {value_property_local: number}, e.g. {"mapMeanL2": 0.41}
                              or {"coverageF": 0.62}. For each key the VALUE and its
                              `…Status` are written, the status computed by
                              `status_for` from the QualificationSettings recorded in
                              THIS SAME FILE — which means both sections of it: the
                              student's thresholds live above the marker.
                used_frames — iterable of integer frame indices, "selected" only:
                              one hw1:usedFrame triple each.
                frame_count — how many frames the run consumed -> hw1:runFrameCount.
                metadata    — optional statusless mechanism counts from
                              `_RUN_METADATA`, e.g. {"gatedSteps": 3,
                              "spliceCount": 2, "maxGapLength": 17}.
        Out:    None. The machine section is rewritten; the declaration is not.

    Called by `reconstruct.py`, by `completeness.py` and by any other evaluator.
    Nobody else emits a run triple, which is what keeps "where did this number come
    from" answerable from one function.

    WHY THIS IS THE ONLY MUTATION AN ASSESSED FILE ACCEPTS (§3.1).
    Experiments are write-once: `experiment` refuses a file that already carries the
    marker, and every tuning is a brand-new declaration under a brand-new name. Runs
    are the exception because they are the experiment's OUTCOME rather than its
    design — reconstructing the same experiment twice must replace its numbers, not
    fork the notebook. The seal is verified before any of that: a declaration edited
    after assessment is a different experiment, and this function refuses to write
    into it.

    IDEMPOTENT PER (EXPERIMENT, MODE, KEY) — the single most important behaviour
    here, unchanged from v2. The triples removed before writing are exactly the ones
    about to be written: the value and status of each key in `values`, plus
    hw1:usedFrame and hw1:runFrameCount when those arguments are supplied. Everything
    else on the run node is left alone. That is what lets `completeness.py` add
    `coverageF` to a run that already carries `mapMeanL2` without erasing it — the
    two numbers come from two different programs and neither owns the node.

    TOTAL ON EVERY CALL: `rdf:type`, `hw1:selectionMode` and the `hw1:hasRun`
    back-link from the experiment (§4.4). A reader may therefore require all three
    without OPTIONAL, while every value property is optional because of the
    idempotency rule above. Re-adding them costs nothing — an rdflib Graph is a
    set, so adding a triple that is already there is a no-op.

    GRADE THE VALUE YOU STORE, NOT THE VALUE YOU COMPUTED (§4.5).
    Every value goes through `_storable` BEFORE `status_for` sees it, because
    rdflib re-derives every xsd:double at 7 significant digits and the status is a
    denormalised cache of (value + settings): grading the unrounded number would
    write `hw1:mapMeanL2 4e-01 ; hw1:mapMeanL2Status hw1:Fail` for a computed
    0.4000000001 against a threshold of 0.4 — a record that contradicts itself, and
    that a student re-grading the stored value gets the opposite answer from.

    FAIL CLOSED. `inf` is written as-is, `"INF"^^xsd:double` (§4.6), and grades
    `Fail` by the ordinary rule of §4.5 — no special case anywhere (`_storable`
    passes non-finite values through untouched). `mean_l2` returns `inf` when the
    ground truth is missing, so a run that could not be scored reads as FAILED
    rather than as excellent. NaN likewise always Fails.

    WHAT THE REWRITE NORMALISES, worth knowing before you diff a file: the machine
    section is parsed, mutated and re-serialized, so ITS comments go and ITS
    prefixes are reordered, and rdflib writes every xsd:double via "%e" at seven
    significant digits (`0.41` -> `4.1e-01`). Deterministic, uniform across every
    file this project writes — and applied to the machine half ONLY. Above the
    marker not one byte moves.

    Raises ValueError on an unknown `mode`, on `used_frames` with mode="baseline"
    (hw1:usedFrame is a GoodSegments-only property, §4.4), on a file with no machine
    section, on a broken seal, on a `values` key that no hw1:QualityFactor grades,
    and — through `status_for` — on a missing threshold in the file's own settings.
    """
    if mode not in _SELECTION_MODE:
        raise ValueError(
            f"mode must be one of {sorted(_SELECTION_MODE)}, got {mode!r}; the mode is "
            f"the last segment of the run IRI (§2) and is not free text")
    if used_frames is not None and mode not in ("selected", "masked"):
        raise ValueError(
            f"used_frames given with mode={mode!r}: hw1:usedFrame is asserted by "
            f"selected or masked subset runs only. A full-batch run states "
            f"its size with hw1:runFrameCount and lists no frames.")
    if mode == "masked" and not mask_factor:
        raise ValueError("mode='masked' requires mask_factor provenance")
    if mode != "masked" and mask_factor is not None:
        raise ValueError("mask_factor is legal only for mode='masked'")
    metadata = {} if metadata is None else dict(metadata)
    unknown_metadata = sorted(set(metadata) - set(_RUN_METADATA))
    if unknown_metadata:
        raise ValueError(
            f"unknown run metadata {unknown_metadata}; allowed statusless mechanism "
            f"properties: {', '.join(sorted(_RUN_METADATA))}")
    for key, value in metadata.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"run metadata {key} must be a non-negative integer, "
                             f"got {value!r}")

    text = _read_text(exp_path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{exp_path}: no machine section — a run cannot be written into a "
            f"declaration that was never assessed. Run `api.py experiment "
            f"{exp_path}` first.")

    # The WHOLE file answers "which experiment, under which thresholds" (the
    # student's own settings are above the marker); the MACHINE half is what gets
    # mutated and written back. Two parses of one small file, and no possibility of
    # grading against half a setting vector.
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, exp_path)
    _verify_seal(full, exp, student_text, exp_path)
    expname = _experiment_name_from_iri(exp, exp_path)
    settings = _experiment_settings(full, exp, exp_path)
    run = run_iri(expname, mode)

    # The status PROPERTY comes from the factor's hw1:statusProperty, never from
    # string concatenation. The naming rule (value + "Status") is frozen, but a
    # concatenating writer would happily invent `hw1:somethingStatus` for a key no
    # factor grades, and the resulting triple would be invisible to every reader —
    # they all find statuses through `?factor hw1:statusProperty ?p`.
    factors = load_quality_factors()
    if mask_factor is not None:
        selected = {_local(f) for f in full.objects(exp, HW1.evaluatesFactor)}
        if mask_factor not in selected:
            raise ValueError(
                f"mask_factor {mask_factor!r} was not selected by this experiment")
    try:
        status_property = {f["over"]: f["status"] for f in factors.values()}
    except KeyError as exc:
        raise ValueError(
            f"load_quality_factors() must return the §8 shape "
            f"{{'over', 'status', 'polarity', 'qualifiedBy'}}; key {exc} is missing"
        ) from None

    # Resolve EVERY key, and every status, before touching the graph: a typo in the
    # second key must not leave the first one half-written into the file.
    planned = []
    for key in values:
        if key not in status_property:
            raise ValueError(
                f"{key!r} is not graded by any hw1:QualityFactor; write_run writes a "
                f"value together with its status, so a value nothing grades has no "
                f"business in a run node. Graded value properties: "
                f"{', '.join(sorted(status_property))}")
        # ROUND FIRST, GRADE SECOND (§4.5). `_storable` is the value as
        # the .ttl will actually hold it; grading the unrounded argument would bake a
        # verdict the stored number re-grades the other way. mapMeanL2 = 0.4000000001
        # against maxMapMeanL2 = 0.4 is the case: it stores as `4e-01`, so a `Fail`
        # graded from the raw double sits next to a number that reads Pass.
        stored = _storable(values[key])
        planned.append((HW1[key], HW1[status_property[key]], stored,
                        status_for(key, stored, settings, factors)))

    # hw1:usedFrame points at the SHARED structural frames of the batch (§2), not at
    # anything experiment-scoped, so the frame IRIs are minted from the batch name —
    # which is why an experiment without hw1:onBatch cannot record provenance at all.
    frames = None
    if used_frames is not None:
        b = full.value(exp, HW1.onBatch)
        if b is None:
            raise ValueError(
                f"{exp_path}: experiment {exp} has no hw1:onBatch, so the frame IRIs "
                f"for hw1:usedFrame cannot be minted")
        name = _batch_name_from_batch_iri(b)
        frames = [frame_iri(name, int(idx)) for idx in used_frames]

    # Everything a run node consists of lives BELOW the marker, so the mutation
    # happens on the machine section alone and the declaration is never re-serialized.
    g = Graph()
    g.parse(data=machine_text, format="turtle")

    # ── remove exactly what is about to be written, and nothing else ──────────
    for value_p, status_p, _, _ in planned:
        g.remove((run, value_p, None))
        g.remove((run, status_p, None))
    if frames is not None:
        g.remove((run, HW1.usedFrame, None))
    if frame_count is not None:
        g.remove((run, HW1.runFrameCount, None))
    if mask_factor is not None:
        g.remove((run, HW1.maskFactor, None))
    if diagnostic_file is not None:
        g.remove((run, HW1.diagnosticFile, None))
    for key in metadata:
        g.remove((run, HW1[key], None))
    # `mode` is mandatory, so the selection mode is always "supplied" (§8.2) and the
    # removal is unconditional: a run node that somehow carried the OTHER mode would
    # otherwise end up carrying both, and hw1:selectionMode is the one predicate a
    # reader is allowed to assume is single-valued and present.
    g.remove((run, HW1.selectionMode, None))

    # ── write ─────────────────────────────────────────────────────────────────
    g.add((run, RDF.type, HW1.ReconstructionRun))
    g.add((run, HW1.selectionMode, _SELECTION_MODE[mode]))
    g.add((exp, HW1.hasRun, run))
    for value_p, status_p, v, status in planned:
        # `_double_literal` is the one spelling of an xsd:double in this project, and
        # it rounds through `_storable` too — so the number written here is, byte for
        # byte, the number `status` was computed from.
        g.add((run, value_p, _double_literal(v)))
        g.add((run, status_p, status))
    if frame_count is not None:
        g.add((run, HW1.runFrameCount, Literal(int(frame_count), datatype=XSD.integer)))
    if frames is not None:
        for f in frames:
            g.add((run, HW1.usedFrame, f))
    if mask_factor is not None:
        g.add((run, HW1.maskFactor, HW1[mask_factor]))
    if diagnostic_file is not None:
        g.add((run, HW1.diagnosticFile, Literal(str(diagnostic_file))))
    for key, value in metadata.items():
        g.add((run, HW1[key], Literal(int(value), datatype=_RUN_METADATA[key])))

    _write_machine_section(exp_path, student_text, g)


def write_pair_measurements(exp_path, factor_local, measurements):
    """Write a deferred, pre-fit pair factor from the actual consumer loop.

    ``measurements`` is an iterable of dictionaries with ``source``, ``target``,
    ``value``, ``count`` and optional uint8 ``mask``.  This is intentionally a
    separate writer from :func:`write_run`: pair evidence belongs on FramePair,
    while trajectory outcomes belong on ReconstructionRun.
    """
    factors = load_quality_factors()
    if factor_local not in factors:
        raise ValueError(f"unknown QualityFactor {factor_local!r}")
    value_local = factors[factor_local]["over"]
    spec = _PAIR_OBSERVABLES.get(value_local)
    if spec is None or not spec.get("deferred"):
        raise ValueError(
            f"{factor_local} is not a deferred pair factor; experiment measures it")

    text = _read_text(exp_path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(f"{exp_path}: cannot write pair evidence before assessment")
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, exp_path)
    _verify_seal(full, exp, student_text, exp_path)
    selected = {_local(f) for f in full.objects(exp, HW1.evaluatesFactor)}
    if factor_local not in selected:
        raise ValueError(
            f"{exp_path}: cannot write {factor_local}; it was not selected")
    settings = _experiment_settings(full, exp, exp_path)

    by_pair = {(int(m["source"]), int(m["target"])): dict(m)
               for m in measurements}
    g = Graph()
    g.parse(data=machine_text, format="turtle")
    expname = _experiment_name_from_iri(exp, exp_path)
    artifact_root = os.path.splitext(os.path.abspath(exp_path))[0]
    relative_to = os.path.dirname(os.path.abspath(exp_path))
    count_property = spec.get("count_property")

    written = 0
    for pair in g.objects(exp, HW1.producesPair):
        src = g.value(pair, HW1.sourceFrame)
        tgt = g.value(pair, HW1.targetFrame)
        key = (frame_index_from_iri(src), frame_index_from_iri(tgt))
        measurement = by_pair.get(key)
        value = (float("inf") if measurement is None
                 else float(measurement.get("value", float("inf"))))
        count = 0 if measurement is None else int(measurement.get("count", 0))

        g.remove((pair, HW1[value_local], None))
        g.remove((pair, HW1[factors[factor_local]["status"]], None))
        if count_property:
            g.remove((pair, HW1[count_property], None))
        _write_observable(g, pair, value_local, value, settings, factors)
        if count_property:
            g.add((pair, HW1[count_property], Literal(count, datatype=XSD.integer)))

        # Remove only this factor's prior artifact reference; other selected
        # factor masks share hw1:maskFile on the same pair node.
        for old in list(g.objects(pair, HW1.maskFile)):
            if _mask_factor_from_path(old, exp_path) == factor_local:
                g.remove((pair, HW1.maskFile, old))
        mask = None if measurement is None else measurement.get("mask")
        mask_file = _write_mask_artifact(
            mask, artifact_root, relative_to, factor_local,
            f"{key[0]}_{key[1]}.png")
        if mask_file is not None:
            g.add((pair, HW1.maskFile, Literal(mask_file)))
        written += 1

    selected_pair_values = [
        factors[name]["over"] for name in selected
        if name in factors and factors[name]["over"] in _PAIR_OBSERVABLES]
    for pair in g.objects(exp, HW1.producesPair):
        passed = []
        for over in selected_pair_values:
            status = g.value(pair, _status_property(over, factors))
            passed.append(status == HW1.Pass if status is not None else False)
        g.set((pair, HW1.qualificationStatus,
               HW1.Pass if all(passed) else HW1.Fail))

    _write_machine_section(exp_path, student_text, g)
    return written


# =============================================================================
# batch2ttl  — the STRUCTURE of one capture, and not one measured number
# =============================================================================
def _resolve_generation_settings(gen_args, decls):
    """`--gen NAME=VALUE` list + TBox declarations -> {param_local: value}.

    ONLY GenerationSetting parameters are accepted, and that is a hard error rather
    than a tolerated confusion: `--gen tauHi=250` would record a MEASUREMENT number
    on the Batch, where nothing reads it and where it would claim to describe the
    pixels. The role decides the node (§4.2), so the role has to be
    checked at the door.

    NOTHING IS EVER DEFAULTED HERE (§5). A `brightnessGain = 1.0`
    filled in from a default would assert that this code inspected the capture and
    found it ungained — a claim nothing verified, about pixels this program never
    opened. Absence means "not asserted", not "identity", and the TBox backs that by
    declaring the two Generation parameters with no `hw1:paramDefault` at all. This
    is the one place the completeness rule of §5 deliberately does not apply.

    A repeated `--gen` is an error, not last-one-wins: whichever value this function
    picked, the other one is in the shell history of somebody who believes it was
    the treatment.
    """
    given = {}
    for arg in gen_args or []:
        name, value = parse_setting_arg(arg, decls)
        role = decls[name]["role"]
        if role != "GenerationSetting":
            gen_names = sorted(n for n in decls if decls[n]["role"] == "GenerationSetting")
            raise ValueError(
                f"--gen {name}=... : {name!r} is declared hw1:paramRole hw1:{role}, not "
                f"hw1:GenerationSetting. A Generation setting describes the PIXELS and "
                f"lives on the Batch; Measurement and Qualification settings describe a "
                f"measurement pass and live in the DECLARATION, as hw1:FactorSetting "
                f"nodes on the experiment. "
                f"Generation parameters are: {', '.join(gen_names) or '(none)'}")
        if name in given:
            raise ValueError(
                f"--gen {name!r} given twice ({given[name]!r} then {value!r}). One "
                f"capture directory was produced under one level of each generation "
                f"parameter; two levels are two captures.")
        given[name] = value
    return given


def build_batch_graph(data_dir, floor, generation=None, derived_from=None):
    """Build the in-memory rdflib.Graph of one batch directory. STRUCTURE ONLY.

    CONTRACT
        In:     data_dir — a directory with rgb/ and depth/ subdirs (see
                _pair_frames; only stems present in BOTH are frames of this batch).
                A semantic/ subdir, if present, is tolerated but NOT ingested.
                floor — the floor label, an int; see `batch_name`.
                generation — {param_local: value} of GenerationSettings to record,
                already resolved by `_resolve_generation_settings`. Recorded ONLY
                when given (§5).
                derived_from — the batchName of the capture this one was derived
                from, or None; emits one `prov:wasDerivedFrom`.
        Out:    (graph, name, batch_iri) where name = batch_name(data_dir, floor)
                and batch_iri = batch_iri(name).

        The graph must satisfy the TBox in ontology/hw1.ttl exactly. Put a node on
        the wrong subject and a query's pattern simply will not match
        (§4.1):

            Batch       batchName (= name), batchPath, floor, hasFrame,
                        hasGenerationSetting, prov:wasDerivedFrom
            Frame       frameIndex, one per paired stem
            RGBImage    schema:contentUrl,  linked by hasRGBImage
            DepthImage  schema:contentUrl,  linked by hasDepthImage

        `floor` and `frameIndex` are xsd:integer. Store the string properties as
        PLAIN literals: in RDF 1.1 a plain literal IS an xsd:string, which is what
        makes hw1:batchName "name" match under rdflib's strict in-memory matcher.

    WHAT IS DELIBERATELY ABSENT
        Every measured value, every threshold, every measurement parameter.

        No `meanValue`, no `clipHiFraction`, no `validDepthFraction`, no
        `depthRoughness`, no `medianDepthDifference`, no `depthEdgeOverlap`, and no
        `tauHi` / `tauLo`: those are measurement, they depend on the setting vector,
        and they belong to an EXPERIMENT file — the student's declaration, with its
        machine section appended (§4.3). This function opens no PNG. It
        runs in the time it takes to list two directories, and it is the reason a
        second experiment at a second tau does not rewrite the batch.

        NO FramePair NODES AND NO hw1:hasNextFrame — both deleted in v2
        (§11). v1 wrote a FramePair skeleton per consecutive pair here
        and let each experiment decorate it with observables. Under one graph that
        is a collision: two experiments write two contradictory
        `hw1:medianDepthDifference` triples onto one shared node. Pairs are minted
        per experiment now (`pair_iri(expname, i, j)`) — a pair is two links and an
        index, so the skeleton was never worth sharing. `hasNextFrame` went with
        them: it was the successor chain the skeletons hung off, and ordering is
        carried by `hw1:frameIndex` / `hw1:pairIndex`, which a reader sorts by as
        integers instead of walking a property path.

        NO TBox, and no flag to bundle one. `--no-ontology` is gone with the whole
        three-graph layout: the TBox is `hw1/ontology/hw1.ttl` and every reader in
        this project loads it from there (§3), so duplicating it into
        every batch file would only create several copies to disagree with each
        other.

    STEM GAPS ARE STILL LOGGED, EVEN THOUGH THE PAIRS HAVE LEFT
        The warning is about the FRAMES, not the pairs. If 42 is missing from
        depth/ while 43 is present in both, then every experiment over this batch
        will pair 41 with 43 — a real pair over a real two-step motion, whose
        observables honestly report the larger displacement. That is the right
        graph, but it is invisible unless somebody says so. A silently dropped
        frame is the classic way an ICP gate sized for consecutive frames starts
        failing on data nobody thinks changed, and `batch2ttl` is the first command
        in the loop and therefore the right place to say it.
    """
    g = Graph()
    name = batch_name(data_dir, floor)
    b = batch_iri(name)

    g.add((b, RDF.type, HW1.Batch))
    g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    g.add((b, HW1.floor, Literal(int(floor), datatype=XSD.integer)))

    # Corruption provenance, on the BATCH because it describes the pixels and
    # survives every re-measurement of them (§4.2). Applied by nothing
    # here: the capture directory already embodies it, and pretending otherwise is
    # what the role exists to prevent.
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    for param in sorted(generation or {}):
        decl = decls[param]
        s = generation_setting_iri(name, param)
        g.add((b, HW1.hasGenerationSetting, s))
        g.add((s, RDF.type, HW1.FactorSetting))
        g.add((s, HW1.settingParameter, decl["iri"]))
        # The role is DENORMALISED onto the setting node on purpose: the attribution
        # query then reads the verdict ("regenerate the data") off the node it
        # already has, without a second join into the TBox.
        g.add((s, HW1.settingRole, HW1[decl["role"]]))
        # settingForFactor is SINGLE-VALUED — the primary factor only, here exactly
        # as on experiment-scoped settings (§4.5). v2 wrote the affected
        # factors out too, so that attribution could blame brightnessGain for CRUSHED
        # shadows and not only for blown highlights — the failure a low_light batch
        # actually produces. v3 keeps that blame and drops the duplication: `explore`
        # reaches the affected factors by traversing hw1:paramAffectsFactor in the
        # TBox, which is the one edge that was being copied here.
        g.add((s, HW1.settingForFactor, decl["primaryIri"]))
        g.add((s, HW1.settingValue, _setting_value_literal(generation[param], decl["kind"])))

    # prov:wasDerivedFrom BETWEEN BATCHES is the only surviving prov: term
    # (§6): "these pixels came from those pixels, corrupted". It is NOT
    # asserted between experiments — there are no derived experiments in v2.
    if derived_from:
        g.add((b, PROV.wasDerivedFrom, batch_iri(derived_from)))

    frames = _pair_frames(data_dir)
    for stem, rgb_path, depth_path in frames:
        f = frame_iri(name, stem)
        g.add((b, HW1.hasFrame, f))
        g.add((f, RDF.type, HW1.Frame))
        g.add((f, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))

        # An image node carries schema:contentUrl and NOTHING ELSE in v2
        # (§4.3): the observables moved to the experiment-scoped
        # FrameAnnotation, because two experiments cannot both own one image node.
        rc = component_iri(name, stem, "rgb")
        g.add((f, HW1.hasRGBImage, rc))
        g.add((rc, RDF.type, HW1.RGBImage))
        g.add((rc, SCHEMA.contentUrl, Literal(rgb_path)))

        dc = component_iri(name, stem, "depth")
        g.add((f, HW1.hasDepthImage, dc))
        g.add((dc, RDF.type, HW1.DepthImage))
        g.add((dc, SCHEMA.contentUrl, Literal(depth_path)))

    _warn_stem_gaps(frames, prefix="[batch2ttl]")
    return g, name, b


def _warn_stem_gaps(frames, prefix):
    """Log consecutive-stem holes in a paired capture. Shared by declare/batch2ttl."""
    gaps = [(int(s0), int(s1))
            for (s0, _, _), (s1, _, _) in zip(frames, frames[1:])
            if int(s1) - int(s0) != 1]
    if not gaps:
        return gaps
    shown = ", ".join(f"{i}->{j}" for i, j in gaps[:12])
    more = f" ... (+{len(gaps) - 12} more)" if len(gaps) > 12 else ""
    print(f"{prefix} WARNING: {len(gaps)} stem gap(s) in the paired sequence: "
          f"{shown}{more}")
    print(f"{prefix} WARNING: every experiment over this batch will pair across "
          f"those gaps, spanning more than one capture step. A stem missing from "
          f"rgb/ or depth/ changes which pairs exist, and the pair observables will "
          f"read as larger motion — correctly, but check it is the capture and not a "
          f"lost file.")
    return gaps


def cmd_batch2ttl(args):
    """DEPRECATED. Optional sidecar for generation provenance only.

    `declare --data-dir` and `experiment` now read the capture directory
    themselves. This command still writes `<data_dir>/batch.ttl` so a Generation
    setting (`--gen NAME=VALUE`) or `prov:wasDerivedFrom` has somewhere to live
    that is not an experiment file. Overwrites the file wholesale, which is safe
    because it holds no measurement.

    `--gen NAME=VALUE` records the corruption that is ALREADY BAKED into these
    pixels, as a GenerationSetting on the Batch node (§4.2). Nothing
    applies it; it is recorded so the treatment stays identifiable after the fact,
    which is the whole reason `explore`'s verdict section can return the verdict
    "regenerate the data" instead of sending a student to tune a threshold against
    pixels that were broken before measurement started.
    """
    print("[batch2ttl] DEPRECATED: `declare` and `experiment` take the capture "
          "directory directly (`declare --data-dir <dir> --floor <n>`). This "
          "command is now only the optional writer of generation provenance into "
          "<data_dir>/batch.ttl.", file=sys.stderr)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    generation = _resolve_generation_settings(getattr(args, "gen", None), decls)

    g, name, _ = build_batch_graph(args.data_dir, args.floor,
                                   generation=generation,
                                   derived_from=args.derived_from)
    out = args.out or os.path.join(args.data_dir, "batch.ttl")
    g.serialize(destination=out, format="turtle")
    n_frames = len(set(g.subjects(RDF.type, HW1.Frame)))
    print(f"[batch2ttl] batch {name!r}: {n_frames} frames, 0 measured values and 0 "
          f"pairs (by contract — pairs are minted per experiment)")
    if generation:
        print(f"[batch2ttl] generation settings recorded (provenance only, applied by "
              f"nothing): "
              f"{', '.join(f'{k}={generation[k]!r}' for k in sorted(generation))}")
    else:
        print(f"[batch2ttl] generation settings recorded: (none). Absence means NOT "
              f"ASSERTED, not 'uncorrupted' — pass --gen NAME=VALUE for a corrupted "
              f"capture, because nothing defaults a claim about pixels.")
    if args.derived_from:
        print(f"[batch2ttl] prov:wasDerivedFrom {batch_iri(args.derived_from)}")
    print(f"[batch2ttl] wrote {len(g)} triples -> {out}")
    return 0



# =============================================================================
# declare  — scaffold a declaration Turtle so nobody starts from a blank page
#
#   The declaration is still the one piece of RDF the student AUTHORS
#   (§4.2): this command only writes the boilerplate — prefixes, the
#   Experiment node whose IRI tail equals the file stem, the batch join
#   (hw1:batchFile = the capture directory, hw1:onBatch derived from --floor +
#   basename), a factor selection, and the PREDICTION block as comments.
#   Everything it writes is the student's to edit until `api.py experiment`
#   seals the file; it never assesses and never overwrites an existing
#   declaration.
# =============================================================================
def _factor_menu_comment(factors, decls):
    """The §4.2 factor menu as Turtle comment lines, generated from the TBox.

    Generated rather than pasted so the scaffold can never disagree with
    `ontology/hw1.ttl` about names, polarities or threshold defaults.
    """
    menu = _selectable_factors(factors)
    width = max(len(name) for name in menu)
    lines = []
    for name in sorted(menu):
        info = factors[name]
        op = "<=" if info["polarity"] == "lower" else ">="
        default = decls[info["qualifiedBy"]]["default"]
        lines.append(f"#   hw1:{name:<{width}}  Pass iff hw1:{info['over']} "
                     f"{op} hw1:{info['qualifiedBy']} (default {default})")
    return lines


def _declare_capture_args(args):
    """`--data-dir` (preferred) or deprecated `--batch-file` -> (dir, floor, path text)."""
    data_dir_arg = getattr(args, "data_dir", None)
    batch_file_arg = getattr(args, "batch_file", None)
    floor = int(getattr(args, "floor", 1) or 1)
    if data_dir_arg and batch_file_arg:
        raise SystemExit(
            "[declare] pass --data-dir (the capture directory) or the deprecated "
            "--batch-file, not both.")
    if data_dir_arg:
        resolved = _resolve_batch_file(data_dir_arg)
        if not _is_capture_dir(resolved):
            raise SystemExit(
                f"[declare] --data-dir {data_dir_arg!r} does not resolve to a capture "
                f"directory with rgb/ and depth/ subdirs ({resolved!r}; relative "
                f"paths resolve against the current working directory).")
        return resolved, floor, data_dir_arg.replace(os.sep, "/")
    if not batch_file_arg:
        raise SystemExit(
            "[declare] --data-dir is required (the capture directory containing "
            "rgb/ and depth/). `batch2ttl` is no longer a step in this loop.")
    print("[declare] --batch-file is deprecated: pass --data-dir <capture-dir> "
          "instead. The scaffold will name the capture directory in hw1:batchFile.",
          file=sys.stderr)
    resolved = _resolve_batch_file(batch_file_arg)
    if _is_capture_dir(resolved):
        return resolved, floor, batch_file_arg.replace(os.sep, "/")
    if os.path.isfile(resolved):
        _g, _batch, _name, stored_path, stored_floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file_arg)
        if stored_floor is not None:
            floor = stored_floor
        # Write the capture directory into the declaration, not the sidecar.
        return data_dir, floor, os.path.relpath(data_dir, os.getcwd()).replace(os.sep, "/")
    raise SystemExit(
        f"[declare] --batch-file {batch_file_arg!r} does not resolve to a capture "
        f"directory or a batch.ttl ({resolved!r}). Pass --data-dir <capture-dir>.")


def cmd_declare(args):
    """Write a boilerplate declaration to hw1/experiments/<name>.ttl. Never assesses.

    `--data-dir` is the capture directory (rgb/ + depth/). The scaffold writes
    that path into `hw1:batchFile` and derives `hw1:onBatch` from `--floor` plus
    the directory basename. `batch2ttl` is not a step.

    The scaffold is complete enough for `explore` to preview immediately (all
    selected factors, every setting `WILL BE DEFAULTED`), and the file is
    validated through `read_declaration` before this command reports success, so
    a generated declaration can never be one of the malformed ones §8.1 rejects.

    What stays the STUDENT'S JOB, by design: the PREDICTION block (a TODO until
    replaced), the rdfs:label, trimming the factor selection to the hypothesis,
    and any hw1:FactorSetting override — the command scaffolds the syntax as
    comments and sets nothing itself, because a generated setting would record a
    treatment nobody chose.

    Refuses to overwrite: an existing file under the same name is a hard error,
    assessed or not, because `hw1/experiments/` is an append-only lab notebook
    (§3.1) and a fresh scaffold over a hand-edited declaration would
    delete design work.
    """
    name = args.name
    if not _EXPNAME_RE.match(name):
        raise SystemExit(
            f"[declare] {name!r} is not a legal experiment name (§2: "
            f"[A-Za-z0-9_-]+). It becomes both the file stem and the tail of the "
            f"Experiment IRI, which must stay equal.")

    out = args.out or os.path.join(_EXPERIMENT_DIR, f"{name}.ttl")
    stem = os.path.splitext(os.path.basename(out))[0]
    if stem != name:
        raise SystemExit(
            f"[declare] --out names the file {stem!r}.ttl but --name says {name!r}. "
            f"The file stem must equal the Experiment IRI tail (§2), so "
            f"the two flags must agree.")
    if os.path.exists(out):
        raise SystemExit(
            f"[declare] {out} already exists and hw1/experiments/ is append-only "
            f"(§3.1): a new tuning idea is a NEW declaration under a "
            f"new name, and an existing file — assessed or not — is never "
            f"regenerated over.")

    data_dir, floor, batch_file_text = _declare_capture_args(args)
    try:
        frames = _pair_frames(data_dir)
    except ValueError as exc:
        raise SystemExit(f"[declare] {exc}")
    _warn_stem_gaps(frames, prefix="[declare]")
    name_of_batch = batch_name(data_dir, floor)
    batch = batch_iri(name_of_batch)

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    menu = _selectable_factors(factors)
    if args.factor:
        selected = []
        for raw in args.factor:
            local = raw[len("hw1:"):] if raw.startswith("hw1:") else raw
            if local not in menu:
                raise SystemExit(
                    f"[declare] hw1:{local} is not a factor of the menu. The menu is "
                    f"{', '.join('hw1:' + f for f in sorted(menu))} "
                    f"(§4.2); factor names come from the TBox, not the command line.")
            if local not in selected:
                selected.append(local)
        selected.sort()
    else:
        # The full menu: a legal 8-factor selection that assesses as-is. The
        # printed hint (and the TODO in the file) says to trim it — choosing the
        # selection is part of DESIGNING the experiment, not part of the scaffold.
        selected = sorted(menu)

    selection_text = " ,\n        ".join(f"hw1:{f}" for f in selected)
    lines = [
        "# =============================================================================",
        f"# EXPERIMENT DECLARATION — {name}   (scaffolded by `api.py declare`)",
        "#",
        "# This file is YOURS until `api.py experiment` seals it: edit the label, trim",
        "# the factor selection, add overrides. After assessment the student section is",
        "# digest-sealed — a new idea is a NEW file under a NEW name (§3.1).",
        "#",
        "# PREDICTION (write BEFORE assessing — the seal makes it non-retractable):",
        "#   input:    TODO — which frames/factors you expect to fail, and why",
        "#   baseline: TODO — expected mapMeanL2 band and verdict for the full-batch run",
        "#   selected: TODO — expected differential vs baseline, and the mechanism",
        "# =============================================================================",
        "",
        "@prefix hw1:  <http://taica.course/hw1/ontology#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"<{NS}experiment/{name}>",
        "    a hw1:Experiment ;",
        f"    rdfs:label \"TODO: one line naming the condition this experiment tests\"@en ;",
        f"    hw1:batchFile \"{batch_file_text}\" ;",
        f"    hw1:onBatch <{batch}> ;   # {name_of_batch}",
        "    hw1:evaluatesFactor",
        f"        {selection_text} .",
        "",
        "# THE FACTOR MENU (from ontology/hw1.ttl — select 1..8 above; TODO: trim the",
        "# selection to the factors your hypothesis actually needs):",
        *_factor_menu_comment(factors, decls),
        "#",
        "# OVERRIDES — every setting you do not declare is filled from its TBox default",
        "# at assessment time (preview with `explore` before committing). To override",
        "# one, replace the final \".\" above with \";\" and append a blank node, e.g.:",
        "#",
        "#     hw1:hasFactorSetting [",
        "#         a hw1:FactorSetting ;",
        "#         hw1:settingParameter hw1:maxHighFrequencyDepthResidual ;",
        "#         hw1:settingRole hw1:QualificationSetting ;",
        "#         hw1:settingForFactor hw1:HighFrequencyDepthResidual ;",
        "#         hw1:settingValue \"0.010\"^^xsd:double",
        "#     ] .",
        "#",
        "# hw1:settingRole must match the parameter's TBox hw1:paramRole:",
        "#   MeasurementSetting    how the observable is computed (e.g. hw1:tauHi)",
        "#   QualificationSetting  where the Pass line is cut (e.g. hw1:maxClipHiFraction)",
        "# Run-level re-cut: hw1:settingParameter hw1:maxMapMeanL2 with",
        "# hw1:settingForFactor hw1:ReconstructionAccuracy.",
        "",
    ]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # The generator's own output must pass the validator it will later be read
    # by; a scaffold this command cannot re-read is a bug here, not user error.
    try:
        decl = read_declaration(out)
    except ValueError as exc:
        os.remove(out)
        raise SystemExit(
            f"[declare] scaffold failed its own validation and was removed — this "
            f"is a bug in `declare`, not in your input: {exc}")

    defaulted = sorted(set(decl["required"]) - set(decl["given"]))
    print(f"[declare] wrote {out}")
    print(f"[declare] selection ({len(selected)}/{len(menu)}): "
          + ", ".join(f"hw1:{f}" for f in selected)
          + ("" if args.factor else "  — the FULL menu; trim it to your hypothesis"))
    print(f"[declare] {len(defaulted)} setting(s) WILL BE DEFAULTED at assessment; "
          f"preview them with:  api.py explore {out}")
    print(f"[declare] before assessing: replace the PREDICTION TODOs and the "
          f"rdfs:label — the seal makes them non-retractable. Then:  "
          f"api.py experiment {out}")
    return 0


# =============================================================================
# experiment  — ONE declaration in, ONE machine section out, ONCE
#
#   This is the command the whole layout exists for, and in v3 it is the command
#   that reads what a STUDENT wrote. The declaration (§4.2) names a
#   capture directory and says which frames to grade, on which factors, under
#   which thresholds; this command measures exactly that from the rasters on
#   disk and appends the numbers, the verdicts, the defaulted settings and the
#   tamper seal to the declaration's OWN FILE, below the marker (§3.1/§4.3).
#
#   WRITE-ONCE. A file that already carries `MACHINE_MARKER` is a hard error and
#   there is no `--force`: an experiment is one treatment, and the way to try
#   another threshold is to copy the declaration to a new name. That is what turns
#   `hw1/experiments/` into an append-only lab notebook whose series of files IS
#   the report, and it restores v2's "nothing is ever overwritten" guarantee by
#   immutability instead of by digest identity (§6).
#
#   EVERYTHING v2 CARRIED ON THE COMMAND LINE IS DELETED (§7): `--batch-dir`,
#   `--floor`, `--set`, `--exp-id`, `--label`, `--no-pairs`, `--out`. The command
#   takes exactly one positional argument, the declaration path; the batch comes
#   from `hw1:batchFile`, the settings are FactorSettings, the label is
#   `rdfs:label`, and pairs are always minted.
# =============================================================================

# How often the measuring loops print a progress line. 387 frames x up to 5
# observables plus 386 pairs x 2 is well under a minute but far too long to look
# alive, and a command that prints nothing for forty seconds gets killed by the
# person running it. Progress goes to STDERR, not stdout: it is a status display,
# not a result, and `api.py experiment ... > log` must keep the summary readable.
_PROGRESS_EVERY = 25

# WHICH OBSERVABLE GOES ON WHICH NODE, AND WHAT MEASURES IT (§4.3,
# frozen). Two tables, read three ways:
#
#   * PLACEMENT. `clipHiFraction` / `clipLoFraction` are rgb-annotation
#     properties, the three active depth-frame observables are depth-annotation
#     properties, and the three pair observables live on the FramePair. That placement is
#     structural rather than cosmetic: an annotation exists per frame PER
#     MODALITY, and a modality with no selected factor gets no node at all.
#   * MEASUREMENT. `params` is what the measurer needs out of the setting vector
#     — checked against the vector before a single PNG is opened — and `measure`
#     is the one call site of each measurer in this file.
#   * THE MENU. `_selectable_factors` derives the eight-factor menu of §4.2 from
#     these keys: a QualityFactor whose `hw1:overProperty` appears here is
#     selectable; one whose does not (mapMeanL2, coverageF) is a RUN factor,
#     never selected and always evaluated. Adding a menu factor is therefore
#     adding a TBox declaration and one row here, and no per-factor branch
#     anywhere else.
#
# `hw1:meanValue` is deliberately absent: no QualityFactor is declared over it, it
# carries no status and it takes no part in any aggregate. It is written beside
# the clip factors as the baseline they must beat (§4.3), and putting it in this
# table would quietly promote the deliberately-weak baseline to a criterion.
_FRAME_OBSERVABLES = {
    "highFrequencyDepthResidual": {
        "modality": "depth",
        "params": ("residualMaskK",),
        "measure_mask": lambda rgb, depth, s: _high_frequency_depth_residual(
            depth, float(s["residualMaskK"]))},
    "flyingPixelRatio": {
        "modality": "depth",
        "params": ("flyingPixelWindow", "flyingPixelPlanarityTol"),
        "measure_mask": lambda rgb, depth, s: _flying_pixel_ratio(
            depth, int(s["flyingPixelWindow"]),
            float(s["flyingPixelPlanarityTol"]))},
    "validTileCoverage": {
        "modality": "depth",
        "params": ("tileSize", "tileValidFloor"),
        "measure_mask": lambda rgb, depth, s: _valid_tile_coverage(
            depth, int(s["tileSize"]), float(s["tileValidFloor"]))},
    "clipHiFraction": {
        "modality": "rgb",
        "params": ("tauHi",),
        "measure": lambda rgb, depth, s: frame_clip_hi_fraction(rgb, float(s["tauHi"]))},
    "clipLoFraction": {
        "modality": "rgb",
        "params": ("tauLo",),
        "measure": lambda rgb, depth, s: frame_clip_lo_fraction(rgb, float(s["tauLo"]))},
}

_PAIR_OBSERVABLES = {
    "identityMedianDepthChange": {
        "params": ("changeMaskK",),
        "count_property": "identityMedianDepthChangePixelCount",
        "measure_mask": lambda d0, d1, s: _identity_median_depth_change(
            d0, d1, float(s["changeMaskK"]))},
    "jointValidDepthRatio": {
        "params": (),
        "count_property": "jointValidDepthPixelCount",
        "measure_mask": lambda d0, d1, s: _joint_valid_depth_ratio(d0, d1)},
    # This Tier-B factor is measured from the actual constant-velocity state by
    # utils.reconstruct before fitting the current link.  experiment still puts
    # its settings in the vector and mints the pair; write_pair_measurements adds
    # the value/status/count/mask after the baseline consumer run.
    "priorWarpDepthResidual": {
        "params": ("priorWarpDepthGate",),
        "count_property": "priorWarpDepthResidualPixelCount",
        "deferred": True},
}

# The declaration whitelist of §4.2, as two tuples. Anything else in
# the student section — an annotation, a status, an observable value, a run — is a
# hard error, because VERDICTS ARE COMPUTED, NEVER DECLARED. A file that could
# assert its own `hw1:qualificationStatus` would let a student write the answer
# they wanted next to the data that disagrees with it, and nothing downstream
# could tell that apart from a measurement.
_DECLARATION_EXPERIMENT_PREDICATES = (
    RDF.type, RDFS.label, HW1.batchFile, HW1.onBatch, HW1.evaluatesFactor,
    HW1.hasFactorSetting)
_DECLARATION_SETTING_PREDICATES = (
    RDF.type, HW1.settingParameter, HW1.settingRole, HW1.settingForFactor,
    HW1.settingValue)


def _fmt_term(term):
    """One RDF term as it would be written in the declaration. For error messages."""
    text = str(term)
    if isinstance(term, URIRef):
        return f"hw1:{text[len(NS):]}" if text.startswith(NS) else f"<{text}>"
    try:
        return term.n3()
    except Exception:                                    # pragma: no cover
        return repr(text)


def _fmt_triple(s, p, o):
    """`<s> <p> <o> .` in prefixed form — the offending triple every §4.2 error names."""
    return f"{_fmt_term(s)} {_fmt_term(p)} {_fmt_term(o)} ."


def _selectable_factors(factors):
    """{factor_local: value_property_local} — THE MENU of §4.2.

    Derived from the placement tables rather than listed: a QualityFactor is
    selectable iff this file knows where its observable goes and what measures it.
    The run-level factors are separately identified by their run observables.
    """
    placed = set(_FRAME_OBSERVABLES) | set(_PAIR_OBSERVABLES)
    return {name: info["over"] for name, info in factors.items()
            if info["over"] in placed}


def _run_level_factors(factors):
    """{factor_local: value_property_local} — the NOT-selectable factors (§4.2)."""
    run_observables = {"mapMeanL2", "coverageF"}
    return {name: info["over"] for name, info in factors.items()
            if info["over"] in run_observables}


def _run_factor_parameters(decls, factors):
    """The parameters every experiment records whatever it selects (§4.2).

    `icpBackend`, `maxMapMeanL2`, `minCoverageF` — derived, not listed: they are
    the Measurement and Qualification parameters whose primary factor is a
    run-level factor. They are required unconditionally because the run factors
    are evaluated unconditionally: `reconstruct.py` reads `icpBackend` out of the
    file, and `write_run` grades `mapMeanL2` and `coverageF` against the other
    two, so an experiment missing any of them cannot be reconstructed at all.

    There were four until 2026-07-31, when `minSegmentLength` was deleted from
    the TBox: nothing is listed here by name, so this function needed no edit —
    the set follows the ontology, which is the point of deriving it.
    """
    run_factors = set(_run_level_factors(factors))
    return sorted(name for name, d in decls.items()
                  if d["role"] in _EXPERIMENT_ROLES and d["primary"] in run_factors)


def _required_parameters(selected, decls, factors):
    """The REQUIRED PARAMETER SET of §4.2 (v3, selection-scoped). -> set.

    For every SELECTED factor: its `hw1:qualifiedBy` parameter (the threshold that
    turns its value into a verdict) plus every Measurement parameter whose primary
    OR affected factor is selected (the numbers that decided what the value even
    is). Plus the three run-factor parameters, always.

    WHY THE VECTOR IS TOTAL OVER THE SELECTION AND NOT OVER EVERYTHING (the v2
    rule this replaces). A recorded setting is a claim that this level was applied
    to this measurement. An experiment evaluating only depth factors that recorded
    `tauHi = 250` would be claiming a highlight threshold nothing read — as false
    as a defaulted Generation setting, and for the same reason. Totality still
    matters where it buys something: within the selection every threshold
    `status_for` needs is present, so a missing one is a bug rather than a state.

    GENERATION PARAMETERS ARE NEVER IN THIS SET. They live on the Batch, they
    describe the pixels, and absence means "not asserted" rather than "identity"
    (§5) — nothing may default a claim about pixels this program never produced.
    """
    required = set(_run_factor_parameters(decls, factors))
    for factor in selected:
        required.add(factors[factor]["qualifiedBy"])
    for name, d in decls.items():
        if d["role"] != "MeasurementSetting":
            continue
        if set((d["primary"],) + tuple(d["affects"])) & set(selected):
            required.add(name)
    return required


def _resolve_batch_file(batch_file):
    """`hw1:batchFile` -> an absolute path. Relative paths resolve against the CWD.

    Frozen in §4.2: "resolved against the current working directory
    when relative". Not against the declaration's own directory, which would be
    the other defensible rule — the declarations live in `hw1/experiments/` and
    the captures in `eval/`, so a CWD-relative path is the one a student can copy
    out of the `declare --data-dir` command they just ran at the repo root.
    """
    return batch_file if os.path.isabs(batch_file) else os.path.abspath(batch_file)


def _is_capture_dir(path):
    """True iff `path` is a capture directory: rgb/ and depth/ subdirs exist."""
    return (os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "rgb"))
            and os.path.isdir(os.path.join(path, "depth")))


_FLOOR_BATCH_NAME_RE = re.compile(r"^floor(\d+)_(.+)$")


def _floor_from_batch_name(name):
    """`floor1_baseline` -> 1. The name is floor-qualified by `batch_name`."""
    match = _FLOOR_BATCH_NAME_RE.match(name)
    if not match:
        raise ValueError(
            f"batch name {name!r} is not floor-qualified (expected "
            f"'floor<N>_<capture-dir-basename>', e.g. 'floor1_baseline')")
    return int(match.group(1))


def _sidecar_batch_ttl(data_dir):
    """Optional `<data_dir>/batch.ttl` written by the deprecated `batch2ttl`."""
    path = os.path.join(data_dir, "batch.ttl")
    return path if os.path.isfile(path) else None


def _parse_batch_ttl(path):
    """Read one Batch out of a (legacy / sidecar) batch.ttl.

    Returns (graph, batch_iri, batch_name, batch_path_literal, floor_or_None).
    """
    g = Graph()
    g.parse(path, format="turtle")
    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    if len(batches) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Batch subject, found {len(batches)}"
            + (f": {', '.join(str(b) for b in batches)}" if batches else "")
            + ". A batch file describes exactly one capture (§4.1).")
    batch = batches[0]
    name_lit = g.value(batch, HW1.batchName)
    path_lit = g.value(batch, HW1.batchPath)
    floor_lit = g.value(batch, HW1.floor)
    if name_lit is None or path_lit is None:
        raise ValueError(
            f"{path}: the batch carries no hw1:batchName and/or no hw1:batchPath, "
            f"so neither the frame IRIs nor the pixels can be reached from it "
            f"(§4.1).")
    floor = int(floor_lit) if floor_lit is not None else None
    return g, URIRef(str(batch)), str(name_lit), str(path_lit), floor


def _capture_dir(batch_path, batch_file):
    """The directory the PIXELS are in, given a (possibly stale) hw1:batchPath.

    Used for the deprecated `hw1:batchFile` -> batch.ttl hop: `batchPath` is
    whatever string was passed to `batch2ttl --data-dir`, resolved against the
    CWD; when that fails, the directory CONTAINING batch.ttl is tried, because
    that file lives inside the capture directory. Both misses are a hard error
    naming both candidates — measuring the wrong directory is the failure this
    whole resolution chain exists to prevent.
    """
    candidates = [os.path.abspath(batch_path),
                  os.path.dirname(_resolve_batch_file(batch_file))]
    for candidate in candidates:
        if _is_capture_dir(candidate):
            return candidate
    raise ValueError(
        f"the batch's hw1:batchPath {batch_path!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs. Tried "
        f"{', '.join(repr(c) for c in candidates)} (relative paths resolve against "
        f"the current working directory, {os.getcwd()!r}). Point hw1:batchFile at "
        f"the capture directory itself.")


def _resolve_declared_capture(batch_file, on_batch, path, exp, bf):
    """hw1:batchFile + hw1:onBatch -> (data_dir, batch_iri, batch_name).

    Preferred: `hw1:batchFile` names the capture directory (rgb/ + depth/).
    `hw1:onBatch` must equal `batch_iri(batch_name(data_dir, floor))`, with
    `floor` parsed from the onBatch name — that is the 'measured capture A,
    wrote into capture B' check, without a batch.ttl in the middle.

    Deprecated: `hw1:batchFile` names a batch.ttl. The capture is that file's
    `hw1:batchPath` (dirname-of-the-ttl fallback), and `hw1:onBatch` must equal
    the Batch IRI inside the file.
    """
    resolved = _resolve_batch_file(batch_file)
    if _is_capture_dir(resolved):
        if on_batch is None:
            expected = batch_iri(batch_name(resolved, 1))
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI of the capture at {batch_file!r} — it is checked against the "
                f"directory, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(expected)} .")
        declared_name = _batch_name_from_batch_iri(on_batch)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError as exc:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)}, which is not "
                f"a floor-qualified batch IRI of this assignment "
                f"(expected {NS}batch/floor<N>_<capture-dir-basename>). {exc} "
                f"Offending triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        expected_name = batch_name(resolved, floor)
        expected_iri = batch_iri(expected_name)
        if URIRef(str(on_batch)) != expected_iri:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but the capture "
                f"at {batch_file!r} is {_fmt_term(expected_iri)}. That is the "
                f"'measured capture A, wrote into capture B' bug at declaration "
                f"time — an error, not a warning (§4.2). Offending "
                f"triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return resolved, expected_iri, expected_name

    if os.path.isfile(resolved):
        _g, batch, name, stored_path, _floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file)
        if on_batch is None:
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI you believe {batch_file!r} describes — it is checked against "
                f"the file, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(batch)} .")
        if URIRef(str(on_batch)) != batch:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but {batch_file!r} "
                f"describes {_fmt_term(batch)}. That is the 'measured capture A, "
                f"wrote into capture B' bug at declaration time — an error, not a "
                f"warning (§4.2). Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return data_dir, batch, name

    raise ValueError(
        f"{path}: hw1:batchFile {batch_file!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs, or to a (deprecated) batch.ttl "
        f"({resolved!r}; relative paths resolve against the current working "
        f"directory, {os.getcwd()!r}). Point it at the capture directory. "
        f"Offending triple:\n    {_fmt_triple(exp, HW1.batchFile, bf)}")


def _declaration_setting_value(node, param_local, decl, lit, path):
    """One student `hw1:settingValue` literal -> float | int | str, typed per §4.2.

    The declared `hw1:paramValueKind` decides, never the Python type and never the
    text: a "double" parameter yields a float, an "integer" parameter an int, a
    "string" parameter the text verbatim — the same rule `parse_setting_arg`
    applies to `--gen`, and the reason `icpBackend` is an ordinary parameter
    rather than a special case.

    WHAT IS ACCEPTED, AND WHY IT IS NOT "EXACTLY THE DECLARED DATATYPE". Turtle's
    native forms are `0.03` (xsd:decimal), `20` (xsd:integer), `1.0e0`
    (xsd:double) and `"open3d"` (xsd:string), so demanding `"0.03"^^xsd:double`
    and nothing else would reject the spelling most students write first while
    changing no meaning at all. So: a "double" accepts any numeric literal, an
    "integer" accepts an INTEGER literal only (a count has no honest fractional
    reading — the same strictness `parse_setting_arg` applies on the CLI side),
    and a "string" accepts a string literal. A literal
    of the wrong FAMILY — a string for a double, a fraction for a count — is an
    error naming the triple, because that one does change the meaning.
    """
    kind = decl["kind"]
    dt = lit.datatype
    triple = _fmt_triple(node, HW1.settingValue, lit)
    numeric = (XSD.double, XSD.decimal, XSD.float, XSD.integer)
    if kind == "string":
        if dt is not None and dt != XSD.string:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"string\", "
                f"but its hw1:settingValue carries datatype {dt}. Offending triple:\n"
                f"    {triple}")
        return str(lit)
    if kind == "integer":
        if dt != XSD.integer:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"integer\" "
                f"— write it as a bare integer (e.g. 20) or "
                f"\"20\"^^xsd:integer, not as {dt or 'an untyped literal'}. Offending "
                f"triple:\n    {triple}")
        return int(lit)
    if dt not in numeric:
        raise ValueError(
            f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"double\", but "
            f"its hw1:settingValue is {dt or 'an untyped literal'}. Write "
            f"\"0.05\"^^xsd:double (or the bare Turtle form 0.05). Offending triple:\n"
            f"    {triple}")
    return float(lit)


def read_declaration(path):
    """Validate the STUDENT SECTION of a declaration. §8.1. Never writes.

    Returns

        {"exp_iri":     URIRef,
         "exp_name":    str,                    # == the file stem == the IRI tail
         "batch_file":  str,                    # as declared (capture dir, or legacy ttl)
         "batch_iri":   URIRef,
         "batch_name":  str,
         "batch_path":  str,                    # resolved capture directory
         "selected":    [factor_local, …],      # ≥ 1, menu factors only, sorted
         "given":       {param_local: value},   # student-given settings
         "required":    {param_local: value}}   # given ∪ defaults, §4.2's rule

    Reads ONLY what is above the marker, so it answers the same question for an
    unassessed declaration (`explore` view 2) and for an assessed experiment
    (`explore` view 3's given-vs-defaulted split). `experiment` calls it too;
    NEITHER re-implements a rule, which is the point of it being one function.

    EVERY §4.2 RULE, AND EVERY VIOLATION NAMES THE OFFENDING TRIPLE:

      * exactly one `hw1:Experiment` subject;
      * its IRI tail equals the file stem and matches [A-Za-z0-9_-]+ (§2) — a
        mismatch is an error rather than a choice, because picking either one
        would silently rename the student's experiment;
      * `hw1:batchFile` resolves (against the CWD) to a capture directory
        (rgb/ + depth/) whose `batch_name(dir, floor)` IRI equals the declared
        `hw1:onBatch` — the "measured capture A, wrote into capture B" bug,
        caught at declaration time instead of never. A legacy path to a
        batch.ttl is still accepted and checked the same way against that file;
      * `hw1:evaluatesFactor` is non-empty and names menu factors only; a
        run-level factor there is an error, because selection scopes INPUT quality
        and the run factors are evaluated unconditionally;
      * each FactorSetting names a TBox-declared parameter, agrees with its
        `hw1:paramRole` and its `hw1:paramPrimaryFactor`, carries a value typed by
        its `hw1:paramValueKind`, and TOUCHES THE SELECTION — a setting for a
        parameter no selected factor is primary or affected by, and which is not a
        run-factor parameter, records a level that changed nothing;
      * the WHITELIST: nothing outside the Experiment and FactorSetting predicates
        above, because verdicts are computed, never declared.

    STUDENT SETTINGS MAY BE BLANK NODES (§2/§4.5), which is why nothing here mints
    or re-opens them and why the machine section never touches them: it records
    its own defaults under the §2 scheme IRIs and lets the marker split say which
    is which.
    """
    text = _read_text(path)
    student_text, _machine_text = _split_sections(text)

    g = Graph()
    try:
        g.parse(data=student_text, format="turtle")
    except Exception as exc:
        raise ValueError(
            f"{path}: the declaration (everything above the machine marker) is not "
            f"valid Turtle: {exc}") from None

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    menu = _selectable_factors(factors)
    run_factors = _run_level_factors(factors)

    # ── the one Experiment subject, and its name ──────────────────────────────
    exp = _sole_experiment(g, path)
    exp_name = _experiment_name_from_iri(exp, path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if exp_name != stem:
        raise ValueError(
            f"{path}: the experiment names itself {exp_name!r} but the file is called "
            f"{stem!r}.ttl. The IRI tail IS the experiment's identity and so is the "
            f"file name (§2/§6); they must agree, and choosing one for "
            f"you would silently rename your experiment. Offending triple:\n"
            f"    {_fmt_triple(exp, RDF.type, HW1.Experiment)}")

    # ── the whitelist: verdicts are computed, never declared ──────────────────
    setting_nodes = set(g.objects(exp, HW1.hasFactorSetting))
    for s, p, o in g:
        if s == exp:
            if p not in _DECLARATION_EXPERIMENT_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a declared "
                    f"hw1:Experiment. A declaration states its batch, its factor "
                    f"selection and its settings; values, statuses, annotations, "
                    f"pairs and runs are COMPUTED by `api.py experiment` and written "
                    f"below the marker (§4.2). Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_EXPERIMENT_PREDICATES)}. "
                    f"Offending triple:\n    {_fmt_triple(s, p, o)}")
        elif s in setting_nodes:
            if p not in _DECLARATION_SETTING_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a hw1:FactorSetting. "
                    f"Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_SETTING_PREDICATES)} "
                    f"(§4.2). Offending triple:\n    {_fmt_triple(s, p, o)}")
        else:
            raise ValueError(
                f"{path}: {_fmt_term(s)} is neither the declared hw1:Experiment nor "
                f"one of its hw1:hasFactorSetting nodes, so nothing in this project "
                f"reads it. A declaration describes ONE experiment and its settings "
                f"and nothing else (§4.2). Offending triple:\n"
                f"    {_fmt_triple(s, p, o)}")

    # ── the batch: declaration -> batchFile -> capture directory -> pixels ──
    bf = g.value(exp, HW1.batchFile)
    if bf is None:
        raise ValueError(
            f"{path}: the experiment declares no hw1:batchFile, so nothing says which "
            f"capture to measure. Add e.g.\n"
            f"    {_fmt_term(exp)} hw1:batchFile \"eval/first_floor\" .\n"
            f"(§4.2: the path to the capture directory, resolved against "
            f"the current working directory when relative)")
    batch_file = str(bf)
    on_batch = g.value(exp, HW1.onBatch)
    data_dir, batch, batch_name_str = _resolve_declared_capture(
        batch_file, on_batch, path, exp, bf)

    # ── the selection ─────────────────────────────────────────────────────────
    selected_terms = list(g.objects(exp, HW1.evaluatesFactor))
    if not selected_terms:
        raise ValueError(
            f"{path}: the experiment selects no factor. hw1:evaluatesFactor is "
            f"required and takes 1..{len(menu)} of the menu (§4.2): "
            f"{', '.join('hw1:' + f for f in sorted(menu))}. An empty selection would "
            f"measure nothing and qualify nothing.")
    selected = []
    for term in selected_terms:
        local = _local(term)
        if local in run_factors:
            raise ValueError(
                f"{path}: hw1:{local} is a RUN-LEVEL factor and is not selectable — it "
                f"is always evaluated, by `reconstruct.py`, on the reconstruction "
                f"outcome rather than on the input rasters. hw1:evaluatesFactor scopes "
                f"INPUT-quality factors only (§4.2). Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.evaluatesFactor, term)}")
        if local not in menu:
            raise ValueError(
                f"{path}: hw1:{local} is not a factor of the menu. The menu is "
                f"{', '.join('hw1:' + f for f in sorted(menu))} (§4.2); "
                f"factor names come from the TBox, not from the declaration. Offending "
                f"triple:\n    {_fmt_triple(exp, HW1.evaluatesFactor, term)}")
        selected.append(local)
    selected = sorted(set(selected))

    # ── the settings ──────────────────────────────────────────────────────────
    run_params = set(_run_factor_parameters(decls, factors))
    given = {}
    for node in sorted(setting_nodes, key=str):
        param_term = g.value(node, HW1.settingParameter)
        if param_term is None:
            raise ValueError(
                f"{path}: a hw1:FactorSetting has no hw1:settingParameter, so nothing "
                f"says what it sets. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        local = _local(param_term)
        if local not in decls:
            raise ValueError(
                f"{path}: {_fmt_term(param_term)} is not a declared hw1:Parameter. "
                f"Parameter names come from the TBox ({_ONTOLOGY_TTL}), not from the "
                f"declaration; a typo surviving as an extra FactorSetting would record "
                f"a treatment nobody ran. Declared parameters: "
                f"{', '.join(sorted(decls))}. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")
        decl = decls[local]

        if decl["role"] == "GenerationSetting":
            raise ValueError(
                f"{path}: hw1:{local} is declared hw1:paramRole hw1:GenerationSetting. "
                f"It describes how the PIXELS were produced, so it lives on the Batch: "
                f"`api.py batch2ttl --data-dir <dir> --floor <n> --gen {local}=<value>` "
                f"(optional sidecar). Generation settings never enter an experiment, "
                f"and they are never defaulted — absence means NOT ASSERTED "
                f"(§5). Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")

        role_term = g.value(node, HW1.settingRole)
        if role_term is None or _local(role_term) != decl["role"]:
            raise ValueError(
                f"{path}: the setting of hw1:{local} declares hw1:settingRole "
                f"{_fmt_term(role_term) if role_term is not None else '(absent)'}, but "
                f"the TBox declares hw1:paramRole hw1:{decl['role']}. The role says "
                f"WHERE a setting lives and WHAT TO FIX when it is the culprit, so it "
                f"is checked rather than copied. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingRole, role_term)}")

        for_factors = list(g.objects(node, HW1.settingForFactor))
        if len(for_factors) != 1 or URIRef(str(for_factors[0])) != decl["primaryIri"]:
            found = ", ".join(_fmt_term(f) for f in for_factors) or "(none)"
            primary = _fmt_term(decl["primaryIri"])
            raise ValueError(
                f"{path}: the setting of hw1:{local} must carry exactly one "
                f"hw1:settingForFactor, its parameter's hw1:paramPrimaryFactor "
                f"{primary} — it is single-valued in v3 (§4.5) and "
                f"readers traverse hw1:paramAffectsFactor in the TBox for the rest. "
                f"Found: {found}. Write:\n"
                f"    {_fmt_triple(node, HW1.settingForFactor, decl['primaryIri'])}")

        lit = g.value(node, HW1.settingValue)
        if lit is None or not isinstance(lit, Literal):
            raise ValueError(
                f"{path}: the setting of hw1:{local} carries no hw1:settingValue "
                f"literal, so it records no level at all. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        value = _declaration_setting_value(node, local, decl, lit, path)

        touched = set((decl["primary"],) + tuple(decl["affects"]))
        if not touched & set(selected) and local not in run_params:
            raise ValueError(
                f"{path}: hw1:{local} is set, but none of the factors it touches "
                f"({', '.join('hw1:' + f for f in sorted(touched))}) is selected by "
                f"hw1:evaluatesFactor ({', '.join('hw1:' + f for f in selected)}). That "
                f"records a level that changed nothing, which is as false as a "
                f"defaulted Generation setting (§4.2) — select the factor, "
                f"or drop the setting. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")

        stored = _storable(value) if decl["kind"] == "double" else value
        if local in given and given[local] != stored:
            raise ValueError(
                f"{path}: hw1:{local} is set twice, to {given[local]!r} and to "
                f"{stored!r}. One parameter has one level per experiment; two levels "
                f"are two experiments, and the second one needs its own declaration "
                f"file (§3.1). Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingValue, lit)}")
        given[local] = stored

    # ── completeness: the selection-scoped required set (§4.2) ────────────────
    # `given` is a subset of this by construction — the scope check above rejects
    # any setting that touches nothing selected — so the union only makes
    # "student-given wins" total rather than adding anything.
    required_names = _required_parameters(selected, decls, factors) | set(given)
    required = {}
    for param in sorted(required_names):
        if param in given:
            required[param] = given[param]
            continue
        decl = decls[param]
        if decl["default"] is None:
            raise ValueError(
                f"{path}: hw1:{param} is required by this selection "
                f"(§4.2) but declares no hw1:paramDefault, and the declaration does not "
                f"set it. Either add a hw1:FactorSetting for it or declare a default "
                f"in {_ONTOLOGY_TTL}.")
        required[param] = (_storable(decl["default"]) if decl["kind"] == "double"
                           else decl["default"])

    return {"exp_iri": URIRef(str(exp)),
            "exp_name": exp_name,
            "batch_file": batch_file,
            "batch_iri": URIRef(str(batch)),
            "batch_name": batch_name_str,
            "batch_path": data_dir,
            "selected": selected,
            "given": given,
            "required": required}


def _require_settings(settings, names, what):
    """Fail loudly if a parameter a measurer needs is absent from the setting vector.

    Reachable only when the required-parameter rule of §4.2 and the measurement
    tables above disagree — a TBox that dropped a `hw1:paramAffectsFactor` link, or
    a table row whose `params` name a parameter no factor points at. The message
    points at the ontology rather than at the declaration, because an experiment
    cannot silently measure `clipHiFraction` at "whatever tau_hi happened to be":
    there is no such quantity.
    """
    missing = [n for n in names if n not in settings]
    if missing:
        raise ValueError(
            f"{what} needs parameter(s) {', '.join(missing)}, which the selection-scoped "
            f"required set (§4.2) did not put in the setting vector. That "
            f"means the TBox in {_ONTOLOGY_TTL} does not declare them as affecting the "
            f"selected factor — fix the declaration links there, not with a default here.")


def _status_property(value_local, factors):
    """The `hw1:…Status` property of the factor declared over `value_local`.

    READ from the factor's `hw1:statusProperty`, never built by appending "Status"
    to the value property's name. The naming rule happens to be mechanical, and
    that is exactly why the concatenation must not be coded: a factor whose status
    property is named any other way would silently stop being written, and the
    symptom would be "that factor always passes" (§4.5).
    """
    for info in factors.values():
        if info["over"] == value_local:
            return HW1[info["status"]]
    raise ValueError(
        f"no hw1:QualityFactor declares hw1:overProperty hw1:{value_local} in "
        f"{_ONTOLOGY_TTL}, so it has no hw1:statusProperty to write.")


def _write_observable(g, node, value_local, value, settings, factors):
    """Write one observable's VALUE and its baked STATUS onto `node`. -> True iff Pass.

    The two triples are written together, here, and nowhere else — which is what
    makes the totality promise of §4.3 mechanical rather than a habit:
    a fail-closed measurer's `inf` or `0.0` goes in exactly like any other number
    and grades itself through `status_for`. There is no path through this function
    that writes a value without a status, and none that writes a status without the
    value that justifies it.

    THE GRADED NUMBER IS THE STORED NUMBER. `_storable` is applied BEFORE the
    comparison, not only before the write, so the verdict is reproducible from the
    file: re-grade what the .ttl holds and you get back the status the .ttl holds.
    Grading the un-rounded value instead would produce `2e-01 … hw1:Fail` against a
    threshold of 0.2 — see `_storable` for why that is not hypothetical.
    """
    stored = _storable(value)
    g.add((node, HW1[value_local], _double_literal(stored)))
    status = status_for(value_local, stored, settings, factors)
    g.add((node, _status_property(value_local, factors), status))
    return status == HW1.Pass


def _fail_closed_value(value_local, factors):
    """The value to store when a measurer RAISES: the one that fails every threshold.

    §4.3: "A measurer that fails writes its fail-closed value, never
    nothing." The measurers implement that themselves for the failures they can
    see (`inf` for a lower-is-better factor, `0.0` for a higher-is-better one, §9);
    this is the outer guard for the ones they cannot — an unreadable PNG, a raster
    of the wrong dtype. It reads the direction off the TBox and returns the
    infinity on the failing side, so the guard cannot accidentally be more generous
    than a threshold somebody happened to set to 0.0.

    `hw1:meanValue` has no factor and no status, so its fallback is NaN: a value
    that is visibly not a measurement, next to no verdict at all.
    """
    for info in factors.values():
        if info["over"] == value_local:
            return float("-inf") if info["polarity"] == "higher" else float("inf")
    return float("nan")


def _measured(value_local, factors, where, measure, *args):
    """Call one measurer; on an exception store the fail-closed value and say so.

    Never swallows the failure quietly: the WARNING names the node, the observable
    and the exception on stderr, and the file records a value that FAILS. The
    alternative — letting the exception out — would throw away a whole measuring
    pass over one unreadable frame, and the alternative to that — writing nothing —
    would leave a hole that reads as "no problem here" in every table.
    """
    try:
        return measure(*args)
    except Exception as exc:                             # noqa: BLE001 — deliberate
        value = _fail_closed_value(value_local, factors)
        print(f"[experiment] WARNING: hw1:{value_local} on {where} raised "
              f"{type(exc).__name__}: {exc} — storing the fail-closed value {value!r} "
              f"(§4.3: a failing measurer writes its fail-closed value, "
              f"never nothing)", file=sys.stderr, flush=True)
        return value


def _measured_with_mask(value_local, factors, where, measure, *args):
    """Measure a scalar and its already-derived drop mask as one atomic result."""
    try:
        result = measure(*args)
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise TypeError("mask measurer must return (value, mask[, count])")
        value, mask = result[:2]
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise TypeError("mask measurer must return an HxW uint8 mask")
        count = None if len(result) == 2 else int(result[2])
        return value, array, count
    except Exception as exc:                             # noqa: BLE001 — fail closed
        value = _fail_closed_value(value_local, factors)
        print(f"[experiment] WARNING: hw1:{value_local} on {where} raised "
              f"{type(exc).__name__}: {exc} — storing {value!r} without a mask",
              file=sys.stderr, flush=True)
        return value, None, 0


def _factor_for_observable(value_local, factors):
    matches = [name for name, info in factors.items() if info["over"] == value_local]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one QualityFactor over hw1:{value_local}, got {matches}")
    return matches[0]


def _write_mask_artifact(mask, mask_root, relative_to, factor_local, filename):
    """Write one 0/255 PNG and return its portable experiment-relative path."""
    if mask_root is None or mask is None:
        return None
    factor_dir = os.path.join(mask_root, "masks", factor_local)
    os.makedirs(factor_dir, exist_ok=True)
    path = os.path.join(factor_dir, filename)
    Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(path)
    base = os.getcwd() if relative_to is None else relative_to
    return os.path.relpath(path, base).replace(os.sep, "/")


def _check_batch_file(batch_ttl, name, expected_frames):
    """Cross-check the declared batch.ttl against the frames about to be measured.

    WHY THE IRIs ARE RE-DERIVED AND NOT READ OUT OF THE BATCH FILE
        `experiment` builds its subject IRIs the way `batch2ttl` did — from
        `_pair_frames(data_dir)` and the helpers of the IRI section — rather than
        by parsing batch.ttl for frame nodes. Deriving them twice from the same
        rule is not duplication: the rule lives in ONE place (`frame_iri`,
        `component_iri`) and both commands call it. Reading them out of the file
        instead would quietly measure whatever set of frames the last `batch2ttl`
        saw rather than the frames on disk.

    SO WHY LOOK AT THE FILE AT ALL
        Because IRI identity between the two files is the ONLY join there is
        (§2), and nothing in RDF enforces it. If batch.ttl is stale —
        a frame added to `rgb/` since, a `--floor` typo, a directory renamed — the
        experiment ends up full of annotations whose `hw1:annotatesFrame` points at
        subjects no batch file declares. Every table then joins to nothing and
        shows ZERO ROWS against two perfectly well-formed files. That failure is
        invisible, and it is worth one Turtle parse to make it impossible.

        The check runs BEFORE any pixel is read, so a mismatch costs a second
        rather than a full measuring pass.

    PAIRS ARE NOT CHECKED, because batch.ttl declares none (§4.1):
    the experiment mints its own pairs under its own IRIs, so there is nothing to
    be stale about. Frames are shared structure, and they are the whole join.
    """
    g = Graph()
    g.parse(batch_ttl, format="turtle")
    declared_names = sorted({str(o) for o in g.objects(None, HW1.batchName)})
    if declared_names != [name]:
        raise ValueError(
            f"{batch_ttl} declares hw1:batchName {declared_names!r}, but this "
            f"experiment measures batch {name!r}. A batch file describes exactly one "
            f"batch, and that name is inside every frame IRI the two files join on.")

    have_frames = {str(f) for f in g.objects(batch_iri(name), HW1.hasFrame)}
    want_frames = {str(f) for f in expected_frames}

    problems = []
    missing = sorted(want_frames - have_frames)
    extra = sorted(have_frames - want_frames)
    if missing:
        problems.append(f"{len(missing)} frame(s) on disk that batch.ttl does not "
                        f"declare (e.g. {missing[0]})")
    if extra:
        problems.append(f"{len(extra)} frame(s) in batch.ttl that are not on disk "
                        f"(e.g. {extra[0]})")
    if problems:
        raise ValueError(
            f"{batch_ttl} is stale: " + "; ".join(problems) + ". Annotations join to "
            f"frames on these IRIs and on nothing else, so a mismatch makes every "
            f"report show zero rows without erroring. Re-run `api.py batch2ttl "
            f"--data-dir <dir> --floor <n>`.")


def build_machine_graph(decl, data_dir, digest, decls=None, factors=None,
                        artifact_root=None, artifact_relative_to=None):
    """Measure what the declaration selected. -> (graph, counts). The MACHINE SECTION.

    CONTRACT
        In:     decl — `read_declaration`'s dict: the validated declaration.
                data_dir — the capture directory (`_capture_dir`), which is where
                the pixels are; `_pair_frames` addresses it exactly as `batch2ttl`
                did, so the frame IRIs of the two files are the same IRIs.
                digest — the §3.1 seal over the student section, stamped here.
                decls / factors — TBox declarations; loaded if omitted.
        Out:    (graph, {"frames": n, "annotations": n, "pairs": n}).

        Emits, per §4.3 — and ONLY about the student's Experiment
        IRI, which the machine section re-opens rather than re-declares:

            <exp> hw1:declarationDigest "…64 hex…" ;
                  hw1:hasFactorSetting <…/setting/<factor>/<param>> … ;   # DEFAULTS ONLY
                  hw1:producesAnnotation <…/annotation/<n>/rgb> … ;
                  hw1:producesPair <…/pair/<i>_<j>> … .

    THE THREE RULES THAT MAKE THIS SELECTION-AWARE
      1. ONE ANNOTATION PER FRAME PER MODALITY THAT HAS A SELECTED FACTOR. A
         modality with none gets NO node — not an empty one. Totality is per
         selection: inside the declared scope a missing property is a bug, not a
         state, and outside it there is no node to be missing anything.
      2. PAIRS ARE ALWAYS MINTED, for every consecutive pair, even when no pair
         factor is selected — the pair nodes are what carries ADJACENCY into the
         experiment file, and `reconstruct.py` reads nothing else. Their
         observables follow the selection; their `hw1:qualificationStatus` is
         total, vacuously Pass when there is no pair factor to fail.
      3. ONLY DEFAULTS ARE WRITTEN AS SETTINGS. The student's own settings are
         above the marker already — re-emitting them here would give one parameter
         two nodes and, if a hand edit ever disagreed, two levels. Given vs
         defaulted is recoverable from which side of the marker a setting sits on
         (§4.2), which is exactly what `explore` prints.

    ONE DICT, READ ONCE — A SETTING DRIFT IS A LIE IN THE FILE
        The `tau_hi` handed to `frame_clip_hi_fraction`, the `hw1:settingValue` on
        the tauHi FactorSetting, and the `maxClipHiFraction` that decided the
        status all come out of ONE dict (`decl["required"]`), once. Re-deriving any
        of them — a default in a function signature, a second parse, a "sensible"
        round-trip through str — produces a file that RECORDS one treatment and
        REPORTS another. Nothing errors, every number looks plausible, and the
        experiment is unreproducible forever.

    WHAT IS DELIBERATELY ABSENT
        No `hw1:hasFrame`, no `schema:contentUrl`, no `hw1:Batch` type, no floor:
        those are STRUCTURE, derived from the capture directory at measure time
        (and optionally recorded in a deprecated batch.ttl sidecar). The
        annotations reach structure through `hw1:annotatesFrame` and
        `hw1:describesImage`, and that IRI identity is the entire join. No run node
        either — `hw1:mapMeanL2` and `hw1:coverageF` are written later by
        `write_run`; measurement does not know the reconstruction outcome and must
        not pretend to.
    """
    decls = load_parameter_declarations(_ONTOLOGY_TTL) if decls is None else decls
    factors = load_quality_factors(_ONTOLOGY_TTL) if factors is None else factors

    exp = decl["exp_iri"]
    expname = decl["exp_name"]
    name = decl["batch_name"]
    settings = decl["required"]
    selected = decl["selected"]

    # The selection, translated once from factor names to the observables that
    # carry them, in table order so the file is deterministic.
    menu = _selectable_factors(factors)
    chosen = {menu[f] for f in selected}
    frame_values = [v for v in _FRAME_OBSERVABLES if v in chosen]
    pair_values = [v for v in _PAIR_OBSERVABLES if v in chosen]
    modalities = [k for k in _ANNOTATION_KINDS
                  if any(_FRAME_OBSERVABLES[v]["modality"] == k for v in frame_values)]

    # Every parameter every selected measurer needs, checked against the vector
    # BEFORE the first PNG is opened.
    for value_local in frame_values:
        _require_settings(settings, _FRAME_OBSERVABLES[value_local]["params"],
                          f"hw1:{value_local}")
    for value_local in pair_values:
        _require_settings(settings, _PAIR_OBSERVABLES[value_local]["params"],
                          f"hw1:{value_local}")

    g = Graph()

    # ── the seal (§3.1) ──────────────────────────────────────────
    # First triple of the machine section, and the reason the rest of it can be
    # trusted to describe the declaration above it.
    g.add((exp, HW1.declarationDigest, Literal(digest)))

    # ── the defaulted half of the setting vector (§4.2) ───────────────────────
    for param in sorted(set(settings) - set(decl["given"])):
        d = decls[param]
        s = setting_iri(expname, d["primary"], param)
        g.add((exp, HW1.hasFactorSetting, s))
        g.add((s, RDF.type, HW1.FactorSetting))
        g.add((s, HW1.settingParameter, d["iri"]))
        # The role is DENORMALISED onto the node on purpose: `explore` then reads
        # the verdict ("regenerate the data" / "re-measure" / "re-qualify") off the
        # node it already has, with no second join into the TBox.
        g.add((s, HW1.settingRole, HW1[d["role"]]))
        # SINGLE-VALUED in v3 (§4.5): the primary factor only. Readers traverse
        # hw1:paramAffectsFactor in the TBox for the rest, which is also what makes
        # a student's blank-node setting legal — nothing has to re-open it.
        g.add((s, HW1.settingForFactor, d["primaryIri"]))
        g.add((s, HW1.settingValue, _setting_value_literal(settings[param], d["kind"])))

    # ── one FrameAnnotation per frame per SELECTED modality ───────────────────
    frames = _pair_frames(data_dir)
    total = len(frames)
    n_annotations = 0
    for n, (stem, rgb_path, depth_path) in enumerate(frames, start=1):
        for kind in modalities:
            ann = annotation_iri(expname, stem, kind)
            g.add((exp, HW1.producesAnnotation, ann))
            g.add((ann, RDF.type, HW1.FrameAnnotation))
            g.add((ann, HW1.annotatesFrame, frame_iri(name, stem)))
            # frameIndex is repeated here although the Frame already carries it: it
            # is what lets a report order by capture order without loading the batch
            # file at all (§2).
            g.add((ann, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))
            # describesImage (new in v3): the annotation's link to the raster it
            # measured. annotatesFrame stays for frame-level joins; this one says
            # WHICH OF THE TWO IMAGES the numbers on this node are about.
            g.add((ann, HW1.describesImage, component_iri(name, stem, kind)))

            passed = []
            for value_local in frame_values:
                if _FRAME_OBSERVABLES[value_local]["modality"] != kind:
                    continue
                spec = _FRAME_OBSERVABLES[value_local]
                if "measure_mask" in spec:
                    value, mask, _count = _measured_with_mask(
                        value_local, factors, f"frame {stem} ({kind})",
                        spec["measure_mask"], rgb_path, depth_path, settings)
                    factor_local = _factor_for_observable(value_local, factors)
                    mask_file = _write_mask_artifact(
                        mask, artifact_root, artifact_relative_to, factor_local,
                        f"{stem}.png")
                    if mask_file is not None:
                        g.add((ann, HW1.maskFile, Literal(mask_file)))
                else:
                    value = _measured(value_local, factors,
                                      f"frame {stem} ({kind})", spec["measure"],
                                      rgb_path, depth_path, settings)
                passed.append(_write_observable(g, ann, value_local, value,
                                                settings, factors))
            if kind == "rgb":
                # meanValue: the value and NO status, whenever an rgb annotation
                # exists (§4.3). There is no QualityFactor over it, so there is no
                # polarity and no threshold — asking `status_for` for one raises by
                # design. It ships so a student can check "is the picture dark on
                # average?" against the factors that actually predict reconstruction
                # failure, and discover that it does not.
                g.add((ann, HW1.meanValue, _double_literal(
                    _measured("meanValue", factors, f"frame {stem} (rgb)",
                              lambda p: frame_mean_value(p), rgb_path))))
            # The aggregate of §4.5, on the node that says what was aggregated:
            # Pass iff every SELECTED factor of THIS modality passed.
            g.add((ann, HW1.qualificationStatus,
                   HW1.Pass if all(passed) else HW1.Fail))
            n_annotations += 1
        if n % _PROGRESS_EVERY == 0 or n == total:
            print(f"[experiment] frames {n}/{total}", file=sys.stderr, flush=True)

    # ── one FramePair per consecutive pair, ALWAYS ────────────────────────────
    steps = list(zip(frames, frames[1:]))
    total_pairs = len(steps)
    # `enumerate(..., start=0)` is the hw1:pairIndex: a 0-based ORDINAL over the
    # pairs in ascending order, NOT a frame stem (§4.3). The stems are
    # in the IRI, where a gap in the capture stays visible as `41_43`; the ordinal
    # is what a reader sorts by, because unpadded stems sort lexicographically as
    # 0, 1, 10, 100, 11 and a time series read in that order looks like noise.
    for pair_index, ((s0, _, d0), (s1, _, d1)) in enumerate(steps):
        i, j = int(s0), int(s1)
        p = pair_iri(expname, i, j)
        g.add((exp, HW1.producesPair, p))
        g.add((p, RDF.type, HW1.FramePair))
        g.add((p, HW1.sourceFrame, frame_iri(name, s0)))
        g.add((p, HW1.targetFrame, frame_iri(name, s1)))
        g.add((p, HW1.pairIndex, Literal(pair_index, datatype=XSD.integer)))

        pair_passed = []
        for value_local in pair_values:
            spec = _PAIR_OBSERVABLES[value_local]
            if spec.get("deferred"):
                continue
            value, mask, count = _measured_with_mask(
                value_local, factors, f"pair {i}_{j}", spec["measure_mask"],
                d0, d1, settings)
            pair_passed.append(_write_observable(g, p, value_local, value,
                                                 settings, factors))
            if spec.get("count_property") is not None:
                g.add((p, HW1[spec["count_property"]],
                       Literal(int(count or 0), datatype=XSD.integer)))
            factor_local = _factor_for_observable(value_local, factors)
            mask_file = _write_mask_artifact(
                mask, artifact_root, artifact_relative_to, factor_local,
                f"{i}_{j}.png")
            if mask_file is not None:
                g.add((p, HW1.maskFile, Literal(mask_file)))
        # `all([])` is True, and that is the contract: a pair carries a TOTAL
        # qualificationStatus, vacuously Pass when no pair factor was selected
        # (§4.3/§4.5) — the pair is minted for its adjacency, not for a verdict
        # nobody asked for.
        g.add((p, HW1.qualificationStatus,
               HW1.Pass if all(pair_passed) else HW1.Fail))
        if pair_values and ((pair_index + 1) % _PROGRESS_EVERY == 0
                            or pair_index + 1 == total_pairs):
            print(f"[experiment] pairs {pair_index + 1}/{total_pairs}",
                  file=sys.stderr, flush=True)

    return g, {"frames": total, "annotations": n_annotations, "pairs": total_pairs}


def cmd_experiment(args):
    """Assess ONE student declaration, ONCE, into its own file (§3.1).

    The six steps, in order, each of which is a contract somewhere:

      1. WRITE-ONCE CHECK. A file that already carries `MACHINE_MARKER` is a hard
         error — no `--force`, no re-assess path. One assessment per file; every
         tuning is a brand-new experiment under a brand-new name, and the notebook
         stays honest because nothing in it is ever superseded in place.
      2. VALIDATE THE DECLARATION (`read_declaration`, §4.2): one Experiment, the
         name, the batch, the selection, the settings, the whitelist. Every
         violation names the offending triple.
      3. RESOLVE THE PIXELS: declaration -> hw1:batchFile (the capture
         directory, or a deprecated batch.ttl) -> rgb/ + depth/, and cross-check
         hw1:onBatch against the directory BEFORE reading a pixel — a mismatched
         batch name makes every downstream table join to nothing, and finding
         that out afterwards costs the whole pass.
      4. SEAL. sha256 of the file's bytes up to the marker line, stamped as
         `hw1:declarationDigest`. Without it, write-once is unenforceable.
      5. MEASURE AND QUALIFY what was declared, from the ONE settings dict
         (`build_machine_graph`).
      6. APPEND — never rewrite. The declaration is opened in APPEND mode, so
         there is no code path in this command that can damage the half of the
         file a human wrote.
    """
    path = args.declaration
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such declaration file")

    text = _read_text(path)
    if _marker_offset(text) is not None:
        raise ValueError(
            f"{path} HAS ALREADY BEEN ASSESSED — it carries the machine marker, so it "
            f"already holds one experiment's values and verdicts. Experiments are "
            f"WRITE-ONCE (§3.1): there is no --force and no re-assess "
            f"path, because a re-assessment would silently replace the evidence in "
            f"your lab notebook. To try a different threshold, factor selection or "
            f"batch: COPY THE DECLARATION — every line ABOVE the machine marker — "
            f"into a NEW file, edit it there (including the hw1:Experiment IRI tail, "
            f"which must equal the new file's stem), and run "
            f"`api.py experiment <new-name>.ttl`. The old file stays as it is: that "
            f"is what makes hw1/experiments/ a lab notebook.")

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    decl = read_declaration(path)

    required = decl["required"]
    given = decl["given"]
    defaulted = sorted(set(required) - set(given))
    shown_given = ", ".join("{}={!r}".format(k, given[k]) for k in sorted(given))
    shown_default = ", ".join("{}={!r}".format(k, required[k]) for k in defaulted)
    print(f"[experiment] declaration {path}")
    print(f"[experiment] experiment {decl['exp_name']!r} on batch "
          f"{decl['batch_name']!r} (via {decl['batch_file']})")
    print(f"[experiment] evaluates {len(decl['selected'])} factor(s): "
          f"{', '.join(decl['selected'])}")
    print(f"[experiment] settings given:      {shown_given or '(none)'}")
    print(f"[experiment] settings defaulted:  {shown_default or '(none)'}")
    n_meas = sum(1 for k in required if decls[k]["role"] == "MeasurementSetting")
    n_qual = sum(1 for k in required if decls[k]["role"] == "QualificationSetting")
    print(f"[experiment] setting vector: {len(required)} setting(s) — {n_meas} "
          f"Measurement (how pixels are scored) + {n_qual} Qualification (where the "
          f"Pass line sits), total over THIS SELECTION plus the three run-factor "
          f"parameters (§4.2). Generation settings live on the batch.")

    data_dir = decl["batch_path"]
    frames = _pair_frames(data_dir)
    # Legacy declarations still name a batch.ttl: keep the stale-file check so a
    # sidecar that drifted from the pixels cannot silently join to nothing.
    declared = _resolve_batch_file(decl["batch_file"])
    if os.path.isfile(declared) and not _is_capture_dir(declared):
        _check_batch_file(declared, decl["batch_name"],
                          [frame_iri(decl["batch_name"], stem)
                           for stem, _, _ in frames])

    # THE SEAL COVERS THE BYTES THAT WILL BE ABOVE THE MARKER. If the declaration
    # does not end in a newline, one is appended so the marker starts its own line —
    # and the digest is computed over the text INCLUDING that newline, because that
    # is what the file will hold and what `read_experiment` will re-hash.
    student_text = text if text.endswith("\n") else text + "\n"
    digest = _declaration_digest(student_text.encode("utf-8"))

    artifact_root = os.path.splitext(os.path.abspath(path))[0]
    g, counts = build_machine_graph(
        decl, data_dir, digest, decls=decls, factors=factors,
        artifact_root=artifact_root,
        artifact_relative_to=os.path.dirname(os.path.abspath(path)))

    body = g.serialize(format="turtle")
    if isinstance(body, bytes):                          # rdflib < 6 returned bytes
        body = body.decode("utf-8")
    # APPEND, so no failure in this command can touch the student's declaration.
    with open(path, "a", encoding="utf-8") as fh:
        if not text.endswith("\n"):
            fh.write("\n")
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")

    # The verdict counts, printed because they are the number a student actually
    # wants next and because a selection that will turn out empty is visible here
    # rather than three commands later.
    annotations = set(g.subjects(RDF.type, HW1.FrameAnnotation))
    pairs = set(g.subjects(RDF.type, HW1.FramePair))
    bad_ann = sum(1 for a in annotations
                  if g.value(a, HW1.qualificationStatus) == HW1.Fail)
    bad_pairs = sum(1 for p in pairs
                    if g.value(p, HW1.qualificationStatus) == HW1.Fail)
    n_ann, n_pairs = len(annotations), len(pairs)
    modalities = ", ".join(_modalities_of(g)) or "no modality selected"
    print(f"[experiment] batch {decl['batch_name']!r}: {counts['frames']} frames -> "
          f"{n_ann} annotation(s) [{modalities}], {n_pairs} pair(s)")
    print(f"[experiment] verdicts: {n_ann - bad_ann}/{n_ann} annotations Pass, "
          f"{n_pairs - bad_pairs}/{n_pairs} pairs Pass (under the {n_qual} thresholds "
          f"recorded above)")
    print(f"[experiment] experiment IRI  {decl['exp_iri']}")
    print(f"[experiment] declarationDigest {digest}")
    print(f"[experiment] appended {len(g)} triples below the marker -> {path}")
    print(f"[experiment] this file is now SEALED: edit it above the marker and every "
          f"reader will refuse it. Next: `api.py explore {path}`, then "
          f"`reconstruct.py --data_root {data_dir} --experiment {path}`.")
    return 0


def _modalities_of(g):
    """The annotation modalities present in a machine graph, for the summary line."""
    kinds = []
    for kind in _ANNOTATION_KINDS:
        if any(str(a).endswith(f"/{kind}")
               for a in g.subjects(RDF.type, HW1.FrameAnnotation)):
            kinds.append(kind)
    return kinds


# =============================================================================
# explore  — read-only terminal tables (§7.1)
#   Four views: a batch file, an unassessed declaration, an assessed experiment
#   (settings, per-frame table, summary and the computed VERDICT section), and
#   several experiments side by side. It measures nothing and writes nothing, and
#   it re-implements no rule: `read_declaration` and `read_experiment` above are
#   its two inputs, and the TBox wiring (`overProperty` / `statusProperty` /
#   `polarity` / `qualifiedBy` / `paramPrimaryFactor` / `paramAffectsFactor`) is
#   how it resolves factors and culprit settings generically — exactly as the
#   deleted SPARQL queries did (§10).
#
#   READ-ONLY, AND THAT IS AN INVARIANT OF THIS WHOLE SECTION. Nothing below
#   opens a file for writing, calls `write_run`, serializes a graph to disk or
#   mutates a parsed graph. The graphs it parses are throwaway projections of
#   files on disk; the files themselves are never reopened after being read.
#
#   THE SEAL IS NOT SHORT-CIRCUITED. View 3 goes through `read_experiment`,
#   which verifies `hw1:declarationDigest` before returning anything (§3.1), so a
#   file whose declaration was edited after assessment raises here instead of
#   printing a pretty table of stale numbers.
#
#   THE WIDE-TABLE DECISION (§7.1 asks for one, so it is stated here rather than
#   left to the reader): the per-frame table PRINTS EVERY ROW and keeps every
#   COLUMN narrow. A 6-factor selection over a 387-frame batch is 387 lines of
#   117 characters (measured) — one line per frame, value and status merged into
#   cell ("0.0312 P"), long observable names abbreviated with a legend printed
#   above the table. Rows are cheap: a terminal has scrollback, and plain ASCII
#   one-line-per-frame means `explore … | grep ' F'` finds every failing frame.
#   Columns are not cheap: a wrapped line destroys the alignment that makes a
#   column scannable at all. The alternative — eliding all-Pass rows — was
#   REJECTED because `query` is deleted (§10) and the frozen command surface (§7)
#   gives `explore` no flag with which to ask for an elided row back, so an
#   elision here would hide a measured value with no way to recover it. Nothing
#   in this section truncates anything silently; where a list is long it is
#   printed in full or the count of what was left out is printed with it.
# =============================================================================

# What to do about a culprit setting, by its parameter's hw1:paramRole
# (§4.2/§7.1). Keyed by role — NOT by factor and NOT by parameter:
# adding a menu factor, or a parameter for one, must require no edit here.
_ROLE_FIX = {
    "GenerationSetting": "regenerate the data (this level is baked into the pixels)",
    "MeasurementSetting": "change the number, re-measure -> a NEW experiment",
    "QualificationSetting": "change the threshold, re-qualify -> a NEW experiment",
}

# Width at which an observable's local name is abbreviated in a per-frame column
# header. `_abbrev` falls back to full names if truncation would collide, so this
# is a cosmetic bound and never an ambiguity.
_COL_ABBREV = 10

_INDENT = "  "

# Prose is wrapped at this column; tables never are (see the section header).
_WRAP = 92


# ── output primitives: plain ASCII, aligned columns, no dependency ────────────
def _rule(title):
    """A top-level section heading."""
    print()
    print(title)
    print("=" * len(title))


def _sub(title):
    """A block heading inside a view."""
    print()
    print(title)
    print("-" * len(title))


def _kv(label, value):
    """One `label   value` line of a header block.

    Deliberately NOT wrapped: the values here are IRIs and file paths, and a
    wrapped path cannot be copied out of a terminal in one go.

    The gap is a MINIMUM, not a fixed width: experiment names are student-chosen
    and routinely overrun 20 columns, and a name run flush into the path it labels
    is unreadable exactly where the reader most needs to tell them apart.
    """
    gap = max(20 - len(label), 2)
    print(f"{_INDENT}{label}{' ' * gap}{value}")


def _note(text, indent=_INDENT):
    """Prose, greedy-wrapped to `_WRAP` columns. Tables are never wrapped; notes are.

    A wrapped TABLE loses the column alignment that makes it scannable, so the
    per-frame table stays on one line per frame however wide it gets (see the
    section header). A wrapped SENTENCE loses nothing, and the explanations
    around these tables carry the contract references a reader needs.
    """
    line = ""
    for word in text.split():
        if line and len(indent) + len(line) + 1 + len(word) > _WRAP:
            print(indent + line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(indent + line)


def _render_table(headers, rows, right=()):
    """Aligned ASCII table -> a list of lines. `right` = column indices to right-align.

    Deliberately hand-rolled: `explore` must run under the same rdflib-only stack
    as everything else (§7/§10 removed the last non-stdlib
    dependency this project had beyond numpy/Pillow/rdflib), and a table is
    twelve lines of `str.ljust`.
    """
    head = [str(h) for h in headers]
    body = [["" if c is None else str(c) for c in row] for row in rows]
    widths = [len(h) for h in head]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        out = [(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]))
               for i, cell in enumerate(cells)]
        return (_INDENT + "  ".join(out)).rstrip()

    lines = [fmt(head), _INDENT + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in body)
    return lines


def _print_table(headers, rows, right=()):
    """`_render_table`, printed. An empty table says so rather than printing nothing."""
    if not rows:
        print(f"{_INDENT}(no rows)")
        return
    for line in _render_table(headers, rows, right):
        print(line)


def _fmt_num(term):
    """One numeric literal as a narrow fixed-shape string. INF / -INF / NaN survive.

    The fail-closed sentinels of §9 are values a reader must SEE — an `inf`
    printed as `1.0e+308` or, worse, silently reformatted to something finite,
    would hide exactly the rows the convention exists to surface.
    """
    if term is None:
        return "-"
    try:
        v = float(term)
    except (TypeError, ValueError):
        return str(term)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "INF"
    if v == float("-inf"):
        return "-INF"
    if v == 0.0:
        return "0.0000"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _fmt_setting(value):
    """One recorded setting level, printed the way its Python type reads."""
    if isinstance(value, float):
        return _fmt_num(value)
    return str(value)


def _fmt_status(term, short=False):
    """hw1:Pass / hw1:Fail / absent -> "Pass" | "Fail" | "-" (or P / F / - ).

    Compared against the two hw1:Status individuals (§4.5), never against a
    string: a status is a term, and `str(term).endswith("Pass")` would also match
    an IRI from some other vocabulary that happens to end that way.
    """
    if term == HW1.Pass:
        return "P" if short else "Pass"
    if term == HW1.Fail:
        return "F" if short else "Fail"
    if term is None:
        return "-"
    return _local(term)


def _abbrev(names, width=_COL_ABBREV):
    """{name: short name} for column headers, or identity if truncation collides."""
    short = {n: n[:width] for n in names}
    if len(set(short.values())) != len(short):
        return {n: n for n in names}
    return short


def _annotation_kind(ann):
    """Annotation IRI -> its modality segment ("rgb" | "depth"), or None.

    The other direction of `annotation_iri`, and narrow on purpose: §8.2's
    `read_experiment` dict does not carry the modality, but §7.1's per-frame
    table asks for a per-modality `hw1:qualificationStatus` column. Reading the
    frozen last segment (§2) is the same one-line rule `_modalities_of` already
    applies; it is not a second implementation of the frame-index tail parse,
    which stays in `frame_index_from_iri`.
    """
    text = str(ann)
    for kind in _ANNOTATION_KINDS:
        if text.endswith("/" + kind):
            return kind
    return None


# ── view dispatch: by CONTENT, never by filename (§7.1) ──────────
def _classify(path):
    """What kind of thing this is: "batch" | "declaration" | "experiment".

    BY CONTENT, not by name. A capture directory (rgb/ + depth/) is a batch;
    a `hw1:Batch` subject is a (deprecated) batch.ttl; a `hw1:Experiment`
    subject is an experiment file, assessed iff the file carries
    `MACHINE_MARKER` (§3.1). Anything else is an error that NAMES what it
    found, because "unrecognised file" with no evidence is the least useful
    error a reader can get.
    """
    if _is_capture_dir(path):
        return "batch"
    text = _read_text(path)
    g = Graph()
    try:
        g.parse(data=text, format="turtle")
    except Exception as exc:
        raise ValueError(f"{path}: not valid Turtle: {exc}") from None

    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    assessed = _marker_offset(text) is not None

    if batches and exps:
        raise ValueError(
            f"{path}: this file declares BOTH a hw1:Batch ({batches[0]}) and a "
            f"hw1:Experiment ({exps[0]}). §3 keeps them in separate "
            f"files — structure in the capture directory, measurement in "
            f"hw1/experiments/<expname>.ttl — so `explore` cannot tell which view "
            f"you want.")
    if batches:
        return "batch"
    if exps:
        return "experiment" if assessed else "declaration"

    types = sorted({_local(o) for o in g.objects(None, RDF.type)})
    raise ValueError(
        f"{path}: neither a capture directory nor an experiment file. `explore` "
        f"needs a directory with rgb/ + depth/ (the batch view) or a subject typed "
        f"hw1:Experiment (the declaration / assessed views). This file declares "
        + (f"{len(g)} triple(s) with rdf:type " + ", ".join('hw1:' + t for t in types)
           if types else f"{len(g)} triple(s) and no rdf:type at all")
        + ".")


# ── the batch view (§7.1 view 1) ────────────────────────────────
def _generation_from_batch_graph(g, b, path):
    """[(param_local, value), ...] from hw1:hasGenerationSetting on a Batch node."""
    generation = []
    for s in sorted(g.objects(b, HW1.hasGenerationSetting), key=str):
        param = g.value(s, HW1.settingParameter)
        lit = g.value(s, HW1.settingValue)
        local = _local(param) if param is not None else "(no settingParameter)"
        value = (_setting_value_to_python(lit, path, local) if lit is not None
                 else None)
        generation.append((local, value))
    return generation


def _batch_facts_from_ttl(path):
    """Batch header/table facts from a (deprecated) batch.ttl sidecar."""
    g, b, name, stored_path, floor = _parse_batch_ttl(path)
    frames = []
    for f in g.objects(b, HW1.hasFrame):
        idx_lit = g.value(f, HW1.frameIndex)
        idx = int(idx_lit) if idx_lit is not None else frame_index_from_iri(f)
        rgb = g.value(f, HW1.hasRGBImage)
        depth = g.value(f, HW1.hasDepthImage)
        frames.append((idx,
                       g.value(rgb, SCHEMA.contentUrl) if rgb is not None else None,
                       g.value(depth, SCHEMA.contentUrl) if depth is not None else None))
    frames.sort(key=lambda row: row[0])
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": path,
            "iri": b,
            "name": name,
            "batch_path": stored_path,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "generation": _generation_from_batch_graph(g, b, path),
            "derived_from": g.value(b, PROV.wasDerivedFrom)}


def _batch_facts_from_capture(data_dir, declared_name=None):
    """Batch header/table facts from the rasters on disk.

    An optional `<data_dir>/batch.ttl` sidecar still supplies generation
    provenance; the frame list always comes from `_pair_frames`.
    """
    sidecar = _sidecar_batch_ttl(data_dir)
    generation, derived_from, floor, name, iri = [], None, None, None, None
    if sidecar is not None:
        try:
            sidecar_facts = _batch_facts_from_ttl(sidecar)
            generation = sidecar_facts["generation"]
            derived_from = sidecar_facts["derived_from"]
            floor = sidecar_facts["floor"]
            name = sidecar_facts["name"]
            iri = sidecar_facts["iri"]
        except Exception:                                # noqa: BLE001 — sidecar is optional
            pass
    frames = [(int(stem), rgb, depth) for stem, rgb, depth in _pair_frames(data_dir)]
    if declared_name is not None:
        name = declared_name
        iri = batch_iri(declared_name)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError:
            pass
    elif name is None:
        floor = 1 if floor is None else int(floor)
        name = batch_name(data_dir, floor)
        iri = batch_iri(name)
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": sidecar if sidecar is not None else data_dir,
            "iri": iri,
            "name": name,
            "batch_path": data_dir,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "generation": generation,
            "derived_from": derived_from}


def _batch_facts(path, declared_name=None):
    """Everything the batch header and frame table need.

    Accepts a capture directory (preferred) or a batch.ttl (deprecated sidecar).
    Shared by view 1 (the batch itself) and view 2 (the capture a declaration
    names in `hw1:batchFile`), so the two print the SAME header from the same
    code.
    """
    resolved = path if os.path.isabs(path) else os.path.abspath(path)
    if _is_capture_dir(resolved):
        return _batch_facts_from_capture(resolved, declared_name=declared_name)
    if os.path.isfile(resolved):
        return _batch_facts_from_ttl(resolved)
    raise ValueError(
        f"{path}: not a capture directory (rgb/ + depth/) and not a batch.ttl "
        f"({resolved!r}).")


def _print_batch_header(facts):
    """The batch header of §7.1: name, floor, path, frame count, generation settings."""
    _kv("batch", facts["name"])
    _kv("floor", facts["floor"])
    _kv("batchPath", facts["batch_path"])
    if os.path.isfile(str(facts["path"])):
        _kv("batch.ttl", facts["path"])
    stems = [idx for idx, _, _ in facts["frames"]]
    span = f", stems {stems[0]}..{stems[-1]}" if stems else ""
    _kv("frames", f"{len(stems)}{span}")
    if facts["derived_from"] is not None:
        _kv("derivedFrom", _fmt_term(facts["derived_from"]))
    if facts["generation"]:
        _kv("generation", ", ".join(f"{k}={_fmt_setting(v)}"
                                    for k, v in facts["generation"]))
    else:
        # Absence is a claim about nothing, not a claim of cleanliness (§5), and
        # it is also why the Generation verdict cannot fire on the shipped
        # corrupted captures (§12 O-F). Say so here rather than let a reader
        # infer "uncorrupted" from a blank line.
        _kv("generation", "none recorded — NOT ASSERTED, not 'uncorrupted' "
                          "(§5)")

    gaps = facts["gaps"]
    if gaps:
        shown = ", ".join(f"{i}->{j} ({j - i - 1} stem(s) missing)" for i, j in gaps)
        _kv("stem gaps", f"{len(gaps)}: {shown}")
        _note("every experiment over this batch pairs ACROSS those gaps, so those "
              "pairs span more than one capture step and their observables read as "
              "larger motion.", indent=_INDENT + " " * 20)
    else:
        _kv("stem gaps", "none — the paired stems are consecutive")


def _view_batch(path):
    """View 1: batch header + the frame table, with stem gaps called out."""
    facts = _batch_facts(path)
    _rule(f"BATCH  {facts['name']}")
    _print_batch_header(facts)

    _sub("frames")
    rows = [(idx, rgb, depth) for idx, rgb, depth in facts["frames"]]
    _print_table(("frame", "rgb", "depth"), rows, right=(0,))
    _note(f"{len(rows)} frame(s), all shown. A capture holds no measured value by "
          f"contract (§4.1); write a declaration naming it in hw1:batchFile and run "
          f"`api.py experiment` to measure one.")
    return 0


# ── the declaration view (§7.1 view 2) ──────────────────────────
def _selection_rows(selected, factors, settings):
    """One row per selected factor: what it measures, which way, against which number."""
    rows = []
    for f in selected:
        info = factors[f]
        threshold = settings.get(info["qualifiedBy"])
        rows.append((f, "hw1:" + info["over"],
                     "higher is better" if info["polarity"] == "higher"
                     else "lower is better",
                     info["qualifiedBy"],
                     "-" if threshold is None else _fmt_setting(threshold)))
    return rows


def _settings_rows(names, values, decls, source_of):
    """One row per recorded/required parameter: value, role, primary factor, source."""
    rows = []
    for param in sorted(names):
        d = decls.get(param)
        rows.append((param,
                     _fmt_setting(values[param]),
                     d["role"] if d else "(undeclared)",
                     d["primary"] if d else "-",
                     source_of(param)))
    return rows


def _view_declaration(path):
    """View 2: an unassessed declaration — batch header, selection, settings table."""
    decl = read_declaration(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)

    _rule(f"DECLARATION  {decl['exp_name']}  (not yet assessed)")
    _kv("file", path)
    _kv("experiment IRI", decl["exp_iri"])
    _kv("state", "no MACHINE SECTION marker — no values and no verdicts exist yet")
    if "PREDICTION" not in _read_text(path).upper():
        _note("Add a `# PREDICTION` comment before assessment: expected input "
              "Fails, baseline outcome, selected outcome, and the mechanism that "
              "would produce that differential. The declaration digest seals it "
              "with the treatment (§14).")

    _sub("batch (via hw1:batchFile)")
    _print_batch_header(_batch_facts(decl["batch_path"],
                                    declared_name=decl["batch_name"]))

    _sub(f"selection — {len(decl['selected'])} factor(s) of the "
         f"{len(_selectable_factors(factors))}-factor menu")
    _print_table(("factor", "observable", "polarity", "qualifiedBy", "threshold"),
                 _selection_rows(decl["selected"], factors, decl["required"]))

    _sub("settings — the required set of §4.2, total over THIS selection")
    given = decl["given"]
    _print_table(
        ("parameter", "value", "role", "primary factor", "source"),
        _settings_rows(decl["required"], decl["required"], decls,
                       lambda p: "declared" if p in given else "WILL BE DEFAULTED"),
        right=(1,))
    _note(f"{len(given)} declared, {len(decl['required']) - len(given)} will be "
          f"filled from hw1:paramDefault and recorded below the marker. Generation "
          f"parameters are never in this set: they live on the batch and are never "
          f"defaulted (§5).")
    print(f"{_INDENT}Next: `api.py experiment {path}` — once, ever (§3.1).")
    return 0


# ── the assessed view (§7.1 view 3) ─────────────────────────────
def _section_settings(path, exp_iri):
    """({param: value} above the marker, {param: value} below it). The §4.2 split.

    Given-versus-defaulted is recorded by WHICH SIDE OF THE MARKER a setting sits
    on (§4.2) — it is not derivable from the setting node itself, because a
    student's setting may be a blank node and a machine-minted one uses the §2
    scheme IRI, and neither shape is a promise about who wrote it. So the two
    halves are parsed separately and read by the SAME reader
    (`_experiment_settings`), which is also what keeps the "one parameter, one
    level" check applying to both.
    """
    student_text, machine_text = _split_sections(_read_text(path))
    sides = []
    for text in (student_text, machine_text or ""):
        g = Graph()
        g.parse(data=text, format="turtle")
        sides.append(_experiment_settings(g, exp_iri, path))
    return sides[0], sides[1]


def _observable_placement(g, exp, factors, selected):
    """{factor: "frame" | "pair"} — read off WHERE each observable actually landed.

    Generic by construction: the placement table of `build_machine_graph` is not
    consulted and no factor is named, so a menu factor added to the TBox and
    measured onto either node class appears in the right column group with no
    edit here. A selected factor whose observable is nowhere in the file falls
    back to the frame group and prints "-" in every row, which is visible rather
    than silent.
    """
    on_ann, on_pair = set(), set()
    for ann in g.objects(exp, HW1.producesAnnotation):
        on_ann.update(_local(p) for p in g.predicates(ann, None))
    for pair in g.objects(exp, HW1.producesPair):
        on_pair.update(_local(p) for p in g.predicates(pair, None))
    placement = {}
    for f in selected:
        over = factors[f]["over"]
        placement[f] = "pair" if (over in on_pair and over not in on_ann) else "frame"
    return placement


def _collect_frame_rows(g, exp, factors, selected, placement, path):
    """The per-frame table's data: annotations by frame+modality, pairs by source frame.

    Values are found by the factor's `hw1:overProperty` and statuses by its
    `hw1:statusProperty`, both read from the TBox — never by concatenating
    "Status" onto a name and never by branching on which factor it is.
    """
    ann_values, ann_aggregate, means, kinds = {}, {}, {}, []
    for ann in g.objects(exp, HW1.producesAnnotation):
        idx = _annotation_frame_index(g, ann, path)
        kind = _annotation_kind(ann)
        if kind is not None and kind not in kinds:
            kinds.append(kind)
        ann_aggregate[(idx, kind)] = g.value(ann, HW1.qualificationStatus)
        for f in selected:
            over = factors[f]["over"]
            value = g.value(ann, HW1[over])
            if value is not None:
                ann_values[(idx, f)] = (value, g.value(ann, HW1[factors[f]["status"]]))
        mean = g.value(ann, HW1.meanValue)
        if mean is not None:
            means[idx] = mean

    pairs = {}
    for pair in g.objects(exp, HW1.producesPair):
        i = frame_index_from_iri(g.value(pair, HW1.sourceFrame))
        j = frame_index_from_iri(g.value(pair, HW1.targetFrame))
        cells = {}
        for f in selected:
            if placement[f] != "pair":
                continue
            value = g.value(pair, HW1[factors[f]["over"]])
            cells[f] = (value, g.value(pair, HW1[factors[f]["status"]]))
        pairs[i] = (j, cells, g.value(pair, HW1.qualificationStatus))

    kinds = [k for k in _ANNOTATION_KINDS if k in kinds]
    return ann_values, ann_aggregate, means, kinds, pairs


def _print_frame_table(exp, g, factors, selected, placement, path):
    """§7.1's per-frame table. EVERY ROW, narrow columns — see the section header."""
    ann_values, ann_agg, means, kinds, pairs = _collect_frame_rows(
        g, exp["exp_iri"], factors, selected, placement, path)

    frame_factors = [f for f in selected if placement[f] == "frame"]
    pair_factors = [f for f in selected if placement[f] == "pair"]
    short = _abbrev([factors[f]["over"] for f in selected])

    print(f"{_INDENT}columns (value and status merged; P = hw1:Pass, F = hw1:Fail, "
          f"- = not measured on this node):")
    for f in selected:
        info = factors[f]
        op = "<=" if info["polarity"] == "lower" else ">="
        print(f"{_INDENT}  {short[info['over']]:<12} hw1:{info['over']} — {f}, "
              f"Pass iff value {op} {info['qualifiedBy']} "
              f"= {_fmt_setting(exp['settings'][info['qualifiedBy']])}")
    if means:
        # The statusless baseline of §4.3, printed next to the clip factors
        # precisely so "did a clip factor beat the mean?" is a lookup (§7.2).
        print(f"{_INDENT}  {'meanValue':<12} hw1:meanValue — the BASELINE: no factor, "
              f"no threshold, deliberately NO status")

    headers = ["frame"]
    right = [0]
    for f in frame_factors:
        headers.append(short[factors[f]["over"]])
    if means:
        headers.append("meanValue")
    for kind in kinds:
        headers.append(kind + "QS")
    if pairs:
        headers.append("->next")
        right.append(len(headers) - 1)
        for f in pair_factors:
            headers.append(short[factors[f]["over"]])
        headers.append("pairQS")

    def cell(entry):
        if entry is None:
            return "-"
        value, status = entry
        return f"{_fmt_num(value)} {_fmt_status(status, short=True)}"

    rows = []
    for idx in sorted(exp["frame_status"]):
        row = [idx]
        for f in frame_factors:
            row.append(cell(ann_values.get((idx, f))))
        if means:
            row.append(_fmt_num(means.get(idx)))
        for kind in kinds:
            row.append(_fmt_status(ann_agg.get((idx, kind)), short=True))
        if pairs:
            step = pairs.get(idx)
            if step is None:
                row.append("-")
                row.extend("-" for _ in pair_factors)
                row.append("-")
            else:
                j, cells, agg = step
                row.append(j)
                for f in pair_factors:
                    row.append(cell(cells.get(f)))
                row.append(_fmt_status(agg, short=True))
        rows.append(row)

    _print_table(headers, rows, right=tuple(right))
    _note(f"{len(rows)} frame(s), all shown — nothing elided. "
          f"`… | grep ' F'` finds every failing row.")


def _factor_counts(g, factors, selected):
    """Pass/total per selected factor, counted over the nodes that carry its value."""
    rows = []
    for f in sorted(selected):
        info = factors[f]
        total = passed = 0
        for node in g.subjects(HW1[info["over"]], None):
            total += 1
            if g.value(node, HW1[info["status"]]) == HW1.Pass:
                passed += 1
        rows.append((f, "hw1:" + info["over"], f"{passed}/{total}",
                     f"{total - passed}"))
    return rows


def _run_nodes(g, exp_iri):
    """Every hw1:ReconstructionRun of this experiment, in a stable mode order."""
    runs = []
    for run in g.objects(exp_iri, HW1.hasRun):
        mode = g.value(run, HW1.selectionMode)
        runs.append((_local(mode) if mode is not None else "?", run))
    runs.sort(key=lambda row: str(row[1]))
    return runs


def _run_factor_locals(g, factors, run):
    """The factors that actually landed on one run node — TBox-resolved, not listed."""
    return sorted(f for f, info in factors.items()
                  if g.value(run, HW1[info["over"]]) is not None)


def _print_summary(exp, g, factors, selected):
    """§7.1's summary: pass counts, usable links, segments, runs."""
    _print_table(("factor", "observable", "Pass", "Fail"),
                 _factor_counts(g, factors, selected), right=(2, 3))

    frames = exp["frame_status"]
    pair_status = exp["pair_status"]
    usable = exp["usable_links"]
    print()
    _kv("frames usable", f"{sum(1 for v in frames.values() if v)}/{len(frames)} "
                         f"(every annotation of the frame Pass; vacuously usable "
                         f"with no annotation)")
    _kv("pairs Pass", f"{sum(1 for v in pair_status.values() if v)}/"
                      f"{len(pair_status)}")
    _kv("usable links", f"{len(usable)}/{len(pair_status)} "
                        f"(pair Pass AND both endpoint frames usable — §4.5)")

    segments = cut_contiguous_segments(usable)
    kept = sum(len(s) for s in segments)
    _kv("segments", f"{len(segments)} maximal segment(s), "
                    f"{kept} frame(s) kept of {len(frames)}")
    if segments:
        shown = ", ".join(f"{s[0]}-{s[-1]}({len(s)})" for s in segments)
        print(f"{_INDENT}{'':<20}{shown}")

    _sub("runs")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        _note("no hw1:ReconstructionRun yet. Run `reconstruct.py --data_root "
              "<capture> --experiment <this file>`; runs are the only mutation an "
              "assessed file accepts (§3.1).")
        return
    value_locals = sorted({v for _, run in runs
                           for v in _run_factor_locals(g, factors, run)})
    metadata_locals = ["gatedSteps", "spliceCount", "maxGapLength"]
    shown_metadata = [p for p in metadata_locals
                      if any(g.value(run, HW1[p]) is not None for _, run in runs)]
    headers = (["run", "selectionMode", "frames"] +
               [factors[f]["over"] for f in value_locals] + shown_metadata)
    rows = []
    for mode_local, run in runs:
        count = g.value(run, HW1.runFrameCount)
        row = [str(run).rsplit("/", 1)[-1], mode_local,
               "-" if count is None else int(count)]
        for f in value_locals:
            info = factors[f]
            value = g.value(run, HW1[info["over"]])
            status = g.value(run, HW1[info["status"]])
            row.append("-" if value is None
                       else f"{_fmt_num(value)} {_fmt_status(status)}")
        for prop in shown_metadata:
            value = g.value(run, HW1[prop])
            row.append("-" if value is None else int(value))
        rows.append(row)
    _print_table(headers, rows, right=(2,))


def _scoped_subjects(g, prop, value, this_run):
    """Subjects carrying `prop` (optionally == `value`), scoped to ONE run.

    Run nodes OTHER than the run being attributed are skipped, so a failing
    baseline does not turn up inside the selected run's attribution and vice
    versa. The test is on rdf:type, not on the factor, so it stays generic — and
    it is applied to the failing count and the total alike, so the two numbers
    count the same population.
    """
    nodes = []
    for node in g.subjects(prop, value):
        if node != this_run and (node, RDF.type, HW1.ReconstructionRun) in g:
            continue
        nodes.append(node)
    return nodes


def _failing_nodes(g, factors, factor_local, this_run):
    """Every node whose `hw1:statusProperty` for this factor reads hw1:Fail.

    THE GENERIC RESOLUTION, and the whole reason the TBox keeps the
    `statusProperty` wiring after the SPARQL layer was deleted (§10): this is
    `?factor hw1:statusProperty ?sp . ?node ?sp hw1:Fail .` in Python, with no
    predicate enumerated and no factor named. A menu factor added to the TBox is
    picked up here with no edit.
    """
    return _scoped_subjects(g, HW1[factors[factor_local]["status"]], HW1.Fail,
                            this_run)


def _culprit_rows(failing, recorded, decls, where_of):
    """Recorded settings whose parameter touches a failing factor. TBox traversal only.

    `hw1:settingForFactor` on the setting node is NOT consulted — it is
    single-valued on an experiment setting (§4.5) and still multi-valued on a
    batch-side generation setting, so trusting it would give two different
    answers to one question. The blame set is
    `settingParameter -> paramPrimaryFactor / paramAffectsFactor`, read from the
    TBox, which is what reaches `brightnessGain` for a ShadowClipping failure
    whose primary factor is HighlightClipping.
    """
    rows = []
    for param in sorted(recorded):
        d = decls.get(param)
        if d is None:
            continue
        touched = sorted(set((d["primary"],) + tuple(d["affects"])) & set(failing))
        if not touched:
            continue
        rows.append((param, _fmt_setting(recorded[param]), d["role"],
                     where_of(param), ", ".join(touched),
                     _ROLE_FIX.get(d["role"], "(no fix declared for this role)")))
    return rows


def _print_verdict(exp, g, factors, decls, given, path):
    """§7.1's VERDICT SECTION — the old failure_attribution.rq, computed in code.

    For every run carrying a Fail: the factors that failed anywhere in this
    experiment with their failing-node counts, then every setting whose
    parameter touches one of them, with the ROLE as the verdict and the role's
    fix spelled out. Nothing here branches on a factor or a parameter name.
    """
    _rule("VERDICT — what failed, and what to fix")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        bad = sum(1 for f, info in factors.items()
                  if any(True for _ in g.subjects(HW1[info["status"]], HW1.Fail)))
        _note(f"No hw1:ReconstructionRun in this file, so there is nothing to "
              f"attribute yet: attribution starts from a FAILING RUN and walks back "
              f"to the input factors ({bad} factor(s) already carry at least one Fail "
              f"above). Run `reconstruct.py --experiment {path}` first.")
        return

    # Run modes form interventions. Say what the differential proves
    # before walking from input Fails to their setting roles; that older walk is
    # true about the pixels but cannot, by itself, explain a regression caused by
    # deleting temporal links.
    by_tail = {str(run).rsplit("/", 1)[-1]: run for _, run in runs}

    def run_verdict(run):
        statuses = [g.value(run, HW1[info["status"]]) for info in factors.values()
                    if g.value(run, HW1[info["over"]]) is not None]
        statuses = [s for s in statuses if s is not None]
        return None if not statuses else all(s == HW1.Pass for s in statuses)

    baseline = by_tail.get("baseline")
    selected_run = by_tail.get("selected")
    if baseline is not None and selected_run is not None:
        b_ok, s_ok = run_verdict(baseline), run_verdict(selected_run)
        splices = g.value(selected_run, HW1.spliceCount)
        gap = g.value(selected_run, HW1.maxGapLength)
        gated = g.value(selected_run, HW1.gatedSteps)
        mechanism = (f"{int(splices) if splices is not None else '?'} splice(s), "
                     f"max gap {int(gap) if gap is not None else '?'} frame(s), "
                     f"{int(gated) if gated is not None else '?'} gated step(s)")
        if b_ok is True and s_ok is False:
            _note("SELECTION EFFECT — the full-batch baseline passed and the "
                  f"selected probe failed ({mechanism}). Deletion/splicing is the "
                  "observed treatment difference, so deletion is not a repair. "
                  "The candidate table below scopes true INPUT Fails; it does not "
                  "explain this run regression. Act through the setting roles.")
        elif b_ok is False and s_ok is True:
            _note("SELECTION EFFECT — the full-batch baseline failed and the "
                  f"selected probe passed ({mechanism}). Under this experiment, "
                  "the rejected inputs are load-bearing for geometric ICP; use the "
                  "culprit setting roles below to design the next intervention.")
        elif b_ok is not None and s_ok is not None:
            b_l2 = g.value(baseline, HW1.mapMeanL2)
            s_l2 = g.value(selected_run, HW1.mapMeanL2)
            raw = ""
            if b_l2 is not None and s_l2 is not None:
                b_value, s_value = float(b_l2), float(s_l2)
                ratio = s_value / b_value if b_value else float("inf")
                raw = (f" Continuous mapMeanL2 still changed {b_value:.4f} -> "
                       f"{s_value:.4f} m ({ratio:.1f}x); the threshold is not the "
                       f"effect size.")
            _note("SELECTION EFFECT — baseline and selected have the same run verdict "
                  f"({mechanism}). Compare raw outcomes before claiming no effect."
                  + raw)

    attributed = 0
    for mode_local, run in runs:
        run_fails = [f for f, info in factors.items()
                     if g.value(run, HW1[info["status"]]) == HW1.Fail]
        if not run_fails:
            continue
        attributed += 1
        _sub(f"run {str(run).rsplit('/', 1)[-1]} ({mode_local}) FAILED")
        for f in sorted(run_fails):
            info = factors[f]
            print(f"{_INDENT}{f}: hw1:{info['over']} = "
                  f"{_fmt_num(g.value(run, HW1[info['over']]))} vs "
                  f"{info['qualifiedBy']} = "
                  f"{_fmt_setting(exp['settings'].get(info['qualifiedBy']))} -> Fail")

        # ── the failing factors, with their failing-node counts ───────────────
        failing, rows = [], []
        for f in sorted(factors):
            nodes = _failing_nodes(g, factors, f, run)
            if not nodes:
                continue
            failing.append(f)
            total = len(_scoped_subjects(g, HW1[factors[f]["over"]], None, run))
            rows.append((f, "hw1:" + factors[f]["status"], len(nodes),
                         total, "yes" if f in exp["selected"] else
                         ("run factor" if f not in _selectable_factors(factors)
                          else "no")))
        print()
        _note("factors carrying hw1:Fail (found through hw1:statusProperty — no "
              "predicate is enumerated, so a menu factor added to the TBox needs no "
              "edit here):")
        _print_table(("factor", "status property", "Fail", "of nodes", "selected"),
                     rows, right=(2, 3))

        # ── the culprit settings, role first ──────────────────────────────────
        recorded = dict(exp["settings"])
        culprits = _culprit_rows(
            failing, recorded, decls,
            lambda p: "declaration" if p in given else "defaulted")

        gen_rows, gen_note = [], None
        batch_file = g.value(exp["exp_iri"], HW1.batchFile)
        if batch_file is not None:
            resolved = _resolve_batch_file(str(batch_file))
            try:
                facts = _batch_facts(resolved)
                gen_rows = _culprit_rows(
                    failing, dict(facts["generation"]), decls,
                    lambda p: "batch (--gen)")
                if not facts["generation"]:
                    would = sorted(
                        n for n, d in decls.items()
                        if d["role"] == "GenerationSetting"
                        and set((d["primary"],) + tuple(d["affects"])) & set(failing))
                    if would:
                        gen_note = (
                            f"the batch records NO hw1:GenerationSetting, so the "
                            f"Generation verdict cannot fire here. "
                            f"{', '.join(would)} would be the culprit(s) — each "
                            f"touches a failing factor — but nothing recorded a "
                            f"level. NEVER INVENT ONE (§12 O-F): "
                            f"recover the --gen values the capture was made with, "
                            f"or regenerate it.")
            except (ValueError, OSError):
                gen_note = (f"hw1:batchFile {str(batch_file)!r} does not resolve "
                            f"({resolved!r}), so the batch's GenerationSettings could "
                            f"not be read from here.")

        print()
        _note("candidate settings — every recorded setting whose PARAMETER's "
              "hw1:paramPrimaryFactor or hw1:paramAffectsFactor is one of those "
              "factors (TBox traversal, §4.5). This graph walk establishes scope, "
              "not causal relevance to the run; use a matched-setting dataset "
              "contrast or a one-setting experiment to establish that:")
        _print_table(
            ("parameter", "value", "role", "recorded on", "failing factor(s)", "FIX"),
            culprits + gen_rows, right=(1,))
        if gen_note:
            _note(f"note: {gen_note}")
        _note("The ROLE is the verdict. Measurement and Qualification fixes both mean "
              "a NEW declaration under a NEW name: experiments are write-once (§3.1).")

    if attributed == 0:
        _note("Every run in this file passed every run-level factor, so there is "
              "nothing to attribute. (Input factors may still carry Fails — see the "
              "summary above; a selection that drops frames is a design choice, not a "
              "failure.)")


def _view_experiment(path):
    """View 3: an ASSESSED experiment — settings, per-frame table, summary, verdict."""
    # `read_experiment` verifies the §3.1 seal FIRST. A file edited above the
    # marker raises here, which is the entire point: printing a table of numbers
    # computed for a treatment the file no longer declares would be the most
    # misleading artefact this program could produce.
    exp = read_experiment(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    g = exp["graph"]
    selected = exp["selected"]

    _rule(f"EXPERIMENT  {exp['exp_name']}  (assessed)")
    _kv("file", path)
    _kv("experiment IRI", exp["exp_iri"])
    _kv("batch", exp["batch_name"])
    _kv("evaluates", f"{len(selected)} factor(s): {', '.join(selected)}")
    label = g.value(exp["exp_iri"], RDFS.label)
    if label is not None:
        _kv("label", str(label))
    _kv("seal", f"hw1:declarationDigest verified "
                f"({str(g.value(exp['exp_iri'], HW1.declarationDigest))[:12]}…)")

    _sub("settings — given vs defaulted, read from WHICH SIDE OF THE MARKER (§4.2)")
    given, defaulted = _section_settings(path, exp["exp_iri"])
    _print_table(
        ("parameter", "value", "role", "primary factor", "source"),
        _settings_rows(exp["settings"], exp["settings"], decls,
                       lambda p: ("declaration (above marker)" if p in given else
                                  "defaulted (below marker)" if p in defaulted else
                                  "(not in either section)")),
        right=(1,))
    print(f"{_INDENT}{len(given)} declared, {len(defaulted)} defaulted.")

    _sub("selection")
    _print_table(("factor", "observable", "polarity", "qualifiedBy", "threshold"),
                 _selection_rows(selected, factors, exp["settings"]))

    _sub("per frame")
    placement = _observable_placement(g, exp["exp_iri"], factors, selected)
    _print_frame_table(exp, g, factors, selected, placement, path)

    _sub("summary")
    _print_summary(exp, g, factors, selected)

    _sub("filter masks")
    if not exp["mask_files"]:
        _note("no exported factor masks in this experiment")
    else:
        rows = []
        for factor, scopes in sorted(exp["mask_files"].items()):
            sample = next(iter(scopes["frames"].values()), None)
            if sample is None:
                sample = next(iter(scopes["pairs"].values()), "-")
            rows.append((factor, len(scopes["frames"]), len(scopes["pairs"]), sample))
        _print_table(("factor", "frame masks", "pair masks", "example maskFile"),
                     rows, right=(1, 2))

    _print_verdict(exp, g, factors, decls, given, path)
    return 0


# ── the comparison view (§7.1 view 4) ───────────────────────────
def _compare_column(path):
    """One column of the comparison table, from whichever reader the file supports."""
    kind = _classify(path)
    if kind == "batch":
        # Named by hw1:batchName, not by the file name: every batch file in this
        # project is called `batch.ttl` (§3), so a basename would label two
        # columns identically.
        return {"path": path, "kind": "batch", "name": str(_batch_facts(path)["name"]),
                "why": "a batch file carries no selection, no settings and no runs "
                       "(§4.1), so there is nothing to compare it on"}
    if kind == "declaration":
        decl = read_declaration(path)
        return {"path": path, "kind": "declaration", "name": decl["exp_name"],
                "batch": decl["batch_name"], "selected": decl["selected"],
                "settings": decl["required"], "given": decl["given"],
                "runs": {}, "counts": None,
                "why": "declared but NOT ASSESSED — its settings are the required "
                       "set §4.2 would record, and it has no values, verdicts or "
                       "runs to compare"}
    exp = read_experiment(path)
    g = exp["graph"]
    factors = load_quality_factors(_ONTOLOGY_TTL)
    runs = {}
    for _mode_local, run in _run_nodes(g, exp["exp_iri"]):
        # Keyed by the run IRI's `<mode>` segment (§2) — "baseline" / "selected",
        # the words the CLI and the student use — not by the hw1:SelectionMode
        # individual, which is the same fact spelled for the ontology.
        mode = str(run).rsplit("/", 1)[-1]
        for f in _run_factor_locals(g, factors, run):
            info = factors[f]
            runs[(mode, info["over"])] = (
                f"{_fmt_num(g.value(run, HW1[info['over']]))} "
                f"{_fmt_status(g.value(run, HW1[info['status']]))}")
    given, _defaulted = _section_settings(path, exp["exp_iri"])
    return {"path": path, "kind": "experiment", "name": exp["exp_name"],
            "batch": exp["batch_name"], "selected": exp["selected"],
            "settings": exp["settings"], "given": given, "runs": runs,
            "factor_counts": {
                factor: (
                    sum(1 for node in g.subjects(HW1[info["over"]], None)
                        if g.value(node, HW1[info["status"]]) == HW1.Pass),
                    sum(1 for _ in g.subjects(HW1[info["over"]], None)))
                for factor, info in factors.items() if factor in exp["selected"]},
            "counts": (sum(1 for v in exp["frame_status"].values() if v),
                       len(exp["frame_status"]), len(exp["usable_links"]),
                       len(exp["pair_status"])),
            "why": None}


def _view_compare(paths):
    """View 4: several files side by side — selection, settings and runs."""
    columns = [_compare_column(p) for p in paths]
    usable = [c for c in columns if c["kind"] != "batch"]
    _rule(f"COMPARE  {len(columns)} file(s)")
    for c in columns:
        _kv(c["name"], f"{c['path']}  [{c['kind']}]"
                       + (f" — {c['why']}" if c["why"] else ""))
    if not usable:
        print(f"{_INDENT}Nothing comparable was given.")
        return 0

    names = [c["name"] for c in usable]
    _sub("identity")
    _print_table(["", *names],
                 [["batch", *[c["batch"] for c in usable]],
                  ["assessed", *["yes" if c["kind"] == "experiment" else "no"
                                 for c in usable]],
                  ["frames usable", *[f"{c['counts'][0]}/{c['counts'][1]}"
                                      if c["counts"] else "-" for c in usable]],
                  ["usable links", *[f"{c['counts'][2]}/{c['counts'][3]}"
                                     if c["counts"] else "-" for c in usable]]])

    _sub("selection (hw1:evaluatesFactor)")
    all_factors = sorted({f for c in usable for f in c["selected"]})
    rows = []
    for f in all_factors:
        marks = ["yes" if f in c["selected"] else "-" for c in usable]
        rows.append([("*" if len(set(marks)) > 1 else " ") + " " + f, *marks])
    _print_table(["  factor", *names], rows)

    _sub("input factor outcomes")
    rows = []
    for factor in all_factors:
        vals = []
        for column in usable:
            counts = column.get("factor_counts", {}).get(factor)
            vals.append("-" if counts is None else f"{counts[0]}/{counts[1]} Pass")
        rows.append([factor, *vals])
    _print_table(["  factor", *names], rows,
                 right=tuple(range(1, len(names) + 1)))

    _sub("settings")
    all_params = sorted({p for c in usable for p in c["settings"]})
    rows = []
    for p in all_params:
        vals = [_fmt_setting(c["settings"][p]) if p in c["settings"] else "-"
                for c in usable]
        annotated = [v + ("" if c["kind"] != "experiment" else
                          (" (d)" if p in c["given"] else ""))
                     for v, c in zip(vals, usable)]
        rows.append([("*" if len(set(vals)) > 1 else " ") + " " + p, *annotated])
    _print_table(["  parameter", *names], rows, right=tuple(range(1, len(names) + 1)))
    _note("* marks a row where the columns disagree — that is the treatment "
          "difference. (d) marks a level the DECLARATION set; everything else was "
          "filled from hw1:paramDefault (§4.2).")

    _sub("runs")
    all_runs = sorted({k for c in usable for k in c["runs"]})
    if not all_runs:
        print(f"{_INDENT}no hw1:ReconstructionRun in any of these files.")
    else:
        rows = []
        for mode_local, over in all_runs:
            vals = [c["runs"].get((mode_local, over), "-") for c in usable]
            rows.append([f"{mode_local} / hw1:{over}", *vals])
        _print_table(["  run / observable", *names], rows,
                     right=tuple(range(1, len(names) + 1)))
    return 0


def cmd_explore(args):
    """Print what a .ttl file says. READ-ONLY: measures nothing, writes nothing.

    Four views, dispatched on WHAT THE ARGUMENT IS rather than on what it is
    called (§7.1, `_classify`):

      1. a batch file            -> header + frame table + stem gaps
      2. an unassessed declaration -> batch header, selection, required settings
      3. an assessed experiment  -> settings, per-frame table, summary, VERDICT
      4. several paths           -> the comparison table

    It re-implements no rule. The status rule is `status_for`'s and was baked at
    measure time; the completeness rule is `read_declaration`'s; the frame
    verdict, the usable-link rule and the seal check are `read_experiment`'s; the
    segment cut is `cut_contiguous_segments`'. What is computed HERE is only the
    projection: which columns, which counts, and the attribution walk of §7.1 —
    and that walk is generic over the TBox (`statusProperty` for failing factors,
    `paramPrimaryFactor` / `paramAffectsFactor` for culprit settings), so adding a
    menu factor requires no edit in this section.
    """
    paths = list(args.paths)
    missing = [p for p in paths if not (os.path.isfile(p) or _is_capture_dir(p))]
    if missing:
        raise ValueError(
            f"no such capture directory or file: {', '.join(repr(p) for p in missing)}. "
            f"`explore` reads a capture directory, a declaration or an assessed "
            f"experiment — and several experiment files at once print the "
            f"comparison view (§7.1).")
    if len(paths) > 1:
        return _view_compare(paths)
    kind = _classify(paths[0])
    if kind == "batch":
        return _view_batch(paths[0])
    if kind == "declaration":
        return _view_declaration(paths[0])
    return _view_experiment(paths[0])


# =============================================================================
# Frame selection — the segment cutter
#   In v1 this was the body of a `select` subcommand that minted a derived
#   experiment. Both are deleted (§11): selection is not an artefact,
#   it is a step inside `reconstruct.py`, which calls this function on the
#   `usable_links` that `read_experiment` derived and then records the outcome as
#   `hw1:usedFrame` on the selected run. So this is a pure list-to-lists function
#   with no RDF in it at all — the grading happened upstream, in `status_for`.
# =============================================================================
def cut_contiguous_segments(usable_links):
    """Usable links -> the maximal runs of frames they chain. §8.3.

    CONTRACT
        In:     usable_links — an iterable of `(i, j)` integer frame-index pairs,
                each one a USABLE LINK in the sense of §4.5: the pair passed and
                both of its endpoint frames passed. `read_experiment` derives that
                list; this function re-derives nothing and grades nothing.
        Out:    [[i, j, k, …], …] — disjoint segments, each an ascending list of
                frame indices, in ascending order. Every consecutive index pair
                inside a segment is a link that was in `usable_links`.

        A run of `k` chained links yields `k + 1` FRAMES: the links are the gaps
        between frames, so two links (7,8),(8,9) are the three frames 7,8,9. Getting
        that off by one silently shifts every length verdict by one frame.

        EVERY maximal chain is returned, whatever its length. The `min_length`
        argument and its `minSegmentLength` parameter were deleted on 2026-07-31:
        the floor's justification — below ~20 frames the constant-velocity prior
        has nothing to average — is a fact about the ICP backend, not about the
        quality of the pixels, and this layer grades pixels. Measured on floor 1
        it never earned its place either: the selected run loses to the full batch
        at every floor tried, because the cost is the SEAMS between segments and a
        length floor does not touch a seam.

        A missing link is a CUT — and so is a BREAK IN THE CHAIN: if one link's `j`
        is not the next link's `i` the sequence has a hole (a pair that failed, or
        one that was never measured at all), and gluing across it would invent a
        transition nobody graded. Same failure, same treatment. Links are sorted and
        de-duplicated first, so the caller's iteration order cannot fabricate a hole
        that is not in the data.

    WHY THIS CUTS SEGMENTS INSTEAD OF DROPPING BAD FRAMES
        This is the reason selection has this shape at all, and the obvious
        alternative — grade the frames, drop the bad ones, reconstruct what is left
        — is wrong in the worst available way: it produces a plausible number
        rather than a crash.

        Drop frame k from the middle of a run and you have not removed a frame. You
        have SILENTLY CREATED a pair (k-1, k+1) that nobody measured. Two things
        break at once:

          1. THE STORED OBSERVABLES START LYING. `<pair/k-1_k>` and `<pair/k_k+1>`
             describe transitions that no longer occur, and the transition that now
             DOES occur has no node, no `medianDepthDifference`, and no grade. Every
             quality claim made about the run afterwards is a claim about pairs that
             were not reconstructed.
          2. THE MOTION ROUGHLY DOUBLES across the splice. `reconstruct.py`
             initialises each registration from the previous transform
             (constant velocity) and gates correspondences at a distance sized for
             consecutive frames; a spliced pair violates both assumptions exactly
             where the data was already worst. Removing the bad frame can therefore
             cost more accuracy than keeping it would have — a genuinely
             counter-intuitive result, and one students reproduce by accident if
             this function is written the obvious way.

        A cut, by contrast, only ever happens where the sequence was going to be
        interrupted anyway. No spliced pair is ever created, because the frames on
        either side of a cut end up in different segments, and each segment is
        reconstructed independently from its own first-frame anchor.
    """
    links = sorted(set((int(i), int(j)) for i, j in usable_links))
    segments = []
    current = []
    prev_j = None
    for i, j in links:
        if current and prev_j == i:
            current.append(j)                 # the chain continues: one more frame
        else:
            if current:
                segments.append(current)      # hole (or first link): start a segment
            current = [i, j]                  # ONE link is already TWO frames
        prev_j = j
    if current:
        segments.append(current)
    return segments


# =============================================================================
# CLI
# =============================================================================
def _build_parser():
    p = argparse.ArgumentParser(
        description="HW1 data-quality CLI (rdflib only). Scaffold a DECLARATION "
                    "over a capture directory (declare), assess it (experiment), "
                    "and read the result back as terminal tables (explore). "
                    "batch2ttl is a deprecated optional sidecar for generation "
                    "provenance. reconstruct.py completes the suite. No server, "
                    "no daemon, no SPARQL: capture directories and .ttl files.",
        epilog="A measured value means nothing without the settings it was "
               "measured and judged under, so the settings live in the same file "
               "as the values — the student's own declaration, which is "
               "WRITE-ONCE.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b2t = sub.add_parser(
        "batch2ttl",
        help="DEPRECATED. Optional sidecar: write generation provenance to "
             "<data_dir>/batch.ttl. declare/experiment/explore take the capture "
             "directory directly.")
    b2t.add_argument("--data-dir", required=True,
                     help="Directory containing rgb/ and depth/ subdirs of integer-stem .png "
                          "frames. A semantic/ subdir, if present, is tolerated but not "
                          "ingested. batch.ttl is written HERE as an optional sidecar "
                          "for --gen / --derived-from. declare and experiment do not "
                          "need this file.")
    b2t.add_argument("--floor", type=int, default=1,
                     help="Floor this capture is from (default 1). Stored on the Batch node AND "
                          "prefixed onto the batch name, so floor1_mixed_dev and floor2_mixed_dev "
                          "are distinct batches rather than one batch that overwrites itself.")
    b2t.add_argument("--gen", action="append", default=[], metavar="NAME=VALUE",
                     help="One GenerationSetting, repeatable: the corruption ALREADY BAKED "
                          "into these pixels, e.g. --gen brightnessGain=0.45. NAME must be "
                          "declared in ontology/hw1.ttl as a hw1:Parameter whose "
                          "hw1:paramRole is hw1:GenerationSetting; a Measurement or "
                          "Qualification name here is an error (those belong in the "
                          "declaration, as hw1:FactorSetting nodes), and an undeclared name is "
                          "an error with the declared list printed. Applied by NOTHING — "
                          "the capture already embodies it — and never filled from a "
                          "default, because a default would assert something false about "
                          "pixels this code never opened. Absence means NOT ASSERTED.")
    b2t.add_argument("--derived-from", default=None, metavar="BATCHNAME",
                     help="The hw1:batchName of the capture this one was derived from, e.g. "
                          "floor1_baseline for a corrupted copy of it; emits one "
                          "prov:wasDerivedFrom. The only surviving prov: term in the "
                          "project (§6) and the only one that spans two "
                          "batches — there are no derived EXPERIMENTS.")
    b2t.add_argument("--out", default=None,
                     help="Output Turtle path (default <data_dir>/batch.ttl).")
    b2t.set_defaults(func=cmd_batch2ttl)

    dec = sub.add_parser(
        "declare",
        help="Scaffold a declaration Turtle (prefixes, Experiment node, batch join, "
             "factor selection, PREDICTION TODOs) so nobody starts from a blank "
             "page. Writes the STUDENT section only; never assesses, never "
             "overwrites.")
    dec.add_argument("--name", required=True,
                     help="Experiment name: the file stem AND the tail of the "
                          "Experiment IRI, which must stay equal (§2). "
                          "[A-Za-z0-9_-]+ — naming your experimental conditions is "
                          "part of designing them.")
    dec.add_argument("--data-dir", default=None,
                     help="Capture directory containing rgb/ and depth/ subdirs of "
                          "integer-stem .png frames. Written into hw1:batchFile "
                          "verbatim and resolved against the CWD; hw1:onBatch is "
                          "derived from --floor + the directory basename.")
    dec.add_argument("--floor", type=int, default=1,
                     help="Floor this capture is from (default 1). Prefixed onto the "
                          "batch name, so floor1_mixed_dev and floor2_mixed_dev are "
                          "distinct batches.")
    dec.add_argument("--batch-file", default=None,
                     help="DEPRECATED. Path to a capture directory or a legacy "
                          "batch.ttl. Prefer --data-dir; the scaffold writes the "
                          "capture directory into hw1:batchFile either way.")
    dec.add_argument("--factor", action="append", default=[], metavar="FACTOR",
                     help="One menu factor to select, repeatable (with or without the "
                          "hw1: prefix). Omitted entirely: the scaffold selects the "
                          "FULL menu and tells you to trim it — the selection is part "
                          "of the design, so the default is deliberately everything "
                          "rather than a guess.")
    dec.add_argument("--out", default=None,
                     help=f"Output path (default {_EXPERIMENT_DIR}/<name>.ttl). The "
                          f"file stem must equal --name. An existing file is a hard "
                          f"error: the notebook is append-only.")
    dec.set_defaults(func=cmd_declare)

    exp = sub.add_parser(
        "experiment",
        help="Assess ONE student-authored declaration: measure what it selects and "
             "append the machine section to that same file. Once, ever.")
    exp.add_argument("declaration",
                     help=f"Path to the declaration Turtle you wrote (convention: "
                          f"{_EXPERIMENT_DIR}/<expname>.ttl, and the file STEM must equal "
                          f"the tail of the hw1:Experiment IRI inside it). It states the "
                          f"batch (hw1:batchFile = capture directory, hw1:onBatch), "
                          f"the factor selection "
                          f"(hw1:evaluatesFactor, 1..8 from the menu) and any threshold or "
                          f"measurement override (hw1:hasFactorSetting); every value, "
                          f"status and run is COMPUTED and appended below the marker. THE "
                          f"ONLY ARGUMENT: --batch-dir, --floor, --set, --exp-id, --label, "
                          f"--no-pairs and --out are deleted (§7) — the "
                          f"design lives in the RDF now. Running this on an "
                          f"already-assessed file is a HARD ERROR: experiments are "
                          f"write-once, and every tuning is a new declaration under a new "
                          f"name (§3.1).")
    exp.set_defaults(func=cmd_experiment)

    exl = sub.add_parser(
        "explore",
        help="Read-only terminal tables: a batch, a declaration, an assessed "
             "experiment (values, verdicts, runs, attribution) or a comparison.")
    exl.add_argument("paths", nargs="+", metavar="PATH",
                     help="One capture directory, one declaration, one assessed "
                          "experiment — or SEVERAL experiment files, which prints "
                          "the comparison view (§7.1). Writes nothing "
                          "and measures nothing.")
    exl.set_defaults(func=cmd_explore)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
