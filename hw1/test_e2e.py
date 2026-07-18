"""
The API contract for hw1/api.py — and an autograded deliverable.

This file is the SPECIFICATION of what the measurers, `declare`, `experiment`
and `explore` must do. It is not a suggestion and it is not a sample: your
`api.py` and your `ontology/hw1.ttl` are correct when this passes. `batch2ttl`
remains as an optional generation-provenance sidecar.

Every assertion below is written against the v3 design, and every test
docstring names the clause it pins. If a test and the code disagree, read the
§ reference in the docstring before touching either.

WHAT V3 CHANGED, AND WHAT THIS FILE THEREFORE TESTS
    An experiment is a STUDENT-AUTHORED DECLARATION now: a Turtle file naming a
    batch, a selection of quality factors and any threshold overrides. `api.py
    experiment DECLARATION.ttl` validates it, measures exactly what was declared,
    and appends a machine section below a frozen marker line — ONCE, ever. So the
    weight of this suite moved:

      * every §4.2 declaration rule gets its own red-path test, and the error has
        to NAME the offending triple (`DeclarationValidation`);
      * the file is write-once and SEALED — a second `experiment` is a hard error
        and an edit above the marker is detected by `hw1:declarationDigest`
        (`MachineSection`);
      * the student's BYTES survive verbatim, comments and spelling included,
        through `experiment` and through `write_run` (`MachineSection`);
      * annotations are PER MODALITY and totality is PER SELECTION — a depth-only
        experiment records no `tauHi` and mints no rgb annotation
        (`SelectionScoping`, `ModalityAnnotations`);
      * pairs are ALWAYS minted, and both vacuous-pass readings of the usable-link
        rule are pinned (`PairMinting`, `VacuousPassLinks`);
      * the SPARQL layer is gone; attribution is `explore`'s verdict section,
        checked on a fixture engineered to fail a named factor (`ExploreVerdict`).

WHAT SURVIVED UNCHANGED
    1. THE FRAME MEASURERS, against fixtures whose values are CLOSED FORM — a
       synthetic frame has an exact clip fraction, an exact mean and an exact
       valid-depth fraction, so these assertions hold for any correct
       implementation. `depth_roughness` is checked differently and more
       strongly: it is handed a frame carrying Gaussian noise of a KNOWN sigma
       under a fixed seed, and it must RECOVER that sigma. An estimator validated
       by recovering the noise it was given cannot be satisfied by a constant that
       happens to fit.  (contracts §9 — frozen verbatim from v1.)
    2. THE PAIR MEASURERS, the same way: two synthetic depth rasters whose median
       difference and whose edge-mask IoU are known in closed form, plus the two
       fail-closed edge cases (`inf` for the lower-is-better one, `0.0` for the
       higher-is-better one) and the Sobel normalisation that makes
       `sobelThreshold` read in metres of depth step per pixel.  (contracts §9.)
    3. THE STATUS RULE of §4.5, implemented once in `api.status_for` (§8), the
       parameter layer of §5, the segment cutter of §8, and `batch2ttl`, which
       still writes STRUCTURE ONLY.
    4. THE RUN WRITE-BACK of §8.2. `write_run` is still the only writer of run
       triples and still idempotent PER KEY: the single most important test in
       this file is that writing `coverageF` afterwards does not erase
       `mapMeanL2`, because `completeness.py` does exactly that and a
       delete-everything implementation destroys the deliverable silently.

NO DIGEST IDENTITY, NO `query`, NO THRESHOLDS FILE
    `experiment_digest`, `hw1:experimentId`, `--exp-id` and the 8-hex file names
    are deleted (contracts §6/§11): an experiment is identified by its NAME, which
    is the declaration file's stem and the IRI tail, and the two must agree. The
    `query` command, `hw1/queries/*.rq` and the pyoxigraph dependency are deleted
    too (§10) — `explore` prints the projections those queries used to compute.

    Qualification thresholds are still never read from the TBox by a test that
    predicts a status: contracts §5 marks all eight PROVISIONAL until the OFAT
    re-cut lands, so every fixture whose verdicts this file predicts DECLARES them
    explicitly from `PINNED_THRESHOLDS` below, and the expectations are arithmetic
    against THOSE. The TBox defaults are checked for existence, kind and role —
    never for value.

FOUR FIXTURES ARE LOAD-BEARING
    These are not extra coverage — each is the executable argument for a design
    decision, and each is the case that a plausible-looking wrong implementation
    fails. Read them before "simplifying" anything:

      test_half_black_half_white_*      why mean(V) is not a quality factor
      test_solid_pure_red_*             why no colorimetric weighting is allowed
      test_the_student_bytes_survive_*  contracts §3.1 — the declaration is the
                                        student's artefact; a machine pass that
                                        re-serialized it would silently rewrite
                                        their comments, their spelling and their
                                        seal
      test_writing_coverage_f_does_not_erase_map_mean_l2
                                        contracts §8.2 — the `completeness.py`
                                        case; a per-run (rather than per-key)
                                        delete destroys the baseline-vs-selected
                                        comparison the assignment is graded on

RUN
    pixi run -e habitat python -m pytest hw1/test_e2e.py -v      # from repo root
    pixi run -e habitat python hw1/test_e2e.py                   # direct

    HW1_API=instructor/solution/api.py pixi run -e habitat python -m pytest hw1/test_e2e.py -v
        # instructor: run the same contract against the reference solution

    Nothing in this module touches `api` at import time. A function contracts §8
    freezes but that has not landed yet surfaces as ONE red test naming it, never
    as a collection error that hides the other thousand assertions.
"""
import contextlib
import hashlib
import importlib.util
import io
import math
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from rdflib import BNode, Graph, Literal, RDF, RDFS, URIRef, XSD

_HERE = os.path.dirname(os.path.abspath(__file__))

# The module under test defaults to hw1/api.py; HW1_API points it elsewhere so the
# same contract can be run against a reference implementation.
_API_PATH = os.environ.get("HW1_API") or os.path.join(_HERE, "api.py")
if not os.path.isabs(_API_PATH):
    _API_PATH = os.path.join(os.path.dirname(_HERE), _API_PATH)
_spec = importlib.util.spec_from_file_location("hw1_api_under_test", _API_PATH)
api = importlib.util.module_from_spec(_spec)
sys.modules["hw1_api_under_test"] = api
_spec.loader.exec_module(api)

# The TBox. Read directly (never written) by the declaration and invariant tests.
_ONTOLOGY_TTL = getattr(api, "_ONTOLOGY_TTL",
                        os.path.join(_HERE, "ontology", "hw1.ttl"))


def _require(name):
    """Fetch a name contracts §8 freezes on `api`, or fail THIS test with the reason.

    Called from inside a test or a setUpClass, never at import time. A track that
    has not landed yet must cost one red test that names the missing function and
    the clause that froze it — not a module-level AttributeError that stops
    collection and takes every unrelated assertion in this file down with it.
    """
    try:
        return getattr(api, name)
    except AttributeError:
        raise AssertionError(
            f"api.{name} does not exist. §8 freezes it as part of the "
            f"public surface that other modules import by name.") from None


# Module-level temp dir populated by setUpModule / torn down by tearDownModule.
_TMP = None

FLOOR = 1
BATCH_DIRNAME = "e2e_batch"
BATCH = f"floor{FLOOR}_{BATCH_DIRNAME}"          # what hw1:batchName must hold
BATCH_DEGEN_DIRNAME = "e2e_batch_degenerate"
BATCH_DEGEN = f"floor{FLOOR}_{BATCH_DEGEN_DIRNAME}"
BATCH_GAP_DIRNAME = "e2e_batch_gap"
BATCH_GAP = f"floor{FLOOR}_{BATCH_GAP_DIRNAME}"
BATCH_SEL_DIRNAME = "e2e_batch_selection"
BATCH_SEL = f"floor{FLOOR}_{BATCH_SEL_DIRNAME}"
BATCH_TEENS_DIRNAME = "e2e_batch_teens"
BATCH_TEENS = f"floor{FLOOR}_{BATCH_TEENS_DIRNAME}"
# A copy of the selection batch whose batch.ttl records a GenerationSetting, so
# `explore`'s verdict section has a Generation culprit to reach (contracts §7.1).
BATCH_VERDICT_DIRNAME = "e2e_batch_verdict"
BATCH_VERDICT = f"floor{FLOOR}_{BATCH_VERDICT_DIRNAME}"

# A second batch NAME, never materialised on disk — the IRI minters are pure
# functions of a name, so a string is all the scheme tests need.
BATCH_B = f"floor{FLOOR}_e2e_batch_b"

# ---------------------------------------------------------------------------
# The tau this suite measures at. Passed explicitly everywhere; never defaulted.
# They are plain constants, not a file: tau is a MeasurementSetting of an
# experiment now, and the measurer contract — a threshold argument with no
# default — is unchanged by that move, which is exactly why every numeric
# expectation below survived it.
# ---------------------------------------------------------------------------
TAU_LO = 16.0
TAU_HI = 250.0

# ---------------------------------------------------------------------------
# Synthetic frames and their CLOSED-FORM observables.
#
# RGB frames are solid or half/half, so the clip fractions and the mean of
# V = max(R,G,B) are exact. Depth frames are constant (exactly one depth, so
# valid_depth_fraction is exactly 1.0 and depth_roughness is exactly 0), constant
# over a strip with the rest dropped out (an exact fraction of no-return pixels),
# or constant plus Gaussian noise at a known sigma (so depth_roughness is that
# sigma).
#
#   stem  rgb          depth                              fails on
#   0     grey 128     clean 3 m                          -- nothing
#   1     black 0      clean 3 m                          clipLoFraction = 1.0
#   2     white 255    clean 3 m                          clipHiFraction = 1.0
#   3     grey 128     3 m over 13 of 64 rows, rest 0     validDepthFraction
#   4     grey 128     3 m + noise, sigma = 0.15 m        depthRoughness
#
# Each factor is failed by exactly ONE frame, and each failing frame clears the
# other three — frame 3 in particular is clean where it has depth at all, so its
# roughness is 0 and only its validity is bad. Frame 0 is the only frame that
# clears everything. That table is what makes the per-modality
# `hw1:qualificationStatus` aggregate (contracts §4.5) testable in both
# directions, on one batch, one modality at a time.
# ---------------------------------------------------------------------------
IMG = 64
_FRAMES = [
    (0, "grey", "clean"),
    (1, "black", "clean"),
    (2, "white", "clean"),
    (3, "grey", "sparse"),
    (4, "grey", "noisy"),
]
MAIN_STEMS = [stem for stem, _, _ in _FRAMES]
MAIN_PAIRS = [(MAIN_STEMS[k], MAIN_STEMS[k + 1]) for k in range(len(MAIN_STEMS) - 1)]

# Depth of the synthetic frames, and the noise the "noisy" frame carries.
DEPTH_Z_M = 3.0
# 0.15 m is three times the 0.05 m roughness threshold this suite pins, so frame 4
# fails DepthRoughness by a wide margin instead of landing on the boundary — a
# fixture that sat ON the cut would flip with the estimator's sampling error.
BATCH_NOISE_SIGMA_M = 0.15
SPARSE_ROWS = 13                                 # of IMG: vdf = 13/64 = 0.203125

# Fail-closed sentinel, named so an assertion reads as "the gate rejects it".
INF = float("inf")

# ---------------------------------------------------------------------------
# The declared v4 parameters of §5/§13, as a table this file owns
# independently of the TBox. Reading the expectation out of ontology/hw1.ttl would
# make the test a tautology; writing it here means a parameter that changes role,
# kind, primary factor or affects-set in the TBox has to be changed HERE too,
# deliberately, with the contract in hand.
#
#   name -> (role, value kind, primary factor, frozenset(also affects), default)
#
# `default` is `None` for the four Generation parameters — absence is the assertion
# (contracts §5: a default there would assert something false about pixels the
# code never touched). For the eight QUALIFICATION parameters the default is
# written as `PROVISIONAL`: contracts §5 marks every one of those numbers as
# subject to the OFAT re-cut, so this file pins that a default EXISTS and is a
# usable finite number, and refuses to pin which number it is. Measurement
# defaults are constants of the pipeline (the makehdr tau pair, the sensor range,
# the Sobel units) and are pinned exactly.
# ---------------------------------------------------------------------------
PROVISIONAL = object()

DECLARED_PARAMS = {
    "tauHi":            ("MeasurementSetting", "double", "HighlightClipping",
                         frozenset(), 250.0),
    "tauLo":            ("MeasurementSetting", "double", "ShadowClipping",
                         frozenset(), 5.0),
    "residualMaskK":    ("MeasurementSetting", "double",
                         "HighFrequencyDepthResidual", frozenset(), 5.0),
    "flyingPixelWindow": ("MeasurementSetting", "integer", "FlyingPixelRatio",
                           frozenset(), 5),
    "flyingPixelPlanarityTol": ("MeasurementSetting", "double",
                                 "FlyingPixelRatio", frozenset(), 0.03),
    "tileSize":         ("MeasurementSetting", "integer", "ValidTileCoverage",
                         frozenset(), 64),
    "tileValidFloor":   ("MeasurementSetting", "double", "ValidTileCoverage",
                         frozenset(), 0.5),
    "changeMaskK":      ("MeasurementSetting", "double",
                         "IdentityMedianDepthChange", frozenset(), 3.0),
    "priorWarpDepthGate": ("MeasurementSetting", "double",
                           "PriorWarpDepthResidual", frozenset(), 0.10),
    "icpBackend":       ("MeasurementSetting", "string", "ReconstructionAccuracy",
                         frozenset(), "open3d"),
    "maxClipHiFraction": ("QualificationSetting", "double", "HighlightClipping",
                          frozenset(), PROVISIONAL),
    "maxClipLoFraction": ("QualificationSetting", "double", "ShadowClipping",
                          frozenset(), PROVISIONAL),
    "maxHighFrequencyDepthResidual": ("QualificationSetting", "double",
                                       "HighFrequencyDepthResidual", frozenset(),
                                       PROVISIONAL),
    "maxFlyingPixelRatio": ("QualificationSetting", "double", "FlyingPixelRatio",
                             frozenset(), PROVISIONAL),
    "minValidTileCoverage": ("QualificationSetting", "double", "ValidTileCoverage",
                              frozenset(), PROVISIONAL),
    "maxIdentityMedianDepthChange": ("QualificationSetting", "double",
                                      "IdentityMedianDepthChange", frozenset(),
                                      PROVISIONAL),
    "minJointValidDepthRatio": ("QualificationSetting", "double",
                                 "JointValidDepthRatio", frozenset(), PROVISIONAL),
    "maxPriorWarpDepthResidual": ("QualificationSetting", "double",
                                   "PriorWarpDepthResidual", frozenset(), PROVISIONAL),
    "maxMapMeanL2":     ("QualificationSetting", "double", "ReconstructionAccuracy",
                         frozenset(), PROVISIONAL),
    "minCoverageF":     ("QualificationSetting", "double", "Coverage",
                         frozenset(), PROVISIONAL),
    "brightnessGain":   ("GenerationSetting", "double", "HighlightClipping",
                         frozenset({"ShadowClipping"}), None),
    "depthNoiseSigma":  ("GenerationSetting", "double",
                         "HighFrequencyDepthResidual",
                         frozenset({"IdentityMedianDepthChange"}), None),
    "depthScale":       ("GenerationSetting", "double",
                         "IdentityMedianDepthChange",
                         frozenset({"PriorWarpDepthResidual"}), None),
    "injectedFrameCount": ("GenerationSetting", "integer",
                           "HighFrequencyDepthResidual", frozenset(), None),
}

MEASUREMENT_PARAMS = frozenset(
    n for n, d in DECLARED_PARAMS.items() if d[0] == "MeasurementSetting")
QUALIFICATION_PARAMS = frozenset(
    n for n, d in DECLARED_PARAMS.items() if d[0] == "QualificationSetting")
GENERATION_PARAMS = frozenset(
    n for n, d in DECLARED_PARAMS.items() if d[0] == "GenerationSetting")
# Every parameter an experiment may record. v2 recorded ALL of them on EVERY
# experiment; v3's completeness rule is selection-scoped (see `_required_params`).
EXPERIMENT_PARAMS = MEASUREMENT_PARAMS | QUALIFICATION_PARAMS

# The keys `load_parameter_declarations` returns per parameter.
#
# CONTRACT GAP, STATED OUT LOUD: contracts §8 freezes the FUNCTION but not the
# shape of its per-parameter dict (unlike the `load_quality_factors` entry, which
# spells out four keys). The names below are `api.py`'s, adopted here rather than
# invented, and pinned in ONE place so a rename is a two-line change instead of a
# hunt through thirty assertions. They carry no contract content — every fact they
# expose is asserted again, independently, against the `hw1:settingRole` /
# `hw1:settingForFactor` / `hw1:settingValue` triples the emitters actually write.
# If contracts grows a §8 sentence for this, it wins.
DECL_ROLE, DECL_KIND = "role", "kind"
DECL_PRIMARY, DECL_AFFECTS = "primary", "affects"
DECL_DEFAULT, DECL_IRI = "default", "iri"
DECL_KEYS = frozenset({DECL_ROLE, DECL_KIND, DECL_PRIMARY, DECL_AFFECTS,
                       DECL_DEFAULT, DECL_IRI})

# ---------------------------------------------------------------------------
# The ten active QualityFactors, likewise owned here.
#   factor -> (value property, status property, polarity, qualifiedBy parameter)
# ---------------------------------------------------------------------------
QUALITY_FACTORS = {
    "HighFrequencyDepthResidual": (
        "highFrequencyDepthResidual", "highFrequencyDepthResidualStatus",
        "lower", "maxHighFrequencyDepthResidual"),
    "FlyingPixelRatio": ("flyingPixelRatio", "flyingPixelRatioStatus",
                          "lower", "maxFlyingPixelRatio"),
    "ValidTileCoverage": ("validTileCoverage", "validTileCoverageStatus",
                           "higher", "minValidTileCoverage"),
    "HighlightClipping": ("clipHiFraction", "clipHiFractionStatus",
                          "lower", "maxClipHiFraction"),
    "ShadowClipping":   ("clipLoFraction", "clipLoFractionStatus",
                         "lower", "maxClipLoFraction"),
    "IdentityMedianDepthChange": (
        "identityMedianDepthChange", "identityMedianDepthChangeStatus",
        "lower", "maxIdentityMedianDepthChange"),
    "JointValidDepthRatio": (
        "jointValidDepthRatio", "jointValidDepthRatioStatus",
        "higher", "minJointValidDepthRatio"),
    "PriorWarpDepthResidual": (
        "priorWarpDepthResidual", "priorWarpDepthResidualStatus",
        "lower", "maxPriorWarpDepthResidual"),
    "ReconstructionAccuracy": ("mapMeanL2", "mapMeanL2Status",
                               "lower", "maxMapMeanL2"),
    "Coverage":         ("coverageF", "coverageFStatus", "higher", "minCoverageF"),
}
_FACTOR_BY_OVER = {v[0]: k for k, v in QUALITY_FACTORS.items()}

# ---------------------------------------------------------------------------
# The MENU (contracts §4.2) — the six factors a declaration may select, split by
# the node class that carries them, plus the two run factors that are never
# selectable and always evaluated.
#
# `FACTOR_MODALITY` is the §4.3 placement rule: which annotation node an observable
# lands on. It is what makes "a modality with no selected factor gets NO node"
# checkable without asking api.py where it put anything.
# ---------------------------------------------------------------------------
FRAME_FACTORS = ("HighFrequencyDepthResidual", "FlyingPixelRatio",
                 "ValidTileCoverage", "HighlightClipping", "ShadowClipping")
PAIR_FACTORS = ("IdentityMedianDepthChange", "JointValidDepthRatio",
                "PriorWarpDepthResidual")
MENU_FACTORS = FRAME_FACTORS + PAIR_FACTORS
RUN_FACTORS = ("ReconstructionAccuracy", "Coverage")
# §4.2/§13: these three are required on EVERY experiment, whatever is
# selected. `minSegmentLength` was the fourth until it was deleted (§13).
RUN_FACTOR_PARAMS = ("icpBackend", "maxMapMeanL2", "minCoverageF")

FACTOR_MODALITY = {"HighFrequencyDepthResidual": "depth",
                   "FlyingPixelRatio": "depth", "ValidTileCoverage": "depth",
                   "HighlightClipping": "rgb", "ShadowClipping": "rgb"}
MODALITIES = ("rgb", "depth")

# The five frame factor value properties, in the order contracts §4.3 lists them.
FRAME_FACTOR_VALUES = ("highFrequencyDepthResidual", "flyingPixelRatio",
                       "validTileCoverage", "clipHiFraction", "clipLoFraction")
FRAME_FACTOR_STATUSES = tuple(v + "Status" for v in FRAME_FACTOR_VALUES)
PAIR_FACTOR_VALUES = ("identityMedianDepthChange", "jointValidDepthRatio",
                      "priorWarpDepthResidual")
PAIR_FACTOR_STATUSES = tuple(v + "Status" for v in PAIR_FACTOR_VALUES)
RUN_FACTOR_VALUES = ("mapMeanL2", "coverageF")

# Every observable predicate. The batch graph must carry NONE of them.
MEASURED_PREDICATES = (("meanValue",) + FRAME_FACTOR_VALUES + PAIR_FACTOR_VALUES
                       + RUN_FACTOR_VALUES)
# v3 has ONE aggregate predicate — `hw1:qualificationStatus` — on annotations and
# pairs alike (contracts §4.5). `frameStatus` and `pairStatus` are deleted, and
# their absence is asserted separately (`TBoxInvariants`, `DeletedSurface`).
AGGREGATE_STATUS = "qualificationStatus"
STATUS_PREDICATES = (FRAME_FACTOR_STATUSES + PAIR_FACTOR_STATUSES
                     + ("mapMeanL2Status", "coverageFStatus", AGGREGATE_STATUS))

# ---------------------------------------------------------------------------
# The qualification thresholds every predicted-status fixture DECLARES.
#
# Written into the declaration as FactorSettings, never taken from the TBox:
# contracts §5 marks all eight PROVISIONAL, and an expectation computed against a
# number that is scheduled to move is an expectation that will be "fixed" by
# loosening the test. These are the contracts §5 values as of today, but nothing
# here depends on the TBox still agreeing with them tomorrow.
# ---------------------------------------------------------------------------
PINNED_THRESHOLDS = {
    "maxClipHiFraction": 0.05,
    "maxClipLoFraction": 0.30,
    "maxHighFrequencyDepthResidual": 0.05,
    "maxFlyingPixelRatio": 0.05,
    "minValidTileCoverage": 0.50,
    "maxIdentityMedianDepthChange": 0.20,
    "minJointValidDepthRatio": 0.30,
    "maxPriorWarpDepthResidual": 0.10,
    "maxMapMeanL2": 0.80,
    "minCoverageF": 0.40,
}

# ---------------------------------------------------------------------------
# What a number survives on the way through a .ttl.
#
# contracts §5, "Lexical form is rdflib's, and it is lossy": rdflib's Turtle
# serializer normalises every xsd:double to 7 significant digits, and
# `write_run`'s in-place rewrite normalises again. That is uniform and
# deterministic — two files still compare exactly, and no verdict can move — but
# no test may demand more of a value it read back from a file than the serializer
# promises. Assertions on numbers that never went through a file (the measurers)
# stay exact.
#
# v3 adds one exemption in the other direction: a STUDENT-GIVEN setting never goes
# through the serializer at all, because the student section is preserved byte for
# byte (§3.1). `MachineSection` pins that directly, on a value spelled in a way
# rdflib would normalise away.
# ---------------------------------------------------------------------------
SERIALIZER_SIG_DIGITS = 7


def _serializer_delta(expected):
    return max(abs(float(expected)), 1.0) * 10.0 ** -(SERIALIZER_SIG_DIGITS - 1)


# ---------------------------------------------------------------------------
# contracts §3.1 — the marker line, transcribed. `api.MACHINE_MARKER` must equal
# this string exactly; `CommandSurface` asserts that, and every helper below
# splits on THIS constant so a drifted marker reddens one test rather than
# silently splitting every file in the wrong place.
# ---------------------------------------------------------------------------
MACHINE_MARKER = (
    "# ============ MACHINE SECTION (regenerated by api.py — do not edit) "
    "============")

# ---------------------------------------------------------------------------
# The main batch's observables and verdicts under PINNED_THRESHOLDS, tauHi=250,
# tauLo=5 and the [0, 10] m depth window. Arithmetic, not measured-and-recorded:
# every number below is forced by the fixture geometry.
# ---------------------------------------------------------------------------
MAIN_FRAME_VALUES = {
    0: {"meanValue": 128.0, "clipHiFraction": 0.0, "clipLoFraction": 0.0,
        "highFrequencyDepthResidual": 0.0, "flyingPixelRatio": 0.0,
        "validTileCoverage": 1.0},
    1: {"meanValue": 0.0, "clipHiFraction": 0.0, "clipLoFraction": 1.0,
        "highFrequencyDepthResidual": 0.0, "flyingPixelRatio": 0.0,
        "validTileCoverage": 1.0},
    2: {"meanValue": 255.0, "clipHiFraction": 1.0, "clipLoFraction": 0.0,
        "highFrequencyDepthResidual": 0.0, "flyingPixelRatio": 0.0,
        "validTileCoverage": 1.0},
    3: {"meanValue": 128.0, "clipHiFraction": 0.0, "clipLoFraction": 0.0,
        "highFrequencyDepthResidual": 0.0, "flyingPixelRatio": 0.0,
        "validTileCoverage": 0.0},
    # Frame 4's high-frequency residual is a NOISE ESTIMATE, not a closed form:
    # it is checked against BATCH_NOISE_SIGMA_M with the estimator's tolerance.
    4: {"meanValue": 128.0, "clipHiFraction": 0.0, "clipLoFraction": 0.0,
        "flyingPixelRatio": 0.106689453125, "validTileCoverage": 1.0},
}
# True == hw1:Pass, in FRAME_FACTOR_VALUES order.
MAIN_FRAME_STATUSES = {
    0: (True, True, True, True, True),
    1: (True, True, True, True, False),  # clipLoFraction 1.0 > 0.30
    2: (True, True, True, False, True),  # clipHiFraction 1.0 > 0.05
    3: (True, True, False, True, True),  # validTileCoverage 0.0 < 0.50
    4: (False, False, True, True, True), # noisy depth trips both noise factors
}
# The same table split the way v3 stores it: one aggregate per MODALITY.
MAIN_MODALITY_STATUSES = {
    stem: {"depth": all(statuses[:3]),
           "rgb": all(statuses[3:])}
    for stem, statuses in MAIN_FRAME_STATUSES.items()
}


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------
def _rgb_solid(rgb):
    return Image.new("RGB", (IMG, IMG), rgb)


def _rgb_half_black_half_white():
    """Top half (0,0,0), bottom half (255,255,255) — exactly half of each."""
    arr = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    arr[IMG // 2:] = 255
    return Image.fromarray(arr)


def _mm(depth_m):
    """Metres -> the uint16 millimetre raster api.py actually reads."""
    return np.round(np.asarray(depth_m) * 1000.0).astype(np.uint16)


def _depth_clean(z_m=DEPTH_Z_M, size=IMG):
    """Constant depth: every pixel valid, and no high-frequency content at all."""
    return _mm(np.full((size, size), float(z_m)))


def _depth_noisy(sigma_m, z_m=DEPTH_Z_M, size=IMG, seed=0):
    """Constant depth plus i.i.d. Gaussian noise of a KNOWN standard deviation.

    The seed is fixed, so the raster — and therefore every observable measured on
    it — is identical on every run and on every machine.
    """
    rng = np.random.default_rng(seed)
    return _mm(z_m + rng.normal(0.0, float(sigma_m), size=(size, size)))


def _depth_sparse(z_m=DEPTH_Z_M, rows=SPARSE_ROWS):
    """Clean depth over `rows` rows, sensor no-return (raw == 0) everywhere else.

    valid_depth_fraction is exactly rows/IMG. The depth that IS returned is
    perfectly clean, so this frame fails ValidDepthRatio and nothing else — the
    two depth factors have to disagree about it or they are not two factors.
    """
    arr = np.zeros((IMG, IMG), dtype=np.uint16)
    arr[:rows] = int(round(z_m * 1000))
    return arr


def _depth_step(col, near_m, far_m, size=IMG):
    """A vertical depth step: `near_m` left of column `col`, `far_m` from it on.

    The one fixture the edge measurer needs. A step of height h produces a Sobel
    gradient magnitude of exactly h/2 metres per pixel under the normalisation of
    §9 (the 3x3 Sobel answers a ramp of slope a with 8a, and the
    response is divided by 8; a step of h is a ramp of h over the two pixels the
    kernel straddles). So a 0.20 m step lands exactly on the 0.10 default
    threshold, which is the sentence the normalisation test executes.

    Axis-aligned on purpose here — unlike the roughness fixtures, the Sobel pair
    is NOT separable-annihilating, and gx alone answers a vertical step. The same
    property makes this raster's `depth_roughness` exactly 0: the Immerkaer mask
    IS separable, so its column pass annihilates a column-constant step.
    """
    arr = np.full((size, size), int(round(near_m * 1000)), dtype=np.uint16)
    arr[:, col:] = int(round(far_m * 1000))
    return arr


_RGB_MAKERS = {"grey": lambda: _rgb_solid((128, 128, 128)),
               "black": lambda: _rgb_solid((0, 0, 0)),
               "white": lambda: _rgb_solid((255, 255, 255))}
_DEPTH_MAKERS = {"clean": _depth_clean,
                 "sparse": _depth_sparse,
                 "noisy": lambda: _depth_noisy(BATCH_NOISE_SIGMA_M, seed=1)}


def _mkbatch(root):
    """Create root/rgb and root/depth, returning both paths."""
    rgb_dir, depth_dir = os.path.join(root, "rgb"), os.path.join(root, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    return rgb_dir, depth_dir


def _write_batch(root):
    """Materialise _FRAMES under root/rgb and root/depth."""
    rgb_dir, depth_dir = _mkbatch(root)
    for stem, rgb_kind, depth_kind in _FRAMES:
        _RGB_MAKERS[rgb_kind]().save(os.path.join(rgb_dir, f"{stem}.png"))
        Image.fromarray(_DEPTH_MAKERS[depth_kind]()).save(
            os.path.join(depth_dir, f"{stem}.png"))          # uint16 -> I;16


def _write_degenerate_batch(root):
    """Two frames, both mid-grey: one with clean depth, one with NO depth at all.

    Frame 1's depth raster is entirely sensor no-return, so it has no fully-valid
    3x3 window and depth_roughness is inf, its valid fraction is 0.0, and the one
    pair has no jointly-valid pixel at all. This is the batch that proves
    contracts §4.3's totality clause: a failing measurer writes its fail-closed
    value, never nothing.
    """
    rgb_dir, depth_dir = _mkbatch(root)
    for stem, depth in ((0, _depth_clean()),
                        (1, np.zeros((IMG, IMG), dtype=np.uint16))):
        _rgb_solid((128, 128, 128)).save(os.path.join(rgb_dir, f"{stem}.png"))
        Image.fromarray(depth).save(os.path.join(depth_dir, f"{stem}.png"))


# The gap batch: rgb/ carries stems 0,1,2,3 but depth/ carries only 0,1,3. Frame 2
# is therefore NOT a frame of this batch at all, and the pair set changes with it.
GAP_RGB_STEMS = (0, 1, 2, 3)
GAP_DEPTH_STEMS = (0, 1, 3)
GAP_PAIRED_STEMS = (0, 1, 3)
GAP_PAIRS = ((0, 1), (1, 3))


def _write_gap_batch(root):
    """A capture with a stem present in rgb/ only — the silent-answer-change case."""
    rgb_dir, depth_dir = _mkbatch(root)
    for stem in GAP_RGB_STEMS:
        _rgb_solid((128, 128, 128)).save(os.path.join(rgb_dir, f"{stem}.png"))
    for stem in GAP_DEPTH_STEMS:
        Image.fromarray(_depth_clean()).save(os.path.join(depth_dir, f"{stem}.png"))


# The selection batch: ten frames of a scene with one depth step (so the frames
# have structure and StructureOverlap is 1.0 throughout and cannot cut anything by
# accident), all identical except that frames SEL_SPLIT.. sit SEL_JUMP_M further
# away. Every pair therefore has a median depth difference of exactly 0 except the
# one pair that straddles the jump, which has exactly SEL_JUMP_M. Every FRAME
# passes all four frame factors. The usable-link set is arithmetic, not luck.
SEL_FRAMES = 10
SEL_SPLIT = 8                    # frames 0..7 | frames 8..9
SEL_JUMP_M = 0.5                 # > the 0.20 m maxMedianDepthDifference pinned here
SEL_STEP_M = 0.3                 # depth step height; 0.15 m/px >= sobelThreshold
SEL_PAIRS = [(i, i + 1) for i in range(SEL_FRAMES - 1)]
SEL_BAD_PAIR = (SEL_SPLIT - 1, SEL_SPLIT)
SEL_USABLE_LINKS = [(i, i + 1) for i in range(SEL_FRAMES - 1) if i != SEL_SPLIT - 1]
SEL_MIN_SEGMENT = 4              # drops the 2-frame tail, keeps the 8-frame head
SEL_EXPECTED_SEGMENT = list(range(SEL_SPLIT))


def _write_selection_batch(root):
    rgb_dir, depth_dir = _mkbatch(root)
    for stem in range(SEL_FRAMES):
        _rgb_solid((128, 128, 128)).save(os.path.join(rgb_dir, f"{stem}.png"))
        z = DEPTH_Z_M + (SEL_JUMP_M if stem >= SEL_SPLIT else 0.0)
        Image.fromarray(_depth_step(IMG // 2, z, z + SEL_STEP_M)).save(
            os.path.join(depth_dir, f"{stem}.png"))


# The teens batch: five identical good frames whose stems STRADDLE A DECADE, so
# ascending integer order and ascending IRI-string order disagree ("10_11" sorts
# before "8_9"). Nothing else in this file can catch a reader that sorts by IRI.
TEENS_STEMS = (8, 9, 10, 11, 12)
TEENS_PAIRS = [(TEENS_STEMS[k], TEENS_STEMS[k + 1]) for k in range(len(TEENS_STEMS) - 1)]


def _write_teens_batch(root):
    rgb_dir, depth_dir = _mkbatch(root)
    for stem in TEENS_STEMS:
        _rgb_solid((128, 128, 128)).save(os.path.join(rgb_dir, f"{stem}.png"))
        Image.fromarray(_depth_step(IMG // 2, DEPTH_Z_M, DEPTH_Z_M + SEL_STEP_M)).save(
            os.path.join(depth_dir, f"{stem}.png"))


def _extra_rgb(name, image):
    """Write a one-off RGB fixture into the temp dir and return its path."""
    path = os.path.join(_TMP, f"{name}.png")
    image.save(path)
    return path


def _extra_depth(name, arr):
    path = os.path.join(_TMP, f"{name}.png")
    Image.fromarray(arr).save(path)
    return path


def setUpModule():
    """Materialise every fixture batch. Touches NOTHING on `api`.

    Deliberate: a setUpModule that called into the module under test would turn
    one unlanded function into an error on all ~200 tests, including the measurer
    contract that has nothing to do with it.
    """
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="hw1_e2e_")
    _write_batch(os.path.join(_TMP, BATCH_DIRNAME))
    _write_degenerate_batch(os.path.join(_TMP, BATCH_DEGEN_DIRNAME))
    _write_gap_batch(os.path.join(_TMP, BATCH_GAP_DIRNAME))
    _write_selection_batch(os.path.join(_TMP, BATCH_SEL_DIRNAME))
    _write_teens_batch(os.path.join(_TMP, BATCH_TEENS_DIRNAME))
    _write_selection_batch(os.path.join(_TMP, BATCH_VERDICT_DIRNAME))
    os.makedirs(os.path.join(_TMP, "decl"), exist_ok=True)
    os.makedirs(os.path.join(_TMP, "out"), exist_ok=True)


def tearDownModule():
    if _TMP and os.path.isdir(_TMP):
        shutil.rmtree(_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Driving the CLI, and reading the .ttl it wrote
#
# contracts §7 freezes THREE api.py commands and nothing else, so every artefact
# this suite reads is produced by one of them — except the DECLARATIONS, which are
# the one thing in v3 a human writes by hand. Those are authored here as literal
# Turtle text, on purpose: the student section is the deliverable's input format,
# and generating it with rdflib would test a dialect no student will ever type.
# ---------------------------------------------------------------------------
def _batch_dir(dirname):
    return os.path.join(_TMP, dirname)


def _out_path(name):
    out_dir = os.path.join(_TMP, "out")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def _decl_path(stem):
    decl_dir = os.path.join(_TMP, "decl")
    os.makedirs(decl_dir, exist_ok=True)
    return os.path.join(decl_dir, f"{stem}.ttl")


class CliError(RuntimeError):
    """argparse's `SystemExit`, re-raised as an ordinary exception.

    `unittest.suite` catches `Exception` around `setUpClass` but lets `SystemExit`
    through, where it aborts the entire run — so a flag contracts §7 freezes but
    that has not been added yet would take this file down instead of reddening the
    handful of tests that use it. Rejection tests assert `_REJECTED` rather than a
    specific type: contracts pins THAT an illegal invocation stops the command,
    not whether the CLI layer or the parser is the one that stops it.
    """


_REJECTED = (CliError, ValueError, RuntimeError, OSError, KeyError, TypeError)


def _main(argv):
    try:
        return api.main(list(argv))
    except SystemExit as exc:
        if exc.code in (0, None):
            return 0
        raise CliError(f"api.main({list(argv)!r}) exited with {exc.code!r}") from exc


def _run_cli(argv):
    """Run a command that must SUCCEED, returning everything it printed.

    stdout and stderr land in one buffer because `explore`'s views are the thing
    under test and a table printed to the wrong stream is still a table the
    student sees.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            api.main(list(argv))
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise CliError(f"{argv!r} exited with {exc.code!r}: "
                               f"{buf.getvalue()}") from exc
    return buf.getvalue()


def _cli_failure(testcase, argv):
    """Run a command that must FAIL, returning the message it failed with.

    The message is everything the command printed PLUS the exception text, because
    contracts §4.2 requires the error to name the offending triple and does not say
    whether the CLI raises it or prints it before exiting non-zero.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            api.main(list(argv))
        except SystemExit as exc:
            if exc.code in (0, None):
                testcase.fail(f"{argv!r} exited 0; a hard error was required")
            return f"{buf.getvalue()}\nexit={exc.code!r}"
        except AssertionError:
            raise
        except Exception as exc:                        # noqa: BLE001 - see docstring
            return f"{buf.getvalue()}\n{type(exc).__name__}: {exc}"
    testcase.fail(f"{argv!r} succeeded;  requires a hard error")


def _error_message(testcase, fn, *args, **kwargs):
    """Call `fn`, require a non-assertion exception, and return its message."""
    try:
        fn(*args, **kwargs)
    except AssertionError:
        raise
    except Exception as exc:                            # noqa: BLE001 - see docstring
        return str(exc)
    testcase.fail("expected a hard error naming the offending triple; none raised")


def _run_batch2ttl(dirname, out=None, gen=(), derived_from=None, floor=FLOOR):
    argv = ["batch2ttl", "--data-dir", _batch_dir(dirname), "--floor", str(floor)]
    for setting in gen:
        argv += ["--gen", setting]
    if derived_from is not None:
        argv += ["--derived-from", derived_from]
    if out is not None:
        argv += ["--out", out]
    _main(argv)
    return out if out is not None else os.path.join(_batch_dir(dirname), "batch.ttl")


_BATCH_TTL_DONE = set()


def _ensure_batch_ttl(dirname, gen=()):
    """Optional `<data_dir>/batch.ttl` sidecar for generation provenance.

    `declare` / `experiment` now name the capture directory itself. This helper
    remains for tests that need a GenerationSetting on the batch (`--gen`), and
    for the still-supported `explore batch.ttl` view. Failures are swallowed
    here because `batch2ttl` has its own strict class below.
    """
    if dirname in _BATCH_TTL_DONE:
        return os.path.join(_batch_dir(dirname), "batch.ttl")
    _BATCH_TTL_DONE.add(dirname)
    try:
        _run_batch2ttl(dirname, gen=gen)
    except Exception:                                   # noqa: BLE001 - see docstring
        pass
    return os.path.join(_batch_dir(dirname), "batch.ttl")


# ---------------------------------------------------------------------------
# Authoring a declaration (contracts §4.2) — the student's half of the file
# ---------------------------------------------------------------------------
_ROLE_OF = {name: decl[0] for name, decl in DECLARED_PARAMS.items()}


def _literal_text(value, kind):
    """A `hw1:settingValue` literal typed per contracts §5's value kind."""
    if kind == "double":
        return f'"{float(value)!r}"^^xsd:double'
    if kind == "integer":
        return f'"{int(value)}"^^xsd:integer'
    return f'"{value}"^^xsd:string'


def _setting_block(param, value, role=None, factor=None, value_text=None):
    """One student FactorSetting, as the blank node contracts §4.2 shows.

    Blank nodes are the default because §4.5 makes them legal on purpose: v3's
    single-valued `settingForFactor` is what lets a student write `[ … ]` without
    the machine pass ever needing to re-open the node. §2 also allows an IRI —
    `DeclarationReader` exercises that through `extra_body` / `extra_nodes`.
    """
    kind = DECLARED_PARAMS.get(param, (None, "double"))[1]
    role = role if role is not None else _ROLE_OF.get(param, "MeasurementSetting")
    factor = factor if factor is not None else DECLARED_PARAMS.get(
        param, (None, None, "HighlightClipping"))[2]
    text = value_text if value_text is not None else _literal_text(value, kind)
    return ("    hw1:hasFactorSetting [\n"
            f"      a hw1:FactorSetting ;\n"
            f"      hw1:settingParameter hw1:{param} ;\n"
            f"      hw1:settingRole      hw1:{role} ;\n"
            f"      hw1:settingForFactor hw1:{factor} ;\n"
            f"      hw1:settingValue     {text} ]")


def _pinned_settings(selected):
    """Every qualification threshold the selection makes legal, at THIS file's value.

    contracts §4.2 forbids a setting for a parameter no selected factor touches,
    so the pinned vector is scoped to the selection — plus the two run-factor
    thresholds, which §4.2 makes required (and therefore legal) on every
    experiment.
    """
    names = {QUALITY_FACTORS[f][3] for f in selected}
    names |= {"maxMapMeanL2", "minCoverageF"}
    return [(name, PINNED_THRESHOLDS[name]) for name in sorted(names)]


def _required_params(selected):
    """contracts §4.2's completeness rule, TRANSCRIBED — a second implementation.

    For every SELECTED factor: its `hw1:qualifiedBy` parameter, plus every
    Measurement parameter whose primary or affected factor is selected. Plus the
    four run-factor parameters, always. Deliberately not a call into api.py: a
    required set computed by the code under test would agree with itself whatever
    it did.
    """
    required = set(RUN_FACTOR_PARAMS)
    for factor in selected:
        required.add(QUALITY_FACTORS[factor][3])
    for name, (role, _kind, primary, affects, _default) in DECLARED_PARAMS.items():
        if role != "MeasurementSetting":
            continue
        if ({primary} | set(affects)) & set(selected):
            required.add(name)
    return required


def _declaration_text(exp_name, batch_ttl, batch_name, selected, settings=(),
                      label=None, iri_tail=None, omit_batch_file=False,
                      extra_body=(), extra_nodes="", prefixes=None):
    """The student section of contracts §3.1, as literal Turtle.

    Deliberately hand-formatted — two-space oddities, a comment header, a blank
    line before the settings — because `MachineSection` asserts these BYTES
    survive `experiment` and `write_run` untouched. Anything rdflib would
    normalise away is a feature of this fixture, not an accident of it.
    """
    tail = iri_tail if iri_tail is not None else exp_name
    lines = [
        "# ---------------------------------------------------------------",
        f"#  {exp_name} — hand-authored declaration (§4.2)",
        "#  Every byte above the machine marker is mine and stays mine.",
        "# ---------------------------------------------------------------",
    ]
    lines += list(prefixes if prefixes is not None else (
        "@prefix hw1:  <http://taica.course/hw1/ontology#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
    ))
    lines.append("")
    lines.append(f"<{api.NS}experiment/{tail}>")

    body = ["    a hw1:Experiment"]
    if label is not None:
        body.append(f'    rdfs:label "{label}"@en')
    if not omit_batch_file:
        body.append(f'    hw1:batchFile "{batch_ttl}"')
    if batch_name is not None:
        body.append(f"    hw1:onBatch <{api.NS}batch/{batch_name}>")
    if selected is not None:
        joined = " ,\n                        ".join(f"hw1:{f}" for f in selected)
        body.append(f"    hw1:evaluatesFactor {joined}")
    body.extend(extra_body)
    for entry in settings:
        body.append(_setting_block(*entry) if isinstance(entry, tuple)
                    else str(entry))

    lines.append(" ;\n\n".join(body) + " .")
    lines.append("")
    if extra_nodes:
        lines.append(extra_nodes)
        lines.append("")
    return "\n".join(lines)


def _write_declaration(stem, text):
    path = _decl_path(stem)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _declare(stem, dirname, selected, settings=(), pinned=True, label=None,
             batch_name=None, batch_ttl=None, floor=FLOOR, **kwargs):
    """Author `<stem>.ttl` against a fixture capture directory and return its path."""
    resolved_batch = batch_ttl
    if resolved_batch is None:
        resolved_batch = _batch_dir(dirname)
        assert os.path.isdir(resolved_batch), (
            f"{resolved_batch} is missing: a declaration names the capture "
            f"directory in hw1:batchFile.")
    if batch_name is None:
        batch_name = f"floor{floor}_{dirname}"
    entries = list(_pinned_settings(selected) if pinned and selected else [])
    entries += list(settings)
    return _write_declaration(
        stem, _declaration_text(stem, resolved_batch, batch_name, selected,
                                settings=entries, label=label, **kwargs))


def _emit_experiment(stem, dirname, selected, **kwargs):
    """Author a declaration and assess it — the whole v3 `experiment` path.

    contracts §7: ONE positional argument. `--set`, `--batch-dir`, `--floor`,
    `--exp-id`, `--label`, `--no-pairs` and `--out` are deleted; everything they
    carried lives in the declaration this helper writes.
    """
    path = _declare(stem, dirname, selected, **kwargs)
    _main(["experiment", path])
    return path


# ---------------------------------------------------------------------------
# Reading a two-section file back (contracts §3.1)
# ---------------------------------------------------------------------------
def _bytes_of(path):
    with open(path, "rb") as fh:
        return fh.read()


def _text_of(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _marker_offset(path):
    """Byte offset of the start of the marker line, or None if it is absent."""
    data = _bytes_of(path)
    index = data.find(MACHINE_MARKER.encode("utf-8"))
    return None if index < 0 else index


def _declaration_bytes(path):
    """The bytes contracts §3.1 seals: offset 0 up to the start of the marker."""
    offset = _marker_offset(path)
    assert offset is not None, f"{path} carries no machine marker"
    return _bytes_of(path)[:offset]


def _machine_text(path):
    """The marker line and everything under it — what §3.1 lets a tool rewrite."""
    text = _text_of(path)
    index = text.find(MACHINE_MARKER)
    assert index >= 0, f"{path} carries no machine marker"
    return text[index:]


def _seal_of(path):
    """contracts §3.1's tamper seal, transcribed: sha256 of the declaration bytes."""
    return hashlib.sha256(_declaration_bytes(path)).hexdigest()


def _append_below_marker(src, stem, text):
    """Copy `src` to a new file with `text` appended to its MACHINE section.

    The declaration bytes are untouched, so `hw1:declarationDigest` still verifies
    and the mutation is visible to the reader as a graph-level defect rather than
    as a broken seal. That is the only way to test a §8.2 rule about the parsed
    graph without tripping the §3.1 rule about the bytes first.
    """
    path = _out_path(f"{stem}.ttl")
    with open(path, "wb") as fh:
        fh.write(_bytes_of(src))
        fh.write(text.encode("utf-8"))
    return path


def _edit_declaration(src, stem, old, new):
    """Copy `src` with one substitution ABOVE the marker — the §3.1 tamper case."""
    data = _bytes_of(src)
    offset = _marker_offset(src)
    assert offset is not None, f"{src} carries no machine marker"
    head, tail = data[:offset], data[offset:]
    assert old.encode("utf-8") in head, f"{old!r} is not in the declaration of {src}"
    path = _out_path(f"{stem}.ttl")
    with open(path, "wb") as fh:
        fh.write(head.replace(old.encode("utf-8"), new.encode("utf-8")) + tail)
    return path


def _graph_of(path):
    g = Graph()
    g.parse(path, format="turtle")
    return g


def _sole_experiment(g):
    subjects = list(g.subjects(RDF.type, api.HW1.Experiment))
    assert len(subjects) == 1, f"expected exactly one hw1:Experiment, got {subjects}"
    return subjects[0]


def _capture_help(argv):
    """Return the help text argparse prints for `argv`, without exploding the test."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            api.main(list(argv))
        except SystemExit:
            pass
        except Exception:                               # noqa: BLE001
            pass
    return buf.getvalue()


def _snapshot(paths):
    """{path: (mtime_ns, bytes)} — what a read-only command must not change."""
    return {p: (os.stat(p).st_mtime_ns, _bytes_of(p)) for p in paths}


def _expected_status(value_property_local, value, settings):
    """The contracts §4.5 status rule, TRANSCRIBED — a second implementation.

    Deliberately not a call to `api.status_for`: cross-checking every stored
    status against the function that wrote it would only prove the code agrees
    with itself. This reading comes from the contract table, and the two have to
    meet on real measured values.
    """
    factor = _FACTOR_BY_OVER[value_property_local]
    _over, _status, polarity, param = QUALITY_FACTORS[factor]
    threshold = settings[param]
    if math.isnan(value):
        return False
    return value >= threshold if polarity == "higher" else value <= threshold


def _annotation_nodes(g, exp):
    """{(frame_index, kind): node} over `hw1:producesAnnotation` (contracts §4.3)."""
    out = {}
    for node in g.objects(exp, api.HW1.producesAnnotation):
        index = int(g.value(node, api.HW1.frameIndex))
        kind = str(node).rsplit("/", 1)[-1]
        out[(index, kind)] = node
    return out


def _pair_nodes(g, exp):
    """{(i, j): node} over `hw1:producesPair`, keyed by the frame stems."""
    out = {}
    for node in g.objects(exp, api.HW1.producesPair):
        i, j = str(node).rsplit("/", 1)[-1].split("_")
        out[(int(i), int(j))] = node
    return out


def _recorded_settings(g, exp):
    """{param_local: literal} over every `hw1:hasFactorSetting` of the experiment."""
    out = {}
    for node in g.objects(exp, api.HW1.hasFactorSetting):
        param = str(g.value(node, api.HW1.settingParameter)).rsplit("#", 1)[-1]
        out[param] = g.value(node, api.HW1.settingValue)
    return out


# =============================================================================
# 1. The frame measurers — closed-form fixtures, tau always explicit
#    contracts §9: frozen VERBATIM from v1. Zero work in the v2 pass; if one of
#    these goes red, a "refactor" changed a measured quantity.
# =============================================================================
class Measurers(unittest.TestCase):
    """Pure checks. No graph needed; these are the autograded measurer contract."""

    # ---- the baseline ----------------------------------------------------
    def test_mean_value_of_solid_frame_is_the_value(self):
        p = _extra_rgb("solid_mid_grey", _rgb_solid((128, 128, 128)))
        self.assertAlmostEqual(api.frame_mean_value(p), 128.0, places=6)

    def test_mean_value_uses_max_channel_not_a_weighted_luma(self):
        # (255,0,0): max(R,G,B) = 255 everywhere. Rec.601 luma would be 76.245.
        p = _extra_rgb("solid_red_mean", _rgb_solid((255, 0, 0)))
        self.assertAlmostEqual(api.frame_mean_value(p), 255.0, places=6)

    # ---- clip fractions, solid mid-grey ---------------------------------
    def test_solid_mid_grey_clips_neither_way(self):
        """Any 5 < tau_lo < 128 < tau_hi < 250 must give clip_hi = clip_lo = 0."""
        p = _extra_rgb("mid_grey", _rgb_solid((128, 128, 128)))
        for tau_lo, tau_hi in ((16.0, 250.0), (6.0, 249.0), (127.0, 129.0)):
            self.assertEqual(api.frame_clip_hi_fraction(p, tau_hi), 0.0)
            self.assertEqual(api.frame_clip_lo_fraction(p, tau_lo), 0.0)
        self.assertAlmostEqual(api.frame_mean_value(p), 128.0, places=6)

    # ---- LOAD-BEARING FIXTURE 1 -----------------------------------------
    def test_half_black_half_white_is_why_the_mean_is_excluded(self):
        """The executable proof that mean(V) cannot be a quality factor.

        contracts §4.3 gives `hw1:meanValue` a value and NO status, alone among
        the frame observables, and §4.5 excludes it from `hw1:frameStatus`. This
        is why. Half the pixels are crushed to black and half are railed to white:
        not one pixel of this frame carries recoverable information. clip_lo and
        clip_hi both report exactly 0.5 and say so. The mean reports 127.5 — dead
        centre of [0,255], indistinguishable from a perfectly exposed mid-grey
        frame.

        A first moment is blind to distribution shape. That is the whole argument,
        and this is where it is checked.
        """
        p = _extra_rgb("half_black_half_white", _rgb_half_black_half_white())
        self.assertAlmostEqual(api.frame_clip_hi_fraction(p, TAU_HI), 0.5, places=9)
        self.assertAlmostEqual(api.frame_clip_lo_fraction(p, TAU_LO), 0.5, places=9)
        self.assertAlmostEqual(api.frame_mean_value(p), 127.5, places=6)

        # ... and the mean is indistinguishable from an actually-fine frame.
        ok = _extra_rgb("even_grey_127", _rgb_solid((128, 128, 128)))
        self.assertLess(abs(api.frame_mean_value(p) - api.frame_mean_value(ok)), 1.0)
        self.assertNotEqual(api.frame_clip_hi_fraction(p, TAU_HI),
                            api.frame_clip_hi_fraction(ok, TAU_HI))

    # ---- LOAD-BEARING FIXTURE 2 -----------------------------------------
    def test_solid_pure_red_is_why_luma_is_excluded(self):
        """The executable proof that no colorimetric weighting may be used.

        Pure red (255,0,0) has its red channel fully railed in every pixel, so
        V = max(R,G,B) = 255 and clip_hi = 1.0 — correct, the frame is clipped.
        Rec.601 luma of the same pixel is 0.299*255 = 76.245: an ordinary mid-tone.
        A weighted reduction HIDES single-channel saturation, and clipping is a
        per-channel phenomenon, so the per-channel test is the correct one.

        It is also the counter-example against the "symmetric" shadow test: pure
        red has G = B = 0, so min(R,G,B) <= tau_lo would call this fully-saturated
        frame crushed. clip_lo on V correctly reports 0.
        """
        p = _extra_rgb("solid_pure_red", _rgb_solid((255, 0, 0)))
        self.assertEqual(api.frame_clip_hi_fraction(p, TAU_HI), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, TAU_LO), 0.0)

        rec601 = 0.299 * 255.0
        self.assertLess(rec601, 128.0)                       # would read mid-tone
        self.assertEqual(api.frame_mean_value(p), 255.0)     # V does not

    # ---- degenerate ends -------------------------------------------------
    def test_all_black_and_all_white_frames(self):
        black = _extra_rgb("all_black", _rgb_solid((0, 0, 0)))
        white = _extra_rgb("all_white", _rgb_solid((255, 255, 255)))
        self.assertEqual(api.frame_clip_lo_fraction(black, TAU_LO), 1.0)
        self.assertEqual(api.frame_clip_hi_fraction(black, TAU_HI), 0.0)
        self.assertEqual(api.frame_clip_hi_fraction(white, TAU_HI), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(white, TAU_LO), 0.0)

    def test_clip_fractions_are_inclusive_at_tau(self):
        """The comparisons are >= and <=, so a pixel exactly at tau counts."""
        p = _extra_rgb("exactly_tau", _rgb_solid((200, 200, 200)))
        self.assertEqual(api.frame_clip_hi_fraction(p, 200.0), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, 200.0), 1.0)
        self.assertEqual(api.frame_clip_hi_fraction(p, 201.0), 0.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, 199.0), 0.0)

    def test_clip_fraction_is_closed_form_at_any_tau(self):
        """A synthetic frame has an exact clip fraction for ANY tau, which is what
        keeps the measurers autogradable independently of the tau any one
        experiment happens to record.

        The fixture is a vertical ramp: column j has V = 4j, so
            clip_hi(tau) = |{j : 4j >= tau}| / 64   exactly.
        """
        values = (np.arange(IMG, dtype=np.uint8) * 4)         # 0, 4, ..., 252
        arr = np.repeat(values[None, :, None], IMG, axis=0).repeat(3, axis=2)
        p = _extra_rgb("value_ramp", Image.fromarray(arr))
        for tau in (0.0, 1.0, 100.0, 200.0, 252.0, 253.0):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(api.frame_clip_hi_fraction(p, tau),
                                       float((values >= tau).sum()) / IMG, places=12)
                self.assertAlmostEqual(api.frame_clip_lo_fraction(p, tau),
                                       float((values <= tau).sum()) / IMG, places=12)


# =============================================================================
# 3. The IRI scheme — contracts §2, minted by the helpers of §8
# =============================================================================
class Namespaces(unittest.TestCase):
    """contracts §1/§8 — the seven namespace bindings other modules import."""

    def test_the_seven_frozen_namespaces_are_exported(self):
        """§8's first line, one assertion per prefix.

        `SKOS` is on that list because the TBox declares four closed vocabularies
        (`hw1:Polarity`, `hw1:Status`, `hw1:SelectionMode`, `hw1:SettingRole`) as
        `skos:Concept` subclasses and reads their labels with `skos:prefLabel`.
        A module that has to write `Namespace("http://…/skos/core#")` inline has
        already forked the prefix table.
        """
        expected = {
            "NS": "http://taica.course/hw1/ontology#",
            "SCHEMA": "https://schema.org/",
            "QUDT": "http://qudt.org/schema/qudt/",
            "UNIT": "http://qudt.org/vocab/unit/",
            "SKOS": "http://www.w3.org/2004/02/skos/core#",
            "PROV": "http://www.w3.org/ns/prov#",
        }
        for name, uri in expected.items():
            with self.subTest(namespace=name):
                self.assertEqual(str(_require(name)), uri)
        self.assertEqual(str(_require("HW1")), str(api.NS))

    def test_dqv_is_gone(self):
        """contracts §1: the Result/Metric shape it carried is deleted.

        Leaving the binding in place is how a `dqv:` triple gets written by
        accident six months from now, at which point the vocabulary has two ways
        to say what a measurement is and every query has to know both.
        """
        self.assertFalse(hasattr(api, "DQV"),
                         "§1 drops the dqv: namespace entirely")


class IriScheme(unittest.TestCase):
    """Every minter of contracts §8 produces exactly the string of §2.

    The scheme is a contract between `batch2ttl` (which mints the structural
    IRIs), `experiment` (which mints everything experiment-scoped over the same
    frames), `reconstruct.py` (which reads them back) and `explore`. Written as
    literal f-strings here on purpose: comparing `api.frame_iri` against
    `api.frame_iri` would pass against any scheme at all.

    v3 changes two rows: `<expname>` is the declaration file's STEM rather than
    an 8-hex digest (§6), and an annotation IRI carries a `<kind>` tail because
    annotations are per modality (§4.3).
    """

    EXP = "strict_clip"

    def test_batch_and_frame_and_image_iris(self):
        self.assertEqual(str(api.batch_iri(BATCH)), f"{api.NS}batch/{BATCH}")
        self.assertEqual(str(api.frame_iri(BATCH, 7)),
                         f"{api.NS}batch/{BATCH}/frame/7")
        self.assertEqual(str(api.component_iri(BATCH, 7, "rgb")),
                         f"{api.NS}batch/{BATCH}/frame/7/rgb")
        self.assertEqual(str(api.component_iri(BATCH, 7, "depth")),
                         f"{api.NS}batch/{BATCH}/frame/7/depth")

    def test_generation_setting_iri_hangs_off_the_batch(self):
        """§2: `<ns>batch/<name>/setting/<param>` — batch-scoped, not experiment.

        A GenerationSetting describes the PIXELS, so it survives every
        re-measurement of them and must not carry an experiment name; `explore`'s
        verdict section reaches it from the BATCH file, not from the experiment
        (§7.1).
        """
        minter = _require("generation_setting_iri")
        self.assertEqual(str(minter(BATCH, "brightnessGain")),
                         f"{api.NS}batch/{BATCH}/setting/brightnessGain")

    def test_the_experiment_iri_tail_is_the_name(self):
        """contracts §6: identity is the NAME, and the name is the file stem.

        `experiment_iri` takes that name. A digest argument would reintroduce the
        machine-chosen filename v3 deleted, and the student would lose the one
        thing naming buys — a lab notebook whose file names mean something.
        """
        self.assertEqual(str(api.experiment_iri(self.EXP)),
                         f"{api.NS}experiment/{self.EXP}")
        self.assertEqual(str(api.experiment_iri("depth_only-2")),
                         f"{api.NS}experiment/depth_only-2")

    def test_experiment_scoped_iris(self):
        """The four minters that carry an experiment name, §2 rows 6-10.

        `setting_iri` takes the PRIMARY factor's local name, not the parameter's
        role and not every affected factor: §2 says only the primary appears in
        the IRI, so `brightnessGain` — which also affects ShadowClipping — has
        exactly one machine-minted setting node.
        """
        exp = self.EXP
        self.assertEqual(
            str(_require("setting_iri")(exp, "HighlightClipping", "tauHi")),
            f"{api.NS}experiment/{exp}/setting/HighlightClipping/tauHi")
        self.assertEqual(str(_require("run_iri")(exp, "baseline")),
                         f"{api.NS}experiment/{exp}/run/baseline")
        self.assertEqual(str(_require("run_iri")(exp, "selected")),
                         f"{api.NS}experiment/{exp}/run/selected")

    def test_an_annotation_iri_names_its_modality(self):
        """SIGNATURE CHANGED (contracts §8): `annotation_iri(expname, idx, kind)`.

        v3 mints one annotation per frame PER MODALITY (§4.3), so the frame index
        alone no longer identifies an annotation — `.../annotation/3/rgb` and
        `.../annotation/3/depth` are two nodes carrying two aggregates over two
        disjoint factor sets. A minter that dropped the tail would have the depth
        pass overwrite the rgb pass on one subject and the file would still parse.
        """
        minter = _require("annotation_iri")
        self.assertEqual(str(minter(self.EXP, 3, "rgb")),
                         f"{api.NS}experiment/{self.EXP}/annotation/3/rgb")
        self.assertEqual(str(minter(self.EXP, 3, "depth")),
                         f"{api.NS}experiment/{self.EXP}/annotation/3/depth")
        self.assertNotEqual(minter(self.EXP, 3, "rgb"), minter(self.EXP, 3, "depth"))

    def test_pair_iri_is_experiment_scoped(self):
        """`pair_iri(expname, i, j)` — a pair belongs to an experiment, not a batch.

        In v1 a FramePair was a skeleton in `batch.ttl` and its IRI was
        batch-scoped, because every experiment shared it. v2 moved it and v3 keeps
        it there: a pair is two links, an index and — when a pair factor is
        selected — two observables, so sharing forced two experiments to hang
        contradictory values on one subject the moment the named graphs went away.
        """
        self.assertEqual(str(api.pair_iri(self.EXP, 0, 1)),
                         f"{api.NS}experiment/{self.EXP}/pair/0_1")
        self.assertNotIn("batch/", str(api.pair_iri(self.EXP, 0, 1)))

    def test_pair_iri_is_ordered_and_unpadded(self):
        """`<i>_<j>` with i before j in ascending stem order, decimal, no padding.

        `1_3` is a real pair of the gap batch — the stems of two ADJACENT frames
        need not differ by one — and `3_1` is not the same node, because
        `hw1:sourceFrame` is the ICP source and the order is the direction of time.
        """
        self.assertEqual(str(api.pair_iri(self.EXP, 1, 3)),
                         f"{api.NS}experiment/{self.EXP}/pair/1_3")
        self.assertNotEqual(api.pair_iri(self.EXP, 1, 3),
                            api.pair_iri(self.EXP, 3, 1))
        self.assertEqual(str(api.pair_iri(self.EXP, 8, 10)),
                         f"{api.NS}experiment/{self.EXP}/pair/8_10")

    def test_nothing_an_experiment_mints_collides_across_experiments(self):
        """contracts §2, the rule that must never be relaxed.

        This is what replaced named graphs. Two experiments over one batch share
        every FRAME IRI — that is the join — and share nothing else. In v3 the
        thing keeping them apart is the student's chosen NAME, which is also why
        write-once matters: two names cannot collide on a filesystem, but one name
        reused for two treatments would collide everywhere.
        """
        a, b = "strict_clip", "loose_clip"
        minted = []
        for exp in (a, b):
            minted.append({
                str(api.experiment_iri(exp)),
                str(_require("setting_iri")(exp, "HighlightClipping", "tauHi")),
                str(_require("annotation_iri")(exp, 3, "rgb")),
                str(api.pair_iri(exp, 3, 4)),
                str(_require("run_iri")(exp, "baseline")),
            })
        self.assertEqual(minted[0] & minted[1], set(),
                         "an experiment-scoped IRI leaked out of its experiment")
        # ... while the shared structure really is shared.
        self.assertEqual(api.frame_iri(BATCH, 3), api.frame_iri(BATCH, 3))

    def test_frame_iri_round_trips_through_both_parsers(self):
        """contracts §8: ONE tail-parse implementation, and it is exact.

        `reconstruct.py` needs integer stems and reads only an experiment file,
        but `hw1:frameIndex` lives in the BATCH file — so the stem has to come back
        out of the IRI. Minting and parsing must be exact inverses, or a frame list
        ends up silently short by the frames it could not parse, which shows up as
        a slightly worse reconstruction score and never as an error.
        """
        for name in (BATCH, BATCH_B, "floor2_mixed_dev"):
            for idx in (0, 7, 42, 1234):
                with self.subTest(name=name, idx=idx):
                    iri = api.frame_iri(name, idx)
                    self.assertEqual(api.frame_index_from_iri(iri), idx)
                    self.assertEqual(api.batch_name_from_frame_iri(iri), name)

    def test_frame_indices_are_unpadded_so_there_is_one_spelling_per_frame(self):
        """`frame/7` and `frame/007` would be two IRIs for one frame.

        It is also why contracts §2 orders by `hw1:frameIndex` and never by IRI
        string: unpadded stems put `frame/10` before `frame/8`.
        """
        self.assertEqual(str(api.frame_iri(BATCH, 7)),
                         f"{api.NS}batch/{BATCH}/frame/7")
        self.assertLess(f"{api.NS}batch/{BATCH}/frame/10",
                        f"{api.NS}batch/{BATCH}/frame/8",
                        "lexicographic order really does disagree with integer order")

    def test_a_component_iri_is_not_a_frame_iri(self):
        """Strict on purpose: `.../frame/7/depth` must raise, not return 7.

        Returning a plausible-looking number here is how an image node quietly gets
        counted as a frame — and in v3 `hw1:describesImage` points annotations
        straight at those nodes, so they are handled far more often than in v2.
        """
        with self.assertRaises(ValueError):
            api.frame_index_from_iri(api.component_iri(BATCH, 7, "depth"))

    def test_an_iri_outside_the_scheme_raises(self):
        """Neither parser guesses. An IRI this project did not mint is an error.

        DELIBERATELY NOT PINNED: whether `frame_index_from_iri` also accepts an
        ANNOTATION IRI. v2's suite required it to raise on one; v3's annotations
        carry the same `<n>` in the same position, so reading it out of either
        shape is one tail parse rather than two — which is what contracts §8's
        "ONE tail-parse implementation" is asking for. Both readings satisfy the
        clause, so this file pins the two cases that are unambiguous instead.
        """
        bad = ["https://example.org/frame/7", f"{api.NS}experiment/strict_clip"]
        for iri in bad:
            with self.subTest(iri=iri):
                with self.assertRaises(ValueError):
                    api.frame_index_from_iri(URIRef(iri))
                with self.assertRaises(ValueError):
                    api.batch_name_from_frame_iri(URIRef(iri))

    def test_an_annotation_iri_carries_no_batch_name(self):
        """`batch_name_from_frame_iri` is scoped to the BATCH shape and only that.

        An annotation IRI is experiment-scoped: the name in it is the
        EXPERIMENT's. Returning it as a batch name would build frame IRIs under a
        batch that does not exist, and every join against the structure file would
        come back empty rather than wrong — which is worse to debug.
        """
        with self.assertRaises(ValueError):
            api.batch_name_from_frame_iri(
                _require("annotation_iri")(self.EXP, 7, "rgb"))

    def test_batch_name_is_floor_qualified(self):
        """floor1_x and floor2_x are different captures and must not collide."""
        self.assertEqual(api.batch_name(_batch_dir(BATCH_DIRNAME), FLOOR), BATCH)
        self.assertNotEqual(api.batch_name(_batch_dir(BATCH_DIRNAME), 2), BATCH)
        self.assertEqual(api.batch_name(_batch_dir(BATCH_DIRNAME), 2),
                         f"floor2_{BATCH_DIRNAME}")


# =============================================================================
# 4. Parameters — the TBox is the AUTHORITY on which ones exist (contracts §5)
# =============================================================================
class ParameterDeclarations(unittest.TestCase):
    """`load_parameter_declarations` — seventeen parameters, five facts each.

    The TABLE is unchanged from v2 (contracts §5, first line). What changed is
    REQUIREDNESS: a parameter is required on an experiment iff the §4.2
    completeness rule selects it, which `SelectionScoping` pins separately.
    """

    @classmethod
    def setUpClass(cls):
        cls.decls = _require("load_parameter_declarations")()

    def test_a_declaration_exposes_the_documented_keys(self):
        """CONTRACT GAP, pinned here rather than improvised in thirty places.

        contracts §8 freezes the FUNCTION `load_parameter_declarations` but — unlike
        the `load_quality_factors` entry, which spells out four keys — says nothing
        about the shape of its per-parameter dict. These six names are this file's
        reading of §5 (the TBox predicate local names minus their `param` prefix).
        Every other test in this class goes through the DECL_* constants at the top
        of this file, so if contracts grows a sentence that names them differently,
        this is a six-line change and not a rewrite.
        """
        self.assertTrue(self.decls, "no parameters declared at all")
        for name in sorted(self.decls):
            with self.subTest(parameter=name):
                self.assertLessEqual(DECL_KEYS, set(self.decls[name]))

    def test_the_seventeen_declared_parameters_are_exactly_these(self):
        """contracts §5's table, as a set equality."""
        self.assertEqual(set(self.decls), set(DECLARED_PARAMS))

    def test_each_parameter_declares_its_role_kind_and_factors(self):
        """The four facts attribution reads off a parameter (contracts §4.2/§7.1).

        `role` IS the verdict `explore` prints — Generation means regenerate the
        data, Measurement means change the number and re-measure, Qualification
        means change the threshold and re-qualify — so a parameter that silently
        changes role changes what the tool tells a student to go and fix.
        `primaryFactor` is the `<factor>` segment of the machine-minted setting IRI
        and the one legal `hw1:settingForFactor` value (§4.5), so it must never be
        ambiguous. `affectsFactors` is what lets the verdict section blame
        `brightnessGain` for a ShadowClipping failure even though its primary
        factor is HighlightClipping — v3 deleted the duplicated `settingForFactor`
        triples that used to carry that, so this link is now the ONLY path.
        """
        for name, (role, kind, primary, affects, _default) in DECLARED_PARAMS.items():
            with self.subTest(parameter=name):
                decl = self.decls[name]
                self.assertEqual(decl[DECL_ROLE], role)
                self.assertEqual(decl[DECL_KIND], kind)
                self.assertEqual(decl[DECL_PRIMARY], primary)
                self.assertEqual(frozenset(decl[DECL_AFFECTS]), affects)
                self.assertEqual(URIRef(str(decl[DECL_IRI])), api.HW1[name])

    def test_value_kinds_are_the_three_declared_ones(self):
        """"double" | "integer" | "string" (contracts §5), and the two non-doubles
        are the reason the field exists.

        `icpBackend` is categorical; v4's window and tile sizes are counts.
        """
        by_kind = {}
        for name, decl in self.decls.items():
            by_kind.setdefault(decl[DECL_KIND], set()).add(name)
        self.assertLessEqual(set(by_kind), {"double", "integer", "string"})
        self.assertEqual(by_kind.get("string"), {"icpBackend"})
        self.assertEqual(by_kind.get("integer"),
                         {"flyingPixelWindow", "tileSize", "injectedFrameCount"})

    def test_generation_parameters_declare_no_default(self):
        """contracts §5: absence means "NOT ASSERTED", not "identity".

        Filling `brightnessGain` from a default would have every batch assert a
        gain was applied to pixels this code never touched — a claim nothing
        verified. It is the one exemption from the completeness rule, and it is an
        exemption about honesty rather than about convenience.
        """
        for name in sorted(GENERATION_PARAMS):
            with self.subTest(parameter=name):
                self.assertIsNone(self.decls[name][DECL_DEFAULT])

    def test_every_other_parameter_declares_a_usable_default(self):
        """The other half of the completeness rule (contracts §4.2/§5).

        `experiment` fills every REQUIRED parameter the student did not give from
        `hw1:paramDefault`, below the marker. A required parameter with nothing to
        record would make the recorded vector partial over its own selection, and
        the "which side of the marker" reading of given-vs-defaulted would have a
        third state nothing can express.
        """
        for name in sorted(EXPERIMENT_PARAMS):
            with self.subTest(parameter=name):
                default = self.decls[name][DECL_DEFAULT]
                self.assertIsNotNone(default)
                if DECLARED_PARAMS[name][1] != "string":
                    self.assertTrue(math.isfinite(float(default)),
                                    f"{name} default {default!r} is not a usable number")

    def test_measurement_defaults_are_the_pipeline_constants(self):
        """Measurement defaults are pipeline constants and do not float.

        250/5 are the MATLAB `makehdr` 98%/2% pair — constants of the 8-bit
        pipeline. `open3d` is the reconstruction default. Contrast the
        qualification thresholds, which this file deliberately never pins.
        """
        for name in sorted(MEASUREMENT_PARAMS):
            expected = DECLARED_PARAMS[name][4]
            with self.subTest(parameter=name):
                default = self.decls[name][DECL_DEFAULT]
                if DECLARED_PARAMS[name][1] == "string":
                    # contracts §5: `"open3d"^^xsd:string`, so read it through
                    # str()/.toPython() — rdflib does not equate a plain literal
                    # with an xsd:string-typed one.
                    self.assertEqual(str(default), expected)
                    self.assertIsInstance(str(default), str)
                elif DECLARED_PARAMS[name][1] == "integer":
                    self.assertEqual(int(default), expected)
                else:
                    self.assertAlmostEqual(float(default), expected, places=12)

    def test_qualification_defaults_are_asserted_by_shape_not_by_value(self):
        """Deliberately weak, and the docstring is the reason.

        contracts §5 marks every qualification threshold PROVISIONAL pending the
        OFAT re-cut. A test that pinned these numbers would go red
        the first time one moved, which is the one thing they are SUPPOSED to do,
        and the pressure it creates is to leave a known-wrong threshold in place.
        So: assert that each is present and usable, and let the value float. Every
        status this suite predicts is computed against PINNED_THRESHOLDS instead,
        DECLARED explicitly in the fixture's own student section.
        """
        for name in sorted(QUALIFICATION_PARAMS):
            with self.subTest(parameter=name):
                self.assertIs(DECLARED_PARAMS[name][4], PROVISIONAL)
                value = float(self.decls[name][DECL_DEFAULT])
                self.assertTrue(math.isfinite(value) and value > 0.0,
                                f"{name} default {value!r} is not a usable threshold")

    def test_every_factor_named_by_a_parameter_is_a_declared_factor(self):
        """A dangling `hw1:paramPrimaryFactor` makes a machine-minted setting IRI
        point at nothing and drops the parameter out of the verdict section
        entirely — silently, because a missing join contributes no row rather than
        an error."""
        for name, decl in self.decls.items():
            with self.subTest(parameter=name):
                named = {decl[DECL_PRIMARY]} | set(decl[DECL_AFFECTS])
                self.assertLessEqual(named, set(QUALITY_FACTORS))

    def test_parameter_local_names_are_globally_unique(self):
        """Which is why a student writes `hw1:settingParameter hw1:tauHi` with no
        factor qualifier.

        A dict is unique by construction, so the real assertion is that this file's
        table and the TBox agree on the count — a parameter declared twice under
        two factors would collapse to one entry and the loss would be invisible.
        """
        self.assertEqual(len(self.decls), len(DECLARED_PARAMS))
        self.assertEqual(len(self.decls), 24)


class SettingParsing(unittest.TestCase):
    """`parse_setting_arg` — still frozen in §8, still typed by the DECLARATION.

    v3 deletes `--set`, so this parser's one remaining caller is `batch2ttl --gen`.
    It stays on the public surface because the rule it implements — a level is
    typed by `hw1:paramValueKind` and never by what the text looks like — is the
    same rule the DECLARATION validator applies to `hw1:settingValue` (§4.2).
    """

    @classmethod
    def setUpClass(cls):
        cls.decls = _require("load_parameter_declarations")()
        # staticmethod: a bare function stored on a CLASS becomes a bound method,
        # and `self.parse(arg, decls)` would then hand the parser three arguments.
        cls.parse = staticmethod(_require("parse_setting_arg"))

    def test_a_double_parameter_yields_a_float(self):
        name, value = self.parse("tauHi=250", self.decls)
        self.assertEqual(name, "tauHi")
        self.assertIsInstance(value, float)
        self.assertEqual(value, 250.0)

    def test_a_qualification_threshold_parses_like_any_other_double(self):
        """`tauHi=252` and `maxClipHiFraction=0.02` go through ONE code path.

        Thresholds are ordinary parameters — that is what makes sweeping strictness
        the same exercise as sweeping tau. A parser that special-cased them would
        reintroduce the v1 split between "knobs" and "band bounds".
        """
        self.assertEqual(self.parse("tauHi=252", self.decls), ("tauHi", 252.0))
        name, value = self.parse("maxClipHiFraction=0.02", self.decls)
        self.assertEqual(name, "maxClipHiFraction")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0.02, places=12)

    def test_an_integer_parameter_yields_an_int(self):
        """A count parses to an int, and 25.0 would be the wrong triple.

        contracts §5 types `hw1:settingValue` per the declared kind. A float here
        writes `"25.0"^^xsd:integer` — ill-typed — or `25.0^^xsd:double`.

        NO PARAMETER DECLARES kind "integer" since `minSegmentLength` was deleted
        (§13), so this drives the parser through a synthetic declaration rather
        than through the TBox: the rule is the mechanism's, not one parameter's,
        and the next count-valued parameter must inherit it already working.
        """
        decls = dict(self.decls)
        decls["someCount"] = dict(decls["icpBackend"], kind="integer")
        name, value = self.parse("someCount=25", decls)
        self.assertEqual(name, "someCount")
        self.assertIsInstance(value, int)
        self.assertNotIsInstance(value, bool)
        self.assertEqual(value, 25)

    def test_a_string_parameter_yields_its_text_verbatim(self):
        """A string is DATA: `icpBackend=1` is the string "1", not the number 1.

        This is the case `hw1:paramValueKind` exists for. Guessing the type from
        the text — "it parses as a float, so it is one" — is what made a
        categorical level need its own predicate in v1.
        """
        self.assertEqual(self.parse("icpBackend=my_icp", self.decls),
                         ("icpBackend", "my_icp"))
        name, value = self.parse("icpBackend=1", self.decls)
        self.assertIsInstance(value, str)
        self.assertEqual(value, "1")

    def test_a_generation_parameter_parses_through_the_same_function(self):
        """`--gen` is the one surviving caller (contracts §7)."""
        name, value = self.parse("brightnessGain=1.6", self.decls)
        self.assertEqual(name, "brightnessGain")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 1.6, places=12)

    def test_an_undeclared_name_is_a_hard_error_naming_the_declared_list(self):
        """contracts §5: never a silent extra, and the message is the fix.

        A typo that survives as an extra FactorSetting produces a file whose
        provenance describes a treatment that was never run — silently, and (under
        write-once) forever. `tuaHi=250` must stop the command, and the message
        must say what the legal names are, because "undeclared parameter" alone
        sends a student to grep the TBox.
        """
        with self.assertRaises(ValueError) as cm:
            self.parse("tuaHi=250", self.decls)
        message = str(cm.exception)
        self.assertIn("tuaHi", message)
        for declared in ("tauHi", "tauLo", "icpBackend", "minCoverageF",
                         "maxMapMeanL2"):
            self.assertIn(declared, message)

    def test_a_non_numeric_value_for_a_double_parameter_raises(self):
        with self.assertRaises(ValueError):
            self.parse("tauHi=high", self.decls)

    def test_a_non_integer_value_for_an_integer_parameter_raises(self):
        """"25.7 of a thing you can only have whole" is a mistyped flag.

        Synthetic kind for the same reason as `test_an_integer_parameter_yields_an_int`:
        the TBox has no integer parameter left to drive it with.
        """
        decls = dict(self.decls)
        decls["someCount"] = dict(decls["icpBackend"], kind="integer")
        with self.assertRaises(ValueError):
            self.parse("someCount=25.7", decls)

    def test_an_argument_without_an_equals_sign_raises(self):
        with self.assertRaises(ValueError):
            self.parse("tauHi", self.decls)


# =============================================================================
# 5. The status rule — contracts §4.5, implemented ONCE in §8
# =============================================================================
class QualityFactorDeclarations(unittest.TestCase):
    """`load_quality_factors` — the four links that make attribution generic (§8).

    In v3 these links carry more weight than in v2, not less: `explore`'s verdict
    section resolves failing factors and culprit settings through them IN CODE,
    which is exactly what the deleted `failure_attribution.rq` did in SPARQL
    (§10). Adding a menu factor must still require no per-factor branch anywhere.
    """

    @classmethod
    def setUpClass(cls):
        cls.factors = _require("load_quality_factors")()

    def test_the_active_factors_are_exactly_these(self):
        """Eight menu factors and two run factors.

        `ReconstructionAccuracy` and `Coverage` are what close the loop: a failing
        run is the symptom a student starts from, and it is reachable through the
        same `statusProperty` wiring as a failing frame only because they are
        ordinary factors rather than a parallel Result shape. They are also the two
        that `hw1:evaluatesFactor` may NOT name (§4.2) — not selectable, always
        evaluated.
        """
        self.assertEqual(set(self.factors), set(QUALITY_FACTORS))
        self.assertEqual(set(MENU_FACTORS) | set(RUN_FACTORS), set(QUALITY_FACTORS))
        self.assertEqual(set(MENU_FACTORS) & set(RUN_FACTORS), set())

    def test_each_factor_declares_over_status_polarity_and_threshold(self):
        """The exact dict of contracts §8, per factor.

        `polarity` is `"higher"`/`"lower"` — normalised, not the TBox individual's
        local name — because `status_for` branches on it and a rule that branched
        on an IRI would have to know the namespace.
        """
        for name, (over, status, polarity, param) in QUALITY_FACTORS.items():
            with self.subTest(factor=name):
                decl = self.factors[name]
                self.assertLessEqual({"over", "status", "polarity", "qualifiedBy"},
                                     set(decl))
                self.assertEqual(decl["over"], over)
                self.assertEqual(decl["status"], status)
                self.assertEqual(decl["polarity"], polarity)
                self.assertEqual(decl["qualifiedBy"], param)

    def test_the_status_property_name_is_the_value_property_plus_status(self):
        """contracts §4.5's naming rule, checked as data rather than trusted.

        The rule is mechanical, which is exactly why code must READ
        `hw1:statusProperty` instead of concatenating the string: a factor added
        later under a different spelling still has to be found by the verdict
        section, and a concatenating implementation silently stops matching it.
        """
        for name, decl in self.factors.items():
            with self.subTest(factor=name):
                self.assertEqual(decl["status"], decl["over"] + "Status")

    def test_every_threshold_parameter_is_a_declared_qualification_parameter(self):
        """`hw1:qualifiedBy` must land on a parameter the completeness rule fills.

        It is also the parameter §4.2 makes REQUIRED the moment its factor is
        selected — so if it named a Measurement parameter, selecting a factor would
        record a tau as its threshold and `status_for` would grade against it.
        """
        for name, decl in self.factors.items():
            with self.subTest(factor=name):
                self.assertIn(decl["qualifiedBy"], QUALIFICATION_PARAMS)


class StatusRule(unittest.TestCase):
    """`api.status_for` — the ONE implementation of the §4.5 rule (contracts §8).

    `experiment` calls it for frame and pair observables and `write_run` calls it
    for `mapMeanL2` and `coverageF`. Nobody re-derives `>=` versus `<=`; a second
    implementation is a second answer, and the one that disagrees is whichever one
    the reader did not open.
    """

    def setUp(self):
        self.status_for = _require("status_for")
        self.settings = dict(PINNED_THRESHOLDS)

    def _status(self, over, value, settings=None):
        return self.status_for(over, value, self.settings if settings is None else settings)

    def test_higher_is_better_passes_at_and_above_the_threshold(self):
        """`hw1:HigherIsBetter`: Pass iff `v >= t`, INCLUSIVE at the boundary.

        The boundary case is the one that decides whether a threshold reads as "at
        least this much" or "strictly more than this much". contracts §4.5 says
        `>=`, so a frame whose valid-depth fraction is exactly the minimum is
        usable — and a student who re-cuts the threshold TO an observed value gets
        the frames that produced it, which is the only reading that makes a
        measured re-cut mean anything.
        """
        t = self.settings["minValidTileCoverage"]
        self.assertEqual(self._status("validTileCoverage", t), api.HW1.Pass)
        self.assertEqual(self._status("validTileCoverage", t + 0.5), api.HW1.Pass)
        self.assertEqual(self._status("validTileCoverage", t - 1e-9), api.HW1.Fail)
        self.assertEqual(self._status("validTileCoverage", 0.0), api.HW1.Fail)

    def test_lower_is_better_passes_at_and_below_the_threshold(self):
        """`hw1:LowerIsBetter`: Pass iff `v <= t`, inclusive at the boundary."""
        t = self.settings["maxClipHiFraction"]
        self.assertEqual(self._status("clipHiFraction", t), api.HW1.Pass)
        self.assertEqual(self._status("clipHiFraction", 0.0), api.HW1.Pass)
        self.assertEqual(self._status("clipHiFraction", t + 1e-9), api.HW1.Fail)
        self.assertEqual(self._status("clipHiFraction", 1.0), api.HW1.Fail)

    def test_every_declared_factor_grades_through_the_same_call(self):
        """One entry point, keyed by the VALUE property's local name.

        `status_for` takes the observable, not the factor: the caller writing
        `hw1:coverageF` knows which predicate it is writing and should not also
        have to know that `hw1:Coverage` is the factor over it.
        """
        for factor, (over, _s, polarity, param) in QUALITY_FACTORS.items():
            with self.subTest(factor=factor):
                t = self.settings[param]
                passing = t + 1.0 if polarity == "higher" else max(t - 0.001, 0.0)
                failing = max(t - 1.0, -1.0) if polarity == "higher" else t + 1.0
                self.assertEqual(self._status(over, passing), api.HW1.Pass)
                self.assertEqual(self._status(over, failing), api.HW1.Fail)

    def test_the_fail_closed_sentinels_grade_fail_by_the_ordinary_rule(self):
        """contracts §4.5/§9 — no special case, and that is the point.

        A fail-closed `inf` is on the failing side of a lower-is-better factor
        for any finite threshold, and a fail-closed `0.0` is on the failing side
        of a higher-is-better one for any `t > 0`. Both fall out of `<=` and
        `>=`; an `isinf()` branch would be a second rule to keep in sync with
        the first.
        """
        for over in ("highFrequencyDepthResidual", "identityMedianDepthChange",
                     "mapMeanL2"):
            with self.subTest(sentinel="inf", over=over):
                self.assertEqual(self._status(over, INF), api.HW1.Fail)
        for over in ("jointValidDepthRatio", "coverageF"):
            with self.subTest(sentinel="0.0", over=over):
                self.assertEqual(self._status(over, 0.0), api.HW1.Fail)
                self.assertGreater(self.settings[QUALITY_FACTORS[
                    _FACTOR_BY_OVER[over]][3]], 0.0)      # t > 0, as §4.5 requires

    def test_nan_is_always_fail(self):
        """The one value that is not on either side of the comparison.

        `NaN >= t` and `NaN <= t` are both False in IEEE, so a naive `not (v > t)`
        formulation would silently PASS it. contracts §4.5 states the outcome
        directly instead of leaving it to the comparison operator, and both
        polarities are checked because only one of them is at risk.
        """
        nan = float("nan")
        self.assertEqual(self._status("validTileCoverage", nan), api.HW1.Fail)
        self.assertEqual(self._status("clipHiFraction", nan), api.HW1.Fail)
        self.assertEqual(self._status("coverageF", nan), api.HW1.Fail)
        self.assertEqual(self._status("mapMeanL2", nan), api.HW1.Fail)

    def test_a_non_finite_value_on_the_passing_side_still_passes(self):
        """contracts §4.5: "any non-finite `v` that is NOT on the passing side is
        Fail" — so one that IS on the passing side is not.

        No measurer produces `-inf`, so this is not a data case; it is the
        statement that the rule is the COMPARISON plus a NaN clause, and nothing
        else. An implementation that guarded on `math.isfinite` would fail here and
        would also be free to diverge from the contract wherever a future measurer
        returns a sentinel on the good side.
        """
        self.assertEqual(self._status("highFrequencyDepthResidual",
                                     float("-inf")), api.HW1.Pass)

    def test_the_only_status_individuals_are_pass_and_fail(self):
        """contracts §4.5: exactly two. No Marginal, no gradeRank, no third value.

        Strictness is tunable through the per-experiment threshold now, so a middle
        grade would carry no information a student could act on.
        """
        seen = {self._status("clipHiFraction", 0.0),
                self._status("clipHiFraction", 1.0)}
        self.assertEqual(seen, {api.HW1.Pass, api.HW1.Fail})
        self.assertNotEqual(api.HW1.Pass, api.HW1.Fail)

    def test_mean_value_has_no_status_and_raises(self):
        """contracts §8: a value property with no factor over it RAISES.

        `hw1:meanValue` is the band-less baseline (§4.3) and giving it a status
        would quietly promote it to a criterion — the exact thing the
        half-black-half-white fixture exists to argue against, and the exact thing
        §7.2 says carries the "beat the mean" lesson by sitting statusless next to
        the clip factors. Returning `Pass` "because nothing failed" would be worse
        than raising: it would put a green verdict next to a number nobody
        qualified.
        """
        with self.assertRaises(Exception) as cm:
            self.status_for("meanValue", 128.0, self.settings)
        self.assertNotIsInstance(cm.exception, AssertionError)
        self.assertIn("meanValue", str(cm.exception))

    def test_a_missing_threshold_raises_rather_than_guessing(self):
        """contracts §8: it cannot happen under the §4.2 completeness rule.

        Which is exactly why it must raise rather than default: if it ever DOES
        happen the completeness rule has been broken upstream — most plausibly by a
        selection-scoped vector that forgot a selected factor's threshold — and a
        guessed verdict would bake that break into every status in the file instead
        of stopping the run.
        """
        with self.assertRaises(Exception) as cm:
            self.status_for("clipHiFraction", 0.01, {})
        self.assertNotIsInstance(cm.exception, AssertionError)

    def test_the_factors_argument_is_honoured(self):
        """§8's optional `factors`, so a caller that already loaded the TBox does
        not re-parse it per observable — and so this suite can prove the lookup goes
        through `over` rather than through a hard-coded predicate list."""
        factors = {"Invented": {"over": "inventedValue",
                                "status": "inventedValueStatus",
                                "polarity": "higher",
                                "qualifiedBy": "minInvented"}}
        self.assertEqual(
            self.status_for("inventedValue", 1.0, {"minInvented": 0.5}, factors),
            api.HW1.Pass)
        self.assertEqual(
            self.status_for("inventedValue", 0.4, {"minInvented": 0.5}, factors),
            api.HW1.Fail)


class TBoxInvariants(unittest.TestCase):
    """Structural rules `ontology/hw1.ttl` must satisfy, read straight off the file.

    These are not about api.py. They are the properties the verdict section
    assumes, and a TBox that violates one produces a table that prints wrong
    numbers without erroring.
    """

    @classmethod
    def setUpClass(cls):
        cls.g = Graph()
        cls.g.parse(_ONTOLOGY_TTL, format="turtle")
        cls.factors = list(cls.g.subjects(RDF.type, api.HW1.QualityFactor))

    def test_no_quality_factor_declares_the_aggregate_as_its_status_property(self):
        """contracts §4.5, and it is an invariant of the TBox, not a guard in code.

        `hw1:qualificationStatus` is an aggregate OVER factor statuses and is
        deliberately factor-less. `explore` counts failures with the generic
        `?factor hw1:statusProperty ?sp . ?node ?sp hw1:Fail` pattern; the moment an
        "overall" factor pointed at the aggregate, every count would silently
        double-count its own components — an annotation failing one factor would be
        reported as failing two, and the ranking the student acts on would be wrong
        in a way no row makes visible.
        """
        aggregate = api.HW1[AGGREGATE_STATUS]
        for factor in self.factors:
            with self.subTest(factor=str(factor)):
                declared = set(self.g.objects(factor, api.HW1.statusProperty))
                self.assertNotIn(aggregate, declared)
                over = set(self.g.objects(factor, api.HW1.overProperty))
                self.assertNotIn(aggregate, over,
                                 "an aggregate is not an observable either")

    def test_each_factor_declares_exactly_one_of_each_link(self):
        """Single-valued and mandatory: a factor with two thresholds is two factors.

        A second `hw1:qualifiedBy` would make `status_for`'s threshold lookup
        depend on iteration order, which is the kind of defect that reproduces
        only on someone else's machine.
        """
        for factor in self.factors:
            for prop in ("overProperty", "statusProperty", "polarity", "qualifiedBy"):
                with self.subTest(factor=str(factor), link=prop):
                    self.assertEqual(len(list(self.g.objects(factor, api.HW1[prop]))), 1)

    def test_mean_value_has_no_factor_and_no_status_property(self):
        """contracts §4.3: the one observable with a value and no status.

        Checked from the other direction than `StatusRule` does — over the whole
        TBox rather than through `status_for` — because the way this breaks is a
        well-meaning `hw1:MeanBrightness` factor being added later. §9 restates the
        rejection for v3: `Brightness` and `ImageNoise` stay OFF the menu.
        """
        for factor in self.factors:
            self.assertNotIn(api.HW1.meanValue,
                             set(self.g.objects(factor, api.HW1.overProperty)))
        self.assertEqual(list(self.g.triples((None, None, api.HW1.meanValueStatus))), [])

    def test_the_six_menu_factors_and_the_two_run_factors_are_distinguishable(self):
        """contracts §4.2: `hw1:evaluatesFactor` accepts six of the eight.

        The split is not a hard-coded list in api.py's head — it has to be readable
        off the TBox, or `read_declaration` cannot reject `hw1:evaluatesFactor
        hw1:Coverage` generically and a seventh menu factor added later would need
        an edit in two places. Asserted here only as "all eight are declared";
        WHICH TBox link expresses the split is the ontology track's choice, and
        `DeclarationValidation` pins the behaviour it has to produce.
        """
        declared = {str(f).rsplit("#", 1)[-1] for f in self.factors}
        self.assertEqual(declared, set(QUALITY_FACTORS))

    def test_the_deleted_vocabulary_is_gone(self):
        """contracts §11's ledger, as absence.

        A leftover `hw1:frameStatus` declaration is not inert: it is a second,
        contradictory way to say what a frame's verdict is, and the first table
        someone writes against it prints rows that look right. v3 deletes the two
        per-node aggregates in favour of ONE (`hw1:qualificationStatus`, §4.5) and
        deletes digest identity outright (§6).
        """
        dead_classes = ("Band", "Knob", "MeasurementKnob", "CorruptionKnob",
                        "KnobSetting", "Result", "ResultMeasure")
        for local in dead_classes:
            with self.subTest(term=local):
                self.assertEqual(list(self.g.subjects(RDF.type, api.HW1[local])), [])
        dead_predicates = ("forFactor", "grade", "gradeRank", "minInclusive",
                           "maxInclusive", "knobValue", "knobLabel",
                           "knobValueKind", "defaultValue", "hasKnobSetting",
                           "setsKnob", "parameterisedBy", "hasNextFrame",
                           "fromFrame", "toFrame", "includesFrame",
                           "selectionCriterion", "hasResult",
                           # v3 (contracts §11): the two aggregates and the digest
                           "frameStatus", "pairStatus", "experimentId")
        for local in dead_predicates:
            with self.subTest(term=local):
                self.assertEqual(
                    list(self.g.triples((api.HW1[local], None, None))), [],
                    f"hw1:{local} is deleted in v3; see §11")

    def test_the_new_v3_vocabulary_is_declared(self):
        """contracts §11's other column: five terms arrived, and code imports them.

        `hw1:batchFile` is how a single-argument CLI reaches pixels;
        `hw1:evaluatesFactor` is the selection; `hw1:describesImage` is the
        annotation's link to the raster it measured; `hw1:qualificationStatus` is
        the one aggregate; `hw1:declarationDigest` is the tamper seal without which
        write-once is unenforceable (§3.1). A term used by api.py but never
        declared is a term no reader can look up.
        """
        for local in ("batchFile", "evaluatesFactor", "describesImage",
                      "qualificationStatus", "declarationDigest"):
            with self.subTest(term=local):
                self.assertTrue(
                    list(self.g.triples((api.HW1[local], None, None))),
                    f"hw1:{local} is new in v3 and must be declared; contracts §11")

    def test_exactly_two_status_individuals_are_declared(self):
        """contracts §4.5. The v1 Good/Marginal/Bad scale is deleted, not renamed."""
        statuses = {str(s).rsplit("#", 1)[-1]
                    for s in self.g.subjects(RDF.type, api.HW1.Status)}
        self.assertEqual(statuses, {"Pass", "Fail"})

    def test_the_three_setting_roles_are_declared(self):
        """Three verdicts, and the verdict section prints one of them per culprit."""
        roles = {str(s).rsplit("#", 1)[-1]
                 for s in self.g.subjects(RDF.type, api.HW1.SettingRole)}
        self.assertEqual(roles, {"GenerationSetting", "MeasurementSetting",
                                 "QualificationSetting"})

    def test_the_three_selection_modes_are_declared(self):
        """Full batch, selected segments, and the v4 mask-filtered probe."""
        modes = {str(s).rsplit("#", 1)[-1]
                 for s in self.g.subjects(RDF.type, api.HW1.SelectionMode)}
        self.assertEqual(modes, {"FullBatch", "GoodSegments", "MaskFiltered"})

    def test_parameter_defaults_carry_the_datatype_their_kind_declares(self):
        """contracts §5, "Value kinds" — the DATATYPE is pinned; the spelling is not.

        `hw1:paramValueKind "double"` means the default is an `xsd:double`, and a
        reader that got `xsd:decimal` back could not represent INF or compare
        against a double-valued setting without a cast. What is deliberately NOT
        asserted here is the LEXICAL form: contracts §5 hands serialization to
        rdflib, which normalises doubles to 7 significant digits, so `"250.0"` and
        `"2.5e+02"` are the same declaration and neither is more canonical.

        The string case is the trap §5's last paragraph names: rdflib does not
        treat a plain literal and an `xsd:string`-typed one as equal, so a default
        must be read through `.toPython()` / `str()`. Asserted here as the
        inequality itself, because a reader written the obvious way passes every
        other test in this file and then silently finds no default for
        `icpBackend`.
        """
        for name, (_role, kind, _p, _a, _d) in DECLARED_PARAMS.items():
            literal = self.g.value(api.HW1[name], api.HW1.paramDefault)
            if literal is None:
                continue
            with self.subTest(parameter=name, kind=kind):
                self.assertEqual(literal.datatype,
                                 {"double": XSD.double, "integer": XSD.integer,
                                  "string": XSD.string}[kind])
                if kind == "double":
                    self.assertTrue(math.isfinite(float(literal)))
                elif kind == "integer":
                    self.assertIsInstance(literal.toPython(), int)
                else:
                    self.assertEqual(literal.toPython(), str(literal))
                    self.assertNotEqual(literal, Literal(str(literal)),
                                        "an xsd:string literal is not equal to a "
                                        "plain one; read defaults with .toPython()")

    def test_generation_parameters_link_to_the_factors_they_corrupt(self):
        """contracts §5/§7.1: without these links the Generation verdict is
        unreachable.

        `explore` walks from a failing HighlightClipping to `brightnessGain` on the
        batch and prints role=Generation — "regenerate the data". Drop the link and
        the same walk points at `tauHi` instead, sending the student to tune a
        threshold against pixels that were broken before measurement started. v3
        makes this link LOAD-BEARING rather than a convenience: `settingForFactor`
        is single-valued now (§4.5), so `paramAffectsFactor` in the TBox is the
        only remaining path from a failing factor to a setting that merely affects
        it.
        """
        for name in sorted(GENERATION_PARAMS):
            expected = ({DECLARED_PARAMS[name][2]} | set(DECLARED_PARAMS[name][3]))
            with self.subTest(parameter=name):
                linked = {str(f).rsplit("#", 1)[-1]
                          for f in self.g.objects(api.HW1[name], api.HW1.paramPrimaryFactor)}
                linked |= {str(f).rsplit("#", 1)[-1]
                           for f in self.g.objects(api.HW1[name], api.HW1.paramAffectsFactor)}
                self.assertEqual(linked, expected)


# =============================================================================
# 6. The segment cutter — contracts §8
# =============================================================================
class SegmentCutter(unittest.TestCase):
    """`cut_contiguous_segments(usable_links)`.

    EVERY maximal chain is returned, whatever its length. The `min_length`
    argument went with `minSegmentLength` on 2026-07-31 (contracts §13): a floor
    justified by "below ~20 frames the constant-velocity prior has nothing to
    average" states a fact about the ICP backend, not about the quality of the
    pixels, and it never earned its place on floor 1 either — the selected run
    loses to the full batch at every floor tried, because the cost is the SEAM
    between two segments and a floor does not touch a seam.

    A run of k chained links still yields k+1 FRAMES; the off-by-one now shows up
    only in the lengths this cutter reports, not in a keep/drop decision.
    """

    def setUp(self):
        self.cut = _require("cut_contiguous_segments")

    def test_no_links_is_no_segments(self):
        """An empty selection is a legal outcome, not an error and not one empty
        run. contracts §7: `reconstruct` prints the reason and writes no selected
        run rather than an INF one."""
        self.assertEqual(list(self.cut([])), [])

    def test_one_link_is_two_frames(self):
        """The off-by-one, stated as a single case.

        One link joins two frames, and a cutter that counted LINKS would report
        that segment as length 1. Every length this function reports — and every
        length `explore` prints beside a segment span — is a frame count.
        """
        self.assertEqual(list(self.cut([(0, 1)])), [[0, 1]])
        self.assertEqual(len(list(self.cut([(0, 1)]))[0]), 2)

    def test_a_short_run_is_kept_because_there_is_no_floor_to_drop_it(self):
        """Three chained links are four frames, and four frames are returned.

        The regression this guards is the deleted floor coming back by accident —
        as a constant, a default argument, or a `len(s) >= 2` filter. A segment is
        short because the DATA cut it there; discarding it would silently
        substitute a judgement about the reconstructor for a judgement about the
        pixels.
        """
        links = [(0, 1), (1, 2), (2, 3)]
        self.assertEqual(list(self.cut(links)), [[0, 1, 2, 3]])
        self.assertEqual(list(self.cut([(0, 1)])), [[0, 1]])

    def test_a_mid_sequence_gap_splits_the_run(self):
        """One unusable link in the middle cuts the capture in two.

        Never drop a frame from INSIDE a segment: dropping frame k creates a pair
        (k-1, k+1) whose stored observables no longer describe it and whose motion
        roughly doubles, which breaks the constant-velocity initialisation the
        reconstruction depends on. So the cut falls BETWEEN segments, and a segment
        shorter than `min_length` is dropped whole.
        """
        links = [(0, 1), (1, 2), (3, 4), (4, 5)]        # (2,3) is not usable
        self.assertEqual(list(self.cut(links)), [[0, 1, 2], [3, 4, 5]])

    def test_the_chain_rule_is_j_equals_i_not_j_equals_i_plus_one(self):
        """contracts §8: links chain where the `j` of one equals the `i` of the next.

        The gap batch has adjacent frames 1 and 3 — stem 2 exists in `rgb/` only,
        so it is not a frame of the batch at all and (1,3) is a perfectly ordinary
        consecutive pair. A cutter that tested `j + 1 == i` would split every
        capture at every dropped frame, silently, and blame the trajectory.
        """
        self.assertEqual(list(self.cut([(0, 1), (1, 3), (3, 4)])),
                         [[0, 1, 3, 4]])

    def test_the_selection_fixture_cuts_where_the_arithmetic_says(self):
        """The end-to-end shape, on the batch whose usable-link set is forced.

        Ten frames, all four frame factors Pass throughout; every pair's median
        depth difference is exactly 0 except the one straddling the 0.5 m jump.
        Segments [0..7] and [8,9] — and with no floor, BOTH are reconstructed.
        """
        self.assertEqual(list(self.cut(SEL_USABLE_LINKS)),
                         [SEL_EXPECTED_SEGMENT, [SEL_SPLIT, SEL_SPLIT + 1]])

    def test_links_arriving_out_of_order_still_chain_ascending(self):
        """An RDF graph is a SET and nothing about it is ordered.

        The caller may hand these over in any order at all; the cutter sorts. A
        reconstruction fed frames out of order would integrate the trajectory
        backwards through part of the capture, which costs accuracy and never
        errors.
        """
        scrambled = [(2, 3), (0, 1), (1, 2)]
        self.assertEqual(list(self.cut(scrambled)), [[0, 1, 2, 3]])

    def test_segments_are_ascending_int_lists(self):
        """The type `reconstruct.py` indexes frames with. A str stem would
        round-trip through every IRI helper and then sort `10` before `8`."""
        segments = list(self.cut([(8, 9), (9, 10), (10, 11)]))
        self.assertEqual(segments, [[8, 9, 10, 11]])
        for value in segments[0]:
            self.assertIsInstance(value, int)


# =============================================================================
# 7. `batch2ttl` — STRUCTURE ONLY, and most assertions are about ABSENCE
# =============================================================================
class BatchGraph(unittest.TestCase):
    """contracts §4.1 — the batch file changes only when the PIXELS change.

    Unchanged from v2 by design (plan.md §3: "Measurers and `batch2ttl`
    untouched"). `batch2ttl` is now an optional sidecar: GenerationSettings
    recorded here are what `explore`'s verdict section reads to print the
    Generation fix (§7.1). `declare` / `experiment` name the capture directory
    itself in `hw1:batchFile`.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_dir = _batch_dir(BATCH_DIRNAME)
        cls.path = _run_batch2ttl(BATCH_DIRNAME)     # default <data_dir>/batch.ttl
        _BATCH_TTL_DONE.add(BATCH_DIRNAME)
        cls.g = _graph_of(cls.path)
        cls.b = api.batch_iri(BATCH)

    def test_the_default_output_path_is_beside_the_pixels(self):
        """contracts §3: `<data_dir>/batch.ttl`, one per capture dir.

        Beside the pixels rather than in a central directory because it describes
        them: copy the capture and its (optional) structure sidecar travels with
        it. `declare` / `experiment` no longer require this file.
        """
        self.assertEqual(self.path, os.path.join(self.data_dir, "batch.ttl"))
        self.assertTrue(os.path.isfile(self.path))

    def test_the_batch_node_carries_name_path_and_floor(self):
        """The three things a batch file is allowed to say about the capture.

        `batchName` is the join key every frame IRI is built from; `batchPath`
        records the capture directory the sidecar describes. `declare` /
        `experiment` now name that directory in `hw1:batchFile` directly.
        """
        self.assertIn((self.b, RDF.type, api.HW1.Batch), self.g)
        self.assertEqual(str(self.g.value(self.b, api.HW1.batchName)), BATCH)
        self.assertEqual(str(self.g.value(self.b, api.HW1.batchPath)), self.data_dir)
        self.assertEqual(int(self.g.value(self.b, api.HW1.floor)), FLOOR)

    def test_every_frame_carries_structure_and_only_structure(self):
        """One Frame per paired stem, each with an rgb and a depth node (§4.1)."""
        frames = set(self.g.objects(self.b, api.HW1.hasFrame))
        self.assertEqual({str(f) for f in frames},
                         {str(api.frame_iri(BATCH, stem)) for stem in MAIN_STEMS})
        for stem in MAIN_STEMS:
            with self.subTest(frame=stem):
                f = api.frame_iri(BATCH, stem)
                self.assertIn((f, RDF.type, api.HW1.Frame), self.g)
                self.assertEqual(int(self.g.value(f, api.HW1.frameIndex)), stem)
                rc = self.g.value(f, api.HW1.hasRGBImage)
                dc = self.g.value(f, api.HW1.hasDepthImage)
                self.assertEqual(rc, api.component_iri(BATCH, stem, "rgb"))
                self.assertEqual(dc, api.component_iri(BATCH, stem, "depth"))
                self.assertIn((rc, RDF.type, api.HW1.RGBImage), self.g)
                self.assertIn((dc, RDF.type, api.HW1.DepthImage), self.g)
                self.assertEqual(str(self.g.value(rc, api.SCHEMA.contentUrl)),
                                 os.path.join(self.data_dir, "rgb", f"{stem}.png"))
                self.assertEqual(str(self.g.value(dc, api.SCHEMA.contentUrl)),
                                 os.path.join(self.data_dir, "depth", f"{stem}.png"))

    def test_an_image_node_carries_a_content_url_and_nothing_else(self):
        """contracts §4.3, stated from the batch side.

        In v1 the image node WAS the observation and carried `clipHiFraction` and
        friends; that worked only because a named graph scoped it. With one graph,
        two experiments would write two contradictory triples onto this subject.
        v3 points at it from the annotation instead — `hw1:describesImage` — which
        gives a query the same join with none of the collision.
        """
        for stem in MAIN_STEMS:
            for kind in MODALITIES:
                node = api.component_iri(BATCH, stem, kind)
                with self.subTest(frame=stem, kind=kind):
                    self.assertEqual({str(p) for p in self.g.predicates(node, None)},
                                     {str(RDF.type), str(api.SCHEMA.contentUrl)})

    def test_the_batch_file_holds_zero_measured_values_and_zero_statuses(self):
        """The single most important assertion in this class.

        A measured value is only meaningful together with the settings it was
        measured under, so it belongs to an experiment file; let one leak back here
        and re-measuring under a second declaration starts rewriting the batch
        file, which is the thing the structure/measurement split exists to prevent.
        Counted, not spot-checked, and counted for the STATUS predicates too — a
        baked verdict is even more context-dependent than the number it grades.
        """
        for prop in MEASURED_PREDICATES + STATUS_PREDICATES:
            with self.subTest(predicate=prop):
                self.assertEqual(
                    len(list(self.g.triples((None, api.HW1[prop], None)))), 0,
                    f"hw1:{prop} is a measurement; it belongs in an experiment file")

    def test_there_are_no_pair_skeletons_and_no_successor_chain(self):
        """Pairs are minted per EXPERIMENT (contracts §4.3), always, even when no
        pair factor is selected — so caching them here would buy nothing and force
        every experiment to share a subject. `hasNextFrame` went with them:
        ordering is `hw1:frameIndex` and `hw1:pairIndex`."""
        self.assertEqual(list(self.g.subjects(RDF.type, api.HW1.FramePair)), [])
        for prop in ("hasNextFrame", "hasPreviousFrame", "fromFrame", "toFrame",
                     "sourceFrame", "targetFrame", "pairIndex"):
            with self.subTest(predicate=prop):
                self.assertEqual(list(self.g.triples((None, api.HW1[prop], None))), [])

    def test_no_tbox_is_bundled_into_the_batch_file(self):
        """contracts §3: the TBox is never bundled into an ABox file.

        Copying class and parameter declarations into every capture directory would
        give the repo N ontologies that drift apart — and `explore` resolves factors
        through the TBox generically (§10), so it would resolve them differently
        per capture.
        """
        self.assertEqual(list(self.g.subjects(RDF.type, RDFS.Class)), [])
        self.assertEqual(list(self.g.subjects(RDF.type, api.HW1.QualityFactor)), [])
        self.assertEqual(list(self.g.subjects(RDF.type, api.HW1.Parameter)), [])
        self.assertEqual(len(list(self.g.triples((None, api.HW1.paramRole, None)))), 0)

    def test_generation_settings_are_never_auto_filled(self):
        """contracts §5: recorded only when actually given.

        This batch was written with no `--gen`, so it asserts nothing about how its
        pixels were produced — which is the honest reading, since nothing in this
        pipeline touched them. A default of "gain 1.0, sigma 0.0" would look
        harmless and would be a claim no one verified. It would also put a spurious
        Generation culprit in every verdict section (§7.1).
        """
        self.assertEqual(list(self.g.objects(self.b, api.HW1.hasGenerationSetting)), [])
        self.assertEqual(list(self.g.subjects(RDF.type, api.HW1.FactorSetting)), [])

    def test_a_generation_setting_records_role_parameter_value_and_factor(self):
        """contracts §4.1/§4.2 — the node shape `explore`'s verdict section reads.

        RESOLVED (owner, 2026-07-30): §4.5's single-valued rule is GLOBAL, batch
        settings included — v2 emitted one triple per affected factor here too, and
        `batch2ttl` was outside plan.md §3's tracks, which is the only reason the
        two disagreed. So this pins the primary factor as the SOLE value.
        `brightnessGain`'s blame for a ShadowClipping failure is not lost by the
        deletion: `explore` reaches it through `hw1:paramAffectsFactor` in the TBox
        (`TBoxInvariants` pins that link) rather than off the setting node.
        """
        path = _run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("gen_batch.ttl"),
                              gen=["brightnessGain=1.6"])
        g = _graph_of(path)
        batch = api.batch_iri(BATCH_DEGEN)
        node = _require("generation_setting_iri")(BATCH_DEGEN, "brightnessGain")

        self.assertIn((batch, api.HW1.hasGenerationSetting, node), g)
        self.assertIn((node, RDF.type, api.HW1.FactorSetting), g)
        self.assertEqual(g.value(node, api.HW1.settingParameter), api.HW1.brightnessGain)
        self.assertEqual(g.value(node, api.HW1.settingRole), api.HW1.GenerationSetting)
        self.assertEqual(list(g.objects(node, api.HW1.settingForFactor)),
                         [api.HW1.HighlightClipping],
                         "§4.5: single-valued on batch settings too — "
                         "ShadowClipping is reached via hw1:paramAffectsFactor")

        value = g.value(node, api.HW1.settingValue)
        self.assertEqual(value.datatype, XSD.double)
        self.assertAlmostEqual(float(value), 1.6, delta=_serializer_delta(1.6))

    def test_a_non_generation_parameter_is_not_a_generation_setting(self):
        """`--gen tauHi=250` must stop the command (contracts §4.1/§5).

        A tau is a MeasurementSetting: it describes how pixels are SCORED and
        belongs to a declaration. Recorded on a batch it would claim the capture
        was produced at tau 250, and the verdict section would print
        role=Generation — "regenerate the data" — for a number that changes nothing
        about the data at all. The wrong verdict is worse than no verdict.
        """
        for arg in ("tauHi=250", "maxClipHiFraction=0.05"):
            with self.subTest(gen=arg):
                with self.assertRaises(_REJECTED):
                    _run_batch2ttl(BATCH_DEGEN_DIRNAME,
                                   out=_out_path("gen_bad_role.ttl"), gen=[arg])

    def test_an_undeclared_generation_name_is_a_hard_error(self):
        """Same rule as `hw1:settingParameter` (contracts §5): the list is closed."""
        with self.assertRaises(_REJECTED):
            _run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("gen_bad_name.ttl"),
                           gen=["brihtnessGain=1.6"])

    def test_derived_from_links_one_batch_to_another(self):
        """contracts §4.1: `prov:wasDerivedFrom` survives BETWEEN BATCHES only.

        A corrupted capture is derived from the clean one it was generated from,
        and that edge is how `explore`'s comparison view pairs them. It is the one
        surviving use of `prov:` — there are still no derived EXPERIMENTS (§6),
        because baseline and selected are two Run nodes inside one experiment.
        """
        path = _run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("derived.ttl"),
                              derived_from=BATCH)
        g = _graph_of(path)
        self.assertEqual(g.value(api.batch_iri(BATCH_DEGEN), api.PROV.wasDerivedFrom),
                         api.batch_iri(BATCH))

    def test_a_stem_missing_from_one_subdir_is_not_a_frame(self):
        """The case that silently changes the answer, pinned to an explicit set.

        `rgb/` holds stems 0,1,2,3 and `depth/` holds only 0,1,3. A frame needs
        BOTH, so frame 2 is not a frame of this batch — and the experiment layer
        will pair 1 with 3, spanning a real two-step motion whose observables
        honestly report the larger displacement. `explore`'s batch view is required
        to CALL THE GAP OUT (§7.1) precisely because the graph is otherwise silent
        about it.
        """
        path = _run_batch2ttl(BATCH_GAP_DIRNAME, out=_out_path("gap_batch.ttl"))
        g = _graph_of(path)
        b = api.batch_iri(BATCH_GAP)
        frames = {int(g.value(f, api.HW1.frameIndex))
                  for f in g.objects(b, api.HW1.hasFrame)}
        self.assertEqual(frames, set(GAP_PAIRED_STEMS))
        self.assertNotIn(2, frames)      # present in rgb/ only, so not a frame

    def test_structure_is_built_without_opening_a_single_pixel(self):
        """A batch of unmeasurable frames still has a perfectly good batch file.

        The degenerate batch's frame 1 carries no depth at all, so every depth
        observable on it is inf or 0. The batch file does not care and must not: it
        lists two frames and no numbers. This is the behavioural statement behind
        "this command opens no PNG" — the reason a second declaration over the same
        capture does not rewrite the batch.
        """
        path = _run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("degen_batch.ttl"))
        g = _graph_of(path)
        self.assertEqual(len(list(g.objects(api.batch_iri(BATCH_DEGEN),
                                            api.HW1.hasFrame))), 2)
        for prop in MEASURED_PREDICATES:
            self.assertEqual(len(list(g.triples((None, api.HW1[prop], None)))), 0)

    def test_the_floor_qualifies_the_batch_name_everywhere(self):
        """floor1_x and floor2_x are two captures, two names, two sets of frame
        IRIs — so a declaration that names the wrong floor cannot silently join
        against the right one. §4.2 turns that into a hard error: the batch IRI
        parsed out of `hw1:batchFile` must equal the declared `hw1:onBatch`."""
        path = _run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("floor2.ttl"), floor=2)
        g = _graph_of(path)
        other = api.batch_iri(f"floor2_{BATCH_DEGEN_DIRNAME}")
        self.assertEqual(str(g.value(other, api.HW1.batchName)),
                         f"floor2_{BATCH_DEGEN_DIRNAME}")
        self.assertEqual(int(g.value(other, api.HW1.floor)), 2)


class BatchGraphFlags(unittest.TestCase):
    """Flags contracts §7 freezes ON and OFF for `batch2ttl`."""

    def test_no_ontology_is_not_a_flag_any_more(self):
        """contracts §3: nothing bundles a TBox, so nothing needs a flag to stop
        it. Accepting a dead flag silently is how a script keeps passing it for a
        year after it stopped meaning anything."""
        with self.assertRaises(_REJECTED):
            _main(["batch2ttl", "--data-dir", _batch_dir(BATCH_DEGEN_DIRNAME),
                   "--floor", str(FLOOR), "--no-ontology",
                   "--out", _out_path("flag_no_ontology.ttl")])


# =============================================================================
# 8. The DECLARATION — contracts §4.2, the student's artefact
#
# The selections every fixture below declares. Each one is chosen to make some
# clause of §4.2/§4.3 observable: DEPTH_ONLY records no `tauHi` and mints no rgb
# annotation, PAIR_ONLY leaves every frame without an annotation (and therefore
# vacuously usable), FRAME_ONLY leaves every pair vacuously Pass, and ALL_MENU is
# v2's total pass — the case v3 has to keep working unchanged.
# =============================================================================
DEPTH_ONLY = ("HighFrequencyDepthResidual", "ValidTileCoverage")
RGB_ONLY = ("HighlightClipping", "ShadowClipping")
FRAME_ONLY = FRAME_FACTORS
PAIR_ONLY = ("IdentityMedianDepthChange",)
ALL_MENU = MENU_FACTORS


class DeclarationValidation(unittest.TestCase):
    """One test per §4.2 rule, each on the RED path (contracts §4.2/§8.1).

    A declaration is the one file in this system a human writes by hand, so every
    one of these is a mistake a student will actually make. The contract asks for
    two things of each: a HARD ERROR, and a message that NAMES THE OFFENDING
    TRIPLE — because "invalid declaration" sends a student to read api.py, and the
    parameter's local name sends them to the line they typed.

    Both entry points are checked. `read_declaration` (§8.1) is where the message
    lives; `experiment DECLARATION.ttl` (§7) is where a student meets it, and it
    must additionally leave the file ALONE — a rejected declaration that came back
    with a machine section appended would be assessed-but-invalid, and write-once
    (§3.1) would then refuse to ever fix it.
    """

    def _rejects(self, path, *needles):
        """Assert both entry points reject `path`, and the message names the triple."""
        message = _error_message(self, _require("read_declaration"), path)
        for needle in needles:
            self.assertIn(needle, message,
                          f"§4.2 requires the error to name the "
                          f"offending triple; {needle!r} is missing from:\n{message}")
        before = _bytes_of(path)
        _cli_failure(self, ["experiment", path])
        self.assertEqual(_bytes_of(path), before,
                         "a rejected declaration was written to; §3.1 "
                         "makes assessment write-once, so a half-assessed file can "
                         "never be repaired")
        self.assertIsNone(_marker_offset(path),
                          "a rejected declaration got a machine marker")
        return message

    # ---- exactly one Experiment, and its IRI tail is the file stem -------
    def test_a_declaration_with_no_experiment_subject_is_an_error(self):
        """§4.2, first rule. Zero means the file is not a declaration at all."""
        path = _write_declaration("decl_zero", (
            "@prefix hw1: <http://taica.course/hw1/ontology#> .\n"
            f'<{api.NS}experiment/decl_zero> hw1:batchFile "nowhere/batch.ttl" .\n'))
        self._rejects(path, "Experiment")

    def test_two_experiment_subjects_are_an_error(self):
        """"The first one" is not an answer.

        Two Experiment nodes in one declaration means two designs were pasted
        together, and every downstream answer — which selection, which thresholds,
        whose annotations — becomes a coin flip that nothing reports.
        """
        path = _declare("decl_two", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        extra_nodes=f"<{api.NS}experiment/decl_two_twin> "
                                    f"a hw1:Experiment .")
        self._rejects(path, "Experiment")

    def test_an_iri_tail_that_is_not_the_file_stem_is_an_error(self):
        """contracts §2: the tail MUST equal the stem; `experiment` never picks one.

        Identity is the name in v3 (§6), and the name has two spellings — the
        filename and the IRI. Letting them disagree means `hw1/experiments/` no
        longer indexes the notebook: `explore a.ttl b.ttl` would print one name and
        the file system another, and nothing would say which is the experiment.
        """
        path = _declare("decl_tail", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        iri_tail="a_different_name")
        self._rejects(path, "a_different_name")

    # ---- batchFile ------------------------------------------------------
    def test_a_missing_batch_file_is_an_error(self):
        """§4.2: `hw1:batchFile` is how a one-argument CLI reaches pixels.

        Without it there is no path from the declaration to `rgb/` and `depth/`,
        and `hw1:onBatch` alone names a batch IRI that nothing can resolve to a
        directory.
        """
        path = _declare("decl_nobatchfile", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        omit_batch_file=True)
        self._rejects(path, "batchFile")

    def test_an_unresolvable_batch_file_is_an_error(self):
        """A path that does not exist is a typo, not an empty batch.

        Measuring nothing and writing an experiment with zero annotations would be
        a perfectly parseable file describing a capture that was never opened.
        """
        missing = os.path.join(_TMP, "not_a_capture", "batch.ttl")
        path = _declare("decl_badbatchfile", BATCH_DIRNAME, DEPTH_ONLY,
                        pinned=False, batch_ttl=missing)
        self._rejects(path, "batch.ttl")

    def test_a_batch_file_that_disagrees_with_on_batch_is_an_error(self):
        """§4.2: "the 'measured capture A, wrote into capture B' bug at declaration
        time".

        The batch IRI derived from the capture directory in `hw1:batchFile` must
        EQUAL `hw1:onBatch`. Nothing downstream can catch this: the annotations
        would carry frame IRIs of the batch that was measured while the experiment
        claimed the other one, and both files parse, load and join cleanly.
        """
        path = _declare("decl_batchmismatch", BATCH_DIRNAME, DEPTH_ONLY,
                        pinned=False, batch_name=BATCH_DEGEN)
        self._rejects(path, BATCH_DEGEN)

    # ---- evaluatesFactor ------------------------------------------------
    def test_an_empty_selection_is_an_error(self):
        """§4.2: 1..6 factors. Zero is not "measure everything" and not "measure
        nothing" — it is a design that says nothing, and the file it would produce
        would carry annotations with no observables and a vacuous Pass on every
        one of them."""
        path = _declare("decl_noselection", BATCH_DIRNAME, None, pinned=False)
        self._rejects(path, "evaluatesFactor")

    def test_a_factor_off_the_menu_is_an_error(self):
        """The menu is closed at six (§4.2, §9).

        `hw1:MeanBrightness` is the exact mistake the menu exists to prevent —
        `hw1:meanValue` has no polarity and no threshold, so "selecting" it would
        ask for a status nothing can compute.
        """
        path = _declare("decl_offmenu", BATCH_DIRNAME,
                        ("HighFrequencyDepthResidual", "MeanBrightness"), pinned=False)
        self._rejects(path, "MeanBrightness")

    def test_a_run_factor_in_the_selection_is_an_error(self):
        """§4.2: `ReconstructionAccuracy` and `Coverage` are NOT selectable.

        They are always evaluated — selection scopes INPUT-QUALITY factors only.
        Naming one is a category error worth stopping on rather than ignoring: the
        student thinks they have turned the run verdict on or off, and they have
        done neither.
        """
        for factor in RUN_FACTORS:
            with self.subTest(factor=factor):
                path = _declare(f"decl_runfactor_{factor}", BATCH_DIRNAME,
                                ("HighFrequencyDepthResidual", factor), pinned=False)
                self._rejects(path, factor)

    # ---- FactorSetting nodes --------------------------------------------
    def test_a_setting_for_an_undeclared_parameter_is_an_error(self):
        """§4.2: `settingParameter` must be TBox-declared.

        `hw1:maxClipHiFractoin` parses, validates as an IRI and means nothing. Left
        alone it would be recorded as a level of a treatment that never happened —
        and under write-once that file is the permanent record.
        """
        path = _declare("decl_badparam", BATCH_DIRNAME, RGB_ONLY, pinned=False,
                        settings=[("maxClipHiFractoin", 0.05, "QualificationSetting",
                                   "HighlightClipping")])
        self._rejects(path, "maxClipHiFractoin")

    def test_a_setting_whose_role_is_not_the_parameters_role_is_an_error(self):
        """§4.2: `settingRole` must EQUAL `hw1:paramRole`.

        The role is not decoration: it is the FIX the verdict section prints
        (§7.1). A threshold recorded as a MeasurementSetting would tell the student
        to re-measure when the number they need to change is a threshold.
        """
        path = _declare("decl_badrole", BATCH_DIRNAME, RGB_ONLY, pinned=False,
                        settings=[("maxClipHiFraction", 0.05, "MeasurementSetting")])
        self._rejects(path, "maxClipHiFraction")

    def test_a_setting_for_the_wrong_factor_is_an_error(self):
        """§4.2/§4.5: `settingForFactor` must equal `hw1:paramPrimaryFactor`, and
        it is SINGLE-VALUED in v3.

        v2 duplicated the affected factors onto the node; v3 deletes that and has
        readers traverse `hw1:paramAffectsFactor` in the TBox instead. So the one
        triple that remains has exactly one legal object, and a declaration naming
        another factor is claiming a link the TBox does not have.
        """
        path = _declare("decl_badfactor", BATCH_DIRNAME, RGB_ONLY, pinned=False,
                        settings=[("maxClipHiFraction", 0.05, None,
                                   "ShadowClipping")])
        self._rejects(path, "ShadowClipping")

    def test_a_mistyped_setting_value_is_an_error(self):
        """§4.2: `settingValue` typed per `hw1:paramValueKind`.

        A threshold spelled as a string is not a threshold: nothing can compare a
        clip fraction against `"tight"`, and a threshold spelled as a string is a
        Pass line that never compares against anything. Deliberately NOT pinned here:
        whether an UNTYPED literal ("0.05" with no `^^`) is coerced or rejected —
        contracts says "typed per the kind" and stops there, so both readings are
        defensible and neither is worth a red test.
        """
        cases = [("mistyped_string", "maxClipHiFraction", '"tight"^^xsd:string',
                  RGB_ONLY, "maxClipHiFraction"),
                 ("mistyped_string_for_depth", "maxHighFrequencyDepthResidual",
                  '"0.98"^^xsd:string', DEPTH_ONLY,
                  "maxHighFrequencyDepthResidual")]
        for stem, param, text, selection, needle in cases:
            with self.subTest(case=stem):
                path = _declare(f"decl_{stem}", BATCH_DIRNAME, selection,
                                pinned=False,
                                settings=[(param, None, None, None, text)])
                self._rejects(path, needle)

    def test_a_setting_that_touches_no_selected_factor_is_an_error(self):
        """§4.2's last setting rule: "it would record a level that touched nothing".

        A depth-only experiment that sets `tauHi` records a highlight-clipping tau
        it never applied to a pixel. The recorded vector is the treatment; a level
        in it that changed no measurement makes the notebook lie about what was
        varied — which is the one thing an OFAT sweep cannot survive.
        """
        path = _declare("decl_untouched", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        settings=[("tauHi", 250.0)])
        self._rejects(path, "tauHi")

    def test_a_run_factor_parameter_is_legal_under_any_selection(self):
        """... and the exemption in the same clause, so the rule above is not a ban.

        `icpBackend`, `maxMapMeanL2` and `minCoverageF` are REQUIRED on every
        experiment (§4.2 completeness), so setting one can never be a level that
        touched nothing — the run factors are always evaluated.
        """
        path = _emit_experiment(
            "decl_runparams", BATCH_DEGEN_DIRNAME, DEPTH_ONLY,
            settings=[("icpBackend", "open3d")])
        settings = _require("read_experiment")(path)["settings"]
        self.assertEqual(settings["icpBackend"], "open3d")
        self.assertAlmostEqual(settings["maxMapMeanL2"],
                               PINNED_THRESHOLDS["maxMapMeanL2"], places=12)

    # ---- the whitelist: verdicts are computed, never declared ------------
    def test_a_declared_annotation_is_an_error(self):
        """§4.2's whitelist, and the sentence it ends with.

        A student who writes their own `hw1:FrameAnnotation` has written down the
        answer. The whole loop is "declare the design, let the tooling compute the
        verdict"; a declaration that carries observations makes the file's contents
        unfalsifiable — nothing downstream can tell a measured value from a typed
        one.
        """
        node = f"{api.NS}experiment/decl_annotation/annotation/0/rgb"
        path = _declare("decl_annotation", BATCH_DIRNAME, RGB_ONLY, pinned=False,
                        extra_body=[f"    hw1:producesAnnotation <{node}>"],
                        extra_nodes=(f"<{node}> a hw1:FrameAnnotation ;\n"
                                     f"    hw1:frameIndex 0 ;\n"
                                     f"    hw1:describesImage "
                                     f"<{api.NS}batch/{BATCH}/frame/0/rgb> ."))
        self._rejects(path, "annotation/0/rgb")

    def test_a_declared_status_is_an_error(self):
        """The same rule at its sharpest: a declared PASS.

        `hw1:qualificationStatus hw1:Pass` written by hand is a green verdict
        nobody measured. Under write-once it is also permanent.
        """
        node = f"{api.NS}experiment/decl_status/pair/0_1"
        path = _declare("decl_status", BATCH_DIRNAME, PAIR_ONLY, pinned=False,
                        extra_body=[f"    hw1:producesPair <{node}>"],
                        extra_nodes=(f"<{node}> a hw1:FramePair ;\n"
                                     f"    hw1:qualificationStatus hw1:Pass ."))
        # The needle is the offending NODE rather than the predicate: a graph is a
        # set, so which of the node's triples the validator reaches first is not
        # determined by anything, and §4.2 asks for one offending triple named —
        # not for a particular one.
        self._rejects(path, "pair/0_1")

    def test_a_declared_observable_is_an_error(self):
        """An observable on the Experiment node itself — the shortest version.

        `hw1:meanValue` is the one observable with no status, which makes it the
        most innocent-looking thing to type and exactly as forbidden as the rest.
        """
        path = _declare("decl_observable", BATCH_DIRNAME, RGB_ONLY, pinned=False,
                        extra_body=['    hw1:meanValue "128.0"^^xsd:double'])
        self._rejects(path, "meanValue")

    def test_a_declared_run_is_an_error(self):
        """A run is an OUTCOME (§3.1), and the only writer of one is `write_run`.

        Declaring it would let a student assert the reconstruction score of a
        reconstruction that never ran — while `write_run`'s idempotence would
        later replace it with the real number and leave no trace of the claim.
        """
        node = f"{api.NS}experiment/decl_run/run/baseline"
        path = _declare("decl_run", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        extra_body=[f"    hw1:hasRun <{node}>"],
                        extra_nodes=(f"<{node}> a hw1:ReconstructionRun ;\n"
                                     f"    hw1:selectionMode hw1:FullBatch ;\n"
                                     f'    hw1:mapMeanL2 "0.10"^^xsd:double .'))
        self._rejects(path, "run/baseline")

    def test_a_declared_digest_is_an_error(self):
        """The seal is the machine's (§3.1), and a student-supplied one would be
        a forged tamper seal — the one triple whose whole purpose is that nobody
        below the marker can have written it."""
        path = _declare("decl_digest", BATCH_DIRNAME, DEPTH_ONLY, pinned=False,
                        extra_body=['    hw1:declarationDigest "00"'])
        self._rejects(path, "declarationDigest")

    # ---- the positive control -------------------------------------------
    def test_the_smallest_legal_declaration_is_accepted(self):
        """Without this the class above proves only that everything is rejected.

        One Experiment, one batch, one factor, no settings, no label. Everything
        else — the whole required parameter set — is filled from `hw1:paramDefault`
        below the marker (§4.2).
        """
        path = _declare("decl_minimal", BATCH_DEGEN_DIRNAME,
                        ("HighFrequencyDepthResidual",), pinned=False)
        info = _require("read_declaration")(path)
        self.assertEqual(info["exp_name"], "decl_minimal")
        self.assertEqual(list(info["selected"]),
                         ["HighFrequencyDepthResidual"])
        self.assertEqual(info["given"], {})


class DeclarationReader(unittest.TestCase):
    """`read_declaration` — contracts §8.1: validates, never writes.

    `explore` view 2 and `experiment` both consume this, and §8.1 says in so many
    words that neither re-implements a rule. That is what stops the declaration
    from having two dialects: one the tool measures and one the tool prints.
    """

    @classmethod
    def setUpClass(cls):
        cls.read = staticmethod(_require("read_declaration"))
        cls.path = _declare("read_decl", BATCH_SEL_DIRNAME, ALL_MENU,
                            label="every factor, thresholds pinned by hand")
        cls.depth_path = _declare("read_decl_depth", BATCH_SEL_DIRNAME, DEPTH_ONLY)

    def test_it_returns_every_documented_key(self):
        """contracts §8.1's dict, key by key."""
        info = self.read(self.path)
        self.assertLessEqual({"exp_iri", "exp_name", "batch_file", "batch_iri",
                              "batch_name", "batch_path", "selected", "given",
                              "required"}, set(info))
        self.assertEqual(info["exp_name"], "read_decl")
        self.assertEqual(info["exp_iri"], api.experiment_iri("read_decl"))
        self.assertEqual(info["batch_name"], BATCH_SEL)
        self.assertEqual(info["batch_iri"], api.batch_iri(BATCH_SEL))
        self.assertEqual(os.path.realpath(info["batch_path"]),
                         os.path.realpath(_batch_dir(BATCH_SEL_DIRNAME)))

    def test_the_selection_comes_back_as_factor_local_names(self):
        """`[factor_local, …]`, at least one, menu factors only (§8.1)."""
        selected = self.read(self.path)["selected"]
        self.assertEqual(set(selected), set(ALL_MENU))
        self.assertLessEqual(set(selected), set(MENU_FACTORS))
        self.assertTrue(all(isinstance(f, str) for f in selected))

    def test_given_holds_the_student_settings_and_nothing_else(self):
        """`given` is what the student typed; `required` is what will be recorded.

        The split is the whole point of §4.2's completeness rule — `explore` view 2
        prints "given" or "will be defaulted" per parameter from exactly these two
        dicts, before a single pixel has been opened.
        """
        info = self.read(self.path)
        self.assertEqual(set(info["given"]),
                         {name for name, _v in _pinned_settings(ALL_MENU)})
        for name, value in _pinned_settings(ALL_MENU):
            with self.subTest(parameter=name):
                self.assertAlmostEqual(float(info["given"][name]), value, places=12)

    def test_required_is_the_selection_scoped_total_vector(self):
        """contracts §4.2's completeness rule, against a transcription of it.

        A depth-only experiment requires no `tauHi`, and
        requires all four run-factor parameters anyway. `required` is given ∪
        defaults, so it is total over the selection — which is what makes every
        recorded status reproducible from the file alone.
        """
        info = self.read(self.depth_path)
        self.assertEqual(set(info["required"]), _required_params(DEPTH_ONLY))
        self.assertNotIn("tauHi", info["required"])
        for name in RUN_FACTOR_PARAMS:
            self.assertIn(name, info["required"])
        self.assertLessEqual(set(info["given"]), set(info["required"]))

    def test_a_student_setting_may_be_an_iri_node(self):
        """contracts §2: student FactorSettings "may be blank nodes or any IRI".

        Only MACHINE-minted settings are bound to the scheme. A validator that
        insisted on blank nodes — or on the scheme — would reject a perfectly
        legal declaration for a reason no clause supports.
        """
        node = "https://example.org/my-settings/tight-highlights"
        path = _declare("read_decl_iri", BATCH_DEGEN_DIRNAME, RGB_ONLY, pinned=False,
                        extra_body=[f"    hw1:hasFactorSetting <{node}>"],
                        extra_nodes=(f"<{node}> a hw1:FactorSetting ;\n"
                                     f"    hw1:settingParameter hw1:maxClipHiFraction ;\n"
                                     f"    hw1:settingRole hw1:QualificationSetting ;\n"
                                     f"    hw1:settingForFactor hw1:HighlightClipping ;\n"
                                     f'    hw1:settingValue "0.03"^^xsd:double .'))
        info = self.read(path)
        self.assertAlmostEqual(float(info["given"]["maxClipHiFraction"]), 0.03,
                               places=12)

    def test_reading_a_declaration_writes_nothing(self):
        """§8.1: "validates, never writes".

        `explore` calls it, and `explore` is read-only (§7.1). A reader that
        normalised the file it read would silently break the seal of every file it
        was pointed at — including, one day, an assessed one.
        """
        before = _snapshot([self.path])
        self.read(self.path)
        self.assertEqual(_snapshot([self.path]), before)
        self.assertIsNone(_marker_offset(self.path))


# =============================================================================
# 9. The two-section file — contracts §3.1: one marker, write-once, sealed
# =============================================================================
class MachineSection(unittest.TestCase):
    """The anatomy of an assessed experiment file, byte by byte.

    contracts §3.1 makes three promises that nothing else in the system can
    enforce: the student section is never rewritten, assessment happens once, and
    an edit above the marker is DETECTED. The third is what makes the second
    enforceable at all — without the seal, "write-once" is a rule about a marker
    line anyone can delete.

    The fixture's declaration is deliberately awkward: a comment header, blank
    lines between predicates, and a threshold spelled `0.030`. rdflib would
    normalise every one of those away, so if the student section is ever
    round-tripped through a parser this class goes red immediately.
    """

    SPELLING = '"0.030"^^xsd:double'

    @classmethod
    def setUpClass(cls):
        cls.settings = [("maxClipHiFraction", 0.03, None, None, cls.SPELLING),
                        ("maxMapMeanL2", 0.40), ("minCoverageF", 0.40)]
        cls.path = _declare("seal_main", BATCH_DEGEN_DIRNAME, ("HighlightClipping",),
                            pinned=False, settings=cls.settings,
                            label="the bytes above the marker are mine")
        cls.declared = _bytes_of(cls.path)
        _main(["experiment", cls.path])

    def test_the_marker_is_the_frozen_line_and_appears_exactly_once(self):
        """contracts §3.1, and `api.MACHINE_MARKER` must BE that line.

        Every writer splits the file on this string, so two of them would put the
        machine section in the middle of itself and a drifted one would make
        `write_run` rewrite the student's Turtle.
        """
        self.assertEqual(_require("MACHINE_MARKER"), MACHINE_MARKER)
        text = _text_of(self.path)
        self.assertEqual(text.count(MACHINE_MARKER), 1)
        lines = text.splitlines()
        self.assertIn(MACHINE_MARKER, lines,
                      "the marker must be a LINE of its own, not a substring")

    def test_the_student_bytes_survive_assessment_verbatim(self):
        """contracts §3.1: "read and validated, byte-for-byte preserved".

        Not "semantically preserved". The declaration is the student's artefact and
        the notebook's evidence; a machine pass that parsed and re-serialized it
        would silently delete their comments, re-order their triples, rename their
        blank nodes and re-spell their numbers — and the diff between two
        experiments in the notebook would stop being a diff between two designs.
        """
        head = _declaration_bytes(self.path)
        self.assertTrue(head.startswith(self.declared),
                        "the student section was rewritten by `experiment`")
        self.assertEqual(head[len(self.declared):].strip(), b"",
                         "`experiment` inserted content above the marker")

    def test_the_spelling_the_student_chose_is_still_in_the_file(self):
        """The half of the rule a byte comparison alone would not explain.

        `0.030` is not how rdflib spells 0.03, and `# ---` comments do not survive
        a parse. Both are still here, above the marker, unchanged.
        """
        head = _declaration_bytes(self.path).decode("utf-8")
        self.assertIn(self.SPELLING, head)
        self.assertIn("hand-authored declaration", head)
        self.assertIn("the bytes above the marker are mine", head)

    def test_each_section_also_parses_on_its_own(self):
        """contracts §3.1: "the whole file parses as one graph and each section
        also parses alone".

        That is what "duplicate `@prefix` directives mid-file are legal Turtle"
        buys, and it is not decoration: it is how a reader tells a GIVEN setting
        from a DEFAULTED one (§4.2) — by which side of the marker it parses from.
        """
        whole = _graph_of(self.path)
        student = Graph()
        student.parse(data=_declaration_bytes(self.path).decode("utf-8"),
                      format="turtle")
        machine = Graph()
        machine.parse(data=_machine_text(self.path), format="turtle")
        self.assertTrue(len(student) > 0 and len(machine) > 0)
        self.assertLessEqual(set(machine.predicates(None, None)) |
                             set(student.predicates(None, None)),
                             set(whole.predicates(None, None)))

    def test_given_and_defaulted_settings_are_told_apart_by_the_marker(self):
        """contracts §4.2: "given vs defaulted is recoverable from which side of
        the marker a setting sits on".

        This is what replaced v2's total vector plus a digest: the file records
        every required parameter, and the SPLIT records which of them the student
        chose. `explore` view 2 and view 3 both print that column, and there is no
        predicate for it anywhere — the position IS the fact.
        """
        student = Graph()
        student.parse(data=_declaration_bytes(self.path).decode("utf-8"),
                      format="turtle")
        machine = Graph()
        machine.parse(data=_machine_text(self.path), format="turtle")

        def params(g):
            return {str(o).rsplit("#", 1)[-1]
                    for o in g.objects(None, api.HW1.settingParameter)}

        given, defaulted = params(student), params(machine)
        self.assertEqual(given, {"maxClipHiFraction", "maxMapMeanL2", "minCoverageF"})
        self.assertIn("tauHi", defaulted)
        self.assertEqual(given & defaulted, set(),
                         "a parameter recorded on both sides of the marker has two "
                         "levels; §4.2 gives it one")
        self.assertEqual(given | defaulted, _required_params(("HighlightClipping",)))

    def test_a_machine_minted_setting_uses_the_scheme_iri(self):
        """contracts §2/§4.2: only the machine's settings are bound to the scheme.

        `<ns>experiment/<expname>/setting/<factor>/<param>`, with the PRIMARY
        factor. A blank node here would be unreachable from anywhere else in the
        file, and the same default filled twice would mint two of them.
        """
        g = _graph_of(self.path)
        node = _require("setting_iri")("seal_main", "HighlightClipping", "tauHi")
        self.assertIn((node, RDF.type, api.HW1.FactorSetting), g)
        self.assertEqual(g.value(node, api.HW1.settingParameter), api.HW1.tauHi)
        self.assertEqual(g.value(node, api.HW1.settingRole),
                         api.HW1.MeasurementSetting)
        self.assertEqual(list(g.objects(node, api.HW1.settingForFactor)),
                         [api.HW1.HighlightClipping],
                         "§4.5 makes settingForFactor single-valued in v3")

    def test_the_seal_is_the_sha256_of_the_bytes_above_the_marker(self):
        """contracts §3.1, transcribed and compared.

        Sixty-four hex characters over `data[0:marker_start]`. Any other reading —
        the parsed graph, the file minus the machine section's trailing newline,
        the text after normalisation — produces a digest that changes when nothing
        changed, and `read_experiment` would then refuse every file it wrote.
        """
        g = _graph_of(self.path)
        exp = _sole_experiment(g)
        stored = str(g.value(exp, api.HW1.declarationDigest))
        self.assertRegex(stored, r"\A[0-9a-f]{64}\Z")
        self.assertEqual(stored, _seal_of(self.path))
        self.assertEqual(stored, hashlib.sha256(self.declared).hexdigest())

    def test_assessing_an_already_assessed_file_is_a_hard_error(self):
        """contracts §3.1: WRITE-ONCE. No `--force`, no re-assess path.

        "Copy the declaration to a new name." Every tuning — a threshold, a factor,
        the batch — is a brand-new experiment, and that is what turns
        `hw1/experiments/` into an append-only lab notebook whose diff is a diff
        between two designs. A second assessment would silently replace the
        verdicts a student already wrote up, under a name that still describes the
        first design.
        """
        before = _bytes_of(self.path)
        message = _cli_failure(self, ["experiment", self.path]).lower()
        self.assertEqual(_bytes_of(self.path), before,
                         "a re-assessment rewrote a file §3.1 seals")
        self.assertIn("already", message,
                      f"the error must say the file is already assessed:\n{message}")
        self.assertTrue("copy" in message or "new" in message,
                        f"§3.1 names the fix — copy the declaration to "
                        f"a new name — and the message is where a student meets "
                        f"it:\n{message}")

    def test_an_edit_above_the_marker_is_detected_by_both_readers(self):
        """contracts §3.1/§7: `read_experiment` AND `write_run` verify the seal
        FIRST.

        An edited declaration is a NEW experiment that must get a new file. Without
        this check the student's most natural move — tweak the threshold, re-run
        `reconstruct` — would produce a file whose declaration says 0.02 and whose
        every status was computed at 0.03, with nothing in the file disagreeing.
        """
        tampered = _edit_declaration(self.path, "seal_tampered",
                                     self.SPELLING, '"0.020"^^xsd:double')
        self.assertNotEqual(_seal_of(tampered), _seal_of(self.path),
                            "fixture is broken: the edit did not change the seal")
        with self.assertRaises(Exception) as cm:
            _require("read_experiment")(tampered)
        self.assertNotIsInstance(cm.exception, AssertionError)
        with self.assertRaises(Exception) as cm:
            _require("write_run")(tampered, "baseline", {"mapMeanL2": 0.41})
        self.assertNotIsInstance(cm.exception, AssertionError)

    def test_a_comment_edit_above_the_marker_is_detected_too(self):
        """The seal is over BYTES, not over triples.

        A comment carries no triple, so a graph-level seal would call this file
        unchanged — and then the one thing the seal exists for, "was this
        declaration edited after it was assessed", would answer no to an edit
        anyone can see.
        """
        tampered = _edit_declaration(self.path, "seal_comment",
                                     "hand-authored declaration",
                                     "hand-authoured declaration")
        with self.assertRaises(Exception) as cm:
            _require("read_experiment")(tampered)
        self.assertNotIsInstance(cm.exception, AssertionError)

    def test_write_run_leaves_the_student_bytes_untouched(self):
        """contracts §3.1/§4.4: `write_run` rewrites ONLY below the marker.

        It is the one mutation an assessed file accepts, because a run is the
        experiment's outcome rather than its design. Re-serializing the whole file
        would be the easy implementation and would destroy the seal it just
        verified — the file would then refuse its own next read.
        """
        path = _emit_experiment("seal_run", BATCH_DEGEN_DIRNAME, DEPTH_ONLY)
        before_head = _declaration_bytes(path)
        before_seal = _seal_of(path)
        _require("write_run")(path, "baseline", {"mapMeanL2": 0.41}, frame_count=2)
        _require("write_run")(path, "selected", {"coverageF": 0.62},
                              used_frames=[0], frame_count=1)
        self.assertEqual(_declaration_bytes(path), before_head)
        self.assertEqual(_seal_of(path), before_seal)
        g = _graph_of(path)
        self.assertEqual(str(g.value(_sole_experiment(g), api.HW1.declarationDigest)),
                         before_seal, "write_run invalidated the seal it verified")


# =============================================================================
# 10. `experiment` — measurement scoped to the declared selection (§4.2/§4.3)
# =============================================================================
class ExperimentGraph(unittest.TestCase):
    """One measurement pass over the main synthetic batch, whole menu selected.

    ALL_MENU is v2's total pass, so this class is where v3 has to keep every
    guarantee v2 had — while the two below it pin what happens when the selection
    is smaller, which is the case v2 could not express at all.
    """

    NAME = "exp_main"

    @classmethod
    def setUpClass(cls):
        cls.path = _emit_experiment(cls.NAME, BATCH_DIRNAME, ALL_MENU,
                                    label="the whole menu, thresholds pinned")
        cls.g = _graph_of(cls.path)
        cls.exp = api.experiment_iri(cls.NAME)
        cls.ann = _annotation_nodes(cls.g, cls.exp)
        cls.pairs = _pair_nodes(cls.g, cls.exp)

    def test_the_experiment_node_carries_its_name_batch_and_selection(self):
        """contracts §4.2/§4.3 — the join points, all of them re-opened subjects.

        The machine section asserts `producesAnnotation`, `producesPair` and
        `hasFactorSetting` about an IRI the STUDENT declared, which is how the two
        sections of §3.1 become one graph without either rewriting the other.
        """
        self.assertIn((self.exp, RDF.type, api.HW1.Experiment), self.g)
        self.assertEqual(self.g.value(self.exp, api.HW1.onBatch),
                         api.batch_iri(BATCH))
        self.assertEqual({str(f).rsplit("#", 1)[-1]
                          for f in self.g.objects(self.exp, api.HW1.evaluatesFactor)},
                         set(ALL_MENU))
        self.assertTrue(list(self.g.objects(self.exp, api.HW1.producesAnnotation)))
        self.assertTrue(list(self.g.objects(self.exp, api.HW1.producesPair)))

    def test_a_label_is_the_students_and_is_never_rewritten(self):
        """`rdfs:label` is optional (§4.2) and lives ABOVE the marker.

        v2 had `--label` on the command line and the machine wrote it; v3 has the
        student write it, which is why it needs no machine support at all — it is
        simply part of the bytes that are preserved.
        """
        self.assertEqual(str(self.g.value(self.exp, RDFS.label)),
                         "the whole menu, thresholds pinned")
        self.assertIn("the whole menu, thresholds pinned",
                      _declaration_bytes(self.path).decode("utf-8"))

    def test_the_recorded_vector_is_exactly_the_required_set(self):
        """The all-menu declaration records exactly the active required vector.

        Deprecated v3 parameters remain parseable in the TBox but are not pulled
        into a new v4 experiment.
        """
        recorded = _recorded_settings(self.g, self.exp)
        self.assertEqual(set(recorded), _required_params(ALL_MENU))

    def test_generation_parameters_are_never_recorded_on_an_experiment(self):
        """contracts §5: they describe PIXELS and live on the batch.

        Recording `brightnessGain` here would claim this measurement pass applied a
        gain, and the verdict section would then print "regenerate the data" for a
        setting the experiment never touched.
        """
        recorded = _recorded_settings(self.g, self.exp)
        for name in GENERATION_PARAMS:
            self.assertNotIn(name, recorded)

    def test_each_machine_minted_setting_has_the_frozen_shape(self):
        """§4.2/§4.5: type, parameter, role, ONE factor, value.

        The single `settingForFactor` is the v3 change and it is load-bearing in
        the other direction: `explore` must traverse `paramAffectsFactor` in the
        TBox to reach `brightnessGain` from a ShadowClipping failure, because the
        setting node no longer says so itself.
        """
        decls = _require("load_parameter_declarations")()
        for node in self.g.objects(self.exp, api.HW1.hasFactorSetting):
            param = str(self.g.value(node, api.HW1.settingParameter)).rsplit("#", 1)[-1]
            if str(node).startswith(str(api.experiment_iri(self.NAME))):
                with self.subTest(parameter=param):
                    self.assertIn((node, RDF.type, api.HW1.FactorSetting), self.g)
                    self.assertEqual(
                        self.g.value(node, api.HW1.settingRole),
                        api.HW1[decls[param][DECL_ROLE]])
                    self.assertEqual(
                        list(self.g.objects(node, api.HW1.settingForFactor)),
                        [api.HW1[decls[param][DECL_PRIMARY]]])
                    self.assertIsNotNone(self.g.value(node, api.HW1.settingValue))

    def test_setting_values_are_typed_per_the_declared_value_kind(self):
        """contracts §5: `xsd:double` / `xsd:integer` / `xsd:string`, from the kind.

        A threshold spelled as a string never compares against a measured double,
        and `icpBackend` as a plain literal does not compare equal to an
        `xsd:string`-typed one in rdflib — the trap §5's last paragraph names.
        """
        recorded = _recorded_settings(self.g, self.exp)
        expected = {"double": XSD.double, "integer": XSD.integer,
                    "string": XSD.string}
        for name, literal in recorded.items():
            with self.subTest(parameter=name):
                self.assertEqual(literal.datatype,
                                 expected[DECLARED_PARAMS[name][1]])

    def test_one_annotation_per_frame_per_modality_linked_both_ways(self):
        """contracts §4.3 — the v3 shape, and the two links that make it joinable.

        `hw1:annotatesFrame` for frame-level joins; `hw1:describesImage` for the
        raster the number actually came from. Two annotations share a frame index,
        so the index alone is no longer a key — which is exactly why the IRI
        carries the modality.
        """
        self.assertEqual(set(self.ann),
                         {(stem, kind) for stem in MAIN_STEMS for kind in MODALITIES})
        for (stem, kind), node in sorted(self.ann.items()):
            with self.subTest(frame=stem, kind=kind):
                self.assertEqual(node, _require("annotation_iri")(self.NAME, stem, kind))
                self.assertIn((node, RDF.type, api.HW1.FrameAnnotation), self.g)
                self.assertEqual(self.g.value(node, api.HW1.annotatesFrame),
                                 api.frame_iri(BATCH, stem))
                self.assertEqual(int(self.g.value(node, api.HW1.frameIndex)), stem)
                self.assertEqual(self.g.value(node, api.HW1.describesImage),
                                 api.component_iri(BATCH, stem, kind))
                self.assertIn((self.exp, api.HW1.producesAnnotation, node), self.g)

    def test_every_selected_factors_value_and_status_is_total_on_its_modality(self):
        """contracts §4.3: "within the declared scope, a missing property is a bug".

        Totality is PER SELECTION now, which makes it a stronger statement rather
        than a weaker one: v2 could always fall back on "the property is there but
        it means nothing here". v3 cannot — if the factor was declared, the number
        is measured and graded on every node of its modality.
        """
        for (stem, kind), node in sorted(self.ann.items()):
            for factor in ALL_MENU:
                if FACTOR_MODALITY.get(factor) != kind:
                    continue
                over, status, _p, _q = QUALITY_FACTORS[factor]
                with self.subTest(frame=stem, kind=kind, factor=factor):
                    self.assertIsNotNone(self.g.value(node, api.HW1[over]))
                    self.assertIn(self.g.value(node, api.HW1[status]),
                                  (api.HW1.Pass, api.HW1.Fail))
            self.assertIn(self.g.value(node, api.HW1[AGGREGATE_STATUS]),
                          (api.HW1.Pass, api.HW1.Fail))

    def test_a_depth_observable_never_lands_on_an_rgb_annotation(self):
        """contracts §4.3's placement rule, as an exclusion.

        `validDepthFraction` on the rgb node would make `hw1:describesImage` a lie
        and the modality aggregate a mix of two rasters' verdicts — and it is the
        obvious way to implement the split wrongly: mint two nodes, write
        everything to both.
        """
        for (stem, kind), node in sorted(self.ann.items()):
            foreign = [f for f in FRAME_FACTORS if FACTOR_MODALITY[f] != kind]
            for factor in foreign:
                over, status, _p, _q = QUALITY_FACTORS[factor]
                with self.subTest(frame=stem, kind=kind, factor=factor):
                    self.assertIsNone(self.g.value(node, api.HW1[over]))
                    self.assertIsNone(self.g.value(node, api.HW1[status]))

    def test_mean_value_sits_on_the_rgb_annotation_and_carries_no_status(self):
        """contracts §4.3/§7.2 — the baseline, deliberately weak, deliberately here.

        It is written whenever an rgb annotation exists and it never gets a status.
        That is the "beat the mean" lesson in the data model: a number with no
        threshold next to numbers with thresholds, so the difference between a
        measurement and a criterion is visible in the file rather than asserted in
        the handout.
        """
        for stem in MAIN_STEMS:
            with self.subTest(frame=stem):
                rgb = self.ann[(stem, "rgb")]
                self.assertAlmostEqual(float(self.g.value(rgb, api.HW1.meanValue)),
                                       MAIN_FRAME_VALUES[stem]["meanValue"],
                                       delta=_serializer_delta(255.0))
                self.assertIsNone(self.g.value(rgb, api.HW1.meanValueStatus))
                self.assertIsNone(self.g.value(self.ann[(stem, "depth")],
                                               api.HW1.meanValue))

    def test_the_stored_values_are_the_closed_form_ones(self):
        """The fixture geometry, recovered through the whole CLI (§4.6 tolerance)."""
        for stem in MAIN_STEMS:
            for over, expected in MAIN_FRAME_VALUES[stem].items():
                if over == "meanValue":
                    continue
                kind = FACTOR_MODALITY[_FACTOR_BY_OVER[over]]
                with self.subTest(frame=stem, observable=over):
                    got = float(self.g.value(self.ann[(stem, kind)], api.HW1[over]))
                    self.assertAlmostEqual(got, expected,
                                           delta=_serializer_delta(expected))
        rough = float(self.g.value(
            self.ann[(4, "depth")], api.HW1.highFrequencyDepthResidual))
        self.assertAlmostEqual(rough, BATCH_NOISE_SIGMA_M, delta=0.02,
                               msg="frame 4 must recover the sigma it was built with")

    def test_each_frame_fails_exactly_the_factor_it_was_built_to_fail(self):
        """The fixture table at the top of this file, executed.

        The fixtures independently exercise RGB clipping, tile dropout and noisy
        depth. Without that, an aggregate that
        happened to be the conjunction of the wrong subset would still look right.
        """
        for stem, expected in MAIN_FRAME_STATUSES.items():
            for factor_value, passes in zip(FRAME_FACTOR_VALUES, expected):
                factor = _FACTOR_BY_OVER[factor_value]
                kind = FACTOR_MODALITY[factor]
                with self.subTest(frame=stem, observable=factor_value):
                    got = self.g.value(self.ann[(stem, kind)],
                                       api.HW1[factor_value + "Status"])
                    self.assertEqual(got, api.HW1.Pass if passes else api.HW1.Fail)

    def test_the_aggregate_is_the_conjunction_of_that_modalitys_factors(self):
        """contracts §4.5: `qualificationStatus` on an annotation is Pass iff EVERY
        SELECTED FACTOR OF THAT ANNOTATION'S MODALITY is Pass.

        The v2 predicate aggregated all four frame factors onto one node; v3
        aggregates two and two. Frame 1 (black rgb, clean depth) is the case that
        separates the readings: its depth annotation passes and its rgb annotation
        fails, and a single frame-level aggregate could not say both.
        """
        for stem in MAIN_STEMS:
            for kind, expected in MAIN_MODALITY_STATUSES[stem].items():
                with self.subTest(frame=stem, kind=kind):
                    self.assertEqual(
                        self.g.value(self.ann[(stem, kind)],
                                     api.HW1[AGGREGATE_STATUS]),
                        api.HW1.Pass if expected else api.HW1.Fail)
        self.assertEqual(self.g.value(self.ann[(1, "depth")],
                                      api.HW1[AGGREGATE_STATUS]), api.HW1.Pass)
        self.assertEqual(self.g.value(self.ann[(1, "rgb")],
                                      api.HW1[AGGREGATE_STATUS]), api.HW1.Fail)

    def test_mean_value_does_not_participate_in_the_aggregate(self):
        """Frame 2 is pure white — mean 255, the most extreme value in the batch —
        and its rgb aggregate is decided by `clipHiFraction` alone. A `meanValue`
        that voted would make the baseline a criterion by the back door."""
        self.assertAlmostEqual(
            float(self.g.value(self.ann[(2, "rgb")], api.HW1.meanValue)), 255.0,
            delta=_serializer_delta(255.0))
        self.assertEqual(self.g.value(self.ann[(2, "rgb")],
                                      api.HW1[AGGREGATE_STATUS]), api.HW1.Fail)
        self.assertEqual(self.g.value(self.ann[(0, "rgb")],
                                      api.HW1[AGGREGATE_STATUS]), api.HW1.Pass)

    def test_one_pair_per_consecutive_frame_pair(self):
        """contracts §4.3: pairs carry adjacency into the experiment file."""
        self.assertEqual(set(self.pairs), set(MAIN_PAIRS))
        for (i, j), node in sorted(self.pairs.items()):
            with self.subTest(pair=(i, j)):
                self.assertIn((node, RDF.type, api.HW1.FramePair), self.g)
                self.assertEqual(self.g.value(node, api.HW1.sourceFrame),
                                 api.frame_iri(BATCH, i))
                self.assertEqual(self.g.value(node, api.HW1.targetFrame),
                                 api.frame_iri(BATCH, j))
                self.assertIn((self.exp, api.HW1.producesPair, node), self.g)

    def test_pair_index_is_a_zero_based_ordinal(self):
        """`hw1:pairIndex` orders pairs; the IRI does not (contracts §2)."""
        indices = {ij: int(self.g.value(node, api.HW1.pairIndex))
                   for ij, node in self.pairs.items()}
        self.assertEqual(sorted(indices.values()), list(range(len(MAIN_PAIRS))))
        self.assertEqual([indices[ij] for ij in MAIN_PAIRS],
                         list(range(len(MAIN_PAIRS))))

    def test_assessment_pair_factors_are_total_and_prior_warp_is_deferred(self):
        """Raster-only pair factors are total; prior warp waits for reconstruction."""
        for ij, node in sorted(self.pairs.items()):
            for over in ("identityMedianDepthChange", "jointValidDepthRatio"):
                with self.subTest(pair=ij, observable=over):
                    self.assertIsNotNone(self.g.value(node, api.HW1[over]))
                    self.assertIn(self.g.value(node, api.HW1[over + "Status"]),
                                  (api.HW1.Pass, api.HW1.Fail))
            self.assertIsNone(self.g.value(node, api.HW1.priorWarpDepthResidual))
            self.assertIsNone(self.g.value(node,
                                           api.HW1.priorWarpDepthResidualStatus))
            self.assertIn(self.g.value(node, api.HW1[AGGREGATE_STATUS]),
                          (api.HW1.Pass, api.HW1.Fail))

    def test_the_pair_aggregate_is_the_conjunction_of_the_selected_pair_factors(self):
        """contracts §4.5, the pair half — same predicate, different carrying node.

        The deferred prior-warp factor does not participate until reconstruction
        writes it back.
        """
        for ij, node in sorted(self.pairs.items()):
            expected = all(
                self.g.value(node, api.HW1[over + "Status"]) == api.HW1.Pass
                for over in ("identityMedianDepthChange", "jointValidDepthRatio"))
            with self.subTest(pair=ij):
                self.assertEqual(self.g.value(node, api.HW1[AGGREGATE_STATUS]),
                                 api.HW1.Pass if expected else api.HW1.Fail)

    def test_every_stored_status_is_the_status_rule_applied_to_the_stored_value(self):
        """The §4.5 rule against a TRANSCRIPTION of it, over every observable here.

        Not a call to `api.status_for` — that would only prove the code agrees with
        itself. And graded against the value THE FILE STORES (§4.6), because the
        writer grades the rounded number it wrote.
        """
        for factor, (over, status, _p, _q) in QUALITY_FACTORS.items():
            if factor in RUN_FACTORS:
                continue
            for node in self.g.subjects(api.HW1[over], None):
                value = float(self.g.value(node, api.HW1[over]))
                with self.subTest(node=str(node), observable=over):
                    self.assertEqual(
                        self.g.value(node, api.HW1[status]),
                        api.HW1.Pass if _expected_status(over, value,
                                                         PINNED_THRESHOLDS)
                        else api.HW1.Fail)


class SelectionScoping(unittest.TestCase):
    """contracts §4.2's completeness rule when the selection is SMALLER than six.

    This is the clause v3 exists for. v2 recorded every parameter on every
    experiment, which meant a depth-only study still carried a `tauHi` it never
    applied — a level that touched nothing, recorded as though it had.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = _emit_experiment("scope_depth", BATCH_DIRNAME, DEPTH_ONLY)
        cls.g = _graph_of(cls.path)
        cls.exp = api.experiment_iri("scope_depth")
        cls.recorded = _recorded_settings(cls.g, cls.exp)

    def test_a_depth_only_experiment_records_no_tau(self):
        """The sentence contracts §5 ends on: "An experiment evaluating only depth
        factors records no `tauHi`".

        Neither the RGB measurement parameters nor the RGB thresholds are here.
        Recording them would be as false as a defaulted Generation setting: a
        number in the treatment column that changed nothing that was measured.
        """
        for name in ("tauHi", "tauLo", "maxClipHiFraction", "maxClipLoFraction"):
            with self.subTest(parameter=name):
                self.assertNotIn(name, self.recorded)

    def test_it_records_exactly_the_required_set(self):
        """qualifiedBy ∪ active Measurement params ∪ run-factor params."""
        self.assertEqual(set(self.recorded), _required_params(DEPTH_ONLY))
        self.assertEqual(set(self.recorded),
                         {"maxHighFrequencyDepthResidual", "residualMaskK",
                          "minValidTileCoverage", "tileSize", "tileValidFloor"}
                         | set(RUN_FACTOR_PARAMS))

    def test_a_pair_only_selection_pulls_in_its_mask_setting(self):
        """Identity change requires its mask level and qualification threshold."""
        path = _emit_experiment("scope_pair", BATCH_SEL_DIRNAME, PAIR_ONLY)
        recorded = _recorded_settings(_graph_of(path), api.experiment_iri("scope_pair"))
        self.assertEqual(set(recorded), _required_params(PAIR_ONLY))
        self.assertIn("changeMaskK", recorded)
        self.assertIn("maxIdentityMedianDepthChange", recorded)
        self.assertNotIn("priorWarpDepthGate", recorded)

    def test_the_selection_is_readable_back_off_the_file(self):
        """`hw1:evaluatesFactor` survives into the assessed file (§4.2/§8.2).

        Every later reader — `read_experiment`, `explore`, `reconstruct` — needs to
        know what the totality guarantee covers. A file that recorded values
        without the selection would leave "is this property missing or was it never
        asked for" unanswerable.
        """
        self.assertEqual({str(f).rsplit("#", 1)[-1]
                          for f in self.g.objects(self.exp, api.HW1.evaluatesFactor)},
                         set(DEPTH_ONLY))


class ModalityAnnotations(unittest.TestCase):
    """contracts §4.3: "a modality with no selected factor gets NO node"."""

    @classmethod
    def setUpClass(cls):
        cls.depth_path = _emit_experiment("mod_depth", BATCH_DIRNAME, DEPTH_ONLY)
        cls.rgb_path = _emit_experiment("mod_rgb", BATCH_DIRNAME, RGB_ONLY)
        cls.pair_path = _emit_experiment("mod_pair", BATCH_SEL_DIRNAME, PAIR_ONLY)

    def _annotations(self, path, name):
        g = _graph_of(path)
        return g, _annotation_nodes(g, api.experiment_iri(name))

    def test_a_depth_only_experiment_mints_no_rgb_annotation(self):
        """Not an empty rgb node — NO node.

        An annotation with no observables would be indistinguishable from one whose
        measurers all failed, and its `qualificationStatus` would be a vacuous Pass
        sitting on a raster nobody looked at.
        """
        _g, ann = self._annotations(self.depth_path, "mod_depth")
        self.assertEqual(set(ann), {(stem, "depth") for stem in MAIN_STEMS})
        self.assertEqual([k for k in ann if k[1] == "rgb"], [])

    def test_an_rgb_only_experiment_mints_no_depth_annotation(self):
        """... and the mirror image, so the rule is about the SELECTION rather than
        about depth being special."""
        _g, ann = self._annotations(self.rgb_path, "mod_rgb")
        self.assertEqual(set(ann), {(stem, "rgb") for stem in MAIN_STEMS})
        self.assertEqual([k for k in ann if k[1] == "depth"], [])

    def test_a_pair_only_experiment_mints_no_annotation_at_all(self):
        """Both modalities empty. The frames are still there — as PAIR endpoints.

        This is the fixture behind §4.5's "vacuously usable when the frame has no
        annotations", and it is why frame usability had to move out of the file and
        into `read_experiment`: there is no node left to carry it.
        """
        _g, ann = self._annotations(self.pair_path, "mod_pair")
        self.assertEqual(ann, {})

    def test_mean_value_is_written_iff_an_rgb_annotation_exists(self):
        """contracts §4.3: "written whenever an rgb annotation exists".

        It is not a selected factor and never has a status, so the only rule that
        can place it is the existence of the node it belongs on.
        """
        g, ann = self._annotations(self.rgb_path, "mod_rgb")
        for (stem, _kind), node in ann.items():
            with self.subTest(frame=stem):
                self.assertIsNotNone(g.value(node, api.HW1.meanValue))
        depth_g, _ann = self._annotations(self.depth_path, "mod_depth")
        self.assertEqual(list(depth_g.triples((None, api.HW1.meanValue, None))), [],
                         "meanValue was written with no rgb annotation to carry it")

    def test_the_aggregate_covers_only_that_modalitys_selected_factors(self):
        """contracts §4.5, on a selection where the two modalities disagree.

        Under DEPTH_ONLY, frames 1 and 2 (black and white rgb) have no rgb
        annotation at all, so nothing they do to the clip factors can reach the
        depth aggregate — and frame 3 fails it on validity alone.
        """
        g, ann = self._annotations(self.depth_path, "mod_depth")
        expected = {stem: MAIN_MODALITY_STATUSES[stem]["depth"] for stem in MAIN_STEMS}
        for stem, passes in expected.items():
            with self.subTest(frame=stem):
                self.assertEqual(g.value(ann[(stem, "depth")],
                                         api.HW1[AGGREGATE_STATUS]),
                                 api.HW1.Pass if passes else api.HW1.Fail)
        self.assertEqual(expected, {0: True, 1: True, 2: True, 3: False, 4: False})

    def test_describes_image_points_at_the_raster_that_was_measured(self):
        """contracts §4.3's new link, on the modality-split file.

        The rgb annotation points at `.../frame/<n>/rgb` and the depth annotation at
        `.../frame/<n>/depth`. Both point at the same FRAME through
        `annotatesFrame`, which is what keeps the frame-level join working.
        """
        g, ann = self._annotations(self.rgb_path, "mod_rgb")
        for (stem, kind), node in ann.items():
            with self.subTest(frame=stem, kind=kind):
                self.assertEqual(g.value(node, api.HW1.describesImage),
                                 api.component_iri(BATCH, stem, kind))
                self.assertEqual(g.value(node, api.HW1.annotatesFrame),
                                 api.frame_iri(BATCH, stem))


class PairMinting(unittest.TestCase):
    """contracts §4.3: "Pairs are always minted"."""

    @classmethod
    def setUpClass(cls):
        cls.path = _emit_experiment("pairs_frameonly", BATCH_SEL_DIRNAME, FRAME_ONLY)
        cls.g = _graph_of(cls.path)
        cls.exp = api.experiment_iri("pairs_frameonly")
        cls.pairs = _pair_nodes(cls.g, cls.exp)

    def test_every_consecutive_pair_exists_with_no_pair_factor_selected(self):
        """"...because the pair nodes are what carries adjacency into the experiment
        file and `reconstruct` reads nothing else".

        Skipping them when no pair factor is selected would be the obvious
        optimisation and would leave `reconstruct` with a frame set and no edges —
        so the segment cut would find one segment of length zero and the selected
        run would silently become the baseline run.
        """
        self.assertEqual(set(self.pairs), set(SEL_PAIRS))
        for ij, node in sorted(self.pairs.items()):
            with self.subTest(pair=ij):
                self.assertIn((node, RDF.type, api.HW1.FramePair), self.g)
                self.assertEqual(self.g.value(node, api.HW1.sourceFrame),
                                 api.frame_iri(BATCH_SEL, ij[0]))
                self.assertEqual(self.g.value(node, api.HW1.targetFrame),
                                 api.frame_iri(BATCH_SEL, ij[1]))

    def test_no_pair_observable_is_written_for_an_unselected_pair_factor(self):
        """"Pair observables appear only for selected pair factors."

        Measuring them anyway would be a value with no factor behind it in this
        experiment — and its status would have to be graded against a threshold the
        completeness rule did not record.
        """
        for ij, node in sorted(self.pairs.items()):
            for over in PAIR_FACTOR_VALUES:
                with self.subTest(pair=ij, observable=over):
                    self.assertIsNone(self.g.value(node, api.HW1[over]))
                    self.assertIsNone(self.g.value(node, api.HW1[over + "Status"]))

    def test_the_pair_aggregate_is_total_and_vacuously_passes(self):
        """contracts §4.5: "vacuously `Pass` when no pair factor is selected".

        Total, not absent: `read_experiment`'s `pair_status` is documented as total
        (§8.2) and the usable-link rule ANDs it with the endpoints, so a missing
        aggregate would have to be defaulted by every reader — three readers, three
        defaults, one of them wrong.
        """
        for ij, node in sorted(self.pairs.items()):
            with self.subTest(pair=ij):
                self.assertEqual(self.g.value(node, api.HW1[AGGREGATE_STATUS]),
                                 api.HW1.Pass)

    def test_pairs_are_built_between_adjacent_frames_not_adjacent_integers(self):
        """The gap batch: stem 2 is in `rgb/` only, so (1,3) is a real pair.

        A pair set built from `range(max_stem)` would invent a frame 2 that has no
        depth raster, and the measurer would be handed a path that does not exist.
        """
        path = _emit_experiment("pairs_gap", BATCH_GAP_DIRNAME, PAIR_ONLY)
        pairs = _pair_nodes(_graph_of(path), api.experiment_iri("pairs_gap"))
        self.assertEqual(set(pairs), set(GAP_PAIRS))

    def test_pair_index_is_an_ordinal_and_not_the_source_stem(self):
        """On the gap batch the two disagree: pair (1,3) is index 1, not 3."""
        path = _emit_experiment("pairs_gap_index", BATCH_GAP_DIRNAME, PAIR_ONLY)
        g = _graph_of(path)
        pairs = _pair_nodes(g, api.experiment_iri("pairs_gap_index"))
        self.assertEqual({ij: int(g.value(node, api.HW1.pairIndex))
                          for ij, node in pairs.items()},
                         {(0, 1): 0, (1, 3): 1})


class ExperimentTotality(unittest.TestCase):
    """contracts §4.3: "A measurer that fails writes its fail-closed value, never
    nothing" — on the batch where every depth measurer fails.

    The degenerate batch's frame 1 has no depth at all. A writer that skipped the
    triple would drop the frame out of every join, which reads as "no such frame" —
    a different and much worse lie than "this frame scored nothing".
    """

    @classmethod
    def setUpClass(cls):
        cls.path = _emit_experiment("total_degen", BATCH_DEGEN_DIRNAME,
                                    DEPTH_ONLY + PAIR_ONLY)
        cls.g = _graph_of(cls.path)
        cls.exp = api.experiment_iri("total_degen")
        cls.ann = _annotation_nodes(cls.g, cls.exp)
        cls.pairs = _pair_nodes(cls.g, cls.exp)

    def test_a_failing_measurer_writes_its_sentinel_not_nothing(self):
        """`inf` for the lower-is-better factors, `0.0` for the higher-is-better
        one — contracts §9's fail-closed rules, surviving all the way to the file."""
        depth = self.ann[(1, "depth")]
        self.assertEqual(float(self.g.value(depth, api.HW1.validTileCoverage)), 0.0)
        self.assertEqual(float(self.g.value(
            depth, api.HW1.highFrequencyDepthResidual)), INF)
        pair = self.pairs[(0, 1)]
        self.assertEqual(float(self.g.value(
            pair, api.HW1.identityMedianDepthChange)), INF)

    def test_inf_serialises_as_the_xsd_double_INF_literal(self):
        """contracts §4.6: `INF`, per XSD — not `inf`, not a missing triple.

        Asserted against the SERIALIZED TEXT rather than the parsed term: rdflib's
        parser normalises an `xsd:double`'s lexical form back through `str(float)`,
        so a term read out of a correct file still reports `'inf'`. §4.6 governs the
        bytes a stricter engine will read.
        """
        text = _text_of(self.path)
        self.assertIn('"INF"^^xsd:double', text)
        self.assertNotIn('"inf"^^xsd:double', text)

    def test_the_sentinels_grade_fail_and_carry_the_aggregate_down(self):
        """No `isinf` branch anywhere: `inf <= t` is False for every finite t."""
        depth = self.ann[(1, "depth")]
        self.assertEqual(self.g.value(depth, api.HW1.validTileCoverageStatus),
                         api.HW1.Fail)
        self.assertEqual(self.g.value(
            depth, api.HW1.highFrequencyDepthResidualStatus),
                         api.HW1.Fail)
        self.assertEqual(self.g.value(depth, api.HW1[AGGREGATE_STATUS]), api.HW1.Fail)
        self.assertEqual(self.g.value(self.pairs[(0, 1)], api.HW1[AGGREGATE_STATUS]),
                         api.HW1.Fail)

    def test_the_clean_frame_of_the_same_batch_is_unaffected(self):
        """Frame 0 of the same batch is perfect, so the sentinels are not a
        file-wide fallback that happens to look right on the broken frame."""
        depth = self.ann[(0, "depth")]
        self.assertEqual(float(self.g.value(depth, api.HW1.validTileCoverage)), 1.0)
        self.assertEqual(float(self.g.value(
            depth, api.HW1.highFrequencyDepthResidual)), 0.0)
        self.assertEqual(self.g.value(depth, api.HW1[AGGREGATE_STATUS]), api.HW1.Pass)


class ExperimentIsolation(unittest.TestCase):
    """LOAD-BEARING: two experiments over one batch, in ONE graph, colliding nowhere.

    This class replaces v1's named-graph scoping tests wholesale. There are no
    named graphs, and the ONLY thing keeping two treatments of one capture apart is
    that every node an experiment mints carries its NAME (contracts §2). If any
    minter drops it, the two files still parse, still load into one store, and
    silently answer every question with whichever triple was read last.
    """

    @classmethod
    def setUpClass(cls):
        # Same batch, same selection; the second measures at a tauHi BELOW the grey
        # frames' value, so the two disagree about clipHiFraction on three of the
        # five frames. Two NAMES, two files — the v3 shape of "nothing is ever
        # overwritten" (§3.1).
        cls.path_a = _emit_experiment("iso_default_tau", BATCH_DIRNAME, RGB_ONLY)
        cls.path_b = _emit_experiment("iso_low_tau", BATCH_DIRNAME, RGB_ONLY,
                                      settings=[("tauHi", 127.0)])
        cls.g_a, cls.g_b = _graph_of(cls.path_a), _graph_of(cls.path_b)
        cls.exp_a = api.experiment_iri("iso_default_tau")
        cls.exp_b = api.experiment_iri("iso_low_tau")

    def test_two_treatments_of_one_batch_get_two_names(self):
        """contracts §6: identity is the name, and the student chooses it.

        Naming your experimental conditions is part of designing an experiment —
        which is also why the file name and the IRI have to agree (§2).
        """
        self.assertNotEqual(self.exp_a, self.exp_b)
        self.assertEqual(str(self.exp_a).rsplit("/", 1)[-1], "iso_default_tau")
        self.assertEqual(str(self.exp_b).rsplit("/", 1)[-1], "iso_low_tau")

    def test_two_experiments_over_one_batch_do_not_collide(self):
        """The rule of contracts §2 that must never be relaxed.

        Everything minted is disjoint; the frames are SHARED, because they are the
        join. Asserted as a set intersection rather than per-node so a single
        forgotten name anywhere in the emitter shows up here.
        """
        def minted(g, exp):
            nodes = set(g.objects(exp, api.HW1.producesAnnotation))
            nodes |= set(g.objects(exp, api.HW1.producesPair))
            nodes |= {n for n in g.objects(exp, api.HW1.hasFactorSetting)
                      if isinstance(n, URIRef)}
            nodes.add(exp)
            return {str(n) for n in nodes}

        a, b = minted(self.g_a, self.exp_a), minted(self.g_b, self.exp_b)
        self.assertTrue(a and b)
        self.assertEqual(a & b, set(),
                         "an experiment-scoped node is shared between two "
                         "experiments; §2 forbids it")

        frames_a = {str(self.g_a.value(n, api.HW1.annotatesFrame))
                    for n in self.g_a.objects(self.exp_a, api.HW1.producesAnnotation)}
        frames_b = {str(self.g_b.value(n, api.HW1.annotatesFrame))
                    for n in self.g_b.objects(self.exp_b, api.HW1.producesAnnotation)}
        self.assertEqual(frames_a, frames_b)
        self.assertEqual(frames_a,
                         {str(api.frame_iri(BATCH, s)) for s in MAIN_STEMS})

    def test_the_union_of_both_files_holds_no_contradictory_triple(self):
        """Both files loaded into ONE graph — which is what `explore a.ttl b.ttl`
        does (§7.1 view 4).

        Every observation subject keeps exactly one value per predicate, and the
        two experiments' disagreement about `clipHiFraction` survives as two
        annotations rather than as two triples on one node.
        """
        merged = Graph()
        merged.parse(self.path_a, format="turtle")
        merged.parse(self.path_b, format="turtle")

        annotations = set(merged.subjects(RDF.type, api.HW1.FrameAnnotation))
        self.assertEqual(len(annotations), 2 * len(MAIN_STEMS))
        for node in annotations:
            for prop in ("meanValue", "clipHiFraction", "clipLoFraction"):
                with self.subTest(node=str(node), predicate=prop):
                    self.assertEqual(
                        len(list(merged.objects(node, api.HW1[prop]))), 1,
                        "two experiments wrote onto one observation subject")

    def test_the_two_experiments_really_do_disagree(self):
        """... so the isolation above is not vacuous.

        The grey frames read V = 128: `clipHiFraction` is 0.0 at tauHi=250 and 1.0
        at tauHi=127, and the two verdicts differ with it. If both files carried the
        same numbers, "nothing collides" would be true of two identical copies and
        would prove nothing.
        """
        minter = _require("annotation_iri")
        for stem in (0, 3, 4):                       # the mid-grey frames
            with self.subTest(frame=stem):
                a = float(self.g_a.value(minter("iso_default_tau", stem, "rgb"),
                                         api.HW1.clipHiFraction))
                b = float(self.g_b.value(minter("iso_low_tau", stem, "rgb"),
                                         api.HW1.clipHiFraction))
                self.assertEqual((a, b), (0.0, 1.0))
                self.assertEqual(
                    self.g_a.value(minter("iso_default_tau", stem, "rgb"),
                                   api.HW1.clipHiFractionStatus), api.HW1.Pass)
                self.assertEqual(
                    self.g_b.value(minter("iso_low_tau", stem, "rgb"),
                                   api.HW1.clipHiFractionStatus), api.HW1.Fail)


class SettingLexicalForm(unittest.TestCase):
    """contracts §5, "Lexical form is rdflib's, and it is lossy" — MACHINE side only.

    v3 splits this rule in two. A STUDENT-given setting never meets the serializer
    at all: its bytes are preserved (§3.1), and `MachineSection` pins the spelling
    the student chose surviving verbatim. What this class pins is the other half —
    that everything rdflib DOES write is written the same way every time, so two
    experiments recording one defaulted value compare equal across two files and
    `explore`'s comparison view cannot report a difference that does not exist.
    """

    @classmethod
    def setUpClass(cls):
        # tauHi at its declared default, defaulted into two different files over
        # two different batches, and the same number written by `batch2ttl` as a
        # GenerationSetting on a third.
        cls.batch_path = _run_batch2ttl(
            BATCH_DEGEN_DIRNAME, out=_out_path("lex_batch.ttl"),
            gen=[f"brightnessGain={DECLARED_PARAMS['tauHi'][4]!r}"])
        cls.one = _emit_experiment("lex_one", BATCH_DIRNAME, RGB_ONLY)
        cls.two = _emit_experiment("lex_two", BATCH_SEL_DIRNAME, RGB_ONLY)

    @staticmethod
    def _setting_text(path, param):
        g = _graph_of(path)
        for node in g.subjects(api.HW1.settingParameter, api.HW1[param]):
            return str(g.value(node, api.HW1.settingValue))
        raise AssertionError(f"no setting for hw1:{param} in {path}")

    def test_one_defaulted_value_is_spelled_the_same_way_in_two_files(self):
        """Two experiments, two batches, one `hw1:paramDefault`, one spelling.

        If the defaulting path preserved the TBox's lexical form in one file and
        re-derived it in another, `explore a.ttl b.ttl` would list `tauHi` as a
        DIFFERENCE between two experiments that ran at the same tau.
        """
        self.assertEqual(self._setting_text(self.one, "tauHi"),
                         self._setting_text(self.two, "tauHi"))

    def test_two_different_writers_spell_one_value_the_same_way(self):
        """`batch2ttl` and `experiment` are two commands and one serializer.

        The batch's `brightnessGain` and the experiment's `tauHi` are different
        parameters with different roles on different subjects, and they carry the
        same double. The verdict section prints Generation and Measurement culprits
        in one table (§7.1), so a spelling that depended on which command wrote it
        would make the two rows incomparable.
        """
        self.assertEqual(self._setting_text(self.batch_path, "brightnessGain"),
                         self._setting_text(self.one, "tauHi"))

    def test_a_value_round_trips_to_within_the_serializers_precision(self):
        """Parse -> read `settingValue` -> float recovers the level, to 7 digits.

        That is the whole guarantee, and it is enough for everything a verdict
        depends on: a threshold whose eighth significant digit moved qualifies
        nothing differently, which is why §5 can call the loss harmless.
        """
        declared = _require("load_parameter_declarations")()
        g = _graph_of(self.one)
        for name in sorted(_required_params(RGB_ONLY)):
            if DECLARED_PARAMS[name][1] != "double":
                continue
            expected = PINNED_THRESHOLDS.get(name, declared[name][DECL_DEFAULT])
            with self.subTest(parameter=name):
                node = next(g.subjects(api.HW1.settingParameter, api.HW1[name]))
                self.assertAlmostEqual(float(g.value(node, api.HW1.settingValue)),
                                       float(expected),
                                       delta=_serializer_delta(expected))


# =============================================================================
# 11. `read_experiment` — contracts §8.2
# =============================================================================
class ExperimentReader(unittest.TestCase):
    """The one reader every consumer of an experiment file goes through.

    `frame_status` and `usable_links` are DERIVED here — §4.5 deleted
    `hw1:frameStatus` and moved the conjunction into code — so `reconstruct.py`, a
    notebook and an evaluator cannot end up with three subtly different selection
    criteria.
    """

    @classmethod
    def setUpClass(cls):
        cls.read = staticmethod(_require("read_experiment"))
        cls.main_path = _emit_experiment("read_main", BATCH_DIRNAME, FRAME_ONLY)
        cls.sel_path = _emit_experiment("read_sel", BATCH_SEL_DIRNAME, ALL_MENU)
        cls.teens_path = _emit_experiment("read_teens", BATCH_TEENS_DIRNAME, ALL_MENU)

    def test_it_returns_every_documented_key(self):
        """contracts §8.2's dict, key by key.

        A reader that returned a superset is fine; one that omitted `graph` would
        make every caller re-parse the file it just read, and one that omitted
        `usable_links` would push the selection criterion back out into the
        callers.
        """
        info = self.read(self.main_path)
        self.assertLessEqual(
            {"exp_iri", "exp_name", "batch_name", "batch_iri", "selected",
             "settings", "frame_status", "pair_status", "usable_links", "graph"},
            set(info))
        self.assertEqual(info["exp_name"], "read_main")
        self.assertEqual(info["exp_iri"], api.experiment_iri("read_main"))
        self.assertEqual(info["batch_name"], BATCH)
        self.assertEqual(info["batch_iri"], api.batch_iri(BATCH))
        self.assertEqual(set(info["selected"]), set(FRAME_ONLY))
        self.assertIsInstance(info["graph"], Graph)

    def test_the_deleted_v2_key_is_gone(self):
        """contracts §6/§11: `exp_id` is deleted along with the digest.

        Leaving it importable means `reconstruct.py` can still key a file by
        something that no longer identifies it — and the two keys would agree right
        up until someone renamed a declaration.
        """
        self.assertNotIn("exp_id", self.read(self.main_path))

    def test_settings_come_back_typed_by_their_declared_kind(self):
        """`float | int | str` (contracts §8.2), and the caller must not re-parse.

        `reconstruct.py` compares `settings["icpBackend"]` against `--version`,
        and `write_run` grades against `settings["maxMapMeanL2"]`; a str where a
        float belongs turns a threshold comparison into a lexicographic one, which
        does not error and does not mean anything.
        """
        settings = self.read(self.main_path)["settings"]
        self.assertEqual(set(settings), _required_params(FRAME_ONLY))
        self.assertIsInstance(settings["tauHi"], float)
        self.assertIsInstance(settings["maxMapMeanL2"], float)
        self.assertIsInstance(settings["icpBackend"], str)
        self.assertEqual(settings["icpBackend"], "open3d")
        for name, expected in _pinned_settings(FRAME_ONLY):
            with self.subTest(parameter=name):
                self.assertAlmostEqual(settings[name], expected, places=12)

    def test_frame_status_is_the_conjunction_of_the_frames_annotations(self):
        """contracts §8.2: `{frame_index: bool}`, DERIVED (§4.5), True == usable.

        v2 read one `hw1:frameStatus` triple. v3 has one aggregate per modality and
        no frame-level node at all, so the reader ANDs them — and the key and shape
        stay identical so `reconstruct.py` did not have to change.
        """
        status = self.read(self.main_path)["frame_status"]
        self.assertEqual(set(status), set(MAIN_STEMS))
        for stem, expected in MAIN_FRAME_STATUSES.items():
            with self.subTest(frame=stem):
                self.assertIsInstance(status[stem], bool)
                self.assertEqual(status[stem], all(expected))

    def test_pair_status_is_total_and_keyed_by_the_frame_index_pair(self):
        """`{(i, j): bool}` — a tuple of ints, not a string and not an IRI.

        Total because pairs are always minted (§4.3): every consecutive pair of the
        batch has an entry whether or not a pair factor was selected.
        """
        status = self.read(self.sel_path)["pair_status"]
        self.assertEqual(set(status), set(SEL_PAIRS))
        self.assertEqual(status[SEL_BAD_PAIR], False)
        self.assertTrue(all(status[ij] for ij in SEL_PAIRS if ij != SEL_BAD_PAIR))

    def test_usable_links_are_pair_pass_and_both_endpoints_usable(self):
        """contracts §4.5's usable-link rule, on the batch whose answer is forced.

        Ten frames all passing every frame factor; one pair straddling a 0.5 m jump
        fails IdentityMedianDepthChange. So the usable links are every consecutive
        pair but that one — and the segment cut that follows is [0..7] and [8,9].
        """
        info = self.read(self.sel_path)
        self.assertEqual(list(info["usable_links"]), SEL_USABLE_LINKS)
        self.assertNotIn(SEL_BAD_PAIR, list(info["usable_links"]))

    def test_a_failing_endpoint_frame_removes_the_link_even_when_the_pair_passes(self):
        """The "AND both endpoint frames usable" half of the rule.

        On the main batch under a frame-only selection every pair is vacuously
        Pass, and only frame 0 clears all four frame factors — so every link has at
        least one bad endpoint and NOTHING is usable. A criterion that looked at the
        pair alone would happily reconstruct through a frame with no valid depth.
        """
        info = self.read(self.main_path)
        self.assertEqual(list(info["usable_links"]), [])
        self.assertTrue(all(info["pair_status"].values()),
                        "fixture is broken: no pair factor is selected, so every "
                        "pair must be vacuously Pass")

    def test_usable_links_are_ordered_by_pair_index_not_by_iri(self):
        """contracts §2: ORDER BY `hw1:pairIndex`, never by IRI string.

        The teens batch's stems straddle a decade, so the pair IRIs sort
        `10_11 < 11_12 < 8_9 < 9_10` while the trajectory runs 8,9,10,11,12. A
        reader that sorted by IRI hands `reconstruct.py` a frame list that jumps
        backwards through the capture — which costs accuracy and never errors.
        """
        links = list(self.read(self.teens_path)["usable_links"])
        self.assertEqual(links, TEENS_PAIRS)
        by_iri = sorted(TEENS_PAIRS, key=lambda ij: f"{ij[0]}_{ij[1]}")
        self.assertNotEqual(by_iri, TEENS_PAIRS,
                            "fixture is broken: IRI order must differ from index order")

    def test_two_experiment_subjects_raise_rather_than_guess(self):
        """contracts §8.2: exactly one, and "the first one" is not an answer.

        The extra subject is appended BELOW the marker on purpose — the seal covers
        the declaration bytes only (§3.1), so this file's seal still verifies and
        the reader has to reject it on the graph rather than on the digest. Both
        checks matter and only one of them fires here.
        """
        path = _append_below_marker(
            self.main_path, "read_two_subjects",
            f"\n<{api.NS}experiment/read_main_twin> a "
            f"<{api.NS}Experiment> .\n")
        self.assertEqual(_seal_of(path), _seal_of(self.main_path),
                         "fixture is broken: the declaration bytes changed")
        with self.assertRaises(ValueError):
            self.read(path)
        with self.assertRaises(ValueError):
            _require("write_run")(path, "baseline", {"mapMeanL2": 0.41})


class VacuousPassLinks(unittest.TestCase):
    """contracts §4.5's two vacuity clauses, each on the selection that isolates it.

    These are the cases v2's usable-link rule could not have: it graded four frame
    factors and two pair factors, always, so neither side was ever empty. In v3
    "usable" has to mean "nothing I ASKED ABOUT failed", and the two readings of
    that — vacuously usable frames, vacuously passing pairs — are what let a
    student study one axis at a time without the other silently cutting their data.
    """

    @classmethod
    def setUpClass(cls):
        cls.read = staticmethod(_require("read_experiment"))
        cls.pair_only = _emit_experiment("vac_pair_only", BATCH_SEL_DIRNAME,
                                         PAIR_ONLY)
        cls.frame_only = _emit_experiment("vac_frame_only", BATCH_SEL_DIRNAME,
                                          DEPTH_ONLY)
        cls.frame_only_bad = _emit_experiment("vac_frame_bad", BATCH_DIRNAME,
                                              DEPTH_ONLY)

    def test_a_pair_only_selection_makes_every_frame_vacuously_usable(self):
        """"vacuously usable when the frame has no annotations".

        There is no annotation to consult, so the frame cannot have failed
        anything. The links are then decided by the pairs alone — which is exactly
        what a student studying IdentityMedianDepthChange asked for, and the
        answer is the same forced set the fully-selected file produces.
        """
        info = self.read(self.pair_only)
        self.assertEqual(info["frame_status"], {stem: True
                                                for stem in range(SEL_FRAMES)})
        self.assertEqual(list(info["usable_links"]), SEL_USABLE_LINKS)

    def test_a_frame_only_selection_makes_every_pair_vacuously_pass(self):
        """The mirror image: no pair factor, so every pair passes, and the links are
        decided by the frames alone.

        On the selection batch every frame passes both depth factors, so EVERY
        consecutive pair is usable — including the one straddling the 0.5 m jump
        that the pair-only file cuts. The two files disagree because the two
        declarations asked different questions, and that is the feature.
        """
        info = self.read(self.frame_only)
        self.assertTrue(all(info["pair_status"].values()))
        self.assertEqual(list(info["usable_links"]), SEL_PAIRS)
        self.assertIn(SEL_BAD_PAIR, list(info["usable_links"]))

    def test_a_frame_only_selection_still_cuts_on_a_failing_frame(self):
        """... so vacuity is not "everything is usable".

        On the main batch under a depth-only selection, frames 3 and 4 fail. Every
        link touching them goes, and the two links that remain are the ones joining
        frames 0, 1 and 2 — all of which pass the two selected DEPTH factors even
        though two of them are pure black and pure white.
        """
        info = self.read(self.frame_only_bad)
        self.assertEqual(info["frame_status"],
                         {stem: MAIN_MODALITY_STATUSES[stem]["depth"]
                          for stem in MAIN_STEMS})
        self.assertEqual(list(info["usable_links"]), [(0, 1), (1, 2)])


# =============================================================================
# 12. `write_run` — contracts §8.2, the ONE writer of run triples
# =============================================================================
class RunWriteBack(unittest.TestCase):
    """`reconstruct.py`, `completeness.py` and any evaluator go through this.

    Nobody else emits a `hw1:ReconstructionRun` triple, which is what makes the
    idempotence rule enforceable at all: two writers with two delete strategies
    are two answers to "what happens to the other key". v3 adds two behavioural
    deltas and nothing else (§4.4): the seal is verified before any write, and the
    rewrite touches only the bytes below the marker.

    Every case gets its OWN experiment rather than a copy of one: write-once (§3.1)
    means a file cannot be re-assessed, and a COPY of an assessed file would carry
    an IRI tail that no longer matches its stem.
    """

    @classmethod
    def setUpClass(cls):
        cls.write_run = staticmethod(_require("write_run"))
        cls.run_iri = staticmethod(_require("run_iri"))

    def _fresh(self, name):
        stem = f"run_{name}"
        path = _emit_experiment(stem, BATCH_DEGEN_DIRNAME, DEPTH_ONLY)
        return stem, path

    def test_a_run_carries_its_type_mode_and_back_link_on_every_call(self):
        """contracts §4.4: `rdf:type`, `hw1:selectionMode` and `hw1:hasRun` are TOTAL.

        Everything else about a run is optional and readers must tolerate its
        absence — `write_run` is idempotent per key, so a run can legally exist
        carrying `coverageF` and no `mapMeanL2`. The three that are guaranteed are
        guaranteed because they are written unconditionally, which is why this is
        checked after a call that supplies only one value.
        """
        name, path = self._fresh("total")
        self.write_run(path, "baseline", {"mapMeanL2": 0.41})
        g = _graph_of(path)
        run = self.run_iri(name, "baseline")
        self.assertIn((run, RDF.type, api.HW1.ReconstructionRun), g)
        self.assertEqual(g.value(run, api.HW1.selectionMode), api.HW1.FullBatch)
        self.assertIn((api.experiment_iri(name), api.HW1.hasRun, run), g)

    def test_a_selected_run_is_the_good_segments_mode(self):
        """contracts §4.4: `baseline` -> `hw1:FullBatch`, `selected` ->
        `hw1:GoodSegments`. The mode is pinned to the IRI, so the two cannot drift
        apart and `explore`'s summary can name either."""
        name, path = self._fresh("mode")
        self.write_run(path, "selected", {"mapMeanL2": 0.30}, used_frames=[0],
                       frame_count=1)
        g = _graph_of(path)
        self.assertEqual(g.value(self.run_iri(name, "selected"),
                                 api.HW1.selectionMode), api.HW1.GoodSegments)

    def test_it_writes_the_value_and_its_status(self):
        """contracts §8.2: both, computed per §4.5 from the settings in THIS file.

        A run value with no verdict would push grading back to read time for
        exactly the two observables the whole pipeline is judged on, and the
        verdict section starts from a run that carries a `Fail` (§7.1).
        """
        name, path = self._fresh("value_and_status")
        self.write_run(path, "baseline", {"mapMeanL2": 0.81, "coverageF": 0.62})
        g = _graph_of(path)
        run = self.run_iri(name, "baseline")
        self.assertAlmostEqual(float(g.value(run, api.HW1.mapMeanL2)), 0.81, places=12)
        self.assertEqual(g.value(run, api.HW1.mapMeanL2Status), api.HW1.Fail)
        self.assertAlmostEqual(float(g.value(run, api.HW1.coverageF)), 0.62, places=12)
        self.assertEqual(g.value(run, api.HW1.coverageFStatus), api.HW1.Pass)

    def test_statusless_probe_metadata_is_written_and_replaced_per_key(self):
        """contracts §14: mechanism counts survive beside graded outcomes.

        They have no status because a splice or gate is evidence to explain, not
        another pass line for students to tune around.
        """
        name, path = self._fresh("probe_metadata")
        self.write_run(path, "selected", {"mapMeanL2": 0.81},
                       used_frames=[0], frame_count=1,
                       metadata={"gatedSteps": 3, "spliceCount": 2,
                                 "maxGapLength": 17})
        self.write_run(path, "selected", {"coverageF": 0.62},
                       metadata={"gatedSteps": 4})
        g = _graph_of(path)
        run = self.run_iri(name, "selected")
        self.assertEqual(int(g.value(run, api.HW1.gatedSteps)), 4)
        self.assertEqual(int(g.value(run, api.HW1.spliceCount)), 2)
        self.assertEqual(int(g.value(run, api.HW1.maxGapLength)), 17)
        self.assertIsNone(g.value(run, api.HW1.gatedStepsStatus))
        self.assertEqual(g.value(run, api.HW1.mapMeanL2Status), api.HW1.Fail)
        self.assertEqual(g.value(run, api.HW1.coverageFStatus), api.HW1.Pass)

    def test_probe_metadata_rejects_unknown_and_negative_values(self):
        """A typo must not mint an invisible predicate in the experiment file."""
        _name, path = self._fresh("bad_probe_metadata")
        with self.assertRaisesRegex(ValueError, "unknown run metadata"):
            self.write_run(path, "baseline", {"mapMeanL2": 0.2},
                           metadata={"gatedStep": 1})
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            self.write_run(path, "baseline", {"mapMeanL2": 0.2},
                           metadata={"gatedSteps": -1})

    def test_the_threshold_comes_from_the_settings_in_the_same_file(self):
        """contracts §8.2, and this is what makes strictness a treatment axis.

        The same 0.85 m residual is a Fail under `maxMapMeanL2=0.80` (the default
        this fixture pins) and a Pass under 0.90. Reading the threshold from a global default instead would make
        two experiments' statuses incomparable while their values stayed fine — the
        failure mode the per-experiment QualificationSetting exists to remove. In
        v3 the loose file is a SECOND declaration under a second name, which is the
        notebook discipline §3.1 enforces.
        """
        strict_name, strict = self._fresh("threshold_strict")
        self.write_run(strict, "baseline", {"mapMeanL2": 0.85})
        self.assertEqual(_graph_of(strict).value(
            self.run_iri(strict_name, "baseline"), api.HW1.mapMeanL2Status),
            api.HW1.Fail)

        loose = _emit_experiment(
            "run_threshold_loose", BATCH_DEGEN_DIRNAME, DEPTH_ONLY, pinned=False,
            settings=[("maxHighFrequencyDepthResidual", 0.05),
                      ("minValidTileCoverage", 0.50),
                      ("maxMapMeanL2", 0.90), ("minCoverageF", 0.40)])
        self.write_run(loose, "baseline", {"mapMeanL2": 0.85})
        self.assertEqual(_graph_of(loose).value(
            self.run_iri("run_threshold_loose", "baseline"),
            api.HW1.mapMeanL2Status), api.HW1.Pass)

    def test_writing_the_same_key_four_times_leaves_exactly_one_value(self):
        """contracts §8.2 idempotence — a rerun REPLACES; it never accumulates.

        Two contradictory `mapMeanL2` values for one run cannot be repaired by a
        reader: nothing in either triple says which execution wrote it. So the
        writer removes the key's triples before adding, and the status goes with
        the value — a stale `Fail` beside a fresh passing number is worse than
        either alone.
        """
        name, path = self._fresh("idempotent")
        for value in (0.41, 0.42, 0.43, 0.44):
            self.write_run(path, "baseline", {"mapMeanL2": value})
        g = _graph_of(path)
        run = self.run_iri(name, "baseline")
        self.assertEqual(len(list(g.objects(run, api.HW1.mapMeanL2))), 1)
        self.assertEqual(len(list(g.objects(run, api.HW1.mapMeanL2Status))), 1)
        self.assertAlmostEqual(float(g.value(run, api.HW1.mapMeanL2)), 0.44, places=12)
        self.assertEqual(len(list(g.objects(api.experiment_iri(name),
                                            api.HW1.hasRun))), 1)

    def test_writing_coverage_f_does_not_erase_map_mean_l2(self):
        """THE MOST IMPORTANT TEST IN THIS FILE (contracts §8.2).

        `completeness.py` computes the coverage F-score long after `reconstruct.py`
        wrote the residual, and it writes it onto the SAME run node through this
        same function. Idempotence is per (experiment, mode, KEY): "properties it
        is not writing are left intact". An implementation that cleared the run
        node first — the obvious way to make a rerun idempotent — silently deletes
        the mean-L2 of every run the moment coverage is computed, and the
        baseline-versus-selected table the assignment is graded on comes back half
        empty with no error anywhere.

        Checked in BOTH orders, because a writer that special-cased one key would
        pass the natural one.
        """
        name, forward = self._fresh("coverage_after")
        self.write_run(forward, "baseline", {"mapMeanL2": 0.81})
        self.write_run(forward, "baseline", {"coverageF": 0.62})
        g = _graph_of(forward)
        run = self.run_iri(name, "baseline")
        self.assertAlmostEqual(float(g.value(run, api.HW1.mapMeanL2)), 0.81, places=12,
                               msg="writing coverageF erased mapMeanL2")
        self.assertEqual(g.value(run, api.HW1.mapMeanL2Status), api.HW1.Fail)
        self.assertAlmostEqual(float(g.value(run, api.HW1.coverageF)), 0.62, places=12)
        self.assertEqual(g.value(run, api.HW1.coverageFStatus), api.HW1.Pass)

        back_name, backward = self._fresh("residual_after")
        self.write_run(backward, "baseline", {"coverageF": 0.62})
        self.write_run(backward, "baseline", {"mapMeanL2": 0.81})
        h = _graph_of(backward)
        back_run = self.run_iri(back_name, "baseline")
        self.assertAlmostEqual(float(h.value(back_run, api.HW1.coverageF)), 0.62,
                               places=12, msg="writing mapMeanL2 erased coverageF")
        self.assertAlmostEqual(float(h.value(back_run, api.HW1.mapMeanL2)), 0.81,
                               places=12)

    def test_a_run_may_legally_carry_coverage_and_no_residual(self):
        """The `completeness.py`-ran-first case that contracts §4.4 calls legal.

        It is the reason every reader must treat the run VALUES as optional while
        requiring the type, the mode and the back-link. Pinned so the reverse
        assumption — "a run always has a mapMeanL2" — cannot be quietly baked into
        `explore`'s summary table.
        """
        name, path = self._fresh("coverage_only")
        self.write_run(path, "baseline", {"coverageF": 0.62})
        g = _graph_of(path)
        run = self.run_iri(name, "baseline")
        self.assertIsNone(g.value(run, api.HW1.mapMeanL2))
        self.assertIsNone(g.value(run, api.HW1.mapMeanL2Status))
        self.assertIn((run, RDF.type, api.HW1.ReconstructionRun), g)
        self.assertEqual(g.value(run, api.HW1.selectionMode), api.HW1.FullBatch)

    def test_baseline_and_selected_runs_coexist_in_one_file(self):
        """contracts §6: there are still no derived experiments.

        Baseline and selected share a batch and a declaration, so in v1 they
        collided on the digest and `select` minted a `-sel-<hash>` experiment to
        keep them apart. They are two Run nodes of ONE experiment now, so they
        cannot collide by construction — and the comparison the assignment is
        graded on is a single-file read.
        """
        name, path = self._fresh("two_modes")
        self.write_run(path, "baseline", {"mapMeanL2": 0.41}, frame_count=2)
        self.write_run(path, "selected", {"mapMeanL2": 0.22},
                       used_frames=[0], frame_count=1)
        g = _graph_of(path)
        runs = set(g.objects(api.experiment_iri(name), api.HW1.hasRun))
        self.assertEqual({str(r) for r in runs},
                         {str(self.run_iri(name, "baseline")),
                          str(self.run_iri(name, "selected"))})
        self.assertAlmostEqual(
            float(g.value(self.run_iri(name, "baseline"), api.HW1.mapMeanL2)),
            0.41, places=12)
        self.assertAlmostEqual(
            float(g.value(self.run_iri(name, "selected"), api.HW1.mapMeanL2)),
            0.22, places=12)

    def test_a_good_segments_run_asserts_its_frames_and_a_full_batch_run_does_not(self):
        """contracts §4.4: reconstruction provenance lives in the graph.

        "Which frames went into this number" is answerable from the file, with no
        CSV sidecar. A full-batch run asserts none — its frame set is the batch —
        which is why `hw1:runFrameCount` has to be stated separately rather than
        counted from `usedFrame`.
        """
        name, path = self._fresh("used_frames")
        self.write_run(path, "selected", {"mapMeanL2": 0.22},
                       used_frames=[0, 1], frame_count=2)
        self.write_run(path, "baseline", {"mapMeanL2": 0.41}, frame_count=2)
        g = _graph_of(path)
        selected = self.run_iri(name, "selected")
        baseline = self.run_iri(name, "baseline")
        self.assertEqual({str(f) for f in g.objects(selected, api.HW1.usedFrame)},
                         {str(api.frame_iri(BATCH_DEGEN, 0)),
                          str(api.frame_iri(BATCH_DEGEN, 1))})
        self.assertEqual(list(g.objects(baseline, api.HW1.usedFrame)), [])
        self.assertEqual(int(g.value(selected, api.HW1.runFrameCount)), 2)
        self.assertEqual(int(g.value(baseline, api.HW1.runFrameCount)), 2)

    def test_used_frames_are_replaced_not_accumulated(self):
        """contracts §8.2: `usedFrame`/`selectionMode`/`runFrameCount` are removed
        first WHEN SUPPLIED.

        A second selected run over a re-cut segment set that merely added its
        frames would report the union of every selection ever tried, and the count
        would stop matching the frame list — with nothing to say which is stale.
        """
        name, path = self._fresh("used_replaced")
        self.write_run(path, "selected", {"mapMeanL2": 0.3}, used_frames=[0, 1],
                       frame_count=2)
        self.write_run(path, "selected", {"mapMeanL2": 0.2}, used_frames=[1],
                       frame_count=1)
        g = _graph_of(path)
        selected = self.run_iri(name, "selected")
        self.assertEqual({str(f) for f in g.objects(selected, api.HW1.usedFrame)},
                         {str(api.frame_iri(BATCH_DEGEN, 1))})
        self.assertEqual(int(g.value(selected, api.HW1.runFrameCount)), 1)

    def test_inf_is_written_as_is_and_grades_fail(self):
        """A scoreless run grades as FAILED, not as excellent and not as absent.

        `mean_l2` returns inf when ground truth is missing. inf is representable as
        xsd:double (§4.6), so it is written as-is and fails by the ordinary rule.
        Skipping the triple instead would drop the row out of every table, which
        reads as "no such run" — a different and much worse lie than "this run
        scored nothing".
        """
        name, path = self._fresh("inf")
        self.write_run(path, "baseline", {"mapMeanL2": INF})
        self.assertIn('"INF"^^xsd:double', _machine_text(path))
        g = _graph_of(path)
        literal = g.value(self.run_iri(name, "baseline"), api.HW1.mapMeanL2)
        self.assertEqual(literal.datatype, XSD.double)
        self.assertEqual(float(literal), INF)
        self.assertEqual(g.value(self.run_iri(name, "baseline"),
                                 api.HW1.mapMeanL2Status), api.HW1.Fail)

    def test_an_undeclared_run_key_raises(self):
        """Same validation philosophy as `hw1:settingParameter` (contracts §5/§8).

        `write_run` computes the status through `status_for`, which raises for a
        value property no factor is declared over. A silently-written
        `hw1:mapMeanl2` would be a number no factor grades, no table looks for and
        no student ever sees.
        """
        _name, path = self._fresh("bad_key")
        for key in ("mapMeanl2", "meanValue"):
            with self.subTest(key=key):
                with self.assertRaises(Exception) as cm:
                    self.write_run(path, "baseline", {key: 0.41})
                self.assertNotIsInstance(cm.exception, AssertionError)

    def test_the_rest_of_the_experiment_file_survives_the_rewrite(self):
        """contracts §8.2 + §3.1: rewrite the machine section, keep every triple.

        Reserialization of the MACHINE section is allowed to drop its comments and
        reorder its prefixes — it is machine generated. It must not drop TRIPLES,
        and it must not touch the student's bytes at all (`MachineSection` pins the
        second half; this is the first).

        Compared over the GROUNDED triples plus a blank-node count, not over the
        raw triple set: a student's FactorSettings are legal blank nodes (§2), and
        rdflib mints a fresh label for each of them on every parse, so a set
        difference would report every one of them as dropped-and-re-added even
        when the bytes never moved.
        """
        def grounded(g):
            return {(s, p, o) for s, p, o in g
                    if not isinstance(s, BNode) and not isinstance(o, BNode)}

        def anonymous(g):
            return sum(1 for s, _p, o in g
                       if isinstance(s, BNode) or isinstance(o, BNode))

        name, path = self._fresh("preserves")
        before_g = _graph_of(path)
        before, before_anon = grounded(before_g), anonymous(before_g)
        self.write_run(path, "baseline", {"mapMeanL2": 0.41}, frame_count=2)
        after_g = _graph_of(path)
        after, after_anon = grounded(after_g), anonymous(after_g)
        self.assertEqual(before - after, set(),
                         "write_run dropped triples it did not write")
        self.assertEqual(before_anon, after_anon,
                         "write_run dropped or duplicated a student's blank-node "
                         "FactorSetting")
        run = self.run_iri(name, "baseline")
        self.assertTrue(all(s == run or (s == api.experiment_iri(name)
                                         and p == api.HW1.hasRun)
                            for s, p, _o in after - before))

    def test_the_in_place_rewrite_moves_no_setting_and_no_verdict(self):
        """contracts §5: the rewrite re-normalises the machine section's literals,
        and that is why it is safe rather than why it is a problem.

        Normalisation is idempotent, so a machine section that has already been
        through it comes out identical — which is the claim that makes "the loss is
        harmless" true rather than merely hoped for. If a value moved here, a
        threshold could drift under a status that was computed before the drift,
        and a run written today would silently re-qualify yesterday's frames.
        """
        _name, path = self._fresh("rewrite_stable")
        read = _require("read_experiment")
        before = read(path)
        before_statuses = {(str(s), str(p)): str(o) for s, p, o in _graph_of(path)
                           if o in (api.HW1.Pass, api.HW1.Fail)}

        self.write_run(path, "baseline", {"mapMeanL2": 0.41}, frame_count=2)
        self.write_run(path, "selected", {"coverageF": 0.62}, used_frames=[0],
                       frame_count=1)

        after = read(path)
        self.assertEqual(after["settings"], before["settings"])
        self.assertEqual(after["frame_status"], before["frame_status"])
        self.assertEqual(after["pair_status"], before["pair_status"])
        after_statuses = {(str(s), str(p)): str(o) for s, p, o in _graph_of(path)
                          if o in (api.HW1.Pass, api.HW1.Fail)}
        self.assertLessEqual(set(before_statuses.items()), set(after_statuses.items()),
                             "an existing verdict changed when a run was written")


# =============================================================================
# 13. `explore` — contracts §7.1, four views, and it writes NOTHING
# =============================================================================
# The verdict section's three fixes, one per SettingRole (contracts §7.1). Matched
# as lowercase substrings rather than as whole phrases: the table LAYOUT is the
# explore track's choice and this file has no business pinning column widths — but
# the VERB is the contract, because it is the only part a student acts on.
ROLE_FIX_WORDS = {
    "GenerationSetting": ("regenerat",),
    "MeasurementSetting": ("re-measure", "remeasure"),
    "QualificationSetting": ("re-qualif", "requalif"),
}


def _number_is_printed(text, value):
    """Is `value` in `text` under any sane table formatting?

    The file stores `0.55` as `5.5e-01` (rdflib normalises every xsd:double), so a
    table prints whatever `explore` formats the parsed float as. Two, three digits
    or `%g` all read as the same number to a human; the assertion is that the
    NUMBER reached the page, not which of those the explore track chose.
    """
    return any(format(float(value), spec) in text
               for spec in (".2f", ".3f", ".4f", "g"))


class ExploreCommand(unittest.TestCase):
    """The four views of contracts §7.1, and the rule that governs all of them.

    `explore` replaced the whole SPARQL layer (§10): every shipped query became a
    projection over one self-contained file, and projections are this command's
    job. So the assertions here are about CONTENT — the batch is named, the
    selection is listed, the given/defaulted split is visible, the run values are
    printed — never about column widths or box-drawing characters.
    """

    @classmethod
    def setUpClass(cls):
        cls.batch_ttl = _ensure_batch_ttl(BATCH_DIRNAME)
        cls.gap_ttl = _run_batch2ttl(BATCH_GAP_DIRNAME,
                                     out=_out_path("explore_gap_batch.ttl"))
        cls.declaration = _declare("explore_decl", BATCH_DIRNAME, DEPTH_ONLY,
                                   label="depth only, nothing measured yet")
        cls.assessed = _emit_experiment("explore_assessed", BATCH_SEL_DIRNAME,
                                        ALL_MENU, label="the whole menu")
        _require("write_run")(cls.assessed, "baseline", {"mapMeanL2": 0.55,
                                                         "coverageF": 0.62},
                              frame_count=SEL_FRAMES)
        _require("write_run")(cls.assessed, "selected", {"mapMeanL2": 0.21,
                                                         "coverageF": 0.58},
                              used_frames=SEL_EXPECTED_SEGMENT,
                              frame_count=len(SEL_EXPECTED_SEGMENT))
        cls.other = _emit_experiment("explore_other", BATCH_SEL_DIRNAME, DEPTH_ONLY,
                                     label="depth only")

    def test_the_batch_view_prints_the_header_and_the_frame_table(self):
        """contracts §7.1 view 1 — the first thing a student runs after `batch2ttl`.

        Name, floor, path, frame count and one row per frame with both raster
        paths. It is the view that answers "did the capture I think I collected
        actually land on disk", which is why the paths are printed rather than
        summarised.
        """
        text = _run_cli(["explore", self.batch_ttl])
        self.assertIn(BATCH, text)
        self.assertIn(_batch_dir(BATCH_DIRNAME), text)
        for stem in MAIN_STEMS:
            with self.subTest(frame=stem):
                self.assertIn(f"{stem}.png", text)
        self.assertIn(str(len(MAIN_STEMS)), text)

    def test_the_batch_view_calls_out_a_gap_in_the_stem_sequence(self):
        """§7.1 view 1's last clause, and the one thing the graph cannot say.

        Stem 2 exists in `rgb/` only, so it is not a frame — and nothing in
        `batch.ttl` records that it was ever there. A student comparing "300 frames
        captured" against "297 frames listed" needs the command to say so, because
        the file will not.
        """
        text = _run_cli(["explore", self.gap_ttl]).lower()
        self.assertTrue("gap" in text or "missing" in text,
                        f"§7.1 view 1 requires gaps to be called out:\n"
                        f"{text}")

    def test_the_declaration_view_prints_the_selection_and_the_settings_plan(self):
        """§7.1 view 2 — a declaration that has NOT been assessed.

        Every required parameter with its value and whether it is student-given or
        will be defaulted. This is the view that makes write-once survivable: it
        shows the student exactly what their file is about to record, BEFORE the
        one assessment they get.
        """
        text = _run_cli(["explore", self.declaration])
        self.assertIn(BATCH, text)
        for factor in DEPTH_ONLY:
            with self.subTest(factor=factor):
                self.assertIn(factor, text)
        for name in sorted(_required_params(DEPTH_ONLY)):
            with self.subTest(parameter=name):
                self.assertIn(name, text)
        self.assertNotIn("tauHi", text,
                         "a depth-only declaration requires no tauHi (§4.2)")
        lowered = text.lower()
        self.assertTrue("default" in lowered,
                        "§7.1 view 2 prints whether each value is given or defaulted")

    def test_the_declaration_view_does_not_assess_anything(self):
        """`explore` "never measures, never writes" (§7.1).

        A view that quietly assessed the declaration it was pointed at would spend
        the file's single allowed assessment (§3.1) on a read.
        """
        before = _snapshot([self.declaration])
        _run_cli(["explore", self.declaration])
        self.assertEqual(_snapshot([self.declaration]), before)
        self.assertIsNone(_marker_offset(self.declaration))

    def test_the_assessed_view_prints_frames_pairs_settings_and_runs(self):
        """§7.1 view 3 — the per-frame table, the summary and the runs.

        This one view replaces three deleted queries (§10): `frame_quality.rq`,
        `worst_pairs.rq` and the summary half of the old attribution query. So it
        has to carry their content: a row per frame, the pair verdict toward the
        next frame, the usable-link count, the maximal segments, and both runs
        with their values and statuses.
        """
        text = _run_cli(["explore", self.assessed])
        lowered = text.lower()
        self.assertIn("explore_assessed", text)
        for factor in ALL_MENU:
            with self.subTest(factor=factor):
                self.assertTrue(factor in text or QUALITY_FACTORS[factor][0] in text,
                                f"§7.1 view 3 must report the selected factor {factor}")
        self.assertIn("baseline", lowered)
        self.assertIn("selected", lowered)
        self.assertTrue(_number_is_printed(text, 0.55),
                        f"the baseline residual is not in the table:\n{text}")
        self.assertTrue(_number_is_printed(text, 0.21),
                        f"the selected residual is not in the table:\n{text}")
        self.assertTrue("usable" in lowered,
                        "§7.1 view 3 prints the usable-link count")
        self.assertTrue("segment" in lowered,
                        "§7.1 view 3 prints the contiguous segments")

    def test_the_assessed_view_marks_given_settings_apart_from_defaulted_ones(self):
        """§7.1 view 3: "given vs defaulted read from the marker split".

        The same column as view 2, computed from a different place — position
        relative to the marker rather than presence in `given` (§4.2). A view that
        printed the vector without the split would make two experiments look
        identical when one chose its thresholds and the other inherited them.
        """
        text = _run_cli(["explore", self.assessed]).lower()
        self.assertTrue("given" in text or "student" in text)
        self.assertTrue("default" in text)

    def test_the_comparison_view_puts_two_experiments_side_by_side(self):
        """§7.1 view 4, which replaced `compare_experiments.rq` (§10).

        One column per experiment: selection differences, setting differences, run
        values and statuses. The two files here differ in exactly the two ways a
        v3 experiment can differ — a smaller selection, and therefore a smaller
        recorded vector — so a comparison that only diffed values would print
        nothing at all.
        """
        text = _run_cli(["explore", self.assessed, self.other])
        self.assertIn("explore_assessed", text)
        self.assertIn("explore_other", text)
        self.assertIn("PriorWarpDepthResidual", text,
                      "a factor selected by one experiment and not the other is a "
                      "difference §7.1 view 4 must show")

    def test_explore_writes_nothing_at_all(self):
        """§7.1's governing rule, over every view and every file it touched.

        Read-only means the mtimes do not move either: a command that rewrote a
        file with identical content would still break `hw1:declarationDigest` if it
        normalised so much as a newline above the marker.
        """
        watched = [self.batch_ttl, self.gap_ttl, self.declaration, self.assessed,
                   self.other, _ONTOLOGY_TTL]
        before = _snapshot(watched)
        for argv in (["explore", self.batch_ttl],
                     ["explore", self.gap_ttl],
                     ["explore", self.declaration],
                     ["explore", self.assessed],
                     ["explore", self.assessed, self.other]):
            _run_cli(argv)
        self.assertEqual(_snapshot(watched), before,
                         "`explore` is read-only (§7.1)")


class ExploreVerdict(unittest.TestCase):
    """§7.1 view 3's VERDICT SECTION, on a fixture engineered to fail one factor.

    This is the payoff of the whole assignment and the replacement for
    `failure_attribution.rq` (§10). The chain it has to walk:

        a run carrying a Fail
          -> the failing factors, found through `hw1:statusProperty` generically
          -> every recorded setting whose parameter's PRIMARY or AFFECTED factor
             is one of them, via `settingParameter` -> `paramPrimaryFactor` /
             `paramAffectsFactor` in the TBox (§4.5)
          -> plus the BATCH's GenerationSettings for those factors
          -> each printed with its ROLE and the role's FIX.

    The fixture makes every link on that chain necessary. The batch's one bad pair
    fails IdentityMedianDepthChange, whose threshold is a Qualification culprit;
    `depthNoiseSigma` sits on the BATCH and reaches it through
    `paramAffectsFactor`; and a failing `mapMeanL2` adds the run factor, whose
    own three parameters are required on every experiment.
    """

    NAME = "verdict_pairs"

    @classmethod
    def setUpClass(cls):
        cls.batch_ttl = _ensure_batch_ttl(BATCH_VERDICT_DIRNAME,
                                          gen=["depthNoiseSigma=0.02"])
        cls.path = _emit_experiment(cls.NAME, BATCH_VERDICT_DIRNAME, PAIR_ONLY,
                                    label="one bad pair, one bad run")
        _require("write_run")(cls.path, "baseline",
                              {"mapMeanL2": 0.90, "coverageF": 0.62},
                              frame_count=SEL_FRAMES)
        cls.text = _run_cli(["explore", cls.path])

    def test_the_fixture_really_does_fail_what_it_claims_to(self):
        """Without this the whole class could pass against an empty verdict.

        One pair fails IdentityMedianDepthChange; the baseline run fails
        ReconstructionAccuracy at 0.90 m against the declared 0.40; and coverage
        passes, so the verdict has something to leave out as well.
        """
        g = _graph_of(self.path)
        pairs = _pair_nodes(g, api.experiment_iri(self.NAME))
        failing = {ij for ij, node in pairs.items()
                   if g.value(node, api.HW1.identityMedianDepthChangeStatus)
                   == api.HW1.Fail}
        self.assertEqual(failing, {SEL_BAD_PAIR})
        run = _require("run_iri")(self.NAME, "baseline")
        self.assertEqual(g.value(run, api.HW1.mapMeanL2Status), api.HW1.Fail)
        self.assertEqual(g.value(run, api.HW1.coverageFStatus), api.HW1.Pass)

    def test_the_verdict_names_the_failing_factors(self):
        """"the failing factors with their failing-node counts".

        Named by factor or by the observable the factor is over — either reading
        identifies it — and `Coverage`, which passed, must not be in the list.
        """
        self.assertTrue("IdentityMedianDepthChange" in self.text
                        or "identityMedianDepthChange" in self.text,
                        f"the failing pair factor is not named:\n{self.text}")
        self.assertTrue("ReconstructionAccuracy" in self.text
                        or "mapMeanL2" in self.text,
                        f"the failing run factor is not named:\n{self.text}")

    def test_the_verdict_reaches_the_qualification_setting_of_the_failing_factor(self):
        """The primary-factor hop: the identity-change threshold qualifies the factor
        that failed, so it is the first thing a student can change."""
        self.assertIn("maxIdentityMedianDepthChange", self.text)
        self.assertIn("maxMapMeanL2", self.text)

    def test_the_verdict_reaches_the_factor_measurement_setting(self):
        """The pair factor's mask level is exposed as a Measurement culprit."""
        self.assertIn("changeMaskK", self.text)

    def test_the_verdict_reaches_the_batchs_generation_setting(self):
        """"plus the batch's GenerationSettings for those factors" (§7.1).

        `depthNoiseSigma` is recorded on the BATCH, not on the experiment, and it
        reaches IdentityMedianDepthChange through `paramAffectsFactor`. It is also
        the only culprit whose fix is not a number in this file: the pixels have
        to be made again.
        """
        self.assertIn("depthNoiseSigma", self.text)

    def test_each_culprit_carries_its_role_and_the_roles_fix(self):
        """contracts §7.1: the ROLE is the verdict, and the fix follows from it.

        Generation -> regenerate the data. Measurement -> change the number and
        re-measure, as a NEW experiment. Qualification -> change the threshold and
        re-qualify, as a NEW experiment. The "new experiment" half is v3's: under
        write-once (§3.1) neither of the last two can be applied to this file at
        all, so a fix that told the student to re-run `experiment` here would send
        them into a hard error.
        """
        lowered = self.text.lower()
        for role, verbs in ROLE_FIX_WORDS.items():
            with self.subTest(role=role):
                self.assertIn(role.replace("Setting", "").lower(), lowered,
                              f"no {role} row in the verdict:\n{self.text}")
                self.assertTrue(any(verb in lowered for verb in verbs),
                                f"the {role} row does not print its fix "
                                f"({' / '.join(verbs)}):\n{self.text}")

    def test_a_passing_experiment_prints_no_culprits(self):
        """... so the verdict section is not a table of every setting in the file.

        The teens batch passes everything and its run scores 0.10 m against a 0.40
        threshold. A verdict section that listed culprits anyway would train
        students to ignore it.
        """
        path = _emit_experiment("verdict_clean", BATCH_TEENS_DIRNAME, ALL_MENU)
        _require("write_run")(path, "baseline", {"mapMeanL2": 0.10, "coverageF": 0.90},
                              frame_count=len(TEENS_STEMS))
        self.assertFalse(any(o == api.HW1.Fail for _s, _p, o in _graph_of(path)),
                         "fixture is broken: this experiment must carry no Fail")
        lowered = _run_cli(["explore", path]).lower()
        for verbs in ROLE_FIX_WORDS.values():
            for verb in verbs:
                with self.subTest(verb=verb):
                    self.assertNotIn(verb, lowered,
                                     "a clean experiment was given a fix to apply")


# =============================================================================
# 14. The command surface and the single-graph invariant — contracts §3/§7/§8
# =============================================================================
class CommandSurface(unittest.TestCase):
    """Four api.py commands (contracts §7), and the flags v3 deleted.

    `experiment` takes EXACTLY ONE positional argument now. Every flag it used to
    carry described part of the experiment, and every one of them moved into the
    declaration — which is the whole v3 thesis in one signature. `declare` only
    scaffolds the student section of a declaration; it assesses nothing.
    """

    def test_exactly_four_subcommands_are_offered(self):
        """Read off the parser's own help, so a fifth cannot slip in unnoticed."""
        text = _capture_help(["--help"])
        match = re.search(r"\{([A-Za-z0-9_,\-]+)\}", text)
        self.assertIsNotNone(match, f"no subcommand list in --help:\n{text}")
        self.assertEqual(match.group(1).split(","),
                         ["batch2ttl", "declare", "experiment", "explore"])

    def test_the_dead_subcommands_are_rejected(self):
        """contracts §7/§10/§11: `query` is deleted, and so are v1's three.

        Not deprecated — gone. A `query` that still parsed would keep pyoxigraph in
        the dependency set and the `.rq` files in the tree, and the student would
        have two ways to ask the same question, one of which is no longer
        maintained.
        """
        for command in ("query", "load", "select", "result"):
            with self.subTest(command=command):
                with self.assertRaises(_REJECTED):
                    _main([command])

    def test_experiment_takes_exactly_one_positional_argument(self):
        """contracts §7. Two declarations is not a batch job, it is a typo.

        Silently assessing the first would spend the write-once budget of a file
        the student did not name.
        """
        first = _declare("surface_one", BATCH_DEGEN_DIRNAME, DEPTH_ONLY)
        second = _declare("surface_two", BATCH_DEGEN_DIRNAME, DEPTH_ONLY)
        with self.assertRaises(_REJECTED):
            _main(["experiment", first, second])
        self.assertIsNone(_marker_offset(first))
        self.assertIsNone(_marker_offset(second))

    def test_every_deleted_experiment_flag_is_rejected(self):
        """contracts §7/§11: `--set`, `--exp-id`, `--batch-dir`, `--floor`,
        `--label`, `--no-pairs`, `--out`.

        Each one carried a piece of the design on the command line, where nothing
        recorded it and nothing could review it. A flag that still parsed would let
        a sweep script keep passing `--set maxClipHiFraction=0.02` for a year while
        the declaration in the file says 0.05 — and the file is the record.
        """
        path = _declare("surface_flags", BATCH_DEGEN_DIRNAME, DEPTH_ONLY)
        for flag, value in (("--set", "tauHi=250"), ("--exp-id", "abcd1234"),
                            ("--batch-dir", _batch_dir(BATCH_DEGEN_DIRNAME)),
                            ("--floor", "1"), ("--label", "x"),
                            ("--out", _out_path("surface_flags_out.ttl")),
                            ("--no-pairs", None)):
            argv = ["experiment", path, flag] + ([value] if value else [])
            with self.subTest(flag=flag):
                with self.assertRaises(_REJECTED):
                    _main(argv)
        self.assertIsNone(_marker_offset(path))

    def test_the_experiment_usage_line_offers_one_positional_and_no_options(self):
        """contracts §7: `api.py experiment DECLARATION.ttl` and nothing else.

        Read off the USAGE line rather than the whole help text, on purpose: the
        prose underneath is a fine place to tell a student that `--set` is gone and
        where its content went, and a test that forbade the word would forbid the
        explanation. What must not appear is an OPTION — the usage line is what a
        reader copies.
        """
        text = _capture_help(["experiment", "--help"])
        match = re.search(r"usage:.*?(?:\n\n|\npositional)", text, re.S)
        self.assertIsNotNone(match, f"no usage line in:\n{text}")
        offered = set(re.findall(r"--[A-Za-z][A-Za-z0-9-]*", match.group(0)))
        self.assertLessEqual(offered, {"--help"},
                             f"§7 gives `experiment` one positional "
                             f"argument and no flags; usage offers {sorted(offered)}")

    def test_the_marker_is_exported_and_frozen(self):
        """contracts §8: `MACHINE_MARKER` is public because two modules split on it.

        `api.py` writes it and `reconstruct.py` writes below it; a private copy in
        either would be a second definition of where the student's file ends.
        """
        self.assertEqual(_require("MACHINE_MARKER"), MACHINE_MARKER)
        self.assertTrue(MACHINE_MARKER.startswith("#"),
                        "the marker must be a Turtle COMMENT or the file stops "
                        "parsing as one graph")

    def test_the_deleted_public_names_are_gone(self):
        """contracts §8's "Deleted from the surface" list plus §10's, as absence.

        Each of these is a mechanism with a v3 replacement (§11). Leaving one
        importable means another module can still reach the old model, and the two
        models disagree about what identifies an experiment and where a verdict
        comes from.
        """
        for name in ("experiment_digest", "cmd_query", "cmd_load", "cmd_select",
                     "cmd_result", "write_result", "selection_digest",
                     "build_pair_grading_query", "load_knob_declarations",
                     "parse_knob_arg", "load_result_measures", "DQV",
                     "_BANDS_TTL", "to_graph"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(api, name),
                                 f"api.{name} is deleted; see §8/§11")

    def test_the_new_public_names_are_present(self):
        """contracts §8's frozen surface — other modules import exactly these."""
        for name in ("NS", "HW1", "SCHEMA", "QUDT", "UNIT", "SKOS", "PROV",
                     "MACHINE_MARKER", "batch_name", "batch_iri", "frame_iri",
                     "component_iri", "generation_setting_iri", "experiment_iri",
                     "setting_iri", "annotation_iri", "pair_iri", "run_iri",
                     "frame_index_from_iri", "batch_name_from_frame_iri",
                     "load_parameter_declarations", "parse_setting_arg",
                     "load_quality_factors", "status_for", "read_declaration",
                     "read_experiment", "write_run", "cut_contiguous_segments"):
            with self.subTest(name=name):
                _require(name)

    def test_the_deleted_files_are_gone_from_the_tree(self):
        """contracts §3/§10's deletion lists.

        `hw1/queries/` in particular is not inert: four `.rq` files that no command
        can run are four documents a student will try to use, and the first thing
        they will reach for is the `query` command that no longer exists.
        `.oxigraph_db/` is worse — a RocksDB directory that takes a write lock.
        """
        for relative in ("queries", ".oxigraph_db", "thresholds.json",
                         "thresholds.schema.json", "run.py",
                         os.path.join("ontology", "bands.ttl")):
            with self.subTest(path=relative):
                self.assertFalse(os.path.exists(os.path.join(_HERE, relative)),
                                 f"hw1/{relative} is deleted in v3; contracts §3/§10")

    def test_pyoxigraph_is_gone_from_the_stack(self):
        """contracts §10: "the stack is rdflib-only".

        The import order comment in `reconstruct.py` went with it — pyoxigraph had
        to be imported before open3d or the process crashed, a constraint that
        outlived every explanation of it. Checked on the SOURCE rather than by
        importing, so this test says something even where the package is still
        installed in the environment — and matched on the IMPORT rather than on
        the word, because a comment recording what was deleted and why is exactly
        what a deletion should leave behind.
        """
        importer = re.compile(r"^\s*(?:import\s+pyoxigraph|from\s+pyoxigraph\b)",
                              re.MULTILINE)
        for module in ("api.py", "reconstruct.py"):
            path = os.path.join(_HERE, module)
            if not os.path.isfile(path):
                continue
            with self.subTest(module=module):
                self.assertIsNone(importer.search(_text_of(path)),
                                  f"hw1/{module} still imports pyoxigraph; "
                                  f"§10 deletes the dependency")


class SingleGraphSerialization(unittest.TestCase):
    """Nothing this codebase writes names a graph (contracts §2/§3).

    Turtle cannot carry a graph name, which is exactly why v1's loader needed a
    `to_graph` argument per file and why one missed flag emptied every named graph
    silently. v2 removed the possibility rather than the mistake, and v3 goes
    further: with the query layer deleted there is no store to load anything into.
    """

    @classmethod
    def setUpClass(cls):
        cls.paths = [_run_batch2ttl(BATCH_DEGEN_DIRNAME, out=_out_path("sg_batch.ttl")),
                     _emit_experiment("sg_exp", BATCH_DEGEN_DIRNAME, DEPTH_ONLY)]

    def test_no_emitted_ttl_contains_a_graph_or_trig_construct(self):
        """A `{` in a Turtle file is a TriG dataset block and nothing else.

        Blank-node property lists use `[ ]` and collections use `( )`, so a brace
        anywhere in a file this project writes means someone reached for a named
        graph again.
        """
        for path in self.paths:
            with self.subTest(path=os.path.basename(path)):
                self.assertNotIn("{", _text_of(path),
                                 "TriG graph block in a Turtle file")

    def test_an_experiment_file_and_its_batch_file_load_as_one_graph(self):
        """The join v1 needed named graphs for, done by parsing two files into one
        graph.

        The experiment's annotations point at frame IRIs the batch file defines,
        and `hw1:describesImage` points at the image nodes it defines. Both joins
        have to close with no GRAPH clause and no store — which is what makes
        `explore` a hundred lines of rdflib instead of a database.
        """
        merged = Graph()
        for path in self.paths:
            merged.parse(path, format="turtle")
        for node in merged.subjects(RDF.type, api.HW1.FrameAnnotation):
            frame = merged.value(node, api.HW1.annotatesFrame)
            image = merged.value(node, api.HW1.describesImage)
            with self.subTest(annotation=str(node)):
                self.assertIn((frame, RDF.type, api.HW1.Frame), merged)
                self.assertIsNotNone(merged.value(image, api.SCHEMA.contentUrl))


if __name__ == "__main__":
    unittest.main(verbosity=2)
