"""
HW1 data-quality triplestore CLI — ingest frames into Fuseki, query PASS frames via SPARQL.

WHAT THIS FILE IS
    A three-subcommand command-line tool that bridges a directory of captured SLAM
    frames to an Apache Jena Fuseki triplestore over the SPARQL 1.1 protocol
    (Graph Store HTTP + Query), using `rdflib` to build and serialize the RDF.

    * `insert`   — pair the rgb/ + depth/ frames of a batch, measure each frame's
                   per-frame observables, build an RDF graph matching
                   `ontology/hw1.ttl`, and PUT it into Fuseki as ONE named graph
                   per batch (re-inserting a batch REPLACES it).
    * `retrieve` — run a SPARQL SELECT that grades every frame of a named batch
                   against the four one-sided bands and writes the PASS frames to
                   a CSV. The grading lives in the SPARQL FILTER, not in Python.
    * `compare`  — run the cross-named-graph join of `queries/compare_batches.rq`:
                   two batches on one trajectory, joined on frameIndex, per-frame
                   observable deltas out. The query a dataframe filter cannot be.

FOUR QUALITY FACTORS + ONE BASELINE  (all single-frame, all single-sample
computable, therefore all checkable at inference time)

    depth-validity         `valid_depth_fraction`    in [0,1]      band: >= min
    depth roughness        `depth_roughness`         >= 0, METRES  band: <= max
    highlight clipping     `clip_hi_fraction`        in [0,1]      band: <= max
    shadow clipping        `clip_lo_fraction`        in [0,1]      band: <= max

    Exactly ONE of the four points up. `valid_depth_fraction` is higher-is-better;
    the other three are lower-is-better. Read every band direction off this table
    before you write the FILTER — three of the four comparisons are `<=`.

    plus the deliberately weak baseline `mean_value` in [0,255], carried into the
    store so the "did a clip factor beat the mean?" comparison is a query, not a
    hand calculation. It is NOT a quality factor and has NO band.

    Both RGB factors are computed on the value channel V = max(R,G,B). There is no
    colorimetric weighting anywhere in this assignment: see `definitions.md`, and
    see `frame_mean_value` for why the mean ships only as a baseline to beat.

THRESHOLDS  (`hw1/thresholds.json` — you author it; it is NOT in the repo)
    No threshold is hardcoded anywhere in this file. One FLAT JSON file — one band
    set for the whole assignment, not one per floor — carries BOTH kinds of number:

        {"tau_lo": ..., "tau_hi": ...,               <- MEASUREMENT parameters
         "bands": {"valid_depth_fraction_min": ...,   <- GATING parameters
                   "depth_roughness_max": ...,
                   "clip_hi_fraction_max": ...,
                   "clip_lo_fraction_max": ...}}

    See `load_thresholds` (authoritative) and `thresholds.schema.json`.

    tau is read at INSERT time; the bands are read at RETRIEVE time. Consequence:
    change tau and every clip fraction already in the store is STALE — it answers a
    question you are no longer asking — and the batch must be re-inserted. That is
    safe and idempotent because `insert` PUTs one named graph per batch, replacing
    it wholesale. Changing a band needs no re-insert at all: re-run `retrieve`.
    tau_hi/tau_lo are recorded on the Batch node so a stored fraction stays
    interpretable from the store alone. They are the ONLY measurement parameters
    that need recording: the two depth factors are fully determined by the raster
    and the fixed validity rule, and `depth_roughness` has no free parameters.

ONTOLOGY CONTRACT  (must match ontology/hw1.ttl — see NS/HW1 below)
    Classes: Batch, Frame, RGBImage, DepthImage, QualityFactor.
             RGBImage/DepthImage are rdfs:subClassOf schema:ImageObject.
    Object props: hasFrame (Batch->Frame), hasRGBImage (Frame->RGBImage),
                  hasDepthImage (Frame->DepthImage).
    Batch:      batchName, batchPath, floor,
                tauHi, tauLo               <- measurement provenance, you derive these
    RGBImage:   meanValue, clipHiFraction, clipLoFraction, schema:contentUrl.
    DepthImage: validDepthFraction, depthRoughness, schema:contentUrl.
    Five observables per frame, two provenance properties per batch.
    IRI scheme: batch  = <ns>batch/<name>
                frame  = <ns>batch/<name>/frame/<n>
                rgb    = <ns>batch/<name>/frame/<n>/rgb
                depth  = <ns>batch/<name>/frame/<n>/depth
                <name> = f"floor{floor}_{basename(data_dir)}" (see `batch_name`);
                it is ALSO the named-graph IRI's last segment and the value of
                hw1:batchName, i.e. what you pass to `--batch`.

DEPTH FORMAT
    Depth PNGs are uint16 millimetres; metres = raw / 1000.0. A pixel is valid iff
    raw != 0 AND min_range <= metres <= max_range.

ENDPOINT
    Default `http://localhost:3030/ds`. Query `<endpoint>/query`, Graph Store
    `<endpoint>/data`. This module NEVER starts the server and does not require
    one to import.

DEPENDENCIES
    Standard library + numpy + Pillow + rdflib. No scipy, no OpenCV, no Open3D.
    Every quality factor is computed DIRECTLY from the raw depth/RGB PNG — no
    unprojection, no point cloud, no normal estimation — which is what keeps this
    list short and what makes the factors grade the INPUT DATA rather than a
    reconstruction pipeline that may itself be unimplemented. Nothing here imports
    `utils.py` either, so an unfinished ICP cannot block this file and an
    unfinished measurer cannot block the ICP.

SEE ALSO
    queries/valid_frames.rq     — SPARQL SELECT template `retrieve` fills + runs
    queries/compare_batches.rq  — cross-batch join `compare` fills + runs
    ontology/hw1.ttl            — the TBox these triples must satisfy
    definitions.md              — the four measurement contracts and their sources
    test_e2e.py                 — the executable specification of this file
"""

import argparse
import csv
import glob
import json
import os
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLStore

# =============================================================================
# Ontology namespace  (must match ontology/hw1.ttl exactly — do not rename)
# =============================================================================
NS = "http://taica.course/hw1/ontology#"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Path to hw1/ontology/hw1.ttl relative to this file (this file lives in hw1/).
_ONTOLOGY_TTL = os.path.join(_HERE, "ontology", "hw1.ttl")

# Directory of SPARQL query templates (*.rq), decoupled from this module.
_QUERY_DIR = os.path.join(_HERE, "queries")

# The thresholds file YOU author. It is deliberately absent from the repo.
DEFAULT_THRESHOLDS = os.path.join(_HERE, "thresholds.json")

DEFAULT_ENDPOINT = "http://localhost:3030/ds"

# Depth PNG encoding: uint16 millimetres.
_DEPTH_SCALE = 1000.0

# =============================================================================
# Immerkaer noise-estimation constants — the measurement contract of
# `frame_depth_roughness`. FIXED: they are NOT yours to derive and they are NOT
# in thresholds.json (you derive exactly six numbers — tau_lo, tau_hi and the
# four band bounds). Unlike the tau pair these need no batch-level provenance,
# because they are not free parameters: they are the kernel and the two constants
# of one published estimator. Read these names in your measurer; do not paste the
# numbers inline.
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

# Sanity ceiling on the depth-roughness BAND bound in thresholds.json. The
# observable is a noise level in METRES, so a bound above the sensor's whole
# usable range is not a bound anyone means — it is a unit error (millimetres
# written where metres belong).
_ROUGHNESS_SANITY_MAX = 10.0


# =============================================================================
# thresholds.json  — the ONE place a number you derived may live
# =============================================================================
_SCHEMA_HELP = """\
thresholds.json must look exactly like this (all six numbers are yours to derive;
the file is FLAT — one band set for the whole assignment, NOT one per floor):

{
  "tau_lo": 20,                                 # measurement: V <= tau_lo is crushed
  "tau_hi": 240,                                # measurement: V >= tau_hi is saturated
  "bands": {
    "valid_depth_fraction_min": 0.50,           # gate: keep frames with vdf   >= this
    "depth_roughness_max": 0.02,                # gate: keep frames with rough <= this
    "clip_hi_fraction_max": 0.10,               # gate: keep frames with chi   <= this
    "clip_lo_fraction_max": 0.10                # gate: keep frames with clo   <= this
  }
}

Every band key is <observable_name>_<min|max>, so each one names the exact
observable it bounds. The SHORT spellings (valid_depth_min, roughness_max, ...)
are the `retrieve` CLI override flags — a separate namespace. Do not use them here.

0 <= tau_lo < tau_hi <= 255.  valid_depth_fraction_min and the two clip bounds are
fractions in [0,1]; depth_roughness_max is a noise level in METRES, >= 0 (see
frame_depth_roughness).  All four bands are ONE-SIDED, but they do NOT all point
the same way: valid_depth_fraction_min is the only LOWER bound, the other three
are upper bounds.  There is no per-floor variant: applying the floor-1 bands to
your floor-2 capture and reporting whether they transferred IS the phase-2
deliverable.
The numbers above are SHAPE, not answers — deriving them is the assignment.
See thresholds.schema.json for the machine-readable schema."""

# resolve_band's fixed output order — the same order as the query's band tokens
# @@DMIN@@ @@RMAX@@ @@CLIPHI@@ @@CLIPLO@@.
_BAND_KEYS = ("valid_depth_fraction_min", "depth_roughness_max",
              "clip_hi_fraction_max", "clip_lo_fraction_max")
_BAND_RANGES = {"valid_depth_fraction_min": (0.0, 1.0),
                "depth_roughness_max": (0.0, _ROUGHNESS_SANITY_MAX),
                "clip_hi_fraction_max": (0.0, 1.0),
                "clip_lo_fraction_max": (0.0, 1.0)}
# CLI override attribute per band, same order as _BAND_KEYS. These are the
# `retrieve` flag names, NOT the thresholds.json keys (which are _BAND_KEYS).
_BAND_CLI = ("valid_depth_min", "roughness_max", "clip_hi_max", "clip_lo_max")


def load_thresholds(path=None):
    """Load, validate and return the thresholds dict from `path` (default
    DEFAULT_THRESHOLDS = hw1/thresholds.json).  THIS IS THE AUTHORITATIVE
    STATEMENT OF THE SCHEMA:

        {"tau_lo": <0..255>, "tau_hi": <0..255>,
         "bands": {"valid_depth_fraction_min": <0..1>,
                   "depth_roughness_max":      <0..10>,   metres
                   "clip_hi_fraction_max":     <0..1>,
                   "clip_lo_fraction_max":     <0..1>}}

    The band keys are <observable_name>_<min|max>.  The short names on the
    `retrieve` CLI (--valid-depth-min, --roughness-max, ...) are override FLAGS,
    not keys of this file; nothing here accepts them.

    Flat: ONE band set, not one per floor.  `--floor` labels a batch, it does not
    select a band set.  Requires tau_lo < tau_hi.  Every key is required and there
    are no defaults anywhere: a missing file or a missing key raises with the
    schema in the message rather than quietly filling in a number, because a
    silently defaulted threshold is a number nobody derived and nobody can defend.

    Returns the parsed dict (floats), plus `_path` for provenance in logs.
    Raises FileNotFoundError (absent file) or ValueError (malformed).
    """
    path = path or DEFAULT_THRESHOLDS
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"thresholds file not found: {path}\n\n{_SCHEMA_HELP}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            th = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})\n\n{_SCHEMA_HELP}") from exc
    if not isinstance(th, dict):
        raise ValueError(f"{path}: top level must be a JSON object\n\n{_SCHEMA_HELP}")

    def _num(container, key, lo, hi, where):
        if key not in container:
            raise ValueError(f"{path}: missing required key {where}{key!r}\n\n{_SCHEMA_HELP}")
        val = container[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{path}: {where}{key!r} must be a number, got {val!r}")
        val = float(val)
        if not lo <= val <= hi:
            raise ValueError(f"{path}: {where}{key!r} = {val} is outside [{lo}, {hi}]")
        return val

    tau_lo = _num(th, "tau_lo", 0.0, 255.0, "")
    tau_hi = _num(th, "tau_hi", 0.0, 255.0, "")
    if not tau_lo < tau_hi:
        raise ValueError(f"{path}: need tau_lo < tau_hi, got {tau_lo} >= {tau_hi}")

    bands_in = th.get("bands")
    if not isinstance(bands_in, dict):
        raise ValueError(f"{path}: missing required object 'bands'\n\n{_SCHEMA_HELP}")
    bands = {k: _num(bands_in, k, *_BAND_RANGES[k], "bands.") for k in _BAND_KEYS}

    return {"tau_lo": tau_lo, "tau_hi": tau_hi, "bands": bands,
            "_path": os.path.abspath(path)}


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
        over-exposure — with one interval, so any band around it widens until it
        admits both;
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


def frame_valid_fraction(depth_path, min_range=0.0, max_range=10.0):
    """Quality factor 1 — fraction of usable depth pixels in one depth frame.

    CONTRACT
        In:     depth_path — a depth PNG, uint16 MILLIMETRES (metres = raw/1000.0).
                min_range, max_range — the sensor's usable range, in METRES,
                inclusive on both ends.
        Out:    one float in [0, 1]: (number of valid pixels) / (total pixels).
        Valid:  a pixel is valid iff raw != 0 AND min_range <= metres <= max_range.
                raw == 0 is the sensor's "no return" code, not a 0 m reading.
        Band:   one-sided, >= min. More valid depth is better.
        Purity: no randomness, no global state; the same file always gives the
                same number.
    """
    #TODO


def frame_depth_roughness(depth_path, min_range=0.0, max_range=10.0):
    """Quality factor 2 — a REFERENCE-FREE estimate of the depth raster's noise.

    CONTRACT
        In:     depth_path — the same uint16-millimetre depth PNG as above.
                min_range, max_range — usable range in METRES; a pixel is valid
                under exactly the rule frame_valid_fraction uses.
        Out:    one float >= 0: the estimated per-pixel depth noise standard
                deviation, in METRES. On D = raw / 1000.0,

                    M = [[ 1, -2,  1],
                         [-2,  4, -2],
                         [ 1, -2,  1]]        Immerkaer 1996; ||M|| = 6

                    R = D * M                 (see the mask rule below)
                    return median(|R|) / (6 * 0.6745)

                Read _IMMERKAER_M / _IMMERKAER_NORM / _MAD_TO_SIGMA rather than
                pasting the numbers inline.
        Band:   one-sided, <= max. LOWER IS BETTER — the OPPOSITE direction to
                valid_depth_fraction. Of the four bands, that one is the only
                lower bound and this is one of three upper bounds.
        Units:  metres, so the band is a physical noise level you can sanity-check
                against the sensor model instead of against an arbitrary scale.
        Edge:   a frame with NO fully-valid 3x3 window at all — total dropout, or a
                raster smaller than 3x3 — returns float("inf"). See below; this is
                a contract requirement, not an implementation detail.
        Purity: deterministic and pure — numpy + Pillow, no point cloud, no
                normals, no second frame. Same file, same number, every time.

    WHY IT WORKS
        M is the difference of two discrete Laplacians, so it annihilates any
        locally-linear depth surface: a plane at any tilt, at any distance,
        contributes exactly nothing. What survives is the high-frequency residual,
        and for i.i.d. noise of standard deviation sigma the response has standard
        deviation ||M|| * sigma = 6 * sigma. Dividing by 6 * 0.6745 — where 0.6745
        is the median of |N(0,1)| — converts the robust spread of |R| back to
        sigma. Hand this estimator a frame with noise of a known sigma and it
        returns that sigma; that is how test_e2e.py checks it.

    TWO CONTRACT POINTS THAT ARE NOT OPTIONAL
    Get either wrong and you have silently built a different metric that still
    looks plausible and still produces numbers:

      1. THE FULLY-VALID-WINDOW MASK. Evaluate R ONLY at windows whose NINE pixels
         are all valid. Dropout is encoded as raw == 0, i.e. as a 0 m reading
         sitting next to a 3 m one, and the mask reads that as an enormous edge.
         Skip the nine-pixel test and your estimate is driven by how much dropout
         the frame has — factor 2 collapses into a restatement of factor 1, and
         your two depth factors stop being two pieces of evidence.

      2. THE MEDIAN, NOT THE MEAN. The literature form is sqrt(pi/2)/6 * mean|R|.
         Use the median instead: a real depth frame contains real discontinuities
         — an object boundary, a doorway — and each one produces a large Laplacian
         response that is SIGNAL, not noise. A mean is moved by a handful of such
         windows; a median is not. test_e2e.py hands you a frame with a hard step
         edge and no noise: the median form returns 0, either mean form does not.

    WHY THE DEGENERATE FRAME RETURNS inf, NOT 0
        A frame with no fully-valid window has no noise estimate at all, and the
        two available "nothing to report" values are not symmetric. This band is
        <= max, so 0.0 means BEST and would let a frame with no usable depth
        whatsoever sail straight through the roughness gate. inf fails every
        finite band, so the gate rejects it. When a measurement is impossible, a
        quality filter must fail closed. (Do not argue that factor 1 rejects such
        a frame anyway: that reasoning couples the two factors, and keeping them
        independent is the whole point of having two.) inf is representable as
        xsd:double, rdflib serialises it, and a SPARQL FILTER(?r <= 0.05) is false
        against it, so the frame drops out of the PASS set exactly as intended.

    SOURCE
        Immerkaer, "Fast Noise Variance Estimation", CVGIP: Graphical Models and
        Image Processing 58(2), 1996 — the standard reference-free noise
        estimator, and the same kernel Shin et al., IROS 2019 use for the noise
        term of their image quality measure.
    """
    #TODO


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

        Band:   one-sided, <= max. Less blown-out is better.
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
    """
    #TODO


def frame_clip_lo_fraction(rgb_path, tau_lo):
    """RGB quality factor 2 (ShadowClipping) — crushed pixel fraction.

    CONTRACT
        In:     rgb_path — an RGB PNG.
                tau_lo — the shadow threshold on the 0-255 value scale.  NO
                DEFAULT, never hardcoded, for the same reason as tau_hi: it is a
                measurement parameter, and it is recorded on the Batch node.
        Out:    one float in [0, 1]:

                    clip_lo_fraction = |{ V <= tau_lo }| / N ,   V = max(R,G,B)

        Band:   one-sided, <= max. Less crushed is better.
        Purity: deterministic; no randomness, no global state.

    THE TWO CLIP FACTORS USE THE SAME OPERATOR AND MEAN DIFFERENT QUANTIFIERS
    Same operator as the highlight twin, different meaning — do not "fix" it:

        V >= tau_hi   <=>   AT LEAST ONE channel is saturated      (exists)
        V <= tau_lo   <=>   EVERY channel is crushed               (for all)

    SOURCE
        Shin et al., IROS 2019, eq. 7 — same source as the highlight factor.
    """
    # WHY NOT min(R,G,B) <= tau_lo, the apparently symmetric alternative?  Because
    # it is wrong: pure red (255,0,0) has G = B = 0, so a fully saturated pixel
    # would be counted as crushed.  max(R,G,B) <= tau_lo is the test that means
    # "every channel is dark", which is what a crushed pixel actually is.
    #TODO


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
# IRI helpers  (batch name = named-graph IRI too)
# =============================================================================
def batch_name(data_dir, floor):
    """Floor-qualified batch name: f"floor{floor}_{basename(data_dir)}".

    Both floors ship captures with the same directory names, so a basename alone is
    not a unique key: floor 1's `mixed_dev` and floor 2's `mixed_dev` would map to
    ONE named graph and silently overwrite each other — and comparing the two
    floors is exactly what phase 2 asks you to do.  This name is what goes into
    hw1:batchName, into the batch IRI, and into `--batch` on retrieve/compare.
    """
    return f"floor{int(floor)}_{os.path.basename(os.path.normpath(data_dir))}"


def batch_iri(name):
    """IRI of a batch node / its named graph: <ns>batch/<name>."""
    return URIRef(f"{NS}batch/{name}")


def _frame_iri(name, idx):
    return URIRef(f"{NS}batch/{name}/frame/{idx}")


def _component_iri(name, idx, kind):
    # kind in {"rgb", "depth"}
    return URIRef(f"{NS}batch/{name}/frame/{idx}/{kind}")


# =============================================================================
# insert  — build the batch RDF graph and PUT it into Fuseki as a named graph
# =============================================================================
def build_batch_graph(data_dir, floor, thresholds, include_ontology=True):
    """Build the in-memory rdflib.Graph of one batch directory.

    CONTRACT
        In:     data_dir — a directory with rgb/ and depth/ subdirs (see
                _pair_frames; only stems present in BOTH are frames of this batch).
                floor — the floor label, an int; see `batch_name`.
                thresholds — the dict from load_thresholds(); tau_lo/tau_hi are
                read HERE, at insert time, and are the tau every clip fraction in
                this graph is counted against.
                include_ontology — if true and ontology/hw1.ttl exists, parse the
                TBox into the same graph so the pushed named graph is
                self-describing.
        Out:    (graph, name, batch_iri) where name = batch_name(data_dir, floor)
                and batch_iri = batch_iri(name).

        The graph must satisfy the TBox in ontology/hw1.ttl exactly — FIVE
        observables per frame, each on the node whose domain declares it, plus TWO
        provenance properties on the batch. Put an observable on the wrong node and
        the retrieve query's pattern simply will not match:

            Batch       batchName (= name), batchPath, floor,
                        tauHi, tauLo               <- provenance, from `thresholds`
            Frame       frameIndex, one per paired stem, linked by hasFrame
            RGBImage    meanValue, clipHiFraction, clipLoFraction,
                        schema:contentUrl,  linked by hasRGBImage
            DepthImage  validDepthFraction, depthRoughness,
                        schema:contentUrl,  linked by hasDepthImage

        Numeric literals are xsd:double (frameIndex and floor are xsd:integer).
        Store the string properties as PLAIN literals: in RDF 1.1 a plain literal
        IS an xsd:string, and this is what makes hw1:batchName "name" match on both
        Fuseki and rdflib's stricter in-memory matcher.

        WHY THE TWO PROVENANCE PROPERTIES. Without tauHi/tauLo the store holds
        clip fractions whose meaning is unrecoverable from the store itself, and
        two batches measured at different tau look directly comparable when they
        are not. Read them off `thresholds` — the same dict the measurers are
        called with — rather than re-deriving them here: a literal that drifts
        from the value actually applied makes the store assert something FALSE,
        which is strictly worse than recording nothing at all.

        The two DEPTH factors need no such record, and that is a property of the
        factors rather than an omission: both are fully determined by the raster
        plus the fixed validity rule, and depth_roughness has no free parameters
        at all.
    """
    #TODO


def put_named_graph(graph, batch_uri, endpoint):
    """Replace the named graph <batch_uri> in Fuseki with `graph` via a Graph Store
    HTTP PUT of Turtle to <endpoint>/data?graph=<batch_uri>. Requires a running server.

    PUT (not POST) is what makes re-`insert` idempotent, and what makes changing tau
    safe: the batch's graph is replaced wholesale, so stale fractions cannot survive
    alongside fresh ones.
    """
    turtle = graph.serialize(format="turtle")
    if isinstance(turtle, str):
        turtle = turtle.encode("utf-8")
    url = f"{endpoint}/data?graph=" + urllib.parse.quote(str(batch_uri), safe="")
    req = urllib.request.Request(url, data=turtle, method="PUT",
                                 headers={"Content-Type": "text/turtle"})
    with urllib.request.urlopen(req) as resp:  # nosec - localhost triplestore
        return resp.status


def cmd_insert(args):
    th = load_thresholds(getattr(args, "thresholds", None))
    g, name, b = build_batch_graph(args.data_dir, args.floor, th)
    n = len(g)
    put_named_graph(g, b, args.endpoint)
    print(f"[insert] batch {name!r}: measured at tau_lo={th['tau_lo']}, tau_hi={th['tau_hi']}")
    print(f"[insert] pushed {n} triples")
    print(f"[insert] named graph: {b}")
    return 0


# =============================================================================
# retrieve  — SPARQL SELECT of PASS frames (grading lives in the FILTER)
# =============================================================================
def resolve_band(thresholds, args=None):
    """Resolve the effective four-bound band.

    CONTRACT
        In:     thresholds — the dict from load_thresholds(); the bands are read
                HERE, at retrieve time, which is why re-deriving a band costs one
                re-run of `retrieve` and no re-insert at all.
                args — the parsed CLI namespace, or None.  Its four override
                attributes are `valid_depth_min`, `roughness_max`, `clip_hi_max`,
                `clip_lo_max` — the SHORT CLI spellings, not the thresholds.json
                keys; each is None when not given, and a non-None override WINS
                over the file, so a sweep can move one bound without editing
                thresholds.json.
        Out:    a 4-tuple of floats in exactly this order

                    (valid_depth_min, roughness_max, clip_hi_max, clip_lo_max)

                which is the order of the query's band tokens
                @@DMIN@@ @@RMAX@@ @@CLIPHI@@ @@CLIPLO@@ — build_select relies on it.
        Bands:  all four are ONE-SIDED, but they do NOT all point the same way:
                the FIRST is a lower bound (>=) and the other THREE are upper
                bounds (<=). This function carries only the numbers; which side
                each comparison is on lives in the FILTER of
                queries/valid_frames.rq.
        No defaults: every number traces back to thresholds.json or to an explicit
                CLI flag. There is nothing to fall back to.
    """
    #TODO


def load_query(name):
    """Read a SPARQL query template `<name>.rq` from hw1/queries/ and return its text."""
    path = os.path.join(_QUERY_DIR, f"{name}.rq")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def build_select(batch, dmin, rmax, cliphi, cliplo):
    """Build the runnable SPARQL SELECT for one batch from queries/valid_frames.rq.

    CONTRACT
        In:     batch — the floor-qualified batch name (see `batch_name`).
                dmin, rmax, cliphi, cliplo — the four bounds from resolve_band(),
                in that order.
        Out:    the template text of queries/valid_frames.rq with all SIX
                substitution tokens replaced, and therefore no "@@" left in it:

                    @@NS@@       the ontology namespace IRI
                    @@BATCH@@    the batch name, SPARQL-escaped
                    @@DMIN@@     valid-depth-fraction LOWER bound   (float literal)
                    @@RMAX@@     depth-roughness UPPER bound, metres(float literal)
                    @@CLIPHI@@   clip-hi-fraction upper bound       (float literal)
                    @@CLIPLO@@   clip-lo-fraction upper bound       (float literal)

        Safety: the four bounds are floats this module controls, so injecting them
                inline is safe. The batch name is a string from the command line
                and is NOT: escape it (backslash and double-quote) and substitute
                it LAST, after every trusted token has been consumed — otherwise a
                batch named "@@CLIPHI@@" could rewrite the query.
        The result must bind exactly ?idx ?rgb ?depth ?vdf ?rough ?cliphi ?cliplo,
                because cmd_retrieve reads those names off each result row.

        `build_compare` below is the same substitution pattern with three tokens
        and ships implemented — read it if the mechanics are unclear.
    """
    #TODO


def build_compare(batch_a, batch_b):
    """Build the cross-batch SELECT from queries/compare_batches.rq.

    Three substitution tokens: @@NS@@, @@BATCH_A@@ (the reference batch) and
    @@BATCH_B@@ (the compared batch). Both names are floor-qualified, escaped, and
    substituted after @@NS@@, for the same reason build_select escapes its batch
    name last. Neither name is ever hardcoded — the caller passes whichever two
    batches it means. The column set is whatever the query's SELECT declares.
    """
    query = load_query("compare_batches").replace("@@NS@@", NS)
    for token, name in (("@@BATCH_A@@", batch_a), ("@@BATCH_B@@", batch_b)):
        query = query.replace(token, name.replace("\\", "\\\\").replace('"', '\\"'))
    return query


def run_select(query, endpoint):
    """Run a SPARQL SELECT against <endpoint>/query and return the rdflib Result.
    Requires a running server.
    """
    store = SPARQLStore(query_endpoint=f"{endpoint}/query")
    g = Graph(store)
    return g.query(query)


CSV_COLUMNS = ["frame", "rgb_path", "depth_path", "valid_fraction",
               "depth_roughness", "clip_hi_fraction", "clip_lo_fraction"]


def cmd_retrieve(args):
    th = load_thresholds(getattr(args, "thresholds", None))
    dmin, rmax, cliphi, cliplo = resolve_band(th, args)
    print(f"[retrieve] batch {args.batch!r}: effective band valid_depth >= {dmin}, "
          f"roughness <= {rmax}, clip_hi <= {cliphi}, clip_lo <= {cliplo}")

    result = run_select(build_select(args.batch, dmin, rmax, cliphi, cliplo), args.endpoint)

    rows = sorted((int(r.idx), str(r.rgb), str(r.depth), float(r.vdf),
                   float(r.rough), float(r.cliphi), float(r.cliplo)) for r in result)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)

    print(f"[retrieve] {len(rows)} PASS frames -> {args.out}")
    return 0


def cmd_compare(args):
    """Run the two-named-graph join and print (optionally CSV) the per-frame deltas.

    Column-agnostic on purpose: it prints whatever variables your query's SELECT
    declares, so changing compare_batches.rq's projection needs no change here.
    """
    result = run_select(build_compare(args.batch_a, args.batch_b), args.endpoint)
    cols = [str(v) for v in result.vars] if result.vars else []
    rows = [[("" if row[c] is None else str(row[c].toPython())) for c in result.vars]
            for row in result]

    print(f"[compare] {args.batch_a!r} (A) vs {args.batch_b!r} (B), deltas are B - A")
    print("  " + "\t".join(cols))
    for row in rows:
        print("  " + "\t".join(row))
    print(f"[compare] {len(rows)} joined frames")

    if getattr(args, "out", None):
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
        print(f"[compare] -> {args.out}")
    return 0


# =============================================================================
# CLI
# =============================================================================
def _build_parser():
    p = argparse.ArgumentParser(
        description="HW1 data-quality triplestore CLI (Fuseki + rdflib): insert batches, "
                    "retrieve PASS frames, compare two batches — all via SPARQL.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("insert", help="Measure a batch's frames and push them into Fuseki "
                                        "as one named graph.")
    ins.add_argument("--data-dir", required=True,
                     help="Directory containing rgb/ and depth/ subdirs of integer-stem .png frames.")
    ins.add_argument("--floor", type=int, default=1,
                     help="Floor this capture is from (default 1). Stored on the Batch node AND "
                          "prefixed onto the batch name, so floor1_mixed_dev and floor2_mixed_dev "
                          "are distinct graphs. It does NOT select a band set — there is only one.")
    ins.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                     help=f"Fuseki dataset endpoint (default {DEFAULT_ENDPOINT}).")
    ins.add_argument("--thresholds", default=None,
                     help=f"thresholds.json path (default {DEFAULT_THRESHOLDS}). tau_lo/tau_hi are "
                          "read HERE, at insert time; change them and you must re-insert.")
    ins.set_defaults(func=cmd_insert)

    ret = sub.add_parser("retrieve", help="SPARQL-query a batch's PASS frames (inside the band) "
                                          "to a CSV.")
    ret.add_argument("--batch", required=True,
                     help="Floor-qualified batch name (hw1:batchName), e.g. floor1_mixed_dev.")
    ret.add_argument("--out", required=True, help="Output CSV path.")
    ret.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                     help=f"Fuseki dataset endpoint (default {DEFAULT_ENDPOINT}).")
    ret.add_argument("--thresholds", default=None,
                     help=f"thresholds.json path (default {DEFAULT_THRESHOLDS}). The BANDS are read "
                          "here, at retrieve time; changing one needs no re-insert.")
    ret.add_argument("--valid-depth-min", dest="valid_depth_min", type=float, default=None,
                     help="Override bands.valid_depth_fraction_min.")
    ret.add_argument("--roughness-max", dest="roughness_max", type=float, default=None,
                     help="Override bands.depth_roughness_max (metres).")
    ret.add_argument("--clip-hi-max", dest="clip_hi_max", type=float, default=None,
                     help="Override bands.clip_hi_fraction_max.")
    ret.add_argument("--clip-lo-max", dest="clip_lo_max", type=float, default=None,
                     help="Override bands.clip_lo_fraction_max.")
    ret.set_defaults(func=cmd_retrieve)

    cmp_ = sub.add_parser("compare", help="Join two batches on frameIndex across their named "
                                          "graphs and print per-frame observable deltas.")
    cmp_.add_argument("--batch-a", dest="batch_a", required=True,
                      help="Reference batch name (floor-qualified), e.g. floor1_baseline.")
    cmp_.add_argument("--batch-b", dest="batch_b", required=True,
                      help="Compared batch name (floor-qualified), e.g. floor1_mixed_dev.")
    cmp_.add_argument("--out", default=None, help="Optional CSV path for the delta table.")
    cmp_.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                      help=f"Fuseki dataset endpoint (default {DEFAULT_ENDPOINT}).")
    cmp_.set_defaults(func=cmd_compare)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
