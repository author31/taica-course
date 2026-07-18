"""
HW1 data-quality triplestore CLI — ingest frames into Fuseki, query PASS frames via SPARQL.

WHAT THIS FILE IS
    A two-subcommand command-line tool that bridges a directory of captured SLAM
    frames to an Apache Jena Fuseki triplestore over the SPARQL 1.1 protocol
    (Graph Store HTTP + Query), using `rdflib` to build and serialize the RDF.

    * `insert`   — pair the rgb/ + depth/ frames of a batch, measure each frame's
                   per-frame observables (Rec.601 avg luma, valid-depth fraction),
                   build an RDF graph matching `ontology/hw1.ttl`, and PUT it into
                   Fuseki as ONE named graph per batch (re-insert replaces the batch).
    * `retrieve` — run a SPARQL SELECT that grades every frame of a named batch
                   against a brightness/valid-depth band and writes the PASS frames
                   (those inside the band) to a CSV. The grading now lives in SPARQL
                   FILTERs, not Python — the band comes from `THRESHOLDS[floor]`
                   (mirroring `definitions.md`) unless overridden on the CLI.

    The two "observable measurers" below (`frame_luma`, `frame_valid_fraction`) are
    the single-frame twins of `autoresearch.py`'s aggregate measurers, kept pure so a
    query sample can be graded frame-by-frame at ingest time.

TWO API-SERVICEABLE AXES  (single-sample computable → checkable at inference)
    * Brightness  — observable `avg_luma_rec601`: Rec.601 mean luma of the RGB frame,
      in [0, 255]. Band = THRESHOLDS[floor]["luma"] = (lo, hi).
    * Depth       — observable `valid_depth_fraction`: fraction of depth pixels that
      are non-zero AND inside [min_range, max_range] metres, in [0, 1]. Minimum =
      THRESHOLDS[floor]["valid_frac_min"].
    The noise axis is deliberately excluded (needs a clean-depth reference the API
    query sample does not have). See `definitions.md`.

ONTOLOGY CONTRACT  (must match ontology/hw1.ttl — see NS/HW1 below)
    Classes: Batch, Frame, RGBImage, DepthImage, QualityFactor.
             RGBImage/DepthImage are rdfs:subClassOf schema:ImageObject.
    Object props: hasFrame (Batch->Frame), hasRGBImage (Frame->RGBImage),
                  hasDepthImage (Frame->DepthImage).
    Per-node observables: avgLuma on the RGBImage node, validDepthFraction on the
                  DepthImage node; each image node carries schema:contentUrl (path).
    IRI scheme: batch  = <ns>batch/<name>
                frame  = <ns>batch/<name>/frame/<n>
                rgb    = <ns>batch/<name>/frame/<n>/rgb
                depth  = <ns>batch/<name>/frame/<n>/depth
                <name> = os.path.basename(data_dir); it is ALSO the named-graph IRI.

DEPTH FORMAT
    Depth PNGs are uint16 millimetres; metres = raw / 1000.0. A pixel is valid iff
    raw != 0 AND min_range <= metres <= max_range.

ENDPOINT
    Default `http://localhost:3030/ds`. Query `<endpoint>/query`, update
    `<endpoint>/update`, Graph Store `<endpoint>/data`. This module NEVER starts the
    server and does not require one to import.

DEPENDENCIES
    Standard library + numpy + Pillow + rdflib. No OpenCV, no Open3D.

SEE ALSO
    queries/valid_frames.rq      — SPARQL SELECT template `retrieve` fills + runs
    ontology/hw1.ttl             — TBox + floor-1 reference bands (parsed in on insert)
    definitions.md               — source of truth for the thresholds below
    autoresearch.py:145-206      — aggregate measurers frame_* mirror per-frame
"""

import argparse
import csv
import glob
import os
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

import rdflib
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLStore

# =============================================================================
# Ontology namespace  (must match ontology/hw1.ttl exactly — do not rename)
# =============================================================================
NS = "http://taica.course/hw1/ontology#"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")

# Path to hw1/ontology/hw1.ttl relative to this file (this file lives in hw1/).
_ONTOLOGY_TTL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ontology", "hw1.ttl")

# Directory of SPARQL query templates (*.rq), decoupled from this module.
_QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")

DEFAULT_ENDPOINT = "http://localhost:3030/ds"

# =============================================================================
# Thresholds  (single source of truth — copied from definitions.md)
#   Class-B conventions: valid for the hw1 robust-ICP pipeline + coupling ONLY.
#   DEFAULT band source for `retrieve` when CLI overrides are not supplied.
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

    Load the PNG as RGB (PIL `Image.open(p).convert("RGB")`), cast to float64,
    take the per-pixel luma via the Rec.601 dot product, and return its mean over
    all pixels. This is the per-frame twin of autoresearch.measure_luma.
    """
    arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64)
    luma = arr @ _LUMA_WEIGHTS
    return float(luma.mean())


def frame_valid_fraction(depth_path, min_range=0.0, max_range=10.0):
    """SPEC: fraction of valid depth pixels in a single depth frame, in [0,1].

    Depth PNG is uint16 millimetres; metres = raw / 1000.0. A pixel is valid iff
    raw != 0 AND min_range <= metres <= max_range. Return valid_count / total_pixels.
    This is the per-frame twin of autoresearch.measure_valid_ratio.
    """
    raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    meters = raw.astype(np.float64) / 1000.0
    valid = (raw != 0) & (meters >= min_range) & (meters <= max_range)
    return float(valid.sum()) / float(valid.size)


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
# IRI helpers  (batch = named-graph IRI too)
# =============================================================================
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
def build_batch_graph(data_dir, floor, include_ontology=True):
    """Build the in-memory rdflib.Graph for a batch directory.

    Returns (graph, name, batch_iri). Measures per-frame avgLuma / validDepthFraction.
    If `include_ontology` and ontology/hw1.ttl exists, parse it in so a fresh
    in-memory store also carries the TBox + floor-1 reference bands.
    """
    name = os.path.basename(os.path.normpath(data_dir))
    pairs = _pair_frames(data_dir)

    g = Graph()
    g.bind("hw1", HW1)
    g.bind("schema", SCHEMA)

    if include_ontology and os.path.exists(_ONTOLOGY_TTL):
        g.parse(_ONTOLOGY_TTL, format="turtle")

    b = batch_iri(name)
    g.add((b, RDF.type, HW1.Batch))
    # String props are stored as plain literals: in RDF 1.1 a plain literal IS
    # xsd:string, so this honours the ontology's rdfs:range xsd:string while also
    # matching a plain "name" constant in the retrieve query on BOTH Jena/Fuseki
    # (RDF 1.1 compliant) and rdflib's stricter in-memory pattern matcher.
    g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    g.add((b, HW1.floor, Literal(int(floor), datatype=XSD.integer)))

    for stem, rgb_path, depth_path in pairs:
        idx = int(stem)
        f = _frame_iri(name, idx)
        g.add((f, RDF.type, HW1.Frame))
        g.add((f, HW1.frameIndex, Literal(idx, datatype=XSD.integer)))
        g.add((b, HW1.hasFrame, f))

        # avgLuma now lives on the RGBImage node (ontology domain hw1:RGBImage).
        rc = _component_iri(name, idx, "rgb")
        g.add((rc, RDF.type, HW1.RGBImage))
        g.add((rc, HW1.avgLuma, Literal(frame_luma(rgb_path), datatype=XSD.double)))
        g.add((rc, SCHEMA.contentUrl, Literal(rgb_path)))
        g.add((f, HW1.hasRGBImage, rc))

        # validDepthFraction now lives on the DepthImage node (domain hw1:DepthImage).
        dc = _component_iri(name, idx, "depth")
        g.add((dc, RDF.type, HW1.DepthImage))
        g.add((dc, HW1.validDepthFraction,
               Literal(frame_valid_fraction(depth_path), datatype=XSD.double)))
        g.add((dc, SCHEMA.contentUrl, Literal(depth_path)))
        g.add((f, HW1.hasDepthImage, dc))

    return g, name, b


def put_named_graph(graph, batch_uri, endpoint):
    """Replace the named graph <batch_uri> in Fuseki with `graph` via a Graph Store
    HTTP PUT of Turtle to <endpoint>/data?graph=<batch_uri>. Requires a running server.
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
    g, name, b = build_batch_graph(args.data_dir, args.floor)
    n = len(g)
    put_named_graph(g, b, args.endpoint)
    print(f"[insert] batch {name!r}: pushed {n} triples")
    print(f"[insert] named graph: {b}")
    return 0


# =============================================================================
# retrieve  — SPARQL SELECT of PASS frames (grading lives in the FILTER)
# =============================================================================
def resolve_band(args):
    """Resolve the effective (bmin, bmax, dmin) band: CLI override if given, else
    THRESHOLDS[floor]. Returns three floats.
    """
    lo, hi = THRESHOLDS[args.floor]["luma"]
    dmin_default = THRESHOLDS[args.floor]["valid_frac_min"]
    bmin = float(args.brightness_min) if args.brightness_min is not None else float(lo)
    bmax = float(args.brightness_max) if args.brightness_max is not None else float(hi)
    dmin = float(args.valid_depth_min) if args.valid_depth_min is not None else float(dmin_default)
    return bmin, bmax, dmin


def load_query(name):
    """Read a SPARQL query template `<name>.rq` from hw1/queries/ and return its text."""
    path = os.path.join(_QUERY_DIR, f"{name}.rq")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_select(batch_name, bmin, bmax, dmin):
    """Build the SPARQL SELECT string for a batch from the queries/valid_frames.rq
    template, substituting the namespace, the (SPARQL-escaped) batch name, and the
    numeric band literals (floats we control — safe to inject inline).

    Tokens (@@NS@@/@@BMIN@@/@@BMAX@@/@@DMIN@@) are substituted before @@BATCH@@ so a
    batch name that happens to contain a token string cannot corrupt the query.
    """
    esc = batch_name.replace("\\", "\\\\").replace('"', '\\"')
    query = load_query("valid_frames")
    query = (query
             .replace("@@NS@@", NS)
             .replace("@@BMIN@@", repr(float(bmin)))
             .replace("@@BMAX@@", repr(float(bmax)))
             .replace("@@DMIN@@", repr(float(dmin)))
             .replace("@@BATCH@@", esc))
    return query


def run_select(query, endpoint):
    """Run a SPARQL SELECT against <endpoint>/query and return the rdflib Result.
    Requires a running server.
    """
    store = SPARQLStore(query_endpoint=f"{endpoint}/query")
    g = Graph(store)
    return g.query(query)


def cmd_retrieve(args):
    bmin, bmax, dmin = resolve_band(args)
    print(f"[retrieve] batch {args.batch!r}, floor {args.floor}: "
          f"effective band brightness=[{bmin}, {bmax}], valid_depth_min={dmin}")

    query = build_select(args.batch, bmin, bmax, dmin)
    result = run_select(query, args.endpoint)

    rows = []
    for row in result:
        frame = int(row.idx)
        rows.append((frame, str(row.rgb), str(row.depth), float(row.luma), float(row.vdf)))
    rows.sort(key=lambda r: r[0])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "rgb_path", "depth_path", "luma", "valid_fraction"])
        for frame, rgb, depth, luma, vdf in rows:
            w.writerow([frame, rgb, depth, luma, vdf])

    print(f"[retrieve] {len(rows)} PASS frames -> {args.out}")
    return 0


# =============================================================================
# CLI
# =============================================================================
def _build_parser():
    p = argparse.ArgumentParser(
        description="HW1 data-quality triplestore CLI (Fuseki + rdflib): insert batches, "
                    "retrieve PASS frames via SPARQL.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("insert", help="Measure a batch's frames and push them into Fuseki "
                                        "as one named graph.")
    ins.add_argument("--data-dir", required=True,
                     help="Directory containing rgb/ and depth/ subdirs of integer-stem .png frames.")
    ins.add_argument("--floor", type=int, default=1,
                     help="Floor label stored on the Batch node (default 1).")
    ins.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                     help=f"Fuseki dataset endpoint (default {DEFAULT_ENDPOINT}).")
    ins.set_defaults(func=cmd_insert)

    ret = sub.add_parser("retrieve", help="SPARQL-query a batch's PASS frames (inside the band) "
                                          "to a CSV.")
    ret.add_argument("--batch", required=True,
                     help="Batch name (batchName == this value) to query.")
    ret.add_argument("--out", required=True, help="Output CSV path.")
    ret.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                     help=f"Fuseki dataset endpoint (default {DEFAULT_ENDPOINT}).")
    ret.add_argument("--floor", type=int, default=1,
                     help="Floor whose THRESHOLDS defaults define the band (default 1).")
    ret.add_argument("--brightness-min", type=float, default=None,
                     help="Override brightness band low (default THRESHOLDS[floor] luma lo).")
    ret.add_argument("--brightness-max", type=float, default=None,
                     help="Override brightness band high (default THRESHOLDS[floor] luma hi).")
    ret.add_argument("--valid-depth-min", type=float, default=None,
                     help="Override valid-depth minimum (default THRESHOLDS[floor] valid_frac_min).")
    ret.set_defaults(func=cmd_retrieve)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
