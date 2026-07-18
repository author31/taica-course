# HW1 Semantic Layer — Research Synthesis & Refined Design

Date: 2026-07-29. Sources: three research passes (RGB clipping thresholds; frame-pair
ICP factors + pairwise RDF modeling; experiment-layer ontology patterns). Citations at
the end of each section. This document refines `IDEA.md` into concrete ontology deltas
for `ontology/hw1.ttl` and the builder in `api.py`.

---

## 0. Decisions at a glance

| Question | Decision |
|---|---|
| Experiment semantic layer | `hw1:Experiment` node + reified `hw1:KnobSetting` (ML-Schema `HyperParameterSetting` shape); one **named graph per experiment**, graph IRI == experiment IRI |
| Knob taxonomy | `hw1:Knob` with subclasses `hw1:CorruptionKnob` ("fix = regenerate data") and `hw1:MeasurementKnob` ("fix = change the number, re-measure") — this split IS the pedagogical payoff |
| tau_hi / tau_lo | **Universal constants of the 8-bit pipeline: 250 / 5** (98%/2% full scale). Not student-derived per batch. Kept as `MeasurementKnob`s only as a sensitivity-ablation axis |
| Frame-to-frame factors | New `hw1:FramePair` class (W3C n-ary Pattern 1) with 2 raster-only observables: `medianDepthDifference`, `depthEdgeOverlap` |
| Sequence | Assert `hw1:hasNextFrame` forward only. **No `hasPreviousFrame` property** — pyoxigraph does no OWL reasoning; query backwards with SPARQL path `^hw1:hasNextFrame` |
| Result metrics | `hw1:Result` ⊂ `dqv:QualityMeasurement`, `hw1:ResultMeasure` ⊂ `dqv:Metric` — so existing `hw1:Band` machinery grades reconstruction outcomes the same way it grades observables |
| Batch vs Experiment split | Batch graph = invariant structure only (frames, IRIs, paths, order). ALL measured values (frame + pair observables), knobs, and results live in the experiment graph |

---

## 1. Experiment layer

### 1.1 The problem being fixed

`hw1.ttl` currently puts `tauHi`/`tauLo` on `hw1:Batch` while `clipHiFraction` sits on
`RGBImage`. Two measurements of the same batch at different tau produce contradictory
triples on the same subject. The shipped queries already document the pain:
`valid_frames.rq` — "tau is baked into the stored clip fractions at insert time. Change
tau and you must re-insert"; `compare_batches.rq` — "clip fractions of the two batches
are only comparable if both were measured at the same tau". RDF Data Cube IC-12 is the
formal statement of the violation: the identifying dimensions (the knob vector) must be
part of what identifies an observation.

### 1.2 Design

- **`hw1:Experiment`** = one algorithm run = (input batch, knob vector, results). DoE
  calls the knob vector a *treatment combination*, a.k.a. a *run* (NIST/SEMATECH §5.7).
- **`hw1:KnobSetting`** reified node: `[ hw1:setsKnob <knob> ; hw1:knobValue <double> ]`.
  Chosen over flat properties because (i) "list all knobs of ?exp" is one BGP, and
  (ii) it composes with the existing `hw1:parameterisedBy` (range `rdf:Property`):

  ```sparql
  ?factor hw1:parameterisedBy ?knob .
  ?exp hw1:hasKnobSetting [ hw1:setsKnob ?knob ; hw1:knobValue ?tau ] .
  ```

  Under flat properties this needs a variable in predicate position — unteachable.
  Same shape as `mls:HyperParameterSetting` (ML-Schema), MEX `HyperParameter`, and
  ISA's Factor/FactorValue split — three communities converged independently.
- **Punning:** `hw1:tauHi` stays an `rdf:Property` (so `parameterisedBy` keeps its
  range) AND becomes an individual of `hw1:MeasurementKnob` (so it can sit in object
  position). Legal in RDFS; only OWL-DL objects; we run no reasoner. Teaching moment.
- **Do NOT name the knob class `Factor`.** DoE "factor" (a knob) would collide with
  `hw1:QualityFactor` (a measured metric — a DoE *covariate*). Record the DoE synonym
  via `skos:altLabel "factor"`.
- **Named graph per experiment, graph IRI == experiment node IRI.** Per-frame/per-pair
  observables keep their current flat single-triple form; scoping by graph disambiguates
  at zero triple cost (this is `qb:Slice` semantics). `GRAPH ?exp { ?exp a
  hw1:Experiment ... }` binds graph and node in one pattern.
- **Rejected alternatives:** per-observation `dqv:QualityMeasurement` nodes (4–5× triple
  and query-pattern bloat at frame cardinality; breaks `hw1:overProperty`); PROV-O for
  knobs (no value slot — Qualification Pattern needs 4–5 triples to store one float);
  importing `mls:` IRIs (drags Task/Implementation machinery; `mls:hasInput` overloaded).
  Rule of thumb worth teaching: **reify where vocabulary is open and cardinality low
  (knobs, results); stay flat and scope by graph where vocabulary is fixed and
  cardinality is huge (per-frame observables).**

### 1.3 Corrupted batches

A corrupted variant lives in its own directory ⇒ it is its own `hw1:Batch` (own graph,
pixels are different). The experiment that runs on it records the generation knobs
(brightness gain, depth-noise sigma…) as `CorruptionKnob` settings — the knob vector
identifies the treatment even though the pixels already embody it. Optional decoration:
`prov:wasDerivedFrom` base batch on the Batch node.

### 1.4 Store layout

| Graph IRI | Contents | Changes when |
|---|---|---|
| default graph | TBox (`hw1.ttl`) + Band individuals | at query time |
| `<ns>batch/<name>` | Batch, Frames, image nodes, `frameIndex`, paths, `hasNextFrame`, `FramePair` nodes with `fromFrame`/`toFrame` | only when pixels change |
| `<ns>experiment/<id>` | Experiment node, KnobSettings, **all measured observables** (frame + pair), Results | every experiment |

Uniform rule: structure in the batch graph, numbers in the experiment graph — even for
knob-free observables like `validDepthFraction` (uniformity beats micro-optimization;
under corruption knobs most "knob-free" values differ across experiments anyway).

### 1.5 TBox delta (existing `rdf:Property`/`rdfs:comment` house style)

```turtle
# ---- experiment layer ----
hw1:Experiment a rdfs:Class .
hw1:Knob       a rdfs:Class ; skos:altLabel "factor"@en ;
    rdfs:comment "A settable value. DoE calls this a FACTOR, its value a LEVEL. Not hw1:QualityFactor, which is a measured metric."@en .
hw1:CorruptionKnob  rdfs:subClassOf hw1:Knob ;
    rdfs:comment "Changes the pixels. If culprit: regenerate data."@en .
hw1:MeasurementKnob rdfs:subClassOf hw1:Knob ;
    rdfs:comment "Changes only how pixels are scored. If culprit: change the number, re-measure."@en .
hw1:KnobSetting a rdfs:Class ;
    rdfs:comment "One knob at one value, for one experiment. N-ary relation node."@en .

hw1:onBatch        a rdf:Property ; rdfs:domain hw1:Experiment ; rdfs:range hw1:Batch .
hw1:hasKnobSetting a rdf:Property ; rdfs:domain hw1:Experiment ; rdfs:range hw1:KnobSetting .
hw1:setsKnob       a rdf:Property ; rdfs:domain hw1:KnobSetting ; rdfs:range hw1:Knob .
hw1:knobValue      a rdf:Property ; rdfs:domain hw1:KnobSetting ; rdfs:range xsd:double .

# Punned: still rdf:Property (hw1:parameterisedBy range intact) AND Knob individuals.
# Legal RDFS; only OWL-DL objects; no reasoner runs here.
hw1:tauHi           a hw1:MeasurementKnob .
hw1:tauLo           a hw1:MeasurementKnob .
hw1:brightnessGain  a rdf:Property, hw1:CorruptionKnob ; rdfs:range xsd:double .
hw1:depthNoiseSigma a rdf:Property, hw1:CorruptionKnob ; rdfs:range xsd:double ; qudt:unit unit:M .

# ---- results ----
hw1:ResultMeasure a rdfs:Class ; rdfs:subClassOf dqv:Metric .
hw1:Result        a rdfs:Class ; rdfs:subClassOf dqv:QualityMeasurement .
hw1:hasResult     a rdf:Property ; rdfs:domain hw1:Experiment ; rdfs:range hw1:Result .

hw1:MapMeanL2 a hw1:ResultMeasure ; rdfs:label "map mean L2 error"@en ;
    hw1:polarity hw1:LowerIsBetter ; qudt:unit unit:M ; dqv:expectedDataType xsd:double .
hw1:CoverageF a hw1:ResultMeasure ; rdfs:label "coverage-aware F-score"@en ;
    hw1:polarity hw1:HigherIsBetter ; dqv:expectedDataType xsd:double .
```

`ResultMeasure ⊂ dqv:Metric` means result measures reuse `hw1:polarity`,
`dqv:expectedDataType`, `qudt:unit`, and — critically — **`hw1:Band` can grade a result
exactly as it grades an observable**, so "reconstruction failed" is a band lookup, not a
magic number in a FILTER.

REMOVE: `rdfs:domain hw1:Batch` declarations of `hw1:tauHi`/`hw1:tauLo` and their
"Student-derived" comments (superseded — see §3).

### 1.6 ABox example (TriG for illustration; students write plain .ttl, loader supplies graph)

```turtle
hw1:experiment/e07 {
  hw1:experiment/e07 a hw1:Experiment ;
      rdfs:label "floor1 brightness OFAT, step 3"@en ;
      hw1:onBatch hw1:batch/first_floor_mixed ;
      hw1:hasKnobSetting [ hw1:setsKnob hw1:brightnessGain  ; hw1:knobValue 1.8   ] ,
                         [ hw1:setsKnob hw1:depthNoiseSigma ; hw1:knobValue 0.0   ] ,
                         [ hw1:setsKnob hw1:tauHi           ; hw1:knobValue 250.0 ] ,
                         [ hw1:setsKnob hw1:tauLo           ; hw1:knobValue 5.0   ] ;
      hw1:hasResult [ a hw1:Result ; dqv:isMeasurementOf hw1:MapMeanL2 ;
                      dqv:computedOn hw1:experiment/e07 ; dqv:value 0.41 ; qudt:unit unit:M ] .

  # Observables measured UNDER those knobs. Subject IRIs are the SAME nodes the batch
  # graph declares — that identity is the join. Same batch at tauLo=16 => second graph,
  # same subjects, different numbers, no conflict.
  hw1:batch/first_floor_mixed/frame/42/rgb
      hw1:clipHiFraction 0.213 ; hw1:clipLoFraction 0.004 ; hw1:meanValue 198.6 .
  hw1:batch/first_floor_mixed/frame/42/depth
      hw1:validDepthFraction 0.91 ; hw1:depthRoughness 0.021 .
  hw1:batch/first_floor_mixed/pair/42_43
      hw1:medianDepthDifference 0.031 ; hw1:depthEdgeOverlap 0.61 .
}
```

### 1.7 Payoff queries

Failure attribution — which experiments have shadow clipping outside Good AND high error,
and at what tauLo:

```sparql
PREFIX hw1: <http://taica.course/hw1/ontology#>
PREFIX dqv: <http://www.w3.org/ns/dqv#>
SELECT ?exp ?tauLo ?err (COUNT(?img) AS ?badFrames) WHERE {
  ?band hw1:forFactor hw1:ShadowClipping ; hw1:grade hw1:Good ; hw1:maxInclusive ?maxGood .
  GRAPH ?exp {
    ?exp a hw1:Experiment ;
         hw1:hasKnobSetting [ hw1:setsKnob hw1:tauLo ; hw1:knobValue ?tauLo ] ;
         hw1:hasResult      [ dqv:isMeasurementOf hw1:MapMeanL2 ; dqv:value ?err ] .
    ?img hw1:clipLoFraction ?clipLo .
    FILTER (?clipLo > ?maxGood)
  }
}
GROUP BY ?exp ?tauLo ?err
HAVING (COUNT(?img) > 20 && ?err > 0.30)
ORDER BY DESC(?err)
```

If failing rows cluster at low tauLo ⇒ measurement artefact, fix one number. If tauLo is
constant across them ⇒ clipping is real, fix the capture. The generic variant resolves
the knob through the ontology and answers "what do I fix" by knob class:

```sparql
SELECT ?exp ?knob ?level ?class ?err WHERE {
  GRAPH ?exp {
    ?exp hw1:hasKnobSetting [ hw1:setsKnob ?knob ; hw1:knobValue ?level ] ;
         hw1:hasResult      [ dqv:isMeasurementOf hw1:MapMeanL2 ; dqv:value ?err ] .
  }
  hw1:ShadowClipping hw1:parameterisedBy ?knob .   # finds tauLo without naming it
  ?knob a ?class . VALUES ?class { hw1:CorruptionKnob hw1:MeasurementKnob }
} ORDER BY ?knob ?level
```

`?class` is the verdict: `CorruptionKnob` → regenerate; `MeasurementKnob` → re-measure.

### 1.8 Implementation notes (pyoxigraph)

- **BUG (pre-existing):** `api.py` `cmd_query`/loader and `run.py` call `store.load()`
  without `to_graph` — everything lands in the default graph, so the two shipped
  `GRAPH ?g` queries (`valid_frames.rq`, `compare_batches.rq`) return **zero rows**
  today. Fix regardless of this design.
- Loader: `store.load(path=ttl, format=RdfFormat.TURTLE, to_graph=NamedNode(f"{NS}experiment/{exp_id}"))`.
  Turtle has `supports_datasets == False`; students keep writing `.ttl`, loader supplies
  the graph from the filename/id. (Alternative: teach TriG. Not recommended — second syntax.)
- Useful API: `Store.named_graphs()`, `clear_graph()`, `Store.query(..., use_default_graph_as_union=True)`
  — document the union flag only as a debugging aid; it collapses the scoping.
- Re-running with different knobs = insert one more experiment graph; batch graph untouched.

Sources: W3C n-ary Note (Pattern 1); ML-Schema (`Run/HyperParameterSetting/hasValue`,
`ModelEvaluation`); DQV Note (`QualityMeasurement ⊂ qb:Observation`; Appendix D —
parameterised metrics deliberately unstandardised); RDF Data Cube (IC-12, `qb:Slice`);
NIST/SEMATECH e-Handbook §5.7 (DoE glossary); "On Semantics of RDF Datasets" (W3C Note —
local graph-naming convention must be documented); pyoxigraph Store docs; MLSea (ESWC
2024 — setting/evaluation shape still current); EXPO (naming; also the Factor-collision
warning); Carroll et al. WWW 2005 (named graphs).

---

## 2. Frame-pair layer

### 2.1 What drives pairwise ICP failure (literature)

| Driver | Evidence |
|---|---|
| D1 Overlap | <30% = failure regime (3DMatch/3DLoMatch, Predator CVPR'21); Pomerleau IJRR'12 samples 0.3–0.99 |
| D2 Initial misalignment / motion | ICP basin ≈ 10–30° rotation (Pomerleau AR'13 perturbation protocol); KinectFusion assumes small inter-frame motion; constant-velocity init fails when acceleration violates model |
| D3 Geometric degeneracy | Point-to-plane Hessian rank-deficient on flat walls/corridors (KinectFusion limitations; Hinduja & Kaess IROS'19) |
| D4 Depth noise / invalidity | Kinect-style error grows quadratically with range (Khoshelham & Elberink '12) |

Literature thresholds live in **pose space** (overlap %, degrees) — raster Good bands
below are priors to calibrate with the OFAT sweep, not published constants.

### 2.2 Chosen raster-only observables (two PNG pairs, numpy, deterministic)

| Observable | Formula | Proxies | Polarity | Good band (prior) |
|---|---|---|---|---|
| `medianDepthDifference` | median abs(D_t − D_t+1) over jointly-valid px, metres | D2 (initial projective-ICP residual; Kerl DVO / KinectFusion residual at identity) | Lower | ≤ 0.07 m; Bad > 0.20 m (≈ ICP gate) |
| `depthEdgeOverlap` | IoU of dilated (1–2 px) Sobel depth-edge masks | D3 — only proxy covering degeneracy (Choi/Trevor/Christensen IROS'13: depth edges are registration features) | Higher | ≥ 0.5; Bad < 0.25 |

Coverage: D2→`medianDepthDifference`; D3→`depthEdgeOverlap`. D1/D4 carried at the
single-frame level by the existing `validDepthFraction` only (pair-level
`jointValidDepthFraction` and `photometricDifference` were dropped by design decision).
Known blind spots (state in assignment): depth-diff blind to optical-axis rotation and
wall-parallel slide — the pair-level photometric proxy that covered this is dropped,
accepted gap; edge IoU unstable when edges sparse — which is itself a degeneracy signal.
Optical flow = best motion proxy in literature but needs OpenCV — excluded per
raster/numpy-only constraint.

Note: `depthEdgeOverlap`'s Sobel threshold θ and dilation radius are `MeasurementKnob`s
(record their settings on the Experiment). `medianDepthDifference` is parameter-free
(no `parameterisedBy`) — d_max for validity comes from the existing depth-range config.

### 2.3 TBox delta (house style)

```turtle
# ---- frame pairs (W3C n-ary relation Pattern 1) ----
hw1:FramePair a rdfs:Class ;
    rdfs:comment "Ordered pair of temporally adjacent frames; carries pairwise observables predicting frame-to-frame ICP success."@en .
hw1:fromFrame a rdf:Property ; rdfs:domain hw1:FramePair ; rdfs:range hw1:Frame ;
    rdfs:comment "Earlier frame; ICP source."@en .
hw1:toFrame   a rdf:Property ; rdfs:domain hw1:FramePair ; rdfs:range hw1:Frame ;
    rdfs:comment "Later frame; ICP target."@en .
hw1:hasNextFrame a rdf:Property ; rdfs:domain hw1:Frame ; rdfs:range hw1:Frame ;
    rdfs:comment "Asserted forward only. pyoxigraph does no OWL reasoning: query the inverse with SPARQL path ^hw1:hasNextFrame."@en .

hw1:medianDepthDifference   a rdf:Property ; rdfs:domain hw1:FramePair ; rdfs:range xsd:double ; qudt:unit unit:M .
hw1:depthEdgeOverlap        a rdf:Property ; rdfs:domain hw1:FramePair ; rdfs:range xsd:double .

hw1:DepthConsistency a hw1:QualityFactor ;
    rdfs:label "depth consistency"@en ;
    hw1:overProperty hw1:medianDepthDifference ; hw1:polarity hw1:LowerIsBetter ;
    hw1:appliesTo hw1:FramePair .
hw1:StructureOverlap a hw1:QualityFactor ;
    rdfs:label "structure overlap"@en ;
    hw1:overProperty hw1:depthEdgeOverlap ; hw1:polarity hw1:HigherIsBetter ;
    hw1:appliesTo hw1:FramePair .
```

Naming rationale: "FramePair" matches plain-noun class style (Batch, Frame, RGBImage);
"Transition" implies state-machine semantics; `fromFrame`/`toFrame` = ICP source/target.
Pair IRIs: `<ns>batch/<name>/pair/<i>_<i+1>`. Rejected: RDF-star (experimental in
Oxigraph; a quoted triple can't be the object of `hw1:appliesTo`), rdf:Statement
reification (verbose, semantics-free), rdf:Seq/List/OLO (Daga et al. CEUR'19: list
vocabularies complicate SPARQL for nothing when a successor property suffices).
DQV on pair nodes is legitimate: `dqv:computedOn` range is deliberately open
(`rdfs:Resource`) per the DQV Note / Albertoni & Isaac SWJ 2020.

Placement: `FramePair` nodes + `fromFrame`/`toFrame` + `hasNextFrame` = structure →
batch graph. The four measured values → experiment graph (§1.4).

Sequence walks in SPARQL: previous frame `?f ^hw1:hasNextFrame ?prev`; whole chain
`?first hw1:hasNextFrame+ ?later`; ordering via existing `frameIndex` + ORDER BY.

Sources: Rusinkiewicz & Levoy '01; Pomerleau IJRR'12 + AR'13; Newcombe KinectFusion '11;
Huang Predator CVPR'21; Khoshelham & Elberink '12; Choi/Trevor/Christensen IROS'13;
Sturm TUM benchmark '12 (per-sequence velocity as difficulty); Noy & Rector W3C n-ary
Note '06; Daga CEUR'19; Oxigraph SPARQL notes.

---

## 3. tau thresholds & clip bands (literature verdicts)

### 3.1 Constants, not batch knobs

Every source fixes tau as a fraction of full scale; none re-derives it per dataset:
Eilertsen HDRCNN 0.95→243; Santos SIGGRAPH'20 0.96→245; MATLAB `makehdr` 98%/2% →
250/5; Zhang ICRA'17 VO code 252/10; Shin IROS'19 235/15; BT.709 legal range 235/16.
Sub-255 guard-bands exist only for sensor noise + response rolloff (Zhang & Brainard
JOSA'04; Debevec & Malik '97) — Habitat renders have neither; clipping is exact at
255/0.

**Adopt: tau_hi = 250, tau_lo = 5** (citable to `makehdr`; inside envelope
[235,252]×[5,16]). Keep tau as `MeasurementKnob` in the experiment layer purely as a
sensitivity axis: one OFAT sweep over {235,243,250,252,255}×{0,5,10,15,16} showing
band-grade invariance is a cheap, defensible student exercise. This supersedes the
"Student-derived" batch-property design.

### 3.2 Definitions confirmed

- Highlight: `max(R,G,B) >= tau_hi` = "any channel clipped" — matches dominant
  literature definition (Eilertsen; Zhang & Brainard: one clipped channel already
  destroys chromatic info).
- Shadow: `max(R,G,B) <= tau_lo` ≡ ALL channels crushed — correct; `min()` would
  false-flag legit saturated colors like (0,0,255).

### 3.3 Bands (label as course-calibrated; no validated constant exists in literature)

- `clipHiFraction`: **Good ≤ 0.02, Marginal 0.02–0.05, Bad > 0.05.** Upper anchor:
  Eilertsen's test protocol uses exactly 5% saturation as "degraded enough to need HDR
  reconstruction"; AE systems target ~0% (Zhang's code compensates at any over-ratio).
  0.02 itself = extrapolated midpoint — say so.
- `clipLoFraction`: **Good ≤ 0.10, Bad > 0.30.** Deliberately loose and asymmetric:
  Zhang's defaults disable underexposure compensation entirely; QueensCAMP found
  underexposure barely hurts VO; indoor Replica legitimately has large dark regions.
- Validate both against the pipeline's own OFAT ICP-error curves (autoresearch
  program) — the literature explicitly declines to give a fraction→performance map
  (scene-dependent; Shim/Shin optimize gradient/entropy metrics instead for this
  reason).

Key refs: Eilertsen et al. TOG'17 (arxiv 1710.07480); Santos et al. SIGGRAPH'20
(2005.07335); Zhang/Forster/Scaramuzza ICRA'17 + uzh-rpg/active_camera_exposure_control
(252/10, over/under ratio == our clip fractions); Shin et al. IROS'19 (1907.12646);
ITU-R BT.709-6; MATLAB makehdr docs; Zhang & Brainard JOSA-A'04; Debevec & Malik
SIGGRAPH'97; Mertens Exposure Fusion PG'07 (soft-weight counterexample); QueensCAMP
(2410.12520).

---

## 4. Migration checklist

1. `ontology/hw1.ttl`: add §1.5 + §2.3 TBox; move `tauHi`/`tauLo` off `Batch` (now
   punned Knob individuals); add `hasNextFrame`; do NOT add `hasPreviousFrame`.
2. `api.py` builder: split output — `batch.ttl` (structure: frames, images, paths,
   `frameIndex`, `hasNextFrame`, FramePair skeletons) vs `experiment.ttl` (KnobSettings,
   all observable values, Results). Add the 2 pairwise metric functions (numpy only).
3. `api.py`/`run.py` loader: pass `to_graph=` (batch graph for batch files, experiment
   IRI for experiment files). **Fixes existing zero-rows bug in shipped GRAPH queries.**
4. Hard-code tau defaults 250/5 in `thresholds` config; record them as KnobSettings
   per experiment.
5. `queries/`: update `valid_frames.rq`/`compare_batches.rq` to the new layout; add
   `failure_attribution.rq` (§1.7) and a pairwise variant (worst FramePair per
   experiment vs per-frame ICP error, once run.py logs per-frame residuals).
6. Bands: add Band individuals for the 2 pairwise factors + 2 clip fractions using §2.2
   / §3.3 priors; calibrate via OFAT sweep; add Bands for `MapMeanL2`/`CoverageF`
   (result grading — §1.5).
7. Docs: state the graph-naming convention (graph IRI == experiment IRI) in the TBox
   header comment (required — RDF dataset semantics is deliberately unstandardised).
