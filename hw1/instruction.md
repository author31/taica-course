# HW1 — Ontology-Driven Evaluation for Geometric ICP 3D Reconstruction

Onboarding guide for the experiment-centric design. It walks the whole flow on
the current tree, where every student deliverable is filled in with the
reference implementation so the pipeline can be tested end to end.

```bash
# one-time setup, from the repository root
pixi install -e habitat
pixi run -e habitat python -c "import open3d, rdflib; print('ready')"
```

Every command below runs from the repository root with `pixi run -e habitat`.

---

## 1. Goal of Homework 1

Students implement **geometric ICP** (frame-to-frame, point-to-plane,
constant-velocity init, per-step plausibility gate) reconstructing a camera
trajectory and map from RGB-D frames — and then learn the lesson the algorithm
alone cannot teach: **the algorithm is one puzzle piece; the data quality fed
to it decides whether it works.**

On top of the algorithm sits a semantic layer: an **ontology of qualification
factors** — measurable properties of the raw RGB and depth rasters that drive
reconstruction quality. Each factor has a definition, a measurement procedure,
a polarity, and a Pass/Fail threshold. When reconstruction fails, students do
not shrug and recollect everything; they read the layer's attribution —
*which factor failed, under which setting, and what kind of fix that implies*
— and act on it. The mindset shift being trained: from "my code has a bug" to
"my input violates a quality condition I can name, measure, and defend."

Success for one homework iteration is an evidence chain: a sealed experiment
file holding a prediction, measured values, baked verdicts, two reconstruction
outcomes, and an attribution that survives scrutiny.

## 2. Why two phases, and why the algorithm comes with them

| | Phase 1 | Phase 2 |
|---|---|---|
| Data | **provided by us** (first floor) | **collected by the student** (second floor, Habitat sim) |
| Dirty secret | the capture is corrupted; students are *not told* | whatever the student's own driving causes |
| What closes the loop | a supported diagnosis (the provided pixels cannot be regenerated) | a **new capture** that makes the full-batch run pass |

The intention of the split:

- **Phase 1 fixes the data so diagnosis is the only move.** Students implement
  the algorithm and the factor measurers, run them on a capture that secretly
  contains defective frames, and must *discover* the defects through the
  ontology tooling — explore, declare factors, assess, reconstruct, attribute.
  Because they cannot regenerate our pixels, the phase ends at a defensible
  verdict: which frames are bad, which factor proves it, and whether removing
  them actually helps the consumer (it does — see §4.3).
- **Phase 2 makes the loop close for real.** The student drives the agent,
  collects a second-floor capture, and runs the same loop on their own data.
  Now every verdict role is actionable: a Generation verdict means re-drive
  the agent; Measurement/Qualification verdicts mean a number in their own
  declaration. The algorithm implementation is shared across both phases —
  what changes is who owns the data and therefore which fixes are honest.

## 3. Phase 1 — get the corrupted dataset

Download the phase-1 capture (peer note: it is the injected-corruption
handout; students just see "the first-floor capture"):

> https://drive.google.com/file/d/1GKa5nNexuRSCDBXQII_K2Q50Ky6rydx3/view?usp=sharing

Unpack it under `eval/`. The expected layout (one **batch** = one capture
directory):

```text
<capture-dir>/
  rgb/<integer-stem>.png      8-bit colour, one per frame
  depth/<same-stem>.png       16-bit depth, millimetres, 0 = no return
  intrinsics.json             {"width","height","hfov"} — THIS capture's camera
```

RGB and depth files sharing a stem are one frame. A corrupted variant of a
capture is its **own batch** in its own directory; never overwrite a capture
in place.

First contact, always the same command:

```bash
pixi run -e habitat python hw1/api.py explore <capture-dir>
```

`explore` reads the capture directory (rgb/ + depth/) directly — frames, image
paths, stem gaps — and prints it. There is no `batch.ttl` step. Generation
provenance (`api.py batch2ttl --gen name=value --derived-from`, writing an
optional sidecar) is recorded only when the actual generator level is known;
never invent one — an absent GenerationSetting means "not asserted", not
"clean".

## 4. The experiment-centric design

The organizing rule: **an experiment is something you design, not something
the tool runs.** You author a Turtle *declaration* — which factors to
evaluate, under which settings, judged by which thresholds — and the tooling
measures exactly what you declared, bakes Pass/Fail verdicts next to every
value **in the same file**, appends both reconstruction outcomes, and seals
the whole thing. `hw1/experiments/` grows as an append-only lab notebook:
every tuning idea is a new file under a new name, and old files never change.

The workflow loop:

```text
1. explore        see the capture directory before measuring anything
2. declare        scaffold <name>.ttl (`api.py declare --data-dir`), then edit it:
                  trim the factors, add settings, write the PREDICTION
3. experiment     assess once → values + verdicts baked below the marker, file sealed
4. reconstruct    two runs: full-batch baseline + verdict-driven deletion probe
5. explore /      read the verdict walk: failing run ⇒ failing factors ⇒
```

### 4.1 The ontology engine: rdflib only

There is **no triple store, no SPARQL, no server, no named graphs, no OWL
reasoning**. The engine is `hw1/api.py` + [rdflib](https://rdflib.readthedocs.io/):
plain Turtle files parsed into a single graph, validated, extended, and
serialized back. Four subcommands (`reconstruct.py` is the fifth command of
the suite):

| Command | Does | Writes |
|---|---|---|
| `api.py explore <capture-dir or .ttl>…` | read-only terminal tables; several files → comparison view | nothing |
| `api.py declare --data-dir` | scaffold a declaration (prefixes, capture join, factor selection, PREDICTION TODOs); never assesses | `hw1/experiments/<name>.ttl` (refuses to overwrite) |
| `api.py experiment <decl.ttl>` | assess one declaration, **once**, then seal | machine section of the same file |
| `reconstruct.py --experiment` | baseline + selected runs, outcomes into the sealed file | run nodes + diagnostics JSON |
| `api.py batch2ttl` | **deprecated.** optional sidecar for `--gen` / `--derived-from` provenance | `<capture>/batch.ttl` |

Everything the semantic layer computes — statuses, verdicts, attribution — is
baked into the files at assessment time and read back with `explore`. The
`.ttl` files *are* the state; version-control them like lab notes.

### 4.2 The TBox: `hw1/ontology/hw1.ttl`

The ontology file is the single source of truth for *names and defaults*. You
read it (and `definitions.md`) to know what is declarable; you never edit it.
Three things it defines:

**The factor menu** — 8 selectable quality factors, all computed from raw
rasters (numpy + Pillow):

| Factor | Observable | Polarity | Default threshold |
|---|---|---|---|
| `hw1:HighFrequencyDepthResidual` | `highFrequencyDepthResidual` (m) | lower-better | `maxHighFrequencyDepthResidual` 0.05 |
| `hw1:FlyingPixelRatio` | `flyingPixelRatio` | lower-better | `maxFlyingPixelRatio` 0.05 |
| `hw1:ValidTileCoverage` | `validTileCoverage` | higher-better | `minValidTileCoverage` 0.50 |
| `hw1:HighlightClipping` (RGB) | `clipHiFraction` | lower-better | `maxClipHiFraction` 0.05 |
| `hw1:ShadowClipping` (RGB) | `clipLoFraction` | lower-better | `maxClipLoFraction` 0.30 |
| `hw1:IdentityMedianDepthChange` (pair) | `identityMedianDepthChange` (m) | lower-better | `maxIdentityMedianDepthChange` 0.20 |
| `hw1:JointValidDepthRatio` (pair) | `jointValidDepthRatio` | higher-better | `minJointValidDepthRatio` 0.30 |
| `hw1:PriorWarpDepthResidual` (pair) | `priorWarpDepthResidual` (m) | lower-better | `maxPriorWarpDepthResidual` 0.10 |

Plus the statusless RGB baseline `hw1:meanValue` (always written, never Pass
or Fail — the control your chosen factors are supposed to beat), and the two
run-level factors that are always evaluated, never selected:
`hw1:ReconstructionAccuracy` over `mapMeanL2` (`maxMapMeanL2` 0.80 m) and
`hw1:Coverage` over `coverageF` (`minCoverageF` 0.40).

**Parameters and their roles** — every settable number is a declared
`hw1:Parameter` with a role that *is* the verdict vocabulary:

| Role | Lives on | When attribution names it, the fix is |
|---|---|---|
| `GenerationSetting` | Batch | the pixels are bad → regenerate / re-drive the agent |
| `MeasurementSetting` | Experiment | the observable was measured at a poor level → change the number, new experiment |
| `QualificationSetting` | Experiment | the Pass line is mis-cut → change the threshold, new experiment |

**The grading rule** — one rule for everything: higher-better passes iff
`value >= threshold`, lower-better iff `value <= threshold`. A status is a
cache; the raw value is always stored next to it, so any threshold debate can
be re-litigated from the sealed file.

### 4.3 The experiment Turtle file — declaring qualification factors

A declaration is the one piece of RDF you author. You do not start from a
blank page — scaffold it:

```bash
pixi run -e habitat python hw1/api.py declare \
  --name my_first_test \
  --data-dir <capture-dir> \
  --floor 1 \
  --factor HighFrequencyDepthResidual
```

This writes `hw1/experiments/my_first_test.ttl` with the prefixes, the
Experiment node (IRI tail = file stem, `hw1:onBatch` derived from `--floor`
and the capture-directory basename), your factor selection (omit `--factor`
and it selects the full menu for you to trim), the factor menu and override
syntax as comments, and the PREDICTION block as TODOs. Everything in it stays
yours to edit until assessment seals the file; it never overwrites an
existing declaration.
Minimal anatomy of what it scaffolds:

```turtle
# PREDICTION (write BEFORE assessing — the seal makes it non-retractable):
#   input:    which frames/factors will fail, and why
#   baseline: expected mapMeanL2 band and verdict
#   selected: expected differential and the mechanism behind it

@prefix hw1:  <http://taica.course/hw1/ontology#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

<http://taica.course/hw1/ontology#experiment/my_first_test>
    a hw1:Experiment ;
    rdfs:label "depth-glitch hypothesis, stock measurement, re-cut outcome line"@en ;
    hw1:batchFile "<capture-dir>" ;
    hw1:onBatch <http://taica.course/hw1/ontology#batch/floor1_<capture-name>> ;
    hw1:evaluatesFactor hw1:HighFrequencyDepthResidual .
```

Rules that bite: the file stem **must equal** the IRI tail
(`my_first_test.ttl` ↔ `…experiment/my_first_test` — naming your experimental
conditions is part of designing them); `hw1:onBatch` must match
`floor<N>_<capture-dir-basename>` for the directory named in `hw1:batchFile`;
the selection is 1–8 menu factors.

**Overriding a setting** — a `hw1:FactorSetting` blank node per override.
Qualification example (tighten a factor's Pass line):

```turtle
    hw1:hasFactorSetting [
        a hw1:FactorSetting ;
        hw1:settingParameter hw1:maxHighFrequencyDepthResidual ;
        hw1:settingRole hw1:QualificationSetting ;
        hw1:settingForFactor hw1:HighFrequencyDepthResidual ;
        hw1:settingValue "0.010"^^xsd:double
    ] .
```

Measurement example (change how the observable is computed —
e.g. `hw1:tauHi "245.0"` for HighlightClipping), and run-level example
(re-cut the outcome line when your capture's clean floor supports it —
`hw1:settingParameter hw1:maxMapMeanL2` with
`hw1:settingForFactor hw1:ReconstructionAccuracy`). Everything you do not
override is filled from TBox defaults *for the selected factors only*, and
`explore` on the unassessed declaration previews the full setting vector —
declared vs `WILL BE DEFAULTED` — before you commit:

```bash
pixi run -e habitat python hw1/api.py explore hw1/experiments/my_first_test.ttl
```

**Assess once, sealed forever:**

```bash
pixi run -e habitat python hw1/api.py experiment hw1/experiments/my_first_test.ttl
```

This measures the selection, writes per-frame annotations (per modality:
values + statuses + a per-modality `qualificationStatus`), mints frame pairs,
fills defaults, and appends it all below a machine marker with a
`declarationDigest` sealing the student section. Re-assessment is a hard
error. To change anything, copy the student section to a new name.

**Reconstruct — the control and the probe:**

```bash
pixi run -e habitat python hw1/reconstruct.py \
  --experiment hw1/experiments/my_first_test.ttl \
  --data_root <capture-dir> --no-vis
```

Two runs by default: `baseline` (every frame) and `selected` (maximal
usable-link segments — a link is usable iff the pair passed and both endpoint
frames passed). The selected run is a **falsification probe** of "the failed
frames are harming ICP", not a promised repair. Both record `mapMeanL2` and
`gatedSteps`; selected adds `spliceCount` and `maxGapLength`. Interpreting the
differential:

- baseline Fail, selected Pass → the rejected frames are load-bearing; the
  verdicts found real damage;
- baseline Pass, selected Fail → deletion/splicing caused the regression;
- similar outcomes → the factors are not shown to bind the consumer at these
  levels.

**Worked reference** (in the tree, fully reproducible): the sealed experiment
`hw1/experiments/first_floor_uniform_injected_v7.ttl` selects only
`HighFrequencyDepthResidual` with `maxMapMeanL2` re-cut to 0.3, and shows the
loop closing: HFD fails exactly the 10 defective frames (values ≈ 2.25 m vs a
0.0054 m clean-side max), baseline 0.8679 **Fail**, selected 0.0353 **Pass**
with `spliceCount 10, maxGapLength 1`. `explore` prints the whole verdict
walk, ending at a Generation setting on the batch:

```bash
pixi run -e habitat python hw1/api.py explore hw1/experiments/first_floor_uniform_injected_v7.ttl
```

**Inspect visually** — the dashboard projects the same sealed files: factor
timelines with thresholds, linked RGB/depth frame viewer, run outcomes with
splice/gate evidence, settings with declared/defaulted source, and a notebook
comparison table:

```bash
pixi run -e habitat python hw1/dashboard.py          # http://127.0.0.1:8765
```

## 5. What students implement

Right now **everything ships as the reference implementation** so the whole
flow can be tested (this document's purpose). Before handout, the functions
below are carved to documented stubs; their docstring CONTRACT/SPEC blocks
are the assignment. Everything else — the RDF engine, the CLI, sealing,
explore, the dashboard — ships working and is off-limits.

### 5.1 `hw1/utils.py` — the reconstruction stack

Functions marked `SHIPS WORKING` in their docstring stay (depth loading,
preprocessing, FPFH/RANSAC, the Open3D ICP wrappers, frame reconciliation,
`mean_l2`, visualisation helpers). The student pass targets:

| Function | What is being learned |
|---|---|
| `depth_image_to_point_cloud` | the pinhole back-projection itself — Open3D's projection helpers are off-limits; validity masking (0 = no return, never a point at the origin) |
| `my_local_icp_algorithm` | point-to-point ICP from scratch: cKDTree correspondences, Kabsch/Umeyama SVD with reflection fix, convergence handling |
| `reconstruct` | the SLAM loop per its CONTRACT: frame streaming, constant-velocity init, the per-step plausibility gate, frame-0 anchoring, subsetting semantics |

Smoke fixture: `hw1/tests/fixtures/` ships five synthetic frames with exactly
known clouds and a GT trajectory whose perfect `mean_l2` is 0.0 — it separates
"my ICP is wrong" from "my loop is wrong" before any real capture is touched.

### 5.2 `hw1/api.py` — only the qualification-factor measurers

Students implement **only the measurement functions** behind the factor menu —
never the RDF machinery. The eight active measurers (plus their `_mask`
variants where masks are exported):

| Function | Factor |
|---|---|
| `frame_clip_hi_fraction` | HighlightClipping |
| `frame_clip_lo_fraction` | ShadowClipping |
| `frame_high_frequency_depth_residual` (+`_mask`) | HighFrequencyDepthResidual |
| `frame_flying_pixel_ratio` (+`_mask`) | FlyingPixelRatio |
| `frame_valid_tile_coverage` (+`_mask`) | ValidTileCoverage |
| `pair_identity_median_depth_change` (+`_mask`) | IdentityMedianDepthChange |
| `pair_joint_valid_depth_ratio` (+`_mask`) | JointValidDepthRatio |
| `pair_prior_warp_depth_residual` (+`_mask`) | PriorWarpDepthResidual |

Each has an exact contract (formula, units, validity rule, fail-closed
sentinel) in its docstring and in `definitions.md`; the tests in
`hw1/tests/test_factor_measurers.py` and
`hw1/tests/test_factor_mask_pipeline.py` pin the expected behavior.

### Verify your setup end to end

```bash
pixi run -e habitat pytest hw1/tests/test_factor_measurers.py \
  hw1/tests/test_factor_mask_pipeline.py hw1/tests/test_diagnostic_metrics.py -q

# reproduce the worked reference (deterministic — expect 0.8679 / 0.0353):
pixi run -e habitat python hw1/reconstruct.py \
  --experiment hw1/experiments/first_floor_uniform_injected_v7.ttl \
  --data_root eval/first_floor_uniform_injected --baseline-only --no-vis
pixi run -e habitat python hw1/reconstruct.py \
  --experiment hw1/experiments/first_floor_uniform_injected_v7.ttl \
  --data_root eval/first_floor_uniform_injected --selected-only --no-vis
```

| Symptom | Meaning |
|---|---|
| declaration digest mismatch | student section edited after assessment; make a new experiment |
| `experiment` refuses to run | file already sealed — that is the design, not a bug |
| selected run much worse than baseline | real splice/gap effect; read `spliceCount`/`maxGapLength`/`gatedSteps` |
| RGB factor fails, geometry unchanged | true input condition, not load-bearing for geometry-only ICP — say both in the report |
| no Generation verdict on provided data | no generator level recorded on the batch; never invent one |
