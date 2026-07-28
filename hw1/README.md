# Homework 1 — Geometry-only ICP SLAM under Environment Uncertainty

You implement a **geometry-only ICP SLAM** pipeline over RGB-D frames, and then
you decide **which frames are worth feeding it**.

Two things are scored: how well your reconstruction tracks the ground-truth
camera path (mean L2) and covers the floor (F-score), and how well you can
*justify*, from your own measurements, a data-quality gate that improves it.

The twist: the input data is not clean. Parts of the environment are
**degraded** — the simulator injects uncertainty that depends on **where the
agent is**, not on when it got there. Some regions of the floor produce frames
your algorithm cannot use. Nobody tells you which regions, how bad they are, or
where the boundary lies. That is the assignment.

> **The central caveat, and you must handle it explicitly in your report.**
> The pipeline is **geometry-only**: ICP reads depth. It never reads RGB. So no
> RGB measurement can *cause* a reconstruction failure. RGB factors are
> **upstream proxies** — they are indicators of a scene condition that reaches
> the geometry through the simulator's light→depth coupling. A report that
> asserts "brightness breaks ICP" is wrong. A report that shows an RGB factor
> predicts ICP degradation, and says plainly what it is a proxy *for*, is right.

---

## The two phases

| | Phase 1 | Phase 2 |
|---|---|---|
| **Floor** | `apartment_0` first floor | `apartment_0` second floor |
| **Data** | **Provided, frozen.** You do not capture it. | **You capture it**, interactively. |
| **Ships as** | `rgb/`, `depth/`, `GT_pose.npy`, `intrinsics.json` — nothing else | you produce the same four |
| **Config** | **None.** The capture is shipped without its config. | [`configs/second_floor.yaml`](configs/second_floor.yaml), fully commented |
| **The mechanism** | Hidden. You infer it from the data. | **Revealed.** The config exposes the uncertainty zones *and* the light→depth coupling. |
| **Your job** | Build the pipeline; derive the quality factors, their thresholds, and the clipping parameters; show the gate helps | Capture your own floor-2 data; test whether your floor-1 thresholds **transfer**; reconcile what you inferred in phase 1 against the mechanism the config now shows you |

Phase 1 gives you a clean control and a degraded run of the **same trajectory**,
so every degradation is attributable. Phase 2 asks whether anything you learned
survives contact with a different floor and a trajectory you chose yourself.

---

## Environment setup

```bash
# 1. Install pixi (task/environment manager) — https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Fetch the habitat-lab submodule (an editable dependency)
git submodule update --init dependencies/habitat-lab

# 3. Install the project environment (Python 3.9, habitat-sim, open3d, pygame, …)
pixi install -e habitat
pixi run smoke            # sanity check: prints habitat-sim / habitat-lab versions

# 4. Download the Replica apartment_0 scene (needed for phase 2)
pixi run fetch-replica

# 5. Download the provided phase-1 capture
pixi run fetch-hw1-data
```

Step 5 prints the directory it extracted to; `<phase1>` in the commands below
means that directory. The phase-1 capture holds exactly two runs of the same
trajectory — a clean `baseline/` and a degraded `mixed/` — each containing
`rgb/`, `depth/`, `GT_pose.npy` and `intrinsics.json`, and nothing else.

`intrinsics.json` carries exactly three keys — `width`, `height`, `hfov` — the
camera parameters you need to unproject depth into a point cloud. It says
nothing about how the environment degrades; that is still yours to find.

Everything below runs inside the pixi `habitat` environment
(`pixi run -e habitat python ...`), from the **repo root**. The `habitat` env
also ships `openjdk 21`, so `java` is on `PATH` for the triplestore with no
extra setup.

- **Simulator:** [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
  0.3.3, driven through the reusable `packages/simulator` library.
  `hw1/load.py` is a thin CLI driver over it.
- **Triplestore:** Apache Jena Fuseki, launched by a pixi task (below), serving
  an in-memory dataset — it is wiped on restart, so just re-`insert`.

---

## File map

Everything under `hw1/`. **Bold** files are ones you edit.

| File | Role |
|---|---|
| **`utils.py`** | The ICP SLAM library. Unprojection, preprocessing, ICP, the `reconstruct(...)` loop, `mean_l2(...)`. Partly implemented — see [What you implement](#what-you-implement). |
| `reconstruct.py` | Thin CLI over `utils.py` — reconstructs one capture dir, prints mean L2, opens an Open3D window with the estimated (red) and GT (black) trajectories. Ships working. |
| `completeness.py` | Coverage-aware correctness — builds a whole-floor GT map from the **baseline** capture, anchors your reconstruction into the world frame, returns accuracy / completeness / **F-score**. Ships working; it is the grader, so do not edit it. |
| **`api.py`** | The data-quality CLI — `insert` a capture batch into Fuseki as RDF, then `retrieve` the frames that pass your quality bands as a CSV manifest. The four measurers and the graph/query builders are yours. |
| **`ontology/hw1.ttl`** | The ontology (Turtle). The TBox ships; the quality-factor individuals and the new datatype properties are yours. |
| **`queries/*.rq`** | SPARQL templates `api.py` fills and runs. Skeletons — the graph patterns and filters are yours. |
| **`thresholds.json`** | **Does not ship. You author it.** Your derived bands and clipping parameters. See [thresholds.json](#thresholdsjson). |
| **`analysis.md`** | Your written deliverable. Ships as a template of prompts. |
| `definitions.md` | The four quality factors as **measurement contracts** — formulas, units, ranges, invariants, and the literature they come from. Contains **no** thresholds; those are yours to derive. |
| `test_e2e.py` | The **API contract** for `api.py`. Ships working and is autograded — read it as the specification of what `insert` and `retrieve` must do. |
| `configs/second_floor.yaml` | The phase-2 config (heavily commented) — scene, agent, camera, lighting, depth sensor model, the spatial `uncertainties` block, output. |
| `fuseki_bin/` | Fuseki server binary (downloaded on demand; not checked in). |

Supporting code lives outside `hw1/` and ships working — you never edit it:

| Path | Role |
|---|---|
| `packages/simulator/` | The simulation library (pixi editable install; importable as `simulator`). Engine, config loading, the lighting/depth pixel pipeline, the uncertainty zones, pose replay, the pygame viewer. |
| `scripts/evaluate.py` | The two-run orchestrator: collect a clean run and a degraded run of the same trajectory, then score both. |
| `hw1/load.py` | Interactive pygame collection, or deterministic replay of a `.npy` pose trajectory. |

---

## What you implement

Every blank function keeps its signature and a docstring stating the
**contract** — inputs, outputs, units, invariants — and ends in `#TODO`. The two
halves of the assignment do not block each other: an empty `utils.py` still lets
you run `api.py`, and vice versa.

### `utils.py` — the geometry

| Function | |
|---|---|
| `depth_image_to_point_cloud` | **`#TODO`** — unprojection with the intrinsics from the capture's `intrinsics.json`. |
| `reconstruct` | **`#TODO`** — the frame-to-frame loop: pose accumulation, outlier gating, `--frames-csv` subsetting. This is the highest-risk one; a broken loop silently becomes a broken analysis, so use the smoke fixture to tell "my ICP is wrong" from "my loop is wrong" *before* you start the data-quality work. |
| `local_icp_algorithm` | Ships working — an **Open3D wrapper**, and the reference baseline your own ICP is compared against. |
| `preprocess_point_cloud`, `global_registration`, `multiscale_icp` | Ship working. |
| `mean_l2` | Ships working. **It is the score** — it must be byte-identical across submissions, so it is not yours to write. |

Read the intrinsics (`width`, `height`, `hfov`) from the **`intrinsics.json` in
the capture directory you are reconstructing** — never from a config. Every
capture carries its own, including the ones you collect in phase 2, so the same
code path works on both floors and cannot silently unproject one floor's depth
with another floor's camera.

### `api.py` — the four quality factors

Two categories, four factors — two on depth, two on RGB. Full measurement
contracts, units and citations are in [`definitions.md`](definitions.md).

| Measurer | Factor | |
|---|---|---|
| `frame_valid_fraction` | `ValidDepthRatio` | **`#TODO`** — fraction of depth pixels that returned a usable range. |
| `frame_depth_roughness` | `DepthRoughness` | **`#TODO`** — reference-free estimate of the depth raster's noise level, in metres. Factor 1 asks whether a pixel returned a number; this asks how much that number can be trusted. |
| `frame_clip_hi_fraction` | `HighlightClipping` | **`#TODO`** — fraction of pixels railed at the top. |
| `frame_clip_lo_fraction` | `ShadowClipping` | **`#TODO`** — fraction of pixels railed at the bottom. |
| `frame_mean_value` | *not a factor* | **Ships working.** The **naive baseline you must beat.** Its docstring says outright that it is a poor quality metric and why. |

Also `#TODO`: `build_batch_graph` (emit the observables onto the ontology's node
structure), `resolve_band` (four one-sided bounds, read from `thresholds.json`),
and `build_select` (template substitution into `valid_frames.rq`). Fuseki
transport and CLI wiring ship working — they are not the learning objective.

Three things worth knowing before you start:

- **All four measurers read raw PNGs and nothing else.** No unprojection, no
  point cloud, no normals, no Open3D — numpy and Pillow are enough. That is what
  makes the two halves of the assignment genuinely independent: you can finish
  and grade the whole `api.py` side with `utils.py` still empty.
- Both clipping factors are computed on the **value channel** `V = max(R,G,B)`,
  not on any luma. `definitions.md` gives the two independent reasons; one of
  them is a one-line counter-example you should be able to reproduce.
- The two definitions use the **same operator with different quantifiers** —
  `V ≥ τ_hi` means *at least one* channel is saturated, `V ≤ τ_lo` means *every*
  channel is crushed. That asymmetry is deliberate. The symmetric-looking
  alternative is wrong, and `definitions.md` says why.

One warning about `frame_depth_roughness` in particular: it has two contract
details — the fully-valid-window mask and the use of a **median** rather than a
mean — and getting either wrong yields a number that still looks entirely
plausible while measuring something else. `definitions.md` §1.2 spells both out
and ships four closed-form fixtures that catch them. Write those fixtures before
you measure anything real.

### `ontology/hw1.ttl` and `queries/`

The TBox ships. You add the four `QualityFactor` individuals, the datatype
properties carrying the new observables, and — on the `Batch` node — the
**measurement provenance**: the clipping parameters the batch's fractions were
computed under. Without them the store holds fractions whose meaning is
unrecoverable from the store itself, and two batches measured differently look
directly comparable when they are not.

Two queries:

- `valid_frames.rq` — the PASS-frame `SELECT`. Header and substitution-token
  contract ship; the graph pattern and the four `FILTER` terms are yours. Read
  the header comment about `GRAPH ?g` before you debug an empty result set.
- `compare_batches.rq` — joins the **baseline** and **degraded** named graphs on
  `frameIndex` and returns per-frame observable deltas. This is the query that
  answers "why a triplestore instead of a dataframe".

---

## `thresholds.json`

**We ship no numbers.** You derive every threshold yourself and record them in
`hw1/thresholds.json`, which `api.py` loads at runtime. The file must carry:

- **Four one-sided band bounds**, one per quality factor —
  `valid_depth ≥`, `depth_roughness ≤` (in metres), `clip_hi ≤`, `clip_lo ≤`.
  Only the first points up; the other three are all "lower is better".
- **The two clipping parameters `τ_lo` and `τ_hi`.** These are *measurement*
  parameters, not bands, and they are yours to derive and justify like
  everything else.

Every band key is `<observable_name>_<min|max>`:

```json
{
  "tau_lo": null,
  "tau_hi": null,
  "bands": {
    "valid_depth_fraction_min": null,
    "depth_roughness_max": null,
    "clip_hi_fraction_max": null,
    "clip_lo_fraction_max": null
  }
}
```

`api.py`'s `--valid-depth-min` / `--clip-hi-max` style command-line flags are a
separate, shorter namespace for one-off overrides. They are **not** the keys in
the file; a CLI spelling written into `thresholds.json` is a key the loader does
not read.

> ### The gotcha that will cost you an afternoon
>
> **τ is read at `insert` time; the bands are read at `retrieve` time.**
>
> Change `τ_lo` or `τ_hi` and every clip fraction already in the triplestore is
> **stale** — it was measured under the old τ. Retrieving against it silently
> gives you numbers that no longer mean what you think.
>
> **Fix: re-run `api.py insert` after any τ change.** This is safe and
> idempotent — insert replaces the batch's whole named graph rather than
> appending to it, so re-inserting can never double-count or leave a mixture of
> old and new measurements.
>
> Changing only a *band* needs no re-insert. Bands are applied by the query.

A degenerate band is not automatically a failure. If a factor genuinely does not
separate on this scene, **"this metric does not separate here, and here is the
evidence" is a valid and creditable conclusion.** What is not creditable is an
unargued band that gates nothing.

---

## Running the pipeline

### 1. Reconstruct a capture

```bash
# the provided phase-1 clean run
pixi run -e habitat python hw1/reconstruct.py --data_root <phase1>/baseline/

# the provided phase-1 degraded run
pixi run -e habitat python hw1/reconstruct.py --data_root <phase1>/mixed/
```

Prints the mean L2 and opens an Open3D window; `--no-vis` for the metric only.
`-v my_icp` selects your from-scratch ICP instead of the Open3D reference.

### 2. Launch the triplestore

```bash
pixi run -e fuseki fuseki      # fetches the Fuseki jar on first run, then serves
```

Serves an in-memory dataset at `http://localhost:3030/ds`.

### 3. Measure a batch and insert it

```bash
pixi run -e habitat python hw1/api.py insert --data-dir <phase1>/baseline --floor 1
pixi run -e habitat python hw1/api.py insert --data-dir <phase1>/mixed    --floor 1
```

Measures every paired frame with your four measurers (at the τ from
`thresholds.json`), builds the RDF, and PUTs it into the named graph for that
batch, so every capture lands in its own graph and they all coexist in the store.

**Batch names are floor-qualified**: `floor{N}_{directory name}`, giving
`floor1_baseline`, `floor1_mixed`, `floor2_baseline`, `floor2_mixed`. Both
floors use the same two directory names on disk, so the floor prefix is what
keeps your phase-1 and phase-2 captures in separate named graphs instead of one
silently overwriting the other. That qualified name is what you pass to
`retrieve --batch` and what `compare_batches.rq` joins on.

### 4. Retrieve the frames that pass your gate

```bash
pixi run -e habitat python hw1/api.py retrieve --batch floor1_mixed --out valid_frames.csv
```

Runs the `SELECT`, applies your four `FILTER` terms, and writes the manifest.
Band values come from `thresholds.json`; run `api.py retrieve --help` for the
override flags.

### 5. Reconstruct through the gate

```bash
pixi run -e habitat python hw1/reconstruct.py \
    --data_root <phase1>/mixed/ --frames-csv valid_frames.csv
```

Reconstructs **only** the frames listed in the manifest. The GT is subset by the
same frame stems, so the metric stays aligned. **This is the loop your whole
analysis runs in**: change a band → retrieve → reconstruct → observe the effect
on mean L2 and F-score.

### 6. Phase 2 — collect your own floor-2 data

```bash
# interactive pygame collection (drive with the keyboard)
pixi run -e habitat python hw1/load.py

# clean collection, uncertainty disabled
pixi run -e habitat python hw1/load.py --clean
```

Navigation runs in a live pygame window (first-person RGB + depth + a top-down
bird's-eye panel), so you can *see* every frame exactly as it is captured, with
whatever degradation is active at your current position already applied.

> ### Reproducibility — required, not optional
>
> **Interactive capture is not reproducible.** Which zone you are in is
> determined by position, but the flicker phase and the per-frame depth noise
> both key off the wall clock while you drive. Run the same route twice and you
> get different pixels.
>
> So phase 2 must go: **collect interactively → keep `GT_pose.npy` → re-render
> deterministically → run every analysis on the re-rendered capture.**
>
> ```bash
> pixi run -e habitat python hw1/load.py --trajectory <your>/GT_pose.npy
> ```
>
> In replay the time base is `t = frame_index / fps`, so the render is
> reproducible across runs and machines. Without this step no phase-2 number you
> report is reproducible — including by the grader.

### 7. Phase 2 — the two-run evaluation

```bash
pixi run -e habitat python scripts/evaluate.py --config hw1/configs/second_floor.yaml
```

Collects the same trajectory twice — uncertainty off → `baseline/`, on →
`mixed/` — and scores both. The clean run is **required**:
`completeness.build_gt_reference` refuses a degraded directory, because a
reference contaminated by injected faults is not a reference.

### 8. Tests

```bash
pixi run -e habitat test        # hw1/test_e2e.py — the api.py contract, autograded
env -u PYTHONPATH pixi run -e habitat python -m pytest packages/simulator/tests/
```

`test_e2e.py` spins up a real Fuseki server on a free port — nothing is mocked —
synthesises a small batch with **closed-form** expected values for every factor,
inserts it, retrieves it, and asserts the manifest holds exactly the frames
inside the band. Every fixture passes τ in explicitly. Two of the fixtures are
there as executable arguments rather than as coverage; they are worth reading
before you write your report.

The simulator suite needs the Replica scene and **fails loud** if it is missing
rather than skipping.

---

## How the environment degrades

Enough to orient you; the details are what you are being asked to find.

**Uncertainty is a property of place.** The configuration defines circular zones
with **hard edges** — inside one, the scene light flickers; outside all of them,
rendering is identical to a clean run. Zones are checked in list order and the
first match wins. There is no schedule, no seeded window stream, and no
time-of-day: cross the boundary and the effect switches on, cross back and it
switches off.

There is more than one zone and they are not equally severe. The clean
navigable space outside every zone is your **within-capture control**.

Lighting is emulated as a post-process on the RGB frame — the frame is scaled by
an exposure gain, and inside a zone a periodic term modulates that gain. Keep in
mind that a **single-frame** metric fires on the extreme frames of a flicker
zone quite happily; what it cannot do is recognise the modulation as
*periodic*. Flicker is not invisible to a per-frame measurement — its
periodicity is. And the structure you actually need to recover is **spatial**,
which `GT_pose.npy` gives you directly.

For phase 1, that is all you are told. The phase-2 config shows you the rest,
including the causal path by which a photometric effect reaches a geometry-only
pipeline at all.

---

## Coverage-aware correctness — the F-score

Mean L2 scores only the **camera path**, so on its own it rewards doing *less*:
a run that captures a few frames of one corner posts a tiny error while covering
almost nothing. So `completeness.py` also scores the **reconstructed cloud**
against a whole-floor GT map, on two intuitive numbers — **accuracy** (of the
points you reconstructed, what fraction land on the true surface) and
**completeness** (of the whole floor, what fraction you covered) — combined as:

```
F = 2 · A · C / (A + C)
```

A high F needs *both*, so a small accurate sliver still scores low.

The two clouds are placed in the same frame by **one** transform — the first
camera's GT pose — with no trajectory fitting, no ICP, no RANSAC. There is
nothing to overfit, so leftover drift stays visible in the score. The distance
tolerance and the pass rule are pinned in `completeness.py`.

This matters for your gate: a band tight enough to throw away most of the
capture will improve mean L2 and **destroy** the F-score. Report both.

---

## Deliverables

Grading splits into a machine-checkable core and a rubric that rewards reasoning.
Pass bars for the autograded rows ship with the autograder.

### Phase 1 — the provided floor-1 dataset

| Deliverable | How it is checked | Weight |
|---|---|---|
| ICP implementation | Autograded — `reconstruct.py` on the provided clean capture: mean L2 and F-score against fixed pass bars | 25% |
| Ontology API | Autograded — `test_e2e.py` passes against a live Fuseki | 15% |
| `thresholds.json` + the four measurers | Autograded — schema valid (bands **and** τ); measurers pure, deterministic, and matching the closed-form fixtures at the τ the fixture passes in; bands non-degenerate *or* accompanied by an argued degeneracy finding | 10% |
| `analysis.md` — protocol design | Rubric — is your experiment actually capable of answering the question? Controls, confounds, sweep coverage, compute cost acknowledged | 15% |
| `analysis.md` — evidence and reasoning | Rubric — bands supported by measured reconstruction outcomes; at least one **falsifiable prediction** stated and tested; the geometry-only proxy caveat correctly handled; and does either clipping factor predict degradation better than the shipped `frame_mean_value` baseline? **A negative result argued from evidence scores full marks.** | 15% |

### Phase 2 — your floor-2 capture

| Deliverable | How it is checked | Weight |
|---|---|---|
| A capture meeting a coverage floor | Autograded — F-score of your own reconstruction above a floor, so a two-metre capture cannot pass | 5% |
| Band transfer analysis | Rubric — floor-1 bands applied to floor-2 data. Did they transfer? Evidence either way | 10% |
| Mechanism verification | Rubric — the phase-2 config reveals the mechanism. Does your report reconcile it with what you inferred in phase 1? | 5% |

### Two things that are deliberately *not* graded

- **No observable's value.** Because τ is yours, two students'
  `clip_hi_fraction` numbers are not the same quantity. No band bound and no
  observable is a grading target. What is graded is the **τ-agnostic downstream
  outcome**: your gate produces a frame set, that frame set produces a mean L2
  and an F-score on the same provided capture.
- **Absolute phase-2 mean L2.** Trajectories differ per student, so it is not
  comparable across submissions. Only the coverage floor and the reasoning count.

The protocol is **open**. We state the objective; you design the experiment. That
is where the marks are.
