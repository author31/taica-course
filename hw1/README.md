# Homework 1 — Geometry-only ICP SLAM under Environment Uncertainty

You implement a **geometry-only ICP SLAM** pipeline (point-to-plane ICP over
RGB-D frames) and score it by the **mean L2 distance** between your predicted
camera trajectory and the ground truth. The twist: the input data is *not clean*.
A **seeded scheduler injects temporal uncertainty windows** (flicker, low light,
over-exposure) live during a run, and part of the grade is how well your
algorithm holds up inside those windows.

- **Simulator:** [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
  0.3.3, driven through the reusable `packages/simulator` library (shared with
  future homeworks; `hw1/load.py` is a thin CLI driver over it).
- **Environment:** Replica `apartment_0`, **second floor**. The agent spawns at
  `start_position: [0.0, 1.4252348, 0.0]` (navmesh-snapped; `apartment_0`'s
  floor heights are first ≈ −1.57, second ≈ 1.43, and the navmesh tops out
  around `y = 5.4`).
- **One config:** [`configs/second_floor.yaml`](configs/second_floor.yaml) — the
  single hw1 config; every environment condition lives there.
- **Uncertainty regime:** the config's `uncertainties` block drives a seeded
  scheduler that samples an endless `gap → duration → effect type` stream
  (seconds). Inside a window one effect is active at a fixed per-type severity;
  outside windows rendering is bit-identical to a clean run.
- **Pygame integration:** navigation runs in a live pygame window
  (first-person RGB + depth + a top-down bird's-eye panel), so you can *see*
  every frame exactly as it is captured, with any active window's fault already
  applied.

The first floor was dropped from configs / eval data / trajectories in the
temporal-uncertainty refactor. Its empirically derived thresholds are **retained
untouched** (in `api.py THRESHOLDS`, the ontology TTL, and `definitions.md`) as
the reference band for the synthetic ontology test (`test_e2e.py`).

This README is the **index** for `hw1/`: it maps the files, shows how to run the
pipeline, and points at the grounding docs. The empirical data-quality thresholds
and how they were derived live in their own canonical docs (see
[File map](#file-map) and [Empirical grounding](#empirical-grounding)) and are
**not** restated here.

---

## File map

Everything under `hw1/`. One line each; follow the link for detail.

| File | Role |
|---|---|
| `load.py` | Thin CLI driver over `packages/simulator` — interactive pygame collection (uncertainty windows fire **live on the wall clock**; `--clean` disables) or headless-rendered replay preview of a `.npy` pose trajectory; writes `rgb/`, `depth/`, `GT_pose.npy` (+ `windows.json` when the scheduler ran). |
| `configs/second_floor.yaml` | **The** single hw1 config (heavily commented) — scene, agent, camera, baseline lighting/depth models, the `uncertainties` scheduler block, output. |
| `utils.py` | The ICP SLAM library **you implement** — point-cloud unprojection, normal/feature preprocessing, ICP (`local_icp_algorithm` / `my_local_icp_algorithm`), the `reconstruct(...)` loop, and `mean_l2(...)`. |
| `reconstruct.py` | Thin CLI over `utils.py` — reconstructs one capture dir (`--data_root`), prints mean L2, and opens an Open3D window with the estimated (red) and GT (black) trajectories. |
| `completeness.py` | Coverage-aware correctness — builds the whole-floor GT map (from the **baseline** capture only), anchors a reconstruction into the world frame, and returns accuracy / completeness / **F-score**. |
| `api.py` | Ontology-grounded data-quality CLI — `insert` a capture batch into the Fuseki triplestore as RDF, then `retrieve` (SPARQL SELECT) the frames that pass the quality bands to a CSV manifest (see [Ontology-grounded data-quality](#ontology-grounded-data-quality)). |
| `ontology/hw1.ttl` | The ontology (Turtle) — `Batch` / `Frame` / `Component` (RGB + depth image) classes and the `GoodBrightnessRange` / `ValidDepthRatio` quality factors carrying the floor-1 reference bands. |
| `fuseki_bin/` | Apache Jena Fuseki server binary (downloaded on demand by the `fuseki` pixi task; not checked in). |
| `queries/` | The SPARQL query templates (`*.rq`) `api.py` fills and runs — e.g. `valid_frames.rq`, the PASS-frame SELECT. |
| `test_e2e.py` | Ontology regression test of the `api.py` + Fuseki round trip against a live server (see [Tests](#tests)). |
| `definitions.md` | **Canonical**: empirical data-quality definitions and threshold tables (the floor-1 tables are archived as the synthetic-test reference), plus the current temporal-window regime and the `windows.json` schema. |
| `empirical-analysis.md` | **Canonical**: narrative of the empirical process — how the thresholds were derived under the legacy OFAT regime, plus the current temporal-window regime and its caveats. |

The simulation library lives outside `hw1/` in `packages/simulator/` (pixi
editable install — importable as `simulator` inside the habitat env):

| Module | Role |
|---|---|
| `simulator/engine.py` | `Engine` — owns the habitat sim, agent, scheduler, per-frame RNG, and the EGL/DISPLAY workaround; `make_cfg`, `add_start_marker`. |
| `simulator/config.py` | `load_config` — yaml + schema defaults merge. |
| `simulator/effects.py` | Lighting / depth-fault pixel pipeline, `UncertaintyScheduler`, `process_observations`. |
| `simulator/replay.py` | `.npy` pose replay — `load_trajectory`, `replay_poses`, `save_frame`. |
| `simulator/viewer.py` | pygame panels (the only module that imports pygame; SDL env vars set at its import). |
| `tests/` | Unit tests (pure numpy) + the pipeline e2e gate (needs the habitat env + Replica scene). |

The grader's orchestrator, `scripts/evaluate.py`, also lives outside `hw1/`.

---

## Quick start

```bash
# 1. Install pixi (task/environment manager) — https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Fetch the habitat-lab submodule (an editable dependency)
git submodule update --init dependencies/habitat-lab

# 3. Install the project environment (Python 3.9, habitat-sim, open3d, pygame, …)
pixi install -e habitat
pixi run smoke            # sanity check: prints habitat-sim / habitat-lab versions

# 4. Download the Replica apartment_0 scene into replica_v1/
pixi run fetch-replica
```

Everything below runs inside the pixi `habitat` environment
(`pixi run -e habitat python ...`), from the **repo root**.

---

## Running the pipeline

**Collect / replay a capture** with the launcher. The config defaults to
`hw1/configs/second_floor.yaml` (`--config` to override); `--output-root`
overrides `output.root`; `--fps` paces the preview loop:

```bash
# interactive (pygame keyboard) — uncertainty windows fire LIVE on the wall
# clock while you drive, so captures inherit whatever effect is active
pixi run -e habitat python hw1/load.py

# uncorrupted interactive collection (scheduler disabled)
pixi run -e habitat python hw1/load.py --clean

# replay preview of the committed trajectory (deterministic t = frame / fps)
pixi run -e habitat python hw1/load.py --trajectory trajectories/secondfloor.npy
```

When the scheduler ran, the realized windows are saved to
`<output.root>/windows.json` (window ground truth — see
[Ontology-grounded data-quality](#ontology-grounded-data-quality)).

**Reconstruct + visualize one capture** with the thin runner (`--data_root`
defaults to `eval/_data/second_floor/baseline/`):

```bash
pixi run -e habitat python hw1/reconstruct.py
pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/second_floor/mixed/

# restrict to the ontology-selected valid frames (a CSV from api.py retrieve)
pixi run -e habitat python hw1/reconstruct.py \
    --data_root eval/_data/second_floor/baseline/ --frames-csv valid_frames.csv
```

It prints the mean L2 and opens an Open3D window (add `--no-vis` for the metric
only). With `--frames-csv` it reconstructs **only** the frames listed in the CSV
(see [Ontology-grounded data-quality](#ontology-grounded-data-quality)); the GT is
subset by the same frame stems so the metric stays aligned. Omit it to run over
every frame under the capture dir (unchanged default).

---

## Coverage-aware correctness — F-score

Mean L2 scores only the **camera path**, so it quietly rewards doing *less*: a run
that captures a few frames of one corner posts a tiny error while covering almost
nothing. To close that exploit, `hw1/completeness.py` scores the **reconstructed
cloud** against a whole-floor GT map on two intuitive numbers — **accuracy** (of
the points you reconstructed, what fraction land on the true surface) and
**completeness** (of the whole floor, what fraction you covered) — combined into
`F = 2·A·C / (A + C)`. High F needs *both*, so a tiny accurate sliver still scores
low.

The whole-floor GT map is built from the **baseline** (clean, scheduler-off)
capture only — `build_gt_reference` refuses a `mixed/` dir — so the reference is
never contaminated by injected faults. The two clouds are placed in the same
frame by **one** transform (the first camera's GT pose), with no trajectory
fitting / ICP / RANSAC — nothing to overfit, so leftover drift stays visible in
the score. Implementation and the anchoring transform are in
`hw1/completeness.py`; the metric parameters (τ = 0.10 m, the pass rule) are
pinned in [`definitions.md`](definitions.md).

---

## Ontology-grounded data-quality

The data-quality gate is grounded in an **ontology + triplestore**, not a hardcoded
table. A capture batch on disk becomes RDF triples in an Apache Jena Fuseki server;
you then *query* the store (SPARQL) for the frames that satisfy the quality factors
and export them to a CSV manifest — the manifest `reconstruct.py --frames-csv`
consumes.

```
eval/_data/second_floor/<batch>/  --(api.py insert)-->  Fuseki (/ds, in-mem)
        Fuseki  --(api.py retrieve, SPARQL SELECT)-->  valid_frames.csv
        valid_frames.csv  --(reconstruct.py --frames-csv)-->  SLAM + mean L2
```

`<batch>` is the dir basename — under the two-run evaluate flow that means
**`baseline` and `mixed`**, each landing in its own named graph.

### The ontology (`ontology/hw1.ttl`)

- **`Batch`** — one capture dir (one `reconstruct.py` input), keyed by `batchName`
  (dir basename). Many batches coexist in the store, each in its own named graph.
- **`Frame`** — one timestep; carries `frameIndex`, the measured observables
  `avgLuma` + `validDepthFraction`, and `hasComponent` an **`RGBImage`** and a
  **`DepthImage`** (each with an `imagePath`).
- **Quality factors** — `GoodBrightnessRange` (on `avgLuma`) and `ValidDepthRatio`
  (on `validDepthFraction`), holding the floor-1 **reference bands** from
  [`definitions.md`](definitions.md).

### 1. Launch the triplestore

```bash
pixi run -e fuseki fuseki      # downloads the Fuseki jar on first run, then serves
```

Serves an in-memory dataset at `http://localhost:3030/ds` (wiped on restart; just
re-`insert`).

### 2. Insert a batch → triples

```bash
pixi run -e habitat python hw1/api.py insert \
    --data-dir eval/_data/second_floor/baseline --floor 2
pixi run -e habitat python hw1/api.py insert \
    --data-dir eval/_data/second_floor/mixed --floor 2
```

Measures every paired frame (`avg_luma_rec601`, `valid_depth_fraction`), builds the
RDF, and PUTs it into the named graph for that batch (re-insert replaces just that
batch). The two evaluate conditions are two batches, `baseline` and `mixed`.

### 3. Retrieve valid frames → CSV

```bash
pixi run -e habitat python hw1/api.py retrieve \
    --batch mixed --out valid_frames.csv --floor 2 \
    [--brightness-min 18.04] [--brightness-max 252.84] [--valid-depth-min 0.493]
```

Runs a SPARQL SELECT that joins each frame to its component paths and **FILTERs** on
the quality bands. The three band flags are **the knob you explore** — each defaults
to the `--floor`'s reference band from [`definitions.md`](definitions.md) when
omitted, so a plain call reproduces the empirical gate, and overriding them lets
you see how the valid-frame set (and downstream mean L2) moves. The effective band
is printed each run. (`--floor` defaults to 1; pass `--floor 2` for the
second-floor captures above.)

CSV columns (exact header, consumed by `reconstruct.py --frames-csv`):

```
frame,rgb_path,depth_path,luma,valid_fraction
```

### Observables

**Two API-serviceable axes** (both single-sample computable, so checkable at
inference):

- **Brightness** — `avg_luma_rec601` (Rec.601 mean luma of the RGB frame), gated by
  `GoodBrightnessRange`.
- **Valid-depth** — `valid_depth_fraction` (fraction of depth pixels non-zero and
  inside the range band), gated by `ValidDepthRatio`.

The third sweep axis, **depth noise (`realized_sigma_z_m`), is excluded**: it needs
a paired clean-depth reference of the same trajectory, which no arbitrary query
sample has — un-computable at inference, so not API-serviceable.

The threshold **numbers** live in [`definitions.md`](definitions.md) (single source
of truth, mirrored by `api.py THRESHOLDS` — the default the retrieve flags fall back
to); how they were derived is in [`empirical-analysis.md`](empirical-analysis.md).

### Known limitation — the floor-2 brightness band does not gate

The floor-2 `GoodBrightnessRange` is **degenerate** ([18.04, 252.84] — the full
sweep range; flagged at `api.py` `THRESHOLDS[2]`). Consequence: **the SPARQL gate
does not detect the injected brightness effects** (flicker / low_light /
over_exposure) in a second-floor `mixed` batch — every frame passes the
brightness FILTER, and only the **valid-depth axis** (fed by the light→depth
coupling) can fail frames there. The authoritative record of *when* effects were
active is the scheduler's **`windows.json`** (per capture dir), which lives
deliberately **outside** the triplestore — the store holds observables only, so
cross-referencing gate PASS/FAIL against the windows is offline analysis, not
SPARQL. Schema and location are documented in
[`definitions.md`](definitions.md#temporal-window-regime-current).

The floor-1 bands, by contrast, are well-formed on both axes; they are retained
(unchanged) as the reference the synthetic ontology test grades against, even
though floor 1 no longer has configs or eval data.

---

## Tests

Two independent suites.

### Simulator pipeline suite (`packages/simulator/tests/`)

Unit tests (pure numpy: scheduler determinism, golden-array pixel pipeline) plus
the pipeline e2e gate (replay smoke, determinism, baseline invariance outside
windows, effects firing inside windows, hard-error trajectory paths, the
evaluate.py two-run flow, headless/no-pygame):

```bash
env -u PYTHONPATH pixi run -e habitat python -m pytest packages/simulator/tests/
```

The e2e cases need the Replica scene (`pixi run fetch-replica`) and **fail loud**
if it is missing rather than skip; `SIM_E2E_SKIP=1` is the explicit escape hatch
for machines without the scene. They replay a committed 10-pose fixture
(`tests/fixtures/mini_secondfloor.npy`), so the suite stays under ~2 minutes.

### Ontology regression (`hw1/test_e2e.py`)

A black-box integration test of the whole `api.py` + Fuseki round trip — it spins
up a **real** Fuseki server on a free port, so nothing is mocked. It:

1. synthesises a tiny 4-frame batch on disk with **known** per-frame luma and
   valid-depth (solid-grey RGB → exact Rec.601 luma; constant depth → exact valid
   fraction), so PASS/FAIL is deterministic;
2. `insert`s it into the store (Graph Store HTTP PUT, one named graph);
3. `retrieve`s it back through the SPARQL SELECT and asserts the CSV holds exactly
   the frames inside the band.

The 7 cases cover the pure measurers (`frame_luma`, `frame_valid_fraction`), the
`valid_frames.rq` template substitution, the default floor-1 band, a
`--brightness-min` override that narrows the set, an impossible band that returns
empty, and re-insert idempotency (named-graph replace, not append).

```bash
pixi run -e habitat test                          # pixi task → pytest hw1/test_e2e.py -v
pixi run -e habitat python -m unittest hw1.test_e2e   # no pytest needed (stdlib unittest)
```

The pixi `habitat` env ships `openjdk 21`, so `java` is on PATH inside
`pixi run -e habitat` with no extra setup. The Fuseki jar under `hw1/fuseki_bin/`
is fetched once by `pixi run -e fuseki fuseki`. If java or the jar is missing the
module **skips** rather than fails — it is an integration test, not a unit test.

Expected output:

```
============================= test session starts ==============================
platform linux -- Python 3.9.23, pytest-8.4.1, pluggy-1.6.0
collected 7 items

hw1/test_e2e.py::MeasurersAndTemplate::test_build_select_fully_substituted PASSED [ 14%]
hw1/test_e2e.py::MeasurersAndTemplate::test_frame_luma_matches_solid_value PASSED [ 28%]
hw1/test_e2e.py::MeasurersAndTemplate::test_valid_fraction_full_and_empty PASSED [ 42%]
hw1/test_e2e.py::InsertRetrieveE2E::test_reinsert_is_idempotent PASSED           [ 57%]
hw1/test_e2e.py::InsertRetrieveE2E::test_retrieve_brightness_override_narrows_set PASSED [ 71%]
hw1/test_e2e.py::InsertRetrieveE2E::test_retrieve_default_band PASSED            [ 85%]
hw1/test_e2e.py::InsertRetrieveE2E::test_retrieve_impossible_band_is_empty PASSED [100%]

============================== 7 passed in 1.73s ===============================
```

---

## Empirical grounding

The data-quality thresholds are **pipeline-specific conventions**: valid for *this*
geometry-only robust-ICP pipeline + coupling only, not transferable constants.
They were produced by OFAT sweeps under the legacy one-effect-per-config regime
and are documented canonically — this README does not restate the numbers or the
derivation.

- [`definitions.md`](definitions.md) — the resulting definitions, per-floor
  threshold tables (brightness + valid-depth; floor-1 archived as the
  synthetic-test reference), and the current temporal-window regime.
- [`empirical-analysis.md`](empirical-analysis.md) — the *why/how*: method, pass
  rule, knob rationale, per-knob generalization findings, and the current-regime
  caveats.

---

## Technical details

### Config-driven simulated environment, rendered through pygame's canvas

Every environment condition — scene, agent actuation, camera intrinsics /
extrinsics, lighting, depth faults, uncertainty scheduling, output — lives in one
YAML file (`hw1/configs/second_floor.yaml`); no code edit is needed to retune.
Each raw observation is passed through the config **once**
(`simulator.process_observations`) so that what you see in the pygame canvas is
*exactly* what gets written to disk. Habitat renders offscreen on EGL while
pygame owns only the on-screen CPU blit, so the two OpenGL contexts never
collide.

### Temporal uncertainty windows

The config's `uncertainties` block seeds an `UncertaintyScheduler` that lazily
draws an endless deterministic stream — `gap_s → duration_s → effect type →
repeat` — so effects arrive as **temporal windows** with fixed per-type severity
(`types:` in the config), not as separate corrupted configs. One seed
(`uncertainties.seed`) governs both the window sampling and the per-frame depth
RNG, so outside windows a scheduled run is **bit-identical** to a clean run. The
time base is wall-clock seconds in interactive collection and `t = i / fps` in
replay; the realized windows are serialized to `windows.json` per capture.

### Photometric simulation

Lighting is emulated as a **post-process on the RGB frame** (the Replica mesh is
flat/vertex-shaded, so in-sim lights largely no-op). The frame is scaled by an
exposure gain; inside a flicker window a periodic term modulates it:

```
brightness *= 1 + amplitude * sin(2*pi*frequency*t + phase)
```

with `phase = -2π·frequency·window_start_s`, so a window always starts at the
same point of the sine regardless of when it fires; plus ambient colour tint,
contrast, and gamma. In replay `t` is derived from the **frame index**
(`t = i / fps`), not the wall clock, so flicker is reproducible across runs and
machines given a fixed seed.

### Ambient-light coupling on the depth-sensor emulation

Habitat renders light-independent geometric depth, but a real structured-light /
ToF sensor degrades as scene light drives its emitter SNR down. The depth emulation
couples to the current exposure via `stress = |exposure - light_nominal|`:

```
noise_std    *= 1 + light_noise_gain   * stress
dropout_prob += light_dropout_gain     * stress     (clamped ≤ 1)
max_range    *= 1 - light_range_gain    * stress     (clamped ≥ 0)
```

so brighter- or darker-than-nominal lighting injects more depth noise, more
dropout, and a shorter usable range. This is the *only* path by which lighting
reaches the geometry-only reconstruction — which is exactly what makes the
robustness metric move (and the only axis the ontology gate can catch on floor 2;
see the [known limitation](#known-limitation--the-floor-2-brightness-band-does-not-gate)).
The coupling block in `configs/second_floor.yaml`:

```yaml
depth:
  noise_std: 0.005          # nonzero base so the light gain has something to scale
  light_nominal: 1.0
  light_noise_gain: 4.0
  light_dropout_gain: 0.3
  light_range_gain: 0.4
```
