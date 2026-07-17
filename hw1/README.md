# Homework 1 — Geometry-only ICP SLAM under Environment Uncertainty

You implement a **geometry-only ICP SLAM** pipeline (point-to-plane ICP over
RGB-D frames) and score it by the **mean L2 distance** between your predicted
camera trajectory and the ground truth. The twist: the input data is *not clean*.
Lighting and depth-sensor faults are injected through a single config file, and
part of the grade is how well your algorithm holds up.

- **Simulator:** [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) 0.3.3.
- **Environment:** Replica `apartment_0` (one mesh, two navigable storeys).
- **First floor** — agent spawns at ground level, `start_position: [0.0, 0.0, 0.0]`
  (`y = 0`). This is the *development* environment.
- **Second floor** — agent spawns on the upper storey,
  `start_position: [0.979, 1.425, 3.773]`. This is the *generalization*
  environment. (Note: the intuitive `y = 8` has **no navmesh** in `apartment_0` —
  the mesh's navigable region tops out around `y = 5.4` — so the real upper storey
  at `y ≈ 1.5` is used instead.)
- **Pygame integration:** navigation run in a live pygame window
  (first-person RGB + depth + a top-down bird's-eye panel), so you can *see* every
  frame exactly as it is captured, with all the injected faults already applied.

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
| `load.py` | Simulator launcher / data collector — builds the habitat sim + agent, drives it interactively (pygame) or replays a trajectory, and writes each capture (`rgb/`, `depth/`, `GT_pose.npy`) with the config's faults baked in. |
| `utils.py` | The ICP SLAM library **you implement** — point-cloud unprojection, normal/feature preprocessing, ICP (`local_icp_algorithm` / `my_local_icp_algorithm`), the `reconstruct(...)` loop, and `mean_l2(...)`. |
| `reconstruct.py` | Thin CLI over `utils.py` — reconstructs one floor's capture, prints mean L2, and opens an Open3D window with the estimated (red) and GT (black) trajectories. |
| `completeness.py` | Coverage-aware correctness — builds the whole-floor GT map, anchors a reconstruction into the world frame, and returns accuracy / completeness / **F-score**. |
| `api.py` | Data-quality query CLI — grades a directory of frames per-frame on brightness + valid-depth against the empirical thresholds (see [Data-quality API](#data-quality-api)). |
| `definitions.md` | **Canonical**: empirical data-quality definitions, per-floor threshold tables, and reproduce commands. |
| `empirical-analysis.md` | **Canonical**: narrative of the empirical process — how the thresholds were derived (OFAT sweeps, pass rule, per-knob findings). |

The grader's orchestrator, `scripts/evaluate.py`, lives outside `hw1/` (see
[Evaluation dimensions](#evaluation-dimensions)).

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

**Collect / replay a capture** with the launcher (`hw1/load.py` is the canonical
launcher — not `scripts/load.py`). Interactive pygame drive, or headless replay of
a saved trajectory from `trajectories/`:

```bash
# interactive (pygame keyboard)
pixi run -e habitat python hw1/load.py --config scripts/config.yaml

# headless replay of a trajectory
pixi run -e habitat python hw1/load.py --config scripts/config.yaml \
    --trajectory trajectories/firstfloor.npy
```

**Reconstruct + visualize one floor** with the thin runner:

```bash
pixi run -e habitat python hw1/reconstruct.py --floor 1 --version open3d
pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/baseline/
```

It prints the mean L2 and opens an Open3D window (add `--no-vis` for the metric
only).

---

## Evaluation dimensions

Your pipeline is scored along two axes, both driven by the same three lighting
perturbations:

1. **Robustness** — stability under perturbation *inside the development
   environment* (first floor).
2. **Generalization** — behaviour *outside* the development configuration: the new
   environment (second floor) under a **shifted, harsher** perturbation
   distribution.

Registration is **geometry-only** (no colour), so lighting can only move the
metric through the **depth sensor's ambient-light coupling** — exposure brighter
or darker than nominal raises depth noise / dropout and shrinks usable range (see
[Technical details](#technical-details)). Each axis changes **only** the listed
`lighting` value(s) relative to the neutral baseline (`brightness 1.0`,
`amplitude 0.0`):

| Axis | First floor (robustness) | Second floor (generalization, harsher) |
|---|---|---|
| Low light | `brightness: 0.3` | `brightness: 0.2` |
| Over exposure | `brightness: 2.5` | `brightness: 3.0` |
| Flickering | `amplitude: 0.8`, `frequency: 5.0` | `amplitude: 0.9`, `frequency: 6.0` |

The grader's orchestrator `scripts/evaluate.py` replays each (floor, axis)
through its config, reconstructs with `hw1/utils.py`, scores both mean L2 **and**
the coverage-aware F-score (`hw1/completeness.py`), and emits
`eval/results.csv` + `eval/fscore.csv` + one radar chart per floor:

```bash
pixi run -e habitat python scripts/evaluate.py
```

The radar reads as "how far does each perturbation push the error away from the
clean baseline" — a tight polygon near the dashed baseline ring is robust; a large
polygon is fragile. (See [Appendix](#appendix--radar-charts-per-floor).)

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

The two clouds are placed in the same frame by **one** transform (the first
camera's GT pose), with no trajectory fitting / ICP / RANSAC — nothing to overfit,
so leftover drift stays visible in the score. Implementation and the anchoring
transform are in `hw1/completeness.py`; the metric parameters (τ = 0.10 m, the
pass rule) are pinned in [`definitions.md`](definitions.md).

---

## Data-quality API

`hw1/api.py` answers one question for a directory of captured frames: *is each
frame's data quality inside the empirically-derived "Good" range for this
pipeline?* It grades **per frame** on two axes and prints a table, optionally
writing a CSV report and a PASS manifest.

```bash
pixi run -e habitat python hw1/api.py --data-dir DIR --floor {1,2} \
    [--out report.csv] [--manifest manifest.json]
```

- `--data-dir` must contain `rgb/` and `depth/` subdirs of integer-stem `.png` frames.
- `--floor` selects which floor's empirical thresholds to grade against.

**Two API-serviceable axes** (both single-sample computable, so checkable at
inference):

- **Brightness** — observable `avg_luma_rec601` (Rec.601 mean luma of the RGB frame).
- **Valid-depth** — observable `valid_depth_fraction` (fraction of depth pixels that
  are non-zero and inside the range band).

The third sweep axis, **depth noise (`realized_sigma_z_m`), is excluded**: it needs
a paired clean-depth reference of the same trajectory, which no arbitrary query
sample has — un-computable at inference, so not API-serviceable.

Output columns (stdout table and `--out` CSV):

```
frame,luma,valid_fraction,brightness_ok,valid_depth_ok,verdict
```

`verdict` is `PASS` iff **both** axes are ok. `--manifest` writes the sorted stems
of the PASS frames as a JSON list.

The threshold **numbers** live in [`definitions.md`](definitions.md) (single source
of truth, mirrored by `api.py THRESHOLDS`); how they were derived is in
[`empirical-analysis.md`](empirical-analysis.md). Note the floor-2 brightness band
is **degenerate** (full-range, non-gating) — `--floor 2` emits a stderr warning and
only the valid-depth axis is meaningful there.

---

## Empirical grounding

The data-quality thresholds are **pipeline-specific conventions**: valid for *this*
geometry-only robust-ICP pipeline + coupling only, not transferable constants.
They were produced by OFAT sweeps and are documented canonically — this README
does not restate the numbers or the derivation.

- [`definitions.md`](definitions.md) — the resulting definitions, per-floor
  threshold tables (brightness + valid-depth), and reproduce commands.
- [`empirical-analysis.md`](empirical-analysis.md) — the *why/how*: method, pass
  rule, knob rationale, and per-knob generalization findings.

---

## Technical details

### Config-driven simulated environment, rendered through pygame's canvas

Every environment condition — scene, agent actuation, camera intrinsics /
extrinsics, lighting, depth faults, output — lives in one YAML file (see
`configs/*.yaml`); no code edit is needed to retune. Each raw observation is passed
through the config **once** (`process_observations`) so that what you see in the
pygame canvas is *exactly* what gets written to disk. Habitat renders offscreen on
EGL while pygame owns only the on-screen CPU blit, so the two OpenGL contexts never
collide.

### Photometric simulation

Lighting is emulated as a **post-process on the RGB frame** (the Replica mesh is
flat/vertex-shaded, so in-sim lights largely no-op). The frame is scaled by an
exposure gain with an optional periodic flicker:

```
brightness *= 1 + amplitude * sin(2*pi*frequency*t + phase)
```

plus ambient colour tint, contrast, and gamma. In replay `t` is derived from the
**frame index** (`t = i / fps`), not the wall clock, so flicker is reproducible
across runs and machines given a fixed seed.

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
robustness / generalization metric move. The shared coupling block (identical
across all configs) is:

```yaml
depth:
  noise_std: 0.005          # nonzero base so the light gain has something to scale
  light_nominal: 1.0
  light_noise_gain: 4.0
  light_dropout_gain: 0.3
  light_range_gain: 0.4
```

