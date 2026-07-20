# Data-Quality Definitions — Empirical Bands + Temporal-Window Regime

Empirically derived data-quality ranges for the geometry-only robust-ICP SLAM pipeline
(`hw1/utils.reconstruct`). These are **pipeline-specific thresholds**: conventions valid for *this*
pipeline + coupling only, not transferable scientific constants.

> **Regime note.** The bands below were derived under the **legacy one-effect-per-config
> (OFAT) regime** by an `autoresearch.py` sweep harness that is **not committed to this repo**
> (see [Provenance](#provenance-historical)). The pipeline has since moved to **seeded temporal
> uncertainty windows** injected live during a single-config run (see
> [Temporal-window regime (current)](#temporal-window-regime-current)). The threshold *numbers*
> are retained verbatim: they remain the `hw1/api.py THRESHOLDS` defaults, and the floor-1 band
> is the reference the synthetic ontology test (`hw1/test_e2e.py`) grades against. The first
> floor itself no longer has a config, eval data, or a committed trajectory.

- **Metric:** coverage-aware correctness F-score (`hw1/completeness.py`, F = 2AC/(A+C), τ = 0.10 m, higher is better)
- **Pass rule:** `F >= keep * F(axis_neutral)`, `keep = 0.5` (relative to per-axis neutral reference)
- **Range domain:** measured observable (not knob value), knob range kept alongside

## Summary

| Definition | Observable | First floor (archived) | Second floor |
|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` | **[146, 231]** (brightness 0.5–1.0) | *degenerate* — [18, 253] (full sweep) |
| ValidDepthRatio | `valid_depth_fraction` | **≥ 0.570** (max_range ≥ 2.0 m) | **≥ 0.493** (max_range ≥ 1.0 m) |

Clean baseline F at derivation time: first floor **0.505**, second floor **0.240**.

## API threshold table

Single source of truth mirrored by `hw1/api.py THRESHOLDS`. Exact values:

| Floor | GoodBrightnessRange (`avg_luma_rec601`) | ValidDepthRatio (`valid_depth_fraction`) |
|---|---|---|
| 1 | [146.35, 230.87] | ≥ 0.570 |
| 2 | [18.04, 252.84] (DEGENERATE — non-gating) | ≥ 0.493 |

Only brightness + valid-depth are API-serviceable (single-sample computable); the noise axis is
excluded (needs a clean-depth reference).

## First floor — ARCHIVED (synthetic-test reference)

> **Historical.** The first floor was dropped from configs / eval data / trajectories; its
> derivation trajectory (`trajectories/firstfloor.npy`) is not in the repo. The numbers are kept
> **unchanged** because they are load-bearing: `api.py THRESHOLDS[1]` defaults, the floor-1
> individuals in `ontology/hw1.ttl`, and `hw1/test_e2e.py`'s deterministic PASS/FAIL fixture all
> depend on exactly this band. Do not edit them without re-deriving the whole set.

Clean baseline F = 0.505.

| Definition | Observable range | Knob range | Axis-neutral F | Threshold |
|---|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` [146.35, 230.87] | `lighting.brightness` 0.5–1.0 | 0.589 | 0.295 |
| ValidDepthRatio | `valid_depth_fraction` [0.570, 1.000] | `depth.max_range` 2.0–10.0 m | 0.505 | 0.253 |

Both axes give clean, non-degenerate bands. Brightness fails on both dark and bright sides.

## Second floor (`trajectories/secondfloor.npy`)

The active floor — `trajectories/secondfloor.npy` (454 poses) is committed and drives
`scripts/evaluate.py`. Clean baseline F at derivation time = 0.240 (harsher scene — lower
ceiling F across all axes).

| Definition | Observable range | Knob range | Axis-neutral F | Threshold |
|---|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` [18.04, 252.84] ⚠ | `lighting.brightness` 0.1–3.0 | 0.090 | 0.045 |
| ValidDepthRatio | `valid_depth_fraction` [0.493, 1.000] | `depth.max_range` 1.0–10.0 m | 0.240 | 0.120 |

⚠ **GoodBrightnessRange is degenerate.** The brightness-axis neutral F is already very low
(0.090), so the pass threshold (0.045) is passed by all 11 sweep points — "never fails on low
side / high side", band = full sweep range. The definition is **not usable** on this floor: the
scene is too far from the ICP working point for the brightness coupling to produce a measurable
cliff above the noise floor. Depth-ratio axis remains well-formed.

**Consequence under the current regime:** the ontology gate cannot detect the injected
brightness-type windows (flicker / low_light / over_exposure) on floor 2 — only the
valid-depth axis gates there, via the light→depth coupling. The window ground truth lives in
`windows.json` (below), outside the store.

## Why no depth-noise definition

`realized_sigma_z_m` (paired noisy−clean std) was **dropped as a query observable**: measuring it
needs a clean-depth reference of the *same* trajectory, which the sweep has but no arbitrary API
query sample does. Un-computable at inference → un-checkable range → useless for the ontology API.
A reference-free σ_z proxy (local plane-fit residual, temporal std) would need its own re-measured
band, not this one. Only `avg_luma_rec601` and `valid_depth_fraction` are single-sample computable
and thus API-serviceable.

## Generalization notes

- **Depth-ratio** (max_range far-loss) generalizes: both floors keep ~half the valid-depth fraction.
- **Brightness** does **not** generalize — clean on floor 1, degenerate on floor 2. Cross-scene brightness
  thresholds need either a per-scene neutral calibration or a stronger coupling gain tuned to the harsher scene.

## Temporal-window regime (current)

Uncertainties are no longer one-effect-per-config. The single config
(`hw1/configs/second_floor.yaml`, `uncertainties:` block) seeds an `UncertaintyScheduler`
(`packages/simulator`) that lazily draws an endless deterministic stream —
`gap_s → duration_s → effect type → repeat` (seconds, uniform per draw) — so any run contains
**temporal windows** of flicker / low_light / over_exposure at fixed per-type severity. One seed
governs both the window sampling and the per-frame depth RNG; frames outside every window are
bit-identical to a clean run.

- **Interactive collection** (`hw1/load.py`): t = wall-clock seconds since session start —
  windows fire live; `--clean` disables the scheduler.
- **Replay / evaluate** (`scripts/evaluate.py`): t = frame_index / fps (default 30) —
  deterministic. Evaluate does a two-run flow from the one config: scheduler OFF →
  `eval/_data/second_floor/baseline/`, ON → `eval/_data/second_floor/mixed/`.

### Window ground truth — `windows.json`

Whenever the scheduler ran, the realized windows are serialized to
`<capture_root>/windows.json` (e.g. `eval/_data/second_floor/mixed/windows.json`). This file is
the **authoritative record of when effects were active**; it is kept deliberately **outside**
the triplestore (the ontology holds per-frame observables only), so cross-referencing gate
PASS/FAIL against windows is offline analysis, not SPARQL. Schema — a JSON list, times in
seconds:

```json
[
  {
    "start_s": 3.1203,
    "end_s": 5.5037,
    "type": "over_exposure",
    "params": { "brightness": 2.2, "contrast": 1.2 }
  }
]
```

`evaluate.py` maps each window to 1-indexed frame bounds via
`frame = round(t_s * fps)` (replay time base `t = i / fps`) for the per-window rows in
`eval/per_window.csv`.

## Provenance (historical)

The derivation harness and its outputs are **not committed to this repo**: `autoresearch.py`
(OFAT sweep script), `trajectories/firstfloor.npy`, and the `research_out/` /
`research_out_secondfloor/` result trees (results.csv, curve_*.png, definitions.yaml) existed
only in the original derivation workspace. The historical invocations were
`autoresearch.py --trajectory trajectories/{firstfloor,secondfloor}.npy`; they cannot be re-run
from this repo. The tables above are therefore a **recorded convention**, mirrored in code by
`hw1/api.py THRESHOLDS` and regression-locked by `hw1/test_e2e.py`.
