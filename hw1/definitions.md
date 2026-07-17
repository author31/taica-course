# Data-Quality Definitions — Two Floors

Empirically derived data-quality ranges for the geometry-only robust-ICP SLAM pipeline
(`hw1/utils.reconstruct`). These are **pipeline-specific thresholds**: conventions valid for *this*
pipeline + coupling only, not transferable scientific constants. Produced by `autoresearch.py`
via OFAT sweeps.

- **Metric:** coverage-aware correctness F-score (`hw1/completeness.py`, F = 2AC/(A+C), τ = 0.10 m, higher is better)
- **Pass rule:** `F >= keep * F(axis_neutral)`, `keep = 0.5` (relative to per-axis neutral reference)
- **Range domain:** measured observable (not knob value), knob range kept alongside

## Summary

| Definition | Observable | First floor | Second floor |
|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` | **[146, 231]** (brightness 0.5–1.0) | *degenerate* — [18, 253] (full sweep) |
| ValidDepthRatio | `valid_depth_fraction` | **≥ 0.570** (max_range ≥ 2.0 m) | **≥ 0.493** (max_range ≥ 1.0 m) |

Clean baseline F: first floor **0.505**, second floor **0.240**.

## API threshold table

Single source of truth mirrored by `hw1/api.py THRESHOLDS`. Exact values:

| Floor | GoodBrightnessRange (`avg_luma_rec601`) | ValidDepthRatio (`valid_depth_fraction`) |
|---|---|---|
| 1 | [146.35, 230.87] | ≥ 0.570 |
| 2 | [18.04, 252.84] (DEGENERATE — non-gating) | ≥ 0.493 |

Only brightness + valid-depth are API-serviceable (single-sample computable); the noise axis is
excluded (needs a clean-depth reference).

## First floor (`trajectories/firstfloor.npy`)

Clean baseline F = 0.505.

| Definition | Observable range | Knob range | Axis-neutral F | Threshold |
|---|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` [146.35, 230.87] | `lighting.brightness` 0.5–1.0 | 0.589 | 0.295 |
| ValidDepthRatio | `valid_depth_fraction` [0.570, 1.000] | `depth.max_range` 2.0–10.0 m | 0.505 | 0.253 |

Both axes give clean, non-degenerate bands. Brightness fails on both dark and bright sides.

## Second floor (`trajectories/secondfloor.npy`)

Clean baseline F = 0.240 (harsher scene — lower ceiling F across all axes).

| Definition | Observable range | Knob range | Axis-neutral F | Threshold |
|---|---|---|---|---|
| GoodBrightnessRange | `avg_luma_rec601` [18.04, 252.84] ⚠ | `lighting.brightness` 0.1–3.0 | 0.090 | 0.045 |
| ValidDepthRatio | `valid_depth_fraction` [0.493, 1.000] | `depth.max_range` 1.0–10.0 m | 0.240 | 0.120 |

⚠ **GoodBrightnessRange is degenerate.** The brightness-axis neutral F is already very low
(0.090), so the pass threshold (0.045) is passed by all 11 sweep points — "never fails on low
side / high side", band = full sweep range. The definition is **not usable** on this floor: the
scene is too far from the ICP working point for the brightness coupling to produce a measurable
cliff above the noise floor. Depth-ratio axis remains well-formed.

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

## Provenance

- Script: `autoresearch.py`
- Results: `research_out/results.csv` (floor 1), `research_out_secondfloor/results.csv` (floor 2)
- Curves: `research_out/curve_*.png`, `research_out_secondfloor/curve_*.png`
- Definitions: `research_out/definitions.yaml`, `research_out_secondfloor/definitions.yaml`

## Reproduce

Run from repo root:

- Floor 1: `pixi run -e habitat python autoresearch.py --trajectory trajectories/firstfloor.npy`
- Floor 2: `pixi run -e habitat python autoresearch.py --trajectory trajectories/secondfloor.npy --out-dir research_out_secondfloor`
