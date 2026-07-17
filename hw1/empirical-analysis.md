# Empirical Analysis — How the Data-Quality Ranges Were Derived

A short narrative of the empirical process behind the data-quality definitions.
Results and repro commands live in [`definitions.md`](definitions.md); the experiment
itself is [`../autoresearch.py`](../autoresearch.py). This file is the *why* and *how*,
not a re-listing of the tables.

## 1. Question

The ontology API needs Good ranges for two observables (`avg_luma_rec601`,
`valid_depth_fraction`) so a caller can ask "is this capture usable?". These are
scene- and pipeline-dependent conventions, *not* scientific
constants. There is no universal "good brightness"; there is only a brightness band under
which *this* geometry-only robust-ICP pipeline (`hw1/utils.reconstruct`) still
reconstructs the floor it was tuned on. So the bands have to be *measured* against this
pipeline, per floor, rather than assumed.

## 2. Method

**OFAT** (one-factor-at-a-time): sweep one knob, hold everything else at the neutral
(fully-clean) config. Each sweep point replays the trajectory through `hw1/load.py`,
reconstructs geometry-only with light→depth coupling, and scores.

- **Metric:** coverage-aware correctness F-score (`hw1/completeness.py`,
  `F = 2AC/(A+C)`, τ = 0.10 m, higher = better). F rewards accuracy *and* coverage, so it
  can't be gamed by capturing fewer/easier points — a degradation that drops geometry
  shows up as a lower F.
- **Pass rule:** relative to a **per-axis** neutral reference,
  `F >= keep * F(axis_neutral)` with `keep = 0.5`. Per-axis (not global-clean) because the
  brightness axis carries the coupling's base noise, so its neutral F sits below the clean
  baseline; judging each axis against its own neutral isolates the knob's stress from that
  offset. See `main` in `autoresearch.py`.
- **Boundary:** linear-interp crossing of that threshold, scanned outward from the max-F
  anchor (`find_crossing`), so out-of-band noise can't move an edge.
- **Baseline:** the clean neutral pass does double duty — it yields the clean baseline F
  *and* builds the whole-floor GT map (`completeness.build_gt_reference`) that every sweep
  point is scored against.

## 3. Knob rationale

Three knobs were considered; two survived into the API.

- **brightness** (`lighting.brightness`, coupling ON): geometry-only reconstruct never
  sees RGB, so brightness only reaches the metric through depth. The coupling is
  deliberately **noise-only** — `light_noise_gain = 6`, base `noise_std = 0.002`,
  dropout/range gains = 0. Full 3-channel coupling made F non-monotone: mild stress raises
  dropout/range, which *help* (fewer far points, coverage self-heals) and cancel the noise
  hit into mush. This matches the ablation finding that depth noise is ~100% of the real
  lighting-induced failure. Zeroing dropout/range gives a clean, monotone-in-stress,
  two-sided band.
- **depth_ratio** (`depth.max_range` shrink, coupling OFF): implemented as **structured**
  far-loss — the same far pixels vanish every frame — *not* random dropout. Uniform
  dropout is benign here: ICP keeps ample points and coverage self-heals across the ~387
  frames. Only same-pixels-gone-every-frame produces a persistent coverage hole and lost
  ICP constraints that degrade F.
- **depth_noise** (`depth.noise_std`, direct): dropped from the API. The realized σ_z
  observable needs a paired clean-depth reference of the *same* trajectory to measure —
  which the sweep has but an arbitrary live API query sample does not. Un-computable at
  inference → un-checkable → useless as an ontology query field.

## 4. Findings

Per-floor bands are in `definitions.md`; the empirical takeaway is that generalization
differs sharply between the two knobs.

- **Depth-ratio generalizes.** Both floors keep ~half the valid-depth fraction: floor-1
  `valid_depth_fraction ≥ 0.570` (max_range ≥ 2.0 m), floor-2 `≥ 0.493` (max_range ≥
  1.0 m). Well-formed cliff on both.
- **Brightness does not generalize.** Clean two-sided band on floor 1
  (`avg_luma_rec601 ∈ [146, 231]`), but **degenerate** on floor 2. Floor 2 is a harsher
  scene (clean baseline F 0.240 vs 0.505), and its brightness-axis neutral F is already
  only ≈ 0.090 — so the pass threshold (0.5 × 0.090 = 0.045) is cleared by *every* sweep
  point. The result is a full-range, non-gating band ([18, 253]): the scene sits too far
  from the ICP working point for the brightness coupling to carve a measurable cliff above
  the noise floor. It is reported but flagged unusable.

Cross-scene brightness gating would need a per-scene neutral calibration or a stronger
coupling gain tuned to the harsher floor.

## 5. Pointers

- [`../autoresearch.py`](../autoresearch.py) — the experiment (GRIDS, COUPLING_ON,
  build_config, replay, score_capture, find_crossing, main).
- [`definitions.md`](definitions.md) — resulting definitions, per-floor tables, and repro
  commands.
- `research_out/curve_*.png`, `research_out_secondfloor/curve_*.png` — per-knob F curves
  with threshold line and shaded Good band.
- `research_out/results.csv`, `research_out_secondfloor/results.csv` — raw sweep rows.
