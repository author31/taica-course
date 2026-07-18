# Empirical Analysis — How the Data-Quality Ranges Were Derived

> **LEGACY-REGIME NOTICE.** Sections 1–5 document the **one-effect-per-config OFAT regime**
> under which the data-quality thresholds were derived: per-condition configs
> (`baseline` / `flicker` / `low_light` / `over_exposure` batches per floor) swept by an
> `autoresearch.py` harness that is **not committed to this repo**. That regime has been
> replaced by **seeded temporal uncertainty windows** injected live during a single-config run
> — see [section 6](#6-current-regime--temporal-windows--two-run-eval). The narrative is kept
> intact as the derivation record for the thresholds that `hw1/api.py THRESHOLDS` and the
> ontology still carry (the floor-1 band also anchors the synthetic test `hw1/test_e2e.py`).

A short narrative of the empirical process behind the data-quality definitions.
Results live in [`definitions.md`](definitions.md); the experiment itself was
`autoresearch.py` (not in this repo — see the Provenance section of `definitions.md`).
This file is the *why* and *how*, not a re-listing of the tables.

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
(fully-clean) config. Each sweep point replayed the trajectory through the then-current
`hw1/load.py`, reconstructed geometry-only with light→depth coupling, and scored.

- **Metric:** coverage-aware correctness F-score (`hw1/completeness.py`,
  `F = 2AC/(A+C)`, τ = 0.10 m, higher = better). F rewards accuracy *and* coverage, so it
  can't be gamed by capturing fewer/easier points — a degradation that drops geometry
  shows up as a lower F.
- **Pass rule:** relative to a **per-axis** neutral reference,
  `F >= keep * F(axis_neutral)` with `keep = 0.5`. Per-axis (not global-clean) because the
  brightness axis carries the coupling's base noise, so its neutral F sits below the clean
  baseline; judging each axis against its own neutral isolates the knob's stress from that
  offset.
- **Boundary:** linear-interp crossing of that threshold, scanned outward from the max-F
  anchor (`find_crossing`), so out-of-band noise can't move an edge.
- **Baseline:** the clean neutral pass did double duty — it yielded the clean baseline F
  *and* built the whole-floor GT map (`completeness.build_gt_reference`) that every sweep
  point was scored against.

## 3. Knob rationale

Three knobs were considered; two survived into the API.

- **brightness** (`lighting.brightness`, coupling ON): geometry-only reconstruct never
  sees RGB, so brightness only reaches the metric through depth. The coupling was
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

- [`definitions.md`](definitions.md) — resulting definitions, per-floor tables (floor 1
  archived as the synthetic-test reference), `windows.json` schema, and provenance.
- `autoresearch.py`, `research_out*/` (curve PNGs, results.csv, definitions.yaml) — the
  original experiment and its raw outputs; **not committed to this repo** (historical
  derivation workspace only).

## 6. Current regime — temporal windows + two-run eval

The per-condition config batches are gone. The single config
(`hw1/configs/second_floor.yaml`) seeds an `UncertaintyScheduler`
(`packages/simulator/simulator/effects.py`) that draws an endless deterministic stream —
`gap_s → duration_s → effect type → repeat` (seconds) — so flicker / low_light /
over_exposure now arrive as **temporal windows** at fixed per-type severity during any run.
One seed governs the window sampling *and* the per-frame depth RNG, so frames outside every
window are bit-identical to a clean run. Realized windows are serialized to `windows.json`
per capture (ground truth, outside the ontology store).

`scripts/evaluate.py` replaces the old 8-config batch loop with a **two-run flow** from the
one config: scheduler OFF → `eval/_data/second_floor/baseline/` (also feeds the whole-floor
GT reference), scheduler ON → `eval/_data/second_floor/mixed/`. It reports whole-episode
mean L2 + F per condition (`eval/results.csv`) and a **per-window breakdown**
(`eval/per_window.csv`): each window's mixed-run mean L2 next to the baseline run over the
*same* frames, plus a trailing `clean` row for frames outside every window, and a radar
plot per effect type (`eval/radar_secondfloor.png`).

**Caveat — ICP brittleness dominates whole-episode L2.** The geometry-only ICP
reconstruction on the 454-pose second-floor trajectory is brittle: the baseline
whole-episode mean L2 varies strongly with the depth-noise RNG stream (i.e. with the choice
of `uncertainties.seed`), with divergence typically appearing late in the trajectory. A
mixed run can therefore post a whole-episode L2 at or even below the baseline's, purely
from a different noise realization. The **per-window rows** — mixed vs baseline over the
same frame span — are the meaningful robustness signal; treat absolute whole-episode L2 as
seed-conditional, not as a stable pipeline constant.
