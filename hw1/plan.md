# HW1 Implementation Plan

Concrete build plan for the TAICA HW1 assignment, derived from [`IDEA.md`](IDEA.md)
after a design review. This document supersedes `IDEA.md` wherever they disagree —
`IDEA.md` stays as the record of original intent.

**Audience of the artifact being built:** students. Everything under `hw1/` that
students touch ships as runnable boilerplate with `#TODO` docstrings; the reference
solution lives instructor-side and is never published.

---

## 1. Locked design decisions

| # | Question | Decision | Consequence |
|---|---|---|---|
| D1 | Algorithm | **Geometry-only ICP.** RGB is a *declared proxy*, not a cause. | RGB quality axes must be documented as upstream indicators reaching geometry only through the simulated light→depth coupling. Students must be graded on recognizing this, not on asserting "brightness breaks ICP". |
| D2 | Thresholds | **Students derive them.** We ship definitions + measurement contract; no numbers. | `api.py THRESHOLDS` is removed from the student tree. Students produce `hw1/thresholds.json`. Instructor-side reference bands are re-derived but never published. |
| D3 | Semantics | **Dropped.** `semantic/` leaves the dataset spec. | `output.save_semantic: false` stays; `IDEA.md`'s dataset structure section is corrected to `rgb/` + `depth/` only. |
| D4 | Floors | **Phase 1 = floor 1 (provided, frozen). Phase 2 = floor 2 (student-collected).** | Floor-1 assets must be regenerated from scratch — see §3. Floor-2 assets already exist. |
| D5 | Coupling visibility | **Hidden in phase 1, revealed in phase 2.** | Phase-1 dataset ships as a frozen capture with no *uncertainties or coupling* config. It **does** ship `intrinsics.json` (`width`, `height`, `hfov`) — without it `depth_image_to_point_cloud` is unimplementable on the very dataset it is first used against, and camera parameters reveal nothing about the uncertainty regime. **Every** capture carries its own, phase-2 collections included: reconstruction reads intrinsics from the capture being reconstructed, never from a config, so one code path serves both floors and cannot unproject one floor's depth with another floor's camera. The capture-writing path in `packages/simulator` emits it, so `load.py` and `evaluate.py` both inherit it. Phase-2 `configs/second_floor.yaml` exposes the full `uncertainties` + `depth.light_*_gain` block. |
| D6 | Derivation target | **ICP degradation only.** No effect labels or other artifacts shipped. | Phase-1 dataset = `rgb/` + `depth/` + `GT_pose.npy` + `intrinsics.json` (D5). Nothing else. Band quality is judged by what it does to reconstruction, never by effect-detection accuracy. D14 makes this self-enforcing: `windows.json` no longer exists to leak. |
| D7 | Quality axes | **Two categories, four quality factors:** depth (2 — `ValidDepthRatio`, `DepthRoughness`), RGB (2 — `HighlightClipping`, `ShadowClipping`). | See §4.2.1 for formulas and citations. |
| D17 | Factors read raw rasters only | **Every factor is computed directly from the `depth/*.png` and `rgb/*.png` files.** No unprojection, no point clouds, no normal estimation, no Open3D. | Replaced `frame_geometric_conditioning` (λ_min of the 6×6 point-to-plane normal matrix) with `frame_depth_roughness` — see §4.2.0. Restores `api.py`'s own stated doctrine, *"Standard library + numpy + Pillow + rdflib. No OpenCV, no Open3D"*, which the conditioning factor had broken. Factors grade *input data*, so they must depend only on what a data file contains — never on a reconstruction pipeline that may itself be unimplemented. |
| D11 | Metric scope | **Single-frame only, everywhere. Sharpness and noise metrics are excluded *from the RGB factors specifically*.** | Read the scope carefully: D11 is RGB-scoped, and D17 makes `DepthRoughness` — an Immerkær noise estimator — factor 2 *on depth*, which is not a contradiction. This simulator injects neither RGB blur nor RGB noise, so a sharpness metric on RGB would measure exposure gain while claiming to measure focus; it *does* inject depth noise, so the same estimator on depth measures exactly what it claims. Variance of Laplacian and Tenengrad stay out of the student docs. Temporal metrics are struck everywhere, preserving `api.py`'s "single-sample computable" doctrine. |
| D12 | Channel reduction | **`V = max(R, G, B)`. No luma anywhere.** | Rec.601/709 weights encode a colorimetric convention this renderer never promised, and weighting *hides single-channel saturation* (see §4.2.1). `frame_luma` is deleted from the student tree and `avgLuma` leaves the TBox; the naive baseline becomes `mean(V)`. |
| D13 | Clipping thresholds τ | **Student-derived, alongside the bands.** | τ is a *measurement* parameter, so the stored observables now differ per student. Two consequences, both handled: grading moves onto the τ-agnostic downstream reconstruction outcome (§6), and the ontology must record τ as batch-level provenance (§4.3) or the triples become uninterpretable. |
| D14 | Uncertainty regime | **Spatial zones, flicker-only, on both floors.** `low_light` and `over_exposure` are removed; effect depends on **where** the agent is, not when. | Architecture change, not a config edit — see §3.1. `UncertaintyScheduler` and `windows.json` are deleted; `Engine.observe` gains a position lookup; `uncertainties.seed` is demoted to the depth RNG only. Both floors share the mechanism, so only the *scene* changes between phases and a failed band transfer has one cause instead of two. |
| D15 | Zone geometry | **Circles** (`center: [x,z]`, `radius`), **hard edges**, first match wins by list order. | Stateless and trivially testable. The sharp position↔degradation correlation is the phase-2 lesson; a smooth falloff would blur exactly the signal students are meant to find. |
| D16 | Zone severity | **Ladder of three** — benign / moderate / severe (amplitude 0.20 / 0.60 / 0.95) plus the clean region outside all zones. | Gives a within-capture dose-response curve instead of a binary, and the benign zone teaches *effect present ≠ quality bad*. The benign claim must be verified against the depth coupling at M3 (§3.1) or it is false. |
| D8 | Ontology depth | **Flat — SELECT + FILTER.** No CONSTRUCT rules, no subsumption reasoning. | Accepted scope call. One mitigation folded in: a second query joining two batches' named graphs (e.g. `floor1_baseline` and `floor1_mixed`), so the store does at least one thing a single dataframe filter does not. It needs a `compare` subcommand to be runnable — see §4.4. |
| D9 | Derivation protocol | **Open — students design the experiment.** | We state only the objective. Grading shifts weight onto experimental design; the rubric (§6) must make design quality explicit and gradable. |
| D10 | ICP boilerplate | **From-scratch ICP + reconstruct loop are `#TODO`; the Open3D wrapper ships working** as the reference baseline students compare against. | See §4.1. |

### Open item

**O1 — held-out generalization set: TBD.** Two candidates, decide before §7 M4:

- **(a) Instructor holdout capture.** Ship `mixed/` for derivation; keep a
  `mixed_holdout/` (same floor-1 trajectory, **displaced zone centres** — the scheduler
  is seedless after §3.1, so a different seed no longer varies anything)
  instructor-side. Autogradable, directly comparable across students.
- **(b) Phase-2 transfer only.** The floor-1 band is applied to the student's own
  floor-2 capture and the report explains whether it transferred. No extra assets;
  not comparable across students.

Everything else in this plan is agnostic to the choice; (a) only adds one capture
step in §3 and one autograded line in §6.

**O2 — RESOLVED (D11): single-frame only.** `api.py`'s "single-sample computable,
so checkable at inference" doctrine stands unchanged. Consequence to state precisely
in the student docs — precisely, because the loose version is wrong: a per-frame
metric **does** fire on the extreme frames inside a flicker zone (amplitude 0.95 drives
individual frames to both crush and blow-out), it simply **cannot identify the
modulation as periodic**. Flicker is not invisible; its *periodicity* is. Under D14
this matters more, not less, since flicker is now the *only* effect — but it also
matters less in practice, because what the student actually needs to recover is the
**spatial** structure, and position is recoverable from `GT_pose.npy`.

---

## 2. Repository split

Answers currently leak throughout the tree. Two-tree layout:

**Student-facing (published):**

```
hw1/
  README.md              # rewritten — no derived numbers, no degeneracy findings
  load.py                # ships working
  reconstruct.py         # ships working
  completeness.py        # ships working (it is the grader)
  utils.py               # partially #TODO — §4.1
  api.py                 # partially #TODO — §4.2
  ontology/hw1.ttl       # TBox ships; quality-factor individuals #TODO
  queries/*.rq           # skeletons with #TODO graph patterns
  test_e2e.py            # ships working — the API contract students code against
  configs/second_floor.yaml   # phase 2 only; coupling visible
packages/simulator/      # ships working
scripts/evaluate.py      # ships working
docs/, pixi.toml, README.md
```

**Instructor-only (never published — separate private branch or `instructor/` excluded from the student remote):**

```
instructor/
  README.md               # layout, per-file milestone ownership, leak-enforcement rule
  definitions.md          # currently deleted in the worktree — move here, do not restore to hw1/
  empirical-analysis.md   # same
  autoresearch.py         # sweep harness (not currently committed)
  thresholds.reference.json   # an M3 OUTPUT — does not exist until the bands are derived;
                              # until then the reference CLI needs an explicit --thresholds
  solution/               # reference utils.py + api.py implementations
  configs/first_floor.yaml
  trajectories/firstfloor.npy
```

**`hw1/plan.md` and `hw1/IDEA.md` are the largest leak in the tree and must move.**
They sit inside the student-facing directory while containing, between them, every
derived band (`[146.35, 230.87]`, `[18.04, 252.84]`, the valid-depth minima), the
the zone coordinates and amplitude ladder, the
light→depth coupling gains, and the full rationale for every answer students are meant
to derive. Publishing `hw1/` as it stands ships the answer key with the assignment.
Move both to `instructor/` before M6, or the entire §2 split is decorative. Not done
in-flight only because `hw1/plan.md` is the path this work is actively being tracked
against.

**Leak audit before publishing** — the current `hw1/README.md` must lose:

- the `THRESHOLDS` table and every derived number,
- the "Known limitation — the floor-2 brightness band does not gate" section,
- the "Empirical grounding" section and its links,
- the `definitions.md` / `empirical-analysis.md` file-map rows.

**Honest caveat on D5.** Hiding the coupling means *not shipping the config numbers*
with the phase-1 dataset. The mechanism itself is readable in
`packages/simulator/simulator/effects.py:apply_depth_sensor`. A student who reads
the simulator source can infer *that* lighting couples to depth, but not the
severities or the schedule. This is acceptable — inferring the mechanism from source
is itself legitimate investigation — but the plan should not pretend the coupling is
secret.

---

## 3. Simulator change and asset regeneration

### 3.1 Spatial uncertainty zones (replaces the temporal scheduler)

**Both floors move to spatial, flicker-only uncertainty.** `low_light` and
`over_exposure` are removed; the effect an agent sees depends on **where it is**, not
on when it got there.

This is an architecture change, not a config edit. Today `UncertaintyScheduler.active(t)`
(`effects.py:231`) samples a seeded `gap → duration → type` stream and
`Engine.observe(t)` (`engine.py:188`) calls it with time only — position is never
consulted. It is available though: `self.agent.get_state()` is already in scope, so the
change is contained to `Engine.observe` plus one new class.

**Net simplification.** A zone lookup is a pure function of `(position, t)`: no seed, no
RNG, no lazy unroll, no cursor, no realized-window list, no `windows.json`. The entire
stateful machinery of `UncertaintyScheduler` is deleted, and `uncertainties.seed` is
demoted to governing **only** the per-frame depth RNG.

**Config schema**, identical in shape for both floors:

```yaml
uncertainties:
  enabled: true          # evaluate.py flips this false for the baseline run
  mode: spatial
  seed: 20               # NOW ONLY the per-frame depth RNG
  zones:                 # first match wins (list order = priority)
    - name: benign
      center: [x, z]     # world metres, XZ; y is implied by the floor
      radius: 2.0        # hard edge: (x-cx)^2 + (z-cz)^2 <= r^2
      flicker: { amplitude: 0.20, frequency: 0.8 }
    - name: moderate
      center: [x, z]
      radius: 2.0
      flicker: { amplitude: 0.60, frequency: 1.2 }
    - name: severe
      center: [x, z]
      radius: 2.0
      flicker: { amplitude: 0.95, frequency: 2.0 }
# removed: gap_s, duration_s, types
```

**`ZoneScheduler`** replaces `UncertaintyScheduler`:

- `active(t, position) -> {}` outside every zone, else
  `{"lighting": {"amplitude": a, "frequency": f, "phase": 0.0}}`.
- **Stateless and seedless.** `phase` is pinned at 0, so the oscillation is
  `1 + a·sin(2πft)` on the global clock. Zone-entry anchoring (the current window-start
  trick) is deliberately *not* carried over: it would require entry-time state, and
  re-entering a zone would re-anchor it.
- Hard edges (no falloff, no crossfade). Stateless, trivially testable, and the sharp
  position↔degradation correlation is what makes the phase-2 lesson legible.
- **Agent position is authoritative**, not the sensor pose. They differ only by
  `camera.position` `[0, 1.5, 0]`, which is pure `y` — XZ is identical either way, but
  the code should pick one and say so.

**`Engine.observe(t)`** keeps its signature; it looks up
`self.agent.get_state().position` internally and passes XZ to the scheduler.

**Severity ladder.** Three zones spanning benign → severe, plus the clean region
outside all of them:

| Zone | amplitude | gain range | intended outcome |
|---|---|---|---|
| benign | 0.20 | 0.80 – 1.20 | never clips; ICP essentially unaffected |
| moderate | 0.60 | 0.40 – 1.60 | intermittent clipping on both rails |
| severe | 0.95 | 0.05 – 1.95 | heavy clipping on both rails |

The point of the ladder is a **within-capture dose-response curve** instead of a
binary, and a benign zone that teaches *effect present ≠ quality bad*.

**Calibrate the benign zone against the depth coupling (M3).** "Benign" is a claim
about ICP, and the coupling may falsify it: `stress = |gain − 1|`, so amplitude 0.20
already yields `noise_std ×1.8`, `dropout +0.06`, `max_range ×0.92`. Severe yields
`noise_std ×4.8`, `dropout +0.285`, `max_range ×0.62`. Verify empirically that the
benign zone leaves the F-score intact; if it does not, lower its amplitude until it
does, or the lesson it exists to teach does not land.

**Zone membership is not a per-frame degradation label.** With `phase: 0.0` on the
global clock, an in-zone frame whose `sin(2πft)` happens to vanish has gain exactly 1.0
— and since `stress = |gain − 1|` is then 0, its depth is unstressed too. Such a frame
is **bit-identical to baseline despite being inside a zone**. This is a consequence of
dropping zone-entry phase anchoring, and it is physically sensible: a flickering light
is at nominal brightness twice a cycle. Two implications:

- `zone_frame_counts` reports *geometry*, not damage. Frames-per-zone ≥
  frames-degraded, always. Do not use the former as a proxy for the latter when
  validating a zone placement (§3.2 item 4).
- Choose frequencies so zero-crossings do not land systematically on captured frames.
  The e2e suite pins `f = 0.75 Hz` for exactly this reason.

**Keep frequencies ≲3 Hz.** At 30 fps, Nyquist is 15 Hz — above that, frames alias and
flicker reads as random gain jitter. Interactive capture samples irregularly (frames
are written on keypress, `load.py:155`), so aliasing behaves differently there than in
replay. Staying low avoids the whole class of problem.

**Zone placement procedure**, per floor — the one hard constraint is that zones must
intersect the committed trajectory, or nothing fires in replay and the mixed run
silently degenerates to baseline (the same failure the `gap_s [8,25]` comment in
`second_floor.yaml` already records):

1. Load the trajectory, project poses to XZ.
2. Pick three well-separated centers along it (e.g. at arclength quartiles); size each
   radius so the zone covers a contiguous run of roughly 40–80 frames.
3. Replay once and **count frames per zone** (`zone_frame_counts`). Assert every zone is
   entered, no zone swallows the episode, and a meaningful number of frames fall outside
   all zones — those clean frames are the within-capture control the whole analysis
   rests on.
4. **Ship a moving e2e fixture.** The committed
   `packages/simulator/tests/fixtures/mini_secondfloor.npy` is 10 poses at a single XZ
   `(0.0, 0.0)` — a pure in-place rotation — so it cannot produce a within-run mixture
   of in-zone and out-of-zone frames at all. The e2e suite currently works around this
   with two separate runs (zone 50 m away vs zone on the fixture) and asserts the
   fixture's stationarity, so the workaround self-destructs the moment a moving fixture
   lands. Cut one from the new trajectory during M2 and delete the workaround.

**What this does *not* fix.** It does not make interactive capture reproducible.
`Engine.observe` keys the depth RNG on `default_rng([seed, round(t*1000)])` and the
sine reads `t`, both wall-clock in interactive mode. Zone *membership* becomes
position-determined, but phase and depth noise still do not repeat. §6's
"re-render via `--trajectory`" requirement stands unchanged and is still load-bearing.

**Coherence with the RGB factors.** Flicker-only means the sole photometric knob is
exposure gain — exactly what `clip_hi` and `clip_lo` measure. It also retroactively
justifies dropping entropy: entropy earned its place by catching the gamma/contrast
compression that never reaches a rail, and with `low_light` / `over_exposure` gone
there is none. And because `stress = |gain − 1|`, flicker damages depth on **both**
swings, so each clip factor should predict ICP damage on its own side.

**Deletions and test rewrites.** `UncertaintyScheduler`, its `save()`, and every
`windows.json` path come out — which also makes D6 self-enforcing rather than a
packaging check (§3.2 item 5 keeps the check anyway; belt and braces). In
`packages/simulator/tests/`: the scheduler-determinism unit test becomes a purity test
(same `(position, t)` → same override, no hidden state); the golden-array pixel tests
are unaffected; the e2e cases "baseline invariance outside windows" and "effects firing
inside windows" become "outside zones" / "inside zones" and gain the frames-per-zone
assertion from step 3.

### 3.2 Floor-1 asset regeneration

Floor 1 was dropped in the temporal-uncertainty refactor. Nothing usable survives:
`trajectories/firstfloor.npy` was never committed, `configs/baseline.firstfloor.yaml`
was deleted at `4c4beae`, and `data_collection/first_floor/{rgb,depth,semantic}` are
empty directories holding only a stale `GT_pose.npy`.

Work items, in order:

1. **Restore the trajectory search tool.** `scripts/search_traj.py` exists at HEAD but
   is deleted in the worktree — restore it (`git checkout HEAD -- scripts/search_traj.py`).
2. **Author `instructor/configs/first_floor.yaml`.** Base it on
   `git show HEAD~1:configs/baseline.firstfloor.yaml`, then port the current schema:
   the **spatial `uncertainties` block (§3.1)** and the `depth.light_*_gain` coupling
   from `hw1/configs/second_floor.yaml`. Floor-1 navmesh height is ≈ −1.57 (vs ≈ 1.43
   for floor 2) — `agent.start_position` must snap onto it; verify with
   `pathfinder.snap_point` before trusting it. Zone centres are filled in at item 4,
   after the trajectory exists.
3. **Generate `instructor/trajectories/firstfloor.npy`** via `search_traj.py`. Target
   comparable length to `secondfloor.npy` (454 poses ≈ 15 s at 30 fps) and enough
   floor coverage that `completeness.py` F-scores are meaningful.
4. **Place the three zones on the trajectory** by the §3.1 procedure, and run the
   frames-per-zone assertion. This replaces the old schedule-tuning step, and the
   failure it guards against is the same one: under the temporal scheduler,
   `gap_s: [8, 25]` put the first window past the episode end and the mixed run
   degenerated to baseline. A zone that the trajectory misses fails identically, just
   more visibly.
5. **Render the frozen phase-1 dataset** with `scripts/evaluate.py`'s two-run flow:
   - `baseline/` — `uncertainties.enabled: false`. Required:
     `completeness.build_gt_reference` refuses a `mixed/` dir, and students need a clean
     reference to attribute degradation against.
   - `mixed/` — zones on.
   - **Directory names are `baseline/` and `mixed/` on both floors** — no `mixed_dev`.
     That matches `scripts/evaluate.py`'s hardcoded `CONDITIONS = ("baseline", "mixed")`
     and sidesteps the question of whether `build_gt_reference`'s "refuses a `mixed/`
     dir" guard substring-matches a longer name. The two floors therefore have
     identically-named capture dirs; the collision is resolved in the **triplestore**,
     not the filesystem — batch names are floor-qualified (`floor1_mixed`,
     `floor2_mixed`), since `build_batch_graph` would otherwise map both onto one named
     graph and the phase-2 transfer analysis is exactly the workflow that hits it.
   - Ship exactly `rgb/`, `depth/`, `GT_pose.npy`, `intrinsics.json` per capture
     (D5/D6). The packaging check must be an **allowlist, not a `windows.json`
     blocklist**: `windows.json` is gone after §3.1, but `evaluate.py` now emits
     `per_zone.csv`, which is the same hazard reborn — a per-zone frame breakdown is
     precisely the effect-label artifact D6 forbids. An allowlist catches it, and every
     future variant of it, for free.
   - If O1(a) is chosen, render `mixed_holdout/` with **displaced zone centres** and
     keep it instructor-side. Note the change from the old plan: the scheduler is
     seedless now, so a "different seed" no longer produces a different realization —
     the holdout must vary the zone geometry instead.
6. **Re-derive instructor reference bands** for all four quality factors (§4.2) on floor 1 with
   the reference solution, at the instructor's own τ. The historical floor-1 luma band
   ([146.35, 230.87]) carries over **nothing** — the factor set changed and luma is gone
   (D12), so all four bands are derived fresh. These bands are a **feasibility check on
   the assignment, not an answer key** (D13): confirm each factor separates on this
   scene at a sane τ, and that the clip pair beats `mean(V)` at least as convincingly as
   students will be asked to show. If a factor comes out degenerate, that is a *finding
   to teach*, not a bug to hide: the student docs must say plainly that "this metric
   does not separate on this scene" is a valid, creditable conclusion. Otherwise
   students grind against a factor that cannot work.
7. **Delete the stale `data_collection/first_floor/` tree** or repoint it at the new capture.
8. **Publish the dataset as a downloadable archive**, not as repo content — the
   `pixi run fetch-replica` pattern extends naturally to a `fetch-hw1-data` task.

### 3.3 Floor-2 config rewrite

`hw1/configs/second_floor.yaml` is the one config students *do* see (D5), so it is
also the document that teaches the mechanism. Changes:

- Replace the `uncertainties` block with the §3.1 spatial schema. `gap_s`,
  `duration_s` and the whole `types:` map — `flicker`, `low_light`, `over_exposure` —
  are deleted. **This is currently the one thing standing between M1.5 and a working
  degraded run:** stale temporal keys are ignored rather than rejected, so the config as
  it stands resolves to `zones: []` and the `mixed` run degenerates to baseline. Both
  `load.py` and `evaluate.py` print a loud warning in that state, so it fails visibly
  rather than silently — but it does not work until this lands.
- Place three zones on `trajectories/secondfloor.npy` (454 poses, ≈15 s at 30 fps) by
  the §3.1 procedure. The existing comment block explaining the old seed/gap tuning
  should be **replaced, not amended** — it documents a mechanism that no longer exists
  and would actively mislead.
- `lighting.brightness / contrast / gamma / ambient_rgb` stay as the baseline
  photometric model. With `low_light` and `over_exposure` gone, nothing overrides
  `contrast` or `gamma` any more; leave them at 1.0 and say so, rather than deleting
  keys `apply_lighting` still reads.
- The `depth.light_*_gain` coupling block is unchanged and stays visible — it is the
  causal chain phase 2 is meant to reveal (D5).
- `seed` stays, with a corrected comment: it governs the per-frame depth RNG only, no
  longer any scheduling.

---

## 4. Boilerplate specification

Convention for every blanked function: keep the signature, keep a docstring that
states the **contract** (inputs, outputs, units, invariants) without stating the
algorithm, and end the body with `#TODO`. Nothing that ships blank may be
load-bearing for a function that ships working — a student with an empty `utils.py`
must still be able to run `api.py`, and vice versa, so the two halves of the
assignment do not block each other.

### 4.1 `hw1/utils.py`

| Function | Status | Note |
|---|---|---|
| `load_depth_meters` | ships working | I/O detail, not a learning objective |
| `depth_image_to_point_cloud` | **`#TODO`** | Core geometry. Signature is `(rgb, depth_m, width, height, hfov)` — intrinsics are **ordinary arguments**, read from the capture's `intrinsics.json` (D5), never from a config and never module constants. The 2-arg form left module-level constants as the only channel, i.e. hardcoding. |
| `preprocess_point_cloud` | ships working | voxel downsample + normal estimation + FPFH |
| `global_registration` | ships working | RANSAC/FPFH baseline |
| `local_icp_algorithm` | ships working | **Open3D wrapper — the reference baseline** students compare their own ICP against (D10) |
| `multiscale_icp` | ships working | coarse-to-fine driver over the Open3D wrapper |
| `reconstruct` | **`#TODO`** | Frame-to-frame loop, pose accumulation, outlier gating, `--frames-csv` subsetting |
| `mean_l2` | ships working | **It is the score.** Must be byte-identical across submissions; students implementing it invites gaming and grading disputes |
| `_load_gt`, `_sorted_frames`, `_rot_angle_deg`, `make_trajectory`, `remove_ceiling` | ship working | plumbing and visualization |

`reconstruct` is the highest-risk `#TODO` — it is where a broken ICP silently becomes
a broken analysis. Ship a smoke fixture (a handful of frames with a known-good
expected pose delta) so students can tell "my ICP is wrong" from "my loop is wrong"
before they reach the data-quality work. **Three** failure modes, not two: a wrong
unprojection also just "gives a bad trajectory", so the fixture ships per-frame cloud
statistics *and* the reference clouds themselves (`clouds.npz`), letting
`depth_image_to_point_cloud` be checked point-for-point and standalone before anything
downstream runs.

The fixture *generator* lives instructor-side — it composes poses as `poses[-1] @ T_rel`,
which is precisely the composition convention `reconstruct` has to get right. Students
consume the generated capture; they do not regenerate it. `clouds.npz` is what makes
that separation possible: the student-tree test would otherwise have to rebuild clouds
through the very `#TODO` it exists to validate.

**The grader must not depend on carved-out code** (the §4.6 constraint, enforced here).
`completeness.py:build_gt_reference` built the GT reference map out of student code —
not just `depth_image_to_point_cloud` but also `load_depth_meters` and `_sorted_frames`,
all three student-editable and all three on the GT path. A student can move their own
reference map by editing `DEPTH_SCALE` or the frame ordering just as surely as by
editing the unprojection. The rule is therefore the strong one: **`completeness.py`
imports nothing from `utils.py`.** It carries frozen private copies under a
DO-NOT-DE-DUPLICATE banner, and the repeated pinhole math is the cheap side of that
trade. Intrinsics come from the capture's own `intrinsics.json` with **no fallback** — a
missing file raises rather than defaults, because a silently wrong camera yields a
silently wrong GT map.

This defect predates the carve-out and was worse than latent: `completeness.py` used
`import utils as U`, so the module did not import at all except from inside `hw1/`.

**If D10 is ever applied to `my_local_icp_algorithm`, the docstring must go with the
body.** It currently carries a full `SPEC:` block spelling out the Kabsch/Umeyama
recipe *and* the implementation. Blanking only the body would leave the answer sitting
in the docstring. Same for the module docstring's `PIPELINE` section, which states the
per-pair algorithm for the now-blank `reconstruct`.

### 4.2 `hw1/api.py`

Remove `THRESHOLDS` entirely. Replace with `hw1/thresholds.json` **that students
author**, loaded at runtime; ship the loader working, ship the file absent. The loader
must fail loudly and legibly on a missing file or missing key — a silent default is
exactly what D2 forbids.

**Schema, pinned** (the loader docstring is the authoritative statement of it):

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

Key convention: `<observable_name>_<min|max>`, so each key names the exact observable it
bounds. `api.py`'s CLI override flags are a **separate, shorter namespace**
(`--valid-depth-min` and friends) — that is fine, provided no document claims the flag
spellings are the file's keys. They are not.

`tau_*` are read at `insert` time, `bands` at `retrieve` time (D13). **The file is flat
— not keyed per floor.** Per-floor bands would let a student quietly sidestep the
phase-2 deliverable, which is precisely "apply the floor-1 bands to floor-2 data and
report whether they transferred".

`--floor` survives with narrowed meaning: it no longer selects a band set, and does
exactly two things — it is written to `hw1:floor` on the Batch node, and it prefixes
the batch name (`floor{N}_{dirname}`) so the two floors' identically-named captures do
not collide on one named graph.

| Element | Status | Note |
|---|---|---|
| `NS`, `HW1`, `SCHEMA`, IRI helpers, `_stem`, `_pair_frames` | ship working | plumbing |
| `frame_mean_value` | **ships working — the naive baseline, deliberately weak** | `mean(max(R,G,B))` over the frame, in [0,255]. Serves two purposes only: (i) a fully implemented measurer showing the expected shape (pure, numpy + Pillow, documented units and range), (ii) the baseline the student's two RGB factors must **outperform**. Convention-free by construction, so the comparison isolates one variable — first moment vs tail statistic — and cannot be won by accident of colorimetry. Its docstring states outright that it is a poor quality metric and why (§4.2.1). It is **not** itself a quality factor. `frame_luma` is **deleted** (D12) |
| `frame_valid_fraction` | **`#TODO`** | Factor 1 (depth-validity). Contract: fraction in [0,1] of depth pixels non-zero and inside `[min_range, max_range]` m; depth PNG is uint16 mm |
| `frame_depth_roughness` | **`#TODO` — new** | Factor 2. Contract: a **reference-free estimate of the depth raster's noise level**, in metres, from the same uint16-mm PNG. **Lower is better**, so the band is one-sided `≤ max`. Pure numpy + Pillow — no cloud, no normals (D17). Cite Immerkær 1996 in the docstring. See §4.2.1 |
| `frame_clip_hi_fraction` | **`#TODO` — new** | RGB factor 1. Contract: fraction in [0,1] of pixels with `max(R,G,B) ≥ τ_hi`. One-sided band, `≤ max`. τ_hi is student-derived (D13), read from `thresholds.json`, and passed in — never hardcoded. See §4.2.1 |
| `frame_clip_lo_fraction` | **`#TODO` — new** | RGB factor 2. Contract: fraction in [0,1] of pixels with `max(R,G,B) ≤ τ_lo`. One-sided band, `≤ max`. τ_lo student-derived, same handling. See §4.2.1 |
| `build_batch_graph` | **`#TODO`** | Must emit all **five** observables onto the ontology's node structure — `valid_depth_fraction`, `depth_roughness`, `clip_hi_fraction`, `clip_lo_fraction`, `mean_value` — plus the Batch-level provenance of §4.3 |
| `resolve_band` | **`#TODO`** | Four one-sided bounds across the four factors, sourced from `thresholds.json` |
| `build_select` | **`#TODO`** | Template substitution into `valid_frames.rq` |
| `put_named_graph`, `load_query`, `run_select`, `cmd_insert`, `cmd_retrieve`, `cmd_compare`, CLI parser | ship working | Fuseki transport and CLI wiring are not the learning objective. `compare` is the third subcommand — see §4.4 |

### 4.2.0 Depth factor 2 — `DepthRoughness`

Observable `depth_roughness`, in **metres**, band one-sided **`≤ max`** (lower is
better). Computed on the raw uint16-mm depth raster, `D = raw / 1000.0`:

```
M = [[1, -2,  1],
     [-2, 4, -2],
     [1, -2,  1]]                       # Immerkær 1996, ‖M‖ = 6

R = D ∗ M                               # evaluated ONLY where the full 3×3
                                        # neighbourhood is valid
depth_roughness = median(|R|) / (6 · 0.6745)
```

**Grounding.** Immerkær, *Fast Noise Variance Estimation*, CVGIP 1996 — the standard
reference-free noise estimator, and the same kernel Shin et al. IROS 2019 use for their
noise term. The `√(π/2)/6 · mean|R|` form is the literature original; the **median**
variant above is used instead because real depth discontinuities produce large
Laplacian responses that would dominate a mean. `0.6745` is the median of `|N(0,1)|`,
so the constant converts a robust spread back to σ.

**Two details that are not optional.** Evaluate `R` only where **all nine** pixels of
the window are valid — dropout zeros next to real depth otherwise read as enormous
edges, and the factor would end up measuring factor 1. And work in metres, so the band
is a physical noise level a student can sanity-check against the sensor model.

**Why this factor, and why it is the natural fit here. — INSTRUCTOR-ONLY RATIONALE, do
not carry into student docs.** It targets the *dominant* injected depth degradation:
`light_noise_gain: 4.0` is the largest coupling gain in the config, so
`stress = |gain − 1|` drives noise harder than dropout or range loss. That sentence
names a gain value, an effect identity and a ranking of injected effects — three
separate leak-rule violations. The student-facing motivation must be mechanism-free:
factor 1 counts how many depth returns you got, factor 2 says how trustworthy the
returned *values* are, and `valid_depth_fraction = 1.0` is perfectly compatible with
unusable numbers.

It also closes a loop from the original design. The instructor `definitions.md` dropped
the depth-noise axis explicitly — *"`realized_sigma_z_m` needs a paired clean-depth
reference of the same trajectory, which no arbitrary query sample has — un-computable
at inference, so not API-serviceable."* A **reference-free** estimator removes exactly
that objection: σ is recoverable from one frame with no clean twin. The axis that had to
be abandoned becomes API-serviceable, and it is the one most directly coupled to the
light stress the assignment is about.

**Independence from factor 1.** Roughness is computed only over fully-valid windows, so
losing valid pixels makes the estimate *noisier* rather than shifting it wholesale —
far weaker coupling than the replaced conditioning factor, where `λ_min` of a summed `C`
scaled with point count directly. State it no more strongly than that: under heavy
dropout the surviving windows are **not a random subsample** — they sit away from
dropout regions — so a selection effect remains. The §6 requirement stands unchanged:
check empirically whether the two depth factors carry independent information. "My second
factor told me nothing my first had not" remains a full-marks finding.

**Degenerate frame.** With no fully-valid 3×3 window anywhere (an all-zero depth frame),
`frame_depth_roughness` returns **`inf`**. Not `0.0` — zero reads as "perfectly smooth"
and would *pass* a `≤ max` band, letting a frame with no usable depth sail through the
gate. `inf` fails any finite band, is representable as `xsd:double` (`INF`), and SPARQL
evaluates `?rough <= x` false against it, so the frame drops out of the PASS set. Do not
justify this as "factor 1 will catch it anyway" — that reintroduces exactly the coupling
this factor set is built to avoid.

**The general rule, which applies to any factor added later:** when a metric cannot be
computed, return the value on the *failing* side of its band. A degenerate value that
reads as "excellent" is worse than one that reads as "broken", because only the second
is visible.

**What was replaced, and why it is recorded here.** Factor 2 was
`frame_geometric_conditioning` — `λ_min` of the averaged 6×6 point-to-plane normal
matrix. It was correct and well-grounded (Gelfand 2003, Zhang 2016) but carried
unprojection, voxel downsampling, normal estimation, a 6×6 eigendecomposition, three
contract constants, and five invariants students had to respect, all sitting beside a
from-scratch ICP. D17 rules it out on principle as well: it measured a *derived point
cloud*, not the input file. Two consequences worth keeping in view — the degeneracy
lesson ("a frame can be clean and still fail to constrain the pose") leaves the factor
set with it, and if it is ever wanted back, the raster-only form is an inverse-depth
plane residual (`1/D = au + bv + c` is affine across any plane under a pinhole camera),
not the eigen-decomposition.

**Undeclared dependency: `scipy`.** Three modules import it — `hw1/utils.py`,
`hw1/completeness.py` and `packages/simulator/simulator/replay.py` — and `pixi.toml`
declares it nowhere. It resolves today only as a transitive of something else, which
makes the pipeline hostage to an unrelated package's dependency list. Pin it explicitly
in `[feature.habitat.dependencies]`. Deliberately not done in-flight: adding it
re-solves `pixi.lock`, a build-system change that belongs in its own reviewable step.
(`api.py` dropped off this list with D17 — roughness is pure numpy, so the module is
back to numpy + Pillow + rdflib as its docstring always claimed.)

### 4.2.1 The two RGB quality factors

Two factors, both single-frame, both pure numpy + Pillow, both one-sided `≤ max`
bands, both computed on the **value channel**

```
V = max(R, G, B)            # HSV Value. No colorimetric weighting anywhere (D12).
```

**RGB-1 — `HighlightClipping`.** Observable `clip_hi_fraction ∈ [0,1]`:

```
clip_hi_fraction = |{ V ≥ τ_hi }| / N
```

**RGB-2 — `ShadowClipping`.** Observable `clip_lo_fraction ∈ [0,1]`:

```
clip_lo_fraction = |{ V ≤ τ_lo }| / N
```

Grounded in the unsaturated-region mask of Shin et al., IROS 2019 (their eq. 7 —
`U(i) = 1` iff `τ_l ≤ I(i) ≤ τ_h`), which masks out exactly the pixels whose values
carry no recoverable information. Split into two observables rather than one union,
because the gate does not need the direction but the **diagnosis** does: a single
number cannot tell a crushed frame from a blown-out one.

**The two definitions use the same operator and mean different quantifiers.** This is
correct, deliberate, and must be stated in both docstrings or someone will "fix" it:

```
V ≥ τ_hi   ⟺   at least one channel is saturated      (∃)
V ≤ τ_lo   ⟺   every channel is crushed               (∀)
```

The apparently symmetric alternative — `min(R,G,B) ≤ τ_lo` for the shadow side — is
**wrong**: it fires on any saturated colour, since pure red has `G = B = 0`. Put that
counter-example in the code comment.

**Why `max(R,G,B)` and not luma (D12).** Two independent reasons, and the second is
the stronger one:

1. **Luma is a convention this renderer never promised.** Rec.601 weights
   (0.299/0.587/0.114) encode an SDTV standard; Rec.709 (0.2126/0.7152/0.0722) encodes
   another. Picking either bakes an unjustified assumption into a metric students are
   told is grounded. `max` assumes nothing.
2. **Weighting hides single-channel saturation.** Pure red `(255, 0, 0)` has Rec.601
   luma 76 — it reads as an ordinary mid-tone while the red channel is fully railed.
   Clipping is a per-channel phenomenon, so a per-channel test is the correct one.

**Why the mean of anything is excluded.** `mean(V)` ships only as the baseline to beat
(§4.2). It is a first moment, and a first moment is the wrong summary here:

- **Not two-sided.** Under- and over-exposure are separate failures; one scalar over
  one interval cannot hold both, so the band widens until it admits both.
- **Confounded with scene content.** Pointing at a bright window moves the mean as
  much as the injected effect does. Under D6 — derive against ICP degradation — that
  confound is pure noise in the target.
- **Blind to distribution shape.** A frame half crushed and half blown-out has a
  perfectly ordinary mean; the clip pair reports `clip_lo ≈ clip_hi ≈ 0.5`.
- **It already failed empirically.** The floor-2 `GoodBrightnessRange` came out
  degenerate — `[18.04, 252.84]`, the full sweep range, gating nothing.
  **Instructor-side only.** The number, the floor and the finding are all derived
  results that D2 and the §2 leak audit forbid shipping. The student doc keeps the
  reason but states it as a failure mode to *test for on every factor, including their
  own* — never as a result we already have.

The same trap is documented in the literature for the *gradient* family: Zhang et al.,
ICRA 2017 show that the plain and log-mapped gradient sums (`M_sum`, `M_shim`) are
dragged upward by bright regions and select over-exposed images, while percentile-based
metrics picked the frame with the most FAST features in **13 of 18** datasets. Mean and
sum statistics fail the same way; tail and order statistics do not. That is the
transferable lesson, and `mean(V)` shipping as the beatable baseline is how students
meet it.

**τ is student-derived (D13).** τ_lo and τ_hi are *measurement* parameters, not bands,
and they are derived and justified like everything else. Three consequences the
implementation must honour:

- **τ is read at `insert` time, bands at `retrieve` time.** Both live in
  `thresholds.json`; `api.py` reads it in both subcommands. Change τ and the stored
  triples are stale — students must re-`insert`. Named-graph replace makes that safe
  and idempotent, and it is worth saying so in the README, because a silently stale
  batch is the most likely way a student's numbers stop making sense.
- **τ must be recorded on the `Batch` node** (§4.3). Otherwise the store holds
  fractions whose meaning is unrecoverable from the store itself. This is a genuine
  provenance lesson that D13 hands us for free — it is the first place in this
  assignment where the ontology has to describe *how* a number was produced, not just
  what it was.
- **Observables are no longer comparable across students**, so nothing may be graded
  on their values. §6 grades the τ-agnostic downstream reconstruction outcome instead.
  Measurer *correctness* stays autogradable: the §4.5 fixtures pass τ in explicitly,
  and a synthetic frame has a closed-form clip fraction for any τ.

**What is deliberately not in the student docs (D11).** Sharpness metrics (variance of
the Laplacian — Pech-Pacheco, ICPR 2000; Tenengrad — Krotkov 1986) and noise estimators
(Immerkær 1996, as used by Shin et al.) are omitted entirely. This simulator injects
neither RGB blur nor RGB noise. Scaling a frame by exposure gain `g` scales its
Laplacian by `g` and the variance of that Laplacian by `g²` — so here a blur metric
reports contrast while claiming to report focus. Listing them would spend student
budget on metrics that cannot do what their name says. **Instructor-side note:** if the
uncertainty model ever gains motion blur or an RGB noise channel, these become live and
belong back in the menu.

### 4.3 `hw1/ontology/hw1.ttl`

Ships: the TBox as it stands (`Batch`, `Frame`, `RGBImage`, `DepthImage`,
`QualityFactor`, object properties, existing datatype properties).

**Removed:** `avgLuma`, and with it the `GoodBrightnessRange` QualityFactor individual.
`frame_luma` is deleted from the student tree (D12), so the property has no producer —
and `GoodBrightnessRange`, whose sole observable was `avgLuma`, would otherwise survive
by omission despite being absent from D7's exhaustive four-factor set.

On the component nodes — **declarations ship, values are `#TODO`** (same convention as
the τ paragraph below, which spells it out): `depthRoughness` on `DepthImage`;
`clipHiFraction`, `clipLoFraction` and `meanValue` on `RGBImage`, the last being the
baseline, carried so the §6 comparison can be made by query rather than by hand. Only
the `QualityFactor` individuals are genuinely blanked.

`#TODO`, on the `Batch` node — **measurement provenance (D13)**: `tauHi` and `tauLo`,
the thresholds the batch's clip fractions were computed under. Without them the store
holds fractions with no recoverable meaning, and two batches measured at different τ
look directly comparable when they are not. Ship the TBox declaration; students wire
the values.

**`normalRadius` and `voxelSize` are removed** along with the conditioning factor that
needed them (D17). `depth_roughness` has no free parameters: the Immerkær kernel is
fixed, and its validity rule is the one `valid_depth_fraction` already uses. Batch
provenance is therefore back to τ alone.

One loose end this exposes, pre-existing and not introduced here, and worse than it
first looked: `min_range` / `max_range` parameterise the validity mask that **both**
depth factors depend on, and they are recorded nowhere. The same D13 argument applies.
Note there are in fact **two unrelated ranges** in play, which do not match:

- `api.py`'s, as per-call keyword defaults on the measurers (`min_range=0.0`,
  `max_range=10.0`) — overridable silently by any caller;
- the *simulator*'s `depth.min_range` / `depth.max_range`, applied at capture time to
  zero out-of-range pixels before the PNG is ever written.

Today the api-side band is wider than the capture-side one, so it is effectively inert
— but that is a coincidence of the numbers, not a guarantee. If a future config's
capture range exceeds the measurement default, frames get double-filtered and the
stored fractions quietly stop meaning what the ontology says they mean. The capture-side
range additionally *shrinks under light stress* (`light_range_gain`), which is intended
— `valid_depth_fraction` is supposed to register that as lost pixels.

`#TODO`, individuals: the four `QualityFactor`s — `ValidDepthRatio`,
`DepthRoughness`, `HighlightClipping`, `ShadowClipping`. The existing comment —
*"Bands intentionally have no pre-defined min/max here; supply them at query time"* —
stays; it is exactly right under D2.

Band bookkeeping: four factors, all four bands **one-sided** — `valid_depth ≥`,
`roughness ≤`, `clip_hi ≤`, `clip_lo ≤` — so `resolve_band` carries four bounds and
`valid_frames.rq` carries four `FILTER` terms, no wider than the two-sided brightness
band it replaces. Note the direction flip from the replaced factor: conditioning was
higher-is-better (`≥ @@GMIN@@`), roughness is lower-is-better (`≤ @@RMAX@@`). Exactly
one factor now points up — `valid_depth_fraction` — and the other three point down.

### 4.4 `hw1/queries/`

- `valid_frames.rq` — ships as a skeleton: the `PREFIX` block, the `SELECT` header,
  and the substitution-token contract documented in comments. The `GRAPH ?g { ... }`
  pattern body and the `FILTER` are `#TODO`. The existing header comment explaining
  why the pattern must be inside `GRAPH ?g` is valuable and stays — it is a genuine
  Fuseki gotcha, not an answer.
- `compare_batches.rq` — **new, `#TODO`** (the D8 mitigation): one query joining two
  named graphs on `frameIndex`, returning per-frame observable deltas. Cheap to write,
  and it is the query that answers "why a triplestore instead of a dataframe".
  - **Batch names are tokens (`@@BATCH_A@@`, `@@BATCH_B@@`), never hardcoded.**
    `batch_iri` derives the name from the capture directory basename, and the shipped
    batch names are floor-qualified (`floor1_mixed`, `floor2_mixed`) — a literal
    `"mixed"` would match nothing.
  - **It needs a `compare` subcommand to be runnable at all.** Without one nothing
    shipped can execute the query, and the D8 mitigation is decorative. `api.py` gains
    a third subcommand alongside `insert` and `retrieve`, shipping **working** (CLI and
    transport are not the learning objective): load the template, substitute, run the
    SELECT, print rows. The `.rq` body stays `#TODO`.

### 4.5 `hw1/test_e2e.py`

Ships **correct, not passing** — an important distinction. It is the student-facing API
contract, the specification of what `insert`/`retrieve`/`compare` must do, and passing
it is an autograded deliverable. Against the blanked tree it therefore fails 30 of 37
cases *by construction*; that is the signal, not a defect. The module under test is
selectable (`HW1_API=…`), which is how an instructor runs the same contract green
against `instructor/solution/api.py`.

Required rewrite: the current fixture grades against `THRESHOLDS[1]` defaults, which
D2 removes. Change every case to pass explicit band arguments, and extend the
synthetic fixture so every factor has a closed-form expected value:

Every fixture passes τ **explicitly** (D13 makes τ student-derived, so no default may
be assumed) — a synthetic frame has a closed-form clip fraction for any τ, which is
what keeps the measurers autogradable when the observables are not comparable.

| Fixture | Expected |
|---|---|
| Solid mid-grey `(128,128,128)` | `clip_hi = clip_lo = 0` for any `5 < τ_lo < 128 < τ_hi < 250`; `mean_value = 128` |
| Half-black `(0,0,0)` / half-white `(255,255,255)` | `clip_hi = clip_lo = 0.5` — **and `mean_value = 127.5`, which reads as perfectly exposed.** This single fixture is the executable proof of why the mean is excluded; keep it and name it accordingly |
| Solid pure red `(255,0,0)` | `clip_hi = 1.0` — **and Rec.601 luma would be 76, an ordinary mid-tone.** The executable proof of why luma is excluded (D12); worth keeping even though no luma measurer ships |
| All-black / all-white frames | `clip_lo = 1.0` / `clip_hi = 1.0`; the degenerate ends |
| Constant depth | exact `valid_depth_fraction` |
| Constant depth + Gaussian noise at a **known σ**, fixed RNG seed | `depth_roughness ≈ σ` to tolerance — the estimator is checked by *recovering the noise it was given*, which is a far stronger fixture than any closed-form value |
| Constant depth, no noise | `depth_roughness = 0` |
| Constant depth with a hard step edge, no noise | still `≈ 0` — proves the fully-valid-window masking and the median (not mean) actually work; the naive mean form fails this one |

### 4.6 Ships fully working

`load.py`, `reconstruct.py`, `completeness.py`, `packages/simulator/*`,
`scripts/evaluate.py`. `completeness.py` in particular ships complete — it is the
grader, so students must not be able to influence it. That is a real constraint, not a
label: enforcing it meant cutting **every** `build_gt_reference` dependency on
`utils.py` — the unprojection, the depth loader and the frame lister — so the grader now
imports nothing from the student tree at all (§4.1). `reconstruct.py` and `evaluate.py` *do* call the
blanked `reconstruct`, and that is correct — they are drivers of the student's
pipeline, not graders of it.

Not *untouched*, though: `packages/simulator` is rewritten by §3.1 (`ZoneScheduler`
replaces `UncertaintyScheduler`, `Engine.observe` gains a position lookup, the
`windows.json` path is deleted, and the affected tests are rewritten). It ships
working — students never edit it — but it is a prerequisite for M2, not a no-op.

---

## 5. Student-facing documents to write

1. **`hw1/README.md` — rewrite.** Task statement, environment setup, the two phases,
   file map, how to run each stage, deliverables. No derived numbers.
2. **`hw1/definitions.md` — new, student version.** *Not* the instructor file of the
   same name (that moves to `instructor/`). Contains: the four quality factors as
   **measurement contracts** (formulas, units, range, invariants — §4.2.1), their
   literature citations, the two reasons luma is excluded (D12) and the four reasons
   any mean is excluded,
   the D11 scope statement (single-frame only; a per-frame metric fires on the extreme
   frames of a flicker window but cannot see the periodicity), and the D1 caveat —
   *geometric ICP never reads RGB; both RGB factors are upstream proxies and your
   report must say what they are proxies for*. Contains no bands.

   **Citations to carry through to the student docs**, so the factors read as
   engineering practice rather than invented conventions: Shin et al., *Camera Exposure
   Control for Robust Robot Vision with Noise-Aware Image Quality Assessment*, IROS
   2019 (the unsaturated-region mask, eq. 7 — the direct source for both clip factors);
   Zhang, Forster and Scaramuzza, *Active Exposure Control for Robust Visual Odometry
   in HDR Environments*, ICRA 2017 (the 13-of-18 FAST-feature result against sum-based
   metrics — the evidence that mean/sum statistics lose to tail statistics); and
   Immerkær, *Fast Noise Variance Estimation*, CVGIP: Graphical Models and Image
   Processing 1996, for `DepthRoughness`. (Gelfand et al. 3DIM 2003 and Zhang et al.
   ICRA 2016 left the citation list with the conditioning factor under D17.)
3. **`hw1/analysis.md` — the student's deliverable template.** Prompts, not answers:
   protocol design, sweep evidence, derived bands with justification, failure cases,
   what did not work. One section explicitly asks for a **falsifiable prediction**
   checked against measurement.

---

## 6. Deliverables and grading

Because the protocol is open (D9), grading weight moves onto experimental design.
Split: a machine-checkable core that is unambiguous, and a rubric that rewards
reasoning quality.

### Phase 1 — provided floor-1 dataset

| Deliverable | Check | Weight |
|---|---|---|
| ICP implementation | Autograded: `reconstruct.py` on the provided `baseline/` — mean L2 and `completeness.py` F-score against fixed pass bars | 25% |
| Ontology API | Autograded: `test_e2e.py` passes against a live Fuseki | 15% |
| Ontology TBox | **Read, not run.** Under the flat design (D8) nothing joins on the `QualityFactor` individuals, so `hw1.ttl` cannot be graded by execution — the individuals, and the `tauHi`/`tauLo` provenance wiring, are checked by inspection | included in *Ontology API* above |
| `thresholds.json` + the four measurers | Autograded: schema valid (bands **and** τ), measurers pure and deterministic and matching the §4.5 closed-form fixtures at the τ the fixture passes in, bands non-degenerate *or* accompanied by an argued degeneracy finding | 10% |
| Baseline comparison | Rubric: does either clip factor predict reconstruction degradation better than the shipped `mean(V)` baseline? A negative result argued from evidence scores full marks | included in *evidence and reasoning* below |
| `analysis.md` — protocol design | Rubric: is the experiment capable of answering the question? Controls, confounds, sweep coverage, compute cost acknowledged | 15% |
| `analysis.md` — evidence and reasoning | Rubric: bands supported by measured reconstruction outcomes; at least one falsifiable prediction stated and tested; the D1 proxy caveat correctly handled; **the two depth factors checked empirically for independence** before being treated as separate evidence (see §4.2 — averaging decouples them by construction but the scene can still correlate them, and "my second factor told me nothing new" is a full-marks finding) | 15% |

### Phase 2 — student-collected floor-2 dataset

| Deliverable | Check | Weight |
|---|---|---|
| Capture meeting a coverage floor | Autograded: F-score of the student's own reconstruction above a floor, so a two-metre capture cannot pass | 5% |
| Band transfer analysis | Rubric: floor-1 bands applied to floor-2 data; did they transfer? Evidence either way | 10% |
| Mechanism verification | Rubric: phase-2 config reveals the coupling (D5) — does the report reconcile the revealed mechanism with what they inferred in phase 1? | 5% |

**Nothing is graded on an observable's value (D13).** With τ student-derived, two
submissions' `clip_hi_fraction` numbers are not the same quantity, so no band bound and
no observable may be a grading target. What *is* comparable is the τ-agnostic downstream
outcome: the gate produces a frame set, the frame set produces a mean L2 and an F-score
on the same provided capture. Grade that, plus measurer correctness against the §4.5
fixtures, plus the reasoning. The instructor reference bands from §3.2 item 6 are a
**sanity check on the assignment, not an answer key** — they hold only at the
instructor's own τ.

**Cross-student comparability.** Phase-2 trajectories differ per student, so absolute
mean L2 is not comparable and is **not** graded — only the coverage floor and the
reasoning are. If O1(a) is adopted, the instructor holdout carries an autograded band
score and phase-2 transfer stays purely rubric-based.

**Reproducibility requirement — unchanged by D14, and still load-bearing.** Spatial
zones make effect *membership* position-determined, but interactive capture is still
not reproducible: `Engine.observe` keys the depth RNG on
`default_rng([seed, round(t*1000)])` and the flicker sine reads `t`, both wall-clock
while driving. Phase 2 must therefore require: collect interactively → keep
`GT_pose.npy` → re-render deterministically via `load.py --trajectory` (`t = i/fps`) →
run all analysis on the re-rendered capture. Without this, no phase-2 number is
reproducible, including by the grader.

---

## 7. Milestones

| ID | Milestone | Contents | Blocks |
|---|---|---|---|
| **M1** | Repository split | `instructor/` tree created; `definitions.md` / `empirical-analysis.md` moved there; leak audit of `hw1/README.md` done | everything publishable |
| **M1.5** | Simulator refactor | §3.1: `ZoneScheduler`, `Engine.observe` position lookup, `UncertaintyScheduler` + `windows.json` deleted, `packages/simulator/tests/` rewritten to zones. Blocks every capture, so it lands before any asset work | M2, M3 |
| **M2** | Floor-1 + floor-2 configs and assets | §3.2 items 1–5 and §3.3: both configs on the spatial schema, floor-1 trajectory, zone placement with the frames-per-zone assertion, `baseline/` + `mixed/` + `intrinsics.json` frozen | M3, M4 |
| **M3** | Reference solution + bands | Reference `utils.py` / `api.py` with all four measurers; instructor bands re-derived on floor 1; degeneracy check (§3.2 item 6) | M4 pass bars |
| **M4** | Boilerplate carve-out | §4 applied: `utils.py` and `api.py` blanked to spec, `hw1.ttl` and `queries/` skeletonized, `test_e2e.py` rewritten off `THRESHOLDS`, smoke fixture added | M5 |
| **M5** | Student docs | §5: `README.md` rewrite, student `definitions.md`, `analysis.md` template | M6 |
| **M6** | Autograder + packaging | Pass bars fixed from M3 numbers; `fetch-hw1-data` pixi task; ship-check that the archive holds only `rgb/`, `depth/`, `GT_pose.npy`; O1 resolved | release |
| **M7** | Dry run | Run the student path end to end from a clean checkout on a second machine, timing each stage. Budget check: from-scratch ICP + 4 measurers + ontology API + open-protocol sweep + 2 reports is a heavy load — if the dry run exceeds the intended budget, the first thing to cut is the from-scratch ICP (fall back to wiring the loop only, D10 option 3) | release |

---

## 8. Corrections to fold back into `IDEA.md`

- Dataset structure: drop `semantic/` (D3); add that no effect labels ship (D6).
- Phase 1 is floor 1 and phase 2 is floor 2 — consistent with `IDEA.md`, but note
  that floor-1 assets are being rebuilt, not reused.
- **The uncertainty regime is spatial, not temporal** (D14–D16). Uncertainty is a
  property of *place*: three hard-edged circular flicker zones per floor, on a
  benign/moderate/severe amplitude ladder, with clean navigable space outside them.
  `low_light` and `over_exposure` no longer exist, and neither do the seeded temporal
  windows `IDEA.md` predates. The reasoning task students face is therefore "which
  regions of this floor produce data my algorithm cannot use", which is closer to the
  deployment question `IDEA.md` set out to teach than a time-window story was.
- "Edge density = gradient of the depth image" is replaced by **`DepthRoughness`**, a
  reference-free Immerkær noise estimate on the same raster (§4.2.0). It keeps the
  raw-input spirit of the original idea while measuring an injected *corruption* rather
  than scene content — and it revives the depth-noise axis the instructor
  `definitions.md` had to abandon as needing a paired clean reference.
- The open question "what other RGB quality metrics?" is answered: **`HighlightClipping`
  and `ShadowClipping`**, both on `V = max(R,G,B)` (§4.2.1). Average luma is gone
  entirely — no colorimetric weighting anywhere (D12) — and `mean(V)` takes its place
  as the naive baseline students must beat. Blur, noise and temporal metrics are
  explicitly out of scope (D11). The standing caveat holds — under a geometry-only
  pipeline every RGB factor is a proxy.
- Students implement the ontology APIs **and** derive the thresholds; we provide only
  the definitions. `IDEA.md` currently implies we provide the thresholds too.
