# HW1 Analysis Report — *(your name, your student ID)*

> **This file is a template. It is your deliverable.**
>
> Every quoted block like this one is a **prompt**, not content. Replace it with
> your answer and delete the prompt. Keep the headings and their order — the
> rubric is organised around them, and a missing heading reads as a missing
> answer.
>
> Sections marked **REQUIRED** must be present and non-empty. A section you have
> nothing to say in still gets a sentence saying so and why; a silently deleted
> section scores zero for that row.
>
> **You will not find any thresholds in this repository.** Bands and the
> measurement parameters `tau_lo` / `tau_hi` are yours to derive. See
> [`definitions.md`](definitions.md) for what each factor measures and
> [How this is graded](#how-this-is-graded) below for what earns credit.

---

## Contents

| § | Section | Feeds rubric row |
|---|---|---|
| 0 | [Summary](#0-summary) | — |
| 1 | [Protocol design](#1-protocol-design--required) | protocol design (15%) |
| 2 | [Measurement setup: deriving tau](#2-measurement-setup-deriving-tau--required) | evidence and reasoning (15%) |
| 3 | [Sweep evidence](#3-sweep-evidence--required) | evidence and reasoning (15%) |
| 4 | [Derived bands and their justification](#4-derived-bands-and-their-justification--required) | evidence and reasoning (15%) |
| 5 | [Falsifiable prediction](#5-falsifiable-prediction--required) | evidence and reasoning (15%) |
| 6 | [Baseline comparison: your RGB factors vs. mean(V)](#6-baseline-comparison-your-rgb-factors-vs-meanv--required) | evidence and reasoning (15%) |
| 7 | [The proxy caveat](#7-the-proxy-caveat--required) | evidence and reasoning (15%) |
| 8 | [Failure cases](#8-failure-cases--required) | evidence and reasoning (15%) |
| 9 | [What did not work](#9-what-did-not-work--required) | protocol design + evidence |
| 10 | [Phase 2: your own capture](#10-phase-2-your-own-capture--required-for-phase-2) | transfer (10%) + mechanism (5%) |
| 11 | [Reproducibility appendix](#11-reproducibility-appendix--required) | protocol design (15%) |
| — | [How this is graded](#how-this-is-graded) | — |

---

## 0. Summary

> Half a page, written **last**. What question did you set out to answer, what
> did you conclude, and how confident are you? State your headline result in one
> sentence — including the case where the headline result is negative.
>
> If a reader reads only this section, what must they not get wrong?

---

## 1. Protocol design — **REQUIRED**

The experiment is **yours to design**. Nothing in this repository tells you how to
derive a band; that omission is deliberate, and the design is worth as much as
the result.

### 1.1 The question

> State the question your experiment answers, precisely enough that a specific
> measurement could come out either way. "Are the bands good?" is not a question.
> Name the **response variable** you are deriving *against* — what quantity counts
> as "the pipeline did badly" — and justify that choice.

### 1.2 Design

> - What do you vary, and over what range? Why that range and not a wider one?
> - What resolution / how many levels? Justify that this is enough to see the
>   effect you are looking for, and not so many that you cannot afford to run it.
> - What is your unit of analysis — a frame, a run, a subset of frames?
> - How many samples per condition, and what is the spread *within* a condition?

### 1.3 Controls

> - What is your comparison set — what does a "degraded" measurement get compared
>   *against*, and how did you establish that reference?
> - What would a **null result** look like in your design? If you cannot describe
>   one, your design cannot produce one, and it cannot be falsified.
> - Do you have a control that isolates the depth path from the RGB path? (See §7
>   — the two depth factors are computed from the data ICP actually consumes,
>   which makes them useful for exactly this.)

### 1.4 Confounds

> List the confounds you identified — things that move your observable for reasons
> your response variable does not care about, and vice versa. For each: did you
> control it, measure it, or accept it? Accepting a confound is a legitimate
> choice **if you say so and bound its effect**.
>
> At minimum, address: scene content (where the camera is pointing), the
> dependence between your four factors (do they carry independent information?
> measure it, do not assume it either way), and anything about the capture that
> varies along the trajectory.
>
> **The two depth factors get their own answer, and it is graded.** They read the
> same raster — `valid_depth_fraction` from its validity mask, `depth_roughness`
> from the values under that mask — and the definitions argue the coupling is weak
> by construction (`definitions.md` §1.2: roughness is estimated over fully-valid
> windows only, so losing valid pixels makes the estimate *noisier*, not biased).
> An argument is not a measurement. Show what the two observables actually did
> **on your data**, and say whether the second one told you anything the first had
> not. If it did not, say that — *"my second factor carried no information beyond
> my first"*, shown from the evidence, is a **full-marks** finding.

### 1.5 Cost

> What did the experiment cost — number of reconstructions, wall-clock, hardware?
> What did you cut to fit the budget, and what did that cost you in resolution or
> confidence? An honest budget statement is graded; a missing one is not forgiven.
>
> What would you do with 10x the compute? Answer specifically — the answer says
> what you think the limiting uncertainty is.

---

## 2. Measurement setup: deriving tau — **REQUIRED**

`tau_lo` and `tau_hi` are **measurement parameters**, not bands
([`definitions.md` §7](definitions.md#7-what-you-derive)). They decide what number
gets stored; the bands decide which stored numbers pass. Derive them, do not
guess them.

> - How did you choose `tau_lo` and `tau_hi`? Show the evidence, not the conclusion.
> - How sensitive are your clip fractions to tau? Show what happens to the
>   observable distribution as tau moves — this is a cheap sweep and it is the
>   basis for claiming your choice is not arbitrary.
> - Does your choice satisfy `tau_lo < tau_hi`, and what would break if it did not?
> - How sensitive are your **conclusions** to tau? If your headline result flips
>   when tau moves by a few DN, say so.
> - Confirm you re-ran `insert` after fixing your final tau, and say how you
>   verified the store is not stale. (Fractions stored under an old tau are
>   silently wrong: the query still runs and still returns frames.)
> - Confirm `hw1:tauHi` / `hw1:tauLo` are recorded on the `Batch` node, and say in
>   one sentence why a store without them holds uninterpretable numbers.

Record here every other **measurement parameter** you ran with. They are part of
the definition of your numbers, and nobody can reproduce you without them.

`tau_lo` and `tau_hi` are the only measurement parameters you **derive**.
`depth_roughness` has none at all — the kernel and both normalising constants are
fixed by the contract (§1.2 of `definitions.md`), so there is nothing to report
and nothing to tune. The depth range window is *declared*, not derived: state it,
hold it fixed across everything you compare, and record it here, because two
`valid_depth_fraction` values measured under different windows are not the same
quantity.

| Parameter | Value | Derived or given? |
|---|---|---|
| `tau_lo` | | derived (§2) |
| `tau_hi` | | derived (§2) |
| depth range window `[min_range, max_range]` | | declared — state it and hold it fixed |
| *(add rows)* | | |

---

## 3. Sweep evidence — **REQUIRED**

> This section holds the **measurements**, not the interpretation (§4 does that).
>
> - Present the sweep. A table, a plot, or both; label axes and units.
> - For each factor: what range of the observable did you actually cover? If you
>   never observed a factor outside a narrow range, you cannot claim a bound
>   outside it — say what you covered and what you did not.
> - Show the relationship between each observable and your response variable —
>   including the ones where there is no relationship.
> - Quantify the spread, not just the central tendency. How much of the variation
>   in the response variable does the factor explain, and how much is unexplained?

| Factor | Observable range covered | n frames / runs | Response variable | Relationship observed |
|---|---|---|---|---|
| `ValidDepthRatio` | | | | |
| `DepthRoughness` | | | | |
| `HighlightClipping` | | | | |
| `ShadowClipping` | | | | |
| `mean(V)` *(baseline)* | | | | |

> Figures: embed or link them. A figure without a caption stating what it shows
> and what to conclude from it is not evidence.

---

## 4. Derived bands and their justification — **REQUIRED**

All four bands are **one-sided** — one bound each. Fill in the bounds you shipped
in `hw1/thresholds.json`, and justify each one from §3's evidence.

| Factor | Bound (side + value) | Evidence that sets it | Effect of moving it +/-20% |
|---|---|---|---|
| `ValidDepthRatio` (>=) | | | |
| `DepthRoughness` (<=, metres) | | | |
| `HighlightClipping` (<=) | | | |
| `ShadowClipping` (<=) | | | |

For each bound:

> - **What sets it?** Point at the specific measurement. "It looked right on the
>   plot" is not a justification; "the outcome degrades beyond this point and the
>   degradation is larger than the spread within a condition" is.
> - **How sharp is it?** Is there a knee, or did you pick a point on a smooth
>   curve? A smooth curve means the bound is a policy choice — say what policy.
> - **What does it cost?** Every bound rejects frames. How many, and what did
>   rejecting them do to the reconstruction — better, worse, or nothing?
> - **Is it degenerate?** A band that admits every frame you measured gates
>   nothing. Test for this explicitly and report the result.

> **A factor that does not separate is a result, not a failure.** If a band comes
> out degenerate, or moving the bound does not move the outcome, say so, show the
> sweep that demonstrates it, and explain why you think it happens. An argued
> negative result scores **full marks**. A bound asserted without evidence, or a
> degenerate band presented as if it gated something, does not.

---

## 5. Falsifiable prediction — **REQUIRED**

Mandatory, and graded on **structure as much as outcome**. State a prediction that
could have come out wrong, *before* you look, then check it.

### 5.1 The prediction — write this before measuring

> State one prediction about this pipeline, in the form:
>
> > *If [condition], then [measurable quantity] will [direction / magnitude],
> > because [your model of the mechanism].*
>
> It must be **falsifiable**: name the observation that would prove it wrong. If
> no possible measurement could refute it, it is not a prediction and it scores
> zero here — rewrite it.

**Prediction:**

**Mechanism I believe implies it:**

**This is refuted if:**

### 5.2 Pass/fail criterion — fix this before measuring

> What exact threshold on what exact quantity decides confirmed vs. refuted?
> Decide now. Deciding after you see the data is the failure mode this section
> exists to prevent, and it is visible in a report.

### 5.3 The measurement

> The experiment that tests it. Same rigour as §1: what varied, what was
> controlled, sample size.

### 5.4 Verdict

> **Confirmed / Refuted / Inconclusive** — pick one against the criterion in §5.2,
> and defend it.
>
> - If **refuted**: what was wrong with your model, and what do you believe
>   instead now? A refuted prediction reported honestly, with the update it
>   forced, scores **full marks**. This is the cheapest place in the assignment to
>   earn credit and the most common place to lose it by hiding.
> - If **confirmed**: what else would the same mechanism predict, and did you
>   check any of it? One confirmation is weak evidence.
> - If **inconclusive**: what would it have taken to decide? Be specific about
>   sample size or resolution.

> *(Optional, encouraged: a second prediction. Two, one of which failed, is a
> stronger report than one that succeeded.)*

---

## 6. Baseline comparison: your RGB factors vs. mean(V) — **REQUIRED**

`frame_mean_value` ships fully implemented as the **baseline you must beat**
([`definitions.md` §1.5](definitions.md#15-the-baseline-you-must-beat-meanv)). The
comparison isolates one variable — first moment vs. tail statistic — so it is a
fair head-to-head and cannot be won by a nicer channel reduction.

### 6.1 Define "better" — before running it

> What does "predicts degradation better" mean *operationally* in your report?
> Name the comparison. The comparison that grades well is the tau-agnostic
> downstream one: a gate produces a frame set, the frame set produces a
> reconstruction, the reconstruction has a mean L2 and an F-score. Whatever you
> choose, both sides must be run through the **same** protocol.

### 6.2 Result

| Gate | Frames passed | Mean L2 | F-score | Notes |
|---|---|---|---|---|
| No gate (all frames) | | | | |
| `mean(V)` band only | | | | |
| `clip_hi` + `clip_lo` only | | | | |
| All four factors | | | | |
| *(your other conditions)* | | | | |

### 6.3 Interpretation

> - Did the clip pair beat the mean? By how much, and is that larger than the
>   noise in your measurement?
> - Which of `definitions.md` §4's four objections to the mean did you actually
>   observe in your data, and which did not show up? Naming one that did *not*
>   appear is worth as much as confirming the ones that did.
> - If the mean won, or nothing separated: say so and argue it from the evidence.
>   **A negative result argued from evidence scores full marks.** An unsupported
>   claim that the clip pair won does not.
> - The ontology carries `hw1:meanValue` alongside the clip fractions specifically
>   so this comparison can be made by **query** rather than by hand. Did you use
>   `compare_batches.rq` or an equivalent join? Show it.

---

## 7. The proxy caveat — **REQUIRED**

Your reconstruction pipeline is **geometry-only**: ICP consumes points and normals
from the depth image. RGB is not in the objective, not in the correspondence
search, not in the residual. **A clip fraction cannot directly cause a pose
error.** See [`definitions.md` §6](definitions.md#6-the-proxy-caveat-geometric-icp-never-reads-rgb).

> For **each** RGB factor:
>
> - **What is it a proxy for?** Name the quantity you believe it stands in for.
> - **What is the mechanism?** Sketch the chain from that quantity to the
>   reconstruction error. Be concrete about which step you have evidence for and
>   which you are assuming.
> - **What is your evidence?** Correlation between an RGB factor and the outcome
>   is consistent with the proxy story; it is also consistent with several others.
>   Which alternatives did you consider, and what distinguishes them?
> - **What would falsify it?** Name a measurement that would break the chain you
>   propose. If you ran it, report it here or in §5.
>
> A report that concludes "brightness breaks ICP" is **wrong on this pipeline**,
> no matter how clean its correlation plot is. A report that says "clipping does
> not touch ICP; I hypothesise it indicates *X*; here is the evidence and here is
> what would falsify it" is doing the job — even if *X* turns out to be wrong.

---

## 8. Failure cases — **REQUIRED**

> Where does your gate get it wrong? Give **concrete frames or runs**, with
> numbers and, where useful, images.
>
> - **False pass:** a frame your bands admitted that hurt the reconstruction.
>   Which factor should have caught it, and why didn't it?
> - **False reject:** a frame your bands rejected that was fine. What did
>   rejecting it cost?
> - **Invisible failure:** a place the reconstruction went wrong that **none** of
>   the four factors saw at all. This is the most interesting case in the report —
>   it says something about the factor set, not about your bands. What fifth
>   factor would have caught it, and how would you have measured it?
> - **Gate backfire:** did gating ever make the reconstruction *worse* than
>   ungated? Removing frames breaks frame-to-frame continuity; if you saw this,
>   report it, and say what it implies about frame-level gating as a strategy.

---

## 9. What did not work — **REQUIRED**

> Dead ends, abandoned approaches, measurements that came out flat, bugs that
> invalidated a day of results. For each: what you expected, what happened, and
> what you concluded.
>
> This section is **graded, not penalised**. An empty "what did not work" section
> in an open-protocol assignment is not credible — it reads as either a
> too-shallow investigation or an edited one. A clearly argued negative result is
> worth the same as a positive one.

---

## 10. Phase 2: your own capture — **REQUIRED FOR PHASE 2**

### 10.1 Capture and reproducibility

> - How did you collect it, and what coverage did you achieve? (There is an
>   autograded coverage floor — a two-metre stroll does not pass.)
> - **Interactive capture is not reproducible.** Confirm you followed the required
>   flow: collect interactively -> keep `GT_pose.npy` -> **re-render
>   deterministically** via `load.py --trajectory` -> run *all* analysis on the
>   re-rendered capture. State the commands in §11.
> - If you skipped the re-render, none of your phase-2 numbers are reproducible,
>   including by the grader. Say so if that is the case.

### 10.2 Band transfer

> You derived your bands on the provided phase-1 data. Apply them **unchanged** to
> your own phase-2 capture.
>
> - Did they transfer? Show the evidence either way.
> - If they did not: is it the **bands** that failed, the **factors**, or the
>   **scene**? These are different diagnoses with different implications, and
>   distinguishing them is the point of this section.
> - What property would a band have to have to transfer between scenes at all? Do
>   you now believe your bands have it?
> - Would re-deriving on floor 2 give different numbers? If you ran it, show the
>   comparison; if you did not, say what you expect and why.

### 10.3 Mechanism reconciliation

> The phase-2 config is fully visible, and it documents the environment model your
> capture was rendered under.
>
> - Reconcile it with what you **inferred** in phase 1. What did you get right?
> - What did you get **wrong**, and what evidence should have told you sooner?
> - Now that you can read the mechanism, does it change your interpretation of any
>   phase-1 result — including any of your bands, or your §5 prediction? Revising
>   an earlier conclusion here is a good outcome, not an admission.

---

## 11. Reproducibility appendix — **REQUIRED**

> Everything a grader needs to re-run you. Terse is fine; incomplete is not.

- **Commit hash:**
- **Environment:** (pixi env, OS, Python, hardware, GPU if used)
- **Data used:** (which capture directories, and where they came from)
- **`thresholds.json`:** paste it here verbatim, including tau.
- **Commands run, in order:** measurement -> `insert` -> `retrieve` ->
  reconstruction -> scoring. Include the flags.
- **Wall-clock per stage:**
- **Anything nondeterministic** in your pipeline, and how you handled it (seeded,
  averaged, or accepted — and if accepted, the observed spread).

---

## How this is graded

The protocol is open, so grading weight sits on **experimental design and
reasoning**, not on hitting a number.

**Nothing is graded on an observable's value.** `tau` is student-derived, so two
students' `clip_hi_fraction` numbers are not the same quantity. No band bound and
no observable value is a grading target. What is comparable — and what is graded —
is the **tau-agnostic downstream outcome**: your gate produces a frame set, that
frame set produces a reconstruction, and that reconstruction has a mean L2 and an
F-score on the same provided capture.

### Phase 1

| Deliverable | Check | Weight |
|---|---|---|
| ICP implementation | Autograded: `reconstruct.py` on the provided `baseline/` — mean L2 and `completeness.py` F-score against fixed pass bars | 25% |
| Ontology API | Autograded: `test_e2e.py` passes against a live Fuseki | 15% |
| `thresholds.json` + the four measurers | Autograded: schema valid (bands **and** tau), measurers pure and deterministic and matching the shipped closed-form fixtures at the tau the fixture passes in, bands non-degenerate **or** accompanied by an argued degeneracy finding | 10% |
| Baseline comparison (§6) | Rubric: does either clip factor predict reconstruction degradation better than the shipped `mean(V)` baseline? **A negative result argued from evidence scores full marks.** | folded into *evidence and reasoning* |
| `analysis.md` — protocol design (§1, §9, §11) | Rubric: is the experiment capable of answering the question? Controls, confounds, sweep coverage, compute cost acknowledged | 15% |
| `analysis.md` — evidence and reasoning (§2–§8) | Rubric: bands supported by measured reconstruction outcomes; **at least one falsifiable prediction stated and tested**; the proxy caveat correctly handled; **the two depth factors checked empirically for independence** before being treated as separate evidence | 15% |

### Phase 2

| Deliverable | Check | Weight |
|---|---|---|
| Capture meeting a coverage floor | Autograded: F-score of your own reconstruction above a floor, so a two-metre capture cannot pass | 5% |
| Band transfer analysis (§10.2) | Rubric: floor-1 bands applied to floor-2 data; did they transfer? Evidence either way | 10% |
| Mechanism verification (§10.3) | Rubric: the phase-2 config reveals the mechanism — does the report reconcile it with what you inferred in phase 1? | 5% |

Phase-2 trajectories differ per student, so absolute mean L2 is **not** comparable
across submissions and is **not** graded — only the coverage floor and the
reasoning are.

### What earns credit, in one list

- A design that **could have come out the other way**, and says so up front.
- Bands justified by **measured reconstruction outcomes**, not by eyeballing an
  observable's distribution.
- A **falsifiable prediction**, stated with its refutation condition, then
  checked. Refuted-and-updated scores the same as confirmed.
- The **proxy caveat** handled correctly: RGB never enters the ICP objective, so
  any RGB-to-error link is mediated, and you say what by.
- **Negative results argued from evidence** — a factor that does not separate, a
  baseline you failed to beat, a band that did not transfer. Full marks, every
  time, provided the evidence is there.
- **Honest cost and limitation statements.** What you could not afford to run,
  and what that leaves uncertain.

### What loses credit

- A bound with no evidence behind it.
- A degenerate band presented as though it gated something.
- "Brightness breaks ICP", or any causal claim that ignores the geometry-only
  pipeline.
- A prediction written after the measurement, or a pass/fail criterion chosen
  once the numbers were visible.
- An empty §9. In an open-protocol assignment, nothing failing is not credible.
- Numbers that cannot be reproduced from §11.
