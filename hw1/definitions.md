# Data-Quality Definitions — Measurement Contracts

This document defines **what each quality factor measures and how it is computed**.
It defines **no thresholds**. There are no bands, no reference numbers and no
"good" ranges anywhere in this file, and that is deliberate: deriving them is the
assignment (see [What you derive](#what-you-derive)).

Read this as a *specification you implement against*, in the same sense as an API
contract: for each factor it fixes the input, the formula, the units, the range,
the invariants that must hold, and the one thing you are free to choose. Your
implementations live in `hw1/api.py`; the docstrings there restate the same
contract in code, and the fixtures in `hw1/test_e2e.py` are its executable form.

---

## Contents

- [0. Shared conventions](#0-shared-conventions)
- [1. The four quality factors](#1-the-four-quality-factors)
  - [1.1 ValidDepthRatio](#11-validdepthratio--valid_depth_fraction)
  - [1.2 DepthRoughness](#12-depthroughness--depth_roughness)
  - [1.3 HighlightClipping](#13-highlightclipping--clip_hi_fraction)
  - [1.4 ShadowClipping](#14-shadowclipping--clip_lo_fraction)
  - [1.5 The baseline you must beat: mean(V)](#15-the-baseline-you-must-beat-meanv)
- [2. Same operator, different quantifier](#2-same-operator-different-quantifier)
- [3. Why `max(R,G,B)` and not luma](#3-why-maxrgb-and-not-luma)
- [4. Why the mean of anything is excluded](#4-why-the-mean-of-anything-is-excluded)
- [5. Scope: single-frame metrics only](#5-scope-single-frame-metrics-only)
- [6. The proxy caveat: geometric ICP never reads RGB](#6-the-proxy-caveat-geometric-icp-never-reads-rgb)
- [7. What you derive](#7-what-you-derive)
- [8. Literature](#8-literature)

---

## 0. Shared conventions

**Frame.** One frame is a paired `(rgb/<i>.png, depth/<i>.png)`. Every factor
below is computed from **one** frame and nothing else — no neighbouring frames,
no capture-level statistics, no reference image. See [§5](#5-scope-single-frame-metrics-only).

**RGB format.** 8-bit RGB PNG. Channel values are integers in `[0, 255]`
(digital numbers, "DN"), not photometric units. `N = H × W` is the number of
**pixels** (not the number of channel samples): every per-pixel statistic below
reduces the three channels to one number first, then counts pixels.

**Depth format.** uint16 PNG in **millimetres**; `metres = raw / 1000.0`.
`raw == 0` is the sensor's "no return" code, not a distance of zero.

**Value channel.** Every RGB statistic in this assignment is computed on

```
V = max(R, G, B)            # HSV Value, per pixel, in [0, 255]
```

No luma, no Rec.601/709 weights, no colorimetric weighting anywhere. The reason
is [§3](#3-why-maxrgb-and-not-luma); it is a design decision, not an
implementation shortcut, and "fixing" it to a weighted sum breaks the factor.

**Purity.** Every measurer is a **pure, deterministic function** of its declared
inputs: read the named file, compute, return. No RNG, no global state, no
capture-level side channels, no reading a second file. This is not a style
preference — the measurers are autograded against closed-form synthetic fixtures,
and an impure measurer cannot pass them.

**Band shape.** All four factors carry **one-sided** bands: a single bound each,
so `resolve_band` carries four numbers and the SPARQL `FILTER` carries four terms.
Which side is fixed by the factor's semantics and is stated per factor below.
The bound *values* are yours to derive.

**Where the numbers land.** Each observable is stored on the ontology node that
owns it (`RGBImage` or `DepthImage`), and the measurement parameters τ are stored
on the `Batch` node as provenance ([§7](#7-what-you-derive)).

---

## 1. The four quality factors

Two categories, four factors: depth (2), RGB (2).

| # | QualityFactor | Observable | Domain | Range | Band side | Node |
|---|---|---|---|---|---|---|
| 1 | `ValidDepthRatio` | `valid_depth_fraction` | depth | `[0, 1]` | `≥ min` | `DepthImage` |
| 2 | `DepthRoughness` | `depth_roughness` (metres) | depth | `[0, ∞)` + `inf` | `≤ max` | `DepthImage` |
| 3 | `HighlightClipping` | `clip_hi_fraction` | RGB | `[0, 1]` | `≤ max` | `RGBImage` |
| 4 | `ShadowClipping` | `clip_lo_fraction` | RGB | `[0, 1]` | `≤ max` | `RGBImage` |

Note the band directions: **only factor 1 points up.** `valid_depth_fraction` is
"higher is better"; the other three are all "lower is better". Both depth factors
are computed from the depth PNG alone, and both RGB factors from the RGB PNG
alone — nothing here reads a point cloud, a reconstruction, or a second file.

`mean(V)` ([§1.5](#15-the-baseline-you-must-beat-meanv)) is **not** in this table.
It is the shipped baseline, not a quality factor.

---

### 1.1 `ValidDepthRatio` — `valid_depth_fraction`

**Measures:** how much of the depth image carries a usable range return.

**Formula.** With `raw` the uint16 depth array and `metres = raw / 1000.0`:

```
valid(i)             = (raw_i != 0) and (min_range <= metres_i <= max_range)
valid_depth_fraction = |{ i : valid(i) }| / N
```

| | |
|---|---|
| **Input** | one depth PNG, plus the declared range window `[min_range, max_range]` in metres |
| **Units** | dimensionless fraction |
| **Range** | `[0, 1]`, both ends attainable |
| **Band** | one-sided, `valid_depth_fraction ≥ min` |
| **Property** | `hw1:validDepthFraction` on the `DepthImage` node |
| **Function** | `frame_valid_fraction` in `hw1/api.py` |

**Invariants.**

- Exactly `0` on an all-zero depth image; exactly `1` when every pixel returns
  inside the window. Both are attainable, so both must be handled.
- **Monotone in the window:** widening `[min_range, max_range]` can never
  decrease it. The window is therefore part of the measurement, not a detail —
  declare it, hold it fixed across every capture you compare, and record it in
  your report. Two `valid_depth_fraction` values measured under different windows
  are not the same quantity.
- **Completely blind to RGB.** No photometric operation on the colour image can
  change this number by even one pixel. If you observe it moving, something in
  the *depth* path moved.
- Counts pixels, not area or volume: a frame pointed at a near wall and a frame
  pointed down a corridor can share a value while carrying very different
  geometry.
- **Counts returns, not their quality.** A pixel that reports *a* number is
  counted as valid however jittery or wrong that number is. A frame can measure
  `valid_depth_fraction = 1.0` and still hand ICP values it cannot register on.
  This factor cannot see that, by construction — it is a property of the validity
  mask and nothing else. That blind spot is exactly why factor 2 exists.

---

### 1.2 `DepthRoughness` — `depth_roughness`

**Measures:** how noisy the depth raster is — an estimate, in metres, of the
standard deviation of the per-pixel jitter riding on top of the scene's actual
geometry.

Factor 1 asks *did this pixel return a number*. Factor 2 asks *how much can that
number be trusted*. ICP consumes the depth **values**, not the validity mask, so
a frame can be fully valid and still be unusable.

**Reference-free**, and that is the property that makes it admissible here: the
estimate is recovered from the one frame in front of you, with no clean twin of
the same view to difference against. A metric that needed a paired clean
reference could not be evaluated on an arbitrary sample at inference time, which
is the standing requirement on every factor in this assignment
([§5](#5-scope-single-frame-metrics-only)).

**The estimator.** Immerkær (1996). Work on the depth image in metres,
`D = raw / 1000.0`, and convolve it with the 3×3 kernel

```
M = [[ 1, -2,  1],
     [-2,  4, -2],
     [ 1, -2,  1]]
```

`M` is the difference of two Laplacian operators, arranged so that **every row and
every column sums to zero**. A locally planar patch of depth therefore produces a
response of exactly zero, no matter how far away or how steeply tilted that plane
is — the kernel is blind to the smooth part of the scene and sees only what the
scene's own geometry does not explain. Then:

```
R = D ∗ M                               # 3x3 convolution, valid windows only
depth_roughness = median(|R|) / (6 · 0.6745)
```

**The two constants are not tuning knobs.** `6 = ‖M‖ = √(Σ Mᵢⱼ²)`: if the noise
on `D` is i.i.d. with standard deviation `σ`, then `R` has standard deviation
`6σ`. `0.6745` is the median of `|N(0,1)|`, which is what converts a median
absolute value back into a standard deviation. Together they scale a robust
spread of `R` into an estimate of `σ` in the same units as `D` — metres.

#### The two details that decide whether you measured this factor at all

Both of these produce a number that *looks* completely plausible if you get them
wrong. Neither will crash, neither will fail a smoke test, and both silently
turn this into a different metric. Read them twice.

**(a) Evaluate only where all nine pixels of the window are valid.**

The depth raster is not dense: `raw == 0` is the "no return" code
([§0](#0-shared-conventions)), and a zero sitting next to a real return is a
several-metre step. Convolve straight through it and `|R|` at that window is
enormous — not because the depth is noisy, but because part of the window is
missing. Build the validity mask first, erode it by the 3×3 window (equivalently:
keep a window only if **all nine** of its pixels are valid), and take the median
over the surviving windows only. The one-pixel image border has no full
neighbourhood and drops out for the same reason.

Skip this and the factor stops reporting noise and starts reporting dropout — at
which point it is a restatement of factor 1, and the whole reason for having a
second depth factor is gone. You would very likely still see it "work", because
it would be tracking something real; it would just be tracking the thing you
already measured.

Be aware of *when* that failure shows up, because it is not gradual. Detail (b)
below partially covers for a missing mask: while only a minority of windows touch
a zero, the median steps over them and the number looks fine. Once dropout is
extensive enough that most windows touch a zero, the median lands *on* a dropout
edge and the observable jumps by orders of magnitude. So the bug is invisible on
the easy frames and enormous on exactly the frames where you most needed factor 2
to say something factor 1 had not. The two details cover for each other, which is
precisely why getting either wrong is silent.

**(b) Take the median of `|R|`, not the mean.**

The literature form is `√(π/2) / 6 · mean|R|`, and it is correct for its
assumption: an image that is smooth plus noise. A real depth raster is not — it
has genuine discontinuities, every object silhouette and every doorway edge, and
each one produces a large legitimate `|R|`. A mean is dominated by that tail, so
the mean form reports *how much depth structure the frame contains* and calls it
noise. Point the camera at a cluttered corner and you get a large value from a
perfectly clean sensor.

The median is the robust order statistic that survives it: as long as fewer than
half the valid windows sit on a discontinuity, the median tracks the noise floor
and ignores the edges. `0.6745` is exactly the constant that makes the median
version comparable to `σ` again. **Use the median.**

#### Contract summary

| | |
|---|---|
| **Input** | one depth PNG. Nothing else — no intrinsics, no point cloud, no normals, no second frame |
| **Units** | **metres.** It is an estimated standard deviation of depth, so it carries depth's unit. A value computed on the raw millimetre array is the same quantity scaled by 1000 and no band over it transfers |
| **Range** | `[0, ∞)`, plus `inf` as the "could not measure" sentinel (see invariants). Exactly `0` is attainable and legitimate — a frame whose every valid window is perfectly planar. There is no structural upper bound. Never test it with `==` |
| **Band** | one-sided, `depth_roughness ≤ max` (**lower is better**) |
| **Property** | `hw1:depthRoughness` on the `DepthImage` node |
| **Function** | `frame_depth_roughness` in `hw1/api.py` |

**Invariants.**

- **Zero response to planar depth.** Because the rows and columns of `M` sum to
  zero, an affine depth patch gives `R = 0` identically. A head-on wall, a tilted
  wall and a sloping floor all contribute nothing. What is left in `R` is what the
  local plane does not explain: noise, and real discontinuities — hence (b).
- **No free parameters.** The kernel, the `6` and the `0.6745` are all fixed by
  this contract. There is nothing here to tune, nothing to justify in your report,
  and nothing to record as measurement provenance. Contrast `τ_lo` / `τ_hi`
  ([§7](#7-what-you-derive)), which are yours and which *must* be recorded.
- **Deterministic and pure.** Pure numpy: a convolution and a median. No RNG, no
  downsampling, no neighbour search, no iteration. The measurer must return
  bit-identical values on repeated calls with the same input, and there is nothing
  in the definition that could make it not.
- **Lower is better**, which is the *opposite* direction to factor 1. Two depth
  observables, two opposite band sides. Getting this backwards produces a gate
  that admits exactly the frames it should reject, and it is a silent failure —
  the query runs, the manifest is non-empty, the frames are wrong.
- **Independence from factor 1 — weaker than the old coupling, but still check it.**
  Because the estimate is taken over fully-valid windows only, losing valid pixels
  does not *bias* `depth_roughness`; it just leaves fewer windows to take a median
  over, so the estimate becomes **noisier**, not systematically higher or lower.
  That is a genuinely weak coupling: the two factors are computed from disjoint
  aspects of the same raster — one from the mask, one from the values under the
  mask. What it does **not** rule out is correlation through the *scene*: whatever
  drives dropout in a region may also drive jitter there, and then the two factors
  move together for reasons neither of them can see. Weak coupling by construction
  is not the same as independence in your data. **Check it empirically** before
  treating the two as separate evidence — and note that *"my second factor told me
  nothing my first had not"* is a full-marks finding, not a failure.
- **Degenerate input: return `inf`.** A frame with no fully-valid 3×3 window
  anywhere leaves the median undefined, and the contract fixes what happens next —
  `frame_depth_roughness` returns **`inf`**. Not `0.0`, not `NaN`, not an
  exception.

  The reasoning matters more than the value, because the wrong choice here is the
  tempting one. `0.0` is arithmetically defensible — no windows, no measured
  roughness — but it is the *best possible* value on a `≤ max` band, so it
  **passes every band you could write**. A frame carrying no usable depth at all
  would sail through the gate as the smoothest frame in the capture. `inf` fails
  any finite bound, so the gate rejects it, which is what you want from a frame
  the measurer could not measure.

  Generalise it: when a metric cannot be computed, return the value that lands on
  the **failing** side of its band. Fail-safe beats plausible. A degenerate value
  that reads as "excellent" is worse than one that reads as "broken", because only
  the second one is visible.

  Such a frame is in practice already far outside factor 1's band, but "another
  factor will catch it" is not a reason to leave this one undefined — the two
  bands are independent bounds and yours may be run alone.

**Sanity fixtures — write these before you touch real data.** All four are
closed-form and take a few lines each:

1. A synthetic **perfect plane** measures exactly `0.0` — a constant-depth frame,
   or a ramp whose slope is a whole number of millimetres per pixel so that
   nothing is lost to the raster's 1 mm quantisation. If it does not, your kernel
   or your metre conversion is wrong. (Off the millimetre grid you are measuring
   the quantisation floor of the format rather than a bug; know which one you are
   looking at before you go debugging.)
2. That same plane plus i.i.d. Gaussian noise of known `σ` measures `≈ σ`. Pick a
   `σ` comfortably above the 1 mm quantisation step so you are testing your
   normalisation and not the raster format. This is the check that the
   `6 · 0.6745` constant is right; nothing else in the contract tests it.
3. A frame containing one large **depth step** — a plane with a block of very
   different depth in it, no noise added — still measures `0.0`, provided fewer
   than half the valid windows straddle the step. If the step shows up as
   roughness, either you took a mean or your window mask is wrong, and details
   (a) and (b) above tell you which.
4. That same plane with **scattered single-pixel dropouts** punched into it — no
   noise, no step in the real depth, just zeros — also measures `0.0`. Scatter
   enough of them that *most* 3×3 windows touch at least one; a single compact
   block will not do, because the median steps over a minority of bad windows all
   by itself and the fixture would pass with the mask missing. This is the one
   test that isolates detail (a): if it comes out large, you convolved across
   dropout boundaries and your factor 2 is a copy of factor 1.

---

### 1.3 `HighlightClipping` — `clip_hi_fraction`

**Measures:** the fraction of the frame that has saturated *at the top* — pixels
whose true radiance is unrecoverable because at least one channel has railed.

```
clip_hi_fraction = |{ i : V_i >= τ_hi }| / N
```

| | |
|---|---|
| **Input** | one RGB PNG, plus `τ_hi` **passed in** |
| **Units** | dimensionless fraction (`τ_hi` is in DN, `[0, 255]`) |
| **Range** | `[0, 1]`, both ends attainable |
| **Band** | one-sided, `clip_hi_fraction ≤ max` |
| **Property** | `hw1:clipHiFraction` on the `RGBImage` node |
| **Function** | `frame_clip_hi_fraction` in `hw1/api.py` |

**Invariants.**

- `τ_hi` is a **parameter**, never a hardcoded constant. It is read from
  `hw1/thresholds.json` and passed in ([§7](#7-what-you-derive)); a measurer that
  bakes in a number cannot be graded against the fixtures.
- **Monotone in τ:** `clip_hi_fraction` is non-increasing as `τ_hi` rises. At
  `τ_hi = 255` it counts only fully railed pixels; at `τ_hi = 0` it is exactly `1`.
- The comparison is **inclusive** (`≥`). Say so and mean it — off-by-one at the
  rail is the single most common bug here, and it is exactly the pixels at 255
  that the factor exists to count.
- **Permutation-invariant:** it is a bag-of-pixels statistic. It knows *how much*
  of the frame is clipped and nothing at all about *where*. Two frames with the
  same value can look completely different.
- It is a **tail statistic** of the distribution of `V`, not a moment. That is
  the whole point ([§4](#4-why-the-mean-of-anything-is-excluded)).

---

### 1.4 `ShadowClipping` — `clip_lo_fraction`

**Measures:** the fraction of the frame crushed *at the bottom* — pixels carrying
no recoverable signal because every channel has bottomed out.

```
clip_lo_fraction = |{ i : V_i <= τ_lo }| / N
```

| | |
|---|---|
| **Input** | one RGB PNG, plus `τ_lo` **passed in** |
| **Units** | dimensionless fraction (`τ_lo` is in DN, `[0, 255]`) |
| **Range** | `[0, 1]`, both ends attainable |
| **Band** | one-sided, `clip_lo_fraction ≤ max` |
| **Property** | `hw1:clipLoFraction` on the `RGBImage` node |
| **Function** | `frame_clip_lo_fraction` in `hw1/api.py` |

**Invariants.**

- `τ_lo` is a parameter, same handling as `τ_hi`.
- **Monotone in τ:** non-decreasing as `τ_lo` rises. At `τ_lo = 0` it counts only
  pure black.
- The comparison is **inclusive** (`≤`).
- **Disjointness.** Provided `τ_lo < τ_hi`, the two pixel sets are disjoint, so
  `clip_lo_fraction + clip_hi_fraction ≤ 1`. Choosing `τ_lo ≥ τ_hi` is not
  forbidden by the formula, and it is meaningless — the two factors would then
  double-count pixels. Your `thresholds.json` must satisfy `τ_lo < τ_hi`.
- Together, §1.3 and §1.4 are the **two tails of one distribution**, kept as two
  observables rather than one union, because the gate does not need the direction
  but the **diagnosis** does: a single "fraction of unusable pixels" cannot tell a
  crushed frame from a blown-out one, and those two failures have different
  causes and different fixes.

Both clip factors are the direct implementation of the unsaturated-region mask of
Shin et al. (IROS 2019), whose eq. 7 marks a pixel usable iff
`τ_l ≤ I(i) ≤ τ_h` — masking out exactly the pixels whose values carry no
recoverable information. We split their single mask into its two tails.

---

### 1.5 The baseline you must beat: `mean(V)`

```
mean_value = (1/N) · Σ V_i          # in [0, 255]
```

`frame_mean_value` ships **fully implemented** in `hw1/api.py`, stores to
`hw1:meanValue` on the `RGBImage` node, and exists for exactly two reasons:

1. It is a **worked example** of the measurer shape you are implementing four
   times — pure, numpy + Pillow, documented units and range.
2. It is the **baseline your two RGB factors must outperform**. It is
   convention-free by construction (same `V`, no weights), so the comparison
   isolates a single variable: *first moment vs. tail statistic*. You cannot win
   it by accident of colorimetry, and you cannot win it by picking a nicer
   channel reduction.

It is **not** a quality factor, it gets no band of its own in
`thresholds.json`, and [§4](#4-why-the-mean-of-anything-is-excluded) is the
argument for why. It is carried in the ontology so the head-to-head comparison
can be made by SPARQL query rather than by hand.

---

## 2. Same operator, different quantifier

The two clip definitions use the **same** operator on the **same** channel
reduction, and they mean different things. This is correct and deliberate:

```
V ≥ τ_hi   ⟺   at least one channel is saturated      (∃)
V ≤ τ_lo   ⟺   every channel is crushed               (∀)
```

Because `V = max(R,G,B)`, `V ≥ τ_hi` is satisfied as soon as *any* single channel
is high, while `V ≤ τ_lo` requires *all three* to be low. One `max`, two
quantifiers — which is precisely the behaviour you want, since saturation is a
per-channel event but crushing is not.

**The symmetric-looking "fix" is wrong.** It is tempting to mirror the operator
and write `min(R,G,B) ≤ τ_lo` for the shadow side. Don't:

```
pure red = (255, 0, 0)  →  min(R,G,B) = 0  →  flagged as CRUSHED
```

A fully blown-out red pixel would be counted as a shadow-clipped pixel. The
frame is the brightest it can be in that channel and the metric reports darkness.
Keep both definitions on `V`, and keep this counter-example in the code comment
so nobody "fixes" it later.

---

## 3. Why `max(R,G,B)` and not luma

Two independent reasons. The second is the stronger one.

1. **Luma is a convention this renderer never promised.** Rec.601 weights
   (0.299 / 0.587 / 0.114) encode an SDTV standard; Rec.709 (0.2126 / 0.7152 /
   0.0722) encodes a different one. Picking either bakes an unjustified
   assumption about display primaries and viewing conditions into a metric you
   are told is grounded. `max` assumes nothing about colour science; it is a
   statement about the *sensor rail*, which is a real, per-channel thing.

2. **Weighting hides single-channel saturation.** Pure red `(255, 0, 0)` has
   Rec.601 luma `76` — it reads as an ordinary mid-tone while the red channel is
   fully railed and its true value is unrecoverable. Clipping is a per-channel
   phenomenon, so the correct test is per-channel. Any weighted sum can trade a
   railed channel against two dark ones and return something unremarkable.

`hw1/test_e2e.py` ships this as an executable fixture: a solid pure-red frame has
`clip_hi_fraction = 1.0`, and Rec.601 luma would call it a mid-tone. No luma
measurer ships; the fixture is kept anyway, as the proof.

---

## 4. Why the mean of anything is excluded

`mean(V)` is a **first moment**, and a first moment is the wrong summary for this
job. Four reasons:

1. **It is not two-sided.** Under-exposure and over-exposure are separate
   failures with separate causes. One scalar constrained to one interval cannot
   hold both: to admit legitimately dark frames the lower bound must drop, to
   admit legitimately bright frames the upper bound must rise, and the interval
   widens until it excludes nothing.

2. **It is confounded with scene content.** Pointing the camera at a bright
   window moves the mean about as much as a genuine exposure fault does. Since
   you are deriving bands *against reconstruction degradation*, that confound is
   pure noise in the target: the metric moves for reasons the reconstruction does
   not care about.

3. **It is blind to distribution shape.** A frame that is half crushed and half
   blown out — the worst case — has a perfectly ordinary mean. The clip pair
   reports `clip_lo ≈ clip_hi ≈ 0.5` and correctly calls it a disaster.
   `hw1/test_e2e.py` ships exactly this fixture (half black, half white,
   `mean_value = 127.5`, "perfectly exposed"). It is the executable proof.

4. **Two-sided moment bands degenerate in practice.** The observed failure mode
   is a band that widens until it spans essentially the whole observed range and
   gates nothing — technically a band, operationally a no-op. Test for this
   explicitly on every factor you derive, including your own: a band that admits
   every frame is a finding, not a result, and it must be reported as such.

**The same trap is documented in the literature**, for the *gradient* family
rather than the exposure family. Zhang, Forster and Scaramuzza (ICRA 2017) show
that plain and log-mapped gradient **sums** (`M_sum`, `M_shim`) are dragged
upward by bright regions and end up selecting over-exposed images, while a
percentile-based (tail) metric picked the frame with the most FAST features in
**13 of 18** datasets. Means and sums fail this way; tail and order statistics do
not. That is the transferable lesson, and `mean(V)` shipping as the beatable
baseline is how you are meant to meet it.

---

## 5. Scope: single-frame metrics only

**In scope:** the four factors above, each a pure function of exactly one frame.

**Out of scope for this assignment:** multi-frame and temporal metrics of any
kind, and — on the **RGB** side specifically — focus / sharpness metrics and
image-noise estimators.

The reason for the single-frame rule is the deployment question the assignment is
built around: a metric that needs its neighbours cannot be evaluated on the
sample in front of you at inference time. Everything here stays
*single-sample computable*, therefore checkable before you commit to using a
frame. Note that this rule is exactly why §1.2 insists on a **reference-free**
noise estimate: an estimator that had to difference the frame against a clean
capture of the same view would be a two-sample metric wearing a one-sample
costume, and it would be uncomputable on any sample you had not already staged.

The rule on RGB sharpness and RGB noise is a scope decision: **the RGB factors in
this assignment are exposure-range factors.** Focus / sharpness metrics and
photometric noise estimators on the colour image are out of scope, will not be
graded, and are not part of the factor set you are deriving bands for. (This is a
statement about the **RGB** factors only — factor 2 is a noise estimator, on the
*depth* raster, and it is very much in scope.) Spend your budget on the four
factors in §1 and on the experiment that grounds them.

### The single-frame limitation, stated precisely

The loose version of this limitation is wrong, so here is the precise one.

Suppose some part of the capture contains a **periodic** photometric modulation —
flicker is the canonical example. A per-frame metric **does** fire there: at
sufficient amplitude, individual frames are driven to the crush rail and to the
blow-out rail, and `clip_lo_fraction` / `clip_hi_fraction` report exactly that,
frame by frame, with no temporal information required.

What a per-frame metric **cannot** do is identify the modulation *as* periodic.
It sees a sequence of extreme frames; it cannot tell you they oscillate, at what
frequency, or that they belong to one phenomenon rather than several.

> **Flicker is not invisible. Its periodicity is.**

The practical consequence: structure *across* frames is not the metric's job and
never will be. If you want to reason about how degradation is organised — over
time, over the trajectory, over anything else — you must bring that structure in
from outside the metric. The frame index and `GT_pose.npy` are both available and
both fair game. Using them in your **analysis** is encouraged; smuggling them
into a **measurer** breaks the single-frame contract and the purity requirement.

---

## 6. The proxy caveat: geometric ICP never reads RGB

**The reconstruction pipeline you implement is geometry-only.** Point-to-plane
ICP consumes points and normals from the *depth* image. The colour image is not
in the objective function, not in the correspondence search, and not in the
residual. RGB may be carried along for visualisation; it never influences a pose.

Therefore:

> A change in `clip_hi_fraction` or `clip_lo_fraction` **cannot** directly cause a
> change in your reconstruction error. Both RGB factors are **upstream proxies**.

If you nevertheless find that an RGB factor predicts reconstruction degradation —
and you may — that predictive power is *mediated*. Something upstream affects
both the colour image and the depth image, and your factor is reading the colour
side of it. That is a perfectly legitimate and very common engineering situation:
proxies are used constantly, precisely because they are cheap to measure. What is
not legitimate is mistaking one for a cause.

**This is graded.** Your report must state, for each RGB factor, **what it is a
proxy for** — the mechanism you hypothesise, the evidence you have for it, and
the evidence you would need to confirm it. A report asserting "brightness breaks
ICP" is wrong on this pipeline, however good its correlation plots are. A report
that says "clipping does not touch ICP; I hypothesise it indicates *X*, here is
what would falsify that" is doing the job.

The two depth factors have no such caveat: `valid_depth_fraction` and
`depth_roughness` are computed from the very raster your ICP unprojects — one
from its validity mask, one from the values under that mask. Note what that
asymmetry buys you: it makes the depth factors a natural control when you are
testing a claim about an RGB factor.

---

## 7. What you derive

No bands and no thresholds are provided. You derive **exactly six numbers**,
justify them in `hw1/analysis.md`, and ship them in one file.

**Six numbers, one file: `hw1/thresholds.json`.**

| Kind | Item | Used at |
|---|---|---|
| Measurement parameter | `τ_lo` | `insert` time |
| Measurement parameter | `τ_hi` | `insert` time |
| Band bound | `valid_depth_fraction` minimum (`≥`) | `retrieve` time |
| Band bound | `depth_roughness` maximum (`≤`) | `retrieve` time |
| Band bound | `clip_hi_fraction` maximum (`≤`) | `retrieve` time |
| Band bound | `clip_lo_fraction` maximum (`≤`) | `retrieve` time |

**Six, and no more.** τ_lo and τ_hi are the **only** measurement parameters you
derive. Both depth factors are fully pinned by their contracts: `depth_roughness`
has no free parameters at all (§1.2 — the kernel and both constants are fixed),
and factor 1's range window `[min_range, max_range]` is a *declared* parameter —
you must state it, hold it fixed across everything you compare, and report it
(§1.1), but it is not derived against reconstruction outcomes and it is not one
of the six.

The file's shape. Every band key is `<observable_name>_<min|max>`, so the key
names the exact quantity it bounds and the side it bounds it on:

```jsonc
{
  "tau_lo": null,               // DN in [0,255], must be < tau_hi
  "tau_hi": null,               // DN in [0,255]
  "bands": {
    "valid_depth_fraction_min": null,  // one-sided, >=
    "depth_roughness_max":      null,  // one-sided, <=  (metres; lower is better)
    "clip_hi_fraction_max":     null,  // one-sided, <=
    "clip_lo_fraction_max":     null   // one-sided, <=
  }
}
```

**Do not confuse these with `api.py`'s command-line override flags.**
`--valid-depth-min`, `--clip-hi-max` and friends are a separate, deliberately
shorter namespace for one-off experiments on the command line; they are **not**
the keys in the file. Writing a CLI spelling into `thresholds.json` gives you a
key the loader does not read.

### τ is a measurement parameter, not a band

This distinction is the one that bites, so it gets its own section.

- **τ is read at `insert` time; bands are read at `retrieve` time.** τ decides
  *what number gets stored*. The band decides *which stored numbers pass*.
- **Change τ and every stored clip fraction is stale.** The triples in the store
  were computed under the old τ and are now silently wrong — the query will still
  run and still return frames, and the frames will be the wrong ones. **Re-run
  `insert`.** The batch is written as one named graph and a re-insert *replaces*
  it, so re-inserting is safe and idempotent. A silently stale batch is the most
  likely way your numbers stop making sense.
- **τ must be recorded on the `Batch` node** (`hw1:tauHi`, `hw1:tauLo`).
  Without it the store holds fractions whose meaning is unrecoverable *from the
  store itself*, and two batches measured under different τ look directly
  comparable when they are not. This is the first point in the assignment where
  the ontology has to describe **how** a number was produced, not just what it
  was. Treat it as the provenance lesson it is.
- **Consequence for grading:** because τ differs between students, two students'
  `clip_hi_fraction` values are not the same quantity, so **no observable value
  and no band bound is a grading target**. What is graded is the τ-agnostic
  downstream outcome — the gate produces a frame set, the frame set produces a
  reconstruction, the reconstruction has a mean L2 and an F-score — plus
  measurer correctness against the fixtures, plus your reasoning.

### A factor that does not separate is a result

Derive all four bands honestly. If a factor turns out **not to separate** on this
scene — the band comes out spanning the whole observed range, or moving the bound
does not move the reconstruction outcome — then say so, show the evidence, and
explain why you believe it. *"This metric does not separate on this data, here is
the sweep that shows it, here is my explanation"* is a **valid and creditable
conclusion**, worth full marks. What is not creditable is a bound with no
evidence behind it, or a degenerate band presented as if it gated something.

---

## 8. Literature

The factors are standard engineering practice, not conventions invented for this
assignment. In citation order of use:

- **Immerkær**, *Fast Noise Variance Estimation*, **CVGIP: Graphical Models and
  Image Processing, 1996**.
  The reference-free noise estimator and its 3×3 kernel `M`, with `‖M‖ = 6` — the
  source for §1.2. The paper's own summary statistic is the **mean** of `|R|`;
  §1.2(b) says why this assignment takes the median instead, and what the
  replacement constant `0.6745` does.

- **Shin et al.**, *Camera Exposure Control for Robust Robot Vision with
  Noise-Aware Image Quality Assessment*, **IROS 2019**.
  The unsaturated-region mask, their eq. 7 (`U(i) = 1` iff `τ_l ≤ I(i) ≤ τ_h`) —
  the direct source for both clip factors (§1.3, §1.4).

- **Zhang, Forster and Scaramuzza**, *Active Exposure Control for Robust Visual
  Odometry in HDR Environments*, **ICRA 2017**.
  The 13-of-18 FAST-feature result against sum-based metrics — the evidence that
  mean/sum statistics lose to tail statistics (§4).

Cite Immerkær 1996 in your `frame_depth_roughness` docstring, and Shin 2019 in
both clip-factor docstrings.
