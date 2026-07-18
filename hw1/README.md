# Homework 1 — diagnose geometric ICP, not just implement it

You implement frame-to-frame point-to-plane ICP over RGB-D captures, then use a
sealed experiment notebook to explain why a reconstruction succeeds or fails.
The learning objective is causal reasoning across data generation, measurement,
qualification, and the geometry consumer—not threshold tuning for its own sake.

**Start here: [`instruction.md`](instruction.md)** — the onboarding guide
covering the goal, the two-phase design, the phase-1 dataset, the
experiment-centric workflow (declare → assess → reconstruct → attribute), and
exactly which functions students implement.

## The experiment notebook

A student-authored Turtle declaration selects 1–8 raster quality factors and
records any setting overrides. Assessment appends values and baked Pass/Fail
statuses below a marker and seals the declaration. Reconstruction then adds two
or three run nodes to the same file:

- the full-batch **baseline**, which is the actual convergence outcome;
- the selected-segment **falsification probe**, which tests whether failed
  inputs are load-bearing for geometric ICP.
- an optional **mask-filtered probe**, which keeps the temporal sequence and
  removes only the depth pixels flagged by one selected depth factor.

The selected run is not promised to improve the map. Deleting frames breaks
temporal links and creates segment splices that can dominate the result. The
run therefore records gate/splice/gap evidence alongside trajectory error and,
when a clean reference is supplied, map coverage.

## Quality factors

| Scope | Factor | Observable | Better |
|---|---|---|---|
| depth frame | `HighFrequencyDepthResidual` | high-pass depth residual (m) | lower |
| depth frame | `FlyingPixelRatio` | boundary outlier fraction | lower |
| depth frame | `ValidTileCoverage` | sufficiently supported tile fraction | higher |
| RGB frame | `HighlightClipping` | high-clipped fraction | lower |
| RGB frame | `ShadowClipping` | low-clipped fraction | lower |
| depth pair | `IdentityMedianDepthChange` | unwarped median depth change (m) | lower |
| depth pair | `JointValidDepthRatio` | jointly valid depth fraction | higher |
| depth pair | `PriorWarpDepthResidual` | prior-warp residual (m) | lower |

Each depth factor also emits a uint8 drop mask. Geometry-only ICP does not read
RGB, so RGB failures may be accurate indicators of a capture condition while
being irrelevant to this consumer. Distinguishing “true” from “load-bearing”
is part of the report.
