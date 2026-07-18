# HW1 factor definitions — v4

This document defines the selectable input factors implemented by `api.py`.
Every depth factor produces both a scalar for qualification and a uint8 0/255
drop mask for intervention. A mask value of 255 means “drop this pixel”; 0 means
“keep it.” Scalars and masks come from one shared computation.

## Shared conventions

- Depth PNGs are uint16 millimetres; metres = `raw / 1000`.
- Active depth factors use the ICP consumer's validity rule: `raw != 0`. There
  is no configurable range cap.
- Frame factors inspect one raster. Pair factors inspect one ordered adjacent
  depth pair. `PriorWarpDepthResidual` may additionally use capture intrinsics
  and the declared constant-velocity prior, but never the current pair's fitted
  transform.
- Lower-is-better factors pass at `value <= threshold`; higher-is-better factors
  pass at `value >= threshold`.
- A non-measurable lower-is-better factor returns `inf`; a non-measurable
  higher-is-better factor returns `0.0`. These grade Fail without a special case.
- Every pair factor records the number of pixels that contributed to its value.
- Qualification defaults are provisional and must be calibrated on clean and
  controlled-corruption captures.

## Active menu

| Scope | Factor | Observable | Direction | Drop unit |
|---|---|---|---|---|
| RGB frame | `HighlightClipping` | `clipHiFraction` | lower | none |
| RGB frame | `ShadowClipping` | `clipLoFraction` | lower | none |
| depth frame | `HighFrequencyDepthResidual` | `highFrequencyDepthResidual` (m) | lower | residual centre pixel |
| depth frame | `FlyingPixelRatio` | `flyingPixelRatio` | lower | mixed boundary pixel |
| depth frame | `ValidTileCoverage` | `validTileCoverage` | higher | under-supported tile |
| depth pair | `IdentityMedianDepthChange` | `identityMedianDepthChange` (m) | lower | change outlier |
| depth pair | `JointValidDepthRatio` | `jointValidDepthRatio` | higher | non-jointly-valid coordinate |
| depth pair | `PriorWarpDepthResidual` | `priorWarpDepthResidual` (m) | lower | prior-warp outlier |

## RGB factors

Let `V = max(R,G,B)` on an 8-bit RGB frame.

`HighlightClipping` is the fraction of pixels with `V >= tauHi` and is qualified
by `maxClipHiFraction`. `ShadowClipping` is the fraction with `V <= tauLo` and is
qualified by `maxClipLoFraction`. Their implementation, parameters, thresholds,
and annotation placement are unchanged from v3.

Geometry-only ICP does not read RGB. These factors therefore describe intrinsic
RGB health and serve as negative controls for this consumer; an RGB-only
corruption is expected to leave `MapMeanL2` unchanged.

## Depth-frame factors

### `HighFrequencyDepthResidual`

On every fully valid 3×3 window, apply the Immerkær kernel

```text
 1 -2  1
-2  4 -2
 1 -2  1
```

and let `R` be its response. The scalar is

```text
median(abs(R)) / (6 * 0.6745)
```

in metres. The mask flags response centres exceeding
`max(residualMaskK * median(abs(R)), 0.001 m)`. The one-millimetre minimum avoids
calling a real step edge noise when a quantised plane has zero median response.
No fully valid 3×3 window gives `inf` and an empty drop mask.

### `FlyingPixelRatio`

For each consumer-valid pixel near a local depth discontinuity, remove the centre
from its configured odd window, split neighbour depths at their largest gap, and
fit one image-coordinate depth plane to each side. The centre is a flying pixel
when the two predicted planes differ by more than twice
`flyingPixelPlanarityTol`, the centre lies between their predictions, and it is
farther than that tolerance from both.

The scalar is flagged pixels divided by all raster pixels. The mask contains the
flagged pixels. A valid planar/no-edge frame returns 0 and an empty mask; a frame
with no valid depth returns `inf`.

### `ValidTileCoverage`

Partition the raster from the top-left into `tileSize` squares, retaining smaller
boundary tiles. A tile is supported when its non-zero depth fraction is at least
`tileValidFloor`. The scalar is supported tiles divided by all tiles. The mask
flags every pixel in an unsupported tile, so filtering removes weak geometry as
a region rather than leaving isolated points.

## Adjacent-depth-pair factors

### `IdentityMedianDepthChange`

On coordinates valid in both rasters, compute `C = abs(D0 - D1)`. The scalar is
`median(C)`. The mask flags joint-valid coordinates above

```text
median(C) + changeMaskK * 1.4826 * MAD(C)
```

with a one-millimetre minimum above the median when MAD is zero. Shape mismatch
or no jointly valid coordinate gives `inf`, an empty mask, and count zero.

### `JointValidDepthRatio`

The scalar is the number of coordinates non-zero in both rasters divided by the
number of raster coordinates. The mask flags the complement of joint validity.
Shape mismatch or an empty raster gives 0 and count zero.

### `PriorWarpDepthResidual`

Using capture intrinsics, unproject frame 0, transform its points into frame 1
with the declared constant-velocity prior, project to frame 1, and compare warped
depth with the observed depth. Points behind the observed surface by more than
`priorWarpDepthGate` are treated as occluded and excluded from the scalar. The
scalar is the median absolute residual over visible correspondences; the mask
flags projected residuals above the gate.

This factor is measured inside the baseline reconstruction loop immediately
before fitting the current pair. `experiment` records its settings and pair
scope, while `write_pair_measurements` later writes its value, status,
contributing count, and mask. No correspondence gives `inf`, an empty mask, and
count zero.

## Mask application

`experiment` writes masks below `<experiment>/masks/<Factor>/`:

```text
<stem>.png       frame factor
<i>_<j>.png      pair factor
```

`reconstruct.py --mask-dir .../<Factor>` indexes both forms. Pair masks are
incident on both endpoint frames. All incident drop masks are ORed and converted
to `keep_mask = (drop == 0)`. Point-cloud validity remains
`(depth_m > 0) & keep_mask`; a mask can never resurrect an invalid depth.

## Validation obligation

A factor is not consumer-binding merely because its mask changes a point cloud.
Promotion requires a matched clean/corrupted full-batch contrast, per-link
mechanism evidence, and held-out corruption validation. Report both the scalar
response and the baseline-versus-masked `MapMeanL2` differential. Preserve null
or adverse filtering results.
