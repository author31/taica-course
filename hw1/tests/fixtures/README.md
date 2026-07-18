# Geometry smoke fixture

A five-frame synthetic capture with **exactly known** camera poses. Use it before
you point `reconstruct` at real data.

`reconstruct` is the highest-risk `#TODO` in `utils.py`, because a wrong ICP and a
wrong accumulation loop look identical from the outside — both give you a bad
trajectory. This fixture separates them, by giving every stage of the pipeline a
closed-form answer to check against.

## Layout

```
smoke/
  rgb/{0..4}.png       512*512 8-bit colour
  depth/{0..4}.png     512*512 uint16, millimetres
  GT_pose.npy          (5,7)  [x, y, z, qw, qx, qy, qz], Habitat world frame
  intrinsics.json      {"width", "height", "hfov"}
  expected.json        the oracle (see below)
  clouds.npz           reference point clouds, one per frame
```

The first four are exactly the layout of a real capture, so it is a drop-in
`--data_root`:

```
pixi run -e habitat python hw1/reconstruct.py --data_root hw1/tests/fixtures/smoke
```

## The oracle

`expected.json` gives you, in the frame-0 camera frame (+X right, +Y down, +Z
forward, metres):

| key | what it checks |
|---|---|
| `cloud_stats[i]` | point count, AABB and centroid of frame `i`'s **full** cloud — check `depth_image_to_point_cloud` on its own, before any registration |
| `clouds.npz` | the same clouds, point for point, after `voxel_down_sample(voxel_size)` — downsample yours the same way for a stricter check than `cloud_stats` |
| `steps[i].T_rel` | the transform that aligns frame `i+1` onto frame `i` — check **one** pairwise registration |
| `poses[i]` | camera `i`'s pose and centre — check the accumulation loop |
| `expected_mean_l2_m` | `0.0` |
| `tolerances` | what a correct pipeline actually achieves here |

**A correct pipeline scores `mean_l2 == 0.0` on this fixture**, to within the
millimetre depth quantisation. That is not a target, it is an identity: the ground
truth is derived from the same poses the depth was rendered from, so there is no
sensor noise, no dropout and no unmodelled error to absorb the difference. Whatever
you score above zero is yours.

## Reading the result

| symptom | where to look |
|---|---|
| `cloud_stats` disagrees | `depth_image_to_point_cloud` — intrinsics, axis convention, or the validity mask. Nothing downstream can be right yet. |
| clouds fine, a single `T_rel` disagrees | your ICP. Compare against `local_icp_algorithm`, which ships working, on the same pair. |
| every `T_rel` fine, trajectory still drifts | your accumulation loop — composition order, or which of the pair is source and which is target. |
| trajectory fine, `mean_l2` large | you aligned frames yourself. `mean_l2` owns that; `reconstruct` returns raw frame-0-camera coordinates. |

## Self-check

```
pixi run -e habitat python -m pytest hw1/tests/fixtures/test_icp_smoke.py -v
```

This verifies the fixture, not your code: that the capture is well formed, that the
oracle is internally consistent, and that the shipped `local_icp_algorithm`
recovers every `T_rel` in `expected.json` from an identity initial guess. It never
builds a cloud from a depth image and never chains a trajectory — those are your
two `#TODO`s.

The fixture is generated, not captured: analytic ray casting through the same
pinhole model the assignment uses, with no Habitat, no GPU and no randomness
anywhere. The generator is not part of the student tree — you consume this fixture,
you do not regenerate it.
