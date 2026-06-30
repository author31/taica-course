#!/usr/bin/env python3
# Copyright (c) 2025. HW1 — Camera Pose Estimation in IsaacLab.
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the Jetbot's wheel radius / separation straight off the USD asset.

Use this to confirm whether the hardcoded kinematic constants in
``scripts/hw1.py`` are right:

    WHEEL_RADIUS     = 0.0325   # [m]
    WHEEL_SEPARATION = 0.118    # [m]

It launches Isaac Sim headless (only so the Omniverse/nucleus asset resolver is
available), opens ``jetbot.usd`` as a plain USD stage, finds the two wheel
revolute joints, and derives:

    * wheel separation  = distance between the two wheel-body origins (track width)
    * wheel radius      = half the largest world-space bounding-box extent of a
                          wheel body (a wheel is a flat cylinder; its two large
                          dims are the diameter, the small one is the tyre width)

Both are read from geometry, so treat them as the ground truth to check the
hand-entered constants against.

Run (inside the IsaacLab python env / container):

    python scripts/inspect_jetbot.py
    python scripts/inspect_jetbot.py --robot_usd /abs/path/to/jetbot.usd
"""

from __future__ import annotations

import argparse

# Launch the app FIRST so the `omniverse://` asset resolver + isaacsim paths exist.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Jetbot wheel geometry from its USD.")
parser.add_argument(
    "--robot_usd",
    type=str,
    default=None,
    help="Jetbot USD path. Defaults to <ISAAC_NUCLEUS_DIR>/Robots/NVIDIA/Jetbot/jetbot.usd.",
)
parser.add_argument("--ref_radius", type=float, default=0.0325, help="Hardcoded WHEEL_RADIUS to check.")
parser.add_argument("--ref_separation", type=float, default=0.118, help="Hardcoded WHEEL_SEPARATION to check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# This is a static inspection; force headless so it never needs a display.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Safe to import the rest now.
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402


def _world_pos(prim: Usd.Prim, xf: UsdGeom.XformCache) -> Gf.Vec3d:
    return xf.GetLocalToWorldTransform(prim).ExtractTranslation()


def _world_bbox_size(prim: Usd.Prim, bbox: UsdGeom.BBoxCache) -> Gf.Vec3d:
    rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
    return rng.GetSize()


def main() -> None:
    usd_path = args_cli.robot_usd or f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    print(f"[inspect] opening: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"could not open stage: {usd_path}")

    xf = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    # 1. Find the wheel revolute joints and the wheel body each one drives (body1).
    wheels: list[tuple[str, Usd.Prim]] = []  # (joint_name, wheel_body_prim)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if "wheel" not in name.lower():
            continue
        targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if not targets:
            print(f"[inspect] joint {name!r} has no body1 target — skipping.")
            continue
        body = stage.GetPrimAtPath(targets[0])
        wheels.append((name, body))
        print(f"[inspect] joint {name!r:>22}  ->  body1 {body.GetPath()}")

    if len(wheels) < 2:
        print(f"[inspect] expected >=2 wheel joints, found {len(wheels)}. "
              "Inspect the stage manually (usdview) or pass --robot_usd.")
        simulation_app.close()
        return

    # 2. Wheel separation: distance between the first two wheel-body origins.
    (n0, b0), (n1, b1) = wheels[0], wheels[1]
    p0, p1 = _world_pos(b0, xf), _world_pos(b1, xf)
    separation = (p0 - p1).GetLength()
    print(f"\n[inspect] {n0} origin: ({p0[0]:+.4f}, {p0[1]:+.4f}, {p0[2]:+.4f})")
    print(f"[inspect] {n1} origin: ({p1[0]:+.4f}, {p1[1]:+.4f}, {p1[2]:+.4f})")

    # 3. Wheel radius: half the largest world bbox extent of each wheel body.
    radii = []
    for name, body in wheels:
        size = _world_bbox_size(body, bbox)
        dims = sorted([size[0], size[1], size[2]], reverse=True)
        r = dims[0] / 2.0  # largest extent = diameter
        radii.append(r)
        print(f"[inspect] {name} bbox size: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})  "
              f"-> radius ~= {r:.4f} m")
    radius = sum(radii) / len(radii)

    # 4. Compare against the hardcoded constants.
    def _check(label: str, measured: float, ref: float) -> None:
        diff = measured - ref
        pct = 100.0 * diff / ref if ref else float("nan")
        verdict = "OK" if abs(pct) < 5.0 else "MISMATCH (>5%)"
        print(f"  {label:<18} measured={measured:.4f} m   hardcoded={ref:.4f} m   "
              f"diff={diff:+.4f} m ({pct:+.1f}%)  [{verdict}]")

    print("\n[inspect] ===== comparison =====")
    _check("WHEEL_RADIUS", radius, args_cli.ref_radius)
    _check("WHEEL_SEPARATION", separation, args_cli.ref_separation)
    print("\n[inspect] note: radius from bbox assumes the wheel's largest extent is its "
          "diameter; cross-check in usdview if a hub/axle inflates the box.")

    simulation_app.close()


if __name__ == "__main__":
    main()
