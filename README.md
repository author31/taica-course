# TAICA Course

Course materials and assignments, built on [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
and the Replica `apartment_0` scene. The environment is managed with
[pixi](https://pixi.sh).

## Setup

```bash
curl -fsSL https://pixi.sh/install.sh | bash        # install pixi
git submodule update --init dependencies/habitat-lab # editable dependency
pixi install -e habitat                              # Python 3.9 + habitat-sim + open3d + pygame
pixi run smoke                                       # sanity check
pixi run fetch-replica                               # download the apartment_0 scene
```

See [`docs/pixi.md`](docs/pixi.md) for details on the pixi workspace.

## Assignments

- **[Homework 1 — Geometry-only ICP SLAM under Environment Uncertainty](hw1/README.md)**
  — implement a geometry-only ICP SLAM pipeline and stay robust to lighting /
  depth-sensor faults injected as seeded temporal windows, scored by mean L2
  (baseline vs mixed run, whole-episode + per-window). Code in [`hw1/`](hw1/),
  config at [`hw1/configs/second_floor.yaml`](hw1/configs/second_floor.yaml),
  evaluator at [`scripts/evaluate.py`](scripts/evaluate.py), shared simulation
  library in [`packages/simulator/`](packages/simulator/).

Everything runs inside the pixi `habitat` environment
(`pixi run -e habitat python ...`).
