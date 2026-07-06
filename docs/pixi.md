# pixi environment

The pixi workspace (`pixi.toml`) replaces the conda instructions from the
official habitat docs:

```bash
conda create -n habitat python=3.12 cmake=3.27
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

with a reproducible, lockfile-backed pixi environment.

## Usage

```bash
pixi install -e habitat      # build the environment
pixi run -e habitat smoke     # verify habitat-sim imports (with bullet)
pixi shell -e habitat         # drop into the env
```

## Notes

**Python version.** The official doc says `python=3.12`, but the `aihabitat`
conda channel only publishes py3.9 builds for habitat-sim 0.3.x (0.3.0 - 0.3.3
are all `py3.9_*_linux`). Pinning 3.12 makes the solve fail to find a
habitat-sim build, so this env pins Python 3.9 instead.

**Channel order.** `channels = ["conda-forge", "aihabitat"]` (conda-forge first)
mirrors `-c conda-forge -c aihabitat`.

**numpy pin.** numpy is pinned to habitat-lab's required version so the conda
side and the editable pypi install agree on a single numpy.

**`withbullet`.** This is an (empty) track-features package on aihabitat that
flips the habitat-sim-mutex to the bullet-enabled build, exactly like
`conda install habitat-sim withbullet`.

**habitat-lab.** Installed editable from the submodule, pinned at tag v0.3.3 to
match habitat-sim 0.3.3.

## isaaclab

The `isaaclab` stack still lives in the Docker flow (see `Makefile` /
`Dockerfile`); a second `[feature.isaaclab]` can be added to `pixi.toml` later
without disturbing the habitat env.
