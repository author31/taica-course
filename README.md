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

