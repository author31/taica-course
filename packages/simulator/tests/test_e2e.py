"""Simulator pipeline e2e (plan.md "test_e2e.py spec" — the REAL integration gate).

Cases 1-8: replay smoke, determinism, baseline invariance, effects firing,
hard-error trajectory paths, evaluate.py two-run flow, headless/no-pygame.

FAIL-LOUD policy: a missing Replica scene or pose fixture ABORTS collection with
an actionable message (run `pixi run fetch-replica`) instead of skipping — a
skipped suite must never read as green. Escape hatch for machines that
legitimately lack the scene: SIM_E2E_SKIP=1 (skips loudly).

Run: env -u PYTHONPATH pixi run -e habitat python -m pytest packages/simulator/tests/
Runtime target < 2 min: 10-pose fixture (tests/fixtures/mini_secondfloor.npy,
first 10 poses of trajectories/secondfloor.npy), 128x128 sensors.
"""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

# Headless: drop DISPLAY before ANY Engine construction — proves the EGL
# offscreen path works with no X display at all (Engine tolerates unset DISPLAY).
os.environ.pop("DISPLAY", None)

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Repo-root resolution (tests may be invoked from any cwd): walk up from this
# file to the directory containing pixi.toml.
# ---------------------------------------------------------------------------
def _find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pixi.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (no pixi.toml above %s)" % __file__)


REPO = _find_repo_root()
SCENE = REPO / "replica_v1" / "apartment_0" / "habitat" / "mesh_semantic.ply"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_secondfloor.npy"
CONFIG_YAML = REPO / "hw1" / "configs" / "second_floor.yaml"
FPS = 30.0
N_POSES = 10

# --- fail-loud gate (module import/collection time) ------------------------
if os.environ.get("SIM_E2E_SKIP") == "1":
    pytest.skip(
        "SIM_E2E_SKIP=1: simulator e2e suite SKIPPED on request — this machine "
        "claims to legitimately lack the Replica scene. The pipeline was NOT "
        "exercised.",
        allow_module_level=True,
    )

_missing = [str(p) for p in (SCENE, FIXTURE, CONFIG_YAML) if not p.exists()]
if _missing:
    raise pytest.UsageError(
        "simulator e2e prerequisites missing:\n  "
        + "\n  ".join(_missing)
        + "\nrun `pixi run fetch-replica` (scene) / restore tests/fixtures "
        "(fixture). Missing assets are a HARD ERROR, not a skip "
        "(set SIM_E2E_SKIP=1 only on machines that legitimately lack the scene)."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_config():
    """second_floor.yaml with test-local overrides: fixture trajectory,
    absolute scene path, 128x128 sensors (speed — allowed per spec)."""
    from simulator import load_config

    cfg = load_config(str(CONFIG_YAML))
    cfg["scene"]["path"] = str(SCENE)
    cfg["trajectory"] = str(FIXTURE)
    cfg["camera"].update(width=128, height=128)
    cfg["birdseye"].update(width=128, height=128)
    return cfg


# Aggressive scheduler scenario: windows start almost immediately (gap 0.05-0.06s)
# and outlast the 10-frame capture (t <= 10/30 s). Types forced to low_light ONLY
# so the photometric direction (darker) is certain.
AGGRESSIVE_UNCERTAINTIES = {
    "gap_s": [0.05, 0.06],
    "duration_s": [2.0, 3.0],
    "types": {"low_light": {"brightness": 0.3, "gamma": 1.4}},
}


def _run_replay(cfg, scheduler_on, out_dir):
    """One Engine + replay_poses pass over the fixture -> rgb/ depth/ GT_pose.npy
    (+ windows.json when the scheduler is on). Returns the captured (N,7) poses.

    One habitat Simulator per process at a time: the Engine is always closed
    (finally) before the caller constructs the next one."""
    from simulator import (Engine, UncertaintyScheduler, load_trajectory,
                           replay_poses, save_frame)

    poses = load_trajectory(cfg["trajectory"])
    out_dir = Path(out_dir)
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    out = dict(cfg["output"], save_rgb=True, save_depth=True, save_semantic=False)
    scheduler = UncertaintyScheduler(cfg["uncertainties"]) if scheduler_on else None

    engine = Engine(cfg, scheduler=scheduler, fps_nominal=FPS)
    try:
        def out_cb(frame, sensor_state, idx):
            save_frame(frame, sensor_state, str(out_dir), out, idx)

        captured = replay_poses(engine, poses, out_cb)
        np.save(out_dir / "GT_pose.npy", np.asarray(captured, dtype=np.float32))
        if scheduler is not None:
            scheduler.save(str(out_dir / "windows.json"))
    finally:
        engine.close()
    return captured


def _png(root, kind, idx):
    return Path(root) / kind / f"{idx}.png"


def _read_windows(run_dir):
    with open(Path(run_dir) / "windows.json") as f:
        return json.load(f)


def _in_any_window(t, windows):
    return any(w["start_s"] <= t < w["end_s"] for w in windows)


_EVALUATE_CACHE = {}


def _load_evaluate():
    """Import scripts/evaluate.py as a module (it is not a package member)."""
    if "mod" not in _EVALUATE_CACHE:
        spec = importlib.util.spec_from_file_location(
            "evaluate_e2e", str(REPO / "scripts" / "evaluate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EVALUATE_CACHE["mod"] = mod
    return _EVALUATE_CACHE["mod"]


# ---------------------------------------------------------------------------
# Module-scoped replay runs (Engines constructed sequentially, each closed
# before the next — determinism/invariance need fresh Engines by design).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e_runs")
    cfg = _base_config()
    out = {"cfg": cfg, "root": root}

    out["baseline"] = root / "baseline"          # scheduler OFF
    out["cap_baseline"] = _run_replay(cfg, False, out["baseline"])

    out["sched1"] = root / "sched1"              # scheduler ON, default gaps
    out["cap_sched1"] = _run_replay(cfg, True, out["sched1"])

    out["sched2"] = root / "sched2"              # identical rerun (determinism)
    out["cap_sched2"] = _run_replay(cfg, True, out["sched2"])

    acfg = copy.deepcopy(cfg)                    # aggressive: windows from ~t=0.05
    acfg["uncertainties"].update(copy.deepcopy(AGGRESSIVE_UNCERTAINTIES))
    out["aggressive"] = root / "aggressive"
    out["cap_aggressive"] = _run_replay(acfg, True, out["aggressive"])
    return out


# ---------------------------------------------------------------------------
# 1. replay_smoke
# ---------------------------------------------------------------------------
def test_replay_smoke(runs):
    run = runs["sched1"]
    for i in range(1, N_POSES + 1):
        assert _png(run, "rgb", i).is_file(), f"missing rgb/{i}.png"
        assert _png(run, "depth", i).is_file(), f"missing depth/{i}.png"

    gt = np.load(run / "GT_pose.npy")
    assert gt.shape == (N_POSES, 7)

    fixture = np.load(FIXTURE)
    # Teleport replay is exact: captured sensor poses match the fixture input
    # (empirically max abs diff ~6e-8; loose-but-meaningful tolerance).
    assert np.allclose(gt, fixture, atol=1e-5), (
        "captured poses drifted from fixture input "
        f"(max abs diff {np.abs(gt - fixture).max():.3e})")

    windows = _read_windows(run)
    assert isinstance(windows, list) and len(windows) >= 1
    valid_types = set(runs["cfg"]["uncertainties"]["types"])
    for w in windows:
        assert set(w) == {"start_s", "end_s", "type", "params"}, f"bad schema: {w}"
        assert isinstance(w["start_s"], float) and isinstance(w["end_s"], float)
        assert w["start_s"] < w["end_s"]
        assert w["type"] in valid_types
        assert isinstance(w["params"], dict) and w["params"]


# ---------------------------------------------------------------------------
# 2. determinism
# ---------------------------------------------------------------------------
def test_determinism(runs):
    a, b = runs["sched1"], runs["sched2"]
    for kind in ("rgb", "depth"):
        for i in range(1, N_POSES + 1):
            assert _png(a, kind, i).read_bytes() == _png(b, kind, i).read_bytes(), (
                f"same config+seed produced different {kind}/{i}.png")
    assert (a / "windows.json").read_bytes() == (b / "windows.json").read_bytes()
    assert np.array_equal(runs["cap_sched1"], runs["cap_sched2"])


# ---------------------------------------------------------------------------
# 3. baseline_invariance (per-frame-RNG invariant: out-of-window frames are
#    bit-identical between scheduler ON and OFF runs)
# ---------------------------------------------------------------------------
def test_baseline_invariance_default_gaps(runs):
    # Default gap_s [8,25]: all 10 frames (t <= 10/30 s) sit inside the first
    # gap, so EVERY frame must match the scheduler-off run bit-for-bit...
    windows = _read_windows(runs["sched1"])
    assert windows[0]["start_s"] > N_POSES / FPS, (
        "expected the first realized window to start after the capture")
    for kind in ("rgb", "depth"):
        for i in range(1, N_POSES + 1):
            assert (_png(runs["baseline"], kind, i).read_bytes()
                    == _png(runs["sched1"], kind, i).read_bytes()), (
                f"{kind}/{i}.png differs between scheduler ON/OFF despite "
                f"t={i / FPS:.3f}s being outside all windows")
    # ...and windows DO exist later in time (the scheduler is not a no-op).
    assert len(windows) >= 1


def test_baseline_invariance_aggressive(runs):
    # Aggressive gaps: the first window starts at ~t=0.05s, so frame 1
    # (t=1/30) is outside and frames >= 2 are inside. Out-of-window frames
    # must be bit-identical to baseline; in-window frames must differ.
    windows = _read_windows(runs["aggressive"])
    inside = [i for i in range(1, N_POSES + 1) if _in_any_window(i / FPS, windows)]
    outside = [i for i in range(1, N_POSES + 1) if i not in inside]
    assert inside, "aggressive scenario realized no in-capture window frames"
    assert outside, "aggressive scenario left no out-of-window frames to compare"

    for i in outside:
        for kind in ("rgb", "depth"):
            assert (_png(runs["baseline"], kind, i).read_bytes()
                    == _png(runs["aggressive"], kind, i).read_bytes()), (
                f"out-of-window {kind}/{i}.png differs from baseline")
    for i in inside:
        assert (_png(runs["baseline"], "rgb", i).read_bytes()
                != _png(runs["aggressive"], "rgb", i).read_bytes()), (
            f"in-window rgb/{i}.png identical to baseline — effect did not fire")


# ---------------------------------------------------------------------------
# 4. effects_fire (low_light only: luma drops; light-coupled depth dropout rises)
# ---------------------------------------------------------------------------
def test_effects_fire(runs):
    import cv2

    windows = _read_windows(runs["aggressive"])
    assert all(w["type"] == "low_light" for w in windows)
    inside = [i for i in range(1, N_POSES + 1) if _in_any_window(i / FPS, windows)]
    assert inside
    i = inside[-1]

    rgb_base = cv2.imread(str(_png(runs["baseline"], "rgb", i)))
    rgb_low = cv2.imread(str(_png(runs["aggressive"], "rgb", i)))
    assert rgb_base is not None and rgb_low is not None
    # brightness 0.3 (even with gamma 1.4 lift) must darken the frame.
    assert rgb_low.mean() < rgb_base.mean(), (
        f"low_light frame {i} not darker: {rgb_low.mean():.2f} vs "
        f"baseline {rgb_base.mean():.2f}")

    d_base = cv2.imread(str(_png(runs["baseline"], "depth", i)), cv2.IMREAD_UNCHANGED)
    d_low = cv2.imread(str(_png(runs["aggressive"], "depth", i)), cv2.IMREAD_UNCHANGED)
    assert d_base is not None and d_low is not None and d_base.dtype == np.uint16
    frac_base = float(np.mean(d_base == 0))
    frac_low = float(np.mean(d_low == 0))
    # Light coupling: brightness 0.3 -> stress 0.7 -> dropout_prob ~= 0.21
    # (light_dropout_gain 0.3) on top of a dropout-free baseline.
    assert frac_low > frac_base + 0.10, (
        f"light-coupled depth dropout did not fire: zero-frac {frac_low:.3f} "
        f"(low_light) vs {frac_base:.3f} (baseline)")


# ---------------------------------------------------------------------------
# 5. missing_trajectory_raises (evaluate collect path — hard error, no [skip])
# ---------------------------------------------------------------------------
def test_missing_trajectory_raises(tmp_path):
    evaluate = _load_evaluate()
    cfg = _base_config()
    cfg["trajectory"] = str(tmp_path / "does_not_exist.npy")
    cfg["output"]["root"] = str(tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="trajectory not found"):
        evaluate.collect(cfg, FPS)


# ---------------------------------------------------------------------------
# 6. json_trajectory_raises (action replay deprecated; pointer to .npy)
# ---------------------------------------------------------------------------
def test_json_trajectory_raises():
    from simulator import load_trajectory

    with pytest.raises(ValueError, match=r"\.npy"):
        load_trajectory("x.json")


# ---------------------------------------------------------------------------
# 7. evaluate_two_run (full main flow: collect baseline+mixed, GT ref from
#    baseline/ only, score both conditions, per-window CSV)
# ---------------------------------------------------------------------------
def test_evaluate_two_run(tmp_path, monkeypatch):
    import yaml

    evaluate = _load_evaluate()
    cfg = _base_config()
    data_root = tmp_path / "data"
    out_dir = tmp_path / "eval"
    cfg["output"]["root"] = str(data_root)
    cfg_path = tmp_path / "e2e_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--config", str(cfg_path),
        "--out-dir", str(out_dir), "--fps", str(FPS)])
    evaluate.main()

    # Both conditions captured: 10 frames + GT_pose each; mixed has windows.json.
    for cond in ("baseline", "mixed"):
        droot = data_root / cond
        for kind in ("rgb", "depth"):
            files = sorted(p.name for p in (droot / kind).glob("*.png"))
            assert len(files) == N_POSES, f"{cond}/{kind}: {files}"
        assert np.load(droot / "GT_pose.npy").shape == (N_POSES, 7)
    windows = _read_windows(data_root / "mixed")
    assert isinstance(windows, list) and len(windows) >= 1
    assert not (data_root / "baseline" / "windows.json").exists()

    # results.csv: one scored row per condition. Non-empty accuracy/f columns
    # prove the F-score step ran against the GT reference built from baseline/.
    with open(out_dir / "results.csv") as f:
        rows = {r["condition"]: r for r in __import__("csv").DictReader(f)}
    assert set(rows) == {"baseline", "mixed"}
    for cond, r in rows.items():
        assert int(r["n_frames"]) == N_POSES
        assert r["mean_l2"] != "", cond
        assert r["accuracy"] != "" and r["f_score"] != "", (
            f"{cond}: F-score empty — GT reference (from baseline/) not consumed")

    # per-window CSV: one row per realized window + the trailing clean row,
    # frame bounds consistent with the seconds -> frame mapping (t = i/fps).
    with open(out_dir / "per_window.csv") as f:
        wrows = list(__import__("csv").DictReader(f))
    assert len(wrows) == len(windows) + 1          # + trailing "clean" row
    assert wrows[-1]["window"] == "clean"
    for w, r in zip(windows, wrows):
        assert r["type"] == w["type"]
        if r["frame_start"] != "":
            f0 = max(1, int(round(w["start_s"] * FPS)))
            assert int(r["frame_start"]) == f0
    # Default gaps put every realized window beyond the 10-frame capture, so
    # all frames are clean.
    assert int(wrows[-1]["n_frames"]) == N_POSES


# ---------------------------------------------------------------------------
# 8. headless — must stay LAST in this module: after every Engine run above,
#    pygame was never imported (viewer is opt-in; nothing here touches it).
# ---------------------------------------------------------------------------
def test_z_no_pygame(runs):
    assert "DISPLAY" not in os.environ
    assert "pygame" not in sys.modules, (
        "pygame was imported during the e2e suite — the headless pipeline must "
        "never touch simulator.viewer")
