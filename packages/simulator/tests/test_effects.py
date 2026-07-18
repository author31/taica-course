"""Lane B unit tests: pixel pipeline + UncertaintyScheduler (pure numpy).

Loads simulator/effects.py straight from its file path (not via the
`simulator` package) so these tests never import habitat_sim / pygame and run
outside the sim environment too.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_EFFECTS_PATH = Path(__file__).resolve().parents[1] / "simulator" / "effects.py"
_spec = importlib.util.spec_from_file_location("_effects_under_test", _EFFECTS_PATH)
effects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(effects)


# --- fixtures -----------------------------------------------------------------
def sched_cfg(**over):
    cfg = {
        "enabled": True,
        "seed": 42,
        "gap_s": [8.0, 25.0],
        "duration_s": [1.0, 4.0],
        "types": {
            "flicker": {"amplitude": 0.9, "frequency": 6.0},
            "low_light": {"brightness": 0.3, "gamma": 1.4},
            "over_exposure": {"brightness": 2.2, "contrast": 1.2},
        },
    }
    cfg.update(over)
    return cfg


DEPTH_CFG = {
    "enabled": True,
    "stuck": False,
    "noise_std": 0.05,
    "quantization": 0.05,
    "min_range": 0.3,
    "max_range": 5.0,
    "dropout_prob": 0.05,
    "light_nominal": 1.0,
    "light_noise_gain": 1.5,
    "light_dropout_gain": 0.2,
    "light_range_gain": 0.3,
}


def fixed_depth():
    return np.linspace(0.0, 6.0, 48, dtype=np.float32).reshape(6, 8)


def fixed_rgb():
    return (np.arange(4 * 5 * 3, dtype=np.uint8) * 4 % 256).reshape(4, 5, 3)


# --- scheduler determinism ----------------------------------------------------
def test_same_seed_same_timeline_regardless_of_granularity():
    a = effects.UncertaintyScheduler(sched_cfg())
    b = effects.UncertaintyScheduler(sched_cfg())
    for t in np.arange(0.0, 120.0, 1.0 / 30.0):
        a.active(float(t))
    a.active(120.0)
    for t in np.arange(0.0, 120.0, 1.0 / 10.0):
        b.active(float(t))
    b.active(120.0)
    assert a.windows == b.windows
    assert len(a.windows) > 0


def test_same_seed_same_timeline_regardless_of_unroll_depth():
    a = effects.UncertaintyScheduler(sched_cfg())
    for t in np.arange(0.0, 120.0, 1.0 / 30.0):
        a.active(float(t))
    a.active(120.0)

    # single jump straight to t=120 realizes the identical timeline
    c = effects.UncertaintyScheduler(sched_cfg())
    c.active(120.0)
    assert c.windows == a.windows

    # unrolling much further only appends — the shared prefix is bit-identical
    d = effects.UncertaintyScheduler(sched_cfg())
    d.active(600.0)
    assert len(d.windows) > len(a.windows)
    assert d.windows[: len(a.windows)] == a.windows


def test_non_monotonic_queries_consistent():
    grid = [float(t) for t in np.arange(0.0, 120.0, 0.25)] + [120.0]
    ref = effects.UncertaintyScheduler(sched_cfg())
    ref_answers = {t: ref.active(t) for t in grid}  # monotonic reference

    s = effects.UncertaintyScheduler(sched_cfg())
    order = np.random.default_rng(7).permutation(len(grid))
    for i in order:
        t = grid[int(i)]
        assert s.active(t) == ref_answers[t]
    # repeated queries at past t are pure lookups: timeline unchanged
    n = len(s.windows)
    for t in (3.0, 50.0, 3.0, 119.0, 0.0):
        assert s.active(t) == ref_answers[t]
    assert len(s.windows) == n
    assert s.windows == ref.windows


def test_gaps_durations_in_range_and_non_overlapping():
    cfg = sched_cfg()
    s = effects.UncertaintyScheduler(cfg)
    s.active(600.0)
    assert len(s.windows) >= 10
    prev_end = 0.0  # timeline starts at t=0 with a gap
    for start, end, name, params in s.windows:
        gap = start - prev_end
        assert cfg["gap_s"][0] <= gap <= cfg["gap_s"][1]
        assert cfg["duration_s"][0] <= end - start <= cfg["duration_s"][1]
        assert start >= prev_end  # non-overlapping by construction
        assert name in cfg["types"]
        assert params == cfg["types"][name]
        prev_end = end


def test_flicker_phase_is_window_relative():
    cfg = sched_cfg(types={"flicker": {"amplitude": 0.9, "frequency": 6.0}})
    s = effects.UncertaintyScheduler(cfg)
    s.active(300.0)
    assert len(s.windows) >= 3
    base = {"ambient_rgb": [1.0, 1.0, 1.0], "brightness": 1.0, "contrast": 1.0, "gamma": 1.0}
    for start, end, name, params in s.windows:
        ov = s.active(start)
        assert ov, "window start must be inside the window (half-open [start, end))"
        lighting = {**base, **ov["lighting"]}
        f = lighting["frequency"]
        assert lighting["phase"] == -2.0 * math.pi * f * start
        # oscillation term is zero at window start -> gain factor == 1
        term = math.sin(2.0 * math.pi * f * start + lighting["phase"])
        assert abs(term) < 1e-9
        assert abs(effects.light_exposure(lighting, t=start) - lighting["brightness"]) < 1e-9


def test_active_outside_windows_and_boundaries():
    cfg = sched_cfg()
    s = effects.UncertaintyScheduler(cfg)
    s.active(200.0)
    start, end, name, params = s.windows[0]
    assert s.active(0.0) == {}                      # inside the initial gap
    assert s.active(start - 1e-6) == {}             # just before the window
    assert s.active(start) != {}                    # start is inclusive
    assert s.active(end) == {}                      # end is exclusive
    # non-flicker windows override lighting with their fixed params
    for w_start, w_end, w_name, w_params in s.windows:
        ov = s.active((w_start + w_end) / 2.0)
        assert set(ov) == {"lighting"}
        if w_name == "flicker":
            assert ov["lighting"] == {**w_params,
                                      "phase": -2.0 * math.pi * w_params["frequency"] * w_start}
        else:
            assert ov["lighting"] == w_params


def test_disabled_scheduler_always_baseline():
    s = effects.UncertaintyScheduler(sched_cfg(enabled=False))
    for t in (0.0, 5.0, 50.0, 500.0, 3.0):
        assert s.active(t) == {}
    assert s.windows == []


def test_save_writes_realized_windows(tmp_path):
    import json

    s = effects.UncertaintyScheduler(sched_cfg())
    s.active(100.0)
    path = tmp_path / "windows.json"
    s.save(path)
    data = json.loads(path.read_text())
    assert len(data) == len(s.windows)
    for row, (start, end, name, params) in zip(data, s.windows):
        assert row == {"start_s": start, "end_s": end, "type": name, "params": params}


# --- golden-array pixel tests -------------------------------------------------
def test_apply_lighting_golden():
    rgb = fixed_rgb()
    cfg = {"ambient_rgb": [1.0, 0.9, 0.7], "brightness": 1.3,
           "contrast": 1.15, "gamma": 1.8, "amplitude": 0.0}
    out = effects.apply_lighting(rgb, cfg, t=0.0)

    img = rgb.astype(np.float32) / 255.0
    img *= np.asarray(cfg["ambient_rgb"], dtype=np.float32)
    img *= cfg["brightness"]
    img = (img - 0.5) * cfg["contrast"] + 0.5
    img = np.clip(img, 0.0, 1.0) ** (1.0 / cfg["gamma"])
    expected = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)


def test_apply_lighting_flicker_golden():
    rgb = fixed_rgb()
    cfg = {"ambient_rgb": [1.0, 1.0, 1.0], "brightness": 1.0, "contrast": 1.0,
           "gamma": 1.0, "amplitude": 0.5, "frequency": 2.0, "phase": 0.3}
    t = 0.37
    out = effects.apply_lighting(rgb, cfg, t=t)

    osc = 1.0 + 0.5 * np.sin(2.0 * np.pi * 2.0 * t + 0.3)
    img = rgb.astype(np.float32) / 255.0 * osc
    expected = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    assert np.array_equal(out, expected)
    assert abs(effects.light_exposure(cfg, t=t) - osc) < 1e-12


def test_apply_depth_faults_golden():
    depth = fixed_depth()
    out = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng(9))

    rr = np.random.default_rng(9)  # replay the exact draw sequence
    d = depth.astype(np.float32).copy()
    d += rr.normal(0.0, DEPTH_CFG["noise_std"], size=d.shape).astype(np.float32)
    step = DEPTH_CFG["quantization"]
    d = np.round(d / step) * step
    d[(d < DEPTH_CFG["min_range"]) | (d > DEPTH_CFG["max_range"])] = 0.0
    d[rr.random(d.shape) < DEPTH_CFG["dropout_prob"]] = 0.0
    expected = np.clip(d, 0.0, None)

    assert np.array_equal(out, expected)


def test_apply_depth_sensor_deterministic_with_fixed_rng():
    depth = fixed_depth()
    out1 = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.7, np.random.default_rng(123))
    out2 = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.7, np.random.default_rng(123))
    assert np.array_equal(out1, out2)
    # light coupling actually changes the output at nominal vs stressed exposure
    nominal = effects.apply_depth_sensor(depth, DEPTH_CFG, 1.0, np.random.default_rng(123))
    assert not np.array_equal(out1, nominal)


def test_per_frame_rng_stream_reproducible():
    # Engine keys the per-frame stream as default_rng([seed, ms]) — the same key
    # must give bit-identical depth faults regardless of the window schedule.
    depth = fixed_depth()
    seed, ms = 42, 1234
    f1 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms]))
    f2 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms]))
    assert np.array_equal(f1, f2)
    f3 = effects.apply_depth_faults(depth, DEPTH_CFG, np.random.default_rng([seed, ms + 1]))
    assert not np.array_equal(f1, f3)


def test_depth_faults_disabled_or_stuck_returns_zeros():
    depth = fixed_depth()
    rng = np.random.default_rng(0)
    assert np.array_equal(effects.apply_depth_faults(depth, {**DEPTH_CFG, "enabled": False}, rng),
                          np.zeros_like(depth))
    assert np.array_equal(effects.apply_depth_faults(depth, {**DEPTH_CFG, "stuck": True}, rng),
                          np.zeros_like(depth))


def test_depth_to_vis():
    depth = np.array([[0.0, 2.5], [5.0, 10.0]], dtype=np.float32)
    vis = effects.depth_to_vis(depth, 5.0)
    assert vis.shape == (2, 2, 3)
    assert vis.dtype == np.uint8
    assert np.array_equal(vis[:, :, 0], np.array([[0, 127], [255, 255]], dtype=np.uint8))
    assert np.array_equal(vis[:, :, 0], vis[:, :, 1])
    assert np.array_equal(vis[:, :, 0], vis[:, :, 2])


def test_no_global_np_random_in_module():
    src = _EFFECTS_PATH.read_text()
    for banned in ("np.random.normal", "np.random.random", "np.random.seed",
                   "np.random.uniform", "np.random.choice"):
        assert banned not in src, f"global RNG call {banned} must not appear in effects.py"
