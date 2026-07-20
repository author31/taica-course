"""Pixel pipeline: lighting emulation, depth-sensor faults, uncertainty scheduling.

Contract highlights (plan.md):
- All randomness takes an explicit numpy Generator (`rng`) — NO global np.random.
- UncertaintyScheduler is a TIME-based (seconds) lazy deterministic stream:
  draws (gap_s, duration_s, effect_type) forever from its seed, unrolling only
  as far as the largest `t` queried. No episode horizon.
- Flicker phase inside a window is window-relative:
  phase = -2*pi*frequency*window_start_s.

This module is importable with numpy alone; PIL and habitat_sim are imported
lazily inside semantic_to_vis (the only function that needs them).
"""

import bisect
import json
import math

import numpy as np


# =============================================================================
# Sensor post-processing (this is where "real-world uncertainty" is injected)
# =============================================================================
def apply_lighting(rgb, cfg, t=0.0):
    """Photometric emulation of lighting conditions on an RGB (H,W,3) uint8 image.

    `t` is a time in seconds used to flicker brightness periodically:
        brightness *= 1 + amplitude * sin(2*pi*frequency*t + phase)
    amplitude 0 (default) leaves brightness steady. In interactive mode `t` is
    wall-clock; in replay it is derived from the frame index (i / fps_nominal) so
    flicker is reproducible across runs/machines.
    """
    img = rgb[:, :, :3].astype(np.float32) / 255.0
    img *= np.asarray(cfg["ambient_rgb"], dtype=np.float32)   # colour tint / temperature
    amplitude = float(cfg.get("amplitude", 0.0))
    osc = 1.0
    if amplitude != 0.0:
        osc = 1.0 + amplitude * np.sin(
            2.0 * np.pi * float(cfg.get("frequency", 0.0)) * t + float(cfg.get("phase", 0.0)))
    img *= float(cfg["brightness"]) * osc                      # exposure gain (flicker)
    img = (img - 0.5) * float(cfg["contrast"]) + 0.5           # contrast around mid-grey
    img = np.clip(img, 0.0, 1.0)
    gamma = float(cfg["gamma"])
    if gamma != 1.0:
        img = img ** (1.0 / gamma)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def light_exposure(cfg, t=0.0):
    """Current photometric exposure gain = brightness * flicker osc, matching the
    factor apply_lighting applies to RGB. Used to couple the depth sensor to
    scene light (see apply_depth_sensor)."""
    amplitude = float(cfg.get("amplitude", 0.0))
    osc = 1.0
    if amplitude != 0.0:
        osc = 1.0 + amplitude * np.sin(
            2.0 * np.pi * float(cfg.get("frequency", 0.0)) * t + float(cfg.get("phase", 0.0)))
    return float(cfg["brightness"]) * osc


def apply_depth_faults(depth_m, cfg, rng):
    """Emulate depth-sensor faults on a raw depth map (meters, float32).

    `rng` is an explicit np.random.Generator — every random draw (noise, dropout)
    comes from it, so a per-frame stream (default_rng([seed, ms])) makes faults
    bit-reproducible and independent of any global RNG state."""
    if not cfg.get("enabled", True) or cfg.get("stuck", False):
        return np.zeros_like(depth_m)

    d = depth_m.astype(np.float32).copy()

    if float(cfg["noise_std"]) > 0.0:
        d += rng.normal(0.0, float(cfg["noise_std"]), size=d.shape).astype(np.float32)

    if float(cfg["quantization"]) > 0.0:
        step = float(cfg["quantization"])
        d = np.round(d / step) * step

    # Out-of-range readings return nothing (0).
    out_of_range = (d < float(cfg["min_range"])) | (d > float(cfg["max_range"]))
    d[out_of_range] = 0.0

    if float(cfg["dropout_prob"]) > 0.0:
        drop = rng.random(d.shape) < float(cfg["dropout_prob"])
        d[drop] = 0.0

    return np.clip(d, 0.0, None)


def apply_depth_sensor(depth_m, cfg, light, rng):
    """Depth-sensor emulation WITH ambient-light coupling.

    Habitat renders light-independent geometric depth, but a real structured-
    light / ToF sensor degrades as scene light drives its emitter SNR down:
    brighter-or-darker-than-nominal exposure -> more noise, more dropout, shorter
    usable range. This models that coupling, then hands off to apply_depth_faults
    for the actual fault injection.

        stress       = |light - light_nominal|
        noise_std   *= 1 + light_noise_gain   * stress
        dropout_prob += light_dropout_gain    * stress   (clamped <=1)
        max_range   *= 1 - light_range_gain   * stress   (clamped >=0)

    `light` is the current exposure factor (see light_exposure); `rng` is the
    explicit Generator threaded down to apply_depth_faults. All gains 0 =>
    depth is light-independent, identical to plain apply_depth_faults."""
    if not cfg.get("enabled", True) or cfg.get("stuck", False):
        return apply_depth_faults(depth_m, cfg, rng)   # dead/off sensor: light irrelevant

    stress = abs(float(light) - float(cfg.get("light_nominal", 1.0)))
    stressed = dict(cfg)
    stressed["noise_std"] = float(cfg["noise_std"]) * (
        1.0 + float(cfg.get("light_noise_gain", 0.0)) * stress)
    stressed["dropout_prob"] = min(1.0, float(cfg["dropout_prob"]) + (
        float(cfg.get("light_dropout_gain", 0.0)) * stress))
    stressed["max_range"] = float(cfg["max_range"]) * max(0.0, (
        1.0 - float(cfg.get("light_range_gain", 0.0)) * stress))
    return apply_depth_faults(depth_m, stressed, rng)


# --- visualisation helpers (all return RGB uint8 for pygame; BGR is only for cv2 saves) ---
def depth_to_vis(depth_m, max_range):
    d = np.clip(depth_m / max(max_range, 1e-6), 0.0, 1.0)
    gray = (d * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)  # (H,W,3) RGB


def semantic_to_vis(semantic_obs):
    # Lazy imports: keep this module importable (and the rest of the pixel
    # pipeline testable) without habitat_sim / PIL installed.
    from PIL import Image
    from habitat_sim.utils.common import d3_40_colors_rgb

    img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    img.putpalette(d3_40_colors_rgb.flatten())
    img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    return np.asarray(img.convert("RGB"))  # (H,W,3) RGB


# =============================================================================
# Processed frame: apply the config to raw observations once, reuse for
# both display and saving so the preview matches what gets written.
# =============================================================================
def process_observations(obs, config, t=0.0, rng=None, overrides=None):
    """Raw obs -> processed frame dict (rgb / birdseye / depth_m / depth_vis / semantic).

    `overrides` (typically UncertaintyScheduler.active(t)) has the shape

        {"lighting": {<lighting-key>: value, ...},
         "depth":    {<depth-key>:    value, ...}}

    with both sections optional. Each present section is merged over the
    corresponding config section (override entries win) for THIS frame only —
    `config` is never mutated. Section values are flat key->scalar/list maps,
    so the per-section merge is the deep merge.

    `rng` is the per-frame Generator for depth faults; if None it defaults to
    np.random.default_rng(0) so legacy no-rng calls stay deterministic.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lighting_cfg = config["lighting"]
    depth_cfg = config["depth"]
    if overrides:
        if overrides.get("lighting"):
            lighting_cfg = {**lighting_cfg, **overrides["lighting"]}
        if overrides.get("depth"):
            depth_cfg = {**depth_cfg, **overrides["depth"]}
    light = light_exposure(lighting_cfg, t)                     # shared by RGB + depth
    depth_m = apply_depth_sensor(obs["depth_sensor"], depth_cfg, light, rng)
    return {
        "rgb": apply_lighting(obs["color_sensor"], lighting_cfg, t),  # RGB uint8
        "birdseye": obs["birdseye_sensor"][:, :, :3],                 # top-down RGB uint8
        "depth_m": depth_m,
        "depth_vis": depth_to_vis(depth_m, float(depth_cfg["max_range"])),
        "semantic": semantic_to_vis(obs["semantic_sensor"]),          # RGB uint8
    }


# =============================================================================
# Uncertainty scheduler: seeded temporal windows of effect injection
# =============================================================================
class UncertaintyScheduler:
    """Seeded, endless, deterministic temporal-window stream (seconds).

    Config keys (uncertainty_cfg):
        enabled     bool, default True — False makes active() always return {}
        seed        int  — seeds the window-sampling Generator
        gap_s       [lo, hi] clean seconds between windows, sampled uniform
        duration_s  [lo, hi] window length in seconds, sampled uniform
        types       {name: params_dict} fixed per-type severity (not sampled)

    Stream (starting from t=0 with a gap first):
        gap ~ U(gap_s) -> window start; duration ~ U(duration_s);
        effect type ~ uniform choice over `types` keys; repeat forever.

    Lazy unroll: active(t) extends the realized window list only until it
    covers `t`; queries at already-covered times are pure lookups, so
    non-monotonic `t` is fine and the realized timeline depends ONLY on the
    seed — never on the sequence or granularity of active() calls.

    .windows — list of (start_s, end_s, effect_type, params_dict) realized so far.
    """

    def __init__(self, uncertainty_cfg):
        cfg = uncertainty_cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._gap_lo, self._gap_hi = (float(x) for x in cfg.get("gap_s", [8.0, 25.0]))
        self._dur_lo, self._dur_hi = (float(x) for x in cfg.get("duration_s", [1.0, 4.0]))
        self._types = {name: dict(params or {})
                       for name, params in dict(cfg.get("types", {})).items()}
        self._type_names = list(self._types.keys())
        self._rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.windows = []       # (start_s, end_s, effect_type, params_dict) realized so far
        self._starts = []       # parallel start_s list for bisect lookup
        self._cursor = 0.0      # end of the realized stream; timeline begins with a gap

    def _unroll_to(self, t):
        """Realize windows until the stream covers time `t` (cursor > t)."""
        while self._cursor <= t:
            gap = self._rng.uniform(self._gap_lo, self._gap_hi)
            duration = self._rng.uniform(self._dur_lo, self._dur_hi)
            name = self._type_names[int(self._rng.integers(len(self._type_names)))]
            start = self._cursor + gap
            end = start + duration
            self.windows.append((start, end, name, dict(self._types[name])))
            self._starts.append(start)
            self._cursor = end

    def active(self, t):
        """Param overrides for time `t`; {} outside windows (baseline).

        Inside a window the return value is a process_observations `overrides`
        dict: flicker -> {"lighting": {amplitude, frequency,
        phase=-2*pi*frequency*start_s}} (window-relative phase, so the window
        always starts at the same point of the sine); low_light /
        over_exposure -> {"lighting": {**type params}}.
        Windows are half-open: start_s <= t < end_s.
        """
        if not self.enabled or not self._type_names:
            return {}
        t = float(t)
        self._unroll_to(t)
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            return {}
        start, end, name, params = self.windows[i]
        if t >= end:
            return {}
        lighting = dict(params)
        if name == "flicker":
            lighting["phase"] = -2.0 * math.pi * float(params.get("frequency", 0.0)) * start
        return {"lighting": lighting}

    def save(self, path):
        """Write windows.json: [{"start_s":..., "end_s":..., "type":..., "params": {...}}].

        Saves the windows realized (unrolled) SO FAR — call it after the
        episode so the timeline covers everything that was observed. Times are
        seconds; this is window ground truth kept outside the ontology store.
        """
        payload = [
            {"start_s": start, "end_s": end, "type": name, "params": params}
            for start, end, name, params in self.windows
        ]
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
