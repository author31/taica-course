"""Pixel pipeline: lighting emulation, depth-sensor faults, spatial uncertainty.

Contract highlights (plan.md §3.1):
- All randomness takes an explicit numpy Generator (`rng`) — NO global np.random.
- ZoneScheduler is SPATIAL, not temporal: the effect an agent sees depends on
  WHERE it is, not on when it got there. active(t, position) is a pure function
  of (position, t) — no seed, no RNG, no cursor, no realized-window list.
- Zones are circles with HARD edges ((x-cx)^2 + (z-cz)^2 <= r^2), first match
  wins by config list order, and flicker phase is pinned at 0 (the oscillation
  rides the global clock, so re-entering a zone never re-anchors it).
- `uncertainties.seed` is NOT read here: it governs only the per-frame depth
  RNG the Engine builds (see engine.Engine.observe).

This module is importable with numpy alone; PIL and habitat_sim are imported
lazily inside semantic_to_vis (the only function that needs them).
"""

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

    `overrides` (typically ZoneScheduler.active(t, position)) has the shape

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
# Spatial uncertainty: hard-edged circular flicker zones (plan.md §3.1)
# =============================================================================
OUTSIDE = "outside"   # reserved key/label for "in no zone at all"


def _parse_zones(zones):
    """Validate + normalize a config `zones:` list into an ordered list of dicts.

    Each entry: {"name": str, "center": (x, z), "radius": float,
                 "flicker": {"amplitude": float, "frequency": float}}.
    List ORDER IS PRIORITY (first match wins), so it is preserved exactly.

    Names must be unique and may not be the reserved label "outside" — the
    frames-per-zone counters key on them, and a duplicate would silently merge
    two zones into one row. A missing name defaults to "zone<i>"."""
    parsed = []
    seen = set()
    for i, raw in enumerate(zones or []):
        z = dict(raw or {})
        name = str(z.get("name", f"zone{i}"))
        if name == OUTSIDE:
            raise ValueError(
                f"zone {i}: {OUTSIDE!r} is reserved for the clean region outside "
                "every zone; pick another name")
        if name in seen:
            raise ValueError(f"duplicate zone name {name!r} (names must be unique)")
        seen.add(name)

        center = [float(v) for v in (z.get("center") or [])]
        if len(center) != 2:
            raise ValueError(
                f"zone {name!r}: center must be [x, z] in world metres "
                f"(y is implied by the floor), got {z.get('center')!r}")
        radius = float(z.get("radius", 0.0))
        if radius < 0.0:
            raise ValueError(f"zone {name!r}: radius must be >= 0, got {radius}")

        flicker = dict(z.get("flicker") or {})
        parsed.append({
            "name": name,
            "center": (center[0], center[1]),
            "radius": radius,
            "flicker": {"amplitude": float(flicker.get("amplitude", 0.0)),
                        "frequency": float(flicker.get("frequency", 0.0))},
        })
    return parsed


def _zone_index(zones, x, z):
    """Index of the FIRST zone (list order = priority) containing world point
    (x, z), else None. Hard circular edge, boundary INCLUSIVE:

        (x - cx)^2 + (z - cz)^2 <= r^2

    No falloff, no crossfade — the sharp position<->degradation correlation is
    the lesson, and a smooth edge would blur exactly the signal (D15)."""
    for i, zone in enumerate(zones):
        cx, cz = zone["center"]
        if (x - cx) ** 2 + (z - cz) ** 2 <= zone["radius"] ** 2:
            return i
    return None


def _uncertainty_cfg(config):
    """Accept either a full config dict or its `uncertainties` block."""
    cfg = config or {}
    inner = cfg.get("uncertainties")
    return (inner or {}) if isinstance(inner, dict) else cfg


def _poses_to_xz(poses):
    """(N,7) [x,y,z,qw,qx,qy,qz] or (N,3) [x,y,z] or (N,2) [x,z] -> (N,2) XZ."""
    arr = np.asarray(poses, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"poses must be 2-D (N, k), got shape {arr.shape}")
    if arr.shape[1] == 2:
        return arr                      # already XZ
    if arr.shape[1] >= 3:
        return arr[:, [0, 2]]           # [x, y, z, ...] -> XZ
    raise ValueError(
        f"poses must have >= 2 columns ([x,z] or [x,y,z,...]), got {arr.shape}")


class ZoneScheduler:
    """Spatial uncertainty: hard-edged circular flicker zones.

    Config keys (uncertainty_cfg — a full config or its `uncertainties` block):
        enabled  bool, default True — False makes active() always return {}
        mode     "spatial" (the only mode; anything else is a hard error)
        zones    ordered list, FIRST MATCH WINS:
                   {name, center: [x, z], radius, flicker: {amplitude, frequency}}
        seed     NOT read here. It governs ONLY the per-frame depth RNG
                 (engine.Engine.observe); this class is seedless.

    PURE, STATELESS, SEEDLESS. active() is a function of (position, t) alone:
    no RNG, no lazy unroll, no cursor, no realized-window list, nothing to
    persist (there is no windows.json any more). Calling it twice with the same
    arguments — in any order, on any instance built from the same config —
    returns equal overrides.

    `phase` is pinned at 0.0, so the oscillation is 1 + a*sin(2*pi*f*t) on the
    GLOBAL clock. Zone-entry anchoring is deliberately not carried over from the
    temporal scheduler: it would need entry-time state, and re-entering a zone
    would re-anchor it.

    .zones — the normalized zone list (see _parse_zones), in priority order."""

    def __init__(self, uncertainty_cfg):
        cfg = _uncertainty_cfg(uncertainty_cfg)
        mode = str(cfg.get("mode", "spatial"))
        if mode != "spatial":
            raise ValueError(
                f"uncertainties.mode must be 'spatial' (got {mode!r}) — the "
                "temporal window scheduler was removed in plan.md §3.1")
        self.enabled = bool(cfg.get("enabled", True))
        self.zones = _parse_zones(cfg.get("zones"))

    def active(self, t, position):
        """Param overrides at world position `position`; {} outside every zone.

        `position` is (x, z) in world metres — the AGENT position, not the
        sensor pose (they differ only in y; see engine.Engine.observe).
        Inside a zone the return value is a process_observations `overrides`
        dict: {"lighting": {"amplitude": a, "frequency": f, "phase": 0.0}}.

        `t` is accepted for interface symmetry and is deliberately UNUSED:
        membership is purely spatial and the flicker phase is pinned to the
        global clock, which apply_lighting / light_exposure read from their own
        `t` argument."""
        del t                                   # spatial only — see docstring
        if not self.enabled or not self.zones:
            return {}
        try:
            x, z = (float(v) for v in position)
        except (TypeError, ValueError):
            raise ValueError(
                f"position must be (x, z) world metres, got {position!r}") from None
        i = _zone_index(self.zones, x, z)
        if i is None:
            return {}
        zone = self.zones[i]
        # Fresh dict per call: callers merge it into config sections, and a
        # shared dict would leak mutations between frames.
        return {"lighting": {"amplitude": zone["flicker"]["amplitude"],
                             "frequency": zone["flicker"]["frequency"],
                             "phase": 0.0}}


def zone_frame_labels(config, poses):
    """Zone name per pose, None where the pose is outside every zone.

    `config` is a full config or its `uncertainties` block; `poses` is an (N,7)
    trajectory ([x,y,z,qw,qx,qy,qz], the GT_pose.npy layout), (N,3) [x,y,z], or
    (N,2) [x,z]. Only XZ is read — zone membership ignores the floor height and
    is identical for agent and sensor poses (they differ only in y).

    GEOMETRY ONLY: `uncertainties.enabled` is NOT consulted, so this answers
    "which zones does this trajectory pass through", not "was the effect
    switched on for that run"."""
    zones = _parse_zones(_uncertainty_cfg(config).get("zones"))
    xz = _poses_to_xz(poses)
    labels = []
    for x, z in xz:
        i = _zone_index(zones, float(x), float(z))
        labels.append(None if i is None else zones[i]["name"])
    return labels


def zone_frame_counts(config, poses):
    """Frames per zone + frames outside every zone, for a whole trajectory.

    Returns {zone_name: n_frames, ..., "outside": n_frames_in_no_zone} — every
    configured zone is present (0 if the trajectory misses it) and the counts
    sum to len(poses). Zone placement is validated with this (plan.md §3.1
    step 3): assert every zone is entered, that no zone swallows the episode,
    and that a meaningful number of frames stay outside — those clean frames
    are the within-capture control the whole analysis rests on. A zone the
    trajectory never enters silently degenerates the mixed run to baseline.

    Arguments are as for zone_frame_labels (geometry only)."""
    zones = _parse_zones(_uncertainty_cfg(config).get("zones"))
    counts = {zone["name"]: 0 for zone in zones}
    counts[OUTSIDE] = 0
    for label in zone_frame_labels(config, poses):
        counts[OUTSIDE if label is None else label] += 1
    return counts
