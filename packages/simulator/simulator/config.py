"""Config loading: yaml + schema defaults merge.

NOTE (contract): no defaults merge exists in the legacy code (hw1/load.py:149 is
a bare yaml.safe_load) — the defaults table here is NEW, covering the
`uncertainties` block and the lighting/depth keys the legacy pipeline accessed
unchecked. See plan.md "Interface contract".
"""

import copy

import yaml

# Every key the pixel pipeline reads gets a default here. The legacy code
# (hw1/load.py:268-357) indexed most lighting/depth keys UNCHECKED
# (`cfg["brightness"]`, `cfg["noise_std"]`, ...), so a sparse yaml crashed with
# KeyError deep inside the frame loop. With this table a config only needs to
# state what it changes; file values always win over defaults (deep merge).
#
# Defaults are chosen to be inert: neutral lighting, fault-free depth, zero
# light-coupling gains — merging them into a config never changes behavior the
# config already specifies.
_DEFAULTS = {
    "lighting": {
        "brightness": 1.0,        # exposure gain (1.0 = neutral)
        "contrast": 1.0,          # contrast around mid-grey
        "gamma": 1.0,             # 1.0 = no gamma correction
        "ambient_rgb": [1.0, 1.0, 1.0],  # colour tint / temperature
        "amplitude": 0.0,         # flicker amplitude (0 = steady)
        "frequency": 0.0,         # flicker Hz
        "phase": 0.0,             # flicker phase (rad)
    },
    "depth": {
        "enabled": True,
        "stuck": False,           # dead sensor -> all zeros
        "noise_std": 0.0,         # gaussian noise sigma (m)
        "quantization": 0.0,      # step size (m); 0 = off
        "min_range": 0.0,         # readings below return 0
        "max_range": 10.0,        # readings above return 0
        "dropout_prob": 0.0,      # random per-pixel dropout
        "redwood": False,         # habitat in-sim Redwood noise model
        "redwood_multiplier": 1.0,
        # --- ambient-light coupling (stress = |exposure - light_nominal|) ---
        "light_nominal": 1.0,
        "light_noise_gain": 0.0,
        "light_dropout_gain": 0.0,
        "light_range_gain": 0.0,
    },
    # Temporal-window uncertainty injection (plan.md schema). A seeded scheduler
    # draws gap -> window -> gap ... forever; severity is fixed per-type here,
    # never sampled. `enabled: false` (evaluate's baseline run) skips the
    # scheduler entirely.
    "uncertainties": {
        "enabled": True,
        "seed": 42,               # governs BOTH window sampling and per-frame depth RNG
        "gap_s": [8.0, 25.0],     # clean time between windows, uniform per gap
        "duration_s": [1.0, 4.0],  # window length, uniform per window
        "types": {
            "flicker": {"amplitude": 0.9, "frequency": 6.0},
            "low_light": {"brightness": 0.3, "gamma": 1.4},
            "over_exposure": {"brightness": 2.2, "contrast": 1.2},
        },
    },
}


def _deep_merge(base, override):
    """Recursively merge `override` into `base` in place; override wins.

    Nested dicts merge key-by-key (so a config may state only the keys it
    changes, e.g. one severity knob under uncertainties.types.flicker); any
    non-dict value replaces the default wholesale (lists included — ranges like
    gap_s are atomic, not element-merged)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    """Load a YAML config and merge schema defaults. Returns a dict.

    File values win over defaults; defaults only fill gaps (deep merge, see
    _DEFAULTS). Blocks with no defaults (scene, agent, camera, ...) pass
    through untouched."""
    with open(path, "r") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(copy.deepcopy(_DEFAULTS), loaded)
