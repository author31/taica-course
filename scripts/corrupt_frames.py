"""One-off: inject MILD sensor artifacts to nudge reconstruct.py off, not break it.

Goal: make scripts/reconstruct.py's Mean L2 a little WORSE, without fatal
crashes or the SLAM front-end collapsing. So every perturbation is small and
bounded, and depth stays inside its valid metric range.

Modalities (each its own physics):
  - rgb   (uint8, photometric): mild exposure gain + contrast. Only degrades the
           COLOR term of colored-ICP; geometry still carries alignment, so effect
           is small by design.
  - depth (uint16 mm, or uint8 range-preview -> both LINEAR in meters): per-frame
           multiplicative scale jitter + small Gaussian noise. Jitter makes each
           frame slightly non-metric and INCONSISTENT with its neighbors, so
           colored-ICP odometry drifts a little every step -> L2 grows. Kept
           small (few %) so the physical gate (_MAX_STEP_T/R) never trips and
           points stay < DEPTH_MAX_RANGE (still valid, not dropped).

Zero-depth pixels (no return) are preserved as zero. Overwrites in place.
Set SEED for reproducibility.
"""
import random
from pathlib import Path

import cv2
import numpy as np

BASE = Path("sample_data_collection/first_floor")
MODALITIES = ["rgb", "depth"]
FRACTION = 1.0                 # apply across all frames for steady, gentle drift
SEED = 42

# --- RGB (photometric, mild) ---
RGB_CONTRAST = 1.2             # >1 = higher contrast
RGB_EXPOSURE = (0.75, 1.25)    # gain sweep across the window (dark .. bright)

# --- depth (metric, mild) ---
DEPTH_SCALE = 1000.0           # uint16 mm -> m  (must match reconstruct.py)
DEPTH_MAX_RANGE = 10.0         # keep depth strictly below this (stays valid)
DEPTH_SCALE_JITTER = 0.03      # per-frame multiplicative bias, +/- this fraction
DEPTH_NOISE_FRAC = 0.015       # per-pixel Gaussian noise, fraction of depth value


def list_frames(d: Path):
    frames = [p for p in d.glob("*.png") if p.stem.isdigit()]
    return sorted(frames, key=lambda p: int(p.stem))


def corrupt_rgb(img: np.ndarray, gain: float) -> np.ndarray:
    """Mild exposure + contrast around mid-gray. Photometric only."""
    info = np.iinfo(img.dtype)
    hi = float(info.max)
    f = img.astype(np.float32) / hi
    f *= gain
    f = (f - 0.5) * RGB_CONTRAST + 0.5
    f = np.clip(f, 0.0, 1.0)
    return (f * hi).round().astype(img.dtype)


def corrupt_depth(img: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Per-frame scale bias + small Gaussian noise. Both uint16 (mm) and uint8
    (range-preview) encode depth LINEARLY, so a linear perturb is metric-correct.
    Zeros (no-return) stay zero; result clipped to stay a valid in-range depth."""
    info = np.iinfo(img.dtype)
    hi = float(info.max)
    valid = img > 0
    f = img.astype(np.float32)
    f[valid] *= scale
    f[valid] += rng.normal(0.0, DEPTH_NOISE_FRAC, size=valid.sum()) * f[valid]

    # Ceiling: keep strictly below DEPTH_MAX_RANGE so points aren't dropped.
    if img.dtype == np.uint16:
        cap = DEPTH_MAX_RANGE * DEPTH_SCALE - 1.0        # mm
    else:
        cap = hi - 1.0                                   # 255 preview == max range
    f[valid] = np.clip(f[valid], 1.0, cap)
    f[~valid] = 0.0
    return f.round().astype(img.dtype)


def main():
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    common = set.intersection(*(
        {p.name for p in list_frames(BASE / m)} for m in MODALITIES
    ))
    ref = sorted(common, key=lambda s: int(Path(s).stem))
    n = max(1, round(len(ref) * FRACTION))
    if len(ref) < n:
        raise SystemExit(f"need >= {n} frames, found {len(ref)}")
    start = rng.randint(0, len(ref) - n)
    names = ref[start:start + n]

    gains = np.linspace(RGB_EXPOSURE[0], RGB_EXPOSURE[1], n)
    # Per-frame depth scale: independent jitter around 1.0 -> neighbor-inconsistent.
    depth_scales = 1.0 + nprng.uniform(-DEPTH_SCALE_JITTER, DEPTH_SCALE_JITTER, n)

    print(f"corrupting {n} frames ({FRACTION:.0%}): {names[0]}..{names[-1]}")
    for mod in MODALITIES:
        d = BASE / mod
        for i, name in enumerate(names):
            p = d / name
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  skip missing {p}")
                continue
            if mod == "depth":
                out = corrupt_depth(img, float(depth_scales[i]), nprng)
            else:
                out = corrupt_rgb(img, float(gains[i]))
            cv2.imwrite(str(p), out)
        print(f"  {mod}: done")


if __name__ == "__main__":
    main()
