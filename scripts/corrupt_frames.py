"""One-off: corrupt 10 random consecutive RGB frames.

Applies two degradations to a random 10-frame window:
  - exposure ramp: low-light (dark) at window start -> over-exposure (bright) at end
  - high contrast

Overwrites the frames in place. Set SEED for reproducibility.
"""
import random
from pathlib import Path

import cv2
import numpy as np

RGB_DIR = Path("sample_data_collection/first_floor/rgb")
N = 10
SEED = 42
CONTRAST = 1.8          # >1 = higher contrast
EXPOSURE_RANGE = (0.25, 3.0)  # gain: dark -> bright across the window


def list_frames(d: Path):
    # numeric-named .png frames, sorted by frame index
    frames = [p for p in d.glob("*.png") if p.stem.isdigit()]
    return sorted(frames, key=lambda p: int(p.stem))


def apply(img: np.ndarray, gain: float) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    # exposure gain (low-light .. over-exposure)
    f *= gain
    # high contrast around mid-gray
    f = (f - 0.5) * CONTRAST + 0.5
    f = np.clip(f, 0.0, 1.0)
    return (f * 255.0).round().astype(np.uint8)


def main():
    rng = random.Random(SEED)
    frames = list_frames(RGB_DIR)
    if len(frames) < N:
        raise SystemExit(f"need >= {N} frames, found {len(frames)}")

    start = rng.randint(0, len(frames) - N)
    window = frames[start:start + N]
    gains = np.linspace(EXPOSURE_RANGE[0], EXPOSURE_RANGE[1], N)

    print(f"corrupting frames {window[0].name}..{window[-1].name}")
    for p, g in zip(window, gains):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"failed to read {p}")
        cv2.imwrite(str(p), apply(img, float(g)))
        print(f"  {p.name}: gain={g:.2f}")


if __name__ == "__main__":
    main()
