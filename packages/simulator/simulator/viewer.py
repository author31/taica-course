"""pygame preview panels. The ONLY module in the package that imports pygame.

SDL must be forced to a pure-software X11 window BEFORE `import pygame`, or
pygame's GLX context collides with habitat's GL and crashes with
`X Error ... X_GLXMakeCurrent BadAccess` (see hw1 root_cause_analysis). Callers
must also construct Engine BEFORE initializing the viewer window, and must never
import pygame directly — always through this module.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")

import pygame  # noqa: E402  (env vars above must precede this import)

import numpy as np  # noqa: E402
import cv2  # noqa: E402


def build_canvas(frame, display_cfg):
    """Stack the enabled panels (first-person RGB + depth + bird's-eye)
    horizontally into one (H, W, 3) RGB image."""
    panels = []
    if display_cfg.get("show_rgb", True):
        panels.append(frame["rgb"])
    if display_cfg.get("show_depth", True):
        panels.append(frame["depth_vis"])
    if display_cfg.get("show_birdseye", True):
        panels.append(frame["birdseye"])
    if not panels:
        panels.append(frame["rgb"])
    # Panels may have different resolutions; match heights before hstacking.
    h = panels[0].shape[0]
    panels = [
        p if p.shape[0] == h
        else cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h))
        for p in panels
    ]
    return np.concatenate(panels, axis=1)


def draw_counter(screen, count, font):
    """Hover text overlay (top-left) showing how many frames were captured."""
    label = font.render(f"Captured frames: {count}", True, (255, 255, 0))
    pad = 6
    bg = pygame.Surface((label.get_width() + 2 * pad, label.get_height() + 2 * pad))
    bg.set_alpha(140)
    bg.fill((0, 0, 0))
    screen.blit(bg, (8, 8))
    screen.blit(label, (8 + pad, 8 + pad))


def draw(screen, frame, display_cfg, count, font):
    canvas = build_canvas(frame, display_cfg)
    # pygame surfaces are (W, H, 3); our arrays are (H, W, 3) -> swap axes 0/1.
    surface = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
    scale = float(display_cfg["scale"])
    if scale != 1.0:
        w, h = surface.get_size()
        surface = pygame.transform.scale(surface, (int(w * scale), int(h * scale)))
    screen.blit(surface, (0, 0))
    draw_counter(screen, count, font)
    pygame.display.flip()
