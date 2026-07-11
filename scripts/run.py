import os
import json
import argparse
import shutil

# SDL must render the pygame window in pure software. By default pygame/SDL
# creates a GLX-accelerated window surface, which collides with habitat-sim's
# OpenGL context on the same X display and crashes with
# `X Error ... X_GLXMakeCurrent BadAccess`. Forcing the software X11 path keeps
# pygame off GLX entirely (habitat owns GL). Must be set before `import pygame`.
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")
os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")

import numpy as np
import yaml
import cv2
import pygame
from PIL import Image

from scipy.spatial.transform import Rotation as Rot

import magnum as mn
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb, quat_from_coeffs
from timeit import default_timer as timer


def apply_sin_lighting(rgb, amplitude, frequency, ts, phase):
    scale = 1 + amplitude * np.sin(2 * np.pi * frequency * ts + phase)
    return np.clip(rgb * scale, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
    init_time = timer()
    frames_displayed = 0

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        canvas = np.zeros((screen.get_height(), screen.get_width(), 3), dtype=np.uint8)
        canvas[:] = [128, 0, 128]
        canvas = apply_sin_lighting(canvas, 0.9, 0.1, frames_displayed / 100, 0)

        surface = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        pygame.draw.circle(screen, "red", player_pos, 40)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_pos.y -= 300 * dt
        if keys[pygame.K_s]:
            player_pos.y += 300 * dt
        if keys[pygame.K_a]:
            player_pos.x -= 300 * dt
        if keys[pygame.K_d]:
            player_pos.x += 300 * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
        frames_displayed+=1

    print("Average frame rate:", frames_displayed/(timer()-init_time), "fps")
    pygame.quit()

