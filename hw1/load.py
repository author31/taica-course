"""Interactive Habitat-Sim data collector for hw1 — thin driver over packages/simulator.

All simulation, pixel-pipeline, replay, and viewer logic lives in the `simulator`
package (packages/simulator; contract in plan.md). This file only parses the CLI,
wires the pieces together, and runs the pygame event loop.

HOW TO RUN
    Interactive collection (pygame window, keyboard-driven):
        pixi run -e habitat python hw1/load.py
    Trajectory replay preview (exact .npy pose replay; frames are saved through
    the same pipeline):
        pixi run -e habitat python hw1/load.py --trajectory trajectories/secondfloor.npy
    Config defaults to hw1/configs/second_floor.yaml (--config to override);
    --output-root overrides output.root; --fps paces the preview loop.

KEYBINDINGS (interactive; the pygame window must have focus)
    w / s  move forward / backward      c / SPACE  capture frame
    a / d  turn left / right            q / ESC    quit (aborts replay too)

UNCERTAINTIES ARE LIVE during interactive collection: a seeded scheduler (config
`uncertainties`) fires temporal effect windows (flicker / low_light /
over_exposure) in real time — t is wall-clock seconds since session start — so
captures inherit whatever effect is active. Replay mode instead uses the
deterministic t = frame_index / fps. `--clean` (or `uncertainties.enabled:
false` in the config) disables the scheduler for uncorrupted collection. When
the scheduler is active, the realized windows are saved to
<output.root>/windows.json (window ground truth, outside the ontology store).

OUTPUTS (under output.root)
    rgb/<n>.png  depth/<n>.png  [semantic/<n>.png]  per capture, plus
    GT_pose.npy: (N, 7) captured poses [x, y, z, qw, qx, qy, qz].

GL ORDERING — DO NOT REORDER (condensed; details in simulator.engine / viewer)
    habitat-sim and pygame both want a GL context on the same X display and
    crash with `X_GLXMakeCurrent BadAccess` if they share it. Engine hides
    DISPLAY during Simulator construction (habitat renders offscreen on EGL);
    simulator.viewer forces SDL software rendering via env vars at ITS import,
    before pygame. Therefore Engine is constructed FIRST, `simulator.viewer` is
    imported LAZILY after that, and this file never imports pygame directly.
"""

import argparse
import os
import shutil
import time

import numpy as np

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "second_floor.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="hw1 data collector — thin driver over packages/simulator")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--trajectory", default=None,
                        help="optional .npy (N,7) pose trajectory to replay-preview "
                             "instead of interactive keyboard collection")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="preview frame rate (0 = as fast as possible)")
    parser.add_argument("--output-root", default=None,
                        help="override output.root from the config")
    parser.add_argument("--clean", action="store_true",
                        help="disable the uncertainty scheduler (uncorrupted collection)")
    return parser.parse_args()


def prepare_output_dirs(data_root, out_cfg):
    """Create output.root and the capture subdirs the config saves into."""
    if out_cfg.get("clear_existing", False) and os.path.isdir(data_root):
        shutil.rmtree(data_root)  # WARNING: deletes the whole directory
    subs = ["rgb", "depth"]
    if out_cfg.get("save_semantic", False):
        subs.append("semantic")
    for sub in subs:
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)


def run_replay(engine, viewer, screen, font, traj_path, fps, data_root, out_cfg,
               display_cfg):
    """Replay a .npy pose trajectory, previewing AND saving every frame.

    Deterministic time base (t = i / engine.fps_nominal) lives inside
    replay_poses; this callback only does output I/O, preview, abort, pacing."""
    from simulator import load_trajectory, replay_poses, save_frame

    pygame = viewer.pygame
    poses = load_trajectory(traj_path)
    delay_ms = int(1000.0 / fps) if fps and fps > 0 else 0
    print(f"replaying {len(poses)} poses — q / ESC to abort")

    def on_frame(frame, sensor_state, idx):
        save_frame(frame, sensor_state, data_root, out_cfg, idx)
        viewer.draw(screen, frame, display_cfg, idx, font)
        for event in pygame.event.get():          # let the user abort mid-replay
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                return False
        if delay_ms:
            pygame.time.wait(delay_ms)
        return True

    captured = replay_poses(engine, poses, on_frame)   # caller saves GT_pose.npy
    np.save(os.path.join(data_root, "GT_pose.npy"), captured)
    print(f"replay: saved {len(captured)} poses to "
          f"{os.path.join(data_root, 'GT_pose.npy')}")


def run_interactive(engine, viewer, screen, font, fps, data_root, out_cfg,
                    display_cfg):
    """Keyboard-driven collection; scheduler windows fire live on wall clock."""
    from simulator import save_frame

    pygame = viewer.pygame
    KEY_ACTION = {
        pygame.K_w: "move_forward",
        pygame.K_s: "move_backward",
        pygame.K_a: "turn_left",
        pygame.K_d: "turn_right",
    }
    CAPTURE_KEYS = (pygame.K_c, pygame.K_SPACE)
    QUIT_KEYS = (pygame.K_q, pygame.K_ESCAPE)

    print("#############################")
    print("use the keyboard to control the agent")
    print("  w / s : forward / backward")
    print("  a / d : turn left / right")
    print("  c or SPACE : capture the current frame")
    print("  q or ESC   : finish and quit")
    print("#############################")

    cam_extr = []   # captured GT poses [x, y, z, qw, qx, qy, qz]
    count = 0
    clock = pygame.time.Clock()
    t0 = time.monotonic()   # session start: uncertainties run on wall-clock delta

    def observe_now():
        """Re-render the CURRENT sensors at the CURRENT wall-clock t, so
        time-based effects (scheduler windows, flicker) advance continuously
        even while the agent is standing still."""
        frame = engine.observe(time.monotonic() - t0)
        sensor_state = engine.agent.get_state().sensor_states["color_sensor"]
        return frame, sensor_state

    frame, sensor_state = observe_now()
    running = True
    while running:
        for event in pygame.event.get():   # non-blocking: keep rendering between inputs
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in QUIT_KEYS:
                    running = False
                elif event.key in CAPTURE_KEYS:
                    count += 1
                    pose = save_frame(frame, sensor_state, data_root, out_cfg, count)
                    cam_extr.append(pose)
                    print(f"captured frame {count} @ pose "
                          f"({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) "
                          f"({pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f})")
                elif event.key in KEY_ACTION:
                    engine.step(KEY_ACTION[event.key])
                # any other key is ignored
        frame, sensor_state = observe_now()   # continuous time-driven render
        viewer.draw(screen, frame, display_cfg, count, font)
        clock.tick(fps)

    np.save(os.path.join(data_root, "GT_pose.npy"),
            np.asarray(cam_extr, dtype=np.float32))
    print(f"saved {len(cam_extr)} poses to {os.path.join(data_root, 'GT_pose.npy')}")


def main():
    args = parse_args()   # argparse first: --help never touches habitat or pygame

    from simulator import Engine, UncertaintyScheduler, load_config

    config = load_config(args.config)
    out_cfg = config["output"]
    if args.output_root:
        out_cfg["root"] = args.output_root
    data_root = out_cfg["root"]
    prepare_output_dirs(data_root, out_cfg)

    unc = config["uncertainties"]
    scheduler = None
    if args.clean or not unc.get("enabled", True):
        print("uncertainty scheduler disabled (clean collection)")
    else:
        scheduler = UncertaintyScheduler(unc)

    fps = args.fps if args.fps and args.fps > 0 else 30.0
    engine = Engine(config, scheduler=scheduler, fps_nominal=fps)

    # Engine constructed -> only now may the viewer set SDL vars + import pygame.
    from simulator import viewer
    pygame = viewer.pygame

    pygame.init()
    font = pygame.font.SysFont(None, 28)
    display_cfg = config["display"]
    # Size the window from a real first frame (true start_position, t=0).
    canvas = viewer.build_canvas(engine.observe(0.0), display_cfg)
    scale = float(display_cfg["scale"])
    screen = pygame.display.set_mode(
        (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))
    pygame.display.set_caption("Habitat data collector")

    try:
        if args.trajectory:
            run_replay(engine, viewer, screen, font, args.trajectory, args.fps,
                       data_root, out_cfg, display_cfg)
        else:
            run_interactive(engine, viewer, screen, font, fps, data_root,
                            out_cfg, display_cfg)
        if scheduler is not None:
            scheduler.save(os.path.join(data_root, "windows.json"))
            print(f"{len(scheduler.windows)} uncertainty windows -> "
                  f"{os.path.join(data_root, 'windows.json')}")
    finally:
        pygame.quit()
        engine.close()


if __name__ == "__main__":
    main()
