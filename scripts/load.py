"""
Interactive Habitat-Sim data collector.

================================================================================
CLAUDE.md  (embedded)  —  orientation for anyone (human or agent) editing this file
================================================================================

WHAT THIS IS
    An interactive Habitat-Sim data collector for the course assignment. The
    goal is not only to let students implement an algorithm, but to make them
    confront real-world *environment uncertainty*: lighting, camera geometry,
    and depth-sensor faults are all tunable so the algorithm must stay robust.

    Two design pillars:
      1. Environment configurable — every environment condition lives in a
         single YAML file (scripts/config.yaml); no code edit needed to retune.
      2. Interactivity — navigation runs in a pygame window driven by the
         keyboard, and the student chooses which frames to capture.

    This is a POC and is intentionally kept in ONE file so it can be read
    top-to-bottom in a single pass. Prefer clarity over cleverness when editing.

HOW TO RUN
    pixi run -e habitat python scripts/load.py [--config scripts/config.yaml]

    Requires the pixi `habitat` environment (Python 3.9, habitat-sim 0.3.3,
    pygame, pyyaml) and the Replica scene at the path in config.yaml
    (`pixi run fetch-replica` downloads apartment_0).

KEYBINDINGS  (the pygame window must have focus)
    w  move forward       a  turn left        c / SPACE  capture frame
    s  move backward      d  turn right       q / ESC    quit

    Movement only refreshes the live preview. Capture is DECOUPLED from
    movement: only captured frames are written to disk, so the student chooses
    the trajectory that gets recorded.

CONFIGURATION  (scripts/config.yaml — the whole point of the assignment)
    scene     : mesh path.
    agent     : spawn position + per-key actuation amounts (m / deg).
    camera    : intrinsics (width, height, hfov) + extrinsics (position,
                orientation = pitch/yaw/roll in radians).
    birdseye  : a second top-down camera (own intrinsics/extrinsics) shown as
                the bird's-eye preview panel; preview only, not saved.
    marker    : a box spawned in the world at agent.start_position, visible in
                BOTH views so the two viewpoints can be correlated (enabling it
                turns on habitat physics).
    lighting  : photometric emulation applied to the RGB frame
                (brightness / contrast / gamma / ambient tint). NOTE this is a
                post-process, not an in-sim light setup — the Replica mesh is
                flat/vertex-shaded so real sim lights largely no-op; photometric
                reliably demonstrates lighting conditions.
    depth     : emulated depth-sensor faults — gaussian noise, quantization,
                min/max range dropout, random pixel dropout, and a `stuck` flag
                (dead sensor -> all zeros).
    display   : which preview panels to show (rgb / birdseye) and window scale.
    output    : where captures go and which streams to save.
    seed      : optional RNG seed for reproducible sensor noise/dropout.

CODE MAP
    load_config              read the YAML.
    make_sensor_spec/make_cfg build the habitat Simulator + agent config,
                             including a `move_backward` action (not in the
                             habitat default action space).
    add_start_marker         spawn a box at agent.start_position so the same
                             landmark appears in both views.
    apply_lighting           inject lighting condition into RGB.
    apply_depth_faults       inject depth-sensor faults into the depth map.
    depth_to_vis/semantic_to_vis  colourise depth/semantic for display + save.
    process_observations     apply the config to raw observations ONCE, so the
                             preview matches exactly what gets saved.
    build_canvas/draw        compose enabled panels and blit them to pygame.
    main                     dirs -> simulator -> pygame window -> event loop.

DATA OUTPUTS  (under output.root, e.g. data_collection/first_floor/)
    rgb/<n>.png  depth/<n>.png  semantic/<n>.png   one set per capture.
    GT_pose.npy  : (N, 7) array of captured poses [x, y, z, qw, qx, qy, qz].

CRITICAL GOTCHA — DO NOT REMOVE THE GL WORKAROUNDS
    habitat-sim and pygame both want an OpenGL context on the same X display,
    which crashes fatally with `X Error ... X_GLXMakeCurrent BadAccess`. Two
    complementary fixes are required (both verified; either alone still
    crashes) — see root_cause_analysis.md for the full write-up:
      1. Force SDL to a pure-software X11 window via the SDL_* env vars set
         BELOW, before `import pygame`, so pygame never creates a GLX context.
      2. Hide DISPLAY while `habitat_sim.Simulator(...)` is constructed (in
         main()) so habitat renders offscreen on EGL instead of GLX.
    Result: habitat owns all GL (offscreen/EGL); pygame only CPU-blits finished
    frames. Keep both in place.
"""

import os
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

import magnum as mn
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb


# =============================================================================
# Config
# =============================================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Simulator / agent construction
# =============================================================================
def make_sensor_spec(uuid, sensor_type, cam):
    """Build one CameraSensorSpec from the `camera` config block."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [cam["height"], cam["width"]]
    spec.position = [float(v) for v in cam["position"]]          # extrinsics: translation
    spec.orientation = np.array(cam["orientation"], dtype=np.float32)  # extrinsics: rotation
    spec.hfov = float(cam["hfov"])                                # intrinsics: field of view
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return spec


def make_cfg(config):
    """Assemble the full habitat_sim.Configuration from the loaded YAML."""
    cam = config["camera"]

    # --- simulator backend ---
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = config["scene"]["path"]
    # Physics is only needed to spawn the start-position marker object.
    sim_cfg.enable_physics = bool((config.get("marker") or {}).get("enabled", False))

    # --- agent + sensors ---
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        make_sensor_spec("color_sensor", habitat_sim.SensorType.COLOR, cam),
        make_sensor_spec("depth_sensor", habitat_sim.SensorType.DEPTH, cam),
        make_sensor_spec("semantic_sensor", habitat_sim.SensorType.SEMANTIC, cam),
        # Top-down bird's-eye camera (own intrinsics/extrinsics), preview only.
        make_sensor_spec("birdseye_sensor", habitat_sim.SensorType.COLOR, config["birdseye"]),
    ]

    # Discrete action space. move_backward is added on top of the usual three.
    a = config["agent"]
    ActionSpec = habitat_sim.agent.ActionSpec
    ActuationSpec = habitat_sim.agent.ActuationSpec
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=float(a["move_forward"]))),
        "move_backward": ActionSpec("move_backward", ActuationSpec(amount=float(a["move_backward"]))),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=float(a["turn_left"]))),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=float(a["turn_right"]))),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def add_start_marker(sim, config):
    """Spawn a box in the world at the agent's start_position.

    Because it is a real 3D object, the SAME landmark shows up in both the
    first-person and the bird's-eye views, which makes it easy to correlate the
    two (otherwise mismatched) viewpoints. Requires physics — enabled in
    make_cfg whenever `marker.enabled`. Returns the object, or None if the
    marker is disabled or unavailable.
    """
    mcfg = config.get("marker") or {}
    if not mcfg.get("enabled", False):
        return None

    obj_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()

    handles = obj_mgr.get_template_handles("cubeSolid")
    if not handles:
        print("marker: no 'cubeSolid' primitive template available; skipping")
        return None

    template = obj_mgr.get_template_by_handle(handles[0])
    template.scale = mn.Vector3(*[float(v) for v in mcfg["size"]])
    obj_mgr.register_template(template, "start_marker")

    obj = rigid_mgr.add_object_by_template_handle("start_marker")
    if obj is None:
        print("marker: failed to add object (is physics enabled?); skipping")
        return None

    start = [float(v) for v in config["agent"]["start_position"]]
    start[1] += float(mcfg.get("height_offset", 0.0))
    # Agent spawns facing -Z, so push the marker forward along -Z. At exactly
    # start_position the box sits under the (1.5m-high, horizontal) first-person
    # camera and falls outside its FOV — visible in bird's-eye but not first
    # person. Offsetting forward makes the same landmark co-visible in BOTH views.
    start[2] -= float(mcfg.get("forward_offset", 0.0))
    obj.translation = mn.Vector3(*start)                    # place before freezing
    obj.motion_type = habitat_sim.physics.MotionType.STATIC  # don't let it fall
    print(f"marker: placed start-position marker at "
          f"({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f})")
    return obj


# =============================================================================
# Sensor post-processing (this is where "real-world uncertainty" is injected)
# =============================================================================
def apply_lighting(rgb, cfg):
    """Photometric emulation of lighting conditions on an RGB (H,W,3) uint8 image."""
    img = rgb[:, :, :3].astype(np.float32) / 255.0
    img *= np.asarray(cfg["ambient_rgb"], dtype=np.float32)   # colour tint / temperature
    img *= float(cfg["brightness"])                            # exposure gain
    img = (img - 0.5) * float(cfg["contrast"]) + 0.5           # contrast around mid-grey
    img = np.clip(img, 0.0, 1.0)
    gamma = float(cfg["gamma"])
    if gamma != 1.0:
        img = img ** (1.0 / gamma)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_depth_faults(depth_m, cfg):
    """Emulate depth-sensor faults on a raw depth map (meters, float32)."""
    if not cfg.get("enabled", True) or cfg.get("stuck", False):
        return np.zeros_like(depth_m)

    d = depth_m.astype(np.float32).copy()

    if float(cfg["noise_std"]) > 0.0:
        d += np.random.normal(0.0, float(cfg["noise_std"]), size=d.shape).astype(np.float32)

    if float(cfg["quantization"]) > 0.0:
        step = float(cfg["quantization"])
        d = np.round(d / step) * step

    # Out-of-range readings return nothing (0).
    out_of_range = (d < float(cfg["min_range"])) | (d > float(cfg["max_range"]))
    d[out_of_range] = 0.0

    if float(cfg["dropout_prob"]) > 0.0:
        drop = np.random.random(d.shape) < float(cfg["dropout_prob"])
        d[drop] = 0.0

    return np.clip(d, 0.0, None)


# --- visualisation helpers (all return RGB uint8 for pygame; BGR is only for cv2 saves) ---
def depth_to_vis(depth_m, max_range):
    d = np.clip(depth_m / max(max_range, 1e-6), 0.0, 1.0)
    gray = (d * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)  # (H,W,3) RGB


def semantic_to_vis(semantic_obs):
    img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    img.putpalette(d3_40_colors_rgb.flatten())
    img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    return np.asarray(img.convert("RGB"))  # (H,W,3) RGB


# =============================================================================
# Processed frame: apply the config to raw observations once, reuse for
# both display and saving so the preview matches what gets written.
# =============================================================================
def process_observations(obs, config):
    depth_m = apply_depth_faults(obs["depth_sensor"], config["depth"])
    return {
        "rgb": apply_lighting(obs["color_sensor"], config["lighting"]),     # RGB uint8
        "birdseye": obs["birdseye_sensor"][:, :, :3],                        # top-down RGB uint8
        "depth_m": depth_m,
        "depth_vis": depth_to_vis(depth_m, float(config["depth"]["max_range"])),
        "semantic": semantic_to_vis(obs["semantic_sensor"]),                # RGB uint8
    }


# =============================================================================
# pygame preview window
# =============================================================================
def build_canvas(frame, display_cfg):
    """Stack the enabled panels (first-person RGB + bird's-eye) horizontally
    into one (H, W, 3) RGB image."""
    panels = []
    if display_cfg.get("show_rgb", True):
        panels.append(frame["rgb"])
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


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)

    if config.get("seed") is not None:
        np.random.seed(int(config["seed"]))

    # --- output dirs ---
    out = config["output"]
    data_root = out["root"]
    if out["clear_existing"] and os.path.isdir(data_root):
        shutil.rmtree(data_root)  # WARNING: deletes the whole directory
    for sub in ("rgb", "depth", "semantic"):
        os.makedirs(os.path.join(data_root, sub), exist_ok=True)

    # --- simulator ---
    # Habitat and pygame both want an OpenGL context. If habitat opens a GLX
    # context on the X display, it collides with pygame's window context
    # (X_GLXMakeCurrent BadAccess). So we hide DISPLAY while the Simulator is
    # built, forcing habitat onto an offscreen EGL context, then restore DISPLAY
    # so pygame can own the on-screen window. Rendering happens on EGL and only
    # the finished frames are blitted by pygame — no shared GL context.
    saved_display = os.environ.pop("DISPLAY", None)
    try:
        sim = habitat_sim.Simulator(make_cfg(config))
    finally:
        if saved_display is not None:
            os.environ["DISPLAY"] = saved_display
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(config["agent"]["start_position"], dtype=np.float32)
    agent.set_state(agent_state)

    # Landmark at the spawn point, visible in both views (see add_start_marker).
    add_start_marker(sim, config)

    print("Discrete action space:", list(sim.config.agents[0].action_space.keys()))

    # --- keybindings ---
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

    # --- pygame window ---
    pygame.init()
    font = pygame.font.SysFont(None, 28)

    cam_extr = []   # captured GT poses [x, y, z, qw, qx, qy, qz]
    count = 0

    # Render one frame up front so the window can be sized to the real canvas
    # (RGB + bird's-eye panels), then keep it as the first preview. Read the
    # sensors WITHOUT stepping so the first preview is the true start_position
    # (sim.step would silently advance the agent before the student touches a key).
    obs = sim.get_sensor_observations()
    frame = process_observations(obs, config)
    sensor_state = agent.get_state().sensor_states["color_sensor"]
    canvas = build_canvas(frame, config["display"])
    scale = float(config["display"]["scale"])
    screen = pygame.display.set_mode(
        (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))
    pygame.display.set_caption("Habitat data collector")

    def render(frame):
        draw(screen, frame, config["display"], count, font)

    def step_and_render(action):
        """Step the sim, refresh the preview, return the processed frame + pose."""
        obs = sim.step(action)
        frame = process_observations(obs, config)
        render(frame)
        sensor_state = agent.get_state().sensor_states["color_sensor"]
        return frame, sensor_state

    def capture(frame, sensor_state):
        nonlocal count
        count += 1
        if out["save_rgb"]:
            cv2.imwrite(os.path.join(data_root, "rgb", f"{count}.png"),
                        frame["rgb"][:, :, ::-1])  # RGB -> BGR for cv2
        if out["save_depth"]:
            cv2.imwrite(os.path.join(data_root, "depth", f"{count}.png"), frame["depth_vis"])
        if out["save_semantic"]:
            cv2.imwrite(os.path.join(data_root, "semantic", f"{count}.png"),
                        frame["semantic"][:, :, ::-1])
        p, r = sensor_state.position, sensor_state.rotation
        cam_extr.append([p[0], p[1], p[2], r.w, r.x, r.y, r.z])
        render(frame)  # refresh the capture-counter overlay immediately
        print(f"captured frame {count} @ pose "
              f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
              f"({r.w:.3f}, {r.x:.3f}, {r.y:.3f}, {r.z:.3f})")

    # initial preview
    render(frame)

    running = True
    while running:
        event = pygame.event.wait()  # block until the next input (low CPU)
        if event.type == pygame.QUIT:
            break
        if event.type != pygame.KEYDOWN:
            continue

        if event.key in QUIT_KEYS:
            running = False
        elif event.key in CAPTURE_KEYS:
            capture(frame, sensor_state)
        elif event.key in KEY_ACTION:
            frame, sensor_state = step_and_render(KEY_ACTION[event.key])
        # any other key is ignored

    # --- shutdown ---
    np.save(os.path.join(data_root, "GT_pose.npy"), np.asarray(cam_extr))
    print(f"saved {len(cam_extr)} poses to {os.path.join(data_root, 'GT_pose.npy')}")
    pygame.quit()
    sim.close()


if __name__ == "__main__":
    main()
