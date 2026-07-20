"""Engine: owns the habitat_sim.Simulator, agent, scheduler, time base, RNG, and
the GL workaround (hide DISPLAY during Simulator construction so habitat renders
offscreen on EGL; restore in a finally; tolerate DISPLAY unset for pure headless).

Contract highlights (plan.md):
- Engine exposes PUBLIC `.sim` and `.agent` (search_traj needs pathfinder, the raw
  agent for GreedyGeodesicFollower, and raw observations).
- observe(t): caller supplies time — replay passes i/fps_nominal, interactive passes
  wall-clock seconds. Per-frame rng = np.random.default_rng([seed, round(t*1000)]).
- make_cfg / add_start_marker / make_sensor_spec stay importable module functions.
"""

import os

import numpy as np

import magnum as mn
import habitat_sim

from simulator import effects


def make_sensor_spec(uuid, sensor_type, cam, noise_model=None, noise_kwargs=None):
    """Build one CameraSensorSpec from the `camera` config block.

    `noise_model` (str) attaches a habitat built-in sensor noise model applied
    IN-SIM (e.g. "RedwoodDepthNoiseModel" for depth); `noise_kwargs` are passed
    to it. Left unset = a noise-free sensor."""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [cam["height"], cam["width"]]
    spec.position = [float(v) for v in cam["position"]]          # extrinsics: translation
    spec.orientation = np.array(cam["orientation"], dtype=np.float32)  # extrinsics: rotation
    spec.hfov = float(cam["hfov"])                                # intrinsics: field of view
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    if noise_model:
        spec.noise_model = noise_model                           # in-sim sensor noise
        if noise_kwargs:
            spec.noise_model_kwargs = noise_kwargs
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
    # Depth sensor optionally carries habitat's built-in Redwood noise model
    # (Choi et al.), applied in-sim before observations are returned.
    dcfg = config.get("depth") or {}
    depth_noise = "RedwoodDepthNoiseModel" if dcfg.get("redwood", False) else None
    depth_noise_kwargs = (
        {"noise_multiplier": float(dcfg.get("redwood_multiplier", 1.0))}
        if depth_noise else None)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        make_sensor_spec("color_sensor", habitat_sim.SensorType.COLOR, cam),
        make_sensor_spec("depth_sensor", habitat_sim.SensorType.DEPTH, cam,
                         noise_model=depth_noise, noise_kwargs=depth_noise_kwargs),
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


class Engine:
    """Config-driven simulator session: Simulator + agent + scheduler + RNG.

    CRITICAL GOTCHA — DO NOT REMOVE THE GL WORKAROUND
        habitat-sim and pygame both want an OpenGL context on the same X
        display, which crashes fatally with `X Error ... X_GLXMakeCurrent
        BadAccess`. DISPLAY is hidden while `habitat_sim.Simulator(...)` is
        constructed so habitat renders offscreen on EGL instead of GLX, then
        restored (in a finally) so a viewer can own the on-screen window.
        DISPLAY may legitimately be unset (pure headless) — tolerated.
        The complementary fix — SDL software rendering — lives at the top of
        simulator.viewer; callers must construct Engine BEFORE the viewer.

    Seed policy (single seed): `uncertainties.seed` governs both the
    scheduler's window sampling (scheduler is built by the caller from the same
    config) and the per-frame depth RNG here. The old top-level `seed:` +
    `np.random.seed` path is dead — no global np.random anywhere.
    """

    def __init__(self, config, scheduler=None, fps_nominal=30.0):
        self.config = config
        self.scheduler = scheduler
        self.fps_nominal = float(fps_nominal)
        self.seed = int((config.get("uncertainties") or {}).get("seed", 42))

        # GL workaround: hide DISPLAY so habitat constructs on offscreen EGL.
        saved_display = os.environ.pop("DISPLAY", None)
        try:
            self.sim = habitat_sim.Simulator(make_cfg(config))
        finally:
            if saved_display is not None:
                os.environ["DISPLAY"] = saved_display

        self.agent = self.sim.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(
            config["agent"]["start_position"], dtype=np.float32)
        self.agent.set_state(agent_state)

        # Landmark at the spawn point, visible in both views (see add_start_marker).
        add_start_marker(self.sim, config)

    def step(self, action):
        """Step one discrete action, return raw observations."""
        return self.sim.step(action)

    def observe(self, t):
        """Read the sensors (WITHOUT stepping) at time `t` -> processed frame.

        The caller supplies the time base: replay passes i / fps_nominal (its
        only deterministic clock), interactive passes wall-clock seconds since
        session start. Per-frame rng = default_rng([seed, round(t*1000)]) makes
        depth noise a function of (seed, t) alone — schedule-independent, so
        frames outside uncertainty windows are bit-identical to a baseline
        (scheduler-off) run on the same t grid."""
        obs = self.sim.get_sensor_observations()
        rng = np.random.default_rng([self.seed, round(t * 1000)])
        overrides = self.scheduler.active(t) if self.scheduler is not None else {}
        return effects.process_observations(obs, self.config, t, rng, overrides)

    def close(self):
        self.sim.close()
