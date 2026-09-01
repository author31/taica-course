"""Engine: owns the habitat_sim.Simulator, agent, scheduler, time base, RNG, and
the GL workaround (hide DISPLAY during Simulator construction so habitat renders
offscreen on EGL; restore in a finally; tolerate DISPLAY unset for pure headless).

Contract highlights (plan.md):
- Engine exposes PUBLIC `.sim` and `.agent` (search_traj needs pathfinder, the raw
  agent for GreedyGeodesicFollower, and raw observations).
- observe(t): caller supplies time — replay passes i/fps_nominal, interactive passes
  wall-clock seconds. Per-frame rng = np.random.default_rng([seed, round(t*1000)]).
- make_cfg / add_start_marker / make_sensor_spec stay importable module functions.

PERFORMANCE
- make_cfg attaches only the sensors something consumes: colour + depth always,
  semantic only when output.save_semantic, bird's-eye only when
  display.show_birdseye (default true). Each dropped 512x512 sensor is one
  fewer render pass + GPU->CPU readback per frame.
- Engine caches the raw sensor readout keyed on the agent pose. The scene is
  static, so a readout only changes when the agent moves: standing still (the
  common interactive state) costs no render at all, and the readout sim.step
  already produced is reused by the observe() that follows it. Disabled when
  depth.redwood is on (that noise is re-sampled inside the readout).
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


def wants_semantic(config):
    """Semantic sensor is attached iff the capture saves semantic PNGs."""
    return bool((config.get("output") or {}).get("save_semantic", False))


def wants_birdseye(config):
    """Bird's-eye sensor is attached iff the preview shows it (default true)."""
    return bool((config.get("display") or {}).get("show_birdseye", True))


def make_cfg(config):
    """Assemble the full habitat_sim.Configuration from the loaded YAML.

    Sensors: color_sensor + depth_sensor always; semantic_sensor / birdseye_sensor
    per wants_semantic / wants_birdseye."""
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
    specs = [
        make_sensor_spec("color_sensor", habitat_sim.SensorType.COLOR, cam),
        make_sensor_spec("depth_sensor", habitat_sim.SensorType.DEPTH, cam,
                         noise_model=depth_noise, noise_kwargs=depth_noise_kwargs),
    ]
    # Optional sensors cost a full render pass + readback per frame each, so
    # they are attached only when something consumes them (see module
    # docstring). process_observations yields None for the ones left out.
    if wants_semantic(config):
        specs.append(make_sensor_spec("semantic_sensor", habitat_sim.SensorType.SEMANTIC, cam))
    if wants_birdseye(config):
        # Top-down bird's-eye camera (own intrinsics/extrinsics), preview only.
        specs.append(make_sensor_spec("birdseye_sensor", habitat_sim.SensorType.COLOR,
                                      config["birdseye"]))
    agent_cfg.sensor_specifications = specs

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

    Seed policy: `uncertainties.seed` governs the per-frame depth RNG here and
    NOTHING ELSE — the spatial ZoneScheduler is seedless (plan.md §3.1). The old
    top-level `seed:` + `np.random.seed` path is dead — no global np.random
    anywhere.
    """

    def __init__(self, config, scheduler=None, fps_nominal=30.0, cache_observations=True):
        self.config = config
        self.scheduler = scheduler
        self.fps_nominal = float(fps_nominal)
        self.seed = int((config.get("uncertainties") or {}).get("seed", 42))

        # Raw-readout cache (see module docstring). Off under in-sim Redwood
        # depth noise: habitat re-samples it inside get_sensor_observations, so
        # a cached readout would freeze the noise while the agent stands still.
        redwood = bool((config.get("depth") or {}).get("redwood", False))
        self.cache_observations = bool(cache_observations) and not redwood
        self._obs = None          # last raw readout
        self._obs_pose = None     # agent pose it was rendered at

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

    def _pose_key(self):
        """Hashable agent pose (position + rotation) — the cache key."""
        st = self.agent.get_state()
        p, r = st.position, st.rotation
        return (float(p[0]), float(p[1]), float(p[2]),
                float(r.w), float(r.x), float(r.y), float(r.z))

    def step(self, action):
        """Step one discrete action, return raw observations.

        The readout sim.step renders is kept as the cached one, so the
        observe() that typically follows a keypress does not render again."""
        obs = self.sim.step(action)
        if self.cache_observations and isinstance(obs, dict) and "color_sensor" in obs:
            self._obs, self._obs_pose = obs, self._pose_key()
        return obs

    def raw_observations(self):
        """Current raw sensor readout, rendered only if the agent pose changed
        since the last one (or every call when caching is off)."""
        if not self.cache_observations:
            return self.sim.get_sensor_observations()
        pose = self._pose_key()
        if self._obs is None or pose != self._obs_pose:
            self._obs = self.sim.get_sensor_observations()
            self._obs_pose = pose
        return self._obs

    def observe(self, t):
        """Read the sensors (WITHOUT stepping) at time `t` -> processed frame.

        The caller supplies the time base: replay passes i / fps_nominal (its
        only deterministic clock), interactive passes wall-clock seconds since
        session start. Per-frame rng = default_rng([seed, round(t*1000)]) makes
        depth noise a function of (seed, t) alone — zone-independent, so frames
        outside every uncertainty zone are bit-identical to a baseline
        (scheduler-off) run on the same t grid.

        Uncertainty is SPATIAL (plan.md §3.1): the signature keeps `t` only, and
        the position is looked up here.

        The raw readout comes from raw_observations (cached while the agent
        stands still); the time-dependent effects (flicker, per-frame depth
        noise) are re-applied on every call regardless."""
        obs = self.raw_observations()
        rng = np.random.default_rng([self.seed, round(t * 1000)])
        overrides = {}
        if self.scheduler is not None:
            # The AGENT position is authoritative, not the sensor pose. They
            # differ only by camera.position ([0, 1.5, 0] — pure y), so the XZ
            # handed to the scheduler is identical either way; the code picks
            # the agent and says so.
            pos = self.agent.get_state().position
            overrides = self.scheduler.active(t, (float(pos[0]), float(pos[2])))
        return effects.process_observations(obs, self.config, t, rng, overrides)

    def close(self):
        self.sim.close()
