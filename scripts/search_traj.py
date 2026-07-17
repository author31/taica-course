"""
Headless whole-map trajectory search for the Habitat data collector.

================================================================================
WHAT THIS IS
    scripts/load.py is an *interactive* collector: a human drives the agent in a
    pygame window and hand-picks frames. scripts/reconstruct.py chains
    frame-to-frame ICP over those captures and scores the run by the mean L2
    distance between the predicted camera trajectory and the ground-truth one.

    This script closes the loop automatically AND covers the whole apartment.
    Instead of a handful of hand-driven frames it:

      1. INSPECTS THE MAP   — loads the Replica navmesh
         (replica_v1/apartment_0/habitat/mesh_semantic.navmesh), reports its
         bounds / navigable extent, and samples navigable points on the agent's
         floor.
      2. PLANS COVERAGE     — farthest-point-samples those into a few waypoints
         that span the reachable area, orders them into a short tour, and
         CLOSES the loop by returning to the start (revisit=closed) so the
         reconstructor's pose-graph back-end gets a strong loop closure.
      3. CONTROLS THE AGENT — a GreedyGeodesicFollower walks the geodesic path
         between goals, emitting discrete actions (move_forward / turn_left /
         turn_right). Frames (RGB-D + GT pose) are captured every step; depth is
         saved as 16-bit millimetres so the geometry is precise (reconstruct.py
         auto-detects uint16 depth).
      4. EVALUATES          — reconstructs the run with reconstruct.py (robust
         colored-ICP SLAM + pose-graph loop closure) and measures mean L2.
      5. REPEATS            — if L2 >= target (default 0.3 m) it escalates the
         waypoint count (3 -> 4 -> 5 -> 6); the shortest tour that still covers
         the floor wins. Stops at the first run under target.

WHY SHORT PATH + A CLOSING LOOP IS THE LEVER
    reconstruct.py's SLAM front-end (colored ICP) drifts ~linearly with distance
    travelled (no catastrophic blowups), so the mean L2 scales with PATH LENGTH:
    fewer waypoints => shorter tour => less drift. A single closing loop
    (return-to-start) then gives the pose-graph back-end one strong loop closure
    that folds the residual drift out. Empirically 3 waypoints + closed loop over
    the apartment reaches ~0.14 m mean L2. Retracing or piling on waypoints only
    lengthens the path and RAISES drift, so neither is the default.

HOW TO RUN
    pixi run -e habitat python scripts/search_traj.py \
        [--config scripts/config.yaml] [--target 0.3] [--revisit closed] \
        [--waypoints N] [--iters 4] [--version open3d|my_icp] \
        [--out best_trajectory.npy] [--save-best-data]

OUTPUTS
    <out>.npy            : (N,7) best GT trajectory [x,y,z, qw,qx,qy,qz]
                           (same layout as load.py's GT_pose.npy).
    <out>.actions.json   : winning action sequence + score/metadata.
    (--save-best-data)   : re-writes the winner's rgb/ depth/ GT_pose.npy under
                           output.root so reconstruct.py can render it.
"""

import os
import sys
import json
import shutil
import argparse
import tempfile

import numpy as np
import cv2

# Import the collector and SLAM utils as libraries from hw1/. load.py sets the
# SDL software-render env vars and imports habitat at module load; harmless here
# (no pygame window is ever opened). utils.py pulls in open3d.
_HW1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hw1")
sys.path.insert(0, _HW1)
import load
import utils

try:                                    # exact follower failure type when present
    from habitat_sim.errors import GreedyFollowerError as _GFE
except Exception:                       # fall back to broad catch if API differs
    _GFE = Exception


# Fixed actuation. Empirically 0.20m / 6deg with stride 1 gives colored-ICP
# odometry enough frame overlap to register robustly; going finer only adds
# frames (finer voxels/steps do NOT lower reconstruct.py's drift, and can hurt).
FIXED_GRAN = {"turn": 6.0, "fwd": 0.20, "stride": 1}

# The real lever on whole-map mean-L2 is PATH LENGTH (drift ~ linear in metres),
# not per-frame density. Fewer coverage waypoints => shorter tour => less drift;
# a closing loop (revisit=closed) then gives the pose-graph back-end one strong
# loop closure that folds the residual drift down. So the search escalates the
# waypoint count only if a shorter tour misses target. (3 + closed ~= 0.14m.)
WAYPOINT_SCHEDULE = [3, 4, 5, 6]


# =============================================================================
# Headless simulator
# =============================================================================
def build_sim(config):
    """Construct the habitat Simulator + agent like load.main(), without pygame.
    DISPLAY is hidden during construction so habitat renders offscreen on EGL
    (see the GL workaround note in load.py)."""
    import habitat_sim

    saved_display = os.environ.pop("DISPLAY", None)
    try:
        sim = habitat_sim.Simulator(load.make_cfg(config))
    finally:
        if saved_display is not None:
            os.environ["DISPLAY"] = saved_display

    agent = sim.initialize_agent(0)
    load.add_start_marker(sim, config)
    return sim, agent


def load_navmesh(sim, navmesh_path):
    """Ensure sim.pathfinder has a navmesh. Prefer the .navmesh shipped next to
    the mesh; fall back to recomputing one from the scene geometry."""
    import habitat_sim

    pf = sim.pathfinder
    if navmesh_path and os.path.exists(navmesh_path):
        pf.load_nav_mesh(navmesh_path)
    if not pf.is_loaded:
        print("navmesh: none found, recomputing from scene geometry...")
        settings = habitat_sim.nav.NavMeshSettings()
        settings.set_defaults()
        sim.recompute_navmesh(pf, settings)
    if not pf.is_loaded:
        raise RuntimeError("could not obtain a navmesh for pathfinding")
    return pf


# =============================================================================
# 1. Map inspection
# =============================================================================
def inspect_map(pathfinder, start_xyz, n_samples=3000, floor_tol=0.5, seed=0):
    """Report navmesh extent and return navigable points on the agent's floor
    that are reachable from `start_xyz`."""
    lo, hi = pathfinder.get_bounds()
    print("=" * 60)
    print("MAP INSPECTION")
    print(f"  navmesh bounds : x[{lo[0]:.2f},{hi[0]:.2f}] "
          f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")

    start = np.asarray(pathfinder.snap_point(start_xyz), dtype=np.float32)
    print(f"  start (snapped): ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")

    pathfinder.seed(int(seed))          # so get_random_navigable_point varies per iter
    pts = []
    for _ in range(n_samples):
        p = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if abs(p[1] - start[1]) > floor_tol:           # keep the agent's floor
            continue
        if not np.isfinite(geodesic(pathfinder, start, p)):  # reachable only
            continue
        pts.append(p)
    pts = np.asarray(pts, dtype=np.float32) if pts else np.empty((0, 3), np.float32)

    if len(pts):
        span = pts.max(0) - pts.min(0)
        print(f"  navigable pts  : {len(pts)} on floor "
              f"(footprint {span[0]:.2f}m x {span[2]:.2f}m)")
    else:
        print("  navigable pts  : NONE reachable — check start position")
    print("=" * 60)
    return start, pts


def geodesic(pathfinder, a, b):
    """Geodesic (on-navmesh) distance between two points, inf if unreachable."""
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    pathfinder.find_path(path)
    return path.geodesic_distance


# =============================================================================
# 2. Coverage planning
# =============================================================================
def farthest_point_sample(points, k, seed_point):
    """Pick `k` points that spread across `points`, greedily maximizing the
    minimum distance to already-chosen points (start seeds the set)."""
    if len(points) == 0:
        return np.empty((0, 3), np.float32)
    k = min(k, len(points))
    chosen = [np.asarray(seed_point, dtype=np.float32)]
    d = np.linalg.norm(points - chosen[0], axis=1)
    for _ in range(k):
        i = int(np.argmax(d))
        chosen.append(points[i].copy())
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.asarray(chosen[1:], dtype=np.float32)     # drop the seed itself


def order_tour(pathfinder, start, waypoints):
    """Greedy nearest-neighbor tour (by geodesic distance) from `start`."""
    remaining = list(range(len(waypoints)))
    tour, cur = [], np.asarray(start, dtype=np.float32)
    while remaining:
        j = min(remaining, key=lambda i: geodesic(pathfinder, cur, waypoints[i]))
        tour.append(j)
        cur = waypoints[j]
        remaining.remove(j)
    return [waypoints[i] for i in tour]


def build_goal_sequence(start, tour, mode):
    """Expand a coverage tour into a goal sequence that REVISITS geometry, so
    the reconstructor's pose-graph back-end has loop closures to bound drift:

      none    : the tour as-is (no revisits; drift is uncorrectable).
      closed  : tour then back to start (one big loop closure).
      retrace : tour out, then retrace it back to start (every place revisited
                -> dense loop closures throughout). Best drift bounding.
      hub     : return to start after every waypoint (star pattern; the start
                region is revisited many times).
    """
    start = np.asarray(start, dtype=np.float32)
    if mode == "none":
        return list(tour)
    if mode == "closed":
        return list(tour) + [start]
    if mode == "hub":
        goals = []
        for w in tour:
            goals += [w, start]
        return goals
    # retrace (default)
    return list(tour) + list(reversed(tour[:-1])) + [start]


# =============================================================================
# 3. Agent control — follow the geodesic path, capture RGB-D + poses
# =============================================================================
def capture_frame(sim, agent, config):
    obs = sim.get_sensor_observations()
    frame = load.process_observations(obs, config)
    ss = agent.get_state().sensor_states["color_sensor"]
    p, r = ss.position, ss.rotation
    return frame, [p[0], p[1], p[2], r.w, r.x, r.y, r.z]


def traverse(sim, agent, config, pathfinder, start, waypoints, stride,
             max_steps_per_leg=600):
    """Walk start -> each waypoint using a GreedyGeodesicFollower, capturing a
    frame every `stride` steps. Returns a list of (frame, pose)."""
    import habitat_sim

    # Reset the agent onto the (snapped, navigable) start pose.
    state = habitat_sim.AgentState()
    state.position = np.asarray(start, dtype=np.float32)
    agent.set_state(state)

    follower = habitat_sim.nav.GreedyGeodesicFollower(
        pathfinder, agent,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right",
        fix_thrashing=True,
    )

    captures = [capture_frame(sim, agent, config)]      # start frame
    actions = []
    step_i = 0
    for goal in waypoints:
        follower.reset()
        goal = np.asarray(goal, dtype=np.float32)
        for _ in range(max_steps_per_leg):
            try:
                action = follower.next_action_along(goal)
            except _GFE:
                break                                   # give up on this leg
            if action is None:                          # reached the waypoint
                break
            sim.step(action)
            actions.append(action)
            step_i += 1
            if step_i % stride == 0:
                captures.append(capture_frame(sim, agent, config))
        captures.append(capture_frame(sim, agent, config))  # frame at waypoint
    return captures, actions


# =============================================================================
# Capture I/O + scoring (reconstruct.py compatible)
# =============================================================================
def write_capture(captures, data_root, out_cfg):
    """Persist a rollout in the layout reconstruct.py expects:
    rgb/<n>.png, depth/<n>.png (1..N), GT_pose.npy (N,7).

    Depth is written as 16-bit millimetres so the geometry is precise;
    reconstruct.load_depth_meters detects uint16 and divides by DEPTH_SCALE."""
    for sub in ("rgb", "depth"):
        d = os.path.join(data_root, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    poses = []
    for i, (frame, pose) in enumerate(captures, start=1):
        if out_cfg.get("save_rgb", True):
            cv2.imwrite(os.path.join(data_root, "rgb", f"{i}.png"),
                        frame["rgb"][:, :, ::-1])              # RGB -> BGR
        if out_cfg.get("save_depth", True):
            depth_mm = np.clip(frame["depth_m"] * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(data_root, "depth", f"{i}.png"), depth_mm)
        poses.append(pose)

    poses = np.asarray(poses, dtype=np.float32)
    np.save(os.path.join(data_root, "GT_pose.npy"), poses)
    return poses


def score_capture(data_root, floor, version):
    """Reconstruct the captured run and return its mean L2 error (lower=better).
    Delegates to hw1/utils (geometry-only ICP); build_cloud=False skips the heavy
    global-map accumulation since only the trajectory is scored."""
    _, pred_cam_pos, gt_poses = utils.reconstruct(
        data_root, version, verbose=False, build_cloud=False)
    return utils.mean_l2(pred_cam_pos, gt_poses)


def path_length(poses):
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)))


# =============================================================================
# Rollout of one density setting (build sim -> plan -> traverse -> capture)
# =============================================================================
def set_actuation(config, gran):
    a = config["agent"]
    a["move_forward"] = a["move_backward"] = float(gran["fwd"])
    a["turn_left"] = a["turn_right"] = float(gran["turn"])


def rollout(config, args, waypoints, wp_seed, work_root, navmesh):
    """Build a sim, plan `waypoints` coverage points, traverse (with revisits),
    capture, and score. Returns (l2, poses, actions, waypoints)."""
    set_actuation(config, FIXED_GRAN)
    start_xyz = np.asarray(config["agent"]["start_position"], dtype=np.float32)

    sim, agent = build_sim(config)
    try:
        pf = load_navmesh(sim, navmesh)
        start, pts = inspect_map(pf, start_xyz, seed=wp_seed)
        if len(pts) == 0:
            return float("inf"), None, None, None
        wp = order_tour(pf, start, farthest_point_sample(pts, waypoints, start))
        goals = build_goal_sequence(start, wp, args.revisit)
        print(f"planned {len(wp)} coverage waypoints -> {len(goals)} goals "
              f"(revisit={args.revisit})")
        captures, actions = traverse(sim, agent, config, pf, start, goals, FIXED_GRAN["stride"])
    finally:
        sim.close()

    poses = write_capture(captures, work_root, config.get("output", {}))
    disp = path_length(poses)
    print(f"captured {len(poses)} frames over {disp:.2f}m of travel "
          f"({waypoints} waypoints, revisit={args.revisit})")
    l2 = score_capture(work_root, args.floor, args.version)
    return l2, poses, actions, wp


# =============================================================================
# 5. Refine-until-target workflow (density schedule)
# =============================================================================
def run_workflow(config, args):
    navmesh = args.navmesh or _default_navmesh(config)
    work_root = tempfile.mkdtemp(prefix="traj_search_")
    wp_seed = int(config.get("seed") or 0)

    best = {"l2": float("inf"), "actions": None, "poses": None,
            "waypoints": None, "num_waypoints": None}
    # Search from the shortest tour up; a single --waypoints overrides the schedule.
    if args.waypoints is not None:
        schedule = [args.waypoints]
    else:
        schedule = WAYPOINT_SCHEDULE[:max(1, args.iters)]

    try:
        for it, wpk in enumerate(schedule):
            print(f"\n########## iteration {it + 1}/{len(schedule)} "
                  f"waypoints={wpk} revisit={args.revisit} ##########")
            l2, poses, actions, wp = rollout(config, args, wpk, wp_seed,
                                             work_root, navmesh)
            if poses is None:
                print("no navigable points; aborting")
                break

            improved = l2 < best["l2"]
            print(f"mean L2 = {l2:.4f}m"
                  + ("  <-- new best" if improved else "")
                  + ("  [TARGET MET]" if l2 < args.target else ""))
            if improved:
                best.update(l2=l2, actions=actions, poses=poses.copy(),
                            waypoints=[w.tolist() for w in wp], num_waypoints=wpk)
            if l2 < args.target:
                break
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    return best


def _default_navmesh(config):
    mesh = config["scene"]["path"]
    cand = os.path.splitext(mesh)[0] + ".navmesh"
    return cand if os.path.exists(cand) else None


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--navmesh", default=None,
                        help="explicit .navmesh path (default: next to the mesh)")
    parser.add_argument("--target", type=float, default=0.3,
                        help="stop once mean L2 drops below this (meters)")
    parser.add_argument("--waypoints", type=int, default=None,
                        help="fix the coverage-waypoint count (overrides the "
                             "search; default: escalate 3,4,5,6 until target met)")
    parser.add_argument("--revisit", default="closed",
                        choices=("none", "closed", "retrace", "hub"),
                        help="how the tour revisits geometry to create loop "
                             "closures (closed = return to start; one strong "
                             "loop closure that folds out drift — best value)")
    parser.add_argument("--iters", type=int, default=4,
                        help="max waypoint-schedule steps to try")
    parser.add_argument("--version", default="open3d", choices=("open3d", "my_icp"),
                        help="ICP backend used for scoring (reconstruct.py)")
    parser.add_argument("--floor", type=int, default=1)
    parser.add_argument("--out", default="best_trajectory.npy")
    parser.add_argument("--save-best-data", action="store_true",
                        help="re-capture the winner under output.root so "
                             "reconstruct.py can render it")
    args = parser.parse_args()

    config = load.load_config(args.config)
    # Target the Replica apartment_0 habitat scene regardless of stale config.
    config["scene"]["path"] = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
    if config.get("seed") is not None:
        np.random.seed(int(config["seed"]))

    best = run_workflow(config, args)

    if best["poses"] is None:
        print("\nno valid trajectory found (raise --iters / --waypoints)")
        return

    np.save(args.out, best["poses"])
    with open(args.out + ".actions.json", "w") as f:
        json.dump({"mean_l2": best["l2"],
                   "num_frames": int(len(best["poses"])),
                   "actuation": FIXED_GRAN,
                   "revisit": args.revisit,
                   "num_waypoints": best["num_waypoints"],
                   "target": args.target,
                   "target_met": best["l2"] < args.target,
                   "waypoints": best["waypoints"],
                   "actions": best["actions"]}, f, indent=2)
    status = "TARGET MET" if best["l2"] < args.target else "BELOW-TARGET FLOOR"
    print(f"\n[{status}] best mean L2 = {best['l2']:.4f}m "
          f"over {len(best['poses'])} frames")
    print(f"saved trajectory -> {args.out}")
    print(f"saved metadata   -> {args.out}.actions.json")

    if args.save_best_data:
        navmesh = args.navmesh or _default_navmesh(config)
        set_actuation(config, FIXED_GRAN)
        sim, agent = build_sim(config)
        try:
            pf = load_navmesh(sim, navmesh)
            start = np.asarray(pf.snap_point(
                config["agent"]["start_position"]), dtype=np.float32)
            wp = [np.asarray(w, dtype=np.float32) for w in best["waypoints"]]
            goals = build_goal_sequence(start, wp, args.revisit)
            captures, _ = traverse(sim, agent, config, pf, start, goals,
                                   FIXED_GRAN["stride"])
            write_capture(captures, config["output"]["root"], config.get("output", {}))
        finally:
            sim.close()
        print(f"saved best-run data -> {config['output']['root']}")


if __name__ == "__main__":
    main()
