"""P0 golden test: the public import surface every lane must keep green.

Asserts importability only — bodies may raise NotImplementedError until lanes land.
Also asserts that importing the package does NOT pull in pygame (viewer is opt-in).
"""

import sys


def test_import_surface():
    import simulator
    from simulator import (  # noqa: F401
        Engine,
        UncertaintyScheduler,
        add_start_marker,
        agent_state_from_sensor_pose,
        apply_depth_faults,
        apply_depth_sensor,
        apply_lighting,
        depth_to_vis,
        light_exposure,
        load_config,
        load_trajectory,
        make_cfg,
        make_sensor_spec,
        process_observations,
        replay_poses,
        save_frame,
        semantic_to_vis,
    )

    assert simulator.__all__


def test_no_pygame_on_package_import():
    assert "pygame" not in sys.modules, (
        "importing `simulator` must not import pygame; only `simulator.viewer` may"
    )
