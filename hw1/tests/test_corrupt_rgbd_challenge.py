"""Safety and business-rule tests for the instructor challenge generator."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "corrupt_rgbd_challenge.py"
sys.path.insert(0, str(REPO_ROOT / "hw1"))
import api  # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location("corrupt_rgbd_challenge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capture(root: Path, count: int = 12) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "semantic").mkdir()
    for stem in range(1, count + 1):
        rgb = np.full((8, 8, 3), 40 + stem, dtype=np.uint8)
        # A real discontinuity makes the flying-pixel recipe test meaningful.
        depth = np.full((8, 8), 1000 + 2 * stem, dtype=np.uint16)
        depth[:, 4:] += 700
        depth[0, 0] = 0
        Image.fromarray(rgb).save(root / "rgb" / f"{stem}.png")
        Image.fromarray(depth).save(root / "depth" / f"{stem}.png")
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
            root / "semantic" / f"{stem}.png")
    np.save(root / "GT_pose.npy", np.zeros((count, 7), dtype=np.float32))
    (root / "intrinsics.json").write_text(
        json.dumps({"width": 8, "height": 8, "hfov": 90.0}))
    (root / "batch.ttl").write_text("stale source batch")


def _run(source: Path, output: Path, stems: str = "2,3,4,5,6,7",
         *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output),
         "--inject-stems", stems, "--tile-size", "4", "--seed", "7", *extra],
        cwd=REPO_ROOT, text=True, capture_output=True)


def test_default_plan_is_manually_blocked_and_exactly_100_frames():
    generator = _load_generator()
    stems = generator.DEFAULT_INJECTED_STEMS
    assert len(stems) == 100
    assert len(set(stems)) == 100
    assert generator.DEFAULT_INJECTION_BLOCKS[0] == (5, 9)
    assert generator.DEFAULT_INJECTION_BLOCKS[-1] == (176, 180)
    assert all(end - start + 1 == 5
               for start, end in generator.DEFAULT_INJECTION_BLOCKS)


def test_depth_recipes_cross_their_stock_factor_thresholds(tmp_path):
    """Each edit is defined by the semantic factor it is meant to trip."""
    generator = _load_generator()
    base = np.full((96, 96), 1600, dtype=np.uint16)
    base[:, 48:] = 2300
    base_path = tmp_path / "base.png"
    Image.fromarray(base).save(base_path)
    args = Namespace(
        hf_noise_sigma_mm=80.0,
        flying_gap_mm=600,
        flying_stripe_period=8,
        tile_size=64,
        tile_drop_fraction=0.65,
        tile_remaining_valid_fraction=0.20,
        identity_bias_mm=350,
        joint_dropout_fraction=0.75,
        prior_bias_mm=150,
    )
    rng = np.random.default_rng(20260803)
    values = {}
    for factor in generator.DEPTH_FACTORS:
        edited, _parameters = generator._apply_factor_edit(base, factor, rng, args)
        path = tmp_path / f"{factor}.png"
        Image.fromarray(edited).save(path)
        if factor == "HighFrequencyDepthResidual":
            values[factor] = api.frame_high_frequency_depth_residual(str(path))
        elif factor == "FlyingPixelRatio":
            values[factor] = api.frame_flying_pixel_ratio(str(path))
        elif factor == "ValidTileCoverage":
            values[factor] = api.frame_valid_tile_coverage(str(path))
        elif factor == "IdentityMedianDepthChange":
            values[factor] = api.pair_identity_median_depth_change(
                str(base_path), str(path))
        elif factor == "JointValidDepthRatio":
            values[factor] = api.pair_joint_valid_depth_ratio(
                str(base_path), str(path))
        else:
            values[factor] = api.pair_prior_warp_depth_residual(
                str(base_path), str(path), np.eye(4),
                {"width": 96, "height": 96, "hfov": 90.0})

    assert values["HighFrequencyDepthResidual"] > 0.05
    assert values["FlyingPixelRatio"] > 0.05
    assert values["ValidTileCoverage"] < 0.50
    assert values["IdentityMedianDepthChange"] > 0.20
    assert values["JointValidDepthRatio"] < 0.30
    assert values["PriorWarpDepthResidual"] > 0.10


def test_generator_clones_t_minus_one_then_edits_depth_by_all_factors(tmp_path):
    source = tmp_path / "clean"
    output = tmp_path / "challenge"
    _capture(source)
    source_snapshots = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*") if path.is_file()
    }

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    assert not (output / "batch.ttl").exists()
    assert not (output / "corruption.json").exists()
    assert (output / "GT_pose.npy").read_bytes() == source_snapshots[Path("GT_pose.npy")]
    assert (output / "semantic" / "2.png").is_file()
    for relative, before in source_snapshots.items():
        assert (source / relative).read_bytes() == before

    for stem in range(2, 8):
        # RGB is the exact final output at t-1; consecutive targets cascade.
        assert (output / "rgb" / f"{stem}.png").read_bytes() == (
            output / "rgb" / f"{stem - 1}.png").read_bytes()
        # Depth was cloned from t-1, then a factor treatment changed it.
        assert (output / "depth" / f"{stem}.png").read_bytes() != (
            output / "depth" / f"{stem - 1}.png").read_bytes()
    assert (output / "rgb" / "8.png").read_bytes() == (source / "rgb" / "8.png").read_bytes()
    assert (output / "depth" / "8.png").read_bytes() == (source / "depth" / "8.png").read_bytes()

    manifest = json.loads((tmp_path / "challenge.corruption.json").read_text())
    rule = manifest["business_rule"]
    assert manifest["schema_version"] == 2
    assert rule["injected_frame_count"] == 6
    assert rule["default_plan_is_exactly_100"] is False
    assert [row["primary_factor"] for row in rule["frames"]] == rule[
        "factor_assignment_order"]
    assert [row["cloned_from_frame"] for row in rule["frames"]] == list(range(1, 7))
    assert all(row["rgb"]["final_is_exact_t_minus_1_clone"] for row in rule["frames"])
    assert all(row["depth"]["clone_t_minus_1_sha256"] !=
               row["depth"]["final_sha256"] for row in rule["frames"])


def test_dry_run_reports_the_default_100_frame_business_rule(tmp_path):
    source = tmp_path / "clean"
    output = tmp_path / "challenge"
    _capture(source, count=193)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output), "--dry-run"],
        cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["injected_frame_count"] == 100
    assert plan["injected_stems"][0:5] == [5, 6, 7, 8, 9]
    assert sum(plan["factor_assignment_counts"].values()) == 100
    assert not output.exists()


def test_generator_refuses_to_replace_existing_output(tmp_path):
    source = tmp_path / "clean"
    output = tmp_path / "challenge"
    _capture(source)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("owned by user")
    result = _run(source, output)
    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert sentinel.read_text() == "owned by user"


def test_generator_rejects_unmatched_modalities(tmp_path):
    source = tmp_path / "clean"
    output = tmp_path / "challenge"
    _capture(source)
    (source / "depth" / "7.png").unlink()
    result = _run(source, output)
    assert result.returncode != 0
    assert "stems do not match" in result.stderr
    assert not output.exists()


def test_generator_requires_a_literal_t_minus_one_frame(tmp_path):
    source = tmp_path / "clean"
    output = tmp_path / "challenge"
    _capture(source)
    result = _run(source, output, "1")
    assert result.returncode != 0
    assert "t-1 predecessor" in result.stderr
    assert not output.exists()


def test_generator_refuses_output_nested_in_clean_source(tmp_path):
    source = tmp_path / "clean"
    _capture(source)
    result = _run(source, source / "challenge")
    assert result.returncode != 0
    assert "must not be nested" in result.stderr
    assert not (source / "challenge").exists()
