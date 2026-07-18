import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rdflib import RDF


HW1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW1_DIR))

import api  # noqa: E402
import utils  # noqa: E402


EXPECTED_MENU = {
    "HighFrequencyDepthResidual",
    "FlyingPixelRatio",
    "ValidTileCoverage",
    "HighlightClipping",
    "ShadowClipping",
    "IdentityMedianDepthChange",
    "JointValidDepthRatio",
    "PriorWarpDepthResidual",
}


def _write_capture(root):
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for stem in (0, 1):
        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        depth = np.full((8, 8), 2000, dtype=np.uint16)
        if stem == 1:
            depth[:4, :4] = 0
            depth[6, 6] = 2600
        Image.fromarray(rgb).save(root / "rgb" / f"{stem}.png")
        Image.fromarray(depth).save(root / "depth" / f"{stem}.png")


def test_new_menu_and_retired_factors_are_gone():
    factors = api.load_quality_factors()
    assert set(api._selectable_factors(factors)) == EXPECTED_MENU
    assert set(api._run_level_factors(factors)) == {
        "ReconstructionAccuracy", "Coverage"}
    for old in ("ValidDepthRatio", "DepthRoughness", "DepthConsistency",
                "StructureOverlap"):
        assert old not in factors


def test_machine_graph_writes_masks_and_pair_counts(tmp_path):
    capture = tmp_path / "capture"
    _write_capture(capture)
    factors = api.load_quality_factors()
    declarations = api.load_parameter_declarations()
    selected = [
        "HighFrequencyDepthResidual", "ValidTileCoverage",
        "IdentityMedianDepthChange", "JointValidDepthRatio",
        "PriorWarpDepthResidual",
    ]
    required_names = api._required_parameters(selected, declarations, factors)
    settings = {name: declarations[name]["default"] for name in required_names}
    declaration = {
        "exp_iri": api.experiment_iri("mask_test"),
        "exp_name": "mask_test",
        "batch_name": "floor1_capture",
        "selected": selected,
        "required": settings,
        "given": {},
    }
    artifacts = tmp_path / "mask_test"
    graph, counts = api.build_machine_graph(
        declaration, str(capture), "0" * 64,
        decls=declarations, factors=factors,
        artifact_root=str(artifacts), artifact_relative_to=str(tmp_path))

    assert counts == {"frames": 2, "annotations": 2, "pairs": 1}
    mask_files = [str(value) for value in graph.objects(None, api.HW1.maskFile)]
    # Two immediate frame factors x two frames + two immediate pair factors.
    # PriorWarpDepthResidual is intentionally deferred to reconstruction.
    assert len(mask_files) == 6
    assert all((tmp_path / path).is_file() for path in mask_files)
    pair = next(graph.subjects(RDF.type, api.HW1.FramePair))
    assert int(graph.value(pair, api.HW1.identityMedianDepthChangePixelCount)) == 48
    assert int(graph.value(pair, api.HW1.jointValidDepthPixelCount)) == 48
    assert graph.value(pair, api.HW1.priorWarpDepthResidual) is None


def test_exported_drop_mask_changes_point_count(tmp_path):
    depth = np.full((4, 4), 2.0)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    drop = np.zeros((4, 4), dtype=np.uint8)
    drop[0, :3] = 255
    mask_dir = tmp_path / "HighFrequencyDepthResidual"
    mask_dir.mkdir()
    Image.fromarray(drop).save(mask_dir / "0.png")

    index = utils._mask_index(str(mask_dir))
    keep = utils._keep_mask_for_frame(index, 0, depth.shape)
    cloud = utils.depth_image_to_point_cloud(
        rgb, depth, width=4, height=4, hfov=90.0, keep_mask=keep)
    assert len(cloud.points) == 13
    assert cv2.imread(str(mask_dir / "0.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint8


def test_deferred_prior_warp_round_trip_and_masked_run_provenance(tmp_path):
    capture = tmp_path / "capture"
    _write_capture(capture)
    api.main(["batch2ttl", "--data-dir", str(capture), "--floor", "1"])

    experiment = tmp_path / "prior_test.ttl"
    experiment.write_text(
        "@prefix hw1: <http://taica.course/hw1/ontology#> .\n"
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
        "<http://taica.course/hw1/ontology#experiment/prior_test>\n"
        "    a hw1:Experiment ;\n"
        f"    hw1:batchFile \"{capture / 'batch.ttl'}\" ;\n"
        "    hw1:onBatch <http://taica.course/hw1/ontology#batch/floor1_capture> ;\n"
        "    hw1:evaluatesFactor hw1:PriorWarpDepthResidual .\n",
        encoding="utf-8")
    api.main(["experiment", str(experiment)])

    before = api.read_experiment(str(experiment))
    assert "PriorWarpDepthResidual" not in before["mask_files"]
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 255
    api.write_pair_measurements(
        str(experiment), "PriorWarpDepthResidual",
        [{"source": 0, "target": 1, "value": 0.2,
          "count": 64, "mask": mask}])

    after = api.read_experiment(str(experiment))
    assert after["pair_status"][(0, 1)] is False
    mask_ref = after["mask_files"]["PriorWarpDepthResidual"]["pairs"][(0, 1)]
    assert (tmp_path / mask_ref).is_file()

    api.write_run(
        str(experiment), "masked", {"mapMeanL2": 0.1}, frame_count=2,
        mask_factor="PriorWarpDepthResidual",
        diagnostic_file="prior_test/diagnostics/masked.json")
    final = api.read_experiment(str(experiment))["graph"]
    run = api.run_iri("prior_test", "masked")
    assert final.value(run, api.HW1.selectionMode) == api.HW1.MaskFiltered
    assert final.value(run, api.HW1.maskFactor) == api.HW1.PriorWarpDepthResidual
    assert str(final.value(run, api.HW1.diagnosticFile)).endswith("masked.json")


def test_smoke_reconstruction_exports_per_link_diagnostics():
    smoke = HW1_DIR / "tests" / "fixtures" / "smoke"
    _cloud, pred, _gt, diagnostics = utils.reconstruct(
        str(smoke), build_cloud=False, verbose=False,
        return_diagnostics=True, collect_prior_warp=True)
    assert len(pred) == 5
    assert diagnostics["schema_version"] == 1
    assert len(diagnostics["links"]) == 4
    assert len(diagnostics["prior_warp_measurements"]) == 4
    required = {
        "source", "target", "prior", "pre_gate_transform",
        "applied_transform", "gate_fired", "fitness", "inlier_rmse_m",
        "correspondence_count", "rpe_translation_m", "rpe_rotation_deg",
        "drift_increment_m",
    }
    assert required <= set(diagnostics["links"][0])


def test_hf_mask_recovers_smoke_trajectory_from_sparse_depth_impulses(tmp_path):
    """The mask path is an ICP intervention, not only a point-count option."""
    smoke = HW1_DIR / "tests" / "fixtures" / "smoke"
    capture = tmp_path / "capture"
    shutil.copytree(smoke, capture)
    mask_dir = tmp_path / "HighFrequencyDepthResidual"
    mask_dir.mkdir()

    rng = np.random.default_rng(19)
    for path in sorted((capture / "depth").glob("*.png")):
        raw = np.asarray(Image.open(path)).copy()
        corrupted = rng.random(raw.shape) < 0.03
        offsets = rng.choice(np.array([-600, 600]), size=raw.shape)
        noisy = np.clip(
            raw.astype(np.int32) + corrupted * offsets, 1, 65535
        ).astype(np.uint16)
        Image.fromarray(noisy).save(path)
        mask = api.frame_high_frequency_depth_residual_mask(str(path), 5.0)
        Image.fromarray(mask).save(mask_dir / path.name)

    def score(root, masks=None):
        _cloud, pred, gt = utils.reconstruct(
            str(root), build_cloud=False, verbose=False,
            mask_root=None if masks is None else str(masks))
        return utils.mean_l2(pred, gt)

    clean_score = score(smoke)
    noisy_score = score(capture)
    masked_score = score(capture, mask_dir)

    assert noisy_score > clean_score * 2.0
    assert masked_score < noisy_score * 0.5
    assert masked_score < clean_score * 1.3
