import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.candidates import CandidateSpec, build_candidate_bundle, frozen_snapshot
from human_punching.wbc.references import PROJECT_ROOT, FAMILIES
from human_punching.wbc.runtime.bundle import load_runtime_bundle
from human_punching.wbc.snapshots import SnapshotBank, load_verified_g1_model

BANK = PROJECT_ROOT.parent / "robust_sav/constants/boxing/snapshots/v1_20260921"
ASSETS = PROJECT_ROOT / ".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1"
PLANT = PROJECT_ROOT.parent / "robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json"
RELEASE = PROJECT_ROOT / "outputs/four-punch-milestone"


@pytest.fixture(scope="module")
def actual_state():
    bank = SnapshotBank(BANK)
    model, _ = load_verified_g1_model(ASSETS, bank, PLANT)
    state = bank.state(int(bank.eligible_ids("train")[0]))
    return state, model


def test_recipe_identity_and_search_parameters_are_explicit():
    spec = CandidateSpec(4, "jb", .8, .1, .7)
    assert spec.recipe_id == CandidateSpec(4, "jab", .8, .1, .7).recipe_id
    assert spec.recipe_id != replace(spec, ttc_s=.75).recipe_id
    for values in ((True, "jab", .8, .1, .7), (4, "jab", np.nan, .1, .7),
                   (4, "jab", .8, .1, 0)):
        with pytest.raises(ValueError):
            CandidateSpec(*values)


def test_actual_snapshot_placement_preserves_head_variation_and_velocity(actual_state):
    state, model = actual_state
    spec = CandidateSpec(state.slot_id, "jab", .8, .1, .7)
    frozen = frozen_snapshot(state, model, spec)
    np.testing.assert_allclose(frozen.relative_placement(), [.8, .1], atol=1e-12)
    assert frozen.g1_qpos == state.qpos and frozen.g1_qvel == state.qvel
    assert frozen.head_position_w[2] > 1.20
    assert np.linalg.norm(frozen.head_velocity_w) > 0
    assert frozen.human_origin_w[2] == 0
    # Changing placement moves the human, not the frozen real head or its velocity.
    other = frozen_snapshot(state, model, replace(spec, lateral_m=-.1))
    assert frozen.head_position_w == other.head_position_w
    assert frozen.head_velocity_w == other.head_velocity_w
    np.testing.assert_allclose(np.linalg.norm(np.array(frozen.human_origin_w)-other.human_origin_w), .2)
    with pytest.raises(ValueError, match="matching"):
        frozen_snapshot(state, model, replace(spec, snapshot_slot_id=state.slot_id+1))


def test_invalid_bundles_fail_before_writing(tmp_path, actual_state):
    state, _ = actual_state
    spec = CandidateSpec(state.slot_id, "jab", .8, 0, .7)
    destination = tmp_path / "should-not-exist"
    kwargs = dict(release_root=RELEASE, bank_root=BANK, g1_asset_root=ASSETS,
                  nominal_plant_path=PLANT, output_root=destination)
    with pytest.raises(ValueError, match="four"):
        build_candidate_bundle([spec], **kwargs)
    with pytest.raises(ValueError, match="Duplicate"):
        build_candidate_bundle([spec, spec], **kwargs)
    assert not destination.exists()


def test_actual_four_family_bundle_keeps_failures_and_never_qualifies(tmp_path, actual_state, monkeypatch):
    state, _ = actual_state
    def forbidden(*args, **kwargs):
        raise AssertionError("Candidate generation must not integrate dynamics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    times = (.70, .70, 1.10, .95)
    specs = [CandidateSpec(state.slot_id, family, .8, 0, ttc)
             for family, ttc in zip(FAMILIES, times, strict=True)]
    result = build_candidate_bundle(specs, release_root=RELEASE, bank_root=BANK,
        g1_asset_root=ASSETS, nominal_plant_path=PLANT, output_root=tmp_path / "bundle",
        ik_iterations=8, guard_iterations=40)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    report = json.loads(Path(result["report_path"]).read_text())
    assert not result["qualified_for_training"] and not manifest["qualified_for_training"]
    assert not report["physics_executed"]
    assert report["requested_count"] == 4
    assert report["generated_count"] == 4
    assert report["all_basic_command_screens_passed"]
    assert not report["all_kinematic_screens_passed"]  # Known selected self collisions.
    assert len(report["requests"]) == 4
    assert len(report["guards"]) == 1
    for request in report["requests"]:
        assert request["snapshot_provenance"]["payload_sha256"] == state.payload_sha256
        assert request["snapshot_provenance"]["source_group_id"] == state.source_group_id
        assert request["launch_snapshot"]["g1_qpos"] == list(state.qpos)
        assert request["status"] == "generated"
        assert request["derivative_diagnostics"]["peak_joint_speed_rad_s"] > 0
        assert "geometry_screen" in request
    for clip in manifest["clips"]:
        assert clip["launch_time_s"] == 0
        assert clip["requested_ttc_s"] == next(s.ttc_s for s in specs if s.family == clip["family"])
        assert clip["feedforward_mode"] == "disabled_zero"
        assert clip["guard_source"] == "explicit_common_guard"
        assert len(clip["launch_features"]) == 77
        assert not clip["common_guard_validated"]
        root = Path(result["manifest_path"]).parent
        with np.load(root / clip["master_path"], allow_pickle=False) as arrays:
            np.testing.assert_allclose(np.diff(arrays["time_s"]), .005, atol=1e-12)
            assert np.all(arrays["joint_feedforward"] == 0)
            np.testing.assert_array_equal(arrays["qpos"][0], arrays["qpos"][-1])
    if result["all_families_generated"]:
        bundle = load_runtime_bundle(result["manifest_path"])
        with pytest.raises(ValueError, match="unqualified"):
            bundle.require_training_qualification()
    else:
        with pytest.raises(ValueError, match="four"):
            load_runtime_bundle(result["manifest_path"])


def test_outside_bounds_request_is_recorded_not_clamped(tmp_path, actual_state):
    state, _ = actual_state
    specs = [CandidateSpec(state.slot_id, family, 2., 0, ttc)
             for family, ttc in zip(FAMILIES, (.7, .7, 1.1, .95), strict=True)]
    result = build_candidate_bundle(specs, release_root=RELEASE, bank_root=BANK,
        g1_asset_root=ASSETS, nominal_plant_path=PLANT, output_root=tmp_path / "rejected")
    report = json.loads(Path(result["report_path"]).read_text())
    assert result["generated_count"] == 0
    assert all(row["reason"] == "outside_search_bounds" for row in report["requests"])
    assert all(row["spec"]["forward_m"] == 2. for row in report["requests"])
    assert not report["all_kinematic_screens_passed"]


def test_optimizer_failure_is_not_proof_of_infeasibility(tmp_path, actual_state, monkeypatch):
    import human_punching.wbc.candidates as candidates
    state, _ = actual_state
    def failed(*args, **kwargs):
        raise ValueError("test optimizer did not converge")
    monkeypatch.setattr(candidates, "retarget_candidate", failed)
    specs = [CandidateSpec(state.slot_id, family, .8, 0, ttc)
             for family, ttc in zip(FAMILIES, (.7, .7, 1.1, .95), strict=True)]
    result = build_candidate_bundle(specs, release_root=RELEASE, bank_root=BANK,
        g1_asset_root=ASSETS, nominal_plant_path=PLANT, output_root=tmp_path / "failed",
        guard_iterations=40)
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["generated_count"] == 0
    assert all(row["status"] == "planner_failed" for row in report["requests"])
    assert all(row["proves_physical_infeasibility"] is False for row in report["requests"])
