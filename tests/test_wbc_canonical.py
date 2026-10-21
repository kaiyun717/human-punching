"""Canonical packaging preserves actual source trajectories; no CPU dynamics."""
import json

import mujoco
import numpy as np
import pytest

from human_punching.wbc.canonical import (
    SELECTION_SCHEMA, _verify_phase_evidence, assemble_canonical_bundle,
    write_presampled_reference,
)
from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.references import sha256
from human_punching.wbc.runtime.bundle import load_runtime_bundle


def record(path):
    return {"path": str(path.resolve().relative_to(PROJECT_ROOT)), "sha256": sha256(path)}


@pytest.fixture(scope="module")
def source_bundle(task10_common_bundle):
    return task10_common_bundle


@pytest.fixture
def selection(source_bundle, tmp_path):
    manifest = source_bundle.root/"manifest.json"
    path = tmp_path/"selection.json"
    path.write_text(json.dumps({"schema": SELECTION_SCHEMA, "references": [
        {"family": c["family"], "reference_id": "canonical_"+c["family"],
         "parent_reference_id": c["reference_id"],
         "parent_manifest": record(manifest), "phase_evidence": record(manifest),
         "source_master": record(source_bundle.path(c["master_path"])),
         "outward_end_s": c["outward_end_s"], "guard_time_s": c["guard_time_s"]}
        for c in source_bundle.clips]}))
    return path


def assemble(selection, destination):
    return assemble_canonical_bundle(selection, output_root=destination,
        bank_root=PROJECT_ROOT.parent/"robust_sav/constants/boxing/snapshots/v1_20260921",
        g1_asset_root=PROJECT_ROOT/".cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1",
        nominal_plant_path=PROJECT_ROOT.parent/"robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json")


def test_four_family_conversion_preserves_sources_and_exact_views(source_bundle, selection, tmp_path, monkeypatch):
    def no_step(*args, **kwargs):
        pytest.fail("Canonical conversion must never integrate physics")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, no_step)
    destination = tmp_path/"canonical"
    result = assemble(selection, destination)
    assert result["families"] == 4 and result["qualified_for_training"] is False
    bundle = load_runtime_bundle(destination)
    with pytest.raises(ValueError):
        bundle.require_training_qualification()
    for old, converted in zip(source_bundle.clips, bundle.clips, strict=True):
        with np.load(source_bundle.path(old["master_path"]), allow_pickle=False) as source, \
             np.load(bundle.path(converted["master_path"]), allow_pickle=False) as master, \
             np.load(bundle.path(converted["motion_path"]), allow_pickle=False) as view:
            for key in source.files:
                np.testing.assert_array_equal(source[key], master[key], err_msg=key)
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                np.testing.assert_array_equal(view[key], master[key][::4])
            assert view["fps"] == 50 and not np.any(master["joint_feedforward"])
            assert master["outward_end_s"] == converted["outward_end_s"]
        assert converted["launch_snapshot"] == old["launch_snapshot"]
        assert converted["launch_features"] == old["launch_features"]
        assert converted["qualified_for_training"] is False
    report = json.loads((destination/"assembly_report.json").read_text())
    assert report["physics_executed"] is report["optimization_executed"] is False
    assert report["domain_qualified"] is report["guard_stability_qualified"] is False
    assert len(list((destination/"provenance/snapshots").glob("*/payload.npz"))) == 1
    with pytest.raises(FileExistsError):
        assemble(selection, destination)


def test_phase_evidence_binds_exact_source_and_unrounded_endpoints(selection):
    row = json.loads(selection.read_text())["references"][0]
    _verify_phase_evidence(row)
    row["outward_end_s"] += 1e-8
    with pytest.raises(ValueError, match="phase endpoints"):
        _verify_phase_evidence(row)
    row["source_master"]["sha256"] = "f"*64
    with pytest.raises(ValueError, match="exact selected master"):
        _verify_phase_evidence(row)


@pytest.mark.parametrize("mutation", ["clock", "velocity", "fk", "phase"])
def test_presampled_arrays_are_verified_not_silently_repaired(source_bundle, tmp_path, mutation):
    clip = source_bundle.clips[0]
    with np.load(source_bundle.path(clip["master_path"]), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if mutation == "clock":
        arrays["time_s"][1] += 1e-6
    elif mutation == "velocity":
        arrays["qvel"][5, 7] += .01
    elif mutation == "fk":
        arrays["body_pos_w"][5, 0, 0] += .01
    else:
        arrays["cycle_phase_id"][1] = 2
    source = tmp_path/"changed.npz"
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="clock|qvel|field differs"):
        write_presampled_reference(source, source_bundle, clip["reference_id"],
            output_root=tmp_path/"converted", reference_id="new", outward_end_s=clip["outward_end_s"],
            guard_time_s=clip["guard_time_s"], expected_source_sha256=sha256(source))
    assert not (tmp_path/"converted").exists()


def test_hash_gate_precedes_source_loading(source_bundle, tmp_path):
    clip = source_bundle.clips[0]
    with pytest.raises(ValueError, match="source master hash"):
        write_presampled_reference(source_bundle.path(clip["master_path"]), source_bundle,
            clip["reference_id"], output_root=tmp_path/"converted", reference_id="new",
            outward_end_s=clip["outward_end_s"], guard_time_s=clip["guard_time_s"],
            expected_source_sha256="f"*64)


def test_selection_rejects_incomplete_or_duplicate_family_before_snapshot_access(selection, tmp_path):
    data = json.loads(selection.read_text())
    data["references"][1]["family"] = data["references"][0]["family"]
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="one identified reference per family"):
        assemble(selection, tmp_path/"canonical")
