"""Saved-state media remains bounded, complete, and independent of outcomes."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase0_media", ROOT / "scripts/task10b_phase0_media.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def record():
    return dict(schema=media.SCHEMA, physics_integration=False, learning=False, trial_id=0,
        fps=50, playback_speeds=[1.0], allocation_wall_seconds=1800, application_wall_seconds=1680,
        products_and_caches_bytes=2 * 1024**3, selected_conditions=media.selection(),
        excluded_nodes=["node3006"], clips=media.selection())


def test_all_matched_trial_zero_conditions_are_predeclared():
    rows = media.selection()
    assert len(rows) == len({row["condition"] for row in rows}) == 16
    assert {row["trial_id"] for row in rows} == {0}
    assert {row["family"] for row in rows} == set(media.FAMILIES)
    media.check_scope(record())


@pytest.mark.parametrize("key,value", [("trial_id", 1), ("fps", 30), ("playback_speeds", [1., .5]),
    ("physics_integration", True), ("learning", True), ("allocation_wall_seconds", 1801), ("clips", [])])
def test_changed_media_scope_rejected(key, value):
    data = record()
    data[key] = value
    with pytest.raises(ValueError):
        media.check_scope(data)


def test_rendered_clip_cannot_change_the_predeclared_world():
    data = record()
    data["clips"][0]["trial_id"] = 1
    with pytest.raises(ValueError, match="Clip identities"):
        media.check_scope(data)


@pytest.mark.parametrize("world,steps", [(0, 4), (1, 4), (0, 8)])
def test_only_original_horizon_and_predeclared_world(tmp_path, world, steps):
    path = tmp_path / "rollout.npz"
    np.savez(path, t=np.arange(5) * .005, qpos=np.zeros((5, 55)), qvel=np.zeros((5, 54)),
             physics_step_count=4, evaluation_bank_world=world)
    if world == 0 and steps == 4:
        result = media.recorded_horizon(path, steps)
        assert result["source_state_count"] == 5 and result["expected_video_frames"] == 2
    else:
        with pytest.raises(ValueError):
            media.recorded_horizon(path, steps)


def test_media_helper_has_no_physics_integration_calls():
    parsed = ast.parse((ROOT / "scripts/task10b_phase0_media.py").read_text())
    calls = [node.func.attr for node in ast.walk(parsed) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)]
    assert not {"mj_step", "mj_step1", "mj_step2", "step"}.intersection(calls)


def test_prepare_hash_binds_all_sixteen_saved_trials_without_native_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "ROOT", tmp_path)
    helper = tmp_path / "media_helper.py"
    helper.write_text("# synthetic source identity\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("synthetic interpreter identity\n")
    monkeypatch.setattr(media, "SOURCES", ("media_helper.py",))
    monkeypatch.setattr(media, "PYTHON", interpreter)
    monkeypatch.setattr(media, "writable_path", lambda path: Path(path).resolve())
    def write(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("x") as stream:
            json.dump(value, stream)
    monkeypatch.setattr(media, "write_json", write)
    source = tmp_path / "outputs/task10b/parity/packet.json"
    source.parent.mkdir(parents=True)
    parent = dict(phase="parity", output="outputs/task10b/parity/run", release="release")
    source.write_text(json.dumps(parent))
    conditions = []
    for chosen in media.selection():
        folder = source.parent / "run/parity" / chosen["condition"] / "historical_evaluation"
        folder.mkdir(parents=True)
        model = folder / "historical_evaluation_runtime.xml"
        model.write_text("synthetic playback model identity\n")
        rollout = folder / "rollout_00.npz"
        np.savez(rollout, t=np.arange(5) * .005, qpos=np.zeros((5, 55)), qvel=np.zeros((5, 54)),
                 physics_step_count=4, evaluation_bank_world=0)
        reference = source.parent / "source_snapshot/release" / chosen["family"] / "reference.npz"
        if not reference.exists():
            reference.parent.mkdir(parents=True)
            np.savez(reference, head_center=np.array([1., 0., 1.5]))
        trial = dict(trial_id=0, strict_historical_release_pass=False, failed_checks=["synthetic_failure"],
                     rollout_sha256=media.sha256_file(rollout))
        summary = dict(trials=[trial], runtime_model_sha256=media.sha256_file(model),
                       source_inputs=dict(reference=media.sha256_file(reference)), historical_physics_steps=4)
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "trial_00.json").write_text(json.dumps(trial))
        conditions.append(dict(name=chosen["condition"], status="passed", historical_evaluation=summary))
    audit = source.parent / "run/parity/audit.json"
    audit.write_text(json.dumps(dict(status="passed", mode="parity", conditions=conditions, unattempted_conditions=0)))
    (source.parent / "execution.json").write_text(json.dumps(dict(status="passed", completed_modes=["parity"],
        packet_sha256=media.sha256_file(source), modes=[dict(audit_sha256=media.sha256_file(audit))])))
    destination = tmp_path / "outputs/task10b/media"
    result = media.prepare(source, destination)
    assert result["clips"] == 16 and result["planned_frames"] == 32
    packet = media.verify(destination / "packet.json")
    assert all(not clip["source_strict_historical_pass"] for clip in packet["clips"])
    assert "--exclude=node3006" in packet["submission_command"]
    assert "--time=00:30:00" in packet["submission_command"]
    with pytest.raises(FileExistsError):
        media.prepare(source, destination)
    (tmp_path / packet["clips"][0]["rollout"]).write_bytes(b"changed playback")
    with pytest.raises(ValueError, match="Frozen media input changed"):
        media.verify(destination / "packet.json")
