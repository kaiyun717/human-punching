"""CPU contract tests only: no simulation, integration, training or GPU allocation."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching.wbc.native_cycle_probe import (
    ProbeMetrics, ProbeWriteLedger, build_cycle_probe_cfg, deadline_prefix,
    deterministic_clip_ids, initialize_launch_drive, load_phase_endpoints,
    placed_request, reference_diagnostic,
)
from human_punching.wbc.provenance import configure_local_caches, sha256_file
from human_punching.wbc.runtime.bundle import load_runtime_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle(task10_common_bundle):
    return task10_common_bundle


@pytest.fixture(scope="module")
def phases(bundle):
    return load_phase_endpoints(bundle)


def test_round_robin_is_family_balanced_and_order_independent(bundle):
    ids = deterministic_clip_ids(bundle.clips, np.arange(16))
    assert [bundle.clips[i]["family"] for i in ids] == ["jab", "cross", "left_hook", "right_hook"]*4
    reversed_clips = tuple(reversed(bundle.clips))
    reverse_ids = deterministic_clip_ids(reversed_clips, np.arange(16))
    assert [reversed_clips[i]["reference_id"] for i in reverse_ids] == [bundle.clips[i]["reference_id"] for i in ids]
    variants = [dict(c, reference_id=c["reference_id"]+suffix) for suffix in ("_z", "_a") for c in bundle.clips]
    chosen = deterministic_clip_ids(variants, np.arange(8))
    assert all(variants[i]["reference_id"].endswith("_a") for i in chosen[:4])
    assert all(variants[i]["reference_id"].endswith("_z") for i in chosen[4:])


def test_actual_launch_payload_and_reference_translate_together(bundle, phases):
    origin = np.array([13., -4., 0.])
    for clip in bundle.clips:
        original = placed_request(clip, np.zeros(3), 0.)
        shifted = placed_request(clip, origin, 0.)
        assert original.snapshot.snapshot_id == shifted.snapshot.snapshot_id
        np.testing.assert_allclose(np.asarray(shifted.snapshot.g1_qpos[:3])-original.snapshot.g1_qpos[:3], origin)
        np.testing.assert_allclose(shifted.snapshot.world_to_human_point(shifted.snapshot.head_position_w),
                                   original.snapshot.world_to_human_point(original.snapshot.head_position_w), atol=1e-12)
        assert shifted.snapshot.head_velocity_w == original.snapshot.head_velocity_w
        a = reference_diagnostic(bundle, clip, 1201, np.zeros(3), phases[clip["reference_id"]])
        b = reference_diagnostic(bundle, clip, 1201, origin, phases[clip["reference_id"]])
        np.testing.assert_allclose(b-a, np.broadcast_to(origin, b.shape), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(b-np.asarray(shifted.snapshot.head_position_w), axis=-1),
                                   np.linalg.norm(a-np.asarray(original.snapshot.head_position_w), axis=-1), atol=1e-12)
        np.testing.assert_array_equal(b[-1], b[-200])  # final guard held beyond clip end


def test_missing_or_mutated_snapshot_fails_closed(bundle):
    clip = dict(bundle.clips[0])
    clip.pop("launch_snapshot")
    with pytest.raises(ValueError, match="actual launch"):
        placed_request(clip, np.zeros(3), 0.)
    clip = dict(bundle.clips[0], target_w=[0., 0., 0.])
    with pytest.raises(ValueError, match="target differs"):
        placed_request(clip, np.zeros(3), 0.)
    clip = dict(bundle.clips[0], request_fingerprint="0"*64)
    with pytest.raises(ValueError, match="request fingerprint"):
        placed_request(clip, np.zeros(3), 0.)


def test_phase_file_is_bound_to_exact_master_and_manifest(bundle, phases, tmp_path):
    assert set(phases) == {c["reference_id"] for c in bundle.clips}
    without_phase = replace(bundle, clips=tuple({k:v for k,v in clip.items() if k != "outward_end_s"}
                                               for clip in bundle.clips))
    with pytest.raises(ValueError, match="exact reference outward"):
        load_phase_endpoints(without_phase)
    payload = {"schema": "human_wbc_reference_phases_v1",
        "bundle_manifest_sha256": sha256_file(bundle.root/"manifest.json"),
        "references": [dict(clip) for clip in bundle.clips]}
    valid_path = tmp_path/"valid_phase.json"
    valid_path.write_text(json.dumps(payload))
    assert load_phase_endpoints(without_phase, valid_path) == phases
    payload["references"][0]["master_sha256"] = "0"*64
    path = tmp_path / "bad_phase.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact reference outward"):
        load_phase_endpoints(bundle, path)
    payload["bundle_manifest_sha256"] = "0"*64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bound to this bundle"):
        load_phase_endpoints(bundle, path)


def test_write_ledger_records_initialization_and_rejects_later_mutation():
    ledger = ProbeWriteLedger(3)
    with pytest.raises(RuntimeError, match="exactly one"):
        ledger.seal()
    ledger.event("reset", np.arange(3))
    ledger.event("write", np.arange(3))
    ledger.seal()
    with pytest.raises(RuntimeError, match="after committed launch"):
        ledger.event("write", [1])
    np.testing.assert_array_equal(ledger.blocked, [False, True, False])
    np.testing.assert_array_equal(ledger.writes, [1, 2, 1])
    with pytest.raises(RuntimeError, match="after committed launch"):
        ledger.event("reset", [2])
    np.testing.assert_array_equal(ledger.resets, [1, 1, 2])


def test_deadline_prefix_preserves_full_trace_and_constants(bundle):
    request = placed_request(bundle.clips[0], np.zeros(3), 0.)
    times = np.arange(1201)*.005
    trace = dict(time_s=times, glove_position_w=np.zeros((1201, 2, 3)),
                 joint_limits=np.zeros((48, 2)), effort_limits=np.ones(48), physics_step_count=1200)
    panel = deadline_prefix(trace, request)
    assert len(trace["time_s"]) == 1201
    assert panel["time_s"][-1] <= request.ttc_s+4.+1e-10
    assert panel["physics_step_count"] == len(panel["time_s"])-1
    assert panel["joint_limits"] is trace["joint_limits"]
    assert panel["effort_limits"] is trace["effort_limits"]


def test_prelaunch_drive_initialization_never_advances_reference_or_state():
    import torch
    calls = []
    command = SimpleNamespace(joint_pos=torch.tensor([[3., -.2]]), joint_vel=torch.tensor([[.2, .1]]),
                              physics_steps=torch.tensor([0]), sample_master=lambda name: torch.tensor([[4., 5.]]))
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1., 1.], [-1., 1.]]])),
        set_joint_position_target=lambda value: calls.append(("position", value.clone())),
        set_joint_velocity_target=lambda value: calls.append(("velocity", value.clone())),
        set_joint_effort_target=lambda value: calls.append(("effort", value.clone())))
    class Scene(dict):
        def write_data_to_sim(self):
            calls.append(("write", None))
    state = SimpleNamespace(qpos=torch.zeros((1, 9)), qvel=torch.zeros((1, 8)), time=torch.zeros(1))
    env = SimpleNamespace(scene=Scene(robot=robot),
                          command_manager=SimpleNamespace(get_term=lambda name: command),
                          sim=SimpleNamespace(data=state, forward=lambda: calls.append(("forward", None))))
    report = initialize_launch_drive(env)
    assert [key for key, _ in calls] == ["position", "velocity", "effort", "write", "forward"]
    np.testing.assert_allclose(calls[0][1], [[1., -.2]])
    assert report["integration_steps"] == 0
    assert not report["physical_guard_settling"]
    assert command.physics_steps.item() == 0


def test_cfg_disables_all_midcycle_reset_and_assistance_paths(bundle):
    configure_local_caches()
    metrics = ProbeMetrics()
    cfg = build_cycle_probe_cfg(bundle, metrics)
    assert cfg.scene.num_envs == 16
    assert cfg.decimation == 4 and cfg.sim.mujoco.timestep == .005
    assert not cfg.events and not cfg.terminations and not cfg.rewards and not cfg.auto_reset
    assert not cfg.commands["motion"].assistive_wrench_enabled
    assert type(cfg.commands["motion"]).__name__ == "ProbeCommandCfg"
    assert cfg.metrics["cycle_trace"].per_substep
    cfg.metrics["cycle_trace"].func.reset(np.arange(16))
    np.testing.assert_array_equal(metrics.native.reset_counts, np.ones(16))


def test_compiled_model_saved_losslessly_without_dynamics(bundle, tmp_path):
    import mujoco
    from human_punching.wbc.native_cycle_probe import save_compiled_model
    from human_punching.wbc.provenance import sha256_file
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    model.opt.timestep = .005
    model.opt.iterations = 10
    path = tmp_path / "compiled.mjb"
    receipt = save_compiled_model(model, path)
    restored = mujoco.MjModel.from_binary_path(str(path))
    assert receipt["sha256"] == sha256_file(path)
    assert restored.opt.timestep == .005 and restored.opt.iterations == 10
    assert restored.nq == model.nq and restored.nv == model.nv
    np.testing.assert_array_equal(restored.actuator_biasprm, model.actuator_biasprm)
    np.testing.assert_array_equal(restored.actuator_forcerange, model.actuator_forcerange)
    np.testing.assert_array_equal(restored.body_inertia, model.body_inertia)
    assert [restored.geom(i).name for i in range(restored.ngeom)] == [model.geom(i).name for i in range(model.ngeom)]
    with pytest.raises(FileExistsError):
        save_compiled_model(model, path)
