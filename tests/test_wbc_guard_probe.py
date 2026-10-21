"""CPU-only guards, FK fixtures and instrumentation; never physical evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from human_punching.wbc.cycle_trace import TraceBindings, derive_kinematics
from human_punching.wbc.guard_probe import (
    _guard_command_types, audit_static_drive, build_guard_probe_cfg, common_guard_identity, evaluate_guard_trace,
)
from human_punching.wbc.native_cycle_probe import ProbeMetrics
from human_punching.wbc.provenance import configure_local_caches
from human_punching.wbc.runtime.bundle import load_runtime_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle(task10_common_bundle):
    return task10_common_bundle


def test_real_four_family_common_guard_identity(bundle):
    identity = common_guard_identity(bundle)
    assert identity["maximum_source_qpos_difference"] < 1e-9
    assert len(identity["qpos"]) == 55 and identity["qvel"] == [0.]*54
    assert identity["unique_guard_configurations"] == 1
    assert not identity["independent_guard_replicates"]
    assert identity["feedforward_mode"] == "zero_joint_feedforward"


def test_family_specific_guard_is_rejected(bundle):
    clips = tuple(dict(clip, guard_source="task09_family_guard") for clip in bundle.clips)
    with pytest.raises(ValueError, match="explicit common guard"):
        common_guard_identity(replace(bundle, clips=clips))


def test_guard_command_holds_frame_zero_and_zeroes_all_derivatives_and_feedforward():
    configure_local_caches()
    import torch
    cls, _ = _guard_command_types()
    command = object.__new__(cls)
    command.master_starts = torch.tensor([0, 3])
    command.trajectory_ids = torch.tensor([0, 1])
    command.physics_steps = torch.tensor([71, 999])
    command.master = {name: torch.arange(6*width).reshape(6, width).float()
        for name, width in (("qpos", 55), ("qvel", 54), ("joint_feedforward", 48),
                            ("body_lin_vel_w", 3), ("body_ang_vel_w", 3),
                            ("anchor_lin_acc_w", 3), ("anchor_ang_acc_w", 3))}
    for offset in (0, 1, 2000, -3):
        np.testing.assert_array_equal(command.sample_master("qpos", offset), command.master["qpos"][[0, 3]])
        for name in set(command.master)-{"qpos"}:
            assert not command.sample_master(name, offset).any()
    np.testing.assert_array_equal(command.physics_steps, [71, 999])


def test_guard_config_keeps_native_clocks_and_blocks_resets(bundle):
    configure_local_caches()
    cfg = build_guard_probe_cfg(bundle, ProbeMetrics())
    assert type(cfg.commands["motion"]).__name__ == "StaticGuardCommandCfg"
    assert cfg.scene.num_envs == 16 and cfg.decimation == 4 and cfg.sim.mujoco.timestep == .005
    assert not cfg.events and not cfg.terminations and not cfg.rewards and not cfg.auto_reset
    assert not cfg.commands["motion"].assistive_wrench_enabled


def test_explicit_static_feedforward_is_constant_without_changing_derivatives(bundle):
    configure_local_caches()
    import torch
    cls, _ = _guard_command_types()
    command = object.__new__(cls)
    values = np.linspace(-1., 1., 48)
    command.cfg = SimpleNamespace(static_guard_feedforward_nm=tuple(values))
    command.master_starts = torch.tensor([0, 3])
    command.trajectory_ids = torch.tensor([0, 1])
    command.master = {"joint_feedforward": torch.ones(6, 48)*999., "qvel": torch.ones(6, 54)}
    for offset in (0, 1, 2000):
        np.testing.assert_allclose(command.sample_master("joint_feedforward", offset),
                                   np.broadcast_to(values, (2, 48)), atol=1e-7)
        assert not command.sample_master("qvel", offset).any()
    cfg = build_guard_probe_cfg(bundle, ProbeMetrics(), feedforward=values)
    assert cfg.commands["motion"].static_guard_feedforward_nm == tuple(values)
    with pytest.raises(ValueError, match="48 finite capped"):
        build_guard_probe_cfg(bundle, ProbeMetrics(), feedforward=np.asarray(bundle.body["effort"])*2)


def test_independent_static_drive_checks_feedforward_and_never_torque_clips_drive(bundle):
    target = np.broadcast_to(np.asarray(common_guard_identity(bundle)["qpos"])[7:], (2, 3, 48)).copy()
    ff = np.linspace(-2., 2., 48)
    native = {"joint_position_target": target, "joint_velocity_target": np.zeros_like(target),
              "joint_effort_target": np.broadcast_to(ff, target.shape).copy(),
              "ctrl": np.asarray(bundle.body["kp"])*target+ff}
    audit = audit_static_drive(native, bundle, ff)
    assert audit["status"] == "passed" and audit["maximum_drive_error_nm"] == 0.
    native["joint_effort_target"][1, 0, 3] += .01
    assert audit_static_drive(native, bundle, ff)["status"] == "failed"
    native["joint_effort_target"][1, 0, 3] -= .01
    native["ctrl"] = np.clip(native["ctrl"], -np.asarray(bundle.body["effort"]), np.asarray(bundle.body["effort"]))
    assert audit_static_drive(native, bundle, ff)["status"] == "failed"


@pytest.fixture
def trace(bundle):
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    guard = common_guard_identity(bundle)
    q = np.asarray(guard["qpos"])[None]
    v = np.zeros((1, model.nv))
    fk = derive_kinematics(model, q, v,
        TraceBindings(prefix="", floor_geoms=("floor",), human_geom_prefix="physical/"))
    a = {key: np.repeat(value, 1201, axis=0) for key, value in fk.items()}
    names = bundle.joint_names
    joint_ids = [model.joint(name).id for name in names]
    a.update(time_s=np.arange(1201)*.005,
        joint_position=np.repeat(q[:, model.jnt_qposadr[joint_ids]], 1201, axis=0),
        joint_velocity=np.zeros((1201, 48)), joint_limits=model.jnt_range[joint_ids].copy(),
        effort=np.zeros((1201, 48)), effort_limits=np.asarray(bundle.body["effort"]),
        support_contact=np.ones((1201, 2), bool), support_patch_contact=np.ones((1201, 2), bool),
        external_wrench=np.zeros((1201, model.nbody, 6)), external_generalized_force=np.zeros((1201, 54)),
        root_actuator_force=np.zeros((1201, 6)), external_support=np.zeros(1201, bool),
        reset_event=np.r_[True, np.zeros(1200, bool)], numerical_failure=np.zeros(1201, bool),
        contact_overflow=np.zeros(1201, bool), forbidden_collision=np.zeros(1201, bool),
        human_g1_collision=np.zeros(1201, bool), evidence_kind="synthetic_guard_fixture")
    return a


def assess(trace, bundle, native=False):
    return evaluate_guard_trace(trace, bundle.clips[0]["target_w"], native_contract_passed=native)


def test_synthetic_standing_guard_never_becomes_physical_admission(trace, bundle):
    result = assess(trace, bundle)
    assert not result["guard_admission_passed"]
    assert not result["punch_execution_tested"] and not result["ttc_tested"]
    assert result["checks"]["last_second_double_support"]
    assert result["checks"]["last_second_glove_guard"]
    assert not result["checks"]["native_guard_hold_evidence"]
    assert not result["qualified_for_training"]


@pytest.mark.parametrize("field,mutation,failed", [
    ("support_contact", lambda a: a.__setitem__((-1, 0), False), "last_second_double_support"),
    ("root_linear_velocity_w", lambda a: a.__setitem__((1001, 0), .051), "last_second_root_quiet"),
    ("joint_velocity", lambda a: a.__setitem__((1100, 0), .501), "last_second_joints_quiet"),
    ("glove_position_guard_frame", lambda a: a.__setitem__((1100, 0, 0), a[1100, 0, 0]+.031), "last_second_glove_guard"),
    ("support_anchor_position_w", lambda a: a.__setitem__((500, 0, 0), a[500, 0, 0]+.031), "static_footprint_all_times"),
    ("reset_event", lambda a: a.__setitem__(100, True), "one_initial_reset"),
    ("forbidden_collision", lambda a: a.__setitem__(10, True), "no_forbidden_collision"),
    ("contact_overflow", lambda a: a.__setitem__(30, True), "no_contact_overflow"),
])
def test_one_substep_violation_is_retained(trace, bundle, field, mutation, failed):
    mutation(trace[field])
    result = assess(trace, bundle)
    assert failed in result["failure_categories"]
    assert not result["guard_admission_passed"]


def test_opposing_body_assistance_cannot_cancel(trace, bundle):
    trace["external_wrench"][600, 1, 2] = 10.
    trace["external_wrench"][600, 2, 2] = -10.
    result = assess(trace, bundle)
    assert not result["checks"]["zero_external_body_wrenches"]


def test_missing_truncated_and_float_contact_evidence_fails_closed(trace, bundle):
    missing = dict(trace)
    missing.pop("effort")
    assert assess(missing, bundle)["failure_categories"] == ["missing_evidence"]
    truncated = dict(trace, time_s=trace["time_s"][:-1])
    assert assess(truncated, bundle)["failure_categories"] == ["incomplete_physics_clock"]
    malformed = dict(trace, support_contact=trace["support_contact"].astype(float))
    assert assess(malformed, bundle)["failure_categories"] == ["malformed_evidence"]


def test_force_and_joint_caps_are_checked_over_entire_six_seconds(trace, bundle):
    trace["effort"][400, 3] = trace["effort_limits"][3]+.01
    trace["joint_position"][300, 1] = trace["joint_limits"][1, 1]+.021
    result = assess(trace, bundle)
    assert not result["checks"]["effort_caps"]
    assert not result["checks"]["joint_limits"]


def test_initial_unloaded_contact_is_descriptive_during_physical_settling(trace, bundle):
    baseline = assess(trace, bundle)
    trace["support_contact"][:3] = False
    trace["support_patch_contact"][:3] = False
    result = assess(trace, bundle)
    assert result["checks"] == baseline["checks"]
    assert not result["metrics"]["initial_double_support"]
    assert not result["metrics"]["initial_support_is_admission_gate"]
    assert result["checks"]["last_second_double_support"]
    # A later unsupported sample in the quiet window still fails qualification.
    trace["support_contact"][-1] = False
    assert not assess(trace, bundle)["checks"]["last_second_double_support"]
