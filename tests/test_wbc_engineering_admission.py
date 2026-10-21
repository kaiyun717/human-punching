"""Engineering scope/force-law tests; synthetic arrays are never native evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching.wbc import engineering_admission as module
from human_punching.wbc.engineering_admission import (
    EngineeringAdmission, SCHEMA, SCOPE, audit_controller_trace, engineering_runner,
    verify_engineering_admission,
)
from human_punching.wbc.measurement_contract import digest


def admission_fixture():
    payload = {"schema": SCHEMA, "scope": SCOPE, "qualified_for_training": False,
               "target_ttc_domain_qualified": False,
               "native_controller": {"report": {"path": "synthetic-not-native"}}}
    return EngineeringAdmission.from_dict({**payload, "sha256": digest(payload)})


def trace_fixture():
    body = {"kp": [220.]*48, "kd": [15.]*48, "effort": [35.]*48}
    target = np.full((4, 2, 48), 1.5)
    qpos = np.zeros((4, 2, 55)); qpos[..., 7:] = 1.49
    velocity = np.full((4, 2, 48), .2)
    qvel = np.zeros((4, 2, 54)); qvel[0, :, 6:] = -.2; qvel[1, :, 6:] = .3
    ctrl = 220*target+15*velocity
    force = np.zeros_like(target)
    force[1:] = np.clip(ctrl[1:]-220*qpos[:-1, :, 7:]-15*qvel[:-1, :, 6:], -35, 35)
    return body, {"joint_position_target": target, "joint_velocity_target": velocity,
        "joint_effort_target": np.zeros_like(target), "ctrl": ctrl, "actuator_force": force,
        "qpos": qpos, "qvel": qvel}


def test_affine_drive_is_not_torque_and_force_uses_preintegration_state():
    body, trace = trace_fixture()
    assert np.max(trace["ctrl"]) > body["effort"][0]
    assert audit_controller_trace(trace, body)["passed"]
    trace["actuator_force"][1:] = np.clip(trace["ctrl"][1:]-220*trace["qpos"][1:, :, 7:]
        -15*trace["qvel"][1:, :, 6:], -35, 35)
    result = audit_controller_trace(trace, body)
    assert not result["checks"]["native_force_from_preintegration_state"]


@pytest.mark.parametrize("name", ["ctrl", "joint_effort_target", "actuator_force"])
def test_controller_semantics_changes_are_rejected(name):
    body, trace = trace_fixture()
    trace[name][1, 0, 0] += 1.
    assert not audit_controller_trace(trace, body)["passed"]


def test_typed_admission_cannot_change_scope_or_claim_qualification():
    admission = admission_fixture()
    for key, value in (("scope", "production"), ("qualified_for_training", True),
                       ("target_ttc_domain_qualified", True)):
        changed = admission.as_dict(); changed[key] = value
        changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="schema, scope or digest"):
            EngineeringAdmission.from_dict(changed)
    admission.payload["extra"] = True
    with pytest.raises(ValueError, match="digest"):
        admission.as_dict()
    with pytest.raises(TypeError, match="typed"):
        verify_engineering_admission(None, {})


def test_verifier_recomputes_evidence_instead_of_trusting_label(monkeypatch):
    admission = admission_fixture()
    monkeypatch.setattr(module, "read_artifact", lambda record: record["path"])
    monkeypatch.setattr(module, "prepare_engineering_admission", lambda *args: admission)
    verify_engineering_admission(None, admission)
    changed = deepcopy(admission.payload); changed["extra"] = "different evidence"
    replacement = EngineeringAdmission.from_dict({**changed, "sha256": digest(changed)})
    monkeypatch.setattr(module, "prepare_engineering_admission", lambda *args: replacement)
    with pytest.raises(ValueError, match="evidence changed"):
        verify_engineering_admission(None, admission)


def test_derived_launch_features_accept_only_float64_roundoff(monkeypatch):
    actual = np.linspace(-1., 1., 77)
    monkeypatch.setattr(module, "launch_features", lambda _: actual)
    saved = actual.copy(); saved[38] += 5.551115123125783e-17
    evidence = module.launch_feature_roundoff(None, saved)
    assert evidence["maximum_abs_error"] == 5.551115123125783e-17
    assert evidence["absolute_tolerance"] == 64*np.finfo(np.float64).eps
    assert evidence["relative_tolerance"] == 0.
    for changed in (1e-8, float("nan"), float("inf")):
        saved[38] = changed
        with pytest.raises(ValueError, match="derived launch features"):
            module.launch_feature_roundoff(None, saved)
    with pytest.raises(ValueError, match="malformed"):
        module.launch_feature_roundoff(None, saved[:-1])


def test_admission_reverification_preserves_identity_but_allows_bounded_roundoff_diagnostic(monkeypatch):
    payload = admission_fixture().payload
    payload["references"] = [{"master_sha256": "a"*64,
        "launch_feature_roundoff": {"maximum_abs_error": 5.551115123125783e-17,
            "absolute_tolerance": module.LAUNCH_FEATURE_ABSOLUTE_TOLERANCE, "relative_tolerance": 0.,
            "scope": "derived fixture"}, "geometry": {"guard_clearance": .16}}]
    def make(value):
        return EngineeringAdmission.from_dict({**value, "sha256": digest(value)})
    admission = make(payload)
    changed = deepcopy(payload)
    changed["references"][0]["launch_feature_roundoff"]["maximum_abs_error"] = 0.
    monkeypatch.setattr(module, "read_artifact", lambda record: record["path"])
    monkeypatch.setattr(module, "prepare_engineering_admission", lambda *args: make(changed))
    verify_engineering_admission(None, admission)
    # No rounding tolerance extends to original source identities or other
    # geometry diagnostics, and an invalid recorded/recomputed error is rejected.
    for key, value in (("master_sha256", "b"*64), ("geometry", {"guard_clearance": .16000000000000003})):
        old = changed["references"][0][key]; changed["references"][0][key] = value
        with pytest.raises(ValueError, match="evidence changed"):
            verify_engineering_admission(None, admission)
        changed["references"][0][key] = old
    for value in (1e-8, -1e-17):
        changed["references"][0]["launch_feature_roundoff"]["maximum_abs_error"] = value
        with pytest.raises(ValueError, match="roundoff evidence"):
            verify_engineering_admission(None, admission)


def test_engineering_runner_delegates_only_learning_and_forbids_checkpoint_initialization(monkeypatch):
    from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
    from human_punching.wbc.runtime.runner import HumanMotionTrackingRunner
    admission = admission_fixture()
    cls = engineering_runner(admission)
    assert issubclass(cls, HumanMotionTrackingRunner)
    assert cls.__init__ is HumanMotionTrackingRunner.__init__
    runner = object.__new__(cls)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(bundle_path="synthetic")))))
    calls = []
    monkeypatch.setattr(module, "verify_engineering_admission", lambda *args: calls.append("admission"))
    monkeypatch.setattr(PolicyOnlyMotionTrackingRunner, "learn", lambda self, *a, **k: calls.append("inherited PPO"))
    runner.learn(2)
    assert calls == ["admission", "inherited PPO"]
    saves = []
    monkeypatch.setattr(HumanMotionTrackingRunner, "save", lambda self, path, infos: saves.append((path, infos)))
    runner.save("capacity.pt", {"retained": True})
    assert saves[0][1]["retained"]
    assert saves[0][1]["engineering_capacity"]["capacity_weights_may_initialize_training"] is False
    with pytest.raises(RuntimeError, match="starts from scratch"):
        runner.load("capacity_weights.pt")


def test_auto_reset_override_requires_explicit_boolean():
    with pytest.raises(TypeError, match="explicit boolean"):
        module.build_engineering_env_cfg(None, admission_fixture(), num_envs=4096, auto_reset=1)


def test_evidence_identity_survives_distinct_mount_aliases_but_not_changed_copies(tmp_path, monkeypatch):
    from human_punching.wbc.provenance import sha256_file
    local = tmp_path/"local/human-punching"
    alias = tmp_path/"other_mount/human-punching"
    for root in (local, alias):
        (root/"outputs").mkdir(parents=True)
        (root/"outputs/report.json").write_text('{"same":true}')
    monkeypatch.setattr(module, "PROJECT_ROOT", local)
    expected = sha256_file(local/"outputs/report.json")
    # These are different files/inodes, as observed for the actual NFS aliases.
    assert not (local/"outputs/report.json").samefile(alias/"outputs/report.json")
    record = module._evidence(alias/"outputs/report.json")
    assert record == {"path": "outputs/report.json", "sha256": expected}
    assert module.read_artifact(record) == local/"outputs/report.json"
    assert module.read_artifact({"path": str(alias/"outputs/report.json"), "sha256": expected}) == local/"outputs/report.json"
    (alias/"outputs/report.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="digest differs"):
        module.read_artifact({"path": str(alias/"outputs/report.json"), "sha256": expected})
    with pytest.raises(ValueError, match="hash-identical local"):
        module._evidence(alias/"outputs/report.json")
