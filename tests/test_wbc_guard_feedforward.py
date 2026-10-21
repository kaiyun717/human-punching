"""Native inverse statics and LP algebra only; no CPU dynamics or learning."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.guard_feedforward import (
    _digest, generate_guard_feedforward, load_guard_feedforward, verify_equilibrium,
)
from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.runtime.bundle import load_runtime_bundle


@pytest.fixture(scope="module")
def bundle(task10_common_bundle):
    return task10_common_bundle


@pytest.fixture
def artifact(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("No CPU dynamics"))
    path = tmp_path/"guard_feedforward.json"
    result = generate_guard_feedforward(bundle, path)
    assert result["physics_executed"] is result["qualified_for_training"] is False
    return path


def test_real_static_support_solution_has_no_base_actuation(bundle, artifact):
    data = load_guard_feedforward(artifact, bundle)
    torque = np.asarray(data["joint_feedforward_nm"])
    assert torque.shape == (48,) and np.any(abs(torque) > 1.)
    assert data["applied_base_wrench"] == [0.]*6
    assert data["external_support_applied"] is False
    assert data["guard_admission_passed"] is False
    check = verify_equilibrium(data)
    assert check["maximum_equilibrium_residual"] < 1e-5
    assert .15 < check["peak_joint_effort_ratio"] < .25
    np.testing.assert_array_equal(data["qvel"], np.zeros(54))
    np.testing.assert_array_equal(data["qacc"], np.zeros(54))
    with pytest.raises(FileExistsError):
        generate_guard_feedforward(bundle, artifact)


@pytest.mark.parametrize("key,value", [
    ("qpos_sha256", "f"*64), ("model_sha256", "f"*64),
    ("source_controller_sha256", "f"*64), ("source_sha256", {}),
    ("executed_solver_contract", {}), ("joint_names", ["wrong"]*48),
])
def test_changed_source_pose_or_runtime_rejected_even_with_new_digest(bundle, artifact, key, value):
    data = json.loads(artifact.read_text())
    data[key] = value
    data["payload_sha256"] = _digest({k:v for k,v in data.items() if k != "payload_sha256"})
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contract mismatch"):
        load_guard_feedforward(artifact, bundle)


def test_injected_base_wrench_or_changed_torque_is_rejected(bundle, artifact):
    data = json.loads(artifact.read_text())
    data["applied_base_wrench"][2] = 735.
    with pytest.raises(ValueError, match="never apply a base wrench"):
        verify_equilibrium(data)
    data["applied_base_wrench"] = [0.]*6
    data["joint_feedforward_nm"][0] += 1.
    with pytest.raises(ValueError, match="equilibrium or existing effort"):
        verify_equilibrium(data)


def test_edited_artifact_digest_rejected_before_runtime_construction(bundle, artifact, monkeypatch):
    data = json.loads(artifact.read_text())
    data["joint_feedforward_nm"][0] += .2
    artifact.write_text(json.dumps(data))
    import human_punching.wbc.guard_feedforward as module
    monkeypatch.setattr(module, "solver_contract", lambda *a: pytest.fail("Hash gate before config"))
    with pytest.raises(ValueError, match="payload digest"):
        load_guard_feedforward(artifact, bundle)
