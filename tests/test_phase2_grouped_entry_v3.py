"""Finite receipt identity regressions; no physics, actors, or training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from human_punching import phase2_grouped_entry_v2 as previous
from human_punching import phase2_grouped_entry_v3 as current


def payload():
    return dict(status="bounded_stop",
        grouped_schema="task10b_phase2_grouped_training_receipt_v1",
        target_candidate_identities=((2674, "a" * 64), (4251, "b" * 64)),
        reload_parity=dict(parameters_equal=True, actions_equal=True,
                          maximum_action_error=0.0))


def test_actual_declaration_container_survives_finite_json_roundtrip():
    returned = payload()
    saved = json.loads(json.dumps(returned))
    assert saved != returned  # The preserved v2 gate fails on this real shape.
    assert current.receipt_matches(saved, returned)
    assert current.canonical_receipt(saved) == current.canonical_receipt(returned)


@pytest.mark.parametrize("replacement", [False, True, 2674.0, "2674", 4251])
def test_changed_group_id_or_scalar_type_is_not_normalized(replacement):
    returned = payload()
    saved = json.loads(json.dumps(returned))
    saved["target_candidate_identities"][0][0] = replacement
    assert not current.receipt_matches(saved, returned)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                    {1: "non-string key"}, {"set"}, b"bytes"])
def test_non_json_or_nonfinite_values_are_rejected(value):
    returned = payload()
    returned["extra"] = value
    with pytest.raises((ValueError, TypeError)):
        current.canonical_receipt(returned)


def test_missing_extra_changed_and_reordered_identity_values_fail():
    returned = payload()
    saved = json.loads(json.dumps(returned))
    for altered in (dict(saved, extra=1),
                    {k: v for k, v in saved.items() if k != "status"},
                    dict(saved, target_candidate_identities=list(reversed(saved["target_candidate_identities"]))),
                    dict(saved, grouped_schema="other")):
        assert not current.receipt_matches(altered, returned)


@pytest.mark.parametrize("module,expected", [(previous, False), (current, True)])
def test_full_gate_tuple_receipt_preserves_original_bytes(tmp_path, monkeypatch, module, expected):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(module, "ROOT", root)
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}")
    output = tmp_path / "run"
    original_bytes = []
    def synthetic(path, digest, folder):
        assert path == configuration and digest == module._sha(configuration)
        (folder / "training").mkdir(parents=True)
        checkpoint = folder / "training/checkpoint_final.pt"
        checkpoint.write_bytes(b"synthetic checkpoint, never loaded")
        returned = deepcopy(payload())
        returned["checkpoint"] = module._binding(checkpoint)
        raw = json.dumps(returned).encode()
        original_bytes.append(raw)
        (folder / "grouped_training_receipt.json").write_bytes(raw)
        return returned
    monkeypatch.setattr(module, "_original_execute_configuration", synthetic)
    result = module.execute_configuration(configuration, module._sha(configuration), output)
    assert result["grouped_integrity_pass"] is expected
    assert result["status"] == ("bounded_stop" if expected else "failed")
    assert (output / "grouped_training_receipt.json").read_bytes() == original_bytes[0]
    assert len(original_bytes) == 1
    with pytest.raises(FileExistsError):
        module.execute_configuration(configuration, module._sha(configuration), output)


def test_serialized_identity_never_repairs_failed_checkpoint_parity():
    returned = payload()
    returned["reload_parity"]["actions_equal"] = False
    saved = json.loads(json.dumps(returned))
    assert current.receipt_matches(saved, returned)
    assert not current.checkpoint_integrity(returned)["passed"]
