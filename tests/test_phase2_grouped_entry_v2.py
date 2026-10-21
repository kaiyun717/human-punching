"""Saved-receipt failure propagation only; no CUDA, actors, or model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from human_punching import phase2_grouped_entry_v2 as entry


def good():
    return dict(status="bounded_stop", reload_parity=dict(
        parameters_equal=True, actions_equal=True, maximum_action_error=0.0))


@pytest.mark.parametrize("parity", [None, {}, True, [],
    dict(parameters_equal=False, actions_equal=True, maximum_action_error=0.0),
    dict(parameters_equal=1, actions_equal=True, maximum_action_error=0.0),
    dict(parameters_equal=True, actions_equal=1, maximum_action_error=0.0),
    dict(parameters_equal=True, actions_equal=False, maximum_action_error=0.0),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error=True),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error="0"),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error=1e-30),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error=float("nan")),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error=float("inf")),
    dict(parameters_equal=True, actions_equal=True, maximum_action_error=-float("inf")),
])
def test_missing_inexact_or_nonfinite_parity_never_passes(parity):
    result = good(); result["reload_parity"] = parity
    verdict = entry.checkpoint_integrity(result)
    assert verdict["passed"] is False
    json.dumps(verdict, allow_nan=False)


def test_exact_parity_does_not_repair_original_failure():
    result = good(); result["status"] = "failed"
    assert entry.checkpoint_integrity(result)["passed"] is False
    assert entry.checkpoint_integrity(good())["passed"] is True


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    # ROOT is only widened for synthetic owned fixtures, never production paths.
    monkeypatch.setattr(entry, "ROOT", Path(__file__).resolve().parents[1])
    configuration = tmp_path / "configuration.json"
    configuration.write_text("{}")
    output = tmp_path / "run"
    state = dict(result=good(), calls=0, mutate_checkpoint=False, omit_receipt=False)
    def original(path, digest, folder):
        state["calls"] += 1
        assert path == configuration and digest == entry._sha(configuration)
        (folder / "training").mkdir(parents=True)
        checkpoint = folder / "training/checkpoint_final.pt"
        checkpoint.write_bytes(b"synthetic saved checkpoint, never loaded")
        result = deepcopy(state["result"])
        result["grouped_schema"] = "task10b_phase2_grouped_training_receipt_v1"
        result["checkpoint"] = entry._binding(checkpoint)
        raw = json.dumps(result, sort_keys=True).encode()
        state["original_bytes"] = raw
        if not state["omit_receipt"]:
            (folder / "grouped_training_receipt.json").write_bytes(raw)
        if state["mutate_checkpoint"]:
            checkpoint.write_bytes(b"changed checkpoint")
        return result
    monkeypatch.setattr(entry, "_original_execute_configuration", original)
    return configuration, output, state


def run_fake(fixture):
    configuration, output, state = fixture
    result = entry.execute_configuration(configuration, entry._sha(configuration), output)
    receipt = json.loads((output / "grouped_integrity_v2.json").read_text())
    return result, receipt


def test_good_saved_files_pass_without_rewriting_original(fake_run):
    result, receipt = run_fake(fake_run)
    _, output, state = fake_run
    assert result["status"] == "bounded_stop" and result["grouped_integrity_pass"] is True
    assert receipt["passed"] and all(receipt["checks"].values())
    assert (output / "grouped_training_receipt.json").read_bytes() == state["original_bytes"]
    assert state["calls"] == 1 and receipt["physical_qualification_credit"] is False
    with pytest.raises(FileExistsError):
        run_fake(fake_run)
    assert state["calls"] == 1


@pytest.mark.parametrize("failure", ["bad_parity", "failed_status", "mutate_checkpoint", "omit_receipt"])
def test_failed_integrity_stays_failed_and_preserves_original(fake_run, failure):
    _, output, state = fake_run
    if failure == "bad_parity":
        state["result"]["reload_parity"]["actions_equal"] = False
    elif failure == "failed_status":
        state["result"]["status"] = "failed"
    else:
        state[failure] = True
    result, receipt = run_fake(fake_run)
    assert result["status"] == "failed" and result["grouped_integrity_pass"] is False
    assert receipt["status"] == "failed" and state["calls"] == 1
    if not state["omit_receipt"]:
        assert (output / "grouped_training_receipt.json").read_bytes() == state["original_bytes"]


def test_original_early_exception_gets_separate_failed_integrity(fake_run, monkeypatch):
    def broken(*args):
        raise RuntimeError("synthetic original failure")
    monkeypatch.setattr(entry, "_original_execute_configuration", broken)
    result, receipt = run_fake(fake_run)
    assert result["status"] == "failed" and receipt["original_exception"]["type"] == "RuntimeError"
    assert receipt["original_receipt"] is None and receipt["checkpoint"] is None


def test_wrong_configuration_hash_prevents_delegation(fake_run):
    configuration, output, state = fake_run
    with pytest.raises(ValueError, match="identity"):
        entry.execute_configuration(configuration, "0" * 64, output)
    assert state["calls"] == 0 and not output.exists()


def test_output_outside_task10b_prevents_delegation(fake_run):
    configuration, _, state = fake_run
    output = entry.ROOT / "outputs/synthetic_unowned_grouped_entry_output"
    with pytest.raises(ValueError):
        entry.execute_configuration(configuration, entry._sha(configuration), output)
    assert state["calls"] == 0 and not output.exists()


def test_cli_returns_failure_for_parity_failure(fake_run, capsys):
    configuration, output, state = fake_run
    state["result"]["reload_parity"] = None
    code = entry.main(["--configuration", str(configuration), "--configuration-sha256",
        entry._sha(configuration), "--output", str(output)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
