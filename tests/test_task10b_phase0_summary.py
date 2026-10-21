"""Saved-evidence aggregation: preserve trial pairing and missing/failure denominators."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("task10b_phase0_summary", ROOT/"scripts/task10b_phase0_summary.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def rows():
    return [dict(trial_id=i, reference_tracking={"active_glove_position_rms_m": i/10000,
        "attack_active_glove_position_rms_m": .01}, contact_timing={"absolute_interpolated_contact_error_s": .015},
        final_guard={"trailing_recovered_dwell_s": .8}, joint_limit_violation_max_rad=.001,
        actual_joint_torque_max_nm=40.) for i in range(64)]


def test_pairing_uses_ids_not_saved_order_and_retains_missing_out_of_64():
    before, after = rows(), rows()
    after.reverse()
    for row in after:
        row["reference_tracking"]["active_glove_position_rms_m"] += .002
    after[0]["contact_timing"] = None
    result = summary.pair_panel_metrics(before, after)
    assert result["full_cycle_glove_rms_m"]["paired_difference_summary"]["mean"] == pytest.approx(.002)
    timing = result["historical_contact_error_s"]
    assert timing["accepted"][-1] is None
    assert timing["paired_difference_summary"]["requested"] == 64
    assert timing["paired_difference_summary"]["observed"] == 63
    assert timing["paired_difference_summary"]["missing"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different"])
def test_unmatched_trial_denominators_fail(mutation):
    after = rows()
    if mutation == "missing":
        after.pop()
    elif mutation == "duplicate":
        after[-1]["trial_id"] = 0
    else:
        after[-1]["trial_id"] = 64
    with pytest.raises(ValueError, match="matched trial IDs"):
        summary.pair_panel_metrics(rows(), after)


def test_nonfinite_saved_metric_cannot_become_summary_number():
    after = rows()
    after[5]["actual_joint_torque_max_nm"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.pair_panel_metrics(rows(), after)


def test_failed_or_unfinished_packet_is_rejected_before_any_output(tmp_path):
    packet = tmp_path/"packet.json"
    packet.write_text(json.dumps(dict(task="10B", program_phase=0, phase="preflight", training=False)))
    (tmp_path/"execution.json").write_text(json.dumps(dict(status="failed", completed_modes=[],
                                                         packet_sha256=summary.sha256(packet))))
    with pytest.raises(ValueError, match="complete passed"):
        summary.read_passed_packet(packet, "preflight", {})


def test_gate_failures_remain_in_64_trial_denominator():
    trials = [dict(acceptance_checks={"no_fall": i > 31, "guard_recovered": False}) for i in range(64)]
    result = summary.count_gates(trials)
    assert result == {"guard_recovered": 0, "no_fall": 32}
    trials[-1]["acceptance_checks"].pop("no_fall")
    with pytest.raises(ValueError, match="gate schema"):
        summary.count_gates(trials)
