"""Completed profiling survives independent terminal-panel setup/failure."""
import json
from types import SimpleNamespace

import pytest
import torch

from human_punching.wbc.capacity_panel import prepare_terminal_panel, coexistence_memory


def test_rank_records_precede_collective_teardown_without_waiting_for_panel(tmp_path, monkeypatch):
    events = []
    record = {"status": "passed", "rank": 1, "measured_fresh_transitions": 294912}
    def barrier(*, device_ids):
        assert json.loads((tmp_path/"capacity_completed.json").read_text()) == record
        assert device_ids == [1]
        events.append("all_measurements_complete")
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: events.append("destroyed"))
    prepare_terminal_panel(SimpleNamespace(is_distributed=True, device="cuda:1"), tmp_path, record)
    assert events == ["all_measurements_complete", "destroyed"]
    # An exception in the subsequent independent panel cannot erase this file.
    with pytest.raises(RuntimeError):
        raise RuntimeError("synthetic panel startup failed")
    assert json.loads((tmp_path/"capacity_completed.json").read_text())["status"] == "passed"


def test_failed_capacity_does_not_enter_terminal_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: pytest.fail("No collective"))
    with pytest.raises(ValueError, match="completed capacity"):
        prepare_terminal_panel(SimpleNamespace(is_distributed=True), tmp_path, {"status": "failed"})
    assert not (tmp_path/"capacity_completed.json").exists()


def test_single_rank_persists_without_distributed_operations(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: pytest.fail("No collective"))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: pytest.fail("No group"))
    prepare_terminal_panel(SimpleNamespace(is_distributed=False), tmp_path, {"status": "passed"})
    assert (tmp_path/"capacity_completed.json").is_file()


def memory_fixture():
    # Synthetic bytes establish arithmetic only, not native capacity evidence.
    return ({"total_bytes": 1000, "driver_used_bytes": 500},
            {"torch_peak_reserved_bytes": 600, "driver_used_bytes": 650},
            {"available": True, "total_vram_bytes": 1000,
             "conservative_peak_used_bytes": 700, "maximum_sampled_non_torch_bytes": 200})


def test_coexistence_peak_uses_actual_panel_memory_schema():
    before, after, native = memory_fixture()
    result = coexistence_memory(before, after, native)
    assert result["coexistence_peak_used_bytes"] == 800
    assert result["headroom_fraction"] == pytest.approx(.2)
    assert result["total_vram_bytes"] == 1000
    native["conservative_peak_used_bytes"] = 900
    assert coexistence_memory(before, after, native)["headroom_fraction"] == pytest.approx(.1)


@pytest.mark.parametrize("mutation", ("missing", "different_total", "boolean", "negative", "unavailable"))
def test_coexistence_memory_rejects_missing_or_invalid_native_samples(mutation):
    before, after, native = memory_fixture()
    if mutation == "missing": del native["maximum_sampled_non_torch_bytes"]
    elif mutation == "different_total": native["total_vram_bytes"] += 1
    elif mutation == "boolean": after["torch_peak_reserved_bytes"] = True
    elif mutation == "negative": before["driver_used_bytes"] = -1
    else: native["available"] = False
    with pytest.raises(ValueError, match="coexistence memory"):
        coexistence_memory(before, after, native)
