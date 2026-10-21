import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).parents[1]/"scripts/task10_cycle_qualification.py"
    spec = importlib.util.spec_from_file_location("task10_cycle_packet_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("steps", [True, 299, 1001, 400.])
def test_invalid_step_budget_precedes_any_reference_or_write(wrapper, steps, tmp_path):
    output = tmp_path/"should_not_exist"
    with pytest.raises(ValueError, match="policy steps"):
        wrapper.prepare(tmp_path/"missing_reference", output, steps)
    assert not output.exists()


def test_allocation_required_before_native_calls(wrapper, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    path = tmp_path/"packet.json"
    path.write_text(json.dumps({"schema": "human_wbc_cycle_qualification_packet_v1"}))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **k: pytest.fail("No native call expected"))
    with pytest.raises(ValueError, match="allocated"):
        wrapper.execute(path)


@pytest.mark.parametrize("field,value", [
    ("physics_steps", 400), ("environments", 8192), ("training", True),
    ("strict_readiness", False), ("application_wall_seconds", 4000),
    ("products_and_caches_bytes", 20*1024**3),
])
def test_prepared_scope_cannot_expand_at_execution(wrapper, tmp_path, monkeypatch, field, value):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    record = dict(schema="human_wbc_cycle_qualification_packet_v1", policy_steps=400,
        physics_steps=1600, environments=16, training=False, strict_readiness=True,
        application_wall_seconds=1600, products_and_caches_bytes=6*1024**3)
    record[field] = value
    path = tmp_path/"packet.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **k: pytest.fail("No native call expected"))
    with pytest.raises(ValueError, match="scope"):
        wrapper.execute(path)
