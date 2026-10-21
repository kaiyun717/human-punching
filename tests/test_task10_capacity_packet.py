"""Allocation topology must match the actual localhost-only worker launcher."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).parents[1]/"scripts/task10_capacity.py"
    spec = importlib.util.spec_from_file_location("task10_capacity_packet_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gpus", (1, 2, 4))
def test_capacity_command_pins_all_gpus_to_one_task_on_one_node(wrapper, tmp_path, gpus, monkeypatch):
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **kw: pytest.fail("No submission"))
    command = wrapper.submission_command(tmp_path, gpus=gpus, exclude_node="node3006")
    assert command[:2] == ["sbatch", "--parsable"]
    assert command.count("--nodes=1") == command.count("--ntasks=1") == 1
    assert f"--gres=gpu:l40s:{gpus}" in command and "--time=00:30:00" in command
    assert not any(item.startswith("--gpus=") for item in command)
    assert "--exclude=node3006" in command and "--no-requeue" in command
    assert command[-2:] == [str(wrapper.ROOT/"scripts/task10_capacity.sbatch"), str(tmp_path/"packet.json")]


def test_allocation_receipt_reads_only_explicit_resource_fields(wrapper, monkeypatch):
    for name in wrapper.ALLOCATION_ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PRIVATE_UNRELATED_ENVIRONMENT", "must_not_record")
    record = wrapper.allocation_environment()
    assert record == {"schema": "human_wbc_capacity_allocation_environment_v1", "environment": {
        "SLURM_JOB_ID": "synthetic", "SLURM_JOB_NUM_NODES": "1", "SLURM_GPUS_ON_NODE": "2",
        "CUDA_VISIBLE_DEVICES": "0,1"}}
