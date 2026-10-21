"""Bounded-session source guards fail before any learner process starts."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/task10_guard_learning.py"
    spec = importlib.util.spec_from_file_location("task10_guard_learning_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_input_rejected_before_process(launcher, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(launcher, "writable_path", lambda p: Path(p))
    source = tmp_path / "source.py"
    source.write_text("old\n")
    digest = launcher.sha256_file(source)
    source.write_text("changed\n")
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema": launcher.SCHEMA, "production_training": False,
        "punch_training": False, "scratch_initialization": True,
        "files": {str(source): digest}}))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned learner"))
    with pytest.raises(ValueError, match="Prepared source/input changed"):
        launcher.execute(packet)


def test_no_allocation_rejected_before_read(launcher, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned learner"))
    with pytest.raises(RuntimeError, match="allocated Slurm"):
        launcher.execute(Path("does-not-exist"))


def test_long_compiler_path_rejected_before_creation(launcher, tmp_path, monkeypatch):
    from human_punching.wbc import guard_training
    monkeypatch.setattr(guard_training, "prepare_guard_admission", lambda *a: {})
    monkeypatch.setattr(launcher, "writable_path", lambda p: Path(p))
    output = tmp_path / ("a" * 120)
    with pytest.raises(ValueError, match="shallow output"):
        launcher.prepare(tmp_path, tmp_path, tmp_path, output)
    assert not output.exists()
