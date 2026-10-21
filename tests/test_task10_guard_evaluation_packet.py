"""Replay input confinement and pre-allocation failure checks; no simulation."""
import hashlib
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def module():
    path=Path(__file__).resolve().parents[1]/'scripts/task10_guard_evaluation.py'
    spec=importlib.util.spec_from_file_location('guard_replay_packet',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_verified_mount_alias_uses_local_content_identical_file(module,tmp_path,monkeypatch):
    root=tmp_path/'local/human-punching';root.mkdir(parents=True)
    remote=tmp_path/'compute/human-punching';remote.mkdir(parents=True)
    payload=b'checkpoint with frozen normalizer'
    (root/'model.pt').write_bytes(payload);(remote/'model.pt').write_bytes(payload)
    monkeypatch.setattr(module,'ROOT',root)
    sha=hashlib.sha256(payload).hexdigest()
    assert module.project_input(remote/'model.pt',sha)==root/'model.pt'
    (root/'model.pt').write_bytes(b'different checkpoint')
    with pytest.raises(ValueError,match='verified project file'):module.project_input(remote/'model.pt',sha)


def test_alias_cannot_escape_via_symlink(module,tmp_path,monkeypatch):
    root=tmp_path/'local/human-punching';root.mkdir(parents=True)
    remote=tmp_path/'compute/human-punching';remote.mkdir(parents=True)
    source=remote/'model.pt';source.write_bytes(b'data')
    (root/'model.pt').symlink_to(source);monkeypatch.setattr(module,'ROOT',root)
    with pytest.raises(ValueError,match='verified project file'):
        module.project_input(source,hashlib.sha256(b'data').hexdigest())


def test_execute_requires_allocation_before_reading_packet(module,tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Allocated GPU'):module.execute(tmp_path/'absent.json')
