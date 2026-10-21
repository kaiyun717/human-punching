"""Packet preparation is local and cannot submit a session."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest


def module():
    source=Path(__file__).parents[1]/'scripts/task10_telemetry.py'
    spec=importlib.util.spec_from_file_location('task10_telemetry_test',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_prepare_hashes_relative_paths_freezes_copies_and_records_bounded_no_training(monkeypatch,tmp_path):
    m=module();root=tmp_path/'repo';source=root/'src/human_punching/wbc';source.mkdir(parents=True)
    (source/'x.py').write_text('value=1\n');scripts=root/'scripts';scripts.mkdir()
    (scripts/'task10_telemetry.py').write_text('packet source\n');(scripts/'task10_telemetry.sbatch').write_text('shell\n')
    bundle=root/'bundle';bundle.mkdir();(bundle/'manifest.json').write_text('{}')
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'__file__',str(scripts/'task10_telemetry.py'))
    import human_punching.wbc.runtime.bundle as runtime_bundle
    monkeypatch.setattr(runtime_bundle,'load_runtime_bundle',lambda value:SimpleNamespace(root=bundle))
    # No scheduler/network API can be entered by prepare.
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch anything'))
    monkeypatch.setattr(m.os,'fsencode',lambda value:b'short-fixture-path')
    result=m.prepare(bundle,root/'out')
    assert result['training'] is False and result['acceptance_qualified'] is False
    assert result['environments']==16 and result['seconds']==6
    assert '--time=00:30:00' in result['submission_command']
    assert all(flag in result['submission_command'] for flag in ('--nodes=1', '--ntasks=1', '--gres=gpu:l40s:1'))
    assert result['application_wall_seconds']<1800
    assert all(not Path(p).is_absolute() for p in result['files'])
    for p,h in result['files'].items():
        assert m.sha256_file(root/'out/source_snapshot'/p)==h
    with pytest.raises(FileExistsError):m.prepare(bundle,root/'out')


def test_slurm_script_uses_submit_directory_not_spool_source():
    path=Path(__file__).parents[1]/'scripts/task10_telemetry.sbatch'
    text=path.read_text()
    assert 'SLURM_SUBMIT_DIR' in text and 'BASH_SOURCE' not in text


def test_budget_walk_tolerates_disappearing_compiler_scratch(tmp_path):
    m=module()
    stable=tmp_path/'stable';stable.write_bytes(b'abcd')
    class Disappearing:
        def is_file(self): return True
        def is_symlink(self): return False
        def stat(self): raise FileNotFoundError('compiler removed scratch')
    root=SimpleNamespace(rglob=lambda _: iter((stable,Disappearing())))
    assert m.tree_bytes(root)==4
