"""Finite V2 dispatch and namespace/source guards; no scheduler or native calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from human_punching import phase2_runtime_pack_entry_v2 as entry
from human_punching.wbc.provenance import sha256_file

SPEC = importlib.util.spec_from_file_location('isolated_phase2_expand_v2',
    Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_expand_v2.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_isolated_entry_reuses_budget_watchdog_and_source_verifier():
    assert launcher.legacy.SCHEMA == launcher.SCHEMA
    assert launcher.legacy.MODES == ('runtime-pack', 'grouped-pilot')
    assert not launcher.legacy.CONDITIONAL_MODES
    assert launcher.legacy.validate_spec is launcher.validate_spec
    assert launcher.legacy.required_inputs is launcher.required_inputs
    assert launcher.legacy.native_command is launcher.native_command
    assert launcher.legacy.submit.__module__ == 'task10b_phase2_frozen_expansion_v1'
    assert launcher.legacy.execute.__module__ == 'task10b_phase2_frozen_expansion_v1'
    assert launcher.legacy.verify.__module__ == 'task10b_phase2_frozen_expansion_v1'


@pytest.mark.parametrize('mode', ['shared', 'timing-pilot', 'grouped-finetune', 'target-references'])
def test_no_scope_extension(mode):
    with pytest.raises(ValueError): launcher.validate_spec(mode, {}, 480, 400)
    with pytest.raises(ValueError): launcher.native_command({'mode': mode})


def test_scope_preserves_original_validation_and_no_conditional_unlock(monkeypatch):
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs)); return {'seed': 21021, 'fresh_transition_cap': 262144}
    monkeypatch.setattr(launcher, '_validate_spec', original)
    result = launcher.validate_spec('grouped-pilot', {'configuration': 'exact'}, 1800, 1680)
    assert calls == [(('grouped-pilot', {'configuration': 'exact'}, 1800, 1680), {'proof': None})]
    assert result['fresh_transition_cap'] == 262144 and result['final_checkpoint_integrity_required']
    with pytest.raises(ValueError): launcher.validate_spec('grouped-pilot', {}, 1800, 1680, proof={})


def test_prepare_requires_explicit_namespace_proof():
    with pytest.raises(ValueError, match='namespace'): launcher.prepare('unused', 'runtime-pack', {})


def test_semantic_dependency_injection_restores_original_on_failure(tmp_path, monkeypatch):
    plan = dump(tmp_path/'plan.json', {})
    old = launcher.legacy.semantic_files
    monkeypatch.setattr(launcher.legacy.base, 'local', lambda p: Path(p).resolve())
    def original(record, **kwargs):
        return launcher.legacy.semantic_files([plan])
    monkeypatch.setattr(launcher, '_required_inputs', original)
    def refused(*args, **kwargs): raise ValueError('exact closure rejection')
    monkeypatch.setattr(launcher, 'files', refused)
    record = {'mode': 'runtime-pack', 'inputs': {'plan': str(plan)}, 'runtime_namespace_proof': {'bound': True}}
    with pytest.raises(ValueError, match='exact closure'): launcher.required_inputs(record)
    assert launcher.legacy.semantic_files is old
    record.pop('runtime_namespace_proof')
    with pytest.raises(ValueError, match='isolated'): launcher.required_inputs(record)


def test_batch_script_slot_changes_only(monkeypatch):
    original = ['sbatch', '--gres=gpu:1', '--time=5',
                str(launcher.ROOT/'scripts/task10b_phase2_expand.sbatch'), '/exact/packet.json']
    monkeypatch.setattr(launcher, '_submission_command', lambda *args: original.copy())
    actual = launcher.submission_command({}, 'unused')
    assert actual[:-2] == original[:-2] and actual[-1] == original[-1]
    assert actual[-2] == str(launcher.ROOT/'scripts/task10b_phase2_expand_v2.sbatch')


def test_grouped_command_selects_integrity_entry_without_budget_change(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, '_native_command', lambda *args:
        ['python', '-B', '-m', 'human_punching.phase2_grouped_learning',
         '--configuration', 'same', '--configuration-sha256', 'same-sha', '--output', 'same-output'])
    actual = launcher.native_command({'mode': 'grouped-pilot'}, 1600)
    assert actual == ['python', '-B', '-m', 'human_punching.phase2_grouped_entry_v2',
        '--configuration', 'same', '--configuration-sha256', 'same-sha', '--output', 'same-output']


def test_runtime_command_binds_packet_alias_and_same_adaptive_wall(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    proof = dump(tmp_path/'proof.json', {'local_root': '/exact/logical/root'})
    packet = dump(tmp_path/'outputs/p/packet.json', {'immutable': 'packet'})
    monkeypatch.setattr(launcher.legacy, 'bound', lambda _: proof)
    monkeypatch.setattr(launcher, '_native_command', lambda *args:
        ['python', '-B', str(tmp_path/'scripts/task10b_phase2_runtime_pack.py'),
         '--plan', 'same', '--output', str(packet.parent/'run'), '--wall-seconds', '191.2'])
    actual = launcher.native_command({'mode': 'runtime-pack', 'runtime_namespace_proof': {},
        'output': 'outputs/p/run'}, 191.2)
    assert actual[2:4] == ['-m', 'human_punching.phase2_runtime_pack_entry_v2']
    assert actual[actual.index('--wall-seconds')+1] == '191.2'
    assert actual[-2:] == ['--packet-sha256', sha256_file(packet)]
    assert actual[actual.index('--logical-project-root')+1] == '/exact/logical/root'


@pytest.fixture
def logical_packet(tmp_path, monkeypatch):
    root = tmp_path/'physical'; root.mkdir()
    logical = tmp_path/'logical'; logical.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(entry, 'PROJECT_ROOT', root)
    def owned(value):
        result = Path(value).resolve()
        if not result.is_relative_to(root): raise ValueError('outside root')
        return result
    monkeypatch.setattr(entry, 'writable_path', owned)
    proof = dump(root/'proof.json', {'local_root': str(logical)})
    namespace = {'nonce': 'exact synthetic owned proof'}
    monkeypatch.setattr(entry, 'evidence', lambda r, p: namespace)
    plan = dump(root/'plan.json', {'unchanged': 'packing plan'})
    entry_path = root/'src/human_punching/phase2_runtime_pack_entry_v2.py'
    entry_path.parent.mkdir(parents=True); entry_path.write_text('# current entry bytes')
    monkeypatch.setattr(entry, '__file__', str(entry_path))
    files = ['plan.json', 'proof.json', str(entry_path.relative_to(root)),
             'src/human_punching/phase2_runtime_pack.py', 'scripts/task10b_phase2_expand_v2.py']
    packet_dir = root/'outputs/task10b/p'; packet_dir.mkdir(parents=True)
    inventory = {}
    for name in files:
        source = root/name
        if not source.exists(): source.parent.mkdir(parents=True, exist_ok=True); source.write_text('# fixed source')
        saved = packet_dir/'source_snapshot'/name
        saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(source.read_bytes())
        inventory[name] = sha256_file(source)
    record = dict(schema='task10b_phase2_expansion_packet_v2', mode='runtime-pack',
        training=False, fresh_transition_cap=0, gpus=1, automatic_followup=False,
        promotion=False, g1_actor_execution=False, combined_scene=False, inputs={'plan': str(logical/'plan.json')},
        output='outputs/task10b/p/run', runtime_namespace_proof={'path': 'proof.json', 'sha256': sha256_file(proof)},
        application_wall_seconds=240, allocation_wall_seconds=300,
        files=inventory, runtime_namespace_evidence=namespace)
    packet = dump(packet_dir/'packet.json', record)
    def invoke(**kwargs):
        return entry.execute(plan, packet_dir/'run', wall_seconds=kwargs.pop('wall_seconds', 190),
            logical_project_root=kwargs.pop('logical_project_root', str(logical)), runtime_namespace_proof=proof,
            packet_path=packet, packet_sha256=sha256_file(packet), **kwargs)
    return root, logical, packet, record, invoke


def test_pack_binds_sources_and_serializes_same_owned_location(logical_packet, monkeypatch):
    root, logical, packet, _, invoke = logical_packet
    old_output, old_input = entry.packing.writable_path, entry.packing.owned_input
    monkeypatch.setattr(entry.packing, 'owned_input', lambda binding: Path(binding['path']).resolve())
    patched_input = entry.packing.owned_input
    calls = []
    def fake_pack(plan, output, **kwargs):
        calls.append((plan, output, kwargs))
        folder = entry.packing.writable_path(output)
        source = entry.packing.owned_input({'path': root/'plan.json'})
        assert str(folder).startswith(str(logical)) and folder.resolve() == (root/'outputs/task10b/p/run')
        assert source == logical/'plan.json' and source.resolve() == root/'plan.json'
        folder.mkdir()
        return {'status': 'packed'}
    monkeypatch.setattr(entry.packing, 'pack', fake_pack)
    assert invoke() == {'status': 'packed'}
    assert len(calls) == 1 and calls[0][2] == {'wall_seconds': 190}
    assert entry.packing.writable_path is old_output and entry.packing.owned_input is patched_input
    receipt = json.loads((packet.parent/'run/logical_entry_v2.json').read_text())
    assert receipt['scientific_packing_math_changed'] is False


@pytest.mark.parametrize('change', ['entry_archive', 'plan_live', 'foreign_logical', 'wrong_plan', 'training', 'missing_source', 'too_late', 'expanded_child'])
def test_packet_source_scope_namespace_and_deadline_fail_before_pack(logical_packet, monkeypatch, change):
    root, logical, packet, record, invoke = logical_packet
    monkeypatch.setattr(entry.packing, 'pack', lambda *a, **k: pytest.fail('must reject before packing'))
    options = {}
    if change == 'entry_archive': (packet.parent/'source_snapshot/src/human_punching/phase2_runtime_pack_entry_v2.py').write_text('changed')
    elif change == 'plan_live': (root/'plan.json').write_text('changed')
    elif change == 'foreign_logical': options['logical_project_root'] = str(root)
    elif change == 'wrong_plan':
        other = dump(root/'other.json', {}); record['inputs']['plan'] = str(other); dump(packet, record)
    elif change == 'training': record['training'] = True; dump(packet, record)
    elif change == 'missing_source': record['files'].pop('src/human_punching/phase2_runtime_pack.py'); dump(packet, record)
    elif change == 'too_late': options['wall_seconds'] = 120
    else: options['wall_seconds'] = 211
    with pytest.raises(ValueError): invoke(**options)


def test_adapter_restored_when_original_pack_fails(logical_packet, monkeypatch):
    _, _, _, _, invoke = logical_packet
    old_output, old_input = entry.packing.writable_path, entry.packing.owned_input
    def fail(*args, **kwargs): raise RuntimeError('preserved underlying exception')
    monkeypatch.setattr(entry.packing, 'pack', fail)
    with pytest.raises(RuntimeError, match='underlying'): invoke()
    assert entry.packing.writable_path is old_output and entry.packing.owned_input is old_input
