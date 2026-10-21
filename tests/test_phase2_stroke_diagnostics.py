"""Saved-state CPU FK/review evidence; no physics, actor or GPU execution."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.phase2_stroke_diagnostics import (diagnostic_arrays,
    sequence_joints, write_review_plots)
from human_punching.wbc.provenance import sha256_file
from test_phase2_expanded_evaluation import fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('phase2_stroke_cli', ROOT/'scripts/task10b_phase2_stroke_diagnostics.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


@pytest.fixture
def stored():
    model = mujoco.MjModel.from_xml_path(str(ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    q = np.tile(model.qpos0, (6, 1))
    q[:, 2] = 1.
    v = np.zeros((6, 54))
    groups = sequence_joints(model)
    arm = model.jnt_qposadr[groups['left_arm'][0]]
    q[:, arm] = np.linspace(0., .01, len(q))
    v[:, model.jnt_dofadr[groups['left_arm'][0]]] = .4
    return model, q, v


def make_arrays(stored):
    model, q, v = stored
    return diagnostic_arrays(model, q, v, q[:-1], v[:-1], reference_ticks=np.arange(6),
        drive_reference_ticks=np.arange(5), target_w=[.8, .2, 1.2], active_hand=0)


def test_complete_exact_native_and_drive_knots_have_no_automatic_classification(stored):
    model, q, v = stored
    original = q.copy()
    arrays, receipt = make_arrays(stored)
    assert arrays['time_s'].tolist() == [0., .005, .01, .015, .02, .025]
    assert len(arrays['drive_time_s']) == 5
    np.testing.assert_array_equal(arrays['qpos'], q)
    np.testing.assert_array_equal(arrays['drive_qpos'], q[:-1])
    np.testing.assert_allclose(arrays['paired_glove_position_error_m'], 0., atol=1.e-15)
    np.testing.assert_array_equal(q, original)
    assert len(arrays['joint_names']) == 31
    assert receipt['missing_paired_drive_intervals'] == 0
    assert receipt['single_outward_stroke_pass'] is None and receipt['movement_exploit'] is None
    assert receipt['physics_integrations'] == receipt['actor_calls'] == 0


def test_moving_chest_glove_velocity_removes_frame_translation(stored):
    model, q, v = stored
    q[:, 7:] = q[0, 7:]
    v[:] = 0.
    v[:, :3] = [.03, -.02, .01]
    arrays, _ = make_arrays((model, q, v))
    np.testing.assert_allclose(arrays['glove_chest_velocity_m_s'], 0., atol=1.e-15)
    np.testing.assert_allclose(arrays['glove_world_velocity_m_s'],
        np.broadcast_to([.03, -.02, .01], (6, 2, 3)), atol=1.e-15)


@pytest.mark.parametrize('change', ['wrong_tick', 'bad_quaternion', 'nonfinite', 'fabricated_final_drive'])
def test_bad_or_unpaired_evidence_cannot_become_stroke_diagnostics(stored, change):
    model, q, v = stored
    rq, rv = q[:-1].copy(), v[:-1].copy()
    ticks = np.arange(5)
    if change == 'wrong_tick':ticks[2] += 1
    elif change == 'bad_quaternion':rq[1, 3] = .5
    elif change == 'nonfinite':v[0, 7] = np.nan
    else:rq, rv, ticks = q, v, np.arange(6)
    with pytest.raises(ValueError):
        diagnostic_arrays(model, q, v, rq, rv, reference_ticks=np.arange(6),
            drive_reference_ticks=ticks, target_w=[.8, .2, 1.2], active_hand=0)


def test_missing_drive_prefix_is_explicit_and_complete_physical_states_remain(stored):
    model, q, v = stored
    arrays, receipt = diagnostic_arrays(model, q, v, q[:2], v[:2], reference_ticks=np.arange(6),
        drive_reference_ticks=np.arange(2), target_w=[.8, .2, 1.2], active_hand=1)
    assert len(arrays['qpos']) == 6 and len(arrays['drive_qpos']) == 2
    assert receipt['missing_paired_drive_intervals'] == 3
    assert receipt['human_trace_review_required']


def test_exact_world_extraction_preserves_prefix_and_rejects_false_audit_hash(tmp_path, stored):
    _, q, v = stored
    q = np.stack((q, q.copy()), axis=1)
    q[:, 1, 0] += .1
    q[4, 1, 7] = np.nan
    v = np.stack((v, v), axis=1)
    path = tmp_path/'raw_trace.npz'
    np.savez(path, qpos=q, qvel=v, pre_qref=q[:-1], pre_vref=v[:-1],
        reference_tick=np.tile(np.arange(6)[:, None], (1, 2)),
        pre_reference_tick=np.tile(np.arange(5)[:, None], (1, 2)))
    binding = dict(world=1, planned_worlds=2, planned_samples=9)
    arrays, prefix = CLI.extract_exact_world(path, sha256_file(path), binding)
    assert len(arrays['qpos']) == 4 and len(arrays['reference_qpos']) == 3
    np.testing.assert_array_equal(arrays['qpos'][:, 0], q[:4, 1, 0])
    assert prefix['truncated_at_native_invalid_state'] and prefix['incomplete_native_horizon']
    assert prefix['unmatched_preintegration_samples'] == 2
    with pytest.raises(ValueError):CLI.extract_exact_world(path, '0'*64, binding)


def test_review_plots_include_all_arrays_without_emitting_truth(tmp_path, stored):
    arrays, receipt = make_arrays(stored)
    paths = write_review_plots(arrays, tmp_path, title='Synthetic stored-state CPU FK only',
        events=dict(preparation_complete=0., actual_launch=.02, outward_reference_complete=.1))
    assert len(paths) == 4 and all(path.stat().st_size > 1000 for path in paths)
    assert receipt['single_outward_stroke_pass'] is None
    with pytest.raises(FileExistsError):write_review_plots(arrays, tmp_path, title='No overwrite', events={})


def test_all_missing_attempts_stay_in_full_declared_ledger_without_native_work(tmp_path, monkeypatch):
    declaration, cells, *_ = fixture()
    run = tmp_path/'run';run.mkdir()
    (run/'declaration.json').write_text(json.dumps(asdict(declaration)))
    catalog = run/'catalog.json'
    catalog.write_text(json.dumps(dict(cells=list(cells.values()))))
    (run/'launch.json').write_text(json.dumps(dict(declaration_identity=declaration.identity,
        catalog=str(catalog), catalog_sha256=sha256_file(catalog))))
    audit = run/'audit.json'
    audit.write_text(json.dumps(dict(schema='task10b_phase2_independent_trace_audit_v1', run=str(run), panels=[])))
    # Ownership is still checked by media. The test's tmp_path is itself inside
    # this session's human-owned fresh output working directory.
    monkeypatch.setattr(CLI, 'load_declaration', lambda path: declaration)
    result = CLI.diagnose(run, audit, sha256_file(audit), tmp_path/'diagnostics', wall_seconds=30.)
    assert len(result['results']) == len(declaration.bindings) == 96
    assert result['counts'] == {'unavailable': 96}
    assert result['manual_review_records_created'] == result['new_physics_steps'] == result['actor_calls'] == 0
    assert all(row['single_outward_stroke_pass'] is None for row in result['results'])


def test_failed_audited_native_prefix_produces_trace_bound_evidence_and_keeps_all_ids(tmp_path, monkeypatch, stored):
    declaration, cells, *_ = fixture()
    run = tmp_path/'native_prefix';run.mkdir()
    batch = run/'batch_0000';batch.mkdir()
    request = declaration.bindings[0]
    candidate, row = next(iter(cells.items()))
    row['active_hand_index'] = 0
    (run/'declaration.json').write_text(json.dumps(asdict(declaration)))
    catalog = run/'catalog.json';catalog.write_text(json.dumps(dict(cells=list(cells.values()))))
    (run/'launch.json').write_text(json.dumps(dict(declaration_identity=declaration.identity,
        catalog=str(catalog), catalog_sha256=sha256_file(catalog))))
    (batch/'plan.json').write_text(json.dumps(dict(trials=[request.trial_id], candidates=[candidate],
        planned_physics_steps=8, actor_sha256='a'*64)))
    model_path = run/'runtime.xml'
    model_path.write_bytes((ROOT/'outputs/four-punch-milestone/jab/model/human.xml').read_bytes())
    _, q, v = stored
    raw = batch/'raw_trace.npz'
    np.savez(raw, qpos=q[:, None], qvel=v[:, None], pre_qref=q[:-1, None], pre_vref=v[:-1, None],
        reference_tick=np.arange(6)[:, None], pre_reference_tick=np.arange(5)[:, None])
    audit = run/'audit.json'
    audit.write_text(json.dumps(dict(schema='task10b_phase2_independent_trace_audit_v1', run=str(run),
        runtime_model=dict(path=str(model_path), sha256=sha256_file(model_path)), native_runtime_model_present=True,
        panels=[dict(path=str(batch), raw_sha256=sha256_file(raw), passed=False,
            trials=[dict(trial_id=request.trial_id, passed=False, execution_complete=False)])])))
    monkeypatch.setattr(CLI, 'load_declaration', lambda path: declaration)
    result = CLI.diagnose(run, audit, sha256_file(audit), tmp_path/'prefix_diagnostics', wall_seconds=30.)
    assert len(result['results']) == 96
    record = result['results'][0]
    assert record['status'] == 'diagnosed_prefix' and not record['audit_trial_pass']
    assert record['single_outward_stroke_pass'] is None and record['movement_exploit'] is None
    assert record['raw_trace_sha256'] == sha256_file(raw)
    assert len(record['diagnostic_evidence']) == 6
    assert all(sha256_file(item['path']) == item['sha256'] for item in record['diagnostic_evidence'])
    assert result['counts'] == {'diagnosed_prefix': 1, 'unavailable': 95}
