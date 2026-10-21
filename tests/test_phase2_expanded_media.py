"""Expanded/shared video selection and saved-frame tests; no GPU rendering."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from human_punching.phase2_deadline import ExecutionDeadline
from human_punching.wbc.provenance import sha256_file
from test_phase2_expanded_evaluation import fixture

SPEC=importlib.util.spec_from_file_location('phase2_expanded_media',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_expanded_media.py')
MEDIA=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MEDIA)


def outcomes(declaration):
    return [dict(trial_id=b.trial_id,status='executed',numeric_complete_cycle=True,execution_complete=True,
        disposition='completed_native_attempt',trace_audit_pass=None,visual_review='pending') for b in declaration.bindings]


@pytest.mark.parametrize('start_mode',['family_guard','shared_jab_saved_state'])
def test_selection_keeps_declared_random_ids_and_all_missing_failed_partial_requests(start_mode):
    declaration,*_=fixture();declaration=replace(declaration,start_mode=start_mode)
    rows=outcomes(declaration);rows[1]['numeric_complete_cycle']=False;rows[2]['status']='rejected'
    rows[3]['execution_complete']=False;missing=rows.pop(4)['trial_id']
    selected=MEDIA.selection(declaration,rows)
    assert selected==MEDIA.selection(declaration,list(reversed(rows)))
    assert [r['trial_id'] for r in selected]==[b.trial_id for b in declaration.bindings[:5]]
    assert selected[0]['reasons']==['predeclared_video_trial_id']
    assert next(r for r in selected if r['trial_id']==missing)['reasons']==['missing_outcome']
    with pytest.raises(ValueError):MEDIA.selection(declaration,rows+[rows[0]])


def test_batch_index_binds_exact_world_reference_and_actual_launch_commitment(tmp_path):
    declaration,cells,*_=fixture();bindings=declaration.bindings[:2]
    candidate=next(iter(cells));batch=tmp_path/'batch_0000';batch.mkdir()
    plan=dict(trials=[b.trial_id for b in bindings],candidates=[candidate]*2,planned_physics_steps=8,actor_sha256='a'*64)
    (batch/'plan.json').write_text(json.dumps(plan))
    (batch/'launch_commitments.json').write_text(json.dumps(dict(commitments=[dict(trial_id=bindings[1].trial_id,
        actual_launch_time_s=.02,disposition='committed')])))
    result=MEDIA.index_batches(tmp_path,declaration,cells)
    assert result[bindings[0].trial_id]['world']==0 and result[bindings[1].trial_id]['world']==1
    assert result[bindings[0].trial_id]['actual_launch_s'] is None
    assert result[bindings[1].trial_id]['actual_launch_s']==.02
    assert result[bindings[1].trial_id]['actual_actor_tensor_digest']=='a'*64
    plan['trials']=[bindings[0].trial_id]*2;(batch/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError):MEDIA.index_batches(tmp_path,declaration,cells)


def test_complete_physical_callback_rows_do_not_invent_missing_status_failures():
    declaration,*_=fixture();rows=outcomes(declaration)
    for row in rows:row.pop('status');row.pop('execution_complete')
    selected=MEDIA.selection(declaration,rows)
    assert [r['trial_id'] for r in selected]==list(declaration.video_trial_ids)


def test_exact_world_extraction_preserves_finite_partial_prefix_and_history(tmp_path):
    q=np.zeros((5,2,55));q[:,:,3]=1.;q[:,1,7]=np.arange(5)+10;q[3,1,8]=np.nan
    v=np.zeros((5,2,54));path=tmp_path/'raw_trace.npz'
    np.savez(path,qpos=q,qvel=v,reset_count=np.ones((5,2)),time=np.arange(5)[:,None]*.005*np.ones((1,2)))
    arrays,prefix,counters=MEDIA.extract_world(path,1,planned_samples=9,planned_worlds=2)
    np.testing.assert_array_equal(arrays['qpos'][:,7],[10,11,12])
    assert prefix['truncated_at_native_invalid_state'] and prefix['incomplete_native_horizon']
    assert prefix['rendered_prefix_samples']==3 and arrays['physics_step_count']==2 and counters['initial_reset_counter']==1
    with pytest.raises(ValueError):MEDIA.extract_world(path,2,planned_samples=9,planned_worlds=2)
    with pytest.raises(ValueError):MEDIA.extract_world(path,0,planned_samples=4,planned_worlds=2)


def test_phase_overlay_uses_actual_launch_and_labels_rejected_requests_honestly():
    row=dict(preparation_complete_tick=80,launch_tick=120,outward_end_tick=240,guard_hold_tick=260,
        duration_s=1.4,requested_ttc_s=.6,target_w=[.8,.2,1.3])
    for instant,phase in ((.2,'PREPARATION'),(.5,'READY HOLD'),(.7,'OUTWARD'),(1.25,'RECOVERY'),(1.35,'GUARD HOLD')):
        label=MEDIA.event_overlay(row,instant,partial=False,actual_launch_s=.6)
        assert label.startswith(phase) and 'request TTC=0.600s' in label and 'target=(0.800,0.200,1.300)m' in label
    rejected=MEDIA.event_overlay(row,.7,partial=True,actual_launch_s=None)
    assert 'NO COMMITTED LAUNCH' in rejected and 'planned launch dt' in rejected and 'PARTIAL TRACE' in rejected


def test_annotation_preserves_one_x_fifty_fps_and_full_decode(tmp_path):
    import imageio.v2 as imageio
    base=tmp_path/'base.mp4'
    with imageio.get_writer(base,fps=50,codec='libx264',macro_block_size=16) as writer:
        for level in (20,80,140):writer.append_data(np.full((80,128,3),level,np.uint8))
    row=dict(preparation_complete_tick=0,launch_tick=0,outward_end_tick=8,guard_hold_tick=8,
        duration_s=.04,requested_ttc_s=.6,target_w=[.8,.2,1.3])
    arrays=dict(time_s=np.arange(9)*.005)
    result=MEDIA.annotate_video(base,tmp_path/'annotated.mp4',arrays,row,partial=True,actual_launch_s=0.,
        deadline=ExecutionDeadline(60.,serialization_reserve_s=10.),product_check=lambda:None)
    assert result['video_frames']==result['decoded_frames']==3 and result['source_sample_indices']==[0,4,8]
    assert result['fps']==50 and result['playback_speeds']==[1.] and not result['integration'] and not result['inference']


def test_media_requires_counted_allocation_before_any_product(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted allocated GPU'):
        MEDIA.render(tmp_path,tmp_path/'never_created')
    assert not (tmp_path/'never_created').exists()


def test_media_follows_only_hash_bound_continuation_ancestry(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    execution=parent/'execution.json';execution.write_text(json.dumps(dict(execution_identity='a'*64)))
    outcomes_path=parent/'outcomes.json';outcomes_path.write_text('[]')
    (child/'execution_identity.json').write_text(json.dumps(dict(identity='a'*64)))
    continuation=dict(parent_run=str(parent),parent_execution=str(execution),parent_execution_sha256=sha256_file(execution),
        parent_outcomes=str(outcomes_path),parent_outcomes_sha256=sha256_file(outcomes_path),same_frozen_execution_identity='a'*64)
    (child/'continuation.json').write_text(json.dumps(continuation))
    assert MEDIA.source_runs(child)==[child,parent]
    outcomes_path.write_text('[{}]')
    with pytest.raises(ValueError,match='evidence changed'):MEDIA.source_runs(child)
