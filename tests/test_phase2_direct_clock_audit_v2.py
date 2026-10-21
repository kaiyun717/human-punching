"""Only saved prefix clock arithmetic; no simulation or actors."""
import json
from pathlib import Path
import numpy as np
import pytest
from human_punching import phase2_direct_clock_audit_v2 as audit


def fixture(tmp_path,monkeypatch,*,planned=12,actual=8):
    (tmp_path/'plan.json').write_text(json.dumps(dict(planned_physics_steps=planned,trials=['a','b'])))
    (tmp_path/'summary.json').write_text(json.dumps(dict(actual_physics_steps=actual)))
    t=np.zeros((actual+1,2),dtype=np.float32)
    for i in range(1,len(t)):t[i]=t[i-1]+np.float32(.005)
    q=np.zeros((actual+1,2,55),dtype=np.float32)
    np.savez(tmp_path/'raw_trace.npz',qpos=q,time=t)
    monkeypatch.setattr(audit.parent,'audit_batch',lambda p:dict(passed=actual==planned,
        checks=dict(complete_raw=actual==planned,all_original_event_checks=True),trials=2))
    return t,q


@pytest.mark.parametrize('actual',[0,4,8,12])
def test_resolved_prefix_is_clock_evidence_without_completed_cycle_credit(tmp_path,monkeypatch,actual):
    fixture(tmp_path,monkeypatch,actual=actual)
    r=audit.audit_batch(tmp_path)
    assert r['passed'] and r['retained_physics_steps']==actual
    assert r['prefix_only']==(actual<12) and r['completed_cycle_credit'] is False
    assert r['qualification'] is False
    assert json.loads(json.dumps(r,allow_nan=False))==r


@pytest.mark.parametrize('change',['count','clock','origin','world_shape','nonfinite','overrun','other_check'])
def test_bad_prefix_or_original_event_checks_cannot_be_repaired(tmp_path,monkeypatch,change):
    t,q=fixture(tmp_path,monkeypatch)
    if change=='count':(tmp_path/'summary.json').write_text(json.dumps(dict(actual_physics_steps=7)))
    if change=='clock':t[4:]+=.005
    if change=='origin':t+=1
    if change=='world_shape':t=t[:,0]
    if change=='nonfinite':t[2]=np.nan
    if change=='overrun':(tmp_path/'plan.json').write_text(json.dumps(dict(planned_physics_steps=4,trials=['a','b'])))
    if change=='other_check':monkeypatch.setattr(audit.parent,'audit_batch',lambda p:dict(passed=False,checks=dict(complete_raw=False,wrong_hand_metadata=False),trials=2))
    np.savez(tmp_path/'raw_trace.npz',qpos=q,time=t)
    assert audit.audit_batch(tmp_path)['passed'] is False


def test_actual_saved_jab_clock_uses_exact_native_accumulation():
    folder=Path(__file__).resolve().parents[1]/'outputs/task10b/p2jg01/run/held_out_validation/validation_00024/batch_0000'
    result=audit.audit_batch(folder)
    assert result['passed'] and result['retained_physics_steps']==908
    assert result['trials']==4 and result['prefix_only'] is False
    assert json.loads(json.dumps(result,allow_nan=False))==result
