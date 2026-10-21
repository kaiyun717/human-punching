"""Independent Phase 1 reconstruction rejects semantic and evidence corruption."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("phase1_audit",Path(__file__).resolve().parents[1]/"scripts/task10b_phase1_audit.py")
audit=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def synthetic(*, contact=False):
    t=np.arange(181)*.005
    boolean=np.ones(len(t),bool)
    distance=np.ones((len(t),2))
    if contact:distance[30,0]=.13
    measured=dict(readiness_admission=boolean.copy(),operational_pose=boolean.copy(),
        operational_support=boolean.copy(),operational_finite=boolean.copy(),operational_upright=boolean.copy(),
        operational_guard=boolean.copy(),frozen_target_glove_distance_m=distance)
    return t,measured


def test_initial_guard_cannot_complete_and_outward_begins_new_dwell():
    t,m=synthetic()
    result,events=audit.independently_classify(t,m,20,60,0)
    assert not events["complete"][0]
    assert not events["complete"][:100].any()
    assert events["complete"][100:].all()
    assert result["recovery_time_s"]==.5
    assert events["phase"][20]=="outward" and events["phase"][60]=="recovery"


def test_recovered_miss_stays_miss_independently_of_contact():
    t,m=synthetic()
    miss,events=audit.independently_classify(t,m,20,60,0)
    m["frozen_target_glove_distance_m"][30,0]=.13
    hit,hit_events=audit.independently_classify(t,m,20,60,0)
    assert miss["functional_complete"] and hit["functional_complete"]
    assert miss["contact_outcome"]=="miss" and miss["first_active_hand_outward_contact_s"] is None
    assert hit["contact_outcome"]=="hit" and hit["first_active_hand_outward_contact_s"]==.15
    assert np.array_equal(events["complete"],hit_events["complete"])


def test_inactive_hand_contact_does_not_convert_commanded_miss():
    t,m=synthetic()
    m["frozen_target_glove_distance_m"][30,1]=.13
    result,_=audit.independently_classify(t,m,20,60,0)
    assert result["functional_complete"] and result["contact_outcome"]=="miss"


def test_support_interruption_restarts_dwell_at_interval_end():
    t,m=synthetic()
    m["operational_support"][80]=False
    result,events=audit.independently_classify(t,m,20,60,0)
    assert events["recovery_dwell_s"][80]==0
    assert result["recovery_time_s"]==.6


def test_pose_interruption_invalidates_adjacent_intervals():
    t,m=synthetic()
    m["operational_pose"][80]=False
    result,events=audit.independently_classify(t,m,20,60,0)
    assert events["recovery_dwell_s"][80]==events["recovery_dwell_s"][81]==0
    assert result["recovery_time_s"]==.605


def test_readiness_is_current_and_rejected_launch_never_completes():
    t,m=synthetic(contact=True)
    m["readiness_admission"][20]=False
    result,events=audit.independently_classify(t,m,20,60,0)
    assert result["readiness_time_s"]==0
    assert result["launch_rejected"] and result["launch_time_s"] is None
    assert result["recovery_time_s"] is None and not events["complete"].any()
    assert result["contact_outcome"]=="hit"  # Observed trace contact is retained, not admission success.


def test_later_fall_does_not_erase_timestamp_but_fails_full_cycle():
    t,m=synthetic()
    m["operational_upright"][-1]=False
    result,_=audit.independently_classify(t,m,20,60,0)
    assert result["recovered_after_outward"] and result["recovery_time_s"]==.5
    assert not result["functional_complete"]


def test_contact_before_launch_is_not_commanded_outward_contact():
    t,m=synthetic()
    m["frozen_target_glove_distance_m"][5,0]=.13
    result,_=audit.independently_classify(t,m,20,60,0)
    assert result["contact_outcome"]=="miss"


def test_reference_schedule_ignores_roundoff_and_preserves_actor_boundary():
    q=np.zeros((120,55)); q[:,3]=1
    v=np.full((120,54),2.6e-17); v[47:,6]=.091
    master=dict(qpos=q,qvel=v)
    result=audit.reference_schedule(master,dict(outward_end_s=.308333333333))
    assert result==dict(first_reference_motion_tick=47,launch_tick=44,outward_tick=62)
    assert result["launch_tick"]*.005==.22


def test_static_reference_is_not_an_executed_punch():
    with pytest.raises(ValueError,match="no outward"):
        audit.reference_schedule(dict(qpos=np.zeros((100,55)),qvel=np.zeros((100,54))),dict(outward_end_s=.3))


def test_comparison_does_not_accept_tampered_completion_or_missing_contact():
    assert not audit.same(False,True)
    assert not audit.same(None,.5)
    assert not audit.same(np.array([False,True]),np.array([False,False]))
    assert not audit.same("hit","miss")


def test_dwell_rejects_duplicate_time_and_reset_row_has_zero_duration():
    assert audit.continuous_interval_dwell([0,.005],[True,True],[True,True]).tolist()==[0,.005]
    with pytest.raises(ValueError,match="increasing"):
        audit.continuous_interval_dwell([0,0],[True,True],[True,True])
