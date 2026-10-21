from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from human_punching.phase2_domain import (
    COUNTS, FINAL_TIMINGS, CANDIDATE_TIMINGS, EventClock, LaunchCommand,
    _maximin, _representatives, feasibility_catalog, measured_domain,
    snapshot_dispositions, start_perturbations,
)
from human_punching.wbc.splits import assign_trajectory_groups


def command(family="jab", ttc=.6, target=(0., 0., 1.2), start="same_saved_state"):
    return LaunchCommand(family, ttc, target, start, "a"*64, "b"*64)


def test_group_assignment_keeps_adjacent_times_and_replica_exclusions():
    groups = np.repeat([29, 30, 31], 6)
    replicas = np.tile(np.repeat([0, 1, 2], 2), 3)
    codes, assigned = snapshot_dispositions(groups, replicas, np.ones(18, bool))
    assert np.sum(codes == 3) == 6
    assert np.sum(codes == 1) == 12
    for group in np.unique(groups):
        assert len(np.unique(assigned[groups == group])) == 1
    assert assign_trajectory_groups([29,30,31]) == assign_trajectory_groups([31,29,30])


def test_standing_replicas_are_calibration_not_split_examples():
    codes, _ = snapshot_dispositions([3307]*3, [0,1,2], [True]*3)
    assert codes.tolist() == [2,2,2]


def test_ineligible_never_becomes_compatible():
    codes, _ = snapshot_dispositions([3,3307], [0,1], [False,False])
    assert codes.tolist() == [0,0]


def test_measured_motion_does_not_use_command_labels():
    v = [[0,0,0], [.1,0,.1], [.1,.1,0], [.50001,0,0], [0,0,-.5], [0,np.nan,0]]
    assert measured_domain(v).tolist() == ["D1","D1","D2","excluded","D2","excluded"]


def test_representatives_are_one_per_command_and_maximin_is_order_invariant():
    cmd = np.array([10,10,20,20,30,30])
    velocity = np.arange(18).reshape(6,3)/100
    reps = _representatives(cmd,np.arange(6),velocity,np.ones(6))
    assert len(reps) == len(set(cmd[reps])) == 3
    rows = np.array([10,20,30,40]); features = np.array([[0,0],[1,1],[2,0],[0,3]])
    assert _maximin(rows,features,3) == _maximin(rows[::-1],features[::-1],3)
    with pytest.raises(ValueError):
        _maximin(rows,features,5)


def test_starts_are_finite_bounded_and_replayable_with_unchanged_root():
    a,b = start_perturbations(21021),start_perturbations(21021)
    assert a == b and a != start_perturbations(21022)
    q,v = np.array(a["joint_position_delta_rad"]),np.array(a["generalized_velocity_delta"])
    assert q.shape == (4,48) and v.shape == (4,54)
    assert not np.any(q[0]) and not np.any(v[0]) and not np.any(v[:,:6])
    assert abs(q).max() <= .015 and abs(v).max() <= .06


def test_counts_and_untrained_interior_are_explicit():
    assert COUNTS == dict(train=32,validation=8,test=16)
    for family,(low,middle,high) in FINAL_TIMINGS.items():
        assert low < middle < high and middle not in CANDIDATE_TIMINGS[family]
        assert low in CANDIDATE_TIMINGS[family] and high in CANDIDATE_TIMINGS[family]


def test_immutable_request_copies_mutable_target_and_changes_identity():
    target = [0.,0.,1.2]; a = command(target=target); target[2] = 100
    assert a.target_w == (0.,0.,1.2)
    with pytest.raises(FrozenInstanceError):
        a.family = "cross"
    assert a.identity != command(ttc=.61).identity
    assert a.identity != command(target=(0,0,1.21)).identity
    assert a.identity != command(start="another_state").identity


def test_no_nearest_reference_or_timing_fallback():
    a,b = command(),command(ttc=.61)
    proof = dict(qualified=True,independent_audit_pass=True,request_identity=a.identity,
                 qualification_receipt_sha256="f"*64)
    catalog = feasibility_catalog([a,b],{a.identity:proof})
    assert [r["legal"] for r in catalog["requests"]] == [True,False]
    assert catalog["coverage_zero_to_four"] == {"0":1,"1":1,"2":0,"3":0,"4":0}


def test_masks_include_zero_coverage_and_require_physical_independent_proof():
    requests = [command(family=f) for f in ("jab","cross","left_hook","right_hook")]
    none = feasibility_catalog(requests,{r.identity:{"screen_passed":True} for r in requests})
    assert none["coverage_zero_to_four"]["0"] == 1
    evidence = {r.identity:dict(qualified=True, independent_audit_pass=True,
        request_identity=r.identity,qualification_receipt_sha256="c"*64) for r in requests}
    all_four = feasibility_catalog(requests,evidence)
    assert all_four["coverage_zero_to_four"]["4"] == 1
    assert len(all_four["four_family_scenarios"]) == 1
    with pytest.raises(ValueError):
        feasibility_catalog([requests[0],requests[0]],{})


def test_launch_clock_excludes_prep_and_preserves_first_early_hit():
    clock = EventClock(0,.6,.62,.63,.64,1.8,first_contact_s=1.24,recovery_s=2.)
    assert clock.timing_valid(.6)
    assert not EventClock(0,.6,.62,.63,.64,1.8,first_contact_s=1.13).timing_valid(.5)
    assert not EventClock(0,.6,.62,.63,.64,1.8,first_contact_s=None).timing_valid(.6)
    assert not EventClock(0,.6,.62,.63,.64,1.8,first_contact_s=.5).timing_valid(.6)


@pytest.mark.parametrize("args", [(0,.6,.62,.63,.645,1.8),(0,.7,.62,.63,.64,1.8),
                                  (0,.6,.62,.63,.64,.5)])
def test_bad_actor_or_phase_clock_rejected(args):
    with pytest.raises(ValueError):
        EventClock(*args)
