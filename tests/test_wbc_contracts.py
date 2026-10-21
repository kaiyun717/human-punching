from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from human_punching.wbc.admission import (
    FeasibleDomain, GuardEvidence, QualificationCell, RejectionReason, admit_request,
)
from human_punching.wbc.contracts import (
    LaunchPlan, LaunchSnapshot, PunchRequest, SearchBounds, TimingContract,
)


def snapshot():
    qpos = np.zeros(36)
    qpos[2:4] = [.75, 1.]
    return LaunchSnapshot("123", qpos, np.zeros(35), (0, 0, 1.23), (.1, .2, 0),
                          (.75, 0, 0), (0, 0, 0, 1), "command:12")


def request():
    return PunchRequest("jb", .6, snapshot(), "guard0")


def domain():
    cell = QualificationCell("jab:0", "jab", (.70, .80), (-.05, .05), (.55, .65),
                             (1.20, 1.25), ("guard0",), ("123",))
    return FeasibleDomain("fixture", (cell,), True, *("a"*64 for _ in range(4)), "fixture", 10)


def guard():
    return GuardEvidence(True, True, True, True, True, (.16, .2))


def test_launch_state_copies_input_and_cannot_chase():
    source = np.zeros(36)
    source[3] = 1
    snap = LaunchSnapshot("x", source, np.zeros(35), [0, 0, 1.23], [0, 0, 0],
                          [.75, 0, 0], [1, 0, 0, 0])
    identity = snap.fingerprint
    source[:] = 7
    assert snap.g1_qpos[3] == 1
    assert identity == snap.fingerprint
    with pytest.raises(FrozenInstanceError):
        snap.head_position_w = (1, 2, 3)
    with pytest.raises(ValueError, match="launch snapshot"):
        LaunchPlan(request(), "ref", "a"*64, (0, 0, 1.24))


def test_coordinate_conventions_and_clocks():
    snap = snapshot()
    assert snap.relative_placement() == (.75, 0)
    assert snap.world_to_human_point((0, 0, 1.23)) == (.75, 0, 1.23)
    assert snap.world_to_human_vector((1, 0, 0)) == (-1, 0, 0)
    for kwargs in ({"physics_dt_s": .001}, {"decimation": 20}, {"integration_substeps": 5}):
        with pytest.raises(ValueError):
            TimingContract(**kwargs)


def test_search_bounds_do_not_establish_feasibility():
    req = request()
    assert SearchBounds().contains(req)
    decision = admit_request(req, FeasibleDomain("unqualified"), guard())
    assert not decision.admitted
    assert RejectionReason.DOMAIN_UNQUALIFIED in decision.reasons
    with pytest.raises(ValueError, match="physical evidence"):
        FeasibleDomain("bad", qualification_passed=True)


def test_coupled_domain_and_guard_evidence_are_required():
    assert admit_request(request(), domain(), guard()).admitted
    assert not admit_request(replace(request(), ttc_s=.7), domain(), guard()).admitted
    assert not admit_request(replace(request(), family="cross"), domain(), guard()).admitted
    assert not admit_request(replace(request(), guard_id="unknown"), domain(), guard()).admitted
    assert not admit_request(request(), domain(), None).admitted
    assert not admit_request(request(), domain(), replace(guard(), glove_distances_m=(.159, .2))).admitted
    assert not admit_request(request(), domain(), replace(guard(), body_collision_free=False)).admitted


def test_bad_launch_payloads_fail():
    with pytest.raises(ValueError):
        replace(snapshot(), g1_qvel=[float("nan")]*35)
    with pytest.raises(ValueError):
        replace(snapshot(), human_quat_wxyz=(0, 1, 0, 0))
    with pytest.raises(ValueError):
        replace(request(), ttc_s=0)
