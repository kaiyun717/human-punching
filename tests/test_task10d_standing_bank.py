"""Changed geometry, circular retention, placement-first sampling and clock."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from human_punching.task10d.placement_phase3 import cache_library
from human_punching.task10d.standing_bank import (
    FixedStrikeClock, build_geometry, calibrate_landmarks, candidate_grid,
    candidate_placement, circular_components, expected_sampler_exposure,
    finalize_bank, primary_envelope, retain_largest_components,
    sample_placement_first,
)


def test_candidate_domain_is_actual_pelvis_and_not_old_residual_limits():
    xy, yaw = candidate_grid()
    assert xy.shape == (22*23, 2)
    assert yaw.shape == (72,)
    np.testing.assert_array_equal(xy.min(0), [.45, -.55])
    np.testing.assert_array_equal(xy.max(0), [1.5, .55])
    assert yaw[0] == -180 and yaw[-1] == 175
    assert len(np.unique(np.mod(yaw, 360))) == 72
    assert primary_envelope(np.array([[.7, 0], [1.5, 0], [.65, .3]])).all()
    assert not primary_envelope(np.array([[.65, 0], [.45, 0], [1.5, .55]])).any()


def test_circular_components_wrap_tie_and_all_valid():
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 1], bool)
    components = circular_components(mask)
    assert [x.tolist() for x in components] == [[3, 4], [6, 7, 0, 1]]
    got = retain_largest_components(mask)
    assert got["retained_component_start"] == 6
    assert got["retained_component_length"] == 4
    np.testing.assert_array_equal(got["retained_valid"], [1, 1, 0, 0, 0, 0, 1, 1])
    tie = retain_largest_components(np.array([0, 1, 1, 0, 1, 1, 0, 0], bool))
    assert tie["retained_component_start"] == 1
    assert circular_components(np.zeros(8, bool)) == []
    np.testing.assert_array_equal(circular_components(np.ones(8, bool))[0], np.arange(8))


def test_largest_valid_run_can_remove_all_aimed_choices_and_unknown_stays_unknown():
    shape = (1, 2, 8)
    hit = np.zeros(shape, bool)
    hit[0, 0, 5] = True
    hit[0, 1, 2] = True
    geometric = np.ones(shape, bool)
    checked = np.ones(shape, bool)
    clean = np.zeros(shape, bool)
    clean[0, 0, [0, 1, 2, 5]] = True
    clean[0, 1, [1, 2, 3]] = True
    geometry = dict(outward_hit=hit, geometric_initial_valid=geometric,
                    primary_xy=np.array([True, True]))
    bank = finalize_bank(geometry, checked, checked, checked, clean)
    assert bank["raw_supported_xy"].tolist() == [True, True]
    assert bank["primary_sampler_xy"].tolist() == [False, True]
    assert bank["aimed_lost_by_retention"].sum() == 1
    assert bank["retained_valid"][0, 0].sum() == 3  # no hit-biased replacement
    unknown = checked.copy()
    unknown[0, 1, 2] = False
    with pytest.raises(ValueError, match="Unrendered"):
        finalize_bank(geometry, checked, checked, unknown, clean)
    clean[0, 1, 2] = False
    bank = finalize_bank(geometry, checked, checked, unknown, clean)
    assert not bank["raw_valid"][0, 1, 2]


@pytest.fixture(scope="module")
def paths():
    return cache_library()


def test_inverse_batched_geometry_matches_production_fk_at_actual_pelvis(paths):
    heading = .47
    state = dict(head_world=np.array([.28, -.14, 1.26]),
        pelvis_world=np.array([.31, -.18, .82]), pelvis_heading_rad=heading,
        torso_rotation=Rotation.from_euler("zyx", [heading, .03, -.02]).as_matrix(),
        camera_position_world=np.array([.36, -.16, 1.23]))
    xy = np.array([[.7, -.2], [1.05, .1], [1.45, .3]])
    yaw = np.array([-175., -50., 80., 175.])
    bank = build_geometry(paths, state, xy_G=xy, yaw_deg=yaw)
    for p, path in enumerate(paths):
        for x,y in ((0,0), (1,1), (2,3)):
            placement = candidate_placement(path,state,xy[x],yaw[y])
            first = path.primitive.reference.sample(0.,placement)
            delta = first.body_pos[0,:2]-state["pelvis_world"][:2]
            independent_xy = Rotation.from_euler("z",-heading).apply([*delta,0.])[:2]
            np.testing.assert_allclose(independent_xy,xy[x],atol=2e-14,rtol=0)
            forward = Rotation.from_quat(first.body_quat[0],scalar_first=True).as_matrix()[:,0]
            expected = heading+np.deg2rad(yaw[y])
            assert abs(np.arctan2(np.sin(np.arctan2(forward[1],forward[0])-expected),
                                  np.cos(np.arctan2(forward[1],forward[0])-expected))) < 1e-13
            np.testing.assert_allclose(first.body_pos[:,2],
                path.primitive.reference.sample(0.,path.primitive.placement).body_pos[:,2],atol=1e-14)
            independent = np.stack([path.primitive.reference.sample(float(t),placement).glove_pos
                                    for t in path.time_s])
            distance = np.linalg.norm(independent-state["head_world"],axis=-1)
            outward = distance[path.outward,path.active_index]
            assert bank["outward_min_distance_m"][p,x,y] == pytest.approx(outward.min(),abs=1e-14)
            assert bank["outward_hit"][p,x,y] == bool(np.any(outward <= .14))
            assert bank["active_recovery_hit"][p,x,y] == bool(np.any(distance[~path.outward,path.active_index] <= .14))
            assert bank["inactive_cycle_hit"][p,x,y] == bool(np.any(distance[:,1-path.active_index] <= .14))
            np.testing.assert_allclose(bank["initial_glove_distances_m"][p,x,y], distance[0], atol=1e-14)


def _sampling_bank():
    # First XY has one yaw/primitive; second has 3 primitives and many yaws.
    # A combination-uniform implementation would almost never sample XY zero.
    mask = np.zeros((3,2,72),bool)
    mask[0,0,0] = True
    mask[:,1,:] = True
    return dict(primary_sampler_xy=np.ones(2,bool),retained_aimed=mask,
        retained_valid=mask.copy(),outward_hit=mask.copy(),
        primitive_ids=np.array(["jab_nominal","jab_variant","cross_nominal"]),
        families=np.array(["jab","jab","cross"]),xy_G=np.array([[.8,0.],[1.1,.2]]),
        yaw_deg=np.arange(-180,180,5),primary_xy=np.ones(2,bool),
        first_outward_sample_s=np.full(mask.shape,.5),outward_min_distance_m=np.full(mask.shape,.1))


def test_sampler_is_xy_uniform_and_reports_nonuniform_family_exposure():
    bank = _sampling_bank()
    draws = sample_placement_first(bank,np.random.default_rng(100),10000)
    assert abs(np.mean([row["xy_index"] == 0 for row in draws])-.5) < .02
    exposure = expected_sampler_exposure(bank)
    assert exposure["family_probability"] == {"cross": .25,"jab": .75}
    assert exposure["primitive_probability"] == {"jab_nominal": .625,"jab_variant": .125,"cross_nominal": .25}
    assert all(row["retained_valid"] and row["frozen_head_outward_hit"] for row in draws)


def test_fixed_clock_uses_saved_landmark_and_terminal_hold():
    clock = FixedStrikeClock(.4,.8,1.5)
    assert clock.alpha == .5
    assert clock.source_time(.8) == .4
    assert clock.source_time(100.) == 1.5
    with pytest.raises(ValueError):
        FixedStrikeClock(0.,.8,1.5)
    with pytest.raises(ValueError):
        clock.source_time(-.1)


def test_calibration_refines_active_outward_entry_not_earlier_inactive_contact():
    # Active center crosses .14 at .43; the inactive hand is already touching.
    class Reference:
        sha256 = "fixture-reference"
        path = "fixture-reference.npz"
        def sample(self,t,placement):
            return SimpleNamespace(glove_pos=np.array([[.57-t,0.,0.],[0.,0.,0.]]))
    from human_punching.task10d.replay import Placement
    primitive = SimpleNamespace(primitive_id="jab_nominal",family="jab",active_hand="left",
        reference=Reference(),metadata_sha256="fixture-json",placement=Placement())
    path = SimpleNamespace(primitive=primitive,time_s=np.array([0.,.4,.435,.5]),
        outward=np.array([True,True,True,False]),active_index=0,
        initial_heading_rad=0.,initial_pelvis=np.zeros(3))
    bank = _sampling_bank()
    bank.update(primitive_ids=np.array(["jab_nominal"]),families=np.array(["jab"]),
        head_world=np.zeros(3),pelvis_world=np.zeros(3),pelvis_heading_rad=np.array(0.),
        xy_G=np.array([[1.05,0.]]),yaw_deg=np.array([0.]),primary_xy=np.array([True]),
        retained_aimed=np.ones((1,1,1),bool),retained_valid=np.ones((1,1,1),bool),
        outward_hit=np.ones((1,1,1),bool),envelope_margin_m=np.array([.25]),
        initial_center_margins=np.ones((1,1,1,2)),outward_min_distance_m=np.full((1,1,1),.1),
        first_outward_sample_s=np.full((1,1,1),.435),previous_outward_sample_s=np.full((1,1,1),.4))
    result=calibrate_landmarks([path],bank,fixture_id="standing-test")
    row=result["landmarks"][0]
    assert row["supported"]
    assert row["s_j_star"] == pytest.approx(.43)
    assert row["first_sampled_entry_s"] == .435
    assert row["crossing_bracket_s"] == [.4,.435]
    assert row["crossing_center_distance_m"] == pytest.approx(.14)
