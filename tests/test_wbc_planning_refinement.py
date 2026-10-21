import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.candidates import CandidateSpec, build_candidate_bundle
from human_punching.wbc.planning_refinement import RefinementSettings, _PoseCurve, _unbiased_support_solve
from human_punching.wbc.references import PROJECT_ROOT, FAMILIES, tangent_velocities
from human_punching.wbc.runtime.bundle import load_runtime_bundle


def test_pose_curve_preserves_quiet_endpoints_and_scalar_bounds():
    times = np.arange(4)*.1
    poses = np.zeros((4, 55))
    poses[:, 3] = 1
    poses[:, 7] = [0., .2, -.1, 0.]
    curve = _PoseCurve(times, poses)
    fine = curve(np.linspace(0., .3, 1001))
    assert fine[:, 7].min() >= -.1 and fine[:, 7].max() <= .2
    np.testing.assert_array_equal(curve(0)[0], poses[0])
    np.testing.assert_allclose(curve(.3)[0], poses[-1], atol=1e-15)
    assert abs((curve(1e-6)[0, 7]-curve(0)[0, 7])/1e-6) < 1e-3
    with pytest.raises(ValueError):
        RefinementSettings(max_source_joint_speed_rad_s=-1)


@pytest.mark.parametrize('backend', ['refined_v1','refined_v2','refined_v3'])
def test_refined_actual_four_family_candidates_remeasure_ttc_and_reduce_spikes(tmp_path, monkeypatch, backend):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline refinement must not integrate dynamics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    specs = [CandidateSpec(15744,family,.8,0,ttc)
             for family,ttc in zip(FAMILIES,(.70,.70,1.10,.95),strict=True)]
    result = build_candidate_bundle(specs, release_root=PROJECT_ROOT/'outputs/four-punch-milestone',
        bank_root=PROJECT_ROOT.parent/'robust_sav/constants/boxing/snapshots/v1_20260921',
        g1_asset_root=PROJECT_ROOT/'.cache/deps/mjlab-8ee51fbc/src/mjlab/asset_zoo/robots/unitree_g1',
        nominal_plant_path=PROJECT_ROOT.parent/'robust_sav/constants/boxing/stance/v1_20260921/nominal_plant.json',
        output_root=tmp_path/'refined', ik_iterations=35, guard_iterations=120, backend=backend)
    report=json.loads(Path(result['report_path']).read_text())
    assert result['generated_count']==4, report['requests']
    bundle=load_runtime_bundle(result['manifest_path'])
    assert not bundle.manifest['qualified_for_training']
    assert report['all_basic_command_screens_passed']
    if backend in ('refined_v1','refined_v2'):
        assert not report['all_kinematic_screens_passed']  # Smoothing cannot certify contacts.
    else:
        assert report['all_kinematic_screens_passed']
    for row in report['requests']:
        assert row['diagnostics']['contact_remeasured_after_refinement']
        assert row['diagnostics']['max_sole_drift_m'] < .015
        assert abs(row['sampled_ttc_error_s']) <= .005+1e-9
        if backend in ('refined_v2','refined_v3'):
            geometry=row['geometry_screen']
            assert geometry['max_floor_penetration_m']==0
            assert geometry['max_selected_self_penetration_m']==0
            assert geometry['max_sole_drift_m'] < .0002
            assert geometry['max_joint_limit_excursion_rad']==0
            if backend=='refined_v2':
                assert not geometry['checks']['initial_guard_velocity_readiness']
            else:
                assert geometry['checks']['initial_guard_velocity_readiness']
                assert geometry['checks']['terminal_guard_velocity_readiness']
    for clip in bundle.clips:
        assert clip['requested_ttc_s'] < clip['outward_end_s'] < clip['guard_time_s']
        with np.load(bundle.path(clip['master_path']),allow_pickle=False) as master:
            np.testing.assert_array_equal(master['cycle_phase_id'],np.where(
                master['time_s'] < clip['outward_end_s'],0,
                np.where(master['time_s'] < clip['guard_time_s'],1,2)))
            assert float(master['outward_end_s'])==clip['outward_end_s']
            if backend=='refined_v3':
                model=mujoco.MjModel.from_xml_path(str(bundle.model_path))
                velocity=tangent_velocities(model,master['time_s'],master['qpos'])
                np.testing.assert_array_equal(master['qvel'],velocity)
                acceleration=np.gradient(velocity,master['time_s'],axis=0,edge_order=2)
                np.testing.assert_allclose(velocity[[0,-1]],0,atol=1e-12)
                np.testing.assert_allclose(acceleration[[0,-1]],0,atol=1e-10)
    # Fixture thresholds are regression diagnostics, not physical feasibility limits.
    right = next(row for row in report['requests'] if row['spec']['family']=='right_hook')
    assert right['derivative_diagnostics']['peak_joint_speed_rad_s'] < 15
    assert right['derivative_diagnostics']['peak_joint_acceleration_rad_s2'] < 1500


def test_support_projection_has_exact_guard_fixed_point():
    fixture=json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())
    guard=np.asarray(fixture['guard_qpos'])
    model=mujoco.MjModel.from_xml_path(str(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    data=mujoco.MjData(model)
    data.qpos[:]=guard
    mujoco.mj_forward(model,data)
    points,rotations=[],[]
    for side in ('Left','Right'):
        name=side+'Foot_sole'
        index=model.site(name).id
        points.append(('site',name,data.site_xpos[index].copy(),40.))
        rotations.append(('site',name,data.site_xmat[index].reshape(3,3).copy(),10.))
    projected=_unbiased_support_solve(model,guard,guard,points,rotations,12)
    np.testing.assert_array_equal(projected,guard)
