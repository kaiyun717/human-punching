import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.reference_screening import ReferenceScreenLimits, screen_reference_geometry
from human_punching.wbc.references import PROJECT_ROOT, sha256

MODEL = PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'


def fixture_pose():
    fixture=json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())
    assert fixture['model_sha256']==sha256(MODEL)
    return np.array(fixture['guard_qpos']),np.array(fixture['recovery_collision_qpos'])


def write_hold(tmp_path, pose, name='hold.npz'):
    path=tmp_path/name
    np.savez(path,time_s=np.arange(5)*.005,qpos=np.tile(pose,(5,1)),qvel=np.zeros((5,54)))
    return path


def test_common_guard_passes_geometry_without_claiming_dynamics(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Reference screen must not step physics')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    guard,_=fixture_pose()
    result=screen_reference_geometry(MODEL,write_hold(tmp_path,guard),guard_time_s=.01)
    assert result['passed'] and all(result['checks'].values())
    assert result['initial_com_support_margin_m'] > .12
    assert not result['physics_executed']
    assert result['dynamic_feasibility']=='unqualified'


def test_ground_penetration_fails_even_when_joint_limits_pass(tmp_path):
    guard,_=fixture_pose()
    guard[2]-=.02
    result=screen_reference_geometry(MODEL,write_hold(tmp_path,guard),guard_time_s=.01)
    assert not result['passed']
    assert result['checks']['joint_limits']
    assert not result['checks']['floor_penetration']
    assert not result['checks']['planted_foot_support_geometry']
    assert result['max_floor_penetration_m'] > .019


def test_floating_guard_fails_without_needing_any_contact(tmp_path):
    guard,_=fixture_pose()
    guard[2]+=.02
    result=screen_reference_geometry(MODEL,write_hold(tmp_path,guard),guard_time_s=.01)
    assert not result['passed']
    assert result['checks']['floor_penetration']
    assert result['checks']['selected_self_collision']
    assert not result['checks']['planted_foot_support_geometry']
    assert result['penetrating_pairs']==[]


def test_real_recovery_collision_names_the_offending_pair(tmp_path):
    _,collision=fixture_pose()
    result=screen_reference_geometry(MODEL,write_hold(tmp_path,collision),guard_time_s=.01)
    assert not result['passed'] and not result['checks']['selected_self_collision']
    row=next(p for p in result['penetrating_pairs'] if p['geom_names']==['physical/LeftHand','physical/RightHand'])
    assert row['max_penetration_m'] > .04
    assert row['at_initial_guard'] and row['penetrating_frame_count']==5


def test_missing_or_malformed_payload_never_passes(tmp_path):
    guard,_=fixture_pose()
    path=tmp_path/'malformed.npz'
    np.savez(path,time_s=np.arange(5)*.01,qpos=np.tile(guard,(5,1)),qvel=np.zeros((5,54)))
    with pytest.raises(ValueError,match='Malformed'):
        screen_reference_geometry(MODEL,path,guard_time_s=.01)
    np.savez(path,time_s=np.arange(5)*.005)
    with pytest.raises(KeyError):
        screen_reference_geometry(MODEL,path,guard_time_s=.01)
    with pytest.raises(ValueError):
        ReferenceScreenLimits(max_floor_penetration_m=np.nan)


@pytest.mark.parametrize('phase,index', [('initial',0),('terminal',3)])
def test_geometrically_valid_guard_with_excess_velocity_is_rejected(tmp_path,phase,index):
    guard,_=fixture_pose()
    path=tmp_path/'moving.npz'
    velocity=np.zeros((5,54))
    velocity[index,0]=.4
    np.savez(path,time_s=np.arange(5)*.005,qpos=np.tile(guard,(5,1)),qvel=velocity)
    result=screen_reference_geometry(MODEL,path,guard_time_s=.01)
    assert not result['passed']
    assert result['checks']['selected_self_collision']
    assert not result['checks'][phase+'_guard_velocity_readiness']
    assert result['checks'][('terminal' if phase=='initial' else 'initial')+'_guard_velocity_readiness']
    metrics=result['initial_guard_speeds'] if phase=='initial' else result['maximum_terminal_guard_speeds']
    assert metrics['root_linear_speed_m_s']==pytest.approx(.4)
    assert metrics['glove_speed_m_s']==pytest.approx(.4)


def test_missing_velocity_cannot_be_assumed_quiet(tmp_path):
    guard,_=fixture_pose()
    path=tmp_path/'missing_velocity.npz'
    np.savez(path,time_s=np.arange(5)*.005,qpos=np.tile(guard,(5,1)))
    with pytest.raises(KeyError):
        screen_reference_geometry(MODEL,path,guard_time_s=.01)
