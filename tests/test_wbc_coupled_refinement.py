import json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from human_punching.wbc.coupled_refinement import CoupledSettings,solve_coupled_subproblem,retime_first_contact,_screened_line_search
from human_punching.wbc.references import PROJECT_ROOT,tangent_velocities
from human_punching.wbc.planning import _smoothstep


def problem(joint_derivative):
    world=np.array([[0.,0.,98.1,0,0,0]])
    return solve_coupled_subproblem(np.zeros(1),world,np.zeros((1,6,1)),np.array([[20.]]),
        np.array([[[joint_derivative]]]),np.array([[0],[0],[1],[0],[0],[0]]),np.zeros((1,1,1)),
        np.array([10.]),np.ones((1,1)),np.array([-1.]),np.array([1.]),np.zeros((0,1)),
        np.array([-1.]),np.array([1.]),98.1,CoupledSettings())


def test_coupled_allocation_changes_path_to_respect_effort_without_root_actuator():
    result,audit=problem(-20.)
    assert result.success
    assert audit['step'][0] >= .5-1e-9
    assert audit['predicted_peak_joint_effort_ratio'] <= 1+1e-9
    assert audit['root_residual_max_normalized'] < 1e-9
    np.testing.assert_allclose(audit['contact_edge_loads'],[[98.1]],atol=1e-7)


def test_impossible_joint_effort_is_rejected_instead_of_clipped():
    result,audit=problem(0.)
    assert not result.success and result.status==2
    assert audit is None


def test_line_search_records_invalid_support_trial_and_uses_valid_smaller_step():
    def evaluate(coefficients):
        if coefficients[0]>.5:
            raise ValueError('Foot corner outside declared support clearance')
        return coefficients
    def merit(value,coefficients):
        return float(1-value[0]),{'support_checked':True},None
    best,rejected=_screened_line_search(evaluate,merit,np.zeros(1),np.ones(1))
    assert best[1]==.5 and best[0]==.5
    assert rejected==[{'step_fraction':1.,'reason':'Foot corner outside declared support clearance'}]
    with pytest.raises(ValueError,match='including the unchanged'):
        _screened_line_search(lambda c: (_ for _ in ()).throw(ValueError('No support')),merit,np.zeros(1),np.ones(1))


def test_changed_path_retimes_true_fk_contact_and_fresh_derivatives(monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *args: (_ for _ in ()).throw(AssertionError('No dynamics')))
    path=PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    guard=np.array(json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())['guard_qpos'])
    times=np.arange(281)*.005;poses=np.tile(guard,(len(times),1))
    address=model.joint('LeftArm_z').qposadr[0]
    poses[:,address]+=.7*_smoothstep((times-.1)/.3)*_smoothstep((.7-times)/.3)
    data.qpos[:]=poses[80];mujoco.mj_forward(model,data)
    target=data.site_xpos[model.site('LeftHand_center').id].copy()
    final_time,final_pose,velocity,record=retime_first_contact(path,times,poses,target,0,.55,.6,.9)
    distances=[]
    for pose in final_pose:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        distances.append(np.linalg.norm(data.site_xpos[model.site('LeftHand_center').id]-target))
    first=final_time[np.flatnonzero(np.array(distances)<=.14)[0]]
    assert abs(first-.55)<=.005+1e-10
    assert record['source_first_contact_s'] < .55
    assert record['all_derivatives_recomputed']
    np.testing.assert_array_equal(velocity,tangent_velocities(model,final_time,final_pose))
    np.testing.assert_allclose(velocity[[0,-1]],0,atol=1e-10)
    np.testing.assert_allclose(np.gradient(velocity,final_time,axis=0,edge_order=2)[[0,-1]],0,atol=1e-8)
