import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.wbc.effort_diagnostics import (contact_edges,support_effort_solution,
                                                 unconstrained_inverse)
from human_punching.wbc.references import PROJECT_ROOT


def support_matrix(mu=.5):
    # Four world-frame points at the unactuated base height isolate friction
    # from tipping. One uncoupled scalar actuator makes the torque cap explicit.
    matrices=[]
    pyramid=np.array([[mu,-mu,0,0],[0,0,mu,-mu],[1,1,1,1]])
    for x,y in ((-.2,-.2),(-.2,.2),(.2,-.2),(.2,.2)):
        force=np.zeros((7,3));force[:3]=np.eye(3)
        force[3:6]=np.array([[0,0,y],[0,0,-x],[-y,x,0]])
        matrices.append(force@pyramid)
    return np.concatenate(matrices,axis=1)


def test_free_base_requires_ground_load_and_actuator_caps_remain_separate():
    required=np.array([0.,0.,98.1,0.,0.,0.,20.])
    result=support_effort_solution(required,support_matrix(),np.array([10.]))
    assert result['support_feasible']
    assert not result['effort_within_caps']
    assert result['minimum_peak_effort_ratio']==pytest.approx(2.)
    np.testing.assert_allclose(result['base_residual'],0.,atol=1e-9)
    assert result['edge_loads'].sum()==pytest.approx(98.1)


def test_negative_normal_load_is_reported_infeasible_not_clamped_into_success():
    required=np.array([0.,0.,-98.1,0.,0.,0.,0.])
    result=support_effort_solution(required,support_matrix(),np.array([10.]))
    assert result['status']=='support_infeasible'
    assert not result['support_feasible'] and not result['effort_within_caps']
    assert result['minimum_peak_effort_ratio'] is None
    assert abs(result['base_residual'][2])>=98.1-1e-6


def test_friction_infeasibility_differs_from_effort_overload():
    required=np.array([40.,0.,100.,0.,0.,0.,0.])
    assert support_effort_solution(required,support_matrix(.5),np.array([10.]))['support_feasible']
    required[0]=60.
    result=support_effort_solution(required,support_matrix(.5),np.array([10.]))
    assert result['status']=='support_infeasible'
    assert np.linalg.norm(result['base_residual'])>1


def test_native_inverse_gravity_for_unactuated_free_body_without_integration(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Offline diagnostics must not integrate dynamics')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='disabled'):
        unconstrained_inverse(model,data,model.qpos0,np.zeros(6),np.zeros(6))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    force=unconstrained_inverse(model,data,model.qpos0,np.zeros(6),np.zeros(6))
    np.testing.assert_allclose(force,[0,0,19.62,0,0,0],atol=1e-10)


def test_actual_human_guard_root_wrench_is_balanced_only_by_hypothetical_contacts():
    fixture=json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())
    directory=PROJECT_ROOT/'outputs/four-punch-milestone/jab'
    model=mujoco.MjModel.from_xml_path(str(directory/'model/human.xml'))
    body=json.loads((directory/'body_metadata.json').read_text())
    data=mujoco.MjData(model)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    required=unconstrained_inverse(model,data,np.array(fixture['guard_qpos']),np.zeros(54),np.zeros(54))
    edges,points,mus=contact_edges(model,data)
    result=support_effort_solution(required,edges,np.array(body['effort']))
    assert result['support_feasible']
    np.testing.assert_allclose(result['base_residual'],0.,atol=1e-7)
    assert result['edge_loads'].sum()==pytest.approx(75*9.81)
    assert points.shape==(8,3) and np.all(points[:,2]==0)
    assert np.all(mus>0)
    assert np.all(data.xfrc_applied==0) and np.all(data.qfrc_applied==0)
