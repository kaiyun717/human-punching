import json
from pathlib import Path

import mujoco
import numpy as np

from human_punching.wbc.balance_refinement import contact_wrench_halfspaces, refine_balance_trajectory, BalanceSettings, _linear_subproblem, _collision_linearization
from human_punching.wbc.effort_diagnostics import support_effort_solution
from human_punching.wbc.references import PROJECT_ROOT


def test_wrench_cone_matches_positive_corner_loads_and_rejects_outside_cop():
    corners=np.array([[x,y,0.] for x in (-.4,-.2,.2,.4) for y in (-.2,.2)])
    origin=np.array([.1,-.1,0.])
    mu=.9
    cone=contact_wrench_halfspaces(corners,mu,origin)
    forces=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1]])
    rays=np.array([np.r_[f,np.cross(p-origin,f)] for p in corners for f in forces]).T
    rng=np.random.default_rng(59)
    for _ in range(20):
        wrench=rays@rng.uniform(0,20,len(rays.T))
        assert (cone@wrench).max()<1e-7
        solution=support_effort_solution(np.r_[wrench,0.],np.vstack((rays,np.zeros(len(rays.T)))),np.ones(1))
        assert solution['support_feasible']
    force=np.array([0.,0.,100.])
    wrench=np.r_[force,np.cross(np.array([.7,0,0])-origin,force)]
    assert (cone@wrench).max()>1
    assert not support_effort_solution(np.r_[wrench,0.],np.vstack((rays,np.zeros(len(rays.T)))),np.ones(1))['support_feasible']


def test_balanced_static_guard_is_fixed_point_without_integration(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Offline balance optimization must not integrate physics')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    import human_punching.wbc.balance_refinement as implementation
    original=implementation._unbiased_support_solve
    def checked_projection(model,*args,**kwargs):
        assert not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        return original(model,*args,**kwargs)
    monkeypatch.setattr(implementation,'_unbiased_support_solve',checked_projection)
    fixture=json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())
    guard=np.asarray(fixture['guard_qpos'])
    times=np.arange(201)*.005
    poses=np.tile(guard,(len(times),1))
    result,diagnostics=refine_balance_trajectory(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml',
        times,poses,.8,settings=BalanceSettings(outer_iterations=1))
    np.testing.assert_allclose(result,poses,atol=1e-12)
    assert diagnostics['maximum_preprojection_wrench_violation']<0
    assert not diagnostics['qualified_for_training'] and not diagnostics['physics_executed']


def test_sparse_correction_respects_joint_bound_and_reports_unresolved_slack():
    settings=BalanceSettings(subproblem_solver='linear_program')
    for cap,expected_slack in ((1.,0.),(.1,.1)):
        result=_linear_subproblem(np.zeros(2),-np.eye(2),np.array([.2,.1]),
            np.eye(2),np.full(2,-1.),np.full(2,cap),np.full(2,-.5),np.full(2,.5),
            np.array([[1.,-1.]]),1,settings)
        assert result.success
        assert np.max(result.x[:2])<=cap+1e-8
        assert abs(result.x[-1]-expected_slack)<1e-8


def test_signed_collision_gradient_matches_native_distance_with_constraints_disabled():
    fixture=json.loads((Path(__file__).parent/'fixtures/wbc_geometry_poses.json').read_text())
    pose=np.asarray(fixture['recovery_collision_qpos'])
    model=mujoco.MjModel.from_xml_path(str(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    maps=np.zeros((1,model.nv,9))
    maps[0,model.joint('LeftArm_x').dofadr[0],0]=1.
    maps[0,model.joint('RightArm_y').dofadr[0],1]=1.
    jac,distances=_collision_linearization(model,pose[None],maps,np.ones((1,1)),np.ones(9))
    direction=np.r_[.6,-.3,np.zeros(7)]
    perturbed=pose.copy()
    mujoco.mj_integratePos(model,perturbed,maps[0]@direction,1e-6)
    _,changed=_collision_linearization(model,perturbed[None],maps,np.ones((1,1)),np.ones(9))
    assert len(jac)>0 and np.any(distances<0)
    np.testing.assert_allclose((changed-distances)/1e-6,jac@direction,atol=2e-6,rtol=1e-4)
