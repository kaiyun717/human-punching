from dataclasses import asdict
from types import SimpleNamespace
import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from human_punching.phase2_shape_derivatives import project_derivatives,DT
from human_punching.phase2_shared_shape_target import refine_shared_target_derivatives
from human_punching.phase2_phases import PhaseSchedule
from human_punching.wbc.provenance import PROJECT_ROOT


def estimates(q):
    t=np.arange(len(q))*DT
    v=np.empty_like(q)
    for i in range(len(q)):
        left,right=max(0,i-1),min(len(q)-1,i+1)
        v[i]=(q[right]-q[left])/(t[right]-t[left])
    return v,np.gradient(v,t,axis=0,edge_order=2)


def extrema(coefficients):
    lower=[];upper=[]
    for segment in coefficients:
        lo=[];hi=[]
        for column in segment.T:
            derivative=np.arange(1,6)*column[1:]
            roots=np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
            points=[0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<1]
            values=np.polynomial.polynomial.polyval(points,column)
            lo.append(min(values));hi.append(max(values))
        lower.append(lo);upper.append(hi)
    return np.asarray(lower),np.asarray(upper)


def test_plateau_at_physical_bound_remains_exactly_constant():
    q=np.array([[.8],[.95],[1.],[1.],[1.],[.9],[.7]],dtype=np.float64)
    before=q.copy();v,a=estimates(q)
    result=project_derivatives(q,v,a,np.array([[-1.,1.]]),joint_names=['bound_joint'])
    assert np.array_equal(q,before)
    assert np.all(result.velocity[2:5]==0) and np.all(result.acceleration[2:5]==0)
    assert np.all(result.controls[2:4]==1.)
    lo,hi=extrema(result.coefficients)
    assert np.max(hi)<=1.+1e-10 and np.min(lo)>=-1.-1e-10
    assert result.receipt['changed_node_joint_pairs']>0


def test_feasible_moving_estimates_are_retained_exactly():
    q=np.arange(7,dtype=np.float64)[:,None]*.1
    v=np.full_like(q,20.);a=np.zeros_like(q)
    result=project_derivatives(q,v,a,np.array([[-1.,1.]]))
    assert np.array_equal(result.velocity,v) and np.array_equal(result.acceleration,a)
    assert result.receipt['changed_node_joint_pairs']==0
    assert np.any(result.velocity)


def test_random_bounded_knots_have_analytic_c2_endpoint_and_hull_certificate():
    rng=np.random.default_rng(421)
    q=rng.uniform(-1,1,(25,4));q[8:11,1]=1.;v=rng.normal(0,60,q.shape);a=rng.normal(0,30000,q.shape)
    result=project_derivatives(q,v,a,np.repeat([[-1.,1.]],4,axis=0))
    c=result.coefficients
    q0=c[:,0];q1=c.sum(axis=1)
    v0=c[:,1]/DT;v1=(c*np.arange(6)[None,:,None]).sum(axis=1)/DT
    a0=2*c[:,2]/DT**2;a1=(c*(np.arange(6)*(np.arange(6)-1))[None,:,None]).sum(axis=1)/DT**2
    assert np.allclose(q0,q[:-1],atol=1e-13,rtol=0) and np.allclose(q1,q[1:],atol=1e-12,rtol=0)
    assert np.allclose(v0,result.velocity[:-1],atol=1e-9,rtol=0)
    assert np.allclose(v1,result.velocity[1:],atol=1e-8,rtol=0)
    assert np.allclose(a0,result.acceleration[:-1],atol=1e-7,rtol=0)
    assert np.allclose(a1,result.acceleration[1:],atol=1e-6,rtol=0)
    lo,hi=extrema(c)
    assert np.all(lo>=np.minimum(q[:-1],q[1:])-1e-10)
    assert np.all(hi<=np.maximum(q[:-1],q[1:])+1e-10)


def test_local_projection_does_not_invent_constant_whole_motion():
    q=np.column_stack((np.array([0.,.1,.2,.2,.3,.4]),np.array([0.,.2,.3,.5,.6,.7])))
    v,a=estimates(q);result=project_derivatives(q,v,a,np.repeat([[-1.,1.]],2,axis=0))
    assert result.velocity[0,1]!=0 and result.velocity[-1,1]!=0
    assert np.array_equal(result.velocity[:,1],v[:,1])
    assert result.receipt['unchanged_node_joint_pairs']>0
    assert all(x['joint_name'] in ('0','1') for x in result.receipt['changed_nodes'])


@pytest.mark.parametrize('problem',['out_of_range','nonfinite','float32','49joints','badlimits'])
def test_invalid_knots_are_not_silently_corrected(problem):
    q=np.array([[0.],[.1],[.2]],np.float64);v,a=estimates(q);limits=np.array([[-1.,1.]])
    if problem=='out_of_range':q[1]=1.000001
    elif problem=='nonfinite':a[1]=np.nan
    elif problem=='float32':q=q.astype(np.float32)
    elif problem=='49joints':q=np.repeat(q,49,axis=1);v=np.repeat(v,49,axis=1);a=np.repeat(a,49,axis=1);limits=np.repeat(limits,49,axis=0)
    else:limits=np.array([[1.,-1.]])
    with pytest.raises(ValueError):project_derivatives(q,v,a,limits)


def synthetic_shared():
    model=SimpleNamespace(nq=55,nv=54,nu=48,jnt_type=np.array([0]+[3]*48),
        jnt_limited=np.array([0]+[1]*48),jnt_qposadr=np.array([0]+list(range(7,55))),
        jnt_dofadr=np.array([0]+list(range(6,54))),jnt_range=np.array([[0.,0.]]+[[-1.,1.]]*48),
        joint=lambda index:SimpleNamespace(name=f'hinge{index}'))
    schedule=PhaseSchedule(4,8,12,16,20)
    q=np.zeros((21,55));q[:,2]=1.;q[:,3]=1.;q[10:13,7]=1.;q[9,7]=.8;q[13,7]=.9
    v=np.zeros((21,54));a=np.zeros_like(v);v[:,6:],a[:,6:]=estimates(q[:,7:]);v[:9]=0;a[:9]=0
    candidate=dict(qpos=q,qvel=v,qacc=a,schedule=schedule,connector_coefficients=np.zeros((6,48)),
        connector_qpos_addresses=np.arange(7,55),connector_dof_addresses=np.arange(6,54),
        time_s=np.arange(21)*DT,t=np.arange(21)*DT,source_start_tick=4)
    return model,candidate,dict(schedule=asdict(schedule))


def test_post_ik_projection_preserves_root_connector_knots_and_event_clocks():
    model,candidate,metadata=synthetic_shared();old=copy.deepcopy(candidate)
    out,receipt=refine_shared_target_derivatives(model,candidate,metadata)
    assert out['qpos'].tobytes()==old['qpos'].tobytes()
    assert out['qvel'][:5].tobytes()==old['qvel'][:5].tobytes()
    assert out['qacc'][:5].tobytes()==old['qacc'][:5].tobytes()
    assert np.array_equal(out['qvel'][:,:6],old['qvel'][:,:6])
    assert out['schedule']==old['schedule']
    assert receipt['physical_qualification'] is False and receipt['feedforward_recomputed'] is False
    assert out['source_segment_coefficients'].shape==(16,6,48)
    assert np.array_equal(out['source_segment_qpos_addresses'],np.arange(7,55))
    assert np.array_equal(out['source_segment_dof_addresses'],np.arange(6,54))
    assert receipt['independent_polynomial_extrema_screen_passed']
    assert receipt['polynomial_extrema_screen_intervals']==16


@pytest.mark.parametrize('field',['root','root_tangent','connector_endpoint','clock','addresses','nonfinite','source_start'])
def test_post_ik_projection_rejects_unsupported_reference_contracts(field):
    model,candidate,metadata=synthetic_shared()
    if field=='root':candidate['qpos'][6,0]=.001
    elif field=='root_tangent':candidate['qvel'][6,0]=.001
    elif field=='connector_endpoint':candidate['qvel'][4,6]=.01
    elif field=='clock':candidate['t'][3]+=.001
    elif field=='addresses':model.jnt_dofadr[2]=6
    elif field=='nonfinite':candidate['qacc'][0,0]=np.nan
    else:candidate['source_start_tick']=8
    with pytest.raises(ValueError):refine_shared_target_derivatives(model,candidate,metadata)


def test_new_composition_copy_preserves_all_other_original_algorithms():
    a=ast.parse((PROJECT_ROOT/'src/human_punching/phase2_shared_reference.py').read_text())
    b=ast.parse((PROJECT_ROOT/'src/human_punching/phase2_shared_reference_shape_v3.py').read_text())
    a={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    b={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert set(a)==set(b)
    for name in a:
        if name!='compose_shared_reference':assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
