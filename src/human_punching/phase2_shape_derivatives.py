"""Local C2 quintic derivative projection for bounded scalar hinge knots.

Positions and clocks never change. Every quintic Bezier control lies between
its two position endpoints, a sufficient continuous no-overshoot certificate.
The shared nodal velocity/acceleration makes the assembled curve C2. Root
positions, quaternion interpolation and free-root tangents are outside this API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools

import numpy as np

VERSION="task10b_scalar_hinge_local_c2_bezier_projection_v1"
DT=.005
# Numerical comparison in the two-variable projection, not a physical range
# enlargement. Independent polynomial screening retains its original1e-10rad.
ROUND_OFF=64*np.finfo(np.float64).eps


def _digest(value):
    array=np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def _constraints(q,previous,following):
    """A[d,e]<=b, where d=h*v and e=h²*a both have angular units."""
    rows=[];bounds=[]
    for neighbor,direction in ((previous,-1.),(following,1.)):
        if neighbor is None:continue
        lower,upper=min(q,neighbor),max(q,neighbor)
        for coefficients in ((direction/5.,0.),(2*direction/5.,1/20.)):
            rows.extend((coefficients,tuple(-x for x in coefficients)))
            bounds.extend((upper-q,q-lower))
    return np.asarray(rows,float),np.asarray(bounds,float)


def _project_pair(estimate,A,b):
    """Exact Euclidean projection in2D by feasible faces/vertices enumeration."""
    estimate=np.asarray(estimate,float)
    if np.all(A@estimate<=b+ROUND_OFF):return estimate.copy()
    candidates=[np.zeros(2)]
    for normal,offset in zip(A,b,strict=True):
        candidate=estimate-normal*((normal@estimate-offset)/(normal@normal))
        if np.all(A@candidate<=b+ROUND_OFF):candidates.append(candidate)
    for i,j in itertools.combinations(range(len(b)),2):
        matrix=A[[i,j]]
        if abs(float(np.linalg.det(matrix)))<=1.e-15:continue
        candidate=np.linalg.solve(matrix,b[[i,j]])
        if np.all(A@candidate<=b+ROUND_OFF):candidates.append(candidate)
    chosen=min(candidates,key=lambda x:float(np.sum((x-estimate)**2)))
    if not np.isfinite(chosen).all() or np.any(A@chosen>b+ROUND_OFF):
        raise ValueError("Bounded nodal projection failed its explicit linear certificate")
    return chosen.copy()


def quintic_controls(q,v,a):
    """Six Bezier controls on each exact5ms interval, with shape(N-1,6,J)."""
    return np.stack((q[:-1],q[:-1]+DT*v[:-1]/5.,
        q[:-1]+2*DT*v[:-1]/5.+DT**2*a[:-1]/20.,
        q[1:]-2*DT*v[1:]/5.+DT**2*a[1:]/20.,
        q[1:]-DT*v[1:]/5.,q[1:]),axis=1)


def power_coefficients(q,v,a):
    """Power coefficients in u=t/h; derivatives are analytic, never re-FDed."""
    c0=q[:-1];c1=DT*v[:-1];c2=.5*DT**2*a[:-1]
    r0=q[1:]-c0-c1-c2;r1=DT*v[1:]-c1-2*c2;r2=DT**2*a[1:]-2*c2
    return np.stack((c0,c1,c2,10*r0-4*r1+.5*r2,-15*r0+7*r1-r2,6*r0-3*r1+.5*r2),axis=1)


@dataclass(frozen=True)
class Projection:
    velocity:np.ndarray
    acceleration:np.ndarray
    coefficients:np.ndarray
    controls:np.ndarray
    receipt:dict


def project_derivatives(positions,velocity,acceleration,limits,*,joint_names=None):
    q=np.asarray(positions);v=np.asarray(velocity);a=np.asarray(acceleration);limits=np.asarray(limits,float)
    if (q.dtype!=np.float64 or v.dtype!=np.float64 or a.dtype!=np.float64
            or q.ndim!=2 or not 3<=len(q)<=6001 or q.shape!=v.shape or q.shape!=a.shape
            or not 1<=q.shape[1]<=48 or limits.shape!=(q.shape[1],2)
            or any(not np.isfinite(x).all() for x in (q,v,a,limits))
            or np.any(limits[:,0]>=limits[:,1])):
        raise ValueError("Bounded finite float64 scalar-hinge knots and actual joint ranges required")
    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):
        raise ValueError("Position knots outside physical ranges cannot be repaired by derivative projection")
    names=list(joint_names) if joint_names is not None else [str(i) for i in range(q.shape[1])]
    if len(names)!=q.shape[1] or len(set(names))!=len(names):raise ValueError("Unique explicit hinge names required")
    out_v=v.copy();out_a=a.copy();changes=[];affected=set();flat=np.diff(q,axis=0)==0
    for knot in range(len(q)):
        for joint in range(q.shape[1]):
            previous=q[knot-1,joint] if knot else None
            following=q[knot+1,joint] if knot+1<len(q) else None
            constant=(knot>0 and flat[knot-1,joint]) or (knot<len(q)-1 and flat[knot,joint])
            estimate=np.array([DT*v[knot,joint],DT**2*a[knot,joint]])
            if constant:
                projected=np.zeros(2)
            else:
                A,b=_constraints(q[knot,joint],previous,following)
                projected=_project_pair(estimate,A,b)
            if not np.array_equal(projected,estimate):
                out_v[knot,joint]=projected[0]/DT;out_a[knot,joint]=projected[1]/DT**2
                changes.append(dict(knot=knot,joint_index=joint,joint_name=names[joint],
                    old_velocity_rad_s=float(v[knot,joint]),new_velocity_rad_s=float(out_v[knot,joint]),
                    old_acceleration_rad_s2=float(a[knot,joint]),new_acceleration_rad_s2=float(out_a[knot,joint]),
                    exact_flat_neighbor=bool(constant),scaled_squared_change=float(np.sum((projected-estimate)**2))))
                for interval in (knot-1,knot):
                    if 0<=interval<len(q)-1:affected.add((interval,joint))
    controls=quintic_controls(q,out_v,out_a)
    lower=np.minimum(q[:-1],q[1:]);upper=np.maximum(q[:-1],q[1:])
    endpoint_excess=np.maximum(np.maximum(lower[:,None,:]-controls,controls-upper[:,None,:]),0.)
    if float(endpoint_excess.max(initial=0))>ROUND_OFF:
        raise ValueError("Projected quintic control hull escaped its unchanged endpoint interval")
    coefficients=power_coefficients(q,out_v,out_a)
    receipt=dict(schema=VERSION,physics_dt_s=DT,positions_sha256=_digest(q),positions_unchanged=True,
        input_velocity_sha256=_digest(v),input_acceleration_sha256=_digest(a),
        velocity_sha256=_digest(out_v),acceleration_sha256=_digest(out_a),
        limits_sha256=_digest(limits),coefficient_sha256=_digest(coefficients),control_sha256=_digest(controls),
        knots=len(q),joints=q.shape[1],changed_node_joint_pairs=len(changes),
        unchanged_node_joint_pairs=q.size-len(changes),changed_nodes=changes,
        affected_interval_joint_pairs=[dict(interval=i,joint_index=j,joint_name=names[j]) for i,j in sorted(affected)],
        maximum_absolute_velocity_change_rad_s=float(abs(out_v-v).max(initial=0)),
        maximum_absolute_acceleration_change_rad_s2=float(abs(out_a-a).max(initial=0)),
        maximum_control_endpoint_interval_roundoff_rad=float(endpoint_excess.max(initial=0)),
        numerical_roundoff_rad=ROUND_OFF,physical_joint_ranges_enlarged=False,
        constraint="Each quintic Bezier control is inside its original two-knot interval; physical ranges unchanged",
        objective="Minimize (h*v-h*v_est)^2+(h^2*a-h^2*a_est)^2 separately at each scalar joint/node; retain feasible estimates exactly",
        constant_intervals="Exact repeated position knots keep a constant interval: both endpoint v/a are exactly zero",
        derivative_semantics="Unique shared nodal v/a and analytic quintic intervals give C2 position; no finite-difference acceleration identity is claimed",
        root_tangents_modified=False,event_ticks_modified=False,reference_generated=False,physical_qualification=False)
    return Projection(out_v,out_a,coefficients,controls,receipt)
