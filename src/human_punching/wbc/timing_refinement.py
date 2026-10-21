"""Explicit post-contact time allocation; never a playback-speed TTC command."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import mujoco
import numpy as np

from .planning_refinement import _PoseCurve,_fk_contact
from .references import tangent_velocities


@dataclass(frozen=True)
class PostContactTimingSettings:
    window_start_s: float
    window_end_s: float
    added_duration_s: float

    def __post_init__(self):
        if (not all(np.isfinite(value) for value in asdict(self).values())
                or self.window_start_s<=0 or self.window_end_s<=self.window_start_s
                or self.added_duration_s<=0):
            raise ValueError('Positive finite post-contact window and added duration required')


def _expanded_time(source_time,settings):
    """C2 monotone source-to-final clock; identical before the declared window."""
    u=np.clip((np.asarray(source_time)-settings.window_start_s)/
              (settings.window_end_s-settings.window_start_s),0.,1.)
    quintic=u**3*(10-15*u+6*u*u)
    return np.asarray(source_time)+settings.added_duration_s*quintic


def refine_post_contact_timing(model_path,times,poses,target,active_hand,requested_ttc_s,
                               outward_end_s,guard_time_s,*,settings):
    """Slow one declared post-contact interval, retaining path and launch command.

    This is a planner variant with freshly sampled native kinematics. It does not
    apply forces or claim physical timing. The actual first analytic contact is
    measured before and after; the clock is exactly identity through that event.
    """
    times,poses=np.asarray(times),np.asarray(poses)
    model=mujoco.MjModel.from_xml_path(str(model_path))
    if (times.ndim!=1 or len(times)<3 or poses.shape!=(len(times),model.nq)
            or not np.allclose(np.diff(times),.005,atol=1e-12,rtol=0)
            or active_hand not in (0,1) or not np.isfinite(poses).all()
            or not 0<requested_ttc_s<outward_end_s<guard_time_s<=times[-1]):
        raise ValueError('Complete 200 Hz reference and ordered punch phases required')
    curve=_PoseCurve(times,poses)
    site=model.site(('Left' if active_hand==0 else 'Right')+'Hand_center').id
    old_contact=_fk_contact(model,curve,np.asarray(target),site,outward_end_s)
    if settings.window_start_s<=max(old_contact,requested_ttc_s) or settings.window_end_s>=guard_time_s:
        raise ValueError('Timing window must start after first contact and end before guard')
    final_end=float(np.ceil((_expanded_time(times[-1],settings)-1e-10)/.02)*.02)
    final_times=np.arange(round(final_end/.005)+1)*.005
    final_queries=np.minimum(final_times,float(_expanded_time(times[-1],settings)))
    # Monotone inversion uses array bisection, not interpolation of clock knots.
    low=np.zeros_like(final_queries);high=np.full_like(final_queries,times[-1])
    for _ in range(55):
        midpoint=.5*(low+high)
        left=_expanded_time(midpoint,settings)<final_queries
        low=np.where(left,midpoint,low);high=np.where(left,high,midpoint)
    source_clock=.5*(low+high)
    source_clock[final_times<=settings.window_start_s]=final_times[final_times<=settings.window_start_s]
    final_poses=curve(source_clock)
    final_guard=float(_expanded_time(guard_time_s,settings))
    final_outward=float(_expanded_time(outward_end_s,settings))
    final_poses[0]=poses[0];final_poses[final_times>=final_guard]=poses[-1]
    velocity=tangent_velocities(model,final_times,final_poses)
    acceleration=np.gradient(velocity,final_times,axis=0,edge_order=2)
    if np.max(abs(velocity[[0,-1]]))>1e-10 or np.max(abs(acceleration[[0,-1]]))>1e-8:
        raise ValueError('Changed reference lost zero sampled endpoint derivatives')
    new_contact=_fk_contact(model,_PoseCurve(final_times,final_poses),np.asarray(target),site,final_outward)
    if abs(new_contact-old_contact)>1e-6:
        raise ValueError('Post-contact refinement unexpectedly changed first contact')
    return final_times,final_poses,velocity,{
        'schema':'human_wbc_postcontact_timing_v1','qualified_for_training':False,'physics_executed':False,
        'settings':asdict(settings),'source_first_contact_s':old_contact,'final_first_contact_s':new_contact,
        'requested_ttc_s':requested_ttc_s,'outward_end_s':final_outward,'guard_time_s':final_guard,
        'source_clock_s':source_clock,'first_contact_clock_identity':True,
        'path_geometry':'Unchanged continuous pose curve; only post-contact time allocation changes.',
        'all_derivatives_recomputed':True}
