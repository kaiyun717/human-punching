"""Conservative FK screening of the exact serialized 5 ms reference master.

A passing result is necessary kinematic evidence, never physical feasibility.
Contacts use the model's actual floor geometry and declared self-collision pairs;
this cannot certify unspecified anatomical collisions or dynamic balance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .references import sha256


@dataclass(frozen=True)
class ReferenceScreenLimits:
    max_floor_penetration_m: float = .001
    max_selected_self_penetration_m: float = .002
    max_support_corner_height_m: float = .003
    max_sole_drift_m: float = .015
    max_joint_limit_excursion_rad: float = 1e-9
    minimum_guard_com_margin_m: float = 0.
    max_guard_root_linear_speed_m_s: float = .05
    max_guard_root_angular_speed_rad_s: float = .12
    max_guard_joint_speed_rad_s: float = .50
    max_guard_glove_speed_m_s: float = .15

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Reference screening limits must be finite and nonnegative")


def screen_reference_geometry(model_path: Path, master_path: Path, *,
                              guard_time_s: float,
                              limits: ReferenceScreenLimits | None = None) -> dict:
    """Fail closed on absent/malformed arrays, missing named geometry or violations.

    This backend declares planted feet: every sole's contact corners must remain
    near the floor. Future heel/pivot references require a different explicit
    support schedule. COM containment is required only at launch and held guard;
    its minimum throughout the strike is reported without asserting balance.
    """
    limits = limits or ReferenceScreenLimits()
    model_path, master_path = Path(model_path), Path(master_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv) != (55, 54) or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Screening requires the constrained physical human with a free root")
    if not np.isfinite(guard_time_s) or guard_time_s <= 0:
        raise ValueError("A positive guard-hold boundary is required")
    with np.load(master_path, allow_pickle=False) as archive:
        times, poses = archive['time_s'].copy(), archive['qpos'].copy()
        velocities = archive['qvel'].copy()
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.allclose(np.diff(times), .005, atol=1e-12, rtol=0)
            or poses.shape != (len(times),55) or not np.isfinite(poses).all()
            or velocities.shape != (len(times),54) or not np.isfinite(velocities).all()
            or not np.isfinite(times).all() or guard_time_s > times[-1]
            or not np.allclose(np.linalg.norm(poses[:,3:7],axis=-1),1.,atol=1e-6)):
        raise ValueError("Malformed 5 ms reference poses or guard boundary")
    floor = model.geom('floor').id
    if (model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
            or model.geom_bodyid[floor] != 0
            or not np.allclose(model.geom_pos[floor],0.,atol=1e-12)
            or not np.allclose(model.geom_quat[floor],[1,0,0,0],atol=1e-12)):
        raise ValueError("Screening expects the declared level floor at world z=0")
    foot_ids = [model.geom('physical/'+side+'Foot').id for side in ('Left','Right')]
    sole_ids = [model.site(side+'Foot_sole').id for side in ('Left','Right')]
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in foot_ids):
        raise ValueError("Planted-foot screening requires the declared solid foot boxes")
    lower, upper = model.jnt_range[1:].T
    excursions = np.maximum(np.maximum(lower-poses[:,7:], poses[:,7:]-upper),0.).max(axis=1)
    data = mujoco.MjData(model)
    root_id = model.body('Hips').id
    floor_penetration = np.zeros(len(times))
    self_penetration = np.zeros(len(times))
    nonfoot_floor_penetration = np.zeros(len(times))
    support = np.zeros(len(times),dtype=bool)
    com_margins = np.empty(len(times))
    soles = np.empty((len(times),2,3))
    corner_heights = np.empty((len(times),2,4))
    glove_speeds = np.empty((len(times),2))
    jacp, jacr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    pairs = {}
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        data.qvel[:] = velocities[i]
        mujoco.mj_forward(model,data)
        for hand,side in enumerate(('Left','Right')):
            mujoco.mj_jacSite(model,data,jacp,jacr,model.site(side+'Hand_center').id)
            glove_speeds[i,hand] = np.linalg.norm(jacp@velocities[i])
        soles[i] = data.site_xpos[sole_ids]
        corners = []
        for side,geom in enumerate(foot_ids):
            size = model.geom_size[geom]
            local = np.array([[sx*size[0],sy*size[1],-size[2]] for sx,sy in product((-1,1),repeat=2)])
            world = local@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
            corners.append(world)
            corner_heights[i,side] = world[:,2]
        corners = np.concatenate(corners)
        support[i] = (corners[:,2].max() <= limits.max_support_corner_height_m
                      and corners[:,2].min() >= -limits.max_floor_penetration_m)
        hull = ConvexHull(corners[:,:2])
        com = data.subtree_com[root_id,:2]
        com_margins[i] = float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])))
        for contact in data.contact:
            g1,g2 = int(contact.geom1),int(contact.geom2)
            penetration = max(0.,-float(contact.dist))
            if penetration <= 0:
                continue
            names = tuple(sorted((model.geom(g1).name,model.geom(g2).name)))
            key = ' | '.join(names)
            if key not in pairs:
                pairs[key] = {'geom_names':list(names),'first_penetration_time_s':float(times[i]),
                              'peak_penetration_time_s':float(times[i]),'max_penetration_m':0.,
                              'at_initial_guard':i==0,'frames':set()}
            row = pairs[key]
            row['frames'].add(i)
            if penetration > row['max_penetration_m']:
                row.update(max_penetration_m=penetration,peak_penetration_time_s=float(times[i]))
            if floor in (g1,g2):
                floor_penetration[i] = max(floor_penetration[i],penetration)
                if next(g for g in (g1,g2) if g != floor) not in foot_ids:
                    nonfoot_floor_penetration[i] = max(nonfoot_floor_penetration[i],penetration)
            elif model.geom_bodyid[g1] and model.geom_bodyid[g2]:
                self_penetration[i] = max(self_penetration[i],penetration)
            else:
                raise ValueError("Unrecognized world contact cannot be silently screened")
    guard_frames = (times >= guard_time_s)
    guard_frames[0] = True
    drift = np.linalg.norm(soles-soles[0],axis=-1).max(axis=1)
    speed_metrics = {
        'root_linear_speed_m_s':np.linalg.norm(velocities[:,:3],axis=-1),
        'root_angular_speed_rad_s':np.linalg.norm(velocities[:,3:6],axis=-1),
        'joint_speed_rad_s':np.max(np.abs(velocities[:,6:]),axis=-1),
        'glove_speed_m_s':glove_speeds.max(axis=1),
    }
    speed_limits = {name:getattr(limits,'max_guard_'+name) for name in speed_metrics}
    ready = np.logical_and.reduce([values <= speed_limits[name] for name,values in speed_metrics.items()])
    initial = np.zeros(len(times),dtype=bool)
    initial[0] = True
    terminal = times >= guard_time_s
    frame_checks = {
        'joint_limits':excursions <= limits.max_joint_limit_excursion_rad,
        'floor_penetration':floor_penetration <= limits.max_floor_penetration_m,
        'selected_self_collision':self_penetration <= limits.max_selected_self_penetration_m,
        'nonfoot_floor_clearance':nonfoot_floor_penetration <= limits.max_floor_penetration_m,
        'planted_foot_support_geometry':support,
        'planted_sole_drift':drift <= limits.max_sole_drift_m,
        'guard_com_inside_support':~guard_frames | (com_margins >= limits.minimum_guard_com_margin_m),
        'initial_guard_velocity_readiness':~initial | ready,
        'terminal_guard_velocity_readiness':~terminal | ready,
    }
    checks = {name:bool(values.all()) for name,values in frame_checks.items()}
    counts = {name:int((~values).sum()) for name,values in frame_checks.items()}
    for row in pairs.values():
        row['penetrating_frame_count'] = len(row.pop('frames'))
    return {'schema':'human_wbc_reference_geometry_screen_v2','passed':all(checks.values()),
        'physics_executed':False,'dynamic_feasibility':'unqualified',
        'model_sha256':sha256(model_path),'master_sha256':sha256(master_path),
        'limits':asdict(limits),'checks':checks,'failed_frame_counts':counts,
        'initial_guard_checks':{name:bool(values[0]) for name,values in frame_checks.items()},
        'initial_guard_speeds':{name:float(values[0]) for name,values in speed_metrics.items()},
        'maximum_terminal_guard_speeds':{name:float(values[terminal].max()) for name,values in speed_metrics.items()},
        'max_joint_limit_excursion_rad':float(excursions.max()),
        'max_floor_penetration_m':float(floor_penetration.max()),
        'max_selected_self_penetration_m':float(self_penetration.max()),
        'max_nonfoot_floor_penetration_m':float(nonfoot_floor_penetration.max()),
        'max_sole_drift_m':float(drift.max()),
        'minimum_foot_corner_height_m':float(corner_heights.min()),
        'maximum_foot_corner_height_m':float(corner_heights.max()),
        'minimum_com_support_margin_m':float(com_margins.min()),
        'minimum_guard_com_support_margin_m':float(com_margins[guard_frames].min()),
        'initial_com_support_margin_m':float(com_margins[0]),
        'penetrating_pairs':sorted(pairs.values(),key=lambda x:-x['max_penetration_m']),
        'scope':'Declared selected self pairs and fixed-foot geometry only; no dynamic support or force proof'}
