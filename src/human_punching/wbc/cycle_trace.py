"""Per-physics-step evidence and offline kinematics for complete-cycle evaluation.

A post-substep hook sees *current* qpos/qvel but forces and contacts from the
interval just integrated. These time bases stay explicit. No forward/step call
is made on the live simulation. The optional CPU reconstruction calls only
kinematics/comPos/Jacobians; it never recomputes a contact force.

The recorder is deliberately slow (synchronizing CPU copies) and is an evaluation
tool, not a production-throughput measurement. Wire metric_function() through a
MetricsTermCfg(per_substep=True), then capture_launch(env) immediately before the
first commanded integration. Do not use a 50 Hz post-policy callback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .contracts import PunchRequest, TimingContract


@dataclass(frozen=True)
class TraceBindings:
    """Named physical meanings; addresses are resolved against the actual model.

    The forefoot anchor is a fixed material point, never a moving center of
    pressure. Its initial location remains the drift origin through heel lift
    and unloading. The patch radius is an explicit diagnostic definition.
    """
    prefix: str = "robot/"
    root_body: str = "Hips"
    guard_body: str = "Chest"
    glove_sites: tuple[str, str] = ("LeftHand_center", "RightHand_center")
    foot_bodies: tuple[str, str] = ("LeftFoot", "RightFoot")
    foot_geoms: tuple[str, str] = ("physical/LeftFoot", "physical/RightFoot")
    floor_geoms: tuple[str, ...] = ("terrain",)
    human_geom_prefix: str = "robot/physical/"
    g1_geom_names: tuple[str, ...] = ()
    allowed_self_pairs: tuple[tuple[str, str], ...] = ()
    anchor_local: tuple[float, float, float] = (0., -.185, -.05)
    patch_radius_m: float = .065
    minimum_normal_force_n: float = 1.

    def __post_init__(self):
        if self.patch_radius_m <= 0 or self.minimum_normal_force_n <= 0:
            raise ValueError("Positive declared support-patch parameters are required")


def _array(value) -> np.ndarray:
    value = getattr(value, "_tensor", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def _id(model, kind: str, name: str) -> int:
    try:
        return int(getattr(model, kind)(name).id)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Trace model lacks {kind} {name!r}") from error


def derive_kinematics(model, qpos: np.ndarray, qvel: np.ndarray,
                      bindings: TraceBindings = TraceBindings()) -> dict[str, np.ndarray]:
    """Fresh FK and analytic velocities for one recorded environment, without dynamics.

    Guard features are relative to the *current chest*. Relative linear velocity
    includes the chest's rotating-frame term; simply subtracting velocities is
    wrong during torso rotation. Root angular velocity is returned in world axes.
    """
    import mujoco

    qpos, qvel = np.asarray(qpos, float), np.asarray(qvel, float)
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq
            or qvel.shape != (len(qpos), model.nv)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()):
        raise ValueError("Finite, full-model qpos/qvel at matching samples are required")
    b = bindings
    root = _id(model, "body", b.prefix + b.root_body)
    chest = _id(model, "body", b.prefix + b.guard_body)
    gloves = [_id(model, "site", b.prefix + name) for name in b.glove_sites]
    feet = [_id(model, "body", b.prefix + name) for name in b.foot_bodies]
    shapes = {"glove_position_w": (2, 3), "root_position_w": (3,),
              "root_quat_wxyz": (4,), "root_linear_velocity_w": (3,),
              "root_angular_velocity_w": (3,), "glove_position_guard_frame": (2, 3),
              "glove_quat_guard_frame": (2, 4), "glove_velocity_guard_frame": (2, 3),
              "support_anchor_position_w": (2, 3)}
    result = {key: np.empty((len(qpos), *shape)) for key, shape in shapes.items()}
    data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    for k, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        result["root_position_w"][k] = data.xpos[root]
        result["root_quat_wxyz"][k] = data.xquat[root]
        mujoco.mj_jacBody(model, data, jp, jr, root)
        result["root_linear_velocity_w"][k] = jp @ v
        result["root_angular_velocity_w"][k] = jr @ v
        mujoco.mj_jacBody(model, data, jp, jr, chest)
        chest_v, chest_w = jp @ v, jr @ v
        rotation = data.xmat[chest].reshape(3, 3)
        for side, site in enumerate(gloves):
            position = data.site_xpos[site]
            offset = position - data.xpos[chest]
            result["glove_position_w"][k, side] = position
            result["glove_position_guard_frame"][k, side] = rotation.T @ offset
            relative_rotation = rotation.T @ data.site_xmat[site].reshape(3, 3)
            mujoco.mju_mat2Quat(result["glove_quat_guard_frame"][k, side],
                               relative_rotation.ravel())
            mujoco.mj_jacSite(model, data, jp, jr, site)
            result["glove_velocity_guard_frame"][k, side] = rotation.T @ (
                jp @ v - chest_v - np.cross(chest_w, offset))
        for side, body in enumerate(feet):
            result["support_anchor_position_w"][k, side] = (
                data.xpos[body] + data.xmat[body].reshape(3, 3) @ b.anchor_local)
    return result


def classify_contacts(model, contacts: Mapping[str, np.ndarray], anchor_position_w: np.ndarray,
                      bindings: TraceBindings = TraceBindings()) -> dict[str, np.ndarray | bool]:
    """Classify actual saved contacts; no height-based contact inference.

    contact.force is the public MuJoCo-Warp API's contact-frame wrench, whose
    first component is nonnegative normal force. Positions and anchor positions
    must refer to the same interval-start kinematics. All contact pairs involving
    the human are classified, including unexpected self/body/floor collisions.
    """
    b = bindings
    geom = np.asarray(contacts["geom"], dtype=int)
    pos = np.asarray(contacts["pos"], dtype=float)
    force = np.asarray(contacts["force"], dtype=float)
    distance = np.asarray(contacts["dist"], dtype=float)
    if (geom.ndim != 2 or geom.shape[1] != 2 or pos.shape != (len(geom), 3)
            or force.shape != (len(geom), 6) or distance.shape != (len(geom),)
            or not all(np.isfinite(x).all() for x in (pos, force, distance))):
        raise ValueError("Malformed actual contact records")
    anchors = np.asarray(anchor_position_w, dtype=float)
    if anchors.shape != (2, 3) or not np.isfinite(anchors).all():
        raise ValueError("Two finite interval-start support anchors are required")
    feet = [_id(model, "geom", b.prefix + name) for name in b.foot_geoms]
    floor = {_id(model, "geom", name) for name in b.floor_geoms}
    human = {i for i in range(model.ngeom)
             if (model.geom(i).name or "").startswith(b.human_geom_prefix)}
    g1 = {_id(model, "geom", name) for name in b.g1_geom_names}
    allowed = {frozenset((_id(model, "geom", a), _id(model, "geom", c)))
               for a, c in b.allowed_self_pairs}
    support, patch = np.zeros(2, bool), np.zeros(2, bool)
    forbidden, human_g1 = False, False
    for pair, point, wrench, depth in zip(geom, pos, force, distance, strict=True):
        pair_set = set(pair)
        if not pair_set.intersection(human):
            continue
        active = depth <= 0 or wrench[0] > 0
        if not active:
            continue
        if pair_set.intersection(g1):
            human_g1 = True
            continue
        foot_ground = False
        for side, foot in enumerate(feet):
            if foot in pair_set and pair_set.intersection(floor):
                foot_ground = True
                loaded = wrench[0] >= b.minimum_normal_force_n
                support[side] |= loaded
                patch[side] |= loaded and np.linalg.norm(point[:2] - anchors[side, :2]) <= b.patch_radius_m
        if not foot_ground and frozenset(pair) not in allowed:
            forbidden = True
    return {"support_contact": support, "support_patch_contact": patch,
            "forbidden_collision": forbidden, "human_g1_collision": human_g1}


class CycleTraceRecorder:
    """Raw recorder with immutable per-environment requests and no silent defaults.

    ``event_provider(env)`` must provide Boolean arrays ``external_support`` and
    ``pose_replacement``. These cannot be established from qpos alone. The caller
    owns instrumentation of its state-write/assistance interfaces. Missing events
    prevent construction of acceptance traces. All-body xfrc and all qfrc_applied
    are independently retained; equal and opposite assistance cannot cancel.
    """
    def __init__(self, requests: Sequence[PunchRequest], *,
                 event_provider: Callable | None = None):
        if not requests:
            raise ValueError("At least one frozen launch request is required")
        self.requests = tuple(requests)
        self.event_provider = event_provider
        self.reset_counts = np.zeros(len(requests), dtype=np.int64)
        self.rows: list[dict[str, np.ndarray]] = []
        self.contacts: list[dict[str, np.ndarray]] = []
        self.model = None

    def note_reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        if not isinstance(indices, slice):
            indices = _array(indices).astype(int)
        self.reset_counts[indices] += 1

    def capture_launch(self, env):
        if self.rows:
            raise ValueError("Launch can only be captured once")
        self._capture(env)

    def metric_function(self):
        def record(env):
            import torch
            if not self.rows:
                raise RuntimeError("Capture the launch before the first integration")
            self._capture(env)
            return torch.zeros(env.num_envs, device=env.device)
        record.reset = self.note_reset
        return record

    def _capture(self, env):
        import mujoco_warp as mjwarp
        import warp as wp
        if env.num_envs != len(self.requests):
            raise ValueError("Request/environment counts disagree")
        TimingContract(float(env.sim.mj_model.opt.timestep), float(env.step_dt),
                       env.cfg.decimation, 1)
        if self.model is not None and self.model is not env.sim.mj_model:
            raise ValueError("The compiled scene cannot change within a cycle trace")
        self.model = env.sim.mj_model
        data = env.sim.data
        names = ("time", "qpos", "qvel", "actuator_force", "qfrc_actuator",
                 "qfrc_applied", "xfrc_applied", "overflow")
        row = {name: _array(getattr(data, name)) for name in names}
        row["step_counter"] = np.asarray(env._sim_step_counter)
        row["reset_counts"] = self.reset_counts.copy()
        if self.event_provider is not None:
            for name, value in self.event_provider(env).items():
                if name not in ("external_support", "pose_replacement"):
                    raise ValueError(f"Undeclared trace event: {name}")
                value = _array(value)
                if value.dtype.kind != "b" or value.shape != (env.num_envs,):
                    raise ValueError(f"{name} requires one recorded Boolean per environment")
                row[name] = value
        native = env.sim.wp_data
        count = int(_array(native.nacon)[0])
        capacity = native.contact.geom.shape[0]
        # Never read out of bounds or silently erase a global overflow.
        if count < 0 or count > capacity:
            raise RuntimeError("Native contact count exceeds allocated contact storage")
        contact = {name: _array(getattr(native.contact, name))[:count]
                   for name in ("geom", "pos", "dist", "worldid")}
        if count:
            ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32,
                           device=native.qpos.device)
            forces = wp.zeros(count, dtype=wp.spatial_vector, device=native.qpos.device)
            mjwarp.contact_force(env.sim.wp_model, native, ids, False, forces)
            contact["force"] = forces.numpy()
        else:
            contact["force"] = np.zeros((0, 6))
        self.rows.append(row)
        self.contacts.append(contact)

    def raw_arrays(self) -> dict[str, np.ndarray]:
        """NPZ-ready measured data, with lossless offsets for ragged contacts.

        This remains useful when offline conversion fails. It records no
        successful-admission or physical-qualification claim.
        """
        if not self.rows:
            return {}
        names = set(self.rows[0])
        if any(set(row) != names for row in self.rows):
            raise ValueError("Trace event fields changed during recording")
        result = {key: np.stack([row[key] for row in self.rows]) for key in names}
        result["contact_offsets"] = np.r_[0, np.cumsum([len(row["geom"]) for row in self.contacts])]
        for key in self.contacts[0]:
            result["contact_" + key] = np.concatenate([row[key] for row in self.contacts], axis=0)
        result["request_sha256"] = np.asarray([request.fingerprint for request in self.requests])
        return result

    def episode_arrays(self, env_index: int, *, reference_glove_position_w: np.ndarray,
                       joint_names: Sequence[str], outward_end_s: float,
                       bindings: TraceBindings = TraceBindings(), admission_passed: bool = False,
                       domain_version: str = "", inter_character_collision_enabled: bool = False,
                       evidence_kind: str = "physics_rollout") -> dict[str, np.ndarray | str | bool | float]:
        """Build evaluator schema only when all required measured inputs exist.

        Support/contact flags at row k describe [t[k-1], t[k]]; row 0 is the
        launch forward state. Their geometry is classified against FK at k-1.
        Effort is likewise interval evidence. Contact timestamps are never used
        as analytic target timestamps; analytic targeting uses fresh FK at k.
        """
        if not self.rows or self.model is None or not 0 <= env_index < len(self.requests):
            raise ValueError("A recorded environment is required")
        missing = {name for name in ("external_support", "pose_replacement")
                   if any(name not in row for row in self.rows)}
        if missing:
            raise ValueError(f"Missing recorded state-write/assistance evidence: {sorted(missing)}")
        model, request = self.model, self.requests[env_index]
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        ticks = arrays["step_counter"] - arrays["step_counter"][0]
        if not np.array_equal(ticks, np.arange(len(self.rows))):
            raise ValueError("Dropped, repeated, or hidden physics steps in cycle trace")
        raw_time = arrays["time"][:, env_index].astype(float)
        tolerance = max(5e-8, 8*np.finfo(np.float32).eps*float(abs(raw_time).max()))
        if abs(raw_time[0] - request.launch_time_s) > tolerance:
            raise ValueError("Request launch timestamp differs from the recorded simulator launch")
        if np.max(abs(np.diff(raw_time) - .005), initial=0) > tolerance:
            raise ValueError("Native clock is inconsistent with one 5 ms integration per tick")
        q, v = arrays["qpos"][:, env_index], arrays["qvel"][:, env_index]
        result = derive_kinematics(model, q, v, bindings)
        joint_ids = [_id(model, "joint", bindings.prefix + name) for name in joint_names]
        if any(model.jnt_type[j] != 3 for j in joint_ids):
            raise ValueError("Evaluated human joints must be scalar hinges")
        actuator_ids = []
        for joint in joint_ids:
            ids = np.flatnonzero((model.actuator_trntype == 0) & (model.actuator_trnid[:, 0] == joint))
            if len(ids) != 1 or not model.actuator_forcelimited[ids[0]]:
                raise ValueError("Each evaluated hinge needs one force-limited joint actuator")
            actuator_ids.append(ids[0])
        root_id = _id(model, "joint", bindings.prefix + bindings.root_body)
        if model.jnt_type[root_id] != 0:
            raise ValueError("Trace requires the named free, unactuated human root")
        root_dof = model.jnt_dofadr[root_id]
        contacts = []
        for k, row in enumerate(self.contacts):
            selected = row["worldid"] == env_index
            subset = {name: values[selected] for name, values in row.items() if name != "worldid"}
            contacts.append(classify_contacts(model, subset,
                result["support_anchor_position_w"][max(k-1, 0)], bindings))
        for name in contacts[0]:
            result[name] = np.asarray([row[name] for row in contacts])
        reset = arrays["reset_counts"][:, env_index]
        generalized_assistance = np.max(abs(arrays["qfrc_applied"][:, env_index]), axis=1) > 1e-9
        overflow = arrays["overflow"].reshape(len(self.rows), -1).any(axis=1)
        ref = np.asarray(reference_glove_position_w, dtype=float)
        if ref.shape != result["glove_position_w"].shape or not np.isfinite(ref).all():
            raise ValueError("A fresh reference sample at every recorded physics timestamp is required")
        result.update(time_s=request.launch_time_s + ticks*.005,
            raw_native_time_s=raw_time, force_interval_start_s=request.launch_time_s + np.maximum(ticks-1, 0)*.005,
            force_interval_end_s=request.launch_time_s + ticks*.005,
            reference_glove_position_w=ref, joint_position=q[:, model.jnt_qposadr[joint_ids]],
            joint_velocity=v[:, model.jnt_dofadr[joint_ids]], joint_limits=model.jnt_range[joint_ids].copy(),
            effort=arrays["actuator_force"][:, env_index, actuator_ids],
            effort_limits=np.minimum(-model.actuator_forcerange[actuator_ids, 0], model.actuator_forcerange[actuator_ids, 1]),
            external_wrench=arrays["xfrc_applied"][:, env_index],
            external_generalized_force=arrays["qfrc_applied"][:, env_index],
            root_actuator_force=arrays["qfrc_actuator"][:, env_index, root_dof:root_dof+6],
            external_support=arrays["external_support"][:, env_index] | generalized_assistance,
            reset_event=np.r_[reset[0] > 0, np.diff(reset) != 0] | arrays["pose_replacement"][:, env_index],
            numerical_failure=np.zeros(len(self.rows), bool), contact_overflow=overflow,
            physics_dt_s=.005, policy_dt_s=.020, decimation=4, integration_substeps=1,
            physics_step_count=len(self.rows)-1, outward_end_s=outward_end_s,
            inter_character_collision_enabled=inter_character_collision_enabled,
            evidence_kind=evidence_kind, admission_passed=admission_passed, domain_version=domain_version,
            request_sha256=request.fingerprint,
            guard_frame_definition="current chest body frame; rigid-frame transport velocity removed",
            support_patch_definition=f"fixed forefoot material point {bindings.anchor_local}, radius {bindings.patch_radius_m} m, load >= {bindings.minimum_normal_force_n} N",
            interval_evidence_semantics="forces and contacts at row k describe preceding integration; launch row is forward state")
        # derive_kinematics already rejects nonfinite state; all recorded arrays
        # remain available to diagnose any other native numerical failure.
        for values in arrays.values():
            if np.issubdtype(values.dtype, np.number):
                bad = ~np.isfinite(values).reshape(len(self.rows), -1).all(axis=1)
                result["numerical_failure"] |= bad
        return result
