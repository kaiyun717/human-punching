"""Entity and native affine PD controller preserving Task 09 physical parameters."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .bundle import RuntimeBundle, load_runtime_bundle


def build_human_terrain_cfg(bundle: RuntimeBundle) -> TerrainEntityCfg:
    """Move the source floor into the scene without changing contact material."""
    spec = mujoco.MjSpec.from_file(str(bundle.model_path))
    floor = spec.geom("floor")
    if floor is None or not np.allclose(floor.pos, 0.) or not np.allclose(floor.quat, [1., 0., 0., 0.]):
        raise ValueError("Expected the source human floor at the world ground plane")
    return TerrainEntityCfg(terrain_type="plane", collisions=(CollisionCfg(
        geom_names_expr=("terrain",), contype=int(floor.contype),
        conaffinity=int(floor.conaffinity), condim=int(floor.condim),
        priority=int(floor.priority), friction=tuple(floor.friction),
        solref=tuple(floor.solref), solimp=tuple(floor.solimp),
        margin=float(floor.margin), gap=float(floor.gap), solmix=float(floor.solmix),
    ),))


@dataclass(kw_only=True)
class NativeAffinePdCfg(ActuatorCfg):
    gains: tuple[tuple[str, float, float, float], ...]

    def build(self, entity, target_ids, target_names):
        return NativeAffinePd(self, entity, target_ids, target_names)


class NativeAffinePd(Actuator):
    """One native actuator per hinge; only physical output force is clipped.

    ``ctrl = kp*q_target + kd*v_reference + feedforward``. Native MuJoCo
    affine bias supplies ``-kp*q-kd*v`` and the implicit damping derivative.
    """

    def edit_spec(self, spec, target_names):
        lookup = {name: (kp, kd, cap) for name, kp, kd, cap in self.cfg.gains}
        ordered = []
        for target in target_names:
            kp, kd, cap = lookup[target.rsplit("/", 1)[-1]]
            joint = spec.joint(target)
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Human actuator may target scalar hinges only")
            actuator = spec.add_actuator()
            actuator.name = f"{target}_native_pd"
            actuator.target = target
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gear[:] = [1., 0., 0., 0., 0., 0.]
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[:] = 0.
            actuator.gainprm[0] = 1.
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[:] = 0.
            actuator.biasprm[1:3] = [-kp, -kd]
            actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange[:] = [-cap, cap]
            self._mjs_actuators.append(actuator)
            ordered.append((kp, kd))
        self._ordered_gains = np.asarray(ordered)

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.kp = torch.as_tensor(self._ordered_gains[:, 0], dtype=torch.float32, device=device)
        self.kd = torch.as_tensor(self._ordered_gains[:, 1], dtype=torch.float32, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        return self.kp * cmd.position_target + self.kd * cmd.velocity_target + cmd.effort_target


def build_human_entity_cfg(bundle: RuntimeBundle | str) -> EntityCfg:
    bundle = load_runtime_bundle(bundle)
    body = bundle.body

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(bundle.model_path))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        # The scene owns the floor. Preserve all human collision geoms/pairs.
        floor = spec.geom("floor")
        if floor is not None:
            spec.delete(floor)
        for key in list(spec.keys):
            spec.delete(key)
        return spec

    with np.load(bundle.path(bundle.clips[0]["master_path"]), allow_pickle=False) as data:
        initial = data["qpos"][0].copy()
    # mjlab infers the dtype of default joint-state tensors from these config
    # values. NumPy float64 scalars would create double defaults and promote
    # joint_pos_rel observations, while the actor and native state are float32.
    # Keep the archived master at full precision; convert at this runtime boundary.
    initial_values = initial.tolist()
    gains = tuple(zip(bundle.joint_names, body["kp"], body["kd"], body["effort"], strict=True))
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(initial_values[:3]), rot=tuple(initial_values[3:7]),
            joint_pos=dict(zip(bundle.joint_names, initial_values[7:], strict=True)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(NativeAffinePdCfg(target_names_expr=bundle.joint_names, gains=gains),),
            soft_joint_pos_limit_factor=1.0,
        ),
    )
