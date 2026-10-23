"""Read-only Task 08 bridge: fixed G1 plant, Fresh96 and real snapshot resets.

The accepted factories, actor, normalization and measurement implementation are
imported from robust_sav. This adapter owns only the extended scene and its reset
coordinates; it never edits or trains the accepted implementation. CUDA dynamics
are deliberately separate from the CPU compilation/FK checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
ROBUST = PROJECT.parent / "robust_sav"
BUNDLE = ROBUST / "constants/boxing/stand_policy/Fresh96FlatPolicy/v1_20261007"
BANK = ROBUST / "constants/boxing/snapshots/v1_20260921"
TARGET_POOL = PROJECT / "outputs/task10c/setup/target_pool.npz"
RESET_KEYS = ("sim/data/qpos", "sim/data/qvel", "action/_action",
              "action/_prev_action", "action/_prev_prev_action")


def _imports() -> None:
    """Make the explicitly permitted read-only source available, without install."""
    source = str((ROBUST / "src").resolve(strict=True))
    if source not in sys.path:
        sys.path.insert(0, source)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def execution_dependencies():
    _imports()
    from boxing.perception.dependencies import verify_execution_nominal
    # The accepted runtime verifier hardcodes robust_sav's interpreter location.
    # Verify its immutable inputs here; validate the local stack below instead.
    return verify_execution_nominal(BUNDLE, bank=BANK, verify_runtime=False)


def runtime_identity() -> dict[str, Any]:
    """Check installed revisions and actual installed G1 assets once at setup."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    versions = {name: importlib.metadata.version(name) for name in
                ("mujoco", "mujoco-warp", "mjlab", "warp-lang", "torch")}
    expected = {"mujoco": "3.11.0", "mujoco-warp": "3.11.0",
                "mjlab": "1.6.0", "warp-lang": "1.14.0"}
    if any(versions[key] != value for key, value in expected.items()):
        raise ValueError(f"Task 08 simulator revision mismatch: {versions}")
    selected = execution_dependencies()["model_action_fingerprint"]
    root = G1_XML.parent.parent
    for name, digest in selected["g1_asset_sha256"].items():
        if _digest(root / name) != digest:
            raise ValueError(f"Installed G1 asset differs from Fresh96: {name}")
    import mjlab
    import mujoco_warp
    return dict(versions=versions, mjlab_source=str(Path(mjlab.__file__).resolve()),
                mujoco_warp_source=str(Path(mujoco_warp.__file__).resolve()),
                accepted_mjlab_commit=selected["mjlab_commit"],
                policy=execution_dependencies()["execution_policy"],
                installed_g1_assets_verified=len(selected["g1_asset_sha256"]))


def _base_cfg(count: int, seed: int):
    _imports()
    from boxing.single_punch.plant import make_nominal_cfg
    from boxing.training.sensors import specialize_terrain_sensor
    from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
    cfg = make_nominal_cfg(count=count, seed=seed)
    cfg.scene.sensors = tuple(specialize_terrain_sensor(s)
        if isinstance(s, TerrainHeightSensorCfg) else s for s in cfg.scene.sensors)
    return cfg


@lru_cache(maxsize=1)
def compiled_baseline():
    """Compile the actual Task 08 pre-camera factory and compare Fresh96 data."""
    from mjlab.scene import Scene
    cfg = _base_cfg(1, 101001)
    model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(model)
    selected = execution_dependencies()["model_action_fingerprint"]
    for name, values in selected["compiled_host_arrays"].items():
        np.testing.assert_array_equal(getattr(model, name), values,
                                      err_msg=f"Task08 factory mismatch: {name}")
    for name, value in selected["simulation_options"].items():
        np.testing.assert_array_equal(getattr(model.opt, name), value,
                                      err_msg=f"Task08 option mismatch: {name}")
    for name, value in selected["model_dimensions"].items():
        if getattr(model, name) != value:
            raise ValueError(f"Task08 baseline changed {name}")
    return model


def make_task08_cfg(*, count: int = 1, seed: int = 101001,
                    extra_entities: Mapping[str, Any] | None = None,
                    scene_spec_fn=None, output_dir: Path | str | None = None):
    """Accepted physical cfg plus caller-owned noncontact visual extensions.

    ``scene_spec_fn`` may attach the accepted camera mount and modify H2 visuals.
    No obsolete Task08 floating gloves are added by this bridge. Original solver
    statistics are retained because adding mocap geometry changes auto statistics.
    """
    cfg = _base_cfg(count, seed)
    baseline = compiled_baseline()
    if extra_entities:
        if set(extra_entities) & set(cfg.scene.entities):
            raise ValueError("Additional entities must not replace accepted G1")
        cfg.scene.entities.update(extra_entities)

    def extend(spec):
        if scene_spec_fn is not None:
            scene_spec_fn(spec)
        for name in ("meaninertia", "meanmass", "meansize", "extent", "center"):
            setattr(spec.stat, name, np.asarray(getattr(baseline.stat, name)).copy())

    cfg.scene.spec_fn = extend
    destination = Path(output_dir or PROJECT / "outputs/task10d/phase1").resolve()
    if not destination.is_relative_to(PROJECT.resolve()):
        raise ValueError("Task10D output/cache must remain inside human-punching")
    cfg.sim.nan_guard.output_dir = str(destination / "nan_dumps")
    return cfg


def audit_extended_model(env) -> dict[str, Any]:
    """Ensure the extension leaves the existing G1 mechanics and action map intact."""
    _imports()
    from boxing.standing.schemas import model_action_fingerprint
    baseline = compiled_baseline()
    selected = execution_dependencies()["model_action_fingerprint"]
    actual = model_action_fingerprint(env)
    fields = ("model_dimensions", "timing", "simulation_options", "joint_order",
              "action_target_joint_order", "compiled_actuator_order",
              "compiled_actuator_joint_order", "action_to_compiled_actuator_ids",
              "default_joint_positions", "action_scale", "action_offset",
              "raw_action_clipping", "processed_target_clipping", "physical_target_formula",
              "previous_action", "encoder_bias_convention", "pd_kp_action_order",
              "pd_kd_action_order", "g1_asset_sha256")
    for name in fields:
        if actual[name] != selected[name]:
            raise ValueError(f"Task10D changed accepted G1 {name}")
    named_indices = {}
    for kind, count in (("body", baseline.nbody), ("geom", baseline.ngeom),
                        ("jnt", baseline.njnt), ("actuator", baseline.nu)):
        accessor = "joint" if kind == "jnt" else kind
        indices = []
        for index in range(count):
            name = getattr(baseline, accessor)(index).name
            if name:
                mapped = getattr(env.sim.mj_model, accessor)(name).id
            elif kind == "geom":
                # Native G1 visual meshes have no names. Match their ordinal
                # among their body's unnamed geoms, never look up geom("").
                before_body = int(baseline.geom_bodyid[index])
                body_name = baseline.body(before_body).name
                after_body = env.sim.mj_model.body(body_name).id
                before = [i for i in range(baseline.ngeom)
                          if baseline.geom_bodyid[i] == before_body
                          and not baseline.geom(i).name]
                after = [i for i in range(env.sim.mj_model.ngeom)
                         if env.sim.mj_model.geom_bodyid[i] == after_body
                         and not env.sim.mj_model.geom(i).name]
                if len(before) != len(after):
                    raise ValueError(f"Task10D changed unnamed geometry on {body_name}")
                mapped = after[before.index(index)]
                for field in ("geom_type", "geom_size", "geom_pos", "geom_quat"):
                    np.testing.assert_array_equal(getattr(env.sim.mj_model, field)[mapped],
                                                  getattr(baseline, field)[index])
            else:
                raise ValueError(f"Cannot identify unnamed accepted {kind} {index}")
            indices.append(mapped)
        named_indices[kind] = indices
    for name, before in selected["compiled_host_arrays"].items():
        old, new = np.asarray(before), np.asarray(actual["compiled_host_arrays"][name])
        kind = name.split("_")[0]
        indices = named_indices.get(kind, np.arange(len(old)))
        np.testing.assert_array_equal(new[indices], old,
                                      err_msg=f"Task10D changed accepted {name}")
    for name in ("meaninertia", "meanmass", "meansize", "extent", "center"):
        np.testing.assert_array_equal(getattr(env.sim.mj_model.stat, name),
                                      getattr(baseline.stat, name))
    # Mocap-only H2 must add no generalized coordinates, constraints or actuators.
    model = env.sim.mj_model
    for name in ("nq", "nv", "nu", "njnt", "neq", "ntendon", "npair", "nexclude"):
        if getattr(model, name) != getattr(baseline, name):
            raise ValueError(f"Kinematic extension added physical {name}")
    for geom in set(range(model.ngeom)) - set(named_indices["geom"]):
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            raise ValueError("H2 visual geometry must be collision-disabled")
    return dict(passed=True, fingerprint_fields=list(fields),
                compiled_array_fields=list(selected["compiled_host_arrays"]),
                simulation_options=actual["simulation_options"], timing=actual["timing"],
                g1_dimensions=actual["model_dimensions"],
                original_model_statistics_preserved=True)


def transform_snapshot(qpos: np.ndarray, qvel: np.ndarray,
                       world_from_collection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid planar registration preserving all momentum and joint coordinates.

    MuJoCo free-root translation velocity is world-frame, whereas its rotational
    qvel is local to the body. Only qvel[:3] rotates under left world rotation.
    """
    from scipy.spatial.transform import Rotation
    q, v = np.array(qpos, copy=True), np.array(qvel, copy=True)
    transform = np.asarray(world_from_collection, dtype=float)
    if q.shape != (36,) or v.shape != (35,) or transform.shape != (4, 4):
        raise ValueError("Expected G1 qpos36/qvel35 and transform[4,4]")
    np.testing.assert_allclose(transform[2], [0, 0, 1, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
    rotation = Rotation.from_matrix(transform[:3, :3])
    q[:3] = transform[:3, :3] @ q[:3] + transform[:3, 3]
    q[3:7] = (rotation * Rotation.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
    v[:3] = transform[:3, :3] @ v[:3]
    return q, v


def load_snapshot_rows(snapshot_ids: Sequence[int], *, split: str = "train",
                       target_pool: Path | str = TARGET_POOL, family: str = "jab") -> dict:
    """Read only requested real rows and their already-selected calibration."""
    _imports()
    from boxing.single_punch.artifacts import PortableBank
    from human_punching.task10c.targets import TargetPool, canonical_from_collection, FAMILIES
    ids = np.asarray(snapshot_ids, dtype=np.int64)
    bank = PortableBank(cache_size=1)
    rows = bank.rows(ids, split=split)
    arrays = {key: np.stack([state.arrays[key][row].copy() for state, row in rows])
              for key in RESET_KEYS}
    pool = TargetPool.load(target_pool)
    scene = pool.arrays["scene_from_canonical"][FAMILIES.index(family)]
    transforms = np.stack([scene @ canonical_from_collection(q)
                           for q in arrays["sim/data/qpos"]])
    return dict(snapshot_ids=ids, arrays=arrays, transforms=transforms,
                split=split, family=family, target_pool=str(Path(target_pool).resolve()),
                bank_manifest_sha256=bank.manifest_sha256)


class DynamicG1:
    """Small composition over an already constructed Task10D CUDA environment."""

    def __init__(self, env, *, noise_seed: int = 101002,
                 observation_noise_enabled: bool = True):
        _imports()
        import torch
        import warp as wp
        from boxing.single_punch.plant import NOISE_TERMS, SinglePunchPlant, validate_nominal_cfg
        from boxing.stance.chart import StanceChart
        from boxing.stance.mc_chart import TorchStanceChart
        from boxing.training.plant import uniform_action_vector
        if not torch.device(env.device).type == "cuda":
            raise ValueError("Task10D G1 dynamics require allocated CUDA")
        validate_nominal_cfg(env.cfg)
        self.env, self.device, self.num_envs = env, torch.device(env.device), env.num_envs
        self.model, self.data, self.robot = env.sim.mj_model, env.sim.data, env.scene["robot"]
        self.dtype = self.data.qpos.dtype
        self.runtime_identity = runtime_identity()
        self.model_audit = audit_extended_model(env)
        self._execution_stream = torch.cuda.ExternalStream(
            wp.get_stream(env.sim.wp_device).cuda_stream, device=self.device)
        self.command = env.command_manager.get_term("twist")
        self.action = env.action_manager.get_term("joint_pos")
        self.default = uniform_action_vector(self.action.offset, "offset")
        self.scale = uniform_action_vector(self.action.scale, "scale")
        selected = execution_dependencies()["model_action_fingerprint"]
        ids = np.asarray(selected["action_to_compiled_actuator_ids"])
        bounds = self.model.jnt_range[self.model.actuator_trnid[ids, 0]].copy()
        self.ctrl_lower = torch.as_tensor(bounds[:, 0], device=self.device, dtype=self.dtype)
        self.ctrl_upper = torch.as_tensor(bounds[:, 1], device=self.device, dtype=self.dtype)
        self._joint_range = torch.as_tensor(self.model.jnt_range[1:].copy(), device=self.device,
                                            dtype=self.dtype)
        self._zero_command = torch.zeros((env.num_envs, 3), device=self.device, dtype=self.dtype)
        self._halfwidth67 = torch.empty(67, device=self.device, dtype=self.dtype)
        for first, last, width in NOISE_TERMS:
            self._halfwidth67[first:last] = width
        self.noise_seed, self.observation_noise_enabled = noise_seed, observation_noise_enabled
        self.noise_generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        self.head_geom_id = self.model.geom("robot/head_collision").id
        self.head_body_id = int(self.model.geom_bodyid[self.head_geom_id])
        self.root_body_id = int(self.robot.indexing.root_body_id)
        self.torso_body_id = self.model.body("robot/torso_link").id
        if self.model.geom_size[self.head_geom_id, 0] != .06:
            raise ValueError("Task08 G1 head proxy must remain 0.06 m")
        self.nominal = execution_dependencies().nominal.to(self.device).eval()
        for parameter in self.nominal.parameters():
            parameter.requires_grad_(False)
        stance = Path(execution_dependencies()["original_dependencies"]["stance_bundle_path"])
        with np.load(stance / "stance.npz", allow_pickle=False) as z:
            self.chart = TorchStanceChart(StanceChart(z["qpos_star"], z["qvel_star"], z["T"]),
                                          device=self.device, dtype=torch.float64)
            self.P = torch.as_tensor(z["P"].copy(), device=self.device, dtype=torch.float64)
        self._reference_conjugate = torch.cat((self.chart.qpos_star[3:4], -self.chart.qpos_star[4:7]))
        SinglePunchPlant._initialize_self_contacts(self)
        with self.ordered_execution():
            self.command.freeze_zero()
            self.robot.data.encoder_bias.zero_()

    @contextmanager
    def ordered_execution(self):
        import torch
        caller = torch.cuda.current_stream(self.device)
        self._execution_stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self._execution_stream):
                yield
        finally:
            caller.wait_stream(self._execution_stream)

    def refresh(self):
        self.env.sim.forward()
        self.env.sim.sense()
        for sensor in self.env.scene.sensors.values():
            sensor._invalidate_cache()

    def reset(self, snapshot_rows: dict, *, transforms=None) -> dict:
        """Task08 derived reset: qpos/qvel/three raw histories, no settling step."""
        import torch
        from boxing.training.plant import tensor_view
        arrays = {key: np.array(value, copy=True) for key, value in snapshot_rows["arrays"].items()}
        transforms = snapshot_rows.get("transforms") if transforms is None else transforms
        if len(arrays["sim/data/qpos"]) != self.num_envs:
            raise ValueError("Snapshot count must equal environment count")
        if transforms is not None:
            for row, transform in enumerate(transforms):
                arrays["sim/data/qpos"][row], arrays["sim/data/qvel"][row] = transform_snapshot(
                    arrays["sim/data/qpos"][row], arrays["sim/data/qvel"][row], transform)
        with self.ordered_execution(), torch.no_grad():
            self.command.freeze_zero()
            self.env.reset()
            manager = self.env.action_manager
            destinations = dict(zip(RESET_KEYS, (tensor_view(self.data.qpos), tensor_view(self.data.qvel),
                manager.action, manager.prev_action, manager.prev_prev_action), strict=True))
            for name, destination in destinations.items():
                destination.copy_(torch.as_tensor(arrays[name], device=self.device, dtype=destination.dtype))
            for name in ("time", "qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "act", "history"):
                if hasattr(self.data, name):
                    tensor_view(getattr(self.data, name)).zero_()
            self.robot.data.encoder_bias.zero_()
            self.robot.data.joint_vel_target.zero_()
            self.robot.data.joint_effort_target.zero_()
            self.action._raw_actions.copy_(manager.action)
            self.action._processed_actions.copy_(self.default + self.scale * manager.action)
            manager.apply_action()
            self.env.scene.write_data_to_sim()
            self.refresh()
            self.noise_generator.manual_seed(self.noise_seed)
            self.refresh_measurement()
            errors = {key: float((destination - torch.as_tensor(arrays[key],
                      device=self.device, dtype=destination.dtype)).abs().max())
                      for key, destination in destinations.items()}
        self.snapshot_ids = np.asarray(snapshot_rows["snapshot_ids"]).copy()
        return dict(snapshot_ids=self.snapshot_ids.tolist(), restored_max_abs_error=errors,
                    transformed_qvel_norm=np.linalg.norm(arrays["sim/data/qvel"], axis=1).tolist(),
                    transforms=None if transforms is None else np.asarray(transforms).tolist(),
                    commands="permanently zero", physics_steps=0,
                    reset_claim="derived fixed nominal reset; preserves momentum and raw history")

    def refresh_measurement(self):
        from boxing.training.plant import GPUTrainingPlant, shared_mechanical
        self.clean99, self.clean111 = GPUTrainingPlant._clean_mechanical(self)
        self.actor99 = shared_mechanical(self.clean99, self._halfwidth67, self.noise_generator,
                                         enabled=self.observation_noise_enabled)

    def nominal_observation(self, actor948=None):
        if actor948 is not None:
            from boxing.training.residual_observations import nominal_observation
            return nominal_observation(actor948)
        from boxing.standing.nominal_evaluation import nominal_observation
        return nominal_observation(self.actor99, actor_features=96)

    def control_action(self):
        """Frozen exported normalizer runs exactly once; return limited radians."""
        import torch
        with self.ordered_execution(), torch.no_grad():
            raw = self.nominal(self.nominal_observation())
            target = self.default + self.scale * raw
            return torch.maximum(torch.minimum(target, self.ctrl_upper), self.ctrl_lower)

    def apply_physical_target(self, target):
        self.env.action_manager.process_action((target - self.default) / self.scale)

    def step_substep(self):
        """One actual 5 ms physics step; caller synchronizes H2 at this boundary."""
        self.env.action_manager.apply_action()
        self.env.scene.write_data_to_sim()
        self.env.sim.step()
        self.env.scene.update(dt=.005)
        self.refresh()
        # Accepted Task08 draws the shared measurement after every 5 ms step;
        # the fourth sample is the next actor's observation without a redraw.
        self.refresh_measurement()

    def geometry(self):
        import torch
        from boxing.single_punch.plant import SinglePunchPlant
        from boxing.training.plant import stance_energy_tensor
        q, v = self.data.qpos[:], self.data.qvel[:]
        head, speed = SinglePunchPlant.head_kinematics(self)
        margin = torch.minimum(q[:, 7:] - self._joint_range[:, 0],
                               self._joint_range[:, 1] - q[:, 7:])
        rotation = self.data.xmat[:, self.torso_body_id]
        finite = (torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1)
                  & torch.isfinite(self.data.actuator_force[:]).all(-1)
                  & torch.isfinite(rotation).all(dim=(-2, -1))
                  & torch.isfinite(self.data.qacc[:]).all(-1)
                  & torch.isfinite(self.data.qacc_warmstart[:]).all(-1))
        self_collision = SinglePunchPlant.self_collision(self)
        energy = stance_energy_tensor(q, v, self.chart, self.P, self._reference_conjugate)
        return dict(qpos=q, qvel=v, head_position=head, head_velocity=speed, height=q[:, 2],
                    torso_tilt=torch.acos(rotation[:, 2, 2].clamp(-1., 1.)),
                    joint_margin=margin, finite=finite, self_collision=self_collision,
                    stance_energy=energy, mechanically_safe=finite & (q[:, 2] >= .55)
                    & (margin >= 0).all(-1) & ~self_collision)
