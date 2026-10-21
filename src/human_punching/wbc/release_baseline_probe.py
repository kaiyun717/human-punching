"""Task 09 controller migration diagnostics in WBC; no learner is constructed.

The four frozen specialists are measurement instruments only. Their weights
never enter a WBC runner, optimizer, training checkpoint, or reference bundle.
References retain the original reset clock, family guard, initial hold and
offline joint feedforward. This is not the Task 10 snapshot/TTC acceptance task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .native_probe import NativeProbeRecorder, _configure_caches, _require_allocated_cuda, audit_probe_arrays
from .provenance import PROJECT_ROOT, runtime_record, sha256_file, writable_path

SCHEMA = "human_wbc_task09_release_probe_v2"
DIAGNOSTIC_SCOPE = "task09_reference_controller_migration"
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
ENVIRONMENTS = 16
TRIAL_IDS = tuple(range(ENVIRONMENTS))
SOLVERS = {
    "historical": dict(timestep=.005, integrator="implicitfast", solver="newton",
                       cone="elliptic", iterations=30, ls_iterations=15, tolerance=1.e-7),
    "accepted": dict(timestep=.005, integrator="implicitfast", solver="newton",
                     cone="pyramidal", iterations=10, ls_iterations=20, tolerance=1.e-8),
}
CONDITIONS = tuple((family, solver, feedforward) for family in FAMILIES
                   for solver, feedforward in (("historical", True), ("accepted", True), ("accepted", False)))


def _write_report(path, payload):
    """Encode strictly before reserving evidence, so type errors leave no partial JSON."""
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    destination = writable_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return destination


def _tensor_view(value):
    """Explicit zero-copy Torch access to native Warp arrays and mjlab proxies.

    ``np.array(wp.array)`` attempts unsupported item indexing. A direct Warp
    array must be converted before indexing; the mjlab proxy already owns that
    conversion. This helper does not replace or copy the simulation storage.
    """
    import torch
    value = getattr(value, "_tensor", value)
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    raise TypeError(f"Expected Torch tensor, mjlab TorchArray or Warp array, got {type(value).__name__}")


def _copy(value):
    """Synchronizing evidence copy, including raw native contact arrays."""
    value = getattr(value, "_tensor", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def _verified_release_file(root: Path, manifest: dict, relative: str) -> Path:
    path = (root / relative).resolve()
    expected = manifest.get("artifacts", {}).get(relative, {}).get("sha256")
    if not path.is_relative_to(root) or not expected or sha256_file(path) != expected:
        raise ValueError(f"Missing or changed Task 09 release artifact: {relative}")
    return path


def load_release_inputs(bundle, release_root: Path, *, num_envs=ENVIRONMENTS) -> dict[str, dict]:
    """Hash-check all diagnostic inputs before native allocation or deserialization."""
    from .runtime.bundle import load_runtime_bundle
    if num_envs not in (16, 64):
        raise ValueError("Release diagnostics use 16 worlds; parity uses the complete 64-state bank")
    bundle = load_runtime_bundle(bundle)
    root = Path(release_root).resolve()
    if (bundle.manifest.get("diagnostic_scope") != DIAGNOSTIC_SCOPE
            or bundle.manifest.get("production_qualified") is not False
            or bundle.manifest.get("feedforward_mode") != "task09_source_inverse_statics"
            or bundle.manifest.get("release_manifest_sha256") != sha256_file(root / "manifest.json")):
        raise ValueError("An explicitly unqualified Task 09 migration bundle is required")
    manifest = json.loads((root / "manifest.json").read_text())
    if len(bundle.clips) != 4 or {clip["family"] for clip in bundle.clips} != set(FAMILIES):
        raise ValueError("Exactly one preserved reference per family is required")
    result = {}
    for clip in bundle.clips:
        family = clip["family"]
        if (clip.get("guard_source") != "task09_family_guard"
                or clip.get("g1_snapshot_present") is not False
                or clip.get("launch_features") != [] or clip.get("requested_ttc_s") is not None):
            raise ValueError("Diagnostic clips must retain family guards without invented G1/TTC context")
        paths = {key: _verified_release_file(root, manifest, f"{family}/{relative}")
                 for key, relative in {
                     "checkpoint": "checkpoint.pt", "schema": "controller_schema.json",
                     "arguments": "arguments.json", "bank": "evaluation/evaluation_bank.npz",
                     "bank_manifest": "evaluation/evaluation_bank_manifest.json",
                     "reference": "reference.npz", "model": "model/human.xml",
                     "source": "jab_learning_source.py",
                 }.items()}
        schema = json.loads(paths["schema"].read_text())
        arguments = json.loads(paths["arguments"].read_text())
        expected = dict(controller_mode="implicit_pd", action_scale_rad=.25,
                        physics_dt=.005, actor_dt=.020, speed=1., guard_hold_s=1.,
                        balance_feedback=False, refresh_kinematics=True,
                        feedforward="offline inverse statics joint torques",
                        observation_clip=20, phase_clock="elapsed from reset/reference sample")
        if any(schema.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen specialist controller schema is unsupported")
        if (schema.get("joint_names") != list(bundle.joint_names)
                or schema.get("qpos_addresses") != list(range(7, 55))
                or schema.get("dof_addresses") != list(range(6, 54))
                or schema.get("model_sha256") != sha256_file(paths["model"])
                or schema.get("reference_sha256") != sha256_file(paths["reference"])
                or not np.array_equal(schema.get("kp"), bundle.body["kp"])
                or not np.array_equal(schema.get("kd"), bundle.body["kd"])
                or not np.array_equal(np.asarray(schema.get("torque_limits"))[:, 1], bundle.body["effort"])):
            raise ValueError("Frozen specialist body/controller identity differs from WBC")
        if arguments.get("no_feedforward") is not False or arguments.get("balance_feedback") is not False:
            raise ValueError("Unsupported original feedforward or balance feedback configuration")
        with np.load(paths["bank"], allow_pickle=False) as data:
            qpos, qvel, ids = data["qpos"].copy(), data["qvel"].copy(), data["trial_ids"].copy()
            if (qpos.shape != (64, 55) or qvel.shape != (64, 54)
                    or not np.array_equal(ids, np.arange(64))
                    or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
                    or str(data["model_sha256"].item()) != schema["model_sha256"]
                    or str(data["reference_sha256"].item()) != schema["reference_sha256"]):
                raise ValueError("Malformed or mismatched preserved guard bank")
        legacy_path = bundle.path(clip["legacy_runtime_path"])
        if sha256_file(legacy_path) != clip["legacy_runtime_sha256"]:
            raise ValueError("Source-grid runtime hash mismatch")
        with np.load(legacy_path, allow_pickle=False) as source:
            legacy = {key: source[key].copy() for key in source.files}
        n = len(legacy["t"])
        for key, shape in (("t", (n,)), ("qpos", (n, 55)), ("qvel", (n, 54)),
                           ("torque", (n, 48)), ("glove_pos", (n, 2, 3))):
            if legacy[key].shape != shape or not np.isfinite(legacy[key]).all():
                raise ValueError("Malformed frozen source-grid runtime")
        if (n < 2 or legacy["t"][0] != 0 or np.any(np.diff(legacy["t"]) <= 0)
                or not np.isclose(clip["legacy_episode_duration_s"], clip["source_duration_s"]+1., atol=1e-10)):
            raise ValueError("Invalid original reset-clock duration")
        result[family] = dict(clip=clip, paths=paths, schema=schema, legacy=legacy,
                              initial_qpos=qpos[:num_envs], initial_qvel=qvel[:num_envs],
                              input_sha256={key: sha256_file(path) for key, path in paths.items()})
    return result


def interpolate_legacy(times, values, query):
    """The source Torch interpolation, including its float32 sample grid."""
    import torch
    clipped = query.clamp(times[0], times[-1])
    upper = torch.searchsorted(times, clipped.contiguous(), right=True).clamp(1, len(times)-1)
    lower = upper-1
    alpha = (clipped-times[lower])/(times[upper]-times[lower])
    result = values[lower] + alpha.reshape(*alpha.shape, *([1]*(values.ndim-1))) * (values[upper]-values[lower])
    return result


def legacy_observations(qpos, qvel, gloves, previous_raw_action, clock, reference, duration):
    """369 source observations; no normalizer and no fabricated G1 features."""
    import torch
    def ref(name, offset=0.):
        values = interpolate_legacy(reference["t"], reference[name], clock+offset)
        if name == "qpos":
            values = values.clone()
            values[:, 3:7] /= values[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1.e-8)
        return values
    desired, desired_v = ref("qpos"), ref("qvel")
    phase = clock/duration
    parts = [qpos[:, 2:7], qvel[:, :6]*.2, qpos[:, 7:], qvel[:, 6:]*.1,
             (desired[:, :3]-qpos[:, :3])*5, desired[:, 3:7], desired[:, 7:]-qpos[:, 7:],
             (desired_v[:, 6:]-qvel[:, 6:])*.1,
             torch.stack((phase, torch.sin(phase*2*np.pi), torch.cos(phase*2*np.pi)), -1),
             previous_raw_action]
    for offset in (.1, .25):
        future = ref("qpos", offset)
        parts.extend(((future[:, :3]-qpos[:, :3])*5, future[:, 7:]-qpos[:, 7:]))
    parts.append((ref("glove_pos")-gloves).flatten(1)*5)
    values = torch.cat(parts, -1)
    # Do not hide nonfinite physical inputs through the historical nan_to_num.
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite source-compatible specialist observation")
    if values.shape[-1] != 369:
        raise ValueError("Frozen specialist observation dimension differs")
    return values.clamp(-20, 20)


def load_frozen_actor(inputs, device):
    """Load only actor tensors, never optimizer/critic state or a training runner."""
    import torch
    from torch import nn
    checkpoint = torch.load(inputs["paths"]["checkpoint"], map_location="cpu", weights_only=True)
    if (checkpoint.get("observations") != 369 or checkpoint.get("actions") != 48
            or checkpoint.get("controller_schema") != inputs["schema"]
            or checkpoint.get("model_sha256") != inputs["schema"]["model_sha256"]
            or checkpoint.get("reference_sha256") != inputs["schema"]["reference_sha256"]):
        raise ValueError("Frozen specialist checkpoint identity mismatch")
    actor = nn.Sequential(nn.Linear(369, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 48))
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in checkpoint["policy"].items()
                           if key.startswith("actor.")}, strict=True)
    return actor.to(device).eval().requires_grad_(False)


def _actor_sha256(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(_copy(value).tobytes())
    return digest.hexdigest()


def _probe_types():
    import torch
    from .native_cycle_probe import _probe_command_types
    from .runtime.action import PunchJointPositionAction, PunchJointPositionActionCfg
    base_command, base_cfg = _probe_command_types()

    class ReleaseCommand(base_command):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            matches = [i for i, clip in enumerate(self.clips) if clip["family"] == cfg.family]
            if len(matches) != 1:
                raise ValueError("Exactly one selected historical family reference is required")
            self._fixed_clip_ids.fill_(matches[0])
            with np.load(cfg.initial_bank, allow_pickle=False) as bank:
                self._initial_qpos = torch.tensor(bank["qpos"][:env.num_envs], dtype=torch.float32, device=self.device)
                self._initial_qvel = torch.tensor(bank["qvel"][:env.num_envs], dtype=torch.float32, device=self.device)

        def _write_reference_state_to_sim(self, env_ids, *args, **kwargs):
            del args, kwargs
            if not getattr(self, "_reset_in_progress", False):
                raise RuntimeError("Preserved bank state writes are restricted to episode initialization")
            self.write_ledger.event("write", env_ids)
            qpos = self._initial_qpos[env_ids].clone()
            qpos[:, :3] += self._env.scene.env_origins[env_ids]
            # The bank contains native MuJoCo qvel, including local root angular
            # tangent coordinates. Copying named full-human arrays preserves it.
            self._env.sim.data.qpos[env_ids] = qpos
            self._env.sim.data.qvel[env_ids] = self._initial_qvel[env_ids]

        def sample_master(self, name, offset_steps=0):
            local = torch.minimum((self.physics_steps+offset_steps).clamp(min=0),
                                  self.master_lengths[self.trajectory_ids]-1)
            value = self.master[name][self.master_starts[self.trajectory_ids]+local]
            # Preserve recorded endpoint derivatives and FF. No strict-bridge
            # settling override or clip-boundary reset is involved here.
            if name == "joint_feedforward" and not self.cfg.feedforward_enabled:
                return torch.zeros_like(value)
            return value

    @dataclass(kw_only=True)
    class ReleaseCommandCfg(base_cfg):
        family: str
        initial_bank: str
        feedforward_enabled: bool

        def build(self, env):
            return ReleaseCommand(self, env)

    class ReleaseAction(PunchJointPositionAction):
        def apply_actions(self):
            self._env._release_baseline_recorder.before_interval(self._env)
            super().apply_actions()

    @dataclass(kw_only=True)
    class ReleaseActionCfg(PunchJointPositionActionCfg):
        def build(self, env):
            return ReleaseAction(self, env)

    return ReleaseCommand, ReleaseCommandCfg, ReleaseActionCfg


class ReleaseRecorder(NativeProbeRecorder):
    def __init__(self, num_envs=ENVIRONMENTS):
        super().__init__(num_envs)
        self.pre, self.policies, self.contacts = [], [], []

    def before_interval(self, env):
        command = env.command_manager.get_term("motion")
        self.pre.append({name: _copy(value) for name, value in dict(
            qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, time=env.sim.data.time,
            qref=command.sample_master("qpos"), vref=command.sample_master("qvel"),
            feedforward=command.sample_master("joint_feedforward"),
            reference_tick=command.physics_steps, actor_action=env._release_baseline_raw_action).items()})

    def capture(self, env):
        import mujoco_warp as mjwarp
        import warp as wp
        super().capture(env)
        native = env.sim.wp_data
        count = int(_copy(native.nacon)[0])
        if not 0 <= count <= native.contact.geom.shape[0]:
            raise RuntimeError("Native contact storage overflow")
        contact = {name: _copy(getattr(native.contact, name))[:count]
                   for name in ("geom", "pos", "dist", "worldid")}
        if count:
            ids = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=native.qpos.device)
            forces = wp.zeros(count, dtype=wp.spatial_vector, device=native.qpos.device)
            mjwarp.contact_force(env.sim.wp_model, native, ids, False, forces)
            contact["force"] = forces.numpy()
        else:
            contact["force"] = np.zeros((0, 6))
        self.contacts.append(contact)

    def all_arrays(self):
        result = self.arrays()
        for prefix, rows in (("pre_", self.pre), ("policy_", self.policies)):
            if rows:
                result.update({prefix+name: np.stack([row[name] for row in rows]) for name in rows[0]})
        if self.contacts:
            result["contact_offsets"] = np.r_[0, np.cumsum([len(row["geom"]) for row in self.contacts])]
            result.update({"contact_"+name: np.concatenate([row[name] for row in self.contacts])
                           for name in self.contacts[0]})
        return result


def build_release_probe_cfg(bundle, recorder, inputs, *, solver, feedforward):
    """Only the new diagnostic module selects the historical solver profile."""
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .runtime.task import build_human_env_cfg
    if solver not in SOLVERS or type(feedforward) is not bool:
        raise ValueError("Unknown diagnostic solver/feedforward condition")
    _, command_type, action_type = _probe_types()
    cfg = build_human_env_cfg(bundle, num_envs=recorder.num_envs, evaluation=True)
    old = cfg.commands["motion"]
    cfg.commands["motion"] = command_type(**{field.name: getattr(old, field.name) for field in fields(old)},
        family=inputs["clip"]["family"], initial_bank=str(inputs["paths"]["bank"]), feedforward_enabled=feedforward)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = action_type(**{field.name: getattr(old, field.name) for field in fields(old)})
    cfg.events, cfg.terminations, cfg.rewards = {}, {}, {}
    cfg.auto_reset = False
    cfg.episode_length_s = 20.
    cfg.metrics = {"release_probe": MetricsTermCfg(func=recorder.metric_function(), per_substep=True)}
    for key, value in SOLVERS[solver].items():
        setattr(cfg.sim.mujoco, key, value)
    # Hold capacities constant across profiles: these prevent overflows rather
    # than silently changing forces. They exceed the historical 96/256 slots.
    cfg.sim.nconmax, cfg.sim.njmax = 128, 512
    cfg.seed = 10010
    return cfg


def verify_release_model(model, bundle, *, solver):
    """Check the compiled body, contacts and native actuator—not config labels."""
    import mujoco
    from .trainedactor_evaluation import verify_compiled_controller
    verify_compiled_controller(model, bundle)
    source = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    checks = {"native_affine_controller": True, "no_equality_support": model.neq == source.neq == 0}
    errors = {}
    def compare(key, pairs, attributes):
        maximum = 0.
        passed = True
        for original, actual in pairs:
            for attribute in attributes:
                a, b = np.asarray(getattr(original, attribute)), np.asarray(getattr(actual, attribute))
                if a.shape != b.shape:
                    passed = False
                else:
                    error = float(np.max(np.abs(a.astype(float)-b.astype(float)), initial=0.))
                    maximum = max(maximum, error)
                    passed &= error <= 1.e-10
        checks[key], errors[key] = bool(passed), maximum
    compare("body_mass_inertia_and_local_frames",
        [(source.body(i), model.body("robot/"+source.body(i).name)) for i in range(1, source.nbody)],
        ("mass", "inertia", "pos", "quat", "ipos", "iquat"))
    source_bodies = np.arange(1, source.nbody)
    actual_bodies = np.asarray([model.body("robot/"+source.body(int(i)).name).id for i in source_bodies])
    checks["body_gravity_compensation"] = bool(np.array_equal(source.body_gravcomp[source_bodies], model.body_gravcomp[actual_bodies]))
    compare("joint_axes_limits_and_passive_parameters",
        [(source.joint(i), model.joint("robot/"+source.joint(i).name)) for i in range(source.njnt)],
        ("type", "pos", "axis", "range", "limited", "stiffness", "solref", "solimp", "margin"))
    for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
        checks[field] = bool(np.allclose(getattr(source, field), getattr(model, field), atol=1.e-10, rtol=0))
    material = ("friction", "solref", "solimp", "contype", "conaffinity", "condim", "priority", "margin", "gap", "solmix")
    compare("human_collision_geometry_and_materials",
        [(source.geom(i), model.geom("robot/"+source.geom(i).name)) for i in range(source.ngeom) if source.geom(i).name != "floor"],
        ("type", "size", "pos", "quat", *material))
    compare("floor_plane_and_contact_materials", [(source.geom("floor"), model.geom("terrain"))],
            ("type", "pos", "quat", *material))
    def pairs(m, prefix):
        result = {}
        for i in range(m.npair):
            names = tuple(sorted(m.geom(int(g)).name.removeprefix(prefix) for g in (m.pair_geom1[i], m.pair_geom2[i])))
            result[names] = np.concatenate([np.asarray(getattr(m, field)[i]).reshape(-1) for field in
                ("pair_dim", "pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap")])
        return result
    original_pairs, actual_pairs = pairs(source, ""), pairs(model, "robot/")
    checks["selected_self_collision_pairs"] = (original_pairs.keys() == actual_pairs.keys() and all(
        np.allclose(value, actual_pairs[key], atol=1.e-10, rtol=0) for key, value in original_pairs.items()))
    enum = dict(integrator=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
                solver=int(mujoco.mjtSolver.mjSOL_NEWTON),
                cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC if solver == "historical" else mujoco.mjtCone.mjCONE_PYRAMIDAL))
    actual_options = {}
    for key, expected in SOLVERS[solver].items():
        actual = getattr(model.opt, key)
        actual_options[key] = int(actual) if key in enum else float(actual)
        checks["actual_solver_"+key] = actual == enum.get(key, expected)
    for key in ("gravity", "density", "viscosity", "impratio", "enableflags", "disableflags"):
        value = np.asarray(getattr(model.opt, key))
        actual_options[key] = value.tolist()
        checks["source_global_"+key] = bool(np.array_equal(value, getattr(source.opt, key)))
    result = dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                  maximum_compiled_differences=errors, actual_options=actual_options,
                  contact_capacities_note="128/512 WBC allocation versus historical 96/256; capacity, not a force-law change")
    if not all(checks.values()):
        raise ValueError("Compiled release migration differs: "+", ".join(key for key, ok in checks.items() if not ok))
    return result


def audit_release_arrays(arrays, *, bundle, inputs, policy_steps, feedforward, origins=None):
    """Reconstruct learned-action targets and drive at every actual integration."""
    audit = audit_probe_arrays(arrays, policy_steps=policy_steps, physics_dt_s=.005,
                               effort_limits=np.asarray(bundle.body["effort"]))
    # This inherited check specifically describes the old zero-action fixture.
    # Keep its physics checks, then positively verify our recorded actor action.
    checks = dict(audit["checks"])
    checks.pop("zero_learned_residual", None)
    physics_audit = {**audit, "checks": dict(checks),
                     "status": "passed" if all(checks.values()) else "failed",
                     "zero_residual_fixture_check": "not_applicable_to_frozen_actor"}
    n = policy_steps*4
    num_envs = len(inputs["initial_qpos"])
    required = {"pre_qpos", "pre_qvel", "pre_qref", "pre_vref", "pre_feedforward", "pre_reference_tick", "pre_actor_action",
                "policy_observation", "policy_action", "policy_previous_raw_action", "policy_reference_tick"}
    if required-set(arrays):
        return dict(status="failed", checks={**checks, "learned_action_evidence": False},
                    missing_fields=sorted(required-set(arrays)))
    actions = np.asarray(arrays["policy_action"])
    pre_action = np.asarray(arrays["pre_actor_action"])
    qref, vref, ff = (np.asarray(arrays["pre_"+name]) for name in ("qref", "vref", "feedforward"))
    shapes = (actions.shape == (policy_steps, num_envs, 48)
              and pre_action.shape == (n, num_envs, 48)
              and qref.shape == (n, num_envs, 55) and vref.shape == (n, num_envs, 54)
              and ff.shape == (n, num_envs, 48)
              and np.asarray(arrays["policy_observation"]).shape == (policy_steps, num_envs, 369)
              and np.asarray(arrays["pre_qpos"]).shape == (n, num_envs, 55)
              and np.asarray(arrays["pre_qvel"]).shape == (n, num_envs, 54))
    checks["learned_action_evidence"] = shapes
    if not shapes:
        return dict(status="failed", checks=checks)
    expected_pre = np.repeat(actions, 4, axis=0)
    checks["actor_held_for_four_physics_steps"] = bool(np.array_equal(pre_action, expected_pre))
    checks["recorded_actor_reaches_action_manager"] = bool(np.array_equal(arrays["raw_action"][1:], expected_pre))
    previous = np.asarray(arrays["policy_previous_raw_action"])
    checks["previous_raw_action_semantics"] = bool(previous.shape == actions.shape and np.all(previous[0] == 0)
                                                   and np.array_equal(previous[1:], actions[:-1]))
    checks["actor_tick_schedule"] = bool(np.array_equal(arrays["policy_reference_tick"],
        np.broadcast_to(np.arange(policy_steps)[:, None]*4, (policy_steps, num_envs))))
    checks["preintegration_reference_clock"] = bool(np.array_equal(arrays["pre_reference_tick"],
        np.broadcast_to(np.arange(n)[:, None], (n, num_envs))))
    low, high = np.asarray(bundle.body["joint_limits_rad"]).T
    expected_target = np.clip(qref[..., 7:]+.25*np.clip(pre_action, -1., 1.), low, high)
    expected_drive = np.asarray(bundle.body["kp"])*expected_target + np.asarray(bundle.body["kd"])*vref[..., 6:] + ff
    caps = np.asarray(bundle.body["effort"])
    expected_force = np.clip(arrays["ctrl"][1:]-np.asarray(bundle.body["kp"])*arrays["pre_qpos"][..., 7:]
                             -np.asarray(bundle.body["kd"])*arrays["pre_qvel"][..., 6:], -caps, caps)
    errors = {
        "joint_target_error_rad": float(abs(arrays["joint_position_target"][1:]-expected_target).max()),
        "joint_velocity_target_error_rad_s": float(abs(arrays["joint_velocity_target"][1:]-vref[..., 6:]).max()),
        "joint_feedforward_target_error_nm": float(abs(arrays["joint_effort_target"][1:]-ff).max()),
        "native_drive_error_nm": float(abs(arrays["ctrl"][1:]-expected_drive).max()),
        "native_affine_force_error_nm": float(abs(arrays["actuator_force"][1:]-expected_force).max()),
    }
    checks.update(joint_target_clipping=errors["joint_target_error_rad"] <= 2.e-6,
                  reference_velocity_target=errors["joint_velocity_target_error_rad_s"] <= 2.e-6,
                  feedforward_reaches_actuator=errors["joint_feedforward_target_error_nm"] <= 5.e-6,
                  native_affine_drive=errors["native_drive_error_nm"] <= 2.e-4,
                  native_affine_force_law=errors["native_affine_force_error_nm"] <= 5.e-4)
    with np.load(bundle.path(inputs["clip"]["master_path"]), allow_pickle=False) as master:
        ids = np.minimum(np.arange(n), len(master["time_s"])-1)
        expected_ff = np.asarray(master["joint_feedforward"], np.float32)[ids]
        if not feedforward:
            expected_ff = np.zeros_like(expected_ff)
        checks["immutable_master_feedforward_including_initial_and_terminal_hold"] = bool(
            np.allclose(ff, expected_ff[:, None], atol=5.e-6, rtol=0))
        checks["immutable_master_joint_reference"] = bool(np.allclose(
            qref[..., 7:], np.asarray(master["qpos"], np.float32)[ids, None, 7:], atol=2.e-6, rtol=0))
        checks["immutable_master_velocity_reference"] = bool(np.allclose(
            vref, np.asarray(master["qvel"], np.float32)[ids, None], atol=2.e-6, rtol=0))
    checks["fixed_initial_bank_joint_state"] = bool(np.array_equal(arrays["qpos"][0, :, 7:], inputs["initial_qpos"][:, 7:].astype(np.float32)))
    checks["fixed_initial_bank_velocity"] = bool(np.array_equal(arrays["qvel"][0], inputs["initial_qvel"].astype(np.float32)))
    origins = np.asarray(origins) if origins is not None else np.zeros((num_envs, 3), np.float32)
    expected_initial = inputs["initial_qpos"].astype(np.float32).copy()
    if origins.shape == (num_envs, 3):
        expected_initial[:, :3] += origins.astype(np.float32)
    checks["fixed_initial_bank_full_pose_and_origin"] = bool(origins.shape == (num_envs, 3)
        and np.array_equal(arrays["qpos"][0], expected_initial))
    finite = all(np.isfinite(value).all() for value in errors.values())
    checks["finite_controller_audit"] = finite
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                controller_errors=errors, physics_audit=physics_audit,
                zero_action_fixture_check_applicable=False)


def diagnostic_episodes(model, arrays, inputs, origins):
    """Fresh FK plus measured interval contacts; no Task 10 admission claim."""
    from .cycle_trace import TraceBindings, classify_contacts, derive_kinematics
    bindings = TraceBindings()
    n = len(arrays["time"])
    times = np.arange(n)*.005
    clip = inputs["clip"]
    active = 0 if clip["family"] in ("jab", "left_hook") else 1
    episodes, reports = [], []
    offsets = arrays["contact_offsets"]
    for world in range(ENVIRONMENTS):
        q, v = arrays["qpos"][:, world], arrays["qvel"][:, world]
        kin = derive_kinematics(model, q, v)
        nominal_q = np.asarray(inputs["legacy"]["qpos"][:1], float).copy()
        nominal_q[:, :3] += origins[world]
        nominal = derive_kinematics(model, nominal_q, np.zeros((1, 54)))
        contact_rows = []
        for tick in range(n):
            span = slice(offsets[tick], offsets[tick+1])
            selected = arrays["contact_worldid"][span] == world
            contact_rows.append(classify_contacts(model,
                {key: arrays["contact_"+key][span][selected] for key in ("geom", "pos", "dist", "force")},
                kin["support_anchor_position_w"][max(tick-1, 0)], bindings))
        contacts = {key: np.asarray([row[key] for row in contact_rows]) for key in contact_rows[0]}
        target = np.asarray(clip["target_w"])+origins[world]
        distance = np.linalg.norm(kin["glove_position_w"]-target, axis=-1)
        outward = (times >= clip["launch_time_s"]) & (times <= clip["outward_end_s"])
        hits = np.flatnonzero(outward & (distance[:, active] <= .14))
        tilt = np.arccos(np.clip(1-2*(q[:, 4]**2+q[:, 5]**2), -1, 1))
        root_error = np.linalg.norm(kin["root_position_w"]-nominal["root_position_w"][0], axis=-1)
        root_angle = 2*np.arccos(np.clip(abs((kin["root_quat_wxyz"]*nominal["root_quat_wxyz"][0]).sum(-1)), 0, 1))
        joint_error = abs(q[:, 7:]-nominal_q[0, 7:]).max(-1)
        glove_error = np.linalg.norm(kin["glove_position_guard_frame"]-nominal["glove_position_guard_frame"][0], axis=-1).max(-1)
        glove_angle = 2*np.arccos(np.clip(abs((kin["glove_quat_guard_frame"]*nominal["glove_quat_guard_frame"][0]).sum(-1)), 0, 1)).max(-1)
        drift = np.linalg.norm(kin["support_anchor_position_w"][..., :2]-nominal["support_anchor_position_w"][0, ..., :2], axis=-1)
        quiet_guard = ((root_error <= .03) & (root_angle <= .10) & (joint_error <= .12)
            & (glove_error <= .03) & (glove_angle <= np.deg2rad(15.))
            & (np.linalg.norm(kin["root_linear_velocity_w"], axis=-1) <= .05)
            & (np.linalg.norm(kin["root_angular_velocity_w"], axis=-1) <= .12)
            & (abs(v[:, 6:]).max(-1) <= .50)
            & (np.linalg.norm(kin["glove_velocity_guard_frame"], axis=-1).max(-1) <= .15)
            & contacts["support_contact"].all(-1) & (distance >= .16).all(-1)
            & (drift <= .03).all(-1))
        trailing = float(0. if not quiet_guard[-1] else (n-1-(np.flatnonzero(~quiet_guard)[-1]+1 if (~quiet_guard).any() else 0))*.005)
        first = None if not len(hits) else float(times[hits[0]])
        limits = model.jnt_range[1:]
        joint_violation = np.maximum(limits[:, 0]-q[:, 7:], q[:, 7:]-limits[:, 1]).max(initial=0.)
        fall = bool(((q[:, 2] < .75*nominal_q[0, 2]) | (tilt > .60)).any())
        loaded = contacts["support_patch_contact"]
        report = dict(trial_id=TRIAL_IDS[world], analytic_contact=first is not None,
            first_contact_reset_clock_s=first,
            first_contact_from_inserted_strike_start_s=None if first is None else first-clip["launch_time_s"],
            commanded_ttc=False, fell=fall, final_quiet_guard_dwell_s=trailing,
            wrong_hand_analytic_contact=bool((distance[:, 1-active][outward] <= .14).any()),
            any_cycle_active_contact=bool((distance[:, active] <= .14).any()),
            final_one_second_guard=bool(trailing >= 1.-1e-9),
            minimum_active_glove_target_distance_m=float(distance[:, active][outward].min()),
            maximum_root_tilt_rad=float(tilt.max()), maximum_joint_limit_excursion_rad=float(joint_violation),
            maximum_loaded_forefoot_drift_m=float(drift[loaded].max(initial=0.)),
            forbidden_collision=bool(contacts["forbidden_collision"].any()),
            final_joint_guard_error_rad=float(joint_error[-1]), final_root_guard_error_m=float(root_error[-1]),
            final_glove_guard_error_m=float(glove_error[-1]),
            complete_cycle_acceptance=False, admission_passed=False)
        episode = dict(time_s=times, raw_native_time_s=arrays["time"][:, world], qpos=q, qvel=v,
            effort=arrays["actuator_force"][:, world], ctrl=arrays["ctrl"][:, world],
            target_w=target, glove_target_distance=distance, quiet_guard=quiet_guard,
            nominal_guard_qpos=nominal_q[0], **kin, **contacts)
        episodes.append(episode)
        reports.append(report)
    return episodes, reports


def execution_counts(recorder, *, completed_policy_steps, initial_time=None, final_time=None,
                     final_reference_tick=None, final_sim_step_counter=None):
    """Measure completed integration from native time, including exceptions.

    mjlab increments its simulation counter and our reference tick *before*
    calling the integrator. Neither is evidence of a completed step on its own.
    Keep those counters, pre-interval records and completed callbacks separate.
    """
    elapsed = None
    ticks = None
    if initial_time is not None and final_time is not None:
        elapsed = np.asarray(final_time, float)-np.asarray(initial_time, float)
        if elapsed.shape != (recorder.num_envs,) or not np.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("Invalid actual native elapsed clock")
        ticks = np.rint(elapsed/.005).astype(np.int64)
        # float32 accumulates one rounding error per native addition, not one
        # error at the final timestamp. Keep the bound below half a physics tick.
        tolerance = min(.0024, max(5.e-8, float(ticks.max())*np.finfo(np.float32).eps*max(.005, float(elapsed.max()))))
        if not np.allclose(elapsed, ticks*.005, atol=tolerance, rtol=0):
            raise ValueError("Actual native time does not match 5 ms integration ticks")
    count = None if ticks is None or not np.all(ticks == ticks[0]) else int(ticks[0])
    return dict(actual_actor_updates=len(recorder.policies),
        actual_completed_policy_steps=int(completed_policy_steps),
        actual_physics_steps=count, actual_physics_steps_per_world=None if ticks is None else ticks.tolist(),
        actual_integrated_duration_s=None if count is None else count*.005,
        actual_native_elapsed_s=None if elapsed is None else elapsed.tolist(),
        actual_preintegration_records=len(recorder.pre),
        actual_postintegration_records=max(0, len(recorder.rows)-1),
        actual_contact_sample_records=len(recorder.contacts),
        actual_reference_ticks=None if final_reference_tick is None else np.asarray(final_reference_tick).tolist(),
        actual_sim_step_counter=None if final_sim_step_counter is None else int(final_sim_step_counter),
        execution_count_semantics="Native elapsed time establishes completed physics; reference/simulation counters can advance before a failed integration")


def zero_step_contract(env, recorder, inputs, observations, origins):
    """Inspect the real initialized native state before any policy or integration."""
    expected = inputs["initial_qpos"].astype(np.float32).copy()
    expected[:, :3] += origins
    command = env.command_manager.get_term("motion")
    checks = dict(
        initial_state_preserved=np.array_equal(_copy(env.sim.data.qpos), expected),
        initial_velocity_preserved=np.array_equal(_copy(env.sim.data.qvel), inputs["initial_qvel"].astype(np.float32)),
        zero_native_time=bool(np.all(_copy(env.sim.data.time) == 0)),
        zero_reference_ticks=bool(np.all(_copy(command.physics_steps) == 0)),
        zero_sim_steps=env._sim_step_counter == 0,
        zero_policy_updates=not recorder.policies,
        zero_preintegration_records=not recorder.pre,
        initial_state_and_native_contacts_recorded=len(recorder.rows) == len(recorder.contacts) == 1,
        source_observation_shape=tuple(observations.shape) == (recorder.num_envs, 369),
        finite_source_observations=bool(np.isfinite(_copy(observations)).all()),
        one_physical_initialization=command.write_ledger.resets.tolist() == [1]*recorder.num_envs
            and command.write_ledger.writes.tolist() == [1]*recorder.num_envs)
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                integration_steps=0, actor_updates=0)


def probe_conditions(mode):
    if mode in ("contract", "diagnostic"):
        return tuple((*row, "learned") for row in CONDITIONS)
    if mode == "parity":
        return tuple((family, solver, True, residual) for family in FAMILIES
                     for solver in SOLVERS for residual in ("learned", "zero"))
    raise ValueError("Probe mode must be contract, diagnostic or parity")


def historical_physics_steps(duration):
    """Preserve the archived evaluator's literal ceil(duration / .005) rule.

    Cross has the saved duration 4.800000000000001 and historically integrated
    961 steps. Subtracting an epsilon changes that accepted evidence contract.
    The new manager environment may round *up* to complete four-step actor
    intervals; the independent evaluator still crops to this historical count.
    """
    if isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Historical evaluation requires a finite positive duration")
    return int(np.ceil(float(duration)/.005))


def run_release_baseline_probe(bundle, release_root, output, *, mode="diagnostic"):
    """Allocated Phase 0 contract checks, 12-condition probe, or 64-world parity.

    No learning is constructed. Contract mode checks initialization and one
    20 ms actor interval (four genuine 5 ms integrations) per condition. Parity
    uses every original bank state, both solver profiles and paired zero actors.
    """
    conditions = probe_conditions(mode)
    num_envs = 64 if mode == "parity" else ENVIRONMENTS
    trial_ids = list(range(num_envs))
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Release migration requires the pinned Task 10 runtime")
    from .runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    sources = load_release_inputs(checked, Path(release_root), num_envs=num_envs)
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from .native_cycle_probe import initialize_launch_drive, save_compiled_model
    from .telemetry.cycle_qualification import native_stream_scope
    wp.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(schema=SCHEMA, diagnostic_scope=DIAGNOSTIC_SCOPE, status="starting",
        runtime=runtime, slurm_job_id=os.environ["SLURM_JOB_ID"], includes_ppo=False,
        training=False, specialist_weights_are_initialization=False,
        production_qualified=False, acceptance_eligible=False, g1_entity_present=False,
        fixed_target_only=True, trial_ids=trial_ids, conditions=[], mode=mode,
        planned_conditions=len(conditions), planned_attempts=len(conditions)*num_envs,
        reference_manifest_sha256=sha256_file(checked.root/"manifest.json"),
        source_inputs={family: data["input_sha256"] for family, data in sources.items()},
        trace_semantics={"state_geometry": "current integrated state; offline fresh FK",
            "forces_contacts": "preceding integration interval; initial row is reset forward",
            "reference_clock": "integer physical tick since reset; original initial guard hold retained",
            "historical_clock_difference": "historical evaluator accumulated float32 .005; WBC samples integer ticks. Source-equivalent interpolation at declared queries, not a bitwise rollout reproduction",
            "phase_denominator": "original source duration + 1.0 seconds",
            "target_timing": "fixed Task 09 target; reset-clock crossing and strike-start offset, no commanded TTC"},
        tf32_enabled=False)
    _write_report(destination/"launch.json", report)
    started = time.monotonic()
    for family, solver, feedforward, residual in conditions:
        name = f"{family}_{solver}_{'ff' if feedforward else 'zero_ff'}"
        if mode == "parity":
            name += "_"+residual
        folder = writable_path(destination/name)
        folder.mkdir(exist_ok=False)
        data = sources[family]
        duration = float(data["clip"]["legacy_episode_duration_s"])
        historical_steps = historical_physics_steps(duration)
        policy_steps = 1 if mode == "contract" else (historical_steps+3)//4
        recorder = ReleaseRecorder(num_envs)
        cfg = build_release_probe_cfg(checked, recorder, data, solver=solver, feedforward=feedforward)
        cfg.sim.nan_guard.output_dir = str(folder/"nan_guard")
        condition = dict(name=name, family=family, solver_profile=solver, feedforward_enabled=feedforward,
            simulation_contract=asdict(cfg.sim.mujoco), nconmax=128, njmax=512,
            requested_duration_s=duration, planned_integrated_duration_s=policy_steps*.020,
            planned_terminal_padding_s=None if mode == "contract" else policy_steps*.020-duration,
            planned_policy_steps=policy_steps, planned_physics_steps=policy_steps*4,
            original_evaluator_physics_steps=historical_steps,
            planned_historical_padding_steps=None if mode == "contract" else policy_steps*4-historical_steps,
            horizon_semantics="Original literal ceil(duration/.005), then whole four-step actor intervals; parity evaluates only the original prefix",
            residual_mode=residual,
            actor_role="frozen historical diagnostic only", checkpoint_sha256=data["input_sha256"]["checkpoint"],
            initial_bank_sha256=data["input_sha256"]["bank"], trial_ids=trial_ids)
        _write_report(folder/"launch.json", condition)
        env = model = None
        error = None
        completed_policy_steps = 0
        initial_time = final_time = final_reference_tick = final_sim_step_counter = None
        condition_started = time.monotonic()
        try:
            with native_stream_scope(), torch.no_grad():
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                env._release_baseline_recorder = recorder
                env._release_baseline_raw_action = torch.zeros((num_envs, 48), device="cuda:0")
                actor = load_frozen_actor(data, "cuda:0")
                actor_before = _actor_sha256(actor)
                reference = {key: torch.as_tensor(value, dtype=torch.float32, device="cuda:0")
                             for key, value in data["legacy"].items() if np.issubdtype(value.dtype, np.number)}
                env.reset(seed=10010)
                initial_time = _copy(env.sim.data.time)
                command = env.command_manager.get_term("motion")
                command.write_ledger.seal()
                initial_drive = initialize_launch_drive(env)
                model = env.sim.mj_model
                condition["compiled_physics_audit"] = verify_release_model(model, checked, solver=solver)
                action_term = env.action_manager.get_term("joint_pos")
                if (tuple(action_term.target_names) != checked.joint_names or action_term.action_dim != 48
                        or not np.all(_copy(action_term._scale) == .25)):
                    raise ValueError("Actual diagnostic action interface differs from the preserved controller")
                origins = _copy(env.scene.env_origins)
                hands = [model.site("robot/"+side+"Hand_center").id for side in ("Left", "Right")]
                condition["compiled_model"] = save_compiled_model(model, folder/"compiled_model.mjb")
                previous = torch.zeros((num_envs, 48), device="cuda:0")
                recorder.capture(env)
                for actor_index in range(policy_steps):
                    qpos = _tensor_view(env.sim.data.qpos).clone()
                    qpos[:, :3] -= env.scene.env_origins
                    gloves = _tensor_view(env.sim.data.site_xpos)[:, hands]-env.scene.env_origins[:, None]
                    clock = command.physics_steps.to(torch.float32)*.005
                    observations = legacy_observations(qpos, _tensor_view(env.sim.data.qvel), gloves,
                                                       previous, clock, reference, duration)
                    if actor_index == 0:
                        condition["zero_step_contract"] = zero_step_contract(env, recorder, data, observations, origins)
                        _write_report(folder/"zero_step_contract.json", condition["zero_step_contract"])
                        if condition["zero_step_contract"]["status"] != "passed":
                            raise RuntimeError("Zero-step native state/observation contract failed")
                    action = actor(observations) if residual == "learned" else torch.zeros_like(previous)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Frozen specialist emitted nonfinite action")
                    recorder.policies.append({key: _copy(value) for key, value in dict(
                        observation=observations, action=action, previous_raw_action=previous,
                        reference_tick=command.physics_steps).items()})
                    env._release_baseline_raw_action = action
                    env.step(action)
                    completed_policy_steps += 1
                    previous = action.clone()
                condition["actor_unchanged"] = actor_before == _actor_sha256(actor)
                condition["initial_drive"] = initial_drive
                condition["physical_reset_counts"] = command.write_ledger.resets.tolist()
                condition["physical_write_counts"] = command.write_ledger.writes.tolist()
                condition["actual_physics_dt_s"] = float(model.opt.timestep)
                condition["actual_policy_dt_s"] = env.step_dt
                condition["actual_decimation"] = env.cfg.decimation
        except Exception as exception:
            error = dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
        finally:
            if env is not None:
                try:
                    final_time = _copy(env.sim.data.time)
                    final_reference_tick = _copy(env.command_manager.get_term("motion").physics_steps)
                    final_sim_step_counter = env._sim_step_counter
                except Exception as exception:
                    error = error or dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
                try:
                    env.close()
                except Exception as exception:
                    error = error or dict(type=type(exception).__name__, message=str(exception))
        try:
            condition.update(execution_counts(recorder, completed_policy_steps=completed_policy_steps,
                initial_time=initial_time, final_time=final_time, final_reference_tick=final_reference_tick,
                final_sim_step_counter=final_sim_step_counter))
        except Exception as exception:
            error = error or dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
            condition.update(execution_counts(recorder, completed_policy_steps=completed_policy_steps))
            condition["unresolved_native_time"] = None if final_time is None else np.asarray(final_time).tolist()
        condition["actual_network_inference_calls"] = len(recorder.policies) if residual == "learned" else 0
        arrays = recorder.all_arrays()
        if arrays:
            with (folder/"raw_trace.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            condition["raw_trace_sha256"] = sha256_file(folder/"raw_trace.npz")
        try:
            if error is None and model is not None:
                audit = audit_release_arrays(arrays, bundle=checked, inputs=data,
                    policy_steps=policy_steps, feedforward=feedforward, origins=origins)
                checks = audit["checks"]
                checks["actor_parameters_unchanged"] = condition["actor_unchanged"]
                checks["one_physical_initialization"] = condition["physical_reset_counts"] == [1]*num_envs and condition["physical_write_counts"] == [1]*num_envs
                checks["actual_native_clock_configuration"] = (condition["actual_physics_dt_s"] == .005 and condition["actual_policy_dt_s"] == .020 and condition["actual_decimation"] == 4)
                checks["executed_counts_match_plan"] = (condition["actual_physics_steps"] == policy_steps*4
                    and condition["actual_completed_policy_steps"] == policy_steps
                    and condition["actual_actor_updates"] == policy_steps)
                checks["zero_step_contract"] = condition["zero_step_contract"]["status"] == "passed"
                checks["zero_residual_control"] = residual != "zero" or bool(np.all(arrays["policy_action"] == 0))
                audit["status"] = "passed" if all(checks.values()) else "failed"
                condition["audit"] = audit
                if mode == "parity" and all(checks.values()):
                    from .release_baseline_evaluation import evaluate_historical_panel
                    condition["historical_evaluation"] = evaluate_historical_panel(model, arrays, data, origins,
                        folder/"historical_evaluation", residual_mode=residual, solver_profile=solver)
                elif mode == "diagnostic":
                    episodes, diagnostics = diagnostic_episodes(model, arrays, data, origins)
                    for world, episode in enumerate(episodes):
                        with (folder/f"episode_{world:02d}.npz").open("xb") as stream:
                            np.savez_compressed(stream, **episode)
                    condition["diagnostics"] = diagnostics
                condition["status"] = "passed" if all(checks.values()) else "failed"
            else:
                condition["status"] = "failed"
        except Exception as exception:
            error = error or dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
            condition["status"] = "failed"
        if error:
            condition["exception"] = error
        condition["wall_time_s"] = time.monotonic()-condition_started
        _write_report(folder/"audit.json", condition)
        report["conditions"].append(condition)
        # A broken controller/clock contract is not a reason to consume the rest
        # of a GPU allocation. Poor physical performance alone remains measured.
        if condition["status"] != "passed":
            break
    report["wall_time_s"] = time.monotonic()-started
    report["status"] = "passed" if len(report["conditions"]) == len(conditions) and all(c["status"] == "passed" for c in report["conditions"]) else "failed"
    report["unattempted_conditions"] = len(conditions)-len(report["conditions"])
    report["unattempted_episodes"] = report["unattempted_conditions"]*num_envs
    report["conditions_with_unresolved_physics_counts"] = sum(c["actual_physics_steps"] is None for c in report["conditions"])
    report["known_physics_steps_per_world_summed_conditions"] = sum(c["actual_physics_steps"] or 0 for c in report["conditions"])
    report["known_world_physics_transitions"] = sum(sum(c["actual_physics_steps_per_world"] or []) for c in report["conditions"])
    report["actual_physics_steps_per_world_summed_conditions"] = (None if report["conditions_with_unresolved_physics_counts"]
        else report["known_physics_steps_per_world_summed_conditions"])
    report["actual_world_physics_transitions"] = (None if report["conditions_with_unresolved_physics_counts"]
        else report["known_world_physics_transitions"])
    report["actual_completed_policy_steps_summed_conditions"] = sum(c["actual_completed_policy_steps"] for c in report["conditions"])
    report["physical_success_is_contract_pass"] = False
    _write_report(destination/"audit.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--release", "--release-root", dest="release_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("contract", "diagnostic", "parity"), default="diagnostic")
    args = parser.parse_args(argv)
    report = run_release_baseline_probe(args.bundle, args.release_root, args.output, mode=args.mode)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
