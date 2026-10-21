"""Bounded Task 10B Phase 2 extension of the original specialist PPO route.

This module uses the original ActorCritic, GAE and PPO objective with mjlab's
plain Simulation. It does not instantiate a WBC environment or PPO runner.
Reference-phase training resets are rehearsal, never commanded-cycle evidence.
Only uninterrupted, readiness-gated validation supplies cycle measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from torch import nn

from .jab_learning import (ActorCritic, advantages, actuator_layout, load_reference,
    support_feedforward, interpolate_reference, points_in_body_frame, quat_up,
    maximum_joint_limit_excursion)
from .scalar_control import configure_implicit_pd, native_pd_drive, export_runtime_model
from .functional_recovery import (FunctionalGuardThresholds, derive_guard_nominal,
    measure_functional_guard, evaluate_functional_trace)
from .contact import first_contact_time
from .wbc.provenance import writable_path, sha256_file, write_json
from .wbc.release_baseline_probe import _tensor_view, _copy, SOLVERS


@dataclass(frozen=True)
class PilotBudget:
    environments: int = 128
    rollout_steps: int = 64
    maximum_updates: int = 32
    maximum_wall_s: float = 1200.
    validation_every: int = 8
    no_improvement_checks: int = 2
    shutdown_reserve_s: float = 90.

    def __post_init__(self):
        if self.environments != 128 or self.rollout_steps != 64:
            raise ValueError("Initial Phase 2 original PPO route is exactly 128 x 64")
        if not 1 <= self.maximum_updates <= 32 or self.validation_every != 8:
            raise ValueError("Pilot cap is 32 updates with validation every eight updates")
        if not 0 < self.maximum_wall_s <= 4*3600 or self.shutdown_reserve_s <= 0:
            raise ValueError("Finite session wall limit <= four hours and shutdown reserve required")
        if self.no_improvement_checks != 2:
            raise ValueError("Pilot stops after two physical validation checks without improvement")

    @property
    def transition_cap(self):
        return self.environments*self.rollout_steps*self.maximum_updates


def policy_digest(policy):
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def family_arguments(release_root, family):
    """Use the selected family's actual reward/controller settings, not defaults."""
    root = Path(release_root).resolve(strict=True)
    if family not in ("jab", "cross", "left_hook", "right_hook"):
        raise ValueError("Unknown specialist family")
    manifest = json.loads((root/"manifest.json").read_text())["artifacts"]
    paths = {}
    for key, name in (("arguments", "arguments.json"), ("schema", "controller_schema.json"),
                      ("checkpoint", "checkpoint.pt"), ("model", "model/human.xml"),
                      ("body_metadata", "body_metadata.json"), ("reference", "reference.npz"),
                      ("bank", "evaluation/evaluation_bank.npz")):
        path = (root/family/name).resolve(strict=True)
        if not path.is_relative_to(root) or sha256_file(path) != manifest[f"{family}/{name}"]["sha256"]:
            raise ValueError(f"Accepted release input changed: {family}/{name}")
        paths[key] = path
    args = json.loads(paths["arguments"].read_text())
    args.setdefault("guard_joint_weight", 0.)
    for key, value in dict(controller="implicit_pd", timestep=.005, policy_dt=.020,
                           speed=1., hold=1., gain_scale=1., action_scale=.25,
                           no_feedforward=False, balance_feedback=False,
                           refresh_kinematics=True, horizon=64, epochs=4,
                           minibatch=2048, learning_rate=1.e-4, target_kl=.03).items():
        if args.get(key) != value:
            raise ValueError(f"Accepted original PPO/controller field differs: {key}")
    args.update(model=paths["model"], body_metadata=paths["body_metadata"], reference=paths["reference"])
    return SimpleNamespace(**args), paths


class ReferenceLibrary:
    """Shared resident arrays; sampling never tiles references across worlds."""
    def __init__(self, model, rows, *, device="cpu", byte_limit=2*1024**3):
        if not rows or len({row["family"] for row in rows}) != 1:
            raise ValueError("A specialist requires one family's explicit reference rows")
        self.rows, self.device, self.arrays = list(rows), device, []
        self.bytes = 0
        for row in rows:
            if not row.get("kinematic_admission", False):
                raise ValueError(f"Rejected reference cannot be executed: {row['candidate_id']}")
            if int(row["launch_tick"]) % 4 or row["outward_end_tick"] <= row["launch_tick"]:
                raise ValueError("Reference requires actor-boundary launch and later outward completion")
            for name in ("reference", "legacy_runtime"):
                if sha256_file(row[name+"_path"]) != row[name+"_sha256"]:
                    raise ValueError("Reference changed after catalog commitment")
            data = load_reference(model, Path(row["reference_path"]), 1.)
            with np.load(row["legacy_runtime_path"], allow_pickle=False) as archive:
                # Original controls use their exact source-grid torque; candidates
                # contain consistently recomputed derivatives and source feedforward.
                if not np.allclose(data["t"], archive["t"], atol=1.e-10, rtol=0):
                    raise ValueError("Runtime and kinematic reference grids differ")
                data["torque"] = archive["torque"].copy()
            duration = float(row["phase_denominator_s"])
            if not np.isclose(duration, data["t"][-1]+1., atol=1.e-8):
                raise ValueError("Phase denominator must include full reference plus one-second hold")
            data["episode_duration"] = duration
            data["initial_root_height"] = float(data["qpos"][0, 2])
            data["launch"] = int(row["launch_tick"])*.005
            data["outward_end"] = int(row["outward_end_tick"])*.005
            with np.load(row["reference_path"], allow_pickle=False) as source:
                attack_window = source["attack_window_reference_s"] if "attack_window_reference_s" in source else [data["launch"], data["outward_end"]]
            data["attack_start"], data["attack_end"] = map(float, attack_window)
            cooked = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                      for k, v in data.items() if v is not None}
            self.bytes += sum(a.numel()*a.element_size() for a in cooked.values())
            if self.bytes > byte_limit:
                raise ValueError("Reference resident-byte bound exceeded")
            self.arrays.append(cooked)
        self.durations = torch.tensor([float(r["phase_denominator_s"]) for r in rows], device=device)
        self.launch_ticks = torch.tensor([r["launch_tick"] for r in rows], dtype=torch.long, device=device)
        self.outward_ticks = torch.tensor([r["outward_end_tick"] for r in rows], dtype=torch.long, device=device)
        self.commitment = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def content_digest(self):
        digest = hashlib.sha256(json.dumps(self.rows, sort_keys=True).encode())
        for reference in self.arrays:
            for key, value in sorted(reference.items()):
                digest.update(key.encode())
                digest.update(_copy(value).tobytes())
        return digest.hexdigest()

    def sample(self, key, ids, clock):
        if ids.ndim != 1 or clock.shape != ids.shape:
            raise ValueError("One reference ID and reference clock per world required")
        output = None
        for index, reference in enumerate(self.arrays):
            values = reference[key]
            sampled = (interpolate_reference(reference["t"], values, clock)
                       if values.ndim and len(values) == len(reference["t"]) and
                       key not in ("head_center", "guard_glove_chest_pos")
                       else values.expand(len(ids), *values.shape))
            if output is None:
                output = torch.empty_like(sampled)
            selected = ids == index
            output[selected] = sampled[selected]
        if output is None or bool(((ids < 0)|(ids >= len(self.arrays))).any()):
            raise ValueError("Unsupported reference ID")
        if key == "qpos":
            output[:, 3:7] /= output[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1.e-8)
        return output


class NativeSpecialistEnv:
    """Original plain Simulation controller with family reference batching."""
    def __init__(self, args, rows, *, num_envs=128, device="cuda:0", solver="accepted"):
        import warp as wp
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from .wbc.telemetry.contacts import NativeContactTelemetry
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Native Phase 2 execution requires an allocated Slurm CUDA GPU")
        if solver not in SOLVERS or num_envs < 1:
            raise ValueError("An explicit qualified solver and positive world count are required")
        self.args, self.device, self.n, self.dt, self.decimation = args, device, num_envs, .005, 4
        self.model = mujoco.MjModel.from_xml_path(str(args.model))
        self.ids, self.qadr, self.vadr = actuator_layout(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (55, 54, 48):
            raise ValueError("Accepted physical dimensions changed")
        body = json.loads(Path(args.body_metadata).read_text())
        self.kp = torch.tensor(body["kp"], dtype=torch.float32, device=device)
        self.kd = torch.tensor(body["kd"], dtype=torch.float32, device=device)
        self.torque_limits = configure_implicit_pd(self.model, body["kp"], body["kd"])
        self.model.opt.timestep = self.dt
        profile = SOLVERS[solver]
        self.solver = solver
        self.library = ReferenceLibrary(self.model, rows, device=device)
        self.low = torch.tensor(self.torque_limits[:, 0], dtype=torch.float32, device=device)
        self.high = torch.tensor(self.torque_limits[:, 1], dtype=torch.float32, device=device)
        self.joint_low = torch.tensor(self.model.jnt_range[self.ids, 0], dtype=torch.float32, device=device)
        self.joint_high = torch.tensor(self.model.jnt_range[self.ids, 1], dtype=torch.float32, device=device)
        wp.init()
        cache = writable_path(os.environ["WARP_CACHE_PATH"])
        cache.mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(cache)
        cfg = SimulationCfg(nconmax=128, njmax=512, mujoco=MujocoCfg(**profile))
        cfg.nan_guard.output_dir = str(writable_path(Path.cwd()/"nan_guard"))
        self.sim = Simulation(num_envs, cfg, self.model, device)
        self.model = self.sim.mj_model
        self.hands = [self.model.site(side+"Hand_center").id for side in ("Left", "Right")]
        self.feet = [self.model.geom("physical/"+side+"Foot").id for side in ("Left", "Right")]
        self.chest = self.model.body("Chest").id
        self.hips = self.model.body("Hips").id
        self.ids_ref = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ticks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_action = torch.zeros((num_envs, 48), device=device)
        self.rejected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.launched = torch.zeros_like(self.rejected)
        self.reset_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.completed = self.fallen = self.physics_steps_executed = self.actor_updates_executed = 0
        self.hard_faults = []
        self.last_support = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        # Contacts use public MuJoCo-Warp decoding, preserving cone dependence.
        bind = SimpleNamespace(prefix="", foot_geoms=("physical/LeftFoot", "physical/RightFoot"),
            floor_geoms=("floor",), human_geom_prefix="physical/", g1_geom_names=(),
            allowed_self_pairs=(), patch_radius_m=.065,
            # Reducer's >= next representable float32 equals the accepted >1e-6
            # comparison for its float32 native forces; shared code is unchanged.
            minimum_normal_force_n=float(np.nextafter(np.float32(1.e-6), np.float32(np.inf))))
        self.contacts = NativeContactTelemetry(self.model, self.sim.wp_model, self.sim.wp_data,
                                               bind, torch_device=device)
        self.initial_up = torch.zeros((num_envs, 3), device=device)
        self.initial_up[:, 2] = 1.
        self.active_hand_index = 0 if rows[0]["family"] in ("jab", "left_hook") else 1
        self.nominal_qpos = load_reference(self.model, args.reference)["qpos"][0]
        self.nominal = derive_guard_nominal(self.model, self.nominal_qpos, family=rows[0]["family"])
        self.nominal_centers = torch.tensor(self.nominal["glove_centers_chest_m"], dtype=torch.float32, device=device)

    @property
    def clock(self):
        return self.ticks.float()*self.dt

    @property
    def duration(self):
        return self.library.durations[self.ids_ref]

    def ref(self, key, offset=0.):
        # A rejected request stays on its pre-outward boundary without pose or
        # history replacement; future features cannot leak an unlaunched stroke.
        query = self.clock+torch.where(self.rejected, 0., offset)
        return self.library.sample(key, self.ids_ref, query)

    def reset(self, env_ids=None, *, random_phase=True, perturb=True, reference_ids=None,
              initial_qpos=None, initial_qvel=None):
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        count = len(env_ids)
        self.sim.reset(env_ids)
        self.ids_ref[env_ids] = (torch.randint(len(self.library.rows), (count,), device=self.device)
            if reference_ids is None else torch.as_tensor(reference_ids, dtype=torch.long, device=self.device))
        self.ticks[env_ids] = 0
        self.rejected[env_ids], self.launched[env_ids] = False, False
        if random_phase:
            starts = torch.rand(count, device=self.device)
            endpoint = self.duration[env_ids]-1.
            self.ticks[env_ids] = (torch.where(starts < .5, 0., (starts-.5)*1.6*endpoint)/self.dt).long()
        pose, velocity = self.ref("qpos")[env_ids].clone(), self.ref("qvel")[env_ids].clone()
        if perturb:
            # D3 initial local caps; Gaussian draw retains the original shape.
            pose[:, self.qadr] += (.005*torch.randn((count, 48), device=self.device)).clamp(-.015, .015)
            pose[:, self.qadr] = pose[:, self.qadr].clamp(self.joint_low, self.joint_high)
            velocity += (.02*torch.randn_like(velocity)).clamp(-.06, .06)
        if initial_qpos is not None or initial_qvel is not None:
            if initial_qpos is None or initial_qvel is None:
                raise ValueError("Explicit initial state must supply both qpos and qvel")
            pose = torch.as_tensor(initial_qpos, dtype=torch.float32, device=self.device)
            velocity = torch.as_tensor(initial_qvel, dtype=torch.float32, device=self.device)
            if pose.shape != (count, 55) or velocity.shape != (count, 54):
                raise ValueError("Malformed explicit initial-state bank")
        _tensor_view(self.sim.data.qpos)[env_ids] = pose
        _tensor_view(self.sim.data.qvel)[env_ids] = velocity
        self.last_action[env_ids] = 0
        self.reset_counts[env_ids] += 1
        self.sim.forward()
        rotation = _tensor_view(self.sim.data.xmat)[env_ids, self.hips].reshape(count, 3, 3)
        self.initial_up[env_ids] = rotation[:, 2, :]
        self.last_support = self.support_contact()
        return self.observe()

    def support_contact(self):
        anchors = _tensor_view(self.sim.data.geom_xpos)[:, self.feet]
        result = self.contacts.sample(anchors)
        if bool((result["numerical_failure"]|result["contact_overflow"]).any()):
            raise RuntimeError("Native contact telemetry fault or capacity overflow")
        return result["support_contact"].clone()

    def observe(self):
        q, v = _tensor_view(self.sim.data.qpos), _tensor_view(self.sim.data.qvel)
        desired, desired_v = self.ref("qpos"), self.ref("qvel")
        phase = self.clock/self.duration
        parts = [q[:, 2:7], v[:, :6]*.2, q[:, self.qadr], v[:, self.vadr]*.1,
            (desired[:, :3]-q[:, :3])*5, desired[:, 3:7], desired[:, self.qadr]-q[:, self.qadr],
            (desired_v[:, self.vadr]-v[:, self.vadr])*.1,
            torch.stack((phase, torch.sin(phase*2*np.pi), torch.cos(phase*2*np.pi)), -1), self.last_action]
        for offset in (.1, .25):
            future = self.ref("qpos", offset)
            parts.extend(((future[:, :3]-q[:, :3])*5, future[:, self.qadr]-q[:, self.qadr]))
        parts.append((self.ref("glove_pos")-_tensor_view(self.sim.data.site_xpos)[:, self.hands]).flatten(1)*5)
        observation = torch.cat(parts, -1)
        if observation.shape != (self.n, 369) or not bool(torch.isfinite(observation).all()):
            raise RuntimeError("Malformed or nonfinite original specialist observation")
        return observation.clamp(-20, 20)

    def readiness(self):
        q, v = _tensor_view(self.sim.data.qpos), _tensor_view(self.sim.data.qvel)
        root_rot = _tensor_view(self.sim.data.xmat)[:, self.hips].reshape(self.n, 3, 3)
        upright_z = torch.einsum("nij,nj->ni", root_rot, self.initial_up)[:, 2]
        chest_rot = _tensor_view(self.sim.data.xmat)[:, self.chest].reshape(self.n, 3, 3)
        relative = points_in_body_frame(_tensor_view(self.sim.data.site_xpos)[:, self.hands],
                                        _tensor_view(self.sim.data.xpos)[:, self.chest], chest_rot)
        return (torch.isfinite(q).all(-1)&torch.isfinite(v).all(-1)&self.last_support.all(-1)
            &(q[:, 2] >= .75*self.nominal["root_height_m"])&(upright_z >= math.cos(.60))
            &((relative-self.nominal_centers).norm(dim=-1) <= .05).all(-1))

    def launch_gate(self):
        requested = (self.ticks == self.library.launch_ticks[self.ids_ref])&~self.launched&~self.rejected
        ready = self.readiness()
        self.launched |= requested&ready
        self.rejected |= requested&~ready
        return requested, ready

    def advance_substep(self, action, *, record=None, stop_on_fall=False):
        desired, desired_v, ff = self.ref("qpos"), self.ref("qvel"), self.ref("torque")
        target = (desired[:, self.qadr]+.25*action.clamp(-1, 1)).clamp(self.joint_low, self.joint_high)
        drive = native_pd_drive(target, desired_v[:, self.vadr], self.kp, self.kd, ff)
        _tensor_view(self.sim.data.ctrl)[:] = drive
        if record is not None:
            record.before(self, action, desired, desired_v, ff, target, drive)
        self.sim.step()
        self.physics_steps_executed += 1
        self.ticks += (~self.rejected).long()
        # Contact/force data belong to this preceding interval. Capture before
        # any forward() refresh at the next actor boundary.
        self.last_support = self.support_contact()
        if record is not None:
            record.after(self)
        self.check_physics(stop_on_fall=stop_on_fall)

    def check_physics(self, *, stop_on_fall=False):
        q, v = _tensor_view(self.sim.data.qpos), _tensor_view(self.sim.data.qvel)
        force = _tensor_view(self.sim.data.actuator_force)
        faults = {}
        faults["nonfinite"] = ~(torch.isfinite(q).all(-1)&torch.isfinite(v).all(-1)&torch.isfinite(force).all(-1))
        faults["root_assistance"] = (_tensor_view(self.sim.data.qfrc_actuator)[:, :6].abs().amax(-1)>1.e-9)
        faults["external_assistance"] = ((_tensor_view(self.sim.data.qfrc_applied).abs().amax(-1)>1.e-9)
            |(_tensor_view(self.sim.data.xfrc_applied).flatten(1).abs().amax(-1)>1.e-9))
        faults["effort_cap"] = ((force < self.low-1.e-4)|(force > self.high+1.e-4)).any(-1)
        # World-up tilt is checked exactly again from saved states; accepted
        # initial orientations are not necessarily identity.
        vector, scalar = q[:, 4:7], q[:, 3:4]
        cross = torch.linalg.cross(vector, self.initial_up)
        current_up = self.initial_up+2*(scalar*cross+torch.linalg.cross(vector, cross))
        self._last_physical_fallen = ((q[:, 2] < .65*self.ref("initial_root_height"))
            |(quat_up(q[:, 3:7]) < .3)|(v.abs().amax(-1) > 100))
        if stop_on_fall:
            faults["fall"] = (q[:, 2] < .75*self.nominal["root_height_m"])|(current_up[:, 2] < math.cos(.60))
        failed = {key: value.nonzero().flatten().detach().cpu().tolist() for key, value in faults.items() if bool(value.any())}
        if failed:
            receipt = dict(physics_steps=self.physics_steps_executed, reference_ticks=_copy(self.ticks).tolist(), faults=failed)
            self.hard_faults.append(receipt)
            raise RuntimeError("Phase 2 hard physical stop: "+json.dumps(receipt, sort_keys=True))

    def reward(self, action, peak_limit):
        """Original family objective with reference-indexed windows/guard poses."""
        q, v = _tensor_view(self.sim.data.qpos), _tensor_view(self.sim.data.qvel)
        desired, desired_v = self.ref("qpos"), self.ref("qvel")
        joint_error = (q[:, self.qadr]-desired[:, self.qadr]).square().mean(-1)
        velocity_error = (v[:, self.vadr]-desired_v[:, self.vadr]).square().mean(-1)
        root_error = (q[:, :3]-desired[:, :3]).square().sum(-1)
        root_rotation = 1-(q[:, 3:7]*desired[:, 3:7]).sum(-1).square().clamp(0, 1)
        body_error = (_tensor_view(self.sim.data.xpos)[:, 1:]-self.ref("body_pos")[:, 1:]).square().sum(-1).mean(-1)
        glove_errors = (_tensor_view(self.sim.data.site_xpos)[:, self.hands]-self.ref("glove_pos")).square().sum(-1)
        rotation = _tensor_view(self.sim.data.site_xmat)[:, self.hands].reshape(self.n, 2, 9)
        glove_rotation = (rotation-self.ref("glove_rotation")).square().mean((-1, -2))
        reward = (.15*torch.exp(-12*joint_error)+.05*torch.exp(-.05*velocity_error)
            +.10*torch.exp(-60*body_error)+.20*torch.exp(-60*root_error)
            +.10*torch.exp(-12*root_rotation)+.35*torch.exp(-self.args.glove_sharpness*glove_errors.mean(-1))
            +.05*torch.exp(-6*glove_rotation))
        attack = (self.clock >= self.ref("attack_start"))&(self.clock <= self.ref("attack_end"))
        weight = self.args.attack_tracking_weight*attack.float()
        active = 1/(1+glove_errors[:, self.active_hand_index]/self.args.attack_tracking_scale**2)
        reward = (1-weight)*reward+weight*active
        guard = self.clock >= self.ref("guard_hold_start")
        rotation = _tensor_view(self.sim.data.xmat)[:, self.chest].reshape(self.n, 3, 3)
        relative = points_in_body_frame(_tensor_view(self.sim.data.site_xpos)[:, self.hands],
                                        _tensor_view(self.sim.data.xpos)[:, self.chest], rotation)
        error = (relative-self.ref("guard_glove_chest_pos")).square().sum(-1).mean(-1)
        weight = self.args.guard_relative_weight*guard.float()
        reward = (1-weight)*reward+weight/(1+error/.02**2)
        if self.args.guard_joint_weight:
            initial = self.library.sample("qpos", self.ids_ref, torch.zeros_like(self.clock))
            error = (q[:, self.qadr]-initial[:, self.qadr]).abs().amax(-1)
            weight = self.args.guard_joint_weight*guard.float()
            reward = (1-weight)*reward+weight*torch.exp(-(error/.1).square())
        reward -= .05*guard.float()*(1-torch.exp(-.2*v.square().mean(-1)))
        reward -= .001*action.square().mean(-1)+.002*(action-self.last_action).square().mean(-1)
        reward -= self.args.joint_limit_penalty*(peak_limit/.01).square()
        return reward

    def step(self, action):
        peak = torch.zeros(self.n, device=self.device)
        fallen = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        for _ in range(4):
            self.advance_substep(action)
            fallen |= self._last_physical_fallen
            peak = torch.maximum(peak, maximum_joint_limit_excursion(
                _tensor_view(self.sim.data.qpos)[:, self.qadr], self.joint_low, self.joint_high))
        self.sim.forward()
        reward = self.reward(action, peak)-2*fallen.float()
        self.last_action[:] = action
        self.actor_updates_executed += 1
        completed = self.clock >= self.duration
        done = fallen|completed
        self.fallen += int(fallen.sum())
        self.completed += int((completed&~fallen).sum())
        if bool(done.any()):
            self.reset(done.nonzero().flatten())
        return self.observe(), reward, done


class TraceRecorder:
    def __init__(self):
        self.rows, self.pre, self.actor, self.contacts = [], [], [], []

    def after(self, env):
        row = {key: _copy(getattr(env.sim.data, key)) for key in ("qpos", "qvel", "time", "ctrl",
            "actuator_force", "qfrc_actuator", "qfrc_applied", "xfrc_applied", "overflow")}
        row.update(reference_tick=_copy(env.ticks), support_contact=_copy(env.last_support),
                   reset_count=_copy(env.reset_counts), launched=_copy(env.launched), rejected=_copy(env.rejected))
        row["efc_force"] = _copy(env.sim.data.efc.force)
        self.rows.append(row)
        count = int(_copy(env.contacts.count)[0])
        if not 0 <= count <= env.contacts.capacity:
            raise RuntimeError("Native contact storage count exceeded allocated capacity")
        contact = {key: _copy(value)[:count] for key, value in env.contacts.arrays.items()}
        contact.update(force=_copy(env.contacts.force_tensor)[:count],
                       dimension=_copy(env.contacts.dimension)[:count],
                       efc_address=_copy(env.contacts.address)[:count])
        self.contacts.append(contact)

    def before(self, env, action, qref, vref, ff, target, drive):
        self.pre.append({key: _copy(value) for key, value in dict(qpos=_tensor_view(env.sim.data.qpos),
            qvel=_tensor_view(env.sim.data.qvel), reference_tick=env.ticks, raw_action=action,
            qref=qref, vref=vref, feedforward=ff, target=target, drive=drive).items()})

    def arrays(self):
        result = {key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]} if self.rows else {}
        for prefix, rows in (("pre_", self.pre), ("policy_", self.actor)):
            if rows:
                result.update({prefix+key: np.stack([r[key] for r in rows]) for key in rows[0]})
        if self.contacts:
            result["contact_offsets"] = np.r_[0, np.cumsum([len(row["geom"]) for row in self.contacts])]
            result.update({"contact_"+key: np.concatenate([row[key] for row in self.contacts]) for key in self.contacts[0]})
        return result


def first_crossing(time_s, distance, radius=.14):
    """First sampled crossing with interpolation; never replace an early hit."""
    t, d = np.asarray(time_s), np.asarray(distance)
    hits = np.flatnonzero(d <= radius)
    if not len(hits):
        return None
    i = int(hits[0])
    if i == 0 or d[i-1] <= radius:
        return float(t[i])
    return float(t[i-1]+(t[i]-t[i-1])*(d[i-1]-radius)/(d[i-1]-d[i]))


def first_glove_crossing(time_s, glove_positions, target, active_hand):
    """Segment/sphere first crossing; both endpoints may lie outside the sphere."""
    positions = np.asarray(glove_positions)[:, active_hand]
    value = first_contact_time(time_s, np.repeat(positions[:, None, :], 2, axis=1), target)
    return float(value) if np.isfinite(value) else None


def timing_result(first_contact_s, launch_s, requested_ttc_s):
    if first_contact_s is None or launch_s is None:
        return dict(hit=False, actual_ttc_s=None, timing_error_s=None, early_hit=False, timing_pass=False)
    actual = float(first_contact_s-launch_s)
    if requested_ttc_s is None:
        return dict(hit=True, actual_ttc_s=actual, timing_error_s=None, early_hit=actual < .50,
                    timing_pass=None, control_only=True)
    earliest = max(.50, float(requested_ttc_s)-.020)
    error = actual-float(requested_ttc_s)
    early = actual < earliest-1.e-9
    return dict(hit=True, actual_ttc_s=actual, timing_error_s=error, early_hit=early,
                earliest_admissible_s=earliest, timing_pass=not early and abs(error) <= .020+1.e-9)


def audit_trace(arrays, env):
    """Independent interval-paired law/history/clock audit from preserved arrays."""
    if len(arrays.get("pre_qpos", ()))+1 != len(arrays.get("qpos", ())):
        return dict(passed=False, reason="Incomplete interval/state pairing")
    a = arrays
    expected = a["pre_drive"]-a["pre_qpos"][:, :, env.qadr]*_copy(env.kp)-a["pre_qvel"][:, :, env.vadr]*_copy(env.kd)
    expected = np.clip(expected, env.torque_limits[:, 0], env.torque_limits[:, 1])
    # Recorded force is native actuator force computed before integration.
    force_error = float(np.max(abs(expected-a["actuator_force"][1:]), initial=0))
    checks = dict(force_law=force_error <= 1.e-3,
        native_drive=np.allclose(a["ctrl"][1:], a["pre_drive"], atol=1.e-4, rtol=0),
        no_root_assistance=bool(np.max(abs(a["qfrc_actuator"][:, :, :6]), initial=0) <= 1.e-9),
        no_external_assistance=bool(np.max(abs(a["qfrc_applied"]), initial=0) <= 1.e-9 and np.max(abs(a["xfrc_applied"]), initial=0) <= 1.e-9),
        caps=bool(np.all(a["actuator_force"] >= env.torque_limits[:, 0]-1.e-4) and np.all(a["actuator_force"] <= env.torque_limits[:, 1]+1.e-4)),
        interval_state_pairing=bool(np.array_equal(a["pre_qpos"], a["qpos"][:-1]) and np.array_equal(a["pre_qvel"], a["qvel"][:-1])),
        native_5ms_clock=bool(np.allclose(np.diff(a["time"], axis=0), .005, atol=2.e-6, rtol=0)),
        one_initialization=bool(np.all(a["reset_count"] == a["reset_count"][0])),
        actor_boundary_ticks=bool(np.array_equal(a["policy_episode_tick"], np.arange(len(a["policy_episode_tick"]))*4)),
        raw_previous_action=bool(np.array_equal(a["policy_previous_raw_action"][1:], a["policy_action"][:-1]) and np.all(a["policy_previous_raw_action"][0] == 0)),
        four_step_residual=bool(np.array_equal(a["pre_raw_action"], np.repeat(a["policy_action"], 4, axis=0))))
    return dict(passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()}, maximum_force_error_nm=force_error)


def evaluate_policy(env, policy, output, *, reference_ids, initial_qpos=None, initial_qvel=None,
                    seed=10010, predicate=None, trial_ids=None):
    """One reset, uninterrupted gated cycles; no reference or actor-history swaps."""
    folder = writable_path(output)
    folder.mkdir(parents=True, exist_ok=False)
    reference_ids = np.asarray(reference_ids, dtype=np.int64)
    if reference_ids.shape != (env.n,):
        raise ValueError("Evaluation requires one declared reference per simulated world")
    if trial_ids is None:
        trial_ids = [f"pilot_{i:04d}" for i in range(env.n)]
    if len(trial_ids) != env.n or len(set(trial_ids)) != env.n:
        raise ValueError("One unique declared trial ID per world required")
    cfg = FunctionalGuardThresholds(**(predicate or {}).get("thresholds", {}))
    torch.manual_seed(seed)
    before, references_before = policy_digest(policy), env.library.content_digest()
    recorder = TraceRecorder()
    chosen = [env.library.rows[i] for i in reference_ids]
    horizon = int(math.ceil(max(float(r["episode_duration_s"]) for r in chosen)/.020-1.e-10))*4
    planned = dict(schema="task10b_phase2_execution_plan_v1", seed=seed, trials=trial_ids,
        reference_ids=reference_ids.tolist(), candidates=[r["candidate_id"] for r in chosen],
        planned_physics_steps=horizon, planned_policy_steps=horizon//4,
        planned_simulated_world_steps=horizon*env.n, horizon_semantics="full longest reference+1s, ceil to actor boundary; per-world gates stop at own padded horizon",
        reference_commitment=env.library.commitment, actor_sha256=before,
        phase_denominators_s=[r["phase_denominator_s"] for r in chosen],
        event_semantics="request reset; prep/readiness/final immutable commit checked on current actor-boundary launch; outward endpoint reviewed source-playback proxy; no state/history replacement",
        planning_scope="D0 cached fixed-target row; continuity from initial reference hold. No arbitrary live-state planner, retargeting, homing or real-time latency claim.",
        fixed_target_only=True, g1_actor_executed=False, independent_domain_qualification=False)
    write_json(folder/"plan.json", planned)
    started, error, commitments = time.monotonic(), None, []
    try:
        with torch.no_grad():
            env.reset(random_phase=False, perturb=initial_qpos is None, reference_ids=reference_ids,
                      initial_qpos=initial_qpos, initial_qvel=initial_qvel)
            recorder.after(env)
            for episode_tick in range(0, horizon, 4):
                env.sim.forward()
                boundary_started = time.monotonic()
                requested_launch, current_ready = env.launch_gate()
                obs = env.observe()
                action = policy.actor(obs)
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Nonfinite deterministic policy output")
                for world in _copy(requested_launch.nonzero().flatten()).astype(int):
                    row = chosen[world]
                    committed = bool(_copy(current_ready)[world])
                    state = {key: _copy(getattr(env.sim.data, key))[world].tolist()
                             for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")}
                    commitments.append(dict(trial_id=trial_ids[world], world=int(world),
                        requested_family=row["family"], requested_ttc_s=row.get("requested_ttc_s"),
                        requested_style=row["candidate_id"], candidate_id=row["candidate_id"],
                        requested_episode_tick=episode_tick, reference_tick=int(_copy(env.ticks)[world]),
                        actual_launch_time_s=episode_tick*.005 if committed else None,
                        final_planning_simulation_time_s=episode_tick*.005,
                        disposition="committed" if committed else "current_readiness_rejection",
                        reference_sha256=row["reference_sha256"], legacy_runtime_sha256=row["legacy_runtime_sha256"],
                        target_w=row["target_w"], catalog_content_sha256=references_before,
                        measured_human_state=state, previous_raw_action=_copy(env.last_action)[world].tolist(),
                        outgoing_raw_action=_copy(action)[world].tolist(),
                        state_and_history_replaced=False, no_closest_reference_substitution=True,
                        static_D0_cached_reference_only=True))
                recorder.actor.append(dict(observation=_copy(obs), action=_copy(action),
                    previous_raw_action=_copy(env.last_action), reference_tick=_copy(env.ticks),
                    episode_tick=np.asarray(episode_tick), readiness=_copy(env.readiness()),
                    qpos=_copy(env.sim.data.qpos), qvel=_copy(env.sim.data.qvel),
                    ctrl=_copy(env.sim.data.ctrl), qacc_warmstart=_copy(env.sim.data.qacc_warmstart),
                    reference_id=_copy(env.ids_ref), phase_denominator_s=_copy(env.duration),
                    launched=_copy(env.launched), rejected=_copy(env.rejected),
                    admission_observation_actor_wall_s=np.asarray(time.monotonic()-boundary_started)))
                for _ in range(4):
                    env.advance_substep(action, record=recorder, stop_on_fall=True)
                env.last_action[:] = action
                env.actor_updates_executed += 1
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception))
    arrays = recorder.arrays()
    if arrays:
        with (folder/"raw_trace.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    write_json(folder/"launch_commitments.json", dict(schema="task10b_phase2_launch_commitments_v1",
        commitments=commitments, clock="simulation seconds since initial physical reset",
        wall_latency_field="policy_admission_observation_actor_wall_s includes gate, observation, actor and evidence copies; it is not isolated final-planning latency",
        no_target_updates_after_commitment=True, arbitrary_live_state_planning_qualified=False))
    audit = audit_trace(arrays, env) if error is None else dict(passed=False, reason="execution exception")
    audit["actor_unchanged"] = before == policy_digest(policy)
    audit["reference_catalog_unchanged"] = env.library.content_digest() == references_before
    audit["passed"] = bool(audit["passed"] and audit["actor_unchanged"] and audit["reference_catalog_unchanged"])
    reports = []
    if error is None:
        for world, row in enumerate(chosen):
            own_steps = (int(np.ceil(row["episode_duration_s"]/.005)) if row["kind"] == "original_control"
                         else int(math.ceil(row["episode_duration_s"]/.020-1.e-10))*4)
            n = own_steps+1
            q, v = arrays["qpos"][:n, world], arrays["qvel"][:n, world]
            t = np.arange(n)*.005
            measures = measure_functional_guard(env.model, q, v, arrays["support_contact"][:n, world],
                                                env.nominal, row["target_w"], thresholds=cfg)
            launch_s, outward_s = row["launch_tick"]*.005, row["outward_end_tick"]*.005
            functional, events = evaluate_functional_trace(t, measures, launch_time_s=launch_s,
                outward_complete_time_s=outward_s, active_hand=env.active_hand_index, thresholds=cfg)
            launched = bool(arrays["launched"][-1, world])
            # Gated rejected references cannot be posthoc reclassified as launched.
            if not launched:
                functional.update(launch_time_s=None, launch_rejected=True, functional_complete=False,
                                  recovered_after_outward=False, recovery_time_s=None)
            active = env.active_hand_index
            first = first_glove_crossing(t, measures["glove_position_w"], row["target_w"], active)
            other = first_glove_crossing(t, measures["glove_position_w"], row["target_w"], 1-active)
            timing = timing_result(first, functional["launch_time_s"], row.get("requested_ttc_s"))
            ref = env.library.arrays[reference_ids[world]]
            desired = interpolate_reference(ref["t"], ref["glove_pos"], torch.tensor(t, dtype=torch.float32, device=env.device))
            if row["kind"] == "original_control":
                expected_contact = first_glove_crossing(_copy(ref["t"]), _copy(ref["glove_pos"]), row["target_w"], active)
                schedule_error = None if first is None or expected_contact is None else first-expected_contact
                timing.update(original_reference_contact_s=expected_contact,
                    original_schedule_error_s=schedule_error,
                    original_schedule_precision_pass=schedule_error is not None and abs(schedule_error) <= .020+1.e-9,
                    timing_pass=schedule_error is not None and abs(schedule_error) <= .020+1.e-9,
                    commanded_ttc_qualification=False)
            first_within_outward = bool(first is not None and launch_s <= first <= outward_s)
            timing["first_contact_in_outward"] = first_within_outward
            squared = np.sum((measures["glove_position_w"][:, active]-_copy(desired)[:, active])**2, axis=-1)
            outward_mask = (t >= launch_s)&(t <= outward_s)
            full_rms, outward_rms = float(np.sqrt(np.mean(squared))), float(np.sqrt(np.mean(squared[outward_mask])))
            limits = np.maximum(env.model.jnt_range[env.ids, 0]-q[:, env.qadr], q[:, env.qadr]-env.model.jnt_range[env.ids, 1])
            excursion = float(np.max(limits, initial=0))
            numeric = bool(functional["functional_complete"] and timing["hit"] and
                (timing["timing_pass"] is not False) and first_within_outward
                and not (other is not None and (first is None or other <= first))
                and full_rms <= .020 and outward_rms <= .020 and excursion <= .020 and audit["passed"])
            result = dict(trial_id=trial_ids[world], candidate_id=row["candidate_id"], family=row["family"],
                reference_id=int(reference_ids[world]), numeric_complete_cycle=numeric,
                behavior_visual_review_required=True, complete_cycle_qualified=False,
                first_active_contact_s=first, first_inactive_contact_s=other, timing=timing,
                contact_geometry_semantics="First segment/sphere crossing of saved 5ms centers; interpolation is diagnostic, not finer physics measurement. Sampled contact remains in functional diagnostics.",
                full_active_glove_rms_m=full_rms, outward_active_glove_rms_m=outward_rms,
                maximum_joint_limit_excursion_rad=excursion, functional=functional,
                request_time_s=0., preparation_complete_time_s=launch_s if launched else None,
                final_planning_time_s=launch_s if launched else None,
                reference_commitment=row["reference_sha256"], target_at_launch=row["target_w"] if launched else None,
                elapsed_wall_s=time.monotonic()-started)
            reports.append(result)
            with (folder/f"trial_{world:04d}.npz").open("xb") as stream:
                np.savez_compressed(stream, time_s=t, qpos=q, qvel=v,
                    support_contact=arrays["support_contact"][:n, world], **{"event_"+k: v for k, v in events.items()})
            write_json(folder/f"trial_{world:04d}.json", result)
    result = dict(schema="task10b_phase2_native_panel_v1", status="failed" if error or not audit["passed"] else "executed",
        planned_trials=env.n, executed_trials=len(reports), incomplete_trials=env.n-len(reports),
        actual_physics_steps=len(recorder.pre), actual_policy_updates=len(recorder.actor),
        numeric_complete_cycles=sum(r["numeric_complete_cycle"] for r in reports),
        hard_faults=env.hard_faults, exception=error, audit=audit, trials=reports,
        wall_time_s=time.monotonic()-started, actor_sha256=before,
        no_training=True, formal_acceptance=False, full_domain_qualification=False)
    write_json(folder/"summary.json", result)
    return result


def fresh_policy(paths, device):
    saved = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if saved["observations"] != 369 or saved["actions"] != 48:
        raise ValueError("Accepted specialist checkpoint has incompatible dimensions")
    policy = ActorCritic(369, 48).to(device)
    policy.load_state_dict(saved["policy"], strict=True)
    return policy


def save_policy(path, policy, optimizer, *, ancestry, updates, fresh_transitions, library):
    destination = writable_path(path)
    with destination.open("xb") as stream:
        torch.save(dict(schema="task10b_phase2_original_ppo_v1", policy=policy.state_dict(),
            optimizer=optimizer.state_dict(), observations=369, actions=48,
            ancestry=ancestry, updates=updates, fresh_transitions=fresh_transitions,
            reference_commitment=library.commitment, reference_rows=library.rows,
            controller="original_native_affine_5ms_20ms", phase_denominator="per-reference full duration+1s",
            actor_parameters_sha256=policy_digest(policy)), stream)
    return dict(path=str(destination), sha256=sha256_file(destination))


def restored_policy_parity(path, policy, observations):
    saved = torch.load(path, map_location=observations.device, weights_only=True)
    restored = ActorCritic(369, 48).to(observations.device)
    restored.load_state_dict(saved["policy"], strict=True)
    with torch.no_grad():
        expected, actual = policy.actor(observations), restored.actor(observations)
    return dict(parameters_equal=policy_digest(restored) == policy_digest(policy),
                actions_equal=bool(torch.equal(expected, actual)), maximum_action_error=float((expected-actual).abs().max()))


def validation_score(panel):
    if panel.get("status") != "executed" or panel.get("hard_faults"):
        return None
    trials = panel["trials"]
    if not trials:
        return None
    # Physical criteria, never PPO reward or final test outcomes.
    successes = sum(r["numeric_complete_cycle"] for r in trials)/len(trials)
    hit_timing = sum(r.get("timing", {}).get("timing_pass") is True for r in trials)/len(trials)
    rms_values = [r["outward_active_glove_rms_m"] for r in trials if "outward_active_glove_rms_m" in r]
    rms = float(np.mean(rms_values)) if rms_values else float("inf")
    return successes, hit_timing, -rms


def validation_without_rng_side_effects(callback, policy, update, *, device):
    """Deterministic validation must not rewind a fine-tuning seed's RNG stream."""
    numpy_state = np.random.get_state()
    selected = torch.device(device)
    devices = [selected.index if selected.index is not None else torch.cuda.current_device()] if selected.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            return callback(policy, update)
    finally:
        np.random.set_state(numpy_state)


def train_pilot(env, policy, output, *, paths, seed, validate, budget=PilotBudget()):
    """Original PPO math; finite transition/wall caps and physical validation."""
    if env.n != budget.environments:
        raise ValueError("Training environment count differs from declared pilot budget")
    folder = writable_path(output)
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=env.args.learning_rate)
    ancestry = dict(checkpoint=str(paths["checkpoint"]), sha256=sha256_file(paths["checkpoint"]),
        seed=seed, fresh_optimizer=True, fresh_simulation=True, independent_scratch_replication=False,
        initial_policy_digest=policy_digest(policy))
    write_json(folder/"training_plan.json", dict(budget=asdict(budget), transition_cap=budget.transition_cap,
        ancestry=ancestry, validation_updates=list(range(0, budget.maximum_updates+1, 8)),
        training_reset_semantics="original half guard/half sampled-reference-phase rehearsal; no commanded-cycle credit",
        reference_commitment=env.library.commitment, algorithm="original PPO .2 clip, GAE .99/.95, value .5, entropy .001, gradnorm1, family reward settings"))
    started, updates, transitions, stagnation = time.monotonic(), 0, 0, 0
    stop, exception, panels, best = "transition_cap", None, [], None
    observation = None
    initial = policy_digest(policy)
    try:
        panel = validation_without_rng_side_effects(validate, policy, 0, device=env.device)
        panels.append(panel)
        best = validation_score(panel)
        if best is None:
            raise RuntimeError("Initial physical validation had a hard failure")
        observation = env.reset()
        with (folder/"training.jsonl").open("x") as log:
            for update in range(1, budget.maximum_updates+1):
                if time.monotonic()-started >= budget.maximum_wall_s-budget.shutdown_reserve_s:
                    stop = "wall_limit"
                    break
                history = {k: [] for k in ("obs", "action", "logp", "value", "reward", "done")}
                with torch.no_grad():
                    for _ in range(budget.rollout_steps):
                        distribution = policy.distribution(observation)
                        action = distribution.sample()
                        value = policy.critic(observation).squeeze(-1)
                        successor, reward, done = env.step(action)
                        for key, value in (("obs", observation), ("action", action),
                            ("logp", distribution.log_prob(action).sum(-1)), ("value", value),
                            ("reward", reward), ("done", done)):
                            history[key].append(value)
                        observation = successor
                        transitions += env.n
                    data = {k: torch.stack(v) for k, v in history.items()}
                    advantage, returns = advantages(data["reward"], data["value"], data["done"],
                                                    policy.critic(observation).squeeze(-1))
                    advantage = (advantage-advantage.mean())/(advantage.std()+1.e-8)
                flat = {k: v.flatten(0, 1) for k, v in data.items()}
                adv, target = advantage.flatten(), returns.flatten()
                losses, kls, early = [], [], False
                for _ in range(env.args.epochs):
                    for indices in torch.randperm(len(adv), device=env.device).split(env.args.minibatch):
                        distribution = policy.distribution(flat["obs"][indices])
                        log_ratio = distribution.log_prob(flat["action"][indices]).sum(-1)-flat["logp"][indices]
                        ratio = log_ratio.exp()
                        kl = float(((ratio-1)-log_ratio).mean().detach())
                        if kl > 1.5*env.args.target_kl:
                            early = True
                            break
                        actor = -torch.minimum(ratio*adv[indices], ratio.clamp(.8, 1.2)*adv[indices]).mean()
                        value_loss = (policy.critic(flat["obs"][indices]).squeeze(-1)-target[indices]).square().mean()
                        loss = actor+.5*value_loss-.001*distribution.entropy().sum(-1).mean()
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite PPO loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                        if not bool(torch.isfinite(norm)):
                            raise RuntimeError("Nonfinite PPO gradient")
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        kls.append(kl)
                    if early:
                        break
                updates = update
                receipt = dict(update=update, fresh_transitions=transitions, elapsed_s=time.monotonic()-started,
                    mean_reward=float(data["reward"].mean()), mean_loss=float(np.mean(losses)) if losses else None,
                    approximate_kl=float(np.mean(kls)) if kls else None, optimizer_steps=len(losses),
                    kl_early_stopped=early, actor_digest=policy_digest(policy))
                log.write(json.dumps(receipt, allow_nan=False)+"\n")
                log.flush()
                print(json.dumps(receipt), flush=True)
                if update % budget.validation_every == 0:
                    checkpoint = save_policy(folder/f"checkpoint_{update:05d}.pt", policy, optimizer,
                        ancestry=ancestry, updates=updates, fresh_transitions=transitions, library=env.library)
                    parity = restored_policy_parity(checkpoint["path"], policy, observation)
                    write_json(folder/f"reload_{update:05d}.json", parity)
                    if not parity["parameters_equal"] or not parity["actions_equal"]:
                        raise RuntimeError("Saved checkpoint reload parity failed")
                    panel = validation_without_rng_side_effects(validate, policy, update, device=env.device)
                    panels.append(panel)
                    score = validation_score(panel)
                    if score is None:
                        raise RuntimeError("Physical validation hard failure")
                    if score > best:
                        best, stagnation = score, 0
                    else:
                        stagnation += 1
                    if stagnation >= budget.no_improvement_checks:
                        stop = "physical_validation_no_improvement"
                        break
                    # Validation may use this environment; a new training episode
                    # is explicit. Never relabel this reset as cycle continuity.
                    observation = env.reset()
    except Exception as error:
        stop, exception = "hard_failure", dict(type=type(error).__name__, message=str(error))
    checkpoint = save_policy(folder/"checkpoint_final.pt", policy, optimizer, ancestry=ancestry,
                             updates=updates, fresh_transitions=transitions, library=env.library)
    parity = None if observation is None else restored_policy_parity(checkpoint["path"], policy, observation)
    result = dict(schema="task10b_phase2_original_ppo_pilot_v1", status="bounded_stop" if exception is None else "failed",
        stop_reason=stop, exception=exception, fresh_transitions=transitions, completed_updates=updates,
        transition_cap=budget.transition_cap, wall_time_s=time.monotonic()-started,
        weight_changed=initial != policy_digest(policy), initial_policy_digest=initial,
        final_policy_digest=policy_digest(policy), checkpoint=checkpoint, reload_parity=parity,
        ancestry=ancestry, validation_panels=panels, hard_faults=env.hard_faults,
        phase2_complete=False, formal_acceptance=False)
    write_json(folder/"summary.json", result)
    return result


def run_cell_panel(env, policy, rows, output, *, initial_qpos, initial_qvel, seed, predicate):
    """Declared D0 bank panel, retaining offline rejection rows in its denominator."""
    folder = writable_path(output)
    folder.mkdir(parents=True, exist_ok=False)
    reports, panels, faults = [], [], []
    active_ids = {r["candidate_id"]: i for i, r in enumerate(env.library.rows)}
    exception = None
    for row in rows:
        trial_ids = [f"{row['candidate_id']}_bank_{i:02d}" for i in range(env.n)]
        if row["candidate_id"] not in active_ids:
            reports.extend(dict(trial_id=trial_id, candidate_id=row["candidate_id"],
                disposition="offline_rejection", reasons=row.get("rejection_reasons", []),
                numeric_complete_cycle=False) for trial_id in trial_ids)
            continue
        if exception is not None:
            reports.extend(dict(trial_id=trial_id, candidate_id=row["candidate_id"],
                disposition="unattempted_after_hard_failure", numeric_complete_cycle=False) for trial_id in trial_ids)
            continue
        panel = evaluate_policy(env, policy, folder/row["candidate_id"],
            reference_ids=np.full(env.n, active_ids[row["candidate_id"]]),
            initial_qpos=initial_qpos, initial_qvel=initial_qvel, seed=seed,
            predicate=predicate, trial_ids=trial_ids)
        panels.append(dict(candidate_id=row["candidate_id"], path=str(folder/row["candidate_id"]),
                           status=panel["status"], wall_time_s=panel["wall_time_s"]))
        reports.extend(dict(r, disposition="executed") for r in panel["trials"])
        if panel["status"] != "executed":
            exception = panel["exception"] or dict(type="TraceAuditFailure", message="Physical trace audit failed")
            done = {r["trial_id"] for r in panel["trials"]}
            reports.extend(dict(trial_id=trial_id, candidate_id=row["candidate_id"],
                disposition="incomplete_hard_failure", numeric_complete_cycle=False) for trial_id in trial_ids if trial_id not in done)
        faults.extend(panel["hard_faults"])
    executed = [r for r in reports if r["disposition"] == "executed"]
    result = dict(schema="task10b_phase2_d0_panel_v1", status="failed" if exception else "executed",
        trials=reports, panels=panels, planned_trials=len(rows)*env.n, executed_trials=len(executed),
        numeric_complete_cycles=sum(r["numeric_complete_cycle"] for r in reports),
        offline_rejections=sum(r["disposition"] == "offline_rejection" for r in reports),
        unattempted_trials=sum(r["disposition"] == "unattempted_after_hard_failure" for r in reports),
        incomplete_trials=sum(r["disposition"] == "incomplete_hard_failure" for r in reports),
        hard_faults=faults, exception=exception, independent_domain_qualification=False,
        scope="D0 eight preserved historical family-bank states per requested cell; training/validation development only; not the final held-out domain panel")
    write_json(folder/"summary.json", result)
    return result


def execute_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("frozen", "pilot"), required=True)
    parser.add_argument("--family", choices=("jab", "cross", "left_hook", "right_hook"), required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--max-updates", type=int, default=32)
    parser.add_argument("--wall-seconds", type=float, default=1680.)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--release", type=Path, default=root/"outputs/four-punch-milestone")
    parser.add_argument("--predicate", type=Path, default=root/"outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json")
    options = parser.parse_args()
    if not np.isfinite(options.wall_seconds) or not 0 < options.wall_seconds <= 4*3600:
        raise ValueError("Every execution session requires a finite wall limit <= four hours")
    from .wbc.provenance import runtime_record
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Phase 2 runtime pins differ from preserved accepted interpreter")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use the Phase 2 launcher inside an allocated Slurm GPU job")
    catalog = json.loads(options.catalog.read_text())
    if catalog.get("schema") != "task10b_phase2_reference_catalog_v1":
        raise ValueError("Unknown Phase 2 reference catalog schema")
    rows = sorted([row for row in catalog["cells"] if row["family"] == options.family],
                  key=lambda row: (row["kind"] != "original_control", row.get("requested_ttc_s") or 0))
    admitted = [row for row in rows if row["kinematic_admission"]]
    if not rows or not admitted:
        raise ValueError("No requested/admitted references for this family")
    folder = writable_path(options.output)
    folder.mkdir(parents=True, exist_ok=False)
    predicate = json.loads(options.predicate.read_text())
    args, paths = family_arguments(options.release, options.family)
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = (PilotBudget(maximum_updates=options.max_updates, maximum_wall_s=options.wall_seconds)
              if options.mode == "pilot" else None)
    write_json(folder/"launch.json", dict(schema="task10b_phase2_original_route_launch_v1",
        mode=options.mode, family=options.family, seed=options.seed, slurm_job_id=os.environ["SLURM_JOB_ID"],
        runtime=runtime, catalog=str(options.catalog.resolve()), catalog_sha256=sha256_file(options.catalog),
        predicate_sha256=sha256_file(options.predicate), sources={str(Path(__file__).resolve()): sha256_file(__file__)},
        release_inputs={key: dict(path=str(path), sha256=sha256_file(path)) for key, path in paths.items()},
        planned_cells=[r["candidate_id"] for r in rows], diagnostic_bank_ids=list(range(8)),
        planned_trials_per_panel=len(rows)*8, budget=asdict(budget) if budget else None,
        fresh_transition_cap=budget.transition_cap if budget else 0,
        no_test_split_access=True, no_g1_actor=True, no_policy_promotion=True))
    started = time.monotonic()
    import warp as wp
    wp.init()
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    with native_stream_scope(), torch.no_grad():
        evaluator = NativeSpecialistEnv(args, admitted, num_envs=8)
        policy = fresh_policy(paths, evaluator.device)
        export_runtime_model(args.model, folder/"runtime.xml", evaluator.model)
        with np.load(paths["bank"], allow_pickle=False) as archive:
            bank_q, bank_v = archive["qpos"][:8].copy(), archive["qvel"][:8].copy()
    def validate(policy, update):
        with native_stream_scope():
            return run_cell_panel(evaluator, policy, rows, folder/f"validation_{update:05d}",
                initial_qpos=bank_q, initial_qvel=bank_v, seed=options.seed+1000, predicate=predicate)
    if options.mode == "frozen":
        result = validate(policy, 0)
    else:
        candidates = sorted([r for r in admitted if r["kind"] != "original_control"], key=lambda r: r["requested_ttc_s"])
        if len(candidates) < 2:
            raise ValueError("A timing pilot requires two explicitly admitted training bins")
        selected = candidates
        write_json(folder/"training_selection.json", dict(candidate_ids=[r["candidate_id"] for r in selected],
            rule="All three initially declared candidate TTCs for D0; final interpolation TTCs remain separately prespecified and untrained",
            original_control_training=False, untrained_cells=[r["candidate_id"] for r in rows if r not in selected]))
        with native_stream_scope():
            env = NativeSpecialistEnv(args, selected, num_envs=128)
            remaining = options.wall_seconds-(time.monotonic()-started)
            if remaining <= budget.shutdown_reserve_s:
                raise RuntimeError("Compilation consumed the declared session wall budget")
            budget = PilotBudget(maximum_updates=options.max_updates, maximum_wall_s=remaining)
            result = train_pilot(env, policy, folder/"training", paths=paths, seed=options.seed,
                                  validate=validate, budget=budget)
    write_json(folder/"execution.json", dict(mode=options.mode, status=result["status"],
        total_wall_s=time.monotonic()-started, summary_path=str(folder/("training/summary.json" if options.mode == "pilot" else "validation_00000/summary.json")),
        slurm_job_id=os.environ["SLURM_JOB_ID"], phase2_complete=False))
    return 0 if result["status"] in ("executed", "bounded_stop") else 1


def main():
    # Preserve early constructor/instrumentation failures without touching an
    # output directory that existed before this process started.
    output, existed = None, True
    if "--output" in sys.argv:
        index = sys.argv.index("--output")
        if index+1 < len(sys.argv):
            output = writable_path(sys.argv[index+1])
            existed = output.exists()
    started = time.monotonic()
    try:
        return execute_cli()
    except Exception as error:
        receipt = dict(schema="task10b_phase2_early_failure_v1", status="failed",
            exception=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
            wall_time_s=time.monotonic()-started, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            native_counts="Inspect retained partial panel traces; absent attempts remain unattempted",
            no_automatic_retry=True, phase2_complete=False)
        if output is not None and not existed and output.exists():
            write_json(output/"execution_error.json", receipt)
        print(json.dumps(receipt, allow_nan=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
