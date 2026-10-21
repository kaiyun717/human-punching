"""Failure-inclusive, frozen-actor canonical panels; never production acceptance.

One physical initialization per predeclared recipe. Strict readiness supplies t0;
unready attempts retain no launch timestamp. The native observer's 5 ms arrays
are re-read and reduced on CPU before a panel can select an engineering checkpoint.
This module does not receive or modify a learner, invoke distributed collectives,
update normalization, perform optimization, or submit jobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .provenance import PROJECT_ROOT, writable_path, write_json, sha256_file, runtime_record
from .runtime.bundle import FAMILIES, load_runtime_bundle
from .native_probe import NativeProbeRecorder, _copy, _require_allocated_cuda
from .native_cycle_probe import _save_npz, save_compiled_model, allocated_device_metadata, placed_request
from .trainedactor_evaluation import (actor_identity, verify_actor_contract, native_interfaces,
                                     verify_compiled_controller)
from .cycle_trace import derive_kinematics
from .telemetry.cycle_qualification import (QualificationCommand, QualificationCommandCfg,
    QualificationActionCfg, QualificationRecorder, native_stream_scope, stack)
from .telemetry.cycle_runtime import (CycleRuntime, JsonReceiptSink, attach_cycle_runtime,
    get_cycle_runtime)
from .telemetry.runtime_telemetry import get_observer
from .telemetry.snapshot_providers import make_snapshot_providers
from .telemetry.readiness_clock import Phase, Failure, ReadinessConfig
from .telemetry.posthoc_qualification import rotation_error_rad

SCHEMA = "human_wbc_engineering_cycle_panel_v1"
RECIPE_SCHEMA = "human_wbc_engineering_cycle_recipes_v1"
SCOPE = "canonical_four_family_scratch_engineering"


def exception_receipt(error, *, stage):
    """Retain a bounded failure location without losing the original outcome."""
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__, limit=12))
    maximum = 16384
    return {"type": type(error).__name__, "message": str(error), "stage": stage,
        "traceback": formatted[-maximum:], "traceback_truncated": len(formatted) > maximum,
        "traceback_character_limit": maximum}


@dataclass(frozen=True)
class EngineeringPanelExpectation:
    completed_iterations: int
    training_seed: int
    policy_sha256: str
    normalization_sha256: str
    checkpoint_contract_sha256: str
    reference_manifest_sha256: str
    recipe_manifest_sha256: str
    engineering_admission_sha256: str
    measurement_contract_sha256: str
    episodes_per_family: int = 16
    scope: str = SCOPE

    def __post_init__(self):
        if (type(self.completed_iterations) is not int or self.completed_iterations < 0
                or type(self.training_seed) is not int or self.training_seed < 0
                or type(self.episodes_per_family) is not int or not 1 <= self.episodes_per_family <= 64
                or self.scope != SCOPE):
            raise ValueError("Exact bounded engineering panel counts and scope required")
        for item in fields(self):
            if item.name.endswith("sha256"):
                value = getattr(self, item.name)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Every engineering panel identity requires a SHA256")


def build_engineering_recipes(bundle, episodes_per_family=16) -> dict:
    checked = load_runtime_bundle(bundle)
    if type(episodes_per_family) is not int or not 1 <= episodes_per_family <= 64:
        raise ValueError("Specify 1..64 recipes per family")
    rows = []
    for family in FAMILIES:
        clips = sorted((c for c in checked.clips if c["family"] == family), key=lambda c: c["reference_id"])
        if not clips:
            raise ValueError("Canonical panel requires all four families")
        for repeat in range(episodes_per_family):
            clip = clips[repeat % len(clips)]
            request = placed_request(clip, np.zeros(3), clip["launch_time_s"])
            rows.append({"recipe_id": f"{family}_{repeat:03d}", "family": family,
                "reference_id": clip["reference_id"], "master_sha256": clip["master_sha256"],
                "request_sha256": request.fingerprint, "repeat_index": repeat//len(clips),
                "condition_index": repeat % len(clips)})
    return {"schema": RECIPE_SCHEMA, "scope": SCOPE, "split": "canonical_engineering",
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "episodes_per_family": episodes_per_family, "recipes": rows,
        "repeated_conditions_are_independent_generalization_samples": False,
        "acceptance_qualified": False}


def engineering_recipe_bytes(payload) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def write_engineering_recipes(path, payload) -> Path:
    return write_json(path, payload)


def _recipes(bundle, path, expectation):
    data = json.loads(Path(path).read_text())
    if (sha256_file(path) != expectation.recipe_manifest_sha256
            or data != build_engineering_recipes(bundle, expectation.episodes_per_family)
            or data["reference_manifest_sha256"] != expectation.reference_manifest_sha256):
        raise ValueError("Predeclared engineering recipes/reference identity changed")
    return tuple(data["recipes"])


class PlannedCommand(QualificationCommand):
    """Only replace random episode selection with the exact predeclared recipe."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lookup = {clip["reference_id"]: i for i, clip in enumerate(self.clips)}
        self.planned_ids = torch.tensor([lookup[name] for name in cfg.planned_reference_ids],
                                       dtype=torch.long, device=self.device)
        if len(self.planned_ids) != env.num_envs:
            raise ValueError("Recipe count differs from actual environments")

    def reset(self, env_ids):
        if np.any(self.physical_resets[_copy(env_ids).astype(int)]):
            raise RuntimeError("Engineering panels forbid a second physical initialization")
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        if not getattr(self, "_reset_in_progress", False):
            raise RuntimeError("Reference selection is allowed only at initial reset")
        self.trajectory_ids[env_ids] = self.planned_ids[env_ids]
        self.physics_steps[env_ids] = 0
        self.time_steps[env_ids] = self.motion.segment_start_idx[self.trajectory_ids[env_ids]]
        self.episode_assist_gain[env_ids] = 0.
        self._episode_start_step[env_ids] = self.time_steps[env_ids]
        self._episode_start_bin[env_ids] = 0
        self._episode_similarity_sum[env_ids] = 0.
        self._episode_step_count[env_ids] = 0
        self._write_reference_state_to_sim(env_ids, self.body_pos_w[env_ids, 0], self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids], self.joint_vel[env_ids])
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class PlannedCommandCfg(QualificationCommandCfg):
    planned_reference_ids: tuple[str, ...] = ()

    def build(self, env):
        return PlannedCommand(self, env)


class PanelRecorder(QualificationRecorder):
    def __init__(self, count):
        self.native = NativeProbeRecorder(count)
        self.pre, self.endpoints, self.policies = [], [], []
        self.initial, self.nominal, self.packet, self.receipts = {}, {}, {}, []

    def policy_boundary(self, env, observations, reward, terminated, timed_out):
        super().policy_boundary(env, observations, reward, terminated, timed_out)
        self.policies[-1]["original_fresh_reward_values"] = _copy(original_fresh_rewards(env))


def fresh_reward_names():
    from .runtime.imitation_rewards import UPSTREAM_NAMES
    from .runtime.task_rewards import DEFAULT_WEIGHTS
    return ["wbc_raw:"+name for name in UPSTREAM_NAMES]+["task_returned:"+name for name in DEFAULT_WEIGHTS]


def original_fresh_rewards(env):
    """Read real reward-time values; never recompute after a launch boundary.

    WBC values precede mjlab's nan_to_num. Our task rates intentionally sanitize
    invalid-state intermediates; these are its returned rates, accompanied by
    independent native invalid-state/overflow/support diagnostics.
    """
    from .runtime.imitation_rewards import CACHE_ATTRIBUTE, UPSTREAM_NAMES
    from .runtime.task_rewards import DEFAULT_WEIGHTS
    observer = get_observer(env)
    imitation = getattr(env, CACHE_ATTRIBUTE, None)
    task = getattr(get_cycle_runtime(env), "_task_reward_cache", None)
    if (imitation is None or task is None or imitation.serial != observer.actual_samples
            or task.serial != observer.actual_samples or set(imitation.raw_values) != set(UPSTREAM_NAMES)
            or set(task.cache) != set(DEFAULT_WEIGHTS)):
        raise RuntimeError("Panel requires complete original fresh rewards at the actual policy window")
    return torch.stack([imitation.raw_values[name] for name in UPSTREAM_NAMES]
                       + [task.cache[name] for name in DEFAULT_WEIGHTS], -1)


def build_engineering_panel_cfg(bundle, recipes, recorder, *, engineering_admission):
    from .capacity_evidence import configure_engineering_capacity
    from .runtime.training_initialization import require_strict_initialization
    from mjlab.managers.metrics_manager import MetricsTermCfg
    cfg = configure_engineering_capacity(bundle, engineering_admission, num_envs=len(recipes))
    require_strict_initialization(cfg.commands["motion"].training_initialization)
    # Keep exactly the training observation/reward semantics. Terminal rows are
    # recorded once by the bridge; continuing other rows must never reset them.
    cfg.auto_reset = False
    cfg.terminations = {}
    for collection, key, cls in ((cfg.commands, "motion", PlannedCommandCfg),
                                  (cfg.actions, "joint_pos", QualificationActionCfg)):
        previous = collection[key]
        args = {f.name: getattr(previous, f.name) for f in fields(previous) if f.init}
        if key == "motion":
            args["planned_reference_ids"] = tuple(row["reference_id"] for row in recipes)
        collection[key] = cls(**args)
    cfg.metrics["engineering_panel_raw"] = MetricsTermCfg(func=recorder.metric(), per_substep=True)
    return cfg


def _cpu(values):
    return {key: torch.from_numpy(np.asarray(value).copy()) for key, value in values.items()}


def replay_attempts(arrays, body, *, readiness=ReadinessConfig()):
    """Recompute launch/failure/contact/recovery from raw endpoints, not booleans."""
    initial, nominal, packet = (_cpu(arrays[key]) for key in ("initial", "nominal", "packet"))
    n = len(initial["root_position_w"])
    result = []
    bridge = CycleRuntime(n, torch.tensor(body["joint_limits_rad"], dtype=torch.float32),
        torch.tensor(body["effort"], dtype=torch.float32), context_width=packet["g1_launch_state"].shape[-1],
        nominal_provider=lambda: nominal, packet_provider=lambda: packet,
        request_ids=lambda: torch.arange(n, dtype=torch.int64)+n,
        reference_gloves=lambda: torch.zeros((n, 2, 3)), receipt_sink=lambda row: result.append(JsonReceiptSink._plain(row)),
        config=readiness, acceptance_eligible=False)
    bridge.after_pending_reset(torch.arange(n), initial, torch.zeros(n, dtype=torch.int64),
                               torch.as_tensor(arrays["pre_interval"]["time"][0], dtype=torch.float64))
    endpoints = arrays["endpoint"]
    for i in range(len(endpoints["root_position_w"])):
        if i % 4 == 0:
            bridge.begin_window()
        state = {name: torch.from_numpy(value[i].copy()) for name, value in endpoints.items()
                 if name not in ("ready_endpoint", "quiet_intervals")}
        bridge.after_substep(torch.full((n,), i+1, dtype=torch.int64), state)
        if i % 4 == 3:
            bridge.command_boundary()
    return bridge, result


def _receipt_decisions(receipts):
    result = {}
    for batch in receipts:
        for j, world in enumerate(batch["environment_ids"]):
            if world in result:
                raise ValueError("Duplicate terminal recipe receipt")
            r, e = batch["readiness"], batch["events"]
            result[world] = {"readiness": {key: r[key][j] for key in
                ("phase", "request_id", "launch_tick", "failure_category", "failure_tick")},
                "events": {key: e[key][j] for key in ("first_contact_tick", "withdrawal_tick",
                    "first_failure_tick", "first_failure_category", "first_failure_mask", "recovery_tick")}}
    return result


def summarize_attempts(bridge, arrays, recipes):
    r, e = bridge.readiness, bridge.events
    results = []
    for world, recipe in enumerate(recipes):
        phase, launch = int(r.phase[world]), int(r.launch_tick[world])
        contact, recovery = int(e.first_contact[world]), int(e.recovery_tick[world])
        terminal = phase in (int(Phase.COMPLETE), int(Phase.FAILED))
        end = (int(r.failure_tick[world]) if phase == Phase.FAILED else
               launch+recovery if phase == Phase.COMPLETE else len(arrays["endpoint"]["root_position_w"]))
        end = max(1, end)
        state = arrays["endpoint"]
        target = arrays["packet"]["target_w"][world]
        hand = int(recipe["family"] in ("cross", "right_hook"))
        outward = float(arrays["packet"]["outward_end_s"][world])
        ttc = float(arrays["packet"]["requested_ttc_s"][world])
        timing = contact*.005-ttc if contact >= 0 else None
        points = state["glove_position_w"][launch:min(end, launch+int(math.floor(outward/.005))+1), world, hand] if launch >= 0 else []
        minimum = float(np.linalg.norm(points-target, axis=-1).min()) if len(points) else None
        nominal = arrays["nominal"]
        row = {**recipe, "environment_id": world, "terminal": terminal, "terminal_phase": Phase(phase).name,
            "terminal_physics_tick": end if terminal else None, "recorded_physics_ticks": len(state["root_position_w"]),
            "post_terminal_diagnostic_ticks": len(state["root_position_w"])-end if terminal else 0,
            "launched": launch >= 0, "launch_tick": launch if launch >= 0 else None,
            "requested_ttc_s": ttc,
            "launch_time_s": float(r.time_origin_s[world])+launch*.005 if launch >= 0 else None,
            "failure_category": Failure(int(r.failure[world])).name,
            "failure_tick": int(r.failure_tick[world]) if int(r.failure_tick[world]) >= 0 else None,
            "event_failure_mask": int(e.first_failure_mask[world]),
            "first_contact_cycle_tick": contact if contact >= 0 else None,
            "contact_ttc_error_s": timing, "outward_minimum_target_distance_m": minimum,
            "recovery_cycle_s": recovery*.005 if recovery >= 0 else None,
            "strict_cycle_success": bool(phase == Phase.COMPLETE and contact >= 0 and timing is not None
                and abs(timing) <= .02000001 and int(e.first_failure_tick[world]) < 0),
            "terminal_guard_root_speed_m_s": float(np.linalg.norm(state["root_linear_velocity_w"][end-1, world])),
            "terminal_guard_glove_error_m": float(np.linalg.norm(state["glove_position_guard_frame"][end-1, world]
                -nominal["glove_position_guard_frame"][world], axis=-1).max()),
            "terminal_footprint_error_m": float(np.linalg.norm(state["support_anchor_position_w"][end-1, world, :, :2]
                -nominal["support_anchor_position_w"][world, :, :2], axis=-1).max()),
            "acceptance_qualified": False}
        results.append(row)
    families = {}
    for family in FAMILIES:
        rows = [row for row in results if row["family"] == family]
        item = {"attempts": len(rows), "terminal": sum(row["terminal"] for row in rows),
            "launched": sum(row["launched"] for row in rows),
            "strict_cycle_success": sum(row["strict_cycle_success"] for row in rows),
            "readiness_failures": sum(row["terminal"] and not row["launched"] for row in rows),
            "failures": {name: sum(row["failure_category"] == name for row in rows)
                         for name in sorted({row["failure_category"] for row in rows})}}
        item["launch_rate"] = item["launched"]/item["attempts"]
        item["strict_cycle_success_rate"] = item["strict_cycle_success"]/item["attempts"]
        for key in ("contact_ttc_error_s", "outward_minimum_target_distance_m", "recovery_cycle_s",
                    "terminal_guard_root_speed_m_s", "terminal_guard_glove_error_m", "terminal_footprint_error_m"):
            values = [row[key] for row in rows if row[key] is not None]
            item[key] = {"observed_count": len(values), "sum": sum(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "maximum_absolute": max(abs(v) for v in values) if values else None}
        families[family] = item
    return {"episodes": results, "families": families, "attempts": len(results),
        "launched": sum(row["launched"] for row in results),
        "strict_cycle_success": sum(row["strict_cycle_success"] for row in results),
        "all_terminal": all(row["terminal"] for row in results),
        "acceptance_qualified": False, "conditional_metrics_omit_unobserved_events": True}


def audit_panel_arrays(arrays, bundle, model, recipes, *, physical_writes, physical_resets, live_receipts):
    verify_compiled_controller(model, bundle)
    native, pre, endpoint, policy = (arrays[key] for key in ("native", "pre_interval", "endpoint", "policy_boundary"))
    count, n = native["qpos"].shape[:2]
    steps = count//4
    checks = {"complete_native_windows": count > 0 and count % 4 == 0 and len(pre["qpos"]) == count,
        "complete_policy_windows": len(policy["qpos"]) == steps,
        "predeclared_recipe_count": n == len(recipes), "native_timestep": float(model.opt.timestep) == .005,
        "one_physical_initialization": physical_writes == [1]*n and physical_resets == [1]*n,
        "all_native_values_finite": all(np.isfinite(v).all() for group in arrays.values() for v in group.values()
                                       if np.issubdtype(v.dtype, np.number)),
        "no_environment_reset_or_termination": not np.any(policy["terminated"] | policy["timed_out"]),
        "no_external_support": not np.any(native["qfrc_applied"]) and not np.any(native["xfrc_applied"]),
        "free_unactuated_root": bool(np.all(abs(native["qfrc_actuator"][..., :6]) <= 1e-9)),
        "no_contact_overflow": not np.any(native["overflow"]),
        "no_native_numerical_fault": not np.any(endpoint["numerical_failure"] | endpoint["contact_overflow"]),
        "finite_observations": bool(policy["observation_finite"].all()),
        "phase_aware_actor_schema": arrays["actor"]["actor_observation"].shape == (steps, n, 398),
        "all_original_fresh_rewards_present": policy["original_fresh_reward_values"].shape == (steps, n, len(fresh_reward_names())),
        "original_fresh_rewards_finite": bool(np.isfinite(policy["original_fresh_reward_values"]).all()),
        "native_step_counter": np.array_equal(native["sim_step_index"], np.arange(1, count+1)),
        "policy_step_counter": np.array_equal(native["policy_step_index"], np.arange(count)//4)}
    kp, kd, caps = (np.asarray(bundle.body[key]) for key in ("kp", "kd", "effort"))
    limits = np.asarray(bundle.body["joint_limits_rad"], np.float32)
    expected = np.clip(pre["qref"].astype(float)+.25*native["raw_action"], limits[:, 0], limits[:, 1])
    errors = {"target_rad": float(abs(expected-native["joint_position_target"]).max()),
        "velocity_rad_s": float(abs(pre["vref"]-native["joint_velocity_target"]).max()),
        "feedforward_nm": float(abs(pre["feedforward"]-native["joint_effort_target"]).max()),
        "drive_nm": float(abs(kp*native["joint_position_target"]+kd*pre["vref"]+pre["feedforward"]-native["ctrl"]).max()),
        "force_nm": float(abs(np.clip(native["ctrl"].astype(float)-kp*pre["qpos"][..., 7:]-kd*pre["qvel"][..., 6:], -caps, caps)
                               -native["actuator_force"]).max()),
        "clock_s": float(abs(native["time"].astype(float)-pre["time"].astype(float)-.005).max())}
    for name, bound in {"target_rad": 2e-6, "velocity_rad_s": 2e-6, "feedforward_nm": 5e-6,
                         "drive_nm": 2e-4, "force_nm": 2e-3, "clock_s": 2e-6}.items():
        checks[name] = errors[name] <= bound
    actions = arrays["actor"]["executed_action"]
    checks["clipped_actor_action_held_four_ticks"] = (actions.shape == (steps, n, 48) and np.all(abs(actions) <= 1.)
        and np.array_equal(native["raw_action"], np.repeat(actions, 4, axis=0)))
    checks["guard_and_lookahead_held_before_launch"] = all(np.array_equal(pre[key][~pre["launched"]], pre["guard"][~pre["launched"]])
        for key in ("qref", "future12", "future24")) and not np.any(pre["vref"][~pre["launched"]])
    checks["one_reference_tick_per_real_interval"] = np.array_equal(pre["episode_tick"],
        np.broadcast_to(np.arange(count)[:, None], (count, n)))
    checks["reference_clock_from_actual_launch"] = np.array_equal(pre["reference_tick"],
        np.where(pre["launched"], pre["episode_tick"]-pre["launch_tick"], 0))
    checks["no_invented_t0"] = np.all(pre["launch_tick"][~pre["launched"]] == -1)
    checks["packet_bound_to_recipes"] = packet_matches_recipes(arrays, bundle, recipes)
    frozen = True
    for world in range(n):
        selected = np.flatnonzero(policy["launched"][:, world])
        for key in ("target", "actor_context", "launch_tick", "request_id"):
            if len(selected):
                values = policy[key][selected, world]
                frozen &= bool(np.all(values == values[0]))
    checks["frozen_launched_context"] = frozen
    fk_error, rotation_error = 0., 0.
    for world in range(n):
        fk = derive_kinematics(model, native["qpos"][:, world], native["qvel"][:, world])
        for name, expected_fk in fk.items():
            actual = endpoint[name][:, world]
            if "quat" in name:
                rotation_error = max(rotation_error, float(rotation_error_rad(expected_fk, actual).max()))
            else:
                fk_error = max(fk_error, float(abs(expected_fk-actual).max()))
    checks["fresh_cpu_fk_parity"] = fk_error <= 2e-5 and rotation_error <= 2e-5
    checks["fresh_actor_anchor"] = (np.max(abs(policy["root_position"]-policy["qpos"][..., :3])) <= 2e-6
        and np.max(rotation_error_rad(policy["root_quaternion"], policy["qpos"][..., 3:7])) <= 2e-6)
    bridge, receipts = replay_attempts(arrays, bundle.body)
    checks["independent_reducer_terminal_parity"] = _receipt_decisions(receipts) == _receipt_decisions(live_receipts)
    summary = summarize_attempts(bridge, arrays, recipes)
    checks["every_recipe_terminal"] = summary["all_terminal"]
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
        "controller_errors": errors, "fk_error": fk_error, "rotation_error_rad": rotation_error,
        "summary": summary}


def packet_matches_recipes(arrays, bundle, recipes):
    """Rebind launch packets to immutable clips, including environment origins."""
    clips = {row["reference_id"]: row for row in bundle.clips}
    packet = arrays["packet"]
    for world, recipe in enumerate(recipes):
        clip = clips[recipe["reference_id"]]
        with np.load(bundle.path(clip["master_path"]), allow_pickle=False) as master:
            origin = arrays["nominal"]["root_position_w"][world]-master["qpos"][0, :3].astype(np.float32)
        snap = clip["launch_snapshot"]
        context = np.asarray(sum((list(snap[key]) for key in ("g1_qpos", "g1_qvel", "head_position_w",
            "head_velocity_w", "human_origin_w", "human_quat_wxyz")), []), np.float32)
        for start in (0, 71, 77):
            context[start:start+3] += origin
        expected = {"target_w": np.asarray(clip["target_w"], np.float32)+origin,
            "requested_ttc_s": np.float32(clip["requested_ttc_s"]),
            "outward_end_s": np.float32(clip["outward_end_s"]),
            "family_id": FAMILIES.index(clip["family"]),
            "plan_digest": np.asarray(list(bytes.fromhex(clip["master_sha256"])), np.uint8),
            "actor_g1_context": np.asarray(clip["launch_features"], np.float32), "g1_launch_state": context,
            "status": 1, "plan_matches_guard_context": True}
        for name, value in expected.items():
            # Translation recovery from two float32 world positions can incur
            # two ulps. No target or readiness thresholds are changed here.
            if name in ("target_w", "g1_launch_state"):
                if not np.allclose(packet[name][world], value, rtol=0, atol=2e-5):
                    return False
            elif not np.array_equal(packet[name][world], value):
                return False
    return True


def _memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": free, "total_bytes": total, "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device), "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def memory_summary(samples):
    if not samples:
        return {"available": False}
    total = max(row["total_bytes"] for row in samples)
    driver = max(row["total_bytes"]-row["free_bytes"] for row in samples)
    external = max(max(0, row["total_bytes"]-row["free_bytes"]-row["reserved_bytes"]) for row in samples)
    peak = max(driver, max(row["torch_peak_reserved_bytes"] for row in samples)+external)
    return {"available": True, "total_vram_bytes": total, "conservative_peak_used_bytes": peak,
        "sampled_driver_peak_used_bytes": driver, "maximum_sampled_non_torch_bytes": external,
        "headroom_fraction": (total-peak)/total, "caller_allocations_remain_resident": True,
        "driver_sampling_is_not_continuous_peak_measurement": True}


def verify_panel_observations(actual, expected):
    """Bind shared meanings and every actor term, including processing/order."""
    shared = lambda value: {key: item for key, item in value.items() if key != "groups"}
    if (shared(actual) != shared(expected)
            or actual["groups"]["actor"] != expected["groups"]["actor"]):
        raise ValueError("Native engineering actor observation schema differs")


def run_engineering_cycle_panel(bundle, actor, actor_contract, recipes_path, output, *, expectation,
                                engineering_admission) -> Path:
    """Execute one bounded independent panel. Its caller owns learner/RNG isolation."""
    from .engineering_admission import verify_engineering_admission
    from .learned_guard_evaluation import evaluation_caches
    checked = load_runtime_bundle(bundle)
    verify_engineering_admission(checked, engineering_admission)
    if engineering_admission.sha256 != expectation.engineering_admission_sha256:
        raise ValueError("Panel engineering admission differs from measured learner")
    verify_actor_contract(actor, checked, actor_contract, expectation)
    recipes = _recipes(checked, recipes_path, expectation)
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Pinned runtime required for engineering panel")
    destination = writable_path(output); destination.mkdir(parents=True, exist_ok=False)
    caches = evaluation_caches(destination)
    from mjlab.envs import ManagerBasedRlEnv
    import mujoco
    import warp as wp
    torch.cuda.init(); torch.cuda.set_device(0); wp.init()
    before = actor_identity(actor)
    recorder = PanelRecorder(len(recipes))
    cfg = build_engineering_panel_cfg(checked, recipes, recorder, engineering_admission=engineering_admission)
    cfg.seed = expectation.training_seed
    cfg.sim.nan_guard.output_dir = str(destination/"nan_guard")
    maximum_steps = math.ceil((6.+max(c["requested_ttc_s"] for c in checked.clips)+4.+.04)/.020)
    report = {"schema": SCHEMA, "scope": SCOPE, "expectation": asdict(expectation), "runtime": runtime,
        "reference_bundle_path": str(checked.root.resolve().relative_to(PROJECT_ROOT.resolve())),
        "training": False, "acceptance_qualified": False, "engineering_checkpoint_selection_eligible": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "cache_paths": caches,
        "device": allocated_device_metadata("cuda:0"), "actor_contract": actor_contract,
        "actor_before": before, "maximum_policy_steps": maximum_steps,
        "fresh_reward_names": fresh_reward_names(),
        "fresh_reward_scope": "Eight original WBC outputs before manager sanitization; seventeen intentionally sanitized task rates; no post-boundary recomputation",
        "caller_learning_allocations_may_remain_resident": True,
        "other_rows_continue_after_first_terminal": True, "actual_g1_collision_response": False,
        "limitations": ["Repeated canonical conditions are not target/TTC generalization evidence.",
            "Isolated immutable G1 launch snapshot, no moving G1 entity.",
            "Caller must independently prove learner state/RNG isolation.",
            "Post-terminal diagnostic samples are retained and excluded from episode success."]}
    started = time.perf_counter(); env = None; error = None; actions = []; memories = []
    writes = resets = []
    try:
        with native_stream_scope(), torch.no_grad():
            memories.append({"stage": "before_construction", **_memory("cuda:0")})
            env = ManagerBasedRlEnv(cfg, device="cuda:0"); env._cycle_qualification = recorder
            memories.append({"stage": "after_construction", **_memory("cuda:0")})
            report["compiled_scene"] = save_compiled_model(env.sim.mj_model, destination/"compiled_scene.mjb")
            actual = native_interfaces(env, checked)
            # Critic is retained for provenance but is never input to this actor.
            for name in ("action_contract", "simulation_contract"):
                if actual[name] != actor_contract[name]:
                    raise ValueError("Native panel controller/solver differs from actor contract")
            verify_panel_observations(actual["observation_contract"], actor_contract["observation_contract"])
            report["native_interfaces"] = actual
            disk_sink = JsonReceiptSink(destination/"attempts")
            def sink(row):
                disk_sink(row); recorder.receipts.append(JsonReceiptSink._plain(row))
            providers = make_snapshot_providers(env, engineering_admitted=True)
            bridge = attach_cycle_runtime(env, **providers, receipt_sink=sink)
            observer = get_observer(env)
            original = observer.after_pending_reset
            def initialization(ids, state, tick, timestamp):
                if recorder.initial or len(ids) != len(recipes):
                    raise RuntimeError("Panel requires exactly one full physical initialization")
                original(ids, state, tick, timestamp)
                recorder.initial = {key: _copy(value) for key, value in state.items()}
                recorder.nominal = {key: _copy(value) for key, value in providers["nominal_provider"]().items()}
                recorder.packet = {key: _copy(value) for key, value in providers["packet_provider"]().items()}
            observer.after_pending_reset = initialization
            obs, _ = env.reset(seed=expectation.training_seed)
            memories.append({"stage": "after_reset", **_memory("cuda:0")})
            for step in range(maximum_steps):
                inference_start = time.perf_counter()
                raw = actor(obs["actor"])
                if raw.shape != (len(recipes), 48) or not torch.isfinite(raw).all():
                    raise RuntimeError("Frozen actor returned malformed/nonfinite actions")
                action = raw.clamp(-1., 1.)
                action_row = {"actor_observation": _copy(obs["actor"]), "raw_action": _copy(raw), "executed_action": _copy(action)}
                action_row["inference_plus_host_capture_s"] = np.asarray(time.perf_counter()-inference_start)
                actions.append(action_row)
                obs, reward, terminated, timed_out, _ = env.step(action)
                recorder.policy_boundary(env, obs, reward, terminated, timed_out)
                memories.append({"stage": "policy", "step": step+1, **_memory("cuda:0")})
                if bool(((bridge.readiness.phase == Phase.COMPLETE) | (bridge.readiness.phase == Phase.FAILED)).all()):
                    break
            report["native_execution"] = True
    except Exception as exc:
        error = exception_receipt(exc, stage="native_execution")
    finally:
        if env is not None:
            cmd = env.command_manager.get_term("motion")
            writes, resets = cmd.physical_writes.tolist(), cmd.physical_resets.tolist()
            memories.append({"stage": "before_close", **_memory("cuda:0")})
            try:
                env.close()
            except Exception as exc:
                if error is None:
                    error = exception_receipt(exc, stage="environment_close")
    arrays = {"native": recorder.native.arrays(), "pre_interval": stack(recorder.pre),
        "endpoint": stack(recorder.endpoints), "policy_boundary": stack(recorder.policies),
        "actor": stack(actions), "initial": recorder.initial, "nominal": recorder.nominal, "packet": recorder.packet}
    report["hashes"] = {name: _save_npz(destination/(name+".npz"), group) for name, group in arrays.items()}
    report["receipts"] = [{"path": str(path.relative_to(destination)), "sha256": sha256_file(path)}
                          for path in sorted((destination/"attempts").glob("*.json"))]
    report.update(exception=error, physical_writes=writes, physical_resets=resets,
        native_wall_s=time.perf_counter()-started, completed_policy_steps=len(recorder.policies),
        actual_physics_samples=len(recorder.native.rows), actor_after=actor_identity(actor), memory_samples=memories,
        memory=memory_summary(memories))
    if actions:
        latency = arrays["actor"]["inference_plus_host_capture_s"]
        report["inference_latency"] = {"scope": "actor forward, clipping and host evidence copies; includes synchronization",
            "samples": len(latency), "mean_s": float(latency.mean()), "maximum_s": float(latency.max()),
            "policy_period_s": .020}
    if error is None:
        try:
            report["audit"] = audit_panel_arrays(arrays, checked,
                mujoco.MjModel.from_binary_path(str(destination/"compiled_scene.mjb")), recipes,
                physical_writes=writes, physical_resets=resets, live_receipts=recorder.receipts)
            report["summary"] = report["audit"]["summary"]
            report["engineering_checkpoint_selection_eligible"] = bool(report["audit"]["passed"] and report["actor_after"] == before)
        except Exception as exc:
            report["exception"] = exception_receipt(exc, stage="offline_verification")
    report["status"] = "recorded" if report["engineering_checkpoint_selection_eligible"] else "failed"
    report["total_wall_s"] = time.perf_counter()-started
    return write_json(destination/"report.json", report)


def verify_engineering_panel(report_path, recipes_path, expectation):
    """Reread hash-bound native arrays and replay outcomes independently."""
    import mujoco
    report_path = Path(report_path)
    report = json.loads(report_path.read_text()); root = report_path.parent
    if (report.get("schema") != SCHEMA or report.get("scope") != SCOPE
            or report.get("expectation") != asdict(expectation) or report.get("training") is not False
            or report.get("acceptance_qualified") is not False or not report.get("native_execution")
            or report.get("exception") is not None or report.get("status") != "recorded"
            or report.get("actor_before") != report.get("actor_after")
            or report.get("actor_before") != {"policy_sha256": expectation.policy_sha256,
                                             "normalization_sha256": expectation.normalization_sha256}):
        raise ValueError("Engineering panel execution/identity is incomplete")
    if memory_summary(report.get("memory_samples", [])) != report.get("memory"):
        raise ValueError("Panel memory summary differs from retained device samples")
    if report.get("fresh_reward_names") != fresh_reward_names():
        raise ValueError("Panel does not bind the complete fresh reward schema")
    # The immutable body/reference root is explicitly recorded by the recipe's
    # actor contract source; callers supply only the exact saved bundle path.
    contract = report["actor_contract"]
    from .validation import digest
    if contract.get("sha256") != digest({key: value for key, value in contract.items() if key != "sha256"}):
        raise ValueError("Actor interface contract digest differs")
    for key in ("policy_sha256", "normalization_sha256", "checkpoint_contract_sha256", "reference_manifest_sha256"):
        if contract.get(key) != getattr(expectation, key):
            raise ValueError("Panel actor contract differs from expectation")
    actual = report["native_interfaces"]
    for key in ("action_contract", "simulation_contract"):
        if actual[key] != contract[key]:
            raise ValueError("Panel native interface differs from actor contract")
    verify_panel_observations(actual["observation_contract"], contract["observation_contract"])
    bundle = load_runtime_bundle(PROJECT_ROOT/report["reference_bundle_path"])
    recipes = _recipes(bundle, recipes_path, expectation)
    if contract["reference_manifest_sha256"] != expectation.reference_manifest_sha256:
        raise ValueError("Panel reference identity differs")
    arrays = {}
    required = {"native", "pre_interval", "endpoint", "policy_boundary", "actor", "initial", "nominal", "packet"}
    if set(report["hashes"]) != required:
        raise ValueError("Incomplete native panel trace payloads")
    consumed = [report_path, Path(recipes_path), root/"compiled_scene.mjb", bundle.root/"manifest.json"]
    for name, expected in report["hashes"].items():
        path = root/(name+".npz")
        if sha256_file(path) != expected:
            raise ValueError("Native panel trace hash differs")
        with np.load(path, allow_pickle=False) as data:
            arrays[name] = {key: value.copy() for key, value in data.items()}
        consumed.append(path)
    receipts = []
    for row in report["receipts"]:
        path = (root/row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != row["sha256"]:
            raise ValueError("Panel terminal receipt hash/path differs")
        receipts.append(json.loads(path.read_text()))
        consumed.append(path)
    if sha256_file(root/"compiled_scene.mjb") != report["compiled_scene"]["sha256"]:
        raise ValueError("Compiled panel scene changed")
    audit = audit_panel_arrays(arrays, bundle, mujoco.MjModel.from_binary_path(str(root/"compiled_scene.mjb")), recipes,
        physical_writes=report["physical_writes"], physical_resets=report["physical_resets"], live_receipts=receipts)
    if not audit["passed"] or audit["summary"] != report["summary"]:
        raise ValueError("Independent physical panel audit or reported outcomes differ")
    return {**audit["summary"], "engineering_checkpoint_selection_eligible": True,
        "acceptance_qualified": False, "report_sha256": sha256_file(report_path),
        "input_files": [{"path": str(path.resolve().relative_to(PROJECT_ROOT.resolve())), "sha256": sha256_file(path)}
                        for path in consumed]}
