"""Task 10B Phase 1: frozen original-specialist execution and separate semantics.

The native loop deliberately preserves Phase 0's qualified controller contract.
Only four accepted-solver learned panels run. The historical evaluator remains
unchanged; functional measurements and contact-independent recovery are additive.
No WBC learner is constructed, and no phase event writes physical/policy state.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .wbc.provenance import runtime_record, sha256_file, writable_path
from .wbc.native_probe import _configure_caches, _require_allocated_cuda
from .wbc.release_baseline_probe import (
    FAMILIES, ReleaseRecorder, load_release_inputs, load_frozen_actor,
    _actor_sha256, _copy, _tensor_view, _write_report, legacy_observations,
    build_release_probe_cfg, verify_release_model, zero_step_contract,
    execution_counts, audit_release_arrays, historical_physics_steps,
)

def run_phase1_probe(bundle, release_root, output, *, mode="pilot", predicate=None):
    """Four frozen original-reference panels; no training or state replacement.

    The Phase 1 state adapter observes the saved 5 ms states. Its admission and
    completion classifications never alter the controller or rewrite a trace.
    This isolated scheduled-reference experiment is not a command scheduler.
    """
    if mode not in ("pilot", "qualification") or predicate is None:
        raise ValueError("Phase 1 requires an explicit pilot/qualification predicate")
    predicate_path = Path(predicate).resolve(strict=True)
    predicate_record = json.loads(predicate_path.read_text())
    conditions = [(family, "accepted", True, "learned") for family in FAMILIES]
    num_envs = 16 if mode == "pilot" else 64
    trial_ids = list(range(num_envs))
    _require_allocated_cuda()
    from .wbc.egl_binding import configure_allocated_egl
    egl_binding = configure_allocated_egl()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Release migration requires the pinned Task 10 runtime")
    from .wbc.runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    sources = load_release_inputs(checked, Path(release_root), num_envs=num_envs)
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from .wbc.native_cycle_probe import initialize_launch_drive, save_compiled_model
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    wp.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(schema="task10b_phase1_native_v1", diagnostic_scope="isolated_original_reference_readiness_recovery", status="starting",
        program_phase=1, predicate=predicate_record, predicate_sha256=sha256_file(predicate_path),
        egl_binding=egl_binding,
        phase_adapter="observational at 5ms from saved uninterrupted states; no physical/history changes",
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
        name += "_"+residual
        folder = writable_path(destination/name)
        folder.mkdir(exist_ok=False)
        data = sources[family]
        duration = float(data["clip"]["legacy_episode_duration_s"])
        historical_steps = historical_physics_steps(duration)
        policy_steps = (historical_steps+3)//4
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
            actor_role="frozen accepted Task09 specialist; zero learning", checkpoint_sha256=data["input_sha256"]["checkpoint"],
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
                if all(checks.values()):
                    from .wbc.release_baseline_evaluation import evaluate_historical_panel
                    from .functional_recovery import evaluate_functional_panel
                    condition["historical_evaluation"] = evaluate_historical_panel(model, arrays, data, origins,
                        folder/"historical_evaluation", residual_mode=residual, solver_profile=solver)
                    condition["functional_evaluation"] = evaluate_functional_panel(model, arrays, data, origins,
                        folder/"functional_evaluation", predicate=predicate_record,
                        historical_summary=condition["historical_evaluation"])
                    if mode == "pilot":
                        from .jab_evaluation import render_rollout
                        import imageio.v2 as imageio
                        metrics = json.loads((folder/"historical_evaluation/trial_00.json").read_text())
                        media = render_rollout(folder/"historical_evaluation/historical_evaluation_runtime.xml",
                            folder/"historical_evaluation/rollout_00.npz", folder/"media",
                            head_center=metrics["head_center_w"],
                            label=f"Task10B Phase1 | {family} | frozen policy | nominal trial00",
                            fps=50, playback_speeds=(1.0,), artifact_prefix=family)
                        with imageio.get_reader(media["video"]) as reader:
                            decoded = sum(1 for _ in reader)
                        if decoded != media["video_frames"]:
                            raise ValueError("Pilot media full decode/frame count differs")
                        condition["media"] = dict(media, decoded_frames=decoded,
                            selection="trial00 fixed before pilot; no success selection",
                            artifacts={str(p.relative_to(folder)): dict(sha256=sha256_file(p), bytes=p.stat().st_size)
                                       for p in sorted((folder/"media").iterdir()) if p.is_file()})
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
    for name in ("bundle", "release", "output", "predicate"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "qualification"), required=True)
    args = parser.parse_args(argv)
    report = run_phase1_probe(args.bundle, args.release, args.output,
                             mode=args.mode, predicate=args.predicate)
    return 0 if report["status"] == "passed" else 2

if __name__ == "__main__":
    raise SystemExit(main())
