"""Phase 1 integration evidence: fixed reference, dynamic Fresh96, no learning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .g1 import PROJECT, DynamicG1, load_snapshot_rows, make_task08_cfg
from .replay import CommittedReplay, PrescribedH2


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + "\n")


def cpu(value):
    from .media import cpu as to_cpu
    return to_cpu(value)


def create_plant(config, output, h2=None):
    from mjlab.envs import ManagerBasedRlEnv
    from .camera import add_camera_only
    cfg = make_task08_cfg(count=1, seed=config["seed"], output_dir=output,
        extra_entities=None if h2 is None else {"h2": h2.entity_cfg()},
        scene_spec_fn=None if h2 is None else add_camera_only)
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    g1 = DynamicG1(env, noise_seed=config["seed"] + 1,
                   observation_noise_enabled=config["nominal_observation_noise"])
    return env, g1


def synchronized_geometry(g1, binding, sample):
    """Native combined geometry after both robots reach the same 5 ms boundary."""
    import mujoco_warp as mjwarp
    import warp as wp
    binding.write(sample, g1.data)
    with wp.ScopedDevice(g1.env.sim.wp_device):
        mjwarp.kinematics(g1.env.sim.wp_model, g1.env.sim.wp_data)
        mjwarp.camlight(g1.env.sim.wp_model, g1.env.sim.wp_data)
    row = g1.geometry()
    gloves = cpu(g1.data.geom_xpos[:, binding.glove_geom_ids])[0]
    return row, gloves


def run_scenario(g1, h2, binding, rows, config, output, requested_ttc, camera=None):
    import torch
    from .media import DiagnosticMedia
    sid = int(rows["snapshot_ids"][0])
    label = f"snapshot_{sid}_" + ("source" if requested_ttc is None else "ttc0700")
    print(f"BEGIN {label}", flush=True)
    reset_record = g1.reset(rows)
    with g1.ordered_execution():
        initial, _ = synchronized_geometry(g1, binding, h2.sample(0.))
        frozen = cpu(initial["head_position"])[0]
    if sid == config["nominal_snapshot_id"]:
        expected = h2.reference.metadata["nominal_target_adaptation"]["target"]["target_world"]
        np.testing.assert_allclose(frozen, expected, atol=5e-6, rtol=0.)
    replay = CommittedReplay(h2, tuple(frozen), requested_ttc=requested_ttc)
    duration = replay.duration_s + config["terminal_hold_s"]
    # End on a genuine actor/camera boundary, retaining the entire terminal hold.
    control_steps = int(np.ceil(duration / .020))
    physics_steps = 4 * control_steps
    trace = {name: [] for name in ("time_s", "source_s", "qpos", "qvel", "head", "gloves",
        "frozen_distances", "moving_distances", "mechanically_safe", "stance_energy")}
    actions, camera_rows = [], []
    frozen_first = moving_first = mechanical_first = clearance_first = None
    max_glove_error = 0.
    media = None
    if camera is not None:
        media = DiagnosticMedia(output / label, camera.renderer.host_model, label)
    keyframes = set(np.round(np.array([0., .510, .680, 1.0, 1.420]) / replay.rate / .02).astype(int))
    try:
        for step in range(physics_steps + 1):
            elapsed = step * .005
            sample = replay.sample(elapsed)
            with g1.ordered_execution(), torch.no_grad():
                geometry, gloves = synchronized_geometry(g1, binding, sample)
                head = cpu(geometry["head_position"])[0]
                frozen_distances = np.linalg.norm(gloves - frozen, axis=-1)
                moving_distances = np.linalg.norm(gloves - head, axis=-1)
                max_glove_error = max(max_glove_error, float(np.max(np.abs(gloves - sample.glove_pos))))
                mechanically_safe = bool(cpu(geometry["mechanically_safe"])[0])
                if frozen_first is None and frozen_distances.min() <= .14:
                    frozen_first = elapsed
                if moving_first is None and moving_distances.min() <= .14:
                    moving_first = elapsed
                if mechanical_first is None and not mechanically_safe:
                    mechanical_first = elapsed
                if clearance_first is None and moving_distances.min() < .16:
                    clearance_first = elapsed
                for key, value in dict(time_s=elapsed, source_s=sample.source_time_s,
                    qpos=cpu(g1.data.qpos)[0], qvel=cpu(g1.data.qvel)[0], head=head, gloves=gloves,
                    frozen_distances=frozen_distances, moving_distances=moving_distances,
                    mechanically_safe=mechanically_safe,
                    stance_energy=float(cpu(geometry["stance_energy"])[0])).items():
                    trace[key].append(value)
                if step % 4 == 0:
                    if camera is not None:
                        exposure = camera.capture(elapsed, reset=(step == 0))
                        camera_rows.append({name: value for name, value in exposure.items()
                            if name != "rgb" and ("timestamp" in name or "tick" in name
                            or "padding" in name or "pixel" in name or name == "visual_age_s")})
                        media.append(g1.data, exposure, time_s=elapsed,
                            source_s=sample.source_time_s,
                            hazard=f"glove={clearance_first}, mechanical={mechanical_first}",
                            keyframe=(step // 4 in keyframes or step == physics_steps))
                    if step < physics_steps:
                        action = g1.control_action()
                        actions.append(cpu(action)[0])
                        g1.apply_physical_target(action)
                if step < physics_steps:
                    # H2 has no dynamic participation. Its endpoint pose is set
                    # before stepping and both geometries are refreshed afterward.
                    binding.write(replay.sample((step + 1) * .005), g1.data)
                    g1.step_substep()
            if step % 100 == 0:
                print(f"{label} {step}/{physics_steps} t={elapsed:.3f}", flush=True)
    finally:
        if media is not None:
            media.close()
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    arrays["physical_targets"] = np.asarray(actions)
    np.savez_compressed(output / f"{label}_trace.npz", **arrays)
    if camera_rows:
        dump(output / f"{label}_camera_timestamps.json", camera_rows)
    summary = dict(label=label, snapshot_id=sid, reset=reset_record,
        frozen_launch_head=frozen, h2_placement=asdict(replay.placement),
        source_entry=asdict(replay.entry), requested_ttc_s=requested_ttc,
        clock_rate=replay.rate, extension_center_s=.68 / replay.rate,
        recovery_complete_s=replay.duration_s, first_sampled_complete_s=np.ceil(replay.duration_s/.005)*.005,
        terminal_hold_s=config["terminal_hold_s"], logged_end_s=physics_steps*.005,
        replay_frozen_target_first_entry_s=frozen_first,
        moving_head_first_encounter_s=moving_first,
        first_glove_clearance_violation_s=clearance_first,
        first_mechanical_safety_violation_s=mechanical_first,
        minimum_frozen_center_distance_m=arrays["frozen_distances"].min(),
        minimum_moving_center_distance_m=arrays["moving_distances"].min(),
        maximum_glove_fk_error_m=max_glove_error,
        g1_max_root_displacement_m=np.linalg.norm(arrays["qpos"][:, :3]-arrays["qpos"][0, :3],axis=1).max(),
        g1_min_root_height_m=arrays["qpos"][:,2].min(),
        terminal_glove_distances_m=arrays["moving_distances"][-1],
        diagnostic_continuation=True, safe_arrival_evaluated=False,
        actor_width=96, visual_actor_features=False,
        video_frames=None if media is None else media.count)
    if max_glove_error > 2e-5:
        raise AssertionError(f"Combined-scene glove attachment/FK mismatch: {max_glove_error}")
    if requested_ttc is not None and (frozen_first is None or abs(frozen_first-requested_ttc)>.00501):
        raise AssertionError(f"Requested/replayed frozen-target timing mismatch: {summary}")
    dump(output / f"{label}.json", summary)
    print(f"DONE {label}: frozen={frozen_first}, moving={moving_first}", flush=True)
    return summary, arrays


def compare_without_h2(g1, rows, trace, label):
    """Open-loop paired action replay isolates physical coupling from perception."""
    import torch
    reset = g1.reset(rows)
    q_errors, v_errors = [], []
    with g1.ordered_execution(), torch.no_grad():
        for step in range(len(trace["time_s"])):
            q_errors.append(np.abs(cpu(g1.data.qpos)[0]-trace["qpos"][step]))
            v_errors.append(np.abs(cpu(g1.data.qvel)[0]-trace["qvel"][step]))
            if step == len(trace["time_s"])-1:
                break
            if step % 4 == 0:
                action = torch.as_tensor(trace["physical_targets"][step//4:step//4+1],
                                         device=g1.device, dtype=g1.dtype)
                g1.apply_physical_target(action)
            g1.step_substep()
    q_errors, v_errors = np.asarray(q_errors), np.asarray(v_errors)
    result = dict(label=label, reset=reset, applied_actions="identical saved physical target sequence",
        compared_physics_samples=len(q_errors), max_abs_qpos_difference=float(q_errors.max()),
        max_abs_qvel_difference=float(v_errors.max()),
        max_root_position_difference_m=float(np.linalg.norm(q_errors[:,:3], axis=1).max()),
        final_max_abs_qpos_difference=float(q_errors[-1].max()),
        comparison="Numerical trajectory differences across separately compiled scenes; no bitwise demand")
    print("ISOLATION", json.dumps(result,default=jsonable), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/task10d/phase1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[53874])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT/"outputs/task10d/phase1").resolve()):
        raise ValueError("Phase1 output must remain inside task10d/phase1")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = baseline_env = None
    try:
        import torch
        import warp as wp
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("Phase1 requires one explicitly allocated CUDA GPU")
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
        config = json.loads(args.config.read_text())
        dump(output/"config.json", config)
        h2 = PrescribedH2(PROJECT/config["reference"])
        if h2.sha256 != config["reference_sha256"]:
            raise ValueError("Approved reference hash differs from configuration")
        env, g1 = create_plant(config, output, h2)
        binding = h2.bind(env.sim.mj_model)
        dump(output/"runtime.json", dict(job_id=os.environ["SLURM_JOB_ID"],
            gpu=torch.cuda.get_device_name(0), runtime=g1.runtime_identity,
            compiled_model=g1.model_audit, dimensions={name:getattr(env.sim.mj_model,name)
                for name in ("nq","nv","nu","nbody","ngeom","nmocap")},
            h2_mocap_ids=binding.mocap_ids, h2_glove_geom_ids=binding.glove_geom_ids))
        from .camera import Task08Camera
        camera = Task08Camera(env.sim, binding, seed=config["seed"]+2)
        scenarios, records = [], []
        for sid in args.snapshots:
            rows = load_snapshot_rows([sid])
            for ttc in (None, config["requested_ttc_s"]):
                summary, trace = run_scenario(g1, h2, binding, rows, config, output, ttc,
                    camera=camera if sid==config["nominal_snapshot_id"] else None)
                scenarios.append(summary)
                records.append((rows,summary["label"],trace))
                dump(output/"progress.json", dict(scenarios=scenarios))
        dump(output/"camera.json", camera.metadata())
        baseline_env, baseline = create_plant(config,output)
        isolation = [compare_without_h2(baseline, rows, trace, label) for rows,label,trace in records]
        result = dict(status="completed",job_id=os.environ["SLURM_JOB_ID"],
            application_wall_s=time.monotonic()-started, scenarios=scenarios,
            dynamics_isolation=isolation, reference_sha256=h2.sha256,
            reference_preserved=(hashlib.sha256(h2.path.read_bytes()).hexdigest()==h2.sha256),
            limitations=["Diagnostic full-cycle continuation after first hazard; no safe-arrival result",
                         "Fresh96 nominal controller only; camera stream independently demonstrated",
                         "Small Phase1 sample; no target-bank coverage or TTC-range claim"])
        dump(output/"results.json",result)
        print("PHASE1_COMPLETED",flush=True)
    except Exception:
        dump(output/"failure.json",dict(status="failed",job_id=os.environ.get("SLURM_JOB_ID"),
            application_wall_s=time.monotonic()-started,traceback=traceback.format_exc()))
        raise
    finally:
        if baseline_env is not None:
            baseline_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
