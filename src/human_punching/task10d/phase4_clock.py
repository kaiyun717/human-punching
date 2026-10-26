"""Bounded fixed-landmark clock check in the accepted 5 ms combined simulator.

This checks prescribed whole-motion playback, not TTC, hitting or physical H2
tracking. References, fixture, calibration and the Phase 3C bank stay unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .g1 import PROJECT
from .phase1 import cpu, dump, synchronized_geometry
from .phase3_render import assert_unchanged, create_batch, state_copy
from .placement_phase3 import INDEX, PRIMITIVE_IDS, cache_library
from .standing_bank import FixedStrikeClock, candidate_placement, choice_record
from .standing_fixture import StandingFixture, launch_geometry

BUNDLE = PROJECT.parent / "robust_sav/constants/boxing/kinematic_attack_bank/v1_20261025"
DT = .005
CONTROL_DT = .020
HOLD_S = .100


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_trials(bank, landmarks):
    """Fixed commands and existing placements, selected before dynamic outcomes."""
    trials = []
    for landmark in landmarks:
        for command in (.50, .80):
            pid = landmark["primitive_id"]
            trials.append(dict(primitive_id=pid, T_strike_s=command,
                s_j_star=landmark["s_j_star"], choice=landmark["calibration_choice"],
                placement_kind="stored_calibration",
                label=f"{pid}_T{round(command*1000):04d}",
                save_media=pid in ("jab_nominal", "left_hook_013_03243")))
    cross = next(row for row in landmarks if row["primitive_id"] == "cross_nominal")
    trials.append(dict(primitive_id="cross_nominal", T_strike_s=.40,
        s_j_star=cross["s_j_star"], choice=cross["calibration_choice"],
        placement_kind="stored_calibration", label="cross_nominal_T0400", save_media=False))
    # Different placement, same saved landmark; no contact time is recalculated.
    p = list(bank["primitive_ids"]).index("jab_nominal")
    candidates = np.argwhere(bank["retained_aimed"][p] & bank["primary_xy"][:, None])
    x, y = min(candidates, key=lambda pair: (
        float(np.linalg.norm(bank["xy_G"][pair[0]] - [1.10, 0.])),
        abs(float((bank["yaw_deg"][pair[1]] - 165. + 180.) % 360. - 180.)),
        int(pair[0]), int(pair[1])))
    jab = next(row for row in landmarks if row["primitive_id"] == "jab_nominal")
    choice = choice_record(bank, p, int(x), int(y))
    assert choice["xy_index"] != jab["calibration_choice"]["xy_index"]
    trials.append(dict(primitive_id="jab_nominal", T_strike_s=.653,
        s_j_star=jab["s_j_star"], choice=choice, placement_kind="alternate_tested_aimed",
        label="jab_nominal_T0653_alternate", save_media=False))
    return trials


def quaternion_error(actual, expected):
    """Sign-invariant component error avoids unstable acos near identity."""
    return float(np.minimum(np.max(np.abs(actual-expected), axis=-1),
                            np.max(np.abs(actual+expected), axis=-1)).max())


def run_trial(g1, binding, camera, path, fixture, trial, output):
    import torch
    from .media import DiagnosticMedia

    reference = path.primitive.reference
    clock = FixedStrikeClock(trial["s_j_star"], trial["T_strike_s"], reference.end_time_s)
    duration = reference.end_time_s / clock.alpha
    steps = 4 * int(np.ceil((duration + HOLD_S) / CONTROL_DT))
    reset = fixture.restore(g1, camera_id=camera.ids.camera_id)
    before = state_copy(g1)
    with g1.ordered_execution(), torch.no_grad():
        launch = launch_geometry(g1, camera.ids.camera_id)
    choice = trial["choice"]
    placement = candidate_placement(path, launch, choice["xy_G_m"], choice["yaw_deg"])
    assert_unchanged(g1, before)
    destination = output / trial["label"]
    title = f"{trial['primitive_id']} T_strike={clock.T_strike:.3f}s rate={clock.alpha:.6f}"
    media = DiagnosticMedia(destination, camera.renderer.host_model, title) if trial["save_media"] else None
    if media is None:
        destination.mkdir(exist_ok=False)
    trace = {k: [] for k in ("elapsed_s", "native_time_s", "source_s", "body_pos", "body_quat",
        "gloves", "head", "body_error_m", "quaternion_error", "glove_error_m",
        "post_step_body_error_m", "finite", "mechanically_safe", "moving_min_distance_m")}
    control_steps = []
    first_hazard = None
    pending_body_error = 0.
    try:
        for step in range(steps + 1):
            elapsed = step * DT
            sample = reference.sample(clock.source_time(elapsed), placement)
            with g1.ordered_execution(), torch.no_grad():
                geometry, gloves = synchronized_geometry(g1, binding, sample)
                if step == 0:
                    assert_unchanged(g1, before)
                body_pos = cpu(g1.data.xpos)[0, binding.body_ids]
                body_quat = cpu(g1.data.xquat)[0, binding.body_ids]
                head = cpu(geometry["head_position"])[0]
                moving_min = float(np.linalg.norm(gloves - head, axis=-1).min())
                if first_hazard is None and moving_min < .16:
                    first_hazard = elapsed
                values = dict(elapsed_s=elapsed, native_time_s=float(cpu(g1.data.time)[0]),
                    source_s=sample.source_time_s, body_pos=body_pos, body_quat=body_quat,
                    gloves=gloves, head=head,
                    body_error_m=float(np.max(np.abs(body_pos-sample.body_pos))),
                    quaternion_error=quaternion_error(body_quat, sample.body_quat),
                    glove_error_m=float(np.max(np.abs(gloves-sample.glove_pos))),
                    post_step_body_error_m=pending_body_error,
                    finite=bool(cpu(geometry["finite"])[0]),
                    mechanically_safe=bool(cpu(geometry["mechanically_safe"])[0]),
                    moving_min_distance_m=moving_min)
                for key, value in values.items():
                    trace[key].append(value)
                if not values["finite"]:
                    raise RuntimeError("Nonfinite combined scene; clock run incomplete")
                if step % 4 == 0:
                    if media is not None:
                        exposure = camera.capture(elapsed, reset=step == 0)
                        if step == 0:
                            np.testing.assert_array_equal(exposure["padding"], True)
                            np.testing.assert_array_equal(exposure["capture_timestamps"], 0)
                            assert_unchanged(g1, before)
                        media.append(g1.data, exposure, time_s=elapsed, source_s=sample.source_time_s,
                            hazard=first_hazard, keyframe=step in (0, steps)
                                or abs(elapsed-clock.T_strike) < .009999)
                    if step < steps:
                        g1.apply_physical_target(g1.control_action())
                        control_steps.append(step)
                if step < steps:
                    next_sample = reference.sample(clock.source_time((step+1)*DT), placement)
                    binding.write(next_sample, g1.data)
                    g1.step_substep()
                    # Read after real integration, before the next loop's FK refresh.
                    pending_body_error = float(np.max(np.abs(
                        cpu(g1.data.xpos)[0, binding.body_ids] - next_sample.body_pos)))
    finally:
        if media is not None:
            media.close()
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays["control_step_indices"] = np.array(control_steps, dtype=np.int32)
    np.savez_compressed(destination / "trace.npz", **arrays)
    t, source, native = (arrays[k] for k in ("elapsed_s", "source_s", "native_time_s"))
    first_landmark_index = int(np.flatnonzero(source >= clock.s_j_star - 1e-12)[0])
    first_end_index = int(np.flatnonzero(source >= reference.end_time_s - 1e-12)[0])
    landmark_sample = float(t[first_landmark_index])
    landmark_error = landmark_sample - clock.T_strike
    expected_source = np.minimum(t * (trial["s_j_star"] / trial["T_strike_s"]), reference.end_time_s)
    source_error = float(np.abs(source - expected_source).max())
    native_error = float(np.abs(native-t).max())
    native_step_error = float(np.abs(np.diff(native)-DT).max())
    hold_pos_error = float(np.max(np.abs(arrays["body_pos"][first_end_index:] - arrays["body_pos"][first_end_index])))
    hold_quat_error = float(np.max(np.abs(arrays["body_quat"][first_end_index:] - arrays["body_quat"][first_end_index])))
    hold_glove_error = float(np.max(np.abs(arrays["gloves"][first_end_index:] - arrays["gloves"][first_end_index])))
    checks = dict(
        clock_at_requested_time=abs(clock.source_time(clock.T_strike)-clock.s_j_star) < 1e-12,
        source_schedule=source_error < 1e-12,
        source_monotone=bool(np.all(np.diff(source) >= 0)),
        landmark_quantization=-1e-12 <= landmark_error < DT + 1e-12,
        actual_physics_cadence=native_step_error < 2e-6 and native_error < 1e-4,
        control_decimation=bool(np.array_equal(arrays["control_step_indices"], np.arange(0, steps, 4))),
        full_body_native_pose=float(arrays["body_error_m"].max()) < 2e-5,
        full_body_native_orientation=float(arrays["quaternion_error"].max()) < 2e-5,
        integrated_endpoint_pose=float(arrays["post_step_body_error_m"].max()) < 2e-5,
        attached_glove_native_pose=float(arrays["glove_error_m"].max()) < 2e-5,
        scaled_completion=0 <= t[first_end_index]-duration < DT + 1e-12,
        complete_hold=float(t[-1]-duration) >= HOLD_S-1e-12,
        terminal_hold_pose=max(hold_pos_error, hold_quat_error, hold_glove_error) == 0.,
        finite=bool(arrays["finite"].all()),
        full_media=media is None or media.count == steps//4+1)
    summary = dict(**trial, fixture_id=fixture.metadata["fixture_id"],
        reference_sha256=reference.sha256, reset=reset, placement=asdict(placement),
        clock="fixed source landmark; no per-placement first-entry recalibration",
        alpha=clock.alpha, source_end_s=reference.end_time_s, duration_s=duration,
        sampled_landmark_s=landmark_sample, landmark_quantization_error_s=landmark_error,
        sampled_completion_s=float(t[first_end_index]), simulated_duration_s=float(t[-1]),
        terminal_hold_s=float(t[-1]-duration), physics_steps=steps,
        control_updates=len(control_steps), physics_dt_s=DT, control_dt_s=CONTROL_DT,
        h2_body_count=len(binding.body_ids), maximum_source_error_s=source_error,
        maximum_native_time_error_s=native_error, maximum_native_step_error_s=native_step_error,
        maximum_body_error_m=float(arrays["body_error_m"].max()),
        maximum_quaternion_error=float(arrays["quaternion_error"].max()),
        maximum_glove_error_m=float(arrays["glove_error_m"].max()),
        maximum_post_step_body_error_m=float(arrays["post_step_body_error_m"].max()),
        hold_body_error_m=hold_pos_error, hold_quaternion_error=hold_quat_error,
        hold_glove_error_m=hold_glove_error, video_frames=0 if media is None else media.count,
        mechanically_safe_all_samples=bool(arrays["mechanically_safe"].all()),
        first_clearance_violation_s=first_hazard,
        moving_min_distance_m=float(arrays["moving_min_distance_m"].min()),
        encounter_is_not_a_pass_condition=True, no_arrival_evaluation=True,
        controller="frozen Fresh96FlatPolicy/v1_20261007", avoidance_residual=0.,
        observation_noise_enabled=bool(g1.observation_noise_enabled), checks=checks,
        passed=all(checks.values()))
    dump(destination / "summary.json", summary)
    print(f"CLOCK {trial['label']}: rate={clock.alpha:.6f} landmark={landmark_sample:.3f} "
          f"end={duration:.6f} pass={summary['passed']}", flush=True)
    if not summary["passed"]:
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--describe", action="store_true", help="Read-only commands/landmarks check without simulation")
    args = parser.parse_args()
    fixture = StandingFixture.load(BUNDLE / "fixture")
    bank = dict(np.load(BUNDLE / "bank.npz", allow_pickle=False))
    landmarks = json.loads((BUNDLE / "fixed_landmarks.json").read_text())["landmarks"]
    assert list(bank["primitive_ids"]) == list(PRIMITIVE_IDS)
    assert [row["primitive_id"] for row in landmarks] == list(PRIMITIVE_IDS)
    assert str(bank["fixture_id"]) == fixture.metadata["fixture_id"]
    trials = make_trials(bank, landmarks)
    if args.describe:
        print(json.dumps(trials, indent=2))
        return
    import torch
    import warp as wp
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("One allocated CUDA GPU is required")
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase4").resolve()):
        raise ValueError("Use a new Phase 4 output directory inside human-punching")
    output.mkdir(parents=True, exist_ok=False)
    paths = cache_library()  # Explicit accepted eight IDs, unchanged native source FK.
    for p, path in enumerate(paths):
        assert path.primitive.reference.sha256 == str(bank["reference_sha256"][p])
    files = list(Path(__file__).parent.glob("*.py")) + [INDEX, PROJECT / "scripts/task10d_env.sh",
        PROJECT / "scripts/task10d_standing.sbatch", PROJECT / "scripts/task10c_env.sh"]
    files += [p for p in BUNDLE.rglob("*") if p.is_file()]
    files += [file for path in paths for file in (path.primitive.reference.path, path.primitive.reference.path.with_suffix(".json"))]
    identity = {str(path): digest(path) for path in files}
    started = time.monotonic()
    dump(output / "invocation.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        source_identity=identity, bundle=str(BUNDLE), fixture_id=fixture.metadata["fixture_id"],
        trials=trials, scope="Quick Phase4 clock check only; no Phase3D, TTC/hit requirement or training",
        physics_dt_s=DT, control_dt_s=CONTROL_DT, terminal_hold_s=HOLD_S,
        pdf_sha256="5cb252daf7b303fc47af176098cab085707a5c5c3d4d179e45855fcc05362cbc"))
    env, g1, binding, camera = create_batch(dict(seed=102503), output, paths[0].primitive.reference, 25., 1)
    results = []
    try:
        dump(output / "runtime.json", dict(runtime=g1.runtime_identity, model=g1.model_audit,
            camera=camera.metadata()))
        by_id = {path.primitive.primitive_id: path for path in paths}
        for trial in trials:
            results.append(run_trial(g1, binding, camera, by_id[trial["primitive_id"]], fixture, trial, output))
            dump(output / "progress.json", dict(completed=len(results), total=len(trials), latest=trial["label"]))
    finally:
        env.close()
    unchanged = all(digest(Path(path)) == value for path, value in identity.items())
    summary = dict(passed=all(row["passed"] for row in results) and unchanged,
        trial_count=len(results), job_id=os.environ["SLURM_JOB_ID"],
        elapsed_wall_s=time.monotonic()-started, source_and_inputs_unchanged=unchanged,
        rates=[min(row["alpha"] for row in results), max(row["alpha"] for row in results)],
        maximum_landmark_quantization_error_s=max(row["landmark_quantization_error_s"] for row in results),
        maximum_body_error_m=max(row["maximum_body_error_m"] for row in results),
        maximum_quaternion_error=max(row["maximum_quaternion_error"] for row in results),
        maximum_glove_error_m=max(row["maximum_glove_error_m"] for row in results),
        maximum_native_time_error_s=max(row["maximum_native_time_error_s"] for row in results),
        total_physics_steps=sum(row["physics_steps"] for row in results),
        total_control_updates=sum(row["control_updates"] for row in results),
        complete_video_pairs=sum(bool(row["save_media"]) for row in results), results=results,
        interpretation="Whole-motion fixed-landmark clock works on tested cases; no TTC or moving-head hit guarantee.")
    dump(output / "summary.json", summary)
    if not summary["passed"]:
        raise AssertionError("Clock trial or preservation check failed")
    print(f"COMPLETE {len(results)} clock checks, {summary['complete_video_pairs']} complete video pairs", flush=True)


if __name__ == "__main__":
    main()
