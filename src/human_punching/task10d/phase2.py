"""Explicit family-aware Phase 2 previews using the accepted combined scene.

Only a declared primitive's reference, reset placement and landmarks determine
its prescribed motion. The real snapshot's head is frozen at launch; neither
moving-head feedback nor family-specific G1 calibration is introduced here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import numpy as np

from .g1 import PROJECT, load_snapshot_rows
# These three accepted helpers have no jab-dependent landmarks or placement.
from .phase1 import cpu, create_plant, dump, synchronized_geometry
from .replay import CommittedReplay, Placement, PrescribedH2

FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Primitive:
    record: dict[str, Any]
    reference: PrescribedH2
    placement: Placement
    metadata_sha256: str

    @property
    def primitive_id(self) -> str:
        return self.record["primitive_id"]

    @property
    def family(self) -> str:
        return self.record["family"]

    @property
    def active_hand(self) -> str:
        return self.record["active_hand"]

    @property
    def landmarks(self) -> dict[str, float]:
        """Declared scalar source landmarks; first entry is measured separately."""
        return {key: float(value) for key, value in self.record["annotations"].items()
                if key.endswith("_s") and isinstance(value, (float, int))}


def load_primitive(record: dict[str, Any]) -> Primitive:
    primitive_id = record["primitive_id"]
    if not isinstance(primitive_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", primitive_id):
        raise ValueError("Primitive ID must be a safe explicit filename component")
    if record["family"] not in FAMILIES or record["active_hand"] not in ("left", "right"):
        raise ValueError("Primitive requires a known family and anatomical active hand")
    path = (PROJECT / record["reference_npz"]).resolve(strict=True)
    if not path.is_relative_to(PROJECT.resolve()):
        raise ValueError("Phase 2 references must be inside human-punching")
    reference = PrescribedH2(path)
    if reference.sha256 != record["reference_sha256"]:
        raise ValueError(f"Reference hash mismatch: {primitive_id}")
    for key in ("family", "active_hand"):
        if reference.reference.metadata[key] != record[key]:
            raise ValueError(f"Manifest/reference {key} mismatch: {primitive_id}")
    placement_record = record["placement"]
    if set(placement_record) != {"xy", "yaw"}:
        raise ValueError("Reset placement must explicitly contain only xy and yaw; no vertical shift")
    placement = Placement(tuple(placement_record["xy"]), placement_record["yaw"])
    target = record["target"]
    if target["snapshot_id"] != 53874:
        raise ValueError("Phase 2 representative previews use only TRAIN snapshot 53874")
    head = np.asarray(target["head_world"], dtype=float)
    if head.shape != (3,) or not np.isfinite(head).all():
        raise ValueError("Primitive target requires the real finite head XYZ")
    annotations = record["annotations"]
    if "extension_center_s" not in annotations:
        raise ValueError("Each family needs its own explicit extension_center_s annotation")
    metadata_sha256 = digest(path.with_suffix(".json"))
    if record.get("reference_json_sha256", metadata_sha256) != metadata_sha256:
        raise ValueError(f"Reference JSON hash mismatch: {primitive_id}")
    primitive = Primitive(record, reference, placement, metadata_sha256)
    for key, value in primitive.landmarks.items():
        if not np.isfinite(value) or not 0 <= value <= reference.end_time_s:
            raise ValueError(f"Invalid source landmark {key}: {value}")
    if "extension_center_s" not in primitive.landmarks:
        raise ValueError("extension_center_s must be a numeric source time")
    return primitive


def load_library(path: Path, primitive_ids: list[str] | None = None) -> list[Primitive]:
    payload = json.loads(path.read_text())
    records = payload["primitives"]
    identities = [record["primitive_id"] for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Library primitive IDs must be unique")
    if primitive_ids is None:
        records = [record for record in records if record.get("role") == "nominal"]
        if sorted(record["family"] for record in records) != sorted(FAMILIES):
            raise ValueError("Default Phase 2 preview requires exactly one nominal per family")
    else:
        if len(set(primitive_ids)) != len(primitive_ids) or not set(primitive_ids) <= set(identities):
            raise ValueError("Requested primitive IDs must be distinct existing library entries")
        records = [records[identities.index(identity)] for identity in primitive_ids]
    return [load_primitive(record) for record in records]


def validate_config(config: dict[str, Any]) -> None:
    expected = {"nominal_snapshot_id": 53874, "requested_ttc_s": .7,
                "physics_dt_s": .005, "control_dt_s": .02,
                "glove_radius_m": .08, "head_radius_m": .06,
                "encounter_distance_m": .14, "safe_clearance_m": .16,
                "nominal_observation_noise": True, "visual_noise": True,
                "visual_delay_ticks": [2, 3, 4]}
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"Phase 2 accepted preview contract changed: {key}")
    if not np.isfinite(config["terminal_hold_s"]) or config["terminal_hold_s"] < .4:
        raise ValueError("Complete-cycle preview requires at least 0.4 s terminal hold")


def landmark_frames(primitive: Primitive, replay: CommittedReplay) -> set[int]:
    landmarks = [0., *primitive.landmarks.values(), primitive.reference.end_time_s]
    return set(np.rint(np.asarray(landmarks) / replay.rate / .02).astype(int))


def run_scenario(g1, primitive, binding, rows, config, output, requested_ttc, camera):
    import torch
    from .media import DiagnosticMedia
    label = primitive.primitive_id + ("_source" if requested_ttc is None else "_ttc0700")
    print(f"BEGIN {label}", flush=True)
    reset_record = g1.reset(rows)
    with g1.ordered_execution():
        initial, _ = synchronized_geometry(g1, binding,
            primitive.reference.sample(0., primitive.placement))
        frozen = cpu(initial["head_position"])[0]
    np.testing.assert_allclose(frozen, primitive.record["target"]["head_world"], atol=5e-6, rtol=0.)
    replay = CommittedReplay(primitive.reference, tuple(frozen),
                             primitive.placement, requested_ttc)
    if replay.entry.interpolated_s is None or replay.entry.launch_overlap:
        raise ValueError(f"Selected primitive requires a future frozen-target entry: {label}")
    physics_steps = 4 * int(np.ceil((replay.duration_s + config["terminal_hold_s"]) / .02))
    keys = ("time_s", "source_s", "g1_root_position", "head", "gloves",
            "frozen_distances", "moving_distances", "mechanically_safe", "stance_energy")
    trace = {key: [] for key in keys}
    camera_rows = []
    first_frozen = [None, None]
    first_moving = [None, None]
    mechanical_first = clearance_first = None
    max_glove_error = 0.
    keyframes = landmark_frames(primitive, replay)
    media = DiagnosticMedia(output / label, camera.renderer.host_model, label)
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
                for hand in range(2):
                    if first_frozen[hand] is None and frozen_distances[hand] <= .14:
                        first_frozen[hand] = elapsed
                    if first_moving[hand] is None and moving_distances[hand] <= .14:
                        first_moving[hand] = elapsed
                if mechanical_first is None and not mechanically_safe:
                    mechanical_first = elapsed
                if clearance_first is None and np.any(moving_distances < .16):
                    clearance_first = elapsed
                row = dict(time_s=elapsed, source_s=sample.source_time_s,
                    g1_root_position=cpu(g1.data.qpos)[0, :3], head=head, gloves=gloves,
                    frozen_distances=frozen_distances, moving_distances=moving_distances,
                    mechanically_safe=mechanically_safe,
                    stance_energy=float(cpu(geometry["stance_energy"])[0]))
                for key, value in row.items():
                    trace[key].append(value)
                if step % 4 == 0:
                    exposure = camera.capture(elapsed, reset=(step == 0))
                    camera_rows.append({name: value for name, value in exposure.items()
                        if name != "rgb" and ("timestamp" in name or "tick" in name
                        or "padding" in name or "pixel" in name or name == "visual_age_s")})
                    media.append(g1.data, exposure, time_s=elapsed,
                        source_s=sample.source_time_s,
                        hazard=f"glove={clearance_first}, mechanical={mechanical_first}",
                        keyframe=(step // 4 in keyframes or step == physics_steps))
                    if step < physics_steps:
                        g1.apply_physical_target(g1.control_action())
                if step < physics_steps:
                    binding.write(replay.sample((step + 1) * .005), g1.data)
                    g1.step_substep()
            if step % 100 == 0:
                print(f"{label} {step}/{physics_steps} t={elapsed:.3f}", flush=True)
    finally:
        media.close()
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    np.savez_compressed(output / f"{label}_trace.npz", **arrays)
    dump(output / f"{label}_camera_timestamps.json", camera_rows)
    frozen_first = min((value for value in first_frozen if value is not None), default=None)
    moving_first = min((value for value in first_moving if value is not None), default=None)
    summary = dict(label=label, primitive_id=primitive.primitive_id,
        family=primitive.family, active_hand=primitive.active_hand,
        stance=primitive.record.get("stance"), snapshot_id=53874, reset=reset_record,
        reference_npz=str(primitive.reference.path), reference_sha256=primitive.reference.sha256,
        reference_json_sha256=primitive.metadata_sha256, frozen_launch_head=frozen,
        h2_runtime_placement=asdict(replay.placement),
        source_to_world_placement=primitive.record.get("source_to_world_placement"),
        source_annotations=primitive.record["annotations"],
        source_entry=asdict(replay.entry), requested_ttc_s=requested_ttc,
        clock_rate=replay.rate,
        episode_landmarks_s={key: value/replay.rate for key,value in primitive.landmarks.items()},
        extension_center_s=primitive.landmarks["extension_center_s"]/replay.rate,
        recovery_complete_s=replay.duration_s,
        first_sampled_complete_s=np.ceil(replay.duration_s/.005)*.005,
        terminal_hold_s=config["terminal_hold_s"], logged_end_s=physics_steps*.005,
        replay_frozen_target_first_entry_s=frozen_first,
        replay_frozen_target_first_entry_by_hand_s=dict(zip(("left","right"), first_frozen)),
        moving_head_first_encounter_s=moving_first,
        moving_head_first_encounter_by_hand_s=dict(zip(("left","right"), first_moving)),
        first_glove_clearance_violation_s=clearance_first,
        first_mechanical_safety_violation_s=mechanical_first,
        minimum_frozen_center_distance_by_hand_m=arrays["frozen_distances"].min(axis=0),
        minimum_moving_center_distance_by_hand_m=arrays["moving_distances"].min(axis=0),
        maximum_glove_fk_error_m=max_glove_error,
        g1_max_root_displacement_m=np.linalg.norm(arrays["g1_root_position"]-arrays["g1_root_position"][0],axis=1).max(),
        g1_min_root_height_m=arrays["g1_root_position"][:,2].min(),
        terminal_glove_distances_m=arrays["moving_distances"][-1],
        terminal_both_gloves_clear=bool(np.all(arrays["moving_distances"][-1] >= .16)),
        diagnostic_continuation=True, safe_arrival_evaluated=False,
        actor_width=96, visual_actor_features=False, video_frames=media.count,
        event_resolution_s=.005)
    dump(output / f"{label}.json", summary)
    if max_glove_error > 2e-5:
        raise AssertionError(f"Combined-scene glove attachment/FK mismatch: {max_glove_error}")
    if requested_ttc is not None and (frozen_first is None or abs(frozen_first-requested_ttc)>.00501):
        raise AssertionError(f"Requested/replayed frozen-target timing mismatch: {summary}")
    print(f"DONE {label}: frozen={frozen_first}, moving={moving_first}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/task10d/phase2.json")
    parser.add_argument("--library", type=Path,
        default=PROJECT/"outputs/task10d/phase2/nominal/nominal_manifest.json")
    parser.add_argument("--primitive-ids", nargs="+")
    parser.add_argument("--clocks", nargs="+", choices=("source", "ttc0700"),
                        default=["source", "ttc0700"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT/"outputs/task10d/phase2").resolve()):
        raise ValueError("Phase 2 output must remain inside task10d/phase2")
    output.mkdir(parents=True, exist_ok=False)
    started, env = time.monotonic(), None
    try:
        import torch
        import warp as wp
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("Phase 2 combined previews require one explicitly allocated CUDA GPU")
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
        config = json.loads(args.config.read_text())
        validate_config(config)
        primitives = load_library(args.library, args.primitive_ids)
        dump(output/"config.json", config)
        dump(output/"selected_primitives.json", dict(library=str(args.library.resolve()),
            library_sha256=digest(args.library), clocks=args.clocks,
            primitives=[primitive.record for primitive in primitives]))
        dump(output/"invocation.json", dict(argv=sys.argv, cwd=str(Path.cwd()),
            config_path=str(args.config.resolve()), config_sha256=digest(args.config)))
        code_paths = [Path(__file__).with_name(name + ".py")
                      for name in ("phase2", "phase1", "replay", "g1", "camera", "media")]
        dump(output/"source_identity.json", {str(path.relative_to(PROJECT)): digest(path) for path in code_paths})
        env, g1 = create_plant(config, output, primitives[0].reference)
        binding = primitives[0].reference.bind(env.sim.mj_model)
        dump(output/"runtime.json", dict(job_id=os.environ["SLURM_JOB_ID"],
            gpu=torch.cuda.get_device_name(0), runtime=g1.runtime_identity,
            compiled_model=g1.model_audit,
            dimensions={name:getattr(env.sim.mj_model,name) for name in ("nq","nv","nu","nbody","ngeom","nmocap")},
            h2_mocap_ids=binding.mocap_ids, h2_glove_geom_ids=binding.glove_geom_ids))
        from .camera import Task08Camera
        camera = Task08Camera(env.sim, binding, seed=config["seed"]+2)
        # Explicitly retain the accepted nominal target's calibration for every
        # family. Loading a family's old calibration would move G1/the target.
        rows = load_snapshot_rows([53874], family="jab")
        scenarios = []
        for primitive in primitives:
            for clock in args.clocks:
                requested_ttc = None if clock == "source" else config["requested_ttc_s"]
                scenarios.append(run_scenario(g1, primitive, binding, rows, config,
                                               output, requested_ttc, camera))
                dump(output/"progress.json", dict(scenarios=scenarios))
        dump(output/"camera.json", camera.metadata())
        preserved = {primitive.primitive_id:
            digest(primitive.reference.path)==primitive.reference.sha256 and
            digest(primitive.reference.path.with_suffix(".json"))==primitive.metadata_sha256
            for primitive in primitives}
        if not all(preserved.values()):
            raise AssertionError("Reference NPZ/JSON changed during preview")
        dump(output/"results.json", dict(status="completed", job_id=os.environ["SLURM_JOB_ID"],
            application_wall_s=time.monotonic()-started, scenarios=scenarios,
            reference_pairs_preserved=preserved,
            limitations=["Diagnostic continuation after first hazard; no safe-arrival result",
                "Fresh96 is proprioceptive; camera preview does not establish avoidance",
                "Camera mount/FOV/range/delay/occlusion unchanged; angle revision deferred to Task 11",
                "One real nominal target and two basic clocks, no target-bank/TTC-range claim",
                "Each primitive is an isolated reset scenario, not a live selectable combination",
                "Accepted Phase 1 dynamics isolation reused; no repeated paired qualification"]))
        print("PHASE2_PREVIEWS_COMPLETED", flush=True)
    except Exception:
        dump(output/"failure.json", dict(status="failed", job_id=os.environ.get("SLURM_JOB_ID"),
            application_wall_s=time.monotonic()-started, traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
