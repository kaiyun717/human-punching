"""One standing Fresh96 launch fixture, selected before attack inspection.

The capture time is fixed at 2 s of normal initialization with ordinary shared
proprioceptive noise. No snapshot bank is read to construct this fixture. Restore
uses the accepted local reset and forward interfaces, then restores the captured
controller context. Camera history starts only when a committed attack is shown.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .g1 import PROJECT, RESET_KEYS, execution_dependencies
from .phase1 import cpu, dump

SCHEMA = "task10d_standing_fixture_v1"
CAPTURE_SECONDS = 2.0
FIXTURE_ID = "fresh96_standing_seed102503_t2000ms"
EXTRA_DATA_FIELDS = ("qacc", "qacc_warmstart", "ctrl", "qfrc_applied",
                     "xfrc_applied", "act", "history")


def launch_geometry(g1, camera_id: int) -> dict[str, np.ndarray]:
    """Read synchronized native geometry without integrating or drawing noise."""
    row = g1.geometry()
    root = g1.root_body_id
    rotation = cpu(g1.data.xmat[:, root])[0].reshape(3, 3)
    optical = cpu(g1.data.cam_xmat[:, camera_id])[0].reshape(3, 3)
    optical[:, 1:] *= -1  # Native camera -Z forward -> optical +Z forward.
    return dict(head_world=cpu(row["head_position"])[0],
        head_velocity_world=cpu(row["head_velocity"])[0],
        pelvis_world=cpu(g1.data.xpos[:, root])[0], pelvis_rotation=rotation,
        pelvis_heading_rad=np.asarray(np.arctan2(rotation[1, 0], rotation[0, 0])),
        torso_world=cpu(g1.data.xpos[:, g1.torso_body_id])[0],
        torso_rotation=cpu(g1.data.xmat[:, g1.torso_body_id])[0].reshape(3, 3),
        camera_position_world=cpu(g1.data.cam_xpos[:, camera_id])[0],
        camera_rotation=optical, baseline_camera_rotation=optical.copy())


def _capture_arrays(g1, camera_id: int) -> dict[str, np.ndarray]:
    manager = g1.env.action_manager
    result = dict(zip(RESET_KEYS, map(cpu, (g1.data.qpos, g1.data.qvel,
        manager.action, manager.prev_action, manager.prev_prev_action)), strict=True))
    for name in EXTRA_DATA_FIELDS:
        if hasattr(g1.data, name):
            result[f"sim/data/{name}"] = cpu(getattr(g1.data, name))
    for name in ("clean99", "clean111", "actor99"):
        result[f"measurement/{name}"] = cpu(getattr(g1, name))
    result["noise_generator_state"] = cpu(g1.noise_generator.get_state())
    result["capture_native_time"] = cpu(g1.data.time)
    result.update({f"geometry/{key}": value for key, value in launch_geometry(g1, camera_id).items()})
    return result


@dataclass(frozen=True)
class StandingFixture:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "StandingFixture":
        path = Path(path).resolve(strict=True)
        if path.is_dir():
            path = path / "fixture.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["schema"] != SCHEMA:
            raise ValueError("Unsupported standing fixture schema")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["payload_sha256"]:
            raise ValueError("Standing fixture payload identity mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        if arrays[RESET_KEYS[0]].shape != (1, 36) or arrays[RESET_KEYS[1]].shape != (1, 35):
            raise ValueError("Expected exactly one untransformed G1 standing state")
        return cls(arrays, metadata)

    @property
    def geometry(self) -> dict[str, np.ndarray]:
        return {key.removeprefix("geometry/"): value.copy()
                for key, value in self.arrays.items() if key.startswith("geometry/")}

    def rows(self, count: int = 1) -> dict:
        """Compatibility view for old reset callers; use restore for full context."""
        if count < 1:
            raise ValueError("Positive fixture replication count required")
        return dict(snapshot_ids=np.full(count, -1, dtype=np.int64),
            fixture_id=self.metadata["fixture_id"], transforms=None,
            arrays={key: np.repeat(self.arrays[key], count, axis=0) for key in RESET_KEYS})

    def restore(self, g1, *, camera_id: int | None = None) -> dict:
        """Restore captured state and shared measurement at a new t=0; no step.

        Noise-generator continuation is exact for a single world. Replicated
        worlds start from identical physical/controller state but consume a
        batched noise stream; batched dynamics are not claimed to be identical.
        """
        import torch
        from boxing.training.plant import tensor_view
        result = g1.reset(self.rows(g1.num_envs))
        with g1.ordered_execution(), torch.no_grad():
            for name in EXTRA_DATA_FIELDS:
                key = f"sim/data/{name}"
                if key not in self.arrays:
                    continue
                target = tensor_view(getattr(g1.data, name))
                value = np.repeat(self.arrays[key], g1.num_envs, axis=0)
                target.copy_(torch.as_tensor(value, device=g1.device, dtype=target.dtype))
            g1.refresh()
            # Forward may alter these solver working values. Keep the captured
            # warmstart for the first genuine attack substep after FK refresh.
            for name in ("qacc", "qacc_warmstart"):
                key = f"sim/data/{name}"
                target = tensor_view(getattr(g1.data, name))
                target.copy_(torch.as_tensor(np.repeat(self.arrays[key], g1.num_envs, axis=0),
                                             device=g1.device, dtype=target.dtype))
            for name in ("clean99", "clean111", "actor99"):
                value = np.repeat(self.arrays[f"measurement/{name}"], g1.num_envs, axis=0)
                setattr(g1, name, torch.as_tensor(value, device=g1.device, dtype=g1.dtype))
            g1.noise_generator.set_state(torch.as_tensor(self.arrays["noise_generator_state"],
                                                        dtype=torch.uint8, device="cpu"))
            g1.command.freeze_zero()
            if np.any(cpu(g1.data.time) != 0):
                raise AssertionError("Fixture reset advanced simulation time")
            if camera_id is not None:
                actual = launch_geometry(g1, camera_id)
                error = {key: float(np.max(np.abs(actual[key]-expected)))
                         for key, expected in self.geometry.items()}
                if max(error.values()) > 2e-6:
                    raise AssertionError(f"Fixture native geometry changed: {error}")
                result["geometry_max_abs_error"] = error
        result.update(fixture_id=self.metadata["fixture_id"], capture_time_s=CAPTURE_SECONDS,
            restored_solver_and_measurement_context=True, camera_history_restored=False,
            launch_clock_s=0.0, physics_steps=0)
        return result


def capture_fixture(g1, camera_id: int, output: Path | str, *, preview: bool = True) -> StandingFixture:
    """Capture the predeclared 2 s standing state; never examine attack outcomes."""
    import torch
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT.resolve()):
        raise ValueError("Fixture products must stay inside human-punching")
    if g1.num_envs != 1:
        raise ValueError("Standing construction uses exactly one dynamic world")
    output.mkdir(parents=True, exist_ok=False)
    with g1.ordered_execution(), torch.no_grad():
        g1.command.freeze_zero()
        g1.env.reset(seed=102503)
        g1.robot.data.encoder_bias.zero_()
        g1.noise_generator.manual_seed(g1.noise_seed)
        g1.refresh()
        g1.refresh_measurement()
    writer = renderer = None
    if preview:
        import imageio.v2 as imageio
        from .media import ExternalRenderer
        renderer = ExternalRenderer(g1.model)
        # This renderer owns a model copy. Hide H2 only in the standing-only
        # preview; no simulation model, opponent geometry or camera is changed.
        for index in range(renderer.model.ngeom):
            name = renderer.model.body(int(renderer.model.geom_bodyid[index])).name
            if name.startswith("h2/"):
                renderer.model.geom_group[index] = 3
        writer = imageio.get_writer(output / "standing_preview.mp4", fps=50,
            codec="libx264", quality=7, macro_block_size=1,
            ffmpeg_params=["-threads", "1"])
    traces = {key: [] for key in ("time_s", "qpos", "qvel", "head_world",
                                  "head_velocity_world", "mechanically_safe")}
    try:
        for step in range(401):
            with g1.ordered_execution(), torch.no_grad():
                geo = g1.geometry()
                for key, value in dict(time_s=step * .005,
                    qpos=cpu(g1.data.qpos)[0], qvel=cpu(g1.data.qvel)[0],
                    head_world=cpu(geo["head_position"])[0],
                    head_velocity_world=cpu(geo["head_velocity"])[0],
                    mechanically_safe=bool(cpu(geo["mechanically_safe"])[0])).items():
                    traces[key].append(value)
                if step % 4 == 0:
                    if renderer is not None:
                        views = renderer.render(g1.data, step * .005)
                        writer.append_data(np.concatenate(views, axis=1))
                    if step < 400:
                        g1.apply_physical_target(g1.control_action())
                if step < 400:
                    g1.step_substep()
        with g1.ordered_execution():
            arrays = _capture_arrays(g1, camera_id)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    trace = {key: np.asarray(value) for key, value in traces.items()}
    np.savez_compressed(output / "standing_trace.npz", **trace)
    np.savez_compressed(output / "fixture.npz", **arrays)
    last = trace["time_s"] >= 1.5
    selected = execution_dependencies()["model_action_fingerprint"]
    metadata = dict(schema=SCHEMA, fixture_id=FIXTURE_ID,
        selection="Fixed capture at 2.000 s declared before any attack search; no outcome selection",
        initialization="Accepted make_nominal_cfg normal reset_scene_to_default; no snapshot restore",
        environment_seed=102503, noise_seed=g1.noise_seed,
        observation_noise_enabled=g1.observation_noise_enabled, zero_velocity_commands=True,
        avoidance_residual=0, capture_time_s=CAPTURE_SECONDS, captured_before_next_control=True,
        physics_dt_s=.005, control_dt_s=.020, copied_raw_action_history=list(RESET_KEYS[2:]),
        solver_context=list(EXTRA_DATA_FIELDS), retained_measurements=["clean99", "clean111", "actor99"],
        policy_recurrent_state="None; frozen feedforward Fresh96 including its exported normalizer",
        camera_history="Not constructed during settling; attack capture(reset=True) starts with truthful padding",
        frame="World coordinates unchanged from normal reset; G origin is ground projection of initial pelvis, +X initial heading, +Y left, +Z up",
        joint_order=selected["joint_order"], action_joint_order=selected["action_target_joint_order"],
        runtime_identity=g1.runtime_identity,
        final_half_second=dict(pelvis_planar_displacement_m=float(np.linalg.norm(
            trace["qpos"][last, :2]-trace["qpos"][last, :2][0], axis=1).max()),
            pelvis_planar_speed_max_m_s=float(np.linalg.norm(trace["qvel"][last, :2],axis=1).max()),
            root_height_range_m=[float(trace["qpos"][last,2].min()),float(trace["qpos"][last,2].max())],
            mechanically_safe_all=bool(trace["mechanically_safe"][last].all())),
        geometry={key.removeprefix("geometry/"): value for key, value in arrays.items()
                  if key.startswith("geometry/")},
        payload_sha256=hashlib.sha256((output/"fixture.npz").read_bytes()).hexdigest())
    dump(output / "fixture.json", metadata)
    fixture = StandingFixture(arrays, metadata)
    # One actor interval suffices to test the changed reset/controller contract.
    # It is standing-only verification, not a second candidate launch state.
    with g1.ordered_execution(), torch.no_grad():
        expected_action = cpu(g1.control_action())
        g1.apply_physical_target(g1.control_action())
        for _ in range(4):
            g1.step_substep()
        expected_q, expected_v = cpu(g1.data.qpos), cpu(g1.data.qvel)
    restore_record = fixture.restore(g1, camera_id=camera_id)
    with g1.ordered_execution(), torch.no_grad():
        actual_action = cpu(g1.control_action())
        g1.apply_physical_target(g1.control_action())
        for _ in range(4):
            g1.step_substep()
        errors = dict(action=float(np.max(np.abs(actual_action-expected_action))),
            qpos=float(np.max(np.abs(cpu(g1.data.qpos)-expected_q))),
            qvel=float(np.max(np.abs(cpu(g1.data.qvel)-expected_v))))
    dump(output / "restore_check.json", dict(reset=restore_record,
        one_control_interval_max_abs_error=errors, tolerance=2e-6,
        passed=bool(max(errors.values()) <= 2e-6)))
    if max(errors.values()) > 2e-6:
        raise AssertionError(f"Standing fixture continuation mismatch: {errors}")
    fixture.restore(g1, camera_id=camera_id)
    return fixture


def main() -> None:
    import argparse
    import os
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from .camera import add_camera_only, mount_config
    from .g1 import DynamicG1, make_task08_cfg
    from .phase2 import load_library
    from .placement_phase3 import INDEX, PRIMITIVE_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("One allocated CUDA GPU required")
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    camera = mount_config(25.)
    # Load every selected ID explicitly. The first reference supplies only the
    # common mocap scene topology; no path/target/placement search occurs here.
    primitives = load_library(INDEX, PRIMITIVE_IDS)
    cfg = make_task08_cfg(count=1, seed=102503, output_dir=args.output,
        extra_entities={"h2": primitives[0].reference.entity_cfg()},
        scene_spec_fn=lambda spec: add_camera_only(spec, camera_config=camera))
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        g1 = DynamicG1(env, noise_seed=102504, observation_noise_enabled=True)
        fixture = capture_fixture(g1, env.sim.mj_model.camera(camera.camera_name).id, args.output)
        print(json.dumps(dict(fixture_id=fixture.metadata["fixture_id"],
                              standing=fixture.metadata["final_half_second"])), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
