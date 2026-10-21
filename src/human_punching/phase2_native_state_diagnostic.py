"""Bounded diagnosis of the failed twelve-interval native-state parity check.

This command observes the preserved adapter. It adds no forward or integration
call and never relaxes its exact comparison. Native execution needs a separately
counted Slurm allocation; CPU helpers and tests perform no native simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np

VERSION = "task10b_native_state_difference_diagnostic_v1"
PARENT_SHA256 = "32ee676197b2de746dd3674680229841095051fee41452f5ce0c0b2a1e6f8758"
MAX_SNAPSHOTS = 96
MAX_SNAPSHOT_BYTES = 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
MAX_CONTACTS = 256
MAX_CONSTRAINTS = 1024
LAYOUT = (("time", 1), ("qpos", 55), ("qvel", 54), ("qacc_warmstart", 54),
          ("ctrl", 48), ("qfrc_applied", 54), ("xfrc_applied", 144))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def split_integration(vector: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(vector)
    if value.shape != (1, 410) or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("One complete finite float64 accepted integration vector required")
    result = {}; start = 0
    for name, count in LAYOUT:
        result[name] = value[:, start:start+count].copy(); start += count
    assert start == 410
    return result


def difference(left: np.ndarray, right: np.ndarray) -> dict:
    a, b = np.asarray(left), np.asarray(right)
    schema = a.shape == b.shape and a.dtype == b.dtype
    result = dict(same_schema=schema, left_shape=list(a.shape), right_shape=list(b.shape),
                  left_dtype=a.dtype.str, right_dtype=b.dtype.str, exact_equal=False)
    if not schema:
        return result
    equal = np.equal(a, b)
    result.update(exact_equal=bool(np.array_equal(a, b)), bytes_equal=a.tobytes() == b.tobytes(),
                  unequal_values=int(np.count_nonzero(~equal)),
                  first_unequal_index=None if equal.all() else np.argwhere(~equal)[0].tolist())
    finite = np.isfinite(a) & np.isfinite(b)
    result["nonfinite_values"] = int(np.count_nonzero(~finite))
    result["maximum_absolute_difference"] = float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)), initial=0))
    if a.dtype.kind == "f" and finite.all():
        # All canonical values came from float32 native storage. Distances are
        # diagnostics only: even one ULP still fails the unchanged exact gate.
        aa, bb = a.astype(np.float32).view(np.int32).astype(np.int64), b.astype(np.float32).view(np.int32).astype(np.int64)
        aa = np.where(aa < 0, -2147483648-aa, aa); bb = np.where(bb < 0, -2147483648-bb, bb)
        result["maximum_float32_ulp_distance"] = int(np.max(np.abs(aa-bb), initial=0))
    return result


def compare_snapshots(left: dict, right: dict) -> dict:
    keys = sorted(set(left) | set(right))
    comparisons = {key: (difference(left[key], right[key]) if key in left and key in right
                         else dict(exact_equal=False, missing_left=key not in left, missing_right=key not in right))
                   for key in keys}
    native = comparisons.get("native.integration", {})
    return dict(exact_native_equal=native.get("exact_equal") is True,
                unequal_fields=[key for key, item in comparisons.items() if item["exact_equal"] is not True],
                fields=comparisons)


class RetainedRecorder:
    """Lossless exclusive files; every completed delegate call has a receipt."""
    def __init__(self, output: Path, capture: Callable):
        self.output = output
        self.capture = capture
        self.snapshots = []
        self.calls = []
        self.latest = {}
        self.total_array_bytes = 0
        self.held_actions = {}

    def observe(self, branch: str, label: str, env) -> dict:
        if len(self.snapshots) >= MAX_SNAPSHOTS:
            raise ValueError("Diagnostic snapshot count exhausted")
        arrays = {key: np.asarray(value).copy() for key, value in self.capture(env).items()}
        if id(env) in self.held_actions:
            arrays["controller.held_raw_action"] = np.asarray(self.held_actions[id(env)]).copy()
        if any(value.dtype.kind not in "biuf" for value in arrays.values()):
            raise ValueError("Only explicit numeric diagnostic arrays are permitted")
        size = sum(value.nbytes for value in arrays.values())
        if size > MAX_SNAPSHOT_BYTES or self.total_array_bytes + size > MAX_TOTAL_BYTES:
            raise ValueError("Diagnostic retained-array byte cap exhausted")
        index = len(self.snapshots)
        path = self.output / f"snapshot_{index:03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        item = dict(index=index, branch=branch, label=label, path=str(path), sha256=digest(path),
                    uncompressed_bytes=size, arrays={k: dict(shape=list(v.shape), dtype=v.dtype.str) for k, v in arrays.items()})
        self.snapshots.append(item); self.latest[branch] = (item, arrays)
        self.total_array_bytes += size
        write_json(self.output / f"snapshot_{index:03d}.json", item)
        return item

    def compare(self, label: str) -> dict:
        li, left = self.latest["source"]; ri, right = self.latest["restored"]
        return dict(label=label, source_snapshot=li["index"], restored_snapshot=ri["index"],
                    **compare_snapshots(left, right))


class ObservedSimulation:
    """Composition, without replacing backend arrays or modifying its methods."""
    def __init__(self, backend, env, recorder: RetainedRecorder, branch: str):
        self.backend, self.env, self.recorder, self.branch = backend, env, recorder, branch

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _call(self, method, *args, **kwargs):
        before = self.recorder.observe(self.branch, method+"_before", self.env)
        index = len(self.recorder.calls)
        item = dict(index=index, branch=self.branch, method=method, before_snapshot=before["index"],
                    call_returned=False, exception=None)
        self.recorder.calls.append(item)
        write_json(self.recorder.output / f"call_{index:03d}_intent.json", item)
        try:
            result = getattr(self.backend, method)(*args, **kwargs)
            item["call_returned"] = True
        except Exception as error:
            item["exception"] = dict(type=type(error).__name__, message=str(error))
            write_json(self.recorder.output / f"call_{index:03d}_terminal.json", item)
            raise
        # Persist return extent before a diagnostic read can itself fail.
        write_json(self.recorder.output / f"call_{index:03d}_returned.json", item)
        item["after_snapshot"] = self.recorder.observe(self.branch, method+"_after", self.env)["index"]
        write_json(self.recorder.output / f"call_{index:03d}_terminal.json", item)
        return result

    def reset(self, *args, **kwargs): return self._call("reset", *args, **kwargs)
    def forward(self, *args, **kwargs): return self._call("forward", *args, **kwargs)
    def step(self, *args, **kwargs): return self._call("step", *args, **kwargs)


def capture_arrays(env) -> dict:
    import torch
    from . import phase2_native_state as parent
    from .wbc.release_baseline_probe import _tensor_view
    def copy(value, index=None):
        tensor = _tensor_view(value)
        return (tensor if index is None else tensor[index]).detach().cpu().numpy().copy()
    native = parent.copy_native_integration(env.model, env.sim.data, 1)
    result = {"native.integration": native}
    result.update({"native."+k: v for k, v in split_integration(native).items()})
    for name in sorted(parent.STATE_FIELDS - {"hard_faults"}):
        if hasattr(env, name):
            value = getattr(env, name)
            result["controller."+name] = np.asarray(value, np.int64) if type(value) in (int, bool) else copy(value)
    result["controller.phase_denominator_s"] = copy(env.duration)
    for name in parent.STATIC_CONTROLLER_ARRAYS:
        value = getattr(env, name)
        result["static."+name] = copy(value) if isinstance(value, torch.Tensor) or hasattr(value, "_tensor") else np.asarray(value)
    data = env.sim.wp_data
    for name in ("solver_niter", "ne", "nf", "nl", "nefc", "nacon", "nisland", "qacc", "qacc_smooth", "qfrc_constraint", "qfrc_smooth", "actuator_force", "qfrc_actuator", "overflow"):
        result["derived."+name] = copy(getattr(data, name))
    nc = int(result["derived.nacon"].reshape(-1)[0]); nefc = int(result["derived.nefc"].reshape(-1)[0])
    if not 0 <= nc <= MAX_CONTACTS or not 0 <= nefc <= MAX_CONSTRAINTS:
        raise ValueError("Active diagnostic contact/constraint bound exceeded")
    for name in ("geom", "dim", "efc_address", "worldid", "geomcollisionid", "type", "dist", "pos", "frame"):
        result["contact."+name] = copy(getattr(data.contact, name), slice(None, nc))
    for name in ("type", "id", "state", "pos", "D", "vel", "aref", "force", "island"):
        result["constraint."+name] = copy(getattr(data.efc, name), (slice(None), slice(None, nefc)))
    return result


def execute(catalog: Path, release: Path, output: Path, wall_seconds: float) -> dict:
    import torch
    from . import phase2_native_state as parent
    from .phase2_deadline import ExecutionDeadline
    from .phase2_learning import NativeSpecialistEnv, family_arguments, fresh_policy
    from .wbc.provenance import PROJECT_ROOT, writable_path
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    from .wbc.release_baseline_probe import _tensor_view
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A counted Slurm CUDA allocation is required")
    if not np.isfinite(wall_seconds) or not 120 < wall_seconds <= 270:
        raise ValueError("Finite diagnostic child wall must be <=270 seconds")
    if digest(Path(parent.__file__)) != PARENT_SHA256:
        raise ValueError("Preserved original native-state adapter changed")
    catalog, release = catalog.resolve(strict=True), release.resolve(strict=True)
    if not catalog.is_relative_to(PROJECT_ROOT.resolve()) or release != (PROJECT_ROOT/"outputs/four-punch-milestone").resolve(strict=True):
        raise ValueError("Owned catalog and unchanged accepted release required")
    rows = json.loads(catalog.read_text())["cells"]
    if (len(rows) != 1 or rows[0].get("family") != "jab" or rows[0].get("kind") != "original_control"
            or rows[0].get("reference_sha256") != "6f045112585abf4c3604252d8baad422d401050b2e70f0f39da86e031ff9bc42"):
        raise ValueError("Exactly the preserved original jab control is required")
    started = time.monotonic()
    deadline = ExecutionDeadline(wall_seconds, serialization_reserve_s=90., session_started_monotonic=started)
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    writer = RetainedRecorder(folder, capture_arrays)
    args, paths = family_arguments(release, "jab")
    comparisons = []; observation_equal = action_equal = None; exception = None
    policy_records = []; environments = {}
    def array(value): return _tensor_view(value).detach().cpu().numpy().copy()
    def construct(branch):
        env = NativeSpecialistEnv(args, rows, num_envs=1)
        # Existing backend storage and captured graphs remain untouched.
        env.sim = ObservedSimulation(env.sim, env, writer, branch)
        environments[branch] = env
        deadline.checkpoint("after_"+branch+"_constructor")
        env.reset(random_phase=False, perturb=False, reference_ids=np.array([0]))
        writer.observe(branch, "completed_original_reset", env)
        return env
    def inference(policy, env, branch):
        observation = env.observe(); action = policy.actor(observation)
        path = folder / f"policy_{len(policy_records):03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, observation=array(observation), raw_action=array(action))
        policy_records.append(dict(branch=branch, path=str(path), sha256=digest(path)))
        writer.held_actions[id(env)] = array(action)
        return observation, action
    def advance(env, action, branch):
        deadline.checkpoint("before_explicit_5ms_interval")
        env.advance_substep(action, stop_on_fall=True)
        writer.observe(branch, "completed_advance_with_updated_counters", env)
    write_json(folder/"declaration.json", dict(version=VERSION, parent_source_sha256=PARENT_SHA256,
        catalog=dict(path=str(catalog), sha256=digest(catalog)),
        accepted_inputs={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        planned_explicit_integration_world_steps=12, state_spec=16383, state_size=410,
        source_prefix_steps=4, next_interval_pairs=4, diagnostic_child_wall_s=wall_seconds,
        max_snapshots=MAX_SNAPSHOTS, max_snapshot_bytes=MAX_SNAPSHOT_BYTES, max_total_array_bytes=MAX_TOTAL_BYTES,
        max_active_contacts=MAX_CONTACTS, max_active_constraints=MAX_CONSTRAINTS,
        exact_gate_unchanged=True, parent_adapter_unmodified=True, new_physics_performed_by_preparation=False,
        synchronized_diagnostic_reads_between_calls=True,
        constructor_scope="Original constructors run unchanged; composition observes every explicit reset/forward/step thereafter",
        shared_start_feasibility_qualified=False))
    try:
        with deadline.signal_handlers(), native_stream_scope(), torch.no_grad():
            left = construct("source")
            policy = fresh_policy(paths, left.device)
            _, first = inference(policy, left, "source_prefix")
            for _ in range(4): advance(left, first, "source")
            left.last_action[:] = first; left.actor_updates_executed += 1
            writer.observe("source", "completed_source_actor_interval", left)
            saved = parent.capture_comparison(left, held_raw_action=array(left.last_action),
                reference_phase="ready_hold", support_interval_end_s=array(left.sim.data.time))
            saved_path = parent.save_snapshot(saved, folder/"saved_start")
            right = construct("restored")
            restore = parent.restore_comparison(right, parent.load_snapshot(saved_path),
                trial_id="diagnostic_independent_destination", reference_ids=np.array([0]))
            writer.held_actions[id(right)] = array(right.last_action)
            writer.observe("restored", "immediately_after_exact_restore", right)
            comparisons.append(writer.compare("immediately_after_exact_restore"))
            left.sim.forward(); right.sim.forward()
            comparisons.append(writer.compare("after_common_forward_refresh"))
            obs_left, action_left = inference(policy, left, "source_continuation")
            obs_right, action_right = inference(policy, right, "restored_continuation")
            observation_equal = torch.equal(obs_left, obs_right)
            action_equal = torch.equal(action_left, action_right)
            for i in range(4):
                advance(left, action_left, "source"); advance(right, action_right, "restored")
                comparisons.append(writer.compare(f"paired_integration_{i+1}"))
    except Exception as error:
        exception = dict(type=type(error).__name__, message=str(error))
        restore = locals().get("restore")
    steps = [item for item in writer.calls if item["method"] == "step"]
    paired = [item for item in comparisons if item["label"].startswith("paired_integration_")]
    exact = bool(exception is None and observation_equal and action_equal and len(paired) == 4
                 and all(item["exact_native_equal"] for item in paired))
    result = dict(schema=VERSION, diagnostic_complete=exception is None, exact_parity_passed=exact,
        exception=exception, observations_equal=observation_equal, actor_outputs_equal=action_equal,
        restore=restore, comparisons=comparisons, snapshots=writer.snapshots, native_calls=writer.calls,
        policy_records=policy_records, explicit_step_calls_issued=len(steps),
        explicit_step_calls_returned=sum(item["call_returned"] for item in steps),
        unattempted_explicit_step_calls=max(0, 12-len(steps)),
        unknown_extent_step_calls=sum(not item["call_returned"] for item in steps),
        retained_array_bytes=writer.total_array_bytes,
        environment_counters={k: dict(physics_steps=v.physics_steps_executed, actor_updates=v.actor_updates_executed,
                                     hard_faults=v.hard_faults) for k, v in environments.items()},
        session_wall_s=time.monotonic()-started, source_sha256=digest(Path(__file__)),
        parent_source_sha256=digest(Path(parent.__file__)), slurm_job_id=os.environ["SLURM_JOB_ID"],
        synchronization_may_change_nondeterministic_interleaving=True,
        no_automatic_retry=True, exact_gate_unchanged=True, shared_start_feasibility_qualified=False,
        g1_actor_executed=False, policy_promotion=False)
    write_json(folder/"diagnostic.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "release", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args()
    result = execute(args.catalog, args.release, args.output, args.wall_seconds)
    print(json.dumps({k: result[k] for k in ("diagnostic_complete", "exact_parity_passed", "explicit_step_calls_returned", "exception")}))
    return 0 if result["exact_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
