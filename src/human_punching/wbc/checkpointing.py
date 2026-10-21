"""Full learner continuation at a completed WBC/RSL iteration boundary.

Call saving on EVERY rank after inherited ``learn`` returns. Restore after model,
optimizer and environment construction, then restore rank-local RNG immediately
before an explicit environment reset. Physical trajectories start fresh: these
artifacts never claim exact simulator/collector continuation.

The inherited runner broadcasts model state at every learn entry. Pinned RSL
normalization already all-reduces its updates, so learned parameters AND buffers
must agree across saved ranks. The helper enforces that parity and leaves the
native startup broadcast and PPO unchanged. Only RNG/sampler state is rank-local.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable

from .provenance import sha256_file, writable_path, write_json
from .runtime.bundle import SOURCE_PINS

SCHEMA = "human_wbc_boundary_checkpoint_v1"
CONTINUATION = "full_learner_with_fresh_physical_trajectories_not_exact_simulator_continuation"
STAGES = ("canonical", "domain", "guard_start")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class CheckpointContract:
    body_sha256: str
    controller_sha256: str
    actor_observation_schema_sha256: str
    action_schema_sha256: str
    simulation_contract_sha256: str
    learner_configuration_sha256: str
    reference_manifest_sha256: str
    implementation_sources_sha256: str
    source_pins: dict[str, str]

    def __post_init__(self):
        values = asdict(self)
        if not all(_hash(value) for key, value in values.items() if key != "source_pins"):
            raise ValueError("Checkpoint contract requires SHA256 identities for all interfaces/assets")
        if self.source_pins != SOURCE_PINS:
            raise ValueError("Checkpoint requires the exact pinned WBC/mjlab source revisions")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


def build_checkpoint_contract(*, body_path: Path, controller: dict,
                              actor_observation_schema: dict, action_schema: dict,
                              simulation_contract: dict, learner_configuration: dict, reference_manifest: Path,
                              implementation_sources: dict[str, Path]) -> CheckpointContract:
    """Compute expected identity from CURRENT artifacts, never from a saved claim.

    The caller's source map must cover its executing human extension; the launch
    packet separately verifies that WBC/mjlab imports use the pinned exports.
    Observation schema must include ordered terms/dimensions and normalization;
    a network width alone is not a sufficient actor-interface description.
    """
    if not implementation_sources or not actor_observation_schema or not action_schema:
        raise ValueError("Source inventory and named observation/action schemas are required")
    source_hashes = {name: sha256_file(path.resolve(strict=True))
                     for name, path in implementation_sources.items()}
    return CheckpointContract(
        sha256_file(body_path.resolve(strict=True)), _digest(controller),
        _digest(actor_observation_schema), _digest(action_schema), _digest(simulation_contract),
        _digest(learner_configuration), sha256_file(reference_manifest.resolve(strict=True)),
        _digest(source_hashes), dict(SOURCE_PINS))


@dataclass(frozen=True)
class BoundaryProgress:
    completed_iterations: int
    stage: str
    stage_completed_iterations: int
    completed_validation_panels: int = 0
    curriculum: dict | None = None
    validation: dict | None = None

    def __post_init__(self):
        for value in (self.completed_iterations, self.stage_completed_iterations, self.completed_validation_panels):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Boundary counters must be nonnegative completed counts")
        if self.completed_iterations < 1 or self.stage_completed_iterations > self.completed_iterations:
            raise ValueError("Checkpoint must follow at least one completed iteration")
        if self.stage not in STAGES:
            raise ValueError("Unknown Task 10 training stage")
        _canonical(asdict(self))


@dataclass(frozen=True)
class StageTransition:
    transition_id: str
    from_stage: str
    to_stage: str
    from_reference_sha256: str
    to_reference_sha256: str
    reason: str

    def __post_init__(self):
        if (self.from_stage, self.to_stage) not in tuple(zip(STAGES, STAGES[1:])):
            raise ValueError("Only canonical-to-domain or domain-to-guard_start transitions are supported")
        if not self.transition_id.strip() or not self.reason.strip():
            raise ValueError("A named stage transition and reason are required")
        if not _hash(self.from_reference_sha256) or not _hash(self.to_reference_sha256):
            raise ValueError("Stage transition must name exact old/new reference manifests")


def _validated_device(device):
    import torch
    device = torch.device(device)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.index != local_rank:
            raise ValueError("Checkpoint may access only this rank's local CUDA device")
        if not torch.cuda.is_initialized():
            raise RuntimeError("Construct the allocated CUDA learner before saving/restoring its RNG")
    elif device.type != "cpu":
        raise ValueError("Only CPU checks and allocated local CUDA learners are supported")
    return device


def _device(runner):
    return _validated_device(runner.device)


def capture_rng(device) -> dict:
    """Capture local RNG only; never use get_rng_state_all or initialize peers."""
    import numpy as np
    import torch
    device = _validated_device(device)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda_local": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
            "device_type": device.type, "device_index": device.index}


def restore_rng(state: dict, device) -> None:
    import numpy as np
    import torch
    device = _validated_device(device)
    if state["device_type"] != device.type or state["device_index"] != device.index:
        raise ValueError("RNG restoration requires the same rank-local device mapping")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda_local"].cpu(), device)


def _parameter_digest(algorithm) -> str:
    """Learned parameter identity, recorded separately from normalization."""
    import torch
    digest = hashlib.sha256()
    for model_name, model in (("actor", algorithm._raw_actor), ("critic", algorithm._raw_critic)):
        for name, parameter in model.named_parameters():
            data = parameter.detach().cpu().contiguous()
            digest.update(_canonical([model_name, name, str(data.dtype), list(data.shape)]))
            digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _buffer_digest(algorithm) -> str:
    import torch
    digest = hashlib.sha256()
    for model_name, model in (("actor", algorithm._raw_actor), ("critic", algorithm._raw_critic)):
        for name, buffer in model.named_buffers():
            data = buffer.detach().cpu().contiguous()
            digest.update(_canonical([model_name, name, str(data.dtype), list(data.shape)]))
            digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_checkpoint_rank_parity(receipts: list[dict]) -> None:
    if not receipts:
        raise ValueError("Checkpoint has no rank receipts")
    for key in ("contract_sha256", "progress_sha256", "sampler_identity_sha256",
                "learned_parameters_sha256", "model_buffers_sha256"):
        if any(not _hash(row.get(key)) for row in receipts) or len({row[key] for row in receipts}) != 1:
            raise ValueError(f"Checkpoint ranks disagree on {key}")


def _rank(runner) -> tuple[int, int]:
    rank = int(getattr(runner, "gpu_global_rank", 0))
    world_size = int(getattr(runner, "gpu_world_size", 1))
    if world_size not in (1, 2, 4) or rank not in range(world_size):
        raise ValueError("Invalid checkpoint rank/world size")
    if world_size > 1:
        import torch
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() != rank
                or torch.distributed.get_world_size() != world_size):
            raise RuntimeError("All checkpoint ranks must use the learner's active distributed group")
    return rank, world_size


def _atomic_torch_save(path: Path, value: dict) -> None:
    import torch
    destination = writable_path(path)
    if destination.exists():
        raise FileExistsError(destination)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to overwrite evidence.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _collective_phase(world_size: int, phase: str):
    """Make recoverable per-rank write/validation errors fail every participant."""
    error = None
    try:
        yield
    except Exception as exception:
        error = exception
    if world_size == 1:
        if error is not None:
            raise error
        return
    import torch
    errors = [None]*world_size
    message = None if error is None else f"{type(error).__name__}: {error}"
    torch.distributed.all_gather_object(errors, message)
    if any(value is not None for value in errors):
        raise RuntimeError(f"Checkpoint {phase} failed across ranks: {errors}") from error


def save_boundary_checkpoint(output: Path, runner, *, contract: CheckpointContract,
                             progress: BoundaryProgress, sampler_identity: dict,
                             sampler_state: dict | None = None) -> dict:
    """Collective helper: EVERY rank calls once after inherited learn returns.

    Publication renames the complete staging directory only after all rank files,
    parameter identity and completed counts agree. Failed staging evidence remains
    separate and cannot be loaded as a completed checkpoint.
    """
    import torch
    destination = writable_path(output)
    staging = writable_path(destination.with_name(destination.name+".incomplete"))
    rank, world_size = _rank(runner)
    with _collective_phase(world_size, "boundary validation"):
        device = _device(runner)
        CheckpointContract(**asdict(contract))
        if not sampler_identity:
            raise ValueError("An explicit sampler identity is required")
        _canonical(sampler_identity)
        _canonical(sampler_state or {})
        if runner.current_learning_iteration != progress.completed_iterations-1:
            raise ValueError("RSL last iteration index does not match the completed-iteration count")
        if getattr(runner.alg.storage, "step", None) != 0:
            raise ValueError("Checkpoint must follow the PPO update with an empty rollout buffer")
    with _collective_phase(world_size, "staging creation"):
        if rank == 0:
            if destination.exists():
                raise FileExistsError(destination)
            staging.mkdir(parents=True, exist_ok=False)
    with _collective_phase(world_size, "rank serialization"):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        learner = runner.alg.save()
        required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
        if not required.issubset(learner):
            raise ValueError("Inherited algorithm save omitted full actor/critic/optimizer state")
        logger = getattr(runner, "logger", None)
        unwrapped = getattr(getattr(runner, "env", None), "unwrapped", None)
        payload = {
            "schema": SCHEMA, "rank": rank, "world_size": world_size,
            "contract": asdict(contract), "contract_sha256": contract.sha256,
            "progress": asdict(progress), "sampler_identity": sampler_identity,
            "sampler_state": sampler_state or {}, "learner": learner,
            "logger_counters": {name: getattr(logger, name, 0) for name in ("tot_timesteps", "tot_time")},
            "environment_common_step_counter": getattr(unwrapped, "common_step_counter", None),
            "continuation": CONTINUATION, "physical_state_saved": False,
            "rng": capture_rng(device),
        }
        filename = f"rank_{rank:02d}.pt"
        _atomic_torch_save(staging/filename, payload)
        receipt = {"rank": rank, "path": filename, "sha256": sha256_file(staging/filename),
                   "contract_sha256": contract.sha256, "progress_sha256": _digest(asdict(progress)),
                   "sampler_identity_sha256": _digest(sampler_identity),
                   "learned_parameters_sha256": _parameter_digest(runner.alg),
                   "model_buffers_sha256": _buffer_digest(runner.alg)}
    receipts = [receipt]
    if world_size > 1:
        receipts = [None]*world_size
        torch.distributed.all_gather_object(receipts, receipt)
    with _collective_phase(world_size, "publication"):
        if rank == 0:
            assert_checkpoint_rank_parity(receipts)
            manifest = {
                "schema": SCHEMA, "status": "complete", "world_size": world_size,
                "contract": asdict(contract), "contract_sha256": contract.sha256,
                "progress": asdict(progress), "sampler_identity": sampler_identity,
                "ranks": receipts, "continuation": CONTINUATION, "physical_state_saved": False,
                "native_distributed_resume_qualified": False,
            }
            write_json(staging/"manifest.json", manifest)
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(staging, destination)
    return {"path": str(destination), "manifest_sha256": sha256_file(destination/"manifest.json"),
            "continuation": CONTINUATION}


def _verify_transition(saved: CheckpointContract, expected: CheckpointContract,
                       progress: BoundaryProgress, stage: str,
                       transition: StageTransition | None) -> None:
    left, right = asdict(saved), asdict(expected)
    changed = {key for key in left if left[key] != right[key]}
    if transition is None:
        if changed or stage != progress.stage:
            raise ValueError("Checkpoint contract/stage mismatch; reference changes require a named stage transition")
        return
    if changed - {"reference_manifest_sha256"}:
        raise ValueError("Stage transition cannot change body, controller, actor schema, physics or executing sources")
    if (transition.from_stage != progress.stage or transition.to_stage != stage
            or transition.from_reference_sha256 != saved.reference_manifest_sha256
            or transition.to_reference_sha256 != expected.reference_manifest_sha256):
        raise ValueError("Named stage transition does not match the saved and requested contracts")


def load_boundary_checkpoint(checkpoint: Path, runner, *, manifest_sha256: str,
                             expected_contract: CheckpointContract, stage: str,
                             sampler_identity: dict, reset_environment: Callable[[], Any],
                             restore_sampler: Callable[[dict], None] | None = None,
                             transition: StageTransition | None = None,
                             transition_sampler_state: dict | None = None,
                             transition_curriculum: dict | None = None) -> dict:
    """Restore the full local learner, then RNG, then explicitly reset physics.

    The caller constructs the environment/runner first (including any constructor
    reset); this function's final explicit reset is the admitted fresh start.
    The caller must derive expected_contract from current, hash-verified assets.
    A transition deliberately resets stage-local counts and sampler state; global
    completed iterations and validation history remain recorded.
    """
    import torch
    root = checkpoint.resolve(strict=True)
    manifest_path = root/"manifest.json"
    if not _hash(manifest_sha256) or sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Checkpoint manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Checkpoint is not a completed WBC boundary artifact")
    saved = CheckpointContract(**manifest["contract"])
    if manifest.get("contract_sha256") != saved.sha256:
        raise ValueError("Checkpoint contract hash mismatch")
    progress = BoundaryProgress(**manifest["progress"])
    _verify_transition(saved, expected_contract, progress, stage, transition)
    if manifest.get("sampler_identity") != sampler_identity:
        raise ValueError("Checkpoint sampler identity mismatch")
    rank, world_size = _rank(runner)
    device = _device(runner)
    if manifest.get("world_size") != world_size:
        raise ValueError("Boundary continuation requires the same rank count")
    rows = manifest.get("ranks", [])
    if len(rows) != world_size or {row.get("rank") for row in rows} != set(range(world_size)):
        raise ValueError("Checkpoint does not contain exactly one file per rank")
    assert_checkpoint_rank_parity(rows)
    files = {}
    receipts = {}
    for row in rows:
        path = (root/row["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or sha256_file(path) != row["sha256"]:
            raise ValueError("Checkpoint rank file path/hash mismatch")
        if (row.get("contract_sha256") != saved.sha256 or
                row.get("progress_sha256") != _digest(asdict(progress)) or
                row.get("sampler_identity_sha256") != _digest(sampler_identity)):
            raise ValueError("Checkpoint rank receipt does not match its contract/counters")
        files[row["rank"]] = path
        receipts[row["rank"]] = row
    if not callable(reset_environment):
        raise ValueError("An explicit fresh environment reset callback is required")
    # Load only after all files and the externally supplied manifest identity have
    # passed. These are this project's trusted full-state files, not arbitrary PT.
    payload = torch.load(files[rank], map_location=device, weights_only=False)
    if (payload.get("schema") != SCHEMA or payload.get("rank") != rank
            or payload.get("world_size") != world_size or payload.get("contract") != asdict(saved)
            or payload.get("contract_sha256") != saved.sha256
            or payload.get("progress") != asdict(progress)
            or payload.get("sampler_identity") != sampler_identity
            or payload.get("physical_state_saved") is not False
            or payload.get("continuation") != CONTINUATION):
        raise ValueError("Checkpoint payload disagrees with its verified manifest")
    sampler = payload["sampler_state"] if transition is None else (transition_sampler_state or {})
    if sampler and restore_sampler is None:
        raise ValueError("Nonempty saved sampler state requires an explicit restoration callback")
    if (payload["rng"]["device_type"] != device.type
            or payload["rng"]["device_index"] != device.index):
        raise ValueError("Saved RNG device mapping differs from the current rank")
    loaded_iteration = runner.alg.load(payload["learner"], load_cfg=None, strict=True)
    if loaded_iteration is not True:
        raise RuntimeError("Inherited algorithm did not restore the full learner/iteration state")
    if _parameter_digest(runner.alg) != receipts[rank]["learned_parameters_sha256"]:
        raise ValueError("Restored learned parameters differ from their checkpoint identity")
    if _buffer_digest(runner.alg) != receipts[rank]["model_buffers_sha256"]:
        raise ValueError("Restored normalizers/model buffers differ from their checkpoint identity")
    runner.current_learning_iteration = progress.completed_iterations  # next index, not saved last index
    logger = getattr(runner, "logger", None)
    for key, value in payload["logger_counters"].items():
        if logger is not None:
            setattr(logger, key, value)
    unwrapped = getattr(getattr(runner, "env", None), "unwrapped", None)
    if unwrapped is not None and payload["environment_common_step_counter"] is not None:
        unwrapped.common_step_counter = payload["environment_common_step_counter"]
    if restore_sampler is not None:
        restore_sampler(sampler)
    restore_rng(payload["rng"], device)
    reset_environment()
    continued = progress if transition is None else replace(
        progress, stage=stage, stage_completed_iterations=0, curriculum=transition_curriculum or {},
        validation={"stage_panels_completed": 0, "selection_history_reset": True})
    return {"progress": asdict(continued), "prior_progress": asdict(progress),
            "transition": asdict(transition) if transition is not None else None,
            "continuation": CONTINUATION, "next_learning_iteration": progress.completed_iterations,
            "physical_state_restored": False,
            "distributed_startup_requirement": "native RSL broadcast unchanged; saved parameters and normalizer buffers agree"}
