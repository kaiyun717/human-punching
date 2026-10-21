"""Inspectable identity shared by PPO capacity measurements and training gates."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

from .budget import TrainingScale
from .conditioning import Conditioning, apply_conditioning
from .runtime.engineering_recipe import LEGACY_RECIPE_ID, resolve_engineering_recipe
from .provenance import PROJECT_ROOT, SOURCE_PINS, configure_local_caches, runtime_record, sha256_file

SCHEMA = "human_wbc_measurement_contract_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def configuration_value(value: Any, *, bundle_root: Path | None = None) -> Any:
    """Serialize values deterministically, preserving mapping/observation order.

    Callable identity is module+qualname and is backed by the source inventory.
    Mutable object repr/address is never an acceptable configuration identity.
    """
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": value.value}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # Configuration infinity (e.g. resampling upper bound) is a literal,
            # never interpreted as a successful native numeric measurement.
            return {"float_literal": str(value)}
        return value
    if isinstance(value, (str, Path)):
        text = str(value)
        if bundle_root is not None:
            text = text.replace(str(bundle_root), "<reference_bundle>")
        return text
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if is_dataclass(value) and not isinstance(value, type):
        return {"dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": [[field.name, configuration_value(getattr(value, field.name), bundle_root=bundle_root)]
                           for field in fields(value)]}
    if isinstance(value, dict):
        return {"ordered_items": [[str(key), configuration_value(item, bundle_root=bundle_root)]
                                  for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [configuration_value(item, bundle_root=bundle_root) for item in value]
    if callable(value):
        module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
        if module and name:
            return {"callable": f"{module}.{name}"}
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return {"tensor_dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.detach().cpu().tolist()}
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        return {"array_dtype": str(value.dtype), "values": value.tolist()}
    raise TypeError(f"Unsupported deterministic configuration type: {type(value).__module__}.{type(value).__qualname__}")


def _without_fields(serialized: dict, names: set[str]) -> dict:
    return {**serialized, "fields": [[name, value] for name, value in serialized["fields"] if name not in names]}


def source_inventory() -> dict:
    """Hash current human extension and verify imported upstream code exports.

    Upstream wheels may add a generated _version.py; it is recorded separately.
    No repository, source copy or installed package is changed by this audit.
    """
    human_files = sorted((PROJECT_ROOT/"src/human_punching").rglob("*.py"))
    human_files += [PROJECT_ROOT/name for name in ("pyproject.toml", "requirements-wbc.in", "requirements-wbc.lock")]
    human = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in human_files}
    upstream = {}
    for package, project in (("wbc_mjlab", "wbc-mjlab"), ("mjlab", "mjlab")):
        revision = SOURCE_PINS[project]
        exported = PROJECT_ROOT/".cache/deps"/f"{project}-{revision[:8]}"
        metadata_path = exported/".task10-source.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("revision") != revision:
            raise ValueError(f"{project} exported revision differs from the accepted pin")
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot locate imported {package}")
        imported_root = Path(spec.origin).resolve().parent
        prefix = f"src/{package}/"
        expected = {name[len(prefix):]: expected_hash for name, expected_hash in metadata["files"].items()
                    if name.startswith(prefix) and name.endswith(".py")}
        if not expected:
            raise ValueError(f"Source export has no {package} Python files")
        actual = {str(path.relative_to(imported_root)): sha256_file(path)
                  for path in sorted(imported_root.rglob("*.py"))}
        if any(actual.get(name) != expected_hash for name, expected_hash in expected.items()):
            raise ValueError(f"Imported {package} code differs from its pinned source export")
        extra = set(actual)-set(expected)
        if extra-{"_version.py"}:
            raise ValueError(f"Untracked imported {package} Python sources: {sorted(extra)}")
        upstream[project] = {"revision": revision, "imported_python_sha256": digest(actual),
                             "generated_python_files": {name: actual[name] for name in sorted(extra)},
                             "source_export_manifest_sha256": sha256_file(metadata_path)}
    return {"human_files": human, "human_files_sha256": digest(human), "upstream": upstream}


def contract_from_parts(*, bundle_identity: dict, scale: TrainingScale, conditioning: str,
                        environment_configuration: Any, learner_configuration: Any,
                        actor_observation_schema: Any, critic_observation_schema: Any,
                        sources: dict, runtime: dict) -> dict:
    """Pure constructor used by the real builder and CPU drift tests."""
    conditioning = Conditioning(conditioning).value
    payload = {"reference_and_body": bundle_identity,
               "scale": {"global_environments": scale.global_environments,
                         "gpu_count": scale.world_size, "environments_per_rank": scale.environments_per_rank,
                         "rollout_steps": scale.rollout_steps, "epochs": scale.epochs,
                         "minibatches": scale.minibatches},
               "conditioning": conditioning, "environment_configuration": environment_configuration,
               "learner_configuration": learner_configuration,
               "actor_observation_schema": actor_observation_schema,
               "critic_observation_schema": critic_observation_schema,
               "sources": sources, "runtime": runtime}
    return {"schema": SCHEMA, "payload": payload, "sha256": digest(payload)}


def build_measurement_contract(bundle, *, scale: TrainingScale,
                               conditioning: str = Conditioning.COMBINED.value,
                               env_cfg=None, agent_cfg=None, engineering_admission=None,
                               observer_report: Path | None = None, bridge_report: Path | None = None,
                               engineering_recipe=LEGACY_RECIPE_ID,
                               training_initialization_report: Path | None = None) -> dict:
    """Build before environment construction; no simulation or learning occurs.

    The caller supplies freshly configured objects when measuring a nondefault
    variant. Seed, run length, save/log locations and resume paths are excluded;
    all dynamics, contact capacities, actions, PPO/network/normalizer settings and
    ordered observation terms remain bound. Native widths/hardware are additional
    evidence rather than guessed pre-construction values.
    """
    # Direct API users need the same confinement as the CLI. Preserve already
    # configured per-rank cache directories instead of overwriting them.
    if any(not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(PROJECT_ROOT)
           for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR", "WARP_CACHE_PATH", "CUDA_CACHE_PATH")):
        configure_local_caches()
    from .runtime.bundle import load_runtime_bundle
    from .runtime.task import build_human_env_cfg, human_ppo_cfg
    checked = load_runtime_bundle(bundle)
    entry = {}
    recipe = resolve_engineering_recipe(engineering_recipe)
    if recipe.recipe_id == LEGACY_RECIPE_ID and training_initialization_report is not None:
        raise ValueError("Legacy recipe cannot borrow revised training initialization evidence")
    if engineering_admission is None:
        if (observer_report is not None or bridge_report is not None or training_initialization_report is not None
                or recipe.recipe_id != LEGACY_RECIPE_ID):
            raise ValueError("Observer engineering receipt requires explicit engineering admission")
        checked.require_training_qualification()
    else:
        from .engineering_admission import SCOPE, verify_engineering_admission
        from .capacity_evidence import (configure_engineering_capacity, observer_capacity_evidence, bridge_capacity_evidence,
                                        require_engineering_configuration, training_initialization_capacity_evidence)
        if observer_report is None:
            raise ValueError("Engineering capacity requires completed native observer instrumentation evidence")
        if bridge_report is None:
            raise ValueError("Engineering capacity requires completed native bridge instrumentation evidence")
        if recipe.recipe_id != LEGACY_RECIPE_ID and training_initialization_report is None:
            raise ValueError("Revised recipe requires completed native training initialization evidence")
        if Conditioning(conditioning) is not Conditioning.COMBINED:
            raise ValueError("Engineering cycle capacity requires combined phase-aware conditioning")
        verify_engineering_admission(checked, engineering_admission)
        observer = observer_capacity_evidence(checked, observer_report)
        bridge = bridge_capacity_evidence(checked, bridge_report)
        initialization = (training_initialization_capacity_evidence(checked, training_initialization_report,
            engineering_recipe=recipe) if training_initialization_report is not None else None)
        entry = {"scope": SCOPE, "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
                 "training_initialization_evidence": initialization,
                 "training_initialization_evidence_sha256": digest(initialization) if initialization is not None else None,
                 "engineering_admission_sha256": engineering_admission.sha256,
                 "observer_evidence": observer, "observer_evidence_sha256": digest(observer),
                 "bridge_evidence": bridge, "bridge_evidence_sha256": digest(bridge),
                 "production_reference_qualified": False}
    if env_cfg is None:
        env_cfg = (build_human_env_cfg(checked, num_envs=scale.environments_per_rank, evaluation=False)
                   if engineering_admission is None else configure_engineering_capacity(checked,
                       engineering_admission, num_envs=scale.environments_per_rank, conditioning=conditioning, engineering_recipe=recipe))
        if engineering_admission is None:
            apply_conditioning(env_cfg, conditioning)
        env_cfg.sim.nan_guard.enabled = True
    if engineering_admission is not None:
        require_engineering_configuration(env_cfg, engineering_recipe=recipe)
    if agent_cfg is None:
        agent_cfg = human_ppo_cfg()
    if env_cfg.scene.num_envs != scale.environments_per_rank:
        raise ValueError("Configuration uses a different per-rank environment count")
    if (agent_cfg.num_steps_per_env, agent_cfg.algorithm.num_learning_epochs,
            agent_cfg.algorithm.num_mini_batches) != (scale.rollout_steps, scale.epochs, scale.minibatches):
        raise ValueError("PPO configuration differs from global rollout/optimizer accounting")
    if not env_cfg.sim.nan_guard.enabled:
        raise ValueError("Capacity/production configuration requires the NaN guard")
    # Both the capacity/production contract require enabled NaN guard; its output
    # location belongs to the individual rank, not the scientific configuration.
    environment = configuration_value(env_cfg, bundle_root=checked.root)
    environment = _without_fields(environment, {"seed", "viewer"})
    for name, value in environment["fields"]:
        if name == "sim":
            for sim_name, sim_value in value["fields"]:
                if sim_name == "nan_guard":
                    sim_value["fields"] = [[key, val] for key, val in sim_value["fields"] if key != "output_dir"]
    learner = _without_fields(configuration_value(agent_cfg, bundle_root=checked.root), {
        "seed", "max_iterations", "save_interval", "experiment_name", "run_name", "resume",
        "load_run", "load_checkpoint", "wandb_project", "wandb_tags", "neptune_project",
    })
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise ValueError("Measurement runtime does not match the accepted candidate pins")
    return contract_from_parts(
        bundle_identity={"reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
                         "body_sha256": sha256_file(checked.model_path),
                         "body_metadata_sha256": sha256_file(checked.path(checked.manifest["body_metadata_path"])),
                         "controller_sha256": checked.controller_sha256(), **entry},
        scale=scale, conditioning=conditioning, environment_configuration=environment,
        learner_configuration=learner,
        actor_observation_schema=configuration_value(env_cfg.observations["actor"], bundle_root=checked.root),
        critic_observation_schema=configuration_value(env_cfg.observations["critic"], bundle_root=checked.root),
        sources=source_inventory(), runtime={"python": runtime["python"], "packages": runtime["packages"]})


def verify_measurement_contract(measured: dict, expected: dict) -> None:
    for value in (measured, expected):
        if (value.get("schema") != SCHEMA or not isinstance(value.get("payload"), dict)
                or value.get("sha256") != digest(value.get("payload"))):
            raise ValueError("Measurement contract schema/digest mismatch")
    if measured["sha256"] != expected["sha256"]:
        raise ValueError("Capacity evidence belongs to a different source/runtime/configuration contract")


def gpu_hardware_identity(device) -> dict:
    """Read the actual CUDA device; never substitute marketing memory capacity."""
    import torch
    import warp as wp
    import importlib.metadata
    import subprocess
    properties = torch.cuda.get_device_properties(device)
    command = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             check=True, text=True, capture_output=True)
    drivers = {line.strip() for line in command.stdout.splitlines() if line.strip()}
    if len(drivers) != 1:
        raise RuntimeError("Cannot establish a unique NVIDIA driver version for the allocated host")
    toolkit, driver_api = wp.get_cuda_toolkit_version(), wp.get_cuda_driver_version()
    if toolkit is None or driver_api is None:
        raise RuntimeError("Allocated Warp CUDA toolkit/driver identity is unavailable")
    return {"name": properties.name, "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "multiprocessor_count": int(properties.multi_processor_count),
            "torch_cuda_build": torch.version.cuda, "nvidia_driver_version": drivers.pop(),
            "warp_cuda_toolkit_build": list(toolkit), "warp_cuda_driver_api": list(driver_api),
            "cuda_nvrtc_package_version": importlib.metadata.version("nvidia-cuda-nvrtc-cu12")}


def verify_gpu_hardware(measured: dict, actual: dict) -> None:
    keys = ("name", "total_memory_bytes", "compute_capability", "multiprocessor_count", "torch_cuda_build",
            "nvidia_driver_version", "warp_cuda_toolkit_build", "warp_cuda_driver_api", "cuda_nvrtc_package_version")
    if any(key not in measured or key not in actual or measured[key] != actual[key] for key in keys):
        raise ValueError("Current GPU hardware differs from the capacity measurement")
