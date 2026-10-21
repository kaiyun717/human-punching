"""Integrity-bound complete tracker handoff, without deployment promotion.

TorchScript contains the actor and its running normalizer and needs only Torch
for inference. The reference/planner/controller source is included for review;
its inclusion does not make the current WBC environment portable. A robust_sav
observation/controller adapter and paired physical qualification remain required.
This module packages existing artifacts; it never trains or submits a job.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any
import xml.etree.ElementTree as ET

from .checkpointing import (BoundaryProgress, CheckpointContract, CONTINUATION,
                            SCHEMA as CHECKPOINT_SCHEMA, assert_checkpoint_rank_parity)
from .provenance import PROJECT_ROOT, RUNTIME_PINS, PYTHON_PIN, sha256_file, writable_path, write_json
from .runtime.bundle import load_runtime_bundle, SOURCE_PINS

SCHEMA = "human_wbc_tracker_export_v1"
LABELS = ("engineering_fixture", "trained_candidate")
INTERFACES = {"actor_observation_schema": "actor_observation_schema_sha256",
              "action_schema": "action_schema_sha256",
              "simulation_contract": "simulation_contract_sha256",
              "learner_configuration": "learner_configuration_sha256"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _member(root: Path, relative: str) -> Path:
    """Require portable literal relative names, with no links or traversal."""
    # Cluster login/compute nodes expose the same NFS tree under /home and
    # /orcd/home. Resolve the root alias, then inspect only member components.
    root = root.resolve(strict=True)
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative
            or str(pure) != relative):
        raise ValueError(f"Invalid portable member path: {relative}")
    candidate = root / relative
    component = root
    for part in pure.parts:
        component = component/part
        if component.is_symlink():
            raise ValueError(f"Symlink is not a portable artifact: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise ValueError(f"Member escapes bundle or is not a file: {relative}")
    return resolved


def _json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    # Reject nonfinite JSON accepted by the standard decoder.
    _digest(result)
    return result


def _boundary(root: Path, expected_hash: str) -> tuple[dict, dict]:
    """Verify every rank before loading trusted project-local full learner state."""
    import torch
    manifest_path = _member(root, "manifest.json")
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError("Boundary checkpoint manifest hash mismatch")
    manifest = _json(manifest_path)
    if (manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("continuation") != CONTINUATION
            or manifest.get("physical_state_saved") is not False):
        raise ValueError("Export requires a completed full learner boundary checkpoint")
    contract = CheckpointContract(**manifest["contract"])
    progress = BoundaryProgress(**manifest["progress"])
    if manifest.get("contract_sha256") != contract.sha256:
        raise ValueError("Boundary contract digest mismatch")
    rows, world_size = manifest.get("ranks", []), manifest.get("world_size")
    if (world_size not in (1, 2, 4) or len(rows) != world_size
            or {row.get("rank") for row in rows} != set(range(world_size))):
        raise ValueError("Boundary checkpoint is missing rank state")
    assert_checkpoint_rank_parity(rows)
    for row in rows:
        if (row.get("path") != f"rank_{row['rank']:02d}.pt"
                or sha256_file(_member(root, row["path"])) != row["sha256"]
                or row["contract_sha256"] != contract.sha256
                or row["progress_sha256"] != _digest(asdict(progress))
                or row["sampler_identity_sha256"] != _digest(manifest["sampler_identity"])):
            raise ValueError("Boundary rank file or receipt hash mismatch")
    # The externally supplied manifest identity must refer to trusted local
    # project output. Hash checking is integrity, not arbitrary pickle safety.
    payload = torch.load(_member(root, "rank_00.pt"), map_location="cpu", weights_only=False)
    for name, expected in (("schema", CHECKPOINT_SCHEMA), ("rank", 0), ("world_size", world_size),
                           ("contract", asdict(contract)), ("contract_sha256", contract.sha256),
                           ("progress", asdict(progress)), ("sampler_identity", manifest["sampler_identity"]),
                           ("continuation", CONTINUATION), ("physical_state_saved", False)):
        if payload.get(name) != expected:
            raise ValueError(f"Boundary rank-zero payload mismatch: {name}")
    learner = payload.get("learner", {})
    if not {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}.issubset(learner):
        raise ValueError("Full learner state is required; actor weights alone are insufficient")
    return manifest, payload


def _normalizer(actor_state: dict) -> dict:
    import torch
    fields = {}
    for name in ("_mean", "_var", "_std", "count"):
        key = "obs_normalizer." + name
        value = actor_state.get(key)
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint lacks finite embedded actor normalization: {key}")
        fields[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
    if (actor_state["obs_normalizer._mean"].shape != actor_state["obs_normalizer._std"].shape
            or torch.any(actor_state["obs_normalizer._std"] < 0)
            or torch.any(actor_state["obs_normalizer._var"] < 0)
            or actor_state["obs_normalizer.count"].numel() != 1
            or actor_state["obs_normalizer.count"].item() < 0):
        raise ValueError("Invalid normalizer shape or scale")
    return {"schema": "human_wbc_actor_normalizer_v1", "state": fields,
            "embedded_in_jit_and_onnx": True, "epsilon": .01,
            "formula": "(observation - mean) / (standard_deviation + epsilon)"}


def _policy_check(jit_path: Path, onnx_path: Path, actor_state: dict, obs_dim: int,
                  action_dim: int) -> dict:
    """Check embedded state and native CPU FP32 JIT/ONNX computation on fixtures.

    ONNX's reference evaluator avoids adding ONNX Runtime to accepted runtime
    dependencies. This small deterministic check is not the 10,000-observation
    portability panel and does not establish physical rollout parity.
    """
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import torch
    actor = torch.jit.load(str(jit_path), map_location="cpu").eval()
    if actor.obs_normalizer.eps != .01:
        raise ValueError("Exported normalizer epsilon differs from pinned RSL MLP semantics")
    saved = actor.state_dict()
    expected = {key: tensor for key, tensor in actor_state.items()
                if key.startswith(("mlp.", "obs_normalizer."))}
    if not expected or saved.keys() != expected.keys():
        raise ValueError("TorchScript actor/normalizer keys differ from full checkpoint")
    if any(not torch.equal(saved[key].cpu(), tensor.cpu()) for key, tensor in expected.items()):
        raise ValueError("TorchScript actor/normalizer differs from full checkpoint")
    if actor_state["obs_normalizer._mean"].numel() != obs_dim:
        raise ValueError("Ordered observation schema width differs from actor normalization")
    model = onnx.load(str(onnx_path), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("ONNX external tensor data is not supported; export a self-contained actor")
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one actor observation input and one residual output")
    shapes = [value.type.tensor_type.shape.dim for value in (model.graph.input[0], model.graph.output[0])]
    if any(len(shape) != 2 for shape in shapes) or [shape[1].dim_value for shape in shapes] != [obs_dim, action_dim]:
        raise ValueError("ONNX observation/action dimensions differ from contract")
    evaluator = ReferenceEvaluator(model)
    generator = np.random.default_rng(10010)
    inputs = np.concatenate((np.zeros((1, obs_dim), dtype=np.float32),
                             generator.normal(size=(31, obs_dim)).astype(np.float32) * 4), axis=0)
    maximum = 0.
    with torch.inference_mode():
        # Exported batch size may be fixed at one; test individual rows.
        for values in inputs:
            values = values[None]
            actual = actor(torch.from_numpy(values)).cpu().numpy()
            other = evaluator.run(None, {model.graph.input[0].name: values})[0]
            if actual.shape != (1, action_dim) or not np.isfinite(actual).all() or not np.isfinite(other).all():
                raise ValueError("Nonfinite or incorrectly shaped exported actor result")
            maximum = max(maximum, float(np.max(np.abs(actual-other))))
    if maximum > 1e-5:
        raise ValueError(f"ONNX/JIT fixture action parity failed: {maximum}")
    return {"sample_count": 32, "kind": "deterministic_synthetic_cpu_fp32",
            "onnx_evaluator": "onnx.reference.ReferenceEvaluator", "max_action_abs_error": maximum,
            "checkpoint_actor_state_exact": True, "ten_thousand_real_observation_gate_passed": False,
            "paired_physical_rollout_gate_passed": False}


def _xml_assets(bundle) -> None:
    """Require the exported reference root to contain the XML asset closure."""
    pending, visited = [bundle.model_path], set()
    while pending:
        xml = pending.pop()
        if xml in visited:
            continue
        visited.add(xml)
        tree = ET.parse(xml)
        compiler = tree.getroot().find("compiler")
        # Supporting directory rewriting would need a separately qualified
        # relocation adapter. Current constrained body is self-contained.
        if compiler is not None and any(compiler.get(key) for key in ("assetdir", "meshdir", "texturedir")):
            raise ValueError("XML asset search directories need an explicit portable relocation adapter")
        for element in tree.iter():
            value = element.get("file")
            if value:
                relative = (xml.parent / value).relative_to(bundle.root).as_posix()
                asset = _member(bundle.root, relative)
                if element.tag == "include":
                    pending.append(asset)


def _domain(path: Path, bundle, *, trained: bool) -> dict:
    domain = _json(path)
    if (domain.get("schema") != "human_wbc_domain_manifest_v1"
            or domain.get("reference_manifest_sha256") != sha256_file(bundle.root/"manifest.json")
            or domain.get("model_sha256") != sha256_file(bundle.model_path)
            or domain.get("controller_sha256") != bundle.controller_sha256()):
        raise ValueError("Domain manifest is not bound to these references/body/controller")
    if domain.get("families") != ["jab", "cross", "left_hook", "right_hook"]:
        raise ValueError("Domain must account for all four punch families")
    if trained and (domain.get("qualified") is not True or not domain.get("cells")):
        raise ValueError("A trained candidate requires a physically qualified target/guard/TTC domain")
    if trained:
        from .admission import QualificationCell
        cells = [QualificationCell(**cell) for cell in domain["cells"]]
        if len({cell.cell_id for cell in cells}) != len(cells):
            raise ValueError("Domain contains duplicate qualification cells")
        if {cell.family.value for cell in cells} != set(domain["families"]):
            raise ValueError("Qualified domain must include cells for all four punch families")
        evidence = domain.get("qualification_evidence", {})
        if not evidence.get("path") or not evidence.get("sha256"):
            raise ValueError("Qualified domain must identify physical evidence by hash")
        report_path = _member(path.parent, evidence["path"])
        if sha256_file(report_path) != evidence["sha256"]:
            raise ValueError("Domain physical evidence hash mismatch")
        report = _json(report_path)
        expected = {"schema": "human_wbc_domain_qualification_v1", "status": "passed",
                    "native_execution": True, "model_sha256": domain["model_sha256"],
                    "controller_sha256": domain["controller_sha256"],
                    "reference_manifest_sha256": domain["reference_manifest_sha256"],
                    "cells_sha256": _digest(domain["cells"]), "source_pins": SOURCE_PINS}
        if (any(report.get(key) != value for key, value in expected.items())
                or not report.get("slurm_job_id") or report.get("physical_trial_count", 0) < 1):
            raise ValueError("Domain qualification lacks bound native physical evidence")
        rows = report.get("per_cell", [])
        if (len(rows) != len(cells) or {row.get("cell_id") for row in rows} != {cell.cell_id for cell in cells}
                or any(row.get("status") != "passed" or type(row.get("physical_trial_count")) is not int
                       or row["physical_trial_count"] < 1 for row in rows)
                or sum(row["physical_trial_count"] for row in rows) != report["physical_trial_count"]):
            raise ValueError("Domain physical evidence must cover every declared cell with counted trials")
    return domain


def _planner(path: Path) -> dict:
    planner = _json(path)
    if (planner.get("schema") != "human_wbc_planner_bundle_v1" or not planner.get("entry_point")
            or not planner.get("runtime_dependencies") or not planner.get("portability_limitations")):
        raise ValueError("Planner requires an explicit entry point, dependencies and portability assessment")
    files = planner.get("files", [])
    if not files or len({row["path"] for row in files}) != len(files):
        raise ValueError("Planner must list a unique, nonempty code/data dependency closure")
    for row in files:
        if row.get("role") not in ("source", "data", "admission", "configuration"):
            raise ValueError("Unknown planner dependency role")
        if sha256_file(_member(path.parent, row["path"])) != row["sha256"]:
            raise ValueError("Planner dependency hash mismatch")
    return planner


def package_tracker(output: Path, *, policy_jit: Path, policy_onnx: Path,
                    boundary_checkpoint: Path, boundary_manifest_sha256: str,
                    reference_bundle: Path, interface_contracts: dict,
                    planner_manifest: Path, domain_manifest: Path,
                    qualification: str = "engineering_fixture",
                    training_evidence: Path | None = None,
                    implementation_sources: dict[str, Path] | None = None) -> dict:
    """Create an exclusive portable handoff directory from trusted local outputs.

    ``engineering_fixture`` is explicitly untrained, even if fixture checkpoint
    counters are positive. ``trained_candidate`` requires physical reference
    qualification and hashed native learning evidence. Neither label means
    acceptance, portability qualification, coupled-runtime readiness or promotion.
    Interface dictionaries must be the exact values hashed at training time.
    """
    if qualification not in LABELS:
        raise ValueError("Unknown export label; acceptance/promotion cannot be set by packaging")
    destination = writable_path(output)
    staging = writable_path(destination.with_name(destination.name+".incomplete"))
    if destination.exists() or staging.exists():
        raise FileExistsError(destination)
    paths = [policy_jit, policy_onnx, boundary_checkpoint, reference_bundle,
             planner_manifest, domain_manifest]
    # Full checkpoint pickle/JIT is trusted code, so refuse arbitrary external
    # inputs; explicit input hashes below additionally protect artifact identity.
    if any(not Path(path).resolve(strict=True).is_relative_to(PROJECT_ROOT) for path in paths):
        raise ValueError("Export inputs must be trusted project-local artifacts")
    bundle = load_runtime_bundle(reference_bundle)
    manifest, payload = _boundary(boundary_checkpoint.resolve(strict=True), boundary_manifest_sha256)
    contract = CheckpointContract(**manifest["contract"])
    if set(interface_contracts) != set(INTERFACES):
        raise ValueError("All observation/action/simulation/learner interfaces are required")
    for key, field in INTERFACES.items():
        if _digest(interface_contracts[key]) != getattr(contract, field):
            raise ValueError(f"Export interface differs from checkpoint: {key}")
    if (contract.body_sha256 != sha256_file(bundle.model_path)
            or contract.reference_manifest_sha256 != sha256_file(bundle.root/"manifest.json")
            or contract.controller_sha256 != bundle.controller_sha256()):
        raise ValueError("Checkpoint references/body/controller mismatch")
    sources = implementation_sources or {str(path.relative_to(PROJECT_ROOT)): path
               for path in (PROJECT_ROOT/"src/human_punching").rglob("*.py")}
    source_hashes = {name: sha256_file(Path(path).resolve(strict=True)) for name, path in sources.items()}
    if _digest(source_hashes) != contract.implementation_sources_sha256:
        raise ValueError("Executing source inventory differs from checkpoint")
    for name in sources:
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("Source inventory names must be portable relative paths")
    if qualification == "trained_candidate" and implementation_sources is not None:
        actual = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                  for path in (PROJECT_ROOT/"src/human_punching").rglob("*.py")}
        if source_hashes != actual:
            raise ValueError("Trained export must include the complete current human source inventory")
    trained = qualification == "trained_candidate"
    if trained:
        bundle.require_training_qualification()
    _xml_assets(bundle)
    domain = _domain(domain_manifest, bundle, trained=trained)
    planner = _planner(planner_manifest)
    if domain.get("reference_generator_sha256") != sha256_file(planner_manifest):
        raise ValueError("Domain does not identify the exact packaged planner")
    evidence = None
    if trained:
        if training_evidence is None:
            raise ValueError("Trained candidate label requires native learning evidence")
        evidence = _json(training_evidence)
        expected = {"schema": "human_wbc_export_training_evidence_v1", "status": "passed",
                    "native_execution": True, "includes_ppo": True,
                    "checkpoint_manifest_sha256": boundary_manifest_sha256,
                    "reference_manifest_sha256": contract.reference_manifest_sha256,
                    "completed_iterations": manifest["progress"]["completed_iterations"]}
        if any(evidence.get(key) != value for key, value in expected.items()) or not evidence.get("slurm_job_id"):
            raise ValueError("Native training evidence does not match the exported full learner")
    elif training_evidence is not None:
        raise ValueError("Engineering fixtures must not attach apparent training evidence")
    actor_schema = interface_contracts["actor_observation_schema"]
    actor_group = actor_schema.get("groups", {}).get("actor", {})
    shape = actor_group.get("shape", [])
    terms = actor_group.get("terms", [])
    if (len(shape) != 1 or not terms or any(not term.get("name") or not term.get("shape") for term in terms)
            or sum(math.prod(term["shape"]) for term in terms) != shape[0]):
        raise ValueError("Export needs the complete ordered actor observation schema")
    action_schema = interface_contracts["action_schema"]
    if (action_schema.get("dimension") != 48 or action_schema.get("joint_names") != list(bundle.joint_names)):
        raise ValueError("Action order differs from the physical human joint order")
    normalizer = _normalizer(payload["learner"]["actor_state_dict"])
    parity = _policy_check(policy_jit, policy_onnx, payload["learner"]["actor_state_dict"], shape[0], 48)
    staging.mkdir(parents=True, exist_ok=False)
    inventory = {}

    def copy(source: Path, relative: str, role: str):
        target = staging/relative
        if target.exists():
            raise ValueError(f"Duplicate export member: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256_file(source)
        shutil.copyfile(source, target)
        if sha256_file(target) != expected:
            raise ValueError("Source changed while publishing export")
        inventory[relative] = {"sha256": expected, "bytes": target.stat().st_size, "role": role}

    def generated(relative: str, value: Any, role: str):
        write_json(staging/relative, value)
        path = staging/relative
        inventory[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role}

    copy(policy_jit, "policy/policy_jit.pt", "actor_with_normalizer_torchscript")
    copy(policy_onnx, "policy/policy.onnx", "actor_with_normalizer_onnx")
    copy(boundary_checkpoint/"manifest.json", "learner/manifest.json", "full_learner_manifest")
    for row in manifest["ranks"]:
        copy(boundary_checkpoint/row["path"], "learner/"+row["path"], "full_learner_rank")
    for path in sorted(bundle.root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(bundle.root).as_posix()
            copy(_member(bundle.root, relative), "references/"+relative, "reference_and_body")
    copy(planner_manifest, "planner/manifest.json", "planner_dependency_manifest")
    for row in planner["files"]:
        copy(_member(planner_manifest.parent, row["path"]), "planner/"+row["path"], "planner_"+row["role"])
    copy(domain_manifest, "domain/manifest.json", "qualified_or_fixture_domain")
    if trained:
        domain_evidence = domain["qualification_evidence"]["path"]
        copy(_member(domain_manifest.parent, domain_evidence), "domain/"+domain_evidence,
             "domain_physical_qualification")
    for name, path in sources.items():
        copy(Path(path), "sources/"+name, "training_implementation_source")
    for name in ("pyproject.toml", "requirements-wbc.in", "requirements-wbc.lock"):
        copy(PROJECT_ROOT/name, "dependencies/"+name, "dependency_declaration")
    for name, directory in (("wbc_mjlab", "wbc-mjlab-eb6c024b"), ("mjlab", "mjlab-8ee51fbc")):
        source_manifest = PROJECT_ROOT/".cache/deps"/directory/".task10-source.json"
        if _json(source_manifest).get("revision") != SOURCE_PINS[name]:
            raise ValueError("Upstream source export manifest does not match the source pin")
        copy(source_manifest,
             "dependencies/"+name+"_source_export.json", "pinned_source_export_inventory")
    if evidence is not None:
        copy(training_evidence, "evidence/training.json", "native_learning_evidence")
    generated("policy/normalization.json", normalizer, "normalization_state")
    generated("interfaces/contracts.json", interface_contracts, "ordered_interfaces")
    generated("interfaces/controller.json", bundle.controller_contract(), "native_affine_controller")
    generated("interfaces/source_pins.json", SOURCE_PINS, "upstream_source_pins")
    generated("dependencies/runtime_pins.json", {"python": PYTHON_PIN, "packages": RUNTIME_PINS}, "runtime_pins")
    generated("evidence/synthetic_export_parity.json", parity, "limited_export_parity")
    export = {"schema": SCHEMA, "status": "packaged", "qualification": qualification,
        "trained_policy": trained, "acceptance_qualified": False, "portability_qualified": False,
        "coupled_runtime_qualified": False, "promoted": False,
        "checkpoint_manifest_sha256": boundary_manifest_sha256,
        "checkpoint_contract_sha256": contract.sha256,
        "reference_manifest_sha256": contract.reference_manifest_sha256,
        "source_pins": SOURCE_PINS, "files": inventory,
        "continuation": CONTINUATION,
        "inference": {"entry_point": "torch.jit.load('policy/policy_jit.pt', map_location='cpu')",
            "runtime_requirements": ["torch"], "normalization_embedded": True,
            "output": "raw_normalized_residual_clip_to_minus_one_plus_one_before_controller",
            "observation_adapter": "required_not_yet_portability_qualified",
            "reference_controller_adapter": "required_pre_integration_200hz_not_yet_portability_qualified",
            "current_planner_portability": "source_and_dependency_closure_only_not_an_installed_portable_runtime"},
        "remaining_gates": ["10000_recorded_observation_normalized_input_action_parity",
            "reference_joint_target_native_drive_force_parity", "paired_physical_rollouts",
            "accepted_runtime_driver_and_simulator_qualification", "held_out_full_cycle_acceptance"]}
    write_json(staging/"manifest.json", export)
    result = verify_tracker(staging, manifest_sha256=sha256_file(staging/"manifest.json"))
    os.rename(staging, destination)
    return {**result, "path": str(destination)}


def verify_tracker(root: Path, *, manifest_sha256: str) -> dict:
    """Verify the exact complete file inventory without executing pickle/JIT."""
    root = root.resolve(strict=True)
    if sha256_file(_member(root, "manifest.json")) != manifest_sha256:
        raise ValueError("Export manifest hash mismatch")
    manifest = _json(root/"manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "packaged"
            or manifest.get("qualification") not in LABELS
            or manifest.get("trained_policy") != (manifest["qualification"] == "trained_candidate")
            or any(manifest.get(key) is not False for key in ("acceptance_qualified", "portability_qualified",
                                                           "coupled_runtime_qualified", "promoted"))):
        raise ValueError("Export qualification labels are invalid")
    rows = manifest.get("files", {})
    if not rows:
        raise ValueError("Export inventory is empty")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Export contains a nonportable symlink")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    if actual != set(rows) | {"manifest.json"}:
        raise ValueError("Export inventory contains missing or untracked files")
    for relative, row in rows.items():
        path = _member(root, relative)
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Export member hash/size mismatch: {relative}")
    required = {"policy/policy_jit.pt", "policy/policy.onnx", "policy/normalization.json",
                "learner/manifest.json", "learner/rank_00.pt", "references/manifest.json",
                "planner/manifest.json", "domain/manifest.json", "interfaces/contracts.json",
                "interfaces/controller.json", "interfaces/source_pins.json", "dependencies/runtime_pins.json",
                "evidence/synthetic_export_parity.json"}
    if not required.issubset(rows):
        raise ValueError("Export omits required supporting components")
    # Cross-check the identities INSIDE copied artifacts too. This also detects
    # an input changed between its initial validation and the copy operation.
    learner = _json(root/"learner/manifest.json")
    if (sha256_file(root/"learner/manifest.json") != manifest["checkpoint_manifest_sha256"]
            or learner.get("contract_sha256") != manifest["checkpoint_contract_sha256"]):
        raise ValueError("Export full learner identity mismatch")
    contract = CheckpointContract(**learner["contract"])
    if contract.sha256 != learner["contract_sha256"]:
        raise ValueError("Export full learner contract digest mismatch")
    assert_checkpoint_rank_parity(learner["ranks"])
    for rank in learner["ranks"]:
        if sha256_file(_member(root/"learner", rank["path"])) != rank["sha256"]:
            raise ValueError("Export full learner rank payload mismatch")
    bundle = load_runtime_bundle(root/"references")
    if (sha256_file(bundle.root/"manifest.json") != contract.reference_manifest_sha256
            or contract.reference_manifest_sha256 != manifest["reference_manifest_sha256"]
            or sha256_file(bundle.model_path) != contract.body_sha256
            or bundle.controller_sha256() != contract.controller_sha256
            or _digest(_json(root/"interfaces/controller.json")) != contract.controller_sha256):
        raise ValueError("Export reference/body/controller identity mismatch")
    interfaces = _json(root/"interfaces/contracts.json")
    for key, field in INTERFACES.items():
        if _digest(interfaces[key]) != getattr(contract, field):
            raise ValueError("Export interface identity mismatch")
    source_hashes = {name.removeprefix("sources/"): row["sha256"]
                     for name, row in rows.items() if row["role"] == "training_implementation_source"}
    if _digest(source_hashes) != contract.implementation_sources_sha256:
        raise ValueError("Export implementation source identity mismatch")
    _planner(root/"planner/manifest.json")
    domain = _domain(root/"domain/manifest.json", bundle, trained=manifest["trained_policy"])
    if domain["reference_generator_sha256"] != sha256_file(root/"planner/manifest.json"):
        raise ValueError("Export planner/domain identity mismatch")
    if manifest["trained_policy"]:
        bundle.require_training_qualification()
        if "evidence/training.json" not in rows:
            raise ValueError("Trained export lacks native learning evidence")
    return {"schema": "human_wbc_tracker_integrity_v1", "status": "passed",
            "manifest_sha256": manifest_sha256, "file_count": len(rows),
            "qualification": manifest["qualification"], "portability_qualified": False,
            "acceptance_qualified": False, "total_bytes": sum(row["bytes"] for row in rows.values())}
