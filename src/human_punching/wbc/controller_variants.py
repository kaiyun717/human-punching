"""Explicit, unqualified controller experiments; immutable physical assets/caps.

Copies a reference bundle into a new directory and changes only named native-PD
gains. The source model and references remain identical. A new controller always
invalidates qualification and requires new native execution evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from .provenance import sha256_file, writable_path, write_json
from .runtime.bundle import load_runtime_bundle


def gain_variant(source: Path, output: Path, scales: dict[str, dict[str, float]], *, reason: str) -> dict:
    bundle = load_runtime_bundle(source)
    if not scales or not reason.strip():
        raise ValueError("Explicit named gain changes and a reason are required")
    body = deepcopy(bundle.body)
    schema = json.loads(bundle.path(bundle.manifest["controller_schema_path"]).read_text())
    if schema["joint_names"] != list(bundle.joint_names):
        raise ValueError("Legacy controller schema joint order differs from body")
    changes = []
    for name, factors in sorted(scales.items()):
        if name not in bundle.joint_names or set(factors) != {"kp", "kd"}:
            raise ValueError("Each named human joint requires explicit kp and kd factors")
        index = bundle.joint_names.index(name)
        before = {key: body[key][index] for key in ("kp", "kd")}
        for key, factor in factors.items():
            if isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
                raise ValueError("Gain factors must be positive finite numbers")
            body[key][index] *= float(factor)
            schema[key][index] = body[key][index]
        changes.append({"joint": name, "before": before,
                        "after": {key: body[key][index] for key in ("kp", "kd")}})
    destination = writable_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    if any(path.is_symlink() for path in bundle.root.rglob("*")):
        raise ValueError("Source diagnostic bundles must not contain symlinks")
    shutil.copytree(bundle.root, destination)
    manifest = deepcopy(bundle.manifest)
    for field, payload in (("body_metadata_path", body), ("controller_schema_path", schema)):
        path = destination / manifest[field]
        path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        manifest["asset_sha256"][path.name] = sha256_file(path)
    manifest.update(qualified_for_training=False,
        reason="Unqualified explicit controller-gain experiment; all physical and training gates reopened",
        legacy_controller_schema_role="Historical Task09 observation metadata; gains updated for this experiment, never the Task10 actor observation contract")
    manifest.pop("qualification_evidence", None)
    for clip in manifest["clips"]:
        clip["common_guard_validated"] = False
    record = {"schema": "human_wbc_controller_variant_v1", "reason": reason,
        "source_manifest_sha256": sha256_file(bundle.root/"manifest.json"),
        "source_controller_sha256": bundle.controller_sha256(), "gain_changes": changes,
        "physical_xml_sha256": sha256_file(bundle.model_path), "physical_model_changed": False,
        "effort_limits_changed": False, "references_changed": False,
        "root_actuated": False, "physics_executed": False, "qualified_for_training": False}
    record_path = write_json(destination/"controller_change.json", record)
    manifest["controller_change"] = {"path": record_path.name, "sha256": sha256_file(record_path)}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    changed = load_runtime_bundle(destination)
    if changed.body["effort"] != bundle.body["effort"] or sha256_file(changed.model_path) != sha256_file(bundle.model_path):
        raise RuntimeError("Controller variant changed the physical model or force limits")
    return {"path": str(destination), "manifest_sha256": sha256_file(destination/"manifest.json"),
            "controller_sha256": changed.controller_sha256(), **record}
