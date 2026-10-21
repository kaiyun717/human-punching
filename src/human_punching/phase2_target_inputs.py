"""Exact inputs for isolated target construction, with opaque ancestor metadata.

This does not generate references or qualify a target domain. The caller also
retains its accepted release/runtime/source closure. Historical paths embedded
in body, geometry and domain provenance are not execution dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .phase2_target_catalog import resolve_cells
from .wbc.provenance import PROJECT_ROOT, sha256_file


def files(plan_path):
    """Return owned relative file/SHA bindings without walking JSON ancestry.

    Only the standalone family-guard target plan is supported. Referenced models
    must be self-contained, as the accepted SOMA model is; external XML assets
    require a separately reviewed dependency closure. No G1 model or actor is
    loaded. Projected timing parents retain their exact native-generation proof.
    """
    root = PROJECT_ROOT.resolve(strict=True)
    closure = {}

    def add(path, expected=None):
        path = Path(path)
        path = (path if path.is_absolute() else root/path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Target construction input must be an owned file")
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("Target construction input SHA changed")
        relative = str(path.relative_to(root))
        if closure.setdefault(relative, digest) != digest:
            raise ValueError("Target construction input changed during closure")
        return path

    plan_path = add(plan_path)
    plan = json.loads(plan_path.read_text())
    _, cells, paths = resolve_cells(plan)
    for key, path in paths.items():
        add(path, plan[key]["sha256"])
    for _, _, parent, source, model in cells:
        add(source, parent["reference_sha256"])
        model = add(model, parent["source_model_sha256"])
        if any("file" in node.attrib or node.tag == "include"
               for node in ET.fromstring(model.read_text()).iter()):
            raise ValueError("Target model must be self-contained; asset closure requires review")

    parent = json.loads(paths["parent_catalog"].read_text())
    if "timing_control_projection" in parent:
        # Read the existing projection contract, not generic historical paths.
        # projection() only checks saved metadata, hashes and mount inode proof;
        # it does not load a MuJoCo model or regenerate a timing reference.
        from .phase2_timing_control import projection
        meta = parent["timing_control_projection"]
        original = add(parent["unpacked_catalog_path"], parent["unpacked_catalog_sha256"])
        probe = add(meta["alias_probe"]["relative_name"], meta["alias_probe"]["sha256"])
        execution = add(meta["generation_execution"]["relative_name"], meta["generation_execution"]["sha256"])
        add(meta["nonce"]["path"], meta["nonce"]["sha256"])
        expected, receipt = projection(original, probe, execution)
        scientific = {k: v for k, v in parent.items() if k not in
                      ("unpacked_catalog_path", "unpacked_catalog_sha256", "timing_control_projection")}
        if meta != receipt or scientific != expected:
            raise ValueError("Target parent projection changed scientific metadata or proof")
        for item in meta["mapped_files"]:
            add(item["path"], item["sha256"])
        execution_data = json.loads(execution.read_text())
        add(execution.parent/"packet.json", execution_data["packet_sha256"])

    # Hash again after all reads so a concurrent metadata/payload edit cannot
    # produce a closure mixing two identities.
    for relative, digest in closure.items():
        if sha256_file(root/relative) != digest:
            raise ValueError("Target input changed during closure")
    return dict(sorted(closure.items()))
