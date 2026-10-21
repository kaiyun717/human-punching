"""Execution dependencies are exact; obsolete ancestry paths remain opaque."""
import copy
import json
from pathlib import Path

import pytest

from human_punching import phase2_target_inputs as inputs
from human_punching.wbc.provenance import sha256_file
from test_phase2_target_catalog import example, encoded


@pytest.fixture
def target_plan(example, monkeypatch):
    root, plan, parent, target = example
    monkeypatch.setattr(inputs, "PROJECT_ROOT", root)
    model = Path(parent["source_model_path"])
    model.write_text('<mujoco><worldbody/></mujoco>')
    parent["source_model_sha256"] = sha256_file(model)
    plan["parent_catalog"] = encoded(root/"parents2.json", {
        "schema": "task10b_phase2_reference_catalog_v1", "cells": [parent]})
    recipe = json.loads(Path(plan["recipe"]["path"]).read_text())
    recipe["parent_catalog_sha256"] = plan["parent_catalog"]["sha256"]
    plan["recipe"] = encoded(root/"recipe2.json", recipe)
    plan["body_metadata"] = encoded(root/"body2.json", {
        "xml_path": "/home/gridsan/nonexistent/human.xml",
        "provenance": {"path": "/never/read/ancestor.pdf", "sha256": "0"*64}})
    path = root/"plan.json"
    path.write_text(json.dumps(plan))
    return root, path, plan, parent


def test_exact_payloads_and_opaque_body_ancestry(target_plan):
    root, path, plan, parent = target_plan
    closure = inputs.files(path)
    expected = {path, *[Path(plan[k]["path"]) for k in
        ("domain", "target_geometry", "parent_catalog", "body_metadata", "recipe")],
        Path(parent["reference_path"]), Path(parent["source_model_path"])}
    assert closure == {str(p.relative_to(root)): sha256_file(p) for p in sorted(expected)}
    assert len(closure) == 8


@pytest.mark.parametrize("key", ["domain", "target_geometry", "parent_catalog", "body_metadata", "recipe"])
def test_bound_metadata_change_rejected(target_plan, key):
    _, path, plan, _ = target_plan
    Path(plan[key]["path"]).write_text('{}')
    with pytest.raises(ValueError, match="hash"):
        inputs.files(path)


@pytest.mark.parametrize("key", ["reference_path", "source_model_path"])
def test_runtime_payload_change_rejected(target_plan, key):
    _, path, _, parent = target_plan
    Path(parent[key]).write_bytes(b"different")
    with pytest.raises(ValueError, match="hash"):
        inputs.files(path)


def test_hash_bound_external_plan_cannot_escape_owned_root(target_plan):
    root, path, _, _ = target_plan
    external = root.parent/(root.name+"_external.json")
    external.write_bytes(path.read_bytes())
    link = root/"escaped_plan.json"
    link.symlink_to(external)
    with pytest.raises(ValueError, match="owned"):
        inputs.files(link)


def test_unclosed_model_asset_rejected(target_plan):
    root, path, plan, parent = target_plan
    model = Path(parent["source_model_path"])
    model.write_text('<mujoco><include file="unbound.xml"/></mujoco>')
    parent["source_model_sha256"] = sha256_file(model)
    plan["parent_catalog"] = encoded(root/"parents_asset.json", {
        "schema": "task10b_phase2_reference_catalog_v1", "cells": [parent]})
    recipe = json.loads(Path(plan["recipe"]["path"]).read_text())
    recipe["parent_catalog_sha256"] = plan["parent_catalog"]["sha256"]
    plan["recipe"] = encoded(root/"recipe_asset.json", recipe)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="self-contained"):
        inputs.files(path)


def test_shared_plan_cannot_inherit_standalone_closure(target_plan):
    _, path, plan, _ = target_plan
    plan["schema"] = "task10b_phase2_shared_target_catalog_plan_v1"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        inputs.files(path)


@pytest.mark.parametrize("change", ["split", "replica", "target"])
def test_no_split_or_target_substitution(target_plan, change):
    root, path, plan, _ = target_plan
    if change == "split":
        plan["split"] = "validation"
    else:
        geometry = json.loads(Path(plan["target_geometry"]["path"]).read_text())
        row = geometry["geometry_candidates"][0]
        if change == "replica": row["original_replica_id"] = 1
        else: row["head_target_w"][0] += .01
        plan["target_geometry"] = encoded(root/"changed_geometry.json", geometry)
        recipe = json.loads(Path(plan["recipe"]["path"]).read_text())
        recipe["target_geometry_sha256"] = plan["target_geometry"]["sha256"]
        plan["recipe"] = encoded(root/"changed_recipe.json", recipe)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="binding"):
        inputs.files(path)


@pytest.fixture
def projected_plan(target_plan, monkeypatch):
    from human_punching import phase2_timing_control as timing
    root, path, plan, parent = target_plan
    expected = json.loads(Path(plan["parent_catalog"]["path"]).read_text())
    original = encoded(root/"native_catalog.json", expected)
    packet = encoded(root/"packet.json", {"mode": "timing-references", "old_path": "/not/traversed"})
    execution = encoded(root/"execution.json", {"packet_sha256": packet["sha256"]})
    probe = encoded(root/"probe.json", {"compute_root": "/opaque/native/mount"})
    nonce = encoded(root/"nonce.json", {"nonce": "fresh"})
    meta = {"alias_probe": {"relative_name": "probe.json", "sha256": probe["sha256"]},
            "generation_execution": {"relative_name": "execution.json", "sha256": execution["sha256"]},
            "nonce": {"path": "nonce.json", "sha256": nonce["sha256"]},
            "mapped_files": [{"path": "source.npz", "sha256": parent["reference_sha256"]}]}
    projected = dict(expected, unpacked_catalog_path="native_catalog.json",
        unpacked_catalog_sha256=original["sha256"], timing_control_projection=meta)
    plan["parent_catalog"] = encoded(root/"projection.json", projected)
    recipe = json.loads(Path(plan["recipe"]["path"]).read_text())
    recipe["parent_catalog_sha256"] = plan["parent_catalog"]["sha256"]
    plan["recipe"] = encoded(root/"projection_recipe.json", recipe)
    path.write_text(json.dumps(plan))
    calls = []
    def projection(a, b, c):
        calls.append((a, b, c))
        return copy.deepcopy(expected), copy.deepcopy(meta)
    monkeypatch.setattr(timing, "projection", projection)
    return root, path, plan, expected, meta, calls


def test_exact_projection_receipts_are_bound_without_historical_walk(projected_plan):
    root, path, _, _, _, calls = projected_plan
    closure = inputs.files(path)
    assert calls == [(root/"native_catalog.json", root/"probe.json", root/"execution.json")]
    assert set(("native_catalog.json", "packet.json", "execution.json", "probe.json", "nonce.json")) <= set(closure)


def test_projection_scientific_metadata_mismatch_rejected(projected_plan):
    _, path, _, expected, _, _ = projected_plan
    expected["extra_scientific_claim"] = True
    with pytest.raises(ValueError, match="projection changed"):
        inputs.files(path)


@pytest.mark.parametrize("name", ["native_catalog.json", "packet.json", "execution.json", "probe.json", "nonce.json"])
def test_projection_evidence_mutation_rejected(projected_plan, name):
    root, path, _, _, _, _ = projected_plan
    (root/name).write_text('{}')
    with pytest.raises(ValueError, match="SHA changed"):
        inputs.files(path)
