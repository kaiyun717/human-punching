"""Exact64/24 snapshot catalogs using the unchanged .205rad sequential solver.

The measured four-cell probe permits candidate construction only. Its remaining
support/effort/correction-step flags remain diagnostics requiring native control
evaluation before learning. No solver, reference, target or split is substituted.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from . import phase2_target_catalog as original
from . import phase2_sequential_target_catalog_v3 as probe
from .phase2_target_reference import RetargetSettings
from .phase2_target_path_reference_v3 import SequentialSettings, VERSION
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

SCHEMA = "task10b_phase2_sequential_full_catalog_plan_v1"
ENTRY_VERSION = "task10b_phase2_sequential_full_catalog_v1"
LIMITS = {"train": (64, 480, 420, 360), "validation": (24, 300, 240, 180)}
OUTPUT_CAPS = {"train": 256*1024**2, "validation": 128*1024**2}
SOURCES = (*probe.SOURCES, "src/human_punching/phase2_sequential_full_catalog_v1.py")
EVIDENCE = ("probe_plan", "probe_audit", "probe_manifest", "probe_packet", "probe_execution", "probe_accounting")
own = original.owned_input


def binding(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Owned full-catalog input required")
    return dict(path=str(path), sha256=sha256_file(path))


def source_bindings():
    return {name:binding(PROJECT_ROOT/name) for name in SOURCES}


def read(record):
    return json.loads(own(record).read_text())


def validate_evidence(plan):
    data = {name:read(plan[name]) for name in EVIDENCE}
    prior = data["probe_plan"]
    probe.validate_plan(prior)
    audit, manifest = data["probe_audit"], data["probe_manifest"]
    packet, execution, accounting = (data[k] for k in ("probe_packet", "probe_execution", "probe_accounting"))
    if (audit.get("schema") != "task10b_phase2_saved_target_reference_independent_audit_v1"
            or audit.get("passed") is not True or audit.get("physics_steps") != 0
            or audit.get("packet_sha256") != plan["probe_packet"]["sha256"]
            or audit.get("manifest_sha256") != plan["probe_manifest"]["sha256"]
            or manifest.get("plan_sha256") != plan["probe_plan"]["sha256"]
            or manifest.get("path_solver_version") != VERSION
            or manifest.get("complete") is not True or manifest.get("exception") is not None
            or any(manifest.get(k) != 4 for k in ("planned_cells", "executed_cells", "generated_cells", "admitted_candidate_cells"))
            or manifest.get("unattempted_cells") != []
            or packet.get("schema") != "task10b_phase2_sequential_reference_packet_v3"
            or execution.get("status") != "executed" or execution.get("returncode") != 0
            or accounting.get("state") != "COMPLETED"
            or execution.get("packet_sha256") != plan["probe_packet"]["sha256"]
            or accounting.get("packet_sha256") != plan["probe_packet"]["sha256"]
            or execution.get("job_id") != accounting.get("job_id")
            or audit.get("job_id") != execution.get("job_id")):
        raise ValueError("Complete counted and independently audited four-cell .205 probe required")
    ids = {r["candidate_id"] for r in prior["cells"]}
    if ({r.get("candidate_id") for r in manifest.get("cells", [])} != ids
            or len(audit.get("cells", [])) != 4
            or {r.get("candidate_id") for r in audit["cells"]} != ids
            or any(r.get("all_checks_passed") is not True for r in audit["cells"])):
        raise ValueError("All four probe identities and independent checks must remain")
    return prior


def validate_plan(plan):
    split = plan.get("split")
    if split not in LIMITS:
        raise ValueError("Only full training or validation construction is permitted")
    count, _, _, child = LIMITS[split]
    if (plan.get("schema") != SCHEMA or plan.get("entry_version") != ENTRY_VERSION
            or plan.get("family") != "jab" or plan.get("path_solver_version") != VERSION
            or plan.get("maximum_cells") != count or len(plan.get("cells", [])) != count
            or plan.get("maximum_child_seconds") != child
            or plan.get("output_byte_cap") != OUTPUT_CAPS[split]
            or plan.get("physical_qualification") is not False
            or plan.get("native_frozen_control_required_before_learning") is not True
            or set(plan.get("sources", {})) != set(SOURCES)):
        raise ValueError("Exact64/24 finite candidate-only .205 contract required")
    for name, value in plan["sources"].items():
        if own(value) != (PROJECT_ROOT/name).resolve(strict=True):
            raise ValueError("Full catalog source must resolve to its exact owned module")
    settings = SequentialSettings(RetargetSettings(**plan["retarget_settings"]))
    prior = validate_evidence(plan)
    for key in ("domain", "target_geometry", "parent_catalog", "body_metadata", "retarget_settings"):
        if plan[key] != prior[key]:
            raise ValueError("The reviewed probe's exact geometry, source and solver settings are fixed")
    if set(plan.get("original_plans", {})) != {"train", "validation"}:
        raise ValueError("Both original64/24 declarations must be retained")
    for name, item in plan["original_plans"].items():
        old = read(item)
        original.validate_plan(old)
        if old["split"] != name or old["family"] != "jab" or len(old["cells"]) != LIMITS[name][0]:
            raise ValueError("Original full split declarations cannot shrink or be resampled")
        for key in ("domain", "target_geometry", "parent_catalog", "body_metadata"):
            if old[key] != plan[key]:
                raise ValueError("Original split source identities changed")
    old = read(plan["original_plans"][split])
    expected = [dict(row, candidate_id=row["candidate_id"]+"_sequential_v3_full") for row in old["cells"]]
    if plan["cells"] != expected:
        raise ValueError("Exact declared targets/timings/order required, including every prior failure")
    common = deepcopy(plan); common["schema"] = original.SCHEMA
    original.validate_plan(common)
    return settings, common


def resolve_cells(plan):
    settings, common = validate_plan(plan)
    _, cells, paths = original.resolve_cells(common)
    recipe = json.loads(paths["recipe"].read_text())
    if (recipe.get("full_catalog_entry_version") != ENTRY_VERSION
            or recipe.get("path_solver_version") != VERSION
            or recipe.get("solver_sources") != plan["sources"]
            or recipe.get("original_plans") != plan["original_plans"]
            or any(recipe.get(key) != plan[key] for key in EVIDENCE)
            or recipe.get("sequential_preference_weight") != .002
            or recipe.get("interior_training_eligible") is not False
            or recipe.get("native_frozen_control_required_before_learning") is not True):
        raise ValueError("One exact common .205 recipe must bind both full splits and measured ancestry")
    domain = json.loads(paths["domain"].read_text())
    groups = [g["snapshot"]["command_id"] for g in domain["selected_command_groups"] if g["split"] == plan["split"]]
    timings = (.6, .75) if plan["split"] == "train" else (.6, .675, .75)
    actual = [(target["command_id"], parent["requested_ttc_s"]) for _, target, parent, _, _ in cells]
    if len(groups) != {"train":32, "validation":8}[plan["split"]] or actual != [(g,t) for g in groups for t in timings]:
        raise ValueError("Exact deterministic32/8 groups and predeclared timings required; no test or interior training")
    for _, target, parent, _, _ in cells:
        if target["placement"] != dict(forward_m=0., left_m=0., yaw_rad=0.):
            raise ValueError("Only the unchanged nominal registered placement is supported")
        interior = parent["requested_ttc_s"] == .675
        if interior and (plan["split"] != "validation" or parent.get("training_eligible") is not False):
            raise ValueError("Untrained interior reference is validation only")
    paths.update({key:own(plan[key]) for key in EVIDENCE})
    return settings, cells, paths


def _engine():
    """Private module instance; never mutates the frozen imported v3 module."""
    spec = importlib.util.spec_from_file_location("human_punching._sequential_full_engine_v1", Path(probe.__file__))
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    return engine


def generate(plan_path, output, *, wall_seconds):
    plan_path = own(binding(plan_path)); plan = json.loads(plan_path.read_text())
    resolve_cells(plan)
    if type(wall_seconds) not in (int, float) or wall_seconds != plan["maximum_child_seconds"]:
        raise ValueError("Exact split child wall required; no late truncation")
    engine = _engine()
    engine.resolve_cells = resolve_cells
    # Reuse the unchanged allocation/CUDA guard; the full wrapper independently
    # enforces its exact180/360 wall, beyond the old probe's240 maximum.
    engine.require_allocation = lambda value: probe.require_allocation(min(value, probe.MAX_CHILD_SECONDS))
    result = engine.generate(plan_path, output, wall_seconds=wall_seconds)
    folder = writable_path(output)
    write_json(folder/"full_catalog_entry.json", dict(schema=ENTRY_VERSION,
        plan_sha256=sha256_file(plan_path), wrapper_source_sha256=sha256_file(Path(__file__)),
        frozen_generator_source_sha256=sha256_file(Path(probe.__file__)),
        manifest_sha256=sha256_file(folder/"manifest.json"), split=plan["split"],
        child_wall_seconds=wall_seconds, solver_changed=False,
        physical_qualification=False, native_frozen_control_required_before_learning=True))
    return result


def create_plans(probe_plan_path, audit_path, train_path, validation_path, output):
    """Metadata only: retain every declared cell and write one common recipe."""
    prior = json.loads(Path(probe_plan_path).read_text()); probe.validate_plan(prior)
    audit = json.loads(Path(audit_path).read_text())
    # The selected packet is obtained from the audit's exact manifest artifact.
    manifests = [PROJECT_ROOT/name for name,digest in audit["artifacts"].items()
                 if name.endswith('/run/manifest.json') and digest == audit["manifest_sha256"]]
    if len(manifests) != 1: raise ValueError("Unique audited generation manifest required")
    packet_dir = manifests[0].parent.parent
    evidence = {"probe_plan":binding(probe_plan_path), "probe_audit":binding(audit_path),
        "probe_manifest":binding(manifests[0]), "probe_packet":binding(packet_dir/"packet.json"),
        "probe_execution":binding(packet_dir/"execution.json"), "probe_accounting":binding(packet_dir/"terminal_accounting.json")}
    sources = source_bindings()
    originals = dict(train=binding(train_path), validation=binding(validation_path))
    recipe = read(prior["recipe"])
    for key in ("probe_only", "predeclared_decision", "purpose", "selection_rule", "selection_sha256"):
        recipe.pop(key, None)
    recipe.update(**evidence, original_plans=originals, solver_sources=sources,
        prior_probe_recipe=prior["recipe"], full_catalog_entry_version=ENTRY_VERSION,
        native_frozen_control_required_before_learning=True,
        purpose="Construct exact declared64train+24validation candidates with unchanged measured .205 solver; no physical qualification or learning authority from offline admission.",
        residual_diagnostics="Probe retained up to .05rad correction steps and2.479 effort ratio; do not claim smoothness/dynamic feasibility or suppress offline failures.")
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"recipe.json", recipe)
    products = {}
    for split in LIMITS:
        old = read(originals[split]); plan = deepcopy(old)
        plan.update(schema=SCHEMA, entry_version=ENTRY_VERSION, path_solver_version=VERSION,
            retarget_settings=prior["retarget_settings"], sources=sources, **evidence,
            original_plans=originals, recipe=binding(folder/"recipe.json"),
            maximum_child_seconds=LIMITS[split][3], output_byte_cap=OUTPUT_CAPS[split],
            physical_qualification=False, native_frozen_control_required_before_learning=True)
        plan["cells"] = [dict(row,candidate_id=row["candidate_id"]+"_sequential_v3_full") for row in old["cells"]]
        resolve_cells(plan)
        path=folder/(split+"_plan.json"); write_json(path,plan); products[split]=binding(path)
    return products


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    args=parser.parse_args(argv)
    result=generate(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps({k:result[k] for k in ('complete','planned_cells','executed_cells','generated_cells','exception')}))
    return 0 if result['complete'] and result['exception'] is None else 1


if __name__ == '__main__':raise SystemExit(main())
