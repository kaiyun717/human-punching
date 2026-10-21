"""Explicit saved-byte composition of the audited RH54-prefix and exact10 suffix.

No reference arrays are loaded or rewritten. Only explicitly listed row path
fields may change spelling; all other native manifest values remain identical.
This module is intentionally independent of the bound native launch driver.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .wbc.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
VERSION = "task10b_phase2_right_hook_composite_catalog_path_projection_v1"
CATALOG = "task10b_phase2_reference_catalog_v1"
PATH_FIELDS = frozenset(("reference_path", "legacy_runtime_path", "source_model_path",
    "diagnostic_screen_path", "generation_failure_path", "retained_partial_reference_path"))
ANNOTATIONS = frozenset(("unpacked_catalog_path", "unpacked_catalog_sha256", "right_hook_composite_catalog_projection"))
GENERATOR = "src/human_punching/phase2_sequential_target_catalog_v3.py"
SOLVER = "src/human_punching/phase2_target_path_reference_v3.py"
FULL_WRAPPER = "src/human_punching/phase2_right_hook_full_catalog_v1.py"
FULL_DRIVER = "scripts/task10b_phase2_right_hook_full_v1.py"
SOLVER_VERSION = "task10b_sequential_active_arm_target_path_v3"
FULL_SCHEMA = "task10b_phase2_right_hook_full_packet_v1"
FULL_EVIDENCE = ("timing_selection","cache_proposal","probe_plan","probe_audit","probe_manifest",
    "probe_packet","probe_execution","probe_accounting","interior_plan","interior_audit",
    "interior_packet","interior_execution","interior_accounting")
AUDITOR = "outputs/task10b/p2rhfullauditprep01/audit_saved_right_hook_full.py"
SUFFIX_AUDITOR = "outputs/task10b/p2rhsuffixauditprep02/audit_saved_right_hook_suffix_v2.py"
SUFFIX_WRAPPER = "src/human_punching/phase2_right_hook_suffix_catalog_v2.py"
SUFFIX_SCHEMA = "task10b_phase2_right_hook_suffix_packet_v2"
COMPOSITE_SCHEMA = "task10b_phase2_right_hook_composite_reference_history_v1"
PRIOR_KEYS = ("prior_plan", "prior_manifest", "prior_packet", "prior_execution", "prior_accounting", "prior_audit")


def _protocol(packet, kind):
    if packet.get("family") != "right_hook" or packet.get("sequential_split") != "train":
        raise ValueError("Only the exact RH training prefix/suffix protocol is supported")
    if packet.get("native_frozen_control_required_before_learning") is not True:
        raise ValueError("Construction cannot grant native qualification or learning credit")
    if kind == "prior" and packet.get("schema") == FULL_SCHEMA:
        return dict(plan_schema="task10b_phase2_right_hook_full_catalog_plan_v1", count=64,
            executed=55, split="train", allocation=420, application=360, child=300, cap=256*1024**2,
            module="human_punching.phase2_right_hook_full_catalog_v1", driver=FULL_DRIVER,
            wrapper=FULL_WRAPPER, mode="right-hook-full-target-references-v1",
            entry_file="right_hook_full_entry.json", entry_schema="task10b_phase2_right_hook_full_catalog_v1",
            auditor=AUDITOR, audit_schema="task10b_phase2_rh_full_saved_target_reference_audit_v1",
            status="failed", returncode=1, state="FAILED", exit_code="2:0", extra_inputs=list(FULL_EVIDENCE))
    if kind == "suffix" and packet.get("schema") == SUFFIX_SCHEMA:
        return dict(plan_schema="task10b_phase2_right_hook_suffix_catalog_plan_v2", count=10,
            executed=10, split="train", allocation=180, application=150, child=90, cap=64*1024**2,
            module="human_punching.phase2_right_hook_suffix_catalog_v2",
            driver="scripts/task10b_phase2_right_hook_suffix_v2.py", wrapper=SUFFIX_WRAPPER,
            mode="right-hook-suffix-target-references-v2", entry_file="right_hook_suffix_entry_v2.json",
            entry_schema="task10b_phase2_right_hook_suffix_catalog_v2", auditor=SUFFIX_AUDITOR,
            audit_schema="task10b_phase2_rh_suffix_saved_target_reference_audit_v2",
            status="executed", returncode=0, state="COMPLETED", exit_code="0:0",
            extra_inputs=[*FULL_EVIDENCE, *PRIOR_KEYS])
    raise ValueError("Explicit original partial-v1 and corrected suffix-v2 packet schemas required")


def _semantic_files(plan_path, kind):
    if kind == "prior":
        from .phase2_right_hook_full_catalog_v1 import files
    elif kind == "suffix":
        from .phase2_right_hook_suffix_catalog_v2 import files
    else:
        raise ValueError("Unknown component")
    return files(plan_path)


def _resolve_suffix(plan):
    from .phase2_right_hook_suffix_catalog_v2 import resolve_cells
    resolve_cells(plan)


def _relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != str(value) or str(path) == ".":
        raise ValueError("Exact root-relative member required")
    return path


def _owned(value):
    root = ROOT.resolve(strict=True)
    path = Path(value)
    path = (path if path.is_absolute() else root/_relative(value)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Projection input must be an owned file")
    return path


def _read(path):
    return json.loads(_owned(path).read_text())


def _binding(path):
    path = _owned(path)
    return {"relative_name": str(path.relative_to(ROOT.resolve())), "sha256": sha256_file(path)}


def _bound(item):
    path = _owned(item["relative_name"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Projection evidence SHA changed")
    return path


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _component(catalog_path, execution_path, probe_path, *, saved_audit_path, kind):
    """Verify the real state of one immutable counted component, without relabeling it."""
    root = ROOT.resolve(strict=True)
    source_path, execution_path, probe_path = map(_owned, (catalog_path, execution_path, probe_path))
    source, execution, probe = map(_read, (source_path, execution_path, probe_path))
    packet_path = _owned(execution_path.parent/"packet.json")
    packet = _read(packet_path)
    packet_sha = sha256_file(packet_path)
    protocol = _protocol(packet, kind)
    if (packet.get("mode") != protocol["mode"] or packet.get("gpus") != 1
            or packet.get("training") is not False or packet.get("fresh_transition_cap") != 0
            or any(packet.get(k) is not False for k in
                   ("automatic_followup", "promotion", "g1_actor_execution", "combined_scene"))
            or execution.get("schema") != "task10b_phase2_execution_v1"
            or execution.get("status") != protocol["status"] or execution.get("returncode") != protocol["returncode"]
            or execution.get("native_processes_started") != 1
            or execution.get("packet_sha256") != packet_sha
            or not isinstance(execution.get("job_id"), str) or not execution["job_id"].isdigit()
            or source_path != _owned(root/_relative(packet["output"])/"manifest.json")
            or source_path.parent.parent != packet_path.parent):
        raise ValueError("Exact counted partial/suffix generation state required")
    wall, app = packet["allocation_wall_seconds"], packet["application_wall_seconds"]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (wall, app)) or not 120 < app <= wall-30 <= 14400-30:
        raise ValueError("Bounded allocation/application limits required")
    if (wall != protocol["allocation"] or app != protocol["application"]
            or packet.get("sequential_child_wall_seconds") != protocol["child"]
            or packet.get("sequential_engineering_correction_cap_rad") != .205
            or packet.get("physical_joint_limits_relaxed") is not False
            or packet.get("sequential_path_solver_version") != SOLVER_VERSION):
        raise ValueError("Exact counted sequential recipe and split walls required")
    terminal_path = _owned(packet_path.parent/"terminal_accounting.json")
    terminal = _read(terminal_path)
    if (terminal.get("schema") != "task10b_phase2_terminal_accounting_v1"
            or terminal.get("state") != protocol["state"] or terminal.get("exit_code") != protocol["exit_code"]
            or terminal.get("packet_sha256") != packet_sha or terminal.get("job_id") != execution["job_id"]
            or not 0 < terminal.get("allocated_gpu_seconds", 0) <= wall
            or terminal.get("allocated_tres", {}).get("gres/gpu") != "1"):
        raise ValueError("Exact terminal one-GPU accounting required")

    files = packet.get("files", {})
    if not files or any(n not in files for n in (GENERATOR, SOLVER, protocol["driver"])):
        raise ValueError("Archived target generator and CLI identities required")
    archive = packet_path.parent/"source_snapshot"
    archived = {}
    total = 0
    for name, digest in files.items():
        name = _relative(name)
        path = _owned(archive/name)
        if not path.is_relative_to(archive.resolve(strict=True)):
            raise ValueError("Source archive member escaped its packet")
        total += path.stat().st_size
        if total > 3*1024**3 or sha256_file(path) != digest:
            raise ValueError("Archived packet input changed or exceeded its size bound")
        archived[str(name)] = path

    logical, compute = Path(probe["local_root"]), Path(probe["compute_root"])
    if (probe.get("schema") != "task10b_phase2_host_mount_alias_probe_v1" or probe.get("passed") is not True
            or probe.get("fresh_owned_nonce_visible_identically_through_both_mounts") is not True
            or not logical.is_absolute() or not compute.is_absolute() or logical == compute
            or ".." in logical.parts or ".." in compute.parts or root not in (logical, compute)
            or execution.get("runtime_path_context", {}).get("root") != str(compute)):
        raise ValueError("Exact two-root nonce proof and native execution root required")
    proof_relative = str(probe_path.relative_to(root))
    proof_binding = {"path": proof_relative, "sha256": sha256_file(probe_path)}
    if (packet.get("runtime_namespace_proof") != proof_binding
            or execution.get("runtime_namespace_proof") != proof_binding
            or files.get(proof_relative) != proof_binding["sha256"]):
        raise ValueError("Generation packet did not bind this namespace proof")

    def same_file(relative, expected=None):
        relative = _relative(relative)
        owned = _owned(relative)
        left, right = logical/relative, compute/relative
        if (not left.is_file() or not right.is_file()
                or left.stat().st_ino != right.stat().st_ino or left.stat().st_ino != owned.stat().st_ino):
            raise ValueError("Projected aliases do not name the same owned inode")
        digest = sha256_file(owned)
        if sha256_file(left) != digest or sha256_file(right) != digest or (expected is not None and digest != expected):
            raise ValueError("Projected aliases or declared file SHA differ")
        return owned, digest, owned.stat().st_ino

    nonce_path = _owned(probe_path.parent/"nonce.json")
    nonce_relative = str(nonce_path.relative_to(root))
    same_file(nonce_relative, probe["nonce_sha256"])
    nonce = _read(nonce_path)
    if (files.get(nonce_relative) != probe["nonce_sha256"]
            or nonce.get("schema") != "task10b_phase2_host_mount_alias_nonce_v1"
            or nonce.get("root") != str(logical) or nonce.get("compute_root") != str(compute) or not nonce.get("nonce")):
        raise ValueError("Fresh namespace nonce identity differs")
    for item in probe["bindings"]:
        _, digest, inode = same_file(item["relative_path"], item["local_sha256"])
        if digest != item["compute_mount_sha256"] or inode != item["local_inode"] or inode != item["compute_inode"]:
            raise ValueError("Reviewed alias binding changed")

    plan_path = _owned(packet["inputs"]["plan"])
    plan_relative = str(plan_path.relative_to(root))
    plan = _read(archived[plan_relative])
    if (sha256_file(plan_path) != files[plan_relative] or source.get("plan_sha256") != files[plan_relative]
            or plan.get("schema") != protocol["plan_schema"]
            or plan.get("start_mode") != "family_guard" or type(plan.get("maximum_cells")) is not int
            or len(plan.get("cells", [])) != protocol["count"] or plan["maximum_cells"] != protocol["count"]
            or plan.get("family") != "right_hook" or plan.get("split") != protocol["split"]
            or plan.get("path_solver_version") != SOLVER_VERSION
            or plan.get("maximum_child_seconds") != protocol["child"]
            or plan.get("output_byte_cap") != protocol["cap"]
            or source.get("source_sha256") != files[GENERATOR]):
        raise ValueError("Exact archived standalone target plan and generator required")
    for key in ("domain", "target_geometry", "parent_catalog", "body_metadata", "recipe", *protocol["extra_inputs"]):
        path = _owned(plan[key]["path"])
        relative = str(path.relative_to(root))
        if (sha256_file(path) != plan[key]["sha256"] or files.get(relative) != plan[key]["sha256"]
                or source.get("input_hashes", {}).get(key) != plan[key]["sha256"]):
            raise ValueError("Target/domain/recipe ancestry changed")
    settings=plan.get("retarget_settings",{})
    if (settings.get("maximum_joint_delta_rad") != .205 or settings.get("maximum_iterations") != 80
            or settings.get("active_position_tolerance_m") != .0005
            or settings.get("maximum_iteration_delta_rad") != .025
            or source.get("path_solver_version") != SOLVER_VERSION
            or source.get("output_byte_cap") != protocol["cap"]):
        raise ValueError("Unchanged reviewed .205 sequential engineering bounds required")
    if (source.get("schema") != CATALOG or any(k in source for k in ANNOTATIONS|{"target_catalog_projection","timing_control_projection","sequential_catalog_projection","family_catalog_projection"})
            or source.get("complete") is not (kind == "suffix")
            or (kind == "suffix" and source.get("exception") is not None)
            or source.get("planned_cells") != len(plan["cells"]) or source.get("executed_cells") != protocol["executed"]
            or len(source.get("cells", [])) != protocol["executed"]
            or source.get("unattempted_cells") != (plan["cells"][55:] if kind == "prior" else [])
            or any(source.get(k) is not False for k in ("physics_executed", "physical_qualification", "learned_trajectory_generator"))
            or source.get("recipe_sha256") != plan["recipe"]["sha256"]):
        raise ValueError("Exact unqualified native component dispositions required")
    if (source.get("generated_cells") != sum(r.get("generation_disposition") == "generated" for r in source["cells"])
            or source.get("admitted_candidate_cells") != sum(r.get("kinematic_admission") is True for r in source["cells"])):
        raise ValueError("Native generated/admitted counters disagree with the preserved rows")
    command = execution.get("native_command", [])
    expected_command = [str(compute/".venv-task10b/bin/python"), "-B", "-m", protocol["module"],
        "--plan", str(compute/plan_relative), "--output", str(compute/_relative(packet["output"])), "--wall-seconds"]
    if not isinstance(command, list) or command[:-1] != expected_command:
        raise ValueError("Recorded native target command changed")
    child = float(command[-1])
    if not math.isfinite(child) or child != protocol["child"]:
        raise ValueError("Recorded target child wall exceeded the packet")

    sidecar_path=_owned(source_path.parent/protocol["entry_file"])
    sidecar=_read(sidecar_path)
    if (protocol["wrapper"] not in files
            or sidecar.get("schema") != protocol["entry_schema"]
            or sidecar.get("plan_sha256") != files[plan_relative]
            or sidecar.get("wrapper_source_sha256") != files[protocol["wrapper"]]
            or sidecar.get("frozen_generator_source_sha256") != files[GENERATOR]
            or sidecar.get("manifest_sha256") != sha256_file(source_path)
            or sidecar.get("family") != "right_hook" or sidecar.get("engineering_cap_rad") != .205
            or sidecar.get("split") != "train" or sidecar.get("child_wall_seconds") != child
            or sidecar.get("solver_changed") is not False or sidecar.get("physical_qualification") is not False
            or sidecar.get("native_frozen_control_required_before_learning") is not True):
        raise ValueError("Explicit unchanged wrapper/generator sidecar required")
    if kind == "suffix":
        prior_plan=_read(_owned(plan["prior_plan"]["path"]))
        if (sidecar.get("ancestry") != {k:plan[k] for k in PRIOR_KEYS}
                or sidecar.get("recipe_sha256") != plan["recipe"]["sha256"]
                or sidecar.get("retained_prefix_ids") != [r["candidate_id"] for r in prior_plan["cells"][:54]]
                or sidecar.get("regenerated_original_ids") != [r["candidate_id"] for r in plan["cells"]]
                or sidecar.get("prior_outcomes_rewritten") is not False
                or sidecar.get("automatic_further_retry") is not False):
            raise ValueError("Suffix must bind unchanged prefix and original continuation IDs")
    component_sidecar=_binding(sidecar_path)
    audit_path=_owned(saved_audit_path);audit=_read(audit_path)
    audit_source=_owned(protocol["auditor"])
    if (audit.get("schema") != protocol["audit_schema"]
            or audit.get("passed") is not True or audit.get("family") != "right_hook"
            or audit.get("active_hand_index") != 1 or audit.get("split") != protocol["split"]
            or audit.get("job_id") != execution["job_id"] or audit.get("physics_steps") != 0
            or audit.get("project_reference_generation") is not False or audit.get("reference_generation") is not False
            or audit.get("source_sha256") != sha256_file(audit_source)
            or len(audit.get("cells",[])) != protocol["count"]
            or [r.get("candidate_id") for r in audit["cells"]] != [r["candidate_id"] for r in plan["cells"]]
            or any(r.get("all_checks_passed") is not True for r in audit["cells"])):
        raise ValueError("Exact independent saved RH full audit required")
    for key,path in dict(packet=packet_path,plan=plan_path,manifest=source_path,
                         execution=execution_path,accounting=terminal_path).items():
        if _owned(audit[key+"_path"]) != path or audit[key+"_sha256"] != sha256_file(path):
            raise ValueError("Saved audit input path or SHA differs")
    for key,field in dict(planned="planned_cells",executed="executed_cells",generated="generated_cells",
                          admitted="admitted_candidate_cells",unattempted="unattempted_cells").items():
        if audit.get("generation",{}).get(key) != source[field]:
            raise ValueError("Saved audit counters differ from retained native dispositions")
    if kind == "suffix" and (audit.get("historical_outcomes_preserved") is not True
            or audit.get("prior_inputs") != {k:plan[k] for k in PRIOR_KEYS}
            or audit.get("original_plan_suffix_start") != 54
            or audit.get("retained_prefix_ids") != sidecar["retained_prefix_ids"]):
        raise ValueError("Suffix audit must retain exact prior outcomes and inputs")
    projected = json.loads(json.dumps(source, allow_nan=False))
    changes = []
    for index, (row, declaration) in enumerate(zip(projected["cells"], plan["cells"][:protocol["executed"]], strict=True)):
        if (any(row.get(k) != v for k, v in declaration.items())
                or row.get("family") != plan["family"] or row.get("split") != plan["split"]
                or row.get("recipe_sha256") != source["recipe_sha256"]
                or row.get("physical_qualification") is not False
                or row.get("path_solver_version") != SOLVER_VERSION
                or row.get("generation_disposition") not in (("generated", "timeout") if kind == "prior" else ("generated", "failed"))
                or (row.get("generation_disposition") != "generated" and row.get("kinematic_admission") is not False)):
            raise ValueError("Native cell order, declaration or disposition changed")
        for key, value in list(row.items()):
            if not key.endswith("_path") or value is None:
                continue
            if key not in PATH_FIELDS:
                raise ValueError("Unreviewed target row pathname field")
            original = Path(value)
            if not original.is_absolute() or ".." in original.parts or str(original) != value or not original.is_relative_to(compute):
                raise ValueError("Only exact native compute-root pathname fields may change")
            relative = str(original.relative_to(compute))
            expected_sha = row.get(key[:-5]+"_sha256")
            if key in ("reference_path", "legacy_runtime_path", "source_model_path", "generation_failure_path") and not isinstance(expected_sha, str):
                raise ValueError("Declared runtime/reference/failure file SHA required")
            if key == "source_model_path" and files.get(relative) != expected_sha:
                raise ValueError("Native model is absent from the generation input archive")
            _, digest, inode = same_file(relative, expected_sha)
            row[key] = str(logical/relative)
            changes.append(dict(json_location=f"cells/{index}/{key}", old=value, new=row[key],
                relative_name=relative, sha256=digest, inode=inode))
    nonpaths = json.loads(json.dumps(source, allow_nan=False))
    for row in nonpaths["cells"]:
        for key in PATH_FIELDS:
            if key in row: row[key] = "<pathname preserved by explicit file binding>"
    receipt = dict(version=VERSION, saved_audit=_binding(audit_path), saved_auditor=_binding(audit_source), protocol=protocol, packet_schema=packet["schema"], component_entry=component_sidecar, kind=kind, native_catalog=_binding(source_path), generation_execution=_binding(execution_path),
        generation_packet=_binding(packet_path), terminal_accounting=_binding(terminal_path), alias_probe=_binding(probe_path),
        nonce=_binding(nonce_path), archived_plan=_binding(archived[plan_relative]),
        archived_generator=_binding(archived[GENERATOR]), verified_archive_files=len(files),
        verified_archive_files_sha256=_canonical_hash(files), archived_logical_bytes=total,
        helper_sha256=sha256_file(__file__), plan=_binding(plan_path), changes=changes, non_path_metadata_sha256=_canonical_hash(nonpaths),
        scientific_metadata_changed=False, reference_bytes_changed=False, physical_qualification=False, native_frozen_control_required_before_learning=True)
    return projected, receipt


def build_projection(catalog_path, execution_path, probe_path, *, saved_audit_path):
    """Compose final dispositions while retaining both real execution histories.

    The ordinary candidate-row schema is retained for the unchanged packer.
    Catalog completeness means 64 final saved dispositions are available; it
    does not describe either allocation as one successful 64-cell generation.
    """
    suffix, suffix_receipt = _component(catalog_path, execution_path, probe_path,
        saved_audit_path=saved_audit_path, kind="suffix")
    plan = _read(_bound(suffix_receipt["plan"]))
    _resolve_suffix(plan)
    prior, prior_receipt = _component(plan["prior_manifest"]["path"],
        plan["prior_execution"]["path"], probe_path,
        saved_audit_path=plan["prior_audit"]["path"], kind="prior")
    for key, receipt_key in dict(prior_plan="plan", prior_manifest="native_catalog",
            prior_packet="generation_packet", prior_execution="generation_execution",
            prior_accounting="terminal_accounting", prior_audit="saved_audit").items():
        if _binding(plan[key]["path"]) != prior_receipt[receipt_key]:
            raise ValueError("Suffix ancestry names a different counted prior component")
    declaration = _read(_bound(prior_receipt["plan"]))
    if (len(declaration["cells"]) != 64 or plan["cells"] != declaration["cells"][54:]
            or len(prior["cells"]) != 55 or len(suffix["cells"]) != 10
            or prior["unattempted_cells"] != declaration["cells"][55:]
            or any(row.get("generation_disposition") != "generated"
                or row.get("kinematic_admission") is not True for row in prior["cells"][:54])
            or prior["cells"][54].get("generation_disposition") != "timeout"
            or prior["cells"][54].get("details",{}).get("physical_failure") is not False
            or prior["exception"].get("type") != "BoundedExecutionInterruption"
            or prior["recipe_sha256"] != suffix["recipe_sha256"]
            or plan["recipe"] != declaration["recipe"]):
        raise ValueError("Only the exact54+timeout+9 unchanged-recipe continuation is supported")
    rows = prior["cells"][:54] + suffix["cells"]
    if ([r["candidate_id"] for r in rows] != [r["candidate_id"] for r in declaration["cells"]]
            or len({r["candidate_id"] for r in rows}) != 64):
        raise ValueError("Composite may not resample, duplicate or reorder original requests")
    history = dict(schema=COMPOSITE_SCHEMA, construction_kind="audited_partial_plus_exact_suffix",
        prior=dict(catalog=prior_receipt["native_catalog"], execution=prior_receipt["generation_execution"],
            accounting=prior_receipt["terminal_accounting"], audit=prior_receipt["saved_audit"],
            planned=64, executed=55, generated=54, admitted=54,
            execution_status="failed", scheduler_state="FAILED", scheduler_exit_code="2:0",
            exception=prior["exception"], timeout_row=prior["cells"][54],
            unattempted_cells=prior["unattempted_cells"]),
        suffix=dict(catalog=suffix_receipt["native_catalog"], execution=suffix_receipt["generation_execution"],
            accounting=suffix_receipt["terminal_accounting"], audit=suffix_receipt["saved_audit"],
            planned=10, executed=10, generated=suffix["generated_cells"],
            admitted=suffix["admitted_candidate_cells"], execution_status="executed",
            scheduler_state="COMPLETED", scheduler_exit_code="0:0", exception=suffix["exception"],
            original_request_ids=[r["candidate_id"] for r in plan["cells"]]),
        aggregate_native_attempts=65, distinct_original_requests=64,
        original_outcomes_rewritten=False, further_retry_authorized=False)
    origins = [dict(candidate_id=r["candidate_id"], component="prior" if i < 54 else "suffix",
                    component_row=i if i < 54 else i-54) for i,r in enumerate(rows)]
    view = dict(schema=CATALOG, family="right_hook", split="train", cells=rows,
        construction_kind=history["construction_kind"], construction_history=history,
        complete=True, completion_semantics="all64_original_requests_have_final_saved_dispositions_across_two_counted_jobs",
        planned_cells=64, resolved_cells=64, generated_cells=54+suffix["generated_cells"],
        admitted_candidate_cells=54+suffix["admitted_candidate_cells"],
        aggregate_native_attempts=65, final_unresolved_cells=[],
        recipe_sha256=prior["recipe_sha256"], declaration_plan=prior_receipt["plan"],
        path_solver_version=SOLVER_VERSION, physics_executed=False,
        physical_qualification=False, learned_trajectory_generator=False)
    receipt = dict(version=VERSION, components=dict(prior=prior_receipt,suffix=suffix_receipt),
        alias_probe=_binding(probe_path), row_origins=origins,
        original_request_ids=[r["candidate_id"] for r in declaration["cells"]],
        original54_rows_unchanged_except_proven_pathnames=True,
        suffix10_rows_unchanged_except_proven_pathnames=True,
        historical_timeout_and_unattempted_preserved=True,
        scientific_metadata_changed=False, reference_bytes_changed=False,
        catalog_history_version=COMPOSITE_SCHEMA, helper_sha256=sha256_file(__file__),
        composite_content_sha256=_canonical_hash(view), physical_qualification=False,
        native_frozen_control_required_before_learning=True)
    return view, receipt


def project(catalog_path, execution_path, probe_path, output, *, saved_audit_path):
    projected, receipt = build_projection(catalog_path, execution_path, probe_path,
                                         saved_audit_path=saved_audit_path)
    destination = Path(output).resolve()
    if not destination.is_relative_to(ROOT.resolve(strict=True)) or destination.exists():
        raise ValueError("Fresh owned composite projection output directory required")
    destination.mkdir(parents=True,exist_ok=False)
    projected["right_hook_composite_catalog_projection"] = receipt
    for name,data in (("projection.json",receipt),("manifest.json",projected)):
        with (destination/name).open("x") as stream:
            json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write("\n")
    validate(destination/"manifest.json")
    return _binding(destination/"manifest.json")


def validate(path):
    path=_owned(path);view=_read(path);receipt=view["right_hook_composite_catalog_projection"]
    suffix=receipt["components"]["suffix"]
    expected,expected_receipt=build_projection(_bound(suffix["native_catalog"]),
        _bound(suffix["generation_execution"]),_bound(receipt["alias_probe"]),
        saved_audit_path=_bound(suffix["saved_audit"]))
    if (receipt!=expected_receipt or _read(path.parent/"projection.json")!=receipt
            or {k:v for k,v in view.items() if k!="right_hook_composite_catalog_projection"}!=expected):
        raise ValueError("Composite projection changed rows, history, paths or evidence")
    return _binding(path)


def verify_projection(path):
    validate(path);view=_read(path)
    return view,view["right_hook_composite_catalog_projection"]


def files(path):
    """Exact two archives, two audits, original inputs, alias proof and all saved rows."""
    _,receipt=verify_projection(path);closure={}
    def add(value,expected=None):
        owned=_owned(value);digest=sha256_file(owned)
        if expected is not None and digest!=expected:raise ValueError("Composite dependency SHA changed")
        name=str(owned.relative_to(ROOT.resolve(strict=True)))
        if closure.setdefault(name,digest)!=digest:raise ValueError("Composite dependency changed")
        return owned
    add(path);add(_owned(path).parent/"projection.json")
    for kind,component in receipt["components"].items():
        for key in ("native_catalog","generation_execution","generation_packet","terminal_accounting",
                "alias_probe","nonce","archived_plan","archived_generator","saved_audit","saved_auditor",
                "component_entry","plan"):
            add(component[key]["relative_name"],component[key]["sha256"])
        packet_path=_bound(component["generation_packet"]);packet=_read(packet_path)
        for name,digest in packet["files"].items():add(packet_path.parent/"source_snapshot"/_relative(name),digest)
        plan_path=_bound(component["plan"])
        for name,digest in _semantic_files(plan_path,kind).items():add(name,digest)
        for item in _read(_bound(component["alias_probe"]))["bindings"]:
            add(item["relative_path"],item["local_sha256"])
        for item in component["changes"]:add(item["relative_name"],item["sha256"])
    for name,digest in closure.items():
        if sha256_file(_owned(name))!=digest:raise ValueError("Composite input changed during final closure")
    return dict(sorted(closure.items()))

