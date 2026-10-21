"""Metadata-only pathname view of a completed, counted family-probe generation run.

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
VERSION = "task10b_phase2_family_catalog_path_projection_v1"
CATALOG = "task10b_phase2_reference_catalog_v1"
PATH_FIELDS = frozenset(("reference_path", "legacy_runtime_path", "source_model_path",
    "diagnostic_screen_path", "generation_failure_path", "retained_partial_reference_path"))
ANNOTATIONS = frozenset(("unpacked_catalog_path", "unpacked_catalog_sha256", "family_catalog_projection"))
GENERATOR = "src/human_punching/phase2_sequential_target_catalog_v3.py"
SOLVER = "src/human_punching/phase2_target_path_reference_v3.py"
WRAPPER = "src/human_punching/phase2_family_target_probe_v1.py"
CLI = "scripts/task10b_phase2_family_targets_v1.py"
SOLVER_VERSION = "task10b_sequential_active_arm_target_path_v3"
PROBE_SCHEMA = "task10b_phase2_family_target_packet_v1"


def _protocol(packet):
    family=packet.get("family")
    if packet.get("schema") != PROBE_SCHEMA or family not in ("cross","left_hook","right_hook"):
        raise ValueError("Explicit cross/left-hook/right-hook family-probe packet required")
    return dict(plan_schema="task10b_phase2_family_target_probe_plan_v1", count=4,
        family=family,split="train",allocation=360,application=300,child=210,cap=128*1024**2,
        module="human_punching.phase2_family_target_probe_v1",driver=CLI,
        extra_inputs=["timing_selection","selected_d0_plan"])


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


def build_projection(catalog_path, execution_path, probe_path):
    """Check successful generation ancestry and return a pathname-only view.

    Failed/incomplete jobs are retained but not projected by this success-only
    helper. Complete catalogs may contain explicit geometric rejections, which
    remain unchanged and receive no new admission or qualification credit.
    """
    root = ROOT.resolve(strict=True)
    source_path, execution_path, probe_path = map(_owned, (catalog_path, execution_path, probe_path))
    source, execution, probe = map(_read, (source_path, execution_path, probe_path))
    packet_path = _owned(execution_path.parent/"packet.json")
    packet = _read(packet_path)
    packet_sha = sha256_file(packet_path)
    protocol = _protocol(packet)
    if (packet.get("mode") != "family-target-references" or packet.get("gpus") != 1
            or packet.get("training") is not False or packet.get("fresh_transition_cap") != 0
            or any(packet.get(k) is not False for k in
                   ("automatic_followup", "promotion", "g1_actor_execution", "combined_scene"))
            or execution.get("schema") != "task10b_phase2_execution_v1"
            or execution.get("status") != "executed" or execution.get("returncode") != 0
            or execution.get("native_processes_started") != 1
            or execution.get("packet_sha256") != packet_sha
            or not isinstance(execution.get("job_id"), str) or not execution["job_id"].isdigit()
            or source_path != _owned(root/_relative(packet["output"])/"manifest.json")
            or source_path.parent.parent != packet_path.parent):
        raise ValueError("Successful counted standalone target generation packet required")
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
            or terminal.get("state") != "COMPLETED" or terminal.get("exit_code") != "0:0"
            or terminal.get("packet_sha256") != packet_sha or terminal.get("job_id") != execution["job_id"]
            or not 0 < terminal.get("allocated_gpu_seconds", 0) <= wall
            or terminal.get("allocated_tres", {}).get("gres/gpu") != "1"):
        raise ValueError("Successful terminal one-GPU accounting required")

    files = packet.get("files", {})
    if not files or any(n not in files for n in (GENERATOR, SOLVER, WRAPPER, protocol["driver"])):
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
            or plan.get("family") != protocol["family"] or plan.get("split") != protocol["split"]
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
    for name,item in plan.get("sources",{}).items():
        path=_owned(item["path"])
        if path != _owned(name) or files.get(name)!=item["sha256"] or sha256_file(path)!=item["sha256"]:
            raise ValueError("Exact family solver source identity changed")
    if (set(plan.get("sources",{})) != {GENERATOR,SOLVER,WRAPPER,
            "src/human_punching/phase2_target_reference.py","src/human_punching/phase2_target_catalog.py"}
            or plan.get("entry_version") != "task10b_phase2_family_target_probe_v1"
            or plan.get("physical_qualification") is not False
            or plan.get("new_family_engineering_bound_unqualified") is not True
            or plan.get("native_control_required_before_learning") is not True):
        raise ValueError("Explicit unqualified family engineering hypothesis and exact sources required")
    settings=plan.get("retarget_settings",{})
    side='Left' if protocol['family']=='left_hook' else 'Right'
    names=[side+suffix for suffix in ('Shoulder_x','Shoulder_y','Shoulder_z','Arm_x','Arm_y','Arm_z',
        'ForeArm_z','ForeArm_x','Hand_x','Hand_y','Hand_z')]
    if (settings.get("maximum_joint_delta_rad") != .205 or settings.get("maximum_iterations") != 80
            or settings.get("active_position_tolerance_m") != .0005
            or settings.get("maximum_iteration_delta_rad") != .025
            or settings.get('active_side')!=side or settings.get('optimized_joint_names')!=names
            or settings.get('trunk_joint_names')!=[]
            or source.get("path_solver_version") != SOLVER_VERSION
            or source.get("output_byte_cap") != protocol["cap"]):
        raise ValueError("Unchanged reviewed .205 sequential engineering bounds required")
    if (source.get("schema") != CATALOG or any(k in source for k in ANNOTATIONS|{"target_catalog_projection","timing_control_projection","sequential_catalog_projection"})
            or source.get("complete") is not True or source.get("exception") is not None
            or source.get("planned_cells") != len(plan["cells"]) or source.get("executed_cells") != len(plan["cells"])
            or len(source.get("cells", [])) != len(plan["cells"]) or source.get("unattempted_cells") != []
            or any(source.get(k) is not False for k in ("physics_executed", "physical_qualification", "learned_trajectory_generator"))
            or source.get("recipe_sha256") != plan["recipe"]["sha256"]):
        raise ValueError("Complete unqualified native target candidate catalog required")
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

    sidecar_path=_owned(source_path.parent/"family_probe_entry.json")
    sidecar=_read(sidecar_path)
    if (sidecar.get("schema") != "task10b_phase2_family_target_probe_v1"
            or sidecar.get("plan_sha256") != files[plan_relative]
            or sidecar.get("wrapper_source_sha256") != files[WRAPPER]
            or sidecar.get("frozen_generator_source_sha256") != files[GENERATOR]
            or sidecar.get("manifest_sha256") != sha256_file(source_path)
            or sidecar.get("family") != protocol["family"]
            or sidecar.get("unchanged_solver_version") != SOLVER_VERSION
            or sidecar.get("engineering_cap_rad") != .205
            or sidecar.get("new_family_bound_unqualified") is not True
            or sidecar.get("solver_changed") is not False or sidecar.get("physical_limits_changed") is not False
            or sidecar.get("physical_qualification") is not False
            or sidecar.get("native_control_required_before_learning") is not True):
        raise ValueError("Explicit family wrapper/generator sidecar binding required")
    family_sidecar=_binding(sidecar_path)
    projected = json.loads(json.dumps(source, allow_nan=False))
    changes = []
    for index, (row, declaration) in enumerate(zip(projected["cells"], plan["cells"], strict=True)):
        if (any(row.get(k) != v for k, v in declaration.items())
                or row.get("family") != plan["family"] or row.get("split") != plan["split"]
                or row.get("recipe_sha256") != source["recipe_sha256"]
                or row.get("physical_qualification") is not False
                or row.get("path_solver_version") != SOLVER_VERSION
                or row.get("generation_disposition") not in ("generated", "failed")
                or (row.get("generation_disposition") == "failed" and row.get("kinematic_admission") is not False)):
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
    receipt = dict(version=VERSION, protocol=protocol, packet_schema=packet["schema"], family_probe_entry=family_sidecar, native_catalog=_binding(source_path), generation_execution=_binding(execution_path),
        generation_packet=_binding(packet_path), terminal_accounting=_binding(terminal_path), alias_probe=_binding(probe_path),
        nonce=_binding(nonce_path), archived_plan=_binding(archived[plan_relative]),
        archived_generator=_binding(archived[GENERATOR]), verified_archive_files=len(files),
        verified_archive_files_sha256=_canonical_hash(files), archived_logical_bytes=total,
        helper_sha256=sha256_file(__file__), changes=changes, non_path_metadata_sha256=_canonical_hash(nonpaths),
        scientific_metadata_changed=False, reference_bytes_changed=False, physical_qualification=False, native_frozen_control_required_before_learning=True)
    return projected, receipt


def project(catalog_path, execution_path, probe_path, output):
    projected, receipt = build_projection(catalog_path, execution_path, probe_path)
    destination = Path(output).resolve()
    if not destination.is_relative_to(ROOT.resolve(strict=True)) or destination.exists():
        raise ValueError("Fresh owned projection output directory required")
    destination.mkdir(parents=True, exist_ok=False)
    projected.update(unpacked_catalog_path=receipt["native_catalog"]["relative_name"],
        unpacked_catalog_sha256=receipt["native_catalog"]["sha256"], family_catalog_projection=receipt)
    for name, data in (("projection.json", receipt), ("manifest.json", projected)):
        with (destination/name).open("x") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    validate(destination/"manifest.json")
    return _binding(destination/"manifest.json")


def validate(path):
    path = _owned(path)
    view = _read(path)
    receipt = view["family_catalog_projection"]
    source, expected_receipt = build_projection(_bound(receipt["native_catalog"]),
        _bound(receipt["generation_execution"]), _bound(receipt["alias_probe"]))
    if (receipt != expected_receipt or _read(path.parent/"projection.json") != receipt
            or view.get("unpacked_catalog_path") != receipt["native_catalog"]["relative_name"]
            or view.get("unpacked_catalog_sha256") != receipt["native_catalog"]["sha256"]
            or {k: v for k, v in view.items() if k not in ANNOTATIONS} != source):
        raise ValueError("Projected target catalog changed science, paths or evidence")
    return _binding(path)


def verify_projection(path):
    """Return the validated view and its distinct family-probe provenance receipt."""
    validate(path)
    view=_read(path)
    return view, view["family_catalog_projection"]


def files(path):
    """Bind the exact metadata, aliases and archived bytes read by validation."""
    view,receipt=verify_projection(path)
    closure={}
    def add(value,expected=None):
        owned=_owned(value);digest=sha256_file(owned)
        if expected is not None and digest!=expected:
            raise ValueError("Sequential projection input changed during closure")
        name=str(owned.relative_to(ROOT.resolve(strict=True)))
        if closure.setdefault(name,digest)!=digest:
            raise ValueError("Sequential projection dependency changed")
        return owned
    add(path);add(_owned(path).parent/"projection.json")
    for key in ("native_catalog","generation_execution","generation_packet","terminal_accounting",
                "alias_probe","nonce","archived_plan","archived_generator"):
        add(receipt[key]["relative_name"],receipt[key]["sha256"])
    if receipt["family_probe_entry"] is not None:
        add(receipt["family_probe_entry"]["relative_name"],receipt["family_probe_entry"]["sha256"])
    packet_path=_bound(receipt["generation_packet"]);packet=_read(packet_path)
    for name,digest in packet["files"].items():
        add(packet_path.parent/"source_snapshot"/_relative(name),digest)
    plan_path=add(packet["inputs"]["plan"]);plan=_read(plan_path)
    for key in ("domain","target_geometry","parent_catalog","body_metadata","recipe",*receipt["protocol"]["extra_inputs"]):
        add(plan[key]["path"],plan[key]["sha256"])
    for name,item in plan["sources"].items():add(item["path"],item["sha256"])
    proof=_read(_bound(receipt["alias_probe"]))
    for item in proof["bindings"]:add(item["relative_path"],item["local_sha256"])
    for item in receipt["changes"]:add(item["relative_name"],item["sha256"])
    for name,digest in closure.items():
        if sha256_file(_owned(name))!=digest:
            raise ValueError("Sequential projection input changed during final closure")
    return dict(sorted(closure.items()))
