"""Two-root metadata views preserve science and reject unsupported ancestry."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from human_punching import phase2_right_hook_full_projection_v1 as projection
from human_punching.wbc.provenance import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return sha256_file(path)


def fixture_files(tmp_path, monkeypatch, *, alias=True, split="train"):
    count,allocation,application,child,cap=((64,420,360,300,256*1024**2) if split=="train" else (24,240,180,120,128*1024**2))
    root = tmp_path/"logical"
    root.mkdir()
    compute = tmp_path/"compute"
    if alias: compute.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(projection, "ROOT", root)
    folder = root/"outputs/task10b/p2fixture"
    run = folder/"run"
    run.mkdir(parents=True)
    probe_path = root/"proof/probe.json"
    nonce_path = root/"proof/nonce.json"
    marker = root/"marker.txt"
    marker.write_text("fixed alias marker")
    write(nonce_path, dict(schema="task10b_phase2_host_mount_alias_nonce_v1",
        root=str(root), compute_root=str(compute), nonce="synthetic-test-nonce"))
    probe = dict(schema="task10b_phase2_host_mount_alias_probe_v1", passed=True,
        local_root=str(root), compute_root=str(compute), fresh_owned_nonce_visible_identically_through_both_mounts=True,
        nonce_sha256=sha256_file(nonce_path), bindings=[dict(relative_path="marker.txt",
        local_sha256=sha256_file(marker), compute_mount_sha256=sha256_file(marker),
        local_inode=marker.stat().st_ino, compute_inode=marker.stat().st_ino)])
    write(probe_path, probe)
    for name in (projection.GENERATOR, projection.FULL_DRIVER, projection.FULL_WRAPPER, projection.SOLVER, projection.AUDITOR):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic archived generator identity\n")
    model = root/"model.xml"
    model.write_text("<mujoco/>")
    plan = dict(schema="task10b_phase2_right_hook_full_catalog_plan_v1", family="right_hook", split=split,
        start_mode="family_guard", maximum_cells=count, cells=[], maximum_child_seconds=child,
        output_byte_cap=cap,path_solver_version=projection.SOLVER_VERSION,
        retarget_settings=dict(maximum_joint_delta_rad=.205,maximum_iterations=80,
            active_position_tolerance_m=.0005,maximum_iteration_delta_rad=.025))
    inputs = {}
    for key in ("domain", "target_geometry", "parent_catalog", "body_metadata", "recipe", *projection.FULL_EVIDENCE):
        path = root/(key+".json")
        write(path, dict(opaque_ancestry_path="/foreign/obsolete/unchanged.json", role=key))
        plan[key] = dict(path=str(path), sha256=sha256_file(path))
        inputs[key] = sha256_file(path)
    rows = []
    for index in range(count):
        entry = dict(candidate_id=f"rh_target_{index}", parent_candidate_id=f"rh_selected_{index}",
            target_candidate_identity=str(index)*64)
        plan["cells"].append(entry)
        cell = run/entry["candidate_id"]
        cell.mkdir()
        ref = cell/"reference.npz"
        ref.write_bytes(b"Opaque synthetic array bytes; projection must never load them")
        screen = cell/"screen.json"
        write(screen, dict(physical_qualification=False))
        native = lambda p: str(compute/p.relative_to(root))
        rows.append(dict(**entry, family="right_hook", split=split, recipe_sha256=inputs["recipe"],
            command_id=2674+index, requested_ttc_s=1. if index%2 == 0 else 1.2,
            target_w=[.85, .2, 1.23], launch_tick=84, outward_end_tick=445,
            reference_path=native(ref), reference_sha256=sha256_file(ref),
            legacy_runtime_path=native(ref), legacy_runtime_sha256=sha256_file(ref),
            source_model_path=native(model), source_model_sha256=sha256_file(model),
            diagnostic_screen_path=native(screen), physical_qualification=False,path_solver_version=projection.SOLVER_VERSION,
            generation_disposition="generated", kinematic_admission=index != count-1,
            rejection_reasons=[] if index != count-1 else ["synthetic_kinematic_rejection"]))
    plan_path = root/"plan.json"
    write(plan_path, plan)
    relative_names = [projection.GENERATOR, projection.FULL_DRIVER, projection.FULL_WRAPPER, projection.SOLVER, projection.AUDITOR, "model.xml", "plan.json", "proof/probe.json",
        "proof/nonce.json", "marker.txt"]+[key+".json" for key in inputs]
    files = {name: sha256_file(root/name) for name in relative_names}
    archive = folder/"source_snapshot"
    for name in relative_names:
        destination = archive/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, destination)
    proof_binding = dict(path="proof/probe.json", sha256=sha256_file(probe_path))
    packet = dict(schema=projection.FULL_SCHEMA, mode="right-hook-full-target-references-v1", family="right_hook", sequential_split=split, native_frozen_control_required_before_learning=True, gpus=1,
        training=False, fresh_transition_cap=0, automatic_followup=False, promotion=False,
        g1_actor_execution=False, combined_scene=False, allocation_wall_seconds=allocation,
        application_wall_seconds=application,sequential_child_wall_seconds=child,
        sequential_path_solver_version=projection.SOLVER_VERSION,
        sequential_engineering_correction_cap_rad=.205,physical_joint_limits_relaxed=False, output=str(run.relative_to(root)), files=files,
        inputs={"plan": str(plan_path)}, runtime_namespace_proof=proof_binding)
    packet_path = folder/"packet.json"
    write(packet_path, packet)
    execution_path = folder/"execution.json"
    execution = dict(schema="task10b_phase2_execution_v1", status="executed", returncode=0,
        native_processes_started=1, packet_sha256=sha256_file(packet_path), job_id="123",
        runtime_path_context={"root": str(compute)}, runtime_namespace_proof=proof_binding,
        native_command=[str(compute/".venv-task10b/bin/python"), "-B", "-m", "human_punching.phase2_right_hook_full_catalog_v1",
            "--plan", str(compute/"plan.json"), "--output", str(compute/run.relative_to(root)), "--wall-seconds", str(float(child))])
    write(execution_path, execution)
    terminal_path = folder/"terminal_accounting.json"
    terminal = dict(schema="task10b_phase2_terminal_accounting_v1", state="COMPLETED", exit_code="0:0",
        packet_sha256=sha256_file(packet_path), job_id="123", allocated_gpu_seconds=50,
        allocated_tres={"gres/gpu": "1"})
    write(terminal_path, terminal)
    catalog_path = run/"manifest.json"
    catalog = dict(schema=projection.CATALOG, path_solver_version=projection.SOLVER_VERSION, output_byte_cap=cap, complete=True, exception=None, planned_cells=count, executed_cells=count,
        cells=rows, generated_cells=count, admitted_candidate_cells=count-1, unattempted_cells=[], input_hashes=inputs,
        plan_sha256=sha256_file(plan_path), source_sha256=files[projection.GENERATOR], recipe_sha256=inputs["recipe"],
        physics_executed=False, physical_qualification=False, learned_trajectory_generator=False)
    write(catalog_path, catalog)
    sidecar=dict(schema="task10b_phase2_right_hook_full_catalog_v1",plan_sha256=sha256_file(plan_path),
        wrapper_source_sha256=files[projection.FULL_WRAPPER],frozen_generator_source_sha256=files[projection.GENERATOR],
        manifest_sha256=sha256_file(catalog_path),family="right_hook",split=split,child_wall_seconds=child,
        engineering_cap_rad=.205,solver_changed=False,physical_qualification=False,native_frozen_control_required_before_learning=True)
    write(run/"right_hook_full_entry.json",sidecar)
    audit_path=root/"saved_audit.json"
    audit=dict(schema="task10b_phase2_rh_full_saved_target_reference_audit_v1",passed=True,family="right_hook",active_hand_index=1,
        split=split,job_id="123",physics_steps=0,project_reference_generation=False,reference_generation=False,
        source_sha256=sha256_file(root/projection.AUDITOR),cells=[dict(candidate_id=row["candidate_id"],all_checks_passed=True) for row in rows],
        generation=dict(planned=count,executed=count,generated=count,admitted=count-1,unattempted=[]))
    for key,path in dict(packet=packet_path,plan=plan_path,manifest=catalog_path,execution=execution_path,accounting=terminal_path).items():
        audit[key+"_path"]=str(path);audit[key+"_sha256"]=sha256_file(path)
    write(audit_path,audit)
    if not alias: shutil.copytree(root, compute)
    return SimpleNamespace(root=root, compute=compute, folder=folder, run=run, archive=archive,
        plan=plan, plan_path=plan_path, packet=packet, packet_path=packet_path, execution=execution,
        execution_path=execution_path, terminal=terminal, terminal_path=terminal_path,
        catalog=catalog, catalog_path=catalog_path, probe_path=probe_path, nonce_path=nonce_path,audit=audit,audit_path=audit_path,sidecar=sidecar,count=count,split=split)


@pytest.fixture
def data(tmp_path,monkeypatch):return fixture_files(tmp_path,monkeypatch)


def build(data):
    return projection.build_projection(data.catalog_path,data.execution_path,data.probe_path,saved_audit_path=data.audit_path)


def refresh_evidence(data):
    write(data.catalog_path,data.catalog)
    data.sidecar["manifest_sha256"]=sha256_file(data.catalog_path)
    write(data.run/"right_hook_full_entry.json",data.sidecar)
    for key,path in dict(packet=data.packet_path,plan=data.plan_path,manifest=data.catalog_path,
                         execution=data.execution_path,accounting=data.terminal_path).items():
        data.audit[key+"_sha256"]=sha256_file(path)
    data.audit["generation"]={k:data.catalog[v] for k,v in dict(planned="planned_cells",executed="executed_cells",
        generated="generated_cells",admitted="admitted_candidate_cells",unattempted="unattempted_cells").items()}
    write(data.audit_path,data.audit)


@pytest.mark.parametrize("split",["train","validation"])
def test_exact64_24_audited_paths_preserve_all_science(tmp_path,monkeypatch,split):
    data=fixture_files(tmp_path,monkeypatch,split=split)
    before={p:sha256_file(p) for p in data.root.rglob("*") if p.is_file()}
    view,receipt=build(data)
    assert len(view["cells"])==data.count and len(receipt["changes"])==4*data.count
    restored=copy.deepcopy(view)
    for item in receipt["changes"]:
        _,index,key=item["json_location"].split("/");restored["cells"][int(index)][key]=item["old"]
        assert Path(item["old"]).stat().st_ino==Path(item["new"]).stat().st_ino==item["inode"]
    assert restored==data.catalog and all(sha256_file(p)==h for p,h in before.items())
    assert receipt["physical_qualification"] is False and receipt["saved_audit"]["sha256"]==sha256_file(data.audit_path)
    assert receipt["protocol"]["child"]==(300 if split=="train" else 120)


def test_view_roundtrip_and_exact_nested_semantic_closure(data,monkeypatch):
    nested=data.root/"nested_evidence.json";write(nested,{"nested_probe_and_interior":True})
    monkeypatch.setattr(projection,"_semantic_files",lambda plan:{"nested_evidence.json":sha256_file(nested)})
    result=projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/"view",saved_audit_path=data.audit_path)
    path=data.root/result["relative_name"];view,receipt=projection.verify_projection(path);closure=projection.files(path)
    assert view["right_hook_full_catalog_projection"]==receipt
    assert closure["nested_evidence.json"]==sha256_file(nested)
    assert closure["saved_audit.json"]==sha256_file(data.audit_path)
    assert closure[projection.AUDITOR]==sha256_file(data.root/projection.AUDITOR)
    for name,digest in data.packet["files"].items():assert closure[str((data.archive/name).relative_to(data.root))]==digest
    with pytest.raises(ValueError,match="Fresh owned"):
        projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/"view",saved_audit_path=data.audit_path)


@pytest.mark.parametrize("field,value",[("requested_ttc_s",.9),("target_w",[0,0,0]),("kinematic_admission",True),("reference_path","/foreign/ref")])
def test_projected_science_and_paths_cannot_change(data,field,value):
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/"view",saved_audit_path=data.audit_path)
    path=data.root/"view/manifest.json";view=json.loads(path.read_text());view["cells"][-1][field]=value;write(path,view)
    with pytest.raises(ValueError,match="changed science"):projection.validate(path)


@pytest.mark.parametrize("field,value",[("schema","task10b_phase2_family_saved_target_reference_independent_audit_v1"),
    ("passed",False),("family","jab"),("split","test"),("job_id","999"),("physics_steps",1),
    ("source_sha256","0"*64),("manifest_sha256","0"*64),("reference_generation",True)])
def test_unrelated_or_unverified_audit_refused(data,field,value):
    data.audit[field]=value;write(data.audit_path,data.audit)
    with pytest.raises(ValueError):build(data)


@pytest.mark.parametrize("case",["audit_missing_row","audit_failed_row","counter","audit_wrong_input_path","partial","sidecar_family","sidecar_bound","archive","reference","nonce","accounting","unreviewed_path","different_inode"])
def test_preserved_ancestry_and_dispositions_fail_closed(data,case):
    if case=="audit_missing_row":data.audit["cells"].pop();write(data.audit_path,data.audit)
    elif case=="audit_failed_row":data.audit["cells"][0]["all_checks_passed"]=False;write(data.audit_path,data.audit)
    elif case=="counter":data.audit["generation"]["generated"]-=1;write(data.audit_path,data.audit)
    elif case=="audit_wrong_input_path":data.audit["manifest_path"]=str(data.plan_path);write(data.audit_path,data.audit)
    elif case=="partial":data.catalog["complete"]=False;refresh_evidence(data)
    elif case.startswith("sidecar"):
        data.sidecar["family" if case=="sidecar_family" else "engineering_cap_rad"]="jab" if case=="sidecar_family" else .21
        write(data.run/"right_hook_full_entry.json",data.sidecar)
    elif case=="archive":(data.archive/projection.GENERATOR).write_text("changed")
    elif case=="reference":(data.run/"rh_target_0/reference.npz").write_bytes(b"changed")
    elif case=="nonce":data.nonce_path.write_text("changed")
    elif case=="accounting":data.terminal["state"]="TIMEOUT";write(data.terminal_path,data.terminal)
    elif case=="unreviewed_path":data.catalog["cells"][0]["new_path"]=str(data.compute/"model.xml");refresh_evidence(data)
    else:
        data.compute.unlink();shutil.copytree(data.root,data.compute)
    with pytest.raises(ValueError):build(data)


@pytest.mark.parametrize("field,value",[("schema","task10b_phase2_sequential_full_packet_v1"),("family","jab"),
    ("sequential_split","test"),("sequential_child_wall_seconds",299),("allocation_wall_seconds",480),
    ("sequential_engineering_correction_cap_rad",.21),("physical_joint_limits_relaxed",True)])
def test_old_protocol_or_changed_bounds_refused(data,field,value):
    data.packet[field]=value;write(data.packet_path,data.packet);digest=sha256_file(data.packet_path)
    data.execution["packet_sha256"]=digest;data.terminal["packet_sha256"]=digest
    write(data.execution_path,data.execution);write(data.terminal_path,data.terminal);refresh_evidence(data)
    with pytest.raises(ValueError):build(data)


def test_all_negative_complete_catalog_stays_negative(data):
    for i,row in enumerate(data.catalog["cells"]):
        old=copy.deepcopy(row)
        for key in list(row):
            if key in ("reference_path","legacy_runtime_path","legacy_runtime_sha256","source_model_path","source_model_sha256","diagnostic_screen_path"):row.pop(key)
        row.update(generation_disposition="failed",kinematic_admission=False,rejection_reasons=["geometry_unsolved"],
            negative_clock_semantics="Parent clocks only; no target execution",retained_partial_reference_path=old["reference_path"])
        p=data.run/f"rh_target_{i}/failure.json";write(p,{"reason":"geometry_unsolved"})
        row.update(generation_failure_path=str(data.compute/p.relative_to(data.root)),generation_failure_sha256=sha256_file(p),reference_sha256=sha256_file(p))
    data.catalog["generated_cells"]=0;data.catalog["admitted_candidate_cells"]=0;refresh_evidence(data)
    view,_=build(data)
    assert view["generated_cells"]==0 and all(r["generation_disposition"]=="failed" and r["kinematic_admission"] is False for r in view["cells"])
