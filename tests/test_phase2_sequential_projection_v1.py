"""Two-root metadata views preserve science and reject unsupported ancestry."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from human_punching import phase2_sequential_projection_v1 as projection
from human_punching.wbc.provenance import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return sha256_file(path)


def fixture_files(tmp_path, monkeypatch, *, alias=True):
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
    for name in (projection.GENERATOR, projection.CLI, projection.SOLVER):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic archived generator identity\n")
    model = root/"model.xml"
    model.write_text("<mujoco/>")
    plan = dict(schema="task10b_phase2_sequential_target_catalog_plan_v3", family="jab", split="train",
        start_mode="family_guard", maximum_cells=4, cells=[], maximum_child_seconds=240,
        output_byte_cap=128*1024**2,path_solver_version=projection.SOLVER_VERSION,
        retarget_settings=dict(maximum_joint_delta_rad=.205,maximum_iterations=80,
            active_position_tolerance_m=.0005,maximum_iteration_delta_rad=.025))
    inputs = {}
    for key in ("domain", "target_geometry", "parent_catalog", "body_metadata", "recipe", "selection_audit", "engineering_bound_review"):
        path = root/(key+".json")
        write(path, dict(opaque_ancestry_path="/foreign/obsolete/unchanged.json", role=key))
        plan[key] = dict(path=str(path), sha256=sha256_file(path))
        inputs[key] = sha256_file(path)
    rows = []
    for index in range(4):
        entry = dict(candidate_id=f"jab_target_{index}", parent_candidate_id=f"jab_uniform_{index}",
            target_candidate_identity=str(index)*64)
        plan["cells"].append(entry)
        cell = run/entry["candidate_id"]
        cell.mkdir()
        ref = cell/"reference.npz"
        ref.write_bytes(b"Opaque synthetic array bytes; projection must never load them")
        screen = cell/"screen.json"
        write(screen, dict(physical_qualification=False))
        native = lambda p: str(compute/p.relative_to(root))
        rows.append(dict(**entry, family="jab", split="train", recipe_sha256=inputs["recipe"],
            command_id=2674+index, requested_ttc_s=.6 if index%2 == 0 else .75,
            target_w=[.85, .2, 1.23], launch_tick=84, outward_end_tick=445,
            reference_path=native(ref), reference_sha256=sha256_file(ref),
            legacy_runtime_path=native(ref), legacy_runtime_sha256=sha256_file(ref),
            source_model_path=native(model), source_model_sha256=sha256_file(model),
            diagnostic_screen_path=native(screen), physical_qualification=False,path_solver_version=projection.SOLVER_VERSION,
            generation_disposition="generated", kinematic_admission=index != 3,
            rejection_reasons=[] if index != 3 else ["synthetic_kinematic_rejection"]))
    plan_path = root/"plan.json"
    write(plan_path, plan)
    relative_names = [projection.GENERATOR, projection.CLI, projection.SOLVER, "model.xml", "plan.json", "proof/probe.json",
        "proof/nonce.json", "marker.txt"]+[key+".json" for key in inputs]
    files = {name: sha256_file(root/name) for name in relative_names}
    archive = folder/"source_snapshot"
    for name in relative_names:
        destination = archive/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, destination)
    proof_binding = dict(path="proof/probe.json", sha256=sha256_file(probe_path))
    packet = dict(schema=projection.PROBE_SCHEMA, mode="sequential-target-references", gpus=1,
        training=False, fresh_transition_cap=0, automatic_followup=False, promotion=False,
        g1_actor_execution=False, combined_scene=False, allocation_wall_seconds=360,
        application_wall_seconds=300,sequential_child_wall_seconds=240,
        sequential_path_solver_version=projection.SOLVER_VERSION,
        sequential_engineering_correction_cap_rad=.205,physical_joint_limits_relaxed=False, output=str(run.relative_to(root)), files=files,
        inputs={"plan": str(plan_path)}, runtime_namespace_proof=proof_binding)
    packet_path = folder/"packet.json"
    write(packet_path, packet)
    execution_path = folder/"execution.json"
    execution = dict(schema="task10b_phase2_execution_v1", status="executed", returncode=0,
        native_processes_started=1, packet_sha256=sha256_file(packet_path), job_id="123",
        runtime_path_context={"root": str(compute)}, runtime_namespace_proof=proof_binding,
        native_command=[str(compute/".venv-task10b/bin/python"), "-B", "-m", "human_punching.phase2_sequential_target_catalog_v3",
            "--plan", str(compute/"plan.json"), "--output", str(compute/run.relative_to(root)), "--wall-seconds", "240.0"])
    write(execution_path, execution)
    terminal_path = folder/"terminal_accounting.json"
    terminal = dict(schema="task10b_phase2_terminal_accounting_v1", state="COMPLETED", exit_code="0:0",
        packet_sha256=sha256_file(packet_path), job_id="123", allocated_gpu_seconds=50,
        allocated_tres={"gres/gpu": "1"})
    write(terminal_path, terminal)
    catalog_path = run/"manifest.json"
    catalog = dict(schema=projection.CATALOG, path_solver_version=projection.SOLVER_VERSION, output_byte_cap=128*1024**2, complete=True, exception=None, planned_cells=4, executed_cells=4,
        cells=rows, generated_cells=4, admitted_candidate_cells=3, unattempted_cells=[], input_hashes=inputs,
        plan_sha256=sha256_file(plan_path), source_sha256=files[projection.GENERATOR], recipe_sha256=inputs["recipe"],
        physics_executed=False, physical_qualification=False, learned_trajectory_generator=False)
    write(catalog_path, catalog)
    if not alias: shutil.copytree(root, compute)
    return SimpleNamespace(root=root, compute=compute, folder=folder, run=run, archive=archive,
        plan=plan, plan_path=plan_path, packet=packet, packet_path=packet_path, execution=execution,
        execution_path=execution_path, terminal=terminal, terminal_path=terminal_path,
        catalog=catalog, catalog_path=catalog_path, probe_path=probe_path, nonce_path=nonce_path)


@pytest.fixture
def data(tmp_path, monkeypatch):
    return fixture_files(tmp_path, monkeypatch)


def build(data):
    return projection.build_projection(data.catalog_path, data.execution_path, data.probe_path)


def test_same_file_alias_changes_only_whitelisted_strings(data):
    before = {p: sha256_file(p) for p in data.root.rglob("*") if p.is_file()}
    view, receipt = build(data)
    assert len(receipt["changes"]) == 16
    restored = copy.deepcopy(view)
    for change in receipt["changes"]:
        _, index, key = change["json_location"].split("/")
        assert view["cells"][int(index)][key] == change["new"]
        restored["cells"][int(index)][key] = change["old"]
        assert Path(change["old"]).stat().st_ino == Path(change["new"]).stat().st_ino == change["inode"]
    assert restored == data.catalog
    assert all(sha256_file(p) == digest for p, digest in before.items())
    assert view["admitted_candidate_cells"] == 3 and not view["cells"][3]["kinematic_admission"]


def test_fresh_view_and_roundtrip_validation(data):
    result = projection.project(data.catalog_path, data.execution_path, data.probe_path, data.root/"view")
    assert projection.validate(data.root/result["relative_name"]) == result
    with pytest.raises(ValueError, match="Fresh owned"):
        projection.project(data.catalog_path, data.execution_path, data.probe_path, data.root/"view")


@pytest.mark.parametrize("field,value", [("requested_ttc_s", .7), ("target_w", [0, 0, 0]),
    ("launch_tick", 0), ("kinematic_admission", True), ("reference_path", "/different/reference.npz")])
def test_projected_science_or_path_tamper_rejected(data, field, value):
    projection.project(data.catalog_path, data.execution_path, data.probe_path, data.root/"view")
    path = data.root/"view/manifest.json"
    view = json.loads(path.read_text())
    view["cells"][3][field] = value
    write(path, view)
    with pytest.raises(ValueError, match="changed science"):
        projection.validate(path)


@pytest.mark.parametrize("name", ["status", "returncode", "native_processes_started", "packet_sha256"])
def test_failed_or_unrelated_generation_rejected(data, name):
    data.execution[name] = {"status": "budget_stop", "returncode": 1,
        "native_processes_started": 0, "packet_sha256": "0"*64}[name]
    write(data.execution_path, data.execution)
    with pytest.raises(ValueError, match="Successful counted"):
        build(data)


@pytest.mark.parametrize("part", ["archive", "reference", "nonce", "accounting", "unknown_path", "wrong_prefix", "unbounded_child", "partial"])
def test_changed_ancestry_and_unsupported_inventory_fail_closed(data, part):
    if part == "archive": (data.archive/projection.GENERATOR).write_text("changed")
    elif part == "reference": (data.run/"jab_target_0/reference.npz").write_bytes(b"changed")
    elif part == "nonce": data.nonce_path.write_text("changed")
    elif part == "accounting":
        data.terminal["state"] = "TIMEOUT"; write(data.terminal_path, data.terminal)
    elif part == "unknown_path":
        data.catalog["cells"][0]["unreviewed_path"] = str(data.compute/"model.xml"); write(data.catalog_path, data.catalog)
    elif part == "wrong_prefix":
        data.catalog["cells"][0]["reference_path"] = str(data.root/"outputs/task10b/p2fixture/run/jab_target_0/reference.npz"); write(data.catalog_path, data.catalog)
    elif part == "unbounded_child":
        data.execution["native_command"][-1] = "211"; write(data.execution_path, data.execution)
    else:
        data.catalog["complete"] = False; write(data.catalog_path, data.catalog)
    with pytest.raises(ValueError): build(data)


def test_identical_bytes_on_different_inodes_are_not_mount_aliases(tmp_path, monkeypatch):
    data = fixture_files(tmp_path, monkeypatch, alias=False)
    with pytest.raises(ValueError, match="same owned inode"):
        build(data)


def test_projection_cannot_write_outside_owned_root(data):
    with pytest.raises(ValueError, match="Fresh owned"):
        projection.project(data.catalog_path, data.execution_path, data.probe_path, data.root.parent/"escaped")


def test_complete_geometric_failure_and_partial_payload_remain_negative(data):
    previous = data.catalog["cells"][3]
    failure = {key: value for key, value in previous.items() if key not in
        ("reference_path", "legacy_runtime_path", "legacy_runtime_sha256", "source_model_path", "source_model_sha256", "diagnostic_screen_path")}
    failure.update(generation_disposition="failed", kinematic_admission=False,
        rejection_reasons=["target_reference_generation_failure"],
        retained_partial_reference_path=previous["reference_path"],
        negative_clock_semantics="Parent clocks only; no generated or executed timing")
    path = data.run/"jab_target_3/generation_failure.json"
    write(path, failure)
    failure.update(generation_failure_path=str(data.compute/path.relative_to(data.root)),
        generation_failure_sha256=sha256_file(path), reference_sha256=sha256_file(path))
    data.catalog["cells"][3] = failure
    data.catalog["generated_cells"] = 3
    write(data.catalog_path, data.catalog)
    projected, receipt = build(data)
    assert projected["cells"][3]["generation_disposition"] == "failed"
    assert projected["cells"][3]["negative_clock_semantics"] == failure["negative_clock_semantics"]
    assert not projected["cells"][3]["kinematic_admission"]
    assert projected["generated_cells"] == 3 and projected["executed_cells"] == 4
    assert any(c["json_location"].endswith("generation_failure_path") for c in receipt["changes"])


@pytest.mark.parametrize("change", ["counter", "missing_digest"])
def test_inconsistent_native_counters_or_missing_file_identity_rejected(data, change):
    if change == "counter": data.catalog["generated_cells"] = 99
    else: data.catalog["cells"][0].pop("reference_sha256")
    write(data.catalog_path, data.catalog)
    with pytest.raises(ValueError): build(data)


def test_archived_generator_identity_is_independent_of_later_live_source(data):
    (data.root/projection.GENERATOR).write_text("# later live implementation; historical archive unchanged\n")
    projected, receipt = build(data)
    assert projected["source_sha256"] == data.packet["files"][projection.GENERATOR]
    assert receipt["archived_generator"]["sha256"] == projected["source_sha256"]


def full_variant(data,split='train'):
    count,wall,app,child,cap=((64,480,420,360,256*1024**2) if split=='train' else (24,300,240,180,128*1024**2))
    data.plan.update(schema='task10b_phase2_sequential_full_catalog_plan_v1',split=split,maximum_cells=count,
        maximum_child_seconds=child,output_byte_cap=cap)
    entries=[];rows=[]
    for index in range(count):
        entry=copy.deepcopy(data.plan['cells'][index%4]);row=copy.deepcopy(data.catalog['cells'][index%4])
        entry['candidate_id']=f'full_{split}_{index}';row.update(entry);row['split']=split
        entries.append(entry);rows.append(row)
    data.plan['cells']=entries;data.catalog['cells']=rows
    for key in projection.FULL_EVIDENCE:
        p=data.root/(key+'.json');write(p,dict(explicit_synthetic_evidence=key))
        digest=sha256_file(p);data.plan[key]=dict(path=str(p),sha256=digest)
        data.catalog['input_hashes'][key]=digest;data.packet['files'][p.name]=digest
        shutil.copyfile(p,data.archive/p.name)
    for name in (projection.FULL_WRAPPER,projection.FULL_DRIVER):
        p=data.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# explicit full wrapper fixture')
        data.packet['files'][name]=sha256_file(p)
        (data.archive/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,data.archive/name)
    write(data.plan_path,data.plan);shutil.copyfile(data.plan_path,data.archive/'plan.json')
    data.packet['files']['plan.json']=sha256_file(data.plan_path)
    data.packet.update(schema=projection.FULL_SCHEMA,sequential_split=split,
        allocation_wall_seconds=wall,application_wall_seconds=app,sequential_child_wall_seconds=child,
        native_frozen_control_required_before_learning=True)
    write(data.packet_path,data.packet);digest=sha256_file(data.packet_path)
    data.execution['packet_sha256']=digest;data.terminal['packet_sha256']=digest
    data.execution['native_command'][3]='human_punching.phase2_sequential_full_catalog_v1'
    data.execution['native_command'][-1]=str(float(child))
    write(data.execution_path,data.execution);write(data.terminal_path,data.terminal)
    data.catalog.update(plan_sha256=sha256_file(data.plan_path),output_byte_cap=cap,
        planned_cells=count,executed_cells=count,generated_cells=count,admitted_candidate_cells=count*3//4)
    write(data.catalog_path,data.catalog)
    sidecar=dict(schema='task10b_phase2_sequential_full_catalog_v1',plan_sha256=sha256_file(data.plan_path),
        wrapper_source_sha256=data.packet['files'][projection.FULL_WRAPPER],
        frozen_generator_source_sha256=data.packet['files'][projection.GENERATOR],
        manifest_sha256=sha256_file(data.catalog_path),split=split,child_wall_seconds=child,
        solver_changed=False,physical_qualification=False,native_frozen_control_required_before_learning=True)
    write(data.run/'full_catalog_entry.json',sidecar)
    return data


@pytest.mark.parametrize('split',['train','validation'])
def test_explicit_full_schema_counts_and_wrapper_ancestry(data,split):
    full_variant(data,split)
    view,receipt=build(data)
    assert view['source_sha256']==data.packet['files'][projection.GENERATOR]
    assert receipt['packet_schema']==projection.FULL_SCHEMA
    assert len(view['cells'])==(64 if split=='train' else 24)
    assert receipt['full_catalog_entry']['sha256']==sha256_file(data.run/'full_catalog_entry.json')
    assert receipt['scientific_metadata_changed'] is False


@pytest.mark.parametrize('field,value', [('solver_changed',True),('wrapper_source_sha256','0'*64),
    ('manifest_sha256','0'*64),('native_frozen_control_required_before_learning',False)])
def test_full_wrapper_sidecar_fail_closed(data,field,value):
    full_variant(data)
    path=data.run/'full_catalog_entry.json';sidecar=json.loads(path.read_text());sidecar[field]=value;write(path,sidecar)
    with pytest.raises(ValueError,match='sidecar'):build(data)


@pytest.mark.parametrize('field,value',[('schema','task10b_phase2_expansion_packet_v1'),
    ('sequential_engineering_correction_cap_rad',.21),('physical_joint_limits_relaxed',True),
    ('sequential_child_wall_seconds',241)])
def test_no_historical_schema_or_recipe_masquerade(data,field,value):
    data.packet[field]=value;write(data.packet_path,data.packet)
    digest=sha256_file(data.packet_path);data.execution['packet_sha256']=digest;data.terminal['packet_sha256']=digest
    write(data.execution_path,data.execution);write(data.terminal_path,data.terminal)
    with pytest.raises(ValueError):build(data)


def test_projection_runtime_dependency_closure(data):
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/'view')
    path=data.root/'view/manifest.json';view,receipt=projection.verify_projection(path)
    closure=projection.files(path)
    assert closure['view/manifest.json']==sha256_file(path)
    for name,digest in data.packet['files'].items():
        assert closure[str((data.archive/name).relative_to(data.root))]==digest
    for name in ['plan.json','selection_audit.json','engineering_bound_review.json','proof/nonce.json','marker.txt']:
        assert closure[name]==sha256_file(data.root/name)
    assert view['sequential_catalog_projection']==receipt
    assert not any('foreign' in path for path in closure)


def test_real_completed_probe_projection_is_metadata_only():
    root=Path(__file__).resolve().parents[1]
    view,receipt=projection.build_projection(root/'outputs/task10b/p2sq02/run/manifest.json',
        root/'outputs/task10b/p2sq02/execution.json',root/'outputs/task10b/p2hostalias01/probe.json')
    assert len(view['cells'])==4 and view['generated_cells']==4
    assert receipt['reference_bytes_changed'] is False and receipt['physical_qualification'] is False
    assert receipt['packet_schema']==projection.PROBE_SCHEMA
