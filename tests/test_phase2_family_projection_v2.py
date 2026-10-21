"""The .210 view preserves negative rows and binds its distinct cap evidence."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from human_punching import phase2_family_projection_v2 as projection


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return projection.sha256_file(path)


@pytest.fixture
def data(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("old_family_projection_fixture",
        Path(__file__).with_name("test_phase2_family_projection_v1.py"))
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    d = old.data.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(projection, "ROOT", d.root)
    previous = copy.deepcopy(d.plan)
    previous_path = d.root/"previous_probe_plan.json"
    previous_sha = write(previous_path, previous)
    diagnostic_path = d.root/"diagnostic.py"; diagnostic_path.write_text("# saved-pose algebra only\n")
    item = dict(path="diagnostic.py", sha256=projection.sha256_file(diagnostic_path))
    evidence = dict(schema="task10b_phase2_six_failed_family_pose_bound_sensitivity_v1",
        status="completed_saved_pose_algebra_only", proposed_caps_rad=[.205, .21, .225, .25],
        tolerance_m=.0005, accepted_constants_unchanged=True, physical_joint_limits_relaxed=False,
        new_reference_generation=False, cases=[dict() for _ in range(6)],
        input_bindings={"previous_probe_plan.json": previous_sha}, script=item,
        independent_fk_source=item, alias_proof=item)
    evidence_path = d.root/"bound_sensitivity.json"; evidence_sha = write(evidence_path, evidence)
    d.plan.update(schema="task10b_phase2_family_target_probe_plan_v2",
        entry_version="task10b_phase2_family_target_probe_v2", path_solver_version=projection.SOLVER_VERSION,
        previous_probe_plan=dict(path=str(previous_path), sha256=previous_sha),
        bound_sensitivity=dict(path=str(evidence_path), sha256=evidence_sha),
        bound_sensitivity_inputs={"previous_probe_plan.json": dict(path="previous_probe_plan.json", sha256=previous_sha),
                                  "diagnostic.py": item})
    d.plan["retarget_settings"]["maximum_joint_delta_rad"] = .21
    names = [projection.GENERATOR, projection.SOLVER, projection.WRAPPER,
        "src/human_punching/phase2_target_path_reference_v3.py",
        "src/human_punching/phase2_target_reference.py", "src/human_punching/phase2_target_catalog.py"]
    for name in names+[projection.CLI]:
        p=d.root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# synthetic distinct cap source\n")
    d.plan["sources"]={n: dict(path=str(d.root/n),sha256=projection.sha256_file(d.root/n)) for n in names}
    recipe_path = Path(d.plan["recipe"]["path"])
    recipe = json.loads(recipe_path.read_text()); recipe.update(bound_sensitivity=d.plan["bound_sensitivity"],
        previous_probe_plan=d.plan["previous_probe_plan"], selected_engineering_cap_rad=.21,
        other_solver_math_changed=False, larger_caps_selected=False)
    d.plan["recipe"]["sha256"] = write(recipe_path,recipe)
    for key in ("recipe", "bound_sensitivity", "previous_probe_plan"):
        d.catalog["input_hashes"][key] = d.plan[key]["sha256"]
    d.catalog["recipe_sha256"] = d.plan["recipe"]["sha256"]
    d.catalog["path_solver_version"] = projection.SOLVER_VERSION
    for row in d.catalog["cells"]:
        row["recipe_sha256"] = d.plan["recipe"]["sha256"]
        row["path_solver_version"] = projection.SOLVER_VERSION
    write(d.plan_path,d.plan)
    for name in names+[projection.CLI,"plan.json","recipe.json","previous_probe_plan.json","bound_sensitivity.json","diagnostic.py"]:
        d.packet["files"][name]=projection.sha256_file(d.root/name)
        destination=d.archive/name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(d.root/name,destination)
    d.packet.update(schema=projection.PROBE_SCHEMA, mode="family-target-references-v2",
        sequential_engineering_correction_cap_rad=.21, sequential_path_solver_version=projection.SOLVER_VERSION)
    write(d.packet_path,d.packet); digest=projection.sha256_file(d.packet_path)
    d.execution["packet_sha256"]=d.terminal["packet_sha256"]=digest
    d.execution["native_command"][3]="human_punching.phase2_family_target_probe_v2"
    write(d.execution_path,d.execution);write(d.terminal_path,d.terminal)
    d.catalog["source_sha256"]=d.packet["files"][projection.GENERATOR]
    d.catalog["plan_sha256"]=projection.sha256_file(d.plan_path);write(d.catalog_path,d.catalog)
    d.sidecar.update(schema="task10b_phase2_family_target_probe_v2",plan_sha256=projection.sha256_file(d.plan_path),
        manifest_sha256=projection.sha256_file(d.catalog_path),wrapper_source_sha256=d.packet["files"][projection.WRAPPER],
        frozen_generator_source_sha256=d.packet["files"][projection.GENERATOR],unchanged_solver_version=projection.SOLVER_VERSION,
        solver_source_sha256=d.packet["files"][projection.SOLVER],engineering_cap_rad=.21,
        solver_changed=True,solver_math_other_than_cap_changed=False)
    write(d.sidecar_path,d.sidecar)
    return d


def build(data):
    return projection.build_projection(data.catalog_path,data.execution_path,data.probe_path)


def test_exact_v2_roundtrip_binds_all_diagnostic_inputs_without_changing_bytes(data):
    before={p:projection.sha256_file(p) for p in data.root.rglob("*") if p.is_file()}
    result=projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/"view")
    view,receipt=projection.verify_projection(data.root/result["relative_name"])
    restored=copy.deepcopy(view)
    for name in projection.ANNOTATIONS:restored.pop(name)
    for item in receipt["changes"]:
        _,index,key=item["json_location"].split("/");restored["cells"][int(index)][key]=item["old"]
    assert restored==data.catalog
    closure=projection.files(data.root/result["relative_name"])
    assert all(n in closure for n in ("diagnostic.py","bound_sensitivity.json","previous_probe_plan.json"))
    assert all(projection.sha256_file(p)==digest for p,digest in before.items())
    assert receipt["scientific_metadata_changed"] is False and receipt["reference_bytes_changed"] is False


@pytest.mark.parametrize("field,value",[("engineering_cap_rad",.205),("engineering_cap_rad",.225),
    ("solver_changed",False),("solver_math_other_than_cap_changed",True),
    ("solver_source_sha256","0"*64),("schema","task10b_phase2_family_target_probe_v1"),
    ("physical_limits_changed",True)])
def test_sidecar_refuses_old_or_further_changed_recipe(data,field,value):
    data.sidecar[field]=value;write(data.sidecar_path,data.sidecar)
    with pytest.raises(ValueError,match="sidecar"):build(data)


@pytest.mark.parametrize("where",["diagnostic.py","bound_sensitivity.json","previous_probe_plan.json",
    projection.SOLVER,projection.WRAPPER,"plan.json"])
def test_cap_evidence_or_source_mutation_is_rejected(data,where):
    (data.root/where).write_text("changed")
    with pytest.raises((ValueError,json.JSONDecodeError)):build(data)


@pytest.mark.parametrize("field,value",[("requested_ttc_s",.9),("family","right_hook"),
    ("kinematic_admission",True),("target_w",[0,0,0])])
def test_projected_view_cannot_change_science(data,field,value):
    projection.project(data.catalog_path,data.execution_path,data.probe_path,data.root/"view")
    path=data.root/"view/manifest.json";view=json.loads(path.read_text());view["cells"][3][field]=value;write(path,view)
    with pytest.raises(ValueError,match="changed science"):projection.verify_projection(path)


def test_equal_bytes_at_nonalias_prefix_rejected(data):
    data.compute.unlink();shutil.copytree(data.root,data.compute)
    with pytest.raises(ValueError,match="same owned inode"):build(data)


@pytest.mark.parametrize("family,suffix",[("cross","c"),("left_hook","l")])
def test_actual_completed_v2_catalog_retains_two_negative_cells(family,suffix):
    root=Path(__file__).resolve().parents[1];folder=root/f"outputs/task10b/p2ft02{suffix}"
    view,receipt=projection.build_projection(folder/"run/manifest.json",folder/"execution.json",
        root/"outputs/task10b/p2hostalias01/probe.json")
    assert view["planned_cells"]==view["executed_cells"]==4
    assert view["generated_cells"]==view["admitted_candidate_cells"]==2
    assert [r["generation_disposition"] for r in view["cells"]]==["failed","failed","generated","generated"]
    assert receipt["protocol"]["family"]==family and receipt["version"]==projection.VERSION
    assert all(r["physical_qualification"] is False for r in view["cells"])


def test_old205_catalog_is_not_silently_relabelled():
    root=Path(__file__).resolve().parents[1];folder=root/"outputs/task10b/p2ft01l"
    with pytest.raises(ValueError,match="Explicit .210"):
        projection.build_projection(folder/"run/manifest.json",folder/"execution.json",root/"outputs/task10b/p2hostalias01/probe.json")
