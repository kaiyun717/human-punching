"""Exact split dispatch, reserved evaluation capacity and real owned closure."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def driver():
    path=ROOT/"scripts/task10b_phase2_right_hook_full_v1.py"
    spec=importlib.util.spec_from_file_location("rh_full_launcher_test",path)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);return d


def record(split="train"):
    count,alloc,app,child={"train":(64,420,360,300),"validation":(24,240,180,120)}[split]
    return dict(mode="right-hook-full-target-references-v1",family="right_hook",training=False,
        fresh_transition_cap=0,inputs={"plan":f"outputs/task10b/p2rhfullprep01/{split}_plan.json"},
        sequential_cells=count,sequential_split=split,sequential_child_wall_seconds=child,
        allocation_wall_seconds=alloc,application_wall_seconds=app,sequential_engineering_correction_cap_rad=.205,
        physical_joint_limits_relaxed=False,native_frozen_control_required_before_learning=True,
        output="outputs/task10b/future/run")


@pytest.mark.parametrize("split",["train","validation"])
def test_exact_child_and_no_native_actor_or_learning_dispatch(split):
    d=driver();r=record(split);cmd=d.native_command(r)
    assert cmd[2:4]==["-m","human_punching.phase2_right_hook_full_catalog_v1"]
    assert cmd[-1]==str(float(r["sequential_child_wall_seconds"]))
    assert d.legacy.CAP==1024**3


@pytest.mark.parametrize("split",["train","validation"])
@pytest.mark.parametrize("delta",[-.001,31.])
def test_late_or_impossible_setup_cannot_shorten_child(split,delta):
    r=record(split)
    with pytest.raises(RuntimeError):driver().native_command(r,r["sequential_child_wall_seconds"]+delta)


@pytest.mark.parametrize("field,value",[("family","left_hook"),("training",True),("fresh_transition_cap",1),
    ("sequential_cells",63),("sequential_engineering_correction_cap_rad",.21),
    ("physical_joint_limits_relaxed",True),("native_frozen_control_required_before_learning",False)])
def test_exact_science_and_scope_before_dispatch(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError):driver().native_command(r)


def test_evaluation_reserve_is_enforced_even_for_generation(monkeypatch):
    d=driver();calls=[]
    monkeypatch.setattr(d,"_admit_reservation",lambda *args,**kwargs:calls.append((args,kwargs)))
    d.protected_reservation([],420)
    assert calls[0][1]["evaluation_reserve_seconds"]==4500
    assert calls[0][1]["successful_route_frozen"] is False
    d.protected_reservation([],240,evaluation_reserve_seconds=4800)
    assert calls[-1][1]["evaluation_reserve_seconds"]==4800
    with pytest.raises(ValueError):d.protected_reservation([],421)
    with pytest.raises(ValueError):d.protected_reservation([],420,successful_route_frozen=True)


def test_semantic_dependency_restored_on_closure_failure(monkeypatch):
    d=driver();original=d.legacy.semantic_files
    def fail(*args,**kwargs):raise ValueError("synthetic closure error")
    monkeypatch.setattr(d,"_required_inputs",fail)
    r=record();r["runtime_namespace_proof"]={"path":"reviewed"}
    with pytest.raises(ValueError,match="synthetic"):d.required_inputs(r)
    assert d.legacy.semantic_files is original


@pytest.mark.parametrize("split,count,allocation,application",[("train",64,420,360),("validation",24,240,180)])
def test_real_plans_common_recipe_exact_input_and_source_closure(split,count,allocation,application):
    d=driver();path=ROOT/f"outputs/task10b/p2rhfullprep01/{split}_plan.json"
    plan=json.loads(path.read_text())
    actual=d.validate_spec(d.MODE,{"plan":str(path)},allocation,application)
    assert actual["family"]=="right_hook" and actual["sequential_cells"]==count
    closure=d.files(path)
    assert str(path.relative_to(ROOT)) in closure
    assert "src/human_punching/phase2_right_hook_interior_v1.py" in closure
    for key in ("interior_audit","interior_packet","interior_plan","probe_audit","probe_plan"):
        p=Path(plan[key]["path"]);assert closure[str(p.relative_to(ROOT))]==plan[key]["sha256"]
    old=json.loads(Path(plan["probe_plan"]["path"]).read_text())
    for key in ("recipe","timing_selection","selected_d0_plan"):
        p=Path(old[key]["path"]);assert closure[str(p.relative_to(ROOT))]==old[key]["sha256"]
    other=json.loads((path.parent/("validation_plan.json" if split=="train" else "train_plan.json")).read_text())
    assert plan["recipe"]==other["recipe"]
