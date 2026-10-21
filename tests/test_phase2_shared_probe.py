"""CPU provenance/denominator checks; no native integration or actor calls."""
from dataclasses import asdict
import ast
import copy
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from human_punching import phase2_shared_probe as probe
from human_punching.phase2_deadline import BoundedExecutionInterruption
from human_punching.phase2_expanded_evaluation import validate_negative_reference_identity
from human_punching.phase2_qualification import PanelDeclaration, TrialBinding
from human_punching.wbc.provenance import PROJECT_ROOT, sha256_file


PLAN=PROJECT_ROOT/"outputs/task10b/p2sharedprobeprep01/plan03.json"


@pytest.fixture(scope="module")
def context():
    if not PLAN.exists():pytest.skip("Actual counted four-start development preparation is not present")
    return probe.load_context(PLAN,sha256_file(PLAN))


def test_actual_capture_and_geometry_are_exact(context):
    assert context["allocation"]["bound_source_files"]==488
    assert context["witness"]["initial_state_identity_passed"]
    assert not context["witness"]["future_bitwise_parity_passed"]
    assert context["q"].dtype==np.float32
    assert context["v"].shape==(4,54)
    assert context["target"]["command_id"]==2674
    assert context["target"]["snapshot_slot_id"]==42791
    assert np.array_equal(context["q"][:,:7],np.repeat(context["q"][:1,:7],4,axis=0))
    assert not np.any(context["v"][:,:6])


def test_all_sixteen_requests_survive_contact_selection(context):
    rows=probe.requested_ledger(context)
    assert len(rows)==len({r["trial_id"] for r in rows})==16
    assert sum(r["disposition"]=="initial_forward_support_candidate_excluded" for r in rows)==12
    assert all(not r["physical_execution_attempted"] for r in rows)
    assert all(r["cycle_success_credit"]==0 for r in rows)
    for family in probe.FAMILIES:
        chosen=[r for r in rows if r["family"]==family and r["disposition"]=="unattempted"]
        assert len(chosen)==1 and chosen[0]["start_index"]==2
    for start in range(4):
        assert len({r["start_state_sha256"] for r in rows if r["start_index"]==start})==1


def test_shared_probe_solver_recipe_is_not_the_sequential_target_recipe(context):
    for family,settings in context["plan"]["retarget_settings_by_family"].items():
        assert settings["maximum_joint_delta_rad"]==.20 and settings["maximum_iterations"]==40
        changed=copy.deepcopy(context["plan"])
        changed["retarget_settings_by_family"][family]["maximum_joint_delta_rad"]=.205
        changed["retarget_settings_by_family"][family]["maximum_iterations"]=80
        with pytest.raises(ValueError):probe.validate_plan(changed)


@pytest.mark.parametrize("key,value",[("split","test"),("selected_start_index",0),
    ("preparation_s",1.),("requested_timings_s",[.6,.6,1.1,1.2]),
    ("learned_qualification",True),("g1_actor_execution",True),
    ("reference_byte_cap",2*1024**3),("start_quantization_version","legacy")])
def test_recipe_cannot_expand_or_change_hidden_request(context,key,value):
    plan=copy.deepcopy(context["plan"]);plan[key]=value
    with pytest.raises(ValueError):probe.validate_plan(plan)


def test_input_byte_identity_required_before_context_read(context):
    with pytest.raises(ValueError,match="Changed or oversized"):
        probe.load_context(PLAN,"f"*64)


@pytest.mark.parametrize("mode",["generate","execute"])
def test_no_counted_cuda_means_no_output_or_physics(monkeypatch,tmp_path,mode):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    monkeypatch.setattr(probe.torch.cuda,"is_available",lambda:False)
    out=tmp_path/"uncreated"
    with pytest.raises(RuntimeError,match="counted Slurm CUDA"):
        if mode=="generate":probe.generate(PLAN,"f"*64,out,wall_seconds=360)
        else:probe.execute(PLAN,"f"*64,{},out,wall_seconds=360)
    assert not out.exists()


def test_exact_event_offsets_and_complete_denominator(context):
    for row in context["parents"]:
        source=probe.source_schedule(row)
        assert source.preparation_complete_tick==0
        assert source.launch_tick==row["launch_tick"]
        assert source.guard_hold_tick==int(np.ceil(row["guard_time_s"]/.005-1e-10))
        shifted=probe.PhaseSchedule(160,160+source.launch_tick,160+source.outward_complete_tick,
            160+source.guard_hold_tick,160+source.reference_end_tick)
        assert abs(shifted.phase_denominator_s-row["phase_denominator_s"]-.8)<1e-12
        assert shifted.launch_tick%4==0


def test_all_generation_failures_are_immutable_negative_receipts(context,monkeypatch,tmp_path):
    monkeypatch.setattr(probe,"require_native",lambda:None)
    monkeypatch.setattr(probe,"load_context",lambda *a:context)
    def reject(*args,**kwargs):raise ValueError("synthetic bounded construction rejection")
    monkeypatch.setattr(probe,"compose_shared_reference",reject)
    result=probe.generate(PLAN,sha256_file(PLAN),tmp_path/"negative",wall_seconds=360)
    assert result["complete"] and len(result["cells"])==4
    for row in result["cells"]:
        assert validate_negative_reference_identity(row)
        assert "reference_path" not in row and "legacy_runtime_path" not in row
        assert row["generation_failure_sha256"]==sha256_file(row["generation_failure_path"])
        assert not row["physical_qualification"]
    ledger=json.loads((tmp_path/"negative/requested_scenarios.json").read_text())
    assert len(ledger)==16 and all(not r["physical_execution_attempted"] for r in ledger)


def test_bounded_generation_stop_preserves_three_unattempted_families(context,monkeypatch,tmp_path):
    monkeypatch.setattr(probe,"require_native",lambda:None)
    monkeypatch.setattr(probe,"load_context",lambda *a:context)
    def stop(*args,**kwargs):raise BoundedExecutionInterruption({"boundary":"fixture","stop_requested":True,"reason":"wall_budget"})
    monkeypatch.setattr(probe,"compose_shared_reference",stop)
    result=probe.generate(PLAN,sha256_file(PLAN),tmp_path/"partial",wall_seconds=360)
    assert not result["complete"] and result["executed_cells"]==1
    assert result["cells"][0]["generation_disposition"]=="timeout"
    assert result["unattempted_families"]==["cross","left_hook","right_hook"]
    assert len(json.loads((tmp_path/"partial/requested_scenarios.json").read_text()))==16


def test_small_development_declaration_is_not_a_learned_panel():
    trial=TrialBinding("development",2674,0,.6,2,"a"*64,"b"*64,(.8,.2,1.2))
    d=probe.DevelopmentDeclaration("jab","c"*64,"d"*64,"e"*64,(trial,))
    assert d.split=="train" and len(d.bindings)==1
    assert not hasattr(d,"fresh_transitions")
    with pytest.raises(TypeError):PanelDeclaration(**asdict(d))


def test_independent_auditor_preserves_all_parent_physical_algorithms():
    parent=ast.parse((PROJECT_ROOT/"scripts/task10b_phase2_audit_initial_v2.py").read_text())
    child=ast.parse((PROJECT_ROOT/"scripts/task10b_phase2_audit_shared_probe.py").read_text())
    def names(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=names(parent),names(child);assert set(a)==set(b)
    for name in a:
        if name!="audit_run":assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    # Only protocol dispatch and explicit result ancestry labels differ.
    left=copy.deepcopy(a["audit_run"]);right=copy.deepcopy(b["audit_run"])
    def ignored(n):
        if not isinstance(n,ast.Assign):return False
        target=n.targets[0]
        if isinstance(target,ast.Name):return target.id in ("development_probe","expanded","template")
        return isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant) and target.slice.value in (
            "development_probe_only","development_probe_parent_auditor_sha256")
    left.body=[n for n in left.body if not ignored(n)];right.body=[n for n in right.body if not ignored(n)]
    assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)


def test_protected_sources_are_not_mutated(context):
    packet=json.loads((PROJECT_ROOT/"outputs/task10b/p2sc01/packet.json").read_text())
    for name in ("src/human_punching/phase2_initial_state_v2.py","src/human_punching/phase2_native_state.py",
            "src/human_punching/phase2_shared_evaluation_v2.py","src/human_punching/phase2_shared_catalog_v2.py",
            "scripts/task10b_phase2_audit_initial_v2.py"):
        assert sha256_file(PROJECT_ROOT/name)==packet["files"][name]


@pytest.fixture
def transport():
    path=PROJECT_ROOT/'outputs/task10b/p2sharedprobeprep01/driver.py'
    spec=importlib.util.spec_from_file_location('shared_probe_test_transport',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_transport_has_only_the_fixed_generation_entry(transport):
    command=transport.native_command(dict(mode='shared-probe-generation',output='outputs/task10b/example/run',
        inputs=dict(plan=dict(path='outputs/task10b/p2sharedprobeprep01/plan03.json',sha256='a'*64))),360)
    assert command[3:6]==['human_punching.phase2_shared_probe','generate','--plan']
    assert command[-2:]==['--wall-seconds','360']
    assert '--catalog' not in command


@pytest.mark.parametrize('remaining',[180,360.1])
def test_transport_rejects_oversized_or_exhausted_child(transport,remaining):
    with pytest.raises(ValueError):transport.native_command(dict(mode='shared-probe-generation'),remaining)


def test_transport_preserves_evaluation_reserve_and_has_no_submission_side_effect(transport):
    tree=ast.parse(Path(transport.__file__).read_text())
    submit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='submit')
    calls=[n for n in ast.walk(submit) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='admit_reservation']
    assert len(calls)==1 and calls[0].args[1].value==480
    assert {k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}==dict(evaluation_reserve_seconds=4500,successful_route_frozen=False)
