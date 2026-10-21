"""Synthetic request-completeness checks; no execution or qualification evidence."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from human_punching import phase2_requested_inventory_v1 as inv
from human_punching.wbc.splits import assign_trajectory_groups
TEST_GROUP=next(i for i in range(100) if assign_trajectory_groups([i])[i]=="test")


def request(key, **changes):
    row = dict(request_id=key, origin_request_id=key, family="jab", command_id=1, original_replica_id=0,
        split="d0_development", ttc_s=.6, start_index=0, start_mode="family_guard",
        start_state_sha256="a"*64, start_protocol_sha256="b"*64, target_w=[1.,0.,1.5],
        target_input_sha256="c"*64, style="declared_original", checkpoint_sha256="d"*64,
        predicate_sha256="e"*64, reference_sha256="f"*64)
    row.update(changes); return row


def declaration(rows, scope="historical_exploratory"):
    if scope=="final_domain":
        for row in rows:row.update(split="test",command_id=TEST_GROUP,original_replica_id=2)
    return dict(schema=inv.DECLARATION, scope=scope,
                request_ids=[r["request_id"] for r in rows], requests=rows,
                origin=dict(kind="panel_declaration" if scope=="final_domain" else "request_list",
                            binding=dict(path="synthetic/original.json",sha256="0"*64)))


def dispositions(rows, statuses=None):
    statuses = statuses or ["unattempted"]*len(rows)
    return dict(schema=inv.DISPOSITIONS, rows=[dict(request_id=r["request_id"],
        request_identity=inv.validate_request(r), status=s, reason="Synthetic retained disposition",
        evidence=[] if s=="unattempted" else [dict(path="synthetic/evidence.json",sha256="1"*64)])
        for r,s in zip(rows,statuses,strict=True)])


def assemble(rows, statuses=None, scope="historical_exploratory"):
    return inv.build_inventory([r["request_id"] for r in rows], [declaration(rows,scope)], dispositions(rows,statuses))


def test_all_negative_and_unattempted_requests_retained_without_legal_claim():
    states=list(inv.STATES); rows=[request(str(i),reference_sha256=None if s=="generation_failed" else "f"*64)
                                 for i,s in enumerate(states)]
    result=assemble(rows,states)
    assert result["count"]==6 and result["request_ids"]==[str(i) for i in range(6)]
    assert result["scopes"]["historical_exploratory"]["dispositions"]==dict.fromkeys(states,1)
    assert all(r["legal"] is None and r["qualification"]=="not_evaluated" for r in result["requests"])
    assert not result["qualification"] and not result["physical_feasibility_inferred"]


@pytest.mark.parametrize("where",["expected_missing","expected_extra","declaration_missing","declaration_extra",
                                  "disposition_missing","disposition_extra","duplicate_declaration","duplicate_disposition"])
def test_missing_extra_and_duplicate_ids_fail_closed(where):
    rows=[request("a"),request("b")]; expected=["a","b"]; dec=[declaration(rows)]; disp=dispositions(rows)
    if where=="expected_missing":expected.pop()
    elif where=="expected_extra":expected.append("c")
    elif where=="declaration_missing":dec[0]["requests"].pop()
    elif where=="declaration_extra":dec[0]["requests"].append(request("c"))
    elif where=="disposition_missing":disp["rows"].pop()
    elif where=="disposition_extra":disp["rows"].append(dispositions([request("c")])["rows"][0])
    elif where=="duplicate_declaration":dec.append(deepcopy(dec[0]))
    elif where=="duplicate_disposition":disp["rows"].append(deepcopy(disp["rows"][0]))
    with pytest.raises(ValueError):inv.build_inventory(expected,dec,disp)


@pytest.mark.parametrize("field,value",[("target_w",[1.001,0.,1.5]),("ttc_s",.75),
    ("start_state_sha256","2"*64),("start_protocol_sha256","2"*64),("checkpoint_sha256","2"*64),
    ("predicate_sha256","2"*64),("reference_sha256","2"*64),("style","changed")])
def test_disposition_cannot_be_attached_to_changed_physical_request(field,value):
    row=request("a"); disp=dispositions([row]); row[field]=value
    with pytest.raises(ValueError,match="identity"):inv.build_inventory(["a"],[declaration([row])],disp)


def test_distinct_state_history_checkpoint_and_scope_are_never_merged():
    a=request("old"); b=request("new",start_state_sha256="2"*64,checkpoint_sha256="3"*64)
    result=inv.build_inventory(["old","new"],[declaration([a]),declaration([b],"final_domain")],dispositions([a,b]))
    assert len(result["requests"])==2
    assert result["requests"][0]["request_identity"]!=result["requests"][1]["request_identity"]
    assert result["scopes"]["final_domain"]["count"]==1
    assert not result["shared_start_choice_established"]


def test_final_mask_preserves_rejections_and_excludes_exploratory_domain():
    a=request("old",ttc_s=.5,command_id=TEST_GROUP); b=request("final"); c=request("seed2",checkpoint_sha256="2"*64)
    result=inv.build_inventory(["old","final","seed2"],
        [declaration([a]),declaration([b,c],"final_domain")],dispositions([a,b,c]))
    mask=dict(schema="task10b_phase2_declared_evaluation_mask_v1",requests=[dict(admitted=False,**{k:b[k] for k in inv.MASK_FIELDS})])
    receipt=inv.validate_final_mask(result,mask,"4"*64)
    assert receipt["mask_cells"]==1 and list(receipt["request_ids_by_cell"].values())==[["final","seed2"]]
    assert mask["requests"][0]["admitted"] is False and receipt["qualification"] is False
    mask["requests"].append(dict(admitted=True,**{k:a[k] for k in inv.MASK_FIELDS}))
    with pytest.raises(ValueError,match="missing or adds"):inv.validate_final_mask(result,mask,"4"*64)


@pytest.mark.parametrize("field",["start_protocol_sha256","target_input_sha256","predicate_sha256","style"])
def test_mask_cannot_collapse_semantically_distinct_requests(field):
    a=request("a"); b=request("b",**{field:"different" if field=="style" else "2"*64})
    result=assemble([a,b],scope="final_domain")
    mask=dict(schema="task10b_phase2_declared_evaluation_mask_v1",requests=[dict(admitted=False,**{k:a[k] for k in inv.MASK_FIELDS})])
    with pytest.raises(ValueError,match="incompatible"):inv.validate_final_mask(result,mask,"4"*64)


@pytest.mark.parametrize("change",["empty_evidence","executed_no_reference","extra_feasible_flag","nonfinite","wrong_split"])
def test_no_status_or_malformed_request_can_create_feasibility(change):
    row=request("a"); status="executed"
    if change=="executed_no_reference":row["reference_sha256"]=None
    if change=="nonfinite":row["target_w"][0]=float("nan")
    if change=="wrong_split":row.update(command_id=3307,split="test")
    if change=="extra_feasible_flag":row["feasible"]=True
    with pytest.raises(ValueError):
        disp=dispositions([row],[status])
        if change=="empty_evidence":disp["rows"][0]["evidence"]=[]
        inv.build_inventory(["a"],[declaration([row])],disp)


def test_factory_hashes_every_input_and_never_overwrites(tmp_path,monkeypatch):
    from human_punching import phase2_target_catalog
    from human_punching.wbc.provenance import sha256_file
    monkeypatch.setattr(phase2_target_catalog,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(inv,"writable_path",lambda p:p)
    def bind(name,data):
        p=tmp_path/name;p.write_text(json.dumps(data));return dict(path=str(p),sha256=sha256_file(p))
    rows=[request("unattempted")]
    declared=declaration(rows)
    declared["origin"]=dict(kind="request_list",binding=bind("original.json",dict(requests=rows)),
                            rows_key="requests",id_key="origin_request_id")
    plan=bind("plan.json",dict(schema=inv.PLAN,request_ids=["unattempted"],
        declarations=[bind("requests.json",declared)],dispositions=bind("dispositions.json",dispositions(rows)),final_mask=None))
    result=inv.create_inventory(plan["path"],tmp_path/"out")
    assert result["provenance"]["plan"]["sha256"]==plan["sha256"]
    assert len(result["provenance"]["inputs"])==3 and result["original_source_completeness_verified"]
    with pytest.raises(FileExistsError):inv.create_inventory(plan["path"],tmp_path/"out")
    (tmp_path/"requests.json").write_text("{}")
    with pytest.raises(ValueError):inv.create_inventory(plan["path"],tmp_path/"out2")


def full_panel_origin():
    spec=importlib.util.spec_from_file_location("inventory_panel_fixture",Path(__file__).with_name("test_phase2_qualification.py"))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    panel=fixture.declaration()
    rows=[request("seed1/"+b.trial_id,origin_request_id=b.trial_id,family=panel.family,
        command_id=b.command_id,original_replica_id=b.original_replica_id,split=panel.split,
        ttc_s=b.ttc_s,start_index=b.start_index,start_state_sha256=b.start_state_sha256,
        reference_sha256=b.reference_sha256,target_w=list(b.target_w),
        checkpoint_sha256=panel.checkpoint_sha256,predicate_sha256=panel.predicate_sha256)
        for b in panel.bindings]
    d=dict(schema=inv.DECLARATION,scope="final_domain",request_ids=[r["request_id"] for r in rows],requests=rows,
           origin=dict(kind="panel_declaration",binding=dict(path="synthetic/panel.json",sha256="0"*64)))
    return asdict(panel),d


def test_full_original_192_panel_is_the_completeness_authority():
    panel,d=full_panel_origin()
    assert inv.validate_origins([d],lambda _:panel)[0]["original_count"]==192
    # Removing both duplicated lists must not hide omission from original panel.
    d["request_ids"].pop();d["requests"].pop()
    with pytest.raises(ValueError,match="omits or adds"):inv.validate_origins([d],lambda _:panel)


@pytest.mark.parametrize("field,value",[("target_w",[9.,0.,1.]),("ttc_s",.7),
    ("start_state_sha256","2"*64),("checkpoint_sha256","2"*64),("predicate_sha256","2"*64)])
def test_final_source_cannot_be_copied_with_changed_physical_meaning(field,value):
    panel,d=full_panel_origin();d["requests"][0][field]=value
    with pytest.raises(ValueError,match="changes original"):inv.validate_origins([d],lambda _:panel)


@pytest.mark.parametrize("field,value",[("split","validation"),("split","calibration"),("start_index",4),("original_replica_id",1)])
def test_final_scope_rejects_nonfinal_rows(field,value):
    panel,d=full_panel_origin();d["requests"][0][field]=value
    with pytest.raises(ValueError):inv.build_inventory(d["request_ids"],[d],dispositions(d["requests"]))


def test_historical_original_master_missing_whole_scenario_detected():
    rows=[request("a"),request("b")];d=declaration(rows[:1]);d["origin"].update(rows_key="requests",id_key="request_id")
    with pytest.raises(ValueError,match="omits or adds"):inv.validate_origins([d],lambda _:dict(requests=rows))
