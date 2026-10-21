"""Independent launcher regressions with a fake scheduler; never submit jobs."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def packet(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/"scripts/task10b_phase2.py"
    spec = importlib.util.spec_from_file_location("phase2_packet_independent_test",script)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root = tmp_path/"human";out = root/"outputs/task10b/p2fixture"
    prep = root/"outputs/task10b/p2prep"
    out.mkdir(parents=True);prep.mkdir()
    python = root/"python";python.write_text("synthetic interpreter\n")
    source = root/"input.py";source.write_text("preserved input\n")
    archive = out/"source_snapshot/input.py";archive.parent.mkdir();archive.write_bytes(source.read_bytes())
    authority = prep/"authorization.json";authority.write_text("{}\n")
    monkeypatch.setattr(module,"ROOT",root)
    monkeypatch.setattr(module,"PREP",prep)
    monkeypatch.setattr(module,"PYTHON",python)
    monkeypatch.setattr(module,"runtime_record",lambda:{"declared_pins_match":True})
    monkeypatch.setattr(module,"verify_runtime",lambda:None)
    monkeypatch.setattr(module,"required_inputs",lambda r:{"input.py"})
    r=dict(schema=module.SCHEMA,phase=2,mode="references",family=None,
        allocation_wall_seconds=900,application_wall_seconds=840,gpus=1,
        automatic_followup=False,promotion=False,g1_actor_execution=False,combined_scene=False,
        training=False,seed=21021,num_envs=128,rollout_steps=64,max_updates=0,
        fresh_transition_cap=0,products_bytes_cap=3*1024**3,
        partition="mit_normal_gpu",excluded_nodes=["node3006"],runtime={"declared_pins_match":True},
        bundle="outputs/task10/release_baseline_02",release="outputs/four-punch-milestone",
        predicate="outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json",
        output=str((out/"run").relative_to(root)),authority_sha256=module.sha256_file(authority),
        interpreter_sha256=module.sha256_file(python),files={"input.py":module.sha256_file(source)})
    r["submission_command"] = module.submission_command(r,out)
    path=out/"packet.json"
    def save():path.write_text(json.dumps(r))
    save()
    return module,path,r,save


def test_complete_packet_verifies(packet):
    module,path,r,_=packet
    assert module.verify(path)==r


@pytest.mark.parametrize("change",["--gres=gpu:l40s:2","--time=999"])
def test_edited_scheduler_resources_rejected(packet,change):
    module,path,r,save=packet
    key=change.split("=")[0]+"="
    r["submission_command"]=[change if x.startswith(key) else x for x in r["submission_command"]]
    save()
    with pytest.raises(ValueError):module.verify(path)


def test_omitted_required_hash_rejected(packet):
    module,path,r,save=packet;r["files"]={};save()
    with pytest.raises(ValueError):module.verify(path)


@pytest.mark.parametrize("target",["input.py","outputs/task10b/p2fixture/source_snapshot/input.py"])
def test_live_or_archived_source_change_rejected(packet,target):
    module,path,r,save=packet;(module.ROOT/target).write_text("changed")
    with pytest.raises(ValueError):module.verify(path)


@pytest.mark.parametrize("field,value",[("output","outputs/task10b/p2other/run"),
    ("bundle","outputs/task10/another"),("predicate","outputs/task10b/another.json")])
def test_redirected_inputs_or_output_rejected(packet,field,value):
    module,path,r,save=packet;r[field]=value;save()
    with pytest.raises(ValueError):module.verify(path)


def test_timeout_reserves_full_wall_and_prevents_retry(packet,monkeypatch):
    module,path,r,_=packet;calls=[]
    def run(*args,**kwargs):
        calls.append(args);raise subprocess.TimeoutExpired(args[0],30)
    monkeypatch.setattr(module.subprocess,"run",run)
    with pytest.raises(subprocess.TimeoutExpired):module.submit(path)
    assert module.budget_status(module.ledger())["outstanding_reserved_gpu_seconds"]==900
    with pytest.raises(ValueError):module.submit(path)
    assert len(calls)==1


@pytest.mark.parametrize("returncode,stdout",[(0,"unparseable response"),(1,"12345")])
def test_ambiguous_submission_keeps_reservation(packet,monkeypatch,returncode,stdout):
    module,path,r,_=packet
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:subprocess.CompletedProcess(a,returncode,stdout,"error"))
    with pytest.raises(RuntimeError):module.submit(path)
    assert module.budget_status(module.ledger())["outstanding_reserved_gpu_seconds"]==900


def test_successful_submission_reserved_and_failed_allocation_charged(packet,monkeypatch):
    module,path,r,_=packet
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:subprocess.CompletedProcess(a,0,"12345;cluster\n",""))
    module.submit(path)
    assert module.budget_status(module.ledger())["outstanding_reserved_gpu_seconds"]==900
    def account(*a,**k):
        return subprocess.CompletedProcess(a,0,"12345|FAILED|1:0|77|cpu=4,gres/gpu=1|node3200|start|end|\n","")
    monkeypatch.setattr(module.subprocess,"run",account)
    module.account(path)
    result=module.budget_status(module.ledger())
    assert result["outstanding_reserved_gpu_seconds"]==0
    assert result["actual_terminal_gpu_seconds"]==77


def test_hash_inconsistent_terminal_cannot_release_reservation(packet,monkeypatch):
    module,path,r,_=packet
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:subprocess.CompletedProcess(a,0,"12345", ""))
    module.submit(path)
    (path.parent/"terminal_accounting.json").write_text(json.dumps(dict(
        schema="task10b_phase2_terminal_accounting_v1",job_id="12345",packet_sha256="0"*64,
        allocated_gpu_seconds=0,state="FAILED",allocated_tres={"gres/gpu":"1"})))
    with pytest.raises(ValueError):module.ledger()


def test_no_submission_beyond_pending_partition_limit(packet,monkeypatch):
    module,path,r,_=packet
    for i in range(2):
        out=module.ROOT/f"outputs/task10b/p2pending{i}";out.mkdir()
        (out/"submission_intent.json").write_text(json.dumps(dict(
            reserved_gpu_seconds=900,command=r["submission_command"])))
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:pytest.fail("Scheduler called past partition cap"))
    with pytest.raises(ValueError,match="concurrent"):module.submit(path)


@pytest.mark.parametrize("kwargs",[
    dict(reserved_seconds=float("nan")),dict(reserved_seconds=1.5),dict(reserved_seconds=True),
    dict(reserved_seconds=900,elapsed_seconds=float("nan"),terminal=True),
    dict(reserved_seconds=900,elapsed_seconds=True,terminal=True)])
def test_noninteger_accounting_rejected(packet,kwargs):
    module,_,_,_=packet
    with pytest.raises(ValueError):module.Allocation("bad",**kwargs)
