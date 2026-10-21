"""Four-cell sequential target-path probe; separate from frozen generation code.

Shares immutable target/group validation and original finalization/screens. The
new plan schema cannot be executed by the historical per-frame generator.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from . import phase2_target_catalog as original
from .phase2_target_catalog import owned_input, screen_chunks
from .phase2_target_reference import RetargetRejection, RetargetSettings, finalize_original_feedforward
from .phase2_target_path_reference_v2 import VERSION, SequentialSettings, retarget_sequential
from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption
from .phase2_reference import contact_diagnostics
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

SCHEMA = "task10b_phase2_sequential_target_catalog_plan_v2"
CATALOG_SCHEMA = original.CATALOG_SCHEMA
MAX_OUTPUT_BYTES = 128*1024**2
OUTPUT_RESERVE = 8*1024**2
MAX_CHILD_SECONDS = 240.
SOURCES = ("src/human_punching/phase2_sequential_target_catalog.py",
    "src/human_punching/phase2_target_path_reference_v2.py",
    "src/human_punching/phase2_target_catalog.py",
    "src/human_punching/phase2_target_reference.py")


def source_bindings():
    return {name:dict(path=str(PROJECT_ROOT/name),sha256=sha256_file(PROJECT_ROOT/name)) for name in SOURCES}


def select_groups(audit):
    """Select only from the completed training-generation diagnostic ledger."""
    if audit.get("passed") is not True or audit.get("generation",{}).get("planned") != 64:
        raise ValueError("Complete independent64-cell training audit required")
    cells=audit.get("cells",[])
    if len(cells)!=64 or any(not c.get("candidate_id","").startswith("jab_train_") for c in cells):
        raise ValueError("Only the immutable training ledger may select probe groups")
    failed={c["command_id"] for c in cells if c.get("generation_disposition")=="failed"}
    if len(failed)!=1:
        raise ValueError("Exactly one retained failed training group is required")
    failed_group=next(iter(failed))
    candidates=[c for c in cells if "maximum_5ms_correction_step_rad" in c and c["command_id"]!=failed_group]
    if not candidates:
        raise ValueError("A distinct generated training group is required")
    worst=max(candidates,key=lambda c:(c["maximum_5ms_correction_step_rad"],-c["command_id"]))
    return failed_group,worst["command_id"]


def validate_plan(plan):
    if (plan.get("schema")!=SCHEMA or plan.get("family")!="jab" or plan.get("split")!="train"
            or plan.get("path_solver_version")!=VERSION or plan.get("maximum_cells")!=4
            or len(plan.get("cells",[]))!=4 or plan.get("output_byte_cap")!=MAX_OUTPUT_BYTES
            or plan.get("maximum_child_seconds")!=MAX_CHILD_SECONDS or set(plan.get("sources",{}))!=set(SOURCES)):
        raise ValueError("Exact source-bound four-cell sequential training probe required")
    for name,record in plan["sources"].items():
        if owned_input(record)!=(PROJECT_ROOT/name).resolve(strict=True):
            raise ValueError("Sequential source must resolve to its exact owned module")
    settings=SequentialSettings(RetargetSettings(**plan["retarget_settings"]))
    audit=json.loads(owned_input(plan["selection_audit"]).read_text())
    groups=select_groups(audit)
    if plan.get("selected_command_groups")!=list(groups):
        raise ValueError("Probe groups differ from the prespecified training-only selection")
    # Reuse structural validation without rewriting any plan/reference receipt.
    common=deepcopy(plan);common["schema"]=original.SCHEMA
    original.validate_plan(common)
    return settings,groups,common


def resolve_cells(plan):
    settings,groups,common=validate_plan(plan)
    _,cells,inputs=original.resolve_cells(common)
    recipe=json.loads(inputs["recipe"].read_text())
    if (recipe.get("path_solver_version")!=VERSION or recipe.get("sequential_preference_weight")!=.002
            or recipe.get("solver_sources")!=plan["sources"]
            or recipe.get("selection_audit")!=plan["selection_audit"]):
        raise ValueError("Distinct sequential recipe and exact solver source bindings required")
    actual=[(target["command_id"],parent["requested_ttc_s"]) for _,target,parent,_,_ in cells]
    expected=[(g,t) for g in groups for t in (.6,.75)]
    if actual!=expected or any(not row[0]["candidate_id"].endswith("_sequential_v2_probe") for row in cells):
        raise ValueError("Four exact group/timing identities and distinct sequential candidate IDs required")
    inputs["selection_audit"]=owned_input(plan["selection_audit"])
    return settings,cells,inputs


def require_allocation(wall_seconds):
    if (isinstance(wall_seconds,bool) or not isinstance(wall_seconds,(int,float))
            or not math.isfinite(wall_seconds) or not 60<=wall_seconds<=MAX_CHILD_SECONDS):
        raise ValueError("Sequential child wall must be finite60..240seconds")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Sequential construction requires a counted Slurm GPU allocation")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Sequential construction requires the allocated CUDA device")


def output_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


class OutputBoundedDeadline:
    """Bound output between each IK update and each existing16-frame screen chunk."""
    def __init__(self,deadline,folder,limit):
        self.deadline,self.folder,self.limit=deadline,folder,limit
    def checkpoint(self,*args,**kwargs):
        self.deadline.checkpoint(*args,**kwargs)
        # IK iterations do not write files. Rechecking the same tree for every
        # solve step would spend the bounded session on metadata traffic.
        if args and args[0]=="target_IK_frame":
            return
        if output_bytes(self.folder)>self.limit-OUTPUT_RESERVE:
            raise ValueError("Sequential output limit reached;8MiB remains for bounded chunk/failure/manifest flushing")
    def signal_handlers(self):
        return self.deadline.signal_handlers()


def generate(plan_path,output,*,wall_seconds):
    """Run only from an allocated wrapper; never selects a learned route."""
    started=time.monotonic()
    require_allocation(wall_seconds)
    deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=min(90.,wall_seconds/4),
        session_started_monotonic=started)
    plan_path=owned_input(dict(path=plan_path,sha256=sha256_file(plan_path)))
    plan=json.loads(plan_path.read_text());settings,cells,inputs=resolve_cells(plan)
    sources={(source,parent["reference_sha256"]) for _,_,parent,source,_ in cells}
    parent_bytes=sum(_preflight(source,digest,plan["resident_byte_cap"]).uncompressed_array_bytes for source,digest in sources)
    folder=writable_path(output)
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"plan.json",plan)
    deadline=OutputBoundedDeadline(deadline,folder,plan["output_byte_cap"])
    rows=[];reference_bytes=parent_bytes;exception=None
    def progress():
        done={r["candidate_id"] for r in rows}
        write_json(folder/f"progress_{len(rows):04d}.json",dict(cells=rows,complete=False,
            unattempted_cells=[r for r in plan["cells"] if r["candidate_id"] not in done]))
    progress()
    with deadline.signal_handlers():
        for entry,target,parent,source_path,model_path in cells:
            current=None
            try:
                deadline.checkpoint("target_reference_cell",progress={"candidate_id":entry["candidate_id"]})
                current=folder/entry["candidate_id"];current.mkdir(exist_ok=False)
                write_json(current/"attempt_started.json",dict(entry=entry,parent=parent,target=target))
                model=mujoco.MjModel.from_xml_path(str(model_path))
                if (model.nq,model.nv,model.nu)!=(55,54,48):raise ValueError("Accepted SOMA dimensions required")
                with np.load(source_path,allow_pickle=False) as archive:source={k:archive[k].copy() for k in archive.files}
                source["time_s"]=np.asarray(source.get("time_s",source.get("t")))
                if len(source["time_s"])>6001:raise ValueError("Declared maximum reference frame count exceeded")
                if abs(source["time_s"][-1]-parent["duration_s"])>1.e-8:
                    raise ValueError("Parent horizon differs from its bound reference")
                candidate,metadata=retarget_sequential(model,source,target["head_target_w"],
                    source_outward_start_s=parent["launch_tick"]*.005,
                    source_outward_end_s=parent["outward_end_tick"]*.005,settings=settings,
                    checkpoint=lambda:deadline.checkpoint("target_IK_frame"))
                candidate,metadata=finalize_original_feedforward(model,candidate,metadata)
                arrays={key:np.asarray(value) for key,value in candidate.items()}
                if any(value.dtype.kind not in "biufcSU" or
                       (value.dtype.kind in "biufc" and not np.isfinite(value).all()) for value in arrays.values()):
                    raise ValueError("Candidate archives require finite numeric or explicit string arrays")
                payload=sum(value.nbytes for value in arrays.values())
                if reference_bytes+payload>plan["reference_byte_cap"]:raise ValueError("Generated reference byte cap would be exceeded")
                if output_bytes(folder)+payload>plan["output_byte_cap"]-OUTPUT_RESERVE:
                    raise ValueError("Sequential reference output cap reserve exhausted")
                path=current/"reference.npz"
                with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
                reference_bytes+=payload
                events=metadata["events"]
                contact=contact_diagnostics(candidate["t"],candidate["glove_pos"],target["head_target_w"],
                    int(candidate["active_hand_index"]),events["launch_time_s"],events["outward_completion_time_s"],parent["requested_ttc_s"])
                reasons=[]
                if not contact["timing_valid"]:reasons.append("retargeted_first_contact_outside_requested_timing")
                if contact["inactive_hand_first_or_simultaneous"]:reasons.append("wrong_hand_first_or_simultaneous")
                if contact["any_contact_before_launch"]:reasons.append("contact_before_launch")
                screen_chunks(model_path,candidate,np.max(np.abs(model.actuator_forcerange),axis=1),current,deadline)
                write_json(current/"retarget.json",metadata);write_json(current/"events.json",events)
                digest=sha256_file(path)
                rows.append(dict(**entry,**events,family=plan["family"],split=plan["split"],
                    command_id=target["command_id"],original_replica_id=target["original_replica_id"],
                    domain_stage=target["domain_stage"],kind="bounded_active_arm_sequential_target_v2",path_solver_version=VERSION,start_mode="family_guard",
                    preparation_complete_tick=0,target_w=target["head_target_w"],head_center=target["head_target_w"],
                    active_hand_index=int(candidate["active_hand_index"]),requested_ttc_s=parent["requested_ttc_s"],
                    reference_path=str(path),reference_sha256=digest,legacy_runtime_path=str(path),legacy_runtime_sha256=digest,
                    reference_identity_kind="reference_content",
                    source_model_path=str(model_path),source_model_sha256=sha256_file(model_path),
                    duration_s=float(candidate["t"][-1]),episode_duration_s=float(candidate["t"][-1])+1.,
                    phase_denominator_s=float(candidate["t"][-1])+1.,guard_time_s=float(candidate["guard_hold_start"]),
                    kinematic_admission=not reasons,rejection_reasons=reasons,kinematic_contact=contact,
                    diagnostic_screen_path=str(current/"screen.json"),generation_disposition="generated",
                    recipe_sha256=plan["recipe"]["sha256"],plan_sha256=sha256_file(plan_path),physical_qualification=False,fine_tuned=False))
            except Exception as error:
                reason="bounded_timeout" if isinstance(error,BoundedExecutionInterruption) else "target_reference_generation_failure"
                failure=dict(**entry,family=plan["family"],split=plan["split"],command_id=target["command_id"],
                    original_replica_id=target["original_replica_id"],requested_ttc_s=parent["requested_ttc_s"],
                    target_w=target["head_target_w"],head_center=target["head_target_w"],domain_stage=target["domain_stage"],
                    kind="bounded_active_arm_sequential_target_generation_failure_v2",path_solver_version=VERSION,start_mode="family_guard",preparation_complete_tick=0,
                    launch_tick=parent["launch_tick"],outward_end_tick=parent["outward_end_tick"],
                    launch_time_s=parent["launch_tick"]*.005,outward_completion_time_s=parent["outward_end_tick"]*.005,
                    duration_s=parent["duration_s"],episode_duration_s=parent["episode_duration_s"],phase_denominator_s=parent["phase_denominator_s"],
                    negative_clock_semantics="Parent clocks retained only to declare a negative/unexecuted cell; no generated-target event or executed timing is claimed",
                    recipe_sha256=plan["recipe"]["sha256"],kinematic_admission=False,rejection_reasons=[reason],
                    generation_disposition="timeout" if reason=="bounded_timeout" else "failed",physical_qualification=False,
                    retained_partial_reference_path=str(current/"reference.npz") if current is not None and (current/"reference.npz").exists() else None,
                    error=dict(type=type(error).__name__,message=str(error)),
                    details=getattr(error,"disposition",getattr(error,"receipt",None)))
                evidence_path=(current if current is not None else folder)/(entry["candidate_id"]+"_generation_failure.json")
                write_json(evidence_path,failure)
                failure.update(reference_identity_kind="generation_failure_receipt",generation_failure_path=str(evidence_path),
                    generation_failure_sha256=sha256_file(evidence_path),reference_sha256=sha256_file(evidence_path))
                rows.append(failure)
                if current is not None:write_json(current/"failure.json",failure)
                if not isinstance(error,RetargetRejection):exception=failure["error"]
            progress()
            if exception:break
    done={r["candidate_id"] for r in rows}
    result=dict(schema=CATALOG_SCHEMA,path_solver_version=VERSION,output_byte_cap=plan["output_byte_cap"],output_bytes_before_manifest=output_bytes(folder),scope="Isolated family-guard fixed snapshot targets; candidate references only",
        complete=len(rows)==len(cells),planned_cells=len(cells),executed_cells=len(rows),cells=rows,
        generated_cells=sum(r["generation_disposition"]=="generated" for r in rows),
        admitted_candidate_cells=sum(r["kinematic_admission"] is True for r in rows),
        unattempted_cells=[r for r in plan["cells"] if r["candidate_id"] not in done],
        exception=exception,reference_uncompressed_array_bytes=reference_bytes,reference_byte_cap=plan["reference_byte_cap"],
        parent_reference_uncompressed_array_bytes=parent_bytes,generated_reference_uncompressed_array_bytes=reference_bytes-parent_bytes,
        resident_byte_reservation=plan["resident_byte_cap"],input_hashes={k:sha256_file(v) for k,v in inputs.items()},
        recipe_sha256=plan["recipe"]["sha256"],
        source_sha256=sha256_file(Path(__file__)),plan_sha256=sha256_file(plan_path),wall_s=time.monotonic()-started,
        physics_executed=False,learned_trajectory_generator=False,physical_qualification=False,
        no_automatic_route_selection=True,final_test_domain_or_mask_tuned=False)
    write_json(folder/"manifest.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args(argv)
    result=generate(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['complete'] and result['exception'] is None else 1


if __name__=='__main__':raise SystemExit(main())
