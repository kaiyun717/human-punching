"""Bounded offline construction of exact snapshot-target reference cells.

This constructs candidates only. Native execution, learned qualification and
launch admission remain separate. A plan names every target and timing row;
there is no nearest-reference lookup or automatic expansion of the plan.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption
from .phase2_domain import fingerprint, assign_trajectory_groups
from .phase2_reference import contact_diagnostics, screen_diagnostics
from .phase2_target_reference import (RetargetSettings, RetargetRejection,
    retarget_upper_body, finalize_original_feedforward)
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

SCHEMA = "task10b_phase2_target_catalog_plan_v1"
CATALOG_SCHEMA = "task10b_phase2_reference_catalog_v1"
BYTE_CAP = 2*1024**3
RESIDENT_CAP = 256*1024**2
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def owned_input(record):
    path=Path(record["path"]).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
        raise ValueError("Candidate inputs must resolve inside human-punching")
    if sha256_file(path)!=record["sha256"]:
        raise ValueError("Candidate input hash changed")
    return path


def validate_plan(plan):
    if (plan.get("schema")!=SCHEMA or plan.get("family") not in FAMILIES
            or plan.get("split") not in ("train","validation","test")
            or plan.get("start_mode")!="family_guard"
            or type(plan.get("maximum_cells")) is not int or not 1<=plan["maximum_cells"]<=192
            or type(plan.get("reference_byte_cap")) is not int or not 0<plan["reference_byte_cap"]<=BYTE_CAP
            or type(plan.get("resident_byte_cap")) is not int or not 0<plan["resident_byte_cap"]<=RESIDENT_CAP
            or not isinstance(plan.get("cells"),list) or not 0<len(plan["cells"])<=plan["maximum_cells"]):
        raise ValueError("One family/split, bounded exact cells and initial reference budgets required")
    settings=RetargetSettings(**plan["retarget_settings"])
    side="Left" if plan["family"] in ("jab","left_hook") else "Right"
    if settings.active_side!=side:
        raise ValueError("Retarget side differs from requested family")
    seen=set()
    for row in plan["cells"]:
        if set(row)!={"candidate_id","parent_candidate_id","target_candidate_identity"}:
            raise ValueError("Candidate entries contain only exact cell, parent and target identities")
        name=row.get("candidate_id")
        if (not isinstance(name,str) or not name or Path(name).name!=name or name in (".","..")
                or name in seen or not row.get("parent_candidate_id") or not row.get("target_candidate_identity")):
            raise ValueError("Unique safe candidate IDs and exact parent/target identities required")
        seen.add(name)
    return settings


def resolve_cells(plan):
    """Validate immutable geometry/group/parent bindings before creating output."""
    settings=validate_plan(plan)
    paths={key:owned_input(plan[key]) for key in ("domain","target_geometry","parent_catalog","body_metadata","recipe")}
    data={key:json.loads(path.read_text()) for key,path in paths.items()}
    domain,geometry,parent=data["domain"],data["target_geometry"],data["parent_catalog"]
    if (domain.get("schema")!="Task10BPhase2Domain/v1"
            or geometry.get("schema")!="Task10BPhase2RegisteredTargets/v1"
            or parent.get("schema")!=CATALOG_SCHEMA):
        raise ValueError("Frozen canonical domain, measured targets and timing catalog required")
    recipe=data["recipe"]
    if (recipe.get("schema")!="task10b_phase2_target_recipe_v1" or recipe.get("family")!=plan["family"]
            or recipe.get("retarget_settings")!=plan["retarget_settings"]
            or any(recipe.get(key+"_sha256")!=plan[key]["sha256"] for key in ("domain","target_geometry","parent_catalog"))
            or recipe.get("start_mode")!="family_guard" or recipe.get("physics_dt_s")!=.005
            or recipe.get("actor_dt_s")!=.020):
        raise ValueError("Target recipe must bind unchanged settings, clocks and source/domain identities")
    groups={g["snapshot"]["command_id"]:g for g in domain["selected_command_groups"]}
    targets={r["candidate_identity"]:r for r in geometry["geometry_candidates"]}
    parents={r["candidate_id"]:r for r in parent["cells"]}
    if len(targets)!=len(geometry["geometry_candidates"]) or len(parents)!=len(parent["cells"]):
        raise ValueError("Duplicate target or parent identity")
    resolved=[];payloads={};selected_targets={};group_timings=set()
    for entry in plan["cells"]:
        target=targets[entry["target_candidate_identity"]];row=parents[entry["parent_candidate_id"]]
        group=groups[target["command_id"]];snapshot=group["snapshot"]
        if (fingerprint({k:v for k,v in target.items() if k!="candidate_identity"})!=target["candidate_identity"]
                or target["source_domain_sha256"]!=plan["domain"]["sha256"]
                or target["split"]!=plan["split"] or group["split"]!=plan["split"]
                or target["original_replica_id"]!=snapshot["replica_id"]
                or target["snapshot_slot_id"]!=snapshot["slot_id"]
                or target["payload_sha256"]!=snapshot["payload_sha256"]
                or target["command_id"]==3307
                or assign_trajectory_groups([target["command_id"]])[target["command_id"]]!=plan["split"]
                or target["original_replica_id"]!={"train":0,"validation":1,"test":2}[plan["split"]]
                or row["family"]!=plan["family"] or row.get("kinematic_admission") is not True
                or row.get("requested_ttc_s") is None
                or not np.isfinite(row["requested_ttc_s"]) or row["requested_ttc_s"]<.5
                or row.get("preparation_complete_tick",0)!=0
                or row.get("start_mode","family_guard")!="family_guard"):
            raise ValueError("Target/group/split/parent binding outside the declared isolated recipe")
        if selected_targets.setdefault(target["command_id"],target["candidate_identity"])!=target["candidate_identity"]:
            raise ValueError("Initial catalog fixes one placement per command group")
        if (type(row.get("launch_tick")) is not int or row["launch_tick"]%4
                or type(row.get("outward_end_tick")) is not int or not 0<=row["launch_tick"]<row["outward_end_tick"]
                or not all(isinstance(row.get(k),(int,float)) and np.isfinite(row[k]) and row[k]>0
                    for k in ("duration_s","episode_duration_s","phase_denominator_s"))
                or abs(row["episode_duration_s"]-row["duration_s"]-1.)>1.e-8
                or row["phase_denominator_s"]!=row["episode_duration_s"]
                or row["outward_end_tick"]*.005>row["duration_s"]):
            raise ValueError("Parent must declare full finite reference clocks before target generation")
        key=(target["command_id"],float(row["requested_ttc_s"]))
        if key in group_timings:raise ValueError("Duplicate command/timing variant")
        group_timings.add(key)
        source=owned_input(dict(path=row["reference_path"],sha256=row["reference_sha256"]))
        model=owned_input(dict(path=row["source_model_path"],sha256=row["source_model_sha256"]))
        if source not in payloads:payloads[source]=_preflight(source,row["reference_sha256"],plan["resident_byte_cap"])
        preflight=payloads[source]
        if preflight.uncompressed_array_bytes*4>plan["resident_byte_cap"]:
            raise ValueError("Source/candidate/work arrays exceed the conservative resident payload reservation")
        resolved.append((entry,target,row,source,model))
    if sum(r.uncompressed_array_bytes for r in payloads.values())>plan["reference_byte_cap"]:
        raise ValueError("Parent reference data exceed the declared total byte budget")
    # Accepted body metadata carries gains; actuator caps come from the source
    # model's unchanged force ranges instead of a new configurable strength.
    return settings,resolved,paths


def screen_chunks(model_path,candidate,caps,folder,deadline,chunk_frames=16):
    """Keep LP work bounded between checkpoints and preserve each completed chunk."""
    if type(chunk_frames) is not int or not 1<=chunk_frames<=16:
        raise ValueError("At most sixteen screening frames between deadline checkpoints")
    summaries=[]
    for number,begin in enumerate(range(0,len(candidate["t"]),chunk_frames)):
        deadline.checkpoint("target_screen_chunk",progress={"first_frame":begin})
        chunk={k:np.asarray(candidate[k])[begin:begin+chunk_frames] for k in ("t","qpos","qvel","qacc")}
        summary,arrays=screen_diagnostics(model_path,chunk,caps)
        with (folder/f"screen_{number:04d}.npz").open("xb") as stream:np.savez_compressed(stream,**arrays)
        write_json(folder/f"screen_{number:04d}.json",summary)
        summaries.append(summary)
    maximum=("joint_limit_excursion_max_rad","selected_self_penetration_max_m","floor_penetration_max_m",
        "joint_velocity_max_rad_s","joint_acceleration_max_rad_s2","maximum_minimax_effort_ratio",
        "maximum_base_force_residual_n","maximum_base_moment_residual_nm")
    result={k:v for k,v in summaries[0].items() if k not in maximum}
    for key in maximum:
        values=[s[key] for s in summaries if s[key] is not None]
        result[key]=max(values) if values else None
    for key in ("frames","within_effort_caps_frames","over_effort_caps_frames"):
        result[key]=sum(s[key] for s in summaries)
    counts=Counter()
    for summary in summaries:counts.update(summary["support_status_counts"])
    result.update(support_status_counts=dict(counts),chunk_frames=chunk_frames,completed_chunks=len(summaries),
        diagnostic_arrays="screen_NNNN.npz; each chunk remains an immutable result")
    write_json(folder/"screen.json",result)
    return result


def generate(plan_path,output,*,wall_seconds):
    """Run only from an allocated wrapper; never selects a learned route."""
    started=time.monotonic()
    deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=min(90.,wall_seconds/4),
        session_started_monotonic=started)
    plan_path=owned_input(dict(path=plan_path,sha256=sha256_file(plan_path)))
    plan=json.loads(plan_path.read_text());settings,cells,inputs=resolve_cells(plan)
    sources={(source,parent["reference_sha256"]) for _,_,parent,source,_ in cells}
    parent_bytes=sum(_preflight(source,digest,plan["resident_byte_cap"]).uncompressed_array_bytes for source,digest in sources)
    folder=writable_path(output)
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"plan.json",plan)
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
                candidate,metadata=retarget_upper_body(model,source,target["head_target_w"],
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
                    domain_stage=target["domain_stage"],kind="bounded_active_arm_target",start_mode="family_guard",
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
                    kind="bounded_active_arm_target_generation_failure",start_mode="family_guard",preparation_complete_tick=0,
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
    result=dict(schema=CATALOG_SCHEMA,scope="Isolated family-guard fixed snapshot targets; candidate references only",
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
