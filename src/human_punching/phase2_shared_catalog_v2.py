"""Bounded exact shared-jab preparation/target reference construction.

This creates offline candidates. A q/v geometry bank never substitutes for a
native physical/controller snapshot, capture/restore parity or native execution.
"""
from __future__ import annotations

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption
from .phase2_domain import fingerprint
from .phase2_learning import family_arguments
from .phase2_phases import PhaseSchedule
from .phase2_reference import contact_diagnostics
from .phase2_shared_reference import (RECIPE_VERSION, SharedPreparationBounds,
    SharedReferenceRejection, array_digest, compose_shared_reference)
from .phase2_shared_target_reference import (SharedTargetRecipe, compose_shared_target,
    finalize_shared_target_feedforward)
from .phase2_target_reference import RetargetSettings, RetargetRejection
from .phase2_target_catalog import owned_input, screen_chunks, CATALOG_SCHEMA, BYTE_CAP, RESIDENT_CAP
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json
from .wbc.splits import assign_trajectory_groups

SCHEMA="task10b_phase2_shared_catalog_plan_v1"
RECIPE_SCHEMA="task10b_phase2_shared_catalog_recipe_v1"
START_QUANTIZATION="task10b_shared_start_source_bounded_inward_v2"


def _identity(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def validate_plan(plan):
    if (plan.get("schema")!=SCHEMA or plan.get("family") not in ("jab","cross","left_hook","right_hook")
            or plan.get("split") not in ("train","validation","test")
            or plan.get("start_mode")!="shared_jab_saved_state"
            or plan.get("coordinate_precision")!="native_float32_canonical"
            or plan.get("domain_scope") not in ("candidate_subset","full_split")
            or type(plan.get("maximum_cells")) is not int or not 1<=plan["maximum_cells"]<=384
            or type(plan.get("reference_byte_cap")) is not int or not 0<plan["reference_byte_cap"]<=BYTE_CAP
            or type(plan.get("resident_byte_cap")) is not int or not 0<plan["resident_byte_cap"]<=RESIDENT_CAP
            or not isinstance(plan.get("cells"),list) or not 0<len(plan["cells"])<=plan["maximum_cells"]):
        raise ValueError("Explicit bounded shared-jab family/split construction plan required")
    duration=plan.get("preparation_s")
    if (isinstance(duration,bool) or not isinstance(duration,(int,float)) or not np.isfinite(duration)
            or not 0<duration<=1. or abs(round(duration/.020)*.020-duration)>1e-12):
        raise ValueError("Preparation must be explicitly bounded and use complete20ms actor intervals")
    settings=RetargetSettings(**plan["retarget_settings"])
    if settings.active_side!=("Left" if plan["family"] in ("jab","left_hook") else "Right"):
        raise ValueError("Shared target active side differs from family")
    target_recipe=SharedTargetRecipe(settings,**plan["shared_target_settings"])
    preparation_bounds=SharedPreparationBounds(**plan["preparation_bounds"])
    groups=plan.get("command_groups",[]);timings=plan.get("timings_s",[])
    bank_hashes=plan.get('start_bank_sha256_by_split',{})
    if (set(bank_hashes)!={'train','validation','test'} or not all(_identity(value) for value in bank_hashes.values())
            or bank_hashes[plan['split']]!=plan['start_bank']['sha256']):
        raise ValueError("Common frozen recipe must bind all three split-specific shared banks")
    if (not isinstance(groups,list) or not groups or len(set(groups))!=len(groups)
            or any(type(g) is not int or g<0 or g==3307 for g in groups)
            or set(assign_trajectory_groups(groups).values())!={plan["split"]}
            or not isinstance(timings,list) or len(timings)!=(2 if plan["split"]=="train" else 3)
            or len(set(timings))!=len(timings) or timings!=sorted(timings)
            or any(isinstance(t,bool) or not isinstance(t,(float,int)) or not np.isfinite(t) or t<.5 for t in timings)):
        raise ValueError("Exact isolated command split and two train/three held-out timings required")
    names=set()
    for entry in plan["cells"]:
        if set(entry)!={"candidate_id","parent_candidate_id","target_candidate_identity","start_index"}:
            raise ValueError("Shared cell binds only exact candidate/parent/target/start identities")
        name=entry["candidate_id"]
        if (not isinstance(name,str) or not name or Path(name).name!=name or name in (".","..") or name in names
                or type(entry["start_index"]) is not int or entry["start_index"] not in range(4)
                or not entry["parent_candidate_id"] or not _identity(entry["target_candidate_identity"])):
            raise ValueError("Safe unique candidate IDs, exact target and four explicit start indices required")
        names.add(name)
    return target_recipe,preparation_bounds


def construct_shared_starts(nominal,metadata,domain,split):
    """New initial bank: inward native rounding, exact source-radius bounds."""
    original=np.asarray(nominal,float)
    if original.shape!=(55,) or not np.isfinite(original).all():raise ValueError("Complete finite canonical jab required")
    native=original.astype(np.float32).astype(float)
    qa=np.asarray(metadata["qpos_addresses"],int);limits=np.asarray(metadata["joint_limits_rad"],float)
    if set(qa.tolist())!=set(range(7,55)) or limits.shape!=(48,2):raise ValueError("Accepted48 hinge addresses and limits required")
    if np.any(original[qa]<limits[:,0]) or np.any(original[qa]>limits[:,1]):
        raise ValueError("Original accepted source joint lies outside physical limits")
    perturb=domain["starts"][split];dq=np.asarray(perturb["joint_position_delta_rad"],float);dv=np.asarray(perturb["generalized_velocity_delta"],float)
    if (dq.shape!=(4,48) or dv.shape!=(4,54) or not np.isfinite(dq).all() or not np.isfinite(dv).all()
            or np.any(dv[:,:6]) or np.max(abs(dq))>.015+1e-12 or np.max(abs(dv))>.06+1e-12):
        raise ValueError("Frozen joint-only D3 perturbations required")
    low=np.maximum(np.maximum(original[qa]-.015,native[qa]-.015),limits[:,0])
    high=np.minimum(np.minimum(original[qa]+.015,native[qa]+.015),limits[:,1])
    if np.any(low>=high):raise ValueError("Nonempty bounded joint interval required")
    def inward(value,lower,upper,toward):
        result=np.asarray(value,np.float32)
        for _ in range(4):
            outside=(result.astype(float)<lower)|(result.astype(float)>upper)
            if not np.any(outside):return result
            result[outside]=np.nextafter(result[outside],np.asarray(toward,np.float32)[outside])
        raise ValueError("Cannot represent new shared candidate within exact inward bounds")
    interior=(low+high)/2
    baseline=native.astype(np.float32);baseline[qa]=inward(np.clip(native[qa],low,high),low,high,interior)
    corrections=[]
    for i,address in enumerate(qa):
        if native[address]<limits[i,0] or native[address]>limits[i,1]:
            if abs(native[address]-original[address])>abs(float(np.spacing(np.float32(native[address]))))/2+1e-15:
                raise ValueError("Only outward native quantization at a valid source limit is supported")
            corrections.append(dict(joint_name=metadata["joint_names"][i],qpos_address=int(address),source_value_rad=float(original[address]),
                native_value_rad=float(native[address]),inward_baseline_value_rad=float(baseline[address]),
                delta_from_native_rad=float(baseline[address]-native[address]),delta_from_source_rad=float(baseline[address]-original[address])))
    q=np.tile(native.astype(np.float32),(4,1));v=np.zeros((4,54),np.float32)
    for start in range(4):
        q[start,qa]=inward(np.clip(native[qa]+dq[start],low,high),low,high,interior)
        v[start]=inward(np.clip(dv[start],-.06,.06),np.full(54,-.06),np.full(54,.06),np.zeros(54))
    if (np.any(q[:,qa].astype(float)<low) or np.any(q[:,qa].astype(float)>high)
            or not np.array_equal(q[:,:7],np.tile(native[:7].astype(np.float32),(4,1)))):
        raise ValueError("New D3 geometry altered root or exceeded exact source/native bounds")
    import hashlib
    return q,v,dict(quantization_version=START_QUANTIZATION,native_nominal_outward_rounding=corrections,
        source_nominal_sha256=hashlib.sha256(original.tobytes()).hexdigest(),native_baseline_sha256=hashlib.sha256(native.astype(np.float32).tobytes()).hexdigest(),
        inward_baseline_sha256=hashlib.sha256(baseline.tobytes()).hexdigest(),root_native_geometry_unchanged=True,
        actual_joint_position_delta_rad=(q[:,qa].astype(float)-original[qa]).tolist(),actual_generalized_velocity_delta=v.astype(float).tolist(),
        desired_joint_position_delta_rad=dq.tolist(),desired_generalized_velocity_delta=dv.tolist(),
        no_state_coercion="New initial candidate bank only; no accepted reference overwrite, root translation, sole reconciliation or within-trial correction.")


def validate_start_bank(q,v,nominal,metadata,domain,split):
    """Exact frozen D3 construction bank, shared unchanged between families."""
    q=np.asarray(q);v=np.asarray(v)
    if q.shape!=(4,55) or v.shape!=(4,54) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("Exactly four finite complete q/v geometry starts required")
    expected_q,expected_v,_=construct_shared_starts(nominal,metadata,domain,split)
    for start in range(4):
        if not np.array_equal(q[start],expected_q[start]) or not np.array_equal(v[start],expected_v[start]):
            raise ValueError("Shared start differs from canonical nativefloat32 jab plus the frozen D3 row")
    return [array_digest(q[i],v[i]) for i in range(4)]


def resolve_cells(plan):
    recipe,bounds=validate_plan(plan)
    inputs={key:owned_input(plan[key]) for key in ("domain","target_geometry","parent_catalog","recipe","start_bank")}
    domain,geometry,parent,declared=(json.loads(inputs[k].read_text()) for k in ("domain","target_geometry","parent_catalog","recipe"))
    if (domain.get("schema")!="Task10BPhase2Domain/v1" or geometry.get("schema")!="Task10BPhase2RegisteredTargets/v1"
            or parent.get("schema")!=CATALOG_SCHEMA):
        raise ValueError("Canonical domain, registered targets and cached timing catalog required")
    required_recipe=dict(schema=RECIPE_SCHEMA,family=plan["family"],shared_recipe_version=RECIPE_VERSION,
        start_mode=plan["start_mode"],coordinate_precision=plan["coordinate_precision"],physics_dt_s=.005,actor_dt_s=.020,
        preparation_s=plan["preparation_s"],preparation_bounds=plan["preparation_bounds"],
        start_quantization_version=START_QUANTIZATION,
        retarget_settings=plan["retarget_settings"],shared_target_settings=plan["shared_target_settings"],
        source_schedules=plan["source_schedules"],
        start_bank_sha256_by_split=plan['start_bank_sha256_by_split'],
        **{key+"_sha256":plan[key]["sha256"] for key in ("domain","target_geometry","parent_catalog")})
    if any(declared.get(k)!=value for k,value in required_recipe.items()):
        raise ValueError("Shared recipe must bind exact clocks, all settings and source/domain/start identities")
    release=Path(plan["release"]).resolve(strict=True)
    if not release.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError("Accepted release must be human-owned")
    _,family_paths=family_arguments(release,plan["family"]);_,jab_paths=family_arguments(release,"jab")
    if (declared.get("accepted_jab_reference_sha256")!=sha256_file(jab_paths["reference"])
            or declared.get("accepted_jab_body_metadata_sha256")!=sha256_file(jab_paths["body_metadata"])):
        raise ValueError("Shared recipe lacks accepted jab reference and body ancestry")
    with np.load(jab_paths["reference"],allow_pickle=False) as archive:
        nominal=archive["qpos"][0].copy()
    start_header=_preflight(inputs["start_bank"],plan["start_bank"]["sha256"],1024**2)
    with np.load(inputs["start_bank"],allow_pickle=False) as archive:q,v=archive["qpos"].copy(),archive["qvel"].copy()
    metadata=json.loads(jab_paths["body_metadata"].read_text())
    start_ids=validate_start_bank(q,v,nominal,metadata,domain,plan["split"])
    _,_,start_quantization=construct_shared_starts(nominal,metadata,domain,plan["split"])
    saved=None
    if plan.get("native_saved_states") is not None:
        from .phase2_native_state import load_snapshot
        inputs["native_saved_states"]=owned_input(plan["native_saved_states"])
        saved=load_snapshot(inputs["native_saved_states"])
        if len(saved.states)!=4:raise ValueError("Native shared comparison bank must have exactly4 states")
        model=mujoco.MjModel.from_xml_path(str(jab_paths["model"]));data=mujoco.MjData(model)
        for i,state in enumerate(saved.states):
            state.validate_model_size(model);mujoco.mj_setState(model,data,state.native_integration_state,state.native_state_spec)
            if not np.array_equal(data.qpos,q[i]) or not np.array_equal(data.qvel,v[i]):
                raise ValueError("Actual native/controller snapshot does not contain the declared geometry start")
    groups={r["snapshot"]["command_id"]:r for r in domain["selected_command_groups"]}
    if any(groups[g]["split"]!=plan["split"] for g in plan["command_groups"]):
        raise ValueError("Command group not selected in canonical split")
    if plan["domain_scope"]=="full_split" and set(plan["command_groups"])!={g for g,r in groups.items() if r["split"]==plan["split"]}:
        raise ValueError("Full split declaration omits canonical groups")
    targets={r["candidate_identity"]:r for r in geometry["geometry_candidates"]};parents={r["candidate_id"]:r for r in parent["cells"]}
    if len(targets)!=len(geometry["geometry_candidates"]) or len(parents)!=len(parent["cells"]):raise ValueError("Duplicate geometry/parent identity")
    resolved=[];selected={};keys=set();headers={}
    for entry in plan["cells"]:
        target=targets[entry["target_candidate_identity"]];row=parents[entry["parent_candidate_id"]]
        snapshot=groups[target["command_id"]]["snapshot"]
        if (fingerprint({k:v for k,v in target.items() if k!="candidate_identity"})!=target["candidate_identity"]
                or target["source_domain_sha256"]!=plan["domain"]["sha256"]
                or target["split"]!=plan["split"] or target["command_id"] not in plan["command_groups"]
                or target["original_replica_id"]!=snapshot["replica_id"] or target["snapshot_slot_id"]!=snapshot["slot_id"]
                or target["original_replica_id"]!={"train":0,"validation":1,"test":2}[plan["split"]]
                or target["payload_sha256"]!=snapshot["payload_sha256"]
                or row["family"]!=plan["family"] or row.get("kinematic_admission") is not True
                or row.get("requested_ttc_s") not in plan["timings_s"] or row.get("preparation_complete_tick",0)!=0
                or row.get("start_mode","family_guard")!="family_guard"):
            raise ValueError("Shared request target/snapshot/family/timing binding differs")
        if selected.setdefault(target["command_id"],target["candidate_identity"])!=target["candidate_identity"]:
            raise ValueError("One immutable target placement per group across starts/timings required")
        key=(target["command_id"],row["requested_ttc_s"],entry["start_index"])
        if key in keys:raise ValueError("Duplicate shared command/timing/start request")
        keys.add(key)
        schedule=PhaseSchedule(**plan["source_schedules"][row["candidate_id"]])
        if (schedule.preparation_complete_tick!=0 or schedule.launch_tick!=row["launch_tick"]
                or schedule.outward_complete_tick!=row["outward_end_tick"]
                or abs(schedule.reference_end_tick*.005-row["duration_s"])>1e-8
                or abs(row["phase_denominator_s"]-row["duration_s"]-1.)>1e-8
                or not np.isfinite(row.get("guard_time_s",float("nan")))
                or schedule.guard_hold_tick!=int(np.ceil(row["guard_time_s"]/.005-1e-10))):
            raise ValueError("Reviewed parent phase schedule differs; no automatic event relabeling")
        source=owned_input(dict(path=row["reference_path"],sha256=row["reference_sha256"]))
        model=owned_input(dict(path=row["source_model_path"],sha256=row["source_model_sha256"]))
        if sha256_file(model)!=sha256_file(family_paths["model"]):raise ValueError("Parent model differs from accepted family model")
        if source not in headers:headers[source]=_preflight(source,row["reference_sha256"],plan["resident_byte_cap"])
        header=headers[source];shape=next(a.shape for a in header.arrays if a.name=="qpos")
        if (len(shape)!=2 or shape[1]!=55 or not 4<=shape[0]<=6001
                or header.uncompressed_array_bytes*12+(shape[0]+round(plan["preparation_s"]/.005)+1)*8192>plan["resident_byte_cap"]):
            raise ValueError("Bounded shared source/candidate/FK/IK resident allowance exceeded")
        resolved.append((entry,target,row,source,model,schedule))
    if keys!=set(itertools.product(plan["command_groups"],plan["timings_s"],range(4))):
        raise ValueError("Every declared group/timing requires all four starts; no resampling")
    total=sum(h.uncompressed_array_bytes for h in headers.values())+start_header.uncompressed_array_bytes
    if total>plan["reference_byte_cap"]:raise ValueError("Declared parent/start reference pool exceeds2GiB")
    context=dict(nominal=nominal,qpos=q.astype(float),qvel=v.astype(float),start_ids=start_ids,
        native_saved_states=saved,parent_bytes=total,target_recipe=recipe,preparation_bounds=bounds,
        start_quantization=start_quantization,
        accepted_jab_reference_sha256=sha256_file(jab_paths["reference"]))
    return resolved,inputs,context


def _events(schedule):
    return dict(preparation_complete_tick=schedule.preparation_complete_tick,launch_tick=schedule.launch_tick,
        outward_end_tick=schedule.outward_complete_tick,guard_hold_tick=schedule.guard_hold_tick,
        reference_end_tick=schedule.reference_end_tick,launch_time_s=schedule.launch_tick*.005,
        preparation_complete_time_s=schedule.preparation_complete_tick*.005,
        outward_completion_time_s=schedule.outward_complete_tick*.005,
        guard_hold_time_s=schedule.guard_hold_tick*.005,phase_semantics_version=schedule.semantics_version)


def archive_payload(candidate):
    """Save no object arrays; the reviewed schedule is explicit JSON and ticks."""
    result={key:np.asarray(value) for key,value in candidate.items() if key!="schedule"}
    result["phase_schedule_json"]=np.asarray(json.dumps(asdict(candidate["schedule"]),sort_keys=True))
    if any(value.dtype.hasobject for value in result.values()):raise ValueError("Reference archive cannot contain Python objects")
    return result


def generate(plan_path,output,*,wall_seconds):
    """Candidate construction only; caller must account the execution allocation."""
    started=time.monotonic();deadline=ExecutionDeadline(wall_seconds,
        serialization_reserve_s=min(90.,wall_seconds/4),session_started_monotonic=started)
    plan_path=owned_input(dict(path=plan_path,sha256=sha256_file(plan_path)))
    plan=json.loads(plan_path.read_text());cells,inputs,context=resolve_cells(plan)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False);write_json(folder/"plan.json",plan)
    write_json(folder/"start_quantization.json",context["start_quantization"])
    rows=[];reference_bytes=context["parent_bytes"];exception=None
    def progress():
        done={row["candidate_id"] for row in rows}
        write_json(folder/f"progress_{len(rows):04d}.json",dict(cells=rows,complete=False,
            unattempted_cells=[row for row in plan["cells"] if row["candidate_id"] not in done]))
    progress()
    with deadline.signal_handlers():
        for entry,target,parent,source_path,model_path,source_schedule in cells:
            current=None;start=entry["start_index"];saved=context["native_saved_states"]
            common=dict(**entry,family=plan["family"],split=plan["split"],start_mode="shared_jab_saved_state",
                command_id=target["command_id"],original_replica_id=target["original_replica_id"],domain_stage=target["domain_stage"],
                target_w=target["head_target_w"],head_center=target["head_target_w"],requested_ttc_s=parent["requested_ttc_s"],
                recipe_sha256=plan["recipe"]["sha256"],plan_sha256=sha256_file(plan_path),
                start_bank_path=str(inputs["start_bank"]),start_bank_sha256=plan["start_bank"]["sha256"],start_row=start,
                geometry_start_state_sha256=context["start_ids"][start],
                native_saved_state_available=saved is not None,requires_native_controller_capture=saved is None,
                saved_state_path=str(inputs["native_saved_states"]) if saved else None,
                saved_state_sha256=plan["native_saved_states"]["sha256"] if saved else None,
                saved_state_identity=saved.identity if saved else None,
                start_state_sha256=saved.states[start].digest() if saved else None,
                shared_physical_controller_identity=saved.physical_controller_identity if saved else None,
                native_capture_restore_parity_qualified=False,physical_qualification=False,fine_tuned=False)
            schedule=PhaseSchedule(round(plan["preparation_s"]/.005),
                round(plan["preparation_s"]/.005)+source_schedule.launch_tick,
                round(plan["preparation_s"]/.005)+source_schedule.outward_complete_tick,
                round(plan["preparation_s"]/.005)+source_schedule.guard_hold_tick,
                round(plan["preparation_s"]/.005)+source_schedule.reference_end_tick)
            try:
                deadline.checkpoint("shared_reference_cell",progress={"candidate_id":entry["candidate_id"]})
                current=folder/entry["candidate_id"];current.mkdir();write_json(current/"attempt_started.json",dict(entry=entry,parent=parent,target=target))
                model=mujoco.MjModel.from_xml_path(str(model_path))
                if (model.nq,model.nv,model.nu)!=(55,54,48):raise ValueError("Accepted SOMA dimensions required")
                with np.load(source_path,allow_pickle=False) as archive:source={key:archive[key].copy() for key in archive.files}
                source["time_s"]=np.asarray(source.get("time_s",source.get("t")))
                support=dict(evidence_id="shared_geometry_bank:"+common["geometry_start_state_sha256"],
                    state_qpos_qvel_sha256=common["geometry_start_state_sha256"],
                    evidence_kind="provisional_geometry_construction_only",native_support_established=False,native_execution_permitted=False)
                candidate,prep=compose_shared_reference(model,source,context["nominal"],context["qpos"][start],context["qvel"][start],
                    source_schedule=source_schedule,preparation_s=plan["preparation_s"],support_evidence=support,
                    preparation_bounds=context["preparation_bounds"])
                deadline.checkpoint("after_shared_preparation_construction")
                candidate,retarget=compose_shared_target(model,candidate,source,candidate["source_original_glove_pos"][:,int(source["active_hand_index"])],
                    target["head_target_w"],source_schedule=source_schedule,recipe=context["target_recipe"],
                    checkpoint=lambda:deadline.checkpoint("shared_target_IK_frame"))
                candidate,retarget=finalize_shared_target_feedforward(model,candidate,retarget)
                # Original reward formulas must retain own-family guard targets,
                # rather than deriving them from the new jab preparation start.
                guard_data=mujoco.MjData(model);guard_data.qpos[:]=source["qpos"][0]
                mujoco.mj_kinematics(model,guard_data)
                chest=model.body("Chest").id;hands=[model.site(side+"Hand_center").id for side in ("Left","Right")]
                candidate["family_guard_glove_chest_pos"]=(guard_data.site_xpos[hands]-guard_data.xpos[chest])@guard_data.xmat[chest].reshape(3,3)
                candidate["family_guard_qpos"]=candidate["qpos"][schedule.preparation_complete_tick].copy()
                candidate["original_family_guard_qpos"]=np.asarray(source["qpos"][0]).copy()
                payload=archive_payload(candidate);size=sum(value.nbytes for value in payload.values())
                if reference_bytes+size>plan["reference_byte_cap"]:raise ValueError("Generated shared references exceed declared2GiB data cap")
                path=current/"reference.npz"
                with path.open("xb") as stream:np.savez_compressed(stream,**payload)
                reference_bytes+=size
                contact=contact_diagnostics(candidate["t"],candidate["glove_pos"],target["head_target_w"],
                    int(candidate["active_hand_index"]),schedule.launch_tick*.005,schedule.outward_complete_tick*.005,parent["requested_ttc_s"])
                reasons=[]
                if not contact["timing_valid"]:reasons.append("shared_first_contact_outside_requested_timing")
                if contact["inactive_hand_first_or_simultaneous"]:reasons.append("wrong_hand_first_or_simultaneous")
                if contact["any_contact_before_launch"]:reasons.append("contact_before_outward_launch_including_preparation")
                screen_chunks(model_path,candidate,np.max(abs(model.actuator_forcerange),axis=1),current,deadline)
                write_json(current/"preparation.json",prep);write_json(current/"retarget.json",retarget);write_json(current/"events.json",_events(schedule))
                digest=sha256_file(path)
                rows.append(dict(**common,**_events(schedule),kind="shared_jab_continuous_preparation_target",
                    active_hand_index=int(candidate["active_hand_index"]),reference_path=str(path),reference_sha256=digest,
                    legacy_runtime_path=str(path),legacy_runtime_sha256=digest,reference_identity_kind="npz_sha256",
                    source_model_path=str(model_path),source_model_sha256=sha256_file(model_path),
                    duration_s=schedule.reference_end_tick*.005,episode_duration_s=schedule.phase_denominator_s,
                    phase_denominator_s=schedule.phase_denominator_s,guard_time_s=schedule.guard_hold_tick*.005,
                    kinematic_admission=not reasons,rejection_reasons=reasons,kinematic_contact=contact,
                    diagnostic_screen_path=str(current/"screen.json"),generation_disposition="generated",
                    preparation_s=plan["preparation_s"],no_fixed_sole_during_preparation=True,
                    own_family_guard_targets_serialized=True,
                    family_guard_target_sha256=array_digest(candidate["family_guard_glove_chest_pos"],candidate["family_guard_qpos"]),
                    guard_reward_semantics="Retain original family Chest-relative centers; joint-reward diagnostic target is own-family upper posture plus explicit shared lower geometry at preparation completion. Never use newt0jab pose as family guard; no exact-joint recovery gate.",
                    training_noise_semantics="Connector already binds one frozen D3start. Do not add anotherD3resetnoise; any sampled source-phase rehearsal requires a separately declared reset contract.",
                    sampling_contract="Exact5msgrid with future+20/+50ticks; no inherited linear-sampler continuousC2 qualification"))
            except Exception as error:
                timeout=isinstance(error,BoundedExecutionInterruption)
                failure=dict(**common,**_events(schedule),kind="shared_jab_reference_generation_failure",
                    duration_s=schedule.reference_end_tick*.005,episode_duration_s=schedule.phase_denominator_s,
                    phase_denominator_s=schedule.phase_denominator_s,kinematic_admission=False,
                    rejection_reasons=["bounded_timeout" if timeout else "shared_reference_generation_failure"],
                    generation_disposition="timeout" if timeout else "failed",
                    error=dict(type=type(error).__name__,message=str(error)),details=getattr(error,"disposition",getattr(error,"receipt",None)),
                    negative_clock_semantics="Planned offset clocks declare an unexecuted negative cell; no generated or executed contact claim",
                    retained_partial_reference_path=str(current/"reference.npz") if current and (current/"reference.npz").exists() else None)
                evidence=(current if current else folder)/(entry["candidate_id"]+"_generation_failure.json");write_json(evidence,failure)
                failure.update(reference_identity_kind="generation_failure_receipt",generation_failure_path=str(evidence),
                    generation_failure_sha256=sha256_file(evidence),reference_sha256=sha256_file(evidence))
                rows.append(failure)
                if current:write_json(current/"failure.json",failure)
                if not isinstance(error,(SharedReferenceRejection,RetargetRejection)):exception=failure["error"]
            progress()
            if exception:break
    done={row["candidate_id"] for row in rows}
    result=dict(schema=CATALOG_SCHEMA,scope="Shared-jab offline preparation/target candidates only; native saved-state/parity gates separate",
        complete=len(rows)==len(cells),planned_cells=len(cells),executed_cells=len(rows),cells=rows,
        unattempted_cells=[row for row in plan["cells"] if row["candidate_id"] not in done],exception=exception,
        reference_uncompressed_array_bytes=reference_bytes,parent_reference_uncompressed_array_bytes=context["parent_bytes"],
        generated_reference_uncompressed_array_bytes=reference_bytes-context["parent_bytes"],reference_byte_cap=plan["reference_byte_cap"],
        resident_byte_reservation=plan["resident_byte_cap"],recipe_sha256=plan["recipe"]["sha256"],plan_sha256=sha256_file(plan_path),
        input_hashes={key:sha256_file(path) for key,path in inputs.items()},source_sha256=sha256_file(__file__),
        native_saved_state_available=context["native_saved_states"] is not None,
        native_capture_restore_parity_qualified=False,physics_executed=False,physical_qualification=False,
        wall_s=time.monotonic()-started,no_automatic_route_selection=True,learned_trajectory_generator=False)
    write_json(folder/"manifest.json",result)
    return result
