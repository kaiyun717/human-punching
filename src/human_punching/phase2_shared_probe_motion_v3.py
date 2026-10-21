"""Separate source-relative root/lower-motion shared development probe.

Original static-pelvis and shape-only failures remain immutable. The source
pelvis and per-foot relative motion is explicit; lower reference IK alone
adapts the common stance. Exact captured history and all physical gates remain.

New root and lower-joint reference knots require fresh tangent derivatives,
hinge interpolation, screening and qualification. The analytic connector is
retained, and source phase correspondence is checked against the new motion.

Generation and native execution are separate counted allocations.  This small
training-split experiment never constructs a 96/192-attempt declaration or uses
a final-test payload.  The other twelve start/family requests remain explicit
initial-forward contact exclusions, not executed failures or proven instability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .phase2_deadline import BoundedExecutionInterruption, ExecutionDeadline
from .phase2_cache import _preflight
from .phase2_domain import fingerprint
from .phase2_expanded_evaluation import (EvaluationLimits, FrozenSnapshotInput,
    launch_requests, validate_geometry_bindings, write_launch_boundaries)
from .phase2_initial_capture_v2 import validate_capture_allocation
from .phase2_initial_state_v2 import _saved, binding, owned, read_bound, validate_witness
from .phase2_learning import evaluate_policy, family_arguments, fresh_policy, policy_digest
from .phase2_phases import PhaseSchedule
from .phase2_qualification import TrialBinding
from .phase2_shared_catalog_v2 import (START_QUANTIZATION, _events, archive_payload,
    validate_start_bank)
from .phase2_shared_evaluation_v2 import MANIFEST, VERSION as INITIALIZATION, SharedEvaluationProtocol, validate_native_bank
from .phase2_shared_reference_shape_v3 import SharedPreparationBounds, array_digest
from .phase2_shared_motion import VERSION as MOTION_VERSION,LowerMotionSettings
from .phase2_shared_motion_reference import compose_shared_motion_reference as compose_shared_reference,verify_motion_events
from .phase2_shape_derivatives import VERSION as PROJECTION_VERSION
from .phase2_shared_motion_target import refine_motion_target_derivatives as refine_shared_target_derivatives
from .phase2_shared_target_reference import (SharedTargetRecipe, compose_shared_target,
    finalize_shared_target_feedforward)
from .phase2_target_catalog import screen_chunks
from .phase2_target_reference import RetargetSettings
from .phase2_reference import contact_diagnostics
from .scalar_control import export_runtime_model
from .wbc.provenance import PROJECT_ROOT, runtime_record, sha256_file, writable_path, write_json

VERSION = "task10b_phase2_four_family_shared_motion_development_probe_v3"
LAUNCH = "task10b_phase2_shared_development_probe_launch_v1"
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SOURCE_CELLS = ("jab_uniform_0p600", "cross_uniform_0p600",
    "left_hook_uniform_1p100", "right_hook_uniform_1p000")
TIMINGS = (.60, .60, 1.10, 1.00)
INPUTS = ("domain", "target_geometry", "parent_catalog", "geometry",
    "native_bank", "capture", "allocation", "witness", "predicate",
    "predecessor_plan", "predecessor_generation_manifest", "algebraic_review",
    "motion_proposal", "failed_shape_generation_manifest")
REFERENCE_CAP = 64*1024**2
RESIDENT_CAP = 256*1024**2


def retarget_settings(family):
    side="Left" if family in ("jab","left_hook") else "Right"
    names=tuple(side+n for n in ("Shoulder_x","Shoulder_y","Shoulder_z","Arm_x","Arm_y","Arm_z",
        "ForeArm_z","ForeArm_x","Hand_x","Hand_y","Hand_z"))
    return RetargetSettings(side,names)


def validate_plan(plan):
    if (plan.get("schema") != VERSION or plan.get("split") != "train"
            or plan.get("derivative_projection_version") != PROJECTION_VERSION
            or plan.get("motion_version") != MOTION_VERSION
            or plan.get("lower_motion_settings") != asdict(LowerMotionSettings())
            or plan.get("families") != list(FAMILIES)
            or plan.get("parent_candidate_ids") != list(SOURCE_CELLS)
            or plan.get("requested_timings_s") != list(TIMINGS)
            or plan.get("selected_start_index") != 2 or plan.get("preparation_s") != .8
            or plan.get("start_quantization_version") != START_QUANTIZATION
            or plan.get("learned_qualification") is not False
            or plan.get("g1_actor_execution") is not False
            or plan.get("reference_byte_cap") != REFERENCE_CAP
            or plan.get("resident_byte_cap") != RESIDENT_CAP
            or fingerprint(plan.get("retarget_settings_by_family")) != fingerprint(
                {f:asdict(retarget_settings(f)) for f in FAMILIES})
            or plan.get("shared_target_settings") != dict(maximum_combined_path_displacement_m=.18)
            or plan.get("preparation_bounds") != asdict(SharedPreparationBounds())
            or plan.get("reference_anchor") != "explicit_inward_native_nominal_geometry_row0"
            or set(plan.get("inputs", {})) != set(INPUTS)
            or not isinstance(plan.get("target_candidate_identity"), str)):
        raise ValueError("Fixed four-family, one-training-target, row2/.8s development plan required")
    for item in plan["inputs"].values():
        if (set(item) != {"path", "sha256"} or not isinstance(item["path"], str)
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in item["sha256"])):
            raise ValueError("Every immutable probe input needs an exact owned path and SHA256")
    return plan


def load_context(plan_path, expected_sha256):
    plan_path, plan = read_bound(dict(path=str(plan_path), sha256=expected_sha256))
    validate_plan(plan)
    paths = {name: owned(item["path"]) for name, item in plan["inputs"].items()}
    if any(sha256_file(paths[n]) != plan["inputs"][n]["sha256"] for n in INPUTS):
        raise ValueError("Frozen shared-development input changed")
    predecessor=json.loads(paths["predecessor_plan"].read_text())
    failed=json.loads(paths["predecessor_generation_manifest"].read_text())
    algebra=json.loads(paths["algebraic_review"].read_text())
    projection_source=next((r for r in algebra.get("implementation",[])
        if Path(r.get("path","")).name=="phase2_shape_derivatives.py"),{})
    if (predecessor.get("schema")!="task10b_phase2_four_family_shared_development_probe_v1"
            or any(fingerprint(plan.get(k))!=fingerprint(v) for k,v in predecessor.items()
                if k not in ("schema","inputs"))
            or any(plan["inputs"].get(k)!=v for k,v in predecessor.get("inputs",{}).items())
            or failed.get("recipe_sha256")!=plan["inputs"]["predecessor_plan"]["sha256"]
            or failed.get("complete") is not True or len(failed.get("cells",[]))!=4
            or any(r.get("kinematic_admission") is not False or r.get("generation_disposition")!="failed"
                for r in failed["cells"])
            or algebra.get("passed") is not True or algebra.get("projection_version")!=PROJECTION_VERSION
            or algebra.get("failed_recipe_plan",{}).get("sha256")!=plan["inputs"]["predecessor_plan"]["sha256"]
            or projection_source.get("sha256")!=sha256_file(Path(__file__).with_name("phase2_shape_derivatives.py"))
            or len(algebra.get("families",[]))!=4
            or any(r.get("new_failed_interval_joint_pairs")!=0 for r in algebra["families"])):
        raise ValueError("Exact failed predecessor, unchanged requests and successful saved-source algebraic evidence required")
    motion_proposal=json.loads(paths["motion_proposal"].read_text())
    shape_failure=json.loads(paths["failed_shape_generation_manifest"].read_text())
    if (motion_proposal.get("schema")!="task10b_shared_relative_pelvis_lower_body_proposal_v1"
            or motion_proposal.get("preferred_recipe",{}).get("version")!=MOTION_VERSION
            or shape_failure.get("probe_version")!="task10b_phase2_four_family_shared_shape_development_probe_v2"
            or shape_failure.get("complete") is not True or len(shape_failure.get("cells",[]))!=4
            or any(r.get("kinematic_admission") is not False or r.get("details",{}).get("reason")!="shared_geometry_compensation_outside_recipe" for r in shape_failure["cells"])):
        raise ValueError("Exact reviewed motion proposal and preserved shape recipe failures required")
    domain = json.loads(paths["domain"].read_text())
    targets = json.loads(paths["target_geometry"].read_text())
    parent = json.loads(paths["parent_catalog"].read_text())
    if (domain.get("schema") != "Task10BPhase2Domain/v1"
            or targets.get("schema") != "Task10BPhase2RegisteredTargets/v1"
            or parent.get("schema") != "task10b_phase2_reference_catalog_v1"):
        raise ValueError("Canonical domain/registered-target/timing inputs required")
    target = next(t for t in targets["geometry_candidates"] if t["candidate_identity"] == plan["target_candidate_identity"])
    group = next(g for g in domain["selected_command_groups"] if g["snapshot"]["command_id"] == target["command_id"])
    if (target["split"] != "train" or target["domain_stage"] != "D1"
            or group["split"] != "train" or group["domain_stage"] != "D1"
            or target["command_id"] == 3307
            or target["source_domain_sha256"] != plan["inputs"]["domain"]["sha256"]
            or target["original_replica_id"] != 0
            or target["payload_sha256"] != group["snapshot"]["payload_sha256"]
            or target["snapshot_slot_id"] != group["snapshot"]["slot_id"]
            or target["placement"] != dict(forward_m=0., left_m=0., yaw_rad=0.)
            or fingerprint({k:v for k,v in target.items() if k != "candidate_identity"}) != target["candidate_identity"]):
        raise ValueError("One exact standing training snapshot at its declared nominal placement required")
    motion = np.asarray(group["measured_velocity_body_vx_vy_native_gyro_z"], float)
    if not np.isfinite(motion).all() or np.linalg.norm(motion[:2]) > .10 or abs(motion[2]) > .10:
        raise ValueError("Measured D1 motion admission differs")
    allocation = validate_capture_allocation(plan["inputs"]["allocation"], capture=plan["inputs"]["capture"])
    capture = json.loads(paths["capture"].read_text())
    witness = validate_witness(plan["inputs"]["witness"])
    bank = _saved(paths["native_bank"])
    if (capture["split"] != "train" or capture["domain_sha256"] != plan["inputs"]["domain"]["sha256"]
            or capture["native_start_state_bank"]["sha256"] != plan["inputs"]["native_bank"]["sha256"]
            or capture["native_start_state_bank"]["identity"] != bank.identity
            or capture["shared_physical_controller_identity"] != bank.physical_controller_identity
            or capture["geometry_start_bank"]["sha256"] != plan["inputs"]["geometry"]["sha256"]
            or capture["initial_identity_witness"] != witness
            or capture["start_quantization_version"] != START_QUANTIZATION):
        raise ValueError("Exact counted shared-bank capture/witness identity required")
    _, jab_paths = family_arguments(PROJECT_ROOT/"outputs/four-punch-milestone", "jab")
    model = mujoco.MjModel.from_xml_path(str(jab_paths["model"]))
    q, v = validate_native_bank(bank, model, sha256_file(jab_paths["reference"]))
    with np.load(paths["geometry"], allow_pickle=False) as archive:
        if not np.array_equal(q, archive["qpos"]) or not np.array_equal(v, archive["qvel"]):
            raise ValueError("Captured geometry differs from the declared shared bank")
    with np.load(jab_paths["reference"], allow_pickle=False) as archive: nominal = archive["qpos"][0].copy()
    validate_start_bank(q, v, nominal, json.loads(jab_paths["body_metadata"].read_text()), domain, "train")
    if [i for i, s in enumerate(bank.states) if all(s.support_contact)] != [2]:
        raise ValueError("This development selection is frozen to the measured row2 initial-forward contact candidate")
    parents = []
    for family, name, timing in zip(FAMILIES, SOURCE_CELLS, TIMINGS, strict=True):
        row = next(r for r in parent["cells"] if r["candidate_id"] == name)
        if (row["family"] != family or row["requested_ttc_s"] != timing or row["kinematic_admission"] is not True
                or row.get("start_mode", "family_guard") != "family_guard"):
            raise ValueError("Prespecified passing isolated timing parent differs")
        for kind in ("reference", "source_model"):
            if sha256_file(owned(row[kind+"_path"])) != row[kind+"_sha256"]:
                raise ValueError("Parent reference/model changed")
        header=_preflight(owned(row["reference_path"]),row["reference_sha256"],REFERENCE_CAP)
        shape=next(a.shape for a in header.arrays if a.name=="qpos")
        if (len(shape)!=2 or shape[1]!=55 or not 4<=shape[0]<=6001
                or header.uncompressed_array_bytes*12+(shape[0]+161)*8192>RESIDENT_CAP):
            raise ValueError("Bounded source/candidate/screen reference workspace exceeded")
        parents.append(row)
    return dict(plan=plan, plan_path=plan_path, paths=paths, domain=domain, targets=targets,
        target=target, bank=bank, q=q, v=v, parents=parents, allocation=allocation, witness=witness)


def requested_ledger(context):
    target = context["target"]
    return [dict(trial_id=f"shared_probe_{family}_start_{start}", family=family, start_index=start,
        start_state_sha256=context["bank"].states[start].digest(), requested_ttc_s=ttc,
        target_candidate_identity=target["candidate_identity"], target_w=target["head_target_w"],
        support_contact_initial_forward=context["bank"].states[start].support_contact.tolist(),
        disposition="unattempted" if start == 2 else "initial_forward_support_candidate_excluded",
        reason="not_reached" if start == 2 else "not_both_feet_in_initial_forward_contact_candidate",
        support_semantics="Initial native forward classification only; no executed interval support or instability inferred",
        physical_execution_attempted=False, numeric_complete_cycle=False, cycle_success_credit=0)
        for family,ttc in zip(FAMILIES,TIMINGS,strict=True) for start in range(4)]


def require_native():
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Reference screens and continuous probe require a counted Slurm CUDA allocation")


def source_schedule(row):
    return PhaseSchedule(0, row["launch_tick"], row["outward_end_tick"],
        int(math.ceil(row["guard_time_s"]/.005-1.e-10)), int(round(row["duration_s"]/.005)))


def refine_then_finalize(model,candidate,metadata):
    """Final source derivatives precede original joint-only feedforward."""
    candidate,metadata=refine_shared_target_derivatives(model,candidate,metadata)
    events=verify_motion_events(candidate,PhaseSchedule(**metadata["source_schedule"]))
    metadata={**metadata,"motion_event_correspondence_after_target":events}
    return finalize_shared_target_feedforward(model,candidate,metadata)


def generate(plan_path, expected_sha256, output, *, wall_seconds):
    require_native()
    if not 180 < wall_seconds <= 360: raise ValueError("Finite generation child must be180..360seconds")
    deadline = ExecutionDeadline(wall_seconds, serialization_reserve_s=60.)
    context = load_context(plan_path, expected_sha256)
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"plan.json", context["plan"])
    write_json(folder/"requested_scenarios.json", requested_ledger(context))
    rows=[]; exception=None; total=0
    with deadline.signal_handlers():
        for parent in context["parents"]:
            family=parent["family"]; current=folder/family; current.mkdir()
            schedule=source_schedule(parent); prep_ticks=160
            full_schedule=PhaseSchedule(prep_ticks, prep_ticks+schedule.launch_tick,
                prep_ticks+schedule.outward_complete_tick, prep_ticks+schedule.guard_hold_tick,
                prep_ticks+schedule.reference_end_tick)
            common=dict(candidate_id="shared_motion_probe_"+family, family=family, split="train", start_mode="shared_jab_saved_state",
                start_index=2,start_row=2,command_id=context["target"]["command_id"],original_replica_id=0,
                target_candidate_identity=context["target"]["candidate_identity"],target_w=context["target"]["head_target_w"],
                head_center=context["target"]["head_target_w"],requested_ttc_s=parent["requested_ttc_s"],
                recipe_sha256=expected_sha256, start_bank_path=str(context["paths"]["geometry"]),
                start_bank_sha256=context["plan"]["inputs"]["geometry"]["sha256"],
                native_saved_state_available=True,saved_state_sha256=context["plan"]["inputs"]["native_bank"]["sha256"],
                saved_state_identity=context["bank"].identity,start_state_sha256=context["bank"].states[2].digest(),
                **_events(full_schedule), duration_s=full_schedule.reference_end_tick*.005,
                episode_duration_s=full_schedule.phase_denominator_s,phase_denominator_s=full_schedule.phase_denominator_s,
                guard_time_s=full_schedule.guard_hold_tick*.005,source_model_path=str(owned(parent["source_model_path"])),
                source_model_sha256=parent["source_model_sha256"],active_hand_index=parent["active_hand_index"],
                source_parent_candidate_id=parent["candidate_id"],source_parent_reference_sha256=parent["reference_sha256"],
                physical_qualification=False,fine_tuned=False)
            try:
                deadline.checkpoint("shared_probe_reference_generation")
                model=mujoco.MjModel.from_xml_path(str(owned(parent["source_model_path"])))
                with np.load(owned(parent["reference_path"]),allow_pickle=False) as archive: source={k:archive[k].copy() for k in archive.files}
                source["time_s"]=source.get("time_s",source.get("t"))
                support=dict(evidence_id="geometry_candidate_only:"+array_digest(context["q"][2],context["v"][2]),
                    state_qpos_qvel_sha256=array_digest(context["q"][2],context["v"][2]),
                    evidence_kind="provisional_geometry_construction_only",native_support_established=False,native_execution_permitted=False)
                # The explicitly reviewed inward-native row0 is a reference
                # anchor. It never replaces any physical/controller state.
                candidate,prep=compose_shared_reference(model,source,context["q"][0].astype(float),
                    context["q"][2].astype(float),context["v"][2].astype(float),source_schedule=schedule,
                    preparation_s=.8,support_evidence=support,
                    preparation_bounds=SharedPreparationBounds(**context["plan"]["preparation_bounds"]),
                    motion_settings=LowerMotionSettings(**context["plan"]["lower_motion_settings"]),
                    checkpoint=lambda:deadline.checkpoint("shared_probe_lower_motion_IK"))
                settings=RetargetSettings(**context["plan"]["retarget_settings_by_family"][family])
                candidate,retarget=compose_shared_target(model,candidate,source,
                    candidate["source_original_glove_pos"][:,common["active_hand_index"]],np.asarray(common["target_w"]),
                    source_schedule=schedule,recipe=SharedTargetRecipe(settings,**context["plan"]["shared_target_settings"]),
                    checkpoint=lambda:deadline.checkpoint("shared_probe_target_IK"))
                candidate,retarget=refine_then_finalize(model,candidate,retarget)
                guard=mujoco.MjData(model); guard.qpos[:]=source["qpos"][0];mujoco.mj_kinematics(model,guard)
                chest=model.body("Chest").id;hands=[model.site(s+"Hand_center").id for s in ("Left","Right")]
                candidate["family_guard_glove_chest_pos"]=(guard.site_xpos[hands]-guard.xpos[chest])@guard.xmat[chest].reshape(3,3)
                candidate["family_guard_qpos"]=candidate["qpos"][160].copy()
                candidate["original_family_guard_qpos"]=np.asarray(source["qpos"][0]).copy()
                payload=archive_payload(candidate);total+=sum(v.nbytes for v in payload.values())
                if total>REFERENCE_CAP:raise ValueError("Four-reference development pool exceeds64MiB")
                reference=current/"reference.npz"
                with reference.open("xb") as stream:np.savez_compressed(stream,**payload)
                contact=contact_diagnostics(candidate["t"],candidate["glove_pos"],common["target_w"],
                    common["active_hand_index"],full_schedule.launch_tick*.005,full_schedule.outward_complete_tick*.005,
                    common["requested_ttc_s"])
                reasons=[]
                if not contact["timing_valid"]:reasons.append("shared_first_contact_outside_requested_timing")
                if contact["inactive_hand_first_or_simultaneous"]:reasons.append("wrong_hand_first_or_simultaneous")
                if contact["any_contact_before_launch"]:reasons.append("contact_before_outward_launch_including_preparation")
                screen_chunks(owned(parent["source_model_path"]),candidate,np.max(abs(model.actuator_forcerange),axis=1),current,deadline)
                write_json(current/"preparation.json",prep);write_json(current/"retarget.json",retarget)
                rows.append(dict(**common,kind="shared_jab_continuous_preparation_target",reference_identity_kind="npz_sha256",
                    reference_path=str(reference),reference_sha256=sha256_file(reference),legacy_runtime_path=str(reference),
                    legacy_runtime_sha256=sha256_file(reference),kinematic_admission=not reasons,rejection_reasons=reasons,
                    generation_disposition="generated",diagnostic_screen_path=str(current/"screen.json"),kinematic_contact=contact))
            except Exception as error:
                timeout=isinstance(error,BoundedExecutionInterruption)
                failure=dict(**common,kind="shared_jab_reference_generation_failure",kinematic_admission=False,
                    rejection_reasons=["bounded_timeout" if timeout else "shared_probe_generation_failure"],
                    generation_disposition="timeout" if timeout else "failed",
                    error=dict(type=type(error).__name__,message=str(error)),
                    details=getattr(error,"disposition",getattr(error,"receipt",None)))
                evidence=current/"generation_failure.json";write_json(evidence,failure)
                rows.append(failure|dict(reference_identity_kind="generation_failure_receipt",reference_sha256=sha256_file(evidence),
                    generation_failure_path=str(evidence),generation_failure_sha256=sha256_file(evidence)))
                if isinstance(error,BoundedExecutionInterruption):exception=failure["error"]
            write_json(folder/f"progress_{len(rows):02d}.json",dict(cells=rows,unattempted_families=list(FAMILIES[len(rows):])))
            if exception:break
    result=dict(schema="task10b_phase2_reference_catalog_v1",probe_version=VERSION,cells=rows,recipe_sha256=expected_sha256,
        source_plan=binding(plan_path),source_sha256=sha256_file(__file__),complete=len(rows)==4,exception=exception,
        planned_cells=4,executed_cells=len(rows),reference_uncompressed_array_bytes=total,
        unattempted_families=list(FAMILIES[len(rows):]),
        native_physics_steps=0,actor_calls=0,learned_qualification=False,full_domain_qualification=False)
    write_json(folder/"manifest.json",result)
    return result


@dataclass(frozen=True)
class DevelopmentDeclaration:
    family: str
    checkpoint_sha256: str
    predicate_sha256: str
    declared_mask_sha256: str
    bindings: tuple[TrialBinding,...]
    split: str="train"
    start_mode: str="shared_jab_saved_state"

    @property
    def identity(self):return fingerprint(asdict(self))


def execute(plan_path,expected_sha256,catalog_item,output,*,wall_seconds):
    require_native()
    if not 180<wall_seconds<=360:raise ValueError("Finite native probe child must be180..360seconds")
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=60.)
    context=load_context(plan_path,expected_sha256);_,catalog=read_bound(catalog_item)
    if (catalog.get("probe_version")!=VERSION or catalog.get("source_plan")!=binding(plan_path)
            or catalog.get("source_sha256")!=sha256_file(__file__) or catalog.get("recipe_sha256")!=expected_sha256):
        raise ValueError("Generated probe references differ from frozen source/recipe")
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    ledger=requested_ledger(context);write_json(folder/"planned_trials.json",ledger)
    write_json(folder/"generation.json",catalog);panels=[];exception=None
    rows={r["family"]:r for r in catalog["cells"]}
    if len(rows)!=len(catalog["cells"]) or not set(rows)<=set(FAMILIES):raise ValueError("Duplicate/unrequested generated family")
    with deadline.signal_handlers():
        for family in FAMILIES:
            selected=next(r for r in ledger if r["family"]==family and r["start_index"]==2)
            row=rows.get(family);env=None;batch=None;requests=[]
            if row is None:continue
            if row.get("kinematic_admission") is not True:
                selected.update(disposition="generation_failure" if row.get("generation_disposition")!="generated" else "kinematic_rejection",
                    reason=row["rejection_reasons"],reference_sha256=row["reference_sha256"]);continue
            try:
                deadline.checkpoint("before_shared_probe_family")
                args,paths=family_arguments(PROJECT_ROOT/"outputs/four-punch-milestone",family)
                trial=TrialBinding(selected["trial_id"],context["target"]["command_id"],0,row["requested_ttc_s"],2,
                    selected["start_state_sha256"],row["reference_sha256"],tuple(row["target_w"]))
                run=folder/family;run.mkdir();binding_dict=asdict(trial)
                mask=dict(schema="task10b_phase2_declared_evaluation_mask_v1",requests=[binding_dict|dict(family=family,admitted=True,reason="fixed_development_candidate")])
                mask_path=run/"mask.json";write_json(mask_path,mask)
                declaration=DevelopmentDeclaration(family,sha256_file(paths["checkpoint"]),context["plan"]["inputs"]["predicate"]["sha256"],sha256_file(mask_path),(trial,))
                manifest=dict(schema=MANIFEST,initialization_version=INITIALIZATION,declaration_identity=declaration.identity,
                    domain_path=str(context["paths"]["domain"]),domain_sha256=context["plan"]["inputs"]["domain"]["sha256"],
                    target_geometry_path=str(context["paths"]["target_geometry"]),target_geometry_sha256=context["plan"]["inputs"]["target_geometry"]["sha256"],
                    native_start_state_bank=dict(path=str(context["paths"]["native_bank"]),
                        sha256=context["plan"]["inputs"]["native_bank"]["sha256"],identity=context["bank"].identity),
                    native_start_state_capture=dict(path=str(context["paths"]["capture"]),sha256=context["plan"]["inputs"]["capture"]["sha256"]),
                    native_capture_allocation=dict(path=str(context["paths"]["allocation"]),sha256=context["plan"]["inputs"]["allocation"]["sha256"]),
                    native_initial_identity_witness=dict(path=str(context["paths"]["witness"]),sha256=context["plan"]["inputs"]["witness"]["sha256"]),
                    initial_execution_state_identity_version="task10b_initial_execution_state_identity_v2",
                    recipe_sha256=expected_sha256,trials=[dict(trial_id=trial.trial_id,candidate_id=row["candidate_id"],
                        target_candidate_identity=context["target"]["candidate_identity"],start_row=2)])
                protocol=SharedEvaluationProtocol()
                entries,_,starts,scope=protocol.preflight(declaration,manifest,dict(cells=[row]),
                    limits=EvaluationLimits(wall_seconds,shutdown_reserve_s=60.,maximum_worlds=1),
                    mask_decisions={trial.trial_id:dict(admitted=True)},release=PROJECT_ROOT/"outputs/four-punch-milestone",route={})
                validate_geometry_bindings(declaration,manifest,context["domain"],context["targets"],entries,{row["candidate_id"]:row})
                write_json(run/"physical_declaration.json",asdict(declaration));write_json(run/"manifest.json",manifest)
                write_json(run/"catalog.json",dict(cells=[row]))
                # The strict independent auditor consumes actual absolute job
                # paths; never rewrite a recorded runtime namespace later.
                absolute_binding=lambda p:dict(path=str(p),sha256=sha256_file(p))
                configuration=dict(manifest=absolute_binding(run/"manifest.json"),mask=absolute_binding(mask_path),
                    predicate=absolute_binding(context["paths"]["predicate"]))
                write_json(run/"configuration.json",configuration)
                write_json(run/"launch.json",dict(schema=LAUNCH,slurm_job_id=os.environ["SLURM_JOB_ID"],runtime=runtime_record(),
                    catalog=str(run/"catalog.json"),catalog_sha256=sha256_file(run/"catalog.json"),recipe_sha256=expected_sha256,
                    release_inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in paths.items()},
                    configuration=absolute_binding(run/"configuration.json"),physical_request_template_identity=declaration.identity,
                    physical_declaration=absolute_binding(run/"physical_declaration.json"),domain=absolute_binding(context["paths"]["domain"]),
                    target_geometry=absolute_binding(context["paths"]["target_geometry"]),evaluated_checkpoint=dict(path=str(paths["checkpoint"]),sha256=declaration.checkpoint_sha256),
                    native_execution_requested=True,checkpoint_template_is_unbound=False,development_only=True,
                    physical_scope=scope,validation_protocol=protocol.metadata,
                    learned_qualification=False,full_domain_qualification=False,formal_acceptance=False))
                q,v=starts[trial.trial_id];requests=launch_requests(declaration,[trial],row,{context["target"]["candidate_identity"]:context["target"]})
                # Durable intent precedes native construction, so a lost actor
                # or trace can never become a never-attempted request.
                selected.update(physical_execution_attempted=True,disposition="attempt_started",request_identity=trial.identity)
                write_json(run/"attempt_started.json",dict(trial=binding_dict,request_identity=trial.identity))
                with native_stream_scope():
                    env=protocol.environment(args,row,[trial],deadline=deadline,qpos=q[None],qvel=v[None],requests=requests,target_input=FrozenSnapshotInput(requests))
                    policy=fresh_policy(paths,"cuda");policy.eval();env.bind_policy(policy);before=policy_digest(policy)
                    export_runtime_model(args.model,run/"runtime.xml",env.model)
                    batch=run/"batch_0000"
                    panel=evaluate_policy(env,policy,batch,reference_ids=np.zeros(1,dtype=int),initial_qpos=q[None],initial_qvel=v[None],
                        seed=10010,predicate=json.loads(context["paths"]["predicate"].read_text()),trial_ids=[trial.trial_id])
                    initialization=protocol.audit_initialization(batch,env,[trial],q[None],v[None])
                    if policy_digest(policy)!=before:raise RuntimeError("Frozen probe policy mutated")
                selected.update(disposition="completed_native_attempt" if panel["trials"] else "incomplete_native_attempt",
                    native_result=panel["trials"][0] if panel["trials"] else None,numeric_complete_cycle=False,
                    internal_numeric_result_pending_independent_audit=bool(panel["trials"] and panel["trials"][0]["numeric_complete_cycle"]),
                    initialization_internal_pass=initialization["passed"],independent_audit_pending=True,visual_review="pending",
                    raw_trace_path=str(batch/"raw_trace.npz"),reference_sha256=row["reference_sha256"])
                write_json(run/"summary.json",dict(trials=[selected],learned_qualification=False,full_domain_qualification=False))
                panels.append(dict(family=family,path=str(run),status=panel["status"]))
                if panel["status"]!="executed" or not initialization["passed"] or panel["hard_faults"]:
                    raise RuntimeError("Shared development recipe paused after native/identity fault")
            except Exception as error:
                exception=dict(family=family,type=type(error).__name__,message=str(error))
                if selected["physical_execution_attempted"]:selected.update(disposition="incomplete_native_attempt",reason=exception)
                else:selected.update(reason=exception)
            finally:
                if env is not None and batch is not None and batch.exists():
                    boundaries=write_launch_boundaries(batch,env.closed_boundary_snapshots)
                    write_json(batch/"closed_protocol_launch.json",dict(schema="task10b_phase2_closed_protocol_native_receipt_v1",
                        requests=requests,initialization_version=INITIALIZATION,decisions=env.closed_launch_records,
                        boundary_snapshots=boundaries,physical_state_or_history_replaced=False))
                if env is not None:del env
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                write_json(folder/f"ledger_{family}.json",ledger)
            if exception:break
    result=dict(schema=VERSION,mode="native_probe",trials=ledger,panels=panels,exception=exception,
        planned_requested_scenarios=16,initial_forward_candidate_exclusions=12,
        planned_native_cycles=4,physical_attempts=sum(r["physical_execution_attempted"] for r in ledger),
        completed_native_attempts=sum(r["disposition"]=="completed_native_attempt" for r in ledger),
        independent_audit_and_video_pending=True,learned_qualification=False,full_domain_qualification=False,
        four_family_choice_qualified=False,source_plan=binding(plan_path),source_catalog=catalog_item)
    write_json(folder/"summary.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("generate","execute"))
    p.add_argument("--plan",type=Path,required=True);p.add_argument("--plan-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--wall-seconds",type=float,required=True)
    p.add_argument("--catalog",type=Path);p.add_argument("--catalog-sha256");a=p.parse_args()
    if a.mode=="generate":result=generate(a.plan,a.plan_sha256,a.output,wall_seconds=a.wall_seconds)
    else:
        if a.catalog is None or a.catalog_sha256 is None:p.error("execute requires the exact counted generation catalog")
        result=execute(a.plan,a.plan_sha256,dict(path=str(a.catalog),sha256=a.catalog_sha256),a.output,wall_seconds=a.wall_seconds)
    print(json.dumps(result,sort_keys=True));return 0 if result.get("exception") is None else 1


if __name__=="__main__":raise SystemExit(main())
