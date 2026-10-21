"""Thin, explicitly selected D0 timing pilot around the unchanged original PPO.

No route is selected here. A reviewed route/video receipt must name the exact
finite training rows. Interior timings remain excluded, and nonuniform catalog
rows keep their original training_eligible=False marker: only a new, separately
bound route authorization can make those exact rows eligible for this pilot.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .phase2_learning import (PilotBudget, NativeSpecialistEnv, family_arguments,
    fresh_policy, train_pilot)
from .phase2_d0_clock_v2 import run_cell_panel, evaluator_receipt
from .phase2_interiors import (ROOT, BINS, CATALOG, owned, bound, binding, read,
    sha, logical_path, _write_json, TOTAL_BYTES, RESIDENT_BYTES)
from .phase2_cache import _preflight

VERSION="task10b_phase2_selected_d0_timing_pilot_v1"
SCHEMA="task10b_phase2_selected_d0_timing_pilot_plan_v1"
ROUTE_SCHEMA="task10b_phase2_d0_route_selection_v1"
VIDEO_SCHEMA="task10b_phase2_d0_video_review_v1"
RELEASE_FILES={"arguments":"arguments.json","schema":"controller_schema.json","checkpoint":"checkpoint.pt",
    "model":"model/human.xml","body_metadata":"body_metadata.json","reference":"reference.npz",
    "bank":"evaluation/evaluation_bank.npz"}
SOURCES=("src/human_punching/phase2_timing_pilot.py","scripts/task10b_phase2_timing_pilot.py",
    "src/human_punching/phase2_learning.py","src/human_punching/phase2_d0_clock_v2.py",
    "src/human_punching/phase2_interiors.py","src/human_punching/phase2_cache.py",
    "src/human_punching/jab_learning.py","src/human_punching/scalar_control.py",
    "src/human_punching/functional_recovery.py","src/human_punching/contact.py",
    "src/human_punching/wbc/release_baseline_probe.py","src/human_punching/wbc/provenance.py")


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def stopping_rules():
    return dict(validation_updates=[0,8,16],no_improvement_checks=2,
        score=["complete_cycle_fraction","correct_hit_timing_fraction","negative_outward_active_glove_rms_m"],
        initial_validation_hard_failure="stop_before_learning",
        validation_fall_nonfinite_root_assistance_effort_cap="hard_stop",
        training_nonfinite_root_assistance_effort_cap="hard_stop",
        training_fall="original_done_reset_no_commanded_cycle_credit",
        reset_protocol="original_half_guard_half_sampled_reference_phase_rehearsal",
        final_test_access=False,conditional_replication_budget_unlocked=False)


def create_plan(catalog,release,predicate,route_receipt,video_receipt,family, *, wall_seconds=1080.):
    """Metadata-only factory; callers must supply an already reviewed selection."""
    if family not in BINS:raise ValueError("Unknown family")
    release=owned(release)
    route=read(owned(route_receipt));selection=route.get("families",{}).get(family,{})
    data=read(owned(catalog));rows={r["candidate_id"]:r for r in data["cells"]}
    ids=selection.get("training_candidate_ids",[])
    requested=[family+"_original"]+ids
    if any(i not in rows for i in requested):raise ValueError("Selected exact reference is missing; no nearest replacement")
    budget=PilotBudget(maximum_updates=16,maximum_wall_s=float(wall_seconds))
    plan=dict(schema=SCHEMA,version=VERSION,logical_project_root=str(ROOT.resolve()),family=family,
        catalog=binding(catalog),release=binding(release/"manifest.json"),predicate=binding(predicate),
        route_receipt=binding(route_receipt),video_receipt=binding(video_receipt),seed=21021,validation_seed=22021,
        training_candidate_ids=list(ids),validation_candidate_ids=requested,
        selected_row_sha256={i:fingerprint(rows[i]) for i in requested},
        release_inputs={name:binding(release/family/path) for name,path in RELEASE_FILES.items()},
        sources={name:binding(ROOT/name) for name in SOURCES},budget=asdict(budget),fresh_transition_cap=131072,
        application_wall_seconds=float(wall_seconds),maximum_session_allocation_seconds=1200,
        bank_ids=list(range(8)),validation_updates=[0,8,16],planned_trials_per_panel=8*len(requested),
        planned_validation_trials=24*len(requested),stopping_rules=stopping_rules(),
        reference_byte_cap=TOTAL_BYTES,resident_reference_byte_cap=RESIDENT_BYTES,
        ancestry=dict(parent="accepted_Task09_family_specialist",fresh_optimizer=True,fresh_simulation=True,
            independent_scratch_replication=False,prior_training_resumed=False),
        scope="D0 development timing pilot only; no held-out domain qualification or final route/domain freeze",
        actual_route_freeze=False,policy_promotion=False)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    family=plan.get("family")
    if (plan.get("schema")!=SCHEMA or plan.get("version")!=VERSION or family not in BINS
            or plan.get("seed")!=21021 or plan.get("validation_seed")!=22021
            or plan.get("bank_ids")!=list(range(8)) or plan.get("validation_updates")!=[0,8,16]
            or plan.get("fresh_transition_cap")!=131072 or plan.get("actual_route_freeze") is not False
            or plan.get("policy_promotion") is not False or plan.get("stopping_rules")!=stopping_rules()):
        raise ValueError("Fixed D0 pilot protocol changed")
    logical_path(plan,ROOT/"outputs/task10b")
    wall=plan.get("application_wall_seconds")
    if (isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or not 180<=wall<=1080
            or plan.get("maximum_session_allocation_seconds")!=1200):raise ValueError("Pilot child wall must be finite180–1080s within1200s allocation")
    budget=PilotBudget(**plan["budget"])
    if asdict(budget)!=asdict(PilotBudget(maximum_updates=16,maximum_wall_s=float(wall))) or budget.transition_cap!=131072:
        raise ValueError("Exact128x64,16updates,131072fresh-transition cap required")
    if plan.get("ancestry")!=dict(parent="accepted_Task09_family_specialist",fresh_optimizer=True,fresh_simulation=True,
            independent_scratch_replication=False,prior_training_resumed=False):raise ValueError("Fresh transferred ancestry required")
    for key,cap in (("reference_byte_cap",TOTAL_BYTES),("resident_reference_byte_cap",RESIDENT_BYTES)):
        if type(plan.get(key)) is not int or not 0<plan[key]<=cap:raise ValueError("Finite reference byte cap required")
    paths={name:bound(plan[name]) for name in ("catalog","release","predicate","route_receipt","video_receipt")}
    if paths["release"].parent!=owned(ROOT/"outputs/four-punch-milestone"):
        raise ValueError("Only accepted Task09 release ancestry is permitted")
    if set(plan.get("sources",{}))!=set(SOURCES):raise ValueError("Complete original trainer/clock source closure required")
    for name,record in plan["sources"].items():
        if bound(record)!=owned(ROOT/name):raise ValueError("Source path changed")
    manifest=read(paths["release"])
    if set(plan.get("release_inputs",{}))!=set(RELEASE_FILES):raise ValueError("Complete Task09 family inputs required")
    for name,relative in RELEASE_FILES.items():
        item=plan["release_inputs"][name];path=bound(item)
        if (path!=owned(paths["release"].parent/family/relative)
                or manifest.get("artifacts",{}).get(family+"/"+relative,{}).get("sha256")!=item["sha256"]):
            raise ValueError("Accepted family input ancestry changed")
    predicate=read(paths["predicate"])
    if predicate.get("thresholds",{}).get("version")!="task10b_functional_guard_v1":raise ValueError("Reviewed guard predicate required")
    route,video,catalog=map(read,(paths["route_receipt"],paths["video_receipt"],paths["catalog"]))
    if (route.get("schema")!=ROUTE_SCHEMA or route.get("status")!="reviewed" or route.get("scope")!="d0_development"
            or route.get("catalog_sha256")!=plan["catalog"]["sha256"]
            or route.get("video_review_sha256")!=plan["video_receipt"]["sha256"]
            or route.get("final_test_used") is not False or route.get("final_recipe_domain_frozen") is not False):
        raise ValueError("Explicit reviewed D0 route selection bound to this catalog/video review required")
    if (video.get("schema")!=VIDEO_SCHEMA or video.get("status")!="reviewed" or video.get("predeclared_selection") is not True
            or family not in video.get("families",[]) or video.get("all_flagged_failures_reviewed") is not True
            or video.get("movement_exploit_detected") is not False):
        raise ValueError("Reviewed predeclared videos and flagged failures without unresolved exploit required")
    selection=route.get("families",{}).get(family,{})
    selected=selection.get("training_candidate_ids")
    if (not isinstance(selected,list) or len(selected) not in (2,3) or len(set(selected))!=len(selected)
            or selected!=plan.get("training_candidate_ids") or not selection.get("recipe_version")
            or not selection.get("selection_reason") or selection.get("authorize_transferred_pilot") is not True):
        raise ValueError("Exactly two or three explicitly selected timing rows required")
    if catalog.get("schema")!=CATALOG or catalog.get("complete") is not True or catalog.get("predicate_sha256")!=plan["predicate"]["sha256"]:
        raise ValueError("Complete bound reference catalog/predicate required")
    rows=catalog.get("cells",[])
    if len({r["candidate_id"] for r in rows})!=len(rows):raise ValueError("Duplicate catalog identities")
    by_id={r["candidate_id"]:r for r in rows};ids=[family+"_original"]+selected
    if (plan.get("validation_candidate_ids")!=ids or set(plan.get("selected_row_sha256",{}))!=set(ids)
            or plan.get("planned_trials_per_panel")!=8*len(ids) or plan.get("planned_validation_trials")!=24*len(ids)):
        raise ValueError("Full original-plus-selected validation ledger changed")
    chosen=[];ttcs=[];nonuniform=[];payloads={}
    for index,identifier in enumerate(ids):
        row=by_id.get(identifier)
        if row is None or fingerprint(row)!=plan["selected_row_sha256"][identifier]:raise ValueError("Selected scientific row changed")
        if (row.get("family")!=family or row.get("kinematic_admission") is not True or row.get("rejection_reasons")!=[]
                or row.get("start_mode","family_guard")!="family_guard" or row.get("split","d0_development")!="d0_development"):
            raise ValueError("Only admitted isolated family-guard D0 references may be selected")
        if (row.get("source_reference_sha256")!=plan["release_inputs"]["reference"]["sha256"]
                or row.get("source_model_sha256")!=plan["release_inputs"]["model"]["sha256"]
                or owned(row["source_model_path"])!=bound(plan["release_inputs"]["model"])):
            raise ValueError("Selected reference source/model ancestry differs")
        if index==0:
            if row.get("kind")!="original_control" or row.get("requested_ttc_s") is not None:
                raise ValueError("Historical original must remain validation-only")
        else:
            if row.get("kind") not in ("uniform","nonuniform_contact_anchored") or row.get("timing_role")=="prespecified_untrained_interior":
                raise ValueError("No interior or target/shared reference training")
            ttc=row.get("requested_ttc_s")
            if isinstance(ttc,bool) or ttc not in BINS[family]:raise ValueError("Only exact initially declared trained timing bins permitted")
            ttcs.append(ttc)
            if row["kind"]=="nonuniform_contact_anchored":
                if row.get("training_eligible") is not False:raise ValueError("Do not rewrite the nonuniform comparison training marker")
                nonuniform.append(identifier)
            elif row.get("training_eligible") is False:raise ValueError("Explicitly untrained uniform reference excluded")
        launch,outward=row.get("launch_tick"),row.get("outward_end_tick")
        duration=row.get("duration_s")
        if (type(launch) is not int or launch<0 or launch%4 or type(outward) is not int or outward<=launch
                or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=outward*.005
                or row.get("episode_duration_s")!=duration+1. or row.get("phase_denominator_s")!=duration+1.):
            raise ValueError("Explicit actor launch, complete source horizon and full-duration+1s phase required")
        for prefix in ("reference","legacy_runtime"):
            path=bound(dict(path=row[prefix+"_path"],sha256=row[prefix+"_sha256"]))
            if path not in payloads:payloads[path]=_preflight(path,row[prefix+"_sha256"],plan["resident_reference_byte_cap"])
        with np.load(owned(row["reference_path"]),allow_pickle=False) as archive:
            grid=np.asarray(archive["t"] if "t" in archive else archive["time_s"])
            if (grid.ndim!=1 or len(grid)<3 or grid[0]!=0 or not np.isfinite(grid).all()
                    or np.any(np.diff(grid)<=0) or float(grid[-1])!=duration
                    or archive["qpos"].shape!=(len(grid),55) or archive["qvel"].shape!=(len(grid),54)):
                raise ValueError("Reference array dimensions/full-duration clock disagree with row")
            if index and (not np.allclose(np.diff(grid),.005,rtol=0,atol=1e-12)
                    or float(archive["requested_ttc_s"])!=row["requested_ttc_s"]):
                raise ValueError("Candidate exact5ms/requested TTC archive clock changed")
        if index==0 and owned(row["reference_path"])!=bound(plan["release_inputs"]["reference"]):
            raise ValueError("Original source reference was substituted")
        chosen.append(deepcopy(row))
    if len(set(ttcs))!=len(ttcs):raise ValueError("Duplicate requested timings cannot stand in for distinct bins")
    if selection.get("explicit_nonuniform_training_ids",[])!=nonuniform:
        raise ValueError("Nonuniform comparison needs exact new per-row training authorization; no marker mutation")
    payload=sum(p.uncompressed_array_bytes for p in payloads.values())
    # FK-derived body/glove features can expand loaded resident data; reserve 4x
    # raw input payload before native construction and check actual tensors later.
    if payload>plan["reference_byte_cap"] or 4*payload>plan["resident_reference_byte_cap"]:
        raise ValueError("Selected reference payload/resident workspace bound exceeded")
    return paths,chosen,budget


def require_allocation():
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Timing PPO requires an authorized counted Slurm CUDA allocation")


def launch_receipt(plan_path, plan, *, runtime, clock, slurm_job_id, progress):
    """Common saved-trace metadata under the honest selected-pilot schema.

    The independent clock auditor already discovers preserved training
    checkpoints by each panel's actor digest. These fields let it and the D0
    renderer locate the same immutable catalog, accepted body and family.
    """
    if not slurm_job_id:raise ValueError("Counted pilot job identity required")
    descriptor=binding(plan_path)
    if read(bound(descriptor))!=plan:raise ValueError("Launch plan differs from committed input")
    return dict(schema=VERSION,mode="timing_pilot",family=plan["family"],seed=plan["seed"],
        plan=descriptor,plan_sha256=descriptor["sha256"],runtime=runtime,
        slurm_job_id=str(slurm_job_id),catalog=str(bound(plan["catalog"])),
        catalog_sha256=plan["catalog"]["sha256"],predicate=str(bound(plan["predicate"])),
        predicate_sha256=plan["predicate"]["sha256"],release_inputs=deepcopy(plan["release_inputs"]),
        sources={str(bound(item)):item["sha256"] for item in plan["sources"].values()},
        parent_checkpoint=deepcopy(plan["release_inputs"]["checkpoint"]),
        original_training_source=deepcopy(plan["sources"]["src/human_punching/phase2_learning.py"]),
        clock_postprocessing=clock,postprocessing_clock_adapter=clock,
        clock_scope_extension="This separately reviewed wrapper reuses only v2.run_cell_panel for learned D0 validation; v2 copied trainer/CLI unused",
        planned_cells=list(plan["validation_candidate_ids"]),training_candidate_ids=list(plan["training_candidate_ids"]),
        diagnostic_bank_ids=list(plan["bank_ids"]),planned_trials_per_panel=plan["planned_trials_per_panel"],
        validation_updates=list(plan["validation_updates"]),budget=deepcopy(plan["budget"]),
        fresh_transition_cap=plan["fresh_transition_cap"],fresh_optimizer=True,fresh_simulation=True,
        planned_validation=deepcopy(progress),no_test_split_access=True,no_g1_actor=True,
        no_policy_promotion=True,phase2_complete=False)


def require_final_checkpoint_parity(summary):
    """Reject unusable final checkpoints without changing original PPO math."""
    parity=summary.get("reload_parity") or {}
    if (parity.get("parameters_equal") is not True or parity.get("actions_equal") is not True
            or parity.get("maximum_action_error")!=0):
        raise RuntimeError("Final checkpoint restored parameter/action parity failed or missing")


def execute(plan_path,output):
    """Invoke original training; this entrypoint neither submits nor selects."""
    started=time.monotonic();require_allocation()
    plan_path=owned(plan_path);plan=read(plan_path);paths,rows,budget=validate_plan(plan)
    output=owned(output,exists=False)
    if not output.is_relative_to(ROOT.resolve()/"outputs/task10b"):raise ValueError("Fresh outputs/task10b pilot directory required")
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/"plan.json",plan)
    progress={str(u):dict(disposition="unattempted",planned_trials=plan["planned_trials_per_panel"]) for u in (0,8,16)}
    _write_json(output/"validation_declared.json",dict(updates=progress,
        trial_ids=[f"{r['candidate_id']}_bank_{i:02d}" for r in rows for i in range(8)]))
    summary=None;error=None;evaluator=env=None;training_started=False
    try:
        import torch
        import warp as wp
        from .wbc.provenance import runtime_record
        from .wbc.telemetry.cycle_qualification import native_stream_scope
        from .scalar_control import export_runtime_model
        runtime=runtime_record()
        if not runtime["declared_pins_match"]:raise RuntimeError("Preserved runtime pins changed")
        clock=evaluator_receipt()
        args,accepted=family_arguments(paths["release"].parent,plan["family"])
        torch.set_num_threads(1);torch.manual_seed(plan["seed"]);np.random.seed(plan["seed"])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;wp.init()
        _write_json(output/"launch.json",launch_receipt(plan_path,plan,runtime=runtime,clock=clock,
            slurm_job_id=os.environ["SLURM_JOB_ID"],progress=progress))
        with native_stream_scope(),torch.no_grad():
            evaluator=NativeSpecialistEnv(args,rows,num_envs=8)
            policy=fresh_policy(accepted,evaluator.device)
            export_runtime_model(args.model,output/"runtime.xml",evaluator.model)
            with np.load(accepted["bank"],allow_pickle=False) as archive:
                bank_q,bank_v=archive["qpos"][:8].copy(),archive["qvel"][:8].copy()
            if bank_q.shape!=(8,55) or bank_v.shape!=(8,54):raise ValueError("Exact eight accepted bank states required")
        def validate(policy,update):
            if update not in (0,8,16) or progress[str(update)]["disposition"]!="unattempted":raise ValueError("Unexpected validation update")
            if time.monotonic()-started>=plan["application_wall_seconds"]-budget.shutdown_reserve_s:
                raise RuntimeError("Pilot wall reservation exhausted before validation")
            progress[str(update)]["disposition"]="execution_extent_unknown"
            _write_json(output/f"validation_{update:05d}_started.json",dict(update=update,progress=deepcopy(progress)))
            with native_stream_scope():
                panel=run_cell_panel(evaluator,policy,rows,output/f"validation_{update:05d}",
                    initial_qpos=bank_q,initial_qvel=bank_v,seed=plan["validation_seed"],predicate=read(paths["predicate"]))
            progress[str(update)]=dict(disposition="completed" if panel["status"]=="executed" else "failed",
                planned_trials=panel["planned_trials"],executed_trials=panel["executed_trials"],
                numeric_complete_cycles=panel["numeric_complete_cycles"])
            _write_json(output/f"validation_{update:05d}_disposition.json",deepcopy(progress[str(update)]))
            return panel
        with native_stream_scope():
            env=NativeSpecialistEnv(args,rows[1:],num_envs=128)
            if env.library.bytes+evaluator.library.bytes>plan["resident_reference_byte_cap"]:
                raise ValueError("Actual resident reference tensors exceeded declared cap")
            remaining=plan["application_wall_seconds"]-(time.monotonic()-started)
            if remaining<=budget.shutdown_reserve_s:raise RuntimeError("Native setup exhausted pilot wall reserve")
            training_started=True
            summary=train_pilot(env,policy,output/"training",paths=accepted,seed=plan["seed"],
                validate=validate,budget=replace(budget,maximum_wall_s=remaining))
            if summary["status"]=="bounded_stop":require_final_checkpoint_parity(summary)
    except Exception as caught:
        error=dict(type=type(caught).__name__,message=str(caught))
    result=dict(schema=VERSION,status="failed" if error else summary["status"],exception=error,
        training_summary=summary,validation_progress=progress,wall_seconds=time.monotonic()-started,
        plan_sha256=sha(plan_path),fresh_transition_cap=131072,
        fresh_transitions=summary["fresh_transitions"] if summary else (None if training_started else 0),
        fresh_transition_extent="recorded" if summary else ("execution_extent_unknown" if training_started else "known_zero_training_not_started"),
        phase2_complete=False,formal_acceptance=False,final_test_executed=False,policy_promoted=False,
        training_engine="unchanged phase2_learning.train_pilot",validation_engine="clock_v2.run_cell_panel postprocessing")
    _write_json(output/"execution.json",result)
    return result
