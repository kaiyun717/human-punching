"""One bounded contact-anchored clock comparison for six failed uniform cells.

The accepted spatial paths, targets and original feedforward remain unchanged.
PCHIP provides a C1 scalar clock, not a C2 spatial-trajectory claim. Generation
requires a counted CUDA allocation; this module submits nothing.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import phase2_reference as reference
from .phase2_interiors import (ROOT, ANCESTRY, BINS, CATALOG, CELL_BYTES, CELL_PRODUCTS,
    TOTAL_BYTES, RESIDENT_BYTES, owned, binding, bound, sha, read, logical_path,
    _write_json, _archive, _native_context, require_allocation)
from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption

VERSION = "task10b_phase2_contact_anchored_nonuniform_v1"
SCHEMA = "task10b_phase2_nonuniform_plan_v1"
FAILED_CELLS = (("cross", .75), ("cross", .90), ("left_hook", .95),
                ("left_hook", 1.25), ("right_hook", .80), ("right_hook", 1.20))
SOURCES = ("src/human_punching/phase2_nonuniform.py", "scripts/task10b_phase2_nonuniform.py",
    "src/human_punching/phase2_interiors.py", "src/human_punching/phase2_reference.py",
    "src/human_punching/wbc/references.py", "src/human_punching/wbc/effort_diagnostics.py",
    "src/human_punching/wbc/release_baseline.py", "src/human_punching/wbc/release_baseline_probe.py",
    "src/human_punching/wbc/runtime/bundle.py", "src/human_punching/phase2_cache.py",
    "src/human_punching/phase2_deadline.py")


def cell_plan():
    return [dict(family=f, requested_ttc_s=t,
        candidate_id=f+"_nonuniform_postcontact_"+f"{t:.3f}".replace(".", "p"),
        parent_uniform_candidate_id=f+"_uniform_"+f"{t:.3f}".replace(".", "p"),
        kind="nonuniform_contact_anchored", split="d0_development", bank_ids=list(range(8)),
        training_eligible=False, timing_role="bounded_frozen_policy_route_comparison") for f,t in FAILED_CELLS]


def clock_contract(source_knots, new_contact):
    """Check the polynomial on whole intervals, not merely the sampled grid."""
    old=np.asarray(source_knots, dtype=float)
    if old.shape!=(6,) or not np.isfinite(old).all() or old[0]!=0 or np.any(np.diff(old)<=0):
        raise ValueError("Strict source 0 < launch knot < first contact < playback end < guard < terminal required")
    if not np.isfinite(new_contact) or new_contact<=old[1]:
        raise ValueError("New contact must follow the fixed source launch knot")
    new=np.r_[old[:2], new_contact+(old[2:]-old[2])]
    curve=PchipInterpolator(new,old,extrapolate=False)
    rates=[]; velocity_jumps=[]; acceleration_jumps=[]
    for i,h in enumerate(np.diff(new)):
        a,b,c,_=curve.c[:,i]
        x=[0.,float(h)]
        if a and 0 < -b/(3*a) < h:x.append(float(-b/(3*a)))
        rates.extend((3*a*np.asarray(x)**2+2*b*np.asarray(x)+c).tolist())
        if i:
            previous=curve.c[:,i-1]; width=new[i]-new[i-1]
            velocity_jumps.append(abs(c-(3*previous[0]*width**2+2*previous[1]*width+previous[2])))
            acceleration_jumps.append(abs(2*b-(6*previous[0]*width+2*previous[1])))
    minimum,maximum=min(rates),max(rates)
    if minimum < .5-1e-12 or maximum > 1.5+1e-12:
        raise ValueError("Bounded clock local source-rate envelope [.5,1.5] exceeded")
    if max(velocity_jumps,default=0.)>1e-10:
        raise ValueError("Scalar clock velocity discontinuity")
    if not np.allclose(curve(new),old,rtol=0,atol=1e-12):raise ValueError("Clock landmark mismatch")
    return curve, dict(source_knots_s=old.tolist(), final_knots_s=new.tolist(),
        preserved_postcontact_intervals_s=np.diff(old[2:]).tolist(),
        local_source_rate_min=float(minimum),local_source_rate_max=float(maximum),
        maximum_clock_velocity_jump=float(max(velocity_jumps,default=0.)),
        maximum_clock_acceleration_jump=float(max(acceleration_jumps,default=0.)),
        continuity="C1 scalar clock; finite acceleration jumps recorded; no C2 spatial-path claim")


def source_landmarks(model,source,requested_ttc):
    candidate=reference.retime_uniform(model,source,1.)
    gloves=reference.glove_positions(model,candidate["qpos"])
    event=candidate["events"]
    contact=reference.contact_diagnostics(candidate["t"],gloves,candidate["head_center"],
        candidate["active_hand_index"],event["launch_time_s"],event["outward_completion_time_s"],requested_ttc)
    crossing=contact["first_active_contact_time_s"]
    if crossing is None or contact["inactive_hand_first_or_simultaneous"] or contact["any_contact_before_launch"]:
        raise ValueError("Accepted source path needs unambiguous first active contact after launch")
    knots=np.array([0.,float(source["launch_time"]),crossing,float(source["return_start"]),
        float(source["guard_hold_start"]),float(candidate["unrounded_duration_s"])])
    if np.any(np.diff(knots)<=0):raise ValueError("Source clock landmarks do not follow the declared order")
    return knots,contact


def retime_contact_anchored(model,source,source_knots,new_contact):
    curve,clock=clock_contract(source_knots,new_contact)
    new=np.asarray(clock["final_knots_s"]); old_t=np.asarray(source.get("time_s",source.get("t")),float)
    old_q=np.asarray(source["qpos"],float)
    if (old_t.ndim!=1 or len(old_t)<3 or old_t[0]!=0 or np.any(np.diff(old_t)<=0)
            or old_q.shape!=(len(old_t),model.nq) or not np.isfinite(old_q).all()
            or not np.isfinite(old_t).all() or old_t[-1]!=source_knots[-1]):
        raise ValueError("Malformed or mismatched source path")
    t=reference.exact_grid(float(new[-1]))
    if len(t)>6001:raise ValueError("Nonuniform frame cap exceeded")
    query=curve(np.minimum(t,new[-1]))
    if np.any(np.diff(query)<-1e-12) or query[0]!=0 or abs(query[-1]-old_t[-1])>1e-10:
        raise ValueError("Nonmonotone or endpoint-inconsistent source clock")
    q=reference.interpolate_poses(old_t,old_q,query)
    v=reference.tangent_velocities(model,t,q); a=np.gradient(v,t,axis=0,edge_order=2)
    if not all(np.isfinite(x).all() for x in (q,v,a)):raise ValueError("Nonfinite tangent derivatives")
    # Only mapped playback end is supplied; launch is recomputed from q AND v.
    events=reference.reference_events(t,q,v,float(new[3]),1.)
    events["phase_semantics"]=("Complete reset-relative contact-anchored PCHIP reference plus 1s terminal hold; "
        "actual outward actor boundary rederived from first pose/velocity motion; mapped original source-playback "
        "end includes withdrawal; no preparation, state, phase or history reset")
    events.update(source_playback_end_s=float(source_knots[3]),mapped_source_playback_end_s=float(new[3]),
        source_launch_annotation_is_not_actual_launch=True,nonuniform_events_version=VERSION)
    return dict(t=t,time_s=t,qpos=q,qvel=v,qacc=a,source_reference_time_s=query,
        duration_scale=float(new[-1]/old_t[-1]),unrounded_duration_s=float(new[-1]),terminal_padding_s=float(t[-1]-new[-1]),
        launch_time=float(new[1]),return_start=float(new[3]),guard_hold_start=float(new[4]),
        head_center=np.asarray(source["head_center"],float).copy(),active_hand_index=int(source["active_hand_index"]),
        attack_window_reference_s=np.array([events["launch_time_s"],events["outward_completion_time_s"]]),
        source_clock_knots_s=np.asarray(source_knots),final_clock_knots_s=new,events=events),clock


def solve_contact_anchored(model,source,requested_ttc, *, maximum_trials=18,checkpoint=lambda stage:None):
    if not np.isfinite(requested_ttc) or requested_ttc<.5 or maximum_trials!=18:
        raise ValueError("The fixed TTC floor and 18-trial ceiling are required")
    checkpoint("source_first_contact")
    knots,source_contact=source_landmarks(model,source,requested_ttc)
    center=float(knots[1]+requested_ttc); bounds=(center-.040,center+.040)
    candidate_contact=center; best=None; trials=[]
    for trial in range(maximum_trials):
        checkpoint("nonuniform_clock_trial")
        try:
            candidate,clock=retime_contact_anchored(model,source,knots,candidate_contact)
        except ValueError as error:
            trials.append(dict(trial=trial,contact_anchor_s=candidate_contact,rejection=str(error)))
            # A failed fixed clock envelope is an unsupported variant, not a search for a new recipe.
            break
        candidate["glove_pos"]=reference.glove_positions(model,candidate["qpos"])
        events=candidate["events"]
        contact=reference.contact_diagnostics(candidate["t"],candidate["glove_pos"],candidate["head_center"],
            candidate["active_hand_index"],events["launch_time_s"],events["outward_completion_time_s"],requested_ttc)
        error=contact["timing_error_s"]
        trials.append(dict(trial=trial,contact_anchor_s=candidate_contact,clock=clock,events=events,**contact))
        if error is None:break
        if best is None or abs(error)<abs(best[1]["timing_error_s"]):best=candidate,contact,clock
        if abs(error)<=.001:break
        proposal=candidate_contact-error
        if trial>=3:proposal=.5*(proposal+candidate_contact)
        proposal=float(np.clip(proposal,*bounds))
        if abs(proposal-candidate_contact)<1e-8:break
        candidate_contact=proposal
    search=dict(method=VERSION,maximum_trials=maximum_trials,executed_trials=len(trials),trials=trials,
        contact_anchor_bounds_s=list(bounds),source_landmarks_s=knots.tolist(),source_first_contact=source_contact,
        reference_contact_compensation_for_policy_lag=False,qualified_physical_execution=False)
    if best is None:
        error=ValueError("Bounded contact-anchored clock has no constructed candidate")
        error.search_receipt=search
        raise error
    candidate,contact,clock=best;candidate["requested_ttc_s"]=float(requested_ttc)
    search.update(selected_contact=contact,selected_clock=clock,solver_residual_within_1ms=abs(contact["timing_error_s"])<=.001)
    return candidate,search


def create_plan(parent_catalog,domain,predicate,release,bundle,assessment, *, wall_seconds=240.):
    """Metadata and NPZ-header preflight only; no model construction or execution."""
    release,bundle=owned(release),owned(bundle)
    plan=dict(schema=SCHEMA,generator_version=VERSION,logical_project_root=str(ROOT.resolve()),cells=cell_plan(),
        parent_catalog=binding(parent_catalog),domain=binding(domain),predicate=binding(predicate),
        release=binding(release/"manifest.json"),bundle=binding(bundle/"manifest.json"),assessment=binding(assessment),
        source_ancestry={f:{name:binding(release/f/name) for name in ANCESTRY} for f in BINS},
        generator_sources={name:binding(ROOT/name) for name in SOURCES},
        application_wall_seconds=float(wall_seconds),serialization_reserve_seconds=20.,maximum_cell_wall_seconds=90.,
        maximum_search_trials=18,contact_anchor_radius_s=.040,local_clock_rate_bounds=[.5,1.5],
        maximum_frames=6001,screen_chunk_frames=16,physics_dt_s=.005,actor_dt_s=.020,
        reference_byte_cap=TOTAL_BYTES,resident_byte_cap=RESIDENT_BYTES,cell_reference_byte_cap=CELL_BYTES,
        cell_product_byte_cap=CELL_PRODUCTS,
        comparison=dict(split="d0_development",families=["cross","left_hook","right_hook"],bank_ids=list(range(8)),
            seed=21021,planned_cycles=48,checkpoint="unchanged_accepted_Task09_specialist_per_family",
            evaluation_horizon="full_new_reference_duration_plus_1s_then_explicit_actor_boundary_padding",
            selection_order=["complete_cycle_fraction","timing_pass_fraction","negative_mean_outward_glove_rms"],
            generation_allocation_seconds=300,native_allocations_seconds=[480,480,480],aggregate_reserved_ceiling_seconds=1740,
            final_test=False,training=False),
        clock_contract="old[0,L,C,E,G,D] to new[0,L,Cnew,Cnew+E-C,Cnew+G-C,Cnew+D-C]; inverse PCHIP C1, C2 not claimed",
        execution_prerequisite="Root-reviewed timing controls and fresh counted reservations; this plan submits nothing")
    validate_plan(plan)
    return plan


def validate_plan(plan):
    fixed=dict(schema=SCHEMA,generator_version=VERSION,cells=cell_plan(),maximum_search_trials=18,
        contact_anchor_radius_s=.040,local_clock_rate_bounds=[.5,1.5],maximum_frames=6001,screen_chunk_frames=16,
        physics_dt_s=.005,actor_dt_s=.020,serialization_reserve_seconds=20.,maximum_cell_wall_seconds=90.)
    if any(plan.get(k)!=v for k,v in fixed.items()):raise ValueError("Declared six-cell nonuniform clock contract changed")
    logical_path(plan,ROOT/"outputs/task10b")
    wall=plan.get("application_wall_seconds")
    if isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or not 60<=wall<=240:
        raise ValueError("Generation child wall must be finite and <=240s inside its 300s allocation")
    comparison=plan.get("comparison",{})
    if (comparison.get("split")!="d0_development" or comparison.get("bank_ids")!=list(range(8))
            or comparison.get("seed")!=21021 or comparison.get("planned_cycles")!=48
            or comparison.get("families")!=["cross","left_hook","right_hook"]
            or comparison.get("generation_allocation_seconds")!=300 or comparison.get("native_allocations_seconds")!=[480]*3
            or comparison.get("aggregate_reserved_ceiling_seconds")!=1740
            or comparison.get("final_test") is not False or comparison.get("training") is not False
            or comparison.get("selection_order")!=["complete_cycle_fraction","timing_pass_fraction","negative_mean_outward_glove_rms"]):
        raise ValueError("Bounded frozen 48-cycle D0 comparison plan changed")
    for key,cap in (("reference_byte_cap",TOTAL_BYTES),("resident_byte_cap",RESIDENT_BYTES),
                    ("cell_reference_byte_cap",CELL_BYTES),("cell_product_byte_cap",CELL_PRODUCTS)):
        if type(plan.get(key)) is not int or not 0<plan[key]<=cap:raise ValueError("Invalid byte cap "+key)
    paths={name:bound(plan[name]) for name in ("parent_catalog","domain","predicate","release","bundle","assessment")}
    if set(plan.get("generator_sources",{}))!=set(SOURCES):raise ValueError("Complete generator source closure required")
    for name,record in plan["generator_sources"].items():
        if bound(record)!=owned(ROOT/name):raise ValueError("Generator source path changed")
    if set(plan.get("source_ancestry",{}))!=set(BINS):raise ValueError("All four source-control ancestries required")
    for family,records in plan["source_ancestry"].items():
        if set(records)!=set(ANCESTRY):raise ValueError("Incomplete source ancestry")
        for name,record in records.items():
            if bound(record)!=owned(paths["release"].parent/family/name):raise ValueError("Source ancestry path changed")
    assessment=read(paths["assessment"])
    if (assessment.get("schema")!="task10b_phase2_uniform_nonuniform_route_assessment_v1"
            or assessment.get("measured",{}).get("failed_cell_ids")!=[r["parent_uniform_candidate_id"] for r in cell_plan()]):
        raise ValueError("Measured route assessment differs from the six declared failed uniform cells")
    for prefix in ("cells","error_localization"):
        evidence=assessment["measured"]
        bound(dict(path=evidence[prefix+"_path"],sha256=evidence[prefix+"_sha256"]))
    parent,domain,predicate=map(read,(paths["parent_catalog"],paths["domain"],paths["predicate"]))
    if domain.get("schema")!="Task10BPhase2Domain/v1" or domain.get("candidate_timings_s")!=BINS:
        raise ValueError("Original declared timing domain changed")
    if (parent.get("schema")!=CATALOG or parent.get("complete") is not True or len(parent.get("cells",[]))!=16
            or parent.get("predicate_sha256")!=plan["predicate"]["sha256"]
            or predicate.get("thresholds",{}).get("version")!="task10b_functional_guard_v1"):
        raise ValueError("Complete sixteen-cell parent and reviewed predicate required")
    expected={(f+"_original",f,None,"original_control") for f in BINS}
    expected.update((f+"_uniform_"+f"{t:.3f}".replace(".","p"),f,t,"uniform") for f,ttcs in BINS.items() for t in ttcs)
    if {(r.get("candidate_id"),r.get("family"),r.get("requested_ttc_s"),r.get("kind")) for r in parent["cells"]}!=expected:
        raise ValueError("Parent cells substituted, duplicated or relabeled")
    payloads={}
    for row in parent["cells"]:
        ancestry=plan["source_ancestry"][row["family"]]
        if row.get("source_reference_sha256")!=ancestry["reference.npz"]["sha256"]:
            raise ValueError("Parent source reference ancestry changed")
        if bound(dict(path=row["source_model_path"],sha256=row["source_model_sha256"]))!=bound(ancestry["model/human.xml"]):
            raise ValueError("Parent source model changed")
        for prefix in ("reference","legacy_runtime"):
            if row.get(prefix+"_path") is None:
                if row.get("kinematic_admission") is True:raise ValueError("Admitted parent lacks reference")
                continue
            path=bound(dict(path=row[prefix+"_path"],sha256=row[prefix+"_sha256"]))
            if path not in payloads:payloads[path]=_preflight(path,row[prefix+"_sha256"],plan["resident_byte_cap"])
        if row["kind"]=="original_control" and bound(dict(path=row["reference_path"],sha256=row["reference_sha256"]))!=bound(ancestry["reference.npz"]):
            raise ValueError("Original source control was replaced")
    parent_bytes=sum(p.uncompressed_array_bytes for p in payloads.values())
    original_bytes=sum(payloads[bound(plan["source_ancestry"][f]["reference.npz"])].uncompressed_array_bytes for f in BINS)
    if parent_bytes+6*plan["cell_reference_byte_cap"]>plan["reference_byte_cap"]:
        raise ValueError("Full six-cell reservation exceeds total reference budget")
    if original_bytes+6*plan["cell_reference_byte_cap"]>plan["resident_byte_cap"]:
        raise ValueError("Source arrays and candidate workspace exceed resident reservation")
    return paths,parent,parent_bytes


def _generate_cell(plan,entry,paths,inputs,folder,checkpoint):
    import mujoco
    from .wbc.release_baseline import frozen_source
    family=entry["family"];origin=inputs[family]
    model_path=bound(plan["source_ancestry"][family]["model/human.xml"])
    source_path=bound(plan["source_ancestry"][family]["reference.npz"])
    checkpoint("nonuniform_model_and_source")
    model=mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq,model.nv,model.nu)!=(55,54,48):raise ValueError("Accepted SOMA dimensions required")
    with np.load(source_path,allow_pickle=False) as archive:source={k:archive[k].copy() for k in archive.files}
    try:
        candidate,search=solve_contact_anchored(model,source,entry["requested_ttc_s"],checkpoint=checkpoint)
    except ValueError as error:
        if hasattr(error,"search_receipt"):_write_json(folder/"search.json",error.search_receipt)
        raise
    _write_json(folder/"search.json",search)
    lower,upper=model.jnt_range[1:].T
    excursion=float(np.maximum(np.maximum(lower-candidate["qpos"][:,7:],candidate["qpos"][:,7:]-upper),0).max())
    if excursion>1e-9:raise ValueError("Fixed source path crosses an original joint-position limit")
    module,_=frozen_source(paths["release"].parent,family)
    candidate["torque"]=module.support_feedforward(model,candidate,np.asarray(origin["schema"]["dof_addresses"]))
    checkpoint("nonuniform_original_feedforward_completed")
    events=candidate.pop("events")
    candidate.update(punch_family=np.asarray(family),timing_kind=np.asarray("contact_anchored_nonuniform_clock"),
        source_reference_sha256=np.asarray(sha(source_path)),source_timing_modified=np.asarray(True),
        nonuniform_generation_version=np.asarray(VERSION))
    payload=_archive(folder/"reference.npz",candidate,plan["cell_reference_byte_cap"])
    _write_json(folder/"reference_payload.json",dict(uncompressed_array_bytes=payload,reference_sha256=sha(folder/"reference.npz")))
    _write_json(folder/"events.json",events)
    summaries=[]
    for number,begin in enumerate(range(0,len(candidate["t"]),plan["screen_chunk_frames"])):
        checkpoint("nonuniform_screen_chunk")
        chunk={key:candidate[key][begin:begin+plan["screen_chunk_frames"]] for key in ("t","qpos","qvel","qacc")}
        summary,arrays=reference.screen_diagnostics(model_path,chunk,np.asarray(origin["schema"]["torque_limits"])[:,1])
        if sum(p.stat().st_size for p in folder.iterdir() if p.is_file())+sum(np.asarray(a).nbytes for a in arrays.values())+32768>plan["cell_product_byte_cap"]:
            raise ValueError("Screening would exceed cell product cap")
        with (folder/f"screen_{number:04d}.npz").open("xb") as stream:np.savez_compressed(stream,**arrays)
        _write_json(folder/f"screen_{number:04d}.json",summary);summaries.append(summary)
    screen=deepcopy(summaries[0]);counts=Counter()
    for summary in summaries:counts.update(summary["support_status_counts"])
    for key in ("joint_limit_excursion_max_rad","selected_self_penetration_max_m","floor_penetration_max_m",
        "joint_velocity_max_rad_s","joint_acceleration_max_rad_s2","maximum_minimax_effort_ratio",
        "maximum_base_force_residual_n","maximum_base_moment_residual_nm"):
        values=[s[key] for s in summaries if s[key] is not None];screen[key]=max(values) if values else None
    for key in ("frames","within_effort_caps_frames","over_effort_caps_frames"):screen[key]=sum(s[key] for s in summaries)
    screen.update(support_status_counts=dict(counts),completed_chunks=len(summaries),diagnostic_arrays="screen_NNNN.npz")
    _write_json(folder/"screen.json",screen)
    contact=search["selected_contact"];reasons=[]
    if not contact["timing_valid"]:reasons.append("reference_first_contact_timing_outside_contract")
    if not search["solver_residual_within_1ms"]:reasons.append("bounded_reference_contact_solve_exceeded_1ms")
    if contact["inactive_hand_first_or_simultaneous"]:reasons.append("wrong_hand_first_or_simultaneous")
    if contact["any_contact_before_launch"]:reasons.append("contact_before_launch")
    path=folder/"reference.npz";digest=sha(path)
    return dict(**entry,**events,start_mode="family_guard",preparation_complete_tick=0,
        source_reference_sha256=sha(source_path),source_model_path=logical_path(plan,model_path),source_model_sha256=sha(model_path),
        head_center=np.asarray(candidate["head_center"]).tolist(),target_w=np.asarray(candidate["head_center"]).tolist(),
        active_hand_index=int(candidate["active_hand_index"]),reference_path=logical_path(plan,path),reference_sha256=digest,
        legacy_runtime_path=logical_path(plan,path),legacy_runtime_sha256=digest,reference_identity_kind="reference_content",
        duration_s=float(candidate["t"][-1]),episode_duration_s=float(candidate["t"][-1])+1.,phase_denominator_s=float(candidate["t"][-1])+1.,
        guard_time_s=float(candidate["guard_hold_start"]),duration_scale=float(candidate["duration_scale"]),
        generation_disposition="generated",kinematic_admission=not reasons,rejection_reasons=reasons,kinematic_contact=contact,
        scalar_clock=search["selected_clock"],diagnostic_screen_path=logical_path(plan,folder/"screen.json"),
        reference_uncompressed_array_bytes=payload,preserved_original_control=False,physical_qualification=False,fine_tuned=False,
        generator_version=VERSION,source_path_unchanged=True,source_existing_retiming_preserved_as_ancestry=True,
        reference_warning="Construction only; C1 scalar clock, not continuous C2 spatial trajectory or native feasibility")


def generate(plan_path,output, *, wall_seconds=None):
    started=time.monotonic();require_allocation()
    plan_path=owned(plan_path);plan=read(plan_path);paths,parent,parent_bytes=validate_plan(plan)
    if wall_seconds is not None and wall_seconds!=plan["application_wall_seconds"]:raise ValueError("Wall must match immutable plan")
    output=owned(output,exists=False)
    if not output.is_relative_to(ROOT.resolve()/"outputs/task10b"):raise ValueError("Fresh outputs/task10b output required")
    output.mkdir(parents=True,exist_ok=False);_write_json(output/"plan.json",plan)
    deadline=ExecutionDeadline(plan["application_wall_seconds"],serialization_reserve_s=20.,session_started_monotonic=started)
    rows=[dict(**entry,kinematic_admission=False,generation_disposition="unattempted",rejection_reasons=["nonuniform_not_attempted"],
        reference_path=None,physical_qualification=False,fine_tuned=False) for entry in plan["cells"]]
    sequence=0;exception=None;inputs=None
    def progress(stage):
        nonlocal sequence
        _write_json(output/f"progress_{sequence:03d}.json",dict(stage=stage,complete=False,cells=deepcopy(rows),
            interruption_semantics="A generation_started cell without a terminal receipt has unknown generation extent, never completed or unattempted"))
        sequence+=1
    progress("declared")
    with deadline.signal_handlers():
        for index,entry in enumerate(plan["cells"]):
            current=None;cell_started=time.monotonic()
            def checkpoint(stage):
                deadline.checkpoint(stage,progress={"candidate_id":entry["candidate_id"]})
                if time.monotonic()-cell_started>=plan["maximum_cell_wall_seconds"]:
                    raise BoundedExecutionInterruption(dict(reason="nonuniform_cell_wall_limit",boundary=stage))
            try:
                checkpoint("before_nonuniform_cell")
                current=output/entry["candidate_id"];current.mkdir(exist_ok=False)
                rows[index].update(generation_disposition="generation_started",rejection_reasons=["generation_extent_unknown_until_terminal_receipt"])
                _write_json(current/"attempt_started.json",dict(entry=entry,plan_sha256=sha(plan_path),elapsed_wall_s=time.monotonic()-started))
                progress("cell_started")
                if inputs is None:inputs=_native_context(paths)
                checkpoint("nonuniform_context_ready")
                row=_generate_cell(plan,entry,paths,inputs,current,checkpoint)
                checkpoint("nonuniform_cell_complete")
                row.update(nonuniform_plan_sha256=sha(plan_path),parent_catalog_sha256=plan["parent_catalog"]["sha256"],
                    domain_sha256=plan["domain"]["sha256"],generation_source_sha256=plan["generator_sources"][SOURCES[0]]["sha256"])
                if sum(p.stat().st_size for p in current.iterdir() if p.is_file())>plan["cell_product_byte_cap"]:
                    raise ValueError("Completed cell exceeds product cap")
                rows[index]=row
            except Exception as error:
                timeout=isinstance(error,BoundedExecutionInterruption)
                rows[index].update(generation_disposition=("timeout" if current is not None else "unattempted_timeout") if timeout else "failed",
                    kinematic_admission=False,rejection_reasons=["bounded_timeout" if timeout else "nonuniform_reference_generation_failure"],
                    error=dict(type=type(error).__name__,message=str(error)),details=getattr(error,"receipt",None),
                    retained_partial_reference_path=logical_path(plan,current/"reference.npz") if current is not None and (current/"reference.npz").exists() else None)
                _write_json((current if current is not None else output)/(entry["candidate_id"]+"_failure.json"),rows[index])
                if timeout or not isinstance(error,ValueError):exception=rows[index]["error"]
            progress("cell_disposition")
            if exception:break
    terminal=[r for r in rows if r["generation_disposition"] in ("generated","failed","timeout")]
    payload_bytes=0;unknown=[]
    for path in output.glob("*/reference.npz"):
        receipt=path.with_name("reference_payload.json")
        if receipt.exists() and read(receipt).get("reference_sha256")==sha(path):payload_bytes+=read(receipt)["uncompressed_array_bytes"]
        else:unknown.append(logical_path(plan,path))
    result=dict(schema=CATALOG,generator_version=VERSION,scope="Original16 plus six bounded nonuniform comparisons; no physical qualification",
        complete=all(r["generation_disposition"] in ("generated","failed") for r in rows),planned_cells=22,
        executed_cells=16+len(terminal),inherited_cells=16,planned_new_cells=6,executed_new_cells=len(terminal),
        cells=deepcopy(parent["cells"])+rows,unattempted_cells=[deepcopy(r) for r in rows if r["generation_disposition"].startswith("unattempted")],
        exception=exception,parent_catalog=plan["parent_catalog"],domain=plan["domain"],assessment=plan["assessment"],
        predicate_path=logical_path(plan,paths["predicate"]),predicate_sha256=plan["predicate"]["sha256"],
        release=plan["release"],bundle=plan["bundle"],source_ancestry=plan["source_ancestry"],generator_sources=plan["generator_sources"],
        logical_project_root=plan["logical_project_root"],runtime_resolved_project_root=str(ROOT.resolve()),plan_sha256=sha(plan_path),
        reference_uncompressed_array_bytes=parent_bytes+payload_bytes,reference_bytes_measurement_is_lower_bound=bool(unknown),
        unknown_partial_reference_payloads=unknown,reserved_uncompressed_new_reference_bytes=6*plan["cell_reference_byte_cap"],
        comparison=plan["comparison"],wall_seconds=time.monotonic()-started,physics_executed=False,physical_qualification=False,
        learned_trajectory_generator=False,final_test_executed=False,fine_tuned=False,no_nearest_substitution=True,
        bounded_execution_notes="<=240s child including serialization; per-clock-trial and <=16-frame-screen checkpoints; allocation watchdog bounds native-call stalls")
    _write_json(output/"manifest.json",result)
    return result
