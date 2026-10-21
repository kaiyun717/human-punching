"""One prespecified RH1.10s interior between selected uniform/nonuniform clocks.

Only matching original-source landmark times are blended. No pose-array blend,
actor execution, policy-lag compensation or new spatial path is introduced.
Construction needs a counted CUDA allocation and does not establish competence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import phase2_nonuniform as prior
from . import phase2_reference as reference
from .phase2_interiors import (ROOT, ANCESTRY, BINS, CATALOG, CELL_BYTES, CELL_PRODUCTS,
    TOTAL_BYTES, RESIDENT_BYTES, owned, binding, bound, sha, read, logical_path,
    _write_json, _native_context, require_allocation)
from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption

VERSION = "task10b_phase2_right_hook_mixed_scalar_interior_v1"
SCHEMA = "task10b_phase2_right_hook_mixed_interior_plan_v1"
SOURCES = ("src/human_punching/phase2_right_hook_interior_v1.py", *prior.SOURCES)
CHILD_SECONDS = 90
ENTRY = dict(family="right_hook", requested_ttc_s=1.1,
    candidate_id="right_hook_mixed_scalar_interior_1p100_v1", kind="mixed_scalar_clock_interior",
    split="d0_development", timing_role="prespecified_untrained_interior", training_eligible=False)


def blended_knots(uniform, nonuniform):
    """Map the same source landmarks through each parent, then average times."""
    if (uniform.get("candidate_id") != "right_hook_uniform_1p000"
            or nonuniform.get("candidate_id") != "right_hook_nonuniform_postcontact_1p200"
            or uniform.get("kind") != "uniform" or nonuniform.get("kind") != "nonuniform_contact_anchored"
            or uniform.get("requested_ttc_s") != 1.0 or nonuniform.get("requested_ttc_s") != 1.2
            or uniform.get("source_reference_sha256") != nonuniform.get("source_reference_sha256")
            or not isinstance(uniform.get("source_reference_sha256"), str)
            or uniform.get("family") != "right_hook" or nonuniform.get("family") != "right_hook"):
        raise ValueError("Exact selected RH parents with identical accepted spatial ancestry required")
    old = np.asarray(nonuniform["scalar_clock"]["source_knots_s"], float)
    right = np.asarray(nonuniform["scalar_clock"]["final_knots_s"], float)
    scale = uniform.get("duration_scale")
    if (old.shape != (6,) or right.shape != (6,) or not np.isfinite(old).all()
            or not np.isfinite(right).all() or old[0] != 0 or right[0] != 0
            or np.any(np.diff(old) <= 0) or np.any(np.diff(right) <= 0)
            or isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Finite strictly ordered matching source landmarks and exact uniform scale required")
    left = old * scale
    return old, (left + right) * .5, left, right


def clock_contract(source_knots, initial_final_knots, contact_offset_s):
    old = np.asarray(source_knots, float)
    initial = np.asarray(initial_final_knots, float)
    if (old.shape != (6,) or initial.shape != (6,) or not np.isfinite(old).all()
            or not np.isfinite(initial).all() or old[0] != 0 or initial[0] != 0
            or np.any(np.diff(old) <= 0) or np.any(np.diff(initial) <= 0)
            or isinstance(contact_offset_s, (bool, np.bool_)) or not np.isfinite(contact_offset_s)
            or abs(contact_offset_s) > .040 + 1e-12):
        raise ValueError("Exact six landmarks and bounded +/-40ms contact correction required")
    new = initial.copy(); new[2:] += contact_offset_s
    if np.any(np.diff(new) <= 0):
        raise ValueError("Mixed clock lost strict landmark order")
    curve = PchipInterpolator(new, old, extrapolate=False)
    rates=[]; velocity_jumps=[]; acceleration_jumps=[]
    for i,h in enumerate(np.diff(new)):
        a,b,c,_=curve.c[:,i]; x=[0.,float(h)]
        if a and 0 < -b/(3*a) < h:x.append(float(-b/(3*a)))
        rates.extend((3*a*np.asarray(x)**2+2*b*np.asarray(x)+c).tolist())
        if i:
            previous=curve.c[:,i-1]; width=new[i]-new[i-1]
            velocity_jumps.append(abs(c-(3*previous[0]*width**2+2*previous[1]*width+previous[2])))
            acceleration_jumps.append(abs(2*b-(6*previous[0]*width+2*previous[1])))
    if min(rates)<.5-1e-12 or max(rates)>1.5+1e-12:
        raise ValueError("Mixed clock source-rate [.5,1.5] envelope exceeded")
    if (max(velocity_jumps,default=0.)>1e-10
            or not np.allclose(curve(new),old,rtol=0,atol=1e-12)
            or not np.isfinite(acceleration_jumps).all()):
        raise ValueError("Mixed clock landmark or derivative continuity failed")
    return curve,dict(source_knots_s=old.tolist(),initial_blended_final_knots_s=initial.tolist(),
        final_knots_s=new.tolist(),blend_lambda=.5,contact_anchor_offset_s=float(contact_offset_s),
        preserved_blended_postcontact_intervals_s=np.diff(initial[2:]).tolist(),
        original_postcontact_intervals_claimed_preserved=False,
        local_source_rate_min=float(min(rates)),local_source_rate_max=float(max(rates)),
        maximum_clock_velocity_jump=float(max(velocity_jumps,default=0.)),
        maximum_clock_acceleration_jump=float(max(acceleration_jumps,default=0.)),
        continuity="C1 scalar clock with recorded finite acceleration jumps; no C2 spatial path claim")


def retime_mixed(model, source, old, initial, offset):
    curve,clock=clock_contract(old,initial,offset);new=np.asarray(clock["final_knots_s"])
    source_t=np.asarray(source.get("time_s",source.get("t")),float)
    source_q=np.asarray(source["qpos"],float)
    if (source_t.ndim!=1 or len(source_t)<3 or source_t[0]!=0 or np.any(np.diff(source_t)<=0)
            or source_q.shape!=(len(source_t),model.nq) or not np.isfinite(source_q).all()
            or not np.isfinite(source_t).all() or source_t[-1]!=old[-1]):
        raise ValueError("Exact accepted source path and terminal landmark required")
    t=reference.exact_grid(float(new[-1]))
    if len(t)>6001:raise ValueError("Mixed interior frame cap exceeded")
    query=curve(np.minimum(t,new[-1]))
    if (np.any(np.diff(query)<-1e-12) or query[0]!=0 or abs(query[-1]-source_t[-1])>1e-10):
        raise ValueError("Mixed source clock is nonmonotone or endpoint-inconsistent")
    q=reference.interpolate_poses(source_t,source_q,query)
    v=reference.tangent_velocities(model,t,q);a=np.gradient(v,t,axis=0,edge_order=2)
    if not all(np.isfinite(x).all() for x in (q,v,a)):raise ValueError("Nonfinite tangent derivatives")
    events=reference.reference_events(t,q,v,float(new[3]),1.)
    events.update(source_playback_end_s=float(old[3]),mapped_source_playback_end_s=float(new[3]),
        source_launch_annotation_is_not_actual_launch=True,nonuniform_events_version=VERSION,
        phase_semantics="Complete mixed scalar reference plus1s terminal hold; actual20ms outward launch rederived from sampled pose/velocity after positional/support readiness; original source-playback endpoint includes withdrawal; no phase/history reset")
    return dict(t=t,time_s=t,qpos=q,qvel=v,qacc=a,source_reference_time_s=query,
        duration_scale=float(new[-1]/source_t[-1]),unrounded_duration_s=float(new[-1]),
        terminal_padding_s=float(t[-1]-new[-1]),launch_time=float(new[1]),return_start=float(new[3]),
        guard_hold_start=float(new[4]),head_center=np.asarray(source["head_center"],float).copy(),
        active_hand_index=int(source["active_hand_index"]),
        attack_window_reference_s=np.array([events["launch_time_s"],events["outward_completion_time_s"]]),
        source_clock_knots_s=np.asarray(old),final_clock_knots_s=new,events=events),clock


def solve_mixed(model, source, requested_ttc, uniform, nonuniform, *, checkpoint=lambda stage:None):
    if requested_ttc!=1.1:raise ValueError("Only prespecified untrained RH1.10s interior is supported")
    old,initial,left,right=blended_knots(uniform,nonuniform)
    checkpoint("mixed_source_landmarks")
    actual,source_contact=prior.source_landmarks(model,source,requested_ttc)
    if not np.allclose(actual,old,rtol=0,atol=1e-12):raise ValueError("Recomputed accepted source landmarks differ")
    offset=0.;best=None;trials=[]
    for trial in range(18):
        checkpoint("mixed_clock_trial")
        try:candidate,clock=retime_mixed(model,source,old,initial,offset)
        except ValueError as error:
            trials.append(dict(trial=trial,contact_anchor_offset_s=offset,rejection=str(error)));break
        candidate["glove_pos"]=reference.glove_positions(model,candidate["qpos"]);events=candidate["events"]
        contact=reference.contact_diagnostics(candidate["t"],candidate["glove_pos"],candidate["head_center"],
            candidate["active_hand_index"],events["launch_time_s"],events["outward_completion_time_s"],requested_ttc)
        error=contact["timing_error_s"]
        trials.append(dict(trial=trial,contact_anchor_offset_s=offset,clock=clock,events=events,**contact))
        if error is None:break
        if best is None or abs(error)<abs(best[1]["timing_error_s"]):best=candidate,contact,clock
        if abs(error)<=.001:break
        proposal=offset-error
        if trial>=3:proposal=.5*(proposal+offset)
        proposal=float(np.clip(proposal,-.040,.040))
        if abs(proposal-offset)<1e-8:break
        offset=proposal
    search=dict(method=VERSION,maximum_trials=18,executed_trials=len(trials),trials=trials,
        source_landmarks_s=old.tolist(),uniform_mapped_landmarks_s=left.tolist(),
        nonuniform_mapped_landmarks_s=right.tolist(),initial_blended_landmarks_s=initial.tolist(),
        contact_anchor_offset_bounds_s=[-.040,.040],source_first_contact=source_contact,
        reference_contact_compensation_for_policy_lag=False,qualified_physical_execution=False)
    if best is None:
        error=ValueError("Bounded mixed clock produced no candidate");error.search_receipt=search;raise error
    candidate,contact,clock=best;candidate["requested_ttc_s"]=1.1
    search.update(selected_contact=contact,selected_clock=clock,solver_residual_within_1ms=abs(contact["timing_error_s"])<=.001)
    return candidate,search


def create_plan(selection_path, proposal_path, original_nonuniform_plan_path):
    selection=read(owned(selection_path));old=read(owned(original_nonuniform_plan_path))
    plan=dict(schema=SCHEMA,generator_version=VERSION,cells=[deepcopy(ENTRY)],
        logical_project_root=str(ROOT.resolve()),selection=binding(selection_path),proposal=binding(proposal_path),
        original_nonuniform_plan=binding(original_nonuniform_plan_path),
        parent_catalog=selection["families"]["right_hook"]["source_catalog"],domain=selection["domain"],
        **{k:deepcopy(old[k]) for k in ("predicate","release","bundle","source_ancestry")},
        generator_sources={n:binding(ROOT/n) for n in SOURCES},application_wall_seconds=CHILD_SECONDS,
        maximum_cell_wall_seconds=60.,serialization_reserve_seconds=20.,maximum_search_trials=18,
        maximum_frames=6001,screen_chunk_frames=16,physics_dt_s=.005,actor_dt_s=.020,
        reference_byte_cap=TOTAL_BYTES,resident_byte_cap=RESIDENT_BYTES,cell_reference_byte_cap=CELL_BYTES,
        cell_product_byte_cap=CELL_PRODUCTS,contact_anchor_radius_s=.040,blend_lambda=.5,
        local_clock_rate_bounds=[.5,1.5],training=False,final_test=False,physical_qualification=False)
    validate_plan(plan);return plan


def validate_plan(plan):
    fixed=dict(schema=SCHEMA,generator_version=VERSION,cells=[ENTRY],application_wall_seconds=90,
        maximum_cell_wall_seconds=60.,serialization_reserve_seconds=20.,maximum_search_trials=18,
        maximum_frames=6001,screen_chunk_frames=16,physics_dt_s=.005,actor_dt_s=.020,
        reference_byte_cap=TOTAL_BYTES,resident_byte_cap=RESIDENT_BYTES,cell_reference_byte_cap=CELL_BYTES,
        cell_product_byte_cap=CELL_PRODUCTS,contact_anchor_radius_s=.040,blend_lambda=.5,
        local_clock_rate_bounds=[.5,1.5],training=False,final_test=False,physical_qualification=False)
    if any(plan.get(k)!=v for k,v in fixed.items()):raise ValueError("Exact one-cell RH1.10 interior contract required")
    logical_path(plan,ROOT/"outputs/task10b")
    paths={k:bound(plan[k]) for k in ("selection","proposal","original_nonuniform_plan","parent_catalog","domain","predicate","release","bundle")}
    selection=read(paths["selection"]);record=selection["families"]["right_hook"]
    if (selection.get("schema")!="task10b_phase2_all_family_target_development_selection_v1"
            or record.get("trained_timings_s")!=[1.,1.2] or record.get("prespecified_untrained_interior_s")!=1.1
            or record.get("source_catalog")!=plan["parent_catalog"] or selection.get("domain")!=plan["domain"]):
        raise ValueError("Exact independently predeclared RH1.10 interior and selected parents required")
    proposal=read(paths["proposal"])
    if (proposal.get("schema")!="task10b_phase2_right_hook_grouped_cache_proposal_v1"
            or proposal.get("untrained_interior_s")!=1.1 or proposal.get("interior_used_for_training") is not False
            or proposal.get("bindings",{}).get("selection")!=plan["selection"]):
        raise ValueError("Reviewed mixed-clock proposal binding required")
    original=read(paths["original_nonuniform_plan"])
    for k in ("predicate","release","bundle","source_ancestry"):
        if plan[k]!=original[k]:raise ValueError("Original accepted ancestry changed")
    if set(plan.get("generator_sources",{}))!=set(SOURCES):raise ValueError("Exact generator source closure required")
    for name,item in plan["generator_sources"].items():
        if bound(item)!=owned(ROOT/name):raise ValueError("Generator source path changed")
    if set(plan["source_ancestry"])!=set(BINS):raise ValueError("All accepted source-control ancestry required")
    for family,records in plan["source_ancestry"].items():
        if set(records)!=set(ANCESTRY):raise ValueError("Incomplete source-control ancestry")
        for name,item in records.items():
            if bound(item)!=owned(paths["release"].parent/family/name):raise ValueError("Accepted source ancestry path changed")
    parent=read(paths["parent_catalog"])
    if (parent.get("schema")!=CATALOG or parent.get("complete") is not True or len(parent.get("cells",[]))!=22
            or parent.get("predicate_sha256")!=plan["predicate"]["sha256"]
            or read(paths["predicate"]).get("thresholds",{}).get("version")!="task10b_functional_guard_v1"):
        raise ValueError("Complete unchanged22-row parent and reviewed recovery predicate required")
    selected=[next(r for r in parent["cells"] if r["candidate_id"]==c) for c in record["training_candidate_ids"]]
    if selected!=record["selected_source_rows"]:raise ValueError("Selected parent row metadata changed")
    old,initial,_,_=blended_knots(*selected);clock_contract(old,initial,0.)
    if selected[0]["source_reference_sha256"]!=plan["source_ancestry"]["right_hook"]["reference.npz"]["sha256"]:
        raise ValueError("Mixed parents do not share the accepted RH source")
    payloads={}
    for row in parent["cells"]:
        for key in ("reference","legacy_runtime","source_model"):
            if row.get(key+"_path") is None:continue
            p=bound(dict(path=row[key+"_path"],sha256=row[key+"_sha256"]))
            if key!="source_model" and p not in payloads:payloads[p]=_preflight(p,row[key+"_sha256"],RESIDENT_BYTES)
    total=sum(v.uncompressed_array_bytes for v in payloads.values())
    if total+CELL_BYTES>min(TOTAL_BYTES,RESIDENT_BYTES):raise ValueError("Parent and one-candidate byte reservation exceeded")
    return paths,parent,total


def files(plan_path):
    """Exact semantic input closure; documentary source ancestry is opaque."""
    plan_path=owned(plan_path);plan=read(plan_path);paths,parent,_=validate_plan(plan);closure={}
    def add(path):
        p=owned(path);closure[str(p.relative_to(ROOT.resolve()))]=sha(p)
    add(plan_path)
    for p in paths.values():add(p)
    for records in plan["source_ancestry"].values():
        for item in records.values():add(bound(item))
    for item in plan["generator_sources"].values():add(bound(item))
    for row in parent["cells"]:
        for key in ("reference","legacy_runtime","source_model"):
            if row.get(key+"_path") is not None:add(bound(dict(path=row[key+"_path"],sha256=row[key+"_sha256"])))
    return dict(sorted(closure.items()))


def _engine(uniform,nonuniform):
    spec=importlib.util.spec_from_file_location("human_punching._rh_mixed_interior_engine_v1",Path(prior.__file__))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.VERSION=VERSION
    module.solve_contact_anchored=lambda model,source,ttc,checkpoint:solve_mixed(model,source,ttc,uniform,nonuniform,checkpoint=checkpoint)
    return module


def generate(plan_path,output,*,wall_seconds=CHILD_SECONDS):
    started=time.monotonic();require_allocation()
    plan_path=owned(plan_path);plan=read(plan_path);paths,parent,parent_bytes=validate_plan(plan)
    if isinstance(wall_seconds,bool) or wall_seconds!=CHILD_SECONDS:raise ValueError("Exact90second child required")
    output=owned(output,exists=False)
    if not output.is_relative_to(ROOT.resolve()/"outputs/task10b"):raise ValueError("Owned fresh Task10B output required")
    output.mkdir(parents=True,exist_ok=False);_write_json(output/"plan.json",plan)
    row=dict(**ENTRY,kinematic_admission=False,generation_disposition="unattempted",rejection_reasons=["mixed_interior_not_attempted"],reference_path=None,physical_qualification=False)
    deadline=ExecutionDeadline(CHILD_SECONDS,serialization_reserve_s=20.,session_started_monotonic=started)
    _write_json(output/"progress_000.json",dict(stage="declared",cell=row,complete=False))
    folder=None;exception=None;cell_started=time.monotonic()
    def checkpoint(stage):
        deadline.checkpoint(stage,progress={"candidate_id":ENTRY["candidate_id"]})
        if time.monotonic()-cell_started>=60.:
            raise BoundedExecutionInterruption(dict(reason="mixed_interior_cell_wall_limit",boundary=stage))
    with deadline.signal_handlers():
        try:
            checkpoint("before_mixed_interior");folder=output/ENTRY["candidate_id"];folder.mkdir(exist_ok=False)
            row.update(generation_disposition="generation_started",rejection_reasons=["generation_extent_unknown_until_terminal_receipt"])
            _write_json(output/"progress_001.json",dict(stage="started",cell=row,complete=False))
            inputs=_native_context(paths);checkpoint("mixed_context_ready")
            parents=read(paths["selection"])["families"]["right_hook"]["selected_source_rows"]
            row=_engine(*parents)._generate_cell(plan,ENTRY,paths,inputs,folder,checkpoint)
            checkpoint("mixed_interior_complete")
            row.update(mixed_interior_plan_sha256=sha(plan_path),parent_catalog_sha256=plan["parent_catalog"]["sha256"],
                selected_parent_candidate_ids=[p["candidate_id"] for p in parents],blend_lambda=.5,
                domain_sha256=plan["domain"]["sha256"],generation_source_sha256=sha(__file__))
            if sum(p.stat().st_size for p in folder.iterdir() if p.is_file())>CELL_PRODUCTS:raise ValueError("Cell product cap exceeded")
        except Exception as error:
            timeout=isinstance(error,BoundedExecutionInterruption)
            row.update(generation_disposition=("timeout" if folder else "unattempted_timeout") if timeout else "failed",
                kinematic_admission=False,rejection_reasons=["bounded_timeout" if timeout else "mixed_interior_generation_failure"],
                error=dict(type=type(error).__name__,message=str(error)),details=getattr(error,"receipt",None),
                retained_partial_reference_path=logical_path(plan,folder/"reference.npz") if folder and (folder/"reference.npz").exists() else None)
            _write_json((folder or output)/(ENTRY["candidate_id"]+"_failure.json"),row)
            if timeout or not isinstance(error,ValueError):exception=row["error"]
    terminal=row["generation_disposition"] in ("generated","failed","timeout")
    payload=folder/"reference_payload.json" if folder else None
    payload_bytes=read(payload)["uncompressed_array_bytes"] if payload and payload.exists() else 0
    unknown=bool(folder and (folder/"reference.npz").exists() and not payload_bytes)
    result=dict(schema=CATALOG,generator_version=VERSION,scope="Unchanged22parent rows plus one prespecified untrained RH1.10 mixed scalar clock",
        complete=row["generation_disposition"] in ("generated","failed"),planned_cells=23,executed_cells=22+int(terminal),
        inherited_cells=22,planned_new_cells=1,executed_new_cells=int(terminal),cells=deepcopy(parent["cells"])+[row],
        unattempted_cells=[deepcopy(row)] if row["generation_disposition"].startswith("unattempted") else [],exception=exception,
        parent_catalog=plan["parent_catalog"],selection=plan["selection"],proposal=plan["proposal"],domain=plan["domain"],
        predicate_path=logical_path(plan,paths["predicate"]),predicate_sha256=plan["predicate"]["sha256"],
        release=plan["release"],bundle=plan["bundle"],source_ancestry=plan["source_ancestry"],generator_sources=plan["generator_sources"],
        logical_project_root=plan["logical_project_root"],runtime_resolved_project_root=str(ROOT.resolve()),plan_sha256=sha(plan_path),
        reference_uncompressed_array_bytes=parent_bytes+payload_bytes,reference_bytes_measurement_is_lower_bound=unknown,
        reserved_uncompressed_new_reference_bytes=CELL_BYTES,wall_seconds=time.monotonic()-started,
        physics_executed=False,physical_qualification=False,learned_trajectory_generator=False,final_test_executed=False,
        fine_tuned=False,no_nearest_substitution=True,bounded_execution_notes="Exact90s child,60s cell including context;20s serialization reserve;18scalar trials and16-frame screen chunks")
    _write_json(output/"manifest.json",result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--wall-seconds",type=float,required=True)
    args=parser.parse_args(argv);result=generate(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps({k:result[k] for k in ("complete","planned_new_cells","executed_new_cells","exception")}))
    return 0 if result["complete"] and result["exception"] is None else 1


if __name__=="__main__":raise SystemExit(main())
