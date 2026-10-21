"""Four predeclared, untrained timing interiors; immutable parent catalog.

This small factory reuses the existing uniform source-path solver. It does not
change the original generator, select another TTC, qualify a policy, or expose
test snapshot payloads. Full construction requires a counted CUDA allocation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "task10b_phase2_interior_reference_plan_v1"
CATALOG = "task10b_phase2_reference_catalog_v1"
VERSION = "task10b_phase2_predeclared_uniform_interiors_v1"
INTERIORS = {"jab": .675, "cross": .675, "left_hook": 1.025, "right_hook": .900}
BINS = {"jab": [.6,.75,.9], "cross": [.6,.75,.9], "left_hook": [.95,1.10,1.25], "right_hook": [.8,1.,1.2]}
TOTAL_BYTES = 2*1024**3
RESIDENT_BYTES = 256*1024**2
CELL_BYTES = 16*1024**2
CELL_PRODUCTS = 64*1024**2
ANCESTRY = ("reference.npz", "model/human.xml", "controller_schema.json", "body_metadata.json",
            "scalar_control_source.py", "jab_learning_source.py", "checkpoint.pt")
SOURCES = ("src/human_punching/phase2_interiors.py", "scripts/task10b_phase2_interiors.py",
    "src/human_punching/phase2_reference.py", "src/human_punching/wbc/references.py",
    "src/human_punching/wbc/effort_diagnostics.py", "src/human_punching/wbc/release_baseline.py",
    "src/human_punching/wbc/release_baseline_probe.py", "src/human_punching/wbc/runtime/bundle.py",
    "src/human_punching/phase2_deadline.py", "src/human_punching/phase2_cache.py")


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data:=stream.read(1024*1024):digest.update(data)
    return digest.hexdigest()


def owned(path, *, exists=True):
    raw=Path(path).expanduser()
    result=(raw if raw.is_absolute() else ROOT/raw).resolve(strict=exists)
    if not result.is_relative_to(ROOT.resolve()) or result==ROOT.resolve():
        raise ValueError("Interior reference paths must resolve inside human-punching")
    return result


def binding(path):
    path=owned(path)
    if not path.is_file():raise ValueError("Bind an existing input file")
    return dict(path=str(path),sha256=sha(path))


def bound(record):
    if not isinstance(record,dict) or set(record)!={"path","sha256"}:
        raise ValueError("Exact path/SHA256 input binding required")
    path=owned(record["path"])
    if not path.is_file() or sha(path)!=record["sha256"]:
        raise ValueError("Bound interior input changed")
    return path


def read(path):return json.loads(Path(path).read_text())


def logical_path(plan,path):
    """Serialize under the exact prepared root, only after owned alias proof.

    The compute host may resolve the prepared /home root to /orcd. This does not
    admit arbitrary foreign prefixes: the declared root itself must resolve to
    this module's current owned project root. Relative inputs are ROOT-relative.
    """
    logical=Path(plan["logical_project_root"])
    if not logical.is_absolute() or logical.resolve(strict=True)!=ROOT.resolve():
        raise ValueError("Prepared logical root does not resolve to the current owned project")
    return str(logical/owned(path,exists=False).relative_to(ROOT.resolve()))


def cell_plan():
    return [dict(family=family,requested_ttc_s=ttc,
        candidate_id=family+"_uniform_interior_"+f"{ttc:.3f}".replace(".","p"),
        timing_role="prespecified_untrained_interior",training_eligible=False)
        for family,ttc in INTERIORS.items()]


def create_plan(parent_catalog,domain,predicate,release,bundle, *, wall_seconds=300.,cell_byte_cap=CELL_BYTES):
    """Build metadata only; source arrays are header-preflighted, never generated."""
    release,bundle=owned(release),owned(bundle)
    plan=dict(schema=SCHEMA,generator_version=VERSION,cells=cell_plan(),logical_project_root=str(ROOT.resolve()),
        parent_catalog=binding(parent_catalog),domain=binding(domain),predicate=binding(predicate),
        release=binding(release/"manifest.json"),bundle=binding(bundle/"manifest.json"),
        source_ancestry={family:{name:binding(release/family/name) for name in ANCESTRY} for family in INTERIORS},
        generator_sources={name:binding(ROOT/name) for name in SOURCES},
        application_wall_seconds=float(wall_seconds),serialization_reserve_seconds=20.,
        maximum_cell_wall_seconds=120.,maximum_search_trials=18,maximum_frames=6001,screen_chunk_frames=16,
        reference_byte_cap=TOTAL_BYTES,resident_byte_cap=RESIDENT_BYTES,cell_reference_byte_cap=cell_byte_cap,
        cell_product_byte_cap=CELL_PRODUCTS,physics_dt_s=.005,actor_dt_s=.020,
        scope="Only the already declared timing interiors; no target geometry, split examples, training or qualification",
        execution_prerequisite="Root-reviewed measured uniform controls and a fresh counted allocation; this plan is not execution authority")
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (plan.get("schema")!=SCHEMA or plan.get("generator_version")!=VERSION or plan.get("cells")!=cell_plan()
            or plan.get("physics_dt_s")!=.005 or plan.get("actor_dt_s")!=.020
            or plan.get("maximum_search_trials")!=18 or plan.get("maximum_frames")!=6001
            or plan.get("screen_chunk_frames")!=16):
        raise ValueError("Exactly four predeclared interiors and unchanged clock/search contract required")
    logical_path(plan,ROOT/"outputs/task10b")
    for key,cap in (("reference_byte_cap",TOTAL_BYTES),("resident_byte_cap",RESIDENT_BYTES),
                    ("cell_reference_byte_cap",CELL_BYTES),("cell_product_byte_cap",CELL_PRODUCTS)):
        if type(plan.get(key)) is not int or not 0<plan[key]<=cap:raise ValueError("Invalid bounded byte cap: "+key)
    wall=plan.get("application_wall_seconds")
    if (isinstance(wall,bool) or not isinstance(wall,(float,int)) or not math.isfinite(wall) or not 60<=wall<=840
            or plan.get("serialization_reserve_seconds")!=20. or plan.get("maximum_cell_wall_seconds")!=120.
            or 6*plan["cell_reference_byte_cap"]>plan["resident_byte_cap"]):
        raise ValueError("Finite bounded wall/cell/resident reservations required")
    paths={name:bound(plan[name]) for name in ("parent_catalog","domain","predicate","release","bundle")}
    if set(plan.get("generator_sources",{}))!=set(SOURCES):raise ValueError("Complete generator source closure required")
    for name,record in plan["generator_sources"].items():
        if bound(record)!=owned(ROOT/name):raise ValueError("Generator source path differs")
    release=paths["release"].parent
    if set(plan.get("source_ancestry",{}))!=set(INTERIORS):raise ValueError("All four accepted source ancestries required")
    for family,records in plan["source_ancestry"].items():
        if set(records)!=set(ANCESTRY):raise ValueError("Incomplete original specialist ancestry")
        for name,record in records.items():
            if bound(record)!=owned(release/family/name):raise ValueError("Source ancestry path differs")
    domain,parent,predicate=map(read,(paths["domain"],paths["parent_catalog"],paths["predicate"]))
    if (domain.get("schema")!="Task10BPhase2Domain/v1" or domain.get("candidate_timings_s")!=BINS
            or any(domain.get("provisional_final_timings_s",{}).get(family)!=[BINS[family][0],ttc,BINS[family][1]] for family,ttc in INTERIORS.items())):
        raise ValueError("Interior TTC differs from the previously declared domain; no substitution")
    if (parent.get("schema")!=CATALOG or parent.get("complete") is not True or len(parent.get("cells",[]))!=16
            or parent.get("predicate_sha256")!=plan["predicate"]["sha256"]
            or predicate.get("thresholds",{}).get("version")!="task10b_functional_guard_v1"):
        raise ValueError("Complete original sixteen-cell parent and reviewed predicate required")
    expected={(family+"_original",family,None) for family in INTERIORS}
    expected.update((family+"_uniform_"+f"{ttc:.3f}".replace(".","p"),family,ttc) for family,ttcs in BINS.items() for ttc in ttcs)
    actual={(row.get("candidate_id"),row.get("family"),row.get("requested_ttc_s")) for row in parent["cells"]}
    if actual!=expected:raise ValueError("Parent cells were substituted, duplicated or relabeled")
    payloads={}
    for row in parent["cells"]:
        expected_kind="original_control" if row["requested_ttc_s"] is None else "uniform"
        if row.get("kind")!=expected_kind:raise ValueError("Parent original/uniform identity was relabeled")
        ancestry=plan["source_ancestry"][row["family"]]
        if row.get("source_reference_sha256")!=ancestry["reference.npz"]["sha256"]:
            raise ValueError("Parent source-control ancestry differs")
        if bound(dict(path=row["source_model_path"],sha256=row["source_model_sha256"]))!=bound(ancestry["model/human.xml"]):
            raise ValueError("Parent source model differs")
        for prefix in ("reference","legacy_runtime"):
            if row.get(prefix+"_path") is None:
                if row.get("kinematic_admission") is True:raise ValueError("Admitted parent lacks reference")
                continue
            path=bound(dict(path=row[prefix+"_path"],sha256=row[prefix+"_sha256"]))
            if path not in payloads:payloads[path]=_preflight(path,row[prefix+"_sha256"],plan["resident_byte_cap"])
        if row["kind"]=="original_control" and bound(dict(path=row["reference_path"],sha256=row["reference_sha256"]))!=bound(ancestry["reference.npz"]):
            raise ValueError("Original reference was replaced")
    parent_bytes=sum(p.uncompressed_array_bytes for p in payloads.values())
    original_bytes=sum(payloads[bound(plan["source_ancestry"][family]["reference.npz"])].uncompressed_array_bytes for family in INTERIORS)
    if original_bytes+6*plan["cell_reference_byte_cap"]>plan["resident_byte_cap"]:
        raise ValueError("Original source arrays and bounded candidate workspace exceed resident reservation")
    if parent_bytes+len(INTERIORS)*plan["cell_reference_byte_cap"]>plan["reference_byte_cap"]:
        raise ValueError("Full interior reservation exceeds total reference budget")
    return paths,parent,parent_bytes


def require_allocation():
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Interior generation requires an explicitly authorized, counted Slurm CUDA allocation")


def _write_json(path,data):
    path=owned(path,exists=False)
    with path.open("x") as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write("\n")


def _archive(path,arrays,cell_cap):
    payload={key:np.asarray(value) for key,value in arrays.items()}
    if any(a.dtype.kind not in "biufcSU" or (a.dtype.kind in "biufc" and not np.isfinite(a).all()) for a in payload.values()):
        raise ValueError("Reference payload must contain finite numeric or explicit string arrays")
    size=sum(a.nbytes for a in payload.values())
    if size>cell_cap:raise ValueError("Candidate exceeds its uncompressed cell reference byte cap")
    with owned(path,exists=False).open("xb") as stream:np.savez_compressed(stream,**payload)
    return size


def _native_context(paths):
    from .wbc.runtime.bundle import load_runtime_bundle
    from .wbc.release_baseline_probe import load_release_inputs
    bundle=load_runtime_bundle(paths["bundle"].parent)
    return load_release_inputs(bundle,paths["release"].parent,num_envs=16)


def _generate_cell(plan,entry,paths,inputs,folder,checkpoint):
    import mujoco
    from .phase2_reference import solve_uniform_timing,screen_diagnostics
    from .wbc.release_baseline import frozen_source
    family=entry["family"];origin=inputs[family]
    model_path=bound(plan["source_ancestry"][family]["model/human.xml"])
    source_path=bound(plan["source_ancestry"][family]["reference.npz"])
    checkpoint("interior_model_and_source")
    model=mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq,model.nv,model.nu)!=(55,54,48):raise ValueError("Accepted SOMA dimensions required")
    with np.load(source_path,allow_pickle=False) as archive:source={key:archive[key].copy() for key in archive.files}
    if len(source["qpos"])*3.0/.005*np.max(np.diff(source["time_s"]))>plan["maximum_frames"]+4:
        raise ValueError("Bounded scalar search could exceed its reference frame cap")
    checkpoint("interior_uniform_search")
    candidate,search=solve_uniform_timing(model,source,entry["requested_ttc_s"],maximum_trials=plan["maximum_search_trials"])
    checkpoint("interior_uniform_search_completed")
    if len(candidate["t"])>plan["maximum_frames"]:raise ValueError("Interior candidate exceeds frame bound")
    module,_=frozen_source(paths["release"].parent,family)
    candidate["torque"]=module.support_feedforward(model,candidate,np.asarray(origin["schema"]["dof_addresses"]))
    checkpoint("interior_original_feedforward_completed")
    events=candidate.pop("events")
    candidate.update(punch_family=np.asarray(family),timing_kind=np.asarray("uniform_complete_reference_clock"),
        source_reference_sha256=np.asarray(sha(source_path)),source_timing_modified=np.asarray(True),
        interior_generation_version=np.asarray(VERSION))
    payload=_archive(folder/"reference.npz",candidate,plan["cell_reference_byte_cap"])
    _write_json(folder/"reference_payload.json",dict(uncompressed_array_bytes=payload,reference_sha256=sha(folder/"reference.npz")))
    _write_json(folder/"search.json",search);_write_json(folder/"events.json",events)
    summaries=[]
    for number,begin in enumerate(range(0,len(candidate["t"]),plan["screen_chunk_frames"])):
        checkpoint("interior_screen_chunk")
        chunk={key:candidate[key][begin:begin+plan["screen_chunk_frames"]] for key in ("t","qpos","qvel","qacc")}
        summary,arrays=screen_diagnostics(model_path,chunk,np.asarray(origin["schema"]["torque_limits"])[:,1])
        # Diagnostic NaNs mean unsupported allocations and must remain visible.
        upper=sum(np.asarray(a).nbytes for a in arrays.values())+32768
        if sum(p.stat().st_size for p in folder.iterdir() if p.is_file())+upper>plan["cell_product_byte_cap"]:
            raise ValueError("Screening would exceed bounded cell products")
        with (folder/f"screen_{number:04d}.npz").open("xb") as stream:np.savez_compressed(stream,**arrays)
        _write_json(folder/f"screen_{number:04d}.json",summary);summaries.append(summary)
    maximum=("joint_limit_excursion_max_rad","selected_self_penetration_max_m","floor_penetration_max_m",
        "joint_velocity_max_rad_s","joint_acceleration_max_rad_s2","maximum_minimax_effort_ratio",
        "maximum_base_force_residual_n","maximum_base_moment_residual_nm")
    screen=deepcopy(summaries[0]);counts=Counter()
    for summary in summaries:counts.update(summary["support_status_counts"])
    for key in maximum:
        values=[s[key] for s in summaries if s[key] is not None];screen[key]=max(values) if values else None
    for key in ("frames","within_effort_caps_frames","over_effort_caps_frames"):screen[key]=sum(s[key] for s in summaries)
    screen.update(support_status_counts=dict(counts),completed_chunks=len(summaries),diagnostic_arrays="screen_NNNN.npz")
    _write_json(folder/"screen.json",screen)
    contact=search["selected_contact"];reasons=[]
    if not contact["timing_valid"]:reasons.append("uniform_reference_first_contact_timing_outside_contract")
    if contact["inactive_hand_first_or_simultaneous"]:reasons.append("wrong_hand_first_or_simultaneous")
    if contact["any_contact_before_launch"]:reasons.append("contact_before_launch")
    reference=folder/"reference.npz";digest=sha(reference)
    return dict(**entry,**events,kind="uniform_interior",split="d0_development",start_mode="family_guard",preparation_complete_tick=0,
        source_reference_sha256=sha(source_path),source_model_path=logical_path(plan,model_path),source_model_sha256=sha(model_path),
        head_center=np.asarray(candidate["head_center"]).tolist(),target_w=np.asarray(candidate["head_center"]).tolist(),
        active_hand_index=int(candidate["active_hand_index"]),reference_path=logical_path(plan,reference),reference_sha256=digest,
        legacy_runtime_path=logical_path(plan,reference),legacy_runtime_sha256=digest,reference_identity_kind="reference_content",
        duration_s=float(candidate["t"][-1]),episode_duration_s=float(candidate["t"][-1])+1.,phase_denominator_s=float(candidate["t"][-1])+1.,
        guard_time_s=float(candidate["guard_hold_start"]),duration_scale=float(candidate["duration_scale"]),
        generation_disposition="generated",kinematic_admission=not reasons,rejection_reasons=reasons,kinematic_contact=contact,
        diagnostic_screen_path=logical_path(plan,folder/"screen.json"),reference_uncompressed_array_bytes=payload,preserved_original_control=False,
        physical_qualification=False,fine_tuned=False,generator_version=VERSION,
        interior_warning="Untrained timing reference construction only; no final-test execution or physical admission")


def generate(plan_path,output, *, wall_seconds=None):
    started=time.monotonic();require_allocation()
    plan_path=owned(plan_path);plan=read(plan_path);paths,parent,parent_bytes=validate_plan(plan)
    if wall_seconds is not None and wall_seconds!=plan["application_wall_seconds"]:
        raise ValueError("Execution wall must match the immutable plan")
    output=owned(output,exists=False)
    if not output.is_relative_to(ROOT.resolve()/"outputs/task10b"):
        raise ValueError("Fresh interior artifacts belong under outputs/task10b")
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/"plan.json",plan)
    deadline=ExecutionDeadline(plan["application_wall_seconds"],serialization_reserve_s=plan["serialization_reserve_seconds"],session_started_monotonic=started)
    rows=[dict(**entry,kind="uniform_interior",split="d0_development",kinematic_admission=False,
        generation_disposition="unattempted",rejection_reasons=["interior_not_attempted"],reference_path=None,
        physical_qualification=False,fine_tuned=False) for entry in plan["cells"]]
    sequence=0;exception=None;inputs=None
    def progress(stage):
        nonlocal sequence
        _write_json(output/f"progress_{sequence:03d}.json",dict(stage=stage,complete=False,parent_catalog_sha256=plan["parent_catalog"]["sha256"],cells=deepcopy(rows),
            interruption_semantics="A generation_started cell with no terminal receipt has unknown generation extent, never a completed or unattempted cell"))
        sequence+=1
    progress("declared")
    with deadline.signal_handlers():
        for index,entry in enumerate(plan["cells"]):
            current=None;cell_started=time.monotonic()
            def checkpoint(stage):
                deadline.checkpoint(stage,progress={"candidate_id":entry["candidate_id"]})
                if time.monotonic()-cell_started>=plan["maximum_cell_wall_seconds"]:
                    raise BoundedExecutionInterruption(dict(reason="interior_cell_wall_limit",boundary=stage,candidate_id=entry["candidate_id"]))
            try:
                checkpoint("before_interior_cell")
                current=output/entry["candidate_id"];current.mkdir(exist_ok=False)
                rows[index].update(generation_disposition="generation_started",rejection_reasons=["generation_extent_unknown_until_terminal_receipt"])
                _write_json(current/"attempt_started.json",dict(entry=entry,plan_sha256=sha(plan_path),elapsed_wall_s=time.monotonic()-started))
                progress("cell_started")
                if inputs is None:inputs=_native_context(paths)
                checkpoint("interior_context_ready")
                row=_generate_cell(plan,entry,paths,inputs,current,checkpoint)
                checkpoint("interior_cell_complete")
                row.update(interior_plan_sha256=sha(plan_path),parent_catalog_sha256=plan["parent_catalog"]["sha256"],
                    domain_sha256=plan["domain"]["sha256"],generation_source_sha256=plan["generator_sources"][SOURCES[0]]["sha256"])
                if sum(p.stat().st_size for p in current.iterdir() if p.is_file())>plan["cell_product_byte_cap"]:
                    raise ValueError("Completed cell exceeds its product budget")
                rows[index]=row
            except Exception as error:
                timeout=isinstance(error,BoundedExecutionInterruption)
                rows[index].update(generation_disposition=("timeout" if current is not None else "unattempted_timeout") if timeout else "failed",
                    kinematic_admission=False,rejection_reasons=["bounded_timeout" if timeout else "interior_reference_generation_failure"],
                    error=dict(type=type(error).__name__,message=str(error)),details=getattr(error,"receipt",None),
                    retained_partial_reference_path=logical_path(plan,current/"reference.npz") if current is not None and (current/"reference.npz").exists() else None)
                _write_json((current if current is not None else output)/(entry["candidate_id"]+"_failure.json"),rows[index])
                if timeout or not isinstance(error,ValueError):exception=rows[index]["error"]
            progress("cell_disposition")
            if exception:break
    terminal=[row for row in rows if row["generation_disposition"] in ("generated","failed","timeout")]
    payload_bytes=0;unknown_payloads=[]
    for path in output.glob("*/reference.npz"):
        receipt=path.with_name("reference_payload.json")
        if receipt.exists() and read(receipt).get("reference_sha256")==sha(path):
            payload_bytes+=read(receipt)["uncompressed_array_bytes"]
        else:unknown_payloads.append(logical_path(plan,path))
    # Parent row dictionaries retain every field/value and sequence unchanged.
    result=dict(schema=CATALOG,generator_version=VERSION,scope="Existing D0 timing catalog plus four predeclared untrained interiors; no physical qualification",
        complete=all(row["generation_disposition"] in ("generated","failed") for row in rows),planned_cells=20,
        executed_cells=16+len(terminal),inherited_cells=16,planned_new_cells=4,executed_new_cells=len(terminal),
        cells=deepcopy(parent["cells"])+rows,unattempted_cells=[deepcopy(row) for row in rows if row["generation_disposition"].startswith("unattempted")],
        exception=exception,parent_catalog=plan["parent_catalog"],domain=plan["domain"],predicate_path=logical_path(plan,paths["predicate"]),
        predicate_sha256=plan["predicate"]["sha256"],release=plan["release"],bundle=plan["bundle"],source_ancestry=plan["source_ancestry"],
        logical_project_root=plan["logical_project_root"],runtime_resolved_project_root=str(ROOT.resolve()),
        generator_sources=plan["generator_sources"],plan_sha256=sha(plan_path),reference_uncompressed_array_bytes=parent_bytes+payload_bytes,
        reference_bytes_measurement_is_lower_bound=bool(unknown_payloads),unknown_partial_reference_payloads=unknown_payloads,
        new_reference_files_bytes=sum(path.stat().st_size for path in output.glob("*/reference.npz")),
        measured_new_cell_product_bytes={p.name:sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in output.iterdir() if p.is_dir()},
        parent_reference_uncompressed_array_bytes=parent_bytes,initial_reference_byte_cap=plan["reference_byte_cap"],
        reserved_uncompressed_new_reference_bytes=4*plan["cell_reference_byte_cap"],wall_seconds=time.monotonic()-started,
        physics_executed=False,learned_trajectory_generator=False,physical_qualification=False,final_test_executed=False,
        fine_tuning_required_before_expanded_competence_claim=True,no_nearest_substitution=True,
        bounded_execution_notes="Cooperative checkpoints before/after the unchanged <=18-trial uniform solve and each <=16-frame support chunk; parent allocation watchdog must bound any native-call stall. Startup counts against wall limit.")
    _write_json(output/"manifest.json",result)
    return result
