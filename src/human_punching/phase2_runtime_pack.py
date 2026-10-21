"""Versioned bounded runtime pools; full-precision candidate evidence is retained.

The native actor/controller already consumes float32 tensors. Runtime archives
store those three executed arrays at that precision, exact clock/target metadata,
and small recipe fields. Offline accelerations/FK remain in source evidence and
are recomputed/inspected independently; they are not resident runtime inputs.
Packing changes FK rounding and therefore requires the same native qualification
as every other new recipe. It grants no execution or budget authority.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .phase2_cache import _preflight
from .phase2_deadline import ExecutionDeadline
from .phase2_target_catalog import owned_input
from .wbc.provenance import sha256_file,writable_path,write_json

VERSION="task10b_float32_native_runtime_pool_v1"
FAILURE_WRAPPER="task10b_phase2_packed_generation_failure_v1"
CAP=2*1024**3
EXECUTED=frozenset(("qpos","qvel","torque"))
OFFLINE=frozenset(("qacc","glove_pos","source_original_glove_pos","source_reference_time_s",
    "source_time_s","source_clock_time_s","source_polynomial_coefficients","source_segment_coefficients"))
REQUIRED=frozenset(("t","qpos","qvel","torque","head_center","active_hand_index","guard_hold_start"))


def runtime_arrays(source):
    """Keep exact metadata; explicitly report every omitted/quantized array."""
    data={k:np.asarray(v) for k,v in source.items()}
    if "t" not in data and "time_s" in data:data["t"]=data["time_s"]
    if not REQUIRED<=data.keys():raise ValueError("Complete source runtime arrays and metadata required")
    t=data["t"];n=len(t)
    if (t.ndim!=1 or not 3<=n<=6001 or t[0]!=0
            or not np.allclose(np.diff(t),.005,atol=1.e-12,rtol=0)
            or data["qpos"].shape!=(n,55) or data["qvel"].shape!=(n,54)
            or data["torque"].shape!=(n,48) or data["head_center"].shape!=(3,)):
        raise ValueError("Exact5ms accepted human dimensions required")
    if (data["active_hand_index"].shape!=() or data["active_hand_index"].dtype.kind not in "iu"
            or int(data["active_hand_index"]) not in (0,1)
            or data["guard_hold_start"].shape!=() or data["guard_hold_start"].dtype.kind not in "iuf"
            or not 0<=float(data["guard_hold_start"])<=float(t[-1])):
        raise ValueError("Exact active hand and bounded guard metadata required")
    packed={};omitted=[];errors={}
    for key,value in data.items():
        if value.dtype.hasobject or value.dtype.kind not in "biufSU":
            raise ValueError("Only explicit numeric/string arrays may enter a runtime pool")
        if value.dtype.kind in "biuf" and not np.isfinite(value).all():
            raise ValueError("Nonfinite source arrays cannot be packed into a native recipe")
        if key in OFFLINE or key=="time_s":omitted.append(key);continue
        if key in EXECUTED:
            converted=value.astype(np.float32)
            if not np.isfinite(converted).all():raise ValueError("Native precision overflow")
            packed[key]=converted
            errors[key]=float(np.max(np.abs(value.astype(float)-converted.astype(float)),initial=0))
        elif key=="t":packed[key]=value.astype(np.float64)
        elif value.nbytes<=4096:packed[key]=value.copy()
        else:raise ValueError("Unreviewed large runtime array; add an explicit storage contract first: "+key)
    return packed,dict(version=VERSION,quantization_max_absolute_error=errors,
        omitted_arrays=sorted(omitted),exact_metadata_fields=sorted(set(packed)-EXECUTED),
        derivative_scope="Source full-precision derivatives/screens are preserved. Native executes these float32 state/drive knots; no exact finite-difference equality or continuous-C2 claim is inferred from quantization.",
        native_execution_parity_assumed=False,physical_qualification=False)


def verify_rejection(row):
    if (row.get("reference_identity_kind")!="generation_failure_receipt"
            or any(key in row for key in ("reference_path","legacy_runtime_path","legacy_runtime_sha256"))
            or row.get("kinematic_admission") is not False
            or row.get("generation_failure_sha256")!=row.get("reference_sha256")
            or not isinstance(row.get("rejection_reasons"),list) or not row["rejection_reasons"]
            or not all(isinstance(reason,str) and reason.strip() for reason in row["rejection_reasons"])):
        raise ValueError("Absent reference requires exact immutable offline rejection evidence")
    path=owned_input(dict(path=row["generation_failure_path"],sha256=row["generation_failure_sha256"]))
    evidence=json.loads(path.read_text())
    if (not isinstance(evidence,dict) or evidence.get("recipe_sha256")!=row.get("recipe_sha256")
            or evidence.get("rejection_reasons")!=row["rejection_reasons"]):
        raise ValueError("Generation failure recipe/reasons differ from its row")
    if row.get("runtime_storage_version") is not None:
        if row["runtime_storage_version"]!=VERSION or row.get("runtime_failure_wrapper_version")!=FAILURE_WRAPPER:
            raise ValueError("Exact packed failure ancestry required")
        parent=owned_input(dict(path=row["unpacked_generation_failure_path"],sha256=row["unpacked_generation_failure_sha256"]))
        original=json.loads(parent.read_text())
        expected={**original,"recipe_sha256":row["recipe_sha256"],"runtime_storage_version":VERSION,
            "runtime_failure_wrapper_version":FAILURE_WRAPPER,"unpacked_recipe_sha256":row["unpacked_recipe_sha256"],
            "unpacked_generation_failure_path":str(parent),"unpacked_generation_failure_sha256":row["unpacked_generation_failure_sha256"]}
        if original.get("recipe_sha256")!=row["unpacked_recipe_sha256"] or evidence!=expected:
            raise ValueError("Packed failure changed its preserved original evidence")
        if any(key in original for key in ("runtime_storage_version","runtime_failure_wrapper_version",
                "unpacked_generation_failure_path","unpacked_generation_failure_sha256",
                "reference_path","legacy_runtime_path","legacy_runtime_sha256")):
            raise ValueError("Recursive or native-file failure evidence is unsupported")
    elif any(key in evidence for key in ("runtime_storage_version","runtime_failure_wrapper_version",
            "unpacked_generation_failure_path","unpacked_generation_failure_sha256")):
        raise ValueError("Source failure cannot hide a prior packing wrapper")
    return path,evidence


def pool_inventory(catalog_bindings,*,cap=CAP):
    if type(cap) is not int or not 0<cap<=CAP:raise ValueError("Active reference pool ceiling is2GiB")
    seen={};total=0
    for binding in catalog_bindings:
        catalog=json.loads(owned_input(binding).read_text())
        if catalog.get("runtime_storage_version")!=VERSION:raise ValueError("Only versioned runtime catalogs enter this pool")
        for row in catalog["cells"]:
            if not row.get("reference_path"):
                verify_rejection(row)
                continue
            path=owned_input(dict(path=row["reference_path"],sha256=row["reference_sha256"]))
            legacy=owned_input(dict(path=row["legacy_runtime_path"],sha256=row["legacy_runtime_sha256"]))
            if (row.get("runtime_storage_version")!=VERSION or legacy!=path
                    or row["legacy_runtime_sha256"]!=row["reference_sha256"]):
                raise ValueError("Packed active pool state/feedforward identities differ")
            if path in seen:
                if seen[path]["sha256"]!=row["reference_sha256"]:raise ValueError("Conflicting active reference identity")
                continue
            item=_preflight(path,row["reference_sha256"],cap)
            total+=item.uncompressed_array_bytes
            if total>cap:raise ValueError("Combined selected family/split runtime pool exceeds2GiB")
            seen[path]=dict(sha256=row["reference_sha256"],bytes=item.uncompressed_array_bytes)
    return dict(unique_archives=len(seen),uncompressed_array_bytes=total,byte_cap=cap,
        paths={str(k):v for k,v in seen.items()})


def pack(plan_path,output,*,wall_seconds):
    started=time.monotonic();deadline=ExecutionDeadline(wall_seconds,
        serialization_reserve_s=min(90.,wall_seconds/4),session_started_monotonic=started)
    plan_path=Path(plan_path).resolve(strict=True)
    plan=json.loads(owned_input(dict(path=plan_path,sha256=sha256_file(plan_path))).read_text())
    if (plan.get("schema")!="task10b_phase2_runtime_pack_plan_v1" or plan.get("storage_version")!=VERSION
            or type(plan.get("byte_cap")) is not int or not 0<plan["byte_cap"]<=CAP
            or not 1<=len(plan.get("catalogs",[]))<=64 or not 1<=len(plan.get("recipes",[]))<=16):
        raise ValueError("A finite exact runtime pool plan required")
    previous=pool_inventory(plan.get("previous_packed_catalogs",[]),cap=plan["byte_cap"])
    catalogs=[(binding,json.loads(owned_input(binding).read_text())) for binding in plan["catalogs"]]
    recipes={binding["sha256"]:json.loads(owned_input(binding).read_text()) for binding in plan["recipes"]}
    if any(cat.get("schema")!="task10b_phase2_reference_catalog_v1" or cat.get("complete") is not True
            or not cat.get("cells") or cat.get("runtime_storage_version") is not None for _,cat in catalogs):
        raise ValueError("Complete nonempty candidate catalogs required")
    if any(len({r["candidate_id"] for r in cat["cells"]})!=len(cat["cells"]) for _,cat in catalogs):
        raise ValueError("Duplicate source candidate IDs cannot enter a runtime pool")
    planned=sum(len(cat["cells"]) for _,cat in catalogs)
    if planned>4096:raise ValueError("Runtime pool plan exceeds finite4096-cell construction bound")
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"plan.json",plan)
    packed_recipes={}
    for digest,recipe in recipes.items():
        updated={**recipe,"unpacked_recipe_sha256":digest,"runtime_storage":dict(version=VERSION,
            executed_array_dtype="float32",clock_dtype="float64",metadata_preserved=True,
            offline_source_artifacts_preserved=True,requires_native_qualification=True)}
        path=folder/("recipe_"+digest+".json");write_json(path,updated)
        packed_recipes[digest]=dict(path=str(path),sha256=sha256_file(path))
    total=previous["uncompressed_array_bytes"];packed_sources={};packed_failures={};outputs=[];ledger=[];error=None
    with deadline.signal_handlers():
        for index,(binding,source_catalog) in enumerate(catalogs):
            rows=[]
            try:
                for row in source_catalog["cells"]:
                    deadline.checkpoint("runtime_pack_cell",progress={"catalog":index,"candidate_id":row["candidate_id"]})
                    old_recipe=row["recipe_sha256"]
                    if old_recipe not in packed_recipes:raise ValueError("Missing exact source recipe")
                    if row.get("kind")=="original_control":raise ValueError("Accepted original controls must remain byte-identical")
                    result={**row,"unpacked_recipe_sha256":old_recipe,"recipe_sha256":packed_recipes[old_recipe]["sha256"],
                        "runtime_storage_version":VERSION}
                    if row.get("reference_path"):
                        source_path=owned_input(dict(path=row["reference_path"],sha256=row["reference_sha256"]))
                        legacy=owned_input(dict(path=row["legacy_runtime_path"],sha256=row["legacy_runtime_sha256"]))
                        if legacy!=source_path or row["legacy_runtime_sha256"]!=row["reference_sha256"]:
                            raise ValueError("Reviewed packing route requires one exact state/feedforward source archive")
                        source_digest=row["reference_sha256"]
                        if source_digest not in packed_sources:
                            # Copies, conversion and archive work stay within a
                            # conservative256MiB owned working-set reservation.
                            _preflight(source_path,row["reference_sha256"],64*1024**2)
                            with np.load(source_path,allow_pickle=False) as archive:
                                arrays,receipt=runtime_arrays({key:archive[key].copy() for key in archive.files})
                            payload=sum(value.nbytes for value in arrays.values())
                            if total+payload>plan["byte_cap"]:raise ValueError("Combined active runtime reference pool would exceed2GiB")
                            path=folder/("reference_"+row["reference_sha256"]+".npz")
                            with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
                            write_json(path.with_suffix(".json"),dict(source_reference_path=str(source_path),
                                source_reference_sha256=row["reference_sha256"],payload_bytes=payload,**receipt))
                            packed_sources[source_digest]=(path,sha256_file(path),payload)
                            total+=payload
                        path,digest,payload=packed_sources[source_digest]
                        result.update(reference_path=str(path),reference_sha256=digest,legacy_runtime_path=str(path),
                            legacy_runtime_sha256=digest,reference_identity_kind="reference_content",
                            unpacked_reference_path=str(source_path),unpacked_reference_sha256=row["reference_sha256"])
                    else:
                        parent,evidence=verify_rejection(row)
                        key=(row["generation_failure_sha256"],result["recipe_sha256"])
                        ancestry=dict(runtime_storage_version=VERSION,runtime_failure_wrapper_version=FAILURE_WRAPPER,
                            unpacked_recipe_sha256=old_recipe,unpacked_generation_failure_path=str(parent),
                            unpacked_generation_failure_sha256=row["generation_failure_sha256"])
                        if key not in packed_failures:
                            path=folder/f"failure_{len(packed_failures):04d}.json"
                            write_json(path,{**evidence,**ancestry,"recipe_sha256":result["recipe_sha256"]})
                            packed_failures[key]=(path,sha256_file(path))
                        path,digest=packed_failures[key]
                        result.update(**ancestry,generation_failure_path=str(path),generation_failure_sha256=digest,
                            reference_sha256=digest)
                    rows.append(result);ledger.append(dict(catalog=index,candidate_id=row["candidate_id"],status="packed"))
                    write_json(folder/f"progress_{len(ledger):04d}.json",dict(completed=ledger,planned=planned,
                        active_pool_uncompressed_bytes=total))
            except Exception as exception:
                error=dict(type=type(exception).__name__,message=str(exception),catalog=index,
                    completed_in_catalog=len(rows));
            path=folder/f"catalog_{index:02d}.json"
            recipe_ids={row["recipe_sha256"] for row in rows}
            write_json(path,{**source_catalog,"cells":rows,"runtime_storage_version":VERSION,
                "recipe_sha256":next(iter(recipe_ids)) if len(recipe_ids)==1 else None,
                "unpacked_catalog_path":binding["path"],"unpacked_catalog_sha256":binding["sha256"],
                "runtime_pack_complete":len(rows)==len(source_catalog["cells"]),"complete":len(rows)==len(source_catalog["cells"]),
                "unpacked_cells_not_completed":[r["candidate_id"] for r in source_catalog["cells"]][len(rows):],
                "physical_qualification":False})
            outputs.append(dict(path=str(path),sha256=sha256_file(path)))
            if error:break
    completed={(item["catalog"],item["candidate_id"]) for item in ledger}
    all_cells=[dict(catalog=index,candidate_id=row["candidate_id"],
        status="packed" if (index,row["candidate_id"]) in completed else "unattempted")
        for index,(_,cat) in enumerate(catalogs) for row in cat["cells"]]
    result=dict(schema="task10b_phase2_runtime_pool_v1",storage_version=VERSION,status="failed" if error else "packed",
        planned_cells=planned,completed_cells=len(ledger),unattempted_cells=planned-len(ledger),exception=error,
        catalogs=outputs,recipes=packed_recipes,previous_pool=previous,
        active_pool_uncompressed_array_bytes=total,byte_cap=plan["byte_cap"],
        wall_s=time.monotonic()-started,source_sha256=sha256_file(Path(__file__)),
        cell_ledger=all_cells,resident_working_set_reservation_bytes=256*1024**2,
        physics_steps=0,actor_calls=0,physical_qualification=False,
        archived_candidate_evidence_is_not_resident_runtime_data=True)
    write_json(folder/"pool.json",result)
    return result
