"""Isolated capture-body variant for explicitly constructed inward D3 starts."""
from pathlib import Path
import os,json,time
import numpy as np
import torch
from .phase2_native_state import (capture_comparison,save_snapshot,validate_test_bank_freeze,_copy)
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json
VERSION="task10b_phase2_native_shared_geometry_bank_capture_v2"
PARENT_NATIVE_SHA256="32ee676197b2de746dd3674680229841095051fee41452f5ce0c0b2a1e6f8758"

def capture_shared_start_bank(catalog,domain_path,split,geometry_path,release,output,wall_seconds,*,pre_test_freeze=None):
    """Counted four-world native capture before any actor or physics interval."""
    if (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or split not in ("train","validation","test") or not np.isfinite(wall_seconds) or not 120<wall_seconds<=900):
        raise RuntimeError("Native four-start capture requires a counted CUDA allocation and finite <=900s wall")
    started=time.monotonic()
    # This gate deliberately precedes opening domain, geometry or native inputs.
    freeze=validate_test_bank_freeze(pre_test_freeze) if split=="test" else None
    from .phase2_deadline import ExecutionDeadline
    from .phase2_learning import NativeSpecialistEnv,family_arguments
    from .phase2_shared_catalog_v2 import validate_start_bank
    from .phase2_cache import _preflight,_owned_path
    from .wbc.provenance import runtime_record
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=90.,session_started_monotonic=started)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    env=None;bank=None;error=None;claim=None
    try:
        with deadline.signal_handlers():
            domain_path=Path(domain_path).resolve(strict=True)
            if not domain_path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError("Owned frozen domain required")
            domain_sha=sha256_file(domain_path)
            if freeze and freeze["domain_sha256"]!=domain_sha:raise ValueError("Test capture domain differs from prior freeze")
            if split=="test":
                claim=writable_path(PROJECT_ROOT/"outputs/task10b/p2_final_test_bank_claims/shared_jab_four_starts_v1")
                claim.mkdir(parents=True,exist_ok=False)
                write_json(claim/"claim.json",dict(run=str(folder),pre_test_freeze=freeze,scope="Before opening test physical-start payloads",automatic_retry=False))
            domain=json.loads(domain_path.read_text())
            if domain.get("schema")!="Task10BPhase2Domain/v1":raise ValueError("Frozen canonical Phase2 domain required")
            if freeze and set(freeze["reserved_test_command_groups"])!={row["snapshot"]["command_id"] for row in domain["selected_command_groups"] if row["split"]=="test"}:
                raise ValueError("Reserved final command groups changed before native bank capture")
            geometry=_owned_path(geometry_path);geometry_sha=sha256_file(geometry)
            _preflight(geometry,geometry_sha,1024**2)
            with np.load(geometry,allow_pickle=False) as archive:q,v=archive["qpos"].copy(),archive["qvel"].copy()
            args,paths=family_arguments(release,"jab")
            with np.load(paths["reference"],allow_pickle=False) as archive:nominal=archive["qpos"][0].copy()
            validate_start_bank(q,v,nominal,json.loads(paths["body_metadata"].read_text()),domain,split)
            rows=[row for row in json.loads(Path(catalog).read_text())["cells"] if row["family"]=="jab" and row.get("kind")=="original_control"]
            if len(rows)!=1 or rows[0].get("reference_sha256")!=sha256_file(paths["reference"]):
                raise ValueError("One accepted original jab control required")
            runtime=runtime_record()
            if not runtime["declared_pins_match"]:raise ValueError("Fresh Phase2 runtime pins differ")
            with native_stream_scope():
                env=NativeSpecialistEnv(args,rows,num_envs=4);deadline.checkpoint("after_four_world_native_initialization")
                env.reset(random_phase=False,perturb=False,reference_ids=np.zeros(4,dtype=int),initial_qpos=q,initial_qvel=v)
                bank=capture_comparison(env,held_raw_action=_copy(env.last_action),reference_phase="preparation",support_interval_end_s=_copy(env.sim.data.time))
                state_path=save_snapshot(bank,folder/"native_bank")
            result=dict(schema="task10b_phase2_native_shared_geometry_bank_capture_v2",complete_capture=True,exception=None,
                split=split,domain_sha256=domain_sha,geometry_start_bank=dict(path=str(geometry),sha256=geometry_sha),
                native_start_state_bank=dict(path=str(state_path),sha256=sha256_file(state_path),identity=bank.identity),
                shared_physical_controller_identity=bank.physical_controller_identity,
                initial_support_contact=[state.support_contact.tolist() for state in bank.states],
                source_sha256=sha256_file(__file__),runtime=runtime,slurm_job_id=os.environ["SLURM_JOB_ID"],
                pre_test_freeze=freeze,planned_native_states=4,captured_native_states=4,
                physics_steps_executed=env.physics_steps_executed,actor_updates_executed=env.actor_updates_executed,
                support_semantics="Initial native forward/support classification at timezero; no executed interval",
                no_actor_execution=True,no_g1_actor_execution=True,native_state_parity_qualified=False,
                shared_start_feasibility_qualified=False,session_wall_s=time.monotonic()-started)
    except Exception as exception:
        error=dict(type=type(exception).__name__,message=str(exception))
        result=dict(schema="task10b_phase2_native_shared_geometry_bank_capture_v2",complete_capture=False,exception=error,
            split=split,planned_native_states=4,captured_native_states=0 if bank is None else len(bank.states),
            slurm_job_id=os.environ["SLURM_JOB_ID"],source_sha256=sha256_file(__file__),
            physics_steps_executed=getattr(env,"physics_steps_executed",None),actor_updates_executed=getattr(env,"actor_updates_executed",None),
            native_initialization_attempted=env is not None,execution_extent_unknown=env is None,
            session_wall_s=time.monotonic()-started,pre_test_freeze=freeze,shared_start_feasibility_qualified=False)
    write_json(folder/"bank_capture.json",result)
    if claim is not None:write_json(claim/"terminal.json",dict(run=str(folder),**result))
    return result

