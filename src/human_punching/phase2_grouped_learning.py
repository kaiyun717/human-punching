"""Train-only target-reference adapter for the original specialist PPO.

This module submits no jobs. Its execution entrypoint requires a counted CUDA
allocation and a concrete held-out physical validation callback. Reference-phase
resets are rehearsal, never uninterrupted-cycle or shared-start evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .phase2_cache import _owned_path, _preflight
from .phase2_deadline import ExecutionDeadline, BoundedExecutionInterruption
from .phase2_domain import FAMILIES, SEEDS, fingerprint
from .phase2_learning import (NativeSpecialistEnv, PilotBudget, family_arguments,
    fresh_policy, train_pilot, _tensor_view)
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json, runtime_record
from .wbc.splits import assign_trajectory_groups

RESET_PROTOCOL = "task10b_expanded_joint_only_rehearsal_v1"
MAX_INPUT_BYTES = 2*1024**3
MAX_RESIDENT_BYTES = 1024**3
DEFAULT_RESIDENT_BYTES = 512*1024**2


def grouped_stopping_rules(budget):
    """The reviewed original pilot stops, recorded before any fresh transition.

    A falling frozen expanded control stops this strict initial recipe before
    learning. Changing that rule requires a new explicit recipe declaration.
    """
    return dict(schema="task10b_phase2_grouped_stopping_rules_v1",
        validation_updates=list(range(0,budget.maximum_updates+1,8)),
        initial_validation_hard_failure="stop_before_learning",
        validation_fall_nonfinite_root_assistance_effort_cap="hard_stop",
        training_nonfinite_root_assistance_effort_cap="hard_stop",
        training_fall="original_done_reset_no_commanded_cycle_credit",
        score=["complete_cycle_fraction","correct_hit_timing_fraction","negative_outward_active_glove_rms_m"],
        no_improvement_checks=2,validation_every_updates=8,
        fresh_transition_cap=budget.transition_cap,maximum_wall_s=budget.maximum_wall_s,
        maximum_updates=budget.maximum_updates,environments=128,rollout_steps=64,
        initial_validation_is_frozen_control=True,conditional_replication_budget_unlocked=False)


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class GroupedTrainingDeclaration:
    family: str
    seed: int
    command_groups: tuple[int,...]
    trained_timings_s: tuple[float,float]
    interpolation_timing_s: float
    target_candidate_identities: tuple[tuple[int,str],...]
    recipe_sha256: str
    domain_sha256: str
    target_geometry_sha256: str
    predicate_sha256: str
    declared_mask_sha256: str
    parent_checkpoint_sha256: str
    resident_reference_bytes: int = DEFAULT_RESIDENT_BYTES

    def __post_init__(self):
        object.__setattr__(self,"command_groups",tuple(self.command_groups))
        object.__setattr__(self,"trained_timings_s",tuple(self.trained_timings_s))
        object.__setattr__(self,"target_candidate_identities",tuple(tuple(x) for x in self.target_candidate_identities))
        if (self.family not in FAMILIES or self.seed not in SEEDS or len(self.command_groups)!=32
                or len(set(self.command_groups))!=32
                or any(type(g) is not int or g<0 or g==3307 for g in self.command_groups)
                or set(assign_trajectory_groups(self.command_groups).values())!={"train"}):
            raise ValueError("Exactly32 deterministic training command groups and a declared transferred seed required")
        if (len(self.trained_timings_s)!=2 or not all(map(math.isfinite,self.trained_timings_s))
                or not math.isfinite(self.interpolation_timing_s)
                or not .5<=self.trained_timings_s[0]<self.interpolation_timing_s<self.trained_timings_s[1]):
            raise ValueError("Two trained endpoints and excluded interior timing required")
        if (len(self.target_candidate_identities)!=32
                or {g for g,_ in self.target_candidate_identities}!=set(self.command_groups)
                or not all(_sha(h) for _,h in self.target_candidate_identities)
                or not all(_sha(getattr(self,k)) for k in ("recipe_sha256","domain_sha256","target_geometry_sha256",
                    "predicate_sha256","declared_mask_sha256","parent_checkpoint_sha256"))
                or type(self.resident_reference_bytes) is not int
                or not 0<self.resident_reference_bytes<=MAX_RESIDENT_BYTES):
            raise ValueError("Immutable per-group placement and bounded source/recipe identities required")

    @property
    def identity(self):return fingerprint(asdict(self))


def select_training_rows(declaration, catalog, domain, targets):
    """Filter metadata before opening references; never sample a held-out row."""
    if (catalog.get("schema")!="task10b_phase2_reference_catalog_v1"
            or domain.get("schema")!="Task10BPhase2Domain/v1"
            or targets.get("schema")!="Task10BPhase2RegisteredTargets/v1"):
        raise ValueError("Canonical reference/domain/target schemas required")
    groups={r["snapshot"]["command_id"]:r for r in domain["selected_command_groups"]}
    if len(groups)!=len(domain["selected_command_groups"]):raise ValueError("Duplicate domain command group")
    if {g for g,r in groups.items() if r["split"]=="train"}!=set(declaration.command_groups):
        raise ValueError("Training groups differ from canonical32-group selection")
    geometry={r["candidate_identity"]:r for r in targets["geometry_candidates"]}
    if len(geometry)!=len(targets["geometry_candidates"]):raise ValueError("Duplicate registered target candidate")
    placements=dict(declaration.target_candidate_identities)
    selected={};excluded=[];seen=set()
    for row in catalog["cells"]:
        candidate=row["candidate_id"]
        if not isinstance(candidate,str) or not candidate or candidate in seen:
            raise ValueError("Unique explicit reference candidate IDs required")
        seen.add(candidate)
        reason=("other_family" if row.get("family")!=declaration.family else
                "not_a_declared_training_group" if row.get("command_id") not in declaration.command_groups else
                "held_out_or_wrong_split" if row.get("split")!="train" else
                "not_a_trained_timing" if row.get("requested_ttc_s") not in declaration.trained_timings_s else
                "historical_control_not_expanded_training" if row.get("kind")=="original_control" else None)
        if reason:
            excluded.append({"candidate_id":candidate,"reason":reason});continue
        group_id=row["command_id"];timing=row["requested_ttc_s"];key=(group_id,timing)
        if key in selected:raise ValueError("Multiple references for the same immutable training request")
        target=geometry[placements[group_id]];source=groups[group_id]["snapshot"]
        if (row.get("kinematic_admission") is not True or row.get("original_replica_id")!=0
                or row.get("recipe_sha256")!=declaration.recipe_sha256
                or row.get("target_candidate_identity")!=placements[group_id]
                or fingerprint({k:v for k,v in target.items() if k!="candidate_identity"})!=target["candidate_identity"]
                or target["source_domain_sha256"]!=declaration.domain_sha256
                or target["command_id"]!=group_id or target["split"]!="train" or target["original_replica_id"]!=0
                or source["replica_id"]!=0 or target["snapshot_slot_id"]!=source["slot_id"]
                or target["payload_sha256"]!=source["payload_sha256"]
                or tuple(row.get("target_w",()))!=tuple(target["head_target_w"])):
            raise ValueError("Training request lacks exact admitted snapshot/placement/recipe binding")
        if (row.get("start_mode","family_guard")!="family_guard" or row.get("preparation_complete_tick",0)!=0
                or type(row.get("launch_tick")) is not int or row["launch_tick"]<0 or row["launch_tick"]%4
                or type(row.get("outward_end_tick")) is not int or row["outward_end_tick"]<=row["launch_tick"]
                or not all(isinstance(row.get(k),(int,float)) and not isinstance(row[k],bool)
                    and math.isfinite(row[k]) and row[k]>0 for k in ("duration_s","episode_duration_s","phase_denominator_s"))
                or abs(row["episode_duration_s"]-row["duration_s"]-1.)>1.e-8
                or row["phase_denominator_s"]!=row["episode_duration_s"]
                or row["outward_end_tick"]*.005>row["duration_s"]):
            raise ValueError("Original family-guard full-reference event/horizon contract required")
        selected[key]=json.loads(json.dumps(row,allow_nan=False))
    expected=set(itertools.product(declaration.command_groups,declaration.trained_timings_s))
    if set(selected)!=expected:
        raise ValueError("Exactly64 admitted references required; unsupported cells cannot be silently resampled")
    rows=[selected[(g,t)] for g in declaration.command_groups for t in declaration.trained_timings_s]
    receipt=dict(schema="task10b_phase2_grouped_training_selection_v1",declaration_identity=declaration.identity,
        selected_candidate_ids=[r["candidate_id"] for r in rows],selected_rows_sha256=fingerprint(rows),
        excluded_rows=excluded,training_groups=list(declaration.command_groups),
        trained_timings_s=list(declaration.trained_timings_s),interpolation_timing_s=declaration.interpolation_timing_s,
        interpolation_used_for_training=False,sampling_split="train",one_placement_per_group=True,
        no_nearest_reference_substitution=True,reset_protocol=RESET_PROTOCOL)
    return rows,receipt


def preflight_training_references(rows, *, resident_limit=MAX_RESIDENT_BYTES):
    """Hash and inspect selected archives before any full reference load."""
    if type(resident_limit) is not int or not 0<resident_limit<=MAX_RESIDENT_BYTES:
        raise ValueError("Positive <=1GiB reference resident limit required")
    headers={};total_archive=total_arrays=0;estimate=0
    for row in rows:
        for name in ("reference","legacy_runtime"):
            path=_owned_path(row[name+"_path"]);expected=row[name+"_sha256"]
            if path in headers:
                if headers[path].sha256!=expected:raise ValueError("Conflicting selected reference hash")
                continue
            header=_preflight(path,expected,MAX_INPUT_BYTES);headers[path]=header
            total_archive+=header.archive_bytes;total_arrays+=header.uncompressed_array_bytes
            arrays={a.name:a for a in header.arrays};n=arrays["qpos"].shape[0]
            if (not 3<=n<=6001 or arrays["qpos"].shape!=(n,55)
                    or arrays["qvel"].shape!=(n,54)):
                raise ValueError("At most6001 source5ms frames and accepted physical dimensions required")
            if max(total_archive,total_arrays)>MAX_INPUT_BYTES:
                raise ValueError("Selected reference pool exceeds2GiB input or bounded resident allowance")
        # The original resident library materializes every selected row even if
        # two rows point to the same archive. Count those copies before loading.
        for name in ("reference","legacy_runtime"):
            header=headers[_owned_path(row[name+"_path"])]
            n=next(a.shape[0] for a in header.arrays if a.name=="qpos")
            estimate+=header.uncompressed_array_bytes*3+n*4096
        if estimate>resident_limit:
            raise ValueError("Selected reference pool exceeds2GiB input or bounded resident allowance")
        for name in ("reference","legacy_runtime"):
            with np.load(row[name+"_path"],allow_pickle=False) as archive:
                clock=np.asarray(archive["t"] if "t" in archive else archive["time_s"],float)
                n=headers[_owned_path(row[name+"_path"])].arrays
                n=next(a.shape[0] for a in n if a.name=="qpos")
                if (clock.shape!=(n,) or clock[0]!=0 or not np.allclose(np.diff(clock),.005,atol=1.e-12,rtol=0)
                        or abs(clock[-1]-row["duration_s"])>1.e-8
                        or not np.array_equal(archive["head_center"],row["target_w"])
                        or int(archive["active_hand_index"])!=(0 if row["family"] in ("jab","left_hook") else 1)):
                    raise ValueError("Selected reference grid, target or active hand changed")
    return dict(unique_archives=len(headers),archive_bytes=total_archive,array_bytes=total_arrays,
        conservative_resident_allowance_bytes=estimate,resident_limit_bytes=resident_limit,
        reference_payloads_loaded=False,metadata_arrays_loaded=True,
        memory_scope="Bounded reference storage/derived-array allowance only; no total process/GPU-memory claim")


def joint_only_rehearsal_noise(qpos,qvel,qadr,vadr,joint_low,joint_high):
    """Original clipped Gaussian shape; no perturbation of the free root."""
    if (qpos.ndim!=2 or qpos.shape[1]!=55 or qvel.shape!=(len(qpos),54)
            or set(map(int,qadr))!=set(range(7,55)) or set(map(int,vadr))!=set(range(6,54))):
        raise ValueError("Accepted48 actuated coordinates excluding the free root required")
    q=qpos.clone();v=qvel.clone();n=len(q)
    q[:,qadr]=(q[:,qadr]+(.005*torch.randn((n,48),device=q.device,dtype=q.dtype)).clamp(-.015,.015)).clamp(joint_low,joint_high)
    v[:,vadr]+=(.02*torch.randn((n,48),device=v.device,dtype=v.dtype)).clamp(-.06,.06)
    return q,v


class GroupedSpecialistEnv(NativeSpecialistEnv):
    """Reuse native controller/PPO observations, changing training reset noise only."""
    def __init__(self,*args,deadline,resident_limit,**kwargs):
        self.execution_deadline=deadline
        super().__init__(*args,**kwargs)
        if self.library.bytes>resident_limit:
            raise ValueError("Measured resident reference tensor bytes exceed declared limit")

    def reset(self,env_ids=None,*,random_phase=True,perturb=True,**kwargs):
        if "initial_qpos" in kwargs or "initial_qvel" in kwargs:
            if perturb:raise ValueError("Explicit comparison starts cannot receive additional training noise")
        super().reset(env_ids,random_phase=random_phase,perturb=False,**kwargs)
        if perturb:
            ids=torch.arange(self.n,device=self.device) if env_ids is None else env_ids
            q=_tensor_view(self.sim.data.qpos);v=_tensor_view(self.sim.data.qvel)
            q[ids],v[ids]=joint_only_rehearsal_noise(q[ids],v[ids],self.qadr,self.vadr,self.joint_low,self.joint_high)
            self.sim.forward();self.last_support=self.support_contact()
        return self.observe()

    def step(self,action):
        try:
            self.execution_deadline.checkpoint("grouped_training_actor_boundary",progress={
                "physics_steps":self.physics_steps_executed,"actor_updates":self.actor_updates_executed})
        except BoundedExecutionInterruption as error:
            self.deadline_interruption_receipt=error.receipt
            raise
        return super().step(action)


def validate_validation_descriptor(declaration,descriptor,domain):
    groups=tuple(r["snapshot"]["command_id"] for r in domain["selected_command_groups"] if r["split"]=="validation")
    if (descriptor.get("schema")!="task10b_phase2_grouped_validation_v1" or descriptor.get("family")!=declaration.family
            or descriptor.get("seed")!=declaration.seed
            or descriptor.get("split")!="validation" or set(descriptor.get("command_groups",[]))!=set(groups)
            or len(descriptor.get("command_groups",[]))!=8 or len(groups)!=8
            or set(assign_trajectory_groups(groups).values())!={"validation"}
            or descriptor.get("trained_timings_s")!=list(declaration.trained_timings_s)
            or descriptor.get("interpolation_timing_s")!=declaration.interpolation_timing_s
            or descriptor.get("expected_attempts")!=96
            or any(descriptor.get(k)!=getattr(declaration,k) for k in ("recipe_sha256","domain_sha256",
                "target_geometry_sha256","predicate_sha256","declared_mask_sha256"))):
        raise ValueError("Physical stopping callback requires the exact separate8-group validation contract")
    trial_ids=descriptor.get("trial_ids",[])
    if len(trial_ids)!=96 or len(set(trial_ids))!=96 or not all(isinstance(x,str) and x for x in trial_ids):
        raise ValueError("Predeclare all96 validation trial IDs")
    return fingerprint(descriptor)


def validate_callback_panel(panel,descriptor):
    if panel.get("grouped_validation_identity")!=fingerprint(descriptor):
        raise ValueError("Validation callback returned a different frozen validation domain")
    trials=panel.get("trials",[])
    expected=set(itertools.product(descriptor["command_groups"],
        (*descriptor["trained_timings_s"],descriptor["interpolation_timing_s"]),range(4)))
    actual=[(r["command_id"],r["requested_ttc_s"],r["start_index"]) for r in trials]
    if (len(actual)!=96 or set(actual)!=expected
            or {r.get("trial_id") for r in trials}!=set(descriptor["trial_ids"])):
        raise ValueError("Physical validation must retain exactly all96 planned attempts, including failures")
    return panel


def _owned_input(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
        raise ValueError("Grouped input must resolve to a regular human-punching file")
    return path


def _read_bound(path,digest):
    path=_owned_input(path)
    if sha256_file(path)!=digest:raise ValueError("Grouped training input changed")
    return json.loads(path.read_text())


def train_grouped(declaration,*,catalog_path,catalog_sha256,domain_path,target_geometry_path,
                  predicate_path,mask_path,recipe_path,validation_descriptor,validation_callback,
                  release,output,budget=PilotBudget(),stopping_rules=None):
    """One fresh bounded original PPO session; validation callback owns real trials.

    Callback signature is ``(policy, update, *, deadline)``. It must use the same
    session deadline and return the complete descriptor-bound physical panel.
    Missing callback is an error, not a successful validation placeholder.
    """
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Grouped training requires a counted Slurm CUDA allocation")
    if not callable(validation_callback) or not isinstance(budget,PilotBudget):
        raise ValueError("Concrete held-out physical validation callback and original128x64 bounded budget required")
    expected_rules=grouped_stopping_rules(budget)
    if stopping_rules is not None and stopping_rules!=expected_rules:
        raise ValueError("Expanded stopping-rule change requires a separately reviewed recipe; strict original stops required")
    started=time.monotonic();deadline=ExecutionDeadline(budget.maximum_wall_s,
        serialization_reserve_s=budget.shutdown_reserve_s,session_started_monotonic=started)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"declaration.json",asdict(declaration));write_json(folder/"validation_descriptor.json",validation_descriptor)
    write_json(folder/"stopping_rules.json",expected_rules)
    env=None;summary=None;error_receipt=None;memory=None
    try:
        with deadline.signal_handlers():
            domain=_read_bound(domain_path,declaration.domain_sha256)
            targets=_read_bound(target_geometry_path,declaration.target_geometry_sha256)
            catalog=_read_bound(catalog_path,catalog_sha256)
            for path,digest in ((predicate_path,declaration.predicate_sha256),(mask_path,declaration.declared_mask_sha256),
                                (recipe_path,declaration.recipe_sha256)):_read_bound(path,digest)
            rows,selection=select_training_rows(declaration,catalog,domain,targets)
            validate_validation_descriptor(declaration,validation_descriptor,domain)
            memory=preflight_training_references(rows,resident_limit=declaration.resident_reference_bytes)
            write_json(folder/"selection.json",selection);write_json(folder/"reference_preflight.json",memory)
            args,paths=family_arguments(release,declaration.family)
            if sha256_file(paths["checkpoint"])!=declaration.parent_checkpoint_sha256:
                raise ValueError("Grouped fine-tuning must start from the accepted family checkpoint")
            runtime=runtime_record()
            if not runtime["declared_pins_match"]:raise ValueError("Phase2 runtime pins differ")
            write_json(folder/"launch.json",dict(schema="task10b_phase2_grouped_launch_v1",
                declaration_identity=declaration.identity,slurm_job_id=os.environ["SLURM_JOB_ID"],runtime=runtime,
                sources={str(Path(__file__).resolve()):sha256_file(__file__)},budget=asdict(budget),
                fresh_transition_cap=budget.transition_cap,selection_sha256=fingerprint(selection),
                validation_descriptor_identity=fingerprint(validation_descriptor),sampling_split="train",
                reset_protocol=RESET_PROTOCOL,stopping_rules=expected_rules,expanded_native_qualification=False))
            torch.set_num_threads(1);torch.manual_seed(declaration.seed);np.random.seed(declaration.seed)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            from .wbc.telemetry.cycle_qualification import native_stream_scope
            with native_stream_scope():
                deadline.checkpoint("before_grouped_native_initialization")
                env=GroupedSpecialistEnv(args,rows,num_envs=128,deadline=deadline,
                    resident_limit=declaration.resident_reference_bytes)
                deadline.checkpoint("after_grouped_native_initialization")
                policy=fresh_policy(paths,env.device)
                memory["measured_resident_reference_tensor_bytes"]=env.library.bytes
                write_json(folder/"reference_residency.json",memory)
                def validate(policy,update):
                    deadline.checkpoint("before_grouped_physical_validation",progress={"update":update})
                    return validate_callback_panel(validation_callback(policy,update,deadline=deadline),validation_descriptor)
                remaining=budget.maximum_wall_s-(time.monotonic()-started)
                effective=replace(budget,maximum_wall_s=remaining)
                summary=train_pilot(env,policy,folder/"training",paths=paths,seed=declaration.seed,
                    validate=validate,budget=effective)
    except Exception as error:
        error_receipt=dict(type=type(error).__name__,message=str(error))
        if isinstance(error,BoundedExecutionInterruption):error_receipt["deadline_receipt"]=error.receipt
    result={**(summary or {}),"grouped_schema":"task10b_phase2_grouped_training_receipt_v1",
        "declaration_identity":declaration.identity,"recipe_sha256":declaration.recipe_sha256,
        "domain_sha256":declaration.domain_sha256,"target_geometry_sha256":declaration.target_geometry_sha256,
        "predicate_sha256":declaration.predicate_sha256,"declared_mask_sha256":declaration.declared_mask_sha256,
        "trained_timings_s":list(declaration.trained_timings_s),"interpolation_timing_s":declaration.interpolation_timing_s,
        "interpolation_used_for_training":False,"sampling_split":"train","reset_protocol":RESET_PROTOCOL,
        "training_groups":list(declaration.command_groups),"target_candidate_identities":declaration.target_candidate_identities,
        "validation_descriptor_identity":fingerprint(validation_descriptor),"wrapper_exception":error_receipt,
        "stopping_rules":expected_rules,
        "reference_memory":memory,"session_wall_s":time.monotonic()-started,
        "deadline_interruption":getattr(env,"deadline_interruption_receipt",None),
        "fresh_transition_cap":budget.transition_cap,"uninterrupted_cycle_qualification":False,
        "phase2_complete":False,"formal_acceptance":False}
    if error_receipt is not None or summary is None:result.update(status="failed",exception=error_receipt)
    write_json(folder/"grouped_training_receipt.json",result)
    return result


def load_configuration(path, expected_sha256):
    """Read only fixed, hashed inputs; no arbitrary import or callback injection."""
    configuration=_read_bound(path,expected_sha256)
    if configuration.get("schema")!="task10b_phase2_grouped_training_configuration_v1":
        raise ValueError("Immutable grouped training configuration required")
    items={}
    for name in ("declaration","catalog","domain","target_geometry","predicate","mask","recipe","validation_configuration"):
        value=configuration[name]
        if set(value)!={"path","sha256"} or not _sha(value["sha256"]):
            raise ValueError("Every grouped configuration input requires exact path and SHA256")
        items[name]=dict(path=str(_owned_input(value["path"])),sha256=value["sha256"])
        if sha256_file(items[name]["path"])!=value["sha256"]:
            raise ValueError("Frozen grouped configuration input changed")
    declaration=GroupedTrainingDeclaration(**_read_bound(items["declaration"]["path"],items["declaration"]["sha256"]))
    for name,attribute in (("domain","domain_sha256"),("target_geometry","target_geometry_sha256"),
            ("predicate","predicate_sha256"),("mask","declared_mask_sha256"),("recipe","recipe_sha256")):
        if items[name]["sha256"]!=getattr(declaration,attribute):
            raise ValueError("Grouped declaration/input identity differs")
    budget=PilotBudget(**configuration["budget"])
    if configuration.get("stopping_rules")!=grouped_stopping_rules(budget):
        raise ValueError("Exact finite physical stopping rules must be declared in the immutable configuration")
    release=Path(configuration["release"]).resolve(strict=True)
    if not release.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Accepted release must resolve inside human-punching")
    return declaration,items,budget,configuration,release


def execute_configuration(path,expected_sha256,output):
    """Concrete CLI path for one separately accounted, finite training packet."""
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Grouped training configuration execution requires a counted Slurm CUDA allocation")
    started=time.monotonic()
    declaration,items,budget,configuration,release=load_configuration(path,expected_sha256)
    from .phase2_validation_callback import GroupedValidationCallback
    # Callback preflight is included in the session wall: it may inspect reference
    # headers/metadata but never launches a native world on its own.
    callback=GroupedValidationCallback.from_configuration(items["validation_configuration"]["path"],
        items["validation_configuration"]["sha256"],declaration,release,
        writable_path(output)/"held_out_validation")
    remaining=budget.maximum_wall_s-(time.monotonic()-started)
    if remaining<=budget.shutdown_reserve_s:
        raise RuntimeError("Grouped configuration preflight exhausted the session wall limit")
    effective=replace(budget,maximum_wall_s=remaining)
    # Preserve the exact externally declared limit alongside the effective
    # remaining limit used by the original trainer.
    result=train_grouped(declaration,catalog_path=items["catalog"]["path"],catalog_sha256=items["catalog"]["sha256"],
        domain_path=items["domain"]["path"],target_geometry_path=items["target_geometry"]["path"],
        predicate_path=items["predicate"]["path"],mask_path=items["mask"]["path"],recipe_path=items["recipe"]["path"],
        validation_descriptor=callback.descriptor,validation_callback=callback,release=release,output=output,budget=effective)
    write_json(writable_path(output)/"configuration_execution.json",dict(
        schema="task10b_phase2_grouped_configuration_execution_v1",configuration_path=str(_owned_input(path)),
        configuration_sha256=expected_sha256,declared_budget=asdict(budget),effective_budget=asdict(effective),
        declared_stopping_rules=configuration["stopping_rules"],session_wall_s=time.monotonic()-started,
        declared_transition_cap=budget.transition_cap,status=result["status"],phase2_complete=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",type=Path,required=True)
    parser.add_argument("--configuration-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    result=execute_configuration(options.configuration,options.configuration_sha256,options.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result["status"]=="bounded_stop" else 1


if __name__=="__main__":raise SystemExit(main())
