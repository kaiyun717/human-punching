"""Bounded native evaluation of immutable 96/192-attempt learned panels.

This is a callable runner, not authority to submit a job. It reuses the original
specialist actor, native controller, continuous evaluator and physical gates.
Only independent family-guard initializations are supported. Shared saved-state
restoration, training, visual acceptance and policy promotion are absent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .phase2_deadline import BoundedExecutionInterruption, ExecutionDeadline
from .phase2_domain import fingerprint
from .phase2_qualification import PanelDeclaration, TrialBinding, qualify_panel,validate_resolved_audit
from .phase2_cache import _owned_path, _preflight
from .phase2_continuation import execution_identity,load_continuation,claim_test_session,verified_run_ancestry
from .phase2_learning import (NativeSpecialistEnv, evaluate_policy, family_arguments,
    fresh_policy, policy_digest, restored_policy_parity)
from .wbc.release_baseline_probe import _copy
from .scalar_control import export_runtime_model
from .wbc.provenance import PROJECT_ROOT, writable_path, write_json, sha256_file, runtime_record

INITIALIZATION_VERSION = "task10b_independent_family_guard_zero_history_v1"
MAX_REFERENCE_BYTES = 2*1024**3
MAX_RESIDENT_BYTES = 256*1024**2


@dataclass(frozen=True)
class EvaluationLimits:
    wall_s: float
    shutdown_reserve_s: float = 120.
    maximum_worlds: int = 16
    maximum_reference_frames: int = 6001
    resident_reference_bytes: int = MAX_RESIDENT_BYTES

    def __post_init__(self):
        if (not np.isfinite(self.wall_s) or not 0 < self.shutdown_reserve_s < self.wall_s <= 14400
                or type(self.maximum_worlds) is not int or not 1 <= self.maximum_worlds <= 16
                or type(self.maximum_reference_frames) is not int or not 3 <= self.maximum_reference_frames <= 6001
                or type(self.resident_reference_bytes) is not int or not 0 < self.resident_reference_bytes <= MAX_RESIDENT_BYTES):
            raise ValueError("Finite <=four-hour wall, serialization reserve, <=16 worlds and bounded resident references required")


def owned_file(path, expected=None):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
        raise ValueError("Evaluation input must resolve to a human-punching file")
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"Frozen input hash differs: {path}")
    return path


def read_json(path, expected=None):
    path = owned_file(path, expected)
    return json.loads(path.read_text())


def load_declaration(path):
    data = read_json(path)
    return PanelDeclaration(**{**data, "bindings":tuple(TrialBinding(**row) for row in data["bindings"])})


def panel_domain_identity(declaration):
    """Freeze physical requests without a circular route/checkpoint hash."""
    return fingerprint(dict(family=declaration.family, split=declaration.split,
        command_groups=declaration.command_groups, timings=declaration.timings,
        start_mode=declaration.start_mode, predicate_sha256=declaration.predicate_sha256,
        declared_mask_sha256=declaration.declared_mask_sha256,
        requests=[{k:v for k,v in asdict(binding).items() if k != "trial_id"}
                  for binding in declaration.bindings]))


def final_test_access_identity(declaration):
    """One untouched-test access per transferred family and independent seed.

    Routes, output paths, targets and masks can be versioned before test access.
    Their hashes must not create another chance to examine the same reserved
    test groups after an observed failure.
    """
    if declaration.split != "test":
        raise ValueError("Exclusive final-test access applies only to test panels")
    return fingerprint(dict(scope="task10b_phase2_untouched_final_test_v1",
        family=declaration.family, seed=declaration.seed))


def initial_state_identity(qpos, qvel):
    """Bind the actual float32 initialization and original zero-history protocol."""
    q, v = np.asarray(qpos, np.float32), np.asarray(qvel, np.float32)
    if q.shape != (55,) or v.shape != (54,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("One finite native-float32 human position/velocity state required")
    if abs(float(np.linalg.norm(q[3:7]))-1.) > 1.e-6:
        raise ValueError("Normalized free-root quaternion required")
    return fingerprint(dict(version=INITIALIZATION_VERSION, qpos=q.tolist(), qvel=v.tolist(),
        previous_raw_action=[0.]*48, held_action_initialization="original independent sim.reset, before first actor call",
        controller_initialization="fresh native Simulation and original sim.reset; no saved-state restoration"))


def validate_route(declaration, route, *, route_sha256, supported_start_mode="family_guard"):
    """Mechanical train/validation freeze; never requests another human approval."""
    if (route_sha256 != declaration.route_sha256 or route.get("schema") != "task10b_phase2_evaluation_route_v1"
            or route.get("family") != declaration.family or route.get("selection_splits") != ["train", "validation"]
            or route.get("test_used_for_selection") is not False
            or route.get("predicate_sha256") != declaration.predicate_sha256
            or route.get("declared_mask_sha256") != declaration.declared_mask_sha256
            or panel_domain_identity(declaration) not in route.get("panel_domain_identities", [])
            or route.get("trained_timings_s") != list(declaration.trained_timings_s)
            or route.get("interpolation_timing_s") != declaration.interpolation_timing_s
            or route.get("interpolation_used_for_training") is not False
            or not isinstance(route.get("recipe_sha256"),str) or len(route["recipe_sha256"])!=64
            or any(c not in "0123456789abcdef" for c in route["recipe_sha256"])):
        raise ValueError("Route does not bind the exact split/domain/mask/timing/predicate declaration")
    if declaration.start_mode != supported_start_mode:
        raise ValueError("Shared starts require a full physical/controller saved-state restore adapter")
    if declaration.split == "validation":
        if route.get("state") not in ("validation_candidate", "frozen"):
            raise ValueError("Validation requires an explicitly declared learned candidate route")
        return {"mechanical_gate_passed":True, "final_test":False}
    if route.get("state") != "frozen":
        raise ValueError("Final test requires a frozen route selected using measured train/validation evidence")
    evidence = route.get("successful_validation", {})
    reports = {}
    for name in ("qualification", "trace_audit", "launch", "training"):
        row = evidence.get(name, {})
        if not row.get("path") or not row.get("sha256"):
            raise ValueError("Final test lacks successful measured validation evidence")
        reports[name] = read_json(row["path"], row["sha256"])
    report, audit, launch = (reports[name] for name in ("qualification", "trace_audit", "launch"))
    if (report.get("schema") != "Task10BPhase2PanelQualification/v1" or report.get("passed") is not True
            or report.get("split") != "validation" or report.get("family") != declaration.family
            or report.get("expected_attempts") != 96
            or report.get("report_identity") != fingerprint({k:v for k,v in report.items() if k != "report_identity"})
            or report.get("declaration_identity") != launch.get("declaration_identity")
            or not launch.get("slurm_job_id") or launch.get("native_execution_requested") is not True
            or audit.get("passed") is not True or audit.get("schema") != "task10b_phase2_independent_trace_audit_v1"
            or audit.get("closed_protocol_launch_audited") is not True
            or (declaration.start_mode=="shared_jab_saved_state" and audit.get("shared_saved_state_initialization_audited") is not True)
            ):
        raise ValueError("Successful actual native 96-attempt validation and independent audit are required before test access")
    selected=PanelDeclaration(**{**report["declaration"],"bindings":tuple(TrialBinding(**b) for b in report["declaration"]["bindings"])})
    if not validate_resolved_audit(audit,selected.bindings)["passed"]:
        raise ValueError("Validation audit has an unresolved/lost attempt or does not cover the full declared denominator")
    resolutions={row["trial_id"]:row for row in audit["trial_outcome_resolutions"]}
    for trial in report["trials"]:
        resolution=resolutions[trial["trial_id"]];source=trial["source_outcome"]
        if ((trial["passed"] and resolution["outcome_resolution"]!="audited_complete")
                or (source["status"] in ("timeout","generation_failure","rejected") and source.get("audit_resolution")!=resolution)):
            raise ValueError("Qualification credits or adjudication differ from independent attempt-resolution evidence")
    recomputed=qualify_panel(selected,[row["source_outcome"] for row in report["trials"]])
    training=reports["training"]
    if (recomputed["report_identity"]!=report["report_identity"]
            or selected.predicate_sha256!=declaration.predicate_sha256
            or selected.declared_mask_sha256!=declaration.declared_mask_sha256
            or selected.trained_timings_s!=declaration.trained_timings_s
            or selected.interpolation_timing_s!=declaration.interpolation_timing_s
            or selected.start_mode!=declaration.start_mode
            or launch.get("recipe_sha256")!=route["recipe_sha256"]
            or launch.get("domain_sha256")!=route.get("domain_sha256")
            or launch.get("target_geometry_sha256")!=route.get("target_geometry_sha256")
            or selected.training_receipt_sha256!=evidence["training"]["sha256"]
            or training.get("checkpoint",{}).get("sha256")!=selected.checkpoint_sha256
            or training.get("fresh_transitions")!=selected.fresh_transitions
            or training.get("weight_changed") is not True
            or any(training.get(key)!=route.get(key) for key in ("recipe_sha256","domain_sha256","target_geometry_sha256"))
            or selected.parent_checkpoint_sha256!=declaration.parent_checkpoint_sha256):
        raise ValueError("Successful validation does not prove the same selected recipe, predicate, mask, timings and learned lineage")
    audited_ids=set(resolutions)
    if audited_ids!={b.trial_id for b in selected.bindings}:
        raise ValueError("Successful validation independent audit covers different trials")
    run = owned_file(evidence["launch"]["path"]).parent
    if Path(audit.get("run", "")).resolve() != run:
        raise ValueError("Validation qualification, native launch and independent audit refer to different runs")
    ancestry=(verified_run_ancestry(run) if (run/"execution_identity.json").exists() else
        {"run_directories":[str(run)]})
    for panel in audit["panels"]:
        raw = owned_file(Path(panel["path"])/"raw_trace.npz", panel["raw_sha256"])
        if not any(raw.is_relative_to(Path(root)) for root in ancestry["run_directories"]):
            raise ValueError("Validation audit trace escapes its bound native run")
    return {"mechanical_gate_passed":True, "final_test":True, "validation_evidence":evidence,
        "new_human_approval_required":False}


def validate_training_identity(declaration, training, checkpoint, parent_sha256, *, route):
    if (parent_sha256 != declaration.parent_checkpoint_sha256
            or checkpoint.get("schema") != "task10b_phase2_original_ppo_v1"
            or checkpoint.get("observations") != 369 or checkpoint.get("actions") != 48
            or checkpoint.get("fresh_transitions") != declaration.fresh_transitions
            or training.get("fresh_transitions") != declaration.fresh_transitions
            or training.get("weight_changed") is not True
            or training.get("status")!="bounded_stop" or training.get("exception") is not None
            or training.get("hard_faults")
            or checkpoint.get("actor_parameters_sha256")!=training.get("final_policy_digest")
            or training.get("checkpoint",{}).get("sha256") != declaration.checkpoint_sha256
            or training.get("reload_parity",{}).get("parameters_equal") is not True
            or training.get("reload_parity",{}).get("actions_equal") is not True
            or any(training.get(key)!=route.get(key) for key in ("recipe_sha256","domain_sha256","target_geometry_sha256"))
            or training.get("trained_timings_s")!=list(declaration.trained_timings_s)
            or training.get("interpolation_used_for_training") is not False):
        raise ValueError("Learned original-PPO checkpoint/training transition/parity identity differs")
    for owner in (training, checkpoint):
        ancestry = owner.get("ancestry", {})
        if (ancestry.get("sha256") != parent_sha256 or ancestry.get("seed") != declaration.seed
                or ancestry.get("fresh_optimizer") is not True or ancestry.get("fresh_simulation") is not True
                or ancestry.get("independent_scratch_replication") is not False):
            raise ValueError("Independent transferred fine-tuning ancestry is missing or mismatched")


def learned_parameter_changes(policy, parent_checkpoint):
    """Independently compare restored tensor bytes with accepted ancestry."""
    parent=torch.load(parent_checkpoint,map_location="cpu",weights_only=True)["policy"]
    current=policy.state_dict()
    if set(parent)!=set(current):
        raise ValueError("Learned and accepted policy tensor schemas differ")
    changed=[]
    for name,tensor in current.items():
        if tensor.shape!=parent[name].shape or tensor.dtype!=parent[name].dtype:
            raise ValueError("Learned and accepted policy tensor dimensions differ")
        if not torch.equal(tensor.detach().cpu(),parent[name]):changed.append(name)
    if not changed:
        raise ValueError("Learned route still contains unchanged accepted policy tensors")
    return dict(changed_tensor_names=changed,changed_tensor_count=len(changed),
        parent_checkpoint_sha256=sha256_file(parent_checkpoint),independent_byte_comparison=True)


def validate_trial_row(binding, declaration, row):
    if (row.get("family") != declaration.family or row.get("reference_sha256") != binding.reference_sha256
            or row.get("requested_ttc_s") != binding.ttc_s or tuple(row.get("target_w",())) != binding.target_w
            or row.get("command_id") != binding.command_id or row.get("split") != declaration.split
            or row.get("original_replica_id") != binding.original_replica_id
            or not isinstance(row.get("kind"),str) or row["kind"] in ("", "original_control")
            or row.get("preparation_complete_tick",0) != 0
            or row.get("start_mode","family_guard") != "family_guard"):
        raise ValueError("Exact family/group/split/replica/reference/target/TTC family-guard binding differs")
    if (type(row.get("launch_tick")) is not int or row["launch_tick"] % 4
            or type(row.get("outward_end_tick")) is not int or not 0 <= row["launch_tick"] < row["outward_end_tick"]
            or not isinstance(row.get("candidate_id"),str) or not row["candidate_id"]
            or Path(row["candidate_id"]).name != row["candidate_id"]
            or row["candidate_id"] in (".","..")):
        raise ValueError("Explicit safe reference identity and actor-boundary events required")
    for key in ("duration_s","episode_duration_s","phase_denominator_s"):
        if not isinstance(row.get(key),(int,float)) or not np.isfinite(row[key]) or row[key] <= 0:
            raise ValueError("Predeclared finite full-cycle horizons required")
    if (abs(row["episode_duration_s"]-row["duration_s"]-1.) > 1.e-8
            or row["episode_duration_s"] != row["phase_denominator_s"]
            or row["outward_end_tick"]*.005 > row["duration_s"]):
        raise ValueError("Full reference plus one-second hold and outward endpoint must be retained")


def validate_negative_reference_identity(row):
    """A failed generation has an immutable receipt, never an invented NPZ."""
    kind=row.get("reference_identity_kind","reference_content")
    if kind in ("reference_content","npz_sha256"):return False
    if (kind!="generation_failure_receipt" or row.get("kinematic_admission") is not False
            or row.get("generation_failure_sha256")!=row.get("reference_sha256")
            or any(key in row for key in ("reference_path","legacy_runtime_path","legacy_runtime_sha256"))
            or not isinstance(row.get("rejection_reasons"),list) or not row["rejection_reasons"]
            or not all(isinstance(reason,str) and reason.strip() for reason in row["rejection_reasons"])):
        raise ValueError("Failed generation must bind an explicit immutable receipt with no reference-file claim")
    owned_file(row["generation_failure_path"],row["generation_failure_sha256"])
    return True


def validate_geometry_bindings(declaration,manifest,domain,targets,entries,rows):
    """Bind target coordinates to the canonical measured snapshot and placement."""
    if domain.get("schema")!="Task10BPhase2Domain/v1" or targets.get("schema")!="Task10BPhase2RegisteredTargets/v1":
        raise ValueError("Canonical frozen Phase2 domain and registered target artifacts required")
    groups={r["snapshot"]["command_id"]:r for r in domain["selected_command_groups"]}
    candidates={r["candidate_identity"]:r for r in targets["geometry_candidates"]}
    if len(candidates)!=len(targets["geometry_candidates"]):raise ValueError("Duplicate target candidate identity")
    selected={}
    for binding in declaration.bindings:
        entry=entries[binding.trial_id];identity=entry["target_candidate_identity"]
        target=candidates[identity];group=groups[binding.command_id];snapshot=group["snapshot"]
        if (fingerprint({k:v for k,v in target.items() if k!="candidate_identity"})!=identity
                or target["source_domain_sha256"]!=manifest["domain_sha256"]
                or target["command_id"]!=binding.command_id or target["split"]!=declaration.split
                or target["original_replica_id"]!=binding.original_replica_id
                or group["split"]!=declaration.split or snapshot["replica_id"]!=binding.original_replica_id
                or target["snapshot_slot_id"]!=snapshot["slot_id"] or target["payload_sha256"]!=snapshot["payload_sha256"]
                or tuple(target["head_target_w"])!=binding.target_w
                or rows[entry["candidate_id"]].get("target_candidate_identity")!=identity):
            raise ValueError("Target/reference no longer matches its exact canonical snapshot, split and placement")
        if selected.setdefault(binding.command_id,identity)!=identity:
            raise ValueError("Initial panel schema fixes one placement per command group across all starts/timings")
    return dict(one_placement_per_group=True,selected_target_candidate_identities=selected)


def declared_mask_decisions(declaration,mask):
    if mask.get("schema")!="task10b_phase2_declared_evaluation_mask_v1":
        raise ValueError("An explicit pre-outcome physical-request mask is required")
    records={}
    for row in mask["requests"]:
        binding={k:row[k] for k in TrialBinding.__dataclass_fields__ if k!="trial_id"}
        key=fingerprint(dict(family=row["family"],**binding))
        if (key in records or type(row.get("admitted")) is not bool
                or (row.get("admitted") is False and
                    (not isinstance(row.get("reason"),str) or not row["reason"].strip()))):
            raise ValueError("Unique mask requests require boolean admission and explicit rejection reasons")
        records[key]=row
    result={}
    for binding in declaration.bindings:
        physical={k:v for k,v in asdict(binding).items() if k!="trial_id"}
        row=records[fingerprint(dict(family=declaration.family,**physical))]
        result[binding.trial_id]=dict(admitted=row["admitted"],reason=row.get("reason"))
    return result


def batch_requests(declaration, entries, rows, maximum_worlds=16):
    """Same exact reference per batch: every trial has the same full horizon."""
    if type(maximum_worlds) is not int or not 1 <= maximum_worlds <= 16:
        raise ValueError("At most sixteen worlds per native batch")
    grouped = {}
    for binding in declaration.bindings:
        entry = entries[binding.trial_id]; row = rows[entry["candidate_id"]]
        grouped.setdefault(row["candidate_id"], []).append(binding)
    result = []
    for candidate, bindings in grouped.items():
        for begin in range(0,len(bindings),maximum_worlds):
            result.append((rows[candidate],tuple(bindings[begin:begin+maximum_worlds])))
    return result


class DeadlineNativeSpecialistEnv(NativeSpecialistEnv):
    """Stop only before the next complete actor interval, inside evaluator catch."""
    def __init__(self,*args,deadline,**kwargs):
        self.execution_deadline = deadline
        super().__init__(*args,**kwargs)

    def launch_gate(self):
        try:
            self.execution_deadline.checkpoint("evaluation_actor_boundary",progress={
                "physics_steps":self.physics_steps_executed,"actor_updates":self.actor_updates_executed})
        except BoundedExecutionInterruption as error:
            self.deadline_interruption_receipt=error.receipt
            raise
        return super().launch_gate()


def launch_requests(declaration, bindings, row, target_candidates):
    """The exact cached command and measured snapshot ancestry; no nearest lookup."""
    result=[]
    for binding in bindings:
        target=target_candidates[row["target_candidate_identity"]]
        result.append(dict(trial_id=binding.trial_id,request_identity=binding.identity,
            family=row["family"],requested_ttc_s=row["requested_ttc_s"],style=row["candidate_id"],
            candidate_id=row["candidate_id"],reference_sha256=row["reference_sha256"],
            target_candidate_identity=row["target_candidate_identity"],snapshot_payload_sha256=target["payload_sha256"],
            checkpoint_sha256=declaration.checkpoint_sha256,snapshot_slot_id=target["snapshot_slot_id"],
            placement=target["placement"],head_point_velocity_w=target["head_point_velocity_w"],
            target_w=list(row["target_w"])))
    return result


class FrozenSnapshotInput:
    """Re-read the exact immutable snapshot-target input at the launch boundary.

    This is a fixed-target experiment input, not a live G1 actor or moving-head
    predictor. The caller supplies one exact declared placement per group.
    """
    def __init__(self, requests):
        self._encoded=tuple(json.dumps(row,sort_keys=True,allow_nan=False) for row in requests)

    def __call__(self, episode_tick):
        if type(episode_tick) is not int or episode_tick < 0 or episode_tick % 4:
            raise ValueError("Frozen launch input requires an actual nonnegative 20ms actor boundary")
        result=[]
        for encoded in self._encoded:
            request=json.loads(encoded)
            result.append({key:request[key] for key in ("trial_id","request_identity","target_candidate_identity",
                "snapshot_payload_sha256","target_w","checkpoint_sha256","snapshot_slot_id",
                "placement","head_point_velocity_w")}|dict(capture_episode_tick=int(episode_tick),
                    input_episode_time_s=int(episode_tick)*.005,
                    source_kind="fixed_canonical_snapshot",head_geometry="head_collision",
                    head_velocity_method="jacobian_qvel",no_prediction=True))
        return result


class ClosedProtocolNativeSpecialistEnv(DeadlineNativeSpecialistEnv):
    """Measured continuation admission for this exact isolated trial protocol.

    No state, reference, previous action or controller history is replaced. A
    successful gate qualifies only continuation from the declared start/history
    through this exact preselected prefix, never arbitrary current human states.
    """
    def __init__(self,*args,expected_qpos,expected_qvel,requests,target_input,**kwargs):
        from .phase2_launch import ClosedProtocolLaunchGuard
        super().__init__(*args,**kwargs)
        self._requests=json.loads(json.dumps(requests,allow_nan=False))
        self._target_input=target_input
        self._row_identities=tuple(fingerprint(row) for row in self.library.rows)
        self._cache_versions=self.cache_versions()
        self.protocol=ClosedProtocolLaunchGuard(expected_qpos,expected_qvel,self._requests,
            _copy(self.library.launch_ticks)[np.zeros(self.n,dtype=int)],
            _copy(self.library.durations)[np.zeros(self.n,dtype=int)],
            expected_reference_ids=np.zeros(self.n,dtype=int))
        self.closed_launch_records=[]
        self.closed_boundary_snapshots=[]
        self._bound_policy=None

    @staticmethod
    def policy_versions(policy):
        return tuple((key,id(value),value.data_ptr(),value._version)
            for key,value in (*policy.named_parameters(),*policy.named_buffers()))

    def bind_policy(self,policy):
        """Bind inference tensor identities before this independent trial starts."""
        if self._bound_policy is not None or self.physics_steps_executed or self.actor_updates_executed:
            raise ValueError("Bind the immutable evaluated policy once before native execution")
        self._bound_policy=policy
        self._policy_versions=self.policy_versions(policy)

    def current_requests(self):
        result=[]
        for world,identifier in enumerate(_copy(self.ids_ref)):
            row=self.library.rows[int(identifier)]
            result.append(self._requests[world]|dict(family=row["family"],
                requested_ttc_s=row["requested_ttc_s"],style=row["candidate_id"],candidate_id=row["candidate_id"],
                reference_sha256=row["reference_sha256"],target_w=list(row["target_w"]),
                target_candidate_identity=row["target_candidate_identity"]))
        return result

    def cache_versions(self):
        tensors=[(f"{i}:{key}",value) for i,row in enumerate(self.library.arrays) for key,value in sorted(row.items())]
        tensors.extend((key,getattr(self.library,key)) for key in ("durations","launch_ticks","outward_ticks"))
        return tuple((key,id(value),value.data_ptr(),value._version) for key,value in tensors)

    def protocol_snapshot(self):
        data=self.sim.data
        state={key:_copy(getattr(data,key)) for key in
            ("qpos","qvel","ctrl","qacc_warmstart","qfrc_applied","xfrc_applied","act","time")}
        state.update(previous_raw_action=_copy(self.last_action),ticks=_copy(self.ticks),
            reference_ids=_copy(self.ids_ref),reset_counts=_copy(self.reset_counts),support=_copy(self.last_support),
            phase_denominators_s=_copy(self.duration),physics_steps=self.physics_steps_executed,
            actor_updates=self.actor_updates_executed,rejected=_copy(self.rejected),launched=_copy(self.launched))
        return state

    def reset(self,*args,**kwargs):
        if kwargs.get("random_phase",True) or kwargs.get("perturb",True):
            raise ValueError("Closed panel requires explicit frozen initial state and no random-phase/reset noise")
        result=super().reset(*args,**kwargs)
        self.protocol.initialize(self.protocol_snapshot())
        return result

    def advance_substep(self,action,*,record=None,stop_on_fall=False):
        self.protocol.before_step(self.protocol_snapshot(),_copy(action))
        super().advance_substep(action,record=record,stop_on_fall=stop_on_fall)
        self.protocol.after_step(self.protocol_snapshot(),_copy(action))

    def launch_gate(self):
        try:
            self.execution_deadline.checkpoint("evaluation_actor_boundary",progress={
                "physics_steps":self.physics_steps_executed,"actor_updates":self.actor_updates_executed})
        except BoundedExecutionInterruption as error:
            self.deadline_interruption_receipt=error.receipt
            raise
        planning_started=time.perf_counter()
        before=self.protocol_snapshot()
        requested=self.protocol.actor_boundary(before)
        lookup_started=time.perf_counter()
        if (self.cache_versions()!=self._cache_versions
                or tuple(fingerprint(row) for row in self.library.rows)!=self._row_identities):
            raise ValueError("Exact cached reference changed during uninterrupted execution")
        if (self._bound_policy is None
                or self.policy_versions(self._bound_policy)!=self._policy_versions):
            raise ValueError("The evaluated checkpoint tensors changed during uninterrupted execution")
        current_requests=self.current_requests()
        self.protocol.assert_requests(current_requests)
        lookup_wall=time.perf_counter()-lookup_started
        ready=self.readiness()
        if np.any(requested):
            callback_started=time.perf_counter()
            receipts=self._target_input(self.physics_steps_executed)
            callback_wall=time.perf_counter()-callback_started
            after=self.protocol_snapshot()
            decisions=self.protocol.finalize(before,after,receipts,_copy(ready),current_requests=current_requests,
                planning_started_perf_counter=planning_started,callback_wall_s=callback_wall,cache_lookup_wall_s=lookup_wall)
            snapshot_index=len(self.closed_boundary_snapshots)
            # The snapshots already own their native copies. Retaining pointers
            # and serializing them after the batch does not enter the measured
            # final-planning interval reported by finalize().
            self.closed_boundary_snapshots.append(dict(episode_tick=self.physics_steps_executed,
                before=before,after=after))
            allowed=np.zeros(self.n,dtype=bool)
            for world,decision in enumerate(decisions):
                if decision is not None:
                    self.closed_launch_records.append(dict(world=world,boundary_snapshot_index=snapshot_index,**decision))
                    allowed[world]=decision["accepted"]
            mask=torch.as_tensor(requested,dtype=torch.bool,device=self.device)
            accepted=torch.as_tensor(allowed,dtype=torch.bool,device=self.device)
            self.launched|=mask&accepted
            self.rejected|=mask&~accepted
            return mask,accepted
        return torch.as_tensor(requested,dtype=torch.bool,device=self.device),ready


def write_launch_boundaries(folder,snapshots):
    """Retain lossless batch snapshots for an independent launch digest audit."""
    from .phase2_launch import snapshot_identity
    arrays={};records=[]
    for index,snapshot in enumerate(snapshots):
        record=dict(episode_tick=snapshot["episode_tick"])
        for side in ("before","after"):
            state=snapshot[side];array_fields={};scalar_fields={}
            for key,value in state.items():
                if isinstance(value,np.ndarray):
                    name=f"boundary_{index:04d}_{side}__{key}"
                    arrays[name]=value
                    array_fields[key]=name
                else:scalar_fields[key]=value
            record[side]=dict(array_fields=array_fields,scalar_fields=scalar_fields,
                snapshot_sha256=snapshot_identity(state))
        records.append(record)
    folder=Path(folder)
    path=folder/"closed_boundary_snapshots.npz"
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
    receipt=dict(schema="task10b_phase2_lossless_launch_boundaries_v1",path=str(path),
        sha256=sha256_file(path),snapshots=records,
        serialization_timing="After the native batch; excluded from final-planning gate wall measurement")
    write_json(folder/"closed_boundary_snapshots.json",receipt)
    return receipt


def outcome_rows(declaration, bindings, panel, batch_path):
    """Keep physical/partial evidence separate from still-pending human/audit gates."""
    reports = {row["trial_id"]:row for row in panel.get("trials",[])}
    interrupted = panel.get("exception",{}).get("type") == "BoundedExecutionInterruption" if panel.get("exception") else False
    rows = []
    for world,binding in enumerate(bindings):
        result = reports.get(binding.trial_id); faults = {name for fault in panel.get("hard_faults",[])
            for name,worlds in fault["faults"].items() if world in worlds}
        row = dict(trial_id=binding.trial_id,request_identity=binding.identity,
            checkpoint_sha256=declaration.checkpoint_sha256,predicate_sha256=declaration.predicate_sha256,
            raw_trace_path=str(Path(batch_path)/"raw_trace.npz"),world=world,
            status="timeout" if interrupted and result is None else "executed",
            physical_execution_attempted=True,
            execution_complete=result is not None, internal_trace_audit_pass=panel.get("audit",{}).get("passed") is True,
            trace_audit_pass=None,visual_review="pending",single_outward_stroke_pass=None,movement_exploit=None,
            numeric_complete_cycle=False, independent_audit_pending=True,
            hard_flags_scope="Observed retained native intervals; incomplete attempts do not establish full-horizon absence of violations",
            batch_actual_physics_steps=panel.get("actual_physics_steps"),batch_actual_policy_updates=panel.get("actual_policy_updates"),
            fallen="fall" in faults, nonfinite="nonfinite" in faults,
            root_assistance=bool({"root_assistance","external_assistance"}&faults),effort_cap_violation="effort_cap" in faults)
        if result is None:
            row.update(disposition="incomplete_timeout" if interrupted else "incomplete_native_failure",
                reason=panel.get("exception") or "incomplete_or_failed_native_audit",functional_recovery_pass=False)
        else:
            functional = result["functional"]; launch = functional.get("launch_time_s")
            first,other = result["first_active_contact_s"],result["first_inactive_contact_s"]
            contacts = [x for x in (first,other) if x is not None]
            row.update(disposition="completed_native_attempt", numeric_complete_cycle=result["numeric_complete_cycle"],
                readiness_pass=launch is not None and not functional.get("launch_rejected",False),
                immutable_launch_pass=launch is not None and result.get("reference_commitment")==binding.reference_sha256
                    and tuple(result.get("target_at_launch") or ())==binding.target_w,
                continuity_history_pass=panel.get("audit",{}).get("passed") is True,
                functional_recovery_pass=functional["functional_complete"],
                correct_hand=first is not None and (other is None or first<other),
                joint_limit_excursion_rad=result["maximum_joint_limit_excursion_rad"],
                full_active_glove_rms_m=result["full_active_glove_rms_m"],outward_active_glove_rms_m=result["outward_active_glove_rms_m"],
                request_s=0.,preparation_complete_s=0.,readiness_s=functional.get("readiness_time_s"),
                launch_boundary_readiness_s=launch,final_planning_s=launch,launch_s=launch,
                outward_complete_s=functional.get("outward_completion_time_s"),recovery_s=functional.get("recovery_time_s"),
                first_active_contact_s=first,first_inactive_contact_s=other,first_any_contact_s=min(contacts) if contacts else None,
                functional_diagnostics=functional,source_native_result=result)
        rows.append(row)
    return rows


def preflight(declaration, manifest, catalog, *, limits, mask_decisions):
    """Check every binding/header before native construction, loading starts only."""
    if (manifest.get("schema") != "task10b_phase2_expanded_manifest_v1"
            or manifest.get("declaration_identity") != declaration.identity
            or manifest.get("initialization_version") != INITIALIZATION_VERSION):
        raise ValueError("Expanded manifest does not bind this declaration and initialization protocol")
    entries = {entry["trial_id"]:entry for entry in manifest["trials"]}
    if len(entries)!=len(manifest["trials"]) or set(entries)!={b.trial_id for b in declaration.bindings}:
        raise ValueError("Manifest requires exactly all declared trial IDs")
    rows = {row["candidate_id"]:row for row in catalog["cells"]}
    if len(rows)!=len(catalog["cells"]):raise ValueError("Duplicate reference catalog identities")
    headers, total = {}, 0
    starts, start_files = {}, {}
    for binding in declaration.bindings:
        entry=entries[binding.trial_id];row=rows[entry["candidate_id"]]
        validate_trial_row(binding,declaration,row)
        failed_generation=validate_negative_reference_identity(row)
        if failed_generation and mask_decisions[binding.trial_id]["admitted"]:
            raise ValueError("A failed reference generation cannot be admitted for native execution")
        if row.get("kinematic_admission") is True and mask_decisions[binding.trial_id]["admitted"]:
            for name in ("reference","legacy_runtime"):
                path=_owned_path(row[name+"_path"]);expected=row[name+"_sha256"]
                if path not in headers:
                    headers[path]=_preflight(path,expected,MAX_REFERENCE_BYTES)
                    total+=headers[path].uncompressed_array_bytes
                    if total>MAX_REFERENCE_BYTES:raise ValueError("Total unique reference arrays exceed2GiB")
                elif headers[path].sha256!=expected:raise ValueError("Conflicting reference identity")
                schemas={a.name:a for a in headers[path].arrays}
                n=schemas.get("qpos").shape[0] if "qpos" in schemas else 0
                if not 3<=n<=limits.maximum_reference_frames or schemas["qpos"].shape!=(n,55):
                    raise ValueError("Reference frame/dimension bound differs")
                # One reference per native batch. Include conservative host/FK
                # expansion and tensor-copy allowance before any payload load.
                estimate=headers[path].uncompressed_array_bytes*3+n*4096
                if estimate>limits.resident_reference_bytes:
                    raise ValueError("Reference load/derived-data allowance exceeds resident chunk budget")
                with np.load(path,allow_pickle=False) as archive:
                    clock=np.asarray(archive["t"] if "t" in archive else archive["time_s"],float)
                    if (clock.shape!=(n,) or clock[0]!=0 or not np.allclose(np.diff(clock),.005,atol=1e-12,rtol=0)
                            or abs(clock[-1]-row["duration_s"])>1.e-8
                            or not np.array_equal(archive["head_center"],binding.target_w)
                            or int(archive["active_hand_index"])!=(0 if declaration.family in ("jab","left_hook") else 1)):
                        raise ValueError("Bound reference grid/target/active-hand metadata differs")
        bank=owned_file(entry["start_bank_path"],entry["start_bank_sha256"])
        if bank not in start_files:
            header=_preflight(_owned_path(bank),entry["start_bank_sha256"],8*1024**2)
            with np.load(bank,allow_pickle=False) as archive:
                q,v=archive["qpos"].copy(),archive["qvel"].copy()
            if q.ndim!=2 or q.shape[1]!=55 or v.shape!=(len(q),54) or len(q)>64:
                raise ValueError("At most64 complete declared initial states per bounded bank")
            start_files[bank]=(q,v)
        q,v=start_files[bank];index=entry["start_row"]
        if type(index) is not int or not 0<=index<len(q):raise ValueError("Invalid declared start row")
        if initial_state_identity(q[index],v[index])!=binding.start_state_sha256:
            raise ValueError("Declared actual native start/history identity differs")
        starts[binding.trial_id]=(q[index].astype(np.float32),v[index].astype(np.float32))
    return entries,rows,starts,dict(unique_reference_uncompressed_bytes=total,
        unique_reference_archives=len(headers),resident_reference_limit_bytes=limits.resident_reference_bytes,
        reference_metadata_payloads_loaded=True,full_reference_state_arrays_loaded=False,
        initialization_version=INITIALIZATION_VERSION)


def validate_declared_starts(declaration,domain,starts,paths):
    with np.load(paths["reference"],allow_pickle=False) as source:
        nominal_q=source["qpos"][0].copy();nominal_v=source["qvel"][0].copy()
    metadata=read_json(paths["body_metadata"]);qa=np.asarray(metadata["qpos_addresses"],int)
    limits=np.asarray(metadata["joint_limits_rad"],float)
    perturbations=domain["starts"][declaration.split]
    delta_q=np.asarray(perturbations["joint_position_delta_rad"],float)
    delta_v=np.asarray(perturbations["generalized_velocity_delta"],float)
    if (delta_q.shape!=(4,48) or delta_v.shape!=(4,54) or not np.isfinite(delta_q).all()
            or not np.isfinite(delta_v).all() or np.any(delta_v[:,:6])
            or abs(delta_q).max()>.015+1e-12 or abs(delta_v).max()>.06+1e-12):
        raise ValueError("Initial isolated panel uses the frozen bounded four joint-only D3 perturbations")
    for binding in declaration.bindings:
        q=nominal_q.copy();q[qa]=np.clip(q[qa]+delta_q[binding.start_index],limits[:,0],limits[:,1])
        v=nominal_v+delta_v[binding.start_index]
        actual_q,actual_v=starts[binding.trial_id]
        if not np.array_equal(actual_q,q.astype(np.float32)) or not np.array_equal(actual_v,v.astype(np.float32)):
            raise ValueError("Initial state differs from accepted family guard plus frozen clipped start perturbation")


def audit_initialization(trace_path,qpos,qvel):
    with np.load(trace_path,allow_pickle=False) as raw:
        checks=dict(qpos=np.array_equal(raw["qpos"][0],qpos),qvel=np.array_equal(raw["qvel"][0],qvel),
            one_independent_reset=bool(np.all(raw["reset_count"][0]==1)),
            previous_raw_action_zero=bool("policy_previous_raw_action" in raw and not np.any(raw["policy_previous_raw_action"][0])),
            reference_tick_zero=bool("policy_reference_tick" in raw and not np.any(raw["policy_reference_tick"][0])),
            initial_ctrl_zero=bool(not np.any(raw["ctrl"][0])))
    return dict(passed=all(checks.values()),checks=checks,version=INITIALIZATION_VERSION,
        complete_saved_state_restoration=False)


def execute(declaration_path,manifest_path,route_path,checkpoint_path,training_path,predicate_path,mask_path,
            output,*,release,limits:EvaluationLimits,protocol=None,continuation=None):
    """Execute one counted allocation; no submission, retry, training or promotion."""
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Expanded native evaluation requires a counted Slurm CUDA allocation")
    started=time.monotonic();deadline=ExecutionDeadline(limits.wall_s,
        serialization_reserve_s=limits.shutdown_reserve_s,session_started_monotonic=started)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    declaration=load_declaration(declaration_path)
    ledger={b.trial_id:dict(trial_id=b.trial_id,status="unattempted",reason="not_reached",request_identity=b.identity,
        physical_execution_attempted=False)
            for b in declaration.bindings}
    write_json(folder/"declaration.json",asdict(declaration))
    exception=None;panels=[];claim=None;claim_session=None;preflight_done=False;native_attempts_started=0;scope=None
    identity=None;continuation_context=None;eligible={b.trial_id for b in declaration.bindings};logical_origin=str(folder)
    try:
        with deadline.signal_handlers():
            identity=execution_identity(declaration,dict(declaration=declaration_path,manifest=manifest_path,route=route_path,
                checkpoint=checkpoint_path,training=training_path,predicate=predicate_path,mask=mask_path),release=release,
                protocol_version=protocol.version if protocol is not None else INITIALIZATION_VERSION,maximum_worlds=limits.maximum_worlds)
            write_json(folder/"execution_identity.json",identity)
            if continuation is not None:
                continuation_context=load_continuation(continuation,identity,declaration)
                ledger={row["trial_id"]:row for row in continuation_context["outcomes"]}
                eligible=set(continuation_context["trial_ids"]);logical_origin=continuation_context["logical_origin_run"]
                write_json(folder/"continuation.json",{key:value for key,value in continuation_context.items() if key!="outcomes"})
                if declaration.split=="test":
                    reservation=claim_test_session(folder,declaration,identity,continuation_context)
                    claim=Path(reservation["path"]);claim_session=Path(reservation["session_path"])
            write_json(folder/"ledger_initial.json",list(ledger.values()))
            route_path=owned_file(route_path,declaration.route_sha256)
            route=read_json(route_path);gate=validate_route(declaration,route,route_sha256=sha256_file(route_path),
                supported_start_mode=protocol.start_mode if protocol is not None else "family_guard")
            training_path=owned_file(training_path,declaration.training_receipt_sha256)
            checkpoint_path=owned_file(checkpoint_path,declaration.checkpoint_sha256)
            predicate_path=owned_file(predicate_path,declaration.predicate_sha256)
            mask=read_json(mask_path,declaration.declared_mask_sha256)
            decisions=declared_mask_decisions(declaration,mask)
            args,paths=family_arguments(release,declaration.family)
            saved=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
            training_receipt=read_json(training_path)
            validate_training_identity(declaration,training_receipt,saved,sha256_file(paths["checkpoint"]),route=route)
            del saved
            runtime=runtime_record()
            if not runtime["declared_pins_match"]:raise ValueError("Phase2 runtime pins differ")
            manifest=read_json(manifest_path);catalog_path=owned_file(manifest["catalog_path"],manifest["catalog_sha256"])
            catalog=read_json(catalog_path)
            if (manifest.get("recipe_sha256")!=route["recipe_sha256"]
                    or catalog.get("schema")!="task10b_phase2_reference_catalog_v1"):
                raise ValueError("Frozen reference catalog/manifest recipe differs from the selected route")
            for name in ("domain","target_geometry"):
                if manifest[name+"_sha256"]!=route.get(name+"_sha256"):
                    raise ValueError("Route/manifest canonical domain or target provenance differs")
            domain=read_json(manifest["domain_path"],manifest["domain_sha256"])
            targets=read_json(manifest["target_geometry_path"],manifest["target_geometry_sha256"])
            if protocol is not None:
                entries,rows,starts,scope=protocol.preflight(declaration,manifest,catalog,limits=limits,
                    mask_decisions=decisions,release=release,route=route)
            else:entries,rows,starts,scope=preflight(declaration,manifest,catalog,limits=limits,mask_decisions=decisions)
            if any(rows[entry["candidate_id"]].get("recipe_sha256")!=route["recipe_sha256"] for entry in entries.values()):
                raise ValueError("Expanded reference row recipe differs from the selected route")
            scope.update(validate_geometry_bindings(declaration,manifest,domain,targets,entries,rows))
            if protocol is not None:protocol.validate_starts(declaration,domain,starts)
            else:validate_declared_starts(declaration,domain,starts,paths)
            preflight_done=True
            write_json(folder/"preflight.json",dict(gate=gate,reference_scope=scope,limits=asdict(limits),
                starts_bound=len(starts),wall_s=time.monotonic()-started))
            deadline.checkpoint("after_complete_preflight")
            training_out=folder/"training";training_out.mkdir()
            copied=training_out/"checkpoint_evaluated.pt"
            with checkpoint_path.open("rb") as source,copied.open("xb") as destination:shutil.copyfileobj(source,destination)
            if sha256_file(copied)!=declaration.checkpoint_sha256:raise ValueError("Copied learned checkpoint differs")
            write_json(folder/"training_receipt.json",read_json(training_path))
            write_json(folder/"catalog.json",catalog)
            write_json(folder/"launch.json",dict(schema="task10b_phase2_expanded_native_launch_v1",family=declaration.family,
                seed=declaration.seed,split=declaration.split,declaration_identity=declaration.identity,
                slurm_job_id=os.environ["SLURM_JOB_ID"],runtime=runtime,native_execution_requested=True,
                catalog=str(folder/"catalog.json"),catalog_sha256=sha256_file(folder/"catalog.json"),
                release_inputs={key:dict(path=str(path),sha256=sha256_file(path)) for key,path in paths.items()},
                evaluated_checkpoint=dict(path=str(copied),sha256=declaration.checkpoint_sha256),
                predicate_sha256=declaration.predicate_sha256,route_path=str(route_path),route_sha256=declaration.route_sha256,
                recipe_sha256=route["recipe_sha256"],domain_sha256=manifest["domain_sha256"],
                domain_path=str(owned_file(manifest["domain_path"],manifest["domain_sha256"])),
                target_geometry_path=str(owned_file(manifest["target_geometry_path"],manifest["target_geometry_sha256"])),
                target_geometry_sha256=manifest["target_geometry_sha256"],
                declaration_path=str(folder/"declaration.json"),declaration_sha256=sha256_file(folder/"declaration.json"),
                execution_identity=identity["identity"],logical_origin_run=logical_origin,
                continuation=None if continuation_context is None else {key:value for key,value in continuation_context.items()
                    if key not in ("outcomes","trial_ids","inherited_attempted_ids")},
                sources={str(Path(__file__).resolve()):sha256_file(__file__)},limits=asdict(limits),
                no_training=True,initialization_version=protocol.version if protocol is not None else INITIALIZATION_VERSION,g1_actor_executed=False,
                full_saved_state_restore_supported=protocol is not None,automatic_test_retry=False,
                start_mode=declaration.start_mode,protocol_metadata=protocol.metadata if protocol is not None else {},
                closed_protocol_launch_supported=True,arbitrary_live_state_planner_supported=False,
                launch_target_input="Exact canonical fixed snapshot target re-read at actual launch; no G1 execution or prediction"))
            torch.set_num_threads(1);torch.manual_seed(declaration.seed);np.random.seed(declaration.seed)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            from .wbc.telemetry.cycle_qualification import native_stream_scope
            target_candidates={target["candidate_identity"]:target for target in targets["geometry_candidates"]}
            policy=None
            for number,(row,bindings) in enumerate(batch_requests(declaration,entries,rows,limits.maximum_worlds)):
                bindings=tuple(binding for binding in bindings if binding.trial_id in eligible)
                if not bindings:continue
                admitted=[]
                for b in bindings:
                    if row.get("kinematic_admission") is True and decisions[b.trial_id]["admitted"]:
                        admitted.append(b)
                    else:
                        ledger[b.trial_id].update(status="generation_failure" if row.get("reference_identity_kind")=="generation_failure_receipt" else "rejected",
                            reason=decisions[b.trial_id].get("reason") or row.get("rejection_reasons",["reference_not_admitted"]),
                            reference_identity_kind=row.get("reference_identity_kind","reference_content"),
                            physical_execution_attempted=False)
                bindings=tuple(admitted)
                if not bindings:
                    write_json(folder/f"ledger_{number:04d}.json",list(ledger.values()));continue
                deadline.checkpoint("before_native_batch",progress={"completed_batches":len(panels)})
                # Preflight failures and wholly rejected panels consume no native
                # trial. Reserve a unique claim immediately before first native
                # initialization; no automatic replay of partial/completed runs.
                if declaration.split=="test" and claim is None:
                    reservation=claim_test_session(folder,declaration,identity)
                    claim=Path(reservation["path"]);claim_session=Path(reservation["session_path"])
                batch_folder=folder/f"batch_{number:04d}"
                q=np.stack([starts[b.trial_id][0] for b in bindings]);v=np.stack([starts[b.trial_id][1] for b in bindings])
                requests=launch_requests(declaration,bindings,row,target_candidates)
                env=None;attempt_started=False
                try:
                    with native_stream_scope(),torch.no_grad():
                        if protocol is not None:
                            env=protocol.environment(args,row,bindings,deadline=deadline,qpos=q,qvel=v,
                                requests=requests,target_input=FrozenSnapshotInput(requests))
                        else:env=ClosedProtocolNativeSpecialistEnv(args,[row],num_envs=len(bindings),deadline=deadline,
                            expected_qpos=q,expected_qvel=v,requests=requests,target_input=FrozenSnapshotInput(requests))
                        if env.library.bytes>limits.resident_reference_bytes:raise ValueError("Measured resident reference bound exceeded")
                        deadline.checkpoint("after_native_initialization")
                        if policy is None:
                            policy=fresh_policy({"checkpoint":copied},env.device)
                            if policy_digest(policy)!=training_receipt["final_policy_digest"]:
                                raise ValueError("Restored parameter bytes differ from the frozen learned training receipt")
                            write_json(folder/"learned_parameter_changes.json",learned_parameter_changes(policy,paths["checkpoint"]))
                            parity=restored_policy_parity(copied,policy,torch.zeros((1,369),device=env.device))
                            write_json(folder/"restored_checkpoint_parity.json",parity)
                            if not parity["parameters_equal"] or not parity["actions_equal"]:raise ValueError("Restored actor parity differs")
                            export_runtime_model(args.model,folder/"runtime.xml",env.model)
                        env.bind_policy(policy)
                        write_json(folder/f"native_attempt_intent_{number:04d}.json",dict(
                            schema="task10b_phase2_native_attempt_intent_v1",execution_identity=identity["identity"],
                            trial_ids=[b.trial_id for b in bindings],request_identities=[b.identity for b in bindings],
                            batch_path=str(batch_folder),phase="Immediately before the one physical reset and uninterrupted evaluation",
                            never_replay_after_external_stop=True))
                        for binding in bindings:
                            ledger[binding.trial_id].update(status="executed",physical_execution_attempted=True,
                                execution_complete=False,numeric_complete_cycle=False,outcome_resolution="unknown_or_missing",
                                reason="Durable native attempt began; retain trace/outcome or unresolved failure")
                        write_json(folder/f"ledger_attempt_{number:04d}.json",list(ledger.values()))
                        native_attempts_started+=len(bindings)
                        attempt_started=True
                        panel=evaluate_policy(env,policy,batch_folder,reference_ids=np.zeros(len(bindings),dtype=int),
                            initial_qpos=q,initial_qvel=v,seed=declaration.seed,predicate=read_json(predicate_path),
                            trial_ids=[b.trial_id for b in bindings])
                        initialization=(protocol.audit_initialization(batch_folder,env,bindings,q,v) if protocol is not None else
                            audit_initialization(batch_folder/"raw_trace.npz",q,v)
                            if (batch_folder/"raw_trace.npz").exists() else {"passed":False,"reason":"no_retained_initial_state"})
                        write_json(batch_folder/"initialization_audit.json",initialization)
                        if panel["status"]=="executed" and not initialization["passed"]:
                            panel={**panel,"status":"failed","exception":{"type":"InitializationAuditFailure","message":"Independent family-guard initialization differs"},
                                "audit":{**panel["audit"],"passed":False}}
                        write_json(batch_folder/"expanded_scope.json",dict(bindings=[asdict(b) for b in bindings],
                            declaration_identity=declaration.identity,reference_resident_bytes=env.library.bytes,
                            elapsed_session_wall_s=time.monotonic()-started,
                            deadline_interruption=getattr(env,"deadline_interruption_receipt",None),
                            isolated_guard_preparation_complete_s=0. if protocol is None else None,
                            preparation_complete_s=row.get("preparation_complete_tick",0)*.005,equal_full_horizons=True,
                            rejection_semantics="No rejected request receives launch/hit/cycle credit. Inherited rejection freezes reference tick but can retain nonzero boundary velocity/feedforward; preserve actual subsequent motion and first contacts. No physical static-hold claim or pose/history replacement.",
                            historical_inner_planning_label="Inherited evaluator plan text says D0; this receipt binds actual expanded groups, fixed snapshot targets, timings and declared starts. Shared preparation, when present, is explicit in protocol metadata.",
                            continuation_scope="Exact declared physical/controller initial state, uninterrupted preparation/reference prefix, accepted positional/support readiness, same immutable request and fresh exact fixed-snapshot input at launch. No arbitrary-state or quiet-hook readiness claim.",
                            protocol_metadata=protocol.metadata if protocol is not None else {},
                            initialization_version=protocol.version if protocol is not None else INITIALIZATION_VERSION))
                        for outcome in outcome_rows(declaration,bindings,panel,batch_folder):
                            outcome["preparation_complete_s"]=row.get("preparation_complete_tick",0)*.005
                            if protocol is not None:outcome.update(protocol.outcome_metadata(outcome["trial_id"]))
                            decisions=[record for record in env.closed_launch_records if record["trial_id"]==outcome["trial_id"]]
                            outcome["closed_protocol_launch"]=decisions
                            outcome["immutable_launch_pass"]=bool(outcome.get("immutable_launch_pass") and len(decisions)==1 and decisions[0]["accepted"])
                            outcome["continuity_history_pass"]=bool(outcome.get("continuity_history_pass") and len(decisions)==1 and decisions[0]["accepted"])
                            ledger[outcome["trial_id"]]=outcome
                        panels.append(dict(path=str(batch_folder),status=panel["status"],planned_trials=len(bindings),
                            actual_physics_steps=panel["actual_physics_steps"],actual_policy_updates=panel["actual_policy_updates"],
                            wall_s=panel["wall_time_s"],reference_resident_bytes=env.library.bytes))
                        write_json(folder/f"ledger_{number:04d}.json",list(ledger.values()))
                        if panel["status"]!="executed":
                            exception=panel.get("exception") or {"type":"TraceAuditFailure","message":"Preserved native batch audit failed"}
                            break
                except Exception as error:
                    if attempt_started:
                        # The inherited evaluator can fail during postprocessing
                        # after writing a raw trace or some trial receipts. Never
                        # turn those physically attempted worlds into unattempted.
                        partial=dict(trials=[read_json(path) for path in sorted(batch_folder.glob("trial_*.json"))],
                            exception=dict(type=type(error).__name__,message=str(error)),
                            hard_faults=env.hard_faults,audit={"passed":False},
                            actual_physics_steps=env.physics_steps_executed,
                            actual_policy_updates=env.actor_updates_executed)
                        for outcome in outcome_rows(declaration,bindings,partial,batch_folder):ledger[outcome["trial_id"]]=outcome
                        write_json(folder/f"interrupted_batch_{number:04d}.json",partial)
                        write_json(folder/f"ledger_partial_{number:04d}.json",list(ledger.values()))
                    raise
                finally:
                    if env is not None:
                        if batch_folder.exists():
                            boundaries=write_launch_boundaries(batch_folder,env.closed_boundary_snapshots)
                            write_json(batch_folder/"closed_protocol_launch.json",dict(
                                schema="task10b_phase2_closed_protocol_native_receipt_v1",requests=requests,
                                decisions=env.closed_launch_records,initialization_version=protocol.version if protocol is not None else INITIALIZATION_VERSION,
                                boundary_snapshots=boundaries,
                                arbitrary_live_state_admission=False,physical_state_or_history_replaced=False,
                                real_time_operation_qualified=False,
                                wall_scope="Measured target callback/cache lookup/unchanged-state finalization while simulation pauses; actor execution excluded",
                                independent_expanded_gate_audit_required=True))
                        del env
                    gc.collect();torch.cuda.empty_cache()
    except Exception as error:
        exception=dict(type=type(error).__name__,message=str(error))
        if isinstance(error,BoundedExecutionInterruption):exception["deadline_receipt"]=error.receipt
    for row in ledger.values():
        if row["status"]=="unattempted" and row["trial_id"] in eligible:
            row["reason"]="preserved_unattempted_remainder_after_stop" if exception else "not_executed"
    outcomes=list(ledger.values());write_json(folder/"outcomes.json",outcomes)
    qualification=qualify_panel(declaration,outcomes)
    write_json(folder/"qualification_pending_review.json",qualification)
    result=dict(schema="task10b_phase2_expanded_execution_v1",status="failed" if exception else "executed_pending_audit_and_video",
        application_run=str(folder),
        execution_identity=identity["identity"] if identity else None,outcomes_sha256=sha256_file(folder/"outcomes.json"),
        logical_origin_run=logical_origin,continuation=continuation_context is not None,
        continuation_configuration_sha256=continuation_context["configuration_sha256"] if continuation_context else None,
        this_session_eligible_trials=len(eligible),inherited_retained_trials=len(declaration.bindings)-len(eligible),
        exception=exception,planned_trials=len(declaration.bindings),native_attempts_started=native_attempts_started,
        completed_attempts=sum(row.get("execution_complete") is True for row in outcomes),
        incomplete_attempts=sum(row["status"] in ("executed","timeout") and not row.get("execution_complete",False) for row in outcomes),
        unattempted_attempts=sum(row["status"]=="unattempted" for row in outcomes),
        rejected_attempts=sum(row["status"]=="rejected" for row in outcomes),panels=panels,
        generation_failure_attempts=sum(row["status"]=="generation_failure" for row in outcomes),
        preflight_completed=preflight_done,final_test_claim=str(claim) if claim else None,
        automatic_retry=False,resume_unattempted_remainder_implemented=True,
        attempted_or_timed_out_requests_replayed=False,
        total_wall_s=time.monotonic()-started,physics_dt_s=.005,actor_dt_s=.020,
        no_training=True,fresh_training_transitions=0,formal_acceptance=False,phase2_complete=False,
        qualification_passed=False,independent_trace_audit_and_video_review_required=True)
    write_json(folder/"execution.json",result)
    if claim_session is not None:write_json(claim_session/"terminal.json",dict(run=str(folder),**result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("declaration","manifest","route","checkpoint","training","predicate","mask","output","release"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,required=True)
    parser.add_argument("--maximum-worlds",type=int,default=16)
    parser.add_argument("--continuation",type=Path)
    options=parser.parse_args()
    result=execute(options.declaration,options.manifest,options.route,options.checkpoint,options.training,
        options.predicate,options.mask,options.output,release=options.release,
        limits=EvaluationLimits(options.wall_seconds,maximum_worlds=options.maximum_worlds),continuation=options.continuation)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="executed_pending_audit_and_video" else 1


if __name__=="__main__":raise SystemExit(main())
