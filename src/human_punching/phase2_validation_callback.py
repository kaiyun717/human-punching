"""Held-out physical validation within a bounded grouped PPO session.

The declaration template supplies only physical requests. Its future checkpoint
and transition identities are deliberately unbound here: update zero evaluates
the transferred frozen policy and supplies no learned qualification claim.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .phase2_domain import fingerprint
from .phase2_expanded_evaluation import (
    EvaluationLimits, ClosedProtocolNativeSpecialistEnv, FrozenSnapshotInput, INITIALIZATION_VERSION,
    audit_initialization, batch_requests, declared_mask_decisions, launch_requests,
    load_declaration, owned_file, preflight, read_json, validate_declared_starts,
    validate_geometry_bindings,
    write_launch_boundaries,
)
from .phase2_learning import evaluate_policy, family_arguments, policy_digest
from .scalar_control import export_runtime_model
from .wbc.provenance import sha256_file, writable_path, write_json, runtime_record


class GroupedValidationCallback:
    """Exact separate eight-group, three-timing, four-start native callback."""

    @classmethod
    def from_configuration(cls, path, expected_sha256, training_declaration, release, output):
        config = read_json(path, expected_sha256)
        if config.get("schema") != "task10b_phase2_grouped_validation_configuration_v1":
            raise ValueError("Frozen grouped validation configuration required")
        inputs = {}
        for name in ("descriptor", "declaration_template", "manifest", "mask", "predicate"):
            item = config[name]
            inputs[name] = owned_file(item["path"], item["sha256"])
        descriptor = read_json(inputs["descriptor"])
        # Lazy import avoids a module cycle and keeps the callback independently
        # importable for CPU interface tests.
        from .phase2_grouped_learning import validate_validation_descriptor
        validate_validation_descriptor(training_declaration, descriptor,
            read_json(config["domain"]["path"], training_declaration.domain_sha256))
        declaration = load_declaration(inputs["declaration_template"])
        if (declaration.split != "validation" or declaration.start_mode != "family_guard"
                or declaration.family != training_declaration.family or declaration.seed != training_declaration.seed
                or set(declaration.command_groups) != set(descriptor["command_groups"])
                or declaration.trained_timings_s != training_declaration.trained_timings_s
                or declaration.interpolation_timing_s != training_declaration.interpolation_timing_s
                or declaration.predicate_sha256 != training_declaration.predicate_sha256
                or declaration.declared_mask_sha256 != training_declaration.declared_mask_sha256
                or {b.trial_id for b in declaration.bindings} != set(descriptor["trial_ids"])
                or sha256_file(inputs["mask"]) != declaration.declared_mask_sha256
                or sha256_file(inputs["predicate"]) != declaration.predicate_sha256):
            raise ValueError("Validation template differs from the exact held-out physical requests")
        manifest = read_json(inputs["manifest"])
        if (manifest["domain_sha256"] != training_declaration.domain_sha256
                or manifest["target_geometry_sha256"] != training_declaration.target_geometry_sha256
                or manifest.get("recipe_sha256") != training_declaration.recipe_sha256):
            raise ValueError("Validation callback canonical domain or geometry differs")
        domain = read_json(manifest["domain_path"], manifest["domain_sha256"])
        targets = read_json(manifest["target_geometry_path"], manifest["target_geometry_sha256"])
        catalog = read_json(manifest["catalog_path"], manifest["catalog_sha256"])
        if catalog.get("schema")!="task10b_phase2_reference_catalog_v1":
            raise ValueError("Canonical expanded reference catalog required")
        decisions = declared_mask_decisions(declaration, read_json(inputs["mask"]))
        # wall_s is a preflight bound only; every executed batch shares the
        # caller's real session deadline, including compilation and serialization.
        limits = EvaluationLimits(14400., maximum_worlds=config.get("maximum_worlds", 16),
            resident_reference_bytes=min(training_declaration.resident_reference_bytes,256*1024**2))
        entries, rows, starts, scope = preflight(declaration, manifest, catalog, limits=limits,
            mask_decisions=decisions)
        scope.update(validate_geometry_bindings(declaration, manifest, domain, targets, entries, rows))
        args, paths = family_arguments(release, declaration.family)
        validate_declared_starts(declaration, domain, starts, paths)
        for row in rows.values():
            if row["candidate_id"] in {entry["candidate_id"] for entry in entries.values()}:
                if row.get("recipe_sha256") != training_declaration.recipe_sha256:
                    raise ValueError("Validation reference recipe differs from the training declaration")
        self = cls()
        self.descriptor, self.declaration = descriptor, declaration
        self.args, self.paths, self.inputs = args, paths, inputs
        self.catalog, self.entries, self.rows, self.starts = catalog, entries, rows, starts
        self.manifest=manifest
        self.decisions, self.limits, self.scope = decisions, limits, scope
        self.predicate = read_json(inputs["predicate"])
        self.targets = {r["candidate_identity"]: r for r in targets["geometry_candidates"]}
        self.output = writable_path(output)
        self.training_declaration = training_declaration
        self.source_configuration = dict(path=str(owned_file(path)), sha256=expected_sha256)
        return self

    def make_environment(self,row,bindings,*,deadline,qpos,qvel,requests,target_input):
        protocol=getattr(self,'validation_protocol',None)
        if protocol is not None:
            return protocol.environment(self.args,row,bindings,deadline=deadline,qpos=qpos,qvel=qvel,
                requests=requests,target_input=target_input)
        return ClosedProtocolNativeSpecialistEnv(self.args,[row],num_envs=len(bindings),deadline=deadline,
            expected_qpos=qpos,expected_qvel=qvel,requests=requests,target_input=target_input)

    def initialization_audit(self,folder,env,bindings,qpos,qvel):
        protocol=getattr(self,'validation_protocol',None)
        if protocol is not None:return protocol.audit_initialization(folder,env,bindings,qpos,qvel)
        return (audit_initialization(folder/'raw_trace.npz',qpos,qvel)
            if (folder/'raw_trace.npz').exists() else {'passed':False})

    def __call__(self, policy, update, *, deadline):
        from .wbc.telemetry.cycle_qualification import native_stream_scope
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Physical grouped validation requires a counted Slurm CUDA allocation")
        if type(update) is not int or update < 0 or update % 8:
            raise ValueError("Original PPO validation cadence is updates zero/eight/sixteen/etc")
        folder = self.output/f"validation_{update:05d}"
        folder.mkdir(parents=True, exist_ok=False)
        actual_digest = policy_digest(policy)
        ledger = {b.trial_id: dict(trial_id=b.trial_id, command_id=b.command_id,
            requested_ttc_s=b.ttc_s, start_index=b.start_index, numeric_complete_cycle=False,
            disposition="unattempted", reason="not_reached") for b in self.declaration.bindings}
        write_json(folder/"planned_trials.json", list(ledger.values()))
        write_json(folder/"descriptor.json", self.descriptor)
        write_json(folder/"physical_declaration.json",asdict(self.declaration))
        write_json(folder/"catalog.json", self.catalog)
        saved = folder/"training"; saved.mkdir()
        with (saved/"checkpoint_in_memory.pt").open("xb") as stream:
            torch.save(dict(policy=policy.state_dict(), observations=369, actions=48,
                actor_parameters_sha256=actual_digest, update=update,
                physical_validation_only=True, learned_qualification=False), stream)
        write_json(folder/"launch.json", dict(schema="task10b_phase2_grouped_validation_launch_v1",
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), runtime=runtime_record(),
            catalog=str(folder/"catalog.json"), catalog_sha256=sha256_file(folder/"catalog.json"),
            release_inputs={key:dict(path=str(path),sha256=sha256_file(path)) for key,path in self.paths.items()},
            configuration=self.source_configuration, descriptor_identity=fingerprint(self.descriptor),
            physical_request_template_identity=self.declaration.identity,
            physical_declaration=dict(path=str(folder/"physical_declaration.json"),sha256=sha256_file(folder/"physical_declaration.json")),
            domain=dict(path=self.manifest["domain_path"],sha256=self.manifest["domain_sha256"]),
            target_geometry=dict(path=self.manifest["target_geometry_path"],sha256=self.manifest["target_geometry_sha256"]),
            recipe_sha256=self.training_declaration.recipe_sha256,
            domain_sha256=self.training_declaration.domain_sha256,
            target_geometry_sha256=self.training_declaration.target_geometry_sha256,
            checkpoint_template_is_unbound=True, actual_policy_digest=actual_digest,
            start_mode=getattr(self.declaration,'start_mode','family_guard'),
            validation_protocol=getattr(getattr(self,'validation_protocol',None),'metadata',None),
            actual_training_update=update, actual_completed_fresh_transitions=update*128*64,
            no_positive_learned_qualification=True, no_final_test=True,
            evaluated_checkpoint=dict(path=str(saved/"checkpoint_in_memory.pt"),sha256=sha256_file(saved/"checkpoint_in_memory.pt")),
            initial_frozen_control=update==0, sources={str(Path(__file__).resolve()):sha256_file(__file__)}))
        started=time.monotonic(); panels=[]; faults=[]; exception=None
        for number,(row, bindings) in enumerate(batch_requests(self.declaration,self.entries,self.rows,self.limits.maximum_worlds)):
            admitted=[]
            for b in bindings:
                if row.get("kinematic_admission") is True and self.decisions[b.trial_id]["admitted"]:
                    admitted.append(b)
                else:
                    generation_failure=row.get("reference_identity_kind")=="generation_failure_receipt"
                    ledger[b.trial_id].update(disposition="generation_failure" if generation_failure else "rejected",
                        physical_attempt=False,reason=self.decisions[b.trial_id].get("reason") or "reference_not_admitted")
            bindings=tuple(admitted)
            if not bindings:continue
            env=None; attempted=False; batch_folder=folder/f"batch_{number:04d}"
            try:
                deadline.checkpoint("before_grouped_validation_batch", progress={"update":update,"batch":number})
                q=np.stack([self.starts[b.trial_id][0] for b in bindings]);v=np.stack([self.starts[b.trial_id][1] for b in bindings])
                requests=launch_requests(self.declaration, bindings, row, self.targets)
                for request in requests:
                    request["checkpoint_sha256"]=sha256_file(saved/"checkpoint_in_memory.pt")
                with native_stream_scope(), torch.no_grad():
                    env=self.make_environment(row,bindings,deadline=deadline,qpos=q,qvel=v,
                        requests=requests,target_input=FrozenSnapshotInput(requests))
                    if env.library.bytes>self.limits.resident_reference_bytes:
                        raise ValueError("Validation resident reference bound exceeded")
                    env.bind_policy(policy)
                    deadline.checkpoint("after_grouped_validation_initialization")
                    if not (folder/"runtime.xml").exists():export_runtime_model(self.args.model,folder/"runtime.xml",env.model)
                    attempted=True
                    panel=evaluate_policy(env,policy,batch_folder,reference_ids=np.zeros(len(bindings),dtype=int),
                        initial_qpos=q,initial_qvel=v,seed=self.declaration.seed,predicate=self.predicate,
                        trial_ids=[b.trial_id for b in bindings])
                    initialization=self.initialization_audit(batch_folder,env,bindings,q,v)
                    write_json(batch_folder/"initialization_audit.json",initialization)
                    reports={r["trial_id"]:r for r in panel["trials"]}
                    for b in bindings:
                        if b.trial_id in reports:
                            ledger[b.trial_id].update(reports[b.trial_id], disposition="completed_native_attempt")
                            decisions=[r for r in env.closed_launch_records if r["trial_id"]==b.trial_id]
                            closed=bool(len(decisions)==1 and decisions[0]["accepted"])
                            ledger[b.trial_id]["closed_protocol_launch"]=decisions
                            ledger[b.trial_id]["numeric_complete_cycle"] &= initialization["passed"] and closed
                        else:
                            ledger[b.trial_id].update(disposition="incomplete_native_failure", reason=panel.get("exception"))
                    faults.extend(dict(batch=number, **fault) for fault in panel["hard_faults"])
                    panels.append(dict(path=str(batch_folder),status=panel["status"],planned_trials=len(bindings),
                        actual_physics_steps=panel["actual_physics_steps"], actual_policy_updates=panel["actual_policy_updates"]))
                    if panel["status"]!="executed" or not initialization["passed"]:
                        raise RuntimeError("Physical validation failed; inspect retained batch and initialization receipts")
            except Exception as error:
                exception=dict(type=type(error).__name__,message=str(error))
                if attempted:
                    # Postprocessing can fail after integration. Every affected
                    # missing result remains attempted with unknown extent.
                    for b in bindings:
                        if ledger[b.trial_id]["disposition"]=="unattempted":
                            ledger[b.trial_id].update(disposition="incomplete_native_failure", reason=exception,
                                execution_extent_unknown=True)
                    if env is not None:
                        faults.extend(dict(batch=number, **fault) for fault in env.hard_faults
                            if dict(batch=number,**fault) not in faults)
                break
            finally:
                if env is not None and batch_folder.exists():
                    boundaries=write_launch_boundaries(batch_folder,env.closed_boundary_snapshots)
                    write_json(batch_folder/"closed_protocol_launch.json",dict(
                        schema="task10b_phase2_closed_protocol_native_receipt_v1",requests=requests,
                        initialization_version=(self.validation_protocol.version if hasattr(self,"validation_protocol")
                            else INITIALIZATION_VERSION),
                        decisions=env.closed_launch_records,boundary_snapshots=boundaries,physical_state_or_history_replaced=False,
                        arbitrary_live_state_admission=False,real_time_operation_qualified=False))
                if env is not None:del env
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                write_json(folder/f"ledger_{number:04d}.json",list(ledger.values()))
        unchanged=actual_digest==policy_digest(policy)
        if not unchanged:exception=dict(type="PolicyMutation",message="Validation changed policy tensors")
        trials=list(ledger.values())
        complete=all(r["disposition"]=="completed_native_attempt" for r in trials)
        result=dict(schema="task10b_phase2_grouped_physical_validation_v1",
            status="executed" if complete and exception is None and unchanged else "failed",
            grouped_validation_identity=fingerprint(self.descriptor), trials=trials,hard_faults=faults,
            exception=exception,panels=panels,actual_policy_digest=actual_digest,actor_unchanged=unchanged,
            actual_training_update=update,actual_completed_fresh_transitions=update*128*64,
            planned_trials=96,completed_trials=sum(r["disposition"]=="completed_native_attempt" for r in trials),
            checkpoint_template_is_unbound=True,learned_qualification=False,
            full_domain_qualification=False,independent_audit_and_video_pending=True,
            wall_time_s=time.monotonic()-started,source_configuration=self.source_configuration)
        write_json(folder/"summary.json",result)
        return result
