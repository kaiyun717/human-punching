"""One bounded cross learning-curve extension; original PPO remains unchanged.

The parent 16-update plan is an immutable source/reference selection, not the
execution budget. This version explicitly declares a fresh 32-update run from
accepted Task09 weights with same seed, never a continuation or replication.
No native work runs on import. Old packets and their source files stay frozen.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
import argparse
import json
import os
import time
import numpy as np
from . import phase2_timing_pilot as parent
from .phase2_learning import PilotBudget, NativeSpecialistEnv, family_arguments, fresh_policy, train_pilot
from .phase2_d0_clock_v2 import run_cell_panel, evaluator_receipt
from .phase2_interiors import ROOT, owned, bound, binding, read, sha, _write_json

VERSION = "task10b_phase2_cross_learning_curve_v2"
SCHEMA = VERSION + "_plan"
UPDATES = [0, 8, 16, 24, 32]
CAP = 262144
CONTROLS = ["cross_original", "cross_uniform_0p600"]
SELECTED = ["cross_uniform_0p600", "cross_nonuniform_postcontact_0p750"]
SOURCES = ["src/human_punching/phase2_timing_pilot_curve_v2.py",
           "scripts/task10b_phase2_curve_v2.py", "scripts/task10b_phase2_curve_v2.sbatch"]

class ControlRegressionError(RuntimeError):
    """A development control failed; this is not itself a physical hard fault."""


def control_regression(panel):
    rows = panel.get("trials", [])
    checks = {}
    for candidate in CONTROLS:
        selected = [r for r in rows if r.get("candidate_id") == candidate]
        expected = {f"{candidate}_bank_{i:02d}" for i in range(8)}
        checks[candidate] = (len(selected) == 8
            and {r.get("trial_id") for r in selected} == expected
            and all(r.get("numeric_complete_cycle") is True for r in selected))
    return dict(passed=all(checks.values()), checks=checks,
                stop_type="development_control_regression", new_physical_fault_claim=False)


def validate_rendered_media(video, expected_media):
    """Allow an explicit setup-only rendering repair, never a silent substitution."""
    actual = video.get("rendered_media_packets", [])
    if [row.get("update") for row in actual] != [0, 8, 16]:
        raise ValueError("Three successful render packets are required")
    for declared, row in zip(expected_media, actual, strict=True):
        if set(row) != {"update", "packet", "render_summary", "render_execution", "terminal_accounting"}:
            raise ValueError("Complete actual render evidence required")
        old = read(bound(declared["packet"]))
        packet_path = bound(row["packet"])
        packet = read(packet_path)
        if (packet.get("mode") != "media" or packet.get("training") is not False
                or packet.get("family") != "cross" or packet.get("seed") != 21021
                or packet.get("inputs") != old.get("inputs") or packet.get("files") != old.get("files")
                or (packet.get("allocation_wall_seconds"), packet.get("application_wall_seconds"))
                   not in ((240, 180), (300, 240))):
            raise ValueError("Actual render must preserve the exact saved selection and source inputs")
        paths = {key:bound(row[key]) for key in ("render_summary", "render_execution", "terminal_accounting")}
        if (paths["render_summary"] != packet_path.parent/"run/summary.json"
                or paths["render_execution"] != packet_path.parent/"execution.json"
                or paths["terminal_accounting"] != packet_path.parent/"terminal_accounting.json"):
            raise ValueError("Render results must belong to their exact packet")
        summary, execution, terminal = (read(paths[k]) for k in
            ("render_summary", "render_execution", "terminal_accounting"))
        if (summary.get("planned") != 10 or summary.get("rendered") != 10
                or summary.get("failed") != 0 or summary.get("actor_calls") != 0
                or execution.get("status") != "executed" or execution.get("returncode") != 0
                or terminal.get("state") != "COMPLETED"
                or terminal.get("packet_sha256") != row["packet"]["sha256"]
                or execution.get("packet_sha256") != row["packet"]["sha256"]
                or terminal.get("job_id") != execution.get("job_id")):
            raise ValueError("All30 selected videos need successful bound render execution")


def create_plan(parent_plan, numerical_review, video_review):
    plan = dict(schema=SCHEMA, version=VERSION, family="cross", seed=21021,
        parent_plan=binding(parent_plan), numerical_review=binding(numerical_review),
        video_review_v2=binding(video_review), sources_v2={n:binding(ROOT/n) for n in SOURCES},
        budget=asdict(PilotBudget(maximum_updates=32, maximum_wall_s=360.)),
        fresh_transition_cap=CAP, validation_updates=UPDATES,
        planned_trials_per_panel=24, planned_validation_trials=120,
        application_wall_seconds=360., maximum_session_allocation_seconds=480,
        wrapper_wall_seconds=420., final_test_access=False, actual_route_freeze=False,
        conditional_replication_budget_unlocked=False, policy_promotion=False,
        same_seed_learning_curve=True, prior_training_resumed=False,
        stop_on_any_original_or_060_control_failure=True)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    fixed = dict(schema=SCHEMA, version=VERSION, family="cross", seed=21021,
        fresh_transition_cap=CAP, validation_updates=UPDATES,
        planned_trials_per_panel=24, planned_validation_trials=120,
        application_wall_seconds=360., maximum_session_allocation_seconds=480,
        wrapper_wall_seconds=420., final_test_access=False, actual_route_freeze=False,
        conditional_replication_budget_unlocked=False, policy_promotion=False,
        same_seed_learning_curve=True, prior_training_resumed=False,
        stop_on_any_original_or_060_control_failure=True)
    if any(plan.get(k) != v or type(plan.get(k)) != type(v) for k,v in fixed.items()):
        raise ValueError("Exact finite cross learning-curve contract required")
    if set(plan) != set(fixed) | {"parent_plan", "numerical_review", "video_review_v2", "sources_v2", "budget"}:
        raise ValueError("Unexpected curve-plan fields")
    budget = PilotBudget(**plan["budget"])
    if asdict(budget) != asdict(PilotBudget(maximum_updates=32, maximum_wall_s=360.)):
        raise ValueError("Exactly128x64x32 fresh transitions with90s shutdown reserve required")
    selection = read(bound(plan["parent_plan"]))
    paths, rows, _ = parent.validate_plan(selection)
    if selection["family"] != "cross" or selection["training_candidate_ids"] != SELECTED:
        raise ValueError("Only the explicitly reviewed cross .60/.75 route is eligible")
    if set(plan["sources_v2"]) != set(SOURCES):
        raise ValueError("Complete isolated source bindings required")
    for name, item in plan["sources_v2"].items():
        if bound(item) != owned(ROOT/name):raise ValueError("Curve source path changed")
    review = read(bound(plan["numerical_review"]))
    video = read(bound(plan["video_review_v2"]))
    # The exact numerical receipt is selected by root after independent audit.
    # It is not a generic receipt that may unlock another route.
    if plan["numerical_review"]["sha256"] != "f1a8bf7ca8f3ba0effec20a3b4f97a78f1274c4f0fcbb83c536c9eb4ede06465":
        raise ValueError("Exact measured cross pilot review required")
    if (video.get("all_required_clips_reviewed") is not True
            or video.get("movement_exploit_detected") is not False
            or type(video.get("trials_reviewed")) is not int or video["trials_reviewed"] != 30):
        raise ValueError("All30 predeclared/random and flagged cross clips must be reviewed first")
    expected_media = [{"update": row["update"], "packet": row["packet"]}
                      for row in review["media"]["packets"]]
    if (video.get("schema") != "task10b_phase2_cross30_visual_review_v1"
            or video.get("numerical_review") != plan["numerical_review"]
            or video.get("media_packets") != expected_media
            or [row["update"] for row in expected_media] != [0, 8, 16]):
        raise ValueError("Visual review must bind this exact parent pilot's three media packets")
    for item in expected_media:
        bound(item["packet"])
    validate_rendered_media(video, expected_media)
    return paths, rows, budget


def execution_plan(plan):
    """Explicit inherited controller/reference settings plus the NEW finite budget."""
    selection = read(bound(plan["parent_plan"]))
    combined = deepcopy(selection)
    combined.update(plan)
    combined["sources"] = {**selection["sources"], **plan["sources_v2"]}
    combined["stopping_rules"] = {**deepcopy(selection["stopping_rules"]),
        "validation_updates": list(UPDATES),
        "original_and_060_controls": "stop_on_any_of_the_matched16_numeric_cycle_failures",
        "control_regression_classification": "development_stop_not_a_new_physical_fault_claim"}
    return combined


def launch_receipt(plan_path, plan, *, runtime, clock, slurm_job_id, progress):
    descriptor = binding(plan_path)
    original = read(bound(descriptor))
    if plan != execution_plan(original):raise ValueError("Execution view changed after commitment")
    return dict(schema=VERSION, mode="timing_learning_curve", family="cross", seed=21021,
        plan=descriptor, plan_sha256=descriptor["sha256"], runtime=runtime,
        slurm_job_id=str(slurm_job_id), catalog=str(bound(plan["catalog"])),
        catalog_sha256=plan["catalog"]["sha256"], predicate=str(bound(plan["predicate"])),
        predicate_sha256=plan["predicate"]["sha256"], release_inputs=deepcopy(plan["release_inputs"]),
        sources={str(bound(item)):item["sha256"] for item in plan["sources"].values()},
        parent_checkpoint=deepcopy(plan["release_inputs"]["checkpoint"]),
        clock_postprocessing=clock, postprocessing_clock_adapter=clock,
        planned_cells=list(plan["validation_candidate_ids"]), diagnostic_bank_ids=list(range(8)),
        validation_updates=UPDATES, fresh_transition_cap=CAP, budget=deepcopy(plan["budget"]),
        planned_validation=deepcopy(progress), same_seed_learning_curve=True,
        fresh_optimizer=True, fresh_simulation=True, prior_training_resumed=False,
        independent_replication=False, phase2_complete=False, no_test_split_access=True)


require_allocation = parent.require_allocation
require_final_checkpoint_parity = parent.require_final_checkpoint_parity

def execute(plan_path,output):
    """Invoke original training; this entrypoint neither submits nor selects."""
    started=time.monotonic();require_allocation()
    plan_path=owned(plan_path);committed=read(plan_path);paths,rows,budget=validate_plan(committed)
    plan=execution_plan(committed)
    output=owned(output,exists=False)
    if not output.is_relative_to(ROOT.resolve()/"outputs/task10b"):raise ValueError("Fresh outputs/task10b pilot directory required")
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/"plan.json",committed)
    _write_json(output/"execution_plan.json",plan)
    progress={str(u):dict(disposition="unattempted",planned_trials=plan["planned_trials_per_panel"]) for u in plan["validation_updates"]}
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
            if update not in plan["validation_updates"] or progress[str(update)]["disposition"]!="unattempted":raise ValueError("Unexpected validation update")
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
            control = control_regression(panel)
            _write_json(output/f"control_regression_{update:05d}.json", control)
            if not control["passed"]:
                raise ControlRegressionError("Original or .60 s control regressed: " + json.dumps(control))
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
        plan_sha256=sha(plan_path),fresh_transition_cap=budget.transition_cap,
        fresh_transitions=summary["fresh_transitions"] if summary else (None if training_started else 0),
        fresh_transition_extent="recorded" if summary else ("execution_extent_unknown" if training_started else "known_zero_training_not_started"),
        phase2_complete=False,formal_acceptance=False,final_test_executed=False,policy_promoted=False,
        training_engine="unchanged phase2_learning.train_pilot",validation_engine="clock_v2.run_cell_panel postprocessing")
    _write_json(output/"execution.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=execute(args.plan,args.output)
    print(json.dumps({k:result[k] for k in ("status","exception","fresh_transitions","phase2_complete")}))
    return 0 if result["status"]=="bounded_stop" else 1

if __name__=="__main__":raise SystemExit(main())
