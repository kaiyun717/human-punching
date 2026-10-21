#!/usr/bin/env python3
"""Prepare/run one bounded Task 09 migration comparison; never submit a job.

Replays four preserved specialists and paired zero-residual controls on their
original 64-trial banks. Historical per-run Python sources are copied byte-for-
byte into disposable packages. The runtime is the Task 10 candidate, so matching
historical behavior must be measured and is not assumed. No learning, WBC weight
initialization, reference qualification, or downstream job submission occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import runtime_record, sha256_file, writable_path, write_json

FAMILIES = ("jab", "cross", "left_hook", "right_hook")
RELEASE_SHA256 = "16eebb942f71f8350bf3d886f8a083eac549a5bc3486fe9d5304b8cde29b0686"
SCHEMA = "human_wbc_historical_migration_packet_v1"
MAX_BYTES = 4 * 1024**3
CONTROLLER_FIELDS = ("timestep", "policy_dt", "speed", "hold", "gain_scale", "action_scale",
                     "no_feedforward", "balance_feedback", "controller", "refresh_kinematics",
                     "reward_profile", "glove_sharpness", "joint_limit_penalty",
                     "attack_tracking_weight", "attack_tracking_scale", "guard_relative_weight")


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_argv(argv):
    """Reject training/transfer modes even if a prepared packet is hand-edited."""
    if "--initialize-policy" in argv or argv.count("--mode") != 1:
        raise ValueError("Only explicit checkpoint evaluation is permitted")
    if argv[argv.index("--mode") + 1] != "evaluate":
        raise ValueError("Migration never trains")
    for flag, expected in (("--controller", "implicit_pd"), ("--timestep", "0.005"),
                           ("--policy-dt", "0.02"), ("--recorded-worlds", "64"),
                           ("--num-envs", "64")):
        if argv.count(flag) != 1 or argv[argv.index(flag)+1] != expected:
            raise ValueError(f"Migration requires {flag} {expected}")
    if not {"--require-five-ms", "--refresh-kinematics"}.issubset(argv):
        raise ValueError("Five-ms clock and fresh-kinematics guards are mandatory")
    for flag in ("--output", "--model", "--reference", "--body-metadata", "--evaluation-bank"):
        if argv.count(flag) != 1:
            raise ValueError(f"Exactly one {flag} is required")


def build_argv(inputs, arguments, output, *, checkpoint=None):
    argv = ["--mode", "evaluate", "--model", inputs["model"], "--reference", inputs["reference"],
            "--body-metadata", inputs["metadata"], "--evaluation-bank", inputs["bank"],
            "--output", str(output), "--controller", "implicit_pd", "--refresh-kinematics",
            "--require-five-ms", "--timestep", "0.005", "--policy-dt", "0.02",
            "--recorded-worlds", "64", "--num-envs", "64"]
    for key in ("reward_profile", "glove_sharpness", "joint_limit_penalty", "attack_tracking_weight",
                "attack_tracking_scale", "guard_relative_weight", "guard_joint_weight", "speed",
                "hold", "gain_scale", "action_scale", "seed"):
        if key in arguments:
            argv += ["--" + key.replace("_", "-"), str(arguments[key])]
    for key in ("no_feedforward", "balance_feedback"):
        if arguments.get(key, False):
            argv.append("--" + key.replace("_", "-"))
    argv += ["--attack-window", *map(str, arguments["attack_window"])]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    validate_argv(argv)
    return argv


def _checked_release_path(release, manifest, relative):
    path = (release / relative).resolve(strict=True)
    if not path.is_relative_to(release):
        raise ValueError("Release input escapes its root")
    expected = manifest["artifacts"].get(relative)
    if expected is None or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"Release manifest mismatch: {relative}")
    return path


def _inspect_checkpoint(path, inputs):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    args = saved["arguments"]
    for key, required in (("timestep", .005), ("policy_dt", .02),
                          ("controller", "implicit_pd"), ("refresh_kinematics", True)):
        if args.get(key) != required:
            raise ValueError(f"Checkpoint would override required migration setting: {key}")
    if saved["model_sha256"] != sha256_file(inputs["model"]) or saved["reference_sha256"] != sha256_file(inputs["reference"]):
        raise ValueError("Checkpoint/model/reference identity mismatch")
    schema = saved["controller_schema"]
    if schema["controller_mode"] != "implicit_pd" or not schema["refresh_kinematics"]:
        raise ValueError("Historical checkpoint lacks required native-controller semantics")
    return {"update": saved["update"], "observations": saved["observations"],
            "actions": saved["actions"], "arguments": args,
            "controller_schema": schema, "sha256": sha256_file(path),
            "loaded_on": "CPU for metadata inspection; no policy execution"}


def prepare(output, release=ROOT / "outputs/four-punch-milestone"):
    import numpy as np
    release = Path(release).resolve(strict=True)
    manifest_path = release / "manifest.json"
    if sha256_file(manifest_path) != RELEASE_SHA256:
        raise ValueError("Unexpected four-punch release manifest")
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Prepare with the pinned .venv-wbc interpreter")
    destination = writable_path(output)
    if not destination.is_relative_to(ROOT / "outputs/task10"):
        raise ValueError("Migration packets belong under outputs/task10")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    manifest = json.loads(manifest_path.read_text())
    files = {str(manifest_path): RELEASE_SHA256}
    tasks, checkpoints, parity = [], {}, {}
    for family in FAMILIES:
        learned_window = None
        for kind, old_kind in (("learned", "evaluation"), ("baseline", "baseline")):
            base = f"{family}/{old_kind}"
            inputs = {}
            for key, relative in (("model", f"{family}/model/human.xml"),
                                  ("reference", f"{family}/reference.npz"),
                                  ("metadata", f"{family}/body_metadata.json"),
                                  ("bank", f"{base}/evaluation_bank.npz"),
                                  ("bank_manifest", f"{family}/evaluation/evaluation_bank_manifest.json")):
                path = _checked_release_path(release, manifest, relative)
                inputs[key] = str(path)
                files[str(path)] = sha256_file(path)
            # jab_learning copies the source model directory, so include every
            # copied file even when only human.xml is passed to the compiler.
            for path in (release / family / "model").rglob("*"):
                if path.is_file():
                    _checked_release_path(release, manifest, str(path.relative_to(release)))
                    files[str(path)] = sha256_file(path)
            with np.load(inputs["bank"], allow_pickle=False) as bank:
                if bank["qpos"].shape != (64, 55) or bank["qvel"].shape != (64, 54):
                    raise ValueError("Expected the original 64-state constrained-body bank")
                if not np.isfinite(bank["qpos"]).all() or not np.isfinite(bank["qvel"]).all():
                    raise ValueError("Nonfinite historical bank")
                if str(bank["model_sha256"].item()) != sha256_file(inputs["model"]) or str(bank["reference_sha256"].item()) != sha256_file(inputs["reference"]):
                    raise ValueError("Historical bank model/reference mismatch")
            old_args = _checked_release_path(release, manifest, f"{base}/arguments.json")
            files[str(old_args)] = sha256_file(old_args)
            arguments = json.loads(old_args.read_text())
            if arguments["mode"] != "evaluate" or arguments.get("initialize_policy"):
                raise ValueError("Only preserved evaluation argument records are permitted")
            if kind == "learned":
                learned_window = arguments["attack_window"]
                checkpoint = _checked_release_path(release, manifest, f"{family}/checkpoint.pt")
                files[str(checkpoint)] = sha256_file(checkpoint)
                checkpoints[family] = _inspect_checkpoint(checkpoint, inputs)
            else:
                checkpoint = None
            # Evaluation restores checkpoint settings only when --checkpoint is
            # present. Make the zero-residual invocation explicitly identical.
            arguments = dict(checkpoints[family]["arguments"])
            arguments["attack_window"] = learned_window
            for field in CONTROLLER_FIELDS:
                if field not in arguments:
                    raise ValueError(f"Checkpoint lacks explicit parity field {field}")
            package = destination / "source_packages" / family / kind / "human_punching"
            package.mkdir(parents=True)
            for filename in ("__init__.py", "jab_evaluation.py", "contact.py"):
                source = ROOT / "src/human_punching" / filename
                shutil.copy2(source, package / filename)
                files[str(source)] = sha256_file(source)
            for name in ("jab_learning", "scalar_control"):
                # Both sides use the selected learned evaluator implementation;
                # changing only the residual avoids source-version confounding.
                source = _checked_release_path(release, manifest, f"{family}/evaluation/{name}_source.py")
                shutil.copy2(source, package / (name + ".py"))
                files[str(source)] = sha256_file(source)
            run = destination / "runs" / family / kind
            argv = build_argv(inputs, arguments, run, checkpoint=checkpoint)
            tasks.append({"family": family, "kind": kind, "inputs": inputs,
                          "checkpoint": str(checkpoint) if checkpoint else None,
                          "pythonpath": str(package.parent), "output": str(run), "argv": argv,
                          "expected_controller_arguments": {
                              **{k: arguments[k] for k in CONTROLLER_FIELDS},
                              "guard_joint_weight": arguments.get("guard_joint_weight", 0.)},
                          "expected_attack_window": learned_window})
        if sha256_file(tasks[-1]["inputs"]["bank"]) != sha256_file(tasks[-2]["inputs"]["bank"]):
            raise ValueError("Learned and zero-residual evaluations must use an identical bank")
        parity[family] = {"same_bank_sha256": sha256_file(tasks[-1]["inputs"]["bank"]),
                          "same_controller_arguments": tasks[-1]["expected_controller_arguments"],
                          "same_attack_window": learned_window,
                          "same_evaluator_source_sha256": sha256_file(Path(tasks[-1]["pythonpath"]) / "human_punching/jab_learning.py"),
                          "only_behavioral_difference": "selected policy residual versus all-zero residual"}
    # Freeze the independent offline evaluator as well as all executable inputs.
    shutil.copy2(ROOT / "scripts/audit_jab_bank.py", destination / "audit_jab_bank.py")
    for path in destination.rglob("*.py"):
        files[str(path)] = sha256_file(path)
    for path in (ROOT / "scripts/task10_migration.py", ROOT / "scripts/task10_migration.sbatch",
                 ROOT / "scripts/audit_jab_bank.py", ROOT / "requirements-wbc.lock",
                 ROOT / "requirements-wbc.in", ROOT / "src/human_punching/wbc/provenance.py"):
        files[str(path)] = sha256_file(path)
    # mjlab source identity matters even when its package version stays 1.6.0.
    mjlab = ROOT / ".cache/deps/mjlab-8ee51fbc"
    marker = json.loads((mjlab / ".task10-source.json").read_text())
    if marker.get("revision") != "8ee51fbcf806a7419189f706d9e394cbeb7790fa":
        raise ValueError("Unexpected pinned mjlab source export")
    for path in mjlab.rglob("*.py"):
        files[str(path)] = sha256_file(path)
    files[str(mjlab / ".task10-source.json")] = sha256_file(mjlab / ".task10-source.json")
    packet = destination / "packet.json"
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
               "--gpus=l40s:1", "--cpus-per-task=4", "--mem=32G", "--time=00:30:00",
               "--no-requeue", "--job-name=human-t09-migration", f"--chdir={ROOT}",
               f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
               str(ROOT / "scripts/task10_migration.sbatch"), str(packet)]
    record = {"schema": SCHEMA, "status": "prepared_not_submitted", "prepared_utc": now(),
              "tasks": tasks, "files": files, "runtime": runtime, "checkpoint_metadata": checkpoints,
              "paired_controller_contract": parity,
              "release_manifest_sha256": RELEASE_SHA256,
              "resources": {"gpus": 1, "gpu_type": "l40s", "cpus": 4, "memory_gib": 32,
                            "wall_seconds": 1800, "application_wall_seconds": 1740,
                            "new_products_and_run_caches_limit_bytes": MAX_BYTES},
              "historical_solver": {"integrator": "implicitfast", "solver": "newton", "cone": "elliptic",
                                    "iterations": 30, "ls_iterations": 15, "tolerance": 1e-7},
              "physics_dt_s": .005, "policy_dt_s": .020, "trials_per_family_and_mode": 64,
              "total_requested_cycles": 512, "submission_command": command,
              "training": False, "automatic_submissions_or_retries": False,
              "runtime_migration_qualified": False, "task10_acceptance": False,
              "comparison_scope": "Historical fixed-reference, fixed-target, family-specific guards; changed runtime needs measurement",
              "trace_limitations": ["Historical traces are not the Task 10 cycle_trace schema",
                 "No loaded material-patch slip, complete collision classification, or immutable G1 launch context",
                 "Historical support force decoder assumes elliptic constraints",
                 "Independent audit recomputes FK from recorded qpos; historical per-step support may be one interval old",
                 "Source copied unchanged; plots/videos and shared-policy claims are outside this replay"],
              "historical_expected": {family: {"learned_passes": 64, "baseline_passes": 0} for family in FAMILIES}}
    write_json(packet, record)
    print(shlex.join(command))
    return record


def _size(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Compiler caches may rename a temporary file during the scan.
            continue
    return total


def _run(command, *, environment, cwd, log, deadline, packet_root):
    stop = None
    sampled_bytes = _size(packet_root)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                sampled_bytes = max(sampled_bytes, _size(packet_root))
                if sampled_bytes > MAX_BYTES:
                    stop = "storage_limit"
                elif time.monotonic() >= deadline:
                    stop = "session_wall_limit"
                if stop:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    sampled_bytes = max(sampled_bytes, _size(packet_root))
    if sampled_bytes > MAX_BYTES:
        stop = "storage_limit"
    return {"returncode": code, "stop_reason": stop, "maximum_sampled_packet_bytes": sampled_bytes,
            "storage_sampling_sleep_seconds": 1, "sampling_includes_directory_scan_overhead": True,
            "command": command}


def _audit_argv(task, packet_root):
    output = Path(task["output"])
    args = [str(packet_root / "audit_jab_bank.py"), "--model", str(output / "model/human.xml"),
            "--runtime-model", str(output / "model/human_runtime.xml"), "--reference", str(output / "reference.npz"),
            "--rollouts", str(output), "--bank", task["inputs"]["bank"],
            "--bank-manifest", task["inputs"]["bank_manifest"], "--output", str(output / "independent"),
            "--workers", "4", "--require-physics-timestep", ".005", "--require-policy-timestep", ".02"]
    if task["checkpoint"]:
        args += ["--checkpoint", task["checkpoint"]]
    return args


def verify_task_paths(task, output):
    """Bind paths across the login/compute NFS mount aliases, without weakening confinement."""
    for flag, expected in (("--output", str(output)), ("--model", task["inputs"]["model"]),
            ("--reference", task["inputs"]["reference"]), ("--body-metadata", task["inputs"]["metadata"]),
            ("--evaluation-bank", task["inputs"]["bank"])):
        actual = task["argv"][task["argv"].index(flag)+1]
        if Path(actual).resolve() != Path(expected).resolve():
            raise ValueError(f"Prepared task path differs from command: {flag}")


def verify_replayed_contract(task):
    """Check effective restored settings, rather than trusting CLI intentions."""
    import mujoco
    output = Path(task["output"])
    actual = json.loads((output / "arguments.json").read_text())
    for field, expected in task["expected_controller_arguments"].items():
        if actual.get(field, 0. if field == "guard_joint_weight" else None) != expected:
            raise ValueError(f"Effective checkpoint/baseline setting differs: {field}")
    if actual["mode"] != "evaluate" or actual.get("initialize_policy"):
        raise ValueError("Generated run does not record evaluation-only execution")
    if actual["attack_window"] != task["expected_attack_window"]:
        raise ValueError("Effective attack window differs from the paired contract")
    model = mujoco.MjModel.from_xml_path(str(output / "model/human_runtime.xml"))
    expected = dict(timestep=.005, integrator=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
                    solver=int(mujoco.mjtSolver.mjSOL_NEWTON), cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC),
                    iterations=30, ls_iterations=15, tolerance=1e-7)
    for field, value in expected.items():
        if getattr(model.opt, field) != value:
            raise ValueError(f"Historical compiled solver mismatch: {field}")
    return {"effective_controller_arguments_match": True, "compiled_historical_solver": expected}


def strict_release_count(summary):
    """Task 09 release additionally required full/attack RMS and contact timing.

    The historical pilot_pass alone excludes these precision targets. Keep its
    descriptive count separately instead of silently treating it as release pass.
    """
    count = 0
    for row in summary["trials"]:
        precision = row["precision_targets"]
        attack_rms = (row["reference_tracking"] or {}).get("attack_active_glove_position_rms_m")
        side = {"Left": 0, "Right": 1}.get(row["active_hand"])
        distances = row["attack_minimum_glove_distance_m"]
        active_contact = side is not None and distances is not None and distances[side] <= .14
        passed = (row["pilot_pass"] and precision["active_glove_rms_target_met"]
                  and precision["contact_timing_target_met"] and attack_rms is not None
                  and attack_rms <= .02 and active_contact)
        count += bool(passed)
    return count


def execute(packet_path):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use an existing allocated Slurm GPU session")
    started = time.monotonic()
    packet_path = writable_path(packet_path).resolve(strict=True)
    packet = json.loads(packet_path.read_text())
    if packet.get("schema") != SCHEMA or packet.get("training") is not False:
        raise ValueError("Unsupported or training-enabled migration packet")
    if packet["resources"] != {"gpus": 1, "gpu_type": "l40s", "cpus": 4, "memory_gib": 32,
            "wall_seconds": 1800, "application_wall_seconds": 1740,
            "new_products_and_run_caches_limit_bytes": MAX_BYTES}:
        raise ValueError("Migration resource contract changed")
    if len(packet["tasks"]) != 8:
        raise ValueError("Exactly four learned/baseline pairs are required")
    expected = [(f, k) for f in FAMILIES for k in ("learned", "baseline")]
    if [(t["family"], t["kind"]) for t in packet["tasks"]] != expected:
        raise ValueError("Unexpected evaluation sequence")
    for filename, digest in packet["files"].items():
        if sha256_file(filename) != digest:
            raise ValueError(f"Frozen migration source/input changed: {filename}")
    if not runtime_record()["declared_pins_match"]:
        raise RuntimeError("Pinned migration runtime changed")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA device is required")
    destination = packet_path.parent
    run_root = writable_path(destination / "runs")
    if run_root.exists() or (destination / "execution.json").exists():
        raise FileExistsError("Refusing to overwrite or resume a migration attempt")
    run_root.mkdir()
    cache = destination / "cache"
    environment = os.environ.copy()
    for variable, name in (("XDG_CACHE_HOME", "xdg"), ("XDG_CONFIG_HOME", "config"),
                           ("WARP_CACHE_PATH", "warp"), ("CUDA_CACHE_PATH", "cuda"),
                           ("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch")):
        path = cache / name
        path.mkdir(parents=True, exist_ok=False)
        environment[variable] = str(path)
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    receipts = []
    deadline = started + 1740
    for task in packet["tasks"]:
        validate_argv(task["argv"])
        output = writable_path(task["output"])
        if not output.is_relative_to(run_root) or output.exists():
            raise ValueError("Each migration output must be new and inside this packet")
        verify_task_paths(task, output)
        if task["argv"].count("--checkpoint") != int(task["kind"] == "learned"):
            raise ValueError("Only learned replay may load the selected checkpoint")
        if time.monotonic() >= deadline:
            break
        # Historical code writes .cache/physics/warp relative to cwd. A common
        # packet-local cwd shares compilation products while preserving budget.
        cwd = destination / "native_work"
        cwd.mkdir(exist_ok=True)
        environment["PYTHONPATH"] = task["pythonpath"]
        name = task["family"] + "_" + task["kind"]
        command = [sys.executable, "-B", "-m", "human_punching.jab_learning", *task["argv"]]
        receipt = {"family": task["family"], "kind": task["kind"], "started_utc": now(),
                   "training": False, "source_packet_sha256": sha256_file(packet_path)}
        receipt["replay"] = _run(command, environment=environment, cwd=cwd,
            log=destination / "logs" / (name + "_replay.log"), deadline=deadline, packet_root=destination)
        healthy = receipt["replay"]["returncode"] == 0 and receipt["replay"]["stop_reason"] is None
        if healthy:
            try:
                receipt["effective_contract"] = verify_replayed_contract(task)
            except Exception as error:
                receipt["contract_error"] = {"type": type(error).__name__, "message": str(error)}
                healthy = False
        if healthy:
            audit_command = [sys.executable, "-B", *_audit_argv(task, destination)]
            receipt["audit"] = _run(audit_command, environment=environment, cwd=cwd,
                log=destination / "logs" / (name + "_audit.log"), deadline=deadline, packet_root=destination)
            healthy = receipt["audit"]["returncode"] == 0 and receipt["audit"]["stop_reason"] is None
        if healthy:
            summary = json.loads((output / "independent/summary.json").read_text())
            receipt.update(trial_count=summary["trial_count"], pilot_pass_count=summary["pilot_pass_count"],
                           strict_historical_release_pass_count=strict_release_count(summary),
                           precision_target_pass_counts=summary["precision_target_pass_counts"],
                           summary_sha256=sha256_file(output / "independent/summary.json"))
        receipt.update(finished_utc=now(), execution_complete=healthy)
        write_json(destination / (name + "_receipt.json"), receipt)
        receipts.append(receipt)
        if not healthy:
            break  # Preserve the bring-up defect; no automatic reruns.
    completed = len(receipts) == 8 and all(row["execution_complete"] for row in receipts)
    report = {"schema": "human_wbc_historical_migration_execution_v1", "job_id": os.environ["SLURM_JOB_ID"],
              "finished_utc": now(), "wall_seconds": time.monotonic()-started,
              "execution_complete": completed, "receipts": receipts,
              "unattempted": expected[len(receipts):], "packet_sha256": sha256_file(packet_path),
              "training": False, "task10_acceptance": False,
              "runtime_migration_qualified": False,
              "qualification_note": "Completed replay counts require independent review; no automatic promotion",
              "final_packet_bytes": _size(destination)}
    write_json(destination / "execution.json", report)
    return 0 if completed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
        return 0
    return execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
