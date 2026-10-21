#!/usr/bin/env python3
"""Prepare or execute one measured canonical learning session; never submit it.

The user authorizes GPU sessions up to one hour. A longer prepared allocation
needs the user's separate authorization before submission. Preparation does not
grant that authorization, start PPO, chain sessions, or promote a checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
from human_punching.wbc.provenance import sha256_file, writable_path, write_json
from human_punching.wbc.runtime.engineering_recipe import (LEGACY_RECIPE_ID, RECIPE_IDS,
    verify_recipe_manifest)

SCHEMA = "human_wbc_engineering_training_packet_v1"


def resource_budget(wall_minutes, products_gib):
    if type(wall_minutes) is not int or not 10 <= wall_minutes <= 1440:
        raise ValueError("One explicit allocation of 10..1440 minutes is required")
    if type(products_gib) is not int or not 4 <= products_gib <= 32:
        raise ValueError("An explicit 4..32 GiB products/cache budget is required")
    return {"nodes": 1, "tasks": 1, "gpus": 2, "gpu_type": "l40s", "cpus": 8, "memory_gib": 64,
        "wall_seconds": wall_minutes * 60, "application_wall_seconds": wall_minutes * 60 - 120,
        "products_and_caches_bytes": products_gib * 1024**3}


def measured_wall_budget(plan, resources):
    """Reserve one complete measured session, without splitting its budget."""
    projection = plan["projection"]
    names = ("learning_seconds", "measured_initialization_seconds", "panel_seconds")
    values = [projection[name] for name in names]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in values):
        raise ValueError("Positive finite measured learning, startup and validation costs are required")
    estimate = sum(values)
    reserved = math.ceil(1.25 * estimate)
    if resources["application_wall_seconds"] < reserved:
        raise ValueError(f"The complete measured session needs at least {reserved + 120} allocation seconds "
                         "including 25% variability and shutdown reserves; choose a sufficient wall budget")
    return {"measured_projection_seconds": estimate, "variability_multiplier": 1.25,
        "required_application_seconds": reserved, "shutdown_reserve_seconds": 120,
        "qualification": "Capacity-based estimate, not a guaranteed completion time"}


def prepare(args):
    resources = resource_budget(args.wall_minutes, args.products_gib)
    from human_punching.wbc.engineering_training import checked_engineering_plan
    plan = checked_engineering_plan(args.bundle, args.engineering_admission, args.observer_report,
        args.bridge_report, args.capacity_report, iterations=args.iterations, seed=args.seed,
        resume=args.resume, resume_manifest_sha256=args.resume_manifest_sha256,
        engineering_recipe=args.engineering_recipe, training_initialization_report=args.training_initialization_report)
    recipe = verify_recipe_manifest(plan["engineering_recipe"], plan["engineering_recipe_sha256"])
    estimate = measured_wall_budget(plan, resources)
    destination = writable_path(args.output)
    if len(os.fsencode(destination / "run/cache/tmp")) > 120:
        raise ValueError("Use a shallow packet destination for CUDA compiler scratch")
    # The checked plan owns scientific admission and the measured configuration.
    # This wrapper owns allocation, frozen files, and bounded process lifetime.
    files = dict(plan["input_files"])
    for relative in ("scripts/task10_engineering_training.py", "scripts/task10_engineering_training.sbatch"):
        files[relative] = sha256_file(ROOT / relative)
    for relative, expected in files.items():
        path = (ROOT / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != expected:
            raise ValueError("Prepared plan has an invalid or changed project input: " + relative)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    for relative, expected in files.items():
        source, copied = ROOT / relative, destination / "source_snapshot" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        if sha256_file(source) != expected or sha256_file(copied) != expected:
            raise ValueError("Source changed while preparing the learning packet")
    write_json(destination / "compute_plan.json", plan)
    minutes = args.wall_minutes
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--nodes=1", "--ntasks=1", "--gres=gpu:l40s:2", "--cpus-per-task=8", "--mem=64G", f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
        "--no-requeue", "--exclude=node3006", "--job-name=human-wbc-canonical",
        f"--chdir={ROOT}", f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
        str(ROOT / "scripts/task10_engineering_training.sbatch"), str(destination / "packet.json")]
    def local(path):
        if path is None:
            return None
        return str(Path(path).resolve(strict=True).relative_to(ROOT.resolve()))
    record = {"schema": SCHEMA, "status": "prepared_not_submitted",
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "bundle": local(args.bundle), "engineering_admission": local(args.engineering_admission),
        "observer_report": local(args.observer_report), "bridge_report": local(args.bridge_report),
        "capacity_report": local(args.capacity_report), "resume": local(args.resume),
        "engineering_recipe": recipe.recipe_id, "engineering_recipe_sha256": recipe.sha256,
        "training_initialization_report": local(args.training_initialization_report),
        "resume_manifest_sha256": args.resume_manifest_sha256,
        "compute_plan_sha256": sha256_file(destination / "compute_plan.json"),
        "output": str((destination / "run").relative_to(ROOT.resolve())),
        "iterations": args.iterations, "seed": args.seed, "resources": resources,
        "wall_budget_basis": estimate,
        "scope": "canonical_four_family_scratch_engineering", "production_qualified": False,
        "automatic_followup": False, "promotion": False,
        "long_session_approval_required_before_submission": args.wall_minutes > 60,
        "authorization": "Standing authorization covers sessions up to one hour; longer sessions require a separate user instruction",
        "submission_command": command}
    write_json(destination / "packet.json", record)
    print(json.dumps({"packet": str(destination / "packet.json"),
        "packet_sha256": sha256_file(destination / "packet.json"),
        "long_session_approval_required": args.wall_minutes > 60,
        "submission_command": shlex.join(command)}))
    return record


def tree_bytes(root):
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except FileNotFoundError:
            pass  # Compiler temporary files may disappear during the sample.
    return size


def execute(packet_path):
    path = Path(packet_path).resolve(strict=True)
    record = json.loads(path.read_text())
    if (record.get("schema") != SCHEMA or not os.environ.get("SLURM_JOB_ID")
            or record.get("scope") != "canonical_four_family_scratch_engineering"
            or record.get("production_qualified") is not False
            or record.get("automatic_followup") is not False or record.get("promotion") is not False):
        raise ValueError("A prepared engineering-only session inside allocated Slurm is required")
    resources = record["resources"]
    seconds, byte_budget = resources.get("wall_seconds"), resources.get("products_and_caches_bytes")
    if (type(seconds) is not int or seconds % 60 or type(byte_budget) is not int or byte_budget % 1024**3
            or resources != resource_budget(seconds // 60, byte_budget // 1024**3)):
        raise ValueError("Prepared session resource bounds differ")
    plan_path = path.parent / "compute_plan.json"
    if sha256_file(plan_path) != record.get("compute_plan_sha256"):
        raise ValueError("Prepared measured compute plan changed")
    plan = json.loads(plan_path.read_text())
    recipe = verify_recipe_manifest(plan.get("engineering_recipe"), plan.get("engineering_recipe_sha256"))
    expected_initialization = plan.get("inputs", {}).get("training_initialization_report")
    if (record.get("engineering_recipe") != recipe.recipe_id
            or record.get("engineering_recipe_sha256") != recipe.sha256
            or record.get("training_initialization_report") !=
                (expected_initialization["path"] if expected_initialization else None)):
        raise ValueError("Prepared packet recipe or initialization evidence differs from its measured plan")
    if record.get("wall_budget_basis") != measured_wall_budget(plan, resources):
        raise ValueError("Prepared allocation no longer covers the measured complete session")
    for relative, expected in record["files"].items():
        source = (ROOT / relative).resolve(strict=True)
        if Path(relative).is_absolute() or not source.is_relative_to(ROOT.resolve()) or sha256_file(source) != expected:
            raise ValueError("Frozen learning input changed: " + relative)
    output = writable_path(ROOT / record["output"])
    if output.exists() or (path.parent / "execution.json").exists():
        raise FileExistsError("Each attempted session requires a fresh prepared packet")
    command = [sys.executable, "-B", "-u", "-m", "human_punching.wbc.engineering_training"]
    for name in ("bundle", "engineering_admission", "observer_report", "bridge_report", "capacity_report"):
        command += ["--" + name.replace("_", "-"), str(ROOT / record[name])]
    command += ["--output", str(output), "--compute-plan", str(plan_path),
        "--iterations", str(record["iterations"]), "--seed", str(record["seed"]),
        "--engineering-recipe", recipe.recipe_id]
    if record.get("training_initialization_report") is not None:
        command += ["--training-initialization-report", str(ROOT / record["training_initialization_report"])]
    if record.get("resume"):
        command += ["--resume", str(ROOT / record["resume"]),
            "--resume-manifest-sha256", record["resume_manifest_sha256"]]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    write_json(path.parent / "allocation_environment.json", {name: os.environ.get(name) for name in (
        "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_NUM_NODES",
        "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES", "SLURM_TRES_PER_NODE")})
    started, peak, stopped = time.monotonic(), 0, None
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            peak = max(peak, tree_bytes(path.parent))
            if peak > byte_budget:
                stopped = "products_and_caches_budget"
            elif time.monotonic() - started > resources["application_wall_seconds"]:
                stopped = "application_wall_budget"
            if stopped:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(2)
        returncode = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    write_json(path.parent / "execution.json", {"schema": "human_wbc_engineering_training_execution_v1",
        "returncode": returncode, "budget_stop": stopped, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "packet_sha256": sha256_file(path), "wall_s": time.monotonic() - started,
        "peak_sampled_products_bytes": max(peak, tree_bytes(path.parent)),
        "scope": record["scope"], "production_qualified": False, "automatic_followup": False})
    return returncode if returncode else (2 if stopped else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for name in ("bundle", "engineering-admission", "observer-report", "bridge-report", "capacity-report", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--wall-minutes", type=int, required=True)
    p.add_argument("--products-gib", type=int, default=20)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-manifest-sha256")
    p.add_argument("--engineering-recipe", choices=RECIPE_IDS, default=LEGACY_RECIPE_ID)
    p.add_argument("--training-initialization-report", type=Path)
    p = sub.add_parser("execute")
    p.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
        return 0
    return execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
