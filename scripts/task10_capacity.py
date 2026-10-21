#!/usr/bin/env python3
"""Prepare/execute one bounded engineering capacity session; never submit jobs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from human_punching.wbc.provenance import runtime_record, sha256_file, writable_path, write_json
from human_punching.wbc.runtime.engineering_recipe import LEGACY_RECIPE_ID, RECIPE_IDS, resolve_engineering_recipe

SCHEMA = "human_wbc_engineering_capacity_packet_v1"
WALL_SECONDS = 1600
LIMIT_BYTES = 8*1024**3
ALLOCATION_ENVIRONMENT_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_NUM_NODES", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES", "SLURM_TRES_PER_NODE")


def allocation_environment():
    return {"schema": "human_wbc_capacity_allocation_environment_v1",
            "environment": {name: os.environ[name] for name in ALLOCATION_ENVIRONMENT_KEYS if name in os.environ}}


def submission_command(destination, *, gpus, exclude_node=None):
    """The local Torchrunx launcher needs every requested GPU on one node."""
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--nodes=1", "--ntasks=1", f"--gres=gpu:l40s:{gpus}", "--cpus-per-task=8", "--mem=64G",
        "--time=00:30:00", "--no-requeue", "--job-name=human-wbc-capacity", f"--chdir={ROOT}",
        f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
        str(ROOT/"scripts/task10_capacity.sbatch"), str(destination/"packet.json")]
    if exclude_node is not None:
        command.insert(2, "--exclude="+exclude_node)
    return command


def prepare(bundle, admission_path, observer_report, bridge_report, output, *, environments=8192, gpus=2,
            seed=10010, exclude_node=None, post_panel=False, engineering_recipe=LEGACY_RECIPE_ID,
            training_initialization_report=None):
    from human_punching.wbc.capacity import CapacityConfig
    from human_punching.wbc.capacity_evidence import (load_engineering_admission, observer_capacity_evidence,
                                                     bridge_capacity_evidence, training_initialization_capacity_evidence)
    from human_punching.wbc.engineering_admission import _evidence, read_artifact
    from human_punching.wbc.measurement_contract import build_measurement_contract
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    admission = load_engineering_admission(admission_path)
    observer = observer_capacity_evidence(checked, observer_report)
    bridge = bridge_capacity_evidence(checked, bridge_report)
    recipe = resolve_engineering_recipe(engineering_recipe)
    if recipe.recipe_id != LEGACY_RECIPE_ID and not post_panel:
        raise ValueError("Revised capacity requires its separate strict 64-episode panel")
    config = CapacityConfig(global_environments=environments, gpu_count=gpus, seed=seed, engineering_recipe=recipe.recipe_id)
    contract = build_measurement_contract(checked, scale=config.scale,
        engineering_admission=admission, observer_report=observer_report, bridge_report=bridge_report,
        engineering_recipe=recipe, training_initialization_report=training_initialization_report)
    initialization = contract["payload"]["reference_and_body"]["training_initialization_evidence"]
    destination = writable_path(output)
    if len(os.fsencode(destination/"run/cache/tmp")) > 120:
        raise ValueError("Use a shallow packet destination: resolved TMPDIR must be at most 120 bytes")
    if not runtime_record()["declared_pins_match"]:
        raise ValueError("Capacity preparation requires pinned runtime")
    if exclude_node is not None and not re.fullmatch(r"node[0-9]+", exclude_node):
        raise ValueError("Node exclusion must name one explicit cluster node")
    paths = list((ROOT/"src/human_punching").rglob("*.py"))
    paths += [ROOT/name for name in ("scripts/task10_capacity.py", "scripts/task10_capacity.sbatch",
        "requirements-wbc.in", "requirements-wbc.lock", "pyproject.toml")]
    paths += [p for p in checked.root.rglob("*") if p.is_file()]
    admission_record, observer_record = _evidence(admission_path), _evidence(observer_report)
    paths += [read_artifact(admission_record), read_artifact(observer_record)]
    native = admission.payload["native_controller"]
    paths += [read_artifact(native["report"])]
    paths += [read_artifact(record) for record in native["payloads"].values()]
    paths += [read_artifact(observer[name]) for name in ("execution", "packet")]
    paths += [read_artifact(item) for item in observer["additional_provenance"].values()]
    paths += [read_artifact({"path": name, "sha256": expected})
              for name, expected in observer["verified_physical_reference_files"].items()]
    paths += [read_artifact(record) for record in observer["trace_payloads"].values()]
    paths += [read_artifact(bridge[name]) for name in ("report", "execution", "packet")]
    paths += [read_artifact(item) for item in bridge["trace_payloads"].values()]
    if initialization is not None:
        paths += [read_artifact(initialization[name]) for name in ("report", "completion_manifest", "execution", "packet")]
        paths += [read_artifact(item) for item in initialization["trace_payloads"].values()]
    files = {str(path.resolve().relative_to(ROOT.resolve())): sha256_file(path) for path in sorted(set(paths))}
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"logs").mkdir()
    panel_record = None
    if post_panel:
        from human_punching.wbc.engineering_cycle_panel import build_engineering_recipes, write_engineering_recipes
        recipe_file = write_engineering_recipes(destination/"engineering_recipes.json",
            build_engineering_recipes(checked, episodes_per_family=16))
        panel_record = _evidence(recipe_file)
        files[panel_record["path"]] = panel_record["sha256"]
    for relative, expected in files.items():
        source, target = ROOT/relative, destination/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(source) != expected or sha256_file(target) != expected:
            raise ValueError("Prepared source/evidence changed during snapshot")
    packet_path = destination/"packet.json"
    command = submission_command(destination, gpus=gpus, exclude_node=exclude_node)
    record = {"schema": SCHEMA, "status": "prepared_not_submitted",
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "bundle": str(checked.root.relative_to(ROOT.resolve())),
        "engineering_admission": admission_record, "observer_report": observer_record,
        "bridge_report": bridge["report"],
        "engineering_recipe": recipe.manifest(), "engineering_recipe_sha256": recipe.sha256,
        "training_initialization_report": initialization["report"] if initialization is not None else None,
        "output": str((destination/"run").relative_to(ROOT.resolve())),
        "config": asdict(config), "measurement_contract": contract,
        "global_accounting": config.scale.report(config.warmup_iterations+config.measurement_iterations),
        "runtime": runtime_record(), "observer_enabled": True,
        "observer_native_report_status": observer["native_report_status"],
        "observer_strict_rollout_status": observer["strict_rollout_status"],
        "bridge_enabled": True, "task_objectives_enabled": True,
        "bridge_launch_qualification": bridge["launch_qualification"],
        "strict_readiness": bridge["readiness"],
        "post_measurement_panel_recipes": panel_record,
        "post_measurement_panel_episodes": 64 if post_panel else 0,
        "post_measurement_panel_scope": "disposable frozen capacity actor; separate timing/memory; no production acceptance",
        "production_training": False, "capacity_only": True, "scratch_initialization": True,
        "capacity_weights_may_initialize_training": False, "automatic_followup": False,
        "resources": {"nodes": 1, "tasks": 1, "gpus": gpus, "gpu_type": "l40s", "cpus": 8, "memory_gib": 64,
            "wall_seconds": 1800, "application_wall_seconds": WALL_SECONDS, "products_and_caches_bytes": LIMIT_BYTES},
        "authorization": "User standing authorization for GPU tests and sessions no longer than one hour",
        "submission_command": command}
    write_json(packet_path, record)
    print(shlex.join(command))
    return record


def tree_bytes(path):
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def execute(packet_path):
    from human_punching.wbc.engineering_admission import read_artifact
    packet_path = Path(packet_path).resolve(strict=True)
    record = json.loads(packet_path.read_text())
    if (record.get("schema") != SCHEMA or record.get("capacity_only") is not True
            or record.get("production_training") is not False or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("A prepared capacity-only packet inside Slurm is required")
    for relative, expected in record["files"].items():
        read_artifact({"path": relative, "sha256": expected})
    output = writable_path(ROOT/record["output"])
    if output.exists() or (packet_path.parent/"execution.json").exists():
        raise FileExistsError("A fresh packet is required for each attempted session")
    command = [sys.executable, "-B", "-u", "-m", "human_punching.wbc.capacity",
        "--bundle", str(ROOT/record["bundle"]), "--output", str(output),
        "--engineering-admission", str(read_artifact(record["engineering_admission"])),
        "--observer-report", str(read_artifact(record["observer_report"])),
        "--bridge-report", str(read_artifact(record["bridge_report"]))]
    for key, flag in (("global_environments", "global-environments"), ("gpu_count", "gpus"),
                      ("warmup_iterations", "warmup-iterations"), ("measurement_iterations", "measurement-iterations"),
                      ("seed", "seed"), ("conditioning", "conditioning"), ("engineering_recipe", "engineering-recipe")):
        command += ["--"+flag, str(record["config"][key])]
    if record.get("training_initialization_report") is not None:
        command += ["--training-initialization-report", str(read_artifact(record["training_initialization_report"]))]
    if record.get("post_measurement_panel_recipes") is not None:
        command += ["--panel-recipes", str(read_artifact(record["post_measurement_panel_recipes"]))]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    write_json(packet_path.parent/"allocation_environment.json", allocation_environment())
    begin, peak, stop = time.monotonic(), 0, None
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            peak = max(peak, tree_bytes(packet_path.parent))
            if peak > LIMIT_BYTES:
                stop = "products_and_caches_budget"
            elif time.monotonic()-begin >= WALL_SECONDS:
                stop = "application_wall_budget"
            if stop:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(2)
        code = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
    write_json(packet_path.parent/"execution.json", {"schema": "human_wbc_capacity_execution_v1",
        "returncode": code, "budget_stop": stop, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "packet_sha256": sha256_file(packet_path), "wall_s": time.monotonic()-begin,
        "peak_sampled_products_bytes": max(peak, tree_bytes(packet_path.parent)),
        "capacity_only": True, "production_training": False})
    return code if code else (2 if stop else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for name in ("bundle", "engineering-admission", "observer-report", "bridge-report", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--environments", type=int, default=8192)
    p.add_argument("--gpus", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--seed", type=int, default=10010)
    p.add_argument("--exclude-node")
    p.add_argument("--post-panel", action="store_true")
    p.add_argument("--engineering-recipe", choices=RECIPE_IDS, default=LEGACY_RECIPE_ID)
    p.add_argument("--training-initialization-report", type=Path)
    p = sub.add_parser("execute"); p.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "execute":
        return execute(args.packet)
    prepare(args.bundle, args.engineering_admission, args.observer_report, args.bridge_report, args.output,
        environments=args.environments, gpus=args.gpus, seed=args.seed, exclude_node=args.exclude_node,
        post_panel=args.post_panel, engineering_recipe=args.engineering_recipe,
        training_initialization_report=args.training_initialization_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
