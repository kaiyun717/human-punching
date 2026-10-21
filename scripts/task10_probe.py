#!/usr/bin/env python3
"""Prepare a bounded diagnostic packet or execute it inside an existing job.

Preparation prints a reviewable sbatch command but never submits it. The packet
binds the exact source/input files; execution rejects changed files. This probe
does not optimize, certify references, measure PPO capacity, or start later jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import (  # noqa: E402
    configure_local_caches, runtime_record, sha256_file, writable_path, write_json,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory(bundle: Path) -> dict[str, str]:
    # Bind the probe's source closure, not unrelated planners still being
    # developed while this diagnostic waits in the scheduler.
    paths = [ROOT / "src/human_punching/__init__.py"]
    paths += [ROOT / "src/human_punching/wbc" / name for name in (
        "__init__.py", "budget.py", "provenance.py", "cli.py", "native_probe.py")]
    paths += [ROOT / "src/human_punching/wbc/runtime" / name for name in (
        "__init__.py", "action.py", "body.py", "bundle.py", "command.py", "mdp.py", "task.py")]
    paths += [ROOT / name for name in (
        "scripts/task10_probe.py", "scripts/task10_probe.sbatch", "requirements-wbc.lock",
        "requirements-wbc.in", "pyproject.toml")]
    paths += [p for p in bundle.rglob("*") if p.is_file()]
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def prepare(bundle: Path, output: Path) -> dict:
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    bundle = bundle.resolve(strict=True)
    if not (bundle / "manifest.json").is_file():
        raise ValueError("Converted reference manifest is missing")
    (destination / "logs").mkdir()
    packet_path = destination / "packet.json"
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
               "--gpus=l40s:1", "--cpus-per-task=4", "--mem=16G", "--time=00:15:00",
               "--no-requeue", "--job-name=human-wbc-probe", f"--chdir={ROOT}",
               f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
               str(ROOT / "scripts/task10_probe.sbatch"), str(packet_path)]
    record = {
        "schema": "human_wbc_native_probe_packet_v1", "prepared_utc": now(),
        "status": "prepared_not_submitted", "bundle": str(bundle),
        "reference_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "output": str(destination / "run"), "environments": 16, "policy_steps": 50,
        "physics_steps_per_environment": 200, "simulated_seconds_per_environment": 1.0,
        "resources": {"account": "mit_general", "partition": "mit_normal_gpu",
                      "gpu_type": "l40s", "gpus": 1, "cpus": 4, "memory_gib": 16,
                      "wall_seconds": 900, "application_wall_seconds": 840,
                      "new_products_and_run_caches_limit_bytes": 2 * 1024**3},
        "runtime": runtime_record(), "files": inventory(bundle),
        "submission_command": command, "training": False, "automatic_retry": False,
        "quota_check_required_before_submission": True,
        "allocation_approval_required": False,
        "authorization": "2026-10-19 user standing authorization for GPU tests; training/evaluation sessions over one hour require approval",
    }
    snapshots = {}
    for name, digest in record["files"].items():
        source = Path(name)
        copied = destination / "source_snapshot" / source.relative_to(ROOT)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        if sha256_file(copied) != digest:
            raise ValueError(f"Source changed during packet preparation: {name}")
        snapshots[str(copied.relative_to(destination))] = digest
    write_json(destination / "source_snapshot.json", snapshots)
    write_json(packet_path, record)
    print(shlex.join(command))
    return record


def execute(packet_path: Path) -> int:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Execution requires an existing Slurm CUDA allocation")
    packet_path = writable_path(packet_path).resolve(strict=True)
    record = json.loads(packet_path.read_text())
    if record.get("schema") != "human_wbc_native_probe_packet_v1":
        raise ValueError("Unsupported probe packet")
    for name, digest in record["files"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"Prepared source/input changed: {name}")
    if not runtime_record()["declared_pins_match"]:
        raise RuntimeError("Runtime package pins changed after preparation")
    output = writable_path(record["output"])
    if output.exists():
        raise FileExistsError("This probe output already exists; never overwrite an attempt")
    configure_local_caches()
    command = [sys.executable, "-B", "-m", "human_punching.wbc.cli", "native-probe",
               "--bundle", record["bundle"], "--output", str(output),
               "--environments", str(record["environments"]),
               "--policy-steps", str(record["policy_steps"])]
    start = time.monotonic()
    stop_reason = None
    peak_sampled = 0
    # MuJoCo's native warning log follows cwd; keep it with this attempt too.
    process = subprocess.Popen(command, cwd=packet_path.parent)
    try:
        while process.poll() is None:
            size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) if output.exists() else 0
            peak_sampled = max(peak_sampled, size)
            if size > record["resources"]["new_products_and_run_caches_limit_bytes"]:
                stop_reason = "storage_limit"
            elif time.monotonic() - start >= record["resources"]["application_wall_seconds"]:
                stop_reason = "application_wall_limit"
            if stop_reason:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(5)
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    write_json(packet_path.parent / "execution.json", {
        "schema": "human_wbc_native_probe_execution_v1", "job_id": os.environ["SLURM_JOB_ID"],
        "host": os.uname().nodename, "finished_utc": now(), "wall_seconds": time.monotonic()-start,
        "returncode": code, "stop_reason": stop_reason, "maximum_sampled_output_bytes": peak_sampled,
        "storage_sampling_period_seconds": 5, "packet_sha256": sha256_file(packet_path),
        "training": False, "command": command,
    })
    return code if code else (1 if stop_reason else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--bundle", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.bundle, args.output)
        return 0
    return execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
