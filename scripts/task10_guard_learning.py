#!/usr/bin/env python3
"""Prepare or execute one bounded scratch guard session; never submit itself."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = "human_wbc_bounded_guard_session_packet_v1"
LIMIT_BYTES = 6 * 1024**3
WALL_SECONDS = 2580


def prepare(bundle: Path, native_guard_report: Path, support_artifact: Path,
            output: Path, *, environments: int = 256, iterations: int = 64,
            seed: int = 10010, exclude_node: str | None = None) -> dict:
    from human_punching.wbc.guard_training import GuardSmokePlan, prepare_guard_admission
    bundle = bundle.resolve(strict=True)
    report = native_guard_report.resolve(strict=True)
    support = support_artifact.resolve(strict=True)
    plan = GuardSmokePlan(environments=environments, iterations=iterations, seed=seed)
    admission = prepare_guard_admission(bundle, report, support)
    destination = writable_path(output)
    # CUDA NVRTC rejects long temporary-directory names on the installed stack.
    # Require a short native path up front; do not redirect outside this packet.
    predicted_tmp = destination / "run/cache/tmp"
    if len(os.fsencode(predicted_tmp)) > 120:
        raise ValueError("Use a shallow output path: resolved run/cache/tmp must be <=120 bytes")
    if not runtime_record()["declared_pins_match"]:
        raise RuntimeError("Pinned runtime differs before preparation")
    if exclude_node is not None and not re.fullmatch(r"node[0-9]+", exclude_node):
        raise ValueError("Node exclusion must be one explicit cluster node name")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    paths = list((ROOT / "src/human_punching/wbc").rglob("*.py"))
    paths += [ROOT / "src/human_punching/__init__.py"]
    paths += [ROOT / name for name in ("scripts/task10_guard_learning.py",
        "scripts/task10_guard_learning.sbatch", "requirements-wbc.lock", "requirements-wbc.in", "pyproject.toml")]
    paths += [p for p in bundle.rglob("*") if p.is_file()]
    paths += [report, support]
    paths += [Path(value["path"]) for value in admission["native_traces"].values()]
    files = {str(p): sha256_file(p) for p in sorted(set(paths))}
    for filename, expected in files.items():
        original = Path(filename)
        copy = destination / "source_snapshot" / original.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copy)
        if sha256_file(copy) != expected:
            raise ValueError(f"Input changed during preparation: {original}")
    packet = destination / "packet.json"
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--gpus=l40s:1", "--cpus-per-task=4", "--mem=32G", "--time=00:45:00", "--no-requeue",
        "--job-name=human-wbc-guard-learn", f"--chdir={ROOT}",
        f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
        str(ROOT / "scripts/task10_guard_learning.sbatch"), str(packet)]
    if exclude_node is not None:
        command.insert(2, "--exclude=" + exclude_node)
    record = {"schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_submitted", "files": files, "runtime": runtime_record(),
        "bundle": str(bundle), "native_guard_report": str(report), "support_artifact": str(support),
        "output": str(destination / "run"), "admission": admission,
        "plan": {"environments": environments, "iterations": iterations, "seed": seed},
        "smoke_accounting": plan.accounting(),
        "capacity": {"warmup_iterations": 2, "measurement_iterations": 3,
                     "fresh_transitions": 5 * environments * 24},
        "sequence": "capacity; fresh scratch guard learning only if measured resource/finite gates pass; physical guard diagnostic",
        "resources": {"gpus": 1, "gpu_type": "l40s", "memory_gib": 32, "cpus": 4,
                      "excluded_node": exclude_node,
                      "wall_seconds": 2700, "application_wall_seconds": WALL_SECONDS,
                      "new_products_and_run_caches_limit_bytes": LIMIT_BYTES},
        "scratch_initialization": True, "historical_initialization": False,
        "punch_training": False, "production_training": False, "automatic_retry": False,
        "authorization": "User standing authority for training/evaluation sessions no longer than one hour",
        "submission_command": command}
    write_json(packet, record)
    print(shlex.join(command))
    return record


def packet_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Atomic compiler-cache renames are expected.
    return total


def execute(packet: Path) -> int:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use an allocated Slurm GPU session")
    packet = writable_path(packet).resolve(strict=True)
    record = json.loads(packet.read_text())
    if (record.get("schema") != SCHEMA or record.get("production_training") is not False
            or record.get("punch_training") is not False or record.get("scratch_initialization") is not True):
        raise ValueError("Invalid bounded guard packet")
    for filename, expected in record["files"].items():
        if sha256_file(filename) != expected:
            raise ValueError(f"Prepared source/input changed: {filename}")
    if not runtime_record()["declared_pins_match"]:
        raise RuntimeError("Pinned runtime changed")
    output = writable_path(record["output"])
    if output.exists() or (packet.parent / "execution.json").exists():
        raise FileExistsError("Each attempted session requires a fresh output packet")
    preflight = {"host": os.uname().nodename,
        "environment": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "cuda_visibility_changed_by_launcher": False}
    try:
        check = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        preflight.update(nvidia_smi_returncode=check.returncode, nvidia_smi_output=check.stdout,
                         nvidia_smi_error=check.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        preflight["nvidia_smi_exception"] = str(error)
    write_json(packet.parent / "device_preflight.json", preflight)
    command = [sys.executable, "-B", "-u", "-m", "human_punching.wbc.guard_training",
        "--bundle", record["bundle"], "--native-guard-report", record["native_guard_report"],
        "--support-artifact", record["support_artifact"], "--output", str(output)]
    for name in ("environments", "iterations", "seed"):
        command += ["--" + name, str(record["plan"][name])]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started, peak, reason = time.monotonic(), 0, None
    child = subprocess.Popen(command, cwd=packet.parent, env=environment, start_new_session=True)
    try:
        while child.poll() is None:
            peak = max(peak, packet_bytes(packet.parent))
            if peak > LIMIT_BYTES:
                reason = "storage_limit"
            elif time.monotonic() - started >= WALL_SECONDS:
                reason = "wall_limit"
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(2)
        code = child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    peak = max(peak, packet_bytes(packet.parent))
    write_json(packet.parent / "execution.json", {"schema": "human_wbc_bounded_guard_execution_v1",
        "job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "packet_sha256": sha256_file(packet), "returncode": code,
        "wall_seconds": time.monotonic()-started, "stop_reason": reason,
        "maximum_sampled_packet_bytes": peak, "storage_sampling_sleep_seconds": 2,
        "scope": "scratch guard-only engineering learning; no punch qualification"})
    return code if code else int(reason is not None or peak > LIMIT_BYTES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("bundle", "native-guard-report", "support-artifact", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--environments", type=int, default=256)
    prep.add_argument("--iterations", type=int, default=64)
    prep.add_argument("--seed", type=int, default=10010)
    prep.add_argument("--exclude-node")
    run = commands.add_parser("execute")
    run.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.bundle, args.native_guard_report, args.support_artifact, args.output,
                environments=args.environments, iterations=args.iterations, seed=args.seed,
                exclude_node=args.exclude_node)
    else:
        raise SystemExit(execute(args.packet))
