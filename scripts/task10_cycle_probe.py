#!/usr/bin/env python3
"""Prepare/run a bounded full-cycle diagnostic; never submit or train."""
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

SCHEMA = "human_wbc_cycle_probe_packet_v3"
LIMIT_BYTES = 4 * 1024**3


def prepare(bundle: Path, phases: Path | None, output: Path, *, mode: str = "cycle",
            feedforward_artifact: Path | None = None) -> dict:
    if mode not in ("cycle", "guard") or (mode == "guard" and phases is not None):
        raise ValueError("Choose cycle or static guard; guard probes have no punch phases")
    if feedforward_artifact is not None and mode != "guard":
        raise ValueError("Static guard feedforward cannot be applied to moving punches")
    bundle = bundle.resolve(strict=True)
    phases = None if phases is None else phases.resolve(strict=True)
    feedforward_artifact = (None if feedforward_artifact is None else
                           feedforward_artifact.resolve(strict=True))
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    source = ROOT / "src/human_punching"
    paths = [source / "__init__.py"]
    paths += [source / "wbc" / name for name in (
        "__init__.py", "native_cycle_probe.py", "cycle_trace.py", "native_probe.py",
        "contracts.py", "evaluation.py", "provenance.py", "guard_probe.py",
        "guard_feedforward.py", "effort_diagnostics.py", "references.py")]
    paths += [source / "wbc/runtime" / name for name in (
        "__init__.py", "task.py", "action.py", "body.py", "command.py", "bundle.py", "mdp.py")]
    paths += [ROOT / name for name in ("scripts/task10_cycle_probe.py", "scripts/task10_cycle_probe.sbatch",
                                      "requirements-wbc.lock", "requirements-wbc.in", "pyproject.toml")]
    paths += [p for p in bundle.rglob("*") if p.is_file()]
    if phases is not None:
        paths.append(phases)
    if feedforward_artifact is not None:
        paths.append(feedforward_artifact)
    files = {str(p): sha256_file(p) for p in sorted(set(paths))}
    snapshots = {}
    for name, expected in files.items():
        path = Path(name)
        copied = destination / "source_snapshot" / path.relative_to(ROOT)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
        if sha256_file(copied) != expected:
            raise ValueError(f"Input changed while preparing: {path}")
        snapshots[str(copied.relative_to(destination))] = expected
    write_json(destination / "source_snapshot.json", snapshots)
    packet = destination / "packet.json"
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--gpus=l40s:1", "--cpus-per-task=4", "--mem=32G", "--time=00:20:00", "--no-requeue",
        f"--job-name=human-wbc-{mode}", f"--chdir={ROOT}",
        f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
        str(ROOT / "scripts/task10_cycle_probe.sbatch"), str(packet)]
    record = {"schema": SCHEMA, "status": "prepared_not_submitted", "files": files,
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "bundle": str(bundle),
        "phase_manifest": None if phases is None else str(phases),
        "feedforward_artifact": None if feedforward_artifact is None else str(feedforward_artifact),
        "probe_mode": mode, "output": str(destination / "run"),
        "runtime": runtime_record(), "environments": 16, "simulated_seconds": 6.,
        "policy_steps": 300, "physics_steps_per_environment": 1200,
        "resources": {"gpus": 1, "gpu_type": "l40s", "cpus": 4, "memory_gib": 32,
                      "wall_seconds": 1200, "application_wall_seconds": 1140,
                      "new_products_and_run_caches_limit_bytes": LIMIT_BYTES},
        "training": False, "acceptance_evidence": False, "automatic_retry": False,
        "authorization": "User standing authorization for GPU tests and sessions up to one hour",
        "submission_command": command}
    write_json(packet, record)
    print(shlex.join(command))
    return record


def size(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Compiler caches can atomically rename temporary files.
    return total


def execute(packet: Path) -> int:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use an existing Slurm GPU allocation")
    packet = writable_path(packet).resolve(strict=True)
    record = json.loads(packet.read_text())
    if record.get("schema") != SCHEMA or record.get("training") is not False:
        raise ValueError("Invalid diagnostic packet")
    for path, digest in record["files"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Prepared source/input changed: {path}")
    if not runtime_record()["declared_pins_match"]:
        raise RuntimeError("Pinned runtime changed")
    output = writable_path(record["output"])
    if output.exists() or (packet.parent / "execution.json").exists():
        raise FileExistsError("Each diagnostic attempt must use a new output directory")
    mode = record.get("probe_mode")
    if mode not in ("cycle", "guard") or (mode == "guard" and record["phase_manifest"] is not None):
        raise ValueError("Invalid diagnostic mode")
    if record["feedforward_artifact"] is not None and mode != "guard":
        raise ValueError("Static feedforward is restricted to the static guard")
    module = "guard_probe" if mode == "guard" else "native_cycle_probe"
    command = [sys.executable, "-B", "-u", "-m", f"human_punching.wbc.{module}",
               "--bundle", record["bundle"], "--output", str(output),
               "--environments", "16", "--seconds", "6"]
    if record["phase_manifest"] is not None:
        command += ["--phases", record["phase_manifest"]]
    if record["feedforward_artifact"] is not None:
        command += ["--feedforward-artifact", record["feedforward_artifact"]]
    started, peak, reason = time.monotonic(), 0, None
    child = subprocess.Popen(command, cwd=packet.parent, start_new_session=True)
    try:
        while child.poll() is None:
            peak = max(peak, size(packet.parent))
            if peak > LIMIT_BYTES:
                reason = "storage_limit"
            elif time.monotonic() - started >= 1140:
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
    peak = max(peak, size(packet.parent))
    write_json(packet.parent / "execution.json", {
        "schema": "human_wbc_cycle_probe_execution_v1", "job_id": os.environ["SLURM_JOB_ID"],
        "host": os.uname().nodename, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "packet_sha256": sha256_file(packet), "returncode": code,
        "wall_seconds": time.monotonic()-started, "stop_reason": reason,
        "maximum_sampled_packet_bytes": peak, "storage_sampling_sleep_seconds": 2,
        "training": False})
    return code if code else int(reason is not None or peak > LIMIT_BYTES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--bundle", type=Path, required=True)
    prep.add_argument("--phases", type=Path)
    prep.add_argument("--mode", choices=("cycle", "guard"), default="cycle")
    prep.add_argument("--feedforward-artifact", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.bundle, args.phases, args.output, mode=args.mode,
                feedforward_artifact=args.feedforward_artifact)
    else:
        raise SystemExit(execute(args.packet))
