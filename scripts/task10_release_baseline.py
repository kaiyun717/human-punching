#!/usr/bin/env python3
"""Freeze or execute a bounded Task09-controller comparison inside WBC; no PPO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import configure_local_caches, sha256_file, writable_path, write_json

SCHEMA = "human_wbc_release_baseline_packet_v1"


def required_inputs(record):
    """Reconstruct the required local input closure independently of the packet."""
    paths = list((ROOT / "src/human_punching/wbc").rglob("*.py"))
    paths += [ROOT / name for name in (
        "src/human_punching/__init__.py", "src/human_punching/scalar_control.py",
        "scripts/task10_release_baseline.py", "scripts/task10_release_baseline.sbatch",
        "requirements-wbc.lock", "requirements-wbc.in", "pyproject.toml")]
    bundle = (ROOT / record["bundle"]).resolve(strict=True)
    release = (ROOT / record["release"]).resolve(strict=True)
    if not bundle.is_relative_to(ROOT.resolve()) or not release.is_relative_to(ROOT.resolve()):
        raise ValueError("Diagnostic inputs must remain inside the human project")
    paths += [p for p in bundle.rglob("*") if p.is_file()]
    paths.append(release / "manifest.json")
    for family in ("jab", "cross", "left_hook", "right_hook"):
        paths += [release / family / name for name in (
            "checkpoint.pt", "controller_schema.json", "body_metadata.json", "arguments.json",
            "reference.npz", "model/human.xml", "jab_learning_source.py", "scalar_control_source.py",
            "evaluation/evaluation_bank.npz", "evaluation/evaluation_bank_manifest.json")]
    return {str(p.resolve(strict=True).relative_to(ROOT.resolve())) for p in paths}


def prepare(bundle, release, output):
    configure_local_caches()
    from human_punching.wbc.release_baseline import DIAGNOSTIC_SCOPE, FAMILIES
    from human_punching.wbc.release_baseline_probe import load_release_inputs
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    release = Path(release).resolve(strict=True)
    destination = writable_path(output)
    if (checked.manifest.get("diagnostic_scope") != DIAGNOSTIC_SCOPE
            or checked.manifest.get("qualified_for_training") is not False
            or checked.manifest.get("release_manifest_sha256") != sha256_file(release / "manifest.json")):
        raise ValueError("Expected an identified historical diagnostic reference bundle")
    load_release_inputs(checked, release)  # Full CPU preflight before making a runnable packet.
    if len(os.fsencode(destination / "run/cache/tmp")) > 120:
        raise ValueError("Use a shallow output path for native compiler temporary files")
    paths = list((ROOT / "src/human_punching/wbc").rglob("*.py"))
    paths += [ROOT / "src/human_punching/__init__.py", ROOT / "src/human_punching/scalar_control.py",
              Path(__file__), ROOT / "scripts/task10_release_baseline.sbatch",
              ROOT / "requirements-wbc.lock", ROOT / "requirements-wbc.in", ROOT / "pyproject.toml"]
    paths += [p for p in checked.root.rglob("*") if p.is_file()]
    release_manifest = json.loads((release / "manifest.json").read_text())
    paths.append(release / "manifest.json")
    for family in FAMILIES:
        for name in ("checkpoint.pt", "controller_schema.json", "body_metadata.json", "arguments.json",
                     "reference.npz", "model/human.xml", "jab_learning_source.py", "scalar_control_source.py",
                     "evaluation/evaluation_bank.npz", "evaluation/evaluation_bank_manifest.json"):
            relative = f"{family}/{name}"
            path = release / relative
            if sha256_file(path) != release_manifest["artifacts"][relative]["sha256"]:
                raise ValueError("Historical diagnostic input changed: " + relative)
            paths.append(path)
    files = {}
    for path in sorted(set(paths)):
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Input outside human project root")
        files[str(path.relative_to(ROOT.resolve()))] = sha256_file(path)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    for relative, digest in files.items():
        saved = destination / "source_snapshot" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, saved)
        if sha256_file(saved) != digest or sha256_file(ROOT / relative) != digest:
            raise ValueError("Input changed during snapshot")
    command = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--nodes=1", "--ntasks=1", "--gres=gpu:l40s:1", "--cpus-per-task=4", "--mem=32G",
        "--time=00:30:00", "--no-requeue", "--exclude=node3006", "--job-name=human-wbc-release",
        f"--chdir={ROOT}", f"--output={destination}/logs/%j.out", f"--error={destination}/logs/%j.err",
        str(ROOT / "scripts/task10_release_baseline.sbatch"), str(destination / "packet.json")]
    packet = {"schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_submitted", "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "bundle": str(checked.root.relative_to(ROOT.resolve())),
        "release": str(release.relative_to(ROOT.resolve())),
        "output": str((destination / "run").relative_to(ROOT.resolve())),
        "files": files, "conditions": 12, "environments_per_condition": 16,
        "bank_rows": list(range(16)), "training": False, "learning_updates": 0,
        "production_qualified": False, "automatic_followup": False,
        "application_wall_seconds": 1680, "allocation_wall_seconds": 1800,
        "products_and_caches_bytes": 6 * 1024**3,
        "authorization": "User standing authorization for GPU tests and sessions at most one hour",
        "submission_command": command}
    write_json(destination / "packet.json", packet)
    print(json.dumps({"packet": str(destination / "packet.json"), "files": len(files),
                      "sha256": sha256_file(destination / "packet.json")}))
    return packet


def tree_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def execute(packet):
    path = writable_path(packet).resolve(strict=True)
    record = json.loads(path.read_text())
    expected = {"schema": SCHEMA, "conditions": 12, "environments_per_condition": 16,
        "bank_rows": list(range(16)), "training": False, "learning_updates": 0,
        "production_qualified": False, "automatic_followup": False,
        "application_wall_seconds": 1680, "allocation_wall_seconds": 1800,
        "products_and_caches_bytes": 6 * 1024**3}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared diagnostic scope or resource limit differs")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("An allocated Slurm GPU is required")
    if not record.get("files") or not required_inputs(record).issubset(record["files"]):
        raise ValueError("Incomplete frozen input closure")
    for relative, digest in record["files"].items():
        source = (ROOT / relative).resolve(strict=True)
        if not source.is_relative_to(ROOT.resolve()) or sha256_file(source) != digest:
            raise ValueError("Frozen input changed: " + relative)
        archive = (path.parent / "source_snapshot" / relative).resolve(strict=True)
        if not archive.is_relative_to(path.parent.resolve()) or sha256_file(archive) != digest:
            raise ValueError("Frozen archive changed: " + relative)
    output = writable_path(ROOT / record["output"])
    if output.exists():
        raise FileExistsError(output)
    configure_local_caches()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(ROOT / ".venv-wbc/bin/python"), "-B", "-m",
        "human_punching.wbc.release_baseline_probe", "--bundle", str(ROOT / record["bundle"]),
        "--release-root", str(ROOT / record["release"]), "--output", str(output)]
    start = time.monotonic()
    reason, peak = None, 0
    process = subprocess.Popen(command, cwd=path.parent, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            size = tree_bytes(path.parent)
            peak = max(peak, size)
            if size > record["products_and_caches_bytes"]:
                reason = "products_and_caches_budget"
            elif time.monotonic() - start >= record["application_wall_seconds"]:
                reason = "application_wall_budget"
            if reason:
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
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
    write_json(path.parent / "execution.json", {"schema": "human_wbc_release_baseline_execution_v1",
        "packet_sha256": sha256_file(path), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "returncode": code, "budget_stop": reason, "wall_s": time.monotonic() - start,
        "peak_sampled_products_bytes": peak, "learning_updates": 0, "production_qualified": False})
    return code if code else (2 if reason else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for arg in ("bundle", "release", "output"):
        prep.add_argument("--" + arg, required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--packet", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.bundle, args.release, args.output)
        return 0
    return execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
