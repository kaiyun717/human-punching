#!/usr/bin/env python3
"""Freeze, verify and execute Task 10B Phase 0; never submit or train.

The preflight packet executes reset/one-actor-interval checks before the twelve
controlled conditions. Parity is a separate packet with a passed preflight
receipt. Existing Task 10 packets/results are never modified or resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = "human_task10b_phase0_packet_v1"
PYTHON = ROOT / ".venv-task10b/bin/python"
RUNTIME = ROOT / "outputs/task10b/p0prep_20261020_01/runtime.json"
RUNTIME_ENV = RUNTIME.with_name("runtime.env")
LIMITS = {"preflight": (1800, 1680, 6 * 1024**3), "parity": (3600, 3420, 8 * 1024**3)}


def now():
    return datetime.now(timezone.utc).isoformat()


def required_inputs(record):
    spec = importlib.util.spec_from_file_location("phase0_old_input_closure", ROOT / "scripts/task10_release_baseline.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    paths = old.required_inputs(record)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "src/human_punching").rglob("*.py"))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*release_baseline*.py"))
    paths.update(("scripts/task10b_phase0.py", "scripts/task10b_phase0.sbatch",
                  "tests/test_task10b_phase0_packet.py", str(RUNTIME.relative_to(ROOT)),
                  str(RUNTIME_ENV.relative_to(ROOT)), ".venv-task10b/pyvenv.cfg",
                  str(RUNTIME.with_name("runtime_interpreter_manifest.json").relative_to(ROOT)),
                  ".venv-task10b/lib/python3.12/site-packages/task10b_readonly_packages.pth"))
    return paths


def validate_scope(record):
    phase = record.get("phase")
    if phase not in LIMITS:
        raise ValueError("Only preflight and parity Phase 0 packets are supported")
    allocation, application, products = LIMITS[phase]
    expected = dict(schema=SCHEMA, task="10B", program_phase=0, training=False,
                    learning_updates=0, automatic_followup=False,
                    allocation_wall_seconds=allocation, application_wall_seconds=application,
                    products_and_caches_bytes=products,
                    modes=["contract", "diagnostic"] if phase == "preflight" else ["parity"])
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Phase 0 scope or bounded resources changed")
    if record.get("partition") not in ("mit_normal_gpu", "mit_preemptable"):
        raise ValueError("Partition is outside Phase 0 authorization")
    exclusions = record.get("excluded_nodes", [])
    if (not isinstance(exclusions, list) or any(not isinstance(node, str)
            or re.fullmatch(r"node[0-9]+", node) is None for node in exclusions)
            or exclusions != sorted(set(exclusions))):
        raise ValueError("Excluded nodes must be distinct canonical cluster node names")


def verify_runtime_identity(record):
    """Verify local interpreter payload and existing installed metadata.

    Distribution METADATA/RECORD/direct_url.json hashes and declared versions
    bind the reused environment without claiming every installed package file
    was rehashed. The interpreter/stdlib manifest is verified file by file.
    """
    runtime = json.loads(RUNTIME.read_text())
    local = Path(runtime["local_python_root"]).resolve(strict=True)
    if not local.is_relative_to(ROOT.resolve()):
        raise ValueError("Interpreter library root escaped the human project")
    manifest = RUNTIME.with_name("runtime_interpreter_manifest.json")
    if sha256_file(manifest) != runtime["interpreter_manifest_sha256"]:
        raise ValueError("Interpreter manifest changed")
    for item in json.loads(manifest.read_text()):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Interpreter manifest path is not relative")
        member = local / relative
        if not member.resolve(strict=True).is_relative_to(local):
            raise ValueError("Interpreter member escaped its local root")
        if "symlink" in item:
            if not member.is_symlink() or os.readlink(member) != item["symlink"]:
                raise ValueError("Interpreter symlink changed: " + str(relative))
        elif (member.is_symlink() or member.stat().st_size != item["bytes"]
              or sha256_file(member) != item["sha256"]):
            raise ValueError("Interpreter payload changed: " + str(relative))
    for key, expected in runtime["distribution_metadata_hashes"].items():
        package, filename = key.rsplit("/", 1)
        distribution = importlib.metadata.distribution(package)
        matches = [f for f in (distribution.files or ())
                   if str(f).endswith(".dist-info/" + filename)]
        if len(matches) != 1:
            raise ValueError("Installed distribution metadata missing: " + key)
        metadata_path = Path(distribution.locate_file(matches[0])).resolve(strict=True)
        value = distribution.read_text(filename)
        if (not metadata_path.is_relative_to(ROOT.resolve()) or value is None
                or len(value.encode()) != expected["bytes"]
                or hashlib.sha256(value.encode()).hexdigest() != expected["sha256"]):
            raise ValueError("Installed distribution metadata changed: " + key)
    current = runtime_record()
    if not current["declared_pins_match"] or current != record["runtime"]:
        raise ValueError("Phase 0 runtime package pins or identity changed")


def verify_preflight_identity(record):
    """Require the passed diagnostic to cover the same scientific input bytes."""
    gate = (ROOT / record["preflight_receipt"]).resolve(strict=True)
    if not gate.is_relative_to(ROOT) or sha256_file(gate) != record["preflight_sha256"]:
        raise ValueError("Preflight receipt changed")
    result = json.loads(gate.read_text())
    if result.get("status") != "passed" or result.get("completed_modes") != ["contract", "diagnostic"]:
        raise ValueError("Native contract and twelve-condition diagnostic must pass first")
    previous_path = gate.parent / "packet.json"
    if (str(previous_path.relative_to(ROOT)) != record["preflight_packet"]
            or sha256_file(previous_path) != record["preflight_packet_sha256"]
            or result.get("packet_sha256") != record["preflight_packet_sha256"]):
        raise ValueError("Preflight execution does not bind the recorded packet")
    previous = verify(previous_path, check_preflight=False)
    if previous["phase"] != "preflight":
        raise ValueError("Parity requires a preflight packet")
    for key in ("bundle", "release", "runtime", "interpreter_path", "interpreter_sha256", "files"):
        if previous[key] != record[key]:
            raise ValueError("Parity inputs differ from passed preflight: " + key)
    if len(result.get("modes", [])) != 2:
        raise ValueError("Preflight execution lacks its two completed mode receipts")
    for mode, entry in zip(("contract", "diagnostic"), result["modes"]):
        audit = (ROOT / previous["output"] / mode / "audit.json").resolve(strict=True)
        if (not audit.is_relative_to(ROOT) or entry.get("mode") != mode
                or entry.get("returncode") != 0 or entry.get("audit_status") != "passed"
                or sha256_file(audit) != entry.get("audit_sha256")):
            raise ValueError("Preflight native audit identity differs: " + mode)


def verify(packet, *, check_preflight=True):
    path = writable_path(packet).resolve(strict=True)
    record = json.loads(path.read_text())
    validate_scope(record)
    if not required_inputs(record).issubset(record.get("files", {})):
        raise ValueError("Incomplete frozen input closure")
    for relative, digest in record["files"].items():
        source = (ROOT / relative).resolve(strict=True)
        archive = (path.parent / "source_snapshot" / relative).resolve(strict=True)
        if not source.is_relative_to(ROOT.resolve()) or sha256_file(source) != digest:
            raise ValueError("Frozen input changed: " + relative)
        if not archive.is_relative_to(path.parent) or sha256_file(archive) != digest:
            raise ValueError("Frozen archive changed: " + relative)
    interpreter = PYTHON.resolve(strict=True)
    if not interpreter.is_relative_to(ROOT) or sha256_file(interpreter) != record["interpreter_sha256"]:
        raise ValueError("Human-owned Python interpreter identity changed")
    verify_runtime_identity(record)
    if record["phase"] == "parity" and check_preflight:
        verify_preflight_identity(record)
    return record


def prepare(bundle, release, output, phase, *, partition="mit_normal_gpu", preflight=None, excluded_nodes=("node3006",)):
    from human_punching.wbc.release_baseline_probe import load_release_inputs
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    if phase not in LIMITS:
        raise ValueError("Unsupported Phase 0 packet")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "wbc":
        raise ValueError("Task 10B implementation requires human-punching/wbc")
    if not PYTHON.resolve(strict=True).is_relative_to(ROOT):
        raise ValueError("Python interpreter must be human-owned")
    checked = load_runtime_bundle(bundle)
    release = Path(release).resolve(strict=True)
    load_release_inputs(checked, release, num_envs=64 if phase == "parity" else 16)
    destination = writable_path(output)
    destination.relative_to(ROOT / "outputs/task10b")
    if len(os.fsencode(destination / "run/diagnostic/cache/tmp")) > 120:
        raise ValueError("Use a shallow native compiler temporary path")
    allocation, application, products = LIMITS[phase]
    record = dict(schema=SCHEMA, task="10B", program_phase=0, phase=phase,
        prepared_utc=now(), status="prepared_not_submitted", partition=partition,
        excluded_nodes=sorted(set(excluded_nodes)),
        bundle=str(checked.root.relative_to(ROOT)), release=str(release.relative_to(ROOT)),
        output=str((destination / "run").relative_to(ROOT)),
        modes=["contract", "diagnostic"] if phase == "preflight" else ["parity"],
        training=False, learning_updates=0, automatic_followup=False,
        production_qualified=False, allocation_wall_seconds=allocation,
        application_wall_seconds=application, products_and_caches_bytes=products,
        interpreter_path=str(PYTHON.resolve()), interpreter_sha256=sha256_file(PYTHON),
        runtime=runtime_record(),
        authorization="2026-10-20 explicit Task 10B Phase 0 implementation/execution; up to two normal and four preemptable L40S; sessions <=4h; no later phase",
        requested_contract_fixtures=192 if phase == "preflight" else 0,
        requested_full_cycle_attempts=192 if phase == "preflight" else 1024)
    if "node3006" in record["excluded_nodes"]:
        record["node3006_exclusion_reason"] = (
            "Repeated CUDA initialization failures before scene construction: historical Task 10 gl01/job26344908 "
            "and Task 10B p0c04/job26449837. Existing Task 10 launchers already exclude this node; "
            "the next allocation records independent NVIDIA/Torch health before native execution.")
    if phase == "parity":
        if preflight is None:
            raise ValueError("Parity requires an explicit successful preflight execution receipt")
        gate = Path(preflight).resolve(strict=True)
        gate.relative_to(ROOT)
        result = json.loads(gate.read_text())
        if result.get("status") != "passed" or result.get("completed_modes") != ["contract", "diagnostic"]:
            raise ValueError("Native preflight is not complete and passed")
        previous_path = gate.parent / "packet.json"
        record.update(preflight_receipt=str(gate.relative_to(ROOT)), preflight_sha256=sha256_file(gate),
                      preflight_packet=str(previous_path.relative_to(ROOT)),
                      preflight_packet_sha256=sha256_file(previous_path))
    validate_scope(record)
    files = {}
    for relative in sorted(required_inputs(record)):
        source = (ROOT / relative).resolve(strict=True)
        source.relative_to(ROOT)
        files[relative] = sha256_file(source)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    for relative, digest in files.items():
        archived = destination / "source_snapshot" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, archived)
        if sha256_file(archived) != digest or sha256_file(ROOT / relative) != digest:
            raise ValueError("Input changed during source archival: " + relative)
    record["files"] = files
    record["submission_command"] = ["sbatch", "--parsable", "--account=mit_general",
        "--partition=" + partition, "--nodes=1", "--ntasks=1", "--gres=gpu:l40s:1",
        "--cpus-per-task=4", "--mem=32G", "--time=" + ("00:30:00" if phase == "preflight" else "01:00:00"),
        "--no-requeue", "--job-name=t10b-p0-" + phase, "--chdir=" + str(ROOT),
        "--output=" + str(destination / "logs/%j.out"), "--error=" + str(destination / "logs/%j.err"),
        str(ROOT / "scripts/task10b_phase0.sbatch"), str(destination / "packet.json")]
    if record["excluded_nodes"]:
        record["submission_command"].insert(1, "--exclude=" + ",".join(record["excluded_nodes"]))
    write_json(destination / "packet.json", record)
    verify(destination / "packet.json")
    return {"packet": str(destination / "packet.json"), "files": len(files),
            "packet_sha256": sha256_file(destination / "packet.json")}


def tree_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


CUDA_HEALTH_CODE = r'''
import json
import sys
result = {"python": sys.version, "executable": sys.executable, "initialized": False}
try:
    import torch
    result.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
    torch.cuda.init()
    value = torch.ones(1, device="cuda:0")
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    result.update(initialized=True, tensor_roundtrip=float(value.item()) == 1.0,
                  device_name=props.name, total_memory_bytes=props.total_memory,
                  compute_capability=[props.major, props.minor], device_uuid=str(getattr(props, "uuid", "unavailable")))
except Exception as error:
    result["exception"] = {"type": type(error).__name__, "message": str(error)}
print(json.dumps(result))
sys.exit(0 if result.get("initialized") and result.get("tensor_roundtrip") else 2)
'''


def cuda_health_receipt(destination):
    """Record assigned-device visibility and driver initialization, without physics."""
    selected = ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD")
    result = dict(schema="task10b_phase0_cuda_health_v1", utc=now(), hostname=socket.gethostname(),
                  selected_environment={name: os.environ.get(name) for name in selected},
                  physics_steps=0, actor_updates=0, learning_updates=0)
    commands = {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,compute_mode",
                       "--format=csv,noheader,nounits"],
        "torch": [str(PYTHON), "-B", "-c", CUDA_HEALTH_CODE],
    }
    for name, command in commands.items():
        start = time.monotonic()
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=15 if name == "nvidia_smi" else 60, check=False)
            entry = dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            if name == "torch" and completed.stdout.strip():
                try:
                    entry["result"] = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    entry["parse_error"] = "Torch health stdout was not a single JSON record"
        except Exception as error:
            entry = dict(returncode=None, exception=dict(type=type(error).__name__, message=str(error)))
        entry["wall_s"] = time.monotonic() - start
        result[name] = entry
    measured = result["torch"].get("result", {})
    result["status"] = "passed" if (result["nvidia_smi"]["returncode"] == 0
        and result["torch"]["returncode"] == 0 and measured.get("cuda_available") is True
        and measured.get("initialized") is True and measured.get("tensor_roundtrip") is True
        and measured.get("device_count", 0) >= 1) else "failed"
    write_json(destination, result)
    return result


def execute(packet):
    path = writable_path(packet).resolve(strict=True)
    if path.name != "packet.json" or not path.is_relative_to(ROOT / "outputs/task10b"):
        raise ValueError("Execution packet must be inside a Task 10B output directory")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Allocated Slurm CUDA execution is required")
    start = time.monotonic()
    receipt = dict(schema="human_task10b_phase0_execution_v1", status="running", started_utc=now(),
        packet_sha256=sha256_file(path), slurm_job_id=os.environ["SLURM_JOB_ID"],
        training=False, learning_updates=0, completed_modes=[], modes=[], budget_stop=None,
        peak_sampled_products_bytes=0, native_processes_started=0)
    write_json(path.parent / "execution_started.json", receipt)
    code = 2
    try:
        record = verify(path)
        health_path = path.parent / "cuda_health.json"
        health = cuda_health_receipt(health_path)
        receipt.update(cuda_health_status=health["status"], cuda_health_sha256=sha256_file(health_path))
        if health["status"] != "passed":
            raise RuntimeError("Allocated CUDA health check failed before native modes; see cuda_health.json")
        output = writable_path(ROOT / record["output"])
        output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled")
        for mode in record["modes"]:
            command = [str(PYTHON), "-B", "-m", "human_punching.wbc.release_baseline_probe",
                "--mode", mode, "--bundle", str(ROOT / record["bundle"]),
                "--release-root", str(ROOT / record["release"]), "--output", str(output / mode)]
            mode_start = time.monotonic()
            process = subprocess.Popen(command, cwd=output, env=env, start_new_session=True)
            receipt["native_processes_started"] += 1
            try:
                while process.poll() is None:
                    size = tree_bytes(path.parent)
                    receipt["peak_sampled_products_bytes"] = max(size, receipt["peak_sampled_products_bytes"])
                    if size > record["products_and_caches_bytes"]:
                        receipt["budget_stop"] = "products_and_caches_budget"
                    elif time.monotonic() - start >= record["application_wall_seconds"]:
                        receipt["budget_stop"] = "application_wall_budget"
                    if receipt["budget_stop"]:
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
            audit_path = output / mode / "audit.json"
            audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
            mode_result = dict(mode=mode, returncode=code, wall_s=time.monotonic() - mode_start,
                audit_status=audit.get("status"), audit_sha256=sha256_file(audit_path) if audit else None,
                completed_conditions=len(audit.get("conditions", [])),
                unattempted_conditions=audit.get("unattempted_conditions"),
                actual_world_physics_transitions=audit.get("actual_world_physics_transitions"))
            receipt["modes"].append(mode_result)
            write_json(path.parent / (mode + "_execution.json"), mode_result)
            if code or receipt["budget_stop"] or audit.get("status") != "passed":
                break
            receipt["completed_modes"].append(mode)
        receipt["status"] = "passed" if receipt["completed_modes"] == record["modes"] else "failed"
    except Exception as error:
        receipt.update(status="failed", exception=dict(type=type(error).__name__, message=str(error)))
    finally:
        if receipt["native_processes_started"] == 0:
            receipt.update(all_native_modes_unattempted=True, actual_world_physics_transitions=0,
                           actual_actor_updates=0)
        receipt.update(wall_s=time.monotonic() - start, ended_utc=now(), returncode=code)
        write_json(path.parent / "execution.json", receipt)
    return 0 if receipt["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("bundle", "release", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--phase", choices=tuple(LIMITS), required=True)
    prep.add_argument("--partition", choices=("mit_normal_gpu", "mit_preemptable"), default="mit_normal_gpu")
    prep.add_argument("--preflight", type=Path)
    prep.add_argument("--exclude-node", action="append", default=["node3006"],
                      help="Additional canonical cluster node to exclude; node3006 is excluded by default")
    for name in ("verify", "execute"):
        sub.add_parser(name).add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.bundle, args.release, args.output, args.phase,
              partition=args.partition, preflight=args.preflight, excluded_nodes=args.exclude_node)))
        return 0
    if args.mode == "verify":
        record = verify(args.packet)
        print(json.dumps({"status": "passed", "files": len(record["files"]), "phase": record["phase"]}))
        return 0
    return execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
