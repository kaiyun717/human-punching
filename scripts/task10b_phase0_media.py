#!/usr/bin/env python3
"""Prepare or render the fixed Task 10B Phase 0 saved-state evidence gallery.

Trial 00 is selected in every parity condition before outcomes are inspected.
Only cropped recorded states are replayed; no physics integration or learning.
This helper never submits a job or changes an executed experiment packet.
"""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import sha256_file, writable_path, write_json

PYTHON = ROOT / ".venv-task10b/bin/python"
SCHEMA = "task10b_phase0_saved_media_packet_v1"
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SOURCES = (
    "scripts/task10b_phase0_media.py", "scripts/task10b_phase0_media.sbatch",
    "scripts/plot_punch_trial.py", "tests/test_task10b_phase0_media.py",
    "src/human_punching/jab_evaluation.py", "src/human_punching/contact.py",
    "src/human_punching/render.py", "src/human_punching/target_render.py",
    "src/human_punching/wbc/egl_binding.py", "src/human_punching/wbc/provenance.py",
    "outputs/task10b/p0prep_20261020_01/runtime.json",
    "outputs/task10b/p0prep_20261020_01/runtime.env",
    ".venv-task10b/pyvenv.cfg",
    ".venv-task10b/lib/python3.12/site-packages/task10b_readonly_packages.pth",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def selection():
    return [dict(family=family, solver=solver, residual=residual, trial_id=0,
                 condition=f"{family}_{solver}_ff_{residual}")
            for family in FAMILIES for solver in ("historical", "accepted")
            for residual in ("learned", "zero")]


def relative(path):
    return str(Path(path).resolve(strict=True).relative_to(ROOT.resolve()))


def check_scope(packet):
    fixed = dict(schema=SCHEMA, physics_integration=False, learning=False, trial_id=0,
                 fps=50, playback_speeds=[1.0], allocation_wall_seconds=1800,
                 application_wall_seconds=1680, products_and_caches_bytes=2 * 1024**3,
                 selected_conditions=selection(), excluded_nodes=["node3006"])
    if any(packet.get(key) != value for key, value in fixed.items()):
        raise ValueError("Saved-media selection or bounded scope changed")
    if len(packet.get("clips", [])) != 16:
        raise ValueError("All sixteen predeclared parity clips are required")
    if any(any(clip.get(key) != value for key, value in expected.items())
           for clip, expected in zip(packet["clips"], selection(), strict=True)):
        raise ValueError("Clip identities differ from the predeclared selection")


def recorded_horizon(rollout, expected_steps):
    with np.load(rollout, allow_pickle=False) as arrays:
        times = np.asarray(arrays["t"], dtype=float)
        if (times.shape != (expected_steps + 1,) or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(expected_steps + 1) * .005, atol=1e-12, rtol=0)
                or int(arrays["physics_step_count"]) != expected_steps
                or int(arrays["evaluation_bank_world"]) != 0
                or arrays["qpos"].shape != (len(times), 55)
                or arrays["qvel"].shape != (len(times), 54)):
            raise ValueError("Media must use original cropped horizon and predeclared bank row 00")
    # Same sample-time grid as the existing renderer; source states are nearest
    # recorded samples, without interpolating physical state.
    frame_count = len(np.arange(0., float(times[-1]) + 1e-9, 1. / 50))
    return dict(source_state_count=len(times), source_duration_s=float(times[-1]),
                expected_video_frames=frame_count, source_physics_steps=expected_steps)


def prepare(parity_packet, output):
    source = Path(parity_packet).resolve(strict=True)
    source.relative_to(ROOT)
    parent = json.loads(source.read_text())
    execution_path = source.parent / "execution.json"
    execution = json.loads(execution_path.read_text())
    if (parent.get("phase") != "parity" or execution.get("status") != "passed"
            or execution.get("completed_modes") != ["parity"]
            or execution.get("packet_sha256") != sha256_file(source)):
        raise ValueError("A completed and identified parity measurement packet is required")
    native = ROOT / parent["output"] / "parity"
    audit_path = native / "audit.json"
    audit = json.loads(audit_path.read_text())
    if (audit.get("status") != "passed" or audit.get("mode") != "parity"
            or len(audit.get("conditions", [])) != 16 or audit.get("unattempted_conditions") != 0):
        raise ValueError("Incomplete parity cannot be presented as a sixteen-condition gallery")
    mode = execution.get("modes", [])
    if len(mode) != 1 or mode[0].get("audit_sha256") != sha256_file(audit_path):
        raise ValueError("Parity execution does not bind its native audit")
    conditions = {row["name"]: row for row in audit["conditions"]}
    paths = [source, execution_path, audit_path, *(ROOT / name for name in SOURCES)]
    clips = []
    for chosen in selection():
        condition = conditions[chosen["condition"]]
        if condition.get("status") != "passed":
            raise ValueError("Cannot render an unverified native condition")
        folder = native / chosen["condition"] / "historical_evaluation"
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary != condition["historical_evaluation"]:
            raise ValueError("Historical summary differs from native audit")
        rows = [row for row in summary["trials"] if row["trial_id"] == 0]
        if len(rows) != 1:
            raise ValueError("Predeclared trial 00 is missing or duplicated")
        model = folder / "historical_evaluation_runtime.xml"
        rollout = folder / "rollout_00.npz"
        reference = source.parent / "source_snapshot" / parent["release"] / chosen["family"] / "reference.npz"
        if (sha256_file(model) != summary["runtime_model_sha256"]
                or sha256_file(rollout) != rows[0]["rollout_sha256"]
                or sha256_file(reference) != summary["source_inputs"]["reference"]):
            raise ValueError("Saved playback input hash differs from parity evidence")
        with np.load(reference, allow_pickle=False) as arrays:
            target_key = next((key for key in ("head_center", "head_center_w", "target_head_center_w") if key in arrays), None)
            if target_key is None:
                raise ValueError("Explicit frozen reference head center is required")
            head = np.asarray(arrays[target_key], dtype=float)
            if head.shape != (3,) or not np.isfinite(head).all():
                raise ValueError("Malformed frozen reference target")
        clips.append(dict(**chosen, model=relative(model), rollout=relative(rollout),
            reference=relative(reference), summary=relative(summary_path), head_center=head.tolist(),
            source_strict_historical_pass=bool(rows[0]["strict_historical_release_pass"]),
            source_failed_checks=rows[0]["failed_checks"],
            **recorded_horizon(rollout, int(summary["historical_physics_steps"]))))
        paths.extend((model, rollout, reference, summary_path, folder / "trial_00.json"))
    destination = writable_path(output)
    destination.relative_to(ROOT / "outputs/task10b")
    if destination.is_relative_to(source.parent):
        raise ValueError("Media must not write inside its executed parity packet")
    packet = dict(schema=SCHEMA, prepared_utc=now(), status="prepared_not_submitted",
        physics_integration=False, learning=False, trial_id=0, fps=50, playback_speeds=[1.0],
        allocation_wall_seconds=1800, application_wall_seconds=1680,
        products_and_caches_bytes=2 * 1024**3, excluded_nodes=["node3006"],
        selected_conditions=selection(), clips=clips, source_parity_packet=relative(source),
        source_parity_sha256=sha256_file(source), output=str((destination / "run").relative_to(ROOT)),
        source_files=list(SOURCES), interpreter_sha256=sha256_file(PYTHON),
        selection_note="Trial 00 of all sixteen conditions, including failures; fixed before physical outcomes. No success selection.",
        source_horizon="Original historical cropped horizon; actor-rounding padding excluded, last-state still retained.",
        authorization="Task 10B Phase 0 visual evidence only; replay recorded states, no learning or physics integration.")
    packet["files"] = {relative(path): sha256_file(path) for path in paths}
    check_scope(packet)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    for name in SOURCES:
        copied = destination / "source_snapshot" / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, copied)
    packet["submission_command"] = ["sbatch", "--parsable", "--account=mit_general", "--partition=mit_normal_gpu",
        "--nodes=1", "--ntasks=1", "--gres=gpu:l40s:1", "--cpus-per-task=4", "--mem=32G",
        "--time=00:30:00", "--no-requeue", "--exclude=node3006", "--job-name=t10b-p0-media",
        "--chdir=" + str(ROOT), "--output=" + str(destination / "logs/%j.out"),
        "--error=" + str(destination / "logs/%j.err"), str(ROOT / "scripts/task10b_phase0_media.sbatch"),
        str(destination / "packet.json")]
    write_json(destination / "packet.json", packet)
    verify(destination / "packet.json")
    return dict(packet=str(destination / "packet.json"), packet_sha256=sha256_file(destination / "packet.json"),
                clips=len(clips), planned_frames=sum(row["expected_video_frames"] for row in clips))


def verify(path):
    path = writable_path(path).resolve(strict=True)
    packet = json.loads(path.read_text())
    check_scope(packet)
    required = set(SOURCES) | {packet["source_parity_packet"]}
    required.update(clip[key] for clip in packet["clips"] for key in ("model", "rollout", "reference", "summary"))
    if not required.issubset(packet.get("files", {})) or packet.get("source_files") != list(SOURCES):
        raise ValueError("Media packet lacks required source/input hashes")
    if sha256_file(PYTHON) != packet["interpreter_sha256"] or not PYTHON.resolve().is_relative_to(ROOT):
        raise ValueError("Human-owned interpreter identity changed")
    for name, expected in packet["files"].items():
        current = (ROOT / name).resolve(strict=True)
        if not current.is_relative_to(ROOT) or sha256_file(current) != expected:
            raise ValueError("Frozen media input changed: " + name)
        if name in packet["source_files"] and sha256_file(path.parent / "source_snapshot" / name) != expected:
            raise ValueError("Media source archive differs: " + name)
    return packet


def local_caches(output):
    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR", "MPLCONFIGDIR", "TORCH_HOME",
                 "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "WARP_CACHE_PATH", "HF_HOME", "NUMBA_CACHE_DIR"):
        folder = writable_path(output / "cache" / name.lower())
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", MUJOCO_GL="egl")


def worker(path):
    packet = verify(path)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Saved-state rendering requires an allocated GPU")
    output = writable_path(ROOT / packet["output"])
    output.mkdir(parents=True, exist_ok=False)
    local_caches(output)
    from human_punching.wbc.egl_binding import configure_allocated_egl
    binding = configure_allocated_egl()  # Must precede renderer/MuJoCo imports.
    write_json(output / "gpu_binding.json", binding)
    from human_punching.jab_evaluation import render_rollout
    import imageio.v2 as imageio
    import imageio_ffmpeg
    records = []
    for clip in packet["clips"]:
        start = time.monotonic()
        destination = output / clip["condition"]
        destination.mkdir(exist_ok=False)
        label = f"{clip['family']} | {clip['solver']} solver | {clip['residual']} | trial 00 | recorded physics"
        result = render_rollout(ROOT / clip["model"], ROOT / clip["rollout"], destination,
            head_center=clip["head_center"], label=label, fps=50, playback_speeds=(1.0,), artifact_prefix=clip["family"])
        video = Path(result["video"])
        with imageio.get_reader(video) as reader:
            count = sum(1 for _ in reader)
        if count != clip["expected_video_frames"] or result["video_frames"] != count:
            raise ValueError("Encoded frame count differs from the entire declared playback grid")
        subprocess.run([str(PYTHON), "-B", str(ROOT / "scripts/plot_punch_trial.py"),
            "--model", str(ROOT / clip["model"]), "--reference", str(ROOT / clip["reference"]),
            "--rollout", str(ROOT / clip["rollout"]), "--output", str(destination / "plots")], check=True)
        record = dict(condition=clip["condition"], trial_id=0, decoded_frames=count, fps=50,
            source_duration_s=clip["source_duration_s"], video_duration_s=count / 50,
            source_strict_historical_pass=clip["source_strict_historical_pass"],
            source_failed_checks=clip["source_failed_checks"], wall_s=time.monotonic() - start,
            inputs={key: packet["files"][clip[key]] for key in ("model", "rollout", "reference", "summary")},
            artifacts={str(product.relative_to(destination)): dict(sha256=sha256_file(product), bytes=product.stat().st_size)
                       for product in sorted(destination.rglob("*")) if product.is_file()},
            rendering_only=True, physics_integration=False, learning=False)
        write_json(destination / "media_audit.json", record)
        records.append(record)
    write_json(output / "manifest.json", dict(schema="task10b_phase0_saved_media_v1", status="passed",
        packet_sha256=sha256_file(path), source_parity_sha256=packet["source_parity_sha256"],
        selection=packet["selection_note"], gpu_binding=binding,
        renderer_sha256=sha256_file(ROOT / "src/human_punching/jab_evaluation.py"),
        ffmpeg_sha256=sha256_file(imageio_ffmpeg.get_ffmpeg_exe()), clips=records,
        physics_integration=False, learning=False, source_outcomes_recomputed=False))
    return 0


def size_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A disposable renderer/compiler temporary file was removed.
    return total


def execute(path):
    path = writable_path(path).resolve(strict=True)
    packet = verify(path)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("An allocated Slurm GPU is required")
    start = time.monotonic()
    receipt = dict(schema="task10b_phase0_media_execution_v1", status="running", started_utc=now(),
        job_id=os.environ["SLURM_JOB_ID"], packet_sha256=sha256_file(path), physics_integration=False,
        learning=False, budget_stop=None, peak_sampled_bytes=0)
    write_json(path.parent / "execution_started.json", receipt)
    process = subprocess.Popen([str(PYTHON), "-B", str(ROOT / "scripts/task10b_phase0_media.py"),
                                "worker", "--packet", str(path)], start_new_session=True, cwd=ROOT)
    try:
        while process.poll() is None:
            receipt["peak_sampled_bytes"] = max(receipt["peak_sampled_bytes"], size_bytes(path.parent))
            if receipt["peak_sampled_bytes"] > packet["products_and_caches_bytes"]:
                receipt["budget_stop"] = "products_and_caches_budget"
            elif time.monotonic() - start >= packet["application_wall_seconds"]:
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
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    output = ROOT / packet["output"]
    complete = sorted(output.glob("*/media_audit.json"))
    final_bytes = size_bytes(path.parent)
    if final_bytes > packet["products_and_caches_bytes"]:
        receipt["budget_stop"] = "products_and_caches_budget"
    receipt.update(status="passed" if code == 0 and len(complete) == 16 and not receipt["budget_stop"] else "failed",
        ended_utc=now(), wall_s=time.monotonic() - start, returncode=code,
        completed_clips=len(complete), uncompleted_clips=16 - len(complete), final_bytes=final_bytes)
    write_json(path.parent / "execution.json", receipt)
    return 0 if receipt["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--parity-packet", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    for name in ("verify", "execute", "worker"):
        sub.add_parser(name).add_argument("--packet", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.parity_packet, args.output)))
        return 0
    if args.mode == "verify":
        verify(args.packet)
        print('{"status":"passed"}')
        return 0
    return worker(args.packet) if args.mode == "worker" else execute(args.packet)


if __name__ == "__main__":
    raise SystemExit(main())
