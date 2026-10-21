#!/usr/bin/env python3
"""Prepare pinned local sources; optionally resolve/install an isolated runtime.

No sibling repository or environment is modified. Source pins are authoritative;
the generated dependency lock and runtime record are different evidence objects.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from human_punching.wbc.provenance import (  # noqa: E402
    PYTHON_PIN, SOURCE_PINS, sha256_file, writable_path, write_json,
)


def local_environment() -> dict[str, str]:
    env = os.environ.copy()
    locations = {"UV_CACHE_DIR": ".cache/uv-wbc", "PIP_CACHE_DIR": ".cache/pip-wbc",
                 "TMPDIR": ".cache/task10/tmp", "XDG_CACHE_HOME": ".cache/task10/xdg",
                 "MPLCONFIGDIR": ".cache/task10/matplotlib",
                 "WANDB_CACHE_DIR": ".cache/task10/wandb",
                 "WARP_CACHE_PATH": ".cache/task10/warp"}
    for variable, relative in locations.items():
        path = writable_path(ROOT / relative)
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", UV_LINK_MODE="copy")
    return env


def export_source(name: str) -> Path:
    revision = SOURCE_PINS[name]
    source = ROOT.parent / name
    target = writable_path(ROOT / ".cache" / "deps" / f"{name}-{revision[:8]}")
    manifest = target / ".task10-source.json"
    if target.exists():
        if not manifest.exists() or json.loads(manifest.read_text())["revision"] != revision:
            raise RuntimeError(f"Refusing to overwrite unverified source directory {target}")
        recorded = json.loads(manifest.read_text())
        for relative, digest in recorded["files"].items():
            if sha256_file(target / relative) != digest:
                raise RuntimeError(f"Pinned source changed: {target / relative}")
        return target
    archive = subprocess.check_output(["git", "-C", str(source), "archive", revision])
    # Inspect before extraction. Do not introduce link-mediated writes.
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        members = stream.getmembers()
        for member in members:
            if member.isdev():
                raise RuntimeError(f"Unsupported archive device: {member.name}")
            if not (target / member.name).resolve().is_relative_to(target):
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                link_base = (target / member.name).parent if member.issym() else target
                if not (link_base / member.linkname).resolve().is_relative_to(target):
                    raise RuntimeError(f"Archive link escapes local source: {member.name}")
        target.mkdir(parents=True)
        stream.extractall(target, filter="data")
    files = {str(p.relative_to(target)): sha256_file(p)
             for p in sorted(target.rglob("*")) if p.is_file()}
    write_json(manifest, {"revision": revision, "source": str(source), "files": files})
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Exact Python 3.12.12 interpreter")
    parser.add_argument("--install", action="store_true", help="Resolve and install; default only exports sources")
    parser.add_argument("--resume-install", action="store_true", help="Resume a failed local installation")
    parser.add_argument("--resolve", action="store_true", help="Resolve a NEW candidate lock instead of using the preserved lock")
    parser.add_argument("--output", type=Path, required=True, help="New setup evidence directory inside project")
    args = parser.parse_args()
    output = writable_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = local_environment()
    record: dict = {"status": "started", "commands": [], "training_started": False}

    def run(command: list[str], *, capture: bool = False) -> str:
        record["commands"].append(command)
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (output / "setup.log").open("a") as log:
            log.write("COMMAND " + json.dumps(command) + "\n" + completed.stdout + "\n")
        if completed.returncode:
            raise RuntimeError(f"Command exited {completed.returncode}; see {output / 'setup.log'}")
        return completed.stdout if capture else ""

    try:
        branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
        if branch != "wbc":
            raise RuntimeError(f"Task 10 requires the wbc branch, found {branch}")
        version = run([args.python, "-B", "-c", "import platform; print(platform.python_version())"], capture=True).strip()
        if version != PYTHON_PIN:
            raise RuntimeError(f"Expected Python {PYTHON_PIN}, found {version}")
        source_dirs = [export_source(name) for name in ("mjlab", "wbc-mjlab")]
        record["sources"] = {p.name: str(p) for p in source_dirs}
        if args.install or args.resume_install:
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError("uv executable is required")
            venv = writable_path(ROOT / ".venv-wbc")
            if venv.exists():
                if not args.resume_install:
                    raise RuntimeError(".venv-wbc already exists; use --resume-install after inspection")
                actual_prefix = run([str(venv / "bin" / "python"), "-B", "-c",
                                     "import sys; print(sys.prefix)"], capture=True).strip()
                if Path(actual_prefix).resolve() != venv:
                    raise RuntimeError("Existing interpreter is not the project-local WBC environment")
            else:
                run([uv, "venv", "--python", args.python, str(venv)])
            python = str(venv / "bin" / "python")
            candidate = output / "candidate.in"
            candidate.write_text((ROOT / "requirements-wbc.in").read_text() + "\n" +
                                 "\n".join(str(p) for p in source_dirs + [ROOT]) + "\n")
            lock = output / "requirements-wbc.lock"
            preserved_lock = ROOT / "requirements-wbc.lock"
            if args.resolve:
                run([uv, "pip", "compile", str(candidate), "--python", python,
                     "--generate-hashes", "--emit-index-url", "--index-strategy", "unsafe-best-match",
                     "--no-emit-package", "mjlab", "--no-emit-package", "wbc-mjlab",
                     "--no-emit-package", "human-punching", "-o", str(lock)])
            elif preserved_lock.is_file():
                shutil.copyfile(preserved_lock, lock)
                record["preserved_lock_sha256"] = sha256_file(preserved_lock)
            else:
                raise RuntimeError("Preserved requirements-wbc.lock is missing; explicitly use --resolve for a new candidate")
            run([uv, "pip", "install", "--python", python, "--require-hashes",
                 "--index-strategy", "unsafe-best-match", "-r", str(lock)])
            run([uv, "pip", "install", "--python", python, "--no-deps",
                 *map(str, source_dirs), "-e", str(ROOT)])
            run([uv, "pip", "check", "--python", python])
            observed = run([python, "-B", "-c",
                            "import json; from human_punching.wbc.provenance import runtime_record; "
                            "print(json.dumps(runtime_record()))"], capture=True)
            record["runtime"] = json.loads(observed)
            record["dependency_lock_sha256"] = sha256_file(lock)
            if not record["runtime"]["declared_pins_match"]:
                raise RuntimeError("Installed candidate does not match Task10 runtime pins")
            record["status"] = "installed_not_native_qualified"
        else:
            record["status"] = "sources_prepared_no_install"
    except Exception as error:
        record.update(status="failed", error=str(error))
        print(str(error), file=sys.stderr)
        return 1
    finally:
        write_json(output / "setup.json", record)
    print(json.dumps({"status": record["status"], "evidence": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
