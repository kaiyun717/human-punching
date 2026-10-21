"""Source/runtime identity and confined output handling for Task 10."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PINS = {
    "human-punching-baseline": "e9205a63a188181099b2d08be15913153dbf0ccc",
    "wbc-mjlab": "eb6c024b9586bc1de41b3dd22b9f18ad51cb1da9",
    "mjlab": "8ee51fbcf806a7419189f706d9e394cbeb7790fa",
    "robust_sav": "f4672e3ab403308520beb6c6476a4027096764bf",
}
RUNTIME_PINS = {
    "numpy": "2.4.6", "mujoco": "3.11.0", "mujoco-warp": "3.11.0",
    "warp-lang": "1.14.0", "torch": "2.8.0+cu128", "rsl-rl-lib": "5.5.0",
    "tensordict": "0.14.2", "scipy": "1.18.1", "onnx": "1.23.0",
    "mjlab": "1.6.0", "wbc-mjlab": "0.0.6",
}
PYTHON_PIN = "3.12.12"


def configure_local_caches() -> None:
    """Confine imports, compilation and logging caches before native imports."""
    locations = {
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "config", "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch",
        "CUDA_CACHE_PATH": "cuda", "WARP_CACHE_PATH": "warp",
        "WANDB_CACHE_DIR": "wandb", "WANDB_CONFIG_DIR": "wandb_config",
        "HF_HOME": "huggingface",
    }
    for variable, directory in locations.items():
        destination = writable_path(PROJECT_ROOT / ".cache/task10" / directory)
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def writable_path(path: Path | str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve links before writes; never overwrite frozen Task 09 products."""
    root = root.resolve()
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError(f"Task 10 output must be beneath {root}: {candidate}")
    relative = candidate.relative_to(root)
    if relative.parts[0] == ".git" or relative.parts[:2] in (
        ("outputs", "four-punch-milestone"), ("outputs", "master-report"),
        ("assets", "vendor"),
    ):
        raise ValueError(f"Protected source or historical artifact: {candidate}")
    return candidate


def write_json(path: Path | str, payload: Any, *, overwrite: bool = False) -> Path:
    destination = writable_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is the default: an earlier run is evidence, not a cache.
    with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def git_identity(repository: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True
        ).strip()
    return {"revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status": git("status", "--short")}


def runtime_record() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in RUNTIME_PINS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    differences = {name: {"expected": expected, "actual": versions[name]}
                   for name, expected in RUNTIME_PINS.items() if versions[name] != expected}
    if platform.python_version() != PYTHON_PIN:
        differences["python"] = {"expected": PYTHON_PIN, "actual": platform.python_version()}
    return {"python": platform.python_version(), "packages": versions,
            "pin_differences": differences, "declared_pins_match": not differences,
            "native_execution_qualified": False}


def source_record() -> dict[str, Any]:
    parent = PROJECT_ROOT.parent
    repositories = {name: git_identity(parent / name)
                    for name in ("human-punching", "wbc-mjlab", "mjlab", "robust_sav")}
    for name in ("wbc-mjlab", "mjlab", "robust_sav"):
        repositories[name]["expected_revision"] = SOURCE_PINS[name]
        repositories[name]["pin_matches"] = repositories[name]["revision"] == SOURCE_PINS[name]
    return repositories
