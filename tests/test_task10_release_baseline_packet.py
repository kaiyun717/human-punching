"""Bounded migration diagnostics freeze their real inputs without submitting jobs."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import signal

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs/task10/release_baseline_02"
RELEASE = ROOT / "outputs/four-punch-milestone"
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


@pytest.fixture
def wrapper():
    spec = importlib.util.spec_from_file_location(
        "task10_release_packet_test", ROOT / "scripts/task10_release_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Use the preserved conversion, never regenerate references or simulate."""
    if not (BUNDLE / "manifest.json").is_file():
        pytest.skip("Preserved release_baseline_02 conversion is unavailable")
    spec = importlib.util.spec_from_file_location(
        "task10_release_real_packet_test", ROOT / "scripts/task10_release_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path_factory.mktemp("release_packet") / "prepared"
    # MuJoCo/GLFW may spawn a library-version query on first import. Complete
    # dependency imports before forbidding processes from the preparation call.
    module.configure_local_caches()
    importlib.import_module("human_punching.wbc.release_baseline")
    importlib.import_module("human_punching.wbc.release_baseline_probe")
    with pytest.MonkeyPatch.context() as patch:
        # Pytest uses a long unique basetemp. The actual shallow-path gate has
        # its own test below; this fixture does not execute a native compiler.
        patch.setattr(module.os, "fsencode", lambda _: b"short-unit-test-path")
        patch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("No job execution"))
        patch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("No job submission"))
        packet = module.prepare(BUNDLE, RELEASE, destination)
    return module, destination, packet


def test_preparation_is_nonlearning_bounded_and_never_submits(prepared):
    _, destination, packet = prepared
    assert packet["status"] == "prepared_not_submitted"
    assert packet["diagnostic_scope"] == "task09_reference_controller_migration"
    assert packet["conditions"] == 12
    assert packet["environments_per_condition"] == 16
    assert packet["bank_rows"] == list(range(16))
    assert packet["training"] is False and packet["learning_updates"] == 0
    assert packet["automatic_followup"] is False and packet["production_qualified"] is False
    assert packet["application_wall_seconds"] == 1680
    assert packet["allocation_wall_seconds"] == 1800
    assert packet["products_and_caches_bytes"] == 6 * 1024**3
    command = packet["submission_command"]
    assert command[:2] == ["sbatch", "--parsable"]
    for flag in ("--partition=mit_normal_gpu", "--nodes=1", "--ntasks=1",
                 "--gres=gpu:l40s:1", "--time=00:30:00", "--no-requeue"):
        assert command.count(flag) == 1
    assert command[-2:] == [str(ROOT / "scripts/task10_release_baseline.sbatch"),
                            str(destination / "packet.json")]
    assert not (destination / "run").exists()
    assert not (destination / "execution.json").exists()


def test_packet_freezes_complete_local_source_and_dependency_closure(prepared):
    module, destination, packet = prepared
    required = {str(path.relative_to(ROOT))
                for path in (ROOT / "src/human_punching/wbc").rglob("*.py")}
    required.update(("src/human_punching/__init__.py", "src/human_punching/scalar_control.py",
                     "scripts/task10_release_baseline.py", "scripts/task10_release_baseline.sbatch",
                     "requirements-wbc.lock", "requirements-wbc.in", "pyproject.toml"))
    assert required <= packet["files"].keys()
    for relative, digest in packet["files"].items():
        assert (ROOT / relative).resolve().is_relative_to(ROOT)
        assert module.sha256_file(destination / "source_snapshot" / relative) == digest
    # Frozen legacy helper imports one local module outside the WBC namespace.
    # This guards against an unnoticed helper import widening the source closure.
    for family in FAMILIES:
        tree = ast.parse((RELEASE / family / "jab_learning_source.py").read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and node.module and node.module.startswith("human_punching")}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names if alias.name.startswith("human_punching"))
        assert imports == {"human_punching.scalar_control"}
        scalar = RELEASE / family / "scalar_control_source.py"
        assert module.sha256_file(scalar) == packet["files"]["src/human_punching/scalar_control.py"]


def test_packet_binds_legacy_grid_banks_checkpoints_and_helpers(prepared):
    module, _, packet = prepared
    release_manifest = json.loads((RELEASE / "manifest.json").read_text())
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for path in BUNDLE.rglob("*"):
        if path.is_file():
            assert packet["files"][str(path.relative_to(ROOT))] == module.sha256_file(path)
    for clip in manifest["clips"]:
        legacy = BUNDLE / clip["legacy_runtime_path"]
        assert packet["files"][str(legacy.relative_to(ROOT))] == clip["legacy_runtime_sha256"]
    for family in FAMILIES:
        for name in ("checkpoint.pt", "controller_schema.json", "arguments.json", "body_metadata.json",
                     "reference.npz", "model/human.xml", "jab_learning_source.py", "scalar_control_source.py",
                     "evaluation/evaluation_bank.npz", "evaluation/evaluation_bank_manifest.json"):
            source = RELEASE / family / name
            assert packet["files"][str(source.relative_to(ROOT))] == (
                release_manifest["artifacts"][f"{family}/{name}"]["sha256"]
            )


def test_changed_legacy_grid_rejected_before_packet_creation(wrapper, tmp_path, monkeypatch):
    if not (BUNDLE / "manifest.json").is_file():
        pytest.skip("Preserved release_baseline_02 conversion is unavailable")
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    legacy = copied / manifest["clips"][0]["legacy_runtime_path"]
    with legacy.open("ab") as stream:
        stream.write(b"unverified-source-grid-change")
    destination = tmp_path / "must_not_exist"
    wrapper.configure_local_caches()
    importlib.import_module("human_punching.wbc.release_baseline")
    importlib.import_module("human_punching.wbc.release_baseline_probe")
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(ValueError, match="Source-grid runtime hash mismatch"):
        wrapper.prepare(copied, RELEASE, destination)
    assert not destination.exists()


def test_native_compiler_path_bound_precedes_output_creation(wrapper, tmp_path):
    if not (BUNDLE / "manifest.json").is_file():
        pytest.skip("Preserved release_baseline_02 conversion is unavailable")
    destination = tmp_path / ("long_native_compiler_path_" * 6)
    with pytest.raises(ValueError, match="shallow output path"):
        wrapper.prepare(BUNDLE, RELEASE, destination)
    assert not destination.exists()


def scope_record():
    return dict(schema="human_wbc_release_baseline_packet_v1", conditions=12,
                environments_per_condition=16, bank_rows=list(range(16)), training=False,
                learning_updates=0, production_qualified=False, automatic_followup=False,
                application_wall_seconds=1680, allocation_wall_seconds=1800,
                products_and_caches_bytes=6 * 1024**3)


@pytest.mark.parametrize("field,value", [
    ("conditions", 16), ("environments_per_condition", 64), ("bank_rows", list(range(16, 32))),
    ("training", True), ("learning_updates", 1), ("production_qualified", True),
    ("automatic_followup", True), ("application_wall_seconds", 3601),
    ("allocation_wall_seconds", 3601), ("products_and_caches_bytes", 12 * 1024**3),
])
def test_scope_expansion_fails_before_any_process(wrapper, tmp_path, monkeypatch, field, value):
    record = scope_record()
    record[field] = value
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(record))
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(ValueError, match="scope or resource"):
        wrapper.execute(packet)


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"])
def test_allocation_required_before_sources_or_execution(wrapper, tmp_path, monkeypatch, missing):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv(missing)
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(scope_record()))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(RuntimeError, match="allocated Slurm GPU"):
        wrapper.execute(packet)


def execution_fixture(wrapper, tmp_path, monkeypatch):
    """An inert synthetic input packet; it cannot run a policy or simulator."""
    root = tmp_path / "project"
    root.mkdir()
    packet_dir = root / "packet"
    packet_dir.mkdir()
    source = root / "frozen.txt"
    source.write_text("frozen synthetic input\n")
    archive = packet_dir / "source_snapshot/frozen.txt"
    archive.parent.mkdir()
    shutil.copy2(source, archive)
    record = scope_record()
    record.update(files={"frozen.txt": wrapper.sha256_file(source)},
                  bundle="bundle", release="release", output="packet/run")
    packet = packet_dir / "packet.json"
    packet.write_text(json.dumps(record))
    monkeypatch.setattr(wrapper, "ROOT", root)
    monkeypatch.setattr(wrapper, "required_inputs", lambda record: {"frozen.txt"})
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return root, packet, source


def test_changed_frozen_source_prevents_execution(wrapper, tmp_path, monkeypatch):
    _, packet, source = execution_fixture(wrapper, tmp_path, monkeypatch)
    source.write_text("changed after preparation\n")
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(ValueError, match="Frozen input changed"):
        wrapper.execute(packet)


@pytest.mark.parametrize("missing_map", [True, False], ids=["empty-map", "omitted-required-source"])
def test_missing_frozen_source_closure_prevents_execution(wrapper, tmp_path, monkeypatch, missing_map):
    root, packet, source = execution_fixture(wrapper, tmp_path, monkeypatch)
    record = json.loads(packet.read_text())
    if missing_map:
        record["files"] = {}
    else:
        other = root / "unrelated.txt"
        shutil.copy2(source, other)
        record["files"] = {"unrelated.txt": wrapper.sha256_file(other)}
    packet.write_text(json.dumps(record))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(ValueError, match="(?i)(closure|required|empty|missing)"):
        wrapper.execute(packet)
    assert not (packet.parent / "run").exists()
    assert not (packet.parent / "execution.json").exists()


def test_corrupted_source_archive_prevents_execution(wrapper, tmp_path, monkeypatch):
    _, packet, source = execution_fixture(wrapper, tmp_path, monkeypatch)
    archived = packet.parent / "source_snapshot/frozen.txt"
    archived.write_text("corrupted archived source despite unchanged live input\n")
    record = json.loads(packet.read_text())
    assert wrapper.sha256_file(source) == record["files"]["frozen.txt"]
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("No execution"))
    with pytest.raises(ValueError, match="(?i)(archive|snapshot)"):
        wrapper.execute(packet)
    assert not (packet.parent / "run").exists()
    assert not (packet.parent / "execution.json").exists()


def test_execution_starts_only_inference_probe_and_binds_receipt(wrapper, tmp_path, monkeypatch):
    root, packet, _ = execution_fixture(wrapper, tmp_path, monkeypatch)
    seen = []

    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    def launch(command, **kwargs):
        seen.append((command, kwargs))
        return FinishedProcess()

    monkeypatch.setattr(wrapper.subprocess, "Popen", launch)
    assert wrapper.execute(packet) == 0
    assert len(seen) == 1
    command, kwargs = seen[0]
    assert command == [str(root / ".venv-wbc/bin/python"), "-B", "-m",
                       "human_punching.wbc.release_baseline_probe", "--bundle", str(root / "bundle"),
                       "--release-root", str(root / "release"), "--output", str(root / "packet/run")]
    assert kwargs["start_new_session"] is True
    receipt = json.loads((packet.parent / "execution.json").read_text())
    assert receipt["packet_sha256"] == wrapper.sha256_file(packet)
    assert receipt["slurm_job_id"] == "fixture" and receipt["returncode"] == 0
    assert receipt["budget_stop"] is None and receipt["learning_updates"] == 0
    assert receipt["production_qualified"] is False


@pytest.mark.parametrize("budget", ["application_wall_budget", "products_and_caches_budget"])
def test_watchdog_stops_incomplete_probe_and_records_reason(wrapper, tmp_path, monkeypatch, budget):
    _, packet, _ = execution_fixture(wrapper, tmp_path, monkeypatch)

    class RunningProcess:
        pid = 12345
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, **kwargs):
            self.done = True
            return 0

    process = RunningProcess()
    signals = []
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(wrapper.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(wrapper.time, "sleep", lambda _: pytest.fail("Budget stop must be immediate"))
    monkeypatch.setattr(wrapper, "tree_bytes", lambda _: 6 * 1024**3 + 1
                        if budget == "products_and_caches_budget" else 1)
    times = iter((0., 1681., 1682.)) if budget == "application_wall_budget" else iter((0., 1.))
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: next(times))
    assert wrapper.execute(packet) == 2
    assert signals == [(12345, signal.SIGTERM)]
    receipt = json.loads((packet.parent / "execution.json").read_text())
    assert receipt["budget_stop"] == budget
    assert receipt["learning_updates"] == 0 and receipt["returncode"] == 0
