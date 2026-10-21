"""Phase 0 authorization gates cannot become training or an unbounded run."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase0_packet", ROOT / "scripts/task10b_phase0.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def scope(phase):
    allocation, application, products = packet.LIMITS[phase]
    return dict(schema=packet.SCHEMA, task="10B", program_phase=0, phase=phase,
        training=False, learning_updates=0, automatic_followup=False,
        allocation_wall_seconds=allocation, application_wall_seconds=application,
        products_and_caches_bytes=products, partition="mit_normal_gpu",
        modes=["contract", "diagnostic"] if phase == "preflight" else ["parity"])


@pytest.mark.parametrize("phase", ["preflight", "parity"])
def test_bounded_phase0_scope(phase):
    record = scope(phase)
    packet.validate_scope(record)
    assert record["allocation_wall_seconds"] <= 4 * 3600
    assert record["application_wall_seconds"] < record["allocation_wall_seconds"]


@pytest.mark.parametrize("key,value", [("training", True), ("learning_updates", 1),
    ("automatic_followup", True), ("program_phase", 1), ("allocation_wall_seconds", 14401),
    ("modes", ["diagnostic"]), ("partition", "unapproved")])
def test_changed_scope_rejected(key, value):
    record = scope("preflight")
    record[key] = value
    with pytest.raises(ValueError):
        packet.validate_scope(record)


def test_unknown_execution_phase_rejected():
    record = scope("parity")
    record["phase"] = "training"
    with pytest.raises(ValueError):
        packet.validate_scope(record)


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    runtime = tmp_path / "runtime.json"
    monkeypatch.setattr(packet, "RUNTIME", runtime)
    local = tmp_path / "python"
    local.mkdir()
    member = local / "stdlib.py"
    member.write_text("preserved bytes\n")
    link = local / "alias.py"
    link.symlink_to("stdlib.py")
    manifest = runtime.with_name("runtime_interpreter_manifest.json")
    manifest.write_text(json.dumps([
        dict(path="stdlib.py", bytes=member.stat().st_size, sha256=packet.sha256_file(member)),
        dict(path="alias.py", symlink="stdlib.py")]))
    metadata = tmp_path / "site-packages/fake.dist-info/METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Version: 1.0\n")
    distribution = SimpleNamespace(files=[Path("fake.dist-info/METADATA")],
        locate_file=lambda item: metadata.parent.parent / item,
        read_text=lambda name: metadata.read_text())
    monkeypatch.setattr(packet.importlib.metadata, "distribution", lambda name: distribution)
    current = dict(declared_pins_match=True)
    monkeypatch.setattr(packet, "runtime_record", lambda: current)
    runtime.write_text(json.dumps(dict(local_python_root=str(local),
        interpreter_manifest_sha256=packet.sha256_file(manifest),
        distribution_metadata_hashes={"fake/METADATA": dict(bytes=metadata.stat().st_size,
            sha256=packet.sha256_file(metadata))})))
    return dict(runtime=current), member, link, metadata


def test_runtime_checks_interpreter_and_installed_metadata(runtime_fixture):
    record, *_ = runtime_fixture
    packet.verify_runtime_identity(record)


@pytest.mark.parametrize("component", ["stdlib", "symlink", "metadata"])
def test_runtime_payload_or_metadata_change_rejected(runtime_fixture, component):
    record, member, link, metadata = runtime_fixture
    if component == "stdlib":
        member.write_text("changed bytes\n")
    elif component == "symlink":
        link.unlink()
        link.symlink_to("./stdlib.py")
    else:
        metadata.write_text("Version: 2.0\n")
    with pytest.raises(ValueError):
        packet.verify_runtime_identity(record)


@pytest.fixture
def preflight_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    folder = tmp_path / "preflight"
    folder.mkdir()
    previous_path = folder / "packet.json"
    previous = dict(phase="preflight", bundle="bundle", release="release", runtime={"pin": "1"},
        interpreter_path="local/python", interpreter_sha256="python-hash", files={"source.py": "source-hash"},
        output="preflight/run")
    previous_path.write_text(json.dumps(previous))
    monkeypatch.setattr(packet, "verify", lambda path, **kwargs: previous)
    modes = []
    for mode in ("contract", "diagnostic"):
        audit = folder / "run" / mode / "audit.json"
        audit.parent.mkdir(parents=True)
        audit.write_text('{"status": "passed"}')
        modes.append(dict(mode=mode, returncode=0, audit_status="passed", audit_sha256=packet.sha256_file(audit)))
    receipt = dict(status="passed", completed_modes=["contract", "diagnostic"], modes=modes,
        packet_sha256=packet.sha256_file(previous_path))
    gate = folder / "execution.json"
    gate.write_text(json.dumps(receipt))
    record = json.loads(json.dumps(previous))
    record.update(phase="parity", preflight_receipt="preflight/execution.json",
        preflight_sha256=packet.sha256_file(gate), preflight_packet="preflight/packet.json",
        preflight_packet_sha256=packet.sha256_file(previous_path))
    return record, gate, receipt


def test_parity_gate_binds_actual_preflight_packet(preflight_fixture):
    record, *_ = preflight_fixture
    packet.verify_preflight_identity(record)


@pytest.mark.parametrize("key", ["bundle", "release", "runtime", "files", "interpreter_sha256"])
def test_parity_cannot_reuse_preflight_for_changed_inputs(preflight_fixture, key):
    record, *_ = preflight_fixture
    record[key] = "changed"
    with pytest.raises(ValueError, match="Parity inputs differ"):
        packet.verify_preflight_identity(record)


def test_preflight_receipt_cannot_claim_a_different_packet(preflight_fixture):
    record, gate, receipt = preflight_fixture
    receipt["packet_sha256"] = "another-packet"
    gate.write_text(json.dumps(receipt))
    record["preflight_sha256"] = packet.sha256_file(gate)
    with pytest.raises(ValueError, match="does not bind"):
        packet.verify_preflight_identity(record)


def test_verification_failure_keeps_zero_execution_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "writable_path", lambda path: Path(path))
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    path = tmp_path / "outputs/task10b/failure/packet.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"phase":"preflight"}')
    def reject(path):
        raise ValueError("source identity mismatch")
    monkeypatch.setattr(packet, "verify", reject)
    def write(path, value):
        Path(path).write_text(json.dumps(value))
    monkeypatch.setattr(packet, "write_json", write)
    assert packet.execute(path) == 2
    started = json.loads((path.parent / "execution_started.json").read_text())
    final = json.loads((path.parent / "execution.json").read_text())
    assert started["status"] == "running"
    assert final["status"] == "failed"
    assert final["actual_world_physics_transitions"] == final["actual_actor_updates"] == 0
    assert final["all_native_modes_unattempted"]
    assert final["exception"]["message"] == "source identity mismatch"


def test_launcher_uses_bound_job_directory():
    source = (ROOT / "scripts/task10b_phase0.sbatch").read_text()
    assert 'task10b_phase0_root="$(pwd -P)"' in source
    assert 'cd -- "$SLURM_SUBMIT_DIR"' not in source


@pytest.fixture
def shell_fixture(tmp_path):
    root = tmp_path / "physical_project"
    alias = tmp_path / "submitted_alias"
    root.mkdir()
    alias.symlink_to(root, target_is_directory=True)
    wrapper = root / "scripts/task10b_phase0.py"
    wrapper.parent.mkdir()
    wrapper.write_text("# stub wrapper\n")
    runtime = root / "outputs/task10b/p0prep_20261020_01/runtime.env"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# isolated shell regression fixture\n")
    python = root / ".venv-task10b/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$TASK10B_LAUNCH_CAPTURE"\n')
    python.chmod(0o755)
    submitted = alias / "outputs/task10b/packet_fixture/packet.json"
    submitted.parent.mkdir()
    submitted.write_text("{}\n")
    capture = root / "captured_arguments.txt"
    env = dict(os.environ, SLURM_JOB_ID="test-only", SLURM_SUBMIT_DIR=str(tmp_path),
               TASK10B_LAUNCH_CAPTURE=str(capture))
    return root, submitted, capture, env


def test_launcher_resolves_submitted_alias_before_root_check(shell_fixture):
    root, submitted, capture, env = shell_fixture
    result = subprocess.run(["bash", str(ROOT / "scripts/task10b_phase0.sbatch"), str(submitted)],
                            cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ["-B", str(root / "scripts/task10b_phase0.py"),
                                                "execute", "--packet", str(submitted.resolve())]
    assert "resolved_root=" + str(root) in result.stderr
    assert "submitted_packet=" + str(submitted) in result.stderr


def test_launcher_rejects_outside_packet_with_diagnostic(shell_fixture):
    root, submitted, capture, env = shell_fixture
    outside = root / "outside.json"
    outside.write_text("{}\n")
    result = subprocess.run(["bash", str(ROOT / "scripts/task10b_phase0.sbatch"), str(outside)],
                            cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "outside the resolved Task10B output root" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("nodes", [["node3006"], ["node3003", "node3006"], []])
def test_node_exclusions_are_recorded_scope_choices(nodes):
    record = scope("preflight")
    record["excluded_nodes"] = nodes
    packet.validate_scope(record)


def test_known_cuda_failure_node_is_excluded_by_default():
    assert packet.prepare.__kwdefaults__["excluded_nodes"] == ("node3006",)


@pytest.mark.parametrize("nodes", [["--time=100"], ["node3006,node3003"], ["node3006", "node3006"], "node3006"])
def test_malformed_node_exclusions_rejected(nodes):
    record = scope("preflight")
    record["excluded_nodes"] = nodes
    with pytest.raises(ValueError, match="Excluded nodes"):
        packet.validate_scope(record)


@pytest.mark.parametrize("failure", [None, "nvidia", "torch", "malformed", "timeout"])
def test_cuda_health_receipt_preserves_diagnostics(tmp_path, monkeypatch, failure):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=2 if failure == "nvidia" else 0,
                                   stdout="0, NVIDIA L40S, GPU-test, pci-test, driver-test\n", stderr="")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        payload = dict(cuda_available=True, device_count=1, initialized=True, tensor_roundtrip=True)
        if failure == "torch":
            payload.update(cuda_available=False, initialized=False,
                           exception=dict(type="RuntimeError", message="CUDA unknown error"))
        return SimpleNamespace(returncode=2 if failure == "torch" else 0,
            stdout="unexpected output" if failure == "malformed" else json.dumps(payload),
            stderr="CUDA warning" if failure == "torch" else "")
    monkeypatch.setattr(packet.subprocess, "run", run)
    monkeypatch.setattr(packet, "write_json", lambda path, value: Path(path).write_text(json.dumps(value)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("UNRELATED_SECRET_TEST", "must-not-appear")
    output = tmp_path / "health.json"
    result = packet.cuda_health_receipt(output)
    assert result["status"] == ("passed" if failure is None else "failed")
    assert result["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert "must-not-appear" not in output.read_text()
    assert result["physics_steps"] == result["actor_updates"] == result["learning_updates"] == 0
    assert len(calls) == 2
    assert calls[0][1]["timeout"] == 15 and calls[1][1]["timeout"] == 60
    if failure == "torch":
        assert result["torch"]["stderr"] == "CUDA warning"
        assert result["torch"]["result"]["exception"]["message"] == "CUDA unknown error"


def test_cuda_health_failure_never_starts_native_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "writable_path", lambda path: Path(path))
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    path = tmp_path / "outputs/task10b/cuda_failure/packet.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"phase":"preflight"}')
    monkeypatch.setattr(packet, "verify", lambda path: scope("preflight"))
    monkeypatch.setattr(packet, "write_json", lambda path, value: Path(path).write_text(json.dumps(value)))
    def failed_health(destination):
        result = dict(status="failed", physics_steps=0)
        destination.write_text(json.dumps(result))
        return result
    monkeypatch.setattr(packet, "cuda_health_receipt", failed_health)
    def forbidden(*args, **kwargs):
        raise AssertionError("Native process must not start after a CUDA health failure")
    monkeypatch.setattr(packet.subprocess, "Popen", forbidden)
    assert packet.execute(path) == 2
    result = json.loads((path.parent / "execution.json").read_text())
    assert result["cuda_health_status"] == "failed"
    assert result["native_processes_started"] == result["actual_world_physics_transitions"] == 0
    assert result["all_native_modes_unattempted"] and not result["modes"]
