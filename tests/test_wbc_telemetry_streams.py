"""CPU stream ownership regressions; allocated native jobs test CUDA capture."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from human_punching.wbc.telemetry import streams, contacts


def shared_stream(monkeypatch, *, torch_handle=731):
    owner = SimpleNamespace(cuda_stream=731)
    monkeypatch.setattr(wp, "get_stream", lambda device: owner)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=torch_handle))
    # The old implementation registered a duplicate nonowner which subsequently
    # unregistered this still-live stream. No wrapper may be created at all.
    monkeypatch.setattr(wp, "stream_from_torch", lambda *args: pytest.fail("Duplicate Warp stream ownership"))
    return owner


def test_native_owner_is_reused_across_repeated_hook_windows(monkeypatch):
    owner = shared_stream(monkeypatch)
    for _ in range(100):
        assert streams.current_native_stream("cuda:0") is owner


def test_mismatched_streams_fail_without_silent_registration_or_switch(monkeypatch):
    shared_stream(monkeypatch, torch_handle=815)
    monkeypatch.setattr(wp, "set_stream", lambda *args: pytest.fail("Implicit stream switch"))
    with pytest.raises(RuntimeError, match="matching Warp and PyTorch"):
        streams.current_native_stream("cuda:0")


def test_unindexed_cuda_resolves_current_device_and_cpu_is_rejected(monkeypatch):
    owner = shared_stream(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    seen = []
    monkeypatch.setattr(wp, "get_stream", lambda device: seen.append(device) or owner)
    assert streams.current_native_stream("cuda") is owner
    assert seen == ["cuda:1"]
    with pytest.raises(ValueError, match="CUDA"):
        streams.current_native_stream("cpu")


def test_actual_contact_sampling_reuses_owner_for_repeated_native_calls(monkeypatch):
    import mujoco_warp as mjwarp
    owner = shared_stream(monkeypatch)
    scoped, decoded = [], []
    @contextmanager
    def scope(stream):
        assert stream is owner
        scoped.append(stream)
        yield
    monkeypatch.setattr(wp, "ScopedStream", scope)
    monkeypatch.setattr(mjwarp, "contact_force", lambda *args: decoded.append(args))
    sample = contacts.NativeContactTelemetry.__new__(contacts.NativeContactTelemetry)
    sample.force_tensor = torch.ones(1, 6)
    sample.ids, sample.safe_ids = torch.arange(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    sample.count, sample.overflow = torch.ones(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    sample.dimension, sample.address = torch.tensor([3]), torch.zeros(1, 6, dtype=torch.int32)
    sample.pyramidal, sample.data = True, SimpleNamespace(njmax=32)
    sample.model, sample.wp_ids, sample.forces = object(), object(), object()
    sample.arrays = {"worldid": torch.zeros(1, dtype=torch.int32), "geom": torch.zeros(1, 2, dtype=torch.int32),
                     "pos": torch.zeros(1, 3), "dist": torch.zeros(1)}
    sample.reducer = SimpleNamespace(device="cuda:0", reduce=lambda **kw: {"numerical_failure": torch.zeros(1, dtype=torch.bool)})
    for _ in range(8):
        assert not sample.sample(torch.zeros(1, 2, 3))["numerical_failure"].any()
    assert len(scoped) == len(decoded) == 8
    assert torch.count_nonzero(sample.force_tensor) == 0


def test_observer_admission_binds_stream_owner_and_actual_comparators():
    from human_punching.wbc.capacity_evidence import OBSERVER_REQUIRED_SOURCES
    for name in ("streams.py", "native_qualification.py", "posthoc_qualification.py", "cycle_qualification.py"):
        assert "src/human_punching/wbc/telemetry/"+name in OBSERVER_REQUIRED_SOURCES


@pytest.mark.parametrize("corrupt", [False, True])
def test_private_capture_regression_checks_values_and_uses_existing_owner(monkeypatch, corrupt):
    owner = shared_stream(monkeypatch)
    graph, operations = object(), []
    monkeypatch.setattr(wp, "array", lambda data,**kwargs: torch.tensor(data))
    monkeypatch.setattr(wp, "zeros", lambda count,**kwargs: torch.zeros(count))
    monkeypatch.setattr(wp, "synchronize_stream", lambda stream: operations.append(("sync",stream)))
    @contextmanager
    def capture(*,stream):
        assert stream is owner
        operations.append(("capture",stream))
        yield SimpleNamespace(graph=graph)
    monkeypatch.setattr(wp,"ScopedCapture",capture)
    monkeypatch.setattr(wp,"copy",lambda target,source: target.copy_(source+(1 if corrupt else 0)))
    monkeypatch.setattr(wp,"capture_launch",lambda g,*,stream: operations.append(("launch",g,stream)))
    monkeypatch.setattr(wp,"to_torch",lambda value:value)
    if corrupt:
        with pytest.raises(RuntimeError,match="capture/copy roundtrip"):
            streams.verify_native_stream_capture("cuda:0")
    else:
        result=streams.verify_native_stream_capture("cuda:0")
        assert result["passed"] and result["physics_steps"]==0
        assert result["stream_owner_retained"] and result["copied_values_exact"]
    assert [operation[0] for operation in operations]==["sync","capture","launch","sync"]
    assert operations[2]==("launch",graph,owner)
