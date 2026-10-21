"""Preserve ownership of the shared native Warp/PyTorch CUDA stream."""
from __future__ import annotations

import torch


def current_native_stream(device):
    """Return the existing Warp stream only when Torch uses that same handle.

    Task 10 native entry points explicitly place Torch work on Warp's owned
    stream. Calling ``wp.stream_from_torch`` on that already-owned handle would
    create another nonowning Warp wrapper. In Warp 1.14, destroying that wrapper
    unregisters the handle even while the original owner remains alive, so a
    later simulation cannot capture CUDA graphs. Reuse the owner instead.

    This helper neither switches streams nor invents synchronization for a
    mismatched caller. The caller must establish the shared stream scope before
    constructing or using observer state.
    """
    import warp as wp
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native telemetry requires an explicit CUDA stream")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    native = wp.get_stream(str(device))
    current = torch.cuda.current_stream(device)
    if native.cuda_stream != current.cuda_stream:
        raise RuntimeError("Native telemetry requires matching Warp and PyTorch stream handles; enter native_stream_scope first")
    return native


def verify_native_stream_capture(device):
    """Allocated qualification only: test graph registration after observer use.

    A two-value device copy cannot alter model state or integrate physics. It
    exposes the lost-registration defect even when previously captured physics
    graphs can still launch on the live CUDA handle.
    """
    import warp as wp
    stream = current_native_stream(device)
    source = wp.array([1., -2.], dtype=wp.float32, device=str(device))
    target = wp.zeros(2, dtype=wp.float32, device=str(device))
    wp.synchronize_stream(stream)
    with wp.ScopedCapture(stream=stream) as capture:
        wp.copy(target, source)
    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    copied = torch.equal(wp.to_torch(target).cpu(), torch.tensor([1., -2.]))
    retained = current_native_stream(device) is stream
    if not copied or not retained:
        raise RuntimeError("Native stream capture/copy roundtrip changed its owner or values")
    return {"passed": True, "graph_captures": 1, "graph_launches": 1,
            "physics_steps": 0, "stream_owner_retained": retained, "copied_values_exact": copied}
