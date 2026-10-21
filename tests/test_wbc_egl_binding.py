"""Device identity selection only; CPU tests never initialize GL or CUDA."""
import pytest
from human_punching.wbc.egl_binding import select_identity


def test_select_by_uuid_not_order_or_physical_ordinal():
    records = [{"gpu_uuid": "GPU-other", "egl_device_index": 0},
               {"gpu_uuid": "GPU-allocated", "egl_device_index": 5}]
    assert select_identity(records, "GPU-allocated")["egl_device_index"] == 5
    with pytest.raises(RuntimeError):
        select_identity(records, "GPU-absent")
    with pytest.raises(RuntimeError):
        select_identity(records+[records[1]], "GPU-allocated")
