"""CPU exercise of capture reporting switches with real Task08 noise/delay.

The renderer and FK calls are stand-ins; this does not claim native GPU parity.
Noise, delay, capture assembly and array ownership use the actual implementation.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from human_punching.task10d import camera as module


class Renderer:
    def __init__(self, count):
        self.count, self.capture_index = count, 0
        self.depth = torch.full((count, 9, 16), .8)
        self.masks = torch.ones_like(self.depth, dtype=torch.bool)
        self.masks[:, 1, 1] = False
        self.packed_rgb = torch.full((count, 376, 672), (170 << 16) | (20 << 8) | 25,
                                     dtype=torch.int32).view(torch.uint32)
        self.raw_depth = self.depth
        self.segmentation = torch.zeros((count, 9, 16, 2), dtype=torch.int32)
        self.foreground = self.masks
        self.clean_foreground = self.masks

    def capture(self):
        self.depth.fill_(.8 + .1*self.capture_index)
        self.depth[~self.masks] = 3.
        self.capture_index += 1
        return SimpleNamespace(depth=self.depth, masks=self.masks,
            clean_depth=self.depth+.1*self.masks, clean_masks=self.masks,
            camera_position=torch.zeros((self.count, 3)),
            camera_rotation=torch.eye(3).expand(self.count, 3, 3), capture_index=self.capture_index)


def fixture():
    from boxing.perception.image_faults import ForegroundVisualFaultProcess
    from boxing.perception.latency import VisualDelayBuffer
    value = module.Task08Camera.__new__(module.Task08Camera)
    value.count, value.device = 3, torch.device("cpu")
    value._initialized, value._previous_time = False, None
    value._ordered = nullcontext
    value._mask = torch.ones(value.count, dtype=torch.bool)
    value._h2_geom_ids, value._glove_geom_ids = torch.tensor([2, 3, 4]), torch.tensor([3, 4])
    value._rays = torch.ones((9, 16, 3))
    value.ids = SimpleNamespace(geom_object_type=5)
    value.sim = SimpleNamespace(wp_model=object(), wp_data=object(),
        mj_model=SimpleNamespace(ngeom=5),
        data=SimpleNamespace(geom_xpos=torch.zeros((value.count, 5, 3))))
    value.renderer = Renderer(value.count)
    value.faults = ForegroundVisualFaultProcess(value.count, device="cpu", seed=12345)
    value.delay = VisualDelayBuffer(value.count, device="cpu", seed=12346)
    return value


def test_optional_reporting_preserves_exact_faults_and_delayed_packets(monkeypatch):
    import mujoco_warp
    calls = {"fk": 0, "camera": 0, "visibility": 0, "counts": 0}
    def counted(name, result=None):
        def execute(*args, **kwargs):
            calls[name] += 1
            return result
        return execute
    monkeypatch.setattr(mujoco_warp, "kinematics", counted("fk"))
    monkeypatch.setattr(mujoco_warp, "camlight", counted("camera"))
    monkeypatch.setattr(module, "_visibility_counts", counted("visibility", {"visible_test": torch.ones(3)}))
    monkeypatch.setattr(module, "_glove_observation_counts", counted("counts", {"count_test": torch.ones(3)}))
    full, minimal = fixture(), fixture()
    first_owned = None
    for tick in range(6):
        expected = full.capture(tick*.02)  # Unchanged defaults: owned NumPy, RGB, diagnostics.
        actual = minimal.capture(tick*.02, as_numpy=False, diagnostics=False, include_rgb=False)
        assert "rgb" in expected and "rgb" not in actual
        assert any(key.startswith("fault_") for key in expected)
        assert not any(key.startswith("fault_") for key in actual)
        for key, tensor in actual.items():
            assert isinstance(tensor, torch.Tensor)
            np.testing.assert_array_equal(expected[key], tensor.numpy())
        for key, tensor in full.faults.diagnostics().items():
            torch.testing.assert_close(tensor, minimal.faults.diagnostics()[key], rtol=0, atol=0)
        for key, tensor in full.delay.state().items():
            torch.testing.assert_close(tensor, minimal.delay.state()[key], rtol=0, atol=0)
        if tick == 0:
            first_owned = expected["acquired_depth"]
            assert first_owned[0, 0, 0] == np.float32(.8)
        else:
            assert first_owned[0, 0, 0] == np.float32(.8)  # Does not alias later source storage.
    assert calls == {"fk": 12, "camera": 12, "visibility": 6, "counts": 6}
    assert full._previous_time == minimal._previous_time == .1


def test_tensor_rgb_and_optional_source_are_actual_current_exposure(monkeypatch):
    import mujoco_warp
    monkeypatch.setattr(mujoco_warp, "kinematics", lambda *args: None)
    monkeypatch.setattr(mujoco_warp, "camlight", lambda *args: None)
    camera = fixture()
    exposure = camera.capture(0., as_numpy=False, diagnostics=False,
                               include_rgb=True, include_source=True)
    assert exposure["rgb"].shape == (3, 376, 672, 3)
    assert exposure["rgb"].dtype == torch.uint8
    assert exposure["rgb"][1, 100, 200].tolist() == [170, 20, 25]
    assert exposure["source_depth"] is camera.renderer.raw_depth
    assert exposure["capture_frame_ids"].tolist() == [0, 0, 0]
    assert exposure["padding"].all()
    assert exposure["delay_ticks"].tolist() == camera.delay.delay_ticks.tolist()
