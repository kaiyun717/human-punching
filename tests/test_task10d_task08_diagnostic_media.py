"""Selected native media preserves source time, world identity and outcome gaps."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching.task10d.task08_diagnostic_media import (
    Task08DiagnosticMedia, compose_frame, owned_batch_row, select_representatives,
)


class FakeRenderer:
    def __init__(self):
        self.states = []
        self.closed = False

    def render_owned(self, state, *, time_s):
        self.states.append({name: value.copy() for name, value in state.items()})
        return np.full((360, 640, 3), [12, 24, 36], dtype=np.uint8)

    def close(self):
        self.closed = True


def fixture(tick=0):
    worlds = 3
    state = SimpleNamespace(qpos=torch.arange(worlds*36).reshape(worlds, 36).double(),
        qvel=torch.zeros((worlds, 35)), mocap_pos=torch.ones((worlds, 7, 3)),
        mocap_quat=torch.tensor([1., 0., 0., 0.]).expand(worlds, 7, 4).clone())
    exposure = dict(rgb=torch.tensor([170, 20, 25], dtype=torch.uint8).expand(worlds, 376, 672, 3).clone(),
        clean_normalized=torch.full((worlds, 9, 16), .2),
        clean_masks=torch.ones((worlds, 9, 16), dtype=torch.bool),
        corrupted_normalized=torch.full((worlds, 9, 16), .7),
        corrupted_masks=torch.ones((worlds, 9, 16), dtype=torch.bool),
        delivered_normalized=torch.full((worlds, 9, 16), .9),
        delivered_masks=torch.ones((worlds, 9, 16), dtype=torch.bool),
        capture_timestamps=torch.full((worlds,), tick*.02, dtype=torch.float64),
        capture_frame_ids=torch.full((worlds,), tick, dtype=torch.long),
        delay_ticks=torch.tensor([2, 3, 4]),
        delivered_frame_ids=torch.full((worlds,), max(tick-3, 0), dtype=torch.long),
        delivered_capture_timestamps=torch.full((worlds,), max(tick-3, 0)*.02, dtype=torch.float64),
        padding=torch.full((worlds,), tick < 3, dtype=torch.bool),
        visual_age_s=torch.full((worlds,), min(tick, 3)*.02))
    return state, exposure


def test_world_row_is_selected_before_host_copy_and_owns_storage():
    class GPU:
        def __init__(self):
            self.selected = []

        def __getitem__(self, key):
            assert key == 2
            self.selected.append(key)
            return torch.tensor([17., 23.])

        def cpu(self):
            pytest.fail("Never transfer a complete batch")

    value = GPU()
    np.testing.assert_array_equal(owned_batch_row(value, 2), [17., 23.])
    assert value.selected == [2]
    array = np.arange(8).reshape(4, 2)
    row = owned_batch_row(array, 2)
    array[:] = 0
    np.testing.assert_array_equal(row, [4, 5])


def test_panel_order_uses_same_acquisition_corruption_and_explicit_invalid_mask():
    _, exposure = fixture()
    owned = {name: owned_batch_row(value, 1) for name, value in exposure.items()}
    owned["clean_masks"][0, 0] = False
    owned["clean_normalized"][0, 0] = 1
    frame = compose_frame(np.full((360, 640, 3), [12, 24, 36], np.uint8), owned,
        title="Task08 frozen policy", episode=dict(primitive_id="jab", snapshot_id=42), time_s=0)
    assert frame.shape == (864, 1280, 3)
    np.testing.assert_array_equal(frame[200, 100], [12, 24, 36])
    np.testing.assert_array_equal(frame[200, 800], [170, 20, 25])
    assert frame[460, 10, 0] == 255
    assert frame[520, 100, 0] == 46
    assert frame[520, 700, 0] == 161  # Same source .7; delivered .9 remains separate.


def test_selection_is_deterministic_retains_context_and_reports_shortages():
    rows = [dict(primitive_id="jab", snapshot_id=sid, episode_id=sid,
                 batch_index=3, world_index=sid % 64, disposition=kind,
                 safe_arrival=kind == "arrival")
            for sid, kind in [(10, "arrival"), (5, "arrival"), (9, "fist_clearance"), (1, "arrival")]]
    rows += [dict(primitive_id="cross", snapshot_id=7, disposition="running")]
    result = select_representatives(rows, ["jab", "cross"])
    reverse = select_representatives(reversed(rows), ["jab", "cross"])
    assert result == reverse
    assert [row["snapshot_id"] for row in result["selected"]] == [1, 5, 9]
    assert result["selected"][0]["batch_index"] == 3
    assert result["shortages"]["cross"] == {"success": 2, "failure": 1}
    assert result["available_counts"]["cross"]["unknown"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        select_representatives(rows+rows[:1], ["jab", "cross"])


def test_actual_video_and_trace_keep_off_tick_event_and_replay_difference(tmp_path):
    renderer = FakeRenderer()
    expected = dict(primitive_id="jab", snapshot_id=42, disposition="arrival",
                    safe_arrival=True, event_time_s=.045)
    recorder = Task08DiagnosticMedia(tmp_path / "media", None, "Task08 frozen policy",
        episode={**expected, "media_outcome_class": "success"}, renderer=renderer)
    for tick in range(3):
        state, exposure = fixture(tick)
        exposure.update(visual_frame_ids=torch.zeros((3, 4), dtype=torch.long),
            visual_timestamps=torch.zeros((3, 4), dtype=torch.float64),
            visual_padding=torch.ones((3, 4), dtype=torch.bool))
        if tick == 0:
            exposure["fault_dropout_probability"] = torch.tensor([.01, .02, .03])
        recorder.append(state, exposure, index=1, time_s=.02*tick)
    report = recorder.close({**expected, "event_time_s": .05}, expected_outcome=expected)
    assert report["status"] == "completed" and report["decoded_frames"] == 3
    assert report["first_event_s"] == .05 and report["last_capture_s"] == .04
    assert report["exact_reported_outcome_parity"] is False
    assert report["replay_comparisons"]["safe_arrival"] is True
    assert report["scheduled_captures_complete_through_end"]
    assert [row["fault_diagnostics_available"] for row in report["events"]] == [True, False, False]
    assert report["events"][1]["fault_diagnostics"] is None
    assert renderer.closed and len(renderer.states) == 3
    with np.load(tmp_path / "media" / "trace.npz", allow_pickle=False) as trace:
        assert trace["world_index"].tolist() == [1, 1, 1]
        assert trace["time_s"].tolist() == [0., .02, .04]
        assert trace["qpos"][0, 0] == 36
        assert trace["mocap_pos"].shape == (3, 7, 3)
        assert np.allclose(trace["delivered_normalized"], .9)
        assert np.allclose(trace["corrupted_normalized"], .7)
        assert "rgb" not in trace and trace["rgb_sha256"].shape == (3,)
        assert trace["visual_frame_ids"].shape == (3, 4)
        assert trace["visual_padding"].all()
        np.testing.assert_allclose(trace["actor_visual_age_s"],
            np.repeat(np.array([0., .02, .04])[:, None], 4, axis=1))


def test_no_off_tick_fabrication_or_silent_frame_cap(tmp_path):
    state, exposure = fixture()
    recorder = Task08DiagnosticMedia(tmp_path / "validation", None, "Test",
        renderer=FakeRenderer(), max_frames=1)
    with pytest.raises(ValueError, match="capture time"):
        recorder.append(state, exposure, index=0, time_s=.005)
    recorder.append(state, exposure, index=0, time_s=0.)
    with pytest.raises(RuntimeError, match="no silent truncation"):
        recorder.append(state, exposure, index=0, time_s=0.)
    report = recorder.close(dict(disposition="hit", event_time_s=.02))
    assert report["decoded_frames"] == 1
    assert report["scheduled_captures_complete_through_end"] is False
