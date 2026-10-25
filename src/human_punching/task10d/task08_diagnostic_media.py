"""Selected Task08 four-panel clips from actual native evaluation exposures.

Clockwise panels match Task08: isometric, acquired RGB, same-acquisition
corrupted C depth/mask, clean simulator-ID depth/mask. Delivered actor packets
are recorded separately, never relabeled as the current source exposure.
This module does not acquire a camera frame or advance any simulation.
"""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np

FPS = 50
STATE_FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat")
PANEL_ORDER = ("isometric", "source_rgb", "corrupted_C_depth", "uncorrupted_GT_depth")
DEPTH_FIELDS = tuple(f"{kind}_{part}" for kind in
    ("clean", "acquired", "corrupted", "delivered", "delivered_clean")
    for part in ("depth", "normalized", "masks"))
METADATA_FIELDS = (
    "capture_timestamps", "capture_frame_ids", "delivered_capture_timestamps",
    "delivery_timestamps", "delivered_frame_ids", "delivered_source_tick",
    "delay_ticks", "padding", "visual_age_s", "camera_position", "camera_rotation",
    "renderer_capture_index", "visual_frame_ids", "visual_timestamps", "visual_padding",
    "visual_delivery_timestamps", "proprioception_timestamps",
)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value {type(value).__name__}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def owned_batch_row(value: Any, index: int) -> np.ndarray:
    """Select one world before host transfer; never copy the full GPU batch."""
    value = getattr(value, "_tensor", value)
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or index < 0:
        raise ValueError("Media world index must be a nonnegative integer")
    row = value[index]
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    return np.array(row, copy=True)


def _field(data: Any, name: str) -> Any:
    return data[name] if isinstance(data, Mapping) else getattr(data, name)


def _outcome_class(row: Mapping[str, Any]) -> str:
    """Only explicit successes/failures; missing or invalid attempts stay unknown."""
    disposition = str(row.get("disposition", "")).lower()
    if disposition in {"unattempted", "unknown", "running", "pending", "rejected",
                       "numerical_failure", "fatal", "scheduler_unready"}:
        return "unknown"
    explicit = row.get("safe_arrival", row.get("success"))
    if isinstance(explicit, (bool, np.bool_)):
        return "success" if explicit else "failure"
    if disposition in {"safe_arrival", "arrival"}:
        return "success"
    if disposition in {"fist_clearance", "head_hit", "hit", "collision", "unsafe",
                       "fall", "fell", "uprightness", "horizon", "timeout",
                       "censored", "failure"}:
        return "failure"
    return "unknown"


def select_representatives(outcomes: Iterable[Mapping[str, Any]],
                           primitive_ids: Iterable[str], *,
                           successes: int = 2, failures: int = 1) -> dict[str, Any]:
    """First snapshot IDs per outcome class; deterministic, with honest shortages.

    Call after all evaluation outcomes are finalized. Returned records retain
    their batch/world/seed identity so replay can preserve the original context.
    This selection is illustrative and never estimates a success rate.
    """
    primitive_ids = tuple(primitive_ids)
    if len(set(primitive_ids)) != len(primitive_ids) or min(successes, failures) < 0:
        raise ValueError("Require unique primitives and nonnegative requested counts")
    grouped = {pid: {"success": [], "failure": [], "unknown": []} for pid in primitive_ids}
    seen = set()
    for original in outcomes:
        row = deepcopy(dict(original))
        pid, sid = row["primitive_id"], int(row["snapshot_id"])
        if pid not in grouped:
            raise ValueError(f"Unexpected primitive {pid}")
        if (pid, sid) in seen:
            raise ValueError(f"Duplicate evaluation outcome {(pid, sid)}")
        seen.add((pid, sid))
        grouped[pid][_outcome_class(row)].append(row)
    selected, counts, shortages = [], {}, {}
    for pid, classes in grouped.items():
        counts[pid] = {name: len(rows) for name, rows in classes.items()}
        shortages[pid] = {}
        for category, requested in (("success", successes), ("failure", failures)):
            ordered = sorted(classes[category], key=lambda row:
                             (int(row["snapshot_id"]), str(row.get("attempt_id", ""))))
            for ordinal, row in enumerate(ordered[:requested], 1):
                selected.append({**row, "media_outcome_class": category,
                                 "media_ordinal": ordinal})
            shortages[pid][category] = max(0, requested-len(ordered))
    return dict(schema="task10d_task08_representative_selection_v1",
        selection="Per primitive and explicit outcome class: ascending snapshot ID; no replacements or fabricated missing categories",
        outcome_selected_illustrations=True, performance_sample=False,
        requested_per_primitive={"success": successes, "failure": failures},
        requested_total=len(primitive_ids)*(successes+failures), selected_total=len(selected),
        available_counts=counts, shortages=shortages, selected=selected)


class IndexedExternalRenderer:
    """Task08 isometric geometry, including every prescribed H2 mocap link.

    Only the requested world's current state crosses to the host. The separate
    copied display model performs FK, with no physics step or frame cache.
    """
    def __init__(self, host_model: Any):
        import mujoco
        from .media import bind_allocated_egl
        self.binding = bind_allocated_egl()
        self.mj, self.model = mujoco, copy(host_model)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 640, 360
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.iso = mujoco.MjvCamera()
        self.iso.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.iso.distance, self.iso.azimuth, self.iso.elevation = 3.2, 130., -18.
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = (1, 1, 1, 0, 0, 0)

    def render(self, native_g1_data: Any, *, index: int, time_s: float) -> np.ndarray:
        state = {name: owned_batch_row(_field(native_g1_data, name), index)
                 for name in STATE_FIELDS}
        return self.render_owned(state, time_s=time_s)

    def render_owned(self, state: Mapping[str, np.ndarray], *, time_s: float) -> np.ndarray:
        for name in STATE_FIELDS:
            getattr(self.data, name)[:] = state[name]
        self.data.time = float(time_s)
        self.mj.mj_forward(self.model, self.data)
        self.iso.lookat[:] = self.data.qpos[:3] + np.array([.2, 0., .15])
        self.renderer.update_scene(self.data, camera=self.iso, scene_option=self.option)
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = True
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def compose_frame(isometric: np.ndarray, exposure: Mapping[str, np.ndarray], *,
                  title: str, episode: Mapping[str, Any], time_s: float,
                  event: Any = None, status: Any = "running") -> np.ndarray:
    """Exact usual Task08 panel positions, sizing, depth palette and source timing."""
    from PIL import Image, ImageDraw
    from boxing.perception.camera_media import _letterbox, _depth_mask_panel
    if isometric.shape != (360, 640, 3) or exposure["rgb"].shape != (376, 672, 3):
        raise ValueError("Require actual 672x376 RGB and 640x360 isometric RGB")
    canvas = Image.new("RGB", (1280, 864), "#10151c")
    canvas.paste(_letterbox(isometric), (0, 72))
    canvas.paste(_letterbox(exposure["rgb"]), (640, 72))
    canvas.paste(_depth_mask_panel(exposure["clean_normalized"], exposure["clean_masks"]), (0, 456))
    canvas.paste(_depth_mask_panel(exposure["corrupted_normalized"], exposure["corrupted_masks"]), (640, 456))
    draw = ImageDraw.Draw(canvas)
    pid, sid = episode.get("primitive_id", "unknown"), episode.get("snapshot_id", "unknown")
    selection = episode.get("media_outcome_class", "selected")
    selection_label = f"{selection}-selected" if selection in ("success", "failure") else "selected"
    draw.text((8, 6), f"{title} | {pid} | snapshot {sid} | {selection_label} illustration", fill="white")
    delay = int(exposure["delay_ticks"])
    clock = (f"Capture {int(exposure['capture_frame_ids'])} at {time_s:.3f}s | "
             f"intended delivery {time_s+delay/FPS:.3f}s | {20*delay}ms delay | status {status}")
    draw.text((8, 27), clock, fill="white")
    draw.text((8, 49), "Isometric: same native G1 and prescribed whole-body H2 state", fill="white")
    draw.text((648, 49), "Actual mounted 672x376 RGB (aspect preserved)", fill="white")
    draw.text((8, 434), "Uncorrupted simulator-ID GT foreground depth/mask", fill="white")
    draw.text((648, 434), "RGB HSV -> masked depth pool -> corrupted C depth/mask", fill="white")
    draw.text((8, 820), "All panels share capture time. Actor consumes delayed C; actual delivered depths/IDs/ages/padding are archived.", fill="white")
    note = "Black 0.1m to gray 3m; white=missing. First event may fall between real 50Hz captures."
    if event is not None:
        note += " Event: " + str(event)
    draw.text((8, 843), note, fill="white")
    return np.asarray(canvas)


class Task08DiagnosticMedia:
    """Stream one selected clip and retain only compact verification arrays.

    Caller supplies actual 20 ms captures through the first terminal event and
    calls close with the final outcome. No off-tick terminal image is invented.
    Replay the original full vector batch/seeds; this helper cannot establish
    outcome parity by itself. All failures/unknown parity remain in media.json.
    """
    def __init__(self, output: Path, host_model: Any, title: str, *,
                 episode: Mapping[str, Any] | None = None, max_frames: int = 2000,
                 renderer: Any = None, writer_factory: Any = None,
                 verify_video: bool = True):
        import imageio.v2 as imageio
        if max_frames <= 0 or not isinstance(title, str) or not title.isprintable():
            raise ValueError("Positive frame bound and printable title required")
        self.output = Path(output).resolve()
        project = Path(__file__).resolve().parents[3]
        if not self.output.is_relative_to(project):
            raise ValueError("Media outputs must remain inside human-punching")
        self.output.mkdir(parents=True, exist_ok=False)
        self.title, self.episode = title, deepcopy(dict(episode or {}))
        self.max_frames, self.verify_video = int(max_frames), verify_video
        self.renderer = renderer if renderer is not None else IndexedExternalRenderer(host_model)
        factory = writer_factory or imageio.get_writer
        self.video = self.output / "four_panel.mp4"
        self.writer = factory(self.video, fps=FPS, codec="libx264", quality=7,
            macro_block_size=16, ffmpeg_params=["-threads", "1"])
        self.frames: list[dict[str, np.ndarray]] = []
        self.status_rows: list[dict[str, Any]] = []
        self.closed = False
        self.field_names: set[str] | None = None

    def append(self, native_g1_data: Any, exposure: Mapping[str, Any], index: int = 0,
               *, time_s: float, event: Any = None, status: Any = "running") -> None:
        if self.closed:
            raise RuntimeError("Media is already closed")
        if len(self.frames) >= self.max_frames:
            raise RuntimeError("Selected clip exceeded its explicit frame bound; no silent truncation")
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Capture time must be finite and nonnegative")
        required = ("rgb", "clean_normalized", "clean_masks", "corrupted_normalized",
                    "corrupted_masks", "capture_timestamps", "capture_frame_ids", "delay_ticks")
        missing = set(required)-set(exposure)
        if missing:
            raise ValueError(f"Missing actual exposure fields: {sorted(missing)}")
        fields = set(required) | (set(exposure) & set(DEPTH_FIELDS+METADATA_FIELDS))
        owned = {key: owned_batch_row(exposure[key], index) for key in fields}
        # Attribution/fault diagnostics may be requested only at reset. Retain
        # them when actually supplied, without manufacturing subsequent values
        # or changing the fixed-shape state/perception trace schema.
        optional_faults = {key: owned_batch_row(exposure[key], index)
                           for key in exposure if key.startswith("fault_")}
        if not np.isclose(float(owned["capture_timestamps"]), time_s, atol=1e-7, rtol=0):
            raise ValueError("Native state time must equal the actual source capture time")
        if owned["rgb"].dtype != np.uint8 or owned["rgb"].shape != (376, 672, 3):
            raise ValueError("Actual uint8 672x376 acquired RGB required")
        for kind in ("clean", "corrupted"):
            depth, mask = owned[f"{kind}_normalized"], owned[f"{kind}_masks"]
            if depth.shape != (9, 16) or mask.shape != (9, 16) or mask.dtype != np.bool_:
                raise ValueError("Task08 panels require 16x9 depths and explicit boolean masks")
            if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)) or np.any(depth[~mask] != 1):
                raise ValueError("Normalized depths require [0,1] and invalid sentinel1")
        if owned["capture_frame_ids"].dtype.kind not in "iu" or int(owned["capture_frame_ids"]) < 0:
            raise ValueError("Actual source capture ID must be a nonnegative integer")
        if self.frames:
            previous = self.frames[-1]
            if not np.isclose(time_s-float(previous["time_s"]), 1/FPS, atol=1e-7, rtol=0):
                raise ValueError("Only consecutive real 20 ms source captures may be encoded")
            if int(owned["capture_frame_ids"])-int(previous["capture_frame_ids"]) != 1:
                raise ValueError("Actual source frame IDs must be consecutive")
        state = {name: owned_batch_row(_field(native_g1_data, name), index) for name in STATE_FIELDS}
        if any(not np.isfinite(value).all() for value in state.values()):
            raise ValueError("Native media state is nonfinite")
        trace = {key: value for key, value in owned.items() if key != "rgb"}
        if "visual_timestamps" in owned:
            trace["actor_visual_age_s"] = time_s-owned["visual_timestamps"]
        trace.update(state, time_s=np.asarray(time_s), world_index=np.asarray(index),
                     rgb_sha256=np.asarray(hashlib.sha256(owned["rgb"].tobytes()).hexdigest()))
        if self.field_names is not None and set(trace) != self.field_names:
            raise ValueError("Exposure trace fields changed within a selected episode")
        view = self.renderer.render_owned(state, time_s=time_s)
        composed = compose_frame(view, owned, title=self.title, episode=self.episode,
                                  time_s=time_s, event=event, status=status)
        self.writer.append_data(composed)
        if not self.frames:
            from PIL import Image
            Image.fromarray(composed).save(self.output / "first.png")
        self.field_names = set(trace)
        self.frames.append(trace)
        self.status_rows.append(dict(time_s=float(time_s), event=deepcopy(event), status=deepcopy(status),
            fault_diagnostics_available=bool(optional_faults), fault_diagnostics=optional_faults or None))

    def close(self, outcome: Mapping[str, Any] | None = None, *,
              expected_outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("Media is already closed")
        self.closed = True
        try:
            self.writer.close()
        finally:
            self.renderer.close()
        actual = deepcopy(dict(outcome or {}))
        expected = deepcopy(dict(expected_outcome or {}))
        report = dict(schema="task10d_task08_selected_four_panel_v1", status="verifying",
            title=self.title, episode=self.episode, outcome=actual, expected_outcome=expected or None,
            frames=len(self.frames), fps=FPS, dimensions=[1280, 864],
            panel_order_clockwise=list(PANEL_ORDER), source_aligned=True,
            delivered_C_in_trace=True, raw_RGB_trace="SHA256 per source frame; composed MP4 preserves visible RGB panel",
            age_metadata="visual_age_s is supplied packet age; actor_visual_age_s is capture time minus archived actor-history source timestamps, when supplied",
            recording_semantics="Real 20ms captures through first event; exact event time remains separate",
            selection="Outcome-selected illustration; not a performance sample",
            physics_steps=0, camera_acquisitions=0,
            helper_sha256=_sha256(Path(__file__)), files={}, events=self.status_rows)
        comparisons = {}
        for key in ("primitive_id", "snapshot_id", "disposition", "safe_arrival", "success",
                    "observed_hit", "event_time_s", "end_time_s"):
            if key in expected:
                comparisons[key] = None if key not in actual else actual[key] == expected[key]
        report["replay_comparisons"] = comparisons
        report["exact_reported_outcome_parity"] = (
            None if not comparisons or any(value is None for value in comparisons.values())
            else all(comparisons.values()))
        try:
            if not self.frames:
                raise ValueError("Selected episode produced no actual source captures")
            arrays = {key: np.stack([row[key] for row in self.frames]) for key in sorted(self.field_names)}
            path = self.output / "trace.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(path, allow_pickle=False) as stored:
                if set(stored.files) != set(arrays) or any(not np.array_equal(stored[key], value)
                        for key, value in arrays.items()):
                    raise AssertionError("Compact media trace changed on serialization")
            first, last = float(arrays["time_s"][0]), float(arrays["time_s"][-1])
            report.update(first_capture_s=first, last_capture_s=last, trace_roundtrip_exact=True,
                first_event_s=actual.get("event_time_s"), episode_end_s=actual.get("end_time_s"),
                delivered_C_in_trace="delivered_normalized" in arrays)
            end = actual.get("event_time_s", actual.get("end_time_s"))
            if end is not None and last > float(end)+1e-7:
                raise ValueError("Clip extends after first event; standard Task08 media ends at that event")
            report["scheduled_captures_complete_through_end"] = (
                None if end is None else bool(abs(first) <= 1e-7 and
                    last >= np.floor((float(end)+1e-7)*FPS)/FPS-1e-7))
            if self.verify_video:
                from boxing.perception.diagnostics import decode_video
                count = decode_video(self.video, deadline=time.monotonic()+120.)
                if count != len(self.frames):
                    raise ValueError("Independent video decoder frame count differs from actual captures")
                report["decoded_frames"] = count
            else:
                report["decoded_frames"] = None
            for path in (self.video, self.output / "trace.npz", self.output / "first.png"):
                if path.exists():
                    report["files"][path.name] = dict(sha256=_sha256(path), bytes=path.stat().st_size)
            report["status"] = "completed"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with (self.output / "media.json").open("x") as stream:
                json.dump(report, stream, indent=2, default=_json, allow_nan=False)
                stream.write("\n")
            self.frames.clear()
        return report
