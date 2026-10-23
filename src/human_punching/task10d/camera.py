"""The accepted Task 08 RGB-D path as independent replay evidence.

Fresh96 is a proprioceptive nominal controller. This module acquires and delays
the existing camera measurements without introducing visual features, primitive
identity, requested timing, or completion state into that controller. It does
not invoke a residual avoidance policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROBUST_SAV = Path(__file__).resolve().parents[4] / "robust_sav"
HSV_LUT = ROBUST_SAV / "outputs/boxing/diagnostics/task07b_bc_preflight_20261003/shared/hsv_rgb24.bin"
HSV_SHA256 = "861a19b98d286ffc0819b87fb6af8be19e39268b6a4ba51267db04402a8a42d1"


def add_camera_only(spec: Any) -> None:
    """Attach the unchanged Task 08 camera; H2 already owns both glove geoms.

    The scene builder must preserve the baseline G1 statistics while appending
    H2 geometry. Adding this camera introduces no bodies, mass, or contact pairs.
    """
    from boxing.perception.rectified_camera import CAMERA

    torso = spec.body(CAMERA.parent_body)
    if torso is None:
        raise ValueError(f"Missing accepted camera mount body {CAMERA.parent_body}")
    torso.add_camera(
        name=CAMERA.camera_name,
        pos=CAMERA.mount_position,
        quat=CAMERA.mujoco_quaternion_wxyz,
        fovy=CAMERA.vertical_fov_deg,
        resolution=(CAMERA.width, CAMERA.height),
    )


def scene_ids(model: Any, glove_geoms: Any) -> Any:
    """Resolve two H2 glove names/IDs or a binding's ``glove_geom_ids``.

    These are actual attached H2 glove geoms, not independent proxy duplicates.
    The accepted renderer uses their IDs only for its privileged clean stream;
    its C foreground still comes exclusively from rendered RGB classification.
    """
    from boxing.perception.rectified_camera import CAMERA
    from boxing.perception.scene import SceneIds

    glove_geoms = getattr(glove_geoms, "glove_geom_ids", glove_geoms)
    geoms = tuple(int(model.geom(value).id) if isinstance(value, str) else int(value)
                  for value in glove_geoms)
    if len(geoms) != 2 or len(set(geoms)) != 2:
        raise ValueError("The camera requires two distinct attached H2 glove geoms")
    bodies = tuple(int(model.geom_bodyid[index]) for index in geoms)
    mocaps = tuple(int(model.body_mocapid[index]) for index in bodies)
    if min(mocaps) < 0 or len(set(mocaps)) != 2:
        raise ValueError("H2 gloves must attach to the two prescribed wrist mocap bodies")
    ids = SceneIds(
        camera_id=int(model.camera(CAMERA.camera_name).id),
        torso_body_id=int(model.body(CAMERA.parent_body).id),
        fist_geom_ids=geoms,
        fist_body_ids=bodies,
        fist_mocap_ids=mocaps,
    )
    if int(model.cam_bodyid[ids.camera_id]) != ids.torso_body_id:
        raise ValueError("Task 08 camera must remain attached to the dynamic G1 torso")
    np.testing.assert_array_equal(model.cam_pos[ids.camera_id], CAMERA.mount_position)
    np.testing.assert_allclose(model.cam_quat[ids.camera_id], CAMERA.mujoco_quaternion_wxyz,
                               rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(model.cam_resolution[ids.camera_id], (672, 376))
    if float(model.cam_fovy[ids.camera_id]) != CAMERA.vertical_fov_deg:
        raise ValueError("Compiled camera field of view differs from Task 08")
    for index in geoms:
        if model.geom_contype[index] or model.geom_conaffinity[index]:
            raise ValueError("Prescribed H2 glove geometry must not generate contacts")
        if int(model.geom_group[index]) not in (0, 1, 2):
            raise ValueError("H2 glove geometry is excluded from the accepted camera groups")
        np.testing.assert_allclose(model.geom_size[index, 0], 0.08, rtol=0.0, atol=1e-12)
    return ids


def _owned(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _visibility_counts(segmentation, raw_depth, glove_centers, camera_position,
                       camera_rotation, *, h2_geom_ids, glove_geom_ids,
                       geom_object_type, optical_rays, ngeom):
    """Count actual first surfaces and H2 body occlusion of each glove sphere.

    Segmentation and native optical depth identify the rendered occluder. The
    isolated 0.08 m sphere intersection only identifies rays that would reach a
    glove absent occlusion; it never substitutes for C foreground or depth.
    These current-acquisition counts are diagnostics, without depth-range gating.
    """
    import torch

    geom_ids, types = segmentation.unbind(-1)
    valid_geom = (types == geom_object_type) & (geom_ids >= 0) & (geom_ids < ngeom)
    safe_ids = geom_ids.clamp(0, ngeom - 1).long()
    counts = torch.zeros((len(segmentation), ngeom), dtype=torch.int64,
                         device=segmentation.device)
    counts.scatter_add_(1, safe_ids.flatten(1), valid_geom.long().flatten(1))
    body_lookup = torch.zeros(ngeom, dtype=torch.bool, device=segmentation.device)
    body_lookup[h2_geom_ids] = True
    body_lookup[glove_geom_ids] = False
    body_surface = (valid_geom & body_lookup[safe_ids]
                    & torch.isfinite(raw_depth) & (raw_depth > 0))
    centers = torch.einsum("nki,nij->nkj", glove_centers - camera_position[:, None],
                           camera_rotation)
    silhouettes, occluded = [], []
    for hand in range(2):
        center = centers[:, hand]
        along = torch.einsum("hwc,nc->nhw", optical_rays, center)
        discriminant = along.square() - (center.square().sum(-1)[:, None, None] - .08 ** 2)
        root = discriminant.clamp_min(0).sqrt()
        entry, exit_ = along - root, along + root
        distance = torch.where(entry > 0, entry, exit_)
        intersects = (discriminant >= 0) & (distance > 0)
        sphere_depth = distance * optical_rays[..., 2]
        silhouettes.append(intersects.sum((-2, -1)))
        # A 0.1 mm separation excludes numerical equality at tangent/attached
        # surfaces; these counts are evidence of a nearer rendered H2 body.
        occluded.append((intersects & body_surface
                         & (raw_depth < sphere_depth - 1e-4)).sum((-2, -1)))
    return {
        "h2_geom_visible_pixels": counts[:, h2_geom_ids],
        "glove_visible_pixels": counts[:, glove_geom_ids],
        "glove_unoccluded_silhouette_pixels": torch.stack(silhouettes, -1),
        "glove_h2_body_occluded_pixels": torch.stack(occluded, -1),
    }


class Task08Camera:
    """Actual GPU RGB rendering, accepted pooling/noise, and normal visual delay.

    ``capture(time_s)`` acquires synchronized state at each 20 ms boundary. The
    caller updates every H2 mocap and G1 state before this call. The first call
    resets the acquisition; pass ``reset=True`` for each subsequent scenario.
    Returned NumPy arrays own their storage and retain the world batch axis.
    """

    def __init__(self, sim: Any, ids: Any, seed: int = 102300) -> None:
        import torch
        import warp as wp
        from boxing.perception.classification import load_hsv_lut
        from boxing.perception.image_faults import ForegroundVisualFaultProcess
        from boxing.perception.latency import VisualDelayBuffer
        from boxing.perception.rectified_camera import optical_rays
        from boxing.perception.renderer import RenderedDepthCamera

        if not sim.wp_device.is_cuda:
            raise ValueError("Task 08 camera evidence requires the allocated GPU runtime")
        self.sim = sim
        self.ids = ids if hasattr(ids, "fist_geom_ids") else scene_ids(sim.mj_model, ids)
        self.count = int(sim.wp_data.nworld)
        self.device = torch.device(str(sim.wp_device))
        self.seed = int(seed)
        self._initialized = False
        self._previous_time: float | None = None
        model = sim.mj_model
        prefixes = {model.geom(index).name.rsplit("/", 1)[0] + "/"
                    for index in self.ids.fist_geom_ids}
        if len(prefixes) != 1:
            raise ValueError("Both camera gloves must belong to one named H2 entity")
        self.h2_prefix = prefixes.pop()
        self.h2_geom_ids = np.array([index for index in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[index])).name or "").startswith(self.h2_prefix)],
            dtype=np.int64)
        if len(self.h2_geom_ids) <= 2:
            raise ValueError("Full H2 body geometry is required for camera occlusion evidence")
        self.stream = torch.cuda.ExternalStream(wp.get_stream(sim.wp_device).cuda_stream,
                                                device=self.device)
        with self._ordered():
            lut, self.lut_metadata = load_hsv_lut(
                HSV_LUT, device=self.device, expected_sha256=HSV_SHA256)
            self.renderer = RenderedDepthCamera(sim, self.ids, path="C", lut=lut)
            self.faults = ForegroundVisualFaultProcess(
                self.count, device=self.device, seed=self.seed)
            self.delay = VisualDelayBuffer(
                self.count, device=self.device, seed=self.seed + 1)
            self._mask = torch.ones(self.count, device=self.device, dtype=torch.bool)
            self._rays = torch.as_tensor(optical_rays(), dtype=torch.float32,
                                         device=self.device)
            self._h2_geom_ids = torch.as_tensor(self.h2_geom_ids, device=self.device)
            self._glove_geom_ids = torch.as_tensor(self.ids.fist_geom_ids,
                                                  device=self.device)

    @contextmanager
    def _ordered(self) -> Iterator[None]:
        """Use the simulation Warp stream for all aliased Torch/render buffers."""
        import torch
        import warp as wp

        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream), wp.ScopedDevice(self.sim.wp_device):
                yield
        finally:
            caller.wait_stream(self.stream)

    def capture(self, time_s: float, *, reset: bool = False) -> dict[str, np.ndarray]:
        """Return acquired RGB/depth and separately timestamped delayed packets.

        ``acquired_*`` is uncorrupted C (RGB-HSV) foreground. ``clean_*`` is
        simulator-ID ground truth, never used to repair C. ``corrupted_*`` is
        the newly acquired C packet after the accepted noise model, and
        ``delivered_*`` is its delayed version. Depth is optical metres;
        normalized depth is (depth-.1)/2.9 with invalid sentinel 1.
        """
        import torch
        import mujoco_warp as mjwarp
        from boxing.perception.classification import packed_rgb_to_uint8
        from boxing.perception.depth import normalize_depth

        time_s = float(time_s)
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Camera capture time must be finite and nonnegative")
        reset = bool(reset or not self._initialized)
        if not reset and not np.isclose(time_s - self._previous_time, 0.02, atol=1e-8, rtol=0.0):
            raise ValueError("Accepted camera acquisitions must be exactly 20 ms apart")
        with torch.no_grad(), self._ordered():
            # These update derived geometry/camera/light poses only, without a
            # physics integration, controller action, or additional acquisition.
            mjwarp.kinematics(self.sim.wp_model, self.sim.wp_data)
            mjwarp.camlight(self.sim.wp_model, self.sim.wp_data)
            frame = self.renderer.capture()
            timestamps = torch.full((self.count,), time_s, device=self.device, dtype=torch.float64)
            if reset:
                self.faults.reset(self._mask)
            realized = self.faults.apply(frame.depth, frame.masks, self._mask)
            clean = normalize_depth(frame.clean_depth, frame.clean_masks)
            operation = self.delay.reset if reset else self.delay.append
            operation(self._mask, realized["corrupted_normalized"], clean,
                      realized["corrupted_masks"], frame.clean_masks, timestamps)
            packet = self.delay.delivered(timestamps)
            delivered_depth = torch.where(packet["corrupted_valid"],
                                          0.1 + 2.9 * packet["corrupted"], 3.0)
            delivered_clean_depth = torch.where(packet["clean_valid"],
                                                0.1 + 2.9 * packet["clean"], 3.0)
            values = {
                "rgb": packed_rgb_to_uint8(self.renderer.packed_rgb.view(torch.int32)),
                "acquired_depth": frame.depth,
                "acquired_normalized": normalize_depth(frame.depth, frame.masks),
                "acquired_masks": frame.masks,
                "clean_depth": frame.clean_depth,
                "clean_normalized": clean,
                "clean_masks": frame.clean_masks,
                "corrupted_depth": realized["corrupted_depth"],
                "corrupted_normalized": realized["corrupted_normalized"],
                "corrupted_masks": realized["corrupted_masks"],
                "delivered_depth": delivered_depth,
                "delivered_normalized": packet["corrupted"],
                "delivered_masks": packet["corrupted_valid"],
                "delivered_clean_depth": delivered_clean_depth,
                "delivered_clean_normalized": packet["clean"],
                "delivered_clean_masks": packet["clean_valid"],
                "capture_timestamps": timestamps,
                "capture_frame_ids": self.delay.capture_counts - 1,
                "delivered_capture_timestamps": packet["capture_timestamps"],
                "delivery_timestamps": packet["delivery_timestamps"],
                "delivered_frame_ids": packet["frame_ids"],
                "delivered_source_tick": packet["frame_ids"],
                "delay_ticks": packet["delay_ticks"],
                "padding": packet["padding"],
                "visual_age_s": packet["age_s"],
                "camera_position": frame.camera_position,
                "camera_rotation": frame.camera_rotation,
                "renderer_capture_index": torch.full((self.count,), frame.capture_index,
                                                       device=self.device, dtype=torch.int64),
            }
            geom_positions = getattr(self.sim.data.geom_xpos, "_tensor", self.sim.data.geom_xpos)
            values.update(_visibility_counts(
                self.renderer.segmentation, self.renderer.raw_depth,
                geom_positions[:, self._glove_geom_ids], frame.camera_position,
                frame.camera_rotation, h2_geom_ids=self._h2_geom_ids,
                glove_geom_ids=self._glove_geom_ids,
                geom_object_type=self.ids.geom_object_type, optical_rays=self._rays,
                ngeom=self.sim.mj_model.ngeom))
            result = {name: _owned(value) for name, value in values.items()}
            result.update({"fault_" + name: _owned(value)
                           for name, value in self.faults.diagnostics().items()})
        self._initialized = True
        self._previous_time = time_s
        return result

    def metadata(self) -> dict[str, Any]:
        """Compact contract/provenance; no per-frame RGB trace is retained."""
        from boxing.perception.depth import profile as depth_profile
        from boxing.perception.image_faults import fault_profile
        from boxing.perception.latency import latency_profile
        from boxing.perception.rectified_camera import CAMERA

        model = self.sim.mj_model
        return {
            "schema_id": "task10d_task08_independent_camera_evidence_v1",
            "perception_path": "C",
            "camera": CAMERA.record(),
            "depth": depth_profile(),
            "faults": fault_profile(),
            "latency": latency_profile(),
            "visual_fault_seed": self.seed,
            "delay_seed": self.seed + 1,
            "hsv_lut_path": str(HSV_LUT),
            "hsv_lut_sha256": HSV_SHA256,
            "glove_geom_names": [model.geom(index).name for index in self.ids.fist_geom_ids],
            "glove_geom_ids": list(self.ids.fist_geom_ids),
            "glove_mocap_ids": list(self.ids.fist_mocap_ids),
            "enabled_geom_groups": [0, 1, 2],
            "body_occlusion": "all enabled full-body geometry; contact flags do not hide occluders",
            "h2_geom_ids": self.h2_geom_ids.tolist(),
            "h2_geom_names": [model.geom(int(index)).name for index in self.h2_geom_ids],
            "occlusion_counts": {
                "timestamp": "current acquisition; no delayed or actor features",
                "visible": "actual first-surface typed segmentation counts by H2 geom and glove",
                "silhouette": "positive optical ray intersections with isolated 0.08 m glove sphere",
                "h2_body_occluded": "actual H2 non-glove first surface at least 0.1 mm nearer than glove sphere",
                "depth_range_gated": False,
            },
            "controller": "frozen Fresh96 proprioception only; no residual avoidance policy invoked",
            "actor_camera_inputs": False,
            "appearance_audit": self.renderer.appearance_audit,
            "capture_count": self.renderer.capture_count,
        }
