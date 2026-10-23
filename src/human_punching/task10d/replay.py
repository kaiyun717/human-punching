"""Collision-free whole-body H2 replay in the pinned mjlab 1.6 runtime.

Each native link is represented by a world-child mocap body. Its mesh retains
the original link-local transform; both gloves retain the native wrist-local
attachments. The displayed body poses always come from interpolated native
root/joint coordinates followed by FK, never independent link interpolation.
This module does not write generalized positions, velocities, or controls.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation, Slerp

from human_punching.task10c.references import StoredReference
from human_punching.task10c.robot import (
    BODY_NAMES, GLOVE_SITE_NAMES, H2_XML, ROOT, assert_selected_model,
)

APPROVED_JAB = ROOT / "outputs/task10c/jab/nominal_reference_adaptation_20261022_01/final/candidate/jab.npz"
APPROVED_JAB_SHA256 = "50887274783da7364d8b7ce39ee7f956c583174b0339ca602d65bfd71ff600e8"
PHYSICS_DT = 0.005
GLOVE_RADIUS = 0.08
HEAD_RADIUS = 0.06
ENCOUNTER_RADIUS = GLOVE_RADIUS + HEAD_RADIUS
SAFE_CENTER_DISTANCE = 0.16
GLOVE_GEOM_NAMES = ("left_glove_collision", "right_glove_collision")


@dataclass(frozen=True)
class Placement:
    """Episode-fixed planar registration; native floor height is preserved."""

    xy: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if len(self.xy) != 2 or not np.isfinite((*self.xy, self.yaw)).all():
            raise ValueError("Placement requires finite XY translation and yaw")
        object.__setattr__(self, "xy", tuple(float(v) for v in self.xy))
        object.__setattr__(self, "yaw", float(self.yaw))

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_rotvec([0.0, 0.0, self.yaw])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return self.rotation.apply(points) + np.array([*self.xy, 0.0])


@dataclass(frozen=True)
class PoseSample:
    """World body/glove poses plus the unregistered native source qpos."""

    source_time_s: float
    body_pos: np.ndarray
    body_quat: np.ndarray
    glove_pos: np.ndarray
    qpos: np.ndarray


@dataclass(frozen=True)
class EntryMeasurement:
    """FK estimate and its enclosing 5 ms source-grid sample, kept distinct."""

    interpolated_s: float | None
    sampled_s: float | None
    previous_sample_s: float | None
    hand: str | None
    minimum_sampled_distance_m: float
    launch_overlap: bool
    grid_dt_s: float = PHYSICS_DT


@dataclass(frozen=True)
class BoundH2:
    """Compiled names resolve ownership; writing never assumes mocap ordering."""

    body_ids: np.ndarray
    mocap_ids: np.ndarray
    glove_geom_ids: np.ndarray
    prefix: str

    def write(self, sample: PoseSample, sim_data: Any, *, env_ids=None,
              origins=None) -> None:
        """Write only owned mocap poses to mjlab TensorData or native MjData.

        Call the simulator's forward/update path after this method before
        reading geometry or rendering. For a physics step, the caller writes
        the pose at that step's endpoint and reads both robots after forward.
        ``origins`` is an optional (N,3) world shift for vectorized environments.
        """
        if isinstance(sim_data.mocap_pos, np.ndarray):
            if env_ids is not None or origins is not None:
                raise ValueError("Native MjData has one world and no env_ids/origins")
            sim_data.mocap_pos[self.mocap_ids] = sample.body_pos
            sim_data.mocap_quat[self.mocap_ids] = sample.body_quat
            return
        import torch

        pos_buffer, quat_buffer = sim_data.mocap_pos, sim_data.mocap_quat
        device = pos_buffer.device
        ids = torch.as_tensor(self.mocap_ids, dtype=torch.long, device=device)
        if env_ids is None:
            selected = torch.arange(pos_buffer.shape[0], device=device)
        elif isinstance(env_ids, slice):
            selected = torch.arange(pos_buffer.shape[0], device=device)[env_ids]
        else:
            selected = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
        pos = torch.as_tensor(sample.body_pos, dtype=pos_buffer.dtype, device=device)
        quat = torch.as_tensor(sample.body_quat, dtype=quat_buffer.dtype, device=device)
        pos = pos.unsqueeze(0).expand(len(selected), -1, -1)
        if origins is not None:
            shifts = torch.as_tensor(origins, dtype=pos.dtype, device=device)
            if shifts.ndim == 1:
                shifts = shifts[None, :]
            if shifts.shape != (len(selected), 3):
                raise ValueError("origins must contain one XYZ shift per selected env")
            pos = pos + shifts[:, None, :]
        pos_buffer[selected[:, None], ids[None, :]] = pos
        quat_buffer[selected[:, None], ids[None, :]] = quat


class PrescribedH2:
    """Read-only reference and native CPU FK, without a dynamic H2 model in scene."""

    def __init__(self, reference_path: str | Path = APPROVED_JAB):
        self.path = Path(reference_path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.path == APPROVED_JAB.resolve() and self.sha256 != APPROVED_JAB_SHA256:
            raise ValueError("Approved jab payload identity changed")
        self.reference = StoredReference.load(self.path)
        # Replay must never deform its selected source after launch. Keep this
        # protection local: historical Task10C adaptation still owns writable
        # references when it explicitly loads them through StoredReference.
        for array in self.reference.arrays.values():
            array.setflags(write=False)
        self.model = mujoco.MjModel.from_xml_path(str(H2_XML))
        assert_selected_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.array([self.model.body(name).id for name in BODY_NAMES])
        self.site_ids = np.array([self.model.site(name).id for name in GLOVE_SITE_NAMES])
        self.body_names = BODY_NAMES
        self.times = np.asarray(self.reference.time_s, dtype=np.float64)
        self.qposes = np.asarray(self.reference.arrays["qpos"], dtype=np.float64)
        self.times.setflags(write=False)
        self.qposes.setflags(write=False)
        self._root_slerp = Slerp(self.times, Rotation.from_quat(
            self.qposes[:, 3:7], scalar_first=True))
        self.end_time_s = float(self.times[-1])
        self.primitive_id = f"{self.reference.metadata['family']}:{self.sha256[:12]}"

    def sample(self, source_time_s: float, placement: Placement = Placement()) -> PoseSample:
        """Linear scalar coordinates / root SLERP, then exact native link FK."""
        if not np.isfinite(source_time_s):
            raise ValueError("Source time must be finite")
        source_time_s = float(np.clip(source_time_s, self.times[0], self.times[-1]))
        hi = int(np.clip(np.searchsorted(self.times, source_time_s, side="right"),
                         1, len(self.times) - 1))
        lo = hi - 1
        weight = (source_time_s - self.times[lo]) / (self.times[hi] - self.times[lo])
        qpos = self.qposes[lo] * (1.0 - weight) + self.qposes[hi] * weight
        qpos[3:7] = self._root_slerp([source_time_s]).as_quat(scalar_first=True)[0]
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        body_pos = placement.transform_points(self.data.xpos[self.body_ids])
        body_quat = (placement.rotation * Rotation.from_quat(
            self.data.xquat[self.body_ids], scalar_first=True)).as_quat(scalar_first=True)
        glove_pos = placement.transform_points(self.data.site_xpos[self.site_ids])
        return PoseSample(source_time_s, body_pos, body_quat, glove_pos, qpos)

    def entity_cfg(self, placement: Placement = Placement()):
        """One mjlab entity whose 32 root-level mocap links have no DOFs."""
        from mjlab.entity import EntityCfg

        initial = self.sample(0.0, placement)
        return EntityCfg(
            spec_fn=lambda: self.spec(placement),
            init_state=EntityCfg.InitialStateCfg(
                pos=tuple(initial.body_pos[0]), rot=tuple(initial.body_quat[0])),
        )

    def spec(self, placement: Placement = Placement()) -> mujoco.MjSpec:
        """Preserve native visual meshes and gloves; omit physical proxy geoms.

        Both red gloves use group 0, native visible meshes group 2. Group 3
        physical proxy capsules/boxes are not visible in the original visual
        model and are not needed in this prescribed representation.
        """
        source = ET.parse(H2_XML).getroot()
        root = ET.Element("mujoco", model="task10d_prescribed_h2")
        compiler = deepcopy(source.find("compiler"))
        compiler.set("meshdir", str((H2_XML.parent / "meshes").resolve()))
        compiler.set("fusestatic", "false")
        root.append(compiler)
        for tag in ("default", "asset"):
            element = source.find(tag)
            if element is not None:
                root.append(deepcopy(element))
        world = ET.SubElement(root, "worldbody")
        initial = self.sample(0.0, placement)
        source_bodies = {body.attrib["name"]: body for body in source.iter("body")}
        for i, name in enumerate(self.body_names):
            body = ET.SubElement(world, "body", name=name, mocap="true",
                                 pos=" ".join(map(str, initial.body_pos[i])),
                                 quat=" ".join(map(str, initial.body_quat[i])))
            for j, original in enumerate(source_bodies[name].findall("geom")):
                is_glove = original.get("name") in GLOVE_GEOM_NAMES
                if original.get("class") != "visual" and not is_glove:
                    continue
                geom = deepcopy(original)
                geom.set("name", original.get("name", f"{name}_visual_{j}"))
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("density", "0")
                if is_glove:
                    geom.set("rgba", "0.8 0.12 0.1 1")
                    geom.set("group", "0")
                body.append(geom)
            for original in source_bodies[name].findall("site"):
                if original.get("name") in GLOVE_SITE_NAMES:
                    site = deepcopy(original)
                    site.set("rgba", "0 0 0 0")
                    body.append(site)
        return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))

    def bind(self, model: mujoco.MjModel, prefix: str = "h2/") -> BoundH2:
        """Verify every H2 link is mocap-only and every owned geom noncontact."""
        body_ids = np.array([model.body(prefix + name).id for name in self.body_names])
        mocap_ids = model.body_mocapid[body_ids].copy()
        owned_geoms = np.isin(model.geom_bodyid, body_ids)
        if np.any(mocap_ids < 0) or len(np.unique(mocap_ids)) != len(body_ids):
            raise ValueError("Every H2 link must own a distinct mocap body")
        if np.any(model.body_jntnum[body_ids]) or np.any(model.body_dofnum[body_ids]):
            raise ValueError("Prescribed H2 must not add dynamic degrees of freedom")
        if np.any(model.geom_contype[owned_geoms]) or np.any(model.geom_conaffinity[owned_geoms]):
            raise ValueError("H2 geoms must not participate in contact")
        glove_ids = np.array([model.geom(prefix + name).id for name in GLOVE_GEOM_NAMES])
        if not np.allclose(model.geom_size[glove_ids, 0], GLOVE_RADIUS, atol=1e-12, rtol=0):
            raise ValueError("Both gloves must retain the 0.08 m radius")
        if not np.allclose(model.geom_rgba[glove_ids], [0.8, 0.12, 0.1, 1.0]):
            raise ValueError("Both gloves must use the red observation convention")
        return BoundH2(body_ids, mocap_ids, glove_ids, prefix)

    def first_entry(self, frozen_head, placement: Placement = Placement()) -> EntryMeasurement:
        """Small single-target FK scan; no sweep, re-aiming, or moving-head oracle."""
        head = np.asarray(frozen_head, dtype=float)
        if head.shape != (3,) or not np.isfinite(head).all():
            raise ValueError("Frozen head must be a finite world XYZ point")
        count = int(np.ceil(self.end_time_s / PHYSICS_DT))
        grid = np.minimum(np.arange(count + 1) * PHYSICS_DT, self.end_time_s)
        distances = np.array([np.linalg.norm(self.sample(t, placement).glove_pos - head,
                                            axis=1) for t in grid])
        overlaps = np.any(distances <= ENCOUNTER_RADIUS, axis=1)
        minimum = float(distances.min())
        indices = np.flatnonzero(overlaps)
        if len(indices) == 0:
            return EntryMeasurement(None, None, None, None, minimum, False)
        i = int(indices[0])
        if i == 0:
            hand = ("left", "right")[int(distances[0].argmin())]
            return EntryMeasurement(0.0, 0.0, None, hand, minimum, True)
        candidates = []
        for hand in np.flatnonzero(distances[i] <= ENCOUNTER_RADIUS):
            def signed_distance(t):
                return float(np.linalg.norm(self.sample(t, placement).glove_pos[hand] - head)
                             - ENCOUNTER_RADIUS)
            crossing = brentq(signed_distance, grid[i - 1], grid[i], xtol=1e-12)
            candidates.append((float(crossing), int(hand)))
        crossing, hand = min(candidates)
        return EntryMeasurement(crossing, float(grid[i]), float(grid[i - 1]),
                                ("left", "right")[hand], minimum, False)


@dataclass(frozen=True)
class CommittedReplay:
    """Immutable placement, target and constant-rate clock fixed at launch.

    A global linear clock is C1 through the encounter and preserves all source
    timing proportions. Native endpoint guard holds make the terminal switch
    stationary. Only this representative basic clock is validated in Phase 1.
    """

    reference: PrescribedH2
    frozen_head: tuple[float, float, float]
    placement: Placement = Placement()
    requested_ttc: float | None = None
    entry: EntryMeasurement = field(init=False)
    rate: float = field(init=False)

    def __post_init__(self) -> None:
        head = tuple(float(v) for v in self.frozen_head)
        object.__setattr__(self, "frozen_head", head)
        entry = self.reference.first_entry(head, self.placement)
        object.__setattr__(self, "entry", entry)
        rate = 1.0
        if self.requested_ttc is not None:
            if not np.isfinite(self.requested_ttc) or self.requested_ttc <= 0:
                raise ValueError("Requested TTC must be positive and finite")
            if entry.interpolated_s is None or entry.interpolated_s <= 0:
                raise ValueError("Requested TTC requires a future entry into the frozen target")
            rate = entry.interpolated_s / self.requested_ttc
        object.__setattr__(self, "rate", rate)

    @property
    def duration_s(self) -> float:
        return self.reference.end_time_s / self.rate

    def source_time(self, elapsed_s: float) -> float:
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        return min(float(elapsed_s) * self.rate, self.reference.end_time_s)

    def sample(self, elapsed_s: float) -> PoseSample:
        return self.reference.sample(self.source_time(elapsed_s), self.placement)

    def complete(self, elapsed_s: float) -> bool:
        return elapsed_s >= self.duration_s
