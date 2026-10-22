"""Named, fractional stored references with explicit source seconds and terminal hold."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .robot import BODY_NAMES, GLOVE_SITE_NAMES, H2_XML, JOINT_NAMES, joint_addresses


def quaternion_interpolate(time_s, quaternions, query):
    q = np.asarray(quaternions)
    # Uniform actor grids can overshoot an exact endpoint by one floating ulp.
    query = np.clip(np.atleast_1d(query), time_s[0], time_s[-1])
    flat = q.reshape(len(time_s), -1, 4)
    result = np.stack([Slerp(time_s, Rotation.from_quat(flat[:, i], scalar_first=True))(query)
                       .as_quat(scalar_first=True) for i in range(flat.shape[1])], axis=1)
    return result.reshape((len(query),) + q.shape[1:])


@dataclass
class StoredReference:
    arrays: dict[str, np.ndarray]
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> "StoredReference":
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        metadata = json.loads(path.with_suffix(".json").read_text())
        names = tuple(map(str, arrays["joint_names"]))
        if len(names) != 31 or set(names) != set(JOINT_NAMES):
            raise ValueError("Reference joint map differs from selected 31-joint H2")
        order = [names.index(name) for name in JOINT_NAMES]
        arrays["joint_pos"] = arrays["joint_pos"][:, order]
        arrays["joint_vel"] = arrays["joint_vel"][:, order]
        arrays["joint_names"] = np.array(JOINT_NAMES)
        t = arrays["time_s"]
        if len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("Reference needs strictly increasing explicit seconds from zero")
        for key, value in arrays.items():
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite reference {key}")
        if arrays["qpos"].shape != (len(t), 38) or arrays["qvel"].shape != (len(t), 37):
            raise ValueError("Reference does not use selected 31-joint model dimensions")
        return cls(arrays, metadata)

    @property
    def time_s(self):
        return self.arrays["time_s"]

    def sample(self, source_time_s, rate=1.0) -> dict[str, np.ndarray]:
        """Fractional interpolation; all velocities scaled by ds/dt and zero in hold."""
        scalar = np.ndim(source_time_s) == 0
        raw = np.atleast_1d(source_time_s)
        query = np.clip(raw, 0, self.time_s[-1])
        vel_rate = np.broadcast_to(rate, raw.shape) * (raw < self.time_s[-1])
        result = {}
        for key, value in self.arrays.items():
            if value.ndim == 0 or len(value) != len(self.time_s) or value.dtype.kind not in "fci":
                continue
            if "quat" in key:
                out = quaternion_interpolate(self.time_s, value, query)
            else:
                flat = value.reshape(len(self.time_s), -1)
                out = np.stack([np.interp(query, self.time_s, flat[:, j]) for j in range(flat.shape[1])], axis=-1)
                out = out.reshape((len(query),) + value.shape[1:])
                if key == "qpos":
                    out[:, 3:7] = quaternion_interpolate(self.time_s, value[:, 3:7], query)
            if "vel" in key:
                out *= vel_rate.reshape((len(query),) + (1,) * (out.ndim - 1))
            result[key] = out[0] if scalar else out
        return result


def arrays_from_qpos(time_s: np.ndarray, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Compute training/reset kinematics on the actual selected model by name."""
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    data = mujoco.MjData(model)
    qa, va = joint_addresses(model)
    n = len(time_s)
    qvel = np.empty((n, model.nv))
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        mujoco.mj_differentiatePos(model, qvel[i], time_s[hi] - time_s[lo], qpos[lo], qpos[hi])
    body_ids = [model.body(name).id for name in BODY_NAMES]
    site_ids = [model.site(name).id for name in GLOVE_SITE_NAMES]
    result = {"time_s": time_s, "joint_names": np.array(JOINT_NAMES), "body_names": np.array(BODY_NAMES),
              "glove_site_names": np.array(GLOVE_SITE_NAMES), "qpos": qpos, "qvel": qvel,
              "joint_pos": qpos[:, qa], "joint_vel": qvel[:, va],
              "root_pos_w": qpos[:, :3], "root_quat_w": qpos[:, 3:7]}
    for prefix, count in (("body", len(body_ids)), ("glove", 2)):
        for field, dim in (("pos", 3), ("quat", 4), ("lin_vel", 3), ("ang_vel", 3)):
            result[f"{prefix}_{field}_w"] = np.empty((n, count, dim))
    v = np.empty(6)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        for j, body in enumerate(body_ids):
            result["body_pos_w"][i, j] = data.xpos[body]
            result["body_quat_w"][i, j] = data.xquat[body]
            # XBODY is the regular body origin matching xpos. BODY reports
            # the inertial COM frame and gives a different linear velocity.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, v, 0)
            result["body_ang_vel_w"][i, j], result["body_lin_vel_w"][i, j] = v[:3], v[3:]
        for j, site in enumerate(site_ids):
            result["glove_pos_w"][i, j] = data.site_xpos[site]
            result["glove_quat_w"][i, j] = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat(scalar_first=True)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, v, 0)
            result["glove_ang_vel_w"][i, j], result["glove_lin_vel_w"][i, j] = v[:3], v[3:]
    # Useful established TaskNPoint six-array aliases are the same named arrays.
    result["fps"] = np.array(50.)
    return result
