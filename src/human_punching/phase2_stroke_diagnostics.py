"""Stored-state family-corridor and arm/trunk evidence, without a classifier.

All reference poses are the exact recorded pre-integration qref/vref. Their
clock has one sample per retained native interval; the final physical endpoint
has no invented next drive/reference. FK does not integrate or call an actor.
No diagnostic here assigns recognizable-stroke or movement-exploit truth.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

VERSION = "task10b_phase2_stored_stroke_diagnostics_v1"
MAX_SAMPLES = 6205
MAX_WORKSPACE_BYTES = 256 * 1024**2
ARM_PARTS = ("Shoulder", "Arm", "ForeArm", "Hand")
TRUNK_PARTS = ("Spine1", "Spine2", "Chest")


def sequence_joints(model):
    """Return the actual accepted named arm/trunk hinges in model order."""
    groups = {"left_arm": [], "right_arm": [], "trunk": []}
    for joint in range(model.njnt):
        name = model.joint(joint).name or ""
        stem = name.rsplit("_", 1)[0]
        group = ("left_arm" if stem in tuple("Left" + p for p in ARM_PARTS) else
                 "right_arm" if stem in tuple("Right" + p for p in ARM_PARTS) else
                 "trunk" if stem in TRUNK_PARTS else None)
        if group is not None:
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Named arm/trunk sequence coordinates must be scalar hinges")
            groups[group].append(joint)
    if tuple(map(len, groups.values())) != (11, 11, 9):
        raise ValueError("Accepted named 11+11 arm and nine trunk hinges required")
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def _fk(model, poses, velocities, checkpoint):
    """Pure saved-state FK and Jacobian velocity; no forward dynamics/contact."""
    hands = [model.site(side + "Hand_center").id for side in ("Left", "Right")]
    chest = model.body("Chest").id
    data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    count = len(poses)
    out = {"glove_world_m": np.empty((count, 2, 3)),
           "glove_world_velocity_m_s": np.empty((count, 2, 3)),
           "glove_chest_m": np.empty((count, 2, 3)),
           "glove_chest_velocity_m_s": np.empty((count, 2, 3)),
           "chest_world_m": np.empty((count, 3)),
           "chest_world_rotation": np.empty((count, 3, 3)),
           "chest_world_velocity_m_s": np.empty((count, 3)),
           "chest_world_angular_velocity_rad_s": np.empty((count, 3))}
    for index, (pose, velocity) in enumerate(zip(poses, velocities, strict=True)):
        if index % 32 == 0:
            checkpoint("stored_stroke_fk", progress={"sample": index, "samples": count})
        data.qpos[:] = pose
        data.qvel[:] = velocity
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rotation = data.xmat[chest].reshape(3, 3)
        position = data.xpos[chest]
        mujoco.mj_jacBody(model, data, jp, jr, chest)
        linear, angular = jp @ velocity, jr @ velocity
        out["chest_world_m"][index] = position
        out["chest_world_rotation"][index] = rotation
        out["chest_world_velocity_m_s"][index] = linear
        out["chest_world_angular_velocity_rad_s"][index] = angular
        for hand, site in enumerate(hands):
            point = data.site_xpos[site]
            mujoco.mj_jacSite(model, data, jp, jr, site)
            speed = jp @ velocity
            offset = point - position
            out["glove_world_m"][index, hand] = point
            out["glove_world_velocity_m_s"][index, hand] = speed
            out["glove_chest_m"][index, hand] = rotation.T @ offset
            out["glove_chest_velocity_m_s"][index, hand] = rotation.T @ (
                speed - linear - np.cross(angular, offset))
    return out


def diagnostic_arrays(model, qpos, qvel, reference_qpos, reference_qvel, *,
                      reference_ticks, drive_reference_ticks, target_w, active_hand,
                      checkpoint=lambda *_args, **_kwargs: None):
    """Keep every retained physical knot and every actually paired drive knot.

    ``reference_qpos/qvel`` are taken from raw pre_qref/pre_vref, never sampled
    from a nearby cached path. An unmatched pre-drive after a failed integration
    is excluded and reported by the caller; it is not a physical interval.
    """
    q, v = np.asarray(qpos), np.asarray(qvel)
    rq, rv = np.asarray(reference_qpos), np.asarray(reference_qvel)
    ticks, drive_ticks = np.asarray(reference_ticks), np.asarray(drive_reference_ticks)
    target = np.asarray(target_w, float)
    n, m = len(q), len(rq)
    if (model.nq != 55 or model.nv != 54 or not 1 <= n <= MAX_SAMPLES
            or q.shape != (n, 55) or v.shape != (n, 54)
            or rq.shape != (m, 55) or rv.shape != (m, 54)
            or not 0 <= m <= n-1 or ticks.shape != (n,) or drive_ticks.shape != (m,)
            or ticks.dtype.kind not in "iu" or drive_ticks.dtype.kind not in "iu"
            or np.any(ticks < 0) or not np.array_equal(ticks[:m], drive_ticks)
            or target.shape != (3,) or active_hand not in (0, 1)
            or not all(np.isfinite(a).all() for a in (q, v, rq, rv, target))):
        raise ValueError("Finite accepted-size stored native states and paired exact drive clocks required")
    for poses in (q, rq):
        if np.any(abs(np.linalg.norm(poses[:, 3:7], axis=1)-1.) > 1.e-5):
            raise ValueError("Invalid root quaternion in stored native/drive evidence")
    groups = sequence_joints(model)
    joint_ids = np.concatenate(tuple(groups.values()))
    qa, va = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    # Conservative finite array allowance, independent of compressed file size.
    workspace = sum(a.nbytes for a in (q, v, rq, rv, ticks, drive_ticks)) + (n+m)*4096
    if workspace > MAX_WORKSPACE_BYTES:
        raise ValueError("Stored-state diagnostic workspace exceeds256MiB")
    arrays = dict(time_s=np.arange(n)*.005, drive_time_s=np.arange(m)*.005,
        qpos=q.copy(), qvel=v.copy(), drive_qpos=rq.copy(), drive_qvel=rv.copy(),
        reference_tick=ticks.copy(), drive_reference_tick=drive_ticks.copy(),
        target_w=target.copy(), joint_qpos_addresses=qa.copy(), joint_dof_addresses=va.copy(),
        joint_names=np.asarray([model.joint(int(j)).name for j in joint_ids], dtype="U64"),
        joint_group=np.asarray([key for key, ids in groups.items() for _ in ids], dtype="U16"),
        joint_position_rad=q[:, qa].copy(), joint_velocity_rad_s=v[:, va].copy(),
        drive_joint_position_rad=rq[:, qa].copy(), drive_joint_velocity_rad_s=rv[:, va].copy())
    actual = _fk(model, q, v, checkpoint)
    desired = _fk(model, rq, rv, checkpoint)
    arrays.update(actual)
    arrays.update({"drive_" + key: value for key, value in desired.items()})
    arrays["glove_target_distance_m"] = np.linalg.norm(actual["glove_world_m"]-target, axis=-1)
    arrays["drive_glove_target_distance_m"] = np.linalg.norm(desired["glove_world_m"]-target, axis=-1)
    arrays["paired_glove_position_error_m"] = np.linalg.norm(
        actual["glove_world_m"][:m]-desired["glove_world_m"], axis=-1)
    metadata = dict(schema=VERSION, active_hand_index=active_hand,
        retained_physical_samples=n, paired_preintegration_reference_samples=m,
        physical_reference_pairing="Drive at physical interval start k; reference_tick equals recorded pre_reference_tick; no final endpoint drive inferred",
        physical_sample_interval_s=.005, missing_paired_drive_intervals=n-1-m,
        declared_workspace_allowance_bytes=MAX_WORKSPACE_BYTES, estimated_owned_array_workspace_bytes=workspace,
        stored_output_array_bytes=sum(a.nbytes for a in arrays.values()),
        joint_groups={key: [model.joint(int(j)).name for j in ids] for key, ids in groups.items()},
        corridor_semantics="Exact executed family reference and recorded physical glove paths in world and moving Chest coordinates",
        sequence_semantics="Complete named arm/trunk q and qvel, Chest rotation and Jacobian velocity; no learned or threshold classifier",
        single_outward_stroke_pass=None, movement_exploit=None,
        human_trace_review_required=True, physics_integrations=0, actor_calls=0)
    return arrays, metadata


def write_review_plots(arrays, folder, *, title, events, checkpoint=lambda *_args, **_kwargs: None):
    """Plot all retained samples; diagnostic values carry no automatic verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    t, rt = arrays["time_s"], arrays["drive_time_s"]
    paths = []
    def finish(fig, name):
        checkpoint("stored_stroke_plot", progress={"plot": name})
        fig.suptitle(title + "\nStored-state evidence; explicit reviewer judgment required", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .94))
        path = folder/name
        if path.exists():raise FileExistsError(path)
        fig.savefig(path, dpi=125)
        plt.close(fig)
        paths.append(path)
    def event_lines(ax):
        for name, value in events.items():
            if value is not None and np.isfinite(value):
                ax.axvline(value, lw=.8, alpha=.55, label=name)
        ax.set_xlabel("episode time (s)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, frame, coord in zip(axes.flat, ("world", "world", "chest", "chest"),
                                ((0, 1), (0, 2), (0, 1), (0, 2)), strict=True):
        for hand, color in enumerate(("tab:blue", "tab:orange")):
            actual, desired = arrays[f"glove_{frame}_m"][:, hand], arrays[f"drive_glove_{frame}_m"][:, hand]
            ax.plot(actual[:, coord[0]], actual[:, coord[1]], color=color, label=("left", "right")[hand]+" native")
            ax.plot(desired[:, coord[0]], desired[:, coord[1]], color=color, ls="--", label=("left", "right")[hand]+" drive")
        if frame == "world":
            target = arrays["target_w"]
            ax.scatter(target[coord[0]], target[coord[1]], c="black", marker="x", label="frozen head center")
        ax.set_xlabel(f'{frame} {"xyz"[coord[0]]} (m)');ax.set_ylabel(f'{frame} {"xyz"[coord[1]]} (m)')
        ax.axis("equal");ax.grid(alpha=.2);ax.legend(fontsize=7)
    finish(fig, "glove-corridor.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for hand, color in enumerate(("tab:blue", "tab:orange")):
        label = ("left", "right")[hand]
        axes[0].plot(t, arrays["glove_target_distance_m"][:, hand], color=color, label=label+" native")
        axes[0].plot(rt, arrays["drive_glove_target_distance_m"][:, hand], color=color, ls="--", label=label+" drive")
        axes[1].plot(rt, arrays["paired_glove_position_error_m"][:, hand], color=color, label=label)
        axes[2].plot(t, np.linalg.norm(arrays["glove_chest_velocity_m_s"][:, hand], axis=-1), color=color, label=label+" native")
        axes[2].plot(rt, np.linalg.norm(arrays["drive_glove_chest_velocity_m_s"][:, hand], axis=-1), color=color, ls="--", label=label+" drive")
    axes[0].axhline(.14, color="black", lw=.8, ls=":", label="declared contact radius")
    for ax, ylabel in zip(axes, ("head-center distance (m)", "paired drive position error (m)", "Chest-relative glove speed (m/s)"), strict=True):
        ax.set_ylabel(ylabel);event_lines(ax);ax.legend(fontsize=7);ax.grid(alpha=.2)
    finish(fig, "glove-clock.png")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for row, group in enumerate(("left_arm", "right_arm", "trunk")):
        selected = np.flatnonzero(arrays["joint_group"] == group)
        for col, field, units in ((0, "position_rad", "rad"), (1, "velocity_rad_s", "rad/s")):
            ax = axes[row, col]
            for index in selected:
                color = f"C{int(index-selected[0])%10}"
                ax.plot(t, arrays["joint_"+field][:, index], color=color, lw=.85, label=arrays["joint_names"][index])
                ax.plot(rt, arrays["drive_joint_"+field][:, index], color=color, ls="--", lw=.6)
            ax.set_ylabel(group+" ("+units+")");event_lines(ax);ax.grid(alpha=.2);ax.legend(fontsize=5, ncol=3)
    finish(fig, "arm-trunk-sequencing.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for axis, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        for ax, field in zip(axes, ("chest_world_velocity_m_s", "chest_world_angular_velocity_rad_s"), strict=True):
            ax.plot(t, arrays[field][:, axis], color=color, label="xyz"[axis]+" native")
            ax.plot(rt, arrays["drive_"+field][:, axis], color=color, ls="--", label="xyz"[axis]+" drive")
    for ax, label in zip(axes, ("Chest world linear velocity (m/s)", "Chest world angular velocity (rad/s)"), strict=True):
        ax.set_ylabel(label);event_lines(ax);ax.legend(fontsize=7);ax.grid(alpha=.2)
    finish(fig, "chest-sequencing.png")
    return paths
