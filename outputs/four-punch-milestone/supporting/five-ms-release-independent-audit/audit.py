"""Read-only release spot audit of model identity, clocks and actual force law.

Full 64-world acceptance belongs to the independent evaluator. This checks
nominal and last bank worlds independently and reconstructs the zero-residual
drive from the reference without importing the training implementation.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAIRS = {
    "jab": (5931806, 5931807),
    "cross": (5931915, 5931916),
    "left_hook": (5931934, 5931935),
    "right_hook": (5931918, 5931919),
}
BODY_FIELDS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "jnt_range", "jnt_pos", "jnt_axis", "jnt_type", "jnt_limited", "jnt_stiffness",
    "jnt_solref", "jnt_solimp", "jnt_margin", "dof_armature", "dof_damping", "dof_frictionloss",
    "geom_pos", "geom_quat", "geom_size", "geom_type", "geom_friction", "geom_solref", "geom_solimp",
    "geom_contype", "geom_conaffinity", "geom_condim", "geom_margin", "geom_gap",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_feedforward(model, reference, va):
    data = mujoco.MjData(model)
    sites = [i for i in range(model.nsite) if "sole" in (model.site(i).name or "").lower()]
    torque = []
    for pose, velocity in zip(reference["qpos"], reference["qvel"], strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        rows = []
        for site in sites:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site)
            rows.append(jac)
        jac = np.concatenate(rows)
        force = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=1e-5)[0].reshape(-1, 3)
        force[:, 2] = np.maximum(force[:, 2], 0)
        force[:, :2] = np.clip(force[:, :2], -.8*force[:, 2:3], .8*force[:, 2:3])
        torque.append((data.qfrc_bias-jac.T@force.ravel())[va])
    return np.clip(torque, model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])


def interpolate(times, values, query):
    # Emulate the trainer's committed float32 tensors and Torch linear formula.
    times, values, query = [np.asarray(array, dtype=np.float32) for array in (times, values, query)]
    query = np.clip(query, times[0], times[-1])
    upper = np.clip(np.searchsorted(times, query, side="right"), 1, len(times)-1)
    lower = upper-1
    fraction = (query-times[lower])/(times[upper]-times[lower])
    return values[lower]+fraction[:, None]*(values[upper]-values[lower])


results = []
for family, pair in PAIRS.items():
    learned_dir, baseline_dir = [PROJECT / f"outputs/punch-5ms-{job}" for job in pair]
    source = mujoco.MjModel.from_xml_path(str(learned_dir / "model/human.xml"))
    model = mujoco.MjModel.from_xml_path(str(learned_dir / "model/human_runtime.xml"))
    schema = json.loads((learned_dir / "controller_schema.json").read_text())
    baseline_schema = json.loads((baseline_dir / "controller_schema.json").read_text())
    assert schema == baseline_schema
    changed = [field for field in BODY_FIELDS if not np.array_equal(getattr(source, field), getattr(model, field))]
    assert not changed, changed
    assert model.opt.timestep == .005 and schema["actor_dt"] == .02
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_forcelimited) and not np.any(model.actuator_ctrllimited)
    assert np.array_equal(model.actuator_forcerange, source.actuator_ctrlrange)
    assert np.allclose(model.actuator_gear, [1, 0, 0, 0, 0, 0])
    ids = model.actuator_trnid[:, 0]
    qa, va = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    kp, kd = -model.actuator_biasprm[:, 1], -model.actuator_biasprm[:, 2]
    assert np.array_equal(kp, schema["kp"]) and np.array_equal(kd, schema["kd"])
    assert np.min(qa) >= 7 and np.min(va) >= 6
    reference = np.load(learned_dir / "reference.npz")
    ff = reference_feedforward(model, reference, va)
    entry = dict(family=family, learned_job=pair[0], baseline_job=pair[1],
                 compared_body_fields=BODY_FIELDS, changed_body_fields=changed,
                 total_mass_kg=float(model.body_mass.sum()), paired_schemas_identical=True,
                 native_force_flags_verified=True, original_strength_caps_preserved=True,
                 runtime_model_sha256=digest(learned_dir / "model/human_runtime.xml"), worlds=[])
    for run_dir, kind in ((learned_dir, "learned_policy"), (baseline_dir, "reference_pd")):
        assert digest(run_dir / "model/human_runtime.xml") == entry["runtime_model_sha256"]
        for world in (0, 63):
            path = run_dir / f"rollout_{world:02d}.npz"
            r = np.load(path)
            clock = r["simulation_time_s"]
            assert str(r["evaluation_kind"]) == kind
            assert np.allclose(np.diff(clock), .005, atol=1e-6, rtol=0)
            assert np.array_equal(r["policy_update_step_indices"], np.arange(0, len(clock)-1, 4))
            assert int(r["state_reset_count"]) == 1
            assert np.max(abs(r["qfrc_applied"])) == np.max(abs(r["xfrc_applied"])) == 0
            assert np.max(abs(r["qfrc_actuator"][:, :6])) == 0
            expected_force = np.clip(r["ctrl"][1:]-kp*r["qpos"][:-1, qa]-kd*r["qvel"][:-1, va],
                                     *model.actuator_forcerange.T)
            force_error = float(abs(expected_force-r["actuator_force"][1:]).max())
            assert force_error < 5e-4, force_error
            assert np.array_equal(r["actuator_force"], r["qfrc_actuator"][:, va])
            row = dict(kind=kind, world=world, rollout_sha256=digest(path),
                       maximum_actual_dt_error_s=float(abs(np.diff(clock)-.005).max()),
                       actual_native_force_equation_error_nm=force_error,
                       actual_force_transmission_matches=True, zero_external_and_root_assistance=True)
            if kind == "reference_pd":
                qref = interpolate(reference["t"], reference["qpos"][:, qa], clock[:-1])
                vref = interpolate(reference["t"], reference["qvel"][:, va], clock[:-1])
                feed = interpolate(reference["t"], ff, clock[:-1])
                expected_drive = (kp.astype(np.float32)*np.clip(qref, *model.jnt_range[ids].T).astype(np.float32)
                                  + kd.astype(np.float32)*vref+feed)
                drive_error = float(abs(expected_drive-r["ctrl"][1:]).max())
                assert drive_error < .005, drive_error
                row["zero_residual_drive_reconstruction_error_nm"] = drive_error
                assert not str(r["checkpoint"]) and not str(r["checkpoint_sha256"])
            else:
                assert digest(PROJECT / str(r["checkpoint"])) == str(r["checkpoint_sha256"])
            entry["worlds"].append(row)
    results.append(entry)
    print(f"{family}: model identity, force law, actual clocks and paired zero-residual drive verified", flush=True)
(OUT / "report.json").write_text(json.dumps(dict(
    scope="Independent read-only spot audit of nominal and last fresh-test worlds per family; full 64-world acceptance remains in the release evidence.",
    results=results), indent=2)+"\n")
