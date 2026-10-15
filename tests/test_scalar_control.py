"""Force equivalence, native damping stability, and unchanged body/strength."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.scalar_control import (
    actuator_torque_ranges,
    configure_implicit_pd,
    export_implicit_pd_model,
    export_runtime_model,
    native_pd_drive,
    scalar_actuator_layout,
)


def model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".005" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body><freejoint/><geom type="sphere" size="1" mass="1000"/>
        <body pos="0 0 1"><joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size=".1" mass=".1"/>
        </body></body></worldbody>
      <actuator><motor name="motor" joint="hinge" gear="1"
        ctrllimited="true" ctrlrange="-100 100"/></actuator>
    </mujoco>""")


def test_native_pd_force_matches_explicit_clipped_law_and_root_unactuated():
    native = model()
    configure_implicit_pd(native, [30.], [.3])
    d = mujoco.MjData(native)
    for position, velocity, target, desired_v, feedforward in (
        (.1, -2., .3, 1., 7.), (.5, 3., -10., -4., -100.),
        (-.5, -3., 10., 4., 100.),
    ):
        d.qpos[7], d.qvel[6] = position, velocity
        d.ctrl[:] = native_pd_drive(target, desired_v, 30., .3, feedforward)
        mujoco.mj_forward(native, d)
        expected = np.clip(30*(target-position)+.3*(desired_v-velocity)+feedforward, -100, 100)
        np.testing.assert_allclose(d.actuator_force, [expected], atol=1e-12)
        np.testing.assert_allclose(d.qfrc_actuator[:6], 0, atol=1e-12)
        np.testing.assert_allclose(d.qfrc_actuator[6], expected, atol=1e-12)


def test_native_damping_stable_at_true_five_ms_where_explicit_is_unstable():
    velocities = {}
    for kind in ("native", "explicit"):
        m = model()
        if kind == "native":
            configure_implicit_pd(m, [1.], [.3])
        d = mujoco.MjData(m)
        d.qvel[6] = 1.
        history = []
        for _ in range(40):
            drive = 0.
            d.ctrl[:] = drive if kind == "native" else np.clip(
                drive-d.qpos[7]-.3*d.qvel[6], -100, 100)
            mujoco.mj_step(m, d)
            history.append(d.qvel[6])
        assert d.time == pytest.approx(.2)
        velocities[kind] = np.asarray(history)
    assert abs(velocities["native"]).max() < 1
    assert abs(velocities["explicit"]).max() > 10


def test_native_conversion_preserves_body_and_caps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root/"outputs/scalar-body/human.xml"
    if not source.exists():
        pytest.skip("Locally generated scalar body fixture absent")
    original = mujoco.MjModel.from_xml_path(str(source))
    kp, kd = np.full(original.nu, 30.), np.full(original.nu, 2.)
    destination = export_implicit_pd_model(source, tmp_path/"human_native.xml", kp, kd)
    converted = mujoco.MjModel.from_xml_path(str(destination))
    assert (original.nq, original.nv, original.nu) == (converted.nq, converted.nv, converted.nu)
    for key in ("body_mass", "body_inertia", "body_pos", "jnt_range", "dof_armature", "dof_damping"):
        np.testing.assert_array_equal(getattr(converted, key), getattr(original, key))
    np.testing.assert_array_equal(actuator_torque_ranges(converted), actuator_torque_ranges(original))
    assert converted.opt.timestep == .005
    assert np.all(converted.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
    assert np.all(converted.actuator_forcelimited)
    assert not np.any(converted.actuator_ctrllimited)
    np.testing.assert_array_equal(converted.actuator_biasprm[:, 1], -kp)
    np.testing.assert_array_equal(converted.actuator_biasprm[:, 2], -kd)
    np.testing.assert_array_equal(scalar_actuator_layout(converted)[0],
                                  scalar_actuator_layout(original)[0])
    with pytest.raises(ValueError, match="overwrite"):
        export_implicit_pd_model(source, source, kp, kd)


def test_unbounded_actuator_rejected():
    m = model()
    m.actuator_ctrllimited[:] = False
    with pytest.raises(ValueError, match="output torque cap"):
        configure_implicit_pd(m, [30.], [.3])


@pytest.mark.parametrize("native", [False, True])
def test_runtime_xml_preserves_exact_body_and_actuator_enums(tmp_path, native):
    source = Path(__file__).resolve().parents[1]/"outputs/scalar-body/human.xml"
    if not source.exists():
        pytest.skip("Locally generated scalar body fixture absent")
    source_bytes = source.read_bytes()
    m = mujoco.MjModel.from_xml_path(str(source))
    if native:
        configure_implicit_pd(m, np.full(m.nu, 123.456789012345), np.full(m.nu, 7.654321098765))
    m.opt.timestep = .005
    m.opt.iterations, m.opt.ls_iterations, m.opt.ccd_iterations = 30, 15, 50
    m.opt.tolerance, m.opt.ls_tolerance = 1e-7, .01
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    output = export_runtime_model(source, tmp_path/"runtime.xml", m)
    recompiled = mujoco.MjModel.from_xml_path(str(output))
    assert source.read_bytes() == source_bytes
    assert recompiled.opt.timestep == .005
    assert recompiled.opt.iterations == 30 and recompiled.opt.ls_iterations == 15
    np.testing.assert_array_equal(recompiled.body_mass, m.body_mass)
    np.testing.assert_array_equal(recompiled.body_inertia, m.body_inertia)
    np.testing.assert_array_equal(recompiled.actuator_biastype, m.actuator_biastype)
    np.testing.assert_array_equal(recompiled.actuator_ctrllimited, m.actuator_ctrllimited)
    np.testing.assert_array_equal(recompiled.actuator_forcerange, m.actuator_forcerange)
    assert float(recompiled.body_mass.sum()) == float(m.body_mass.sum())
