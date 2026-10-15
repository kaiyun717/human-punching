"""Check the support-allocation diagnostic against analytically known cases."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "reference_dynamics_probe",
    Path(__file__).resolve().parents[1]/"scripts/reference_dynamics_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def corner_jacobian():
    jacobian = np.zeros((12, 7))
    for i, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        jacobian[3*i:3*i+3, :3] = np.eye(3)
        # omega cross point; the last DOF has no effect on these support points.
        jacobian[3*i:3*i+3, 3:6] = [[0, 0, -y], [0, 0, x], [y, -x, 0]]
    return jacobian


@pytest.mark.parametrize("torque,utilization", [(50., .5), (-50., 2.)])
def test_support_lp_recovers_known_asymmetric_strength_utilization(torque, utilization):
    required = np.array([0., 0., 100., 0., 0., 0., torque])
    result = probe.allocate_support(required, corner_jacobian(), np.array([6]),
                                    np.array([[-25., 100.]]), .9)
    assert result["success"]
    assert result["peak_normalized_torque"] == pytest.approx(utilization)
    assert result["root_wrench_residual_max"] < 1e-10
    assert np.asarray(result["forces"])[:, 2].sum() == pytest.approx(100.)


def test_support_lp_rejects_impossible_horizontal_root_force():
    required = np.array([200., 0., 100., 0., 0., 0., 0.])
    result = probe.allocate_support(required, corner_jacobian(), np.array([6]),
                                    np.array([[-100., 100.]]), .9)
    assert not result["success"]
    assert result["status"] == 2
