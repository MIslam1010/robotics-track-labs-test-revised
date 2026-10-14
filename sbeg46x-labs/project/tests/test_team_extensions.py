"""Acceptance-test harness for the TEAM's code. Each test is SKIPPED while the corresponding TODO
stub raises NotImplementedError; once you port your lab code (or set SCENARIO in team_design.py),
the test runs automatically. Thresholds are those of scenario_sheets.md."""
import math
import warnings

import numpy as np
import pytest

from sbeg46x_project import estimation, optctrl, registration, safety, tasks, teleop

warnings.simplefilter("ignore")


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError as e:
        pytest.skip(f"team TODO: {e}")


# ------------------------------------------------------------------------- lab functions you port
def _waves(F, xdot, b):
    """Lab 3 wave transformation: forward u = (b xdot + F)/sqrt(2b), returning v = (b xdot - F)/sqrt(2b)."""
    s = math.sqrt(2 * b)
    return (b * xdot + F) / s, (b * xdot - F) / s


def test_wave_variables_conserve_power():
    b = 5.0
    F_m, u_m = _try(teleop.wave_leader, 0.02, 0.1, b)                  # xdot_m = 0.02, v_m = 0.1
    v_m = 0.1
    assert F_m * 0.02 == pytest.approx((u_m**2 - v_m**2) / 2)          # power in = (u^2 - v^2)/2
    xdot_d, v_s = _try(teleop.wave_follower, 0.3, 0.2, b)
    assert 0.2 * xdot_d == pytest.approx((0.3**2 - v_s**2) / 2)


def test_wave_variables_signs_lab3_convention():
    """Signs, not only squares: the outputs must be the Lab 3 encoding of their power pair."""
    b, s = 5.0, math.sqrt(10.0)
    F_m, u_m = _try(teleop.wave_leader, 0.02, 0.1, b)
    assert F_m == pytest.approx(5.0 * 0.02 - s * 0.1)                  # F_m = b xdot_m - sqrt(2b) v_m < 0
    assert u_m == pytest.approx(s * 0.02 - 0.1)                        # u_m = sqrt(2b) xdot_m - v_m
    assert np.allclose(_waves(F_m, 0.02, b), (u_m, 0.1))
    xdot_d, v_s = _try(teleop.wave_follower, 0.3, 0.2, b)
    assert xdot_d == pytest.approx((s * 0.3 - 0.2) / b)
    assert v_s == pytest.approx(s * xdot_d - 0.3)
    assert np.allclose(_waves(0.2, xdot_d, b), (0.3, v_s))
    # a still leader (xdot_m = 0): F_m = -sqrt(2b) v_m, negative for a positive returning wave
    assert _try(teleop.wave_leader, 0.0, 1.0, b)[0] < 0


def test_tdpa_limits_the_output_energy():
    F, E_in, E_out, r = _try(teleop.tdpa_leader, 1.0, -0.1, 1e-3, 0.0, 0.0, 2e-5)
    assert r == pytest.approx(0.2) and E_out == pytest.approx(2e-5)
    dx, E_in_s, E_out_s, r = _try(teleop.tdpa_follower, 1.0, 1e-4, 0.0, 0.0, 5e-5)
    assert dx == pytest.approx(5e-5) and E_out_s == pytest.approx(5e-5)


def test_pivot_calibration_exact_data():
    rng = np.random.default_rng(0)
    t_true, p_piv = np.array([0.001, -0.002, 0.1]), np.array([0.2, 0.0, 0.05])
    Rs = [registration.rot_z(a) @ np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
          for a, b in rng.uniform(-0.6, 0.6, (12, 2))]
    ps = [p_piv - R @ t_true for R in Rs]
    t, p, res = _try(registration.pivot_calibration, Rs, ps)
    assert np.allclose(t, t_true, atol=1e-9) and np.allclose(p, p_piv, atol=1e-9) and res < 1e-9


def test_registration_and_tre_prediction():
    P = tasks.S2["fiducials_p"]
    T = registration.pose(registration.rot_z(0.3), (0.2, 0.05, 0.04))
    T_est = _try(registration.paired_point_registration, P, registration.apply(T, P))
    assert np.allclose(T_est, T, atol=1e-12)
    # prediction against a Monte-Carlo run with isotropic FLE (sigma per axis)
    sigma, rng, tre = 0.5e-3, np.random.default_rng(1), []
    for _ in range(3000):
        Te = registration.paired_point_registration(P, registration.localise(registration.apply(T, P), sigma, rng))
        tre.append(registration.tre(Te, T, tasks.S2["target_p"]))
    pred = _try(registration.tre_prediction, P, tasks.S2["target_p"], math.sqrt(3) * sigma)
    assert pred == pytest.approx(np.sqrt(np.mean(np.square(tre))), rel=0.06)


def test_kalman_filter_constant_velocity():
    A, B, C = np.array([[1, 0.01], [0, 1]]), np.zeros((2, 1)), np.array([[1.0, 0.0]])
    kf = estimation.KalmanFilter(A, B, C, np.diag([1e-10, 1e-8]), [[1e-6]], [0.0, 0.0], np.eye(2))
    rng = np.random.default_rng(3)
    for k in range(500):
        _try(kf.predict)
        kf.update(0.05 * k * 0.01 + 1e-3 * rng.standard_normal())
    assert abs(kf.x[1] - 0.05) < 2e-3


def test_mpc_matches_lqr_when_unconstrained():
    Ad, Bd = optctrl.c2d([[0, 1], [0, 0]], [[0], [1]], 0.02)
    Q, R = np.diag([100.0, 1.0]), np.array([[0.1]])
    K, _ = optctrl.dlqr(Ad, Bd, Q, R)
    mpc = optctrl.LinearMPC(Ad, Bd, Q, R, N=10)
    u = _try(mpc.solve, np.array([0.1, 0.0]))
    assert np.allclose(u, -K @ np.array([0.1, 0.0]), rtol=1e-3, atol=1e-6)


def test_cbf_filter_1d():
    u = _try(safety.cbf_filter_1d, 1.0, 0.01, 0.0, -1.0, 10.0)        # h = 0.01, h_dot = -u
    assert u == pytest.approx(0.1)
    assert safety.cbf_filter_1d(-1.0, 0.01, 0.0, -1.0, 10.0) == -1.0  # inactive


# ------------------------------------------------------------------------- your scenario
def test_team_scenario_acceptance():
    """All simulation acceptance tests of your scenario (team_design.SCENARIO)."""
    import team_design
    if team_design.SCENARIO is None:
        pytest.skip("set SCENARIO in team_design.py")
    design = _try(team_design.make_design)
    fn = {1: tasks.s1_acceptance, 2: tasks.s2_acceptance, 3: tasks.s3_acceptance, 4: tasks.s4_acceptance,
          5: tasks.s5_acceptance, 6: tasks.s6_acceptance}[team_design.SCENARIO]
    res = _try(fn, design)
    failed = [(a, v) for a, ok, v in res if not ok]
    assert not failed, "acceptance tests failed: " + "; ".join(f"{a}: {v}" for a, v in failed)
