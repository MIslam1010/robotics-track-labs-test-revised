"""Checks of the starter models with hand-computed or textbook answers (run before you change anything)."""
import math

import numpy as np
import modern_robotics as mr
import pytest

from sbeg46x_project import arm, devices, estimation, optctrl, registration, tasks, teleop, tissue


# ----------------------------------------------------------------------------- devices
def test_hapkit_data_are_the_course_values():
    h = devices.HAPKIT
    assert (h["m_h"], h["b_h"], h["k_g"], h["T"], h["force_max"], h["x_range"]) == (0.05705, 0.15, 1.2111, 1e-3, 2.0, 0.040)


def test_hapkit_saturation_and_watchdog():
    d = devices.Hapkit()
    assert d.command(3.0) == 2.0 and d.sat_count == 1
    d.x = 0.041
    assert d.command(1.0) == 0.0 and d.tripped
    d.x = 0.0
    assert d.command(1.0) == 0.0          # latched


def test_free_decay_matches_mass_damper():
    """Hapkit alone, released 10 mm off centre: x'' = -(b x' + k_g x)/m  -> very light damping."""
    d = devices.Hapkit()
    out = devices.simulate_device(lambda t, x, s: 0.0, 1.0, x0=0.010, dev=d)
    wn = math.sqrt(d.k_g / d.m_h)
    zeta = d.b_h / (2 * math.sqrt(d.k_g * d.m_h))
    t = out["t"]
    x_ref = 0.010 * np.exp(-zeta * wn * t) * (np.cos(wn * math.sqrt(1 - zeta**2) * t)
                                               + zeta / math.sqrt(1 - zeta**2) * np.sin(wn * math.sqrt(1 - zeta**2) * t))
    assert np.max(np.abs(out["x"] - x_ref)) < 2e-5


def test_velocity_estimator_ramp():
    est = devices.VelocityEstimator()
    v = [est(0.01 * k * 1e-3) for k in range(400)]
    assert abs(v[-1] - 0.01) < 1e-6


def test_imu_at_rest_gives_the_tilt():
    t = np.arange(0, 1, 0.005)
    d = devices.imu_from_angle(t, np.full(len(t), 0.3), seed=1, gyro_noise=0, acc_noise=0, gyro_bias=0)
    assert np.allclose(estimation.accel_tilt(d["ax"], d["ay"]), 0.3)
    assert np.allclose(d["gyro"], 0.0)


def test_servo_axis_rate_limit_and_settling():
    s = devices.ServoAxis(v_max=0.05, delay=0.0)
    for _ in range(2000):
        s.command(0.02)
        for _ in range(5):
            s.advance(2e-4)
    assert abs(s.x - 0.02) < 1e-5
    s2 = devices.ServoAxis(v_max=0.05, delay=0.0)
    s2.command(0.02)
    assert abs(s2.x_cmd - 0.05e-3) < 1e-12        # one 1 ms step at 50 mm/s


# ----------------------------------------------------------------------------- tissue
def test_needle_puncture_depths_by_hand():
    m = tissue.NeedleTissue()
    z_skin, z_lig = m.puncture_depths()
    assert abs(z_skin - (-100 + math.sqrt(100**2 + 4 * 20000 * 0.8)) / 40000) < 1e-12   # 4.30 mm
    assert abs(z_lig - (0.030 + 0.6 / 250)) < 1e-12                                     # 32.4 mm
    prof = tissue.needle_quasistatic_profile(tissue.NEEDLE_S1, np.array([0.002, 0.004, 0.0045, 0.010, 0.031, 0.0335]))
    assert np.isclose(prof[0], 100 * 0.002 + 20000 * 0.002**2)          # elastic skin
    assert np.isclose(prof[3], 12.0 * 0.010 + 0.20)                     # friction + cutting at 10 mm
    assert np.isclose(prof[4], 12.0 * 0.030 + 0.20 + 250 * 0.001)       # ligament loading
    assert np.isclose(prof[5], 12.0 * 0.030)                            # after the loss of resistance


def test_punctures_are_permanent_and_dura_is_a_wall():
    m = tissue.NeedleTissue()
    for z in np.linspace(0, 0.034, 400):
        m.force(z, 0.005)
    assert [e[0] for e in m.events] == ["skin", "ligament"]
    assert m.force(0.003, -0.005) <= 0.0                 # withdrawing: friction only
    assert np.isclose(m.critical_force(0.038), 2000 * 0.001)


def test_wall_force_pushes_only():
    assert tissue.wall_force(0.011, 0.0, 1000.0, 0.0, 0.010) == pytest.approx(-1.0)
    assert tissue.wall_force(0.011, -1.0, 1000.0, 2.0, 0.010) == 0.0
    assert tissue.wall_force(-0.011, 0.0, 1000.0, 0.0, -0.010, side=-1) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- teleoperation
def test_delay_line_holds_and_drops():
    d = teleop.DelayLine([3, 3, -1, 3, 3, 3, 3, 3], [0.0])
    got = []
    for n in range(8):
        d.send(n, [float(n)])
        got.append(float(d.receive(n)[0]))
    assert got == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 4.0]


def test_position_position_is_transparent_without_delay():
    op = devices.Operator(devices.ramp_to(0.0, 0.010, 0.010, 0.1))
    out = teleop.simulate(teleop.PositionPosition(), op, 2.0)
    assert abs(out["x_s"][-1] - out["x_m"][-1]) < 2e-4
    assert out["E_ch"].min() > -1e-4


def test_team_stubs_raise():
    with pytest.raises(NotImplementedError):
        teleop.wave_leader(0.0, 0.0, 1.0)


# ----------------------------------------------------------------------------- course arm
def test_course_arm_home_and_tool():
    _, M = arm.screw_list(0.0)
    assert np.allclose(M[:3, 3], [0.206, 0.109, 0.0561])
    for L in (0.06, 0.10, 0.12):
        T = arm.fk(np.zeros(6), L)
        assert np.allclose(T[:3, 3], [0.206, 0.109 + L, 0.0561])      # z_b = +y_s at home


def test_course_arm_fk_matches_modern_robotics_body_form():
    S, M = arm.screw_list(0.10)
    B = mr.Adjoint(mr.TransInv(M)) @ S
    q = np.array([0.3, -0.6, 1.2, -2.1, -1.5, 0.4])
    assert np.allclose(arm.fk(q, 0.10), mr.FKinBody(M, B, q))


def test_soft_limits_follow_the_checklist():
    lim = np.degrees(arm.soft_limits())
    assert np.allclose(lim[:5], [[-155, 155]] * 5) and np.allclose(lim[5], [-165, 165])
    assert arm.SPEED_FRACTION * np.degrees(arm.RATED_SPEED) == pytest.approx(80.0)


def test_point_jacobian_by_finite_differences():
    q = np.array([0.1, -0.7, 1.3, -2.2, -1.5, 0.2])
    T_ft = registration.pose(None, (0.001, -0.002, 0.10))
    p = arm.fk_tool(q, T_ft)[:3, 3]
    J = arm.point_jacobian(q, p)
    dq = 1e-7
    Jfd = np.column_stack([(arm.fk_tool(q + dq * e, T_ft)[:3, 3] - p) / dq for e in np.eye(6)])
    assert np.allclose(J, Jfd, atol=1e-5)


def test_rcm_and_segment_distances():
    T = arm.line_pose([0, 0, 0], [0, 0, -1])
    assert arm.shaft_distance(T, [0.003, 0.004, 0.5]) == pytest.approx(0.005)
    assert arm.segment_point_distance([0, 0, 0], [0, 0, 1], [0, 0.002, 2]) == pytest.approx(math.hypot(0.002, 1))


def test_quintic_duration_peaks():
    T = arm.quintic_duration([0.0], [1.0], 1.0)
    assert T == pytest.approx(1.875)
    qd = max(abs(arm.quintic([0.0], [1.0], T, t)[1][0]) for t in np.linspace(0, T, 2001))
    assert qd == pytest.approx(1.0, rel=1e-4)


def test_servo_model_tracks_a_ramp_with_its_delay():
    t = np.arange(0, 2, 1e-3)
    q_cmd = np.column_stack([0.2 * t] * 6)
    out = arm.ServoModel(delay=0.01).simulate(t, q_cmd)
    lag = 0.2 * (0.01 + 2 * 0.8 / 30.0)                 # transport delay + 2 zeta / wn of the loop
    assert abs((q_cmd[-1, 0] - out["q"][-1, 0]) - lag) < 2e-4


# ----------------------------------------------------------------------------- optimisation / estimation
def test_c2d_double_integrator():
    Ad, Bd = optctrl.c2d([[0, 1], [0, 0]], [[0], [1]], 0.01)
    assert np.allclose(Ad, [[1, 0.01], [0, 1]]) and np.allclose(Bd.ravel(), [0.5e-4, 0.01])


def test_dlqr_scalar_riccati():
    a, b, q, r = 1.02, 0.1, 2.0, 0.5
    K, P = optctrl.dlqr([[a]], [[b]], [[q]], [[r]])
    p = P[0, 0]
    assert p == pytest.approx(q + a * a * p - (a * b * p) ** 2 / (r + b * b * p))
    assert K[0, 0] == pytest.approx(a * b * p / (r + b * b * p))


def test_solve_qp_box():
    x, info = optctrl.solve_qp(np.eye(2), [-2.0, -2.0], [[1.0, 1.0]], [1.0], lb=[0, 0], ub=[1, 1])
    assert info["success"] and np.allclose(x, [0.5, 0.5], atol=1e-6)


def test_phase_portrait_of_a_circle():
    p = np.linspace(0, 0.99, 100)
    th = 5 + 20 * np.cos(2 * np.pi * p)
    om = -20 * 2 * np.pi * np.sin(2 * np.pi * p)
    assert np.allclose(estimation.phase_portrait(th, om, 5.0, 2 * np.pi), p, atol=1e-9)


def test_complementary_filter_removes_accelerometer_noise():
    t = np.arange(0, 10, 0.005)
    th = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    d = devices.imu_from_angle(t, th, seed=2, gyro_bias=0.0)
    acc = estimation.accel_tilt(d["ax"], d["ay"])
    cf = estimation.complementary_filter(t, d["gyro"], acc, tau=0.5)
    m = t > 2
    assert np.sqrt(np.mean((cf - th)[m] ** 2)) < 0.5 * np.sqrt(np.mean((acc - th)[m] ** 2))


# ----------------------------------------------------------------------------- registration helpers
def test_fre_tre_of_a_known_transform():
    T = registration.pose(registration.rot_z(0.1), (0.01, 0.02, 0.0))
    P = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
    assert registration.fre(T, P, registration.apply(T, P)) == pytest.approx(0.0, abs=1e-15)
    T2 = T.copy()
    T2[:3, 3] += [0.001, 0, 0]
    assert registration.tre(T2, T, [0.1, 0.2, 0.3]) == pytest.approx(0.001)


# ----------------------------------------------------------------------------- scenario set-ups
def test_s2_geometry_and_pre_insertion_pose():
    S = tasks.S2
    q, ax, e, g = tasks.s2_pre_insertion(S["T_ft_true"], S["T_sp_true"])
    T = arm.fk_tool(q, S["T_ft_true"])
    assert np.allclose(T[:3, 3], e - S["pre_height"] * ax, atol=1e-6)
    assert np.allclose(T[:3, 2], ax, atol=1e-6)
    lim = arm.soft_limits()
    assert np.all(q > lim[:, 0]) and np.all(q < lim[:, 1])
    d = arm.segment_point_distance(e, g, registration.apply(S["T_sp_true"], S["vessel_p"]))
    assert d > S["r_forbidden"]                                      # the planned line is safe


def test_s3_target_and_s6_profiles():
    assert tasks.s3_target(1.25)[0] == pytest.approx(0.020)
    p = np.linspace(0, 1, 1001)
    k = tasks.s6_knee_deg(p)
    assert 4.9 < k.min() < 6 and 59 < k.max() < 61
    assert tasks.s6_x(35.0) == 0.0
    sea = tasks.VirtualSEA(0.0)
    sea.x_mot = 0.001
    assert sea.spring_force(0.0) == pytest.approx(0.3)
