"""Checks of the hardware-run helpers (sbeg46x_project.hwtasks) with known answers: identification of
the device models, replay simulations of a logged command, reference signals and the geometric checks
used by the hardware acceptance tests HWk-1 ... HWk-4 (scenario_sheets.md)."""
import math

import numpy as np
import pytest

from sbeg46x_project import arm, devices, hardware, hwtasks, registration, tasks


def test_thresholds_respect_the_derated_limits():
    assert hwtasks.HAPKIT_FORCE_MAX == 2.0 and hwtasks.HAPKIT_RANGE == 0.040
    assert math.degrees(hwtasks.ARM_SPEED_MAX) == pytest.approx(80.0)
    assert math.degrees(hwtasks.ARM_SPEED_FIRST_RUN) == pytest.approx(40.0)
    assert hwtasks.HW[1]["delays"] == (0.0, 0.1, 0.2)            # G.7: 0-200 ms
    assert min(hwtasks.HW[4]["delays"]) == 0.1 and max(hwtasks.HW[4]["delays"]) == 0.5   # 100-500 ms
    assert hwtasks.HW[5]["delays"] == (0.3, 0.6)                 # 300-600 ms
    for k in (2, 4, 5):
        assert hwtasks.HW[k]["speed_fraction"] == arm.SPEED_FRACTION and hwtasks.HW[k]["current_fraction"] == 0.60
    for k in (1, 3, 6):
        assert hwtasks.HW[k]["t_stop"] == 0.010
    assert 0.1 * 9.81 < hwtasks.HAPKIT_FORCE_MAX                # the 100 g drop test load is below 2 N


def test_free_decay_identifies_the_device():
    """K = 0: exact; K = 100 N/m: the sampled spring reduces the apparent damping, and the K T / 2
    correction brings the estimate back within 10 %."""
    for K, tol_b in ((0.0, 0.01), (100.0, 0.10)):
        o = devices.simulate_device(lambda t, x, s: -K * x, 3.0, x0=0.010)
        f = hwtasks.fit_free_decay(o["t"], o["x_meas"], K=K)
        assert f["m_h"] == pytest.approx(devices.HAPKIT["m_h"], rel=0.01)
        assert f["b_h"] == pytest.approx(devices.HAPKIT["b_h"], rel=tol_b)
    f0 = hwtasks.fit_free_decay(o["t"], o["x_meas"], K=100.0, delay_eff=0.0)
    assert f0["b_h"] < 0.85 * devices.HAPKIT["b_h"]              # uncorrected: clearly too low


def test_servo_step_identification():
    t = np.arange(0, 1.5, 1e-3)
    qc = np.where(t < 0.3, 0.0, math.radians(5)) + np.where(t < 0.9, 0.0, math.radians(5))
    sm = arm.ServoModel(wn=24.0, zeta=0.65, delay=0.016, n=1)
    sm.vmax = np.inf
    q = arm.quantise(sm.simulate(t, qc[:, None], q0=[0.0])["q"][:, 0])
    f = hwtasks.fit_servo_step(t, qc, q)
    assert f["delay"] == pytest.approx(0.016, abs=1e-9)
    assert f["wn"] == pytest.approx(24.0, rel=0.02) and f["zeta"] == pytest.approx(0.65, rel=0.02)


def test_imu_bias_at_rest():
    t = np.arange(0, 12, 0.005)
    d = devices.imu_from_angle(t, np.zeros_like(t), seed=4)
    assert hwtasks.imu_bias_at_rest(t, d["gyro"], 10.0) == pytest.approx(devices.IMU["gyro_bias"], abs=2e-3)


def test_replay_device_reproduces_its_own_log():
    def wall(t, x, s):
        return -300.0 * (x - 0.005)
    sim = devices.simulate_device(wall, 1.0)
    log = dict(t=sim["t"], cmd=np.full((len(sim["t"]), 1), 0.005), meas=sim["x_meas"][:, None])
    out = hardware.compare(log, hwtasks.replay_device(wall, sim["t"]), skip=0.1)
    assert out["rms_sim_meas"][0] < 2e-5


def test_replay_follower_hapkit_matches_the_teleoperation_simulator():
    """The follower of a P-P teleoperator without delay, replayed from the logged set-point r and the
    logged tissue force, reproduces the follower trajectory of the full simulation."""
    run = tasks.s1_run(tasks.teleop.PositionPosition(), "transparency", 0.0)
    rep = hwtasks.replay_follower_hapkit(run["t"], run["r"], 200.0, 2.0, f_env_log=run["f_env"], x0=run["x_s"][0])
    log = dict(t=run["t"], cmd=run["r"][:, None], meas=run["x_s"][:, None])
    out = hardware.compare(log, rep, skip=0.5)
    assert out["rms_sim_meas"][0] < 0.2e-3


def test_replay_servo_axis_and_arm():
    t = np.arange(0, 3, 1e-3)
    r = np.where(t < 0.5, 0.0, np.minimum((t - 0.5) * 0.02, 0.02))
    a = hwtasks.replay_servo_axis(t, r)
    assert abs(a["q"][-1] - 0.02) < 2e-4
    t_log = np.arange(0, 2, 0.02)                                  # 50 Hz servo bus
    Q = np.column_stack([0.2 * t_log] * 6)
    out = hwtasks.replay_arm(t_log, Q)
    assert out["t"][1] - out["t"][0] == pytest.approx(1e-3)
    assert np.max(np.abs(out["q"][-1] - Q[-1])) < 0.2 * (0.01 + 2 * 0.8 / 30 + 0.02) + 1e-3


def test_zero_phase_velocity_of_a_sine():
    t = np.arange(0, 10, 1e-3)
    x = 0.02 * np.sin(2 * np.pi * 0.2 * t)
    xq = devices.quantise(x, 2e-5)
    v = hwtasks.zero_phase_velocity(t, xq, fc=20.0)
    v_true = 0.02 * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
    m = (t > 1) & (t < 9)
    assert np.sqrt(np.mean((v - v_true)[m] ** 2)) < 0.3e-3


def test_phase_and_cadence_from_events():
    e = np.arange(0, 10.01, 1.0)
    t = np.array([-0.5, 0.0, 0.25, 3.5, 9.99, 10.5])
    ph = hwtasks.phase_from_events(t, e)
    assert np.isnan(ph[0]) and ph[1] == 0 and ph[2] == pytest.approx(0.25) and ph[3] == pytest.approx(0.5)
    assert ph[4] == pytest.approx(0.99) and np.isnan(ph[5])
    assert hwtasks.cadence_from_events(np.arange(0, 8, 0.8)) == pytest.approx(1.25)


def test_effort_reduction_and_static_force_check():
    load = np.array([0.0, 0.4, 0.8, 0.4, 0.0])
    assert hwtasks.effort_reduction(0.25 * load, load) == pytest.approx(0.75)
    assert hwtasks.effort_reduction(load, load) == pytest.approx(0.0)
    assert hwtasks.static_force_check([0.50, 0.95], [0.49, 0.98]) == pytest.approx(0.03)


def test_rcm_clearance_and_tre_from_measured_joints():
    S = tasks.S2
    q, ax, e, g = tasks.s2_pre_insertion(S["T_ft_true"], S["T_sp_true"])
    T_in, _ = arm.ik_tool(arm.line_pose(e + 0.010 * ax, ax, (0, -1, 0)), q, S["T_ft_true"])
    Q = np.vstack([q, T_in])
    rc = hwtasks.rcm_errors(Q, S["T_ft_true"], e)
    assert np.isnan(rc[0]) and rc[1] < 1e-6                         # above the entry: not counted
    v = registration.apply(S["T_sp_true"], S["vessel_p"])
    c = hwtasks.shaft_clearance(Q[1:], S["T_ft_true"], v, S["r_forbidden"])
    assert c[0] == pytest.approx(arm.segment_point_distance(e + 0.010 * ax, e - 0.090 * ax, v) - 0.005, abs=1e-6)
    T = S["T_sp_true"]
    P = S["checks_p"]
    meas = registration.apply(T, P) + np.array([[0.001, 0, 0], [0, 0.002, 0]])
    assert np.allclose(hwtasks.tre_at_checks(T, P, meas), [0.001, 0.002])
