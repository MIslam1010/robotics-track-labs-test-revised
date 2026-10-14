"""Acceptance-test harness for the TEAM's code. Each test is SKIPPED while the corresponding stub in
sbeg461_project raises NotImplementedError; once you implement it, the test runs automatically.

Thresholds are the acceptance tests of scenario_sheets.md. Add your own tests for the other
acceptance tests of your scenario (stability maps, contact tests, video evidence...).
"""
import numpy as np
import pytest

from sbeg461_project import control, dyn, metrics, tasks, traj
from sbeg461_project.models import load_scenario


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError as e:
        pytest.skip(f"team TODO: {e}")


def test_at2_computed_torque_gait():
    """AT2-1 / AT2-2: computed torque tracks the gait within 0.5 deg RMS (exact model) and
    2 deg RMS (80 % masses, CoMs 2 cm distal), without saturation."""
    s = load_scenario(2)
    p = s.params
    ref = lambda t: traj.gait_reference(t, p["gait_period"])  # noqa: E731
    for model, tol in ((s, 0.5), (s.scaled(p["model_error"]["mass_scale"], p["model_error"]["com_shift"]), 2.0)):
        ctrl = _try(control.computed_torque, model, 400.0, 40.0, ref)
        sim = dyn.simulate(s, ctrl, *ref(0.0)[:2], 2 * p["gait_period"], 1e-3)
        m = metrics.tracking_metrics(sim, ref, skip=0.5 * p["gait_period"])
        assert np.degrees(m["rms_err"]).max() <= tol
        assert m["sat_frac"] <= 0.01


def test_at4_computed_torque_load():
    """AT4-1: computed torque, nominal 30 kg pack: RMS <= 0.5 deg, no saturation."""
    s = load_scenario(4)
    p = s.params
    ref = lambda t: traj.squat_reference(t, p["cycle_period"], p["knee_range"], p["trunk_lean"])  # noqa: E731
    ctrl = _try(control.computed_torque, s, 400.0, 40.0, ref)
    sim = dyn.simulate(s, ctrl, *ref(0.0)[:2], p["cycle_period"], 1e-3)
    m = metrics.tracking_metrics(sim, ref)
    assert np.degrees(m["rms_err"]).max() <= 0.5
    assert m["sat_frac"] == 0.0


def test_at3_time_optimal_faster_than_stop_and_go():
    """AT3-2: the time-optimal scaling of the planned path is at least 20 % faster than the
    stop-and-go quintic baseline and respects the limits (5 % tolerance for the discretised
    phase-plane grid; report the true peak ratios in your report)."""
    import baseline as B
    s = load_scenario(3)
    r = B.baseline_s3()
    fun, T = _try(traj.time_optimal_scaling, s, r["path"])
    assert T <= 0.8 * r["stop_and_go_time"]
    ts = np.linspace(0, T, 200)
    sam = [fun(t) for t in ts]
    qs, qds, qdds = (np.array(x) for x in zip(*sam))
    lim = traj.check_limits(s, qds, traj.torque_profile(s, qs, qds, qdds), tol=0.05)
    assert lim["ok"]


def test_at6_impedance_contact():
    """AT6-2: unexpected contact with a 20 kN/m wall at 0.10 m/s: peak force <= 50 N, final <= 25 N."""
    s = load_scenario(6)
    p = s.params
    import baseline as B
    q0 = B.s6_start(s)
    x0 = s.fk(q0)[:2, 3]
    v = p["approach_speed"]

    def ref(t):
        tt = min(t, 2.0)
        return np.array([x0[0] + v * tt, x0[1]]), np.array([v if t < 2 else 0.0, 0.0]), np.zeros(2)
    ctrl = _try(control.impedance, s, 0.0, p["B_imp"], p["K_imp"], ref)
    wall = tasks.wall_force_s6(s, 20000.0)
    sim = dyn.simulate(s, ctrl, q0, np.zeros(3), 3.0, 1e-3, f_ext=wall)
    F = np.array([-wall(t, q, qd)[0] for t, q, qd in zip(sim["t"], sim["q"], sim["qd"])])
    assert F.max() <= p["F_peak_max"] and F[-1] <= p["F_static_max"]
