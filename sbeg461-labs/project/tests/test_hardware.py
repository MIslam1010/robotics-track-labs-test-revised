"""Checks of the sim-vs-hardware helpers (sbeg461_project.hardware) on synthetic logs with known
answers. Your own hardware acceptance tests (HWk-1 ... HWk-4) read your logged CSV files from
results/hardware/ with the same functions."""
import numpy as np
import pytest

from sbeg461_project import dyn, hardware, traj
from sbeg461_project.control import pd_gravity
from sbeg461_project.models import load_scenario


def _synthetic(dt=0.002, T=4.0, delay=0.03, offset=(0.01, -0.02)):
    t = np.arange(0.0, T, dt)
    cmd = np.column_stack([0.5 * np.sin(2 * np.pi * 0.5 * t), 0.3 * np.cos(2 * np.pi * 0.5 * t)])
    meas = np.column_stack([0.5 * np.sin(2 * np.pi * 0.5 * (t - delay)),
                            0.3 * np.cos(2 * np.pi * 0.5 * (t - delay))]) + np.asarray(offset)
    return t, cmd, meas


def test_rms_of_constant_offset_is_the_offset():
    t = np.linspace(0, 1, 501)
    cmd = np.column_stack([np.sin(t), np.cos(t)])
    log = dict(t=t, cmd=cmd, meas=cmd + np.array([0.01, -0.03]))
    out = hardware.compare(log)
    assert np.allclose(out["rms_cmd_meas"], [0.01, 0.03])
    assert np.allclose(out["max_cmd_meas"], [0.01, 0.03])


def test_csv_round_trip_and_unsorted_rows(tmp_path):
    t, cmd, meas = _synthetic()
    cur = 0.1 * np.ones_like(cmd)
    f = hardware.write_log
    p = tmp_path / "run.csv"
    f(p, t, cmd, meas, cur=cur, force=np.zeros_like(t))
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0]] + lines[1:][::-1] + [lines[5]]) + "\n")  # reversed + duplicate
    log = hardware.load_log(p)
    assert np.allclose(log["t"], t) and np.allclose(log["cmd"], cmd) and np.allclose(log["meas"], meas)
    assert log["cur"].shape == cmd.shape and log["force"].shape == t.shape


def test_load_log_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("t,cmd_1\n0,0\n1,1\n")
    with pytest.raises(ValueError):
        hardware.load_log(p)


def test_delay_estimate_and_alignment():
    t, cmd, meas = _synthetic(offset=(0.0, 0.0))
    d = hardware.estimate_delay(t, cmd[:, 0], meas[:, 0])
    assert abs(d - 0.03) <= 0.002 + 1e-12
    sim = dict(t=t, q=cmd.copy())  # a "perfect" simulation that tracks the command exactly
    raw = hardware.compare(dict(t=t, cmd=cmd, meas=meas), sim, skip=0.5)
    aligned = hardware.compare(dict(t=t, cmd=cmd, meas=meas), sim, skip=0.5, align_delay=True)
    # a 30 ms lag of a 0.5 Hz sine of amplitude A gives an RMS error of about A*2*pi*f*d/sqrt(2)
    assert np.allclose(raw["rms_sim_meas"], np.array([0.5, 0.3]) * np.pi * 0.03 / np.sqrt(2), rtol=0.02)
    assert np.all(aligned["rms_sim_meas"] < 1e-3)
    assert np.allclose(raw["rms_cmd_sim"], 0.0)


def test_resample_marks_points_outside_the_log():
    y = hardware.resample([0.0, 1.0], [[0.0], [2.0]], [-0.5, 0.25, 1.5])
    assert np.isnan(y[0, 0]) and np.isclose(y[1, 0], 0.5) and np.isnan(y[2, 0])


def test_compare_against_a_simulated_run_on_another_grid():
    """A 'hardware' log at 500 Hz (the simulation itself plus encoder quantisation) is compared
    with a 1 kHz dyn.simulate run of the Scenario 2 PD + gravity baseline."""
    s = load_scenario(2)
    p = s.params
    ref = lambda t: traj.gait_reference(t, p["gait_period"])  # noqa: E731
    ctrl = pd_gravity(s, np.array([400.0, 300.0]), np.array([40.0, 30.0]), ref)
    sim = dyn.simulate(s, ctrl, *ref(0.0)[:2], 1.2, 1e-3)
    q = 2 * np.pi / 4096  # 12-bit encoder
    t_hw = sim["t"][::2]
    meas = np.round(sim["q"][::2] / q) * q
    cmd = np.array([ref(tt)[0] for tt in t_hw])
    out = hardware.compare(dict(t=t_hw, cmd=cmd, meas=meas), sim, skip=0.2)
    assert np.all(out["rms_sim_meas"] <= q / np.sqrt(12) * 1.5 + 1e-9)  # only quantisation is left
    assert np.allclose(out["rms_cmd_sim"], out["rms_cmd_meas"], atol=2e-3)


def test_torque_constant_gravity_check_and_stiffness():
    rng = np.random.default_rng(461)
    tau_model = np.array([0.10, 0.25, 0.40, 0.55, 0.70])
    current = tau_model / 1.8 * (1 + 0.02 * rng.standard_normal(5))
    k = hardware.torque_constant(current, tau_model)
    assert abs(k - 1.8) < 0.1
    assert hardware.normalised_rms(k * current, tau_model) < 0.05
    assert np.isclose(hardware.normalised_rms(np.zeros(5), tau_model), 1.0)
    x = np.array([0.0, 0.0017, 0.0033, 0.0050])
    K, f0 = hardware.rendered_stiffness(300 * x + 0.05, x)
    assert np.isclose(K, 300) and np.isclose(f0, 0.05)


def test_limits_and_stop_time():
    assert hardware.within_limits([0.5, -0.59], 1.0, 0.6)
    assert not hardware.within_limits([[0.5, 0.7]], [1.0, 1.0], 0.6)
    t = np.arange(0, 2, 0.01)
    v = np.where(t < 1.0, 1.0, np.exp(-(t - 1.0) / 0.05))
    ts = hardware.stop_time(t, v, 1.0, 0.01)
    assert abs(ts - 0.05 * np.log(100)) < 0.011
    assert hardware.stop_time(t, np.ones_like(t), 1.0, 0.01) == np.inf
