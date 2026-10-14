"""Checks of the sim-vs-hardware helpers (sbeg46x_project.hardware) on synthetic logs with known
answers. The log format and the first group of helpers are those of the SBEG 461 project; your own
hardware acceptance tests (HWk-1 ... HWk-4) read your logged CSV files from results/hardware/ with
the same functions."""
import numpy as np
import pytest

from sbeg46x_project import devices, hardware


def _synthetic(dt=0.002, T=4.0, delay=0.03, offset=(0.01, -0.02)):
    t = np.arange(0.0, T, dt)
    cmd = np.column_stack([0.5 * np.sin(2 * np.pi * 0.5 * t), 0.3 * np.cos(2 * np.pi * 0.5 * t)])
    meas = np.column_stack([0.5 * np.sin(2 * np.pi * 0.5 * (t - delay)),
                            0.3 * np.cos(2 * np.pi * 0.5 * (t - delay))]) + np.asarray(offset)
    return t, cmd, meas


def test_rms_of_constant_offset_is_the_offset():
    t = np.linspace(0, 1, 501)
    cmd = np.column_stack([np.sin(t), np.cos(t)])
    out = hardware.compare(dict(t=t, cmd=cmd, meas=cmd + np.array([0.01, -0.03])))
    assert np.allclose(out["rms_cmd_meas"], [0.01, 0.03]) and np.allclose(out["max_cmd_meas"], [0.01, 0.03])


def test_csv_round_trip_unsorted_rows_and_aux_columns(tmp_path):
    t, cmd, meas = _synthetic()
    p = tmp_path / "run.csv"
    hardware.write_log(p, t, cmd, meas, cur=0.1 * np.ones_like(cmd), force=np.zeros_like(t),
                       aux=dict(E_ch=np.linspace(0, 1, len(t)), h=np.ones_like(t)))
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0]] + lines[1:][::-1] + [lines[5]]) + "\n")   # reversed + duplicate
    log = hardware.load_log(p)
    assert np.allclose(log["t"], t) and np.allclose(log["cmd"], cmd) and np.allclose(log["meas"], meas)
    assert log["cur"].shape == cmd.shape and log["force"].shape == t.shape
    assert set(log["aux"]) == {"E_ch", "h"} and np.allclose(log["aux"]["E_ch"], np.linspace(0, 1, len(t)))


def test_a_461_log_reads_unchanged(tmp_path):
    """A log in the SBEG 461 format (no aux columns) loads with the same keys as in 461."""
    p = tmp_path / "461.csv"
    p.write_text("t,cmd_1,cmd_2,meas_1,meas_2,cur_1,cur_2\n0,0,0,0,0,0.1,0.2\n0.01,0.1,0.2,0.09,0.19,0.1,0.2\n")
    log = hardware.load_log(p)
    assert set(log) == {"t", "cmd", "meas", "cur"}


def test_load_log_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("t,cmd_1\n0,0\n1,1\n")
    with pytest.raises(ValueError):
        hardware.load_log(p)


def test_delay_estimate_and_alignment():
    t, cmd, meas = _synthetic(offset=(0.0, 0.0))
    assert abs(hardware.estimate_delay(t, cmd[:, 0], meas[:, 0]) - 0.03) <= 0.002 + 1e-12
    sim = dict(t=t, q=cmd.copy())
    raw = hardware.compare(dict(t=t, cmd=cmd, meas=meas), sim, skip=0.5)
    aligned = hardware.compare(dict(t=t, cmd=cmd, meas=meas), sim, skip=0.5, align_delay=True)
    assert np.allclose(raw["rms_sim_meas"], np.array([0.5, 0.3]) * np.pi * 0.03 / np.sqrt(2), rtol=0.02)
    assert np.all(aligned["rms_sim_meas"] < 1e-3)


def test_resample_marks_points_outside_the_log():
    y = hardware.resample([0.0, 1.0], [[0.0], [2.0]], [-0.5, 0.25, 1.5])
    assert np.isnan(y[0, 0]) and np.isclose(y[1, 0], 0.5) and np.isnan(y[2, 0])


def test_compare_haptic_run_against_its_simulation():
    """A 'hardware' log of the haptic device (simulation at 500 Hz with 20 um quantisation) is compared
    with the 1 kHz simulation of the same commanded wall: only the quantisation is left."""
    xc = 0.005

    def wall(t, x, s):
        return -300.0 * (x - xc)
    sim = devices.simulate_device(wall, 1.0, x0=0.0)
    t_hw = sim["t"][::2]
    meas = np.round(sim["x"][::2] / 2e-5) * 2e-5
    out = hardware.compare(dict(t=t_hw, cmd=np.full((len(t_hw), 1), xc), meas=meas[:, None]),
                           dict(t=sim["t"], q=sim["x"]), skip=0.2)
    assert out["rms_sim_meas"][0] <= 2e-5 / np.sqrt(12) * 1.5


def test_torque_constant_gravity_check_and_stiffness():
    rng = np.random.default_rng(461)
    tau_model = np.array([0.10, 0.25, 0.40, 0.55, 0.70])
    current = tau_model / 1.8 * (1 + 0.02 * rng.standard_normal(5))
    k = hardware.torque_constant(current, tau_model)
    assert abs(k - 1.8) < 0.1 and hardware.normalised_rms(k * current, tau_model) < 0.05
    x = np.array([0.0, 0.0017, 0.0033, 0.0050])
    K, f0 = hardware.rendered_stiffness(300 * x + 0.05, x)
    assert np.isclose(K, 300) and np.isclose(f0, 0.05)


def test_limits_and_stop_time():
    assert hardware.within_limits([0.5, -0.59], 1.0, 0.6)
    assert not hardware.within_limits([[0.5, 0.7]], [1.0, 1.0], 0.6)
    t = np.arange(0, 2, 0.01)
    v = np.where(t < 1.0, 1.0, np.exp(-(t - 1.0) / 0.05))
    assert abs(hardware.stop_time(t, v, 1.0, 0.01) - 0.05 * np.log(100)) < 0.011
    assert hardware.stop_time(t, np.ones_like(t), 1.0, 0.01) == np.inf


def test_channel_energy_of_a_lossless_spring_and_of_a_damper():
    t = np.arange(0, 2, 1e-3)
    v = np.cos(2 * np.pi * t)
    E = hardware.channel_energy(t, 3.0 * v, v, 0.0 * v, v)       # damper 3 N s/m at the leader port
    assert np.all(np.diff(E) >= -1e-12) and E[-1] == pytest.approx(3.0 * 1.0, rel=1e-2)
    x = np.sin(2 * np.pi * t) / (2 * np.pi)
    E2 = hardware.channel_energy(t, 100 * x, v, 0.0 * v, v)      # spring: energy returns to ~0
    assert abs(E2[-1]) < 1e-3 and E2.max() > 0


def test_peak_to_peak_barrier_and_rest_time():
    t = np.arange(0, 3, 1e-3)
    x = np.where(t < 1, 0.0, 0.002 * np.exp(-(t - 1) / 0.1) * np.sin(40 * (t - 1)))
    assert hardware.peak_to_peak(t, x, 2.0, 3.0) < 1e-6
    rep = hardware.barrier_violations([0.1, 0.0, -0.001, 0.2], tol=0.0005)
    assert rep["n_violations"] == 1 and rep["h_min"] == -0.001
    assert hardware.rest_time(t, x, 1.0, 0.005) < 0.8


def test_points_from_log():
    log = dict(meas=np.array([[0.0] * 6, [0.1] * 6]))
    P = hardware.points_from_log(log, lambda q: np.block([[np.eye(3), np.array(q[:3])[:, None]], [np.zeros((1, 3)), 1]]))
    assert np.allclose(P, [[0, 0, 0], [0.1, 0.1, 0.1]])
