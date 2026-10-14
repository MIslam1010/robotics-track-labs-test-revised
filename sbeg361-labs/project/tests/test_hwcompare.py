"""Tests of the hardware-vs-simulation comparison helper (hwcompare.py)."""
import numpy as np
import pytest

import hwcompare as hw
from scenarios import desktop_6r

# Course arm (Course resources: Course robot models and hardware safety checklist):
# rated joint speed 160 deg/s on all joints; hardware limits +-165 deg (J1-J5), +-175 deg (J6).
RATED = [160.0] * 6
LIMITS = [[-165, 165]] * 5 + [[-175, 175]]


def test_log_roundtrip(tmp_path):
    lg = hw.synthetic_log(n=5)
    p = tmp_path / "log.csv"
    hw.write_log(p, lg["t"], lg["cmd"], lg["meas"], lg["stop"])
    back = hw.read_log(p)
    assert back["cmd"].shape == lg["cmd"].shape
    assert np.allclose(back["meas"], lg["meas"], atol=1e-4)
    assert np.array_equal(back["stop"], lg["stop"])


def test_joint_rms_known_value():
    cmd = np.zeros((4, 2))
    meas = np.array([[1, 0], [-1, 0], [1, 2], [-1, -2]], float)
    rms, mx = hw.joint_errors(cmd, meas)
    assert np.allclose(rms, [1.0, np.sqrt(2.0)])
    assert np.allclose(mx, [1.0, 2.0])


def test_task_error_zero_for_identical_and_scales_with_offset():
    r = desktop_6r(tool=0.0)
    q = np.tile(np.array([10, -40, 80, -120, -90, 0.0]), (3, 1))
    rms, mx, _ = hw.task_errors(r.Slist, r.M, q, q)
    assert rms == pytest.approx(0.0, abs=1e-12)
    # a 1 deg error on joint 1 moves the flange on a circle about the base axis
    q2 = q.copy()
    q2[:, 0] += 1.0
    T = hw.fk(r.Slist, r.M, np.deg2rad(q[0]))
    radius = np.hypot(T[0, 3], T[1, 3])
    _, mx, _ = hw.task_errors(r.Slist, r.M, q, q2)
    assert mx == pytest.approx(2 * radius * np.sin(np.deg2rad(0.5)), rel=1e-9)


def test_limit_margin_and_speed():
    t = np.linspace(0, 1, 101)
    q = np.c_[30 * t, -10 * t]                       # 30 deg/s and -10 deg/s
    assert np.allclose(hw.limit_margins(q, [[-90, 90], [-20, 20]]), [60, 10])
    assert np.allclose(hw.joint_speeds(t, q), [[30, -10]] * len(t))


def test_report_on_synthetic_log_passes():
    lg = hw.synthetic_log()
    r = desktop_6r(tool=0.0)
    res = hw.hw_report(lg, limits_deg=LIMITS, rated_speed_deg_s=RATED, Slist=r.Slist, M=r.M)
    for key in ("HW-2 joint RMS [deg]", "HW-3 task RMS [mm]", "HW-4 limit margin [deg]",
                "HW-5 speed / rated", "HW-6 halt time [s]", "HW-6 motion after stop [deg]"):
        assert key in res
        value, ok = res[key]
        assert ok, f"{key} = {value}"


def test_report_flags_large_tracking_error_and_no_stop():
    lg = hw.synthetic_log(lag_s=0.5, noise_deg=0.0)
    lg["stop"] = None
    res = hw.hw_report(lg)
    assert res["HW-2 joint max [deg]"][1] is False
    assert not any(k.startswith("HW-6") for k in res)


def test_cli_demo(tmp_path, capsys):
    p = tmp_path / "demo.csv"
    assert hw.main(["--demo", str(p)]) == 0
    lim = [str(x) for pair in LIMITS for x in pair]
    assert hw.main([str(p), "--limits", *lim, "--rated", *(["160"] * 6),
                    "--model", "desktop6r"]) == 0
    out = capsys.readouterr().out
    assert "HW-6 halt time" in out and "FAIL" not in out


def test_hw5_limit_is_25_percent_in_H1_and_50_percent_otherwise(capsys):
    """Safety checklist rule 3: <= 25 % of rated in session H1, <= 50 % otherwise."""
    assert hw.speed_fraction("H1") == 0.25 and hw.speed_fraction("H2") == 0.5
    assert hw.speed_fraction() == 0.5                      # default = "otherwise"
    with pytest.raises(ValueError):
        hw.speed_fraction("H3")
    t = np.linspace(0, 1, 101)
    for v, ok_h1, ok_h2 in [(30.0, True, True),            # 19 % of 160 deg/s
                            (60.0, False, True),           # 37.5 %
                            (100.0, False, False)]:        # 62.5 %
        q = np.c_[v * t, np.zeros((len(t), 5))]
        lg = dict(t=t, cmd=q, meas=q, stop=None)
        key = "HW-5 speed / rated"
        r1 = hw.hw_report(lg, rated_speed_deg_s=RATED, session="H1")[key]
        r2 = hw.hw_report(lg, rated_speed_deg_s=RATED, session="H2")[key]
        assert r1[0] == pytest.approx(v / 160.0) and r1[1] is ok_h1 and r2[1] is ok_h2
        assert hw.hw_report(lg, rated_speed_deg_s=RATED)[key][1] is ok_h2


def test_cli_session_option(tmp_path):
    t = np.linspace(0, 1, 101)
    q = np.c_[60.0 * t, np.zeros((len(t), 5))]           # 37.5 % of rated
    p = tmp_path / "log.csv"
    hw.write_log(p, t, q, q)
    rated = ["--rated", *(["160"] * 6)]
    assert hw.main([str(p), *rated]) == 0
    assert hw.main([str(p), *rated, "--session", "H2"]) == 0
    assert hw.main([str(p), *rated, "--session", "H1"]) == 1
