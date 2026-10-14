"""Each baseline lands in its documented envelope (scenario_sheets.md, 'Baseline'). Most baselines
deliberately FAIL several acceptance tests: your design must pass them all and you must explain why
the baseline does not."""
import warnings

import numpy as np
import pytest

from sbeg46x_project import baselines, tasks

warnings.simplefilter("ignore")


def _table(res):
    return {a: (ok, v) for a, ok, v in res}


@pytest.fixture(scope="module")
def s1():
    return _table(tasks.s1_acceptance(baselines.s1_architecture))


def test_s1_baseline(s1):
    assert [a for a, (ok, _) in s1.items() if ok] == ["AT1-1"]
    assert s1["AT1-1"][1]["rms_force_err"] < 0.07
    assert all(v["watchdog"] for v in s1["AT1-2"][1].values())             # unstable under delay
    assert 0.004 < s1["AT1-5"][1][0.0]["overshoot"] < 0.007                  # 5.5 mm loss-of-resistance lurch
    assert s1["AT1-4"][1][0.0]["f_crit_max"] > 1.0                           # touches the dura


def test_s2_baseline():
    r = _table(tasks.s2_acceptance(baselines.S2Baseline(), n_mc=20))
    assert not any(ok for ok, _ in r.values())
    assert r["AT2-1"][1]["tip_offset_error"] == pytest.approx(np.linalg.norm([0.6e-3, -0.4e-3, 0.8e-3]))
    assert 3.5e-3 < r["AT2-2"][1]["tre_rms"] < 4.5e-3                         # placement error, no registration
    assert 3.0e-3 < r["AT2-3"][1]["rcm_max"] < 5.0e-3
    assert r["AT2-4"][1]["ratio"] == pytest.approx(1.0, abs=1e-3)
    assert r["AT2-5"][1]["clearance_min_nudge"] < -1e-3


def test_s3_baseline():
    r = _table(tasks.s3_acceptance(baselines.S3Baseline()))
    assert [a for a, (ok, _) in r.items() if ok] == ["AT3-5"]
    assert 9e-3 < r["AT3-2"][1]["rmse"] < 10.5e-3
    assert 1.4e-3 < r["AT3-1"][1]["v_rmse"] < 2.0e-3
    assert 0.028 < r["AT3-4"][1]["x_abs_max_spasm"] < 0.030


def test_s4_baseline():
    r = _table(tasks.s4_acceptance(baselines.S4Baseline()))
    assert [a for a, (ok, _) in r.items() if ok] == ["AT4-4"]
    assert r["AT4-1"][1]["tre_rms"] == pytest.approx(0.008, abs=1e-4)
    assert all(v["watchdog"] for v in r["AT4-2"][1].values())
    assert all(abs(v["clearance_min"] - 0.012) < 5e-4 for v in r["AT4-3"][1].values())


def test_s5_baseline():
    r = _table(tasks.s5_acceptance(baselines.S5Baseline()))
    assert [a for a, (ok, _) in r.items() if ok] == ["AT5-3"]
    assert all(v["watchdog"] for v in r["AT5-2"][1].values())
    assert 2.5e-3 < r["AT5-4"][1]["rcm_max"] < 4.5e-3
    assert 3.0e-3 < r["AT5-1"][1]["tre_rms_target"] < 3.7e-3


def test_s6_baseline():
    r = _table(tasks.s6_acceptance(baselines.S6Baseline()))
    assert [a for a, (ok, _) in r.items() if ok] == ["AT6-3"]
    assert r["AT6-1"][1]["phase_rms_1"] < 1e-6 and 0.25 < r["AT6-1"][1]["phase_rms_2"] < 0.32
    assert r["AT6-2"][1]["effort_reduction"] < 0.0
    assert r["AT6-4"][1]["x_max_push"] > 0.0353
