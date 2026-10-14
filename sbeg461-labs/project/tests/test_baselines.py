"""Baseline checks: each scenario's baseline runs and lands in its documented envelope.

The envelopes are the numbers quoted in scenario_sheets.md ("Baseline" rows). Several baselines
deliberately FAIL the project acceptance tests: that gap is what the team must close.
"""
import numpy as np
import pytest

import baseline as B
from sbeg461_project.models import load_scenario


@pytest.fixture(scope="module")
def s1():
    return B.baseline_s1()


@pytest.fixture(scope="module")
def s3():
    return B.baseline_s3()


@pytest.fixture(scope="module")
def s5():
    return B.baseline_s5()


@pytest.fixture(scope="module")
def s6():
    return B.baseline_s6()


def test_s1_rehab_baseline(s1):
    p = load_scenario(1).params
    assert abs(s1["K_rendered"] - p["K_normal"]) / p["K_normal"] < 0.02       # static stiffness exact
    assert 0.015 < s1["max_lateral_dev"] < 0.025                               # ~19.6 mm under 10 N pulse
    assert s1["final_lateral_dev"] < 0.005
    assert abs(s1["reach_progress"] - 0.20) < 0.01                             # reach completed
    assert s1["max_interaction_force"] < p["F_max"]


def test_s2_gait_baseline_misses_target():
    m = B.baseline_s2(cycles=1.5)
    assert m["sat_frac"] == 0.0
    assert np.all(m["peak_tau"] < load_scenario(2).tau_max)
    assert 0.03 < m["rms_err"][0] < 0.09          # hip RMS about 3.4 deg with PD + gravity
    assert np.max(m["rms_err"]) > np.radians(2.0)  # baseline misses the 2 deg (model error) target


def test_s3_surgical_baseline(s3):
    p = load_scenario(3).params
    assert s3["pre_pose_err"] < 1e-3
    assert s3["straight_line_clearance"] < 0          # direct joint motion would hit the light
    assert s3["path_clearance"] >= 0.02 - 1e-3          # planned path keeps the 2 cm margin
    assert 2.0 < s3["stop_and_go_time"] < 10.0
    assert s3["ik_ok"]
    assert s3["reference_rcm_error_max"] < 1e-9        # task reference respects the RCM exactly
    assert s3["joint_line_rcm_error_max"] > p["rcm_tol"]   # joint interpolation violates it


def test_s4_load_exo_baseline_hits_limits():
    m = B.baseline_s4(cycles=1.0)
    s = load_scenario(4)
    assert np.all(m["peak_tau"] <= s.tau_max + 1e-9)
    assert m["sat_frac"] > 0.0                       # 37.5 kg pack drives the knee into its limit
    assert 0.01 < m["rms_err"][0] < 0.05             # ~1.5 deg knee RMS


def test_s5_eod_baseline(s5):
    p = load_scenario(5).params
    for leg in ("stow_to_pregrasp", "pregrasp_to_bin"):
        r = s5[leg]
        assert r["straight_line_clearance"] < 0      # obstacles force a planned path
        assert r["clearance"] >= p["clearance"] - 1e-3
        assert 2.0 < r["stop_and_go_time"] < 15.0
    assert s5["hold_max_tip_dev"] > p["tip_tol_disturbed"]   # PD + level-ground gravity fails
    assert max(s5["hold_peak_tau_ratio"]) < 1.0


def test_s6_casualty_baseline(s6):
    p = load_scenario(6).params
    assert abs(s6["K_rendered_y"] - p["K_imp"]) / p["K_imp"] < 0.02
    assert s6["limb_sag"] > 0.15                      # too soft to lift a 6 kg limb
    assert s6["contact_peak_force"] < p["F_peak_max"]
    assert s6["contact_final_force"] < p["F_static_max"]
