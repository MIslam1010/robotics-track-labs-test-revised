"""Model and baseline tests for the six project scenarios (run: pytest -q)."""
import json
import os

import numpy as np
import modern_robotics as mr
import pytest

import baseline
from kintools import (fk, jacobian_space, jacobian_body_from_space, body_screws,
                      point_jacobian, ik_body, holding_torques, gravity_wrench_space,
                      manipulability, moved_points, trans)
from scenarios import scenario, ALL, ur_type, DEG

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(361)


def robots():
    return [(sid, scenario(sid).robot) for sid in ALL]


@pytest.mark.parametrize("sid,r", robots())
def test_screw_axes_valid(sid, r):
    for i, c in enumerate(r.joint_types):
        w, v = r.Slist[:3, i], r.Slist[3:, i]
        if c == "R":
            assert np.isclose(np.linalg.norm(w), 1.0)
            assert np.isclose(w @ v, 0.0, atol=1e-12)          # pure rotation, pitch h = 0
            assert np.allclose(v, -np.cross(w, r.q_axis[i]))    # v = -w x q
        else:
            assert np.allclose(w, 0.0) and np.isclose(np.linalg.norm(v), 1.0)


@pytest.mark.parametrize("sid,r", robots())
def test_home_pose_and_body_form(sid, r):
    assert np.allclose(fk(r.Slist, r.M, np.zeros(r.n)), r.M)
    B = body_screws(r.Slist, r.M)
    for _ in range(5):
        th = RNG.uniform(-1.5, 1.5, r.n)
        assert np.allclose(fk(r.Slist, r.M, th), mr.FKinBody(r.M, B, th), atol=1e-10)


def test_ur5_matches_MR_example_4_5():
    r = ur_type("UR5", 0.109, 0.082, 0.425, 0.392, 0.089, 0.095)
    T = fk(r.Slist, r.M, [0, -np.pi / 2, 0, 0, np.pi / 2, 0])
    expected = np.array([[0, -1, 0, 0.095], [1, 0, 0, 0.109], [0, 0, 1, 0.988], [0, 0, 0, 1]])
    assert np.allclose(T, expected, atol=1e-3)


@pytest.mark.parametrize("sid,r", robots())
def test_jacobians(sid, r):
    th = RNG.uniform(-1.2, 1.2, r.n)
    Js = jacobian_space(r.Slist, th)
    # finite differences: [V_s] = dT/dtheta_i T^-1
    h = 1e-6
    T0 = fk(r.Slist, r.M, th)
    for i in range(r.n):
        d = np.zeros(r.n); d[i] = h
        dT = (fk(r.Slist, r.M, th + d) - fk(r.Slist, r.M, th - d)) / (2 * h)
        assert np.allclose(mr.se3ToVec(dT @ mr.TransInv(T0)), Js[:, i], atol=1e-6)
    B = body_screws(r.Slist, r.M)
    assert np.allclose(jacobian_body_from_space(r.Slist, r.M, th), mr.JacobianBody(B, th), atol=1e-10)
    # point Jacobian of the tool origin = linear velocity of {b} origin expressed in {s}
    Jp = point_jacobian(r.Slist, r.M, th, [0, 0, 0])
    Jb = jacobian_body_from_space(r.Slist, r.M, th)
    assert np.allclose(Jp, T0[:3, :3] @ Jb[3:], atol=1e-10)


@pytest.mark.parametrize("sid,r", robots())
def test_ik_recovers_random_pose(sid, r):
    th_true = np.clip(RNG.uniform(-1.0, 1.0, r.n), r.limits[:, 0], r.limits[:, 1])
    if sid == 6:
        th_true[6] = 0.05
    T = fk(r.Slist, r.M, th_true)
    th, ok, it, _ = ik_body(r.Slist, r.M, T, th_true + 0.1)
    assert ok
    assert np.allclose(fk(r.Slist, r.M, th), T, atol=1e-6)


def test_statics_wrist_torque_S5():
    """S5 at pose A: the wrist-pitch holding torque is m g x (0.15 m lever) for a level cradle."""
    s = scenario(5)
    r, t = s.robot, s.task
    p = t["poses"]["A_ground"]
    th, ok, _, _ = ik_body(r.Slist, r.M, baseline.trans(p, t["R_d"]), t["seed"])
    tau = holding_torques(r.Slist, th, gravity_wrench_space(p, t["mass"]))
    assert ok and np.isclose(abs(tau[3]), t["mass"] * 9.81 * 0.15, rtol=1e-6)
    assert np.isclose(tau[0], 0.0, atol=1e-9) and np.isclose(tau[4], 0.0, atol=1e-9)


def test_manipulability_2r_closed_form():
    """2R planar arm, L1 = L2 = 1: det(J) = sin(theta2) so mu3 = |sin theta2| (MR Sec. 5.4)."""
    t2 = 0.7
    J = np.array([[-np.sin(0.3) - np.sin(0.3 + t2), -np.sin(0.3 + t2)],
                  [np.cos(0.3) + np.cos(0.3 + t2), np.cos(0.3 + t2)]])
    assert np.isclose(manipulability(J)["mu3"], abs(np.sin(t2)))


EXPECTED_STATUS = {
    "AT1-1": True, "AT1-2": True, "AT2-1": True, "AT2-2": True, "AT2-3": True, "AT2-4": True, "AT2-6": False,
    "AT3-1": True, "AT3-2": True, "AT3-3": False, "AT4-1": True, "AT4-2": False, "AT4-3": True,
    "AT4-4": True, "AT5-1": True, "AT5-2": False, "AT5-3": True, "AT6-1": True, "AT6-2": True,
    "AT6-3": True,
}


def test_baseline_matches_reference():
    """The baseline still reproduces baseline_reference.json (the numbers quoted in
    scenario_sheets.md).  If this fails you have edited scenarios.py - don't: build your
    redesigned robot in team_design.py instead."""
    with open(os.path.join(HERE, "baseline_reference.json")) as f:
        ref = json.load(f)
    out = baseline.run(verbose=False)
    for sid, (s, res, checks) in out.items():
        for c in checks:
            if c.tid in EXPECTED_STATUS:
                assert c.passed == EXPECTED_STATUS[c.tid], c.tid
            else:
                assert c.passed is None, c.tid
        refc = {c["id"]: c for c in ref[f"S{sid}"]["checks"]}
        for c in checks:
            v = c.value
            if isinstance(v, (float, int, np.floating)) and refc[c.tid]["value"] is not None:
                assert np.isclose(float(v), refc[c.tid]["value"], rtol=1e-3, atol=1e-6), c.tid


def test_s5_actuator_mass_at_elbow():
    """scenario_sheets.md S5: the +6 kg / +11 kg catalogue actuator is a point mass on the
    elbow axis (home position (L1, 0, h)), carried by joints 1-2 only; at baseline pose B
    the +6 kg adds -3.32 N m to tau_2 and nothing to joints 3-5."""
    s = scenario(5); r, t = s.robot, s.task
    th = t["seed"]
    for name in ("A_ground", "B_mid"):                  # same IK chain as baseline_5
        th, ok, _, _ = ik_body(r.Slist, r.M, trans(t["poses"][name], t["R_d"]), th,
                               limits=r.limits)
        assert ok
    p_e = moved_points(r.Slist, th, [((0.80, 0, 0.40), 2)])[0]
    extra = np.zeros(r.n)
    extra[:2] = holding_torques(r.Slist[:, :2], th[:2], gravity_wrench_space(p_e, 6.0))
    assert np.isclose(extra[1], -3.32, atol=5e-3) and np.allclose(extra[2:], 0)
    # same result as the full chain with the wrench applied only to the joints that carry it
    Fe = gravity_wrench_space(p_e, 6.0)
    full = holding_torques(r.Slist, th, Fe)
    assert np.allclose(full[:2], extra[:2])
