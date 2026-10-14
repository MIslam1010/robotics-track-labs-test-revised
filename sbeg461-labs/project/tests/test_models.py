"""Model checks for all six scenarios: kinematics, inertia, dynamics properties, cross-checks."""
import modern_robotics as mr
import numpy as np
import pytest

from sbeg461_project import dyn, traj
from sbeg461_project.models import (SCENARIOS, body_inertia, load_scenario, spatial_inertia_to_mci,
                                    trans, transform_inertia)
from sbeg461_project.pin_bridge import HAVE_PINOCCHIO

KEYS = sorted(SCENARIOS)
RNG = np.random.default_rng(461)


@pytest.mark.parametrize("k", KEYS)
def test_shapes_and_screw_axes(k):
    s = load_scenario(k)
    n = s.n
    assert s.Slist.shape == (6, n)
    assert len(s.Mlist) == n + 1 and len(s.Glist) == n
    for x in (s.tau_max, s.qdot_max, s.q_home):
        assert x.shape == (n,)
    assert s.q_limits.shape == (n, 2)
    for i in range(n):
        w, v = s.Slist[:3, i], s.Slist[3:, i]
        assert np.isclose(np.linalg.norm(w), 1.0) or (np.allclose(w, 0) and np.isclose(np.linalg.norm(v), 1))
    assert s.domain in ("biomedical", "defence")
    assert len(set(s.components) & {"dynamics", "trajectory", "planning", "control"}) >= 3


@pytest.mark.parametrize("k", KEYS)
def test_home_configuration(k):
    """FK at q = 0 is M, and the product of Mlist is M (MR Eq. 4.14 and Sec. 8.3.2)."""
    s = load_scenario(k)
    assert np.allclose(s.fk(np.zeros(s.n)), s.M, atol=1e-12)
    P = np.eye(4)
    for Mi in s.Mlist:
        P = P @ Mi
    assert np.allclose(P, s.M, atol=1e-12)
    assert np.all(s.q_home >= s.q_limits[:, 0]) and np.all(s.q_home <= s.q_limits[:, 1])


@pytest.mark.parametrize("k", KEYS)
def test_inertias_positive_definite(k):
    s = load_scenario(k)
    for G in s.Glist:
        assert np.allclose(G, G.T)
        assert np.all(np.linalg.eigvalsh(G) > 0)


@pytest.mark.parametrize("k", KEYS)
def test_mass_matrix_properties(k):
    """M(q) symmetric positive definite; c(q, qd) = C(q, qd) qd with Christoffel C; Mdot - 2C skew
    (MR Sec. 8.1.2, Eq. 8.19 and Proposition 8.1)."""
    s = load_scenario(k)
    n, h = s.n, 1e-6
    for _ in range(3):
        q = RNG.uniform(-1.5, 1.5, n)
        qd = RNG.uniform(-1, 1, n)
        Mq = dyn.mass_matrix(s, q)
        assert np.allclose(Mq, Mq.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(Mq) > 0)
        dM = np.zeros((n, n, n))   # dM[:, :, k] = dM/dq_k
        for kk in range(n):
            e = np.zeros(n)
            e[kk] = h
            dM[:, :, kk] = (dyn.mass_matrix(s, q + e) - dyn.mass_matrix(s, q - e)) / (2 * h)
        # Christoffel form: C_ij = 1/2 sum_k (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) qd_k
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = 0.5 * sum((dM[i, j, kk] + dM[i, kk, j] - dM[j, kk, i]) * qd[kk]
                                    for kk in range(n))
        c_mr = mr.VelQuadraticForces(q, qd, s.Mlist, s.Glist, s.Slist)
        assert np.allclose(C @ qd, c_mr, atol=1e-6)
        Mdot = np.einsum("ijk,k->ij", dM, qd)
        S = Mdot - 2 * C
        assert np.allclose(S, -S.T, atol=1e-6)


def test_hand_computed_gravity_torques():
    """Gravity torques at q = 0 checked by hand (tau = m g x lever)."""
    g = 9.81
    s2 = load_scenario(2)
    assert np.allclose(dyn.gravity(s2, [0, 0]), 0, atol=1e-12)          # leg hanging: zero
    s4 = load_scenario(4)
    assert np.allclose(dyn.gravity(s4, [0, 0]), [-34 * g * 0.15, 34 * g * 0.15], atol=1e-9)
    s5 = load_scenario(5)
    lever = 5.0 * 0.30 + 1.5 * 0.65 + 2.5 * 0.93 + 1.0 * 1.20 + 1.5 * 1.30
    assert np.isclose(dyn.gravity(s5, np.zeros(6))[1], -lever * g, atol=1e-9)
    s6 = load_scenario(6)
    lever6 = 4.0 * 0.225 + 2.5 * 0.65 + 1.2 * 0.925 + 0.5 * 1.0
    assert np.isclose(dyn.gravity(s6, np.zeros(3))[0], lever6 * g, atol=1e-9)


def test_payload_equals_point_mass():
    """with_payload adds exactly m g x lever to the gravity torques (MR Eq. 8.42)."""
    s = load_scenario(6)
    d = dyn.gravity(s.with_payload(1.0), np.zeros(3)) - dyn.gravity(s, np.zeros(3))
    assert np.allclose(d, 9.81 * np.array([1.0, 0.55, 0.15]), atol=1e-9)


def test_spatial_inertia_roundtrip():
    G_b = body_inertia(2.0, [0.1, 0.2, 0.3])
    T = trans([0.1, -0.2, 0.3])
    m, c, Ic = spatial_inertia_to_mci(transform_inertia(G_b, mr.TransInv(T)))
    assert np.isclose(m, 2.0) and np.allclose(c, [0.1, -0.2, 0.3]) and np.allclose(Ic, np.diag([0.1, 0.2, 0.3]))


def test_tip_force_sign_convention():
    """Holding torque tau = g(q) - J_v^T f balances an external force f on the tip (MR Eq. 5.26)."""
    s = load_scenario(6)
    q = s.q_home
    f = np.array([3.0, -7.0, 0.0])
    T = s.fk(q)
    Jv = T[:3, :3] @ s.jacobian_body(q)[3:]
    tau = dyn.gravity(s, q) - Jv.T @ f
    qdd = dyn.forward_dynamics(s, q, np.zeros(3), tau, dyn.ftip_from_force(s, q, f))
    assert np.allclose(qdd, 0, atol=1e-9)


@pytest.mark.skipif(not HAVE_PINOCCHIO, reason="pinocchio not installed")
@pytest.mark.parametrize("k", KEYS)
def test_pinocchio_matches_modern_robotics(k):
    from sbeg461_project.pin_bridge import PinEngine
    s = load_scenario(k)
    e = PinEngine(s)
    for _ in range(3):
        q, qd, qdd = (RNG.uniform(-1, 1, s.n) for _ in range(3))
        F = RNG.uniform(-5, 5, 6)
        assert np.allclose(dyn.inverse_dynamics(s, q, qd, qdd, F), e.inverse_dynamics(q, qd, qdd, F), atol=1e-9)
        assert np.allclose(dyn.mass_matrix(s, q), e.mass_matrix(q), atol=1e-9)


@pytest.mark.skipif(not HAVE_PINOCCHIO, reason="pinocchio not installed")
def test_simulation_engines_agree():
    from sbeg461_project import control
    s = load_scenario(2)
    ref = lambda t: traj.gait_reference(t)  # noqa: E731
    ctrl = control.pd_gravity(s, [400, 300], [40, 30], ref)
    a = dyn.simulate(s, ctrl, *ref(0)[:2], 0.3, 1e-3, engine="mr")
    b = dyn.simulate(s, ctrl, *ref(0)[:2], 0.3, 1e-3, engine="pinocchio")
    assert np.allclose(a["q"], b["q"], atol=1e-9)


def test_energy_conservation_rk4():
    """Unforced, frictionless S6 arm released from rest: RK4 conserves energy (MR Sec. 8.5)."""
    s = load_scenario(6)
    sim = dyn.simulate(s, lambda t, q, qd: np.zeros(3), s.q_home, np.zeros(3), 0.5, 1e-3,
                       saturate=False)
    E = [dyn.energy(s, q, qd) for q, qd in zip(sim["q"], sim["qd"])]
    assert np.ptp(E) < 1e-5 * max(1.0, abs(E[0]))


def test_time_scalings():
    for t in np.linspace(0, 2, 11):
        assert np.isclose(traj.quintic_s(t, 2.0)[0], mr.QuinticTimeScaling(2.0, t))
    h = 1e-6
    for fn in (traj.gait_reference, traj.squat_reference):
        for t in (0.13, 0.71, 1.9):
            q, qd, qdd = fn(t)
            assert np.allclose((fn(t + h)[0] - fn(t - h)[0]) / (2 * h), qd, atol=1e-6)
            assert np.allclose((fn(t + h)[1] - fn(t - h)[1]) / (2 * h), qdd, atol=1e-5)


def test_task_poses():
    """Stored joint solutions reach the specified task poses (to 1 mm and 0.1 deg)."""
    s3 = load_scenario(3)
    p = s3.params
    T = s3.fk(p["q_pre"])
    assert np.linalg.norm(T[:3, 3] - (p["p_rcm"] + [0, 0, p["pre_insertion_height"]])) < 1e-3
    assert np.allclose(T[:3, :3], p["R_pre"], atol=2e-3)
    s5 = load_scenario(5)
    p = s5.params
    for key, target in (("q_pregrasp", p["grasp_point"] + [0, 0, p["pregrasp_offset"]]),
                        ("q_grasp", p["grasp_point"]), ("q_bin", p["bin_point"])):
        T = s5.fk(p[key])
        assert np.linalg.norm(T[:3, 3] - target) < 1e-3, key
        assert np.allclose(T[:3, :3], p["R_grip"], atol=2e-3), key
        assert np.all(p[key] >= s5.q_limits[:, 0]) and np.all(p[key] <= s5.q_limits[:, 1])
