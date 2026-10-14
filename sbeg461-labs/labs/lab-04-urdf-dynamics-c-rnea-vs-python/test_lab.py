"""Auto-checks for SBEG 461 Lab 4 (URDF dynamics, RNEA timing, constrained 2R).

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Optional C++ check: build cpp/ (see lab_handout.md) and set RNEA_CPP=/path/to/rnea_bench, or leave the
binary at cpp/build/rnea_bench. The C++ test is skipped when no binary is found.
Ground truth: modern_robotics (Lynch & Park), Pinocchio, and the numbers printed in MR Example 8.5.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

URDF = HERE / "arm6.urdf"
CSV = HERE / "states_arm6.csv"
G = np.array([0.0, 0.0, -9.81])
RNG = np.random.default_rng(461)
STATES = [RNG.uniform(-2.5, 2.5, (3, 6)) for _ in range(8)]

# MR Example 8.5 state: theta = (-pi/3, 2pi/3), thetadot = (1, 0), g = 0, L = m = 1
TH85 = np.array([-np.pi / 3, 2 * np.pi / 3])
DTH85 = np.array([1.0, 0.0])


@pytest.fixture(scope="module")
def arm6():
    return lab.urdf_to_mr(URDF)


def mr_2r_model(m1=1.0, m2=1.0, L1=1.0, L2=1.0):
    """The MR 2R arm (Fig. 8.1) as Mlist/Glist/Slist with point masses (zero rotational inertia)."""
    Mlist = [mr.RpToTrans(np.eye(3), [L1, 0, 0]), mr.RpToTrans(np.eye(3), [L2, 0, 0]), np.eye(4)]
    Glist = [np.diag([0, 0, 0, m1, m1, m1]), np.diag([0, 0, 0, m2, m2, m2])]
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    return Mlist, Glist, Slist


# ---------------------------------------------------------------- Part A: URDF -> MR parameters

def test_urdf_shapes(arm6):
    Mlist, Glist, Slist = arm6
    assert len(Mlist) == 7 and len(Glist) == 6
    assert np.asarray(Slist).shape == (6, 6)
    for Gi in Glist:
        assert np.allclose(Gi, Gi.T) and np.all(np.linalg.eigvalsh(Gi) > 0)


def test_urdf_forward_kinematics_matches_pinocchio(arm6):
    pin = pytest.importorskip("pinocchio")
    Mlist, _, Slist = arm6
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    fid = model.getFrameId("tool0")
    M = np.eye(4)
    for Mi in Mlist:
        M = M @ Mi
    for th, _, _ in STATES:
        pin.framesForwardKinematics(model, data, th)
        assert np.allclose(mr.FKinSpace(M, Slist, th), data.oMf[fid].homogeneous, atol=1e-9)


def test_urdf_dynamics_match_pinocchio(arm6):
    """The parsed (Mlist, Glist, Slist) must give the same torques as Pinocchio on the same URDF."""
    pin = pytest.importorskip("pinocchio")
    Mlist, Glist, Slist = arm6
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    for th, dth, ddth in STATES:
        tau_pin = pin.rnea(model, data, th, dth, ddth)
        tau_mr = mr.InverseDynamics(th, dth, ddth, G, np.zeros(6), Mlist, Glist, Slist)
        assert np.allclose(tau_mr, tau_pin, atol=1e-8)


# ---------------------------------------------------------------- Part B: RNEA and timing

def test_rnea_matches_modern_robotics(arm6):
    Mlist, Glist, Slist = arm6
    for th, dth, ddth in STATES:
        Ftip = RNG.normal(size=6)
        ref = mr.InverseDynamics(th, dth, ddth, G, Ftip, Mlist, Glist, Slist)
        assert np.allclose(lab.rnea(th, dth, ddth, G, Ftip, Mlist, Glist, Slist), ref, atol=1e-9)


def test_rnea_2r_matches_closed_form():
    Mlist, Glist, Slist = mr_2r_model(m1=2.0, m2=1.5, L1=0.8, L2=0.6)
    for th, dth, ddth in STATES:
        th, dth, ddth = th[:2], dth[:2], ddth[:2]
        g3 = np.array([0, -9.81, 0])
        M, h = lab.twor_dynamics(th, dth, m1=2.0, m2=1.5, L1=0.8, L2=0.6, g=9.81)
        tau = lab.rnea(th, dth, ddth, g3, np.zeros(6), Mlist, Glist, Slist)
        assert np.allclose(tau, M @ ddth + h, atol=1e-9)


def test_time_per_call_is_positive_and_scales():
    def slow(x):
        s = 0.0
        for k in range(2000):
            s += x * k
        return s
    def fast(x):
        return x
    args = [(1.0,)] * 20
    t_slow = lab.time_per_call(slow, args)
    t_fast = lab.time_per_call(fast, args)
    assert t_slow > 0 and t_fast > 0
    assert t_slow > 5 * t_fast


@pytest.mark.local_only  # needs a locally built cpp/build/rnea_bench; not graded
def test_cpp_rnea_agrees_with_python(arm6, tmp_path):
    exe = os.environ.get("RNEA_CPP", str(HERE / "cpp" / "build" / "rnea_bench"))
    if not Path(exe).exists():
        pytest.skip("C++ benchmark not built (set RNEA_CPP or build cpp/ into cpp/build)")
    Mlist, Glist, Slist = arm6
    X = lab.load_states(CSV)[:50]
    mp, sp, tp = tmp_path / "model.txt", tmp_path / "states.txt", tmp_path / "tau.txt"
    lab.export_for_cpp(mp, sp, Mlist, Glist, Slist, X)
    res = subprocess.run([exe, str(mp), str(sp), str(tp), "5"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    tau_cpp = np.loadtxt(tp)
    tau_ref = np.array([mr.InverseDynamics(x[:6], x[6:12], x[12:], G, np.zeros(6), Mlist, Glist, Slist) for x in X])
    assert np.allclose(tau_cpp, tau_ref, atol=1e-9)
    us = [float(line.split()[1]) for line in res.stdout.splitlines() if line.startswith("us_per_call")]
    assert us and us[0] > 0


# ---------------------------------------------------------------- Part C: task-space dynamics

def test_task_space_dynamics_consistent_with_joint_space():
    """F = Lambda Vdot + eta must equal J^-T tau for tau = M thetaddot + h (MR Eqs. 8.82-8.91)."""
    for th, dth, ddth in STATES:
        th, dth, ddth = th[:2], dth[:2], ddth[:2]
        if abs(np.sin(th[1])) < 0.2:
            continue
        M, h = lab.twor_dynamics(th, dth, m1=1.0, m2=1.0, g=9.81)
        J, Jd = lab.twor_jacobian(th, dth)
        Lam, eta = lab.task_space_dynamics(M, h, J, Jd, dth)
        assert np.allclose(Lam, Lam.T, atol=1e-12)
        tau = M @ ddth + h
        Vdot = J @ ddth + Jd @ dth
        assert np.allclose(Lam @ Vdot + eta, np.linalg.solve(J.T, tau), atol=1e-9)


def test_task_space_mass_matrix_value():
    """MR 2R at theta = (0, pi/2): Lambda = diag(1, 2)."""
    M, h = lab.twor_dynamics([0, np.pi / 2], [0, 0])
    J, Jd = lab.twor_jacobian([0, np.pi / 2], [0, 0])
    Lam, eta = lab.task_space_dynamics(M, h, J, Jd, [0, 0])
    assert np.allclose(Lam, np.diag([1.0, 2.0]), atol=1e-12)
    assert np.allclose(eta, 0, atol=1e-12)


# ---------------------------------------------------------------- Part D: constrained 2R

def test_twor_dynamics_matches_modern_robotics():
    Mlist, Glist, Slist = mr_2r_model()
    for th, dth, _ in STATES:
        th, dth = th[:2], dth[:2]
        M, h = lab.twor_dynamics(th, dth, g=9.81)
        assert np.allclose(M, mr.MassMatrix(th, Mlist, Glist, Slist), atol=1e-12)
        h_ref = mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist) + mr.GravityForces(th, [0, -9.81, 0], Mlist, Glist, Slist)
        assert np.allclose(h, h_ref, atol=1e-12)


def test_channel_constraint_is_gradient_of_x():
    eps = 1e-6
    for th, dth, _ in STATES:
        th, dth = th[:2], dth[:2]
        A, Ad = lab.channel_constraint(th, dth)
        x = lambda q: np.cos(q[0]) + np.cos(q[0] + q[1])
        grad = [(x(th + eps * e) - x(th - eps * e)) / (2 * eps) for e in np.eye(2)]
        assert np.asarray(A).shape == (1, 2)
        assert np.allclose(A[0], grad, atol=1e-8)
        A_plus, _ = lab.channel_constraint(th + eps * dth, dth)
        A_minus, _ = lab.channel_constraint(th - eps * dth, dth)
        assert np.allclose(Ad, (A_plus - A_minus) / (2 * eps), atol=1e-7)


def test_mr_example_8_5_forward():
    """MR Example 8.5: lambda = 0.289 tau1 - 1.155 tau2 - 0.167, thdd1 = 0.5 tau1 + 0.289, thdd2 = -1.155."""
    M, h = lab.twor_dynamics(TH85, DTH85)
    A, Ad = lab.channel_constraint(TH85, DTH85)
    assert np.allclose(A, [[0, -np.sqrt(3) / 2]], atol=1e-12)
    assert np.allclose(Ad, [[-1, -0.5]], atol=1e-12)
    for tau in ([0, 0], [1, 0], [0, 1], [2.5, -1.0]):
        ddth, lam = lab.constrained_forward_dynamics(M, h, A, Ad, DTH85, tau)
        t1, t2 = tau
        assert np.allclose(lam, [0.2887 * t1 - 1.1547 * t2 - 0.1667], atol=2e-4)
        assert np.allclose(ddth, [0.5 * t1 + 0.2887, -1.1547], atol=2e-4)
        assert abs((A @ ddth + Ad @ DTH85)[0]) < 1e-12          # MR Eq. 8.95


def test_projector_and_inverse_dynamics():
    M, h = lab.twor_dynamics(TH85, DTH85)
    A, Ad = lab.channel_constraint(TH85, DTH85)
    P = lab.constraint_projector(M, A)
    assert np.allclose(P, [[1, 0], [0.25, 0]], atol=1e-12)          # MR p. 304
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.linalg.matrix_rank(P, tol=1e-9) == 1
    assert np.allclose(P @ np.asarray(A).T, 0, atol=1e-12)
    # inverse dynamics with thdd = (a, -1.155), lambda = f: tau1 = 2a - 0.578, tau2 = 0.5a - 0.866 f - 0.289
    for a, f in ((0.0, 0.0), (1.0, 2.0), (-0.5, 1.0)):
        ddth = np.array([a, -2 / np.sqrt(3)])
        tau = lab.constrained_inverse_dynamics(M, h, A, ddth, [f])
        assert np.allclose(tau, [2 * a - 0.5774, 0.5 * a - 0.8660 * f - 0.2887], atol=2e-4)
        ddth2, lam2 = lab.constrained_forward_dynamics(M, h, A, Ad, DTH85, tau)
        assert np.allclose(ddth2, ddth, atol=1e-10) and np.allclose(lam2, [f], atol=1e-10)


def test_simulation_conserves_energy_and_constraint():
    """tau = 0, g = 0, workless constraint: kinetic energy is constant and the tip stays at x = 1."""
    out = lab.simulate_channel_2r(TH85, DTH85, lambda t, q, dq: np.zeros(2), T=2.0, dt=1e-3, alpha=10.0, beta=10.0)
    th, dth, e = out["theta"], out["dtheta"], out["e"]
    assert th.shape == (2001, 2) and out["lam"].shape == (2001,)
    KE = np.array([0.5 * dq @ lab.twor_dynamics(q, dq)[0] @ dq for q, dq in zip(th, dth)])
    assert np.max(np.abs(KE - KE[0])) < 1e-4
    assert np.max(np.abs(e)) < 1e-6
    x = np.cos(th[:, 0]) + np.cos(th[:, 0] + th[:, 1])
    assert np.max(np.abs(x - 1.0)) < 1e-6
    assert abs(out["lam"][0] - (-1 / 6)) < 1e-3                    # MR: lambda = -0.167 at tau = 0


def test_baumgarte_removes_drift():
    """Start with a small constraint violation; Baumgarte (Featherstone Sec. 8.3) drives it to zero,
    the unstabilised integration keeps it."""
    th0 = TH85 + np.array([0.0, 0.01])
    dth0 = DTH85
    zero = lambda t, q, dq: np.zeros(2)
    plain = lab.simulate_channel_2r(th0, dth0, zero, T=1.0, dt=1e-3)
    stab = lab.simulate_channel_2r(th0, dth0, zero, T=1.0, dt=1e-3, alpha=20.0, beta=20.0)
    assert abs(plain["e"][-1]) > 0.5 * abs(plain["e"][0])
    assert abs(stab["e"][-1]) < 1e-3 * abs(stab["e"][0])
