"""Auto-checks for SBEG 461 Lab 1 (SymPy Lagrangian vs. modern_robotics).

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
Reference values: MR Eq. 8.6 (particle), MR Eq. 8.9-8.10 (2R point masses), MR Fig. 8.3 (M at two poses),
MR Proposition 8.1 (Mdot - 2C skew) and the modern_robotics functions MassMatrix, VelQuadraticForces, GravityForces.
"""
import csv
import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))

with open(HERE / "data" / "arms.json") as fh:
    ARMS = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}


def load_states():
    rows = []
    with open(HERE / "data" / "states.csv") as fh:
        for row in csv.reader(line for line in fh if not line.startswith("#")):
            name = row[0]
            n = len(ARMS[name]["L"])
            vals = [float(x) for x in row[1:]]
            rows.append((name, np.array(vals[:3][:n]), np.array(vals[3:][:n])))
    return rows


STATES = load_states()


@lru_cache(maxsize=None)
def model(name):
    return lab.numeric_model(ARMS[name])


def closed_form_2r(p, th, dth, ddth):
    """MR Eq. 8.9 generalised to links with CoM at r_i and inertia I_i (reduces to MR for r = L, I = 0)."""
    L1 = p["L"][0]
    m1, m2 = p["m"]
    r1, r2 = p["r"]
    I1, I2 = p["I"]
    g = p["g"]
    c2, s2 = np.cos(th[1]), np.sin(th[1])
    h = m2 * L1 * r2
    M = np.array([[I1 + I2 + m1 * r1**2 + m2 * (L1**2 + r2**2 + 2 * L1 * r2 * c2), I2 + m2 * (r2**2 + L1 * r2 * c2)],
                  [I2 + m2 * (r2**2 + L1 * r2 * c2), I2 + m2 * r2**2]])
    c = np.array([-h * s2 * (2 * dth[0] * dth[1] + dth[1] ** 2), h * s2 * dth[0] ** 2])
    gv = np.array([(m1 * r1 + m2 * L1) * g * np.cos(th[0]) + m2 * r2 * g * np.cos(th[0] + th[1]),
                   m2 * r2 * g * np.cos(th[0] + th[1])])
    return M, c, gv, M @ ddth + c + gv


# ---------------------------------------------------------------- checkpoint 1: Euler-Lagrange machinery
def test_make_symbols():
    th, dth, ddth = lab.make_symbols(3)
    assert len(th) == len(dth) == len(ddth) == 3
    assert len(set(th) | set(dth) | set(ddth)) == 9


def test_time_derivative_chain_rule():
    th, dth, ddth = lab.make_symbols(2)
    expr = sp.sin(th[0]) * dth[1] ** 2
    got = lab.time_derivative(expr, th, dth, ddth)
    want = sp.cos(th[0]) * dth[0] * dth[1] ** 2 + 2 * sp.sin(th[0]) * dth[1] * ddth[1]
    assert sp.simplify(got - want) == 0


def test_euler_lagrange_particle_mr_eq_8_6():
    x, dx, ddx = sp.symbols("x dx ddx", real=True)
    m, g = sp.symbols("m g", positive=True)
    f = lab.euler_lagrange(sp.Rational(1, 2) * m * dx**2 - m * g * x, [x], [dx], [ddx])
    assert sp.simplify(f[0] - (m * ddx + m * g)) == 0


def test_euler_lagrange_pendulum():
    th, dth, ddth = sp.symbols("th dth ddth", real=True)
    m, l, g = sp.symbols("m l g", positive=True)
    Lag = sp.Rational(1, 2) * m * l**2 * dth**2 - m * g * l * sp.sin(th)
    f = lab.euler_lagrange(Lag, [th], [dth], [ddth])
    assert sp.simplify(f[0] - (m * l**2 * ddth + m * g * l * sp.cos(th))) == 0


# ---------------------------------------------------------------- checkpoint 2: energies, M, g, Christoffel, C
def test_energy_mr_2r_point():
    p = ARMS["mr_2r_point"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    sub = {th[0]: 0.4, th[1]: -1.1, dth[0]: 0.7, dth[1]: -0.3}
    # MR p. 272: K1, K2, P1, P2 with L = m = 1
    c2 = np.cos(-1.1)
    K_mr = 0.5 * 0.7**2 + 0.5 * ((2 + 2 * c2) * 0.7**2 + 2 * (1 + c2) * 0.7 * -0.3 + 0.3**2)
    P_mr = 9.81 * np.sin(0.4) + 9.81 * (np.sin(0.4) + np.sin(0.4 - 1.1))
    assert float(K.subs(sub)) == pytest.approx(K_mr, abs=1e-12)
    assert float(P.subs(sub)) == pytest.approx(P_mr, abs=1e-12)


def test_mass_matrix_symbolic_mr_eq_8_10():
    p = ARMS["mr_2r_point"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    M = lab.mass_matrix(K, dth)
    want = sp.Matrix([[3 + 2 * sp.cos(th[1]), 1 + sp.cos(th[1])], [1 + sp.cos(th[1]), 1]])
    assert sp.simplify(M - want) == sp.zeros(2, 2)


def test_gravity_symbolic():
    p = ARMS["mr_2r_point"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    gv = lab.gravity_vector(P, th)
    want = sp.Matrix([2 * 9.81 * sp.cos(th[0]) + 9.81 * sp.cos(th[0] + th[1]), 9.81 * sp.cos(th[0] + th[1])])
    diff = (gv - want).subs({th[0]: 0.37, th[1]: 1.9})
    assert all(abs(float(d)) < 1e-12 for d in diff)


def test_christoffel_2r():
    p = ARMS["arm_exo_2r"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    G = lab.christoffel(lab.mass_matrix(K, dth), th)
    h = p["m"][1] * p["L"][0] * p["r"][1]
    t2 = 0.8
    val = lambda e: float(sp.sympify(e).subs({th[0]: 0.3, th[1]: t2}))
    want = {(0, 0, 0): 0, (0, 0, 1): -h * np.sin(t2), (0, 1, 0): -h * np.sin(t2), (0, 1, 1): -h * np.sin(t2),
            (1, 0, 0): h * np.sin(t2), (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 0}
    for (i, j, k), w in want.items():
        assert val(G[i][j][k]) == pytest.approx(w, abs=1e-12), (i, j, k)


def test_christoffel_symmetric_in_jk_3r():
    p = ARMS["eod_3r"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    M = lab.mass_matrix(K, dth)
    G = lab.christoffel(M, th)
    sub = {th[0]: 0.2, th[1]: -0.7, th[2]: 1.3}
    v = lambda e: float(sp.sympify(e).subs(sub))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert v(G[i][j][k] - G[i][k][j]) == pytest.approx(0, abs=1e-12)            # symmetric in j, k
                assert v(G[i][j][k] + G[j][i][k]) == pytest.approx(v(sp.diff(M[i, j], th[k])), abs=1e-12)


def test_coriolis_matrix_times_dtheta_mr_eq_8_10():
    p = ARMS["mr_2r_point"]
    K, P, th, dth, _ = lab.planar_chain_energy(p)
    C = lab.coriolis_matrix(lab.mass_matrix(K, dth), th, dth)
    c = C * sp.Matrix(dth)
    want = sp.Matrix([-sp.sin(th[1]) * (2 * dth[0] * dth[1] + dth[1] ** 2), sp.sin(th[1]) * dth[0] ** 2])
    assert sp.simplify(c - want) == sp.zeros(2, 1)


# ---------------------------------------------------------------- checkpoint 3: numeric model
@pytest.mark.parametrize("th2,M_fig", [(np.pi / 2, [[3, 1], [1, 1]]), (5 * np.pi / 6, [[1.27, 0.13], [0.13, 1.0]])])
def test_mass_matrix_mr_fig_8_3(th2, M_fig):
    M = model("mr_2r_point")["M"]([0.0, th2])
    assert np.allclose(M, M_fig, atol=0.006)   # MR Fig. 8.3 prints 2 decimals


@pytest.mark.parametrize("name", ["mr_2r_point", "arm_exo_2r"])
def test_numeric_model_vs_closed_form(name):
    p = ARMS[name]
    mdl = model(name)
    ddth = np.array([0.4, -1.3])
    for nm, th, dth in STATES:
        if nm != name:
            continue
        M, c, gv, tau = closed_form_2r(p, th, dth, ddth)
        assert np.allclose(mdl["M"](th), M, atol=1e-10)
        assert np.allclose(mdl["c"](th, dth), c, atol=1e-10)
        assert np.allclose(mdl["g"](th), gv, atol=1e-10)
        assert np.allclose(mdl["tau"](th, dth, ddth), tau, atol=1e-10)


@pytest.mark.parametrize("name,th,dth", [(s[0], s[1], s[2]) for s in STATES])
def test_properties_M_pd_and_skew(name, th, dth):
    mdl = model(name)
    M = mdl["M"](th)
    assert np.allclose(M, M.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(M)) > 0
    assert lab.skew_residual(mdl, th, dth) < 1e-10
    assert np.allclose(mdl["C"](th, dth) @ dth, mdl["c"](th, dth), atol=1e-10)


@pytest.mark.parametrize("name,th,dth", [(s[0], s[1], s[2]) for s in STATES])
def test_power_balance(name, th, dth):
    mdl = model(name)
    ddth = np.linspace(-1.0, 1.5, len(th))
    assert lab.power_residual(mdl, th, dth, ddth) < 1e-9
    assert mdl["K"](th, dth) == pytest.approx(0.5 * dth @ mdl["M"](th) @ dth, abs=1e-12)


# ---------------------------------------------------------------- checkpoint 4: modern_robotics model
@pytest.mark.parametrize("name", list(ARMS))
def test_mr_chain_structure(name):
    p = ARMS[name]
    n = len(p["L"])
    Mlist, Glist, Slist, gvec = lab.mr_chain(p)
    assert len(Mlist) == n + 1 and len(Glist) == n
    assert np.asarray(Slist).shape == (6, n)
    assert np.allclose(gvec, [0, -p["g"], 0])
    # home pose of the tip = product of the Mlist entries = Trans(sum L, 0, 0)
    T = np.eye(4)
    for Mi in Mlist:
        T = T @ Mi
    assert np.allclose(T[:3, 3], [sum(p["L"]), 0, 0])
    # FK at a random pose: tip x-y matches planar geometry (bridge to 361)
    th = np.array([0.3, -0.6, 0.9][:n])
    Tsb = mr.FKinSpace(T, Slist, th)
    phi = np.cumsum(th)
    assert np.allclose(Tsb[:2, 3], [np.sum(np.array(p["L"]) * np.cos(phi)), np.sum(np.array(p["L"]) * np.sin(phi))])


@pytest.mark.parametrize("name,th,dth", [(s[0], s[1], s[2]) for s in STATES])
def test_mr_terms_match_library(name, th, dth):
    p = ARMS[name]
    Mlist, Glist, Slist, gvec = lab.mr_chain(p)
    M, c, gv = lab.mr_terms(p, th, dth)
    assert np.allclose(M, mr.MassMatrix(th, Mlist, Glist, Slist), atol=1e-12)
    assert np.allclose(c, mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist), atol=1e-12)
    assert np.allclose(gv, mr.GravityForces(th, gvec, Mlist, Glist, Slist), atol=1e-12)


@pytest.mark.parametrize("name,th,dth", [(s[0], s[1], s[2]) for s in STATES])
def test_mass_matrix_from_jacobians(name, th, dth):
    assert np.allclose(lab.mass_matrix_from_jacobians(ARMS[name], th), model(name)["M"](th), atol=1e-10)


@pytest.mark.local_only  # needs files outside a single-lab repository; not graded
@pytest.mark.parametrize("name", list(ARMS))
def test_361_library_matches_modern_robotics(name):
    """Your sbeg361_kinematics fk_space / jacobian_space against mr.FKinSpace / mr.JacobianSpace (skipped when the
    lab runs on the modern_robotics fallback)."""
    if not str(getattr(lab, "KIN_BACKEND", "")).startswith("sbeg361_kinematics"):
        pytest.skip("sbeg361_kinematics not importable: modern_robotics fallback in use")
    p = ARMS[name]
    n = len(p["L"])
    joint_x = np.concatenate([[0.0], np.cumsum(p["L"])])
    Slist = np.array([[0, 0, 1, 0, -joint_x[i], 0] for i in range(n)], float).T
    Mtip = np.eye(4)
    Mtip[0, 3] = joint_x[n]
    for nm, th, _ in STATES:
        if nm == name:
            assert np.allclose(lab.fk_space(Mtip, Slist, th), mr.FKinSpace(Mtip, Slist, th), atol=1e-12)
            assert np.allclose(lab.jacobian_space(Slist, th), mr.JacobianSpace(Slist, th), atol=1e-12)


# ---------------------------------------------------------------- checkpoint 5: SymPy vs modern_robotics
@pytest.mark.parametrize("name", list(ARMS))
def test_compare_sympy_vs_modern_robotics(name):
    states = [(th, dth) for nm, th, dth in STATES if nm == name]
    err = lab.compare(ARMS[name], states)
    assert set(err) >= {"M", "c", "g", "M_jac"}
    assert max(err.values()) < 1e-9


@pytest.mark.parametrize("name,th,dth", [(s[0], s[1], s[2]) for s in STATES])
def test_inverse_dynamics_vs_modern_robotics(name, th, dth):
    p = ARMS[name]
    Mlist, Glist, Slist, gvec = lab.mr_chain(p)
    ddth = np.linspace(0.5, -0.8, len(th))
    tau_mr = mr.InverseDynamics(th, dth, ddth, gvec, np.zeros(6), Mlist, Glist, Slist)
    assert np.allclose(model(name)["tau"](th, dth, ddth), tau_mr, atol=1e-9)


# ---------------------------------------------------------------- independent reference values (Workshop Problem 4)
def test_eod_3r_gravity_and_mass_matrix_workshop_p4():
    """Hard-coded numbers from Workshop Problem 4b/4d (recomputed independently), so that a consistent but wrong
    pair of models (SymPy and Mlist/Glist with the same mistake) cannot pass."""
    mdl = model("eod_3r")
    th0 = np.zeros(3)
    assert np.allclose(mdl["g"](th0), [106.4385, 32.8635, 3.4335], atol=1e-4)
    assert np.allclose(mdl["M"](th0), [[9.7205, 3.6905, 0.4280], [3.6905, 1.6805, 0.2180], [0.4280, 0.2180, 0.0430]],
                       atol=1e-4)
    assert np.allclose(mdl["g"](np.deg2rad([45, -60, -30])), [82.880, 30.855, 2.428], atol=1e-3)
