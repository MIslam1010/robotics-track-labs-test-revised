"""Auto-checks for the SBEG 461 enrichment lab "Dynamic parameter identification from hardware logs".

    pytest -q test_lab.py                        # in your Lab code folder: tests starter.py (your work)
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test that calls your code fails with
                                                 # NotImplementedError; the data/infrastructure tests pass
                                                 # (student copy: 15 failed, 3 passed, 4 errors in the shared
                                                 # fixture of the log-based tests)

LAB_IMPL is the name of the module to test, imported from this folder (the course team's copy defaults to its
reference module, your Lab code folder to starter).
Ground truth: modern_robotics (InverseDynamics with general spatial inertias, MR Eq. 8.43; ad), the rigid-body
equation G dV - [ad_V]^T G V (MR Eq. 8.40), finite differences, synthetic data with known parameters, and the
weighed mass of the EOD training package. The hidden "true" parameters of the simulated robots are NOT used:
on the logs your model is judged, as on real hardware, by its prediction of held-out data.
"""
import importlib
import os
import py_compile
import sys
from pathlib import Path

import modern_robotics as mr
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LAB_IMPL", "starter")
lab = importlib.import_module(os.environ["LAB_IMPL"])
import hwlog  # noqa: E402
import robots  # noqa: E402

META = lab.load_meta()
RNG = np.random.default_rng(4610)


def random_inertia(rng, m=1.0, L=0.3):
    """A valid spatial inertia written in a frame that is NOT at the CoM and not aligned with its principal axes."""
    R = mr.MatrixExp3(mr.VecToso3(rng.normal(size=3)))
    Ic = R @ np.diag(m * L**2 * rng.uniform(0.05, 0.3, 3)) @ R.T
    Gb = np.zeros((6, 6))
    Gb[:3, :3], Gb[3:, 3:] = Ic, m * np.eye(3)
    Ad = mr.Adjoint(robots.trans(rng.normal(0, 0.05, 3)))
    return Ad.T @ Gb @ Ad                                         # MR Eq. 8.42


def load(name):
    key = META["logs"][name]["robot"]
    return lab.process_log(hwlog.load_log(HERE / "data" / name), META["robots"][key]["torque_constant"])


# ============================================================================ data and infrastructure (provided code)
def test_meta_and_logs():
    assert set(META["robots"]) == {"course_arm", "eod_3r"}
    for name, info in META["logs"].items():
        log = hwlog.load_log(HERE / "data" / name)
        n = robots.get(info["robot"])["n"]
        assert log["cmd"].shape[1] == n and log["cur"].shape[1] == n
        assert np.all(np.diff(log["t"]) > 0)                      # sorted, duplicates dropped
        assert 400 < 1 / np.median(np.diff(log["t"])) < 600       # about 500 Hz
    assert META["package_mass_weighed"] == 1.20


def test_robot_models_are_the_course_models():
    ca, eod = robots.course_arm(), robots.eod_3r()
    assert np.isclose(sum(G[3, 3] for G in ca["Glist"]), 0.67)
    g0 = mr.GravityForces(np.zeros(3), eod["g"], eod["Mlist"], eod["Glist"], eod["Slist"])
    assert np.allclose(g0, [106.44, 32.86, 3.43], atol=0.01)       # value quoted in the course robot models
    assert np.allclose(eod["tau_max"], [155, 70, 15])


def test_starter_compiles():
    py_compile.compile(str(HERE / "starter.py"), doraise=True)


# ============================================================================ Part A: linear-in-parameters form
def test_inertial_parameters_round_trip():
    for G in robots.course_arm()["Glist"] + [random_inertia(RNG) for _ in range(3)]:
        p = lab.inertial_parameters(G)
        assert p.shape == (10,)
        assert np.allclose(lab.spatial_inertia(p), G, atol=1e-14)
    p = lab.inertial_parameters(random_inertia(RNG, m=2.0))
    assert np.isclose(p[0], 2.0)


def test_wrench_regressor_is_the_rigid_body_equation():
    for _ in range(5):
        G = random_inertia(RNG, m=RNG.uniform(0.5, 3))
        V, dV = RNG.normal(size=6), RNG.normal(size=6)
        Phi = lab.wrench_regressor(V, dV)
        assert Phi.shape == (6, 10)
        assert np.allclose(Phi @ lab.inertial_parameters(G), G @ dV - mr.ad(V).T @ G @ V, atol=1e-12)


@pytest.mark.parametrize("key", ["course_arm", "eod_3r"])
def test_regressor_matches_modern_robotics(key):
    r = robots.get(key)
    n = r["n"]
    Gl = [random_inertia(RNG, m=G[3, 3]) for G in r["Glist"]]
    Ia, bv, bc = RNG.uniform(0, 0.1, n), RNG.uniform(0, 0.1, n), RNG.uniform(0, 0.1, n)
    pi = np.concatenate([lab.inertial_parameters(G) for G in Gl] + [Ia, bv, bc])
    for _ in range(4):
        q, dq, ddq = RNG.uniform(-2, 2, (3, n))
        Y = lab.regressor(r, q, dq, ddq)
        assert Y.shape == (n, 13 * n)
        ref = mr.InverseDynamics(q, dq, ddq, r["g"], np.zeros(6), r["Mlist"], Gl, r["Slist"])
        assert np.allclose(Y @ pi, ref + Ia * ddq + bv * dq + bc * np.sign(dq), atol=1e-10)


# ============================================================================ Part B: base parameters
@pytest.mark.parametrize("key,rank", [("eod_3r", 17), ("course_arm", 52)])
def test_base_parameters(key, rank):
    r = robots.get(key)
    n = r["n"]
    b = lab.base_parameters(r)
    assert b["rank"] == rank and b["E"].shape == (rank, 13 * n) and len(b["idx_base"]) == rank
    assert len(b["idx_dep"]) == 13 * n - rank
    pi = np.concatenate([lab.inertial_parameters(random_inertia(RNG)) for _ in range(n)] + [RNG.uniform(0, 1, 3 * n)])
    Q, dQ, ddQ = lab.random_states(r, 20, seed=99)                  # states NOT used to find the base parameters
    W = lab.stack_regressor(r, Q, dQ, ddQ)
    assert np.allclose(W @ pi, W[:, b["idx_base"]] @ (b["E"] @ pi), atol=1e-9)
    assert np.linalg.matrix_rank(W[:, b["idx_base"]]) == rank
    names = lab.param_names(n)
    # joint-1 rotor inertia adds to a link-1 inertia: its column equals another one, and only ONE of the identical
    # columns can be a base parameter (which one, if any, LAPACK's pivoting keeps depends on the platform's BLAS)
    k_ia = names.index("Ia1")
    dup = [k for k in range(W.shape[1]) if k != k_ia and np.allclose(W[:, k], W[:, k_ia], atol=1e-9)]
    assert dup and len({k_ia, *dup} & set(b["idx_base"])) <= 1
    assert all(f"bc{j + 1}" in b["names"] for j in range(n))       # friction always identifiable


# ============================================================================ Part C: logs
def test_process_log_on_a_clean_sine(tmp_path):
    fs, k = 500.0, np.array([2.0, 0.5])
    t = np.arange(0, 12, 1 / fs)
    w = 2 * np.pi * 0.4
    q = np.column_stack([0.5 * np.sin(w * t), 0.3 * np.cos(w * t)])
    cur = np.column_stack([np.cos(w * t), np.sin(w * t)])
    hwlog.write_log(tmp_path / "s.csv", t, q, q, cur=cur)
    d = lab.process_log(hwlog.load_log(tmp_path / "s.csv"), k, fs=fs, cutoff=4.0, decimate=5, trim=1.0)
    assert d["t"][0] >= 1.0 - 1e-9 and d["t"][-1] <= t[-1] - 1.0 + 1e-9
    assert np.allclose(np.diff(d["t"]), 5 / fs)
    tt = d["t"]
    assert np.allclose(d["q"][:, 0], 0.5 * np.sin(w * tt), atol=1e-3)
    assert np.allclose(d["dq"][:, 0], 0.5 * w * np.cos(w * tt), atol=1e-2)
    assert np.allclose(d["ddq"][:, 1], -0.3 * w**2 * np.cos(w * tt), atol=2e-2)
    assert np.allclose(d["tau"], k * np.column_stack([np.cos(w * tt), np.sin(w * tt)]), atol=1e-2)


def test_moving_rows():
    dq = np.array([[0.0, 0.5], [-0.2, 0.05]])
    assert lab.moving_rows(dq, 0.1).tolist() == [False, True, True, False]


# ============================================================================ Part D: estimation
def test_ols_on_synthetic_data():
    Y = RNG.normal(size=(400, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    noise = 0.01 * RNG.normal(size=400)
    out = lab.ols(Y, Y @ beta + noise)
    ref = np.linalg.lstsq(Y, Y @ beta + noise, rcond=None)[0]
    assert np.allclose(out["beta"], ref)
    s2 = np.sum((Y @ beta + noise - Y @ ref) ** 2) / 396
    assert np.isclose(out["sigma2"], s2)
    assert np.allclose(out["cov"], s2 * np.linalg.inv(Y.T @ Y))
    assert np.allclose(out["rel_std"], 100 * np.sqrt(np.diag(out["cov"])) / np.abs(ref))
    assert np.isclose(out["cond"], np.linalg.cond(Y))


def test_wls_finds_the_noise_levels():
    N, n = 3000, 2
    Y = RNG.normal(size=(N * n, 3))
    joint = lab.joint_index(N, n)
    sd = np.array([0.01, 0.5])
    tau = Y @ np.array([1.0, 2.0, 3.0]) + sd[joint] * RNG.normal(size=N * n)
    out = lab.wls(Y, tau, joint, n)
    assert np.isclose(out["sigma"][1], 0.5, rtol=0.05) and out["sigma"][0] < 0.05   # the OLS residual of the
    # quiet joint also contains the OLS estimation error caused by the noisy joint: why weighting is needed
    assert np.all(np.abs(out["beta"] - [1, 2, 3]) < 4 * np.sqrt(np.diag(out["cov"])))
    assert np.sqrt(np.diag(out["cov"]))[0] < np.sqrt(np.diag(lab.ols(Y, tau)["cov"]))[0]   # weighting helps


def test_validate():
    Y = np.eye(4)
    v = lab.validate(Y, np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 2.0, 4.0]), np.array([0, 1, 0, 1]), 2)
    assert np.allclose(v["rms_err"], [np.sqrt(0.5), 0.0])
    assert np.allclose(v["nrms"], [np.sqrt(0.5) / np.sqrt(2.5), 0.0])


@pytest.fixture(scope="module")
def identified():
    out = {}
    for key in ("eod_3r", "course_arm"):
        r = robots.get(key)
        base = lab.base_parameters(r)
        Y, tau, j = lab.build_system(r, load(f"{key}_excite.csv"), base["idx_base"])
        Yv, tv, jv = lab.build_system(r, load(f"{key}_validate.csv"), base["idx_base"])
        fit = lab.wls(Y, tau, j, r["n"])
        out[key] = dict(robot=r, base=base, fit=fit, val=lab.validate(Yv, fit["beta"], tv, jv, r["n"]),
                        nominal=lab.validate(Yv, base["E"] @ lab.nominal_parameters(r), tv, jv, r["n"]))
    return out


def test_identified_eod_arm_predicts_held_out_torque(identified):
    d = identified["eod_3r"]
    assert np.all(d["val"]["nrms"] < 0.05)                         # held-out data, within 5 % per joint
    assert np.all(d["val"]["nrms"] < 0.5 * d["nominal"]["nrms"])   # much better than the CAD model
    names, beta = d["base"]["names"], d["fit"]["beta"]
    assert all(beta[names.index(f"bc{j}")] > 0 and beta[names.index(f"bv{j}")] > 0 for j in (1, 2, 3))
    assert np.all(d["fit"]["rel_std"][[names.index(f"bc{j}") for j in (1, 2, 3)]] < 2.0)   # well identified


def test_identified_course_arm_predicts_held_out_torque(identified):
    d = identified["course_arm"]
    assert np.all(d["val"]["nrms"][[1, 2]] < 0.05)                 # shoulder and elbow
    assert np.all(d["val"]["nrms"] < 0.2)                          # every joint, incl. the noisy wrist
    assert np.all(d["val"]["nrms"] < 0.5 * d["nominal"]["nrms"])
    assert 200 < d["fit"]["cond"] < 5000                           # weighted, base regressor


# ============================================================================ Part E: excitation
def test_fourier_trajectory_derivatives_and_period():
    wf = 2 * np.pi * 0.1
    a, b = RNG.normal(size=(2, 4)), RNG.normal(size=(2, 4))
    t = np.linspace(0, 10, 2001)
    q, dq, ddq = lab.fourier_trajectory(t, [0.1, -0.2], a, b, wf)
    assert q.shape == (2001, 2)
    h = t[1] - t[0]
    assert np.allclose(np.gradient(q, h, axis=0)[2:-2], dq[2:-2], atol=1e-4)
    assert np.allclose(np.gradient(dq, h, axis=0)[2:-2], ddq[2:-2], atol=1e-4)
    assert np.allclose(q[0], q[-1]) and np.allclose(dq[0], dq[-1])          # one period = 10 s
    assert np.allclose(q.mean(axis=0)[None], [0.1, -0.2], atol=0.02)        # centred on q0
    assert np.allclose(dq[0], a.sum(axis=1))


def test_optimised_excitation_beats_a_single_sine():
    r = robots.eod_3r()
    base = lab.base_parameters(r)
    ex = META["robots"]["eod_3r"]["excitation"]
    wf = 2 * np.pi * ex["f_f"]
    t = np.linspace(0, 1 / ex["f_f"], 201)[:-1]
    c_opt = lab.excitation_cost(r, base, *lab.fourier_trajectory(t, ex["q0"], ex["a"], ex["b"], wf), every=2)
    a1 = np.zeros((3, 5))
    a1[:, 0] = 0.4                                                  # every joint the same single sine
    c_sine = lab.excitation_cost(r, base, *lab.fourier_trajectory(t, ex["q0"], a1, np.zeros((3, 5)), wf), every=2)
    assert c_opt < 20 and c_sine > 10 * c_opt


# ============================================================================ Part F: physical consistency
def test_pseudo_inertia_and_consistency():
    p, m = np.array([0.1, 0.0, 0.0]), 2.0                          # point mass
    S = m * (p @ p * np.eye(3) - np.outer(p, p))
    pi = np.array([m, *(m * p), S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2], S[2, 2]])
    J = lab.pseudo_inertia(pi)
    assert np.allclose(J, m * np.outer(np.r_[p, 1], np.r_[p, 1]))  # rank 1: a point is the limit case
    r = robots.course_arm()
    nom = lab.nominal_parameters(r, armature=np.full(6, 1e-3), b_viscous=np.full(6, 0.01), b_coulomb=np.full(6, 0.02))
    assert lab.is_consistent(nom, 6).all()
    bad = nom.copy()
    bad[4:10] = [0.001, 0, 0, 0.001, 0, 0.003]                     # Ixx + Iyy < Izz: no such body
    bad[1:4] = 0
    assert lab.is_consistent(bad, 6).tolist() == [False] + [True] * 5
    neg = nom.copy()
    neg[13 * 6 - 1] = -0.01                                        # negative Coulomb friction on joint 6
    assert not lab.is_consistent(neg, 6)[5]


def test_log_cholesky_parameterisation():
    for _ in range(5):
        pi = lab.log_cholesky_to_pi(RNG.normal(size=10))
        assert np.linalg.eigvalsh(lab.pseudo_inertia(pi)).min() > 0   # always consistent
        assert np.allclose(lab.log_cholesky_to_pi(lab.pi_to_log_cholesky(pi)), pi)


def test_consistent_fit_eod(identified):
    d = identified["eod_3r"]
    r, n = d["robot"], 3
    Y, tau, j = lab.build_system(r, load("eod_3r_excite.csv"))
    prior = lab.nominal_parameters(r, armature=[1.0, 0.5, 0.05], b_viscous=[3, 2, 0.3], b_coulomb=[3, 2, 0.3])
    cf = lab.consistent_fit(Y, tau, j, n, prior, lam=1e-3, sigma=d["fit"]["sigma"])
    assert lab.is_consistent(cf["pi"], n).all()
    Yv, tv, jv = lab.build_system(r, load("eod_3r_validate.csv"))
    v = lab.validate(Yv, cf["pi"], tv, jv, n)
    assert np.all(v["nrms"] < d["val"]["nrms"] * 1.2 + 0.005)      # consistency costs (almost) nothing here


# ============================================================================ Part G: payload
def test_payload_mass_of_the_training_package(identified):
    d = identified["eod_3r"]
    r = d["robot"]
    E = d["base"]["E"]
    model = np.linalg.pinv(E) @ d["fit"]["beta"]                   # any pi with E pi = beta predicts the same torques
    est = lab.payload_estimate(r, load("eod_3r_package.csv"), model)
    assert abs(est["mass"] - META["package_mass_weighed"]) < 0.05 * META["package_mass_weighed"]
    assert 0.0 < est["com"][0] < 0.08                              # package CoM ahead of the link-3 frame
