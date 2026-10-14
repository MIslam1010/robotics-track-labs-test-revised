"""Generate tests/mrcpp_cases.json: inputs and outputs of every public modern_robotics function,
computed with the official Python library (modern_robotics 1.1.1), so that tests/test_mrcpp.cpp can check the C++
port (include/sbeg/mrcpp.hpp) against the same numbers.

Run (any folder):  python make_mrcpp_cases.py        -> writes mrcpp_cases.json next to this file.
Random inputs use numpy.random.default_rng(SEED); re-running reproduces the file byte for byte.

JSON layout: {"_meta": {...}, "<snake_case_function>": [{"in": {...}, "out": ...}, ...], ...}
vectors are flat lists, matrices lists of rows, lists of matrices lists of those.
"""
import json
import warnings
import os

import matplotlib

matplotlib.use("Agg")  # SimulateControl calls plt.show(); keep it non-interactive
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import modern_robotics as mr  # noqa: E402
from modern_robotics import core  # noqa: E402

SEED = 20261013
warnings.filterwarnings("ignore", message=".*non-interactive.*")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mrcpp_cases.json")


def j(x):
    """numpy / python value -> JSON-able (vectors flat, matrices list of rows)."""
    if isinstance(x, (list, tuple)):
        return [j(e) for e in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    a = np.asarray(x, dtype=float)
    return a.tolist()


def rand_unit(rng):
    w = rng.normal(size=3)
    return w / np.linalg.norm(w)


def rand_rot(rng):
    return mr.MatrixExp3(mr.VecToso3(rand_unit(rng) * rng.uniform(0.1, 3.0)))


def rand_T(rng):
    return mr.RpToTrans(rand_rot(rng), rng.uniform(-1, 1, 3))


def rand_screws(rng, n):
    """6 x n screw-axis list: revolute axes through random points, one prismatic axis if n >= 3."""
    cols = []
    for i in range(n):
        if n >= 3 and i == n - 1:
            cols.append(np.r_[0.0, 0.0, 0.0, rand_unit(rng)])
        else:
            cols.append(mr.ScrewToAxis(rng.uniform(-0.5, 0.5, 3), rand_unit(rng), 0.0))
    return np.array(cols).T


def rand_inertia(rng):
    A = rng.normal(size=(3, 3))
    I = A @ A.T + 0.05 * np.eye(3)
    m = rng.uniform(0.5, 3.0)
    G = np.zeros((6, 6))
    G[:3, :3] = I
    G[3:, 3:] = m * np.eye(3)
    return G


def rand_chain(rng, n):
    Mlist = [rand_T(rng) for _ in range(n + 1)]
    Glist = [rand_inertia(rng) for _ in range(n)]
    Slist = rand_screws(rng, n)
    return Mlist, Glist, Slist


def mr_example_chain():
    """The 3-link example of the modern_robotics docstrings (InverseDynamics, MassMatrix, ...)."""
    M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
    M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
    M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
    M34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.14225], [0, 0, 0, 1]])
    G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
    G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
    G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
    Slist = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, -0.089, 0, 0], [0, 1, 0, -0.089, 0, 0.425]]).T
    return [M01, M12, M23, M34], [G1, G2, G3], Slist.astype(float)


def chains(rng):
    yield mr_example_chain()
    for n in (2, 3, 4):
        yield rand_chain(rng, n)


def main():
    rng = np.random.default_rng(SEED)
    C = {"_meta": {"generator": "make_mrcpp_cases.py", "seed": SEED, "modern_robotics": mr.__version__,
                   "numpy": np.__version__}}

    def add(name, inp, out):
        C.setdefault(name, []).append({"in": inp, "out": out})

    # ---------------- helpers
    for z in (0.0, 5e-7, -9.99e-7, 1e-6, 2e-6, -0.3):
        add("near_zero", {"z": z}, bool(core.NearZero(z)))
    for _ in range(4):
        V = rng.normal(size=int(rng.integers(2, 7)))
        add("normalize", {"V": j(V)}, j(mr.Normalize(V)))
    for shape in ((6, 3), (6, 7), (3, 3), (4, 6)):
        A = rng.normal(size=shape)
        add("pinv", {"A": j(A)}, j(np.linalg.pinv(A)))
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # rank 1
    add("pinv", {"A": j(A)}, j(np.linalg.pinv(A)))

    # ---------------- chapter 3
    for _ in range(5):
        R = rand_rot(rng) + rng.normal(scale=0.2, size=(3, 3))
        add("rot_inv", {"R": j(R)}, j(mr.RotInv(R)))
        w = rng.normal(size=3)
        add("vec_to_so3", {"omg": j(w)}, j(mr.VecToso3(w)))
        add("so3_to_vec", {"so3mat": j(mr.VecToso3(w))}, j(mr.so3ToVec(mr.VecToso3(w))))
        ax, th = mr.AxisAng3(w)
        add("axis_ang3", {"expc3": j(w)}, {"omghat": j(ax), "theta": float(th)})
    so3_inputs = [np.zeros(3), np.array([0.0, 0.0, 3e-7]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, np.pi])]
    so3_inputs += [rng.normal(size=3) * 1.5 for _ in range(5)]
    for w in so3_inputs:
        add("matrix_exp3", {"so3mat": j(mr.VecToso3(w))}, j(mr.MatrixExp3(mr.VecToso3(w))))
    log3_inputs = [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]),
                   np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float),
                   mr.MatrixExp3(mr.VecToso3(np.array([1.0, 2.0, 2.0]) / 3 * np.pi))]
    log3_inputs += [rand_rot(rng) for _ in range(6)]
    for R in log3_inputs:
        add("matrix_log3", {"R": j(R)}, j(mr.MatrixLog3(R)))
    for _ in range(4):
        R, p = rand_rot(rng), rng.normal(size=3)
        T = mr.RpToTrans(R, p)
        add("rp_to_trans", {"R": j(R), "p": j(p)}, j(T))
        Rr, pr = mr.TransToRp(T)
        add("trans_to_rp", {"T": j(T)}, {"R": j(Rr), "p": j(pr)})
        add("trans_inv", {"T": j(T)}, j(mr.TransInv(T)))
        add("adjoint", {"T": j(T)}, j(mr.Adjoint(T)))
        V = rng.normal(size=6)
        add("vec_to_se3", {"V": j(V)}, j(mr.VecTose3(V)))
        add("se3_to_vec", {"se3mat": j(mr.VecTose3(V))}, j(mr.se3ToVec(mr.VecTose3(V))))
        add("ad", {"V": j(V)}, j(mr.ad(V)))
        q, s, h = rng.normal(size=3), rand_unit(rng), float(rng.uniform(-1, 1))
        add("screw_to_axis", {"q": j(q), "s": j(s), "h": h}, j(mr.ScrewToAxis(q, s, h)))
    for V in [np.array([0.0, 0, 0, 1, 2, 3]), np.array([1.0, 0, 0, 1, 2, 3])] + [rng.normal(size=6) for _ in range(3)]:
        S, th = mr.AxisAng6(V)
        add("axis_ang6", {"expc6": j(V)}, {"S": j(S), "theta": float(th)})
    exp6_inputs = [np.array([0.0, 0, 0, 1, 2, 3]), np.array([0, 0, 1e-7, 0.5, 0, 0]),
                   np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 1, 2, 3]) * 2.0]
    exp6_inputs += [rng.normal(size=6) for _ in range(5)]
    for V in exp6_inputs:
        add("matrix_exp6", {"se3mat": j(mr.VecTose3(V))}, j(mr.MatrixExp6(mr.VecTose3(V))))
    log6_inputs = [np.eye(4), mr.RpToTrans(np.eye(3), np.array([1.0, -2.0, 0.5])),
                   mr.RpToTrans(np.diag([1.0, -1.0, -1.0]), np.array([0.3, 0.1, 0.0])),
                   np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]], float)]
    log6_inputs += [rand_T(rng) for _ in range(5)]
    for T in log6_inputs:
        add("matrix_log6", {"T": j(T)}, j(mr.MatrixLog6(T)))
    proj_inputs = [np.array([[0.675, 0.150, 0.720], [0.370, 0.771, -0.511], [-0.630, 0.619, 0.472]]),
                   np.diag([1.0, 1.0, -1.0]), np.diag([2.0, 0.5, -0.3])]
    proj_inputs += [rand_rot(rng) + rng.normal(scale=0.1, size=(3, 3)) for _ in range(3)]
    proj_inputs += [rng.normal(size=(3, 3)) for _ in range(4)]
    for M in proj_inputs:
        add("project_to_so3", {"mat": j(M)}, j(mr.ProjectToSO3(M)))
        add("distance_to_so3", {"mat": j(M)}, float(mr.DistanceToSO3(M)))
        add("test_if_so3", {"mat": j(M)}, bool(mr.TestIfSO3(M)))
    se3_inputs = [np.array([[0.675, 0.150, 0.720, 1.2], [0.370, 0.771, -0.511, 5.4],
                            [-0.630, 0.619, 0.472, 3.6], [0.003, 0.002, 0.010, 0.9]]), np.eye(4),
                  mr.RpToTrans(np.diag([1.0, 1.0, -1.0]), np.ones(3))]
    se3_inputs += [rand_T(rng) + rng.normal(scale=s, size=(4, 4)) for s in (1e-5, 0.05, 0.5)]
    for M in se3_inputs:
        add("project_to_se3", {"mat": j(M)}, j(mr.ProjectToSE3(M)))
        add("distance_to_se3", {"mat": j(M)}, float(mr.DistanceToSE3(M)))
        add("test_if_se3", {"mat": j(M)}, bool(mr.TestIfSE3(M)))

    # ---------------- chapters 4-6
    for n in (1, 2, 3, 6):
        for _ in range(2):
            S, B, M = rand_screws(rng, n), rand_screws(rng, n), rand_T(rng)
            th = rng.uniform(-np.pi, np.pi, n)
            add("fkin_space", {"M": j(M), "Slist": j(S), "thetalist": j(th)}, j(mr.FKinSpace(M, S, th)))
            add("fkin_body", {"M": j(M), "Blist": j(B), "thetalist": j(th)}, j(mr.FKinBody(M, B, th)))
            add("jacobian_space", {"Slist": j(S), "thetalist": j(th)}, j(mr.JacobianSpace(S, th)))
            add("jacobian_body", {"Blist": j(B), "thetalist": j(th)}, j(mr.JacobianBody(B, th)))
    # IK: the MR Example 6.1 3R arm, then random 6R arms (target from FK, guess nearby; one unreachable target)
    Blist = np.array([[0, 0, -1, 2, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0.1]]).T.astype(float)
    M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], float)
    T = np.array([[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], float)
    Slist = np.array([[0, 0, 1, 4, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, -1, -6, 0, -0.1]]).T.astype(float)
    ik_cases = [(Blist, Slist, M, T, np.array([1.5, 2.5, 3.0]), 0.01, 0.001)]
    for k in range(5):
        S6 = np.array([mr.ScrewToAxis(rng.uniform(-0.4, 0.4, 3), rand_unit(rng), 0.0) for _ in range(6)]).T
        M6 = rand_T(rng)
        B6 = mr.Adjoint(mr.TransInv(M6)) @ S6
        th = rng.uniform(-1.5, 1.5, 6)
        Tgoal = mr.FKinSpace(M6, S6, th)
        if k == 4:
            Tgoal = Tgoal @ mr.RpToTrans(np.eye(3), np.array([0.0, 0.0, 50.0]))  # far away: IK must fail
        guess = th + rng.normal(scale=0.15, size=6)
        ik_cases.append((B6, S6, M6, Tgoal, guess, 1e-6, 1e-6))
    for B, S, Mh, Tg, guess, eomg, ev in ik_cases:
        thb, okb = mr.IKinBody(B, Mh, Tg, guess, eomg, ev)
        add("ikin_body", {"Blist": j(B), "M": j(Mh), "T": j(Tg), "thetalist0": j(guess), "eomg": eomg, "ev": ev},
            {"thetalist": j(thb), "success": bool(okb)})
        ths, oks = mr.IKinSpace(S, Mh, Tg, guess, eomg, ev)
        add("ikin_space", {"Slist": j(S), "M": j(Mh), "T": j(Tg), "thetalist0": j(guess), "eomg": eomg, "ev": ev},
            {"thetalist": j(ths), "success": bool(oks)})

    # ---------------- chapter 8
    for Mlist, Glist, Slist in chains(rng):
        n = len(Glist)
        chain = {"Mlist": j(Mlist), "Glist": j(Glist), "Slist": j(Slist)}
        th, dth, ddth = rng.uniform(-2, 2, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        g, Ftip, tau = np.array([0, 0, -9.8]), rng.normal(size=6), rng.normal(size=n) * 5
        add("inverse_dynamics", {**chain, "thetalist": j(th), "dthetalist": j(dth), "ddthetalist": j(ddth),
                                 "g": j(g), "Ftip": j(Ftip)},
            j(mr.InverseDynamics(th, dth, ddth, g, Ftip, Mlist, Glist, Slist)))
        add("mass_matrix", {**chain, "thetalist": j(th)}, j(mr.MassMatrix(th, Mlist, Glist, Slist)))
        add("vel_quadratic_forces", {**chain, "thetalist": j(th), "dthetalist": j(dth)},
            j(mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist)))
        add("gravity_forces", {**chain, "thetalist": j(th), "g": j(g)},
            j(mr.GravityForces(th, g, Mlist, Glist, Slist)))
        add("end_effector_forces", {**chain, "thetalist": j(th), "Ftip": j(Ftip)},
            j(mr.EndEffectorForces(th, Ftip, Mlist, Glist, Slist)))
        add("forward_dynamics", {**chain, "thetalist": j(th), "dthetalist": j(dth), "taulist": j(tau),
                                 "g": j(g), "Ftip": j(Ftip)},
            j(mr.ForwardDynamics(th, dth, tau, g, Ftip, Mlist, Glist, Slist)))
        dt = 0.01
        t1, d1 = mr.EulerStep(th, dth, ddth, dt)
        add("euler_step", {"thetalist": j(th), "dthetalist": j(dth), "ddthetalist": j(ddth), "dt": dt},
            {"thetalist": j(t1), "dthetalist": j(d1)})
        N = 6
        thm, dthm, ddthm = (rng.uniform(-1, 1, (N, n)) for _ in range(3))
        Ftipmat = rng.normal(size=(N, 6))
        add("inverse_dynamics_trajectory", {**chain, "thetamat": j(thm), "dthetamat": j(dthm),
                                            "ddthetamat": j(ddthm), "g": j(g), "Ftipmat": j(Ftipmat)},
            j(mr.InverseDynamicsTrajectory(thm, dthm, ddthm, g, Ftipmat, Mlist, Glist, Slist)))
        taumat = rng.normal(size=(N, n))
        thf, dthf = mr.ForwardDynamicsTrajectory(th, dth, taumat, g, Ftipmat, Mlist, Glist, Slist, 0.05, 4)
        add("forward_dynamics_trajectory", {**chain, "thetalist": j(th), "dthetalist": j(dth), "taumat": j(taumat),
                                            "g": j(g), "Ftipmat": j(Ftipmat), "dt": 0.05, "intRes": 4},
            {"thetamat": j(thf), "dthetamat": j(dthf)})
        # chapter 11 on the same chain (model errors: gtilde, scaled inertias)
        eint = rng.normal(size=n) * 0.1
        thd, dthd, ddthd = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        add("computed_torque", {**chain, "thetalist": j(th), "dthetalist": j(dth), "eint": j(eint), "g": j(g),
                                "thetalistd": j(thd), "dthetalistd": j(dthd), "ddthetalistd": j(ddthd),
                                "Kp": 1.3, "Ki": 1.2, "Kd": 1.1},
            j(mr.ComputedTorque(th, dth, eint, g, Mlist, Glist, Slist, thd, dthd, ddthd, 1.3, 1.2, 1.1)))
        Nc = 5
        thmd = mr.JointTrajectory(th, th + 0.5, 1.0, Nc, 5)
        dthmd = np.gradient(thmd, axis=0)
        ddthmd = np.gradient(dthmd, axis=0)
        Ftc = np.zeros((Nc, 6))
        Gt = [1.1 * G for G in Glist]
        gt = np.array([0, 0, -9.7])
        tauc, thc = mr.SimulateControl(th, dth, g, Ftc, Mlist, Glist, Slist, thmd, dthmd, ddthmd, gt, Mlist, Gt,
                                       20.0, 10.0, 18.0, 0.01, 3)
        plt.close("all")
        add("simulate_control", {**chain, "thetalist": j(th), "dthetalist": j(dth), "g": j(g), "Ftipmat": j(Ftc),
                                 "thetamatd": j(thmd), "dthetamatd": j(dthmd), "ddthetamatd": j(ddthmd),
                                 "gtilde": j(gt), "Mtildelist": j(Mlist), "Gtildelist": j(Gt),
                                 "Kp": 20.0, "Ki": 10.0, "Kd": 18.0, "dt": 0.01, "intRes": 3},
            {"taumat": j(tauc), "thetamat": j(thc)})

    # ---------------- chapter 9
    for Tf, t in ((2.0, 0.0), (2.0, 0.6), (2.0, 1.0), (2.0, 2.0), (3.5, 1.1)):
        add("cubic_time_scaling", {"Tf": Tf, "t": t}, float(mr.CubicTimeScaling(Tf, t)))
        add("quintic_time_scaling", {"Tf": Tf, "t": t}, float(mr.QuinticTimeScaling(Tf, t)))
    for method in (3, 5):
        for n, N in ((2, 2), (4, 7)):
            a, b = rng.normal(size=n), rng.normal(size=n)
            add("joint_trajectory", {"thetastart": j(a), "thetaend": j(b), "Tf": 4.0, "N": N, "method": method},
                j(mr.JointTrajectory(a, b, 4.0, N, method)))
        Xs, Xe = rand_T(rng), rand_T(rng)
        add("screw_trajectory", {"Xstart": j(Xs), "Xend": j(Xe), "Tf": 5.0, "N": 4, "method": method},
            j(mr.ScrewTrajectory(Xs, Xe, 5.0, 4, method)))
        add("cartesian_trajectory", {"Xstart": j(Xs), "Xend": j(Xe), "Tf": 5.0, "N": 4, "method": method},
            j(mr.CartesianTrajectory(Xs, Xe, 5.0, 4, method)))

    # Every public function of modern_robotics.core must be covered.
    import inspect
    import re
    public = [n for n, f in inspect.getmembers(core, inspect.isfunction) if f.__module__ == core.__name__]

    def snake(name):
        special = {"VecToso3": "vec_to_so3", "so3ToVec": "so3_to_vec", "VecTose3": "vec_to_se3",
                   "se3ToVec": "se3_to_vec", "FKinBody": "fkin_body", "FKinSpace": "fkin_space",
                   "IKinBody": "ikin_body", "IKinSpace": "ikin_space", "ad": "ad"}
        if name in special:
            return special[name]
        s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()
        return s.replace("project_to_s_o3", "project_to_so3")

    missing = sorted(snake(n) for n in public if snake(n) not in C)
    C["_meta"]["python_functions"] = sorted(public)
    C["_meta"]["helpers_only_in_cpp"] = ["pinv"]
    assert missing == [], f"no cases for: {missing}"

    with open(OUT, "w", encoding="utf8") as f:
        json.dump(C, f, indent=None, separators=(",", ":"))
        f.write("\n")
    total = sum(len(v) for k, v in C.items() if not k.startswith("_"))
    print(f"wrote {OUT}: {len(C) - 1} functions, {total} cases")


if __name__ == "__main__":
    main()
