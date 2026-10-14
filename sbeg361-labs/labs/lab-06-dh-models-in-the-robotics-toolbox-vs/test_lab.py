"""Auto-checks for SBEG 361 Lab 6 -- DH models vs the product of exponentials; C++/Eigen comparison.

Usage:   LAB_IMPL=starter  python -m pytest -q test_lab.py     (your work, file starter.py + cpp/fk_starter.cpp)
         LAB_IMPL=solution python -m pytest -q test_lab.py     (reference; default)
Ground truth: the official `modern_robotics` library (FKinSpace), Corke's printed examples (Sec. 7.1-7.2,
pp. 140-142), MR Example C.1 = MR Example 4.1 (App. C p. 593; pp. 141-142), MR Exercise 5.7 (RRP) and the
Robotics Toolbox for Python.  The C++ test is skipped when no C++17 compiler or Eigen is found.
"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
IMPL = os.environ.get("LAB_IMPL", "starter")
lab = importlib.import_module(IMPL)
from course_arm_model import ENC_RES, LIMITS, M_HOME, Q_HOME, RCM_PORT, RCM_TIP0, SLIST  # noqa: E402

D2R = np.pi / 180
Q_SAMPLES = np.loadtxt(HERE / "data" / "q_samples.csv", delimiter=",", skiprows=1)
RNG = np.random.default_rng(361)


def rz(t):
    return mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0, 0, t])), [0, 0, 0])


def rx(t):
    return mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([t, 0, 0])), [0, 0, 0])


def tz(d):
    return mr.RpToTrans(np.eye(3), [0, 0, d])


def tx(a):
    return mr.RpToTrans(np.eye(3), [a, 0, 0])


def random_table(n, convention, seed):
    rng = np.random.default_rng(seed)
    types = "".join(rng.choice(["R", "P"], size=n, p=[0.7, 0.3]))
    base = mr.FKinSpace(np.eye(4), np.eye(6), rng.normal(size=6))   # any rigid transform
    tool = mr.FKinSpace(np.eye(4), np.eye(6), rng.normal(size=6))
    return dict(convention=convention, types=types, d=rng.normal(size=n) * 0.3, a=rng.normal(size=n) * 0.3,
                alpha=rng.uniform(-np.pi, np.pi, n), offset=rng.uniform(-np.pi, np.pi, n), base=base, tool=tool)


def ref_fk(table, q):
    """Reference FK of a table built from elementary transforms (not from the student's code)."""
    T = np.asarray(table["base"], float)
    for i, jt in enumerate(table["types"]):
        th = table["offset"][i] + (q[i] if jt == "R" else 0.0)
        d = table["d"][i] + (q[i] if jt == "P" else 0.0)
        a, al = table["a"][i], table["alpha"][i]
        T = T @ (rz(th) @ tz(d) @ tx(a) @ rx(al) if table["convention"] == "standard"
                 else rx(al) @ tx(a) @ tz(d) @ rz(th))
    return T @ np.asarray(table["tool"], float)


# MR Example 4.1 (Fig. 4.3), L1 = 0.3, L2 = 0.2, and the modified DH table of MR Example C.1 (p. 593)
L1E, L2E = 0.3, 0.2
S_EX41 = np.array([[0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, -L1E], [1, 0, 0, 0, -L2E, 0]], float).T
M_EX41 = np.array([[0, 0, 1, L1E], [0, 1, 0, 0], [-1, 0, 0, -L2E], [0, 0, 0, 1]], float)
MDH_EXC1 = dict(convention="modified", types="RRR", d=np.zeros(3), a=np.array([0, L1E, L2E]),
                alpha=np.array([0, 90, -90]) * D2R, offset=np.array([0, -90, 0]) * D2R,
                base=np.eye(4), tool=np.eye(4))
# MR Exercise 5.7 RRP robot (Fig. 5.19) and a standard DH table of it (same dict format as course_arm_dh() in this lab)
S_RRP = np.array([[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 2, 0], [0, 0, 0, 0, 1, 0]], float).T
M_RRP = np.array([[-1, 0, 0, 0], [0, 0, 1, 3], [0, 1, 0, 2], [0, 0, 0, 1]], float)
DH_RRP = dict(convention="standard", types="RRP", d=np.array([2.0, 0, 3.0]), a=np.zeros(3),
              alpha=np.array([90, 90, 0]) * D2R, offset=np.array([90, 90, 0]) * D2R,
              base=np.eye(4), tool=np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]))


# ------------------------------------------------------------- provided helpers (library signatures)
def test_provided_helpers_use_library_signatures():
    """fk_space(M, Slist, theta) as in Lab 5; screw_exp(S, theta) = MatrixExp6([S] theta)."""
    S = SLIST[:, 1]
    np.testing.assert_allclose(lab.screw_exp(S, 0.7), mr.MatrixExp6(mr.VecTose3(S * 0.7)), atol=1e-12)
    for q in Q_SAMPLES[:20]:
        np.testing.assert_allclose(lab.fk_space(M_HOME, SLIST, q), mr.FKinSpace(M_HOME, SLIST, q), atol=1e-12)


# ------------------------------------------------------------- Checkpoint 1: link transforms
def test_dh_std_corke_link_example():
    """Corke p. 140: Link([0, 0.1, 0.2, pi/2]).A(0.5)."""
    A = lab.dh_std(0.5, 0.1, 0.2, np.pi / 2)
    expected = np.array([[0.8776, 0, 0.4794, 0.1755], [0.4794, 0, -0.8776, 0.0959],
                         [0, 1, 0, 0.1], [0, 0, 0, 1]])
    np.testing.assert_allclose(A, expected, atol=1e-4)


def test_dh_std_is_rz_tz_tx_rx():
    for _ in range(20):
        th, d, a, al = RNG.uniform(-3, 3, 4)
        np.testing.assert_allclose(lab.dh_std(th, d, a, al), rz(th) @ tz(d) @ tx(a) @ rx(al), atol=1e-12)


def test_dh_mod_is_rx_tx_tz_rz():
    for _ in range(20):
        al, a, d, th = RNG.uniform(-3, 3, 4)
        np.testing.assert_allclose(lab.dh_mod(al, a, d, th), rx(al) @ tx(a) @ tz(d) @ rz(th), atol=1e-12)


# ------------------------------------------------------------- Checkpoint 2: FK from a table
def test_fk_dh_corke_two_link():
    """Corke Sec. 7.2.1 (p. 142): two-link planar robot, a1 = a2 = 1."""
    two = dict(convention="standard", types="RR", d=np.zeros(2), a=np.ones(2), alpha=np.zeros(2),
               offset=np.zeros(2), base=np.eye(4), tool=np.eye(4))
    np.testing.assert_allclose(lab.fk_dh(two, [0, 0])[:3, 3], [2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(lab.fk_dh(two, [np.pi / 4, -np.pi / 4])[:3, 3], [1.7071, 0.7071, 0], atol=1e-4)


def test_fk_dh_mr_example_c1_equals_example_4_1():
    for q in RNG.uniform(-np.pi, np.pi, (50, 3)):
        np.testing.assert_allclose(lab.fk_dh(MDH_EXC1, q), mr.FKinSpace(M_EX41, S_EX41, q), atol=1e-12)


def test_fk_dh_prismatic_mr_exercise_5_7():
    T = lab.fk_dh(DH_RRP, [np.pi / 2, np.pi / 2, 1.0])
    np.testing.assert_allclose(T, [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 6], [0, 0, 0, 1]], atol=1e-9)
    for q in RNG.uniform(-2, 2, (20, 3)):
        np.testing.assert_allclose(lab.fk_dh(DH_RRP, q), mr.FKinSpace(M_RRP, S_RRP, q), atol=1e-9)


def test_fk_dh_random_tables_both_conventions():
    for k, conv in enumerate(["standard", "modified"] * 5):
        tab = random_table(5, conv, seed=k)
        for q in RNG.uniform(-2, 2, (5, 5)):
            np.testing.assert_allclose(lab.fk_dh(tab, q), ref_fk(tab, q), atol=1e-12)
        frames = lab.link_frames(tab, np.zeros(5))
        assert len(frames) == 6
        np.testing.assert_allclose(frames[0], tab["base"], atol=1e-12)
        np.testing.assert_allclose(frames[-1] @ tab["tool"], lab.fk_dh(tab, np.zeros(5)), atol=1e-12)
    with pytest.raises(ValueError):
        lab.fk_dh(dict(random_table(2, "standard", 0), convention="craig-ish"), [0, 0])


# ------------------------------------------------------------- Checkpoint 3: course arm DH tables
def test_course_arm_dh_matches_poe():
    tab = lab.course_arm_dh()
    assert tab["convention"] == "standard" and tab["types"] == "RRRRRR"
    np.testing.assert_allclose(lab.fk_dh(tab, np.zeros(6)), M_HOME, atol=1e-12)
    for q in Q_SAMPLES:
        np.testing.assert_allclose(lab.fk_dh(tab, q), mr.FKinSpace(M_HOME, SLIST, q), atol=1e-12)


def test_course_arm_mdh_matches_poe():
    tab = lab.course_arm_mdh()
    assert tab["convention"] == "modified"
    for q in Q_SAMPLES[:200]:
        np.testing.assert_allclose(lab.fk_dh(tab, q), mr.FKinSpace(M_HOME, SLIST, q), atol=1e-12)


def test_course_arm_ready_pose_checkpoint():
    """Value quoted in the handout (Step 3): flange position at q_home."""
    np.testing.assert_allclose(lab.fk_dh(lab.course_arm_dh(), Q_HOME)[:3, 3], [0.23407, 0.06344, 0.08106],
                               atol=5e-5)


# ------------------------------------------------------------- Checkpoint 4: conversions
def test_poe_from_dh_course_arm_both_conventions():
    for tab in (lab.course_arm_dh(), lab.course_arm_mdh()):
        S, M = lab.poe_from_dh(tab)
        np.testing.assert_allclose(S, SLIST, atol=1e-12)
        np.testing.assert_allclose(M, M_HOME, atol=1e-12)


def test_poe_from_dh_textbook_chains():
    S, M = lab.poe_from_dh(MDH_EXC1)
    np.testing.assert_allclose(S, S_EX41, atol=1e-12)
    np.testing.assert_allclose(M, M_EX41, atol=1e-12)
    S, M = lab.poe_from_dh(DH_RRP)
    np.testing.assert_allclose(S, S_RRP, atol=1e-9)
    np.testing.assert_allclose(M, M_RRP, atol=1e-9)


def test_poe_from_dh_random_tables():
    for k, conv in enumerate(["standard", "modified"] * 4):
        tab = random_table(6, conv, seed=100 + k)
        S, M = lab.poe_from_dh(tab)
        for q in RNG.uniform(-2, 2, (5, 6)):
            np.testing.assert_allclose(mr.FKinSpace(M, S, q), ref_fk(tab, q), atol=1e-10)


def test_std_mod_round_trip():
    for k in range(6):
        std = random_table(6, "standard", seed=200 + k)
        mod = lab.std_to_mod(std)
        back = lab.mod_to_std(mod)
        assert mod["convention"] == "modified" and back["convention"] == "standard"
        assert std["convention"] == "standard"                       # input not modified
        for q in RNG.uniform(-2, 2, (5, 6)):
            T = ref_fk(std, q)
            np.testing.assert_allclose(ref_fk(mod, q), T, atol=1e-12)
            np.testing.assert_allclose(ref_fk(back, q), T, atol=1e-12)
    mod = lab.std_to_mod(lab.course_arm_dh())
    np.testing.assert_allclose(mod["a"], lab.course_arm_mdh()["a"], atol=1e-12)
    np.testing.assert_allclose(mod["alpha"], lab.course_arm_mdh()["alpha"], atol=1e-12)


# ------------------------------------------------------------- Checkpoint 5: RCM robot and toolbox
def rcm_poe():
    p = RCM_PORT
    S = np.c_[np.r_[0, 0, 1, -np.cross([0, 0, 1], p)], np.r_[0, 1, 0, -np.cross([0, 1, 0], p)],
              [0, 0, 0, 0, 0, -1], np.r_[0, 0, -1, -np.cross([0, 0, -1], p)]]
    M = mr.RpToTrans(np.diag([1.0, -1, -1]), p + [0, 0, -RCM_TIP0])
    return S, M


def random_rcm_q(n):
    return np.c_[RNG.uniform(-np.pi, np.pi, n), RNG.uniform(-1.0, 1.0, n), RNG.uniform(0, 0.15, n),
                 RNG.uniform(-np.pi, np.pi, n)]


def test_rcm_dh_matches_poe_and_closed_form():
    tab = lab.rcm_dh()
    assert tab["types"] == "RRPR"
    S, M = rcm_poe()
    for q in random_rcm_q(200):
        T = lab.fk_dh(tab, q)
        np.testing.assert_allclose(T, mr.FKinSpace(M, S, q), atol=1e-12)
        u = np.array([-np.sin(q[1]) * np.cos(q[0]), -np.sin(q[1]) * np.sin(q[0]), -np.cos(q[1])])
        np.testing.assert_allclose(T[:3, 3], RCM_PORT + (RCM_TIP0 + q[2]) * u, atol=1e-12)
    S2, M2 = lab.poe_from_dh(tab)
    np.testing.assert_allclose(S2, S, atol=1e-12)
    np.testing.assert_allclose(M2, M, atol=1e-12)


def test_rcm_property_and_distance():
    tab = lab.rcm_dh()
    d = [lab.rcm_distance(lab.fk_dh(tab, q)) for q in random_rcm_q(500)]
    assert max(d) < 1e-12
    T = mr.RpToTrans(np.eye(3), RCM_PORT + [0.003, -0.004, 0.2])           # shaft parallel to z, 5 mm away
    assert abs(lab.rcm_distance(T) - 0.005) < 1e-12


def test_rtb_course_arm_both_conventions():
    pytest.importorskip("roboticstoolbox")
    for tab in (lab.course_arm_dh(), lab.course_arm_mdh()):
        robot = lab.rtb_robot(tab, "course_arm")
        assert robot.n == 6 and robot.mdh == (tab["convention"] == "modified")
        for q in Q_SAMPLES[:50]:
            np.testing.assert_allclose(robot.fkine(q).A, mr.FKinSpace(M_HOME, SLIST, q), atol=1e-10)


def test_rtb_prismatic_rcm_and_rrp():
    pytest.importorskip("roboticstoolbox")
    S, M = rcm_poe()
    robot = lab.rtb_robot(lab.rcm_dh(), "rcm")
    for q in random_rcm_q(30):
        np.testing.assert_allclose(robot.fkine(q).A, mr.FKinSpace(M, S, q), atol=1e-10)
    rrp = lab.rtb_robot(DH_RRP, "rrp")
    np.testing.assert_allclose(rrp.fkine([np.pi / 2, np.pi / 2, 1.0]).A[:3, 3], [0, 0, 6], atol=1e-9)
    mrob = lab.rtb_robot(lab.std_to_mod(lab.rcm_dh()), "rcm_mdh")
    for q in random_rcm_q(10):
        np.testing.assert_allclose(mrob.fkine(q).A, mr.FKinSpace(M, S, q), atol=1e-10)


# ------------------------------------------------------------- Checkpoint 6: C++ comparison helpers
def test_read_pose_csv_and_pose_errors(tmp_path):
    Ts = np.array([mr.FKinSpace(M_HOME, SLIST, q) for q in Q_SAMPLES[:20]])
    rows = np.c_[Ts[:, :3, :3].reshape(20, 9), Ts[:, :3, 3]]
    f = tmp_path / "poses.csv"
    np.savetxt(f, rows, delimiter=",", fmt="%.17g", header="r11,r12,r13,r21,r22,r23,r31,r32,r33,px,py,pz",
               comments="")
    back = lab.read_pose_csv(f)
    assert back.shape == (20, 4, 4)
    np.testing.assert_allclose(back, Ts, atol=1e-15)
    dp, dr = lab.pose_errors(Ts, back)
    assert dp < 1e-15 and dr < 1e-7
    pert = Ts.copy()
    pert[7] = pert[7] @ mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0, 0, 0.01])), [0.002, 0, 0])
    dp, dr = lab.pose_errors(Ts, pert)
    assert abs(dp - 0.002) < 1e-12 and abs(dr - 0.01) < 1e-9


def test_time_fk_returns_seconds_per_call():
    t = lab.time_fk(lambda q: mr.FKinSpace(M_HOME, SLIST, q), Q_SAMPLES[:50], repeat=2)
    assert isinstance(t, float) and 0 < t < 0.1


def test_encoder_quantisation_checkpoint():
    """Handout Step 6c: worst flange error of 1000 samples when every joint is off by one encoder count."""
    tab = lab.course_arm_dh()
    err = max(np.linalg.norm(lab.fk_dh(tab, q + ENC_RES)[:3, 3] - lab.fk_dh(tab, q)[:3, 3]) for q in Q_SAMPLES)
    assert abs(err - 0.001016) < 2e-6


# ------------------------------------------------------------- C++ / Eigen (optional, auto-skipped)
def _find_eigen():
    cands = [os.environ.get("EIGEN3_INCLUDE_DIR", ""), "/usr/include/eigen3", "/usr/local/include/eigen3",
             "/opt/homebrew/include/eigen3"]
    try:
        import roboticstoolbox
        cands.append(str(Path(roboticstoolbox.__file__).parent / "ets" / "cpp-extensions"))
    except ImportError:
        pass
    for up in [HERE, *HERE.parents][:7]:  # the course's own copy: cpp_support/third_party/eigen
        for sup in (up / "cpp_support", up / "0 Course resources" / "cpp_support"):
            cands.append(str(sup / "third_party" / "eigen"))
    for c in cands:
        if c and (Path(c) / "Eigen" / "Dense").exists():
            return c
    return None


def test_cpp_fk_matches_python(tmp_path):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    eigen = _find_eigen()
    if cxx is None or eigen is None:
        pytest.skip("no C++ compiler or Eigen headers found (set EIGEN3_INCLUDE_DIR)")
    cpp = HERE / "cpp"
    exe = tmp_path / "fk_compare"
    fk_src = cpp / f"fk_{IMPL}.cpp"
    if not fk_src.exists():          # fallback when run from the full course tree
        fk_src = HERE.parent / "instructor" / "cpp" / f"fk_{IMPL}.cpp"
    if not fk_src.exists():
        pytest.skip(f"fk_{IMPL}.cpp not found in the cpp/ folder of your Lab code folder")
    src = [cpp / "main.cpp", cpp / "robot_data.cpp", fk_src]
    # Windows + MinGW: link the C++ runtime statically, otherwise the program may load another program's
    # libstdc++ DLL from PATH and fail to start (exit code 0xC0000139)
    static = ["-static"] if sys.platform == "win32" else []
    build = subprocess.run([cxx, "-std=c++17", "-O2", f"-I{eigen}", f"-I{cpp}", *map(str, src), *static, "-o", str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    run = subprocess.run([str(exe), str(HERE / "data" / "q_samples.csv"), str(tmp_path), "3"],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, "C++ program failed: " + run.stderr[-500:]
    ref = np.array([mr.FKinSpace(M_HOME, SLIST, q) for q in Q_SAMPLES])
    for name in ("poses_poe_cpp.csv", "poses_dh_cpp.csv"):
        dp, dr = lab.pose_errors(ref, lab.read_pose_csv(tmp_path / name))
        assert dp < 1e-12 and dr < 1e-7, name
    timing = np.genfromtxt(tmp_path / "timing_cpp.csv", delimiter=",", skip_header=1, usecols=1)
    assert np.all(timing > 0)
