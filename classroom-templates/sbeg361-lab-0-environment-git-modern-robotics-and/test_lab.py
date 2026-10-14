"""Auto-checks for SBEG 361 Lab 0.

    LAB_IMPL=solution pytest -q test_lab.py      # reference: all pass
    LAB_IMPL=starter  pytest -q test_lab.py      # untouched starter: every test fails (NotImplementedError)

LAB_IMPL is the name of the module to test (default "solution"); it is imported from this folder.
"""
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
lab = importlib.import_module(os.environ.get("LAB_IMPL", "starter"))
DATA = HERE / "data" / "mechanisms.json"


# ---------------------------------------------------------------- checkpoint 1: environment
def test_env_report():
    rep = lab.env_report()
    assert set(rep) >= {"python", "numpy", "modern_robotics"}
    assert all(isinstance(v, str) and v for v in rep.values())
    major, minor = (int(x) for x in rep["python"].split(".")[:2])
    assert (major, minor) >= (3, 10)


# ---------------------------------------------------------------- checkpoint 2: joints and Gruebler
@pytest.mark.parametrize("j,f", [("R", 1), ("P", 1), ("H", 1), ("C", 2), ("U", 2), ("S", 3),
                                 ("RP", 2), ("E", 3), ("F", 0)])
def test_joint_freedoms(j, f):
    assert lab.joint_freedoms(j) == f


def test_joint_freedoms_unknown():
    with pytest.raises(ValueError):
        lab.joint_freedoms("X")


@pytest.mark.parametrize("j,m,c", [("R", 3, 2), ("P", 3, 2), ("R", 6, 5), ("H", 6, 5),
                                   ("C", 6, 4), ("U", 6, 4), ("S", 6, 3)])
def test_joint_constraints_table_2_1(j, m, c):
    assert lab.joint_constraints(j, m) == c


@pytest.mark.parametrize("j", ["H", "C", "U", "S"])
def test_joint_constraints_planar_na(j):
    with pytest.raises(ValueError):
        lab.joint_constraints(j, 3)


@pytest.mark.parametrize("m,N,joints,dof", [
    (3, 4, {"R": 4}, 1),                       # MR Ex 2.3 four-bar
    (3, 3, ["R", "R", "RP"], 1),               # MR Ex 2.3 slider-crank (ii)
    (3, 5, ["R"] * 5, 2),                      # MR Ex 2.4 five-bar
    (3, 8, {"R": 8, "P": 1}, 3),               # MR Ex 2.5
    (3, 5, {"R": 6}, 0),                       # MR Ex 2.6 parallelogram (Gruebler count)
    (6, 17, {"R": 9, "S": 12}, 15),            # MR Ex 2.7 Delta
    (6, 14, {"U": 6, "P": 6, "S": 6}, 6),      # MR Ex 2.8 Stewart-Gough
    (6, 14, [3] * 12 + [1] * 6, 12),           # 6-SPS given as integers f_i
])
def test_grubler_mr_examples(m, N, joints, dof):
    assert lab.grubler(m, N, joints) == dof


@pytest.mark.parametrize("k", range(1, 8))
def test_grubler_kR_chain(k):
    assert lab.grubler(3, k + 1, ["R"] * k) == k   # MR Ex 2.4


def test_grubler_bad_m():
    with pytest.raises(ValueError):
        lab.grubler(4, 3, ["R", "R"])


# ---------------------------------------------------------------- checkpoint 3: mechanism table
EXPECTED = {
    "Four-bar linkage (MR Ex 2.3)": 1,
    "Slider-crank (MR Ex 2.3)": 1,
    "Five-bar linkage (MR Ex 2.4)": 2,
    "Stephenson six-bar (MR Ex 2.4)": 1,
    "Delta robot (MR Ex 2.7)": 15,
    "Stewart-Gough 6-UPS (MR Ex 2.8)": 6,
    "Four-bar (cruciate ligament) knee model": 1,
    "Linkage-driven prosthetic finger": 1,
    "Human arm, shoulder to palm (S-R-R-U)": 7,
    "Hexapod fracture fixator 6-SPS": 12,
    "Knee exoskeleton with self-aligning RP cuffs": 2,
    "EOD robot 6R arm grasping a fixed handle": 0,
    "EOD robot on free chassis, 6R arm grasping a fixed handle": 3,
    "Casualty-litter scissor lift, two stages": 1,
    "Dual 7R arms holding one package": 8,
}


def test_dof_table():
    table = lab.dof_table(DATA)
    assert table == EXPECTED


# ---------------------------------------------------------------- checkpoint 4: C-space topology
def test_wrap_to_pi_scalar_and_array():
    assert math.isclose(float(lab.wrap_to_pi(3 * np.pi / 2)), -np.pi / 2)
    assert math.isclose(float(lab.wrap_to_pi(np.pi)), -np.pi)        # [-pi, pi): pi maps to -pi
    x = np.linspace(-20, 20, 401)
    w = np.asarray(lab.wrap_to_pi(x))
    assert w.shape == x.shape
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.cos(w), np.cos(x)) and np.allclose(np.sin(w), np.sin(x))


def test_torus_distance():
    # 2R arm: 350 deg and 10 deg are 20 deg apart, not 340 deg
    d = lab.torus_distance([math.radians(350), 0.0], [math.radians(10), 0.0])
    assert math.isclose(d, math.radians(20), abs_tol=1e-12)
    d = lab.torus_distance([0.1, -3.0, 2.0], [0.1 + 4 * np.pi, -3.0, 2.0])
    assert math.isclose(d, 0.0, abs_tol=1e-12)
    d = lab.torus_distance([3.0, 3.0], [-3.0, -3.0])
    assert math.isclose(d, math.sqrt(2) * (2 * np.pi - 6.0), abs_tol=1e-12)


@pytest.mark.parametrize("joints,topo", [
    (["P", "R"], "R^1 x S^1"),          # MR Sec. 2.3.1, PR arm
    (["R", "R"], "T^2"),                # 2R arm
    (["R"], "S^1"),                     # door
    (["P", "P", "P"], "R^3"),           # Cartesian gantry
    (["P", "P", "P", "R"], "R^3 x S^1"),  # radiotherapy couch (illustrative)
    (["R"] * 6, "T^6"),                 # 6R arm without joint limits
])
def test_cspace_topology(joints, topo):
    assert lab.cspace_topology(joints) == topo


def test_cspace_topology_rejects():
    with pytest.raises(ValueError):
        lab.cspace_topology(["S"])
    with pytest.raises(ValueError):
        lab.cspace_topology([])


# ---------------------------------------------------------------- checkpoint 5: representation
def test_latlong_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lat = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        lon = rng.uniform(-np.pi, np.pi)
        p = lab.latlong_to_xyz(lat, lon)
        assert math.isclose(np.linalg.norm(p), 1.0, abs_tol=1e-12)
        la, lo = lab.xyz_to_latlong(3.0 * p)
        assert math.isclose(la, lat, abs_tol=1e-9)
        assert math.isclose(float(lab.wrap_to_pi(lo - lon)), 0.0, abs_tol=1e-9)


def test_longitude_rate_singularity():
    w = 0.2
    assert math.isclose(lab.longitude_rate(0.0, w), 0.2)
    assert math.isclose(lab.longitude_rate(math.radians(60), w), 0.4, rel_tol=1e-12)
    assert math.isclose(lab.longitude_rate(math.radians(89), w), 11.4597, rel_tol=1e-4)
    # the Cartesian speed is the same everywhere: compare with finite differences
    for lat in np.radians([0, 45, 80, 89, 89.9]):
        lr = lab.longitude_rate(lat, w)
        h = 1e-7
        v = (lab.latlong_to_xyz(lat, h * lr) - lab.latlong_to_xyz(lat, 0.0)) / h
        assert math.isclose(np.linalg.norm(v), w, rel_tol=1e-5)


# ---------------------------------------------------------------- checkpoint 6: modern_robotics
def test_fk_2r_against_modern_robotics():
    import modern_robotics as mr

    rng = np.random.default_rng(1)
    L1, L2 = 0.6, 0.4
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    M = np.eye(4)
    M[0, 3] = L1 + L2
    for _ in range(50):
        th = rng.uniform(-np.pi, np.pi, 2)
        T = lab.fk_2r_mr(th, L1, L2)
        assert T.shape == (4, 4)
        assert np.allclose(T, mr.FKinSpace(M, Slist, th), atol=1e-12)
        assert np.allclose(T[:2, 3], lab.fk_2r_trig(th, L1, L2), atol=1e-12)
        assert np.isclose(T[2, 3], 0.0)
        # orientation is a rotation by theta1 + theta2 about z
        phi = th[0] + th[1]
        assert np.allclose(T[:2, :2], [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def test_fk_2r_known_values():
    assert np.allclose(lab.fk_2r_trig([0.0, 0.0]), [1.0, 0.0])
    assert np.allclose(lab.fk_2r_trig([np.pi / 2, -np.pi / 2]), [0.4, 0.6])
    assert np.allclose(lab.fk_2r_trig([np.pi / 6, np.pi / 3]), [0.6 * np.cos(np.pi / 6), 0.3 + 0.4])
