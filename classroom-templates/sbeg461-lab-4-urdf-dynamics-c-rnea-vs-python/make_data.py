"""Generate the data files for SBEG 461 Lab 4 (run once; the outputs are committed).

    python make_data.py        -> arm6.urdf, states_arm6.csv

arm6.urdf  A generic 6-DOF revolute arm with UR5-like joint geometry (the robot MR uses in Sections 4.2 and
           8.8) and ILLUSTRATIVE inertial values. It stands in for the manipulator of an EOD/IED-disposal
           robot or an assistive (wheelchair-mounted) arm. Some inertial frames are offset AND rotated, and one
           inertia tensor has products of inertia, so that a parser must handle the full URDF <inertial> element.
states_arm6.csv  200 random states (theta, thetadot, thetaddot) used for the Python/C++ timing and agreement test.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PI2 = np.pi / 2

# (joint name, parent, child, origin xyz, origin rpy, axis) -- UR5-like kinematic layout
JOINTS = [
    ("shoulder_pan", "base_link", "shoulder_link", (0, 0, 0.089159), (0, 0, 0), (0, 0, 1)),
    ("shoulder_lift", "shoulder_link", "upper_arm_link", (0, 0.13585, 0), (0, PI2, 0), (0, 1, 0)),
    ("elbow", "upper_arm_link", "forearm_link", (0, -0.1197, 0.425), (0, 0, 0), (0, 1, 0)),
    ("wrist_1", "forearm_link", "wrist_1_link", (0, 0, 0.39225), (0, PI2, 0), (0, 1, 0)),
    ("wrist_2", "wrist_1_link", "wrist_2_link", (0, 0.093, 0), (0, 0, 0), (0, 0, 1)),
    ("wrist_3", "wrist_2_link", "wrist_3_link", (0, 0, 0.09465), (0, 0, 0), (0, 1, 0)),
]
TOOL = ("tool_fixed", "wrist_3_link", "tool0", (0, 0.0823, 0), (-PI2, 0, 0))

# link: (mass, inertial origin xyz, inertial origin rpy, (ixx, ixy, ixz, iyy, iyz, izz))  -- illustrative values
INERTIA = {
    "shoulder_link": (3.7, (0, 0, 0), (0, 0, 0), (0.0103, 0, 0, 0.0103, 0, 0.0067)),
    "upper_arm_link": (8.393, (0, 0, 0.28), (0, 0, 0), (0.2269, 0, 0, 0.2269, 0, 0.0151)),
    "forearm_link": (2.275, (0, 0.005, 0.25), (0.1, 0, 0), (0.0494, 0.0002, -0.0003, 0.0494, 0.0004, 0.0041)),
    "wrist_1_link": (1.219, (0, 0.01, 0), (0, 0, 0.3), (0.0030, 0, 0, 0.0030, 0, 0.0022)),
    "wrist_2_link": (1.219, (0, 0, 0.01), (0, 0, 0), (0.0030, 0, 0, 0.0030, 0, 0.0022)),
    "wrist_3_link": (0.1879, (0, 0.0623, 0), (PI2, 0, 0), (0.00017, 0, 0, 0.00017, 0, 0.00024)),
}


def fmt(v):
    return " ".join(f"{x:.10g}" for x in v)


def urdf():
    out = ['<?xml version="1.0"?>',
           '<!-- arm6: generic 6-DOF arm, UR5-like geometry, ILLUSTRATIVE inertias (SBEG 461 Lab 4) -->',
           '<robot name="arm6">', '  <link name="base_link"/>']
    for name, (m, xyz, rpy, (ixx, ixy, ixz, iyy, iyz, izz)) in INERTIA.items():
        out += [f'  <link name="{name}">', '    <inertial>',
                f'      <origin xyz="{fmt(xyz)}" rpy="{fmt(rpy)}"/>', f'      <mass value="{m}"/>',
                f'      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>',
                '    </inertial>', '  </link>']
    out.append('  <link name="tool0"/>')
    for name, parent, child, xyz, rpy, axis in JOINTS:
        out += [f'  <joint name="{name}" type="revolute">', f'    <parent link="{parent}"/>',
                f'    <child link="{child}"/>', f'    <origin xyz="{fmt(xyz)}" rpy="{fmt(rpy)}"/>',
                f'    <axis xyz="{fmt(axis)}"/>',
                '    <limit lower="-6.2832" upper="6.2832" effort="150" velocity="3.15"/>', '  </joint>']
    name, parent, child, xyz, rpy = TOOL
    out += [f'  <joint name="{name}" type="fixed">', f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>', f'    <origin xyz="{fmt(xyz)}" rpy="{fmt(rpy)}"/>', '  </joint>',
            '</robot>']
    return "\n".join(out) + "\n"


def states(n=200, seed=461):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, (n, 6))
    thd = rng.uniform(-2, 2, (n, 6))
    thdd = rng.uniform(-5, 5, (n, 6))
    return np.hstack([th, thd, thdd])


if __name__ == "__main__":
    (HERE / "arm6.urdf").write_text(urdf())
    header = ",".join([f"th{i}" for i in range(1, 7)] + [f"dth{i}" for i in range(1, 7)] + [f"ddth{i}" for i in range(1, 7)])
    np.savetxt(HERE / "states_arm6.csv", states(), delimiter=",", header=header, comments="", fmt="%.10f")
    print("wrote arm6.urdf and states_arm6.csv")
