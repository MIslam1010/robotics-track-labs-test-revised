"""scenarios.py - robot models and nominal tasks for the six SBEG 361 project scenarios.

All lengths in metres, angles in radians, masses in kg, torques in N m.
Robot S1 uses the UR5 geometry of MR Example 4.5 (pp. 145-146).  All other robots and
every task number are ILLUSTRATIVE VALUES chosen for teaching; they are not the data of
any specific commercial or military product.

Each robot is described in the space-form PoE (MR Eq. 4.14): screw axes S_i = (w_i, v_i)
in the base frame {s} at the zero ("home") configuration and the home pose M of the tool
frame {b}.  ``points`` lists chain points (position at home, number of joints that carry
it) for drawing and clearance checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kintools import screw_revolute as Rv, screw_prismatic as Pr, trans, rotz, frame_from_z

DEG = np.pi / 180.0


@dataclass
class Robot:
    name: str
    Slist: np.ndarray            # 6 x n screw axes in {s} at theta = 0
    M: np.ndarray                # home pose of the tool frame {b}
    limits: np.ndarray           # n x 2 joint limits [rad or m]
    points: list                 # [(p_home, carried_by_joints), ...] base -> tool
    joint_types: str             # e.g. "RRRRRR", "RRRRRRP"
    q_axis: list                 # one point on each joint axis (revolute) at home
    link_radius: float = 0.04    # radius used for clearance checks
    rate_limit: np.ndarray | None = None    # joint speed limits [rad/s or m/s]
    tau_rated: np.ndarray | None = None     # rated joint torques [N m] (S5)
    notes: str = ""

    @property
    def n(self) -> int:
        return self.Slist.shape[1]


@dataclass
class Scenario:
    sid: int
    title: str
    domain: str                  # "biomedical" or "defence"
    robot: Robot
    task: dict = field(default_factory=dict)

# ============================================================================
# Robot models
# ============================================================================

def ur_type(name, W1, W2, L1, L2, H1, H2, tool=0.0, limits=None, notes="") -> Robot:
    """UR-type 6R arm exactly as in MR Example 4.5 (screw table p. 145).
    `tool` adds a straight tool of that length along the flange z_b axis."""
    S = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, -H1, 0, 0],
        [0, 1, 0, -H1, 0, L1],
        [0, 1, 0, -H1, 0, L1 + L2],
        [0, 0, -1, -W1, L1 + L2, 0],
        [0, 1, 0, H2 - H1, 0, L1 + L2],
    ], float).T
    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]], float)
    M = M @ trans((0, 0, tool))
    q = [(0, 0, 0), (0, 0, H1), (L1, 0, H1), (L1 + L2, 0, H1), (L1 + L2, W1, 0),
         (L1 + L2, 0, H1 - H2)]
    pts = [((0, 0, 0), 0), ((0, 0, H1), 1), ((0, W1, H1), 1), ((L1, W1, H1), 2),
           ((L1 + L2, W1, H1), 3), ((L1 + L2, W1, H1 - H2), 4),
           ((L1 + L2, W1 + W2, H1 - H2), 5), (tuple(M[:3, 3]), 6)]
    if limits is None:
        limits = np.tile([-2 * np.pi, 2 * np.pi], (6, 1))
    return Robot(name, S, M, np.asarray(limits, float), pts, "RRRRRR", q, notes=notes)


def ur5_rehab() -> Robot:
    """S1: UR5 (MR Example 4.5: W1=109, W2=82, L1=425, L2=392, H1=89, H2=95 mm) with a
    0.10 m therapy handle on the flange.  Safe joint limits are illustrative."""
    lim = np.array([[-150, 150], [-180, 0], [-160, 160], [-270, 90], [-180, 180],
                    [-180, 180]], float) * DEG
    return ur_type("UR5 + 0.10 m therapy handle (MR Ex. 4.5)", 0.109, 0.082, 0.425,
                   0.392, 0.089, 0.095, tool=0.10, limits=lim,
                   notes="UR5 geometry from MR Example 4.5; handle and joint limits illustrative")


def desktop_6r(tool=0.06) -> Robot:
    """S3 / course arm: desktop-scale UR-type 6R (illustrative values, ~0.28 m reach)
    with a needle guide of length `tool` on the flange."""
    lim = np.array([[-165, 165], [-165, 165], [-165, 165], [-165, 165], [-165, 165],
                    [-175, 175]], float) * DEG
    r = ur_type("Desktop UR-type 6R + needle guide (illustrative)", 0.0634, 0.0456,
                0.110, 0.096, 0.1312, 0.0751, tool=tool, limits=lim,
                notes="desktop 6R of the course-arm class; illustrative values")
    r.link_radius = 0.025
    return r


def srs_7r() -> Robot:
    """S2: wheelchair-mounted 7R arm, spherical shoulder (z, y, x) - elbow (y) - spherical
    wrist (x, y, z).  Illustrative values: shoulder height 0.25, upper arm 0.40,
    forearm 0.35, wrist-to-spoon-tip 0.12."""
    h, Lu, Lf, Lt = 0.25, 0.40, 0.35, 0.12
    sh, el, wr = np.array([0, 0, h]), np.array([Lu, 0, h]), np.array([Lu + Lf, 0, h])
    w = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    q = [sh, sh, sh, el, wr, wr, wr]
    S = np.column_stack([Rv(wi, qi) for wi, qi in zip(w, q)])
    M = trans(wr + np.array([Lt, 0, 0]))
    lim = np.array([[-170, 170], [-120, 120], [-170, 170], [-150, 5], [-170, 170],
                    [-115, 115], [-170, 170]], float) * DEG
    pts = [((0, 0, 0), 0), (tuple(sh), 1), (tuple(el), 3), (tuple(wr), 4),
           (tuple(M[:3, 3]), 7)]
    r = Robot("Wheelchair-mounted SRS 7R (illustrative)", S, M, lim, pts, "R" * 7,
              [tuple(x) for x in q], link_radius=0.04,
              rate_limit=np.full(7, 40 * DEG), notes="illustrative values")
    return r


def eod_6r(shoulder=(0.10, 0.20), L1=0.80, L2=0.75, Lg=0.25) -> Robot:
    """S4: 6-DOF EOD manipulator on a tracked UGV (illustrative values).
    {s}: on the turret base at deck level (deck 0.45 m above ground), x forward, z up.
    J1 turret yaw (z); J2 shoulder pitch (y) at (a, 0, h); J3 elbow pitch (y);
    J4 forearm roll (x); J5 wrist pitch (y); J6 gripper roll (x).  J4-J6 intersect at the
    wrist centre (spherical wrist).  Home = arm straight forward (a singular pose)."""
    a, h = shoulder
    sh = np.array([a, 0, h]); el = sh + [L1, 0, 0]; wr = el + [L2, 0, 0]
    w = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    q = [(0, 0, 0), sh, el, wr, wr, wr]
    S = np.column_stack([Rv(wi, qi) for wi, qi in zip(w, q)])
    M = trans(wr + [Lg, 0, 0])
    lim = np.array([[-180, 180], [-60, 110], [-160, 160], [-180, 180], [-115, 115],
                    [-180, 180]], float) * DEG
    pts = [((0, 0, 0), 0), ((0, 0, h), 1), (tuple(sh), 1), (tuple(el), 2), (tuple(wr), 3),
           (tuple(M[:3, 3]), 6)]
    return Robot("UGV-mounted EOD 6R arm (illustrative)", S, M, lim, pts, "R" * 6,
                 [tuple(np.asarray(x, float)) for x in q], link_radius=0.04,
                 rate_limit=np.full(6, 30 * DEG), notes="illustrative values")


def lift_5r(h=0.40, L1=0.80, L2=0.75, Lw=0.15) -> Robot:
    """S5: casualty-litter loading arm at the rear of an evacuation vehicle
    (illustrative values).  {s} at the arm base on the rear deck (deck 0.90 m above
    ground).  J1 yaw (z); J2 shoulder pitch (y); J3 elbow pitch (y); J4 wrist pitch (y);
    J5 cradle roll (x).  Tool frame {b} at the cradle lifting point, x_b forward,
    y_b along the litter cross-bar."""
    sh = np.array([0, 0, h]); el = sh + [L1, 0, 0]; wr = el + [L2, 0, 0]
    w = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0)]
    q = [(0, 0, 0), sh, el, wr, wr]
    S = np.column_stack([Rv(wi, qi) for wi, qi in zip(w, q)])
    M = trans(wr + [Lw, 0, 0])
    lim = np.array([[-120, 120], [-30, 120], [-170, 0], [-120, 120], [-30, 30]],
                   float) * DEG
    pts = [((0, 0, 0), 0), (tuple(sh), 1), (tuple(el), 2), (tuple(wr), 3),
           (tuple(M[:3, 3]), 5)]
    return Robot("Litter-loading 5R arm (illustrative)", S, M, lim, pts, "R" * 5,
                 [tuple(np.asarray(x, float)) for x in q], link_radius=0.06,
                 rate_limit=np.full(5, 20 * DEG),
                 tau_rated=np.array([800.0, 2000.0, 1200.0, 400.0, 150.0]),
                 notes="illustrative values")


def telesurgery_6r1p(shaft=0.30) -> Robot:
    """S6: field-hospital telesurgery positioning arm = UR-type 6R (illustrative values)
    + prismatic instrument-insertion stage along the flange z-axis (joint 7).
    At theta = 0 the instrument tip is `shaft` beyond the flange; theta7 >= 0 inserts."""
    W1, W2, L1, L2, H1, H2 = 0.12, 0.09, 0.40, 0.36, 0.16, 0.10
    base = ur_type("tmp", W1, W2, L1, L2, H1, H2, tool=shaft)
    z_b = base.M[:3, 2]                      # flange z-axis at home = +y_s
    S = np.column_stack([base.Slist, Pr(z_b)])
    lim = np.vstack([np.tile([-2 * np.pi, 2 * np.pi], (6, 1)), [0.0, 0.15]])
    flange = base.M[:3, 3] - shaft * z_b
    pts = base.points[:-1] + [(tuple(flange), 6), (tuple(base.M[:3, 3]), 7)]
    r = Robot("Telesurgery 6R + insertion stage P (illustrative)", S, base.M, lim, pts,
              "RRRRRRP", base.q_axis, link_radius=0.05,
              rate_limit=np.r_[np.full(6, 60 * DEG), 0.05], notes="illustrative values")
    r.shaft = shaft
    return r

# ============================================================================
# Scenarios (tasks)
# ============================================================================

def scenario(sid: int) -> Scenario:
    if sid == 1:
        R_handle = np.diag([1.0, -1.0, -1.0])           # handle z_b pointing down
        return Scenario(1, "Upper-limb rehabilitation guide", "biomedical", ur5_rehab(), dict(
            R_d=R_handle,
            circle_centre=np.array([0.50, 0.10, 0.20]), circle_radius=0.15,
            line=(np.array([0.30, -0.20, 0.20]), np.array([0.70, -0.20, 0.20])),
            nominal=np.array([0.50, 0.10, 0.20]),
            seed=np.array([0.0, -1.0, 1.5, -2.07, -1.5708, 0.0]),
            grid=dict(x=(0.10, 0.85), y=(-0.45, 0.50), z=0.20, step=0.05),
            mu1_max=6.0, v_max=0.25,
        ))
    if sid == 2:
        d = np.array([0.0, 1.0, 0.0])                    # spoon axis toward the user
        mouth = np.array([0.40, 0.45, 0.50])
        return Scenario(2, "Feeding / assistive fetch", "biomedical", srs_7r(), dict(
            mouth=mouth, approach_dir=d, stop_distance=0.03,
            R_d=rotz(np.pi / 2),                         # x_b = +y_s, z_b = +z_s (bowl level)
            head_centre=mouth + np.array([0.0, 0.10, 0.05]), head_radius=0.11,
            approach_length=0.15, v_max=0.05, tilt_max=5 * DEG,
            seed=np.array([0.6, -0.4, 0.0, -1.6, 0.0, 0.3, 0.9]),
            clearance_min=0.10, elbow_head_min=0.52,
        ))
    if sid == 3:
        p_e = np.array([0.20, 0.08, 0.03])
        u = np.array([0.0, 0.0, -1.0])
        tilt = 15 * DEG
        dirs = [u, np.array([-np.sin(tilt), 0, -np.cos(tilt)]),
                np.array([0, np.sin(tilt), -np.cos(tilt)])]
        return Scenario(3, "Needle / instrument positioning (RCM)", "biomedical",
                        desktop_6r(), dict(
            entry=p_e, directions=dirs, depth=0.04, standoff=0.005,
            enc_res=360.0 / 4096 * DEG,               # 12-bit encoders
            accuracy_max=1.5e-3,
            seed=np.array([0.3, -0.6, 1.4, -2.4, -1.57, 0.0]),
        ))
    if sid == 4:
        edge = 1.20                                     # UGV standoff: turret axis to vehicle edge
        ground = -0.45
        clearance = 0.30
        obj = np.array([edge + 0.40, 0.0, ground + 0.08])
        return Scenario(4, "EOD arm: under-vehicle reach", "defence", eod_6r(), dict(
            ground_z=ground, edge_x=edge, underside_z=ground + clearance,
            corridor=dict(x=(edge, edge + 0.80), y=(-0.30, 0.30), z=(ground, ground + clearance)),
            object=obj, R_d=np.eye(3), pregrasp_back=0.20, v_approach=0.05,
            margin=0.02, sing_wrist_min=10 * DEG,
            seed_up=np.array([0.0, 0.4, 0.6, 0.0, -0.9, 0.0]),
            seed_down=np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
            standoff_min=0.45,
        ))
    if sid == 5:
        ground = -0.90
        poses = {"A_ground": np.array([0.85, 0.0, ground + 0.15]),
                 "B_mid": np.array([0.95, 0.0, -0.30]),
                 "C_transfer": np.array([0.80, 0.0, 0.20])}
        return Scenario(5, "Casualty-extraction / litter-loading arm", "defence", lift_5r(), dict(
            ground_z=ground, poses=poses, R_d=np.eye(3), mass=100.0, rear_wall_x=-0.10,
            safety_factor=1.5, elbow_min=10 * DEG, v_lift=0.10, tilt_max=5 * DEG,
            seed=np.array([0.0, 0.6, -1.2, 0.6, 0.0]),
        ))
    if sid == 6:
        port = np.array([0.55, 0.30, 0.15])
        depth = 0.08
        return Scenario(6, "Field-hospital telesurgery arm (RCM)", "defence",
                        telesurgery_6r1p(), dict(
            port=port, circle_centre=port + np.array([0, 0, -depth]), circle_radius=0.02,
            rcm_max=2e-3, tip_max=1e-3, scaling=5.0, v_tip_max=0.02,
            seed=np.array([0.3, -0.9, 1.3, -1.9, -1.57, 0.0, 0.05]),
        ))
    raise ValueError(f"unknown scenario {sid}")


ALL = (1, 2, 3, 4, 5, 6)
