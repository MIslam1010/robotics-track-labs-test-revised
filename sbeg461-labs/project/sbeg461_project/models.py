"""Scenario robot models for the SBEG 461 project (Modern Robotics PoE description).

Every robot is described by
  M       home configuration of the end-effector frame {b} in the space frame {s}   (MR Sec. 4.1)
  Slist   6 x n matrix of space screw axes S_i = (w_i, v_i), angular part first       (MR Sec. 4.1.1)
  Mlist   [M_01, M_12, ..., M_n,n+1]: link centre-of-mass frames at home, each
          relative to the previous one; the last entry is the end-effector frame     (MR Sec. 8.3.2)
  Glist   6 x 6 spatial inertia of each link in its CoM frame, G = diag(I_b, m I)    (MR Eq. 8.32)
  g       gravity vector expressed in {s}

All numbers are ILLUSTRATIVE VALUES chosen for teaching, not data of a real product.

Planar robots move in the x-y plane of {s} with joint axes along z, the same convention as
MR Studio and MR Example 4.2.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np

G0 = 9.81


# ----------------------------------------------------------------------------- helpers
def screw_revolute(w, q, h: float = 0.0) -> np.ndarray:
    """Screw axis S = (w, -w x q + h w) of a revolute joint through point q (MR Sec. 3.3.2)."""
    w = np.asarray(w, float)
    q = np.asarray(q, float)
    return np.r_[w, -np.cross(w, q) + h * w]


def trans(p, R=None) -> np.ndarray:
    """Homogeneous transform with rotation R (identity by default) and position p."""
    return mr.RpToTrans(np.eye(3) if R is None else np.asarray(R, float), np.asarray(p, float))


def body_inertia(m: float, Ib) -> np.ndarray:
    """Spatial inertia in a CoM frame, G = diag(I_b, m I3) (MR Eq. 8.32). Ib: 3-vector or 3x3."""
    Ib = np.diag(Ib) if np.ndim(Ib) == 1 else np.asarray(Ib, float)
    G = np.zeros((6, 6))
    G[:3, :3] = Ib
    G[3:, 3:] = m * np.eye(3)
    return G


def rod_inertia_x(m: float, L: float, r: float = 0.03) -> np.ndarray:
    """Uniform solid cylinder of length L along its local x axis (CoM frame at its centre)."""
    Ia = 0.5 * m * r**2
    It = m * (3 * r**2 + L**2) / 12.0
    return body_inertia(m, [Ia, It, It])


def rod_inertia_y(m: float, L: float, r: float = 0.03) -> np.ndarray:
    """Uniform solid cylinder of length L along its local y axis."""
    Ia = 0.5 * m * r**2
    It = m * (3 * r**2 + L**2) / 12.0
    return body_inertia(m, [It, Ia, It])


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Solid box with edge lengths a, b, c along the CoM-frame x, y, z axes."""
    return body_inertia(m, [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12])


def transform_inertia(G_b: np.ndarray, T_ba: np.ndarray) -> np.ndarray:
    """Spatial inertia expressed in frame {a}: G_a = [Ad_Tba]^T G_b [Ad_Tba]  (MR Eq. 8.42)."""
    Ad = mr.Adjoint(T_ba)
    return Ad.T @ G_b @ Ad


def spatial_inertia_to_mci(G: np.ndarray):
    """Inverse of Eq. 8.42 for a frame aligned with the CoM frame: return (m, c, I_c).

    For G_a = [[I_c - m[c][c], m[c]], [-m[c], m I]] (c = CoM position in {a}).
    """
    m = G[3, 3]
    S = G[:3, 3:] / m
    c = np.array([S[2, 1], S[0, 2], S[1, 0]])
    cx = mr.VecToso3(c)
    Ic = G[:3, :3] + m * cx @ cx
    return m, c, Ic


def mlist_from_absolute(com_frames: list, M_ee: np.ndarray) -> list:
    """Build MR's Mlist from absolute home CoM frames M_{s,i} and the end-effector home frame."""
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


# ----------------------------------------------------------------------------- model
@dataclass
class Segment:
    """A collision capsule rigidly attached to link `link` (0 = base), given by its home end points."""
    link: int
    p0: np.ndarray
    p1: np.ndarray
    radius: float


@dataclass
class Scenario:
    key: int
    name: str
    domain: str                      # "biomedical" or "defence"
    description: str
    M: np.ndarray
    Slist: np.ndarray
    Mlist: list
    Glist: list
    g: np.ndarray
    tau_max: np.ndarray
    qdot_max: np.ndarray
    q_limits: np.ndarray             # n x 2
    q_home: np.ndarray
    planar: bool = False
    segments: list = field(default_factory=list)
    obstacles: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    components: tuple = ()           # which of: dynamics, trajectory, planning, control

    # -- basics -------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.Slist.shape[1]

    def fk(self, q) -> np.ndarray:
        """End-effector pose T_sb(q) by the space-form PoE (MR Eq. 4.14)."""
        return mr.FKinSpace(self.M, self.Slist, np.asarray(q, float))

    def jacobian_space(self, q) -> np.ndarray:
        return mr.JacobianSpace(self.Slist, np.asarray(q, float))

    def jacobian_body(self, q) -> np.ndarray:
        Blist = mr.Adjoint(mr.TransInv(self.M)) @ self.Slist
        return mr.JacobianBody(Blist, np.asarray(q, float))

    def link_transforms(self, q) -> list:
        """[T_0, T_1, ..., T_n] with T_i = e^[S1]q1 ... e^[Si]qi (T_0 = I); a point fixed to link i
        that is at p0 in the home configuration is at T_i p0 in configuration q."""
        T = np.eye(4)
        out = [T.copy()]
        for i in range(self.n):
            T = T @ mr.MatrixExp6(mr.VecTose3(self.Slist[:, i] * q[i]))
            out.append(T.copy())
        return out

    def com_frames_home(self) -> list:
        T = np.eye(4)
        out = []
        for i in range(self.n):
            T = T @ self.Mlist[i]
            out.append(T.copy())
        return out

    def total_mass(self) -> float:
        return float(sum(G[3, 3] for G in self.Glist))

    def copy(self) -> "Scenario":
        return copy.deepcopy(self)

    # -- payloads and model error ---------------------------------------------
    def with_payload(self, m: float, Ib=(0.0, 0.0, 0.0), p_e=(0.0, 0.0, 0.0)) -> "Scenario":
        """Return a copy whose last link carries a rigid payload of mass m, rotational inertia Ib
        (about its own CoM, axes aligned with {b}) and CoM at p_e in the end-effector frame {b}."""
        s = self.copy()
        if m == 0.0:
            return s
        T_np = s.Mlist[-1] @ trans(p_e)                 # payload CoM frame in last-link CoM frame
        G_p = body_inertia(m, Ib)
        s.Glist[-1] = s.Glist[-1] + transform_inertia(G_p, mr.TransInv(T_np))
        return s

    def scaled(self, mass_scale=1.0, com_shift=(0.0, 0.0, 0.0)) -> "Scenario":
        """Controller model with parameter error.

        mass_scale : scalar or per-link list; multiplies each link's mass and rotational inertia.
        com_shift  : 3-vector (or per-link list of 3-vectors) in metres; moves each link's CoM
                     frame by this offset expressed in that CoM frame. Other frames do not move.
        """
        s = self.copy()
        k = np.broadcast_to(np.asarray(mass_scale, float), (self.n,))
        d = np.broadcast_to(np.asarray(com_shift, float), (self.n, 3))
        for i in range(self.n):
            s.Glist[i] = s.Glist[i] * k[i]
            if np.any(d[i]):
                shift = trans(d[i])
                s.Mlist[i] = s.Mlist[i] @ shift
                s.Mlist[i + 1] = mr.TransInv(shift) @ s.Mlist[i + 1]
        return s


# ----------------------------------------------------------------------------- scenarios
def scenario_1_rehab() -> Scenario:
    """S1 (biomedical) Planar end-effector upper-limb rehabilitation robot, horizontal plane."""
    L1, L2, m1, m2 = 0.35, 0.35, 2.4, 1.6
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, 1], [L1, 0, 0])]
    M = trans([L1 + L2, 0, 0])
    com = [trans([L1 / 2, 0, 0]), trans([L1 + L2 / 2, 0, 0])]
    G = [rod_inertia_x(m1, L1), rod_inertia_x(m2, L2)]
    A, B = np.array([0.0, 0.35]), np.array([0.0, 0.55])
    return Scenario(
        1, "Rehabilitation robot with a virtual-tunnel impedance", "biomedical",
        "Two-link end-effector robot moving the patient's hand in a horizontal plane "
        "(MIT-Manus-type therapy). Gravity is normal to the plane of motion.",
        M, S, mlist_from_absolute(com, M), G, np.array([0, 0, -G0]),
        tau_max=np.array([20.0, 12.0]), qdot_max=np.array([3.0, 3.0]),
        q_limits=np.array([[0.0, np.pi], [0.2, 2.8]]), q_home=np.array([0.9, 1.6]),
        planar=True,
        segments=[Segment(1, np.zeros(3), np.array([L1, 0, 0]), 0.03),
                  Segment(2, np.array([L1, 0, 0]), np.array([L1 + L2, 0, 0]), 0.03)],
        params=dict(L=(L1, L2), path_A=A, path_B=B, T_reach=2.0,
                    K_normal=500.0, B_normal=15.0, B_tangent=5.0, F_max=40.0,
                    patient_K=100.0, patient_B=5.0, pulse_F=10.0, pulse_T=0.3,
                    elbow_sign=+1),
        components=("dynamics", "trajectory", "control"))


def scenario_2_gait() -> Scenario:
    """S2 (biomedical) Hip-knee exoskeleton leg tracking a gait trajectory (swing leg on a test stand)."""
    L1, L2 = 0.42, 0.43                       # thigh, shank (knee to ankle)
    m1, c1, I1 = 9.0, 0.18, 0.16              # thigh + exo thigh module (mass kg, CoM below hip m, Izz kg m^2)
    m2, c2, I2 = 5.8, 0.24, 0.10              # shank + foot + exo shank module
    # leg hangs along -y at q = 0; hip flexion about +z, knee flexion about -z (both positive = flexion)
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, -1], [0, -L1, 0])]
    M = trans([0, -(L1 + L2), 0])
    com = [trans([0, -c1, 0]), trans([0, -(L1 + c2), 0])]
    G = [body_inertia(m1, [I1, 0.03, I1]), body_inertia(m2, [I2, 0.01, I2])]
    return Scenario(
        2, "Exoskeleton gait tracking", "biomedical",
        "Sagittal-plane hip-knee exoskeleton coupled to the wearer's leg (combined segment "
        "inertias), suspended on a test stand so the swing leg follows a gait-like trajectory.",
        M, S, mlist_from_absolute(com, M), G, np.array([0, -G0, 0]),
        tau_max=np.array([60.0, 45.0]), qdot_max=np.array([8.0, 10.0]),
        q_limits=np.array([[-0.5, 1.2], [0.0, 1.6]]), q_home=np.array([0.0, 0.0]),
        planar=True,
        segments=[Segment(1, np.zeros(3), np.array([0, -L1, 0]), 0.06),
                  Segment(2, np.array([0, -L1, 0]), np.array([0, -L1 - L2, 0]), 0.05)],
        params=dict(L=(L1, L2), gait_period=1.2, n_cycles=10,
                    model_error=dict(mass_scale=0.8, com_shift=(0.0, -0.02, 0.0)),
                    gear_ratio=80.0, rotor_inertia=1.5e-5),
        components=("dynamics", "trajectory", "control"))


def _ur5_core():
    """UR5 screw axes, link frames and inertias as in MR Example 4.5 and the MR Ch. 8 software."""
    W1, W2, L1, L2, H1, H2 = 0.10915, 0.0823, 0.425, 0.39225, 0.089159, 0.09465
    Slist = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, -1, 0],
                      [0, -H1, -H1, -H1, -W1, H2 - H1],
                      [0, 0, 0, 0, L1 + L2, 0],
                      [0, 0, L1, L1 + L2, 0, L1 + L2]], float)
    Mlist = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])]
    Glist = [np.diag([0.010267495893, 0.010267495893, 0.00666, 3.7, 3.7, 3.7]),
             np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393]),
             np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])]
    return Slist, Mlist, Glist


def scenario_3_surgical() -> Scenario:
    """S3 (biomedical) UR5-class arm holding a laparoscope: RRT + time-optimal scaling + task-space
    control under a remote-centre-of-motion (RCM) constraint."""
    Slist, Mlist, Glist = _ur5_core()
    L_tool, m_tool = 0.30, 0.6
    Mlist = [m.copy() for m in Mlist]
    Mlist[-1] = Mlist[-1] @ trans([0, 0, L_tool])          # {b} at the tool tip, z_b along the shaft
    s = Scenario(
        3, "Surgical positioning arm with an RCM constraint", "biomedical",
        "6R UR5-class collaborative arm (MR Example 4.5 kinematics and MR Ch. 8 software inertias) "
        "holding a 0.30 m laparoscope; the shaft must pivot about the trocar port (RCM).",
        np.eye(4), Slist, Mlist, Glist, np.array([0, 0, -G0]),
        tau_max=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]),
        qdot_max=np.full(6, 1.0),                         # safety-reduced speed near the patient
        q_limits=np.tile([-2 * np.pi, 2 * np.pi], (6, 1)),
        q_home=np.array([2.6, -2.5, 2.3, -1.4, -1.57, 0.0]))      # parked, turned away from the table
    M = np.eye(4)
    for Mi in Mlist:
        M = M @ Mi
    s.M = M
    # the tool: 0.6 kg holder + scope, CoM 0.10 m along the shaft from the flange
    s = s.with_payload(m_tool, Ib=(0.004, 0.004, 0.0005), p_e=(0, 0, -(L_tool - 0.10)))
    # collision capsules (home end points from the joint-axis geometry of MR Example 4.5)
    W1, W2, L1, L2, H1, H2 = 0.10915, 0.0823, 0.425, 0.39225, 0.089159, 0.09465
    s.segments = [
        Segment(1, np.array([0, 0, 0.0]), np.array([0, 0, H1]), 0.06),
        Segment(2, np.array([0, 0.07, H1]), np.array([L1, 0.07, H1]), 0.06),
        Segment(3, np.array([L1, 0.02, H1]), np.array([L1 + L2, 0.02, H1]), 0.045),
        Segment(4, np.array([L1 + L2, 0.0, H1]), np.array([L1 + L2, W1, H1]), 0.045),
        Segment(5, np.array([L1 + L2, W1, H1]), np.array([L1 + L2, W1, H1 - H2]), 0.045),
        Segment(6, np.array([L1 + L2, W1, H1 - H2]), np.array([L1 + L2, W1 + W2, H1 - H2]), 0.04),
        Segment(6, np.array([L1 + L2, W1 + W2, H1 - H2]),
                np.array([L1 + L2, W1 + W2 + L_tool, H1 - H2]), 0.006),
    ]
    s.obstacles = dict(
        ground_z=-0.75,
        spheres=[(np.array([0.15, 0.45, 0.45]), 0.12),        # overhead surgical light head
                 (np.array([-0.45, 0.30, 0.25]), 0.12)],      # anaesthesia monitor
        boxes=[(np.array([0.20, 0.15, -0.30]), np.array([0.85, 0.65, 0.05]))])   # patient torso on table
    s.params = dict(L_tool=L_tool, m_tool=m_tool, p_rcm=np.array([0.45, 0.35, 0.05]),
                    R_pre=np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]),   # shaft (z_b) pointing down
                    q_pre=np.array([0.4684, -1.3990, 1.3878, -1.5596, -1.5708, -1.1024]),
                    pre_insertion_height=0.08, insertion_depth=0.08, scan_radius=0.02,
                    scan_period=4.0, rcm_tol=0.002)
    s.components = ("dynamics", "trajectory", "planning", "control")
    return s


def scenario_4_load_exo() -> Scenario:
    """S4 (defence) Soldier load-carriage exoskeleton, stance leg: knee and hip joints carrying a
    backpack frame. Bench-test model: exo shank strut clamped to the boot plate (ankle locked)."""
    Ls, Lt = 0.45, 0.45                       # shank (ankle-knee), thigh (knee-hip) struts
    m1, I1 = 2.5, 0.045                       # exo thigh strut
    m2, I2 = 34.0, 1.20                       # hip frame 4 kg + backpack 30 kg, Izz about CoM
    pack_com = np.array([-0.15, 0.25, 0.0])  # relative to the hip joint (behind and above)
    knee, hip = np.array([0, Ls, 0.0]), np.array([0, Ls + Lt, 0.0])
    # knee flexion rotates the thigh about +z; hip flexion (trunk forward) about -z
    S = np.c_[screw_revolute([0, 0, 1], knee), screw_revolute([0, 0, -1], hip)]
    M = trans(hip + np.array([0, 0.50, 0]))  # {b} on the pack frame, 0.50 m above the hip
    com = [trans(knee + np.array([0, Lt / 2, 0])), trans(hip + pack_com)]
    G = [body_inertia(m1, [I1, 0.005, I1]), body_inertia(m2, [I2, 0.6, I2])]
    return Scenario(
        4, "Load-carriage exoskeleton tracking with load and model error", "defence",
        "Passive-ankle, powered knee-hip load-carriage exoskeleton that transfers a backpack load "
        "to the ground. Stance-leg bench model with the shank strut clamped (illustrative values).",
        M, S, mlist_from_absolute(com, M), G, np.array([0, -G0, 0]),
        tau_max=np.array([200.0, 150.0]), qdot_max=np.array([6.0, 6.0]),
        q_limits=np.array([[0.0, 1.6], [-0.3, 1.8]]), q_home=np.array([0.0, 0.0]),
        planar=True,
        segments=[Segment(1, knee, hip, 0.05), Segment(2, hip, hip + np.array([0, 0.5, 0]), 0.12)],
        params=dict(L=(Ls, Lt), pack_mass_nominal=30.0, pack_mass_range=(20.0, 40.0),
                    pack_com=pack_com, frame_mass=4.0, cycle_period=2.4, knee_range=(0.10, 0.60),
                    trunk_lean=0.15, model_error=dict(pack_mass_actual=37.5, com_shift=(-0.03, 0.0, 0.0))),
        components=("dynamics", "trajectory", "control"))


def scenario_5_eod() -> Scenario:
    """S5 (defence) 6R EOD manipulator on a tracked base: plan, time-scale and track a payload transfer
    under base-motion disturbance."""
    z0 = 0.25                                   # shoulder height above the turret base
    xs, xe, xr, xw, xg = 0.0, 0.60, 0.70, 1.15, 1.35   # shoulder, elbow, roll, wrist pitch, grip centre
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]),         # 1 turret yaw
              screw_revolute([0, 1, 0], [xs, 0, z0]),        # 2 shoulder pitch
              screw_revolute([0, 1, 0], [xe, 0, z0]),        # 3 elbow pitch
              screw_revolute([1, 0, 0], [xr, 0, z0]),        # 4 forearm roll
              screw_revolute([0, 1, 0], [xw, 0, z0]),        # 5 wrist pitch
              screw_revolute([1, 0, 0], [xw, 0, z0])]        # 6 gripper roll
    M = trans([xg, 0, z0])
    com = [trans([0, 0, 0.12]), trans([0.30, 0, z0]), trans([0.65, 0, z0]),
           trans([0.93, 0, z0]), trans([1.20, 0, z0]), trans([1.30, 0, z0])]
    G = [box_inertia(6.0, 0.25, 0.25, 0.25), rod_inertia_x(5.0, 0.60, 0.05),
         rod_inertia_x(1.5, 0.15, 0.05), rod_inertia_x(2.5, 0.45, 0.04),
         box_inertia(1.0, 0.10, 0.08, 0.08), box_inertia(1.5, 0.20, 0.12, 0.06)]
    seg = [Segment(1, np.array([0, 0, 0.12]), np.array([0, 0, z0]), 0.10),
           Segment(2, np.array([0, 0, z0]), np.array([xe, 0, z0]), 0.06),
           Segment(3, np.array([xe, 0, z0]), np.array([xr, 0, z0]), 0.06),
           Segment(4, np.array([xr, 0, z0]), np.array([xw, 0, z0]), 0.05),
           Segment(6, np.array([xw, 0, z0]), np.array([xg, 0, z0]), 0.06)]
    return Scenario(
        5, "EOD manipulator on a tracked base", "defence",
        "Generic 6R explosive-ordnance-disposal arm on a tracked UGV deck (yaw, shoulder, elbow, "
        "forearm roll, wrist pitch, gripper roll). Illustrative values, not a specific product.",
        M, S, mlist_from_absolute(com, M), G, np.array([0, 0, -G0]),
        tau_max=np.array([120.0, 250.0, 120.0, 30.0, 40.0, 15.0]),
        qdot_max=np.array([1.0, 1.0, 1.0, 1.5, 1.5, 2.0]),
        q_limits=np.array([[-np.pi, np.pi], [-1.9, 1.2], [-2.8, 2.8],
                           [-np.pi, np.pi], [-2.0, 2.0], [-np.pi, np.pi]]),
        q_home=np.array([0.0, -0.35, -2.5, 0.0, 1.2, 0.0]),     # stowed: forearm folded over the arm
        segments=seg,
        obstacles=dict(
            ground_z=-0.40,
            spheres=[(np.array([-0.15, 0.40, 0.55]), 0.15)],          # camera mast head
            boxes=[(np.array([-0.90, -0.40, -0.40]), np.array([0.45, 0.40, -0.05])),   # hull + tracks
                   (np.array([0.70, -0.70, -0.40]), np.array([0.90, -0.10, 0.10])),    # culvert wall
                   (np.array([0.80, 0.05, 0.25]), np.array([1.40, 0.65, 0.35]))]),     # overhanging slab
        params=dict(payload_mass=5.0, payload_size=(0.20, 0.15, 0.15), payload_mass_error=0.2,
                    grasp_point=np.array([1.00, 0.35, -0.32]), bin_point=np.array([-0.50, -0.20, 0.20]),
                    pregrasp_offset=0.10,
                    R_grip=np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]),   # approach axis x_b down
                    q_pregrasp=np.array([0.3367, -0.0518, 0.6310, 0.0, 0.9916, 0.3367]),
                    q_grasp=np.array([0.3367, 0.1253, 0.4408, 0.0, 1.0046, 0.3367]),
                    q_bin=np.array([0.3805, -1.8834, -2.1298, 0.0, -0.6993, 0.3805]),
                    slope_deg=10.0, heave_amp=1.0, heave_freq=2.0,
                    tip_tol=0.010, tip_tol_disturbed=0.015, ori_tol_deg=2.0, clearance=0.05),
        components=("dynamics", "trajectory", "planning", "control"))


def scenario_6_casualty() -> Scenario:
    """S6 (defence / medical) Casualty-handling arm with impedance control for safe human contact.
    Planar 3R in the vertical plane; a padded cradle at the tip supports a casualty's limb."""
    L = (0.45, 0.40, 0.15)
    m = (4.0, 2.5, 1.2)
    m_cradle = 0.5
    p = np.cumsum((0.0,) + L)
    S = np.column_stack([screw_revolute([0, 0, 1], [p[i], 0, 0]) for i in range(3)])
    M = trans([p[3], 0, 0])
    com = [trans([p[i] + L[i] / 2, 0, 0]) for i in range(3)]
    G = [rod_inertia_x(m[i], L[i], 0.04) for i in range(3)]
    s = Scenario(
        6, "Casualty-handling arm with impedance control", "defence",
        "Three-link arm on a casualty-evacuation stretcher platform that slides a padded cradle "
        "under a casualty's lower leg and lifts it into a splint. Contact with the person is "
        "expected; impedance control keeps forces safe.",
        M, S, mlist_from_absolute(com, M), G, np.array([0, -G0, 0]),
        tau_max=np.array([150.0, 80.0, 25.0]), qdot_max=np.array([1.5, 1.5, 2.0]),
        q_limits=np.array([[-1.2, 1.2], [-2.5, 2.5], [-2.5, 2.5]]),
        q_home=np.array([0.3, -0.9, 0.6]), planar=True,
        segments=[Segment(i + 1, np.array([p[i], 0, 0]), np.array([p[i + 1], 0, 0]), 0.04)
                  for i in range(3)])
    s = s.with_payload(m_cradle, Ib=(1e-3, 1e-3, 1e-3))
    s.params = dict(L=L, limb_mass=6.0, tissue_K=4000.0, tissue_B=40.0, lift_height=0.15,
                    lift_time=3.0, K_imp=300.0, B_imp=40.0, K_rot=30.0, B_rot=3.0,
                    wall_x=0.80, wall_K=(2000.0, 20000.0), wall_B=50.0, approach_speed=0.10,
                    F_peak_max=50.0, F_static_max=25.0, v_max=0.25)
    s.components = ("dynamics", "trajectory", "control")
    return s


SCENARIOS = {1: scenario_1_rehab, 2: scenario_2_gait, 3: scenario_3_surgical,
             4: scenario_4_load_exo, 5: scenario_5_eod, 6: scenario_6_casualty}


def load_scenario(k: int) -> Scenario:
    """Return a fresh copy of scenario k (1..6)."""
    if k not in SCENARIOS:
        raise KeyError(f"unknown scenario {k}; choose 1..6")
    return SCENARIOS[k]()
