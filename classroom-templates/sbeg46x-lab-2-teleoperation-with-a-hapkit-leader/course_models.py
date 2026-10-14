"""Local copy of the two course robot models used in Lab 2 (numbers identical to Course resources:
"Course robot models and hardware safety checklist").

* hapkit()      -- Hapkit-type 1-DOF haptic paddle (leader, and the simulated follower)
* eod_6r_ugv()  -- UGV-mounted EOD 6R arm (kinematic), used in Part E for the follower speed limit

All values are ILLUSTRATIVE teaching values, not the data of a product.
Notation follows Modern Robotics: screw axes S = (omega, v), space-form PoE T = e^[S1]th1 ... e^[Sn]thn M.
"""
import numpy as np

DEG = np.pi / 180.0
G0 = 9.81


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


def _adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = _skew(p) @ R
    return Ad


def _screw_revolute(w, q):
    w = np.asarray(w, float)
    return np.r_[w, -np.cross(w, np.asarray(q, float))]


def _trans(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


# copied from the course robot models (Course resources), function hapkit; numbers identical
def hapkit() -> dict:
    """Hapkit-type 1-DOF haptic paddle.  Handle pivots about z_s; the grip is rh = 0.090 m from the pivot;
    sector radius rs = 0.073 m, motor pulley rp = 0.0042 m (ratio rs/rp ~ 17.4).  Handle displacement x ~ rh*theta.
    Derated limits: handle force <= 2 N (saturated in firmware), watchdog cuts the motor outside +-40 mm."""
    rh, rs, rp = 0.090, 0.073, 0.0042
    S = _screw_revolute([0, 0, 1], [0, 0, 0]).reshape(6, 1)
    M = _trans([0, rh, 0])
    return dict(
        key="hapkit", Slist=S, M=M, illustrative=True,
        mass=0.10,                         # moving mass of handle + sector (kg)
        com=np.array([0.0, -0.010, 0.0]),  # centre of mass at home in {s} (m), slightly below the pivot
        I_com=np.diag([2.0e-4, 1.0e-4, 1.5e-4]),   # rotational inertia about the CoM (kg m^2)
        g=np.array([0.0, -G0, 0.0]),
        limits=np.array([[-0.040 / rh, 0.040 / rh]]),
        params=dict(rh=rh, rs=rs, rp=rp, ratio=rs / rp, rotor_inertia=1.0e-6,
                    reflected_rotor_inertia=1.0e-6 * (rs / rp) ** 2,
                    force_max=2.0, x_range=0.040, loop_rate_hz=1000.0),
    )


# copied from the course robot models (Course resources), function eod_6r_ugv; numbers identical
def eod_6r_ugv(shoulder=(0.10, 0.20), L1=0.80, L2=0.75, Lg=0.25) -> dict:
    """UGV-mounted EOD 6R arm (yaw, shoulder pitch, elbow pitch, forearm roll, wrist pitch, gripper roll;
    spherical wrist).  {s} on the turret base; at home the arm is stretched along +x_s and {b} (gripper) is
    parallel to {s}, so the gripper approach axis is x_b.  Rated joint speed 30 deg/s; ready pose
    theta_r = (0, 30, -70, 0, 10, 0) deg.  q = 0 is singular: never use it as a working pose."""
    a, h = shoulder
    sh = np.array([a, 0, h])
    el = sh + [L1, 0, 0]
    wr = el + [L2, 0, 0]
    w = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    q = [np.zeros(3), sh, el, wr, wr, wr]
    S = np.column_stack([_screw_revolute(wi, qi) for wi, qi in zip(w, q)])
    M = _trans(wr + [Lg, 0, 0])
    B = _adjoint(np.linalg.inv(M)) @ S
    lim = np.array([[-180, 180], [-60, 110], [-160, 160], [-180, 180], [-115, 115], [-180, 180]], float) * DEG
    return dict(key="eod_6r_ugv", Slist=S, Blist=B, M=M, limits=lim, illustrative=True,
                qdot_max=np.full(6, 30 * DEG),
                params=dict(shoulder=shoulder, L1=L1, L2=L2, Lg=Lg,
                            q_stow=np.array([-90, 70, -150, 0, 50, 0], float) * DEG,
                            q_ready=np.array([0, 30, -70, 0, 10, 0], float) * DEG,
                            qddot_max=60 * DEG))
