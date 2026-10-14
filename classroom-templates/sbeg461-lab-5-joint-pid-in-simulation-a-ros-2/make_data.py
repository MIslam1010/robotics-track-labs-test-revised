"""Regenerate the data files of SBEG 461 Lab 5 (joint PID; ROS 2 controller node).

    python make_data.py

Writes
  data/joint_plants.json    five single-joint plants  tau = M q_dd + b q_d + g(q)  (MR Eq. 11.21) with
                            g(q) = g_cos cos q + g_sin sin q (exact for one rigid body turning about a fixed
                            horizontal axis; zero for a vertical axis):
                              arm_j1   course arm, base yaw joint 1 (vertical axis, g = 0), other joints locked
                                       at q_home, 0.25 kg payload        -- from course_arm.py (= shared model)
                              arm_j2   course arm, shoulder joint 2 (horizontal axis), same locking and payload
                              knee     transfemoral-prosthesis knee, thigh hanging (leg_prosthesis_2r, link 2)
                              eod_wrist  EOD arm wrist pitch with the 2 kg package (eod_3r, link 3), forearm
                                       horizontal (q1 + q2 = 0)
                              mr_link  the single link of MR Sec. 11.4.1 (M = 0.5, m = 1, r = 0.1, b = 0.1)
  data/knee_swing.csv       an illustrative knee swing-phase reference (t, qd, qd_dot, qd_ddot)

M and g_cos/g_sin of the course-arm plants are computed with modern_robotics (MassMatrix, GravityForces) from
course_arm.py, a copy of course_arm of the shared course robot models (Course resources) with identical numbers; the knee and EOD
numbers are those of the shared course robot models leg_prosthesis_2r and eod_3r.  Friction b, torque and speed limits of the
knee and EOD plants are ILLUSTRATIVE VALUES (this file).  The lab itself is self-contained.
"""
import csv
import json
from pathlib import Path

import modern_robotics as mr
import numpy as np

import course_arm as ca

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
G0 = 9.81
PAYLOAD = 0.25                     # kg, course-arm maximum payload (as in Labs 9 and 11)


def arm_joint_plant(j: int) -> dict:
    """One joint of the course arm with the other joints locked at q_home (payload 0.25 kg)."""
    arm = ca.course_arm(payload=PAYLOAD)
    q0 = arm["q_home"].copy()

    def grav(qj):
        q = q0.copy()
        q[j] = qj
        return float(mr.GravityForces(q, arm["g"], arm["Mlist"], arm["Glist"], arm["Slist"])[j])

    Mjj = float(mr.MassMatrix(q0, arm["Mlist"], arm["Glist"], arm["Slist"])[j, j])
    g_cos, g_sin = grav(0.0), grav(np.pi / 2)
    if abs(g_cos) < 1e-12 and abs(g_sin) < 1e-12:
        g_cos = g_sin = 0.0
    return dict(M=Mjj, b=0.01, g_cos=g_cos, g_sin=g_sin,
                tau_max=round(float(arm["derated_current_fraction"] * arm["tau_max"][j]), 12),
                qdot_max=round(float(arm["derated_speed_fraction"] * arm["qdot_max"][j]), 12),
                q_limits=[float(v) for v in arm["limits"][j]],
                locked_q=[float(v) for v in q0], joint_index=j + 1, payload=PAYLOAD)


def main():
    DATA.mkdir(exist_ok=True)
    j1, j2 = arm_joint_plant(0), arm_joint_plant(1)
    j1.update(name="arm_j1", description=(
        "Course arm, base yaw joint 1 (vertical axis: no gravity torque, the MR 'horizontal plane' case), other "
        "joints locked at q_home = (0, -0.6, 1.4, -2.4, -1.57, 0) rad, 0.25 kg payload at the tool point. Torque and "
        "speed limits derated to 60 % / 50 % of the rated values (Course resources: Course robot models and hardware safety checklist). b illustrative."),
        source="course_arm.py = shared course robot model course_arm(payload=0.25); M = MassMatrix[0, 0]")
    j2.update(name="arm_j2", description=(
        "Course arm, shoulder joint 2 (horizontal axis), other joints locked at q_home, 0.25 kg payload. "
        "g(q) = g_cos cos q + g_sin sin q from modern_robotics GravityForces. Derated limits. b illustrative."),
        source="course_arm.py = shared course robot model course_arm(payload=0.25); M = MassMatrix[1, 1]")
    m, r, Izz = 2.4, 0.21, 0.045                    # shared model leg_prosthesis_2r, link 2 (shank + prosthesis)
    knee = dict(name="knee", M=Izz + m * r**2, b=0.3, g_cos=0.0, g_sin=m * G0 * r, tau_max=40.0, qdot_max=8.0,
                q_limits=[-0.1, 2.0], description=(
                    "Powered transfemoral prosthesis, knee joint, thigh hanging vertically (hip q1 = -90 deg locked): "
                    "q = knee flexion = -theta2 of leg_prosthesis_2r (positive = flexion; 0 = straight leg), g(q) = m g r sin q. Shank + prosthesis 2.4 kg, CoM 0.21 m "
                    "below the knee, I_zz = 0.045 kg m^2 about the CoM. b, torque and speed limits illustrative."),
                source="shared course robot model leg_prosthesis_2r (461 Week 3 'leg_2r'); M = I + m r^2")
    m, r, Izz = 3.5, 0.10, 0.008                    # shared model eod_3r, link 3 (wrist + gripper + 2 kg package)
    eod = dict(name="eod_wrist", M=Izz + m * r**2, b=0.2, g_cos=m * G0 * r, g_sin=0.0, tau_max=15.0, qdot_max=2.0,
               q_limits=[-2.0, 2.0], description=(
                   "EOD robot arm (protective use), wrist pitch joint 3 carrying the 2 kg suspect package, shoulder "
                   "and elbow locked with the forearm horizontal (q1 + q2 = 0): g(q) = m g r cos q. Link 3 = 3.5 kg, "
                   "CoM 0.10 m from the wrist axis, I_zz = 0.008 kg m^2. b, torque and speed limits illustrative."),
               source="shared course robot model eod_3r (461 Weeks 1-3); M = I + m r^2")
    mrl = dict(name="mr_link", M=0.5, b=0.1, g_cos=1.0 * G0 * 0.1, g_sin=0.0, tau_max=None, qdot_max=None,
               q_limits=[-np.pi, np.pi], description=(
                   "The single-joint robot of MR Sec. 11.4.1 (Eqs. 11.19-11.21, p. 424): M = 0.5 kg m^2, m = 1 kg, "
                   "r = 0.1 m, b = 0.1 N m s/rad, vertical plane (g = 9.81). Used to reproduce MR Fig. 11.13."),
               source="Modern Robotics Sec. 11.4.1, printed p. 424 (not illustrative)")
    plants = {p["name"]: p for p in (j1, j2, knee, eod, mrl)}
    (DATA / "joint_plants.json").write_text(json.dumps(plants, indent=2))

    # Illustrative knee swing phase: quintic 0.087 -> 1.047 rad (5 -> 60 deg) in 0.25 s, back in 0.25 s, hold 0.2 s.
    dt = 0.001
    t = np.arange(0.0, 0.7 + dt / 2, dt)
    qa, qb, T = np.deg2rad(5.0), np.deg2rad(60.0), 0.25

    def quintic(tau):
        tau = np.clip(tau, 0.0, 1.0)
        return (10 * tau**3 - 15 * tau**4 + 6 * tau**5, (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T,
                (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2)

    s1, v1, a1 = quintic(t / T)
    s2, v2, a2 = quintic((t - T) / T)
    up = t <= T
    qd = np.where(up, qa + (qb - qa) * s1, qb - (qb - qa) * s2)
    vd = np.where(up, (qb - qa) * v1, -(qb - qa) * v2)
    ad = np.where(up, (qb - qa) * a1, -(qb - qa) * a2)
    with open(DATA / "knee_swing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "qd_rad", "qd_dot_rad_s", "qd_ddot_rad_s2"])
        for row in zip(t, qd, vd, ad):
            w.writerow([f"{row[0]:.3f}"] + [f"{x:.9f}" for x in row[1:]])
    print("wrote", DATA / "joint_plants.json", "and", DATA / "knee_swing.csv")
    for p in plants.values():
        print(f"  {p['name']:9s} M={p['M']:.6f} b={p['b']} g_cos={p['g_cos']:.5f} g_sin={p['g_sin']:.5f} "
              f"tau_max={p['tau_max']} qdot_max={p['qdot_max']}")


if __name__ == "__main__":
    main()
