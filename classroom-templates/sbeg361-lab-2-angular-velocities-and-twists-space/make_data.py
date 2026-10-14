"""Generate the synthetic data sets for SBEG 361 Lab 2 (illustrative values, deterministic).

1. nav_tree.json  -- static transform tree of an optical surgical-navigation set-up
   (biomedical case: navigated pedicle-screw insertion in spinal surgery):
       theatre -> tracker                 ceiling/boom-mounted optical tracking camera
       tracker -> patient_ref             dynamic reference frame (marker array) clamped to the spine
       patient_ref -> ct_image            image-to-patient registration result
       ct_image -> plan_entry             planned screw trajectory: origin = entry point, +z = into bone
       tool_marker -> tool_tip            screwdriver calibration (tip offset in the marker frame)
   The edge tracker -> tool_marker is time-varying and comes from tool_track.csv.

2. tool_track.csv -- T_tracker_tool_marker(t) of the navigated screwdriver while the surgeon
   drives the screw: a constant screw motion about the shaft axis with pitch h = lead / (2 pi).
   Columns: t, r11..r33 (row-major), px, py, pz   (metres, seconds).

3. eod_tree.json -- transform tree of a tracked EOD robot (defence case, protective use), with
   the same numbers as Week 2 Workshop Problem 4:
       odom -> base_link -> mast_camera -> package,   base_link -> gripper

Ground truth used by make_data (students should NOT read this before the lab):
   screw lead 2.5 mm/rev, speed pi/2 rad/s for 8 s (2 turns, 5 mm advance);
   actual axis = planned axis tilted by 2.0 deg, entry point offset 1.0 mm from plan.
Run:  python make_data.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import modern_robotics as mr
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
d2r = np.deg2rad

LEAD = 0.0025                       # m per revolution (illustrative pedicle-screw lead)
H = LEAD / (2 * np.pi)              # pitch, m/rad
THETADOT = np.pi / 2                # rad/s
DT, T_END = 0.02, 8.0               # 50 Hz tracker, 8 s
TILT_DEG, OFFSET_MM = 2.0, 1.0      # actual vs planned trajectory
Q_M = np.array([0.05, 0.0, 0.0])    # a point of the shaft axis in the marker frame (m)
S_M = np.array([0.0, 0.0, -1.0])    # shaft direction (towards the tip) in the marker frame
TIP_M = np.array([0.05, 0.0, -0.25])  # calibrated tip position in the marker frame (on the axis)


def rot(axis, ang):
    a = np.asarray(axis, float)
    return mr.MatrixExp3(mr.VecToso3(a / np.linalg.norm(a) * ang))


def tf_entry(parent, child, T):
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {"parent": parent, "child": child,
            "translation": [round(float(v), 9) for v in T[:3, 3]],
            "rotation_xyzw": [round(float(v), 12) for v in (x, y, z, w)]}


def frame_with_z(z_axis, hint=(1.0, 0.0, 0.0)):
    """Rotation matrix whose third column is z_axis (unit), x roughly along hint."""
    z = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    x = np.asarray(hint, float) - (np.asarray(hint, float) @ z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------ navigation scene
T_theatre_tracker = mr.RpToTrans(rot([0, 0, 1], d2r(150)) @ rot([1, 0, 0], d2r(-130)), [1.2, -0.8, 2.1])
T_theatre_pref = mr.RpToTrans(rot([0, 0, 1], d2r(10)), [0.3, 0.1, 1.05])
T_pref_ct = mr.RpToTrans(rot([1, 0, 0], np.pi) @ rot([0, 0, 1], d2r(5)), [-0.12, 0.05, -0.18])
entry_ct = np.array([0.142, 0.085, 0.210])
d_plan = np.array([0.30, -0.25, -0.92]); d_plan /= np.linalg.norm(d_plan)
T_ct_plan = mr.RpToTrans(frame_with_z(d_plan), entry_ct)
T_marker_tip = mr.RpToTrans(np.eye(3), TIP_M)

# actual trajectory: tilt about the plan's x-axis, offset along the plan's y-axis
R_plan = T_ct_plan[:3, :3]
d_act = rot(R_plan[:, 0], d2r(TILT_DEG)) @ d_plan
entry_act = entry_ct + R_plan[:, 1] * OFFSET_MM / 1000

R_ct_marker = frame_with_z(-d_act, hint=(0.0, 1.0, 0.0))          # marker z opposite the shaft
p_ct_marker = entry_act - R_ct_marker @ TIP_M                       # tip starts at the actual entry
T_ct_marker0 = mr.RpToTrans(R_ct_marker, p_ct_marker)

T_tracker_ct = mr.TransInv(T_theatre_tracker) @ T_theatre_pref @ T_pref_ct
T_tracker_pref = mr.TransInv(T_theatre_tracker) @ T_theatre_pref
S_b = np.r_[S_M, -np.cross(S_M, Q_M) + H * S_M]                    # body screw axis of the driver


def main():
    nav = {"description": "Optical surgical navigation for pedicle-screw insertion (illustrative values). "
                          "Edges store T_parent_child; quaternions are tf2 order (x, y, z, w); metres.",
           "transforms": [tf_entry("theatre", "tracker", T_theatre_tracker),
                          tf_entry("tracker", "patient_ref", T_tracker_pref),
                          tf_entry("patient_ref", "ct_image", T_pref_ct),
                          tf_entry("ct_image", "plan_entry", T_ct_plan),
                          tf_entry("tool_marker", "tool_tip", T_marker_tip)]}
    (HERE / "nav_tree.json").write_text(json.dumps(nav, indent=2) + "\n")

    t = np.arange(0.0, T_END + DT / 2, DT)
    T0 = T_tracker_ct @ T_ct_marker0
    with open(HERE / "tool_track.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + [f"r{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)] + ["px", "py", "pz"])
        for tk in t:
            T = T0 @ mr.MatrixExp6(mr.VecTose3(S_b * THETADOT * tk))
            w.writerow([f"{tk:.3f}"] + [f"{v:.12f}" for v in T[:3, :3].ravel()] +
                       [f"{v:.12f}" for v in T[:3, 3]])

    # EOD robot (Week 2 Workshop Problem 4 numbers)
    R_bd = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])
    eod = {"description": "Tracked EOD robot: base, mast camera, gripper and a detected suspect package "
                          "(illustrative values, Week 2 Workshop Problem 4). Edges store T_parent_child; metres.",
           "transforms": [tf_entry("odom", "base_link", mr.RpToTrans(rot([0, 0, 1], d2r(30)), [5.0, 2.0, 0.0])),
                          tf_entry("base_link", "mast_camera", mr.RpToTrans(R_bd, [0.2, 0.0, 0.8])),
                          tf_entry("mast_camera", "package", mr.RpToTrans(R_bd.T, [0.1, 0.5, 1.2])),
                          tf_entry("base_link", "gripper",
                                   mr.RpToTrans(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]]), [0.9, 0.0, 0.4]))]}
    (HERE / "eod_tree.json").write_text(json.dumps(eod, indent=2) + "\n")
    print(f"wrote nav_tree.json, tool_track.csv ({len(t)} samples), eod_tree.json")


if __name__ == "__main__":
    main()
