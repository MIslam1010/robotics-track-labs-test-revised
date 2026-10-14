"""Generate lab/data/tracker_orientations.csv (run once; output is committed).

Simulated optical-tracker readings R_sm of a marker on a laparoscopic instrument (illustrative values).
Most samples are true rotations plus tiny noise (rounded to 6 decimals). Four samples are corrupted
on purpose so that students must detect them:
  index 4  : third column negated   -> det = -1 (left-handed frame, a reflection)
  index 9  : whole matrix scaled by 1.05 -> columns not unit length
  index 13 : first two columns swapped  -> det = -1
  index 17 : large noise (sigma 0.05)   -> not orthogonal
"""
import os
import numpy as np
import modern_robotics as mr

CORRUPTED = (4, 9, 13, 17)


def main():
    rng = np.random.default_rng(361)
    rows = []
    for k in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0.2, 2.5)
        R = mr.MatrixExp3(mr.VecToso3(axis * theta))
        R = R + rng.normal(scale=2e-5, size=(3, 3))
        if k == 4:
            R[:, 2] = -R[:, 2]
        elif k == 9:
            R = 1.05 * R
        elif k == 13:
            R = R[:, [1, 0, 2]]
        elif k == 17:
            R = R + rng.normal(scale=0.05, size=(3, 3))
        rows.append([k] + list(np.round(R.reshape(-1), 6)))
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "data", "tracker_orientations.csv")
    header = "sample,r11,r12,r13,r21,r22,r23,r31,r32,r33"
    np.savetxt(out, np.array(rows), delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.6f"] * 9)
    print("wrote", out)


if __name__ == "__main__":
    main()
