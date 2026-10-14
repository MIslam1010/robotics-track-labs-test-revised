"""Generate the data files of SBEG 361 Lab 6 (read it AFTER the lab).

data/q_samples.csv : 1000 joint vectors of the course arm, uniform inside the joint limits (seed 361),
                     6 columns q1..q6 in rad, header line.  Used by the Python tests and by the C++ program.
"""
from pathlib import Path

import numpy as np

from course_arm_model import LIMITS

HERE = Path(__file__).resolve().parent


def main(n=1000, seed=361):
    rng = np.random.default_rng(seed)
    q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(n, 6))
    (HERE / "data").mkdir(exist_ok=True)
    np.savetxt(HERE / "data" / "q_samples.csv", q, delimiter=",", fmt="%.17g",
               header="q1,q2,q3,q4,q5,q6", comments="")
    print(f"wrote {n} samples to data/q_samples.csv")


if __name__ == "__main__":
    main()
