"""Write data/lab07_config.json for SBEG 461 Lab 7 (Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm).

The values are ILLUSTRATIVE (friction, torque budgets, initial errors); the robots are the shared course models
(robots.py, course_arm.py).  You do not need to run this file; it documents where every number comes from.

    python make_data.py          # rewrites data/lab07_config.json (byte-identical if nothing changed)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIG = {
    "description": "SBEG 461 Lab 7 configuration (illustrative values). Robots: planar 2R (robots.py) and the "
                   "course arm (course_arm.py).",
    "dt": 0.002,                         # s, controller period (500 Hz, torque held between samples, RK4 plant)
    "planar_2r": {
        "b": [0.05, 0.02],               # N m s/rad, viscous joint friction of the plant (illustrative)
        "q_eq": [0.6, 0.9],              # rad, the hold pose (= q_home of the shared model); open-loop UNSTABLE
        "q_hang": [-1.5707963267948966, 0.0],   # rad, hanging straight down; open-loop stable (a pendulum)
        "pd": {"zeta": 1.0, "wn": 8.0},  # PD + gravity compensation gains from the diagonal of M(q_eq)
        "lyapunov": {"q0": [0.9, 0.5], "dq0": [0.0, 0.0], "T": 4.0},   # E1-E3: q0 = q_eq + (0.3, -0.4)
        "wrong_model_mass_scale": 0.8,   # E3: the controller's gravity model is 20 % too light
        "lqr": {"q_max": 0.05, "dq_max": 0.5, "tau_max": [5.0, 2.0],   # Bryson's rule (Astrom-Murray Ex. 6.15)
                "x0": [0.05, -0.05, 0.0, 0.0], "T": 3.0},
    },
    "course_arm": {
        "b": [0.01, 0.01, 0.005, 0.001, 0.001, 0.0002],   # N m s/rad (as Lab 6, illustrative)
        "pd": {"zeta": 1.0, "wn": 8.0},
        "lqr": {"q_max": 0.05, "dq_max": 0.5, "tau_fraction": 0.2},   # feedback budget = 20 % of rated torque
    },
}


def main():
    out = HERE / "data" / "lab07_config.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(CONFIG, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
