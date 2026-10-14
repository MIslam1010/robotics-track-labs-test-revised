"""Regenerate data/lab06_config.json for SBEG 461 Lab 6 (computed torque vs PID with model error).

The robot is THE course arm (lab/course_arm.py, a copy of course_arm of the shared course robot models (Course resources), numbers
identical).  Everything in this file is an ILLUSTRATIVE teaching value: viscous friction, the moves, the model-error
cases and the gains.  You do not need to run this file; it documents where the numbers come from.

    ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python make_data.py
"""
import json
from pathlib import Path

import numpy as np

import course_arm as ca

HERE = Path(__file__).resolve().parent

arm = ca.course_arm(payload=0.25)
q_home = [float(x) for x in arm["q_home"]]

config = {
    "description": "SBEG 461 Lab 6 configuration (illustrative values). Robot: course arm (course_arm.py).",
    "dt": 0.002,                       # controller period (500 Hz servo rate), torque held between samples (ZOH)
    "substeps": 2,                     # RK4 steps of the plant per control period
    "payload": 0.25,                   # kg at the tool point = the course-arm maximum payload
    # viscous friction b_j (N m s/rad); joints 1-2 as the Lab 5 plants arm_j1/arm_j2 (0.01), smaller servos smaller
    "friction_b": [0.01, 0.01, 0.005, 0.001, 0.001, 0.0002],
    "tau_limit_fraction": 0.6,         # hardware rule: <= 60 % of rated torque (Course resources: Course robot models and hardware safety checklist)
    "joint_move": {                    # quintic joint-space move from q_home, then hold
        "q0": q_home,
        "dq": [0.8, 0.4, -0.5, 0.4, 0.4, 0.6],
        "T": 1.5,
        "T_end": 2.0,
    },
    "setpoint": {                      # PD + gravity compensation regulation test (MR Sec. 11.4.2.2)
        "q0": q_home,
        "dq0_offset": [0.1, -0.1, 0.1, -0.1, 0.1, 0.1],   # initial angle error (rad): qd = q0 + offset
        "T_end": 1.5,
    },
    "task_move": {                     # needle-guide approach: tool point 50 mm straight along -z_s, fixed orientation
        "q0": q_home,
        "dp_s": [0.0, 0.0, -0.05],
        "T": 1.0,
        "T_end": 1.5,
    },
    "cases": {                         # controller model vs plant (plant = course arm + 0.25 kg payload + friction)
        "exact": {"model_payload": 0.25, "model_mass_scale": 1.0, "model_friction": True,
                  "text": "model = plant"},
        "payload_unknown": {"model_payload": 0.0, "model_mass_scale": 1.0, "model_friction": True,
                            "text": "the controller does not know the 0.25 kg payload (unknown package / cup)"},
        "masses_120": {"model_payload": 0.25, "model_mass_scale": 1.2, "model_friction": True,
                       "text": "all link masses and inertias 20 % too large in the model (as MR Exercise 11.11)"},
    },
    "gains": {
        "zeta": 1.0,
        "wn": 20.0,                    # rad/s, error-dynamics bandwidth for every controller
        "ct_ki": 2000.0,               # s^-3, computed torque with integral action (0.125 of kp*kd)
        "pid_ki_fraction": 0.1,        # decentralized PID: Ki_j = 0.1 * Ki_max_j (Lab 5 bound, MR Eq. 11.30)
        "gain_reference_q": q_home,    # decentralized gains from M_jj at q_home (nominal model)
    },
}

if __name__ == "__main__":
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "lab06_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("wrote data/lab06_config.json")
    print("diag M(q_home) with payload:", np.round(np.diag(
        __import__("modern_robotics").MassMatrix(arm["q_home"], arm["Mlist"], arm["Glist"], arm["Slist"])), 6))
