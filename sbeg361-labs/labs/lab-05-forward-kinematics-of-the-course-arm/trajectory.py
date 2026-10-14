"""Commanded motion of the SBEG 361 Lab 5 course-arm log (complete helper, shipped with the lab).

`commanded(t)` is the smooth joint motion that was sent to the arm in `data/course_arm_log.csv`
(ILLUSTRATIVE VALUES): each joint moves sinusoidally about the Lab 5 ready pose q_home for T_MOVE
seconds and is then held.  Use it for the dry run of the Part C demonstration:

    import numpy as np, trajectory
    t = np.arange(0.0, trajectory.T_END + 1e-9, 1.0 / trajectory.RATE)
    cmd_servo_deg = np.rad2deg(trajectory.commanded(t)) + offsets_deg     # q_servo = q_model + offset
"""
import numpy as np

Q_HOME = np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0])     # Lab 5 ready pose q_home of the course arm, rad
AMP_DEG = np.array([20.0, 10.0, 15.0, 10.0, 15.0, 30.0])  # amplitude per joint, degrees
RATE, T_MOVE, T_END = 50.0, 18.0, 20.0                    # sample rate (Hz), motion time, log length (s)


def commanded(t):
    """Model angles (rad) [N x 6] of the commanded motion: periodic about q_home, held after T_MOVE."""
    tt = np.minimum(np.asarray(t, float), T_MOVE)
    s = np.sin(2 * np.pi * tt / T_MOVE)[:, None]
    return Q_HOME + np.deg2rad(AMP_DEG) * s
