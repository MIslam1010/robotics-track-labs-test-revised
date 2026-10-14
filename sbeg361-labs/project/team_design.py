"""team_design.py - YOUR team's analysis for the chosen scenario (edit this file).

Fill in ONLY the function for your scenario.  Each function returns a dict
{test_id: (value, passed)} for the acceptance tests marked TEAM in baseline.py and in
scenario_sheets.md.  `pytest -q` runs tests/test_team.py, which skips the scenarios
that still raise NotImplementedError and checks the ones you have implemented.

Rules (project brief, AI-use section): you may use AI tools with disclosure; every
member must be able to explain every line in the oral defence.  You may change the
DESIGN VARIABLES listed in scenario_sheets.md (e.g. by calling the model builders in
scenarios.py with new arguments), never the task requirements.
"""
from __future__ import annotations

import numpy as np

from kintools import (fk, jacobian_body_from_space, jacobian_space, point_jacobian,
                      manipulability,
                      ik_body, resolved_rate_step, trans, frame_from_z, moved_points,
                      sample_segments, point_line_distance, within_limits, pose_error,
                      dls_pinv, holding_torques, gravity_wrench_space)
from scenarios import scenario, DEG

FREE_ROLL = (1, 1, 0, 1, 1, 1)


def path_ik(robot, poses, theta0, mask=None):
    """Solve IK along a list of poses with warm starts (each solution seeds the next).
    Returns (thetas [N x n], all_converged)."""
    # TODO: loop over poses, call ik_body(robot.Slist, robot.M, T, seed, mask=mask,
    #       limits=robot.limits), keep the solution as the next seed, collect results.
    raise NotImplementedError("path_ik")


def evaluate_s1():
    """Rehabilitation guide.  Must return AT1-3 ... AT1-6.
    TODO: (a) circle and line waypoints (>= 36 and >= 21) with the handle orientation R_d;
    (b) path_ik; (c) mu1 of the linear part of J_b along both paths (MR Sec. 5.4);
    (d) coverage of task['grid'] (fraction of grid points that are reachable within the
    safe limits AND have mu1 <= task['mu1_max']); (e) handle speed for your timing law
    and the largest joint step between consecutive waypoints."""
    raise NotImplementedError


def evaluate_s2():
    """Feeding assist.  Must return AT2-5, AT2-6.
    TODO: (a) self-motion: move along the 1-D null space of J_b (MR Sec. 6.3) at the stop
    pose, map the swivel range allowed by the joint limits and pick the configuration that
    puts the elbow farthest from the head (AT2-6, baseline 0.504 m < 0.52 m);
    (b) straight 0.15 m approach along task['approach_dir'] ending at the stop pose,
    v <= 0.05 m/s, using resolved_rate_step (MR Eq. 6.7), tilt and clearance checked."""
    raise NotImplementedError


def evaluate_s3():
    """Needle positioning.  Must return AT3-3 (redesigned), AT3-4, AT3-5.
    TODO: (a) pivot the guide from direction 1 to 2 to 3 while its axis keeps passing
    through the entry point (RCM); report the max point-to-axis distance; (b) advance the
    guide 15 mm along its own axis (20 mm -> 5 mm stand-off) by resolved-rate control and
    report the lateral deviation of the guide tip from the needle line."""
    raise NotImplementedError


def evaluate_s4():
    """EOD under-vehicle reach.  Must return AT4-2..AT4-4 (redesigned), AT4-5, AT4-6.
    TODO: (a) choose the UGV standoff / link lengths (design variables); (b) simulate a
    teleoperated approach: body twist V_b = (0,0,0, v,0,0) for 0.20 m/0.05 m/s with
    resolved_rate_step and damping; (c) corridor + ground margins and |theta5| at every
    step; (d) joint rates vs robot.rate_limit."""
    raise NotImplementedError


def evaluate_s5():
    """Litter-loading arm.  Must return AT5-2 (redesigned), AT5-4, AT5-5.
    TODO: (a) straight-line lift path A -> B -> C with the cradle level; (b) holding
    torques tau = J_s^T F_s along the path (MR Eq. 5.26); (c) joint rates for 0.10 m/s."""
    raise NotImplementedError


def evaluate_s6():
    """Telesurgery RCM arm.  Must return AT6-4, AT6-5.
    TODO: (a) 20 mm circle at the port depth; (b) at each step the shaft must pass the port:
    either solve masked IK per waypoint or build an augmented Jacobian [tip; RCM] and use
    resolved-rate control; (c) report max RCM and tip errors; (d) scale a 100 mm surgeon
    circle (radius 100 mm) by 1/5 and check tip speed and ALL joint rates, including the
    insertion stage (watch for drift of the redundant joints)."""
    raise NotImplementedError


EVALUATORS = {1: evaluate_s1, 2: evaluate_s2, 3: evaluate_s3, 4: evaluate_s4,
              5: evaluate_s5, 6: evaluate_s6}
# TEAM tests + the baseline tests that your redesign must turn from FAIL into PASS
TEAM_TESTS = {1: ["AT1-3", "AT1-4", "AT1-5", "AT1-6"], 2: ["AT2-5", "AT2-6"],
              3: ["AT3-3", "AT3-4", "AT3-5"], 4: ["AT4-2", "AT4-3", "AT4-4", "AT4-5", "AT4-6"],
              5: ["AT5-2", "AT5-4", "AT5-5"], 6: ["AT6-4", "AT6-5"]}
