---
title: "Lab 9 — Trajectory library, torque feasibility and time-optimal time scaling (TOPP) on the course arm"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 10"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before the Week 11 lab) |
| CLOs | **461-4** (point-to-point, via-point and time-optimal trajectories that respect kinematic and torque limits); 461-3 (actuator limits); 461-2 (reuse of the verified RNEA inside a planner) |
| Files (your Lab code folder) | `starter.py` (you edit), `test_lab.py` (auto-checks), `data/course_arm.json`, `data/course_arm.urdf` (the course-arm numbers of Course resources: Course robot models and hardware safety checklist), `data/tasks.json` (Parts A–C), `data/topp_task.json` (Parts D–E) |
| Reference | `modern_robotics` (`CubicTimeScaling`, `QuinticTimeScaling`, `JointTrajectory`, `InverseDynamics`, `InverseDynamicsTrajectory`), `scipy.interpolate.BSpline`, `toppra` 0.6 (TOPP-RA solver), closed-form phase-plane results of Lecture B and the workshop |
| Reading | MR Ch. 9 introduction and §9.1–9.3 (printed pp. 327–338); MR §9.4 (pp. 338–347: Eqs. 9.30–9.37, the algorithm on pp. 343–344, caveats §9.4.4 on p. 346); MR §9.7 notes (p. 349); MR §8.3.2 (RNEA, pp. 292–293); Corke §3.1 (pp. 43–51); LaValle §14.6.3 (printed pp. 846–854) |
| Assessment | Auto-checked (`test_lab.py`) + results tables + plots + short reflection (not a formal report) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation) for everyone; Part E makes a checked command file; running it on the course arm is optional and only with a TA (see Part E) |

The functions you write today form the trajectory and time-scaling modules of your `sbeg461_dynamics` library. Lab 11 (RRT, smoothing and time scaling) and the course project (`sbeg461_project/traj.py`; Scenarios 3 and 5, "time-optimal scaling under the torque and joint-speed limits") reuse `quintic_time_scaling`, `torque_profile`, `check_limits`, `min_time_scale`, `topp`, `command_table` and `check_command`. The RNEA is the one you verified in Labs 2 and 4; a reference copy is provided as `rnea(...)` in `starter.py` (you may paste your own), and `inverse_dynamics(model, ...)` wraps it for any model.

**The robot.** The course arm (the desktop 6R of every hardware session; numbers from Course resources: "Course robot models and hardware safety checklist", copied into `data/course_arm.json`). Its inertial data and ratings are **illustrative values**. Two load cases are used:

* Parts A–C: `load_course_arm()` — the maximum payload of 0.25 kg at the tool point, derated to 50 % of the rated joint speed (160 °/s) and 60 % of the rated torque: $\dot\theta_{\max} = 1.396$ rad/s (all joints), $\tau_{\max} = (1.08,\ 1.80,\ 0.90,\ 0.36,\ 0.36,\ 0.24)$ N m.
* Parts D–E: `course_arm_model(0.1)` — a 0.1 kg foam phantom (within the 0.125 kg checklist payload) and the **rated** limits; `topp_task.json` gives the fractions (torque 60 %; speed 100 %, 50 % or 25 %; $|\ddot\theta| \le 4$ rad/s²).

**Application context.**
*Biomedical:* in a hospital laboratory the arm moves a 0.25 kg sample carrier from its ready pose A to a centrifuge rack B, directly or via a waypoint W above a tray, and seats a sample tube with a short dip of the forearm (A → D); smooth motions keep the samples from spilling ("a glass of water", MR p. 338). In Parts D–E the same arm is a mock-up of a patient-positioning move: a foam phantom goes from a tray, over a clearance waypoint, to a pose above the mock couch, as fast as the motors **and** the safety limits allow.
*Defence (protective/humanitarian):* on an EOD training rig the arm lifts a 0.25 kg **inert mock package** (A → C); the X-Y gantry of a demining sensor head and an elbow exoskeleton are the small models of Part D. Operators of real EOD manipulators use the same tools: smooth or time-optimal scalings with a check that no actuator saturates.

# Learning objectives

By the end of the lab you can:

1. Implement the cubic, quintic and trapezoidal time scalings (MR §9.2.2) and straight-line joint trajectories, and verify them against `modern_robotics` and the peak values of MR Figs. 9.3–9.4.
2. Implement cubic via-point interpolation (MR Eqs. 9.26–9.29), reproduce MR Fig. 9.7(a), and evaluate a uniform cubic B-spline with its convex-hull property (MR p. 338).
3. Compute the joint torques along a trajectory with the RNEA, and use uniform time scaling, $\tau_c = g + (\tau - g)/c^2$, to find the shortest feasible duration and the active limit.
4. Compute the path dynamics $m(s), c(s), g(s)$ (MR Eq. 9.33), the motion cone $[L, U]$ and the velocity limit curve (MR Eqs. 9.36–9.37), and the time-optimal time scaling with `toppra`; compare it with the fastest quintic and explain the difference in time, jerk and actuator use.
5. Turn a time-optimal plan into a safe command file for the course arm (derated speed, joint-limit margin, stretched first run).

# Pre-lab check (complete before the session; 5 questions)

1. A joint moves $\Delta\theta = 1$ rad from rest to rest with a cubic time scaling in $T = 2$ s. Give its peak speed and peak acceleration.
2. A trapezoid has $v = 0.5$ s⁻¹ and $a = 1$ s⁻² (in $s$-units). Is it three-stage? Give $T$ and $t_a$.
3. A rest-to-rest trajectory is replayed twice as slowly. By what factor do $M(\theta)\ddot\theta$, $c(\theta,\dot\theta)$ and $g(\theta)$ change at corresponding points of the path?
4. Write $\dot\theta$ and $\ddot\theta$ along a path in terms of $\theta'$, $\theta''$, $\dot s$, $\ddot s$ (MR Eqs. 9.1–9.2), and show that $m(s) = $ `InverseDynamics(θ, 0, θ', g = 0)` and $c(s) = $ `InverseDynamics(θ, θ', θ'', g = 0)`.
5. The course arm's rated joint speed is 160 °/s. What is the speed limit (rad/s) at 50 % and at 25 % of rated speed? Why does a time-optimal plan touch at least one limit at every instant?

*Answers (for self-check):* 1. $\frac{3}{2T} = 0.75$ rad/s at $t = 1$ s; $\frac{6}{T^2} = 1.5$ rad/s² at $t = 0$ and $T$. 2. $v^2/a = 0.25 \le 1$: three stages; $T = (a + v^2)/(va) = 2.5$ s, $t_a = 0.5$ s. 3. $\dot\theta$ halves and $\ddot\theta$ quarters, so $M\ddot\theta$ and $c = \dot\theta^{\mathsf T}\Gamma\dot\theta$ both become $\tfrac14$ of their values; $g(\theta)$ is unchanged because the configurations are the same. 4. $\dot\theta = \theta'\dot s$, $\ddot\theta = \theta'\ddot s + \theta''\dot s^2$. The RNEA returns $M\ddot\theta + c(\theta,\dot\theta) + g$: with $\dot\theta = 0$, $\ddot\theta = \theta'$ and no gravity it returns $M\theta' = m(s)$; with $\dot\theta = \theta'$, $\ddot\theta = \theta''$ and no gravity it returns $M\theta'' + \theta'^{\mathsf T}\Gamma\theta' = c(s)$ (MR Eq. 9.32). 5. $0.5\times2.793 = 1.396$ rad/s and $0.25\times2.793 = 0.698$ rad/s. If no limit were active, $\ddot s$ could be raised (or lowered) a little and the motion would finish earlier (MR §9.4.1: bang-bang in $\ddot s$, or riding the velocity limit curve).

# Setup

```bash
cd "<your Lab code folder>"   # or your Classroom 50 repo
python -c "import numpy, scipy, modern_robotics, toppra; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py   # 38 failed, 2 errors (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. If `toppra` is missing: `pip install toppra` (it is in the course Docker image and the JupyterHub fallback).

Rules: inside `starter.py` do **not** call `modern_robotics` or SciPy (they are the ground truth of the tests); use NumPy, `toppra` (only inside `topp`) and the provided functions. You write **14 functions**; each has a docstring with its formula and MR reference and a `# TODO` hint. Provided (complete): `rnea`, `inverse_dynamics`, the loaders (`load_course_arm`, `course_arm_model`, `load_tasks`, `load_topp_task`), `trapezoid_from_limits`, `heuristic_via_velocities`, `bspline_basis`, `check_limits`, the paths `StraightLinePath`, `SemicirclePath`, `waypoint_path` (natural cubic spline), the models `mr_2r_model`, `elbow_exo_model`, `gantry_model`, and `quintic_traj`, `peak_usage`, `command_table`, `check_command`, `export_csv`, `write_log`. Paths are called like `toppra` paths: `path(s, 0)` is $\theta$, `path(s, 1)` is $\theta'$, `path(s, 2)` is $\theta''$. A checkpoint is complete when the listed tests pass; ask a TA to sign it off.

# Procedure

## Part A — Time scalings and straight lines (10 min)

1. Implement `cubic_time_scaling(T, t)` and `quintic_time_scaling(T, t)` (return $(s, \dot s, \ddot s)$, work for scalars and arrays, clip $t$ to $[0, T]$), then `trapezoid_params(v, a)` → $(T, t_a, v_{\text{peak}})$ and `trapezoid_time_scaling(v, a, t)` (MR Eqs. 9.16–9.24). Handle the bang-bang case $v^2/a > 1$: the peak speed is then $\sqrt a$ and $t_a = T/2 = 1/\sqrt a$.
2. Implement `joint_trajectory(theta_start, theta_end, T, N, method)` with `method` = 3, 5 or `"trap"` (MR Eqs. 9.3–9.5), returning `(t, Q, Qd, Qdd)`.

**Checkpoint A:** `pytest -q -k "cubic_matches or quintic_matches or scalars or trapezoid or joint_trajectory"`. For A → B with $\ddot\theta_{\text{limit}} = 4$ rad/s² (`tasks.json`) the provided `trapezoid_from_limits` gives $v = 1.1636$ s⁻¹, $a = 3.3333$ s⁻², $T = 1.2085$ s; joint 1 ($\Delta\theta_1 = 1.2$ rad, the largest move) saturates both.

**Plot A:** $s$, $\dot s$, $\ddot s$ of the three scalings for $T = 1$ s (trapezoid $v = 1.5$ s⁻¹, $a = 4.5$ s⁻², which also gives $T = 1$ s); compare with MR Figs. 9.3–9.5.

## Part B — Via points and B-splines (8 min)

3. Implement `via_cubic_coefficients` (MR Eqs. 9.26–9.29), `via_trajectory` and `uniform_cubic_bspline` (global parameter $u\in[0, n-3]$, segment $i = \lfloor u\rfloor$; return the point and its first two derivatives with respect to $u$; the blending functions are given in `bspline_basis`).

**Checkpoint B:** `pytest -q -k "via_cubic or figure_9_7a or bspline"`.

**Plot B:** MR Fig. 9.7(a)/9.8: vias $(0,0), (0,1), (1,1), (1,0)$ at $T = 0, 1, 2, 3$ with velocities $(0,0), (1,0), (0,-1), (0,0)$, as a path in the $(x, y)$ plane and as $x(t), y(t)$. Add the uniform cubic B-spline with the same four points as control points (one segment) and with each end point repeated three times (clamped). Which curves leave the unit square, and why does the convex-hull property (MR p. 338) protect joint limits?

## Part C — Torque feasibility and uniform time scaling (10 min)

4. Implement `torque_profile(robot, Q, Qd, Qdd)` (one `rnea` call per sample, $\mathcal F_{\text{tip}} = 0$) and `min_time_scale(robot, Q, Qd, Qdd, qddot_max=None)`.

The idea of `min_time_scale` (MR §9.7, p. 349, credits it to Hollerbach): if a trajectory $\theta(t)$ is replayed $c$ times slower, $\theta_c(t) = \theta(t/c)$, the configurations are the same, the velocities are divided by $c$ and the accelerations by $c^2$. Because $c(\theta,\dot\theta) = \dot\theta^{\mathsf T}\Gamma(\theta)\dot\theta$ is quadratic in the velocity (MR p. 339),
$$\tau_c = g(\theta) + \frac{\tau - g(\theta)}{c^2} = g + \frac{d}{c^2}.$$
So one evaluation of the trajectory gives the feasible range of $c$ in closed form: $\lvert\dot\theta\rvert/c \le \dot\theta_{\max}$, $\lvert\ddot\theta\rvert/c^2 \le \ddot\theta_{\max}$ (if given) and, sample by sample and joint by joint, $c^2 \ge d/(\tau_{\max} - g)$ if $d > 0$ or $c^2 \ge -d/(\tau_{\max} + g)$ if $d < 0$. If $\lvert g\rvert \ge \tau_{\max}$ anywhere, the arm cannot even hold that configuration and no $c$ helps (raise `ValueError`). For a rest-to-rest trajectory computed with $T = 1$ s, $c$ is the minimum feasible duration in seconds.

**Checkpoint C:** `pytest -q -k "course_arm_data or torque_profile or min_time_scale_e or min_time_scale_c or min_time_scale_r or via_trajectory_on"`. Gravity torque at A with the payload: $g = (0,\ -1.140,\ -0.570,\ -0.242,\ 0,\ 0)$ N m, i.e. 63 %, 63 % and 67 % of the derated limits of joints 2–4 before the arm even moves.

Fill in the table (401 samples per trajectory, computed with $T = 1$ s; "max $\tau$ ratio" is the largest $\lvert\tau_i\rvert/\tau_{\max,i}$ at $T_{\min}$):

| Move (tasks.json) | Profile | $T_{\text{vel}}$ (s) | $T_\tau$ (s) | $T_{\min}$ (s) | Active limit (joint) | Max $\tau$ ratio at $T_{\min}$ |
|---|---|---|---|---|---|---|
| A → B | cubic | | | | | — |
| A → B | quintic | | | | | |
| A → B | trapezoid ($\ddot\theta_{\text{limit}} = 4$) | — | — | | — | |
| A → C | quintic | | | | | — |
| A → D | quintic | | | | | — |
| A → W → B | via cubic (0, 1.2, 2.4 s) | | | | | — |

Reference values (agree to 3 digits): A → B cubic 1.289 / 0.402 s and quintic 1.611 / 0.405 s, both limited by the **speed of joint 1**; at the quintic $T_{\min}$ the largest torque ratio is 0.782 on joint 3, and the trapezoid needs 1.2085 s with 0.781 on joint 3. A → C quintic 1.074 / 0.346 s. A → D quintic 0.134 / 0.235 s, limited by the **torque of joint 4**. The via trajectory is feasible as timed ($c = 0.716$); it could be run in $2.4c = 1.719$ s, limited by the speed of joint 2.

**Plot C:** $\tau_i(t)/\tau_{\max,i}$ for all joints along A → B (quintic) at $T = T_{\min}$.

## Part D — Time-optimal time scaling (TOPP) vs quintic (15 min; finish the table at home)

5. Implement `path_dynamics(model, path, s)` with three calls of `inverse_dynamics` (pre-lab Q4).
6. Implement `accel_limits(m, c, g, sdot, tau_min, tau_max)` (MR Eqs. 9.36–9.37). Swap the bounds when $m_i < 0$. A **zero-inertia** joint ($|m_i| <$ `EPS_M`) gives no bound on $\ddot s$, but makes the state inadmissible if $c_i\dot s^2 + g_i$ is outside the torque range (MR §9.4.4).
7. Implement `velocity_limit(...)`: the kinematic bound $\min_i\dot\theta_{i,\max}/|\theta'_i|$, then bisection on $\dot s$ for the largest admissible value.
8. Implement `topp(model, path, tau_max, qdot_max, qddot_max, N)` with `toppra`: a `JointTorqueConstraint` built on `inverse_dynamics`, optional `JointVelocityConstraint` and `JointAccelerationConstraint`, `algo.TOPPRA(..., parametrizer="ParametrizeConstAccel")`, then `compute_trajectory(0, 0)`. Return the duration, the trajectory and `problem_data.sd_vec` (the optimal $\dot s$ on the grid). The single-switch algorithm of Lecture B (MR §9.4.2) is what you do by hand in the workshop; `toppra` solves the same problem on a grid and also handles multi-switch cases.

**Checkpoint D1** (`pytest -q -k "path_dynamics or accel_limits or velocity_limit"`): the motion cone $[-1.6,\ 1.2]$ of Quiz 8, Q3; $\dot s_{\lim}(0.5) = 2.155$ for the MR 2R arm; $\dot s_{\lim} = 1.007,\ 1.293,\ 1.162$ at $s = 0,\ 0.25,\ 0.5$ for the gantry (the dip at the zero-inertia point $s = 0.5$). On the course-arm task path at $s = 0.5$ (the middle waypoint): $\theta' = (1.5, -0.2, 0.2, 0, 0, 0)$, $g = (0, -0.887, -0.382, -0.130, 0, 0)$ N m, and $\dot s_{\lim} = 5.59$ from 60 % torque alone but $0.931$ with the 50 % speed limit: on this light arm the **speed** limit binds long before the torque limit.

**Checkpoint D2** (`pytest -q -k "topp"`): MR 2R arm, $|\tau|\le(20, 10)$: $T = 1.370$ s (bang-bang by hand: 1.368 s). Elbow exoskeleton, $|\tau|\le10$ N m: $T = 0.388$ s (closed form 0.3885 s); with $|\dot\theta|\le3$ rad/s: 0.7466 s. Gantry semicircle: $T = 1.166$ s, touching the velocity limit curve at $s = 0.5$ (one switch is not enough, MR Fig. 9.12b). With $0.6\tau_{\max}$ the 2R time grows by $1/\sqrt{0.6}$.

9. **TOPP vs quintic on the course arm.** Load the task (`load_topp_task()`, `course_arm_model(task["payload_kg"])`, `waypoint_path(task["waypoints"])`). For the quintic, sample `quintic_traj(path, 1.0)` at 401 instants of $[0, 1]$ and call your `min_time_scale` with `dict(model, tau_max=0.6 * tau_rated, qdot_max=fraction * qdot_rated)` (and `qddot_max` for hw and first run): that is the fastest quintic **along the same path**. Fill the table for three limit sets: **sim** (100 % rated speed, 60 % torque), **hw** (50 % speed, 60 % torque, $|\ddot\theta|\le4$ rad/s²) and **first run** (25 % speed, 60 % torque, 4 rad/s²). Use `peak_usage` for the peak speed and torque ratios and the peak jerk.

| Case | $T_{\text{TOPP}}$ (s) | $T_{\text{quintic}}$ (s) | ratio | active limit of the quintic | peak jerk TOPP / quintic |
|---|---|---|---|---|---|
| sim | 0.615 | 1.100 | 1.79 | | |
| hw | 1.566 | 2.296 | 1.47 | | |
| first run | 2.503 | 4.402 | 1.76 | | |

**Checkpoint D3:** `test_topp_course_arm_respects_limits`, `test_min_time_scale_quintic_elbow_closed_form` and `test_topp_beats_quintic_on_course_arm` pass. In the sim case, check the joint-1 torque ratio from `peak_usage` with $N = 4000$ samples: it slightly exceeds 1 (about 1.01) because `toppra` enforces the constraints only at its grid points. Write one sentence on what this means for a hardware plan.

**Plot D:** for the MR 2R arm, the gantry and the course arm (hw), plot on one $(s,\dot s)$ figure each the velocity limit curve and `toppra`'s $\dot s(s)$, and (2R) 6–8 motion cones as short arrows $(\dot s, L)$ and $(\dot s, U)$. For the course arm add the quintic's $\dot s(s)$ at its $T_{\min}$.

## Part E — A command file for the course arm (5 min, at home)

10. Take the **hw** TOPP plan and stretch it by `slow = 2` (speeds ÷2, accelerations ÷4; torques stay feasible because $\tau_c$ is a convex combination of $\tau$ and $g$). Sample it at the servo period 0.02 s with the provided `command_table(hw["traj"], hw["T"], dt=0.02, slow=2)`, check it with the provided `check_command(t, q, model, 0.25, 10.0)` (25 % rated speed for the first run of a new motion; 10° joint-limit margin, checklist rule 1) and write it with `write_log("cmd_first.csv", t, q)`.

**Checkpoint E** (`test_command_file_for_course_arm`): `check_command` returns `ok = True` with a peak speed ratio of 1.000; the command has 158 samples and ends at 3.131 s. The unstretched plan (`slow = 1`) fails the 25 % check with a ratio of 2.0. Re-solving TOPP directly at 25 % gives 2.503 s instead of $2\times1.566 = 3.13$ s: explain the difference in your reflection.

**Optional hardware run.** The smart servos of the course arm are position-controlled: a command file tests your path, your time scaling and your safety checks, not a torque controller. A run is optional and ungraded, only with a TA present at a station slot, only for students who completed the Week 8 hardware induction, and only with a file that passed `check_command` at 25 % with the 10° margin. The full course-arm checklist (Course resources: Course robot models and hardware safety checklist) applies: keep-out zone, e-stop tested and held by the TA, payload ≤ 0.125 kg, station `servo_zero_offset` and homing applied by the TA. Without a run, simply keep `cmd_first.csv`: the project hardware sessions use the same file format.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 9 into your cumulative library: Parts A–C as `trajectory.py`, Part D as `timescaling.py`. The functions you write (14):

`cubic_time_scaling(T, t)`, `quintic_time_scaling(T, t)`, `trapezoid_params(v, a)`, `trapezoid_time_scaling(v, a, t)`, `joint_trajectory(theta_start, theta_end, T, N, method=5, v=None, a=None)`, `via_cubic_coefficients(beta, beta_dot, times)`, `via_trajectory(points, velocities, times, t)`, `uniform_cubic_bspline(ctrl, u)`, `torque_profile(robot, Q, Qd, Qdd)`, `min_time_scale(robot, Q, Qd, Qdd, qddot_max=None)`, `path_dynamics(model, path, s)`, `accel_limits(m, c, g, sdot, tau_min, tau_max)`, `velocity_limit(model, path, s, tau_min, tau_max, qdot_max=None, sdot_max=100.0, tol=1e-09)`, `topp(model, path, tau_max, qdot_max=None, qddot_max=None, N=200, tau_min=None)`.

Names shared with other labs:

* `torque_profile(robot, Q, Qd, Qdd)` does the same job as Lab 2's `torque_profile(robot, theta, dtheta, ddtheta, Ftip=None)` with $F_{tip} = 0$, and takes its arguments in the same order; Lab 11 uses the same signature again.
* `trapezoid_from_limits` and `check_limits` are provided here and in Lab 11; `check_command` is the same provided function as in Hardware rotation 1 (Week 8); `command_table` is provided here. Copy them into the library with your functions.
* `rnea` and the small $SE(3)$ helpers at the top of `starter.py` are provided copies of your Lab 2/Lab 4 code.

The provided loaders, paths and models do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables (upload to the course site)

1. `starter.py` (all tests pass).
2. Plots A, B, C and D with axis labels and units.
3. The completed tables of Parts C and D, with the active limit of each case, and the answers to the questions below (≤ 200 words in total).
4. `cmd_first.csv` (Part E) and, if you made the optional run, the station log.
5. Reflection (below), ≤ 1 page.

Questions:

1. Why are the long moves of Part C speed-limited and the short move A → D torque-limited? (Hint: $T_{\text{vel}} \propto \lvert\Delta\theta\rvert$, while the inertial torque is $\propto \lvert\Delta\theta\rvert/T^2$.) In A → D only joints 2 and 3 move, yet joint 4 limits the motion: explain with the off-diagonal terms of $M(\theta)$ and the small rating of the wrist motor.
2. The trapezoid takes 25 % less time than the quintic on A → B with almost the same peak torque, and TOPP is 1.47 times faster than the best quintic in the hw case. Where in the motion does the time difference come from (use your phase-plane plot), and why would you still prefer a smoother profile on hardware?

# Reflection (≤ 1 page)

- The minimum times assume the model (masses, payload, friction-free joints) is exact. Which modelling errors from Week 4 (friction, rotor inertia, gear efficiency) would make the real arm saturate earlier, and how much margin would you keep?
- List the caveats of MR §9.4.4 that you met today (static posture, zero-inertia points, discontinuous acceleration, no torque left for feedback — MR p. 347). For each, say how your plan handled it. How would you turn `min_time_scale` or `topp` into a safety rule for a patient- or casualty-handling robot?
- Why does re-solving TOPP at 25 % speed (2.50 s) beat stretching the 50 % plan by 2 (3.13 s)? When would you still prefer the stretched plan on hardware?
- **AI-use note** (required, Course syllabus AI-use policy): state whether you used an AI assistant, for which part, what you asked, and how you verified the answer (tests, hand check, comparison with `modern_robotics`). You must be able to explain every line of `min_time_scale` and `topp` in the lab practical (Week 13).

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| Cubic/quintic positions right but $\dot s$ or $\ddot s$ wrong by a constant factor | forgot the $1/T$ or $1/T^2$ of the chain rule when writing $s$ in terms of $r = t/T$ |
| Trapezoid ends at $s \ne 1$ | used $v$ instead of the peak speed $\sqrt a$ in the bang-bang case |
| `JointTrajectory` comparison fails by one row | `np.linspace(0, T, N)` includes both end points, as MR does ($N$ rows, spacing $T/(N-1)$) |
| Via trajectory jumps at a via point | segment index off by one, or used the global $t$ instead of $\Delta t = t - T_j$ |
| B-spline does not match SciPy | wrong segment at $u = n - 3$ (clamp $i$ to $n - 4$) or wrong derivative of a blending function |
| Torques do not match `modern_robotics` | used the model without the payload on one side only, or the gravity vector with the wrong sign (`g = (0, 0, -9.81)`) |
| `min_time_scale` disagrees with the bisection test | forgot to subtract gravity before dividing by $c^2$, or used $\lvert d\rvert/(\tau_{\max} - \lvert g\rvert)$ (too conservative when $d$ and $g$ have opposite signs) |
| `ValueError` for every trajectory | `tau_max` is the derated value and you multiplied by 0.6 again |
| `accel_limits` gives $L > U$ everywhere | bounds not swapped for $m_i < 0$, or $\dot s$ used where $\dot s^2$ is needed |
| `velocity_limit` returns 0 | the state $(s, 0)$ is inadmissible: the robot cannot hold the path statically with these limits |
| `toppra` finds no solution / `topp` raises | the same static-posture failure, or limits inconsistent with starting and ending at rest; check $g(s)$ against the torque limits |
| TOPP plan slightly exceeds a limit between grid points | expected (constraints are enforced at the grid points only); increase `N` or derate the limit by 2–3 % |
| Command file fails `check_command` with `starts_at_rest = False` | you sampled `traj(t)` with `t` in the stretched time without dividing by `slow` (use the provided `command_table`) |
| `ModuleNotFoundError: toppra` | `pip install toppra` in the course environment, or use the JupyterHub fallback |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, tables, plots and deliverables are the same (deliverable 1 is `cpp/starter.cpp` instead of `starter.py`); the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the closed-form results, and with the numbers of the Python reference (SciPy B-spline values, `modern_robotics` and the real `toppra`), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided functions, as in `starter.py` — do not change), `toppralite.hpp` / `toppralite.cpp` (a small C++ stand-in for the part of `toppra` used in `topp`, see below — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers, including the random inputs that `test_lab.py` draws with NumPy), `plot_results.py` (Plots A–D), `CMakeLists.txt`. The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). `toppra` is not needed for the C++ track.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 9 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*cubic*,*quintic*"      # only some tests (like pytest -k)
python cpp/plot_results.py                     # Plots A-D from the CSV files written by lab_tests
```

With the untouched starter, the program builds and all 31 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The whole C++ test run takes well under a second.

**Functions to write** (in `starter.cpp`, the same 14 as in Python, same order): `cubic_time_scaling`, `quintic_time_scaling`, `trapezoid_params`, `trapezoid_time_scaling`, `joint_trajectory` (Part A); `via_cubic_coefficients`, `via_trajectory`, `uniform_cubic_bspline` (Part B); `torque_profile`, `min_time_scale` (Part C); `path_dynamics`, `accel_limits`, `velocity_limit`, `topp` (Part D). Given in `helpers.cpp`: the SE(3) helpers and `rnea`, `inverse_dynamics`, the loaders, `trapezoid_from_limits`, `heuristic_via_velocities`, `bspline_basis`, `check_limits`, the scalar-time overloads of the three time scalings, the paths and models, `quintic_traj`, `peak_usage`, `command_table`, `check_command`, `export_csv`, `write_log`. Do not call `mrcpp` in `starter.cpp`.

**`topp` with `toppralite`.** `toppralite` is the same TOPP-RA algorithm as `toppra` with the same constraint forms (torque: collocation; acceleration: interpolation; velocity: a bound on $\dot s^2$), the same greedy forward pass and the same constant-acceleration parametrisation; it reproduces the durations of the real `toppra` to about $10^{-8}$ s. The calls translate one to one:

| Python (`toppra`) | C++ (`toppralite`, `namespace ta = toppralite;`) |
|---|---|
| `constraint.JointTorqueConstraint(inv_dyn, np.c_[tau_min, tau_max], np.zeros(n))` | `ta::joint_torque_constraint(inv_dyn, tau_lim, Eigen::VectorXd::Zero(n))`; `tau_lim` is $n\times2$, `inv_dyn` a lambda calling `inverse_dynamics(model, q, qd, qdd)` |
| `constraint.JointVelocityConstraint(np.c_[-v, v])`, `JointAccelerationConstraint(np.c_[-a, a])` | `ta::joint_velocity_constraint(vlim)`, `ta::joint_acceleration_constraint(alim)` ($n\times2$ matrices) |
| `algo.TOPPRA(cons, path, gridpoints=grid, parametrizer="ParametrizeConstAccel")` | `ta::TOPPRA inst(cons, path, grid);` |
| `traj = inst.compute_trajectory(0.0, 0.0)`; `None` if infeasible | `auto traj = inst.compute_trajectory(0.0, 0.0);` `nullptr` if infeasible (then `throw std::runtime_error(...)`) |
| `traj.duration`, `inst.problem_data.sd_vec` | `traj->duration()`, `inst.problem_data().sd_vec` |

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| A — Time scalings, straight lines | `-tc="*cubic_matches*,*quintic_matches*,*scalars*,*trapezoid*,*joint_trajectory*"` | 7 |
| B — Via points, B-splines | `-tc="*via_cubic*,*figure_9_7a*,*bspline*"` | 3 |
| C — Torque feasibility | `-tc="*course_arm_data*,*torque_profile*,*min_time_scale_e*,*min_time_scale_c*,*min_time_scale_r*,*via_trajectory_on*"` | 6 |
| D — Path dynamics, cone, VLC, TOPP vs quintic | `-tc="*path_dynamics*,*accel_limits*,*velocity_limit*,*topp*,*quintic_elbow*"` | 14 |
| E — Command file | `-tc="*command_file*"` | 1 |

All 31 test cases pass with `./build/lab_tests`. The Python track counts 40 tests because pytest counts each value of a parametrized test separately (three $T$ for the cubic and the quintic, two methods for `joint_trajectory`, three moves for the bisection test, three limit sets for TOPP vs quintic); the C++ track checks the same values in one test case each.

**Printing and the Part E file.** For the table values and `cmd_first.csv`, add a temporary `TEST_CASE` at the end of `test_lab.cpp` (include `<iostream>`; remove it before you submit), for example:

```cpp
TEST_CASE("my_part_e") {
    const lab::ToppTask task = lab::load_topp_task();
    const lab::Robot r = lab::course_arm_model(task.payload_kg);
    const lab::Path path = lab::waypoint_path(task.waypoints);
    const lab::ToppResult hw = lab::topp(r, path, 0.6 * r.tau_max, 0.5 * r.qdot_max,
                                         Eigen::VectorXd::Constant(6, 4.0), task.grid_N);
    const auto [t, q] = lab::command_table(hw.traj, hw.T, 0.02, 2.0);
    std::cout << hw.T << " " << lab::check_command(t, q, r, 0.25, 10.0).ok << "\n";
    lab::write_log("cmd_first.csv", t, q);
}
```

For the plots, `./build/lab_tests` writes `lab9_cpp_scaling_{cubic,quintic,trapezoid}.csv` (Plot A), `lab9_cpp_fig_9_7a.csv` and `lab9_cpp_bsplines.csv` (Plot B), `lab9_cpp_tau_AB.csv` (Plot C) and `lab9_cpp_phase_{2r,gantry,course_arm}.csv` (Plot D: velocity limit curve and TOPP $\dot s(s)$) into the folder you run it from, each one when the functions it uses return results of the right size; `python cpp/plot_results.py` saves `lab9_partA_scalings.png`, `lab9_partB_vias.png`, `lab9_partC_torques.png` and `lab9_partD_phase_plane.png`. Add the motion cones and the quintic curve of Plot D yourself (from your own CSV).

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `t` a scalar or an array | `Eigen::VectorXd t`; the scalar-`t` overload is given and returns `lab::TimeScalingPoint` (fields `s, sd, sdd`) |
| tuples and result dicts | small structs with the Python field names: `TimeScaling`, `TrapezoidParams`, `TrapezoidLimits`, `Trajectory` (`t, Q, Qd, Qdd`), `ViaSamples`, `BSplinePoint`, `LimitCheck`, `TimeScaleResult` (`c_vel, c_acc, c_tau, c, binding`), `PathDynamics{m, c, g}`, `ToppResult{T, traj, s_grid, sdot_grid}`, `PeakUsage`, `CommandCheck`; `accel_limits` and `command_table` return a `std::pair` |
| `(N, n)` array, via points `(k, m)` | `Eigen::MatrixXd` with one **row** per sample / via |
| model dicts (course arm, 2R, elbow, gantry) | `lab::Robot` (`name, n, Slist, M, Mlist, Glist, g, qdot_max, tau_max, limits, q_home`); `dict(model, tau_max=...)` = copy the struct and set the field |
| `path(s, order)`, `traj(t, order)` with arrays | `lab::Path`, `lab::TrajectoryFn` (`std::function`), one $s$ or $t$ at a time |
| `method=3`, `5`, `"trap"`; `T=None` | `lab::CUBIC`, `lab::QUINTIC`, `lab::TRAP`; pass any `T` (e.g. `0.0`) with `TRAP` |
| optional `qdot_max=None`, `qddot_max=None`, `tau_min=None`; scalars broadcast | an empty `Eigen::VectorXd()` (the default) means "not given"; always one value per joint: `Eigen::VectorXd::Constant(6, 4.0)` |
| `toppra` inside `topp` | the given `toppralite` (table above) |
| `raise ValueError(...)` / `RuntimeError` | `throw std::invalid_argument("...")` / `throw std::runtime_error("...")` |
| `test_uniform_cubic_bspline_...` calls `scipy.interpolate.BSpline` | compares with SciPy's values stored in `cpp_cases.json` |
| plots with Matplotlib | CSV files written by `lab_tests` + the given `plot_results.py` |

A matrix-size mistake (e.g. `Q` stored as $n \times N$ instead of $N \times n$) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| A time scaling or a blending function is zero everywhere | integer division: `1/6 == 0`, `3/2 == 1` in C++ | write `1.0 / 6.0`, `3.0 / 2.0` |
| `joint_trajectory` shape check fails | `Q` built as $n \times N$, or `LinSpaced(N, 0.0, T)` arguments swapped | one row per sample; `Eigen::VectorXd::LinSpaced(N, low, high)` |
| `min_time_scale` binding joint off by one | printed the 0-based index | `binding.second` is 0-based (joint 4 is `3`), as in Python |
| `topp` fails with "bad input gridpoints" or a dof message | grid not strictly increasing on $[0, 1]$, or limit matrices not $n\times2$ | `Eigen::VectorXd::LinSpaced(N + 1, 0.0, 1.0)`; build `tau_lim` with `.col(0) = tau_min`, `.col(1) = tau_max` |
| `topp` result far from 1.5656 s on the course arm | velocity or acceleration constraint missing | add each constraint only when its vector is non-empty, and pass all three limit sets of the test |
| Wrong results only after `auto x = A * b;` | `auto` keeps an Eigen expression, not a vector | write the type: `Eigen::VectorXd x = A * b;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
