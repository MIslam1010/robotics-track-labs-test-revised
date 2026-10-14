---
title: "Lab 5 — Joint PID in simulation; a ROS 2 controller node"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 5"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 2 h of independent work |
| CLOs | **461-6** (design, tune and evaluate PID motion controllers; stability and robustness); 461-8 (a controller in a ROS 2 software architecture, used again in the project) |
| Reading | MR §11.2 (pp. 407–415), §11.3.1 (pp. 416–420), §11.4.1.1 (pp. 424–429, PID pseudocode Fig. 11.15 on p. 429); Åström–Murray §10.4 (pp. 306–308, windup) and §10.5 (pp. 308–311, implementation) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks, 31 tests), `lab/ros2/joint_sim_node.py` and `lab/ros2/joint_pid_node.py` (ROS 2 nodes, provided), `lab/data/joint_plants.json`, `lab/data/knee_swing.csv`, `lab/course_arm.py` (the local copy of the course arm) and `lab/make_data.py` (regenerates the data from `course_arm.py`; you do not need to run it) |
| Assessment | Auto-checked code (Python parts, 31 tests) + a short lab note (≤ 3 pages: figures, the expected-value table filled with your numbers, answers to the checkpoint questions, ROS 2 evidence, reflection). No formal report this week (formal reports: Labs 3, 8 and 11) |
| AI use | Allowed for code **with disclosure** (AI-use note in the lab note); you must be able to explain every line in an oral check |

## Why this lab matters

In Labs 2–4 the robot moved in simulation only because we applied a known torque (often the exact inverse-dynamics
torque). A real joint needs **feedback**: the dynamics you built in Weeks 1–4 is now the plant of a closed loop. Today you build the controller that the rest of the course uses: a discrete PID with
torque saturation and anti-windup, written exactly as the pseudocode of MR Fig. 11.15, and you put it inside a
**ROS 2 node** that talks to a simulated joint the same way it would talk to a ros2_control (Gazebo or hardware)
joint. Lab 6 (Week 6) adds gravity compensation and computed torque as separate controller laws next to
`JointPID`, and the Week 8 hardware rotation runs your PID on the course arm; the project's `control.py` has its own
implementations. The references today are steps, holds, ramps and a recorded knee swing; the trajectories of Week 10
(Lab 9) plug into the same functions later.

The plants are single joints, $\tau = M\ddot q + b\dot q + g(q)$ (MR Eq. 11.21) with $g(q) = g_\cos\cos q + g_\sin\sin q$, all in
`data/joint_plants.json` (illustrative values unless a source is given):

| Plant | What it is | $M$ (kg m²) | $b$ (N m s/rad) | $g(q)$ (N m) | $\tau_\max$ (N m) |
|---|---|---|---|---|---|
| `arm_j1` | course arm, base yaw (vertical axis: no gravity), other joints locked at $q_\text{home}$, 0.25 kg payload | 0.02484 | 0.01 | 0 | 1.08 (60 % of 1.8) |
| `arm_j2` | course arm, shoulder (horizontal axis), same locking and payload | 0.02405 | 0.01 | $-0.9052\cos q + 0.6961\sin q$ | 1.8 (60 % of 3.0) |
| `knee` | **biomedical**: powered transfemoral-prosthesis knee, thigh hanging, $q = -\theta_2$ of `leg_prosthesis_2r` (positive = flexion) | 0.1508 | 0.3 | $4.944\sin q$ | 40 |
| `eod_wrist` | **defence, protective use**: EOD arm wrist pitch holding a 2 kg suspect package | 0.043 | 0.2 | $3.4335\cos q$ | 15 |
| `mr_link` | the single link of MR §11.4.1 (p. 424), used to reproduce MR Fig. 11.13 | 0.5 | 0.1 | $0.981\cos q$ | none |

The course-arm values come from the shared course model (Course resources: Course robot models and hardware safety
checklist; numbers identical, copied as `lab/course_arm.py`): $M$ is the diagonal entry of the mass matrix at $q_\text{home}$ and $g(q)$ the gravity torque of
`modern_robotics`, both checked by the tests. $M_{jj}$ does not depend on $q_j$ (turning the distal links about axis $j$
does not change their inertia about that axis), so a locked-joint plant has constant inertia. The knee and EOD numbers
are the shared `leg_prosthesis_2r` (Lab 3 key `leg_2r`) shank and the Weeks 1–2 `eod_3r` link 3.

**About the real course arm.** Its smart servos are position-controlled: you cannot send them a torque.
Their internal loop is a controller of exactly the kind you design today: the nominal servo model of the course is a
second-order error dynamics with $\zeta = 0.9$, $\omega_n = 25$ rad/s, and in the Week 8 hardware rotation you identify the values of your own unit. The project hardware sessions (Weeks 10–13) use the position interface; this lab is
simulation-only.

## Learning objectives

By the end of the lab you can:

1. compute $\zeta$, $\omega_n$, roots, overshoot and 2 % settling time of an error dynamics, and measure them on a simulated
   response (MR §11.2);
2. explain the steady-state lag of P control with velocity inputs and remove it with PI or feedforward (MR §11.3.1);
3. design PD/PID gains for a target $\zeta$, $\omega_n$, find the stability bound on $K_i$ and verify it in simulation (MR §11.4.1.1);
4. implement a discrete PID with saturation and anti-windup and measure the effect of windup, gravity, encoder
   quantisation and velocity differencing;
5. run the same controller in a ROS 2 node, change its gains at run time and compare the ROS run with the offline
   simulation.

# Pre-lab check (complete before the session, 5 questions)

1. The error dynamics is $\dot\theta_e + 5\theta_e = 0$. What are the time constant and the 2 % settling time (exact and $4\mathfrak t$)?
2. Give $\zeta$, $\omega_n$ and the overshoot of $\ddot\theta_e + 12\dot\theta_e + 100\theta_e = 0$.
3. For `arm_j1` ($M = 0.02484$ kg m², $b = 0.01$ N m s/rad), which $K_p$, $K_d$ give $\zeta = 1$, $\omega_n = 10$ rad/s?
4. In the PID pseudocode of MR Fig. 11.15, which line estimates the joint velocity, and what can go wrong with it?
5. Why does a PD-controlled joint have a steady-state error in a vertical plane but not in a horizontal one (MR p. 426)?

# Setup

```bash
cd "<course folder>/461/w05/lab"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py   # 29 failed, 2 passed
```

Work in `starter.py` (keep the function names and signatures). Provided, do not edit: `load_plant`, `plant_gravity`,
`plant_rhs`, `rk4_step`, `time_grid`, `step_reference`, `ramp_reference`, `load_knee_swing`. Angles in rad, torques in
N m, time in s. Run a subset of the tests with `-k`, e.g. `LAB_IMPL=starter $PY -m pytest -q test_lab.py -k overshoot`.

# Procedure and checkpoints

### Checkpoint 1 — Error dynamics toolkit (≈ 10 min)

1. `second_order_params(a1, a0)`, `char_roots(coeffs)`, `is_stable(coeffs)` (MR Eqs. 11.3, 11.8).
2. `unit_error_response(zeta, wn, t)`: the three cases of MR p. 413 (treat $\lvert\zeta - 1\rvert < 10^{-9}$ as critical).
3. `overshoot_from_zeta`, `zeta_from_overshoot` (MR p. 415) and `step_metrics(t, e, ess)` with the definitions of MR p. 408.
4. `pytest -k "second_order or char_roots or unit_error or overshoot or step_metrics"`. The tests compare your response
   with the matrix exponential of the companion matrix (MR p. 410) and the overshoot with MR's 73 %, 16 % and 1.5 %.

**Checkpoint question:** for $\zeta = 0.8$ the 2 % settling time you measure is shorter than $4/(\zeta\omega_n)$; for $\zeta = 1$ it is
$5.83/\omega_n$, longer. Why?

### Checkpoint 2 — Velocity inputs (≈ 8 min)

1. `pi_velocity_gains(zeta, wn)` (MR p. 418) and `simulate_velocity_joint(...)` exactly as its docstring (MR Eqs.
   11.12–11.15, explicit Euler with the commanded velocity held over a sample; the integral is clamped while the
   command is at `qdot_max`).
2. `pytest -k "pi_velocity or p_control or feedforward or velocity_saturation"`.
3. **Experiment E1** (MR Fig. 11.8 on the course arm's base joint in velocity mode): reference ramp $q_d = 0.5t$ rad,
   $q(0) = -0.2$ rad, $K_p = 20$; compare P, PI ($K_i = 100$) and feedforward + PI. Plot $e(t)$ for the three.

### Checkpoint 3 — PID with torque inputs (≈ 20 min)

1. `pd_gains`, `ki_max`, `pid_char_poly` (MR Eqs. 11.26, 11.30, p. 427) and `pd_gravity_error` (MR Eq. 11.27, Newton).
2. `JointPID`: `reset(q0)` and `update(qd, qd_dot, q, qdot=None, tau_ff=0)` as in the class docstring — MR Fig. 11.15
   plus the torque limit and the anti-windup clamp. Keep `terms` (P, I, D) and `saturated` up to date: the ROS node
   publishes them.
3. `simulate_joint(plant, ctrl, t, qd, qd_dot, q0, ...)`: controller at the samples of `t`, torque held constant between
   samples (zero-order hold), plant integrated with `substeps` RK4 steps, optional encoder rounding and velocity by
   differencing.
4. `pytest -k "pd_gains or ki_bound or pd_gravity or joint_pid or simulate_joint or mr_fig"`. The test
   `test_mr_fig_11_13_pd_and_pid` reproduces MR Fig. 11.13: $K_p = 2.205$, $K_d = 2$ (critical damping for $M = 0.5$, $b = 0.1$),
   then $K_i = 1$.

**Checkpoint question:** in MR Fig. 11.13 the P term of the PID controller drops to zero at the end. Which term holds
the link against gravity then, and why can a PD controller not do that?

### Checkpoint 4 — Experiments (rest of the session; finish at home)

Use $\Delta t = 1$ ms and measured velocity unless stated. Save each figure with labelled axes and units.

* **E2 — MR Fig. 11.13.** `mr_link` from $q(0) = -\pi/2$ to $q_d = 0$, PD and PID ($K_i = 1$) for 10 s. Plot the errors and the
  P, I, D contributions (store `ctrl.terms` in a loop or rerun `update` on the logged data). Compare the PD final error
  with `pd_gravity_error`.
* **E3 — Bandwidth versus saturation (`arm_j1`).** Step of 0.5 rad, PD with $\zeta = 1$ and $\omega_n \in \{8, 10, 15, 20, 30\}$ rad/s, torque
  limit 1.08 N m. Table: $K_p$, $K_d$, measured settling time, linear prediction $5.83/\omega_n$, peak torque. When does the linear
  prediction stop being useful?
* **E4 — Power-on sag (`arm_j2`).** The shoulder is released at $q = q_d = -0.6$ rad with the controller starting from
  $e = 0$, $\int e = 0$. PD ($\zeta = 1$, $\omega_n = 20$) and PID with $K_i = 0.1$ and $0.3\,K_{i,\max}$. Plot $e(t)$; compare the PD sag with
  `pd_gravity_error`.
* **E5 — The $K_i$ bound.** Same as E4 but with no torque limit in the controller (the actuator still clips at 1.8 N m),
  $K_i = 0.9$ and $1.1\,K_{i,\max}$, 5 s. Plot $e(t)$ and the roots.
* **E6 — Prosthesis swing (biomedical).** Track `knee_swing.csv` (5° → 60° → 5° in 0.5 s, illustrative) with PD, $\zeta = 1$,
  $\omega_n = 30$: (a) measured velocity; (b) velocity by differencing a 12-bit encoder (`velocity="difference"`,
  `enc_res=2*np.pi/4096`); (c) *preview of Week 6*: (a) plus the feedforward torque $\tau_{ff} = M\ddot q_d + b\dot q_d + g(q_d)$. Report the
  largest tracking error, the peak torque and the chatter measure `np.std(np.diff(tau))`.
* **E7 — Windup.** `arm_j1`, step 0.5 rad, $\omega_n = 20$, $K_i = 0.1\,K_{i,\max}$: anti-windup on, off, and no torque limit. Overshoot of each.
* **E8 — EOD wrist (defence, optional).** Hold the package level from power-on, PD vs PID ($K_i = 29$), as in Workshop 5
  Problem 4. Does the result match your hand estimate?

### Checkpoint 5 — The ROS 2 node (guided, ≈ 30 min at home or in the lab)

Use the course ROS 2 Jazzy Docker image (or Ubuntu 24.04 with ROS 2 Jazzy). The nodes are plain `rclpy` scripts, so
no package build is needed. `joint_pid_node.py` imports **your** module (environment variable `LAB_IMPL`, default
`starter`) and uses your `JointPID` and `pd_gains` unchanged; `joint_sim_node.py` uses only provided functions.

```bash
# terminal 1 -- the controller first (it waits for the first joint state)
source /opt/ros/jazzy/setup.bash && cd <course folder>/461/w05/lab
LAB_IMPL=starter python3 ros2/joint_pid_node.py --ros-args -p plant:=arm_j2 -p wn:=20.0 \
    -p reference:=step -p q_target:=-0.3 -p t_step:=1.0 -p dt:=0.005
# terminal 2 -- the simulated joint (publishes /joint_states at 200 Hz, applies /effort_controller/commands)
python3 ros2/joint_sim_node.py --ros-args -p plant:=arm_j2 -p dt:=0.005
# terminal 3 -- inspect and record
ros2 node list ; ros2 topic list ; ros2 topic hz /joint_states
ros2 topic echo /pid_debug --csv > ros_run.csv      # [t, qd, q, e, tau, P, I, D, saturated]; Ctrl-C after ~4 s
ros2 param set /joint_pid ki 38.5                   # add integral action while it runs; watch the sag vanish
```

1. Run the step with PD, record `ros_run.csv`, then repeat with `-p ki:=38.5`.
2. Plot $q(t)$ from the CSV on top of `simulate_joint` with the same settings (`dt = 0.005`, `substeps = 10`, step at
   1 s). They agree to plotting accuracy; any difference comes from messages lost before the first sample and from
   timer jitter (`ros2 topic hz`). The test `test_ros_logic_matches_offline_simulation` shows that the node logic
   itself reproduces `simulate_joint` to $10^{-12}$.
3. Change a gain at run time with `ros2 param set` and describe what you see. Try `-p velocity:=difference`.
4. *Optional:* the controller node publishes `std_msgs/Float64MultiArray` on `/effort_controller/commands` and
   reads `/joint_states`, which are the topics of a ros2_control `effort_controllers/JointGroupEffortController` and
   `joint_state_broadcaster`. With a Gazebo Harmonic model of the course arm (the course-arm URDF published on the course site, plus a
   ros2_control tag) you can drive one joint unchanged by setting `cmd_size` and `cmd_index`. This is a project
   skill, not required today.

**Checkpoint question:** the PID runs once per received joint state. What happens to the controller's notion of time if
the simulator's timer is late or a message is dropped, and how would you fix that in a real system?

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab | 1: $\mathfrak t$, exact / approximate settling | 0.2 s; 0.782 s / 0.8 s |
| | 2: $\zeta$, $\omega_n$, overshoot | 0.6; 10 rad/s; 9.48 % |
| | 3: `arm_j1` $K_p$, $K_d$ | 2.484 N m/rad, 0.487 N m s/rad |
| E1 | P steady lag $c/K_p$; PI final error; PI and ff + PI most negative error | 0.025 rad; < $10^{-5}$ rad; −0.0146 and −0.027 rad |
| E2 | PD final error (= `pd_gravity_error`); PID $e(10\ \text{s})$; PID most negative error; $K_{i,\max}$ | 0.408 rad; 0.003 rad; −0.121 rad; 9.261 |
| E3 | $\omega_n$ = 8, 10, 15, 20, 30 rad/s: $K_p$ (N m/rad) | 1.590, 2.484, 5.590, 9.938, 22.36 |
| | settling time (s) / prediction $5.83/\omega_n$ | 0.73 / 0.73; 0.58 / 0.58; 0.41 / 0.39; 0.33 / 0.29; 0.25 / 0.19 |
| | peak torque (N m), overshoot | 0.795 (unsaturated), 1.08 for the others; none |
| E4 | PD sag; PID $K_i$ = 38.5: worst error, overshoot; $K_i$ = 115.4: worst error, overshoot | −0.1169 rad (−6.7°); −0.0898, +0.0009 rad; −0.0735, +0.0245 rad |
| | peak torque PD / PID 38.5 | 1.28 / 1.33 N m (limit 1.8) |
| E5 | $K_{i,\max}$; real part of the complex pair at $0.9\,K_{i,\max}$ | 384.8; −0.41 s⁻¹ (slowly decaying oscillation); $1.1\,K_{i,\max}$ diverges |
| E6 | (a) max error, peak torque, chatter | 4.46°, 16.1 N m, 0.185 N m |
| | (b) differenced 12-bit encoder: peak torque, chatter | 28.3 N m, 8.86 N m |
| | (c) with feedforward: max error | 0.035° |
| E7 | $K_i$; overshoot with anti-windup / without / no torque limit | 39.75; 16.3 % / 33.7 % / 24.7 % |
| E8 | PD sag; PID worst sag, time | 0.335 rad (19.2°); 0.265 rad at 0.24 s, 0.24° after 1 s |

All values were recomputed by the course team with the reference solution.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 5 into your cumulative library as `control.py`. The functions you write (15):

`second_order_params(a1: float, a0: float)`, `char_roots(coeffs)`, `is_stable(coeffs, tol: float=1e-12)`, `unit_error_response(zeta: float, wn: float, t)`, `overshoot_from_zeta(zeta: float)`, `zeta_from_overshoot(overshoot_pct: float)`, `step_metrics(t, e, ess: float | None=None, band: float=0.02)`, `pi_velocity_gains(zeta: float, wn: float)`, `simulate_velocity_joint(t, qd, qd_dot, kp: float, ki: float=0.0, q0: float=0.0, feedforward: bool=False, qdot_max: float | None=None)`, `pd_gains(M: float, b: float, zeta: float, wn: float)`, `ki_max(M: float, b: float, kp: float, kd: float)`, `pid_char_poly(M: float, b: float, kp: float, ki: float, kd: float)`, `pd_gravity_error(plant: dict, kp: float, qd: float)`, class `JointPID`, `simulate_joint(plant: dict, ctrl: JointPID, t, qd, qd_dot, q0: float, dq0: float=0.0, substeps: int=4, velocity: str='measured', enc_res: float | None=None, tau_ff=None, gravity: bool=True)`.

Names shared with other labs:

* The provided `rk4_step(f, x, h)` integrates $\dot x = f(x)$ (the input is held over the step). Lab 3's `rk4_step(f, t, x, h)` has an extra time argument; the two are different functions in different modules (`control.py` here, `forward_dynamics.py` in Lab 3). Lab 3's version gives the same step with `rk4_step(lambda t, x: f(x), 0.0, x, h)`.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab5_<ID>.py`, passing `LAB_IMPL=lab5_<ID> pytest -q test_lab.py` (31 tests). Add the functions
   to your `sbeg461_dynamics` library as `control.py` (same names).
2. `experiments.py` (or a notebook) that regenerates every figure and number of the lab note.
3. **Lab note** (PDF, ≤ 3 pages): figures for E1–E7, the expected-value table with your numbers, the four checkpoint
   questions answered, the ROS 2 evidence (overlay plot of `ros_run.csv` and the offline simulation, the output of
   `ros2 topic hz /joint_states`, one sentence on the run-time gain change), reflection and AI-use note.

# Reflection prompts (≤ 150 words, in the lab note)

* Which effect surprised you most: the size of the gravity sag, the windup overshoot, or the encoder chatter? What
  would it mean for a prosthesis user or for an operator handling a suspect package?
* Your controller now runs inside a ROS 2 node. Name one thing that can go wrong in the real system that the offline
  simulation cannot show.
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct (for example an anti-windup
  that also froze the integral when the error *leaves* saturation, or a derivative on the error that kicked at the
  setpoint step), and how did you find out?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `unit_error_response` wrong only for $\zeta = 1$ | Division by $\sqrt{\zeta^2 - 1}$; use the critical formula when $\lvert\zeta - 1\rvert < 10^{-9}$. |
| `step_metrics` overshoot negative or huge | Normalise by $e(0) - e_{ss}$ (the sign matters); overshoot is the extreme on the far side of $e_{ss}$, 0 if never crossed. |
| P ramp test off by one sample | Use the error *before* the update: $u_k$ from $e_k$, then $q_{k+1} = q_k + u_k\Delta t$. |
| `test_joint_pid_terms_and_differencing` fails at the first call | After `reset()` without `q0` the first differenced velocity is 0; after `reset(q0)` it is $(q - q_0)/\Delta t$. |
| Anti-windup test fails when leaving saturation | Freeze the integral only if the error has the **same sign** as the unclipped output. |
| PD-under-gravity error has the wrong sign | $\theta_e = \theta_d - \theta$; at rest $K_p\theta_e = g(\theta)$, not $-g$. |
| `simulate_joint` slightly off the analytic response | Torque must be computed once per sample and **held** for all `substeps`; do not recompute it inside the RK4 stages. |
| Huge torque spikes in E6(b) | That is the point of the experiment (encoder steps divided by 1 ms); do not "fix" it in `JointPID`. |
| ROS: `ModuleNotFoundError: starter` | Run from `lab/` or set `LAB_IMPL` to your file name without `.py`; the nodes add `lab/` to the path. |
| ROS: the joint falls before the controller acts | Start `joint_pid_node.py` first; the simulator applies zero torque until a command arrives. |
| ROS: no messages | `ros2 topic list` shows both topics? Same `ROS_DOMAIN_ID` in all terminals; source `/opt/ros/jazzy/setup.bash` everywhere. |

# Pre-lab answers (released after the session)

1. $\mathfrak t = 1/5 = 0.2$ s; exact 2 % settling $0.2\ln 50 = 0.782$ s, approximation $4\mathfrak t = 0.8$ s.
2. $\omega_n = 10$ rad/s, $\zeta = 12/20 = 0.6$, overshoot $e^{-\pi0.6/0.8} = 9.48\,\%$.
3. $K_p = M\omega_n^2 = 2.484$ N m/rad, $K_d = 2\zeta\omega_nM - b = 0.497 - 0.01 = 0.487$ N m s/rad.
4. `qdot = (q - qprev)/dt`. Encoder quantisation divided by a small `dt` gives large velocity steps, which the D term turns
   into torque chatter; it is also a delayed (backward-difference) estimate. MR p. 406 recommends a low-pass filter.
5. In a vertical plane the joint must produce the torque $mgr\cos\theta$ at rest; a PD law produces torque at rest only from a
   position error, so $K_p\theta_e = mgr\cos\theta \ne 0$. In a horizontal plane no torque is needed at rest, so $\theta_e \to 0$.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, deliverables and lab note are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: mass matrix and gravity torques of the course arm), with the matrix-exponential and ODE-solver references and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers: the random cubics drawn by `test_lab.py`, the `expm` unit responses, the `solve_ivp` closed-loop response, the course-arm model), `CMakeLists.txt`; `ros2/node_logic.hpp` (given: the ROS-free classes of the controller and simulator nodes, tested), `ros2/joint_pid_node.cpp` and `ros2/CMakeLists.txt` (given: the C++ controller node for Checkpoint 5, optional build). The data files are the same `data/joint_plants.json` and `data/knee_swing.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). Checkpoint 5 needs ROS 2 Jazzy, as in Python.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 5 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*overshoot*"            # only some tests (like pytest -k)
```

The C++ track has 23 test cases with the Python test names. With the untouched starter the program builds, 21 test cases fail with `TODO: <function>` and 2 pass (they check the given code: the plant data against `mrcpp`, and the ROS-free node classes), exactly as the Python starter gives 29 failed, 2 passed. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 15 as in Python, same order): `second_order_params`, `char_roots`, `is_stable`, `unit_error_response`, `overshoot_from_zeta`, `zeta_from_overshoot`, `step_metrics` (Checkpoint 1); `pi_velocity_gains`, `simulate_velocity_joint` (Checkpoint 2); `pd_gains`, `ki_max`, `pid_char_poly`, `pd_gravity_error`, `JointPID::reset`, `JointPID::update`, `simulate_joint` (Checkpoint 3). Given and complete: `load_plant`, `plant_gravity`, `plant_rhs`, `rk4_step`, `time_grid`, `step_reference`, `ramp_reference`, `load_knee_swing` and the `JointPID` constructor (it calls your `reset`, so every test that builds a controller fails until `reset` is written).

**Checkpoints.** 1: `./build/lab_tests -tc="*second_order*,*char_roots*,*unit_error*,*overshoot*,*step_metrics*"` (5 test cases); 2: `-tc="*pi_velocity*,*p_control*,*feedforward*,*velocity_saturation*"` (4); 3: `-tc="*pd_gains*,*ki_bound*,*pd_gravity*,*joint_pid*,*simulate_joint*,*mr_fig*"` (8); 4: `-tc="*arm_shoulder*,*integral_gain*,*encoder*"` (3); 5: `-tc="*ros*"` (2); finally `./build/lab_tests` must show 23 passed. The Python track counts 31 tests because pytest counts each $\zeta$ of `test_unit_error_response_matches_matrix_exponential` (6) and each plant of `test_ki_bound_is_the_stability_boundary` (4) separately; the C++ test case loops over them.

**Checkpoint 5 in C++** (optional build; needs ROS 2 Jazzy, e.g. the course Docker image). The controller node links **your** `starter.cpp`; the simulator stays the Python node `ros2/joint_sim_node.py` (it uses only the provided Python functions, so it runs with the untouched `starter.py`). Topics, parameters and the debug CSV are the same as in Python; give numeric parameters with a decimal point (`-p wn:=20.0`, `ros2 param set /joint_pid ki 38.5`).

```bash
source /opt/ros/jazzy/setup.bash && cd "<your Lab code folder>"
cmake -S cpp/ros2 -B build_ros && cmake --build build_ros
./build_ros/joint_pid_node --ros-args -p plant:=arm_j2 -p wn:=20.0 \
    -p reference:=step -p q_target:=-0.3 -p t_step:=1.0 -p dt:=0.005          # terminal 1
python3 ros2/joint_sim_node.py --ros-args -p plant:=arm_j2 -p dt:=0.005         # terminal 2
```

**Differences from the Python version.**

| Python | C++ |
|---|---|
| plant `dict`; `p["tau_max"]` may be `None` | `lab::Plant` with the same field names; `p.tau_max` is a `std::optional<double>`: `if (p.tau_max) { ... *p.tau_max ... }`; `dict(load_plant("arm_j1"), tau_max=None)` is `p.tau_max.reset();` |
| 1-D arrays `t`, `qd`, `e`, coefficient lists | `Eigen::VectorXd` (coefficients highest power first); `e[-1]` is `e(e.size() - 1)`; elementwise maths on `t.array()`, e.g. `(-wn * t.array()).exp()` |
| `np.roots(coeffs)` | eigenvalues of the companion matrix with `Eigen::EigenSolver<Eigen::MatrixXd>` (hint in `starter.cpp`); the roots are an `Eigen::VectorXcd` |
| tuple results `(zeta, wn)`, `(Kp, Ki)`, `(Kp, Kd)` | `std::pair<double, double>`: `auto [kp, kd] = lab::pd_gains(M, b, zeta, wn);` |
| dict results of `step_metrics`, `simulate_velocity_joint`, `simulate_joint` | structs `lab::StepMetrics`, `lab::VelocitySim`, `lab::JointSim` with the same field names (`m.settling_time`, `r.e`, `r.tau`) |
| `None` arguments (`ess`, `qdot_max`, `enc_res`, `qdot`, `q0`, `tau_max`); `tau_ff=None` | `std::optional<double>` (`std::nullopt` = `None`; a plain number converts); `tau_ff`: an empty `Eigen::VectorXd` |
| keyword arguments, e.g. `velocity="difference", enc_res=ENC` | positional, in the order of `lab.hpp`: `lab::simulate_joint(p, ctrl, t, qd, vd, q0, 0.0, 4, "difference", ENC)` |
| class `JointPID`; `c.terms["P"]` | struct `lab::JointPID` with public members of the same names; `c.terms.P`. `simulate_joint` takes the controller by reference: create a named controller (`lab::JointPID ctrl(kp, 0.0, kd, 1e-3);`) and pass `ctrl` |
| `lambda s: plant_rhs(...)` for `rk4_step` | a C++ lambda stored in a `lab::StateFunction` (example in `starter.cpp`) |
| `np.clip`, `math.copysign`, `np.round`, `float("nan")` | `std::clamp`, `std::copysign`, `std::nearbyint` (halves to even, as NumPy), `std::numeric_limits<double>::quiet_NaN()` (test with `std::isnan`) |
| `expm` (unit response) and `solve_ivp` (closed loop with gravity) references; `np.random.default_rng(0)` and `(1)` draws | values computed in Python and stored in `cpp_cases.json` |
| `ros2/joint_pid_node.py` (`PidNodeLogic`, `default_controller`), `ros2/joint_sim_node.py` (`JointSimLogic`) | `cpp/ros2/node_logic.hpp` (same classes, ROS-free, given and tested) and `cpp/ros2/joint_pid_node.cpp` (rclcpp node); the simulator node stays in Python |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

**Figures and tables for the lab note.** The plots stay outside the auto-checks. Write the experiments E1–E8 as a C++ program and save the time series with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool; for E2 store `ctrl.terms` after each `update` in your own loop. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab()`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_plant("knee")` and `lab::load_knee_swing()` find the data from any build folder. Deliverable 2 is then `experiments.cpp` plus your plotting script; the ROS overlay plot compares `ros_run.csv` with a CSV written from `lab::simulate_joint` (`dt = 0.005`, `substeps = 10`, step at 1 s). Deliverable 1 for the C++ track is `cpp/starter.cpp` passing all 23 test cases (add the functions to your library as `control.cpp` if you keep a C++ library); the lab note, reflection and AI-use note are unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../joint_plants.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| every controller test fails with `TODO: reset` | the given `JointPID` constructor calls `reset` | write `reset` first |
| compile error "cannot bind non-const lvalue reference ... to a temporary" | `lab::simulate_joint(p, lab::JointPID(...), ...)` | create a named controller and pass it |
| `bad optional access` | `*p.tau_max` (or `*qdot_max`, `*enc_res`) used while it is empty, e.g. for `mr_link` | test `if (p.tau_max)` first |
| "Eigen: wrong matrix size or index out of range" | e.g. the result vectors of `simulate_velocity_joint` / `simulate_joint` not sized (`Eigen::VectorXd v(n);`), or `e(e.size())` instead of `e(e.size() - 1)` | check the sizes; the test program does not crash, the test simply fails |
| Results wrong only in some entries, no compile error | an Eigen expression stored in `auto` (`auto d = e.array() - ess;` keeps references to temporaries) | write the type: `Eigen::VectorXd d = ...;` |
| ROS build: `Could not find a package configuration file provided by "ament_cmake"` | ROS 2 not sourced in this terminal | `source /opt/ros/jazzy/setup.bash`, delete `build_ros`, configure again |
| ROS: `parameter ... type mismatch` (or similar) at start-up | an integer given for a double parameter (`-p wn:=20`) | write `-p wn:=20.0` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
