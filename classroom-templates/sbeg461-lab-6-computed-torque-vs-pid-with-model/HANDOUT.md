---
title: "Lab 6 — Computed torque vs PID with model error on the course arm"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 6"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 2 h of independent work |
| CLOs | **461-6** (design, tune and evaluate PID, computed-torque and task-space motion controllers, analysing stability and robustness to model error); 461-1 (use of $M$, $h$ and their properties); 461-2 (simulation of the closed loop) |
| Reading | MR §11.4.1.2–11.4.3, printed pp. 428–436 (MR Eqs. 11.31–11.47, pseudocode MR Fig. 11.20 on p. 433); MR §8.6 (pp. 298–299, task-space dynamics) and Exercise 8.6 (p. 324, $\dot J$); Murray–Li–Sastry Ch. 4 §5 (pp. 189–195); Corke §9.4.3 (pp. 211–213) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks, 23 tests), `lab/data/lab06_config.json`, `lab/course_arm.py` (the course arm, copied from the shared course model, numbers identical) and `lab/make_data.py` (documents the configuration; you do not need to run it) |
| Assessment | Auto-checked code (23 tests) + a short lab note (≤ 3 pages: figures, the expected-value table with your numbers, answers to the checkpoint questions, reflection). No formal report this week (formal reports: Labs 3, 8 and 11) |
| AI use | Allowed for code **with disclosure** (AI-use note in the lab note); you must be able to explain every line in an oral check |

## Why this lab matters

In Lab 5 every joint was a separate single-joint plant. The real arm is one coupled system: moving the shoulder
changes the inertia seen by the elbow, gravity on the wrist depends on every joint, and a payload changes everything.
Today you control the **whole six-joint course arm** (simulated with its full rigid-body dynamics) with four
controllers of MR §11.4 — decentralized PID, PID + gravity compensation (MR Eq. 11.38), feedforward + PD and
computed torque (MR Eq. 11.37) — and finally in **task space** (MR Eq. 11.47) for a straight needle-guide approach.
Then you break the controller's model on purpose, as every real model is broken: link masses 20 % too large (as in
MR Exercise 11.11) and an unknown 0.25 kg payload (the course arm's maximum: a cup lifted by an assistive arm, a
specimen tube, or the unknown package in an EOD gripper). The question of the week is how much each controller
depends on the model, and why.

**The robot.** `course_arm.py` is the shared course arm (desktop UR-type 6R, myCobot-280 class; illustrative masses and
inertias in `COURSE_ARM_LINKS` of `lab/course_arm.py`; payload, friction, torque limits and test cases in `lab/data/lab06_config.json`). The plant carries the 0.25 kg payload at the tool point and has viscous friction
$b = (0.01, 0.01, 0.005, 0.001, 0.001, 0.0002)$ N m s/rad (illustrative; joints 1–2 as the Lab 5 plants). Torques are limited to
60 % of the rated values, $(1.08, 1.8, 0.9, 0.36, 0.36, 0.24)$ N m. The controller runs at 500 Hz ($\Delta t = 2$ ms, torque held between
samples) and sees the exact joint angles and velocities. The plant is integrated with Pinocchio's articulated-body
algorithm (it reproduces `modern_robotics.ForwardDynamics` to $10^{-8}$, test `test_plant_matches_modern_robotics_forward_dynamics`).

**About the real course arm.** Its smart servos are position-controlled (Week 5 WE5; you use them in the Week 8 hardware rotation): you cannot send them a torque, so
computed torque cannot be run on them directly. This lab is simulation-only. The project hardware runs use the position
interface; the model-based ideas of today reappear there as feedforward on the position command and as the
model-error analysis in your simulation-versus-hardware comparison.

## Learning objectives

By the end of the lab you can:

1. implement decentralized PID, PID + gravity compensation, feedforward + PD and computed torque for an $n$-joint arm,
   using one inverse-dynamics (RNEA) call per cycle for the model terms;
2. design computed-torque gains from $(\zeta, \omega_n)$, decentralized gains from the diagonal of $M$, and state the integral-gain bound;
3. verify the error-energy argument of MR Eqs. 11.40–11.43 in simulation and show where it fails under model error;
4. predict the steady-state error and the actual damping of computed torque with a wrong model ($\tilde Mk_p\theta_e = g - \tilde g$,
   $\zeta' = \zeta\sqrt\alpha$) and confirm the predictions in simulation;
5. implement task-space computed torque (MR Eq. 11.47) with the configuration error $\log(X^{-1}X_d)$, the velocity error and the
   feedforward acceleration, and compare it with joint-space control.

# Pre-lab check (complete before the session, 5 questions)

1. Which computed-torque gains $k_p$, $k_d$ give every joint $\zeta = 1$ and $\omega_n = 20$ rad/s? Up to which $k_i$ is computed torque with
   integral action stable (exact model)?
2. At $q_\text{home}$ the shoulder (joint 2) of the course arm with the payload has $M_{22} = 0.02405$ kg m² and $b_2 = 0.01$ N m s/rad. Which
   decentralized PD gains give $\zeta = 1$, $\omega_n = 20$ rad/s (Lab 5 rule)? Compare $K_{p,2}$ with the torque stiffness $\tilde M_{22}k_p$ of computed torque.
3. Why can $\tilde M(\theta)a + \tilde h(\theta,\dot\theta)$ be computed with a single call `mr.InverseDynamics(theta, dtheta, a, ...)`, and why is this
   cheaper than building $\tilde M$ with `mr.MassMatrix`?
4. At $q_\text{home}$, the wrist joint 4 has $M_{44} = 2.32\times10^{-3}$ kg m² with the payload but $3.91\times10^{-4}$ kg m² without it. If the
   controller does not know the payload, what are $\alpha = \tilde M_{44}/M_{44}$, $\zeta'$ and $\omega_n'$ of computed torque designed for $\zeta = 1$, $\omega_n = 20$ rad/s?
5. With viscous friction $B$ in the plant (not compensated), what becomes of $\dot V = -\dot\theta^{\mathsf T}K_d\dot\theta$ (MR Eq. 11.43)?

# Setup

```bash
cd "<course folder>/461/w06/lab"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py   # 19 failed (NotImplementedError), 4 passed
```

Work in `starter.py` (keep the function names and signatures). Provided, do not edit: `load_config`, `arm_model`,
`make_case`, `tau_limits`, `time_grid`, `quintic_reference`, `cartesian_reference`, `rk4_step`, `Plant`, `simulate_arm`, `_diag`.
`make_case(name)` returns `(plant, model)` for the three model-error cases `"exact"`, `"masses_120"` and `"payload_unknown"`;
a controller always receives the **model**, `simulate_arm` always receives the **plant**. Angles in rad, torques in N m,
time in s. One closed-loop run of 2 s takes about 1 s with the reference solution; if yours takes much longer, you are
probably calling `mr.MassMatrix` inside the loop (see pre-lab question 3).

# Procedure and checkpoints

### Checkpoint 1 — Gains and joint-space laws (≈ 15 min)

1. `ct_gains`, `ct_ki_max`, `decentralized_gains` (MR Eqs. 11.26, 11.30, 11.33).
2. `model_inverse_dynamics` (one RNEA call plus $\tilde b\dot\theta$), `pid_torque` (MR Eq. 11.23 at each joint), `pd_gravity_torque`
   (MR Eq. 11.38), `feedforward_pd_torque` (MR Eq. 11.32 + PID, model evaluated on the **reference**), `computed_torque`
   (MR Eq. 11.37, model evaluated on the **measured** state).
3. `pytest -k "gains or pid_torque or gravity or feedforward or computed_torque"`. The tests compare your
   computed torque with `mr.ComputedTorque` and check the key identity: with an exact model the plant acceleration is
   exactly $\ddot\theta_d + k_p\theta_e + k_d\dot\theta_e$ (feedback linearization).

**Checkpoint question:** `mr.ComputedTorque` multiplies $K_i$ by `eint + e`, not by `eint`. Read its source. What does
it assume about the `eint` you pass, and how does the test account for it?

### Checkpoint 2 — Controller object, energy and metrics (≈ 10 min)

1. `error_energy` (MR Eq. 11.41), `JointController.reset` and `__call__` (update the integral *before* computing the torque,
   as in MR Fig. 11.20), `run_metrics` (MR Fig. 11.19: error norms and the effort $\int\tau^{\mathsf T}\tau\,dt$).
2. `pytest -k "energy or joint_controller or metrics or closed_loop or steady_state or setpoint"`.

**Checkpoint question:** `test_error_energy_and_its_derivative` finds $\dot V = -\dot\theta^{\mathsf T}(K_d + B)\dot\theta$. Where did the friction term come
from, and which term of MR Eq. 11.42 vanished because $\dot M - 2C$ is skew-symmetric?

### Checkpoint 3 — Task space (≈ 15 min)

1. `body_jacobian_dot` (MR Exercise 8.6b: $\partial J_{b,i}/\partial\theta_j = [\mathrm{ad}_{J_{b,i}}]J_{b,j}$ for $i < j$), `task_errors` (MR p. 436),
   `task_feedforward_accel` ($\tfrac{d}{dt}([\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d) = [\mathrm{Ad}_{X^{-1}X_d}]\dot{\mathcal V}_d - [\mathrm{ad}_{\mathcal V}][\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d$; derive it from
   $\tfrac{d}{dt}(X^{-1}X_d)$), `model_task_space_dynamics` (MR Eqs. 8.90–8.91), `task_space_ct` and `TaskController`.
2. `pytest -k "jacobian_dot or task"`. The tests check your $\dot J_b$ and feedforward term against finite differences, and
   that your controller equals MR Eq. 11.47 written with $\Lambda$ and $\eta$.

**Checkpoint question:** show that $J_b^{\mathsf T}(\Lambda a + \eta) = MJ_b^{-1}(a - \dot J_b\dot\theta) + h$. Why does this make the task-space controller
as cheap as joint-space computed torque?

### Checkpoint 4 — Experiments (rest of the session; finish at home)

Use the configuration file (`load_config()`): $\Delta t = 2$ ms, 2 RK4 substeps, torque limits on unless stated. The joint move
is a quintic rest-to-rest move from $q_\text{home} = (0, -0.6, 1.4, -2.4, -1.57, 0)$ by $\Delta q = (0.8, 0.4, -0.5, 0.4, 0.4, 0.6)$ rad in
1.5 s, then a 0.5 s hold. Gains: $\zeta = 1$, $\omega_n = 20$ rad/s for every controller; decentralized gains from the **model's**
$M_{jj}(q_\text{home})$; integral gains $K_{i,j} = 0.1\,K_{i,\max,j}$ (decentralized) and $k_i = 2000$ s⁻³ (computed torque). Save each figure with
labelled axes and units.

* **E1 — Exact model.** Run the six controllers: PID, PD + $\tilde g$, PID + $\tilde g$, feedforward + PD, computed torque (CT) and
  CT + I. Table: max $\lVert\theta_e\rVert$, final $\lVert\theta_e\rVert$, effort. Plot $\lVert\theta_e(t)\rVert$ for all six (log scale).
* **E2 — Masses 20 % too large** (`masses_120`, MR Exercise 11.11). Same six runs and table. Which controllers saturate a
  joint, and why do those that use the model as a torque *gain* behave differently from those that use it only as a
  feedforward?
* **E3 — Unknown payload** (`payload_unknown`). Same six runs and table. Compute $\alpha_j = \tilde M_{jj}/M_{jj}$ at $q_\text{home}$ for every joint and the
  resulting $\zeta_j' = \sqrt{\alpha_j}$, $\omega_{n,j}' = 20\sqrt{\alpha_j}$. Plot the six joint errors of CT. Which joint dominates, and why?
* **E4 — Bandwidth and the static-error prediction.** CT ($k_i = 0$) for `masses_120` and `payload_unknown` with $\omega_n$ = 10, 20, 40 rad/s.
  Plot max error against $\omega_n$ (log–log). Predict the error at the end of the hold by solving $k_p\tilde M(\theta)\theta_e = g(\theta) - \tilde g(\theta)$,
  $\theta = \theta_\text{final} - \theta_e$, with Newton's method (as in `test_ct_steady_state_error_under_model_error`) and compare with the simulation.
* **E5 — Error energy.** Setpoint regulation with PD + $\tilde g$ (no torque limit, 1.5 s) from $q_\text{home}$ to
  $q_\text{home} + (0.1, -0.1, 0.1, -0.1, 0.1, 0.1)$ rad. Plot $V(t)$ (MR Eq. 11.41, evaluated with the **plant** inertia) for `exact` and
  `payload_unknown`. Does $V$ ever increase? Where does the second run end?
* **E6 — Task space** (biomedical: needle-guide approach). Task-space CT moves the tool point 50 mm straight along $-\hat z_s$
  (towards the skin entry point) in 1 s with fixed orientation, then holds 0.5 s; $k_p = 400$, $k_d = 40$, $k_i = 0$ and 2000. Report
  the max and final tool-position error (mm) for the three cases. Would you allow this controller to insert a needle
  with an unidentified tool?
* **E7 (optional) — Fix the model.** For `payload_unknown`, estimate the payload mass from the CT + I run: at the end of
  the hold the integral term supplies the missing gravity torque. Use joint 2 and the payload's lever arm to estimate
  $m$, add it to the model (`arm_model(payload=m_est)`) and rerun E3. This is what a real system does when it weighs the
  payload (EOD gripper, assistive feeding arm).

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab | 1: $k_p$, $k_d$; $k_{i,\max}$ | 400 s⁻², 40 s⁻¹; 16 000 s⁻³ |
| | 2: $K_{p,2}$, $K_{d,2}$ | 9.621 N m/rad, 0.952 N m s/rad (= $\tilde M_{22}k_p$ at $q_\text{home}$) |
| | 4: $\alpha_4$, $\zeta'$, $\omega_n'$ | 0.168, 0.410, 8.21 rad/s |
| E1 exact | max / final $\lVert\theta_e\rVert$ (rad), effort (N² m² s) | PID 0.239 / 0.0069, 3.55; PD + $\tilde g$ 0.0474 / 0.0000, 3.63; PID + $\tilde g$ 0.0250 / 0.0014, 3.62; FF + PD 0.0001 / 0.0000, 3.62; CT 0.0009 / 0.0000, 3.62; CT + I 0.0004 / 0.0000, 3.62 |
| | CT peak torque / limit | 0.70 |
| E2 `masses_120` | max / final, effort | PID 0.199 / 0.0053, 3.65; PD + $\tilde g$ 0.0749 / 0.0579, 3.85; PID + $\tilde g$ 0.0513 / 0.0020, 3.73; FF + PD 0.0482 / 0.0413, 3.82; CT 0.0419 / 0.0417, 3.63; CT + I 0.0308 / 0.0001, 3.62 |
| | peak torque / limit | 1.00 (a joint saturates) for PID, PD + $\tilde g$, PID + $\tilde g$ and FF + PD; 0.81 for CT and CT + I |
| E3 `payload_unknown` | max / final, effort | PID 0.802 / 0.131, 3.22; PD + $\tilde g$ 0.708 / 0.451, 2.33; PID + $\tilde g$ 0.647 / 0.163, 3.34; FF + PD 0.661 / 0.408, 2.48; CT 0.748 / 0.489, 3.04; CT + I 0.681 / 0.0825, 3.55 |
| | $\alpha_j$ (joints 1–6) | 0.408, 0.404, 0.294, 0.168, 0.105, 1.000 |
| E4 | CT max error, $\omega_n$ = 10 / 20 / 40: `masses_120`; `payload_unknown` | 0.1615 / 0.0419 / 0.0108; 0.982 / 0.748 / 0.344 rad |
| | predicted static $\lVert\theta_e\rVert$ at the end (Newton) | 0.0417 rad (simulated 0.0417); 0.491 rad (simulated 0.489) |
| E5 | $V(0)$, $V(1.5\ \text{s})$, final $\lVert\theta_e\rVert$: exact; payload unknown | 0.1271 J, ≈ 0, 0.000 rad (never increases); 0.0476 J, 0.0555 J, 0.544 rad (increases) |
| E6 | max / final tool error, $k_i = 0$; $k_i = 2000$: exact | 0.012 / 0.000 mm; 0.009 / 0.001 mm |
| | `masses_120` | 4.25 / 4.14 mm; 2.93 / 0.003 mm |
| | `payload_unknown` | 66.7 / 51.1 mm; 58.5 / 50.8 mm |

All values were recomputed by the course team with the reference solution. Small differences
($< 10^{-4}$ rad) can come from a different but equivalent order of operations.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 6 into your cumulative library as `control.py` (next to Lab 5's functions). The functions you write (17):

`ct_gains(zeta: float, wn: float)`, `ct_ki_max(kp: float, kd: float)`, `decentralized_gains(model: dict, q_ref, zeta: float, wn: float, ki_fraction: float=0.0)`, `model_inverse_dynamics(model: dict, q, dq, ddq)`, `pid_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd)`, `pd_gravity_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd, model: dict)`, `feedforward_pd_torque(q, dq, eint, qd, dqd, ddqd, Kp, Ki, Kd, model: dict)`, `computed_torque(q, dq, eint, qd, dqd, ddqd, kp, ki, kd, model: dict)`, `error_energy(q, dq, qd, Kp, model: dict)`, class `JointController`, `run_metrics(t, qd, q, tau)`, `body_jacobian_dot(Blist, q, dq)`, `task_errors(X, V, Xd, Vd)`, `task_feedforward_accel(X, V, Xd, Vd, dVd)`, `model_task_space_dynamics(model: dict, q, dq)`, `task_space_ct(q, dq, Xint, Xd, Vd, dVd, kp, ki, kd, model: dict)`, class `TaskController`.

Names shared with other labs:

* **Two names differ from other labs on purpose.** `run_metrics(t, qd, q, tau)` (error norms and effort of a run) is not the `tracking_metrics(t, cmd, meas, max_lag)` you will meet in the Week 8 hardware rotation (sim-vs-hardware lag and RMS). `model_task_space_dynamics(model, q, dq)` is not Lab 4's `task_space_dynamics(M, h, J, Jdot, dthetalist)`; it computes $M$, $h$ and $J_b$ from the model and then applies MR Eqs. 8.90–8.91.
* The provided `rk4_step(f, x, h)` and `time_grid(T, dt)` are the Lab 5 helpers (same signatures); keep one copy in `control.py`.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab06_<ID>.py`, passing `LAB_IMPL=lab06_<ID> pytest -q test_lab.py` (23 tests). Add the functions
   to `sbeg461_dynamics/control.py` next to Lab 5's `JointPID`.
2. `experiments.py` (or a notebook) that regenerates every figure and number of the lab note.
3. **Lab note** (PDF, ≤ 3 pages): figures for E1–E6, the expected-value table with your numbers, the three checkpoint
   questions answered, the E3 $\alpha$ table with your explanation, reflection and AI-use note.

# Reflection prompts (≤ 150 words, in the lab note)

* In E1 computed torque is almost perfect; in E3 it is no better than PID. What exactly did the model give the
  controller in E1 that it lost in E3? Relate your answer to $\alpha_j$ and to the effective stiffness $\tilde M_{jj}k_p$.
* A rehabilitation or assistive arm picks up objects of unknown mass every day. Which controller would you choose,
  and what would you add to it? Would your answer change for an EOD arm handling a suspect package?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct (for example
  computing $\tilde M$ with `MassMatrix` inside the loop, evaluating the feedforward model on the measured state instead of the
  reference, or a $\dot J$ with the sum over $j < i$), and how did you find out?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `test_computed_torque_matches_modern_robotics` off only when $K_i \ne 0$ | `mr.ComputedTorque` uses $K_i(\text{eint} + e)$; your law must use $K_i\,\text{eint}$ with the integral already updated (MR Fig. 11.20). |
| Computed torque correct for scalar gains, wrong for vectors | Use `_diag(K, n) * e` (element-wise), not `K @ e` with a vector `K`. |
| Feedforward test fails by a small amount | The feedforward model is evaluated at $(\theta_d, \dot\theta_d, \ddot\theta_d)$, friction included as $\tilde b\dot\theta_d$. |
| A run takes > 10 s | `mr.MassMatrix` (6 RNEA calls) in the loop. Use `model_inverse_dynamics(model, q, dq, a)`. |
| `JointController` integral off by one sample | Update `eint += e*dt` **before** computing the torque, every call, including the first. |
| `body_jacobian_dot` fails | For the **body** Jacobian column $i$ depends on joints $j > i$ (the space Jacobian is the opposite, MR Exercise 8.6a). |
| Task-space feedforward test fails | Use the **actual** twist $\mathcal V = J_b\dot\theta$ in $[\mathrm{ad}_{\mathcal V}]$ and $[\mathrm{Ad}_{X^{-1}X_d}]$ (not its inverse). |
| E6 diverges | Check the sign of $X_e$: $\log(X^{-1}X_d)$, not $\log(X_d^{-1}X)$; and $V_e = [\mathrm{Ad}]\mathcal V_d - \mathcal V$. |
| E5 energy increases slightly with the exact model | Use the **plant** inertia in $V$ and run without torque limit; the RK4 step must use the held torque. |

# Pre-lab answers (released after the session)

1. $k_p = \omega_n^2 = 400$ s⁻², $k_d = 2\zeta\omega_n = 40$ s⁻¹ (MR Eq. 11.33); $s^3 + 40s^2 + 400s + k_i$ is stable for $0 < k_i < 40\cdot400 = 16\,000$ s⁻³.
2. $K_{p,2} = M_{22}\omega_n^2 = 9.621$ N m/rad, $K_{d,2} = 2\zeta\omega_nM_{22} - b_2 = 0.962 - 0.01 = 0.952$ N m s/rad. At $q_\text{home}$ the computed-torque
   stiffness $\tilde M_{22}k_p$ is the same 9.621 N m/rad: the difference is that computed torque uses the whole, configuration-
   dependent $\tilde M(\theta)$ (coupling and changes along the move) and compensates $\tilde h$.
3. RNEA computes $\tau = M(\theta)\ddot\theta + c(\theta,\dot\theta) + g(\theta)$ for any $\ddot\theta$; with $\ddot\theta = a$ it returns $\tilde Ma + \tilde c + \tilde g$ directly (MR §8.3).
   `MassMatrix` needs $n = 6$ RNEA calls and $\tilde h$ one more, so the single call is about 7 times cheaper.
4. $\alpha_4 = 3.91\times10^{-4}/2.32\times10^{-3} = 0.168$; $\zeta' = \sqrt{0.168} = 0.410$; $\omega_n' = 20\sqrt{0.168} = 8.21$ rad/s — a soft, oscillatory wrist.
5. $\dot V = -\dot\theta^{\mathsf T}(K_d + B)\dot\theta$: friction is extra damping and only helps the argument.

# C++ track

You may do this lab in C++17 instead of Python. The functions, configuration file, checkpoints, experiments, expected values and lab note are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `inverse_dynamics`, `mass_matrix`, `gravity_forces`, `computed_torque`, `forward_dynamics`, `jacobian_body`), with finite differences, with the closed-loop identities of MR §11.4, with an independent Newton solution and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided helpers, translated from `starter.py` — do not edit), `course_arm.hpp` / `course_arm.cpp` (the course arm, numbers identical to `course_arm.py` — do not edit), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random states drawn by `test_lab.py`), `CMakeLists.txt`. The configuration is the same `data/lab06_config.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). Pinocchio is not needed for the C++ track.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 6 --cpp                     # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*task*"                 # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 22 test cases fail with `TODO: <function>`: this is expected (the whole test program runs in about 1 s once your code is written). Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 17 as in Python): `ct_gains`, `ct_ki_max`, `decentralized_gains`, `model_inverse_dynamics`, `pid_torque`, `pd_gravity_torque`, `feedforward_pd_torque`, `computed_torque` (Checkpoint 1); `error_energy`, `JointController::reset` and `JointController::operator()` (the Python `__call__`), `run_metrics` (Checkpoint 2); `body_jacobian_dot`, `task_errors`, `task_feedforward_accel`, `model_task_space_dynamics`, `task_space_ct`, `TaskController::reset` and `TaskController::operator()` (Checkpoint 3). Given and complete (`helpers.cpp`): `load_config`, `arm_model`, `make_case`, `tau_limits`, `time_grid`, `quintic_reference`, `cartesian_reference`, `rk4_step`, `diag_gain` (the Python `_diag`), `Plant`, `simulate_arm` and the constructors of both controllers. As in Python you may use `mr::inverse_dynamics`, `mr::mass_matrix`, `mr::gravity_forces`, `mr::jacobian_body`, `mr::fkin_body`, `mr::trans_inv`, `mr::matrix_log6`, `mr::se3_to_vec`, `mr::adjoint` and `mr::ad`.

**Checkpoints** (22 C++ test cases with the Python test names; the Python track counts 23 because it also has `test_starter_compiles`, which in C++ is the build itself). 1: `./build/lab_tests -tc="*gains*,*pid_torque*,*gravity_torque*,*feedforward*,*computed_torque*"` (8 test cases); 2: `-tc="*energy*,*joint_controller*,*metrics*,*closed_loop*,*steady_state*,*setpoint*"` (6); 3: `-tc="*jacobian_dot*,*task*"` (5); the three data/infrastructure tests (`*config*,*plant*,*references*`) also call `model_inverse_dynamics` or `run_metrics` (marked "C++ only"), so with the starter they fail instead of passing as in Python; at the end `./build/lab_tests` must show 22 passed.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| model dict `model["Mlist"]`, `model["b"]`, … | `lab::Robot` (fields `Slist`, `Blist`, `M`, `Mlist`, `Glist`, `g`, `tau_max`, `q_home`, `b`, `payload`, `mass_scale`): `model.Mlist`, `model.b` |
| `CFG = load_config()`, `CFG["joint_move"]["q0"]` | `sbeg::json CFG = lab::load_config();` `sbeg::as_vec(CFG["joint_move"]["q0"])`, `CFG["dt"].get<double>()` |
| a gain that is a scalar or a vector; `_diag(K, n)` | `lab::Gain`: pass `400.0` or an `Eigen::VectorXd` (both convert automatically); `lab::diag_gain(K, n)` returns the length-$n$ vector |
| `ct_gains` returns `(kp, kd)`; `task_errors` `(Xe, Ve)`; `model_task_space_dynamics` `(Lambda, eta)` | `std::pair`: `auto [kp, kd] = lab::ct_gains(1.0, 20.0);` |
| `decentralized_gains` returns `(Kp, Ki, Kd)`; `quintic_reference` `(qd, dqd, ddqd)`; `cartesian_reference` `(Xd, Vd, dVd)` | structs `lab::PIDGains`, `lab::JointReference`, `lab::CartesianReference` with these field names; `(N, n)` arrays are `Eigen::MatrixXd` (row $k$ = sample $k$), `Xd` is an `mr::TransformList` |
| `run_metrics` and `simulate_arm` return dicts | structs `lab::Metrics{max_err, rms_err, final_err, effort, peak_tau}` and `lab::SimResult{t, q, dq, tau}` |
| classes with `reset()` and `__call__(k, q, dq)`; attributes `eint`, `Xint` | classes derived from `lab::Controller` with `reset()` and `operator()(k, q, dq)`; public members `eint`, `Xint`; `model=None` → leave the argument out (`std::optional`) |
| `qd[k]`, `np.linalg.solve(J, b)`, `np.linalg.inv(J)`, `np.diag(M)`, `a * b` (element-wise) | `qd.row(k).transpose()`, `J.partialPivLu().solve(b)`, `J.inverse()`, `M.diagonal()`, `a.cwiseProduct(b)` |
| `Plant` uses Pinocchio's ABA | `Plant` uses `mr::forward_dynamics` (identical to $10^{-12}$; the closed-loop runs agree with the Python reference to $10^{-6}$) |
| random states of `np.random.default_rng(461)` | stored in `cpp_cases.json` (same numbers as the Python tests) |
| `experiments.py`, matplotlib figures | write `cpp/experiments.cpp` and save each run with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool (see below) |

**Experiments and figures.** E1–E7 and the expected values are the same; a 2 s closed-loop run takes a fraction of a second in C++. To build an experiments program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp course_arm.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_config()` finds the configuration from any build folder. Deliverable 2 is then `experiments.cpp` plus your plotting script; the lab note is unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../lab06_config.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| Every `JointController` test fails with `TODO: JointController::reset` although `operator()` is written | the given constructor calls `reset()` | write `reset()` first |
| "Eigen: wrong matrix size or index out of range" | e.g. `K * e` with a `Gain`, a row of `qd` used as a column, or `eint` not sized in `reset()` | use `lab::diag_gain(K, n).cwiseProduct(e)`, `qd.row(k).transpose()`, `Eigen::VectorXd::Zero(qd.cols())`; the test program does not crash, the test simply fails |
| Results wrong only in some entries, no compile error | an Eigen product stored in `auto` (`auto C = A * B;` keeps references to temporaries) | write the type: `Eigen::MatrixXd C = A * B;` |
| `test_computed_torque_matches_modern_robotics` off only when $K_i \ne 0$ | `mr::computed_torque` uses $K_i(\text{eint} + e)$, like the Python `ComputedTorque` | use $K_i\,\text{eint}$ with the integral already updated (MR Fig. 11.20) |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
