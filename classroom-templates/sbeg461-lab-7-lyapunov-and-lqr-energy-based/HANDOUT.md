---
title: "Lab 7 — Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 7"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 2 h of independent work |
| CLOs | **461-6** (design, tune and evaluate motion controllers, analysing stability and robustness to model error); 461-1 (properties of $M$, $C$ and $g$: Christoffel symbols, skew-symmetry of $\dot M - 2C$, passivity) |
| Reading | MR §8.1.2, printed pp. 275–277 (MR Eqs. 8.13–8.19, Proposition 8.1); MR §11.4.2.2, pp. 434–435 (MR Eqs. 11.38–11.43) and Exercise 11.12 (p. 460, the Krasovskii–LaSalle principle); Åström–Murray §6.3 "Linear quadratic regulators", pp. 190–191 (Eqs. 6.26–6.27) and Exercise 6.15 (Bryson's rule, p. 200); Tedrake, *Underactuated Robotics*, §3.4.1 "Linearizing the manipulator equations", §3.4.3 "LQR feedback" and Ch. 8 §8.1 "Basic derivation" |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks, 23 tests), `lab/data/lab07_config.json`, `lab/robots.py` (the planar 2R and the course arm as used in this lab), `lab/course_arm.py` (the course arm, copied from the shared course model, numbers identical) and `lab/make_data.py` (documents the configuration; you do not need to run it) |
| Assessment | Auto-checked code (23 tests) + a short lab note (≤ 3 pages: figures, the expected-value table with your numbers, answers to the checkpoint questions, reflection). No formal report this week (formal reports: Labs 3, 8 and 11) |
| AI use | Allowed for code **with disclosure** (AI-use note in the lab note); you must be able to explain every line in an oral check |

## Why this lab matters

In Lab 6 you saw that PD + gravity compensation brings the course arm to rest at the setpoint, and you plotted the
"error energy" of MR Eq. 11.41 going down. Today you check **why**, and you learn a second, systematic way of
choosing gains.

**Part A — energy as a Lyapunov function.** You build the Coriolis matrix $C(\theta,\dot\theta)$ from the Christoffel symbols
of $M(\theta)$ (MR Eqs. 8.17, 8.19), verify numerically that $\dot M - 2C$ is skew-symmetric (MR Proposition 8.1) on the planar 2R
and on the six-joint course arm, and use it to show that the manipulator is **passive**: the stored energy can only
grow by the power $\dot\theta^{\mathsf T}\tau$ you put in. Then you evaluate $V$ and $\dot V$ (MR Eqs. 11.41–11.43) along simulated runs,
and break each assumption of the proof in turn: no damping ($K_d = 0$, where the Krasovskii–LaSalle argument fails),
sampling (a digital controller is not the continuous law of the proof), and a wrong gravity model (the arm stops
somewhere else, which you predict with Newton's method).

**Part B — LQR.** Near a hold pose the arm is almost linear. You linearise the dynamics about an equilibrium
(Tedrake §3.4.1), choose weights with Bryson's rule, solve the algebraic Riccati equation (Åström–Murray Eq. 6.27) and get
the gain matrix $K$ that minimises $J = \int(x^{\mathsf T}Qx + u^{\mathsf T}Ru)\,dt$. You also write the PD + gravity-compensation law as a
state-feedback gain $K_0$, compute its cost with a Lyapunov equation, and use $K_0$ as the starting point of Kleinman's
iteration, which reaches the LQR gain by solving only Lyapunov equations. The same quadratic function $x^{\mathsf T}Px$ is then both
the optimal cost and a Lyapunov function of the closed loop: Part A and Part B are the same idea.

**The robots.** The planar 2R of the shared course models (Course resources: Course robot models and hardware safety
checklist; $L = (0.6, 0.4)$ m, uniform rods $m = (2, 1)$ kg, vertical plane, gravity along $-\hat y$; the MR Studio "Planar 2R
arm") stands for the shoulder–elbow of an assistive or rehabilitation arm holding a patient's forearm, or of an EOD
manipulator holding a gripper still while the operator inspects a suspect package. Its plant has viscous friction
$b = (0.05, 0.02)$ N m s/rad (illustrative). The hold pose is $q_\text{eq} = (0.6, 0.9)$ rad (forearm almost vertical), an
**unstable** equilibrium of the uncontrolled arm. The course arm (no payload, friction as in Lab 6) is used for the
skew-symmetry check, the energy rate and a 12-state LQR. Its dynamics comes from `modern_robotics`; the 2R has closed-form
dynamics (checked against `modern_robotics` by the tests), so a 4 s closed-loop run takes about 0.1 s.

**About the real course arm.** Its smart servos are position-controlled (Week 8), so neither law of today can be
sent to them as a torque. This lab is simulation-only; in the Week 8 hardware rotation you add gravity compensation to
the position command and see the offset of experiment E3 on the real arm.

## Learning objectives

By the end of the lab you can:

1. compute the Coriolis matrix from the Christoffel symbols, verify $\dot M - 2C$ skew-symmetric and explain which property of the
   manipulator (passivity) it expresses;
2. evaluate the error energy $V$ and its rate $\dot V$ for PD + gravity compensation, and identify in simulation each assumption of the
   stability proof (damping, exact gravity, continuous time);
3. predict the rest configuration of PD + gravity compensation with a wrong gravity model;
4. linearise the manipulator equations about an equilibrium and decide from the eigenvalues whether it is stable;
5. design an LQR controller (Bryson's rule, Riccati equation), solve Lyapunov equations, and compare LQR with PD in cost, poles,
   peak torque and settling time.

# Pre-lab check (complete before the session, 5 questions)

1. At $q_\text{eq} = (0.6, 0.9)$ the 2R has $M_{11} = 0.80299$ and $M_{22} = 0.05349$ kg m². Which diagonal PD gains give each joint $\zeta = 1$,
   $\omega_n = 8$ rad/s with the Lab 5 rule $K_p = M_{jj}\omega_n^2$, $K_d = 2\zeta\omega_nM_{jj} - b_j$?
2. MR Eq. 11.43 gives $\dot V = -\dot\theta^{\mathsf T}K_d\dot\theta \le 0$. Why is this not yet enough to conclude $\theta \to \theta_d$, and what does the
   Krasovskii–LaSalle principle (MR Exercise 11.12) add?
3. Show that $x^{\mathsf T}Nx = 0$ for every vector $x$ if $N$ is skew-symmetric. Which term of MR Eq. 11.42 vanishes because of this?
4. A single link $M\ddot\theta + b\dot\theta + mgr\cos\theta = \tau$ (MR Eq. 11.21) is held at $\theta^*$ by $\tau^* = mgr\cos\theta^*$. Write $A$ and $B$ of the
   linearisation in $x = (\theta - \theta^*, \dot\theta)$, $u = \tau - \tau^*$. For which $\theta^*$ is the uncontrolled link unstable?
5. With Bryson's rule, $|q_j| \le 0.05$ rad, $|\dot q_j| \le 0.5$ rad/s and a feedback-torque budget $(5, 2)$ N m, what are $Q$ and $R$?

# Setup

```bash
cd "<your Lab code folder>"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py   # 22 failed (NotImplementedError), 1 passed
```

Work in `starter.py` (keep the function names and signatures). Provided, do not edit: `load_config`, `plant_2r`,
`plant_course_arm`, `scaled_model`, `mass_matrix`, `coriolis_vector`, `gravity_torque`, `potential_energy`,
`forward_dynamics`, `pd_gains_from_inertia`, `is_hurwitz`, `time_grid`, `rk4_step` and `simulate`. A robot is a dict with the MR
lists, the friction vector `b` and (2R only) the closed-form parameters; `simulate(plant, policy, q0, dq0, T, dt)` runs
`policy(t, q, dq)` at $1/\Delta t$ Hz with the torque held between samples and returns `t`, `q`, `dq`, `tau`. Gains are $n\times n$
matrices (`np.diag(...)`). Angles in rad, torques in N m, energies in J, time in s.

# Procedure and checkpoints

### Checkpoint 1 — Christoffel symbols, skew-symmetry and passivity (≈ 15 min)

1. `mass_matrix_partials` ($\partial M/\partial\theta_k$ by central differences), `coriolis_matrix` (MR Eqs. 8.17, 8.19), `mass_matrix_dot`
   ($\dot M = \sum_k (\partial M/\partial\theta_k)\dot\theta_k$) and `stored_energy` ($E = \tfrac12\dot\theta^{\mathsf T}M\dot\theta + \mathcal P(\theta)$; kinetic energy MR Eq. 8.14).
2. `pytest -k "models or partials or coriolis or skew or mass_matrix_dot or passive"`. The tests compare $C\dot\theta$ with
   `mr.VelQuadraticForces` on the 2R and on the course arm, check $\dot M - 2C$ for skew-symmetry, and check the power balance
   $E(T) - E(0) = \int_0^T(\dot\theta^{\mathsf T}\tau - \dot\theta^{\mathsf T}B\dot\theta)\,dt$ along an open-loop motion.

**Checkpoint question:** for the 2R, $C_\text{alt} = \begin{bmatrix}0 & h(2\dot\theta_1 + \dot\theta_2)\\ -h\dot\theta_1 & 0\end{bmatrix}$ with $h = -m_2L_1\ell_{c2}\sin\theta_2$ gives the same
$c = C_\text{alt}\dot\theta$ as your Christoffel $C$. Evaluate $\dot M - 2C_\text{alt}$ at $\theta = (0.3, 1.1)$, $\dot\theta = (0.7, -0.4)$: is it skew-symmetric? Is
$\dot\theta^{\mathsf T}(\dot M - 2C_\text{alt})\dot\theta$ zero? Which of the two facts does the energy proof of MR Eq. 11.43 actually need?

### Checkpoint 2 — The Lyapunov function of PD + gravity compensation (≈ 15 min)

1. `pd_gravity_setpoint_torque` (MR Eq. 11.38 with $K_i = 0$, $\dot\theta_d = 0$; gravity of the **model** at the measured $\theta$), `lyapunov_energy`
   (MR Eq. 11.41, $M$ of the **plant**), `lyapunov_rate` (first line of MR Eq. 11.42, with $\ddot\theta$ from `forward_dynamics`; do **not** simplify
   it by hand: the test checks that your general expression collapses to $-\dot\theta^{\mathsf T}(K_d + B)\dot\theta$) and `equilibrium_offset` (Newton's
   method for $K_p(\theta_d - \theta) + \tilde g(\theta) - g(\theta) = 0$).
2. `pytest -k "pd_gravity or lyapunov_energy or lyapunov_rate or energy_decreases or kd_zero or wrong_gravity"`.

**Checkpoint question:** in `test_kd_zero_conserves_energy_lasalle_fails` the continuous-time $\dot V$ is exactly 0, yet the simulated $V$
grows by about 1 J in 2 s, and by a quarter of that when $\Delta t$ is divided by 4. Where does the energy come from? (Hint: over one
sample the torque is held while $\theta$ keeps moving; compare the work done by the held spring torque with the change of the spring
energy $\tfrac12\theta_e^{\mathsf T}K_p\theta_e$.) What does this mean for a real joint with very little friction and $K_d = 0$?

### Checkpoint 3 — Linearisation, Lyapunov equation and LQR (≈ 15 min)

1. `linearize` (Tedrake §3.4.1: $A = \begin{bmatrix}0 & I\\ -M^{-1}\partial g/\partial\theta & -M^{-1}B\end{bmatrix}$, $B = \begin{bmatrix}0\\ M^{-1}\end{bmatrix}$, $u_\text{eq} = g(\theta_\text{eq})$), `bryson_weights`,
   `lyapunov_solve` (Kronecker form, no SciPy), `lqr` (`scipy.linalg.solve_continuous_are`, then $K = R^{-1}B^{\mathsf T}P$),
   `pd_state_feedback` ($K_0 = [K_p - \partial\tilde g/\partial\theta,\ K_d]$), `kleinman_lqr`, `lqr_cost`, `state_feedback_torque` and `run_cost`.
2. `pytest -k "linearize or bryson or lyapunov_solve or lqr or kleinman or state_feedback or course_arm"`. The tests compare your $A$, $B$
   with finite differences of the full nonlinear dynamics, your Lyapunov solution with SciPy, and check the Riccati residual,
   the optimality of $J$ against random perturbations of $K$, and Kleinman's iteration against the Riccati solution.

**Checkpoint question:** why do the Coriolis terms not appear in $A$ (Tedrake §3.4.1), and why does `kleinman_lqr` refuse $K_0 = 0$ at
$q_\text{eq}$ but accept the PD gain? Which result of Part A guarantees that $K_0$ is stabilising?

### Checkpoint 4 — Experiments (rest of the session; finish at home)

Use the configuration file (`load_config()`): $\Delta t = 2$ ms, plant with friction unless stated, PD gains from
`pd_gains_from_inertia(plant, q_eq, 1.0, 8.0)`. Save each figure with labelled axes and units.

* **E1 — Energy decreases.** PD + exact $\tilde g$ from $q_0 = q_\text{eq} + (0.3, -0.4)$ rad, at rest, 4 s. Plot $V(t)$ (log scale) and, on a
  second axis, $\dot V$ from `lyapunov_rate` together with $-\dot\theta^{\mathsf T}(K_d + B)\dot\theta$ and a finite difference of $V$. Report $V$ at 0, 0.5, 1, 2 s.
* **E2 — Remove the damping** (MR Exercise 11.12). (a) $K_d = 0$, frictionless plant, 2 s, $\Delta t$ = 2, 1, 0.5 ms: plot $V(t)$ and the
  energy drift $V(2\,\text{s}) - V(0)$ against $\Delta t$. (b) $K_d = 0$ with the plant friction, 10 s: does the arm converge, and what plays
  the role of $K_d$ in the LaSalle argument?
* **E3 — Wrong gravity model** (biomedical: an assistive arm whose forearm-cuff mass was under-estimated by 20 %;
  `scaled_model(plant, 0.8)`). Predict the rest configuration with `equilibrium_offset`, simulate 4 s and compare. Plot $V(t)$ (still
  measured from $\theta_d$): where does it stop decreasing? Construct a function that *is* a Lyapunov function for this case (hint:
  measure the spring from $q_\text{ss}$ and add the potential of $g - \tilde g$).
* **E4 — Skew-symmetry on the course arm.** For 20 random states of the 2R and of the course arm report
  $\max|N + N^{\mathsf T}|/\max|N|$ with $N = \dot M - 2C$, and the checkpoint-1 numbers for $C_\text{alt}$.
* **E5 — LQR vs PD at the hold pose.** Linearise at $q_\text{eq}$ (open-loop eigenvalues?), design LQR with Bryson's rule (configuration
  `lqr`: 0.05 rad, 0.5 rad/s, $(5, 2)$ N m). Compare LQR and the linearised PD law $K_0$: gains, closed-loop poles, linear cost $J$ from
  $x_0 = (0.05, -0.05, 0, 0)$, and on the **nonlinear** arm (3 s, `state_feedback_torque`): run cost, peak feedback torque $|\tau - u_\text{eq}|$, 2 %
  settling time of $\lVert\theta - q_\text{eq}\rVert$. Run PD + $\tilde g$ itself (`pd_gravity_setpoint_torque`) as well: how close is its cost to that of $K_0$?
* **E6 — Choosing $Q$ and $R$.** Repeat E5 for LQR with $R$ multiplied by $\rho$ = 0.1, 1, 10. Table: state cost $J_x = \int x^{\mathsf T}Qx\,dt$, effort cost
  $J_u = \int u^{\mathsf T}Ru\,dt$, peak feedback torques, slowest and fastest closed-loop pole. Which $\rho$ would you choose for a rehabilitation robot
  attached to a patient's arm, and why?
* **E7 — Course arm, 12 states.** Linearise the course arm at $q_\text{home}$, Bryson weights with 0.05 rad, 0.5 rad/s and 20 % of the rated
  torques. Report the open-loop unstable eigenvalue, the Kleinman iteration count from the PD gains, the slowest and the fastest closed-loop
  pole of LQR and of PD, and the costs from $x_0$ = 0.05 rad on joint 2 and $-0.05$ rad on joint 3. Multiply the fastest LQR pole by $\Delta t$:
  could a 500 Hz controller realise this design? Which weights would you change?

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab | 1: $K_p$; $K_d$ | diag(51.39, 3.423) N m/rad; diag(12.80, 0.836) N m s/rad |
| | 4: $A$; unstable for | $\begin{bmatrix}0 & 1\\ mgr\sin\theta^*/M & -b/M\end{bmatrix}$, $B = (0, 1/M)$; $0 < \theta^* < \pi$ (link above the horizontal) |
| | 5: $Q$; $R$ | diag(400, 400, 4, 4); diag(0.04, 0.25) |
| E1 | $V$ at 0, 0.5, 1, 2 s | 2.5865, 0.0459, 5.45 × 10⁻⁴, 6.0 × 10⁻⁸ J; never increases; $\lVert\theta_e\rVert < 10^{-3}$ rad after 1.50 s |
| | $\dot V$ at $t = 0.2$ s: `lyapunov_rate`; $-\dot\theta^{\mathsf T}(K_d + B)\dot\theta$; finite difference | −6.0017; −6.0017; −5.993 W |
| E2 | (a) drift $V(2\,\text{s}) - V(0)$ at $\Delta t$ = 2, 1, 0.5 ms | 0.998, 0.453, 0.216 J (≈ proportional to $\Delta t$) |
| | (b) $K_d = 0$ with friction: $V(10\,\text{s})$, $\lVert\theta_e(10\,\text{s})\rVert$ | 0.235 J, 0.092 rad (slow convergence: friction is the only damping) |
| E3 | $q_\text{ss} - q_\text{eq}$ (Newton); simulated final $q - q_\text{ss}$ | (−0.03976, −0.01427) rad; < 10⁻⁷ rad |
| | $V$ (from $\theta_d$): minimum, then final | 0.0255 J at 0.58 s, then rises to 0.0410 J |
| E4 | $\max|N + N^{\mathsf T}|/\max|N|$: 2R; course arm | 2 × 10⁻¹⁶; 8 × 10⁻¹⁶ (round-off) |
| | $\dot M - 2C_\text{alt}$ at the checkpoint-1 state; $\dot\theta^{\mathsf T}(\dot M - 2C_\text{alt})\dot\theta$ | $\begin{bmatrix}0.0856 & 0.2567\\ -0.1069 & 0\end{bmatrix}$ (not skew); 0 (to 10⁻¹²) |
| E5 | open-loop eigenvalues at $q_\text{eq}$ | −7.199, −3.258, **+6.571**, **+3.180** s⁻¹ (unstable) |
| | LQR $K$ | $\begin{bmatrix}106.93 & -18.30 & 16.26 & -1.006\\ 10.25 & 41.19 & 1.878 & 4.395\end{bmatrix}$ |
| | PD as $K_0$ | $\begin{bmatrix}60.00 & 1.957 & 12.80 & 0\\ 1.957 & 5.380 & 0 & 0.836\end{bmatrix}$ |
| | closed-loop poles: LQR; PD | −132.7, −10.01, −9.76 ± 4.36j; −37.41, −4.48, −4.94 ± 3.89j |
| | $J$ (linear): LQR; PD | 0.2545; 0.3102 |
| | nonlinear run cost, peak $|\tau - u_\text{eq}|$, 2 % settling: LQR; $K_0$; PD + $\tilde g$ | 0.2524, (6.26, 1.55) N m, 0.43 s; 0.3086, (2.90, 0.17) N m, 1.11 s; 0.3083 |
| | Kleinman iterations from $K_0$ | 9 |
| E6 | $\rho$ = 0.1: $J_x$, $J_u$, peak, slowest / fastest pole | 0.2096, 0.0852, (19.0, 4.9) N m, 10.0 / 420 s⁻¹ |
| | $\rho$ = 1 | 0.2290, 0.0234, (6.26, 1.55) N m, 9.76 / 133 s⁻¹ |
| | $\rho$ = 10 | 0.2796, 0.0073, (2.27, 0.47) N m, 5.30 / 41.9 s⁻¹ |
| E7 | open-loop unstable eigenvalue; Kleinman iterations | +5.977 s⁻¹ (one); 17 |
| | slowest / fastest closed-loop pole: LQR; PD | 10.0 / 20 020 s⁻¹; 3.31 / 67.7 s⁻¹ |
| | $J$: LQR; PD | 0.2045; 0.3103 |
| | fastest LQR pole × $\Delta t$ | ≈ 40 (≫ 1: not realisable at 500 Hz; the wrist joints have tiny inertia but the same weights as the shoulder) |

All values were recomputed by the course team with the reference solution. Small differences
($< 10^{-6}$ in the gains, $< 10^{-4}$ in the run costs) can come from a different but equivalent order of operations.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 7 into your cumulative library as `stability.py` (next to Lab 6's `control.py`). The functions you write (17):

`mass_matrix_partials(robot: dict, q, h: float=1e-6)`, `coriolis_matrix(robot: dict, q, dq, h: float=1e-6)`, `mass_matrix_dot(robot: dict, q, dq, h: float=1e-6)`, `stored_energy(robot: dict, q, dq)`, `pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, model: dict)`, `lyapunov_energy(plant: dict, q, dq, qd, Kp)`, `lyapunov_rate(plant: dict, q, dq, qd, Kp, tau)`, `equilibrium_offset(plant: dict, model: dict, qd, Kp, iters: int=50, tol: float=1e-12, h: float=1e-6)`, `linearize(plant: dict, q_eq, h: float=1e-6)`, `bryson_weights(x_max, u_max)`, `lyapunov_solve(A, Q)`, `lqr(A, B, Q, R)`, `pd_state_feedback(model: dict, q_eq, Kp, Kd, h: float=1e-6)`, `kleinman_lqr(A, B, Q, R, K0, iters: int=50, tol: float=1e-10)`, `lqr_cost(A, B, Q, R, K, x0)`, `state_feedback_torque(q, dq, q_eq, u_eq, K)`, `run_cost(sim: dict, q_eq, u_eq, Q, R)`.

Names shared with other labs:

* `pd_gravity_setpoint_torque(q, dq, qd, Kp, Kd, model)` is the setpoint special case ($K_i = 0$, $\dot\theta_d = 0$, gain matrices) of Lab 6's `pd_gravity_torque(q, dq, eint, qd, dqd, Kp, Ki, Kd, model)`; the two have different names on purpose and both stay in the library.
* The provided `rk4_step(f, x, h)` and `time_grid(T, dt)` are the Lab 5 and Lab 6 helpers (same signatures); keep one copy. The provided `simulate(plant, policy, q0, dq0, T, dt)` is not Lab 6's `simulate_arm`: it takes a function `policy(t, q, dq)` instead of a controller object.

The provided helpers at the top of `starter.py` (loaders, models, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab7_<ID>.py`, passing `LAB_IMPL=lab7_<ID> pytest -q test_lab.py` (23 tests). Add the functions to
   `sbeg461_dynamics/stability.py`.
2. `experiments.py` (or a notebook) that regenerates every figure and number of the lab note.
3. **Lab note** (PDF, ≤ 3 pages): figures for E1–E7, the expected-value table with your numbers, the three checkpoint
   questions answered, the E3 Lyapunov function you constructed, reflection and AI-use note.

# Reflection prompts (≤ 150 words, in the lab note)

* The stability proof of PD + gravity compensation used three facts: exact gravity, $K_d > 0$ (or friction), and continuous
  time. Which experiment broke each of them, and what did you observe? Which of the three is most likely to be violated on a
  low-cost rehabilitation robot, and on an EOD arm carrying an unknown package?
* LQR had a lower cost than PD in E5 and E7, yet it used about twice the peak torque in E5 and needed an impossible bandwidth
  in E7. In your own words: what does "optimal" mean here, and who chooses what is optimal?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct (for example a Coriolis
  matrix that reproduces $c$ but is not built from the Christoffel symbols, `scipy.linalg.solve_continuous_lyapunov(A, Q)`
  with the wrong transpose or sign, or $u = -Kx$ applied without the equilibrium torque $u_\text{eq}$), and how did you find out?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `test_coriolis_matrix...` passes for the 2R but fails for the course arm | An index of the Christoffel symbol is swapped; with $n = 2$ several mistakes cancel. Write $\Gamma_{ijk} = \tfrac12(\partial m_{ij}/\partial\theta_k + \partial m_{ik}/\partial\theta_j - \partial m_{jk}/\partial\theta_i)$ with explicit loops first. |
| $C\dot\theta$ correct but `test_mdot_minus_2c...` fails | You built a $C$ from $c$ directly (as $C_\text{alt}$), not from the Christoffel symbols. |
| `lyapunov_rate` off by $\dot\theta^{\mathsf T}B\dot\theta$ | You used $M\ddot\theta = \tau - c - g$ by hand; use `forward_dynamics`, which includes the plant friction. |
| `test_energy_decreases...` shows $V$ increasing | $V$ must use the **plant** $M$ and the torque of the **same** sample (`sim["tau"][k]`); the controller uses the **model** $\tilde g$. |
| `equilibrium_offset` does not converge | Start Newton at $\theta_d$; the Jacobian is $-K_p + \partial(\tilde g - g)/\partial\theta$ (your finite differences must perturb one joint at a time). |
| `linearize` has the wrong sign in the lower-left block | $M\ddot\theta = \tau - g(\theta) - \dots$, so the block is $-M^{-1}\partial g/\partial\theta$; at $q_\text{eq}$ two eigenvalues must be positive. |
| `lyapunov_solve` disagrees with SciPy by a transpose | Use column stacking (`order="F"`) both when flattening $Q$ and when reshaping $P$; SciPy's `solve_continuous_lyapunov(a, q)` solves $aX + Xa^{\mathsf H} = q$, so the test calls it with $(A^{\mathsf T}, -Q)$. |
| `kleinman_lqr` raises `ValueError` for the PD gain | `pd_state_feedback` must subtract $\partial\tilde g/\partial\theta$ from $K_p$ (the gravity-compensation term linearised); without it $K_0$ is the PD law *without* gravity compensation. |
| Nonlinear LQR run drifts away from $q_\text{eq}$ | The torque is $u_\text{eq} - Kx$; without $u_\text{eq}$ gravity is not held. |

# Pre-lab answers (released after the session)

1. $K_p = \text{diag}(0.80299\cdot64, 0.05349\cdot64) = \text{diag}(51.39, 3.423)$ N m/rad; $K_d = \text{diag}(2\cdot8\cdot0.80299 - 0.05, 2\cdot8\cdot0.05349 - 0.02) = \text{diag}(12.80, 0.836)$
   N m s/rad.
2. $\dot V \le 0$ only says that $V$ does not increase: $V$ could stop decreasing at a state with $\dot\theta = 0$, $\theta \ne \theta_d$. LaSalle: the arm can stay
   only in the largest invariant set where $\dot V = 0$; with $\dot\theta \equiv 0$ the closed loop gives $K_p\theta_e = 0$ (MR Eq. 11.39), so $\theta = \theta_d$ is the only
   such state and the equilibrium is globally asymptotically stable.
3. $x^{\mathsf T}Nx$ is a scalar, so $x^{\mathsf T}Nx = (x^{\mathsf T}Nx)^{\mathsf T} = x^{\mathsf T}N^{\mathsf T}x = -x^{\mathsf T}Nx$, hence 0. The term $\tfrac12\dot\theta^{\mathsf T}(\dot M - 2C)\dot\theta$ of MR Eq. 11.42 vanishes.
4. $A = \begin{bmatrix}0 & 1\\ mgr\sin\theta^*/M & -b/M\end{bmatrix}$, $B = \begin{bmatrix}0\\ 1/M\end{bmatrix}$ (because $\partial(mgr\cos\theta)/\partial\theta = -mgr\sin\theta$). The characteristic polynomial
   $s^2 + (b/M)s - mgr\sin\theta^*/M$ has a positive root when $\sin\theta^* > 0$, i.e. $0 < \theta^* < \pi$: the link held above the horizontal falls over.
5. $Q = \text{diag}(1/0.05^2, 1/0.05^2, 1/0.5^2, 1/0.5^2) = \text{diag}(400, 400, 4, 4)$; $R = \text{diag}(1/25, 1/4) = \text{diag}(0.04, 0.25)$.

# C++ track

You may do this lab in C++17 instead of Python. The functions, configuration file, checkpoints, experiments, expected values and lab note are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `mass_matrix`, `vel_quadratic_forces`, `gravity_forces`), with finite differences, with the identities of MR §8.1.2 and §11.4.2.2 and of LQR theory, and with the numbers of the Python reference (including SciPy's Lyapunov and Riccati solutions), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided helpers, translated from `starter.py`, and the Riccati solver `care_solve` — do not edit), `course_arm.hpp` / `course_arm.cpp` (the course arm, numbers identical to `course_arm.py` — do not edit), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random inputs drawn by `test_lab.py`), `CMakeLists.txt`. The configuration is the same `data/lab07_config.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). SciPy is not needed for the C++ track.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 7 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*lyapunov*"             # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 22 test cases fail with `TODO: <function>`: this is expected (the whole test program runs in well under a second once your code is written). Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 17 as in Python): `mass_matrix_partials`, `coriolis_matrix`, `mass_matrix_dot`, `stored_energy` (Checkpoint 1); `pd_gravity_setpoint_torque`, `lyapunov_energy`, `lyapunov_rate`, `equilibrium_offset` (Checkpoint 2); `linearize`, `bryson_weights`, `lyapunov_solve`, `lqr`, `pd_state_feedback`, `kleinman_lqr`, `lqr_cost`, `state_feedback_torque`, `run_cost` (Checkpoint 3). Given and complete (`helpers.cpp`): `load_config`, `plant_2r`, `plant_course_arm`, `scaled_model`, `mass_matrix`, `coriolis_vector`, `gravity_torque`, `potential_energy`, `forward_dynamics`, `pd_gains_from_inertia`, `is_hurwitz`, `time_grid`, `rk4_step`, `simulate` and `care_solve`. In `lqr`, call `care_solve(A, B, Q, R)` where Python calls `scipy.linalg.solve_continuous_are`; `lyapunov_solve` must be your own Kronecker construction in both languages.

**Checkpoints** (22 C++ test cases with the Python test names; the Python track counts 23 because it also has `test_starter_compiles`, which in C++ is the build itself). 1: `./build/lab_tests -tc="*models*,*partials*,*coriolis*,*skew*,*mass_matrix_dot*,*passive*"` (6 test cases); 2: `-tc="*pd_gravity*,*lyapunov_energy*,*lyapunov_rate*,*energy_decreases*,*kd_zero*,*wrong_gravity*"` (6); 3: `-tc="*linearize*,*bryson*,*lyapunov_solve*,*lqr*,*kleinman*,*state_feedback*,*course_arm*"` (10); at the end `./build/lab_tests` must show 22 passed.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| robot dict `robot["b"]`, `robot["n"]`, `robot["q_home"]` | `lab::Robot` (the course-arm fields `n`, `Slist`, `Mlist`, `Glist`, `g`, `tau_max`, `q_home` plus `key`, `b`, `has_closed_form`, `cf`): `robot.b`, `robot.n` |
| `CFG = load_config()`, `CFG["planar_2r"]["q_eq"]` | `sbeg::json CFG = lab::load_config();` `sbeg::as_vec(CFG["planar_2r"]["q_eq"])`, `CFG["dt"].get<double>()` |
| `plant_2r(cfg)`, `plant_2r(cfg, False)` | `lab::plant_2r(cfg)`, `lab::plant_2r(cfg, false)` (the configuration argument is required) |
| `mass_matrix_partials` returns an $(n, n, n)$ array | `std::vector<Eigen::MatrixXd>`, element `k` = $\partial M/\partial\theta_k$ |
| `pd_gains_from_inertia` returns `(Kp, Kd)`; `linearize` `(A, B, u_eq)`; `kleinman_lqr` `(K, P, iterations)` | structs `lab::PDGains{Kp, Kd}`, `lab::Linearization{A, B, u_eq}`, `lab::KleinmanResult{K, P, iterations}` |
| `bryson_weights` returns `(Q, R)`; `lqr` `(K, P)` | `std::pair`: `auto [Q, R] = lab::bryson_weights(x_max, u_max);` |
| `simulate(plant, policy, ...)` with a Python function; result dict | a lambda `[&](double t, const Eigen::VectorXd& q, const Eigen::VectorXd& dq) { return ...; }`; `lab::SimResult{t, q, dq, tau}` (row $k$ = sample $k$) |
| `scipy.linalg.solve_continuous_are` | `lab::care_solve` (given; matrix-sign-function method on the Hamiltonian matrix, agrees with SciPy to about $10^{-12}$ relative on this lab's systems) |
| `np.kron`, `reshape(order="F")` | write the $n^2\times n^2$ matrix with loops (`L(j * n + i, ...)`), copy columns with `P.col(j) = v.segment(j * n, n)` |
| `np.linalg.eigvals(A)`, `np.trapezoid` | `Eigen::EigenSolver<Eigen::MatrixXd>(A, false).eigenvalues()`; a loop over samples |
| random inputs of `np.random.default_rng(seed)` | stored in `cpp_cases.json` (same numbers as the Python tests) |
| `experiments.py`, matplotlib figures | write `cpp/experiments.cpp` and save each run with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool (see below) |

**Experiments and figures.** E1–E7 and the expected values are the same. To build an experiments program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

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
| `cannot open file: .../cpp_cases.json` or `.../lab07_config.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size or index out of range" | e.g. `K0 << Kp - G, Kd;` with `K0` not sized $n\times 2n$ first, or a row of `sim.q` used as a column | `Eigen::MatrixXd K0(n, 2 * n);`, `sim.q.row(k).transpose()`; the test program does not crash, the test simply fails |
| `lyapunov_solve` wrong by a transpose | row/column index of the stacked vector mixed up | the entry $P(i, j)$ is element `j * n + i` of `vec(P)` (column stacking, as `order="F"`) |
| Results wrong only in some entries, no compile error | an Eigen product stored in `auto` (`auto C = A * B;` keeps references to temporaries) | write the type: `Eigen::MatrixXd C = A * B;` |
| A lambda does not compile ("cannot capture structured binding") | C++17 cannot capture names from `auto [Q, R] = ...` in a lambda | copy them first: `const Eigen::MatrixXd Qm = Q;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
