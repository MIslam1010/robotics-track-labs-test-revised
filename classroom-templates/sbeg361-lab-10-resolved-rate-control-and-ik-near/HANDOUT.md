---
title: "Lab 10 — Resolved-rate control and IK near singularities (Formal report 3)"
subtitle: "SBEG 361 Introduction to Robotics — Week 12"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 40 min in class (after Quiz 10) + about 3 h of independent work for the report; **optional hardware rotation** (Week 12, Course syllabus §5 Labs and formal reports): a group that gets a slot may spend 15–20 min of the session or a booked open-lab slot on the course arm (procedure in Checkpoint 6b). The whole lab and formal report 3 can be completed in simulation with the example hardware log (Course syllabus §5 Labs and formal reports); the hardware run is not needed for full marks. It is separate from, and does not replace, the required project hardware sessions H1 and H2 |
| CLOs | **361-6** (numerical IK, damped least squares, redundancy resolution; convergence); **361-7** (implement and validate against a reference library, a simulator and a hardware log: the example log, or your own if you do the optional run); **361-9** (technical report, honest disclosure of tool use) |
| Reading | MR §6.2.2 (printed pp. 225–229: pseudo-inverse, Eq. 6.6, Newton–Raphson in the body frame), **§6.3 Inverse velocity kinematics (pp. 230–232, Eq. 6.7, weighted pseudo-inverse)**, §6.4 (p. 232, closed loops), §7.1.2 and §7.2.1 (pp. 247–252, Stewart–Gough platform), §11.3.3 (p. 421, Eq. 11.16, task-space feedforward + feedback). Supplementary: Corke §8.2 (pp. 180–186: resolved-rate motion control, over-actuated robots, null-space motion; Exercise 4f, p. 189, damped least squares); MLS Ch. 3 §5.1 (pp. 129–132, redundant manipulators) and §5.4 (pp. 138–143, Stewart platform). MLS writes twists linear-first — swap the blocks before comparing with MR. |
| Files | `starter.py` (your code, 15 functions), `test_lab.py` (30 auto-checks), `data/robots.json`, `data/hexapod.json`, `data/hw_log_example.csv` (robot data copied from the shared course models; the hexapod is problem data) |
| Assessment | Lab 10 counts as one of the three formal reports of the lab category (Course syllabus §4 Assessment): **code and tests 30 % (auto-tests), report 70 %** (Formal report 3: IEEE two-column, ≤ 6 pages, rubric below), due one week after the lab (before the Week 13 lab practical) |
| AI use | Allowed for code **with disclosure** (AI-use note in the report appendix); you must be able to explain every line in an oral check |
| Timing | Week 12 schedule: Session 3 is the project design review; the workshop sheet is a guided take-home problem set. **Quiz 10 runs in the first 10 minutes of this session**, so the lab has 40 minutes in class: finish Checkpoints 4–5 at home |
| Tier | 0 (simulation and the example hardware log, every group; sufficient for full marks) + optional 1 (course arm, supervised) |

The functions you write today become `ik_velocity.py` and `parallel.py` of your `sbeg361_kinematics` library, next to `jacobians.py` (Lab 7) and your IK code (Lab 9). They are also the resolved-rate code your project team uses for the hardware session H2 and the project requirement "a path executed by resolved-rate control with damping near singularities" (project brief §5, item 5).

## Why this lab matters

Lab 9 solved IK for **one** pose. A teleoperated or path-following robot needs joint *velocities* every few milliseconds: $\dot\theta = J^\dagger(\theta)\mathcal V_d$ (MR Eq. 6.7). This works beautifully — until the robot comes near a singularity, where $J^\dagger$ asks for enormous joint rates, or until the robot has more joints than the task needs, where infinitely many $\dot\theta$ do the job and the "free" motion may drift. Three cases drive the lab:

* **Defence — teleoperated approach of an EOD manipulator through a wrist singularity.** The 6R arm of a tracked EOD robot (`eod_6r_ugv`, project scenario S4, Weeks 9–11) retracts its gripper 0.10 m at 0.05 m/s with the wrist almost straight. With the pseudo-inverse the forearm and gripper rolls spin at about $940^\circ$/s — 30 times the rated speed. You make the motion safe with damped least squares (DLS) and adaptive damping.
* **Biomedical — redundancy in assistive and rehabilitation robots.** The wheelchair-mounted 7R feeding arm (`srs_7r_feeding`, scenario S2) swings its elbow away from the user's face while the spoon stays still (null-space self-motion), and the planar 3R rehabilitation robot (`planar_3r`) repeats a therapy circle: with the plain pseudo-inverse its joints do not return to the start (MR §6.4); a null-space joint-centring term fixes it.
* **Biomedical — hexapod patient positioning in radiotherapy.** Leg lengths, the inverse Jacobian (MR §7.2.1) and Newton–Raphson forward kinematics of the couch hexapod of Workshop 12 P5.

Finally you **dry-run** a short resolved-rate line on the course arm (`course_arm`), find that one candidate line is unsafe, and evaluate the safe one with the hardware acceptance tests on a recorded log (the example log; groups with a hardware slot may also run it on the arm — optional).

All robot numbers are those of the shared course robot models (copied into `data/robots.json`); the hexapod is problem data. Geometry is illustrative unless it is a textbook robot.

## Learning objectives

By the end of the lab you can:

1. implement the pseudo-inverse from the SVD, the DLS inverse and an adaptive damping rule, and explain each through the singular values;
2. resolve redundancy with the null-space projector and a weighted pseudo-inverse, and show that joint paths of a redundant robot need not close (MR §6.4);
3. simulate resolved-rate control with feedforward and feedback (MR Eqs. 6.7 and 11.16), rate limiting and damping, and quantify the trade-off between joint rates and tracking error near a singularity;
4. explain how Newton–Raphson IK behaves near singularities and for unreachable targets, with and without damping;
5. compute leg lengths, the inverse Jacobian and the forward kinematics of a Stewart–Gough platform;
6. dry-run a resolved-rate motion on the course arm model, evaluate a recorded run with the hardware acceptance tests (and, optionally, execute it on the arm), and report it in an IEEE-style technical report.

# Pre-lab check (complete before the session, 5 questions)

1. $J$ is $6\times7$ and has rank 6. How many solutions does $J\dot\theta = \mathcal V$ have? Which one does $J^\dagger\mathcal V$ return? What is the dimension of the null space?
2. Write $J^\dagger$ in terms of the SVD $J = U\Sigma V^{\mathsf T}$. What happens to $J^\dagger$ as $\sigma_{\min}\to0$?
3. For $\lambda = 0.02$, compute the DLS gain $\sigma/(\sigma^2+\lambda^2)$ for $\sigma = 0.2$ and $\sigma = 0.002$, and compare with $1/\sigma$.
4. In MR Eq. 11.16, why is the feedforward twist written $[\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d$ rather than just $\mathcal V_d$?
5. A Stewart–Gough leg has base joint $a_i$ and unit direction $\hat n_i$. Show that $\dot s_i = \hat n_i\cdot(v + \omega\times a_i)$ for the space twist $(\omega, v)$, and rewrite it in the form of MR p. 252.

# Setup

```bash
cd "<your Lab code folder>"                                    # or your Classroom 50 repo
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python     # or the course Docker image / JupyterHub
$PY -c "import numpy, modern_robotics; print('ok')"
LAB_IMPL=starter $PY -m pytest -q test_lab.py      # 30 failures (NotImplementedError) is expected at the start
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Work in `starter.py` and keep the function names and signatures. **Do not call `modern_robotics` or `np.linalg.pinv` inside `starter.py`** — the tests compare against them. The helpers at the top of `starter.py` (`vec_to_so3`, `adjoint`, `trans_inv`, `matrix_exp6`, `matrix_log6`, `fk_space`, `body_screws`, `jacobian_space`, `jacobian_body`, the loaders and the log reader/writer) are complete; you may replace them with your own Lab 2–7 versions. (`matrix_exp6` keeps the first-order rotation for angles below $10^{-6}$ rad, unlike `mr.MatrixExp6`; Newton iterations on tiny corrections need it — see Week 9 lecture notes, WE-B7.)

`load_robot(name)` returns `(Slist, M, info)` for `course_arm`, `eod_6r_ugv`, `planar_2r`, `planar_3r`, `srs_7r_feeding`; `info` holds `limits`, `qdot_max` and `q_home`. Run one checkpoint's tests with `-k`, e.g. `LAB_IMPL=starter $PY -m pytest -q test_lab.py -k "pinv or dls"`.

# Procedure and checkpoints

### Checkpoint 1 — Pseudo-inverse and damped least squares (≈ 10 min)

1. `pinv_svd(J, tol)`: $J^\dagger = V\Sigma^\dagger U^{\mathsf T}$, inverting only singular values above `tol`$\cdot\sigma_{\max}$. It must work for fat, square, tall and rank-deficient matrices (MR p. 227).
2. `dls_inverse(J, lam)`: $J^{\mathsf T}(JJ^{\mathsf T}+\lambda^2I)^{-1}$ ($\lambda = 0$ → `pinv_svd`).
3. `dls_singular_gains(J, lam)` and `adaptive_lambda(sigma_min, eps, lam_max)`: $\lambda = 0$ when $\sigma_{\min} \ge \varepsilon$, else $\lambda = \lambda_{\max}\sqrt{1 - (\sigma_{\min}/\varepsilon)^2}$ — damping only near a singularity, growing smoothly to $\lambda_{\max}$ at $\sigma_{\min} = 0$.

**Checkpoint values.** $J = \mathrm{diag}(0.5, 0.01)$, $\lambda = 0.05$: gains $(1.980, 3.846)$ instead of $(2, 100)$ (Quiz 10 Q2); the largest possible gain is $1/(2\lambda) = 10$. `adaptive_lambda(0.03, 0.05, 0.04)` $= 0.032$. EOD arm of Workshop P2 at $(20^\circ, 10^\circ)$: DLS rates $(11.70, -25.82)^\circ$/s for the 0.05 m/s radial command (exact inverse: $(27.37, -60.26)^\circ$/s).
Tests: `test_pinv_*`, `test_dls_*`, `test_adaptive_lambda`.

### Checkpoint 2 — Redundancy and the null space (≈ 10 min)

1. `null_projector(J)` $= I - J^\dagger J$; `weighted_pinv(J, W)` $= W^{-1}J^{\mathsf T}(JW^{-1}J^{\mathsf T})^{-1}$ (MR p. 232); `redundant_rates(J, xdot, qdot0)` $= J^\dagger\dot x + (I - J^\dagger J)\dot\theta_0$; `joint_centering_gradient(theta, limits)` $= \nabla h$ for $h = \tfrac12\sum_i\big((\theta_i - c_i)/r_i\big)^2$ (mid-range $c_i$, range $r_i$).
2. **Feeding arm self-motion.** `srs_7r_feeding` at $\theta = (10, 30, 0, -70, 0, 40, 0)^\circ$: integrate $\dot\theta = P(\theta)\,(0, 0, 0.5, 0, 0, 0, 0)$ rad/s with $J = J_b$ for 1 s ($\Delta t = 0.01$ s, `xdot = 0`). Track the elbow point (the joint-4 axis point, moved by joints 1–3) and the spoon pose.
3. **Therapy circle on the planar 3R** (`planar_3r`, start `q_home` $= (0.4, 0.8, 0.6)$ rad): the handle tracks a circle of radius 0.15 m, period 4 s, 30 loops, starting at the current handle point (centre 0.15 m in $-\hat x_s$ from it), with the $2\times3$ position Jacobian, $\dot x = \dot x_d + 2(x_d - x)$, $\Delta t = 0.01$ s. Therapy-safe limits (illustrative): $\theta_1\in[-90^\circ, 90^\circ]$, $\theta_2\in[0, 2.6]$ rad, $\theta_3\in[-2, 2]$ rad. Run with $\dot\theta_0 = 0$ and with $\dot\theta_0 = -5\nabla h$. Record the joints at the end of each loop.

**Checkpoint values.** Workshop P1: $P = \tfrac16\begin{bmatrix}1&-1&-2\\-1&1&2\\-2&2&4\end{bmatrix}$. Feeding arm: $P e_3 = (0.176, 0, 0.329, 0, -0.372, 0, 0.227)$ — joints 2 and 4 do not take part in this self-motion; after 1 s the elbow has moved **30.6 mm** sideways and the shoulder roll is $9.44^\circ$, while the spoon has moved only 0.056 mm (Euler error; no task feedback). Therapy circle, pseudo-inverse only: after loop 1 the joints differ from the start by $(-0.01, -0.15, 0.30)^\circ$, after 30 loops by $(1.20, -3.84, 5.56)^\circ$, still drifting $0.12^\circ$ per loop in $\theta_3$ — the handle path is closed, the joint path is not. With joint centring ($k_0 = 5$) the loop-to-loop change is zero (to $10^{-3}$ deg) after 10 loops and $h$ falls from 0.0378 to 0.0080.
Tests: `test_null_projector_*`, `test_weighted_pinv`, `test_redundant_rates`, `test_joint_centering_gradient`, `test_srs7r_self_motion`.

### Checkpoint 3 — Resolved-rate control (≈ 15 min)

1. `line_trajectory(T0, displacement, speed, dt)`: straight line, constant speed and orientation; returns times, desired poses and desired **body** twists $\mathcal V_d = (0,\ R_0^{\mathsf T}\hat u\,\text{speed})$ (zero at the last sample).
2. `resolved_rate_sim(...)`: at every step $\mathcal V_b = [\mathrm{Ad}_{X^{-1}X_d}]\mathcal V_d + k_p\mathcal X_e$, $[\mathcal X_e] = \log(X^{-1}X_d)$ (MR Eq. 11.16, $K_i = 0$); $\dot\theta = J_b^\#(\lambda)\mathcal V_b$ (MR Eq. 6.7, damped); optional adaptive $\lambda$ and uniform rate scaling to `qdot_max`; explicit Euler.
3. **Course arm.** From `q_home` $= (0, -0.6, 1.4, -2.4, -1.57, 0)$ (tool point $(0.2341, 0.0634, 0.0811)$, tool pointing down), move 0.06 m along $+\hat y_s$ at 0.02 m/s, $\Delta t = 0.02$ s, with $k_p = 0$ and $k_p = 2$ s$^{-1}$; repeat for $\Delta t = 0.1, 0.05, 0.01$ s.
4. **EOD arm.** From $\theta_0 = (0, -30^\circ, 60^\circ, 3^\circ, 5^\circ, 0)$ retract 0.10 m along $-\hat x_s$ at 0.05 m/s ($\Delta t = 0.02$ s, $k_p = 2$): (i) pseudo-inverse; (ii) pseudo-inverse with the rate limit $30^\circ$/s; (iii) DLS, $\lambda = 0.02$; (iv) adaptive, $\varepsilon = 0.04$, $\lambda_{\max} = 0.04$. Plot $\theta_4$, $\theta_6$, the joint rates, $\sigma_{\min}(J_b)$ and the position error against time.

**Checkpoint values.** Course arm: 151 samples (3 s); peak joint rates $(4.92, 4.97, 11.17, 6.23, 0.14, 4.92)^\circ$/s (7.0 % of the rated $160^\circ$/s); $\sigma_{\min}(J_b) \ge 0.0389$. Final position error (mm):

| $\Delta t$ (s) | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|
| $k_p = 0$ (open loop, Eq. 6.7 only) | 0.375 | 0.189 | 0.076 | 0.038 |
| $k_p = 2$ (Eq. 11.16) | 0.076 | 0.038 | 0.015 | 0.008 |

The open-loop error is proportional to $\Delta t$ (Euler drift, the "small velocity errors accumulate" of MR p. 231); feedback divides it by five.

EOD arm (101 samples, 2 s; $\sigma_{\min} = 0.048$ at the start, $0.0013$ at the closest point):

| Case | Peak wrist roll rate ($^\circ$/s) | Max position error (mm) | Remarks |
|---|---|---|---|
| (i) pseudo-inverse | **944** (at $t = 1.40$ s) | 0.41 | $\theta_4$, $\theta_6$ flip by about $170^\circ$ (end at $173.7^\circ$, $-170.7^\circ$) |
| (ii) pseudo-inverse + $30^\circ$/s limit | 30 | **34.3** (final) | the whole motion is slowed; the gripper falls behind |
| (iii) DLS $\lambda = 0.02$ | 14.6 | 2.72 (max angle error $0.28^\circ$) | passes through the singularity: $\theta_5$ ends at $-1.85^\circ$, no flip |
| (iv) adaptive, $\varepsilon = \lambda_{\max} = 0.04$ | 6.5 | 2.75 | damping on from $t = 0.22$ s; exact tracking far from the singularity |

Tests: `test_line_trajectory`, `test_resolved_rate_*`.

### Checkpoint 4 — IK near singularities (≈ 10 min, may be finished at home)

1. `ik_newton(Slist, M, T_d, theta0, lam, ...)`: the body-frame Newton–Raphson of MR §6.2.2 with the step $J_b^\#(\lambda)\mathcal V_b$; return the error history.
2. Course arm, $\theta_0 = (0, 0.3, -0.6, 0.3, 0, 0)$: (a) $T_d = M$ shifted 0.05 m along $\hat x_s$ (reachable, near the stretched elbow); (b) shifted 0.08 m (unreachable). Try $\lambda = 0, 0.01, 0.02, 0.05$.

**Checkpoint values.** (a) Iterations to $10^{-6}$: 4, 6, 10, 37 for $\lambda = 0, 0.01, 0.02, 0.05$; the $\lambda = 0$ answer equals `mr.IKinBody`. Damping costs speed where the target is reachable — near the solution, DLS converges only linearly. (b) The pseudo-inverse iterates jump about (the linear error reaches 0.460 m) and never settle; DLS with $\lambda = 0.02$ settles at the closest reachable pose with a **14.44 mm** residual and $\theta_3 \to 0$ (elbow straight). An IK that returns "failed" with a sensible nearest pose is what a project acceptance test needs.
Tests: `test_ik_newton_*`.

### Checkpoint 5 — Radiotherapy couch hexapod (≈ 5 min in class, finish at home)

1. `hexapod_leg_lengths(p, R, A, B)` (MR §7.1.2), `hexapod_inverse_jacobian(p, R, A, B)` (MR §7.2.1, row $i$ = $((-\hat n_i\times a_i)^{\mathsf T}, \hat n_i^{\mathsf T})$), `hexapod_forward_kinematics(s, A, B, p0, R0)` (Newton–Raphson on the leg lengths; space twist, so $T \leftarrow e^{[\mathcal V]}T$).

**Checkpoint values.** Home: all $s_i = 0.29762$ m; row 1 of $J_s^{-1}$ = $(-0.04376, -0.24817, -0.10080, -0.47790, -0.25692, 0.84000)$; condition number 8.33. Test pose $p = (0.004, -0.003, 0.252)$, exponential coordinates $(1.0, -0.5, 1.5)^\circ$: $s = (0.29489, 0.30231, 0.30253, 0.30329, 0.29284, 0.30048)$ m; forward kinematics from home converges in 3 iterations to $10^{-12}$ m. The inverse Jacobian agrees with central differences of $s$ along space twists to $10^{-8}$.
Tests: `test_hexapod_*`.

### Checkpoint 6 — From simulation to the course arm (dry run, optional hardware session, evaluation)

**6a — Dry run (everyone, Tier 0).** Two candidate approach lines for a scaled feeding/presentation task start at `q_home`: 0.06 m along $+\hat x_s$ and 0.06 m along $+\hat y_s$, both at 0.02 m/s. Simulate both with $k_p = 0$ and inspect $\sigma_{\min}$, the peak joint rates and the final error.

*Checkpoint values.* The $+\hat x_s$ line runs into the reach boundary (elbow stretch singularity): $\sigma_{\min} < 10^{-3}$, joint rates above $1000^\circ$/s and a 22.5 mm final error — **reject it**; this is exactly the hazard the dry run is for. The $+\hat y_s$ line is approved (Checkpoint 3 values: peak $11.17^\circ$/s ≤ 25 % of rated, $\sigma_{\min} \ge 0.039$, all joints more than $10^\circ$ inside the $\pm165^\circ$ limits; final joints $(0.244, -0.427, 1.004, -2.176, -1.577, 0.244)$ rad). Write the commanded trajectory with `write_joint_log` (degrees, 50 Hz) and add 1 s of hold before and after.

**6b — Hardware session (optional, Tier 1, supervised; Course syllabus §7 Hardware and safety, project brief §4.1).** This part is optional: do it only if your group has a supervised slot. Every mark of Lab 10 can be earned without it (6a and 6c use the example log). If you do run, every rule below is mandatory.

1. *Before powering.* The TA signs the course-arm hardware safety checklist (the shared checklist for all three courses, also quoted in the project brief §4.1). Its rules:
    * software joint limits ≥ $10^\circ$ inside the hardware limits on every joint (hardware ±165°, J6 ±175° → software ≤ ±155°, J6 ≤ ±165°);
    * taped keep-out zone around the base, radius = reach + 0.3 m ≈ 0.58 m;
    * joint speed ≤ 25 % of rated (≤ 40 °/s) on the first run of any new motion and throughout SBEG 361 session H1; ≤ 50 % of rated (≤ 80 °/s) otherwise — today's line is a new motion, so ≤ 40 °/s;
    * payload ≤ 50 % of the rated 0.25 kg, i.e. ≤ 0.125 kg, unless the department approves an exception;
    * every person who touches the arm has completed the hardware safety induction before their first hands-on run;
    * nobody is inside the keep-out zone while the arm can move; a hand may enter for a measurement only with the arm in latched stop and torque disabled (or power removed), with the safety officer's approval;
    * before every session: emergency stop tested; supervisor present for the whole run.

    Also: one operator and one person on the stop; no person as a target (use a foam ball or 3D-printed "mouth" target); derated current ≤ 60 % of rated; logging and video running. This 15–20 min run is a Lab 10 activity, separate from the project sessions H1 and H2.
2. *Calibrate.* Drive the arm to the model zero and record `servo_zero_offset` (convention: $q_\text{model} = q_\text{servo} - \text{offset}$, joint by joint, in the zero and sign of the PoE model). Check HW-1: tool point within 10 mm of the PoE prediction at 3 poses (ruler or grid paper).
3. *Run.* Move to `q_home` in joint space at low speed, then stream the approved commanded joints at 50 Hz (or with the arm SDK's timed joint command); log commanded and measured joints at ≥ 20 Hz in the `hwcompare.py` CSV format. Film the run with a ruler in view.
4. *Stop test (HW-6).* On a second run press the software stop mid-motion; the arm must halt within 0.5 s with ≤ $5^\circ$ joint motion and return to the stow pose only on the operator's command.
5. *Power down* and leave the bench clear. If the arm faults, fill in the fault log (date, arm ID, symptom, what was tried, TA signature) and re-book (project brief §4.1).

**6c — Evaluate (everyone).** `tracking_report(t, cmd_deg, meas_deg, Slist, M)`: per-joint RMS and maximum $\lvert\theta_\text{cmd} - \theta_\text{meas}\rvert$ (HW-2), tool-point RMS and maximum $\lVert p(\theta_\text{cmd}) - p(\theta_\text{meas})\rVert$ in mm (HW-3), peak measured joint speed (HW-5). Run it on `data/hw_log_example.csv` (a synthetic log of the same line: servo lag $\tau = 0.06$ s, $0.3^\circ$ backlash, 12-bit encoders, noise), and also on your own log if you did the optional run in 6b. You may also run the project's `starter/hwcompare.py` on the same file.

*Checkpoint values (example log, 251 samples, 5 s).* HW-2: joint RMS ≤ $0.362^\circ$, max $0.761^\circ$ (limits $2^\circ$ / $5^\circ$: PASS); HW-3: tool point RMS 1.21 mm, max 2.02 mm (limits 5 / 10 mm: PASS); HW-5: peak measured speed $15.4^\circ$/s ≤ $40^\circ$/s (25 % of rated): PASS. Note that the *measured* peak speed is higher than the commanded $11.2^\circ$/s — differentiating quantised, noisy encoder readings amplifies noise.
Tests: `test_tracking_report_*`.

### Checkpoint 7 — Report experiments (at home)

Write `experiments.py` (or a notebook) that regenerates every figure and table of the report:

* **Figure 1** — EOD line, cases (i)–(iv): wrist roll angles $\theta_4$, $\theta_6$ and the largest joint rate against time, with the $30^\circ$/s limit; second panel: $\sigma_{\min}(J_b)$ and $\lambda(t)$ of the adaptive case.
* **Figure 2** — position error against time for (ii)–(iv), and the trade-off curve "peak joint rate vs. maximum tracking error" for $\lambda \in \{0, 0.005, 0.01, 0.02, 0.04, 0.08\}$.
* **Figure 3** — therapy circle on the planar 3R: $\theta_3$ at the end of each loop for $k_0 = 0$ and $k_0 = 5$, and $h(\theta)$ per loop; optional inset: feeding-arm elbow path during the self-motion.
* **Figure 4** — IK error history ($\lVert v_b\rVert$ per iteration, log scale) for the reachable and unreachable targets, $\lambda = 0$ and $0.02$.
* **Figure 5** — course arm: commanded vs. measured joints (the example log, or your own log if you ran the arm) and tool-point error against time; mark the start and end of the line.
* **Table 1** — the Checkpoint 3 $\Delta t$/$k_p$ table; **Table 2** — HW-2, HW-3 and HW-5 from the log with values, limits and PASS/FAIL (add HW-1 and HW-6 if you ran the arm); **Table 3** — hexapod: leg lengths at home and at the correction pose of Workshop P5, first-order vs. exact leg changes.

# Deliverables

1. `starter.py` renamed `lab10_<ID>.py`, passing `LAB_IMPL=lab10_<ID> pytest -q test_lab.py` (30 tests, auto-checked).
2. `experiments.py` / `.ipynb` that regenerates Figures 1–5 and Tables 1–3.
3. The example-log analysis (in `experiments.py` and Section V). Optional, only if you ran the arm: your hardware log (CSV), the signed safety checklist (photo) and the run video link (or the fault log if the session faulted).
4. **Formal report 3** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices).
5. AI-use note (appendix A) and reflection (≤ 200 words, appendix B).
6. Copy the 15 functions into `sbeg361_kinematics/ik_velocity.py` (Checkpoints 1–4, 6) and `sbeg361_kinematics/parallel.py` (Checkpoint 5) and commit with the message `Lab 10: resolved-rate, DLS, redundancy, hexapod`.

# Formal report 3 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Singularity-robust resolved-rate control for an EOD manipulator and redundant assistive robots, validated on a desktop arm*. Suggested length per section in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (pseudo-inverse, DLS, adaptive damping, null-space resolution, hardware-log evaluation), key numbers (peak rates and errors of the four EOD cases, joint drift with and without joint centring, HW-2/HW-3 results), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why joint-rate explosions near singularities are a safety issue for teleoperated EOD manipulation and for robots touching patients; why redundancy is used in assistive robots; objectives and contributions. |
| **II. Models** (½ page) | EOD 6R, course arm, planar 3R, 7R feeding arm, hexapod: a table of the parameters used (state that they are illustrative and where they come from). Where the singularities of each are (geometric description). |
| **III. Methods** (1 page) | Inverse velocity kinematics (MR Eq. 6.7) and the pseudo-inverse via the SVD; DLS as a regularised least-squares problem and its singular-value filter; adaptive damping rule; null-space projector and weighted pseudo-inverse (MR p. 232); task-space feedforward + feedback (MR Eq. 11.16); Newton–Raphson IK with damping; Stewart–Gough inverse Jacobian (MR §7.2.1). Short derivations in your own words. |
| **IV. Simulation results** (1½ pages) | Figures 1–4 and Tables 1 and 3. Every figure is referred to in the text, has labelled axes with units, and a caption that says what to see. |
| **V. Hardware validation** (½ page) | Dry-run decision (why the $+\hat x_s$ line was rejected), Figure 5 and Table 2 from the example log; explain every sim-vs-hardware difference (servo lag, backlash, encoder resolution, and what zero calibration and gravity sag would add on a real arm). If you did the optional run: session procedure and your own log next to the example log. |
| **VI. Discussion** (1 page) | (a) Which damping strategy would you put in the EOD robot's teleoperation software, with which parameters, and what should the operator see? (b) What does the operator lose with DLS, and when is that acceptable? (c) Why do the therapy robot's joints drift, why does it matter for a 30-minute session, and what are the risks of the joint-centring fix? (d) Why is the hexapod's forward kinematics solved iteratively, and how do you make sure it finds the right assembly mode? (e) Limits of the models and of the hardware test. |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | MR (sections and pages), Corke or MLS if used, `modern_robotics` software, the project brief for the hardware tests; IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "no AI tools used"). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clear problem, EOD and biomedical cases motivated, measurable objectives, key numbers in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Models and methods | 20 | Correct MR equations with meaning (minimum norm, null space, singular-value filter of DLS, feedforward frame change in Eq. 11.16); singular configurations described geometrically; DLS derived | Minor slips (e.g. $\lambda$ vs $\lambda^2$, twist frame) | Equations listed without explanation | Wrong or missing |
| Simulation results | 20 | Figures 1–4 and Tables 1, 3 complete, labelled, units, informative captions, values consistent with the checkpoints; trade-off curve interpreted | One figure weak or unlabelled | Several missing or unreadable | Missing |
| Hardware validation | 15 | Dry-run decision justified with numbers; HW-2, HW-3, HW-5 table from the log (example or own) with PASS/FAIL; sim-vs-hardware differences explained with causes | Table complete, analysis of the differences partial | Log shown without analysis | Missing |
| Discussion, safety and limits | 20 | Justified damping choice and operator feedback; DLS cost quantified; drift risk for therapy explained with numbers; assembly-mode issue; honest model and test limits | Choices stated with partial justification | Descriptive only | Missing |
| Writing, format, references | 5 | IEEE format, ≤ 6 pages, concise, correct citations with pages | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 10 | Honest AI note; thoughtful reflection; `experiments.py` regenerates every figure and table; library committed | One item weak | Items superficial | Missing |

Marking: code and tests 30 % (auto-tests), report 70 % (this rubric, scaled), as for formal report 1. Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the code mark at 50 %. A simulation-only report (example log, no hardware run) can earn full marks on every criterion, including "Hardware validation"; an optional hardware run earns no extra marks and is not required.

# Reflection prompts (appendix B, ≤ 200 words)

* Before the lab, what did you expect the pseudo-inverse to do at the EOD wrist singularity? What actually happened, and which plot convinced you?
* The adaptive rule needs two parameters. How would you choose them for a new robot without trial and error?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did a test or a plot reveal it?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `pinv_svd` fails only on tall or rank-deficient matrices | Use `full_matrices=False` and invert only $\sigma_i > $ `tol`$\cdot\sigma_1$; the zero singular values must stay zero in $\Sigma^\dagger$. |
| `dls_inverse` differs from the test by a factor | $\lambda^2$, not $\lambda$, is added to $JJ^{\mathsf T}$; the identity matrix is $m\times m$ (rows of $J$). |
| `null_projector` gives $JP \ne 0$ | You wrote $I - JJ^\dagger$ ($m\times m$) instead of $I - J^\dagger J$ ($n\times n$). |
| Resolved-rate error grows linearly with time even far from singularities | $k_p = 0$ and large $\Delta t$ (Euler drift), or $\mathcal V_d$ given in the space frame while $J_b$ is used. Check that `line_trajectory` returns body twists $R_0^{\mathsf T}\hat u\,v$. |
| The EOD error is large in case (iii) but not in (i) | Expected: DLS trades accuracy for bounded rates near the singularity; check that the error falls again after the singular point. |
| `resolved_rate_sim` rate limit changes the path | Scale the **whole** $\dot\theta$ vector by one factor; clipping each joint separately changes the direction of motion. |
| `ik_newton` returns one iteration too many / too few | Record the error of the initial guess first; `iterations` counts steps taken, `len(err) = iterations + 1`. |
| Hexapod FK stalls at about $10^{-7}$ m | Your `matrix_exp6` drops rotations below $10^{-6}$ rad (`mr.MatrixExp6` does); keep the first-order term, or stop at $10^{-9}$. Multiply the update on the **left** ($e^{[\mathcal V]}T$): $J_s^{-1}$ uses space twists. |
| `tracking_report` task errors of order 100 mm | Angles in the log are **degrees**; convert with `np.deg2rad` before `fk_space`. |
| Hardware arm moves differently from MR Studio | Zero-pose offset not applied, or joint direction reversed in the servo SDK; repeat HW-1 before any long run. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside your Lab code folder, or set `LAB_IMPL` to your file name without `.py`. |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, report and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`) and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers, including the random matrices that `test_lab.py` draws with NumPy), `CMakeLists.txt`. The data files are the same `data/robots.json`, `data/hexapod.json` and `data/hw_log_example.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 10 --cpp                     # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*pinv_*,*dls_*"         # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 26 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 15 as in Python, same order): `pinv_svd`, `dls_inverse`, `dls_singular_gains`, `adaptive_lambda` (Checkpoint 1); `null_projector`, `weighted_pinv`, `redundant_rates`, `joint_centering_gradient` (Checkpoint 2); `line_trajectory`, `resolved_rate_sim` (Checkpoint 3); `ik_newton` (Checkpoint 4); `hexapod_leg_lengths`, `hexapod_inverse_jacobian`, `hexapod_forward_kinematics` (Checkpoint 5); `tracking_report` (Checkpoint 6). Given (complete, as in `starter.py`): `vec_to_so3`, `so3_to_vec`, `vec_to_se3`, `se3_to_vec`, `trans_inv`, `adjoint`, `matrix_exp3`, `matrix_exp6` (with the same first-order rotation below $10^{-6}$ rad as in Python), `matrix_log3`, `matrix_log6`, `fk_space`, `body_screws`, `jacobian_space`, `jacobian_body`, `load_robot`, `load_hexapod`, `read_joint_log`, `write_joint_log`, and, C++ only, `gradient` (= `np.gradient(Y, t, axis=0)`). Do not call `mrcpp` (e.g. `mr::pinv`, `mr::ikin_body`) inside `starter.cpp` — the tests compare against it.

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Pseudo-inverse and DLS | `-tc="*pinv_*,*dls_*,*adaptive*"` | 7 |
| 2 — Redundancy | `-tc="*null_projector*,*weighted*,*redundant*,*centering*,*srs7r*"` | 6 |
| 3 — Resolved-rate control | `-tc="*line_trajectory*,*resolved_rate*"` | 4 |
| 4 — IK near singularities | `-tc="*ik_newton*"` | 3 |
| 5 — Hexapod | `-tc="*hexapod*"` | 4 |
| 6 — Hardware log | `-tc="*tracking_report*"` | 2 |

All 26 test cases pass with `./build/lab_tests`. The Python track counts 30 tests because pytest counts each matrix shape of `test_pinv_svd_matches_numpy` separately; the C++ track checks the same five shapes in one test case. The checkpoint values in the procedure above are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `J`, `Slist`, Jacobians: NumPy arrays; `theta`, `xdot`: 1-D arrays | `Eigen::MatrixXd` ($6\times n$ screw lists: one axis per column, as in Python); `Eigen::VectorXd` |
| `np.linalg.svd(J, full_matrices=False)` | `Eigen::JacobiSVD<Eigen::MatrixXd>` with the options `Eigen::ComputeThinU` and `Eigen::ComputeThinV` (hint in `starter.cpp`); `svd.singularValues()` (descending), `svd.matrixU()`, `svd.matrixV()` is $V$, **not** $V^{\mathsf T}$ |
| `np.linalg.solve(A, B)`, `np.linalg.inv(A)`, `np.diag(s)` | `A.partialPivLu().solve(B)`, `A.inverse()`, `s.asDiagonal()` |
| `dls_singular_gains` returns a tuple; `np.inf` | `std::pair<VectorXd, VectorXd>` (`.first` = $\sigma$, `.second` = gains); `std::numeric_limits<double>::infinity()` |
| `line_trajectory` returns `(t, Xd, Vd)`; `Xd` a list of $4\times4$ arrays | `lab::LineTrajectory` with fields `t` (`VectorXd`), `Xd` (`std::vector<Eigen::Matrix4d>`), `Vd` ($N\times6$, row = sample) |
| `resolved_rate_sim(..., qdot_max=None, eps=None, lam_max=None)` returns a dict | `qdot_max`: empty `Eigen::VectorXd()` = no limit, one entry = the same limit for every joint (`Eigen::VectorXd::Constant(1, 30 * PI / 180)`), or $n$ entries; `eps`, `lam_max`: `std::optional<double>` (pass a number, or leave out); returns `lab::RateSimResult` with fields `theta`, `qdot`, `pos_err`, `ang_err`, `sigma_min`, `lam` |
| `ik_newton` returns `(theta, success, iterations, err)` | `lab::IKNewtonResult` with the same field names; `err` is `std::vector<std::pair<double, double>>` ($\lVert\omega_b\rVert$, $\lVert v_b\rVert$) |
| `load_hexapod()` returns `(A, B, info)`; `hexapod_forward_kinematics` returns `(p, R, iterations)` | `lab::Hexapod` (`A`, `B`: $6\times3$, row $i$ = $a_i$ / $b_i$; `name`, `h0`, `stroke`, `leg_speed_max`, `rated_leg_force`); `lab::HexapodFKResult` (`p`, `R`, `iterations`) |
| `read_joint_log` returns `(t, cmd_deg, meas_deg)`; `tracking_report` returns a dict | `lab::JointLog` (`t`, `cmd_deg`, `meas_deg`, $N\times n$); `lab::TrackingReport` with the same five field names |
| `np.gradient(meas, t, axis=0)`, `np.deg2rad` | `lab::gradient(meas_deg, t)` (given; same numbers), `* PI / 180` with a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

One check differs: in `test_ik_newton_unreachable` the pseudo-inverse iterates are chaotic, so the largest linear error they reach depends on rounding (0.460 m with NumPy, about 0.39 m with Eigen); the C++ test requires a jump above 0.1 m instead of 0.4 m, and in addition checks that the first 8 iterates (which include the first jump, 0.201 m at iteration 3) equal the Python ones. Two C++ test cases also compare the final joints of the course-arm line and every entry of the example-log report with the Python reference.

**Report experiments (Checkpoint 7) in C++.** Write a small `experiments.cpp` (your own `main()`; add it to a second executable or build it by hand next to `starter.cpp`) that runs the simulations of Figures 1–5 and Tables 1–3 and writes the results as CSV files (`sbeg::write_csv` from `<sbeg/labio.hpp>`; the dry-run command log of Checkpoint 6a with `lab::write_joint_log`), and plot them with a short Python script (matplotlib) or any plotting tool you prefer. Plotting stays in Python; the report requirements do not change.

A matrix-size mistake (e.g. multiplying a $6\times n$ Jacobian by a 3-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `pinv_svd` fails only for tall or rank-deficient $J$ | full instead of thin SVD, or used `svd.matrixV()` as if it were $V^{\mathsf T}$ | use the thin options (`Eigen::ComputeThinU`, `Eigen::ComputeThinV`); $J^\dagger = V\,\Sigma^\dagger U^{\mathsf T}$ = `svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose()` |
| `pinv_svd` result contains NaN or inf | divided by a zero singular value | set $1/\sigma_i$ to 0 for $\sigma_i \le$ `tol`$\cdot\sigma_1$ |
| Wrong results only after `auto A = J * J.transpose();` | `auto` keeps an Eigen expression, not a matrix | write the type: `Eigen::MatrixXd A = J * J.transpose();` |
| `resolved_rate_sim` with a rate limit gives "wrong matrix size" | `qdot_max` has one entry and was divided element-wise by the $n$ joint rates | expand it first: `Eigen::VectorXd::Constant(n, qdot_max(0))` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
