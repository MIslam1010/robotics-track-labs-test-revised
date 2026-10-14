---
title: "Lab 8 — PD vs LQR vs MPC on an exoskeleton joint with torque and range limits (Formal report 2)"
subtitle: "SBEG 46X Medical Robotics — Week 9"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work for the report |
| CLOs | **46X-6** (design and evaluate optimal controllers — LQR, trajectory optimisation, MPC — for exoskeletons and rehabilitation robots); **46X-10** (technical report, honest disclosure of tool use) |
| Reading | LQR warm-up: your SBEG 461 Week 7 notes (or Tedrake §8.1 and Åström & Murray §6.3 below). Tedrake, *Underactuated Robotics*: Ch. 8 §8.1 (pp. 129–131), §8.3.1–8.3.2 (pp. 135–137); Ch. 10 §10.2.1–10.2.3 (pp. 188–190), §10.5.2 (p. 198). Åström & Murray, *Feedback Systems*: §6.3, Eqs. (6.26)–(6.27) (pp. 190–191), Exercise 6.15 (Bryson's rule, p. 200). *Modern Robotics* §11.4.1 (Eqs. 11.21–11.27) and Eq. (11.38); §10.7 (pp. 394–395). This week's lecture notes and Workshop 8 |
| Files (your Lab code folder) | `starter.py` (your code), `test_lab.py` (auto-checks), `course_models.py` (the elbow of `arm_exo_2r` and the knee of `load_exo_2r`, copied from Course resources: Course robot models and hardware safety checklist), `data/exo_joint.json` (parameters), `data/ref_elbow_*.csv` (reference runs), `data/expected.json` |
| QP solver | The provided `solve_qp` (NumPy/SciPy only: null-space elimination + least-distance programming with `scipy.optimize.nnls`). The auto-checks use it. For the timing study you may also try OSQP (`pip install osqp`) if your environment allows it; report which solver produced each timing |
| Hardware | **Simulation.** Optional, ungraded: in the project hardware sessions the Hapkit can emulate a 1-DOF joint (its 2 N handle-force limit and ±40 mm watchdog play the roles of the torque and range limits) |
| Assessment | Auto-checked code (tests) + **Formal report 2** (IEEE two-column, ≤ 6 pages, template and rubric below), due one week after the lab, before the Week 10 lab session |
| AI use | Allowed for code **with disclosure** (AI-use note in the report, appendix A); you must be able to explain every line in an oral check. Derivations, numbers, figures and conclusions in the report are your own and come from your runs |

## Why this lab matters

A powered elbow exoskeleton helps a stroke survivor practise flexion–extension. The therapist prescribes the movement
*and* its limits: the joint must stay inside the patient's comfortable range of motion (here 0.20–1.60 rad), must not
move faster than 3 rad/s, and the motor torque is derated to 6 N·m. Three controllers can drive the joint:

* the **PD + gravity compensation** law you tuned in SBEG 461 Weeks 5–6 (gains from $\omega_n$ and $\zeta$);
* **LQR** (SBEG 461 Week 7), which is *the same PD structure* with gains from a cost on error, speed and torque;
* **MPC**, which solves a small quadratic programme (QP) every 10 ms and knows the torque, range and speed limits.

You will see that the first two can be fast and smooth on paper and still break a limit once the motor saturates,
that MPC respects the limits by design, and what it costs (a slower movement, a QP every sample, sensitivity to a wrong
model). An optional defence experiment repeats the comparison on the knee of a soldier load-carriage exoskeleton,
where the same design choices decide whether a loaded knee holds or folds.

## Learning objectives

By the end of the lab you can:

1. build the error model of a gravity-compensated joint and discretise it exactly (zero-order hold);
2. *(warm-up, recap of SBEG 461 Week 7)* solve continuous and discrete Riccati equations (closed form, recursion,
   library) and choose $Q$ and $R$ with Bryson's rule;
3. *(warm-up)* explain why LQR on this joint is a PD law and compare its gains with a hand-tuned PD;
4. write MPC with input and state constraints as a QP in the direct-shooting (condensed) and direct-transcription forms,
   and implement the receding-horizon law;
5. evaluate PD, LQR and MPC against clinical and safety metrics, including robustness to model error, and report the
   comparison in an IEEE-style technical report.

# Pre-lab check (complete before the session, 5 questions)

Questions 1–3 are a recap of SBEG 461 Week 7 (LQR); Questions 4–5 prepare the new part of the lab.

1. For $\dot x = -2x + u$ with $q = 5$, $r = 1$: give $p$, $k$ and the closed-loop pole.
2. Why is the LQR of $M\ddot\theta + b\dot\theta = u$ with the error state $x = (\theta - \theta_d, \dot\theta)$ a PD
   controller? Name the two gains.
3. Bryson's rule with $|\theta - \theta_d|_{\max} = 0.10$ rad, $|\dot\theta|_{\max} = 1.5$ rad/s and
   $|u|_{\max} = 2.5$ N·m: write $Q$ and $R$.
4. MPC with $N = 30$, two states and one input, with two-sided bounds on the input and on both states: how many decision
   variables and inequality rows has the condensed (shooting) QP, and how many variables and equality constraints has
   the transcription QP?
5. The elbow's largest gravity torque is $mgr = 3.3354$ N·m and the motor limit is 6 N·m. What is the largest $|u|$ that
   guarantees $|\tau| \le 6$ N·m for $\tau = g(\theta) + u$ at every angle?

# The model

The joint is the elbow of the shared course model `arm_exo_2r` (forearm + hand + cuff: 2.0 kg, CoM 0.17 m from the
elbow, $I_{zz} = 0.020$ kg·m²) with the upper arm hanging. With $\theta$ the elbow flexion (0 = forearm hanging),
MR Eq. (11.21) becomes

$$M\ddot\theta + b\dot\theta + g(\theta) = \operatorname{sat}(\tau),\qquad g(\theta) = mgr\sin\theta,\qquad
\operatorname{sat}(\tau) = \min\big(\max(\tau, -\tau_{\max}), \tau_{\max}\big).$$

Every controller adds the model's gravity torque, $\tau = g(\theta) + u$ (MR Eq. 11.38 with $K_i = 0$), so that, while
the motor is not saturated, $M\ddot\theta + b\dot\theta = u$. Error state $x = (\theta - \theta_d, \dot\theta)$:

$$\dot x = Ax + Bu,\quad A = \begin{bmatrix}0 & 1\\ 0 & -b/M\end{bmatrix},\ B = \begin{bmatrix}0\\ 1/M\end{bmatrix};\qquad
x[n+1] = A_dx[n] + B_du[n]\ \ (h = 0.01\ \text{s, zero-order hold}).$$

| Quantity | Value (illustrative) | Quantity | Value |
|---|---|---|---|
| $M$ (MassMatrix of `arm_exo_2r`, elbow) | 0.0778 kg·m² | control period $h$, plant integration | 10 ms, RK4 with 10 sub-steps |
| $b$ (joint + exoskeleton damping) | 0.10 N·m·s/rad | task | $\theta(0) = 0.30$ rad; $\theta_d = 1.50$ rad for $0 \le t < 1.5$ s, then 0.40 rad until 3 s |
| $mgr$ (GravityForces of `arm_exo_2r`) | 3.3354 N·m | PD (hand-tuned) | $\omega_n = 15$ rad/s, $\zeta = 0.5$ |
| motor limit $\tau_{\max}$ (derated from 15 N·m) | 6 N·m | LQR (Bryson) | $x_{\max} = (0.10$ rad, $1.5$ rad/s$)$, $u_{\max} = 2.5$ N·m |
| range of motion (prescribed) | $[0.20, 1.60]$ rad | MPC | $N = 30$ (0.3 s), same $Q$, $R$, terminal weight $P$ = DARE solution |
| speed limit | 3 rad/s | MPC input bound | $\lvert u\rvert \le \tau_{\max} - 3.5 = 2.5$ N·m (gravity reserve 3.5 N·m) |

The provided `simulate(kind, j, g_scale=1.0)` runs `kind = "pd"`, `"lqr"` or `"mpc"` on `j = load_params()["elbow"]`
(or `["knee"]`). It calls **your** functions through `make_controller`; the MPC calls `mpc_control` every sample with the
error-state limits of `error_limits(theta_d, j)` and, if the QP is infeasible, solves it again without the state limits
for that sample (status `"state-constraints-dropped"`). `g_scale` makes the real limb heavier or lighter than the model
(the controller still uses the model). It returns arrays `t, theta, theta_dot, theta_d, u, tau_cmd, tau` (one value per
control period) and the controller data.

# Setup

```bash
cd "<your Lab code folder>"
LAB_IMPL=starter python -m pytest -q test_lab.py     # at the start: 21 tests fail with NotImplementedError, 4 data tests pass
```

Use the course Python environment (or the Docker image / JupyterHub). Work in `starter.py` and keep the function names
and signatures.

# Procedure and checkpoints

### Checkpoint 1 — Warm-up: model, PD and Riccati equations (recap of SBEG 461 Week 7, ≈ 12 min)

You wrote LQR code in SBEG 461 Lab 7; here you rebuild it quickly for this joint, including the discrete-time version
that MPC needs. Keep this checkpoint short and move on.

1. `joint_state_space(M, b)`, `discretize_zoh(A, B, h)` (matrix exponential of $\begin{bmatrix}A & B\\ 0 & 0\end{bmatrix}h$)
   and `pd_gains(M, b, zeta, wn)` (SBEG 461 Week 5).
2. `scalar_care(a, b, q, r)` and `scalar_dare(a, b, q, r)`: the closed forms of Workshop Problems 1 and 2.
3. `lqr(A, B, Q, R)` (you may call `scipy.linalg.solve_continuous_are`), `riccati_recursion(A, B, Q, R, Qf, N)`,
   `dlqr(A, B, Q, R)` (iterate the recursion to its fixed point — do **not** call SciPy here; the test compares with it)
   and `bryson_weights(x_max, u_max)`.
4. `python -m pytest -q test_lab.py -k "state_space or zoh or pd_gains or scalar or lqr or riccati or bryson"`.
   One test checks the golden-ratio recursion $1, \tfrac32, \tfrac85, \tfrac{21}{13}$ of Workshop Problem 2.

**Checkpoint question:** your continuous LQR gives $K = (25.00, 2.484)$ and your discrete one $(21.32, 2.216)$ for the
same $Q$, $R$. Why are they different, and which one does the 100 Hz controller need?

### Checkpoint 2 — MPC as a QP (≈ 18 min)

1. `prediction_matrices(A, B, N)` (direct shooting: $X = S_xx_0 + S_uU$).
2. `mpc_qp(...)`: $H$, $f$ and the constraint rows $GU \le w$ for the input limits and the state limits (use the provided
   helper `_limit_rows`).
3. `transcription_qp(...)`: the same problem with $z = (x_1, \dots, x_N, u_0, \dots, u_{N-1})$ and the dynamics as
   equality constraints.
4. `mpc_control(...)`: build, solve with `solve_qp`, return the first input and the status.
5. `python -m pytest -q test_lab.py -k "prediction or mpc or transcription"`. Two tests are worth understanding: with
   inactive constraints and terminal weight = DARE solution, MPC returns exactly $-Kx$ (LQR); and the condensed and
   transcription QPs give the same optimum.

### Checkpoint 3 — Closed-loop runs and metrics (rest of the session, then at home)

1. `metrics(sim, j)`: range-of-motion and speed violations, overshoot, settling time (2 % band of 0.02 rad), peak
   torque, fraction of saturated samples, rms error and effort $h\sum\tau^2$ (definitions in the docstring).
2. Run the whole suite: `python -m pytest -q test_lab.py` — all 25 tests must pass. The run tests compare your
   simulations with `data/ref_elbow_*.csv`.
3. Do the experiments below for the report. Save every figure with axis labels and units.

**Experiment A (warm-up) — LQR is PD with chosen gains.** Keep it to one table and two sentences; the report's weight
is on Experiments B–E.

* A1. **Table 1:** $K_p$, $K_d$, $\omega_n$, $\zeta$ for the hand-tuned PD, the continuous LQR (closed form of Workshop
  Problem 3 *and* `lqr`) and the discrete LQR (`dlqr`, $h = 0.01$ s; also $h = 0.05$ s). Comment on the difference
  between continuous and discrete gains.

**Experiment B — PD vs LQR vs MPC on the elbow (biomedical).**

* B1. **Figure 1:** $\theta(t)$ for the three controllers with $\theta_d(t)$ and the range-of-motion limits.
  **Figure 2:** $\tau(t)$ (with $\pm\tau_{\max}$) and $\dot\theta(t)$ (with $\pm 3$ rad/s) for the three controllers.
* B2. **Table 2:** all `metrics` for the three controllers. Which controller breaks which limit, by how much, and why?
  Relate every violation to the saturated part of Figure 2.

**Experiment C — Choosing $Q$ and $R$.**

* C1. Sweep the error limit of Bryson's rule $x_{\max,1} \in \{0.20, 0.10, 0.05, 0.02\}$ rad (keep the others). For LQR
  and MPC: **Figure 3:** overshoot, range-of-motion violation and settling time versus $x_{\max,1}$. Explain why a more
  "aggressive" LQR becomes *less* safe once the motor saturates, while MPC becomes slightly faster and stays safe.

**Experiment D — Horizon, formulation and computation time.**

* D1. With $x_{\max,1} = 0.02$ rad, run MPC for $N \in \{2, 3, 5, 10, 20, 40\}$. **Figure 4:** overshoot and
  settling time versus $N$, and the mean wall-clock time per `mpc_control` call (use `time.perf_counter`; state your
  computer). Is the 10-ms budget met?
* D2. **Table 3:** for one state ($x_0 = (-1.2, 0)$, $N = 30$): sizes of the condensed and transcription QPs (variables,
  equality and inequality rows, non-zeros of $H$), the solve time of each with `solve_qp`, and the difference between
  the two first inputs.

**Experiment E — Model error: a heavier forearm.**

* E1. Run the three controllers with `g_scale=1.3` (the forearm, cuff and hand are 30 % heavier than the model).
  Measure the steady error at the end of the first step and compare it with your prediction
  $\Delta g/K_p = 0.3\,mgr\sin(1.5)/K_p$. Does MPC still respect every limit? Propose a fix (integral action, a
  disturbance estimate — Week 10's Kalman filter — or a larger gravity reserve) and say what each costs.
* E2 *(optional, defence)*. Repeat B and E1 on the knee of the load-carriage exoskeleton,
  `j = load_params()["knee"]` (squat from 0.10 to 0.50 rad and back; $\tau_{\max} = 200$ N·m; MPC input bound
  $\lvert u\rvert \le 20$ N·m after a 180 N·m gravity reserve), with `g_scale` = 1.0 and 1.1 (a 10 % heavier load than
  modelled). Which controllers keep the knee, which let it fold, and why? (Hint: compare the authority each one has left
  after gravity, and Workshop Problem 6.) Note that the simulated knee has no mechanical end stop.

**Expected values (to check yourself; TAs use them for calibration).**

| Quantity | Expected |
|---|---|
| $A_d$, $B_d$ ($h = 0.01$ s) | $\begin{bmatrix}1 & 0.009936\\ 0 & 0.98723\end{bmatrix}$, $(0.00063993,\ 0.12771)$ |
| $Q$, $R$ (Bryson) | $\operatorname{diag}(100,\ 0.4444)$, $0.16$ |
| PD gains $K_p$, $K_d$ | 17.51 N·m/rad, 1.067 N·m·s/rad |
| Continuous LQR $K$ / discrete LQR $K$ ($h = 0.01$ s) | $(25.00,\ 2.484)$ / $(21.32,\ 2.216)$ |
| PD: range violation, speed violation, overshoot, settling time | 0.0602 rad, 5.985 rad/s, 0.212 rad, 0.61 s |
| LQR: range violation, speed violation, overshoot, settling time | 0, 4.247 rad/s, 0.0004 rad, 0.36 s |
| MPC: range violation, speed violation, peak torque, settling time | 0, 0, 3.874 N·m, 0.53 s |
| Fraction of saturated samples: PD / LQR / MPC | 0.0967 / 0.0567 / 0 |
| Effort $h\sum\tau^2$: PD / LQR / MPC | 28.80 / 25.60 / 19.21 N²·m²·s |
| C1, LQR $K$ for $x_{\max,1}$ = 0.20 / 0.05 / 0.02 rad | $(10.94,\ 1.862)$ / $(40.85,\ 2.773)$ / $(93.03,\ 3.913)$ |
| C1, LQR overshoot for 0.20 / 0.10 / 0.05 / 0.02 rad | 0 / 0.0004 / 0.161 / 0.513 rad (range violation 0.313 rad at 0.02) |
| C1, MPC settling time for 0.20 / 0.10 / 0.05 / 0.02 rad | 0.71 / 0.53 / 0.48 / 0.46 s, no violation |
| D1, MPC overshoot for $N$ = 2 / 3 / 5 / 10 (same for 20 and 40) | 0.0274 / 0.0217 / 0.0134 / 0.0085 rad |
| E1, steady error after the first step: PD / LQR / MPC | −0.0567 / −0.0466 / −0.0466 rad (predicted −0.0570 / −0.0468) |
| E1, MPC speed violation | 0.123 rad/s (the limits hold only for the model) |
| E2, knee discrete LQR $K$ | $(386.7,\ 124.7)$ |
| E2, knee, `g_scale` = 1.0 | all three hold every limit; settling time PD / LQR / MPC 0.98 / 0.80 / 1.03 s |
| E2, knee, `g_scale` = 1.1 | LQR holds (range violation 0); PD and MPC let the knee fold (range violations of more than 2 rad) |

<!-- library-block:begin -->
# Your `sbeg46x_medrob` library

When the tests pass, add the functions you wrote in Lab 8 to a new module `optctrl.py` of your library. The functions you
write (14):

`joint_state_space(M, b)`, `discretize_zoh(A, B, h)`, `pd_gains(M, b, zeta, wn)`, `scalar_care(a, b, q, r)`,
`scalar_dare(a, b, q, r)`, `lqr(A, B, Q, R)`, `riccati_recursion(A, B, Q, R, Qf, N)`, `dlqr(A, B, Q, R)`,
`bryson_weights(x_max, u_max)`, `prediction_matrices(A, B, N)`, `mpc_qp(A, B, Q, R, P, N, x0, u_lim, x_lim)`,
`transcription_qp(A, B, Q, R, P, N, x0, u_lim, x_lim)`, `mpc_control(x0, A, B, Q, R, P, N, u_lim, x_lim)`,
`metrics(sim, j)`.

The provided Lab 8 `solve_qp(H, f, G, w, Aeq, beq, tol)` (null-space elimination + NNLS, returns
`(z, 'optimal'|'infeasible')`) and `_limit_rows` are used by Lab 8 only. Lab 10 (CBF-QP safety filter) and the project
ship a different SciPy-SLSQP `solve_qp(H, f, A_ub, b_ub, lb, ub, x0)` that returns `(x, info)`, so when you port
`mpc_control` into the project's `LinearMPC.solve`, rewrite its status test for the `info` dict.
The other provided helpers (`load_params`, `setpoint`, `gravity`, `error_limits`, `make_controller`, `simulate`) do not
have to go into the library. The full list of library
functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab8_<ID>.py`, passing `LAB_IMPL=lab8_<ID> python -m pytest -q test_lab.py` (25 tests), and
   the 14 functions added to `optctrl.py` of your library.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 2** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–4 and
   Tables 1–3 (plus E1, and E2 if you did it).
4. A reflection (≤ 200 words, appendix B) and an AI-use note (appendix A).

# Formal report 2 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Optimal and constrained
control of a rehabilitation exoskeleton joint: PD, LQR and model-predictive control under torque, speed and
range-of-motion limits*. Suggested length per section in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (gravity-compensated joint model, LQR with Bryson weights, condensed MPC solved as a QP), key numbers (limit violations of PD and LQR, settling time and peak torque of MPC, steady error under a 30 % model error), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why limits matter in robot-assisted therapy (range of motion, speed, torque derating); the clinical task; the defence analogue if you did E2; objectives and contributions. |
| **II. System model** (½–1 page) | Joint model from the course robot (MR Eq. 11.21), gravity compensation, error state, ZOH model with numbers, the limits and the task (state that the values are illustrative). |
| **III. Controller design** (1 page) | PD from $\omega_n$, $\zeta$; LQR as a brief recap of SBEG 461 Week 7 (a few lines: cost, Riccati equation in continuous and discrete time, Bryson's rule, the closed form showing LQR = PD); MPC, the main part: horizon, cost with terminal weight, constraint rows, shooting vs transcription, receding-horizon law, input tightening by the gravity reserve, what happens if the QP is infeasible. Cite Tedrake, Åström & Murray and MR with sections and pages. |
| **IV. Verification** (¼ page) | Unit tests (what each checks), reference runs, and one independent check of your own (e.g. a hand Riccati iteration, or MPC = LQR when no constraint is active). |
| **V. Results** (1½ pages) | Figures 1–4 and Tables 1–3 (+ E1/E2). Every figure is referred to in the text and has a caption that says what to see. |
| **VI. Discussion** (1 page) | Why saturated PD/LQR break limits; the price of MPC (speed, computation, dependence on the model); choice of $Q$, $R$ and $N$; robustness to model error and your proposed fix; which controller you would put on a clinical device and what you would add first (integral action / estimation, a safety filter as in Week 11); limits of the model (1-DOF, rigid cuff, no patient effort, no sensor noise). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | Tedrake (chapters, sections and page of the course PDF), Åström & Murray (sections, pages), *Modern Robotics* (sections, equations); IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "No AI tools were used."). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clinical (and, if done, defence) motivation with the actual limits; measurable objectives; key numbers in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Model and controller design | 20 | Correct model and gravity compensation; LQR recap concise and correct (Riccati equations, Bryson weights, LQR = PD); MPC QP written with all constraint rows, terminal weight and tightening justified; shooting vs transcription explained, not copied | Minor slips in derivation or notation | Equations listed without explanation | Wrong or missing |
| Verification evidence | 10 | All tests pass; reference comparison; at least one independent check of your own, quantified | Tests pass, independent check weak | Some tests fail or no check | No verification |
| Results (figures and tables) | 25 | Figures 1–4 and Tables 1–3 complete, labelled, with units, limits drawn and captions; numbers consistent with the expected values; timing with the computer and solver stated | One figure or table weak | Several missing or unreadable | Missing |
| Discussion, safety and recommendation | 20 | Every violation explained by saturation; cost of MPC quantified; $Q$/$R$/$N$ trade-offs; model-error result explained and a fix proposed with its cost; justified recommendation for a clinical device; model limits | Recommendation with partial justification | Descriptive only | Missing |
| Writing, format, references | 10 | IEEE format, ≤ 6 pages, concise, correct citations with pages | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every figure and table | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, that a "better-tuned" LQR overshoots *more* once the motor saturates, or
  that a 10 % heavier load can make the "safe" controller lose the knee), and why?
* If you were the therapist (or the soldier), which controller would you trust, and what would you want the device to
  tell you when it cannot meet a limit?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it
  was wrong (a test, a hand calculation, a plot)?

# Safety note

This lab is simulation only. If you later run any of these controllers on hardware (Hapkit joint emulator, course arm or
project device), follow Course resources: Course robot models and hardware safety checklist: run the exact command
first in simulation, keep the device's own force/torque saturation and watchdog **in addition to** any MPC constraint
(an optimiser's constraint is not a safety system — it holds only for the model), start with small gains and short
moves, and keep the emergency stop with a team member who is not operating the device.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `discretize_zoh` gives $B_d \approx hB$ only | You used $A_d = e^{Ah}$ and $B_d = hB$ (Euler). Use the block exponential. |
| Riccati recursion test fails at $K[N-1]$ | $K[n]$ must use $S[n+1]$; with $Q_f = 0$ the last gain is 0. |
| `dlqr` does not match SciPy by a factor | You returned $K = R^{-1}B^TS$ (continuous formula). Discrete: $K = (R + B^TSB)^{-1}B^TSA$. |
| MPC cost test fails, constraints test passes | Terminal weight on the wrong block ($P$ belongs on $x_N$), or $f$ without $S_xx_0$. |
| State-limit rows wrong | The rows are $S_uU \le x_{hi} - S_xx_0$ and $-S_uU \le -x_{lo} + S_xx_0$, repeated for $k = 1..N$. |
| MPC first input differs from $-Kx$ in the "inactive" test | $P$ must be the **discrete** Riccati solution, and $H$, $f$ consistent (both with or both without the factor 2). |
| MPC status `infeasible` at many samples | Range/speed bounds with the wrong sign or not shifted by $\theta_d$; use `error_limits`. |
| `metrics` settling time differs by one sample | Settling time = (index of the last sample outside the band + 1) × $h$, measured from the step. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside your Lab code folder, or set `LAB_IMPL` to your file name without `.py`. |
| Everything is slow | One elbow MPC run (300 QPs, $N = 30$) takes well under a second; do not rebuild the controller inside plotting loops — pass `ctrl=make_controller(...)` to `simulate`. |

# Pre-lab answers (released after the session)

1. $p = -2 + \sqrt{4 + 5} = 1$, $k = 1$, closed-loop pole $-3$.
2. $u = -Kx = -K_p(\theta - \theta_d) - K_d\dot\theta$: proportional on the angle error and derivative (damping) on the
   velocity — exactly PD with gravity compensation; LQR only chooses $K_p$ and $K_d$.
3. $Q = \operatorname{diag}(100,\ 0.4444)$, $R = 0.16$.
4. Shooting: 30 variables, $2N + 4N = 180$ inequality rows. Transcription: $N(n + m) = 90$ variables and $Nn = 60$
   equality constraints (plus the same 180 bounds).
5. $6 - 3.3354 = 2.6646$ N·m (the lab rounds the reserve up to 3.5 N·m, $|u| \le 2.5$ N·m).

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, report and deliverables are the same; the auto-checks compare your results with the closed-form results and the direct simulations of `test_lab.py`, with the reference runs `data/ref_elbow_*.csv` and `data/expected.json`, and with the numbers of the Python reference (SciPy's Riccati solvers, the course models), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided functions — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`, `plot_results.py` (plots runs saved as CSV). The data are read from the same `data/exo_joint.json`, `data/ref_elbow_*.csv` and `data/expected.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 8 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*riccati*"              # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 21 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The whole suite, with six closed-loop runs (300 MPC QPs each), takes well under a second once the code is built.

**Functions to write** (in `starter.cpp`, the same 14 as in Python, same order): `joint_state_space`, `discretize_zoh`, `pd_gains`, `scalar_care`, `scalar_dare`, `lqr`, `riccati_recursion`, `dlqr`, `bryson_weights`, `prediction_matrices`, `mpc_qp`, `transcription_qp`, `mpc_control`, `metrics`. Given in `helpers.cpp` (complete, as the "Provided" section of `starter.py`): `load_params`, `setpoint`, `gravity`, `solve_qp` (the same null-space + least-distance + NNLS method, with a C++ port of `scipy.optimize.nnls`), `error_limits`, `make_controller`, `simulate` (the same RK4 with 10 sub-steps) and `limit_rows` (Python `_limit_rows`), plus, C++ only, `expm` (= `scipy.linalg.expm`, for `discretize_zoh`), `solve_continuous_are` (= `scipy.linalg.solve_continuous_are`, for `lqr`) and `write_sim_csv`. As in Python, `dlqr` must iterate the recursion itself (do not solve the DARE another way).

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Model, PD and Riccati equations | `-tc="*params*,*gravity*,*state_space*,*zoh*,*pd_gains*,*scalar*,*lqr_matches*,*riccati*,*dlqr*,*bryson*"` | 11 |
| 2 — MPC as a QP | `-tc="*solve_qp*,*prediction*,*mpc_qp*,*transcription*,*mpc_without*,*mpc_control*"` | 6 |
| 3 — Closed-loop runs and metrics | `-tc="*setpoint*,*metrics*,*runs*,*mpc_run*"` | 4 |

All 21 test cases pass with `./build/lab_tests` (the Python track counts 25 tests: there `test_scalar_care` and `test_runs_match_reference` are parametrized into 3 items each; in C++ each is one test case that loops over the cases). One difference: in Python the 4 data / provided-code tests pass with the untouched starter; in C++ each of them also checks one of your functions (`test_params_match_course_models` → `joint_state_space`, `test_gravity_matches_modern_robotics` → a short PD run through `simulate`, which calls `joint_state_space`, `discretize_zoh` and `pd_gains`, `test_solve_qp_known_problem` → a one-step `mpc_qp`, `test_setpoint_profile` → `metrics` of perfect tracking), so they pass once those functions are written. A few checks are added against the Python reference numbers ($H$ and $f$ of the condensed QP, the transcription optimum, every metric of the synthetic run). The expected values above are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `P = load_params()`, `j = P["elbow"]` (dicts), `j["M"]`, `j["lqr"]["x_max"]`, `j["task"]["steps"]` | `lab::Params P = lab::load_params();`, `const lab::JointParams& j = P.elbow;` with the same field names: `j.M`, `j.lqr.x_max` (`Eigen::VectorXd`), `j.mpc.N`, `j.task.steps` (a `std::vector` of `(t_k, theta_k)` pairs) |
| changing a copy: `j = json.loads(json.dumps(E)); j["lqr"]["x_max"][0] = 0.02` | `lab::JointParams j = P.elbow; j.lqr.x_max(0) = 0.02; j.mpc.N = 10;` |
| controller dict `c["Ad"]`, `c["K"]`, `c["u_lim"]` | `lab::Controller c`: `c.Ad`, `c.K`, `c.u_lim` |
| `sim = simulate("mpc", j, g_scale=1.3)`, `sim["theta"]`, `sim["status"]` | `lab::SimResult sim = lab::simulate("mpc", j, std::nullopt, 1.3);` (arguments in order: `kind, j, T, g_scale, task, substeps, ctrl`; `std::nullopt` = `None`), `sim.theta`, `sim.status` |
| `metrics(...)` returns a dict; `m["settle_time"]` may be `nan` | `lab::Metrics m` with fields `m.rom_violation`, …, `m.settle_time` (`std::isnan` when it never settles) |
| tuples `(A, B)`, `(K, S)`, `(Kp, Kd)`, `(K_list, S_list)` | `std::pair`: `.first`, `.second`; lists are `std::vector<Eigen::MatrixXd>` |
| `mpc_qp` / `transcription_qp` return tuples | `lab::CondensedQP{H, f, G, w}` and `lab::TranscriptionQP{H, f, G, w, Aeq, beq}` |
| `u_lim = (lo, hi)`, `x_lim=None` | `lab::Bounds{lo, hi}` (a pair of `Eigen::VectorXd`); `std::nullopt` (the default) for no state limits |
| `mpc_control` returns `(float, status)` for one input | `std::pair<Eigen::VectorXd, std::string>`: the first input as a vector of length $m$ (the joint torque is `.first(0)`) |
| `bryson_weights(0.5, 2.0)` with scalars | vectors of length 1: `lab::bryson_weights(Eigen::VectorXd::Constant(1, 0.5), Eigen::VectorXd::Constant(1, 2.0))` |
| `scipy.linalg.expm`, `scipy.linalg.solve_continuous_are` | `lab::expm`, `lab::solve_continuous_are` (given) |
| `ValueError`, `RuntimeError` | `throw std::invalid_argument(...)`, `throw std::runtime_error(...)` |
| `time.perf_counter()` (Experiment D) | `std::chrono::steady_clock::now()`; time a Release build (`-DCMAKE_BUILD_TYPE=Release`) and say so in the report |
| figures with matplotlib | save each run with `lab::write_sim_csv("run_mpc.csv", sim);` and plot with `python cpp/plot_results.py run_pd.csv run_lqr.csv run_mpc.csv` (Figures 1–2); write the sweeps of Experiments C–D with `sbeg::write_csv` and plot them with a few lines of Python |

**Experiments in C++.** Write `cpp/experiments.cpp` with your own `main()` that runs the experiments and prints the tables (e.g. `const lab::Params P = lab::load_params(); lab::SimResult sim = lab::simulate("pd", P.elbow);`), and add these lines at the end of `cpp/CMakeLists.txt`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Then `cmake --build build` also builds `./build/experiments` (Windows: `build\experiments.exe`). The report, the figures and tables, the reflection and the AI-use note do not change. For your cumulative library, keep your Lab 8 functions in an `optctrl.cpp`/`optctrl.hpp` pair.

A size or index mistake on an Eigen matrix (e.g. a block of `Su` past its last column) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". A `dlqr` loop without the `max_iter` bound can hang the program if your update is wrong: stop it with Ctrl+C.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../exo_joint.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size or index out of range" in `prediction_matrices` / `mpc_qp` | block sizes: block row $k$ of `Sx` is `Sx.block((k - 1) * n, 0, n, n)`, block $(k, i)$ of `Su` is `Su.block((k - 1) * n, i * m, n, m)` | use `n = A.rows()`, `m = B.cols()` and the 0-based offsets |
| `riccati_recursion` test aborts with a failed `REQUIRE` on the list sizes | `S` must have $N + 1$ entries and `K` $N$ | `std::vector<Eigen::MatrixXd> S(N + 1), K(N);` and fill from the back |
| Wrong `H` or `G` after stacking with `<<` | the target was not resized first | `G.resize(rows1 + rows2, cols); G << G1, G2;` |
| `mpc_control` gives a wrong input for the elbow | `U(0)` used for a multi-input system or `U.head(m)` with the wrong `m` | return `U.head(B.cols())` |
| `metrics` overshoot differs on the first segment | `np.sign(0) == 0`: a segment without a step has no overshoot | compute the sign with 0 for a zero difference |
| `metrics` settling time is `nan` / wrong | `std::numeric_limits<double>::quiet_NaN()` only when the last sample of a segment is outside the band | see the definition in `lab.hpp` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
