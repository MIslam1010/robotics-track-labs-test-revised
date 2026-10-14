---
title: "Lab 10 — CBF-QP safety filter versus saturation on an exoskeleton joint and on the Hapkit"
subtitle: "SBEG 46X Medical Robotics — Week 11"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 2 h of independent work |
| CLOs | **46X-6** (design and evaluate control-barrier-function safety filters and assist-as-needed control for rehabilitation robots and exoskeletons); **46X-9** (project safety layer: these functions are ported to the project package) |
| Reading | Tedrake, *Underactuated Robotics*, Ch. 9, §9.1.6 *Barrier functions*, Theorem 9.5 (course PDF p. 154); Åström & Murray, *Feedback Systems*, §4.4 (invariant sets, p. 118); *Modern Robotics* §11.4.1, Eq. (11.21) (p. 424), Eq. (11.38) (p. 434), §11.7 (pp. 443–447). This week's lecture notes and Workshop 10 |
| Files (your Lab code folder) | `starter.py` (your code), `test_lab.py` (auto-checks), `course_models.py` (the elbow of `arm_exo_2r`, the knee of `load_exo_2r` and the `hapkit`, copied from Course resources: Course robot models and hardware safety checklist), `data/lab10.json` (parameters), `data/expected.json` (reference metrics) |
| Hardware | **Simulation** is complete and unpenalised. Optional: Experiment C on a course **Hapkit** at the supervised Lab 10 station in Session 4 (2 N saturation and ±40 mm watchdog active; Part "Hardware option") |
| Assessment | Auto-checked code (tests) + a short lab memo (≤ 2 pages: Table 1, Figures 1–3, answers to the questions in the checkpoints) + reflection with AI-use note; due before the Week 12 lab |
| AI use | Allowed for code **with disclosure** (AI-use note); you must be able to explain every line in an oral check. Numbers, plots and conclusions come from your own runs |

## Why this lab matters

A rehabilitation exoskeleton is deliberately **soft**: the assist-as-needed controller of Lecture A lets the patient make
errors, because effort drives recovery. But the surgeon's range-of-motion limit is **hard**: after an elbow or tendon
repair, the joint must never pass it — not under an aggressive therapy command, not during a spasm. In SBEG 461 (Lab 8)
and in Lab 4 you rendered a virtual wall, and in Lab 8 you saw actuator saturation. Both act
only *once the limit is reached*, and a joint with inertia then needs a braking distance it no longer has.

A **control barrier function (CBF) safety filter** sits between any nominal controller and the motor. At every sample it
solves a one-variable quadratic programme: *change the command as little as possible so that the barrier condition
holds*. Far from the limit it does nothing; near it, it brakes in proportion to distance and speed. You will build the
filter, show that it keeps the elbow inside its range where saturation fails, see what it needs to know about the
patient's own torque, and put a forbidden region on the Hapkit. The defence experiment repeats this on the knee of a
soldier load-carriage exoskeleton, where a late limit means a buckled knee under a 30 kg pack.

## Learning objectives

By the end of the lab you can:

1. implement the closed-form one-input CBF-QP and the general CBF-QP (several constraints, bounds, weights);
2. derive second-order (exponential) barriers for a torque-driven joint and a force-driven handle and write them as
   bounds on the input, including a known or bounded human torque/force;
3. combine barrier bounds with actuator (and torque-rate) limits, detect infeasibility and choose a fallback;
4. compare the CBF filter with saturation and with a virtual wall on range violation, tracking and intervention;
5. state the assumptions of a software safety filter (model, measured human torque, actuator authority, sampling) and
   what must back it up (end stops, firmware saturation, watchdog).

# Pre-lab check (complete before the session, 5 questions)

1. For $\dot x = u$ and $h = 2 - x$, $\alpha = 3$ s$^{-1}$, $u_{nom} = 4$: give $u^*$ at $x = 1.5$ and at $x = 0$.
2. Why can a position barrier $h(\theta)$ not be enforced directly with the torque of a joint? What is $\psi_1$?
3. Write the upper torque bound for $M\ddot\theta + b\dot\theta + g(\theta) = \tau + \tau_h$, $h = \theta_{max} - \theta$,
   gains $\alpha_1$, $\alpha_2$.
4. The barrier asks for $\tau \le -8$ N m and the motor can give $|\tau| \le 6$ N m. What does a sensible filter output, and
   what should it report?
5. The dead-band assistance has $\delta = 0.05$ rad, $K = 20$ N m/rad, $B = 0.5$ N m s/rad. Give $\tau_a$ for $e = 0.15$ rad,
   $\dot e = -0.2$ rad/s.

# The models

**Elbow exoskeleton (main experiments, as in Lab 8).** The elbow of the shared model `arm_exo_2r` with the upper arm
hanging, with the patient's torque $\tau_h$ (measured by the cuff's force sensor):

$$M\ddot\theta + b\dot\theta + g(\theta) = \tau + \tau_h + \tau_{stop},\qquad g(\theta) = 3.3354\sin\theta\ \text{N m},\qquad |\tau| \le 6\ \text{N m}.$$

$\tau_{stop}$ is the mechanical end stop at 0 and 2.4 rad (stiff spring–damper). Prescribed range of motion $[0.20, 1.60]$
rad; $M = 0.0778$ kg m², $b = 0.10$ N m s/rad (from `course_models.py`). Controller at 1 kHz (zero-order hold), plant
integrated with 4 sub-steps.

**Load-carriage knee (Experiment D, defence, optional).** The knee of `load_exo_2r` with the hip locked (30 kg pack):
$M = 18.80$ kg m², $b = 2$, $g(\theta) = -239.0\sin\theta - 50.03\cos\theta$ N m, $|\tau| \le 200$ N m, torque rate
$\le 2000$ N m/s, squat range $[0.05, 0.60]$ rad, end stops at 0 and 1.6 rad.

**Hapkit (Experiment C).** Handle coordinates $x = r_h\theta$ (as in the course project):
$m_h\ddot x + b_h\dot x + k_gx = F + F_h$, $m_h = 0.05705$ kg (handle + reflected rotor, from `hapkit_linear()`),
$b_h = 0.15$ N s/m, $k_g = 1.2111$ N/m; firmware at 1 kHz with 20 µm position resolution and a filtered velocity
estimate; $|F| \le 2$ N; watchdog (motor off) outside ±40 mm. A forbidden region starts at $x_{max} = 25$ mm. The
operator's hand (0.10 kg, 2 N s/m, grip stiffness 200 N/m) pushes the handle towards a target beyond the region; the
device cannot measure $F_h$, so the filter uses a bound $|F_h| \le 1.7$ N.

| Barrier (upper limit) | Input bound from $\psi_2 \ge 0$ |
|---|---|
| Joint, $h = \theta_{max} - \theta$ | $\tau \le M[\alpha_1\alpha_2(\theta_{max} - \theta) - (\alpha_1 + \alpha_2)\omega] + b\omega + g(\theta) - \tau_h$ |
| Joint, $h = \theta - \theta_{min}$ | $\tau \ge M[-\alpha_1\alpha_2(\theta - \theta_{min}) - (\alpha_1 + \alpha_2)\omega] + b\omega + g(\theta) - \tau_h$ |
| Hapkit, $h = x_{max} - x$ | $F \le m_h[\alpha_1\alpha_2(x_{max} - x) - (\alpha_1 + \alpha_2)\dot x] + b_h\dot x + k_gx - F_{h,max}$ |

Gains: elbow $\alpha_1 = \alpha_2 = 10$ s$^{-1}$; knee 5 s$^{-1}$; Hapkit 50 s$^{-1}$ (all in `data/lab10.json`).

**Provided code.** `simulate_joint(j, experiment, mode, tau_h_mode)` runs the elbow or knee (`experiment = "step"` or
`"spasm"`; `mode = "none"`, `"sat"` or `"cbf"`; `tau_h_mode = "measured"`, `"zero"` or `"robust"`) and
`simulate_hapkit(hk, case, mode)` runs the handle (`case = "push"` or `"hard_push"`; `mode = "none"`, `"wall"` or `"cbf"`).
Both call **your** functions and return logs of numpy arrays. The baselines are provided: `saturate_toward_limit` (the
course-project "saturation": a command that pushes further out is zeroed once the limit is reached; the `"sat"` mode
also clips the set-point to the range) and `virtual_wall_force` (a 200 N/m one-sided spring at $x_{max}$). `solve_qp` is
the course-project QP helper (SciPy SLSQP, signature `solve_qp(H, f, A_ub, b_ub, lb, ub, x0)`); your Lab 8 solver can
be used instead if you stack the bounds as rows.

# Setup

```bash
cd "<your Lab code folder>"
LAB_IMPL=starter python -m pytest -q test_lab.py     # at the start: 17 tests fail with NotImplementedError, 5 data tests pass
```

Use the course Python environment (or the Docker image / JupyterHub). Work in `starter.py`, keep the function names
and signatures, and rename it `lab10_<ID>.py` when you submit.

# Procedure and checkpoints

### Checkpoint 1 — The one-input filter and the general QP (≈ 12 min)

1. `cbf_filter_1d(u_nom, h, Lf_h, Lg_h, alpha)`: the closed form of Workshop Problem 1 (return `u_nom` if `Lg_h` is
   zero). Check by hand: ankle trainer, $\theta = 0.30$, $u_{nom} = 0.5$, $\alpha = 4$ gives 0.2 rad/s.
2. `filter_interval(u_nom, lo, hi)`: several bounds on one input; report `active` and `feasible`.
3. `cbf_qp(u_nom, A, b, W, u_min, u_max)`: several inputs and barriers, through `solve_qp`.

Tests: `test_cbf_filter_1d_*`, `test_filter_interval`, `test_cbf_qp_*`. **Question 1:** why is the constraint of a
CBF-QP always linear in the input, even for a nonlinear robot?

### Checkpoint 2 — Range barriers on the elbow; saturation versus CBF (≈ 15 min)

4. `rom_torque_bounds(theta, omega, j, alpha1, alpha2, tau_h)` (table above) and
   `joint_safety_filter(tau_nom, cbf_bounds, act_bounds)` (actuator bounds always hold; if the intersection is empty
   keep the actuator-feasible torque closest to the barrier interval and set `feasible = False`).
5. `violation_metrics(t, s, lo, hi)`: largest violation, time outside, extremes.
6. Run the **step experiment**: PD + gravity compensation ($\omega_n = 15$ rad/s, $\zeta = 0.5$, as in Lab 8) to a therapy
   set-point of 1.7 rad — beyond the prescribed 1.6 rad — then back to 0.4 rad at 1.5 s, for `mode = "none"`, `"sat"`,
   `"cbf"`. Plot $\theta(t)$ with the limit (Figure 1) and $\tau(t)$, $\tau_{nom}(t)$ for the CBF run.

Expected (reference solution):

| Step experiment | $\theta_{max}$ (rad) | largest violation (rad) | time outside (s) |
|---|---|---|---|
| none (actuator limit only) | 1.864 | 0.264 | 1.39 |
| sat (set-point clipped + saturation at the limit) | 1.758 | 0.158 | 0.74 |
| cbf | 1.600 | 0.000 | 0 |

The CBF changes the command in about 60 % of the samples of this run (it brakes the approach and then holds the joint at
the limit while the set-point lies outside). **Question 2:** the "sat" run clips the set-point to 1.6 rad, yet overshoots
by 0.16 rad. Explain with the damping ratio and with the braking-distance argument of Workshop Problem 5.

### Checkpoint 3 — Spasm during assist-as-needed therapy: what must the filter know? (≈ 15 min)

7. `aan_torque(e, e_dot, K, B, delta)`: the dead-band assistance of Workshop Problem 3 ($K = 20$, $B = 0.5$,
   $\delta = 0.05$). The nominal controller is gravity compensation + this assistance around the slow reference
   $\theta_d = 0.95 - 0.55\cos(2\pi t/4)$ rad. The patient contributes half of a tracking effort and has a 3 N m flexor
   spasm from 1.5 s to 3.0 s, when the reference is near its 1.5 rad peak.
8. Run `experiment = "spasm"` with `("sat", "measured")`, `("cbf", "measured")`, `("cbf", "zero")` and
   `("cbf", "robust")` (bound $|\tau_h| \le 4$ N m). Figure 2: $\theta(t)$ of the four runs with the limit.

| Spasm experiment | $\theta_{max}$ (rad) | largest violation (rad) | RMS tracking error (rad) |
|---|---|---|---|
| sat | 1.682 | 0.082 | 0.085 |
| cbf, $\tau_h$ measured | 1.599 | 0.000 | 0.071 |
| cbf, $\tau_h$ ignored ("zero") | 1.675 | 0.075 | 0.084 |
| cbf, robust bound 4 N m | 1.482 | 0.000 | 0.128 |

**Question 3:** why does the filter that ignores $\tau_h$ fail, although its equations are "correct"? **Question 4:** the
robust filter never violates the limit but stops 0.12 rad short of it and tracks worse. Where does the conservatism come
from, and what hardware would remove it?

### Checkpoint 4 — Forbidden region on the Hapkit (≈ 8 min in class, finish at home)

9. `hapkit_force_bound(x, v, hk, alpha1, alpha2, F_h_max)` and `hapkit_safety_filter(F_nom, x, v, hk, alpha1, alpha2,
   F_h_max)` (nominal $F_{nom} = 0$: a transparent device).
10. Run `simulate_hapkit` for `case = "push"` and `"hard_push"`, `mode = "none"`, `"wall"`, `"cbf"`. Figure 3: $x(t)$ and
    $F(t)$ of the six runs.

| Hapkit run | $x_{max}$ (mm) | violation (mm) | notes |
|---|---|---|---|
| push, none | 32.8 | 7.8 | the hand reaches its target |
| push, wall (200 N/m) | 29.0 | 4.0 | penetration $\approx F_h/K$ |
| push, cbf | 24.3 | 0.0 | peak motor force 1.68 N |
| hard push, none | 42.7 | 17.7 | **watchdog trips** at 40 mm |
| hard push, wall | 33.8 | 8.8 | |
| hard push, cbf | 33.0 | 8.0 | infeasible in 567 samples: the operator out-pushes the 2 N motor |

**Question 5:** in the "hard push" case no filter keeps the handle out. Which layers of protection remain, and what does
this tell you about the claim "the CBF guarantees safety"?

### Experiment D (optional, defence) — Load-carriage knee with a torque-rate limit

11. Run `simulate_joint(load_params()["knee"], "step", mode)` for the three modes: a squat command to 0.65 rad (beyond the
    0.60 rad squat limit) with PD + gravity compensation ($\omega_n = 4$, $\zeta = 0.5$), torque limit 200 N m and torque
    rate 2000 N m/s. Expected: `none` and `sat` overshoot past the holding limit (0.75 rad, Workshop Problem 2d) and the
    knee **buckles onto the 1.6 rad end stop**; `cbf` stops at 0.600 rad, active in about 76 % of the samples.
    **Question 6:** why can saturation not recover once the knee passes about 0.75 rad?

# Hardware option — the filter on a course Hapkit (supervised Lab 10 station, Session 4)

Only after the simulation passes, with a team member holding the emergency stop and the device's own 2 N saturation and
±40 mm watchdog **left unchanged** (Course resources: Course robot models and hardware safety checklist):

1. Re-identify $m_h$ and $b_h$ of the station's unit (free-decay test) and update the constants.
2. Add the filter to the 1 kHz firmware loop after the position read and velocity filter, **before** the existing force
   saturation:

```c
// h = X_MAX - x; robust second-order CBF bound (illustrative constants; use your identified values)
float F_hi = M_H * (A1 * A2 * (X_MAX - x) - (A1 + A2) * v) + B_H * v + K_G * x - FH_MAX;
float F = F_nom;                       // 0 for a transparent handle
if (F > F_hi) F = F_hi;                // CBF-QP with one input = a clip
// existing firmware: saturation at +-2 N and the +-40 mm watchdog follow unchanged
```

3. Start with $\alpha_1 = \alpha_2 = 20$ s$^{-1}$ and `FH_MAX = 0.5` N; push gently towards 25 mm; then increase to the lab
   values. Log $x$ and $F$ over serial, and compare $x_{max}$ and the engagement distance with the simulation (memo,
   Figure 3b). If the handle buzzes near the boundary, the velocity estimate is too noisy for the gains: lower $\alpha$ or
   the velocity-filter cut-off.

# Your `sbeg46x_medrob` library

<!-- library-block:begin -->
When the tests pass, add the functions you wrote in Lab 10 to a new module `safety.py` of your library. The functions
you write (9):

`cbf_filter_1d(u_nom, h, Lf_h, Lg_h, alpha)`, `filter_interval(u_nom, lo, hi)`, `cbf_qp(u_nom, A, b, W, u_min, u_max)`,
`rom_torque_bounds(theta, omega, j, alpha1, alpha2, tau_h)`, `joint_safety_filter(tau_nom, cbf_bounds, act_bounds)`,
`aan_torque(e, e_dot, K, B, delta)`, `violation_metrics(t, s, lo, hi)`, `hapkit_force_bound(x, v, hk, alpha1, alpha2,
F_h_max)`, `hapkit_safety_filter(F_nom, x, v, hk, alpha1, alpha2, F_h_max)`.

`cbf_filter_1d` and `cbf_qp` have the signatures of the course-project package: port them to `safety.py` of your project
(the project's safety-filter TODOs). Also copy `saturate_toward_limit` (a baseline) and
`solve_qp`; the project already provides a two-sided `virtual_wall_force`. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab10_<ID>.py`, passing `LAB_IMPL=lab10_<ID> python -m pytest -q test_lab.py` (22 tests), and
   the 9 functions added to `safety.py` of your library.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates Table 1 (all runs of Checkpoints 2–4 with
   $\theta_{max}$ or $x_{max}$, largest violation, time outside, RMS tracking error, fraction of samples with the filter
   active, number of infeasible samples) and Figures 1–3.
3. A memo (≤ 2 pages): Table 1, Figures 1–3, answers to Questions 1–5 (and 6 if you did Experiment D), and one paragraph
   on how your project will use a safety filter (constraint, what is measured, the bound you assume, and the fallback
   when the QP is infeasible). Hardware option: Figure 3b and the comparison.
4. Reflection (≤ 150 words) with an AI-use note.

**Marking (100 points):** tests 40; Table 1 and figures, consistent with the expected values 20; answers to Questions 1–5
25; project paragraph 10; reflection and AI note 5. Oral check: a TA may ask you to explain any line (2–3 min); failure
caps the code mark at 50 %.

# Reflection prompts (≤ 150 words)

* The CBF filter that ignored the patient's torque used "correct" equations and still failed. What does that teach you
  about the assumptions of a safety function, and how would you write them in your project's hazard analysis
  (ISO 14971, Week 13)?
* As a therapist, would you accept a robust filter that stops 0.12 rad before the prescribed limit? What would you want
  the device to display when the filter is active or infeasible?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it
  was wrong (a test, a hand calculation, a plot)?

# Safety note

A software barrier holds only for its model, its sensors and the force the actuator can give. On hardware it is an
**additional** layer: the Hapkit firmware saturation (2 N) and watchdog (±40 mm), the course-arm limits and the emergency
stop stay in place and are never disabled for an experiment. Test the exact firmware in simulation first, start with small
gains, and stop at once if the handle buzzes or moves unexpectedly.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `cbf_filter_1d` correct when active, wrong when inactive | Return `u_nom` unchanged when $L_fh + L_gh\,u_{nom} + \alpha h \ge 0$; do not always project. |
| Sign errors in `rom_torque_bounds` | For the upper limit $\ddot h = -\ddot\theta$ and $\dot h = -\omega$; the test checks that the bound makes $\psi_2 = 0$ exactly. |
| The CBF run still violates by a few mrad | You used $\psi_2$ with $g$ but forgot $b\omega$ or $\tau_h$; or $\alpha$ too large for the 1 kHz sampling. |
| `joint_safety_filter` infeasible test fails | When the barrier interval and the actuator interval do not overlap, clip $\tau_{nom}$ to the barrier interval **first**, then to the actuator interval. |
| Robust spasm run violates | The upper bound must use $+\tau_{h,max}$ and the lower bound $-\tau_{h,max}$ (worst case for each). |
| Hapkit CBF run pushes the handle away even when the hand is not pushing | Expected with the robust bound: the filter engages about 12 mm before the boundary (Workshop Problem 4d). |
| `ModuleNotFoundError: starter` | Run `pytest` from inside your Lab code folder, or set `LAB_IMPL` to your file name without `.py`. |

# Pre-lab answers (released after the session)

1. Bound $u \le 3(2 - x)$. At $x = 1.5$: $1.5 < 4$, so $u^* = 1.5$. At $x = 0$: $6 \ge 4$, so $u^* = 4$ (unchanged).
2. $\dot h = -\omega$ does not contain $\tau$ ($L_gh = 0$, relative degree two). $\psi_1 = \dot h + \alpha_1h = -\omega +
   \alpha_1(\theta_{max} - \theta)$; the barrier is applied again to $\psi_1$.
3. $\tau \le M[\alpha_1\alpha_2(\theta_{max} - \theta) - (\alpha_1 + \alpha_2)\omega] + b\omega + g(\theta) - \tau_h$.
4. Output $-6$ N m (the hardest braking available) and flag the sample as infeasible; the guarantee is lost, so a
   higher layer (end stop, watchdog, controlled stop) must act if it persists.
5. $\mathrm{dz}(0.15) = 0.10$; $\tau_a = -20(0.10) - 0.5(-0.2) = -2.0 + 0.1 = -1.9$ N m.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, expected values,
hardware option and deliverables are the same; the auto-checks use the same ground truth as `test_lab.py` (grid search,
KKT enumeration, the barrier condition evaluated on the dynamics, the course models, `data/expected.json`) and the numbers
of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and
documentation — do not change), `helpers.cpp` (the provided functions — do not change), `test_lab.cpp` (auto-checks,
doctest), `cpp_cases.json` (reference numbers and the random test inputs of `test_lab.py`), `CMakeLists.txt`. The
parameters and reference metrics are read from the same `data/lab10.json` and `data/expected.json`. `cpp_support`
(Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up
to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++";
macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with
`pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 10 --cpp                     # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*cbf_filter_1d*"        # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 20 test cases fail with `TODO: <function>`: this is expected.
Rebuild (`cmake --build build`) after every change. All 13 closed-loop runs of the tests take well under a second.

**Functions to write** (in `starter.cpp`, the same 9 as in Python, same order): `cbf_filter_1d`, `filter_interval`,
`cbf_qp` (Part A); `rom_torque_bounds`, `joint_safety_filter`, `aan_torque`, `violation_metrics` (Part B);
`hapkit_force_bound`, `hapkit_safety_filter` (Part C). Given in `helpers.cpp` (complete, as the "Provided" section of
`starter.py`): `load_params`, `gravity`, `step_setpoint`, `therapy_reference`, `smooth_pulse`, `patient_torque`,
`operator_target`, `saturate_toward_limit`, `virtual_wall_force`, `solve_qp`, `end_stop_torque`, `pd_torque`,
`simulate_joint` and `simulate_hapkit` (the simulators call your `rom_torque_bounds`, `joint_safety_filter`,
`aan_torque` and `hapkit_safety_filter`).

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — One-input filter and general QP | `-tc="*baselines*,*solve_qp*,*cbf_filter_1d*,*filter_interval*,*cbf_qp*"` | 8 |
| 2 — Range barriers on the elbow | `-tc="*course_models*,*lab8*,*rom_bounds*,*joint_safety*,*violation*,*elbow_step*"` | 7 |
| 3 — Spasm during assist-as-needed therapy | `-tc="*aan*,*spasm*"` | 2 |
| 4 — Forbidden region on the Hapkit | `-tc="*hapkit*"` | 2 |
| Experiment D (optional) — load-carriage knee | `-tc="*knee_load*"` | 1 |

All 20 test cases pass with `./build/lab_tests`. The Python track counts 22 pytest items: the parametrized
`test_rom_bounds_make_second_order_condition_tight` is one C++ test case that loops over the elbow and the knee, and
`test_starter_does_not_import_solution` has no C++ counterpart (the build compiles only your `starter.cpp`). One more
difference: in Python the data and provided-code tests pass with the untouched starter; in C++ each of them also checks
one of your functions (`test_params_match_course_models` → `rom_torque_bounds` at rest in the middle of the range,
`test_same_joint_as_lab8` → `joint_safety_filter` as a plain actuator saturation, `test_provided_baselines` →
`filter_interval` with a one-sided interval, `test_provided_solve_qp` → `cbf_qp` on the same QP), so they pass once
those functions are written. A few checks are added against the Python reference values ($10^{-9}$): the filtered
inputs of the random `cbf_filter_1d` cases, the torque bounds of the random `rom_torque_bounds` cases and the Hapkit force
bounds. The expected values of the tables above are the same in both tracks (the C++ simulators reproduce the Python runs
to about $10^{-16}$).

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `P = load_params()`; `E = P["elbow"]`, `H = P["hapkit"]` (dicts) | `lab::Params P = lab::load_params();` `P.elbow`, `P.knee` (`lab::JointParams`), `P.hapkit` (`lab::HapkitParams`) with the same field names: `E.M`, `E.rom.first` / `E.rom.second`, `E.cbf.alpha1`, `E.spasm.tau_h_max`, `H.x_max` |
| `hk["operator"]`, `hk["push"]` | `H.op` (`operator` is a C++ keyword), `H.cases.at("push")` |
| `j.get("tau_rate")` (absent for the elbow) | `j.tau_rate == 0.0` means no torque-rate limit |
| tuple `(tau_lo, tau_hi)`, `(u, active, feasible)` | `std::pair<double, double>` (`.first`, `.second`); `lab::IntervalResult{u, active, feasible}` |
| `tau, info = joint_safety_filter(...)`, `info["active"]` | `auto [tau, info] = lab::joint_safety_filter(...)`, `info.active` (`lab::FilterInfo`); return `{tau, lab::FilterInfo{active, feasible}}` |
| dict of `violation_metrics` | `lab::ViolationMetrics` (`max_violation`, `time_outside`, `s_max`, `s_min`) |
| `-np.inf`, `np.inf` | `-lab::INF`, `lab::INF` |
| `cbf_qp(u_nom, A, b, W=None, u_min=None, u_max=None)` | empty Eigen objects for `None`: `lab::cbf_qp(u_nom, A, b)` or `lab::cbf_qp(u_nom, A, b, Eigen::MatrixXd(), u_min, u_max)` |
| `solve_qp` (SciPy SLSQP) returns `(z, info)` | same signature, returns `std::pair<Eigen::VectorXd, lab::QPInfo>`; exact active-set enumeration of the KKT conditions for the small QPs of this lab (up to 8 inputs and 20 constraints), so the result is exact rather than iterative |
| log dicts of `simulate_joint` / `simulate_hapkit` | `lab::JointLog` (`t`, `theta`, `omega`, `tau`, `tau_nom`, `tau_h`, `th_d` as `Eigen::VectorXd`, `active`, `feasible` as `std::vector<bool>`) and `lab::HapkitLog` (`t`, `x`, `v`, `F`, `F_h`, `active`, `feasible`, `tripped`) |
| `math.copysign`, `np.maximum(a, b)` | `std::copysign`, `a.array().max(b.array())` (or a loop) |
| `ValueError` | `throw std::invalid_argument("...")` |
| plots with matplotlib (Figures 1–3) | write CSV files and plot them in Python (below) |

**Experiments and plots in C++.** Write a small `experiments.cpp` with your own `main()` (add it as a second
executable, or build it by hand next to `starter.cpp` and `helpers.cpp`) that runs the simulators, computes Table 1 with
your `violation_metrics` (RMS tracking error: `std::sqrt((log.theta - log.th_d).squaredNorm() / n)`; fraction active:
count the `true` entries of `log.active`) and saves each run with `sbeg::write_csv` from `<sbeg/labio.hpp>`, e.g.
`Eigen::MatrixXd M(log.t.size(), 3); M << log.t, log.theta, log.tau;` and
`sbeg::write_csv("elbow_step_cbf.csv", {"t", "theta", "tau"}, M);`. Plot with a few lines of Python:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt("elbow_step_cbf.csv", delimiter=",", skiprows=1)
plt.plot(d[:, 0], d[:, 1]); plt.axhline(1.6, ls="--"); plt.xlabel("t (s)"); plt.ylabel("theta (rad)"); plt.show()
```

The questions, the hardware option (the firmware snippet is already C) and the memo do not change. For your cumulative
library, keep your Lab 10 functions in a `safety.cpp`/`safety.hpp` pair (with `saturate_toward_limit` and `solve_qp`).

A size or index mistake on an Eigen vector (e.g. `t(1)` on an empty vector) does not crash the test program: it is
reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../lab10.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `cbf_qp` tests fail with "Eigen: wrong matrix size" | the empty default `W` multiplied with `u_nom` | `W.size() == 0` → `Eigen::MatrixXd::Identity(n, n)` |
| `aan_torque` wrong sign for negative errors | `std::abs(e) - delta` without the sign of `e` | `std::copysign(std::max(std::abs(e) - delta, 0.0), e)` |
| `time_outside` off by a factor or zero | integer count not converted, or `dt` from the wrong samples | `static_cast<double>(count) * (t(1) - t(0))` |
| `hapkit_safety_filter` returns a force beyond ±2 N in the infeasible test | the barrier bound applied after the actuator clip | follow `joint_safety_filter`: barrier interval first, actuator interval last |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
