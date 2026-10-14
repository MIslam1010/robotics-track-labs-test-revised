---
title: "Lab 8 — Impedance control on a Hapkit-type haptic device and a simulated rehabilitation robot (Formal report 2)"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 9"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work (formal report) |
| CLOs | **461-7** (design impedance/admittance controllers for robots in contact with people and assess their safety); 461-9 (formal report 2); 461-6 (stability of a sampled loop, review) |
| Reading | MR §11.7 (pp. 443–447: MR Eqs. 11.62–11.66), §11.8 (pp. 447–449, low-level torque control), §8.6 (pp. 298–299, task-space inertia, MR Eq. 8.90), §8.9.2 (p. 311, apparent rotor inertia); MLS Ch. 4 §6 (pp. 200–206, background on constrained control) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks, 21 tests), `lab/course_models.py` (the shared `hapkit` and `rehab_2r` models, numbers identical to Course resources: Course robot models and hardware safety checklist), `lab/data/device.json`, `lab/data/free_decay_K50.csv`, `…K100.csv`, `…K150.csv` |
| Assessment | Auto-checked code (21 tests) + **formal report 2** (template and rubric below), due one week after the lab, before the Week 10 lab session |
| AI use | Allowed for code **with disclosure** (AI-use note, report appendix A); you must be able to explain every line in an oral check. Derivations in the report are your own |

## Why this lab matters

Weeks 5–6 made the robot follow a reference motion and reject forces, and Week 7 showed why those controllers are stable. A rehabilitation robot, a haptic simulator or a
casualty-handling arm must do the opposite: **let the person push it** and respond like a chosen spring, damper and
mass (MR §11.7). Today you build that behaviour twice:

1. on a **Hapkit-type 1-DOF haptic paddle** (course hardware, Tier 1; the shared `hapkit` model), where you identify
   the device, render springs, dampers and walls at 1 kHz, and find *why* stiff walls go unstable — the sample-and-hold
   and the loop delay act like negative damping;
2. on the **planar rehabilitation robot** of project scenario S1 (the shared `rehab_2r` model, MIT-Manus-type), where
   you compute the inertia the patient feels, render a task-space impedance without a force sensor (MR Eq. 11.65 with
   $M = 0$) and with a force sensor by admittance control (MR Eq. 11.66), and compare.

The same functions are the core of project scenarios S1 (virtual tunnel) and S6 (casualty-handling arm) and of the
required haptic-device hardware runs (scenario sheets HW1-1 … HW1-4 and the S6 hardware task). The project design review
(Week 12) asks Scenario 1 and 6 teams for their impedance parameters and stability evidence: keep the stability map of E4.

## The two devices (illustrative values)

| Device | Model | Numbers |
|---|---|---|
| Haptic paddle | `course_models.hapkit()`: handle $r_h = 0.090$ m from the pivot, sector $r_s = 0.073$ m, motor pulley $r_p = 0.0042$ m (ratio $r_s/r_p = 17.38$), link 0.10 kg with $\mathcal I_{zz} = 1.5\times10^{-4}$ kg m² about its CoM 10 mm below the pivot, rotor $1.0\times10^{-6}$ kg m² | handle coordinate $x = r_h\theta$; $m_h x'' + b_hx' + k_gx = F_\text{motor} + f_\text{hand}$; limits: force ≤ 2 N, range ±40 mm, loop 1 kHz, position resolution 20 µm (`device.json`), $k_t = 0.010$ N m/A |
| Rehabilitation robot | `course_models.rehab_2r()`: horizontal plane, $L_1 = L_2 = 0.35$ m, rods of 2.4 and 1.6 kg | $M(\theta)$ at `q_home` = (0.9, 1.6): $\begin{bmatrix}0.3545 & 0.0628\\ 0.0628 & 0.0657\end{bmatrix}$ kg m²; $\tau_\max = (20, 12)$ N m; therapy path $A = (0, 0.35) \to B = (0, 0.55)$ m |

The sign conventions are in the header of `starter.py`: $F_\text{motor}$ is the force the device applies to the hand,
$f_\text{hand}$ (or $f_\text{ext}$) the force the hand applies to the device. The rehab dynamics are
$M(\theta)\ddot\theta + c(\theta,\dot\theta) = \tau + J^{\mathsf T}f_\text{ext}$, the same as MR Eq. 11.48 with $\mathcal F_\text{tip} = -f_\text{ext}$.

## Learning objectives

By the end of the lab you can:

1. build the 1-DOF model of a haptic device from its link and rotor inertias (MR §8.9.2), and map handle force to motor
   current (MR §11.8);
2. identify the handle mass and friction from free-decay tests, and recognise the effect of the sample-and-hold on the
   identified damping;
3. render springs, dampers and unilateral walls with a sampled loop and measure the rendered stiffness;
4. compute the exact stability limit of a sampled impedance loop with delay (eigenvalues) and explain it with the
   first-order estimate $K_\max \approx (b + B)/(T/2 + dT)$; check passivity with an energy plot;
5. compute the task-space inertia $\Lambda = J^{-\mathsf T}MJ^{-1}$ and the mass felt in a direction, and show its effect on an
   impedance-controlled rehab robot;
6. implement admittance control and explain when it renders the target impedance exactly and why low virtual mass is
   hard (MR p. 447).

# Pre-lab check (complete before the session, 5 questions)

1. Using the parallel-axis theorem, find the inertia of the paddle link about the pivot. Add the rotor inertia
   reflected through the 17.38 ratio and convert to the handle mass $m_h = I/r_h^2$.
2. The link CoM is 10 mm **below** the pivot. Show that gravity acts like a spring at the handle and compute its
   stiffness $k_g = mgr_c/r_h^2$.
3. Which motor current produces a 2 N handle force (no friction)? Is it within $i_\max = 1.5$ A?
4. A spring is rendered at 1 kHz from the position sampled at the start of each period and held for the period.
   On average, how late is the force? With $b_h = 0.15$ N s/m and no virtual damping, estimate the largest stable
   stiffness.
5. MR p. 446 suggests dropping the mass term of Eq. 11.65 ($M = 0$). What does the user then feel as the mass, and why
   is that acceptable for a lightweight backdrivable robot but not for a heavy one?

# Setup

```bash
cd "<course folder>/461/w09/lab"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py   # 18 failed, 3 passed
```

Work in `starter.py` (keep the names and signatures). Provided, do not edit: `load_device`, `load_free_decay`,
`rk4_step`, `rehab_fk`, `rehab_jacobian`, `rehab_jacobian_dot`, `rehab_ik`, `rehab_dynamics`, `simulate_rehab`,
`spectral_radius`, `oscillation_growth`, `pulse_force`, `one_dof_pulse_response`. Units are SI (m, kg, s, N). Run a subset
with `-k`, e.g. `LAB_IMPL=starter $PY -m pytest -q test_lab.py -k free_decay`.

# Procedure and checkpoints

### Checkpoint 1 — The device as a 1-DOF plant (≈ 8 min)

1. `handle_mass(model)`: $m_h = (I_\text{pivot} + G^2J_\text{rotor})/r_h^2$ with $I_\text{pivot}$ from `mr.MassMatrix` at $\theta = 0$ (MR §8.9.2).
2. `gravity_stiffness(model)`: derivative of `mr.GravityForces` at 0, divided by $r_h^2$.
3. `handle_force_to_current(F, model, k_t)`: $\tau = Fr_h$, $\tau_m = \tau/(r_s/r_p)$, $I = \tau_m/k_t$ (MR §11.8, current control of a
   geared motor).
4. `pytest -k "handle_mass or gravity_stiffness or force_to_current"` (the tests use the parallel-axis theorem and
   `modern_robotics`).

**Checkpoint question:** the rotor's own inertia is less than 1 % of the link's, yet it makes up about 65 % of $m_h$. Why?

### Checkpoint 2 — Identify the device from free decays (≈ 10 min)

The files `free_decay_K{50,100,150}.csv` log the handle released from 12 mm while the device renders a spring $K$
with $B = 0$ (a test you will repeat on the hardware).

1. `fit_free_decay(t, x)`: positive peaks (`scipy.signal.find_peaks` with a minimum spacing of half a period), damped
   period from the peak times, logarithmic decrement from a straight-line fit of $\ln(\text{peak})$, then $\zeta$ and
   $\omega_n$ (Week 5 formulas).
2. `identify_device(Ks, fits)`: fit $\omega_n^2 = K/m_h + k_g/m_h$ (slope gives $m_h$, intercept $k_g$), compute the apparent damping
   $b_\text{app} = 2\zeta\omega_nm_h$ for each test and fit $b_\text{app} = b_h - \tau_\text{eff}K$.
3. `pytest -k "free_decay or identify"`.

**Checkpoint question:** the apparent damping falls as $K$ grows. Explain with a Taylor expansion of
$Kx(t - \tau_\text{eff})$, and compare your $\tau_\text{eff}$ with $T/2$.

### Checkpoint 3 — Rendering springs, dampers and walls (≈ 12 min)

1. `spring_damper_force` and `wall_force` (unilateral wall that can only push).
2. `simulate_device(dev, law, T_end, …)`: exactly as the docstring — quantised position, backward-difference
   velocity, saturation to ±2 N, a delay buffer of `delay_steps` samples, zero-order hold, RK4 sub-steps.
3. `static_push_test(dev, K, B, F_load)`: steady deflection under a constant hand force and $K_\text{rendered} = F/x_{ss}$.
4. `pytest -k "spring_damper or simulate_device or static_push"`.

### Checkpoint 4 — Stability of the sampled loop (≈ 10 min)

1. `closed_loop_matrix(m, b, k_g, K, B, T, d)`: ZOH discretisation of the handle (augmented-matrix exponential), the
   control law with backward-difference velocity and $d$ samples of delay; state
   $z_k = [x_k, v_k, x_{k-1}, c_{k-1}, \dots, c_{k-d}]$.
2. `max_stable_K(...)` by bisection on the spectral radius, and `first_order_K_limit(b, B, T, d)`.
3. `device_energy(dev, sim, K)`: $E = \tfrac12m_hv^2 + \tfrac12Kx^2 + \tfrac12k_gx^2$.
4. `pytest -k "closed_loop or max_stable or energy"`. The tests compare your matrix with an independent loop and with
   your simulator, and the eigenvalue verdict with simulated growth.

**Checkpoint question:** adding virtual damping $B$ raises the stiffness limit almost in proportion to $b_h + B$. Why can
you not add damping without limit? (Hint: look at the velocity estimate.)

### Checkpoint 5 — The rehabilitation robot (≈ 10 min in class, finish at home)

1. `task_inertia(q)` ($\Lambda = J^{-\mathsf T}MJ^{-1}$, MR Eq. 8.90) and `apparent_mass(q, u)` ($m_u = 1/(u^{\mathsf T}\Lambda^{-1}u)$).
2. `impedance_control_torque(q, dq, p_d, v_d, K, B)`: $\tau = J^{\mathsf T}(K(p_d - p) + B(v_d - v))$ — MR Eq. 11.65 with the mass and
   arm-dynamics terms dropped (MR p. 446); $g = 0$ in the horizontal plane.
3. `admittance_control_torque(q, dq, f_meas, p0, M_d, B_d, K_d)`: MR Eq. 11.66, then $\ddot\theta_d = J^{-1}(a_d - \dot J\dot\theta)$ and
   inverse dynamics $\tau = M\ddot\theta_d + c - J^{\mathsf T}f_\text{meas}$.
4. `pytest -k "task_inertia or apparent_mass or impedance_control or admittance"`. The tests use `mr.MassMatrix`,
   `mr.JacobianSpace` and `mr.ForwardDynamics`, and the apparent masses 0.70, 0.81, 0.99 kg quoted in the S1 scenario
   sheet.

**Checkpoint question:** with an exact model the admittance controller renders $M_d\ddot x + B_d\dot x + K_dx = f$ exactly. Which
measured quantity makes that possible, and what happens to the rendering if that measurement is noisy or late?

## Experiments for the report

Use the default settings unless stated ($T = 1$ ms, 20 µm resolution, 2 N limit). Save each figure with labelled axes
and units; the report needs Figures 1–5 and Tables 1–2.

* **E1 — Device model.** $m_h$, $k_g$, the current for 2 N and for the 0.49 N push of E3.
* **E2 — Identification (Table 1).** $\omega_n$, $\zeta$ and $b_\text{app}$ for $K$ = 50, 100, 150 N/m; identified $m_h$, $k_g$, $b_h$, $\tau_\text{eff}$.
  Figure 1: one free decay with the fitted envelope; $b_\text{app}$ against $K$ with the fitted line.
* **E3 — Static push (HW1-1 in simulation).** $K$ = 100 and 300 N/m, $B$ = 1 N s/m, loads ±0.49 and ±0.98 N (50 g and
  100 g). Rendered stiffness and error. Why is it slightly above $K$?
* **E4 — Stability map (Table 2, Figure 2).** Exact $K_\max$ for $d$ = 0, 1, 2, 5 samples and $B$ = 0, 0.5, 1.0 N s/m, with the
  first-order estimate. Plot $K_\max$ against total delay $T/2 + dT$ on log axes.
* **E5 — Simulation check (Figure 3).** $d = 2$, $B = 0.5$: $K = 0.9$ and $1.1\,K_\max$, released from 4 mm. Plot $x(t)$ and $E(t)$;
  report the growth ratio and $E(2\ \text{s})/E(0)$. Does saturation occur? Why does saturation not "fix" instability?
* **E6 — Release test (HW1-3 in simulation).** $K = 300$ N/m, $B = 1$ N s/m, released from 5 mm: peak force and the
  peak-to-peak motion after 1 s.
* **E7 — A virtual wall.** Wall at $x = 0$ with $B = 1$ N s/m and $K_w$ = 1000 and 3000 N/m. The hand is a 200 N/m, 0.5 N s/m
  spring–damper whose rest point moves from −5 mm at 50 mm/s and stops at +6 mm; start at $x = -5$ mm. Report the mean
  penetration and the peak-to-peak "buzz" after 1.5 s. Explain the 3000 N/m result with E4.
* **E8 — Rehab robot, rendered stiffness.** At the path middle (0, 0.45) m, stiffness control with $K = 500$ N/m,
  $B = 15$ N s/m (the S1 tunnel values), constant 5 N pushes along $\hat x$ and $\hat y$ (1.5 s).
* **E9 — Apparent mass along the path.** Tangential ($\hat y$) and lateral ($\hat x$) apparent mass at A, middle and B, and the
  eigenvalues of $\Lambda$ at the middle. Figure 4: the inertia ellipses $\{u: u^{\mathsf T}\Lambda^{-1}u\ \text{const}\}$ at the three points.
* **E10 — Impedance versus admittance (Figure 5).** A 10 N lateral push for 20 ms (0.1–0.12 s) at the middle: stiffness
  control ($K = 500$, $B = 15$) and admittance control with $M_d$ = 2, 1, 0.5 kg (same $K$, $B$). Peak lateral and
  along-path deviation, the 1-DOF prediction (`one_dof_pulse_response`) and the direction of the hand velocity at the
  end of the push. Compare with $\Lambda^{-1}\hat x$.
* **E11 — Low virtual mass and a stiff hand (MR p. 447).** The patient grips firmly: hand force
  $f = 2000(p_\text{mid} - p) - 5v$ N plus a 5 N lateral push for 0.1–0.3 s. Admittance control with $M_d$ = 1, 0.3, 0.1 kg:
  peak joint torque and saturation.
* **E12 (optional) — Force-sensor lag.** Repeat E10/E11 with `force_lag_hz=20` in `simulate_rehab`. What changes?

### Checkpoint 6 — Hardware rotation (this week if a supervised Hapkit station is free; otherwise in your team's project hardware session, Weeks 10–13; not needed for the formal report)

On a Hapkit-type device, under TA supervision and with the watchdog of HW1-4 active: (a) free-decay tests with
$K$ = 50 and 100 N/m, analysed with **your** `fit_free_decay`/`identify_device`; (b) the static push HW1-1 with 50 g and
100 g weights; (c) the largest stable $K$ with $B$ = 0.5 N s/m, raising $K$ in steps of 50 N/m from 100 N/m and stopping at
the first sustained buzz. Compare with your simulated map using the identified $m_h$, $b_h$. Never exceed 2 N or ±40 mm;
release the handle only when the TA says so. The project `hardware.py` tools log and compare the runs.

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab / E1 | $I_\text{pivot}$; reflected rotor; $m_h$ | $1.6\times10^{-4}$ kg m²; $3.02\times10^{-4}$ kg m²; 0.05705 kg |
| | $k_g$; current for 2 N; for 0.49 N | 1.211 N/m; 1.036 A; 0.254 A |
| | first-order $K_\max$ ($b_h = 0.15$, $B = 0$, $d = 0$) | 300 N/m |
| E2 | $\omega_n$ (rad/s) at $K$ = 50, 100, 150 | 29.96, 42.12, 51.48 |
| | $\zeta$; $b_\text{app}$ (N s/m) | 0.0366, 0.0208, 0.0128; 0.125, 0.100, 0.075 |
| | identified $m_h$, $k_g$, $b_h$, $\tau_\text{eff}$ | 0.0571 kg, 1.24 N/m, 0.150 N s/m, 0.500 ms |
| E3 | rendered $K$ for 0.49 / 0.98 N (both signs) | 101.0 / 101.1 N/m ($K$ = 100); 300.6 / 301.5 N/m ($K$ = 300) |
| E4 | exact $K_\max$ (N/m), $B$ = 0 / 0.5 / 1.0 N s/m | $d$ = 0: 300.1 / 1289 / 2261; $d$ = 1: 100.1 / 429.4 / 752.3; $d$ = 2: 60.1 / 257.1 / 449.2; $d$ = 5: 27.3 / 116.0 / 200.8 |
| | first-order estimate | $d$ = 0: 300 / 1300 / 2300; $d$ = 1: 100 / 433.3 / 766.7; $d$ = 2: 60 / 260 / 460; $d$ = 5: 27.3 / 118.2 / 209.1 |
| E5 | $K$ (0.9 and 1.1 $K_\max$); growth ratio; $E(2\,\text{s})/E(0)$ | 231.4 and 282.8 N/m; 0.52 and 1.62; 0.095 and 6.3; saturation only in the unstable run |
| E6 | peak force; peak-to-peak after 1 s | 1.5 N; 0.007 mm |
| E7 | mean penetration; buzz (peak to peak) | $K_w$ = 1000: 0.99 mm, 0.00004 mm; $K_w$ = 3000: 0.38 mm, 0.027 mm (sustained) |
| E8 | rendered stiffness along $\hat x$, $\hat y$ | 499 N/m, 499 N/m (10 mm deflection) |
| E9 | tangential / lateral apparent mass at A, middle, B | 0.703 / 1.208; 0.813 / 0.955; 0.988 / 0.792 kg; eigenvalues at the middle 0.535 and 2.447 kg |
| E10 | stiffness control: peak lateral, peak along-path; 1-DOF prediction with $m_x$ | 4.74 mm, 2.37 mm; 5.81 mm |
| | hand-velocity direction at 0.12 s; direction of $\Lambda^{-1}\hat x$ | 30.8°; 34.7° from $\hat x$ |
| | admittance $M_d$ = 2 / 1 / 0.5 kg: peak lateral (1-DOF prediction); along-path; peak torque | 4.58 (4.55) / 5.78 (5.73) / 7.05 (6.96) mm; < 0.001 mm; 1.81 / 2.75 / 10.0 N m |
| E11 | peak torque $M_d$ = 1 / 0.3 / 0.1 kg | 2.00 / 10.24 / 20.0 N m (saturated) |

All values were recomputed by the course team with the reference solution.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 8 into your cumulative library as `interaction.py`. The functions you write (17):

`handle_mass(model: dict)`, `gravity_stiffness(model: dict)`, `handle_force_to_current(F, model: dict, k_t: float)`, `fit_free_decay(t, x, min_fraction: float=0.1)`, `identify_device(Ks, fits)`, `spring_damper_force(x, v, K: float, B: float, x0: float=0.0, v0: float=0.0)`, `wall_force(x, v, K: float, B: float, x_wall: float, side: int=+1)`, `simulate_device(dev: dict, law, T_end: float, x0: float=0.0, v0: float=0.0, f_hand=None, delay_steps: int=0, quantize: bool=True, substeps: int=10)`, `static_push_test(dev: dict, K: float, B: float, F_load: float, T_end: float=2.0)`, `closed_loop_matrix(m: float, b: float, k_g: float, K: float, B: float, T: float, delay_steps: int=0)`, `max_stable_K(m: float, b: float, k_g: float, B: float, T: float, delay_steps: int=0, K_hi: float=100000.0, tol: float=0.01)`, `first_order_K_limit(b: float, B: float, T: float, delay_steps: int=0)`, `device_energy(dev: dict, sim: dict, K: float, x0: float=0.0)`, `task_inertia(q)`, `apparent_mass(q, u)`, `impedance_control_torque(q, dq, p_d, v_d, K, B)`, `admittance_control_torque(q, dq, f_meas, p0, M_d, B_d, K_d)`.

Names shared with other labs:

* The provided `rk4_step(f, x, h)` is the Lab 5 helper ($\dot x = f(x)$), not Lab 3's `rk4_step(f, t, x, h)`.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab08_<ID>.py`, passing `LAB_IMPL=lab08_<ID> pytest -q test_lab.py` (21 tests). Add the
   functions to your `sbeg461_dynamics` library as `interaction.py` (same names); the project uses them.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 2** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–5 and
   Tables 1–2.
4. A reflection (≤ 200 words, appendix B) and an AI-use note (appendix A).

# Formal report 2 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Impedance and admittance
control for safe physical human–robot interaction: a haptic paddle and an upper-limb rehabilitation robot*.
Suggested length in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (1-DOF device model and identification, sampled impedance rendering with stability analysis, task-space impedance and admittance control of a 2R rehab robot), key numbers (identified $m_h$, $b_h$; $K_\max$ at 0 and 2 ms delay; apparent-mass range; admittance accuracy), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why rehabilitation and haptic devices are impedance-controlled (MR §11.7); one defence/medical case where the same ideas matter (casualty-handling arm, S6); objectives; contributions. |
| **II. Devices and models** (½–1 page) | Paddle: link and rotor inertia, transmission, $m_h$, $k_g$, force-to-current chain (MR §8.9.2, §11.8), limits. Rehab robot: PoE/screw description (or reference to `lab/course_models.py`), $M(\theta)$, $\Lambda = J^{-\mathsf T}MJ^{-1}$. State that all values are illustrative. |
| **III. Methods** (1 page) | Free-decay identification and the ZOH correction; the sampled loop and its exact closed-loop matrix; the first-order delay argument; passivity/energy check; impedance law (MR Eq. 11.65, $M = 0$) and admittance law (MR Eq. 11.66 + inverse dynamics). |
| **IV. Verification** (½ page) | Unit tests against `modern_robotics` and independent code; matrix versus simulation agreement; eigenvalue versus simulated growth. |
| **V. Results** (1½ pages) | Tables 1–2 and Figures 1–5 (E2–E11). Every figure is referred to in the text and has a caption that says what to see. |
| **VI. Discussion and safety** (1 page) | Choose $K$, $B$ for the S1 lateral tunnel on the real paddle with your measured $b_h$ and a realistic delay, with a margin; impedance versus admittance for (a) the rehab robot, (b) a geared casualty-handling arm; what the patient feels ($\Lambda$, anisotropy) and whether inertia shaping is worth the risk; limits of the model (linear friction, no Coulomb friction, no cable compliance, perfect torque source — MR §11.8); hardware precautions (force/range limits, watchdog, standards preview for Week 13). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | MR (sections/pages), MLS where used, `modern_robotics` software; IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "no AI tools used"). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clear problem, biomedical and defence/medical contexts motivated, key numbers in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Devices and models | 15 | Correct $m_h$ (with rotor), $k_g$, current chain, $\Lambda$ with MR notation; illustrative values flagged | Minor slips in notation or parameters | Models stated without derivation | Wrong or missing |
| Methods | 15 | Identification with ZOH correction, exact loop matrix, first-order argument, impedance and admittance laws all correctly stated with MR references | One method unclear | Formulas copied without explanation | Wrong or missing |
| Verification evidence | 15 | All tests pass; matrix–simulation and eigenvalue–growth agreement quantified | Tests pass, verification partly quantified | Some tests fail or no independent check | No verification |
| Results (figures/tables) | 20 | Tables 1–2 and Figures 1–5 complete, labelled, with units and informative captions | One item weak | Several items missing | Missing |
| Discussion and safety | 15 | Defended $K$, $B$ choice with margin from the map; impedance vs admittance justified for both robots; model limits and hardware precautions concrete | Choices stated with partial justification | Descriptive only | Missing |
| Writing, format, references | 5 | IEEE format, ≤ 6 pages, concise, correct citations | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every result | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, that a 1 ms loop limits a 0.06 kg paddle to about 300 N/m without
  virtual damping), and why?
* A patient feels about 0.7 kg along the path at A and 1.2 kg across it. Would you correct this, and at what risk?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct (for example a sign error in
  $J^{\mathsf T}f$ or a delay buffer that was off by one sample), and how did you find out it was wrong?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `handle_mass` about 0.020 kg | Rotor inertia forgotten; it is multiplied by the **square** of the ratio (MR §8.9.2). |
| `fit_free_decay` gives $\zeta$ several times too large | Peaks from quantisation noise near the end; keep peaks above 10 % of the first, use a minimum peak spacing. |
| Identified $b_h$ ≈ 0.10 or depends on $K$ | You fitted one test only; the sample-and-hold removes $KT/2$; fit $b_\text{app}$ against $K$. |
| `simulate_device` matrix test fails by one sample | Delay buffer: the force applied on $[t_k, t_{k+1})$ is the command computed at $k - d$; append then pop. |
| Velocity estimate at $k = 0$ is huge | Use $v_\text{est} = 0$ at the first sample (no previous position). |
| `max_stable_K` returns 0 | Spectral radius computed on the wrong matrix size or $x_{k-1}$ row missing; at $K = 0$ the loop is stable. |
| Wall pulls the hand in | Clip the wall force so it can only push (`side * F <= 0`). |
| Rehab robot drifts in E8 | $J^{\mathsf T}F$ written as $JF$ or $J^{-1}F$; the joint torques must be $J^{\mathsf T}$ times the task force. |
| Admittance response not equal to the 1-DOF model | Inverse dynamics must include $-J^{\mathsf T}f_\text{meas}$ and $\dot J\dot\theta$. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside `lab/`, or set `LAB_IMPL` to your file name without `.py`. |

# Pre-lab answers (released after the session)

1. $I_\text{pivot} = 1.5\times10^{-4} + 0.10(0.010)^2 = 1.6\times10^{-4}$ kg m²; rotor $1.0\times10^{-6}\times17.38^2 = 3.02\times10^{-4}$ kg m²;
   $m_h = 4.62\times10^{-4}/0.090^2 = 0.0570$ kg. The rotor is about two-thirds of what the user feels.
2. A small rotation $\theta$ raises the CoM, so gravity gives a restoring torque $mgr_c\sin\theta \approx mgr_c\theta$; with
   $x = r_h\theta$ the force is $(mgr_c/r_h^2)x$: $k_g = 0.10 \times 9.81 \times 0.010/0.090^2 = 1.21$ N/m (about 1 % of a 100 N/m spring).
3. $I = 2 \times 0.090/17.38/0.010 = 1.04$ A $< 1.5$ A.
4. On average half a period, 0.5 ms (zero-order hold). A late spring $Kx(t - \tau)\approx Kx - K\tau\dot x$ removes damping $K\tau$, so
   $K_\max \approx b_h/(T/2) = 0.15/0.0005 = 300$ N/m.
5. The robot's own apparent inertia $\Lambda(\theta)$ (direction- and configuration-dependent). A lightweight backdrivable
   robot has a small $\Lambda$, so the user hardly notices; a heavy robot would feel heavy and anisotropic, and its mass
   cannot be hidden without measuring acceleration or force (inertia shaping or admittance control).

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, deliverables and formal report are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `mass_matrix`, `gravity_forces`, `vel_quadratic_forces`, `jacobian_space`, `forward_dynamics`), with the hand formulas and S1 apparent masses of the Python tests and with the numbers of the Python reference (SciPy matrix exponentials, the independent sampled loop of `test_lab.py`, the random states drawn by `test_lab.py`), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations with the docstrings of `starter.py` and the small structs `HapkitModel`, `RehabModel`, `Device`, `FreeDecayFit`, `Identification`, `DeviceSim`, `PushTest`, `RehabSim` — do not change), `helpers.cpp` (the provided functions of `starter.py` and the two robot models of `course_models.py`, given — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`. The data files are the same `data/device.json` and `data/free_decay_K{50,100,150}.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 8 --cpp                     # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*free_decay*"           # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 21 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 17 as in Python, same order): `handle_mass`, `gravity_stiffness`, `handle_force_to_current` (Checkpoint 1); `fit_free_decay`, `identify_device` (Checkpoint 2); `spring_damper_force`, `wall_force`, `simulate_device`, `static_push_test` (Checkpoint 3); `closed_loop_matrix`, `max_stable_K`, `first_order_K_limit`, `device_energy` (Checkpoint 4); `task_inertia`, `apparent_mass`, `impedance_control_torque`, `admittance_control_torque` (Checkpoint 5). Given in `helpers.cpp` (complete, as the provided functions of `starter.py`): `hapkit`, `rehab_2r`, `rehab` (= `REHAB`), `load_device`, `load_free_decay`, `rk4_step`, `rehab_fk`, `rehab_jacobian`, `rehab_jacobian_dot`, `rehab_ik`, `rehab_dynamics`, `simulate_rehab`, `spectral_radius`, `oscillation_growth`, `pulse_force`, `one_dof_pulse_response`; C++ only: `expm` (= `scipy.linalg.expm`), `find_peaks` (= `scipy.signal.find_peaks(x, distance=...)`, same peaks; only two peaks of exactly equal height closer than `distance` may be resolved differently), `linear_fit` (= `np.polyfit(x, y, 1)`), and the scalar versions of `handle_force_to_current`, `impedance_control_torque` and `admittance_control_torque` (they call yours). As in Python, `handle_mass` and `gravity_stiffness` may call `mr::mass_matrix` and `mr::gravity_forces`; the other functions use Eigen and the helpers.

**Checkpoints** (21 C++ test cases with the Python test names, as the 21 pytest tests). 1: `./build/lab_tests -tc="*handle_mass*,*gravity_stiffness*,*force_to_current*"`; 2: `-tc="*free_decay*,*identify*"`; 3: `-tc="*spring_damper*,*simulate_device*,*static_push*"`; 4: `-tc="*closed_loop*,*max_stable*,*energy*"`; 5: `-tc="*task_inertia*,*apparent_mass*,*impedance_control*,*admittance*"`; finally `./build/lab_tests` must show 21 passed. In Python the three data tests pass with the untouched starter; in C++ each one also checks one function you write — `test_device_file_and_free_decay_data` calls `handle_mass` (the $m_h$ of `device.json`), `test_rehab_dynamics_matches_modern_robotics` calls `task_inertia` (at the ready pose) and `test_rk4_and_jacobian_dot` calls `spring_damper_force` (one RK4 step of the handle on a rendered spring, compared with the matrix exponential) — so they pass once those functions are written.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| model dicts `cm.hapkit()`, `REHAB`; device dict `dev` (`dev["m_h"]`) | `lab::hapkit()`, `lab::rehab()` (structs with `Slist, M, Mlist, Glist, g, ..., params`); `lab::Device` (`dev.m_h`); a modified copy: `lab::Device d = lab::load_device(); d.T = 0.002;` |
| result dicts of `simulate_device`, `simulate_rehab`, `identify_device`, `static_push_test` | `lab::DeviceSim` (`t, x, v, x_meas, v_est, F_cmd, F_applied`: `Eigen::VectorXd`; `saturated`: `std::vector<bool>`), `lab::RehabSim` (`t`; `q, dq, p, v, tau, f`: $(n+1)\times 2$, one row per sample), `lab::Identification`, `lab::PushTest` |
| tuples `(t, x)`, `(wn, zeta)`, `(M, c)` | `std::pair`, `lab::FreeDecayFit`, `std::pair`: `auto [wn, ze] = lab::fit_free_decay(t, x);` |
| call-backs `law(t, x, v)`, `f_hand`, `controller(t, q, dq, f_meas)`, `f_ext(t, p, v)`; `None` | lambdas (`lab::ForceLaw`, `lab::RehabController`, `lab::HandForce`); `None` = an empty `lab::ForceLaw()` / `lab::HandForce()` |
| keyword arguments (`x0=0.004, delay_steps=2`) | all arguments in the order of `lab.hpp`, e.g. `lab::simulate_device(dev, law, 2.0, 0.004, 0.0, lab::ForceLaw(), 2)` |
| `force_lag_hz=None` / `20` in `simulate_rehab` | `0.0` (no lag) / `20.0` (8th argument) |
| `K`, `B`, `M_d`, ... "matrix or scalar" | you write the `Eigen::Matrix2d` version; the scalar version is given; a mix (matrix $K$, scalar $B$): pass `15.0 * Eigen::Matrix2d::Identity()` |
| `scipy.linalg.expm`, `scipy.signal.find_peaks`, `np.polyfit(x, y, 1)` | `lab::expm`, `lab::find_peaks(x, distance)` (indices), `lab::linear_fit(x, y)` → `(slope, intercept)` |
| `round(x / x_res) * x_res`, `np.signbit` | `std::nearbyint(x / x_res) * x_res` (rounds halves to even, as Python), `std::signbit` |
| delay buffer `buf.append(...)`, `buf.pop(0)` | `std::deque<double>`: `push_back`, `front`, `pop_front` |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| plots with Matplotlib | CSV files (`sbeg::write_csv`) plotted with Python or any tool (below) |

A matrix-size mistake (e.g. `closed_loop_matrix` of size $3\times 3$ when $d = 2$) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**Figures and tables for the report.** The plots stay outside the auto-checks. Write the experiments program (E1–E12) in C++ and save its results with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_device()` and `lab::load_free_decay(50)` find the data from any build folder. Example (E5, one run):

```cpp
const lab::Device dev = lab::load_device();
const double Km = lab::max_stable_K(dev.m_h, dev.b_h, dev.k_g, 0.5, dev.T, 2), K = 0.9 * Km;
const lab::DeviceSim sim = lab::simulate_device(
    dev, [K](double, double x, double v) { return lab::spring_damper_force(x, v, K, 0.5); }, 2.0, 0.004, 0.0,
    lab::ForceLaw(), 2);
Eigen::MatrixXd out(sim.t.size(), 3);
out << sim.t, sim.x, lab::device_energy(dev, sim, K);
sbeg::write_csv("e5_K09.csv", {"t", "x", "E"}, out);
```

Deliverable 1 is `cpp/starter.cpp` (renamed `lab08_<ID>.cpp`), deliverable 2 is `experiments.cpp` plus your plotting script; the formal report, rubric and reflection are unchanged. Your cumulative `sbeg461_dynamics` library stays in Python (the project uses it): if you work in C++, also keep `interaction.py` up to date, or agree a C++ library with your project team.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `fit_free_decay` gives a period that is far off | `static_cast<int>(0.5 * T_guess / dt)` computed with integer `dt`, or peaks not filtered (`x > 0`, `>= min_fraction` of the first) | keep everything `double` until the final cast; filter in two passes as in Python |
| `simulate_device` loop-details test fails | the vectors not resized to $n + 1$, or the deque not pre-filled with `delay_steps` zeros | `n = std::llround(T_end / dev.T)`; `std::deque<double> buf(delay_steps, 0.0);` |
| `static_push_test` gives a negative or huge $K_\text{rendered}$ | `law` and `f_hand` swapped, or the hand force passed as `x0` (arguments are positional in C++) | `lab::simulate_device(dev, law, T_end, 0.0, 0.0, [F_load](double, double, double) { return F_load; })` |
| `closed_loop_matrix` wrong only for $d = 0$ | the term $B_d c_k$ added to the wrong rows | for $d = 0$ add `Bd * crow` to the top two rows: `A.topRows(2) += Bd * crow;` |
| A value is exactly 0 or 1 too often | integer division: `1/2 == 0` in C++ | write `0.5`, `T / 2.0` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
