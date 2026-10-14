---
title: "Lab 3 — Forward-dynamics simulator and integrator comparison (Formal report 1)"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 3"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work for the report |
| CLOs | **461-2** (implement and verify recursive dynamics and forward-dynamics simulation with appropriate numerical integration); 461-1 (structure and properties of $M$, $c$, $g$); **461-9** (technical report) |
| Reading | MR §8.4–8.5 (printed pp. 293–298) and MR Exercises 8.13 and 8.15 (pp. 325–326); Featherstone §6.2 (pp. 104–108), §7.3 (pp. 128–131); Corke §9.3 (pp. 202–204) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks), `lab/data/robots.json`, `lab/data/ur5_fall_reference.csv` |
| Assessment | Auto-checked code (tests) + **Formal report 1** (IEEE two-column, ≤ 6 pages, rubric below), due one week after the lab, before the Week 4 lab session |
| AI use | Allowed for code **with disclosure** (AI-use note in the report appendix); you must be able to explain every line in an oral check |

## Why this lab matters

Inverse dynamics (Lab 2) tells you the torques needed for a motion. **Forward dynamics** answers the opposite question
— *what will the robot do* under given torques — and it is the heart of every simulator you will use in this course
(MR Studio, MuJoCo, Gazebo). Two cases drive the lab:

* **Biomedical — simulating a prosthetic swing.** A transfemoral prosthesis user's leg swings under hip torque and a
  passive (or damped) prosthetic knee. Designers of microprocessor knees simulate the swing to tune damping before any
  patient trial. Model `leg_2r` (illustrative values, as in Workshop Problem 1).
* **Defence — EOD manipulator power loss.** A 6-DOF arm on an explosive-ordnance-disposal robot loses power while
  holding a suspect object. How far does the tool tip fall before a brake engages? This is MR Exercise 8.15(b) (the
  UR5 falling under gravity from its zero configuration), with the UR5 as a stand-in for a 6-DOF EOD arm (model `ur5`,
  MR parameters).

A simulator is only useful if you can trust it, so the second half of the lab is about **verification**: comparing
integrators, measuring their order of accuracy, and using **energy** as a built-in check.

## Learning objectives

By the end of the lab you can:

1. build $M(\theta)$ with $n$ inverse-dynamics calls and with MR Eq. 8.57, and explain why they agree;
2. implement forward dynamics (MR Eq. 8.81) and simulate a robot with explicit Euler (MR p. 297), semi-implicit Euler
   and RK4;
3. verify a simulator against `modern_robotics`, a high-accuracy reference and the energy/power balance
   $\dot E = \dot\theta^{\mathsf T}\tau$;
4. measure an integrator's order of accuracy and choose a time step from an accuracy/cost trade-off;
5. report the results in a short IEEE-style technical report.

# Pre-lab check (complete before the session, 5 questions)

1. Which inputs of `mr.InverseDynamics` give the third column of $M(\theta)$ for a 6R arm?
2. How many inverse-dynamics calls does one `mr.ForwardDynamics` evaluation make for a 6R arm?
3. Write explicit Euler for $x = (\theta, \dot\theta)$ in terms of `ForwardDynamics`.
4. If $\tau = 0$ and there is no friction, what must happen to $E = K + P$ during a simulation? What should happen when
   a hip torque $\tau_1(t)$ is applied?
5. An integrator has global error $e \approx Ch^p$. Your errors are $8\times10^{-3}$ at $h = 0.02$ s and
   $5\times10^{-4}$ at $h = 0.01$ s. What is $p$?

# Setup

```bash
cd "<course folder>/461/w03/lab"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py        # all 25 tests fail with NotImplementedError at the start
```

Work in `starter.py` (keep the function names and signatures). `load_robot(name)` is already written and returns
`(Mlist, Glist, Slist, g)` in `modern_robotics` format: CoM frames $\{i\}$, spatial inertias $\mathcal G_i$ with the
angular block first (MR Eq. 8.32), space-frame screw axes $\mathcal S_i = (\omega_i, v_i)$. You may plug in your own
Lab 2 RNEA anywhere `mr.InverseDynamics` is used (`mass_matrix_rnea` has an `rnea=` argument for that).

# Procedure and checkpoints

### Checkpoint 1 — The mass matrix two ways (≈ 10 min)

1. `mass_matrix_rnea`: $n$ calls with $\dot\theta = 0$, $\ddot\theta = e_j$, $g = 0$, $\mathcal F_{\text{tip}} = 0$
   (MR p. 296).
2. `link_body_jacobians` and `mass_matrix_closed_form`: $M = \sum_i J_{ib}^{\mathsf T}\mathcal G_iJ_{ib}$ (MR Eq. 8.57).
   The screw axes of joints $1..i$ expressed in $\{i\}$ at home are $\mathcal B = [\mathrm{Ad}_{M_{0i}^{-1}}]\mathcal S$,
   with $M_{0i} = M_{01}M_{12}\cdots M_{i-1,i}$.
3. Run `pytest -q test_lab.py -k mass`. Eight tests check both functions against `mr.MassMatrix` on all three robots,
   check that exactly $n$ RNEA calls are made, and check the 2R mass matrix of MR Eq. 8.10 for the MR 2R arm.

**Checkpoint question:** at $\theta = 0$ which UR5 diagonal entry is largest and why?

### Checkpoint 2 — Forward dynamics and energy (≈ 10 min)

1. `forward_dynamics`: one extra call $\texttt{ID}(\theta, \dot\theta, 0, g, \mathcal F_{\text{tip}})$ returns
   $h + J^{\mathsf T}\mathcal F_{\text{tip}}$; solve $M\ddot\theta = \tau - (\cdot)$ with `np.linalg.solve` (never invert $M$).
2. `com_positions` and `energy`: $K = \tfrac12\dot\theta^{\mathsf T}M\dot\theta$, $P = -\sum_i m_i\,g\cdot p_{ci}$
   with $m_i$ = `Glist[i][3, 3]`.
3. `pytest -k "forward or energy"`. The energy test checks $\partial P/\partial\theta = g(\theta)$ against
   `mr.GravityForces` — an independent check of your CoM kinematics.

### Checkpoint 3 — Three integrators and the simulator (≈ 15 min)

1. `euler_step`, `semi_implicit_euler_step`, `rk4_step` for $\dot x = f(t, x)$, $x = (\theta, \dot\theta)$.
2. `simulate(robot, theta0, dtheta0, tf, h, method, tau_fn, Ftip_fn)`; torques may depend on the state,
   `tau_fn(t, theta, dtheta)` (needed for a damper or a controller).
3. `pytest -k "step or simulate"`: your explicit Euler must reproduce `mr.ForwardDynamicsTrajectory` exactly.

### Checkpoint 4 — Verification experiments (rest of the session, then at home)

Run the remaining tests (`pytest -q`, all 25 must pass), then do the experiments below for the report. Save every
figure with axis labels and units.

**Experiment A — prosthetic swing (biomedical).**

* A1. Free swing of `leg_2r` from $\theta_0 = (-60^\circ, -30^\circ)$, $\dot\theta_0 = 0$, for 0.4 s with the three
  integrators and $h \in \{0.02, 0.01, 0.005, 0.0025\}$ s. Compute the error at $t = 0.4$ s against
  `scipy.integrate.solve_ivp(..., method="DOP853", rtol=1e-12, atol=1e-12)`. **Figure 1:** log–log error vs $h$
  with your fitted orders (`observed_order`). **Figure 2:** error vs number of forward-dynamics evaluations
  (Euler 1, RK4 4 per step) — this is the fair cost comparison.
* A2. Add a knee damper $\tau_2 = -b\dot\theta_2$ ($b = 0.6$ N·m·s/rad) and a hip torque
  $\tau_1(t) = 8\sin(2\pi t)$ N·m. Check the power balance $E(t) - E(0) = \int_0^t \dot\theta^{\mathsf T}\tau\,dt$
  (trapezoid rule). **Figure 3:** knee angle with and without damper; energy in, energy dissipated.
  *Note:* the model has no knee extension stop, so the knee can hyperextend; say what a real prosthesis adds.

**Experiment B — EOD arm power loss (defence; MR Exercise 8.15).**

* B1. Simulate the UR5 falling from $\theta = 0$ with $\tau = 0$ for 3 s with $h = 0.01$ s (MR asks for at least 100
  steps per second) with all three integrators. Compare with `data/ur5_fall_reference.csv`. **Figure 4:** total
  energy vs time for the three methods; **Table 1:** max $|\Delta E|$ and $\max_k\lVert\theta_k - \theta_{\text{ref}}\rVert_\infty$.
* B2. Using `mr.FKinSpace` with $M = M_{01}\cdots M_{67}$, compute how far the tool tip falls in 0.25 s and 0.5 s.
  If a brake needs 150 ms to engage, how far has the tip fallen? One paragraph: implications for a robot holding a
  suspect object (protective use only: this is a safety analysis).
* B3. Timing: average wall-clock time of `mass_matrix_rnea` vs `mass_matrix_closed_form` and of one
  `forward_dynamics` call for the UR5. Relate to the $O(n^2)$ / $O(n^3)$ / $O(n)$ discussion of Lecture B (ABA).

**Expected values (to check yourself; TAs use them for calibration).**

| Quantity | Expected |
|---|---|
| UR5 total energy at $\theta = 0$, rest ($P = 0$ at $\{s\}$ origin) | 14.689 J |
| UR5 fall, $h = 0.01$ s, 3 s: max $\lvert\Delta E\rvert$ — Euler / semi-implicit / RK4 | ≈ 23.5 J / ≈ 9.6 J / ≈ $5\times10^{-5}$ J |
| UR5 fall, $h = 0.01$ s: $\lVert\theta - \theta_{\text{ref}}\rVert_\infty$ at 3 s — Euler / semi-implicit / RK4 | ≈ 2.2 rad / ≈ 0.35 rad / ≈ $2\times10^{-5}$ rad |
| UR5 tool-tip drop at 0.15 s / 0.25 s / 0.5 s | ≈ 0.11 m / ≈ 0.31 m / ≈ 0.77 m |
| Leg free swing (A1), $h = 0.01$ s: $E(0.4) - E(0)$ — Euler / semi-implicit / RK4 | ≈ +0.57 J / ≈ −0.003 J / ≈ $-3\times10^{-7}$ J |
| Observed orders (A1) | ≈ 1 (Euler), ≈ 1 (semi-implicit), ≈ 4 (RK4) |

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 3 into your cumulative library as `forward_dynamics.py`. The functions you write (11):

`mass_matrix_rnea(theta, Mlist, Glist, Slist, rnea: Callable=mr.InverseDynamics)`, `link_body_jacobians(theta, Mlist, Slist)`, `mass_matrix_closed_form(theta, Mlist, Glist, Slist)`, `forward_dynamics(theta, dtheta, tau, g, Ftip, Mlist, Glist, Slist)`, `com_positions(theta, Mlist, Slist)`, `energy(theta, dtheta, g, Mlist, Glist, Slist)`, `euler_step(f, t, x, h)`, `semi_implicit_euler_step(f, t, x, h)`, `rk4_step(f, t, x, h)`, `simulate(robot, theta0, dtheta0, tf, h, method='rk4', tau_fn=None, Ftip_fn=None)`, `observed_order(hs, errors)`.

Names shared with other labs:

* `rk4_step(f, t, x, h)` here integrates a time-dependent $\dot x = f(t, x)$. Labs 5–8 and hardware rotation 1 provide a different helper `rk4_step(f, x, h)` for $\dot x = f(x)$; it lives in other modules and is not a replacement for this one.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab3_<ID>.py`, passing `LAB_IMPL=lab3_<ID> pytest -q test_lab.py` (auto-checked, 25 tests).
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 1** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–4 and Table 1.
4. A reflection (≤ 200 words, report appendix B) and an AI-use note (appendix A).

# Formal report 1 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Verification of a
forward-dynamics simulator for a prosthetic leg and an EOD manipulator*. Suggested length per section in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (MR forward dynamics, three integrators), key numbers (orders, energy drift, chosen $h$), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why forward dynamics matters for prosthesis design and for EOD-arm safety; objectives of the study; contributions. |
| **II. Models** (½–1 page) | The two robots (figure or table of parameters; state that `leg_2r` uses illustrative values); dynamics $\tau = M\ddot\theta + h + J^{\mathsf T}\mathcal F_{\text{tip}}$ (MR Eq. 8.77); how $M$ and $h$ are computed ($n + 1$ ID calls; MR Eq. 8.57). |
| **III. Numerical methods** (½–1 page) | Explicit Euler, semi-implicit Euler, RK4 (equations); error order; energy/power balance $\dot E = \dot\theta^{\mathsf T}\tau$ and why it holds (skew-symmetry of $\dot M - 2C$). |
| **IV. Verification** (½ page) | Unit tests vs `modern_robotics`; Euler vs `ForwardDynamicsTrajectory`; reference solutions (method, tolerances). |
| **V. Results** (1½ pages) | Figures 1–4, Table 1, tip-drop numbers, timing. Every figure is referred to in the text and has a caption that states what to see. |
| **VI. Discussion** (1 page) | Which integrator and step size would you use for (a) offline prosthesis design, (b) a 1 kHz real-time simulator? Limits of the models (no friction, no knee stop, rigid links); what the energy check can and cannot prove; safety implication of B2. |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | MR (with sections/pages), Featherstone, `modern_robotics` software; IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "no AI tools used"). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clear problem, both applications motivated, objectives measurable, key numbers in abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Models and theory | 15 | Correct MR equations with numbers, $n + 1$ ID calls explained, parameters tabulated | Minor notation slips | Equations copied without explanation | Wrong or missing |
| Numerical methods | 15 | All three integrators, error orders, stability and energy balance correctly stated | One method or property unclear | Formulas only | Wrong or missing |
| Verification evidence | 15 | All tests pass; Euler = MR trajectory; reference comparison and energy check quantified | Tests pass, comparison partly quantified | Some tests fail or no reference | No verification |
| Results (figures/tables) | 20 | Figures 1–4 and Table 1 complete, labelled, with units, captions and correct values | One figure weak or unlabelled | Several figures missing or unreadable | Missing |
| Discussion and safety | 15 | Justified choice of integrator/time step for both uses; model limits; tip-drop safety implication; energy-check limits | Choice stated with partial justification | Descriptive only | Missing |
| Writing, format, references | 5 | IEEE format, ≤ 6 pages, concise, correct citations | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every result | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, how fast explicit Euler gains energy), and why?
* Where did a test catch a bug in your code? What would have happened in the report without it?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it
  was wrong?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `mass_matrix_rnea` has $g(\theta)$ added to every column | You passed the robot's `g` instead of `np.zeros(3)` in the column calls. |
| Closed-form $M$ wrong for the UR5 only | The screw axes must be expressed in $\{i\}$: $\mathcal B = [\mathrm{Ad}_{M_{0i}^{-1}}]\mathcal S$, using $M_{0i}$ (product up to $i$), not the end-effector $M$. |
| Euler test differs from `ForwardDynamicsTrajectory` after the first step | You updated $\theta$ with the *new* velocity (that is semi-implicit Euler), or used `tau_fn(t + h)`. |
| RK4 order comes out ≈ 1 | `k2`/`k3` evaluated at the wrong state or time, or the reference is not accurate enough (tighten `rtol`). |
| Energy test fails by a constant | The potential reference does not matter, but the sign does: $P = -\sum m_i\,g\cdot p_{ci}$ with $g = (0, 0, -9.81)$. |
| UR5 simulation very slow | Expect about 2 s per simulated second for RK4 with `modern_robotics`; use $h = 0.01$ s, and do not recompute energy inside the integration loop. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside `lab/`, or set `LAB_IMPL` to your file name without `.py`. |

# Pre-lab answers (released after the session)

1. `InverseDynamics(theta, zeros(6), e3, zeros(3), zeros(6), Mlist, Glist, Slist)` with $e_3 = (0,0,1,0,0,0)$.
2. Seven: six for $M$ and one for $h$ (MR p. 296).
3. $\theta_{k+1} = \theta_k + h\dot\theta_k$, $\dot\theta_{k+1} = \dot\theta_k + h\,\texttt{ForwardDynamics}(\theta_k, \dot\theta_k, \tau_k, \mathcal F_k)$.
4. $E$ must stay constant; with $\tau_1(t)$, $\dot E = \dot\theta_1\tau_1$, so $E(t) - E(0) = \int\dot\theta_1\tau_1\,dt$.
5. $p = \log_2(8\times10^{-3}/5\times10^{-4}) = \log_2 16 = 4$.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, deliverables and formal report are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `mass_matrix`, `gravity_forces`, `forward_dynamics`, `forward_dynamics_trajectory`), with MR Eq. 8.10, with the MR Exercise 8.15(b) reference in `data/ur5_fall_reference.csv` and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers, the random states drawn by `test_lab.py` and the high-accuracy DOP853 swing reference), `CMakeLists.txt`. The data files are the same `data/robots.json` and `data/ur5_fall_reference.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 3 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*mass*"                 # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 17 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 11 functions and order as in Python): `mass_matrix_rnea`, `link_body_jacobians`, `mass_matrix_closed_form` (Checkpoint 1); `forward_dynamics`, `com_positions`, `energy` (Checkpoint 2); `euler_step`, `semi_implicit_euler_step`, `rk4_step`, `simulate` (Checkpoint 3); `observed_order` (Checkpoint 4). Given and complete: `load_robot` and `stepper` (the C++ form of the Python `STEPPERS` dictionary). As in Python you may use `mr::inverse_dynamics`, `mr::adjoint`, `mr::trans_inv`, `mr::jacobian_body`, `mr::matrix_exp6` and `mr::vec_to_se3`, and plug in your own Lab 2 RNEA through the `rnea` argument of `mass_matrix_rnea`; do not call `mr::mass_matrix` or `mr::forward_dynamics` inside the functions you write (the tests compare with them).

**Checkpoints** (17 C++ test cases with the Python test names; the Python track counts 25 because pytest counts each robot of a parametrized test separately, the C++ test case loops over the three robots). 1: `./build/lab_tests -tc="*mass*"` (4 test cases); 2: `-tc="*forward_dynamics_matches*,*energy_kinetic*,*energy_values*"` (3); 3: `-tc="*step*,*simulate*"` (5); 4: `./build/lab_tests` must show 17 passed.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `load_robot(name)` returns `(Mlist, Glist, Slist, g)` | returns `lab::Robot{Mlist, Glist, Slist, g}`: `mr::TransformList` ($n+1$ `Eigen::Matrix4d`), `mr::InertiaList` ($n$ $6\times6$), `Eigen::MatrixXd` $6\times n$ (one screw axis per **column**), `Eigen::Vector3d` |
| `theta`, `tau` arrays; `Ftip` 6-vector | `Eigen::VectorXd`; `mr::Vector6d` |
| callables `rnea`, `f(t, x)`, `tau_fn(t, theta, dtheta)`, `Ftip_fn(...)`; `None` | `lab::RneaFunction`, `lab::StateFunction`, `lab::TorqueFunction`, `lab::WrenchFunction` (`std::function`; pass a lambda); an empty function `{}` means zero torque / zero wrench (test it with `if (tau_fn)`) |
| `STEPPERS[method]`; unknown name raises `KeyError` | `lab::stepper(method)` (given); unknown name throws `std::invalid_argument` |
| `energy` returns the tuple `(K, P)` | `std::pair<double, double>`: `auto [K, P] = lab::energy(...);` |
| `link_body_jacobians` returns a list; `com_positions` an `(n, 3)` array | `std::vector<Eigen::MatrixXd>`; an $n\times3$ `Eigen::MatrixXd` (row $i$ = CoM of link $i$) |
| `simulate` returns `(t, theta, dtheta)` | struct `lab::SimResult{t, theta, dtheta}`; `theta`, `dtheta` are $(N+1)\times n$ (row $k$ = step $k$) |
| `np.linalg.solve(M, b)`, `np.r_[a, b]`, `x[:n]`, `x[n:]` | `M.partialPivLu().solve(b)`, `VectorXd x(2 * n); x << a, b;`, `x.head(n)`, `x.tail(n)` |
| `np.polyfit(log h, log e, 1)` | least-squares slope written out: $p = \sum \Delta x\,\Delta y / \sum \Delta x^2$ |
| `solve_ivp(..., "DOP853")` reference (test and Experiment A1) | the test reads the DOP853 result from `cpp_cases.json`; for Experiment A1 use your `rk4` with $h = 10^{-4}$ s as the reference (it agrees with DOP853 to about $10^{-12}$ rad), or compute it in Python |
| `mr.FKinSpace` (B2), `time.perf_counter` (B3) | `mr::fkin_space`; `std::chrono::steady_clock`. C++ timings are much shorter than Python ones: say in the report which language you timed |
| random states of `np.random.default_rng(461)` | stored in `cpp_cases.json` (same numbers as the Python tests) |

**Figures and tables for the report.** The plots stay outside the auto-checks. Write the experiments program in C++ and save its results with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab()`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_robot("ur5")` and `sbeg::read_csv(sbeg::lab_path("../data/ur5_fall_reference.csv"))` find the data from any build folder. Deliverable 2 is then `experiments.cpp` plus your plotting script; the formal report, rubric and reflection are unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../robots.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size or index out of range" | e.g. `J.leftCols(i + 1)` assigned a Jacobian of another width, or a row of `X` mixed up with a column | check the sizes; the test program does not crash, the test simply fails |
| "one RNEA call per column" in `test_mass_matrix_rnea_uses_exactly_n_calls` | `mass_matrix_rnea` calls `mr::inverse_dynamics` directly | call the `rnea` argument |
| `simulate` test stops at a shape check such as `REQUIRE(s.t.size() == N + 1)` or `REQUIRE(s.theta.rows() == N + 1)` | the initial state is not stored as row 0, or $N$ is not `std::llround(tf / h)` | return $N+1$ rows, row 0 = $(\theta_0, \dot\theta_0)$ |
| Results wrong only in some entries, no compile error | an Eigen product stored in `auto` (`auto C = A * B;` keeps references to temporaries) | write the type: `Eigen::MatrixXd C = A * B;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
