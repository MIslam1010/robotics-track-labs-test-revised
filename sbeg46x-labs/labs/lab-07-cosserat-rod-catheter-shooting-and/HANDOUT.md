---
title: "Lab 7 — Cosserat-rod catheter: shooting and BVP solvers, CC vs Cosserat under tip loads, tip-force estimation"
subtitle: "SBEG 46X Medical Robotics — Week 7"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class (simulation, all groups) + about 2 h of independent work |
| CLOs | **46X-5** (model continuum and soft medical robots — constant-curvature kinematics, Cosserat-rod statics — and use the models for tip control and force estimation); 46X-1 (review: kinematics as exponentials) |
| Reading | MR §3.3.2, p. 96 (body twist $T^{-1}\dot T = [\mathcal V_b]$, MR Eq. 3.71 — here with arc length $s$ in place of time); MR §3.3.3, pp. 102–103 (MR Proposition 3.25, Eq. 3.88); MR §3.4, p. 107 (wrenches, MR Eq. 3.93); MR §5.2, p. 188 (statics, MR Eq. 5.26). SciPy documentation of `scipy.integrate.solve_bvp` and `scipy.optimize.least_squares`. Not in the course library (†): Rucker, Jones & Webster, *IEEE Trans. Robot.* 26(5), 2010 (geometrically exact rod model); Webster & Jones, *IJRR* 29(13), 2010 (CC review, Week 6) |
| Files (your Lab code folder) | `starter.py` (your code: 17 functions marked TODO A1 … D5), `test_lab.py` (auto-checks, 24 tests), `data/catheter.json` (catheter and sensor parameters), `data/catheter_log.csv` (bench log: tendon tension, EM-sensor tip pose, reference load-cell force) |
| Library | The finished functions extend module `continuum` of your course library `sbeg46x_medrob` (started in Lab 6 with CC kinematics) |
| Assessment | Auto-checked code (24 tests) + results table + three plots + reflection (one auto-checked lab in the Labs component; not a formal report) |
| AI use | Allowed for code **with disclosure** (AI-use note in the reflection); you must be able to explain every line in an oral check. Derivations are your own |

## Why this lab matters

In Lab 6 the catheter was a constant-curvature (CC) arc: fast, simple and exact — **as long as nothing touches the
tip**. In the heart, the tip presses on the atrial wall; in a confined-space inspection the arm sags under its camera.
Lecture A this week replaced the arc by the **Cosserat rod**: an ODE in arc length for position, angle, internal force
and moment, with conditions at **both** ends (a boundary-value problem). Today you solve it two ways (shooting and
collocation), check it against beams you know, measure how wrong the CC and Euler–Bernoulli models become under
contact, and then turn the model around: from the measured tip pose, **estimate the contact force** — the quantity
an ablation catheter needs to make a good lesion without perforating the wall. The same functions serve project strand
(c) (continuum or soft-robot modelling) and the optional tendon-driven catheter scenario.

## The catheter (illustrative values)

| Quantity | Value |
|---|---|
| Bending section length $L$ | 100 mm, base clamped along $+x$ |
| Bending stiffness $EI$ | $1.0\times10^{-3}$ N m² |
| Tendon offset $r$, nominal tension | 1.2 mm on the $+y$ side; 6 N (≡ tip moment $r\tau = 7.2$ mN m, $\kappa = 7.2$ m⁻¹) |
| EM sensor at the tip | position noise $\sigma_p = 0.1$ mm, angle noise $\sigma_\theta = 0.5°$ |
| Contact force | $F$ applied **to** the tip, base frame; a tissue wall pushes the tip back against its bend, direction `contact_direction(θ)` $= (\sin\theta, -\cos\theta)$ |

Rod equations (state $z = (x, y, \theta, n_x, n_y, m)$): $x' = \cos\theta$, $y' = \sin\theta$, $\theta' = m/EI + \kappa_0$,
$n' = -f$, $m' = -(\cos\theta\,n_y - \sin\theta\,n_x)$; $x(0) = y(0) = \theta(0) = 0$, $n(L) = F$, $m(L) = M + r\tau$.

## Learning objectives

By the end of the lab you can:

1. Implement the planar Cosserat-rod equations and solve the statics BVP by **shooting** (`solve_ivp` + root finding)
   and by **collocation** (`solve_bvp`).
2. **Validate** a rod solver against the CC arc (pure moment), Euler–Bernoulli beams (small loads) and the
   large-deflection elastica (closed form).
3. Quantify the error of the CC and Euler–Bernoulli models of a tendon-driven catheter under tip contact.
4. Estimate the tip contact force from a noisy EM-sensor pose (linear compliance and nonlinear least squares), and
   explain why some force directions are poorly observable.

# Pre-lab check (5 questions, submit before the session)

1. The catheter has $\tau = 6$ N and no contact. What are $\kappa$, the tip angle and the tip position?
2. Which three quantities are known at $s = 0$, which three at $s = L$? What does the shooting method guess?
3. A straight catheter carries a transverse tip force of 0.01 N. Give the Euler–Bernoulli tip deflection and angle.
4. Why is the internal force $n(s)$ constant when only tip loads act, and how does that reduce shooting to one unknown?
5. An EM sensor measures the tip angle with $\sigma_\theta = 0.5°$, and the catheter's tip-angle compliance to a normal
   force is 4.79 rad/N. What force uncertainty results?

# Setup (5 min)

1. Copy your Lab code folder; work in `starter.py` or a copy with your own name (e.g. `continuum_rod.py`).
2. Run `LAB_IMPL=starter python -m pytest -q test_lab.py`: 2 data tests pass and 22 fail with `NotImplementedError`.
   Re-run after every function; the test names tell you which function is being checked.
3. Import your module in a notebook or script (`import starter as lab`) and load the data with `lab.load_catheter()`
   and `lab.load_csv("catheter_log.csv")`.

# Procedure

### Checkpoint 1 — CC and PCC recap (A1–A5, ≈ 7 min)

Implement `cc_tip`, `tendon_curvature`, `cc_transform`, `pcc_tip` and `pneumatic_curvature`.

* E1. Tip pose of the catheter for $\tau$ = 0, 3, 6, 9, 12 N. Check `cc_transform(7.2, 0.1)` against `cc_tip` and say in
  one sentence which MR object the "arc" is (Workshop P1e).
* E2. Reproduce the two-segment soft-gripper finger of Workshop Problem 4 and plot both segments.

### Checkpoint 2 — Shooting (B1–B3, B5, ≈ 12 min)

Implement `rod_rhs`, `shooting_residual`, `solve_rod_shooting` and `euler_bernoulli_tip`.

* E3. **Pure moment check:** $F = 0$, $M = r\tau$ with $\tau = 6$ N must give the CC arc to $10^{-8}$ m and a constant moment.
* E4. **Cantilever check:** straight catheter, transverse tip force with $\alpha = PL^2/EI$ = 0.1, 0.5, 1, 2. Tabulate
  Cosserat and Euler–Bernoulli tip deflections and the error (Workshop Problem 2d). Plot the four rod shapes.
* E5. Plot the scalar shooting residual $R(\sigma) = m(L)$ against the guessed base moment $\sigma$ for the inspection arm of
  Workshop Problem 3 ($L = 0.40$ m, $EI = 0.50$ N m², 0.5 kg at the tip). Mark the rigid guess, the secant step and the root.

### Checkpoint 3 — Collocation with SciPy (B4, ≈ 8 min)

Implement `solve_rod_bvp` with `scipy.integrate.solve_bvp`.

* E6. Solve the $\alpha = 1$ cantilever and compare the tip with shooting and with the elastica value
  $(x_L, y_L)/L = (0.9436, 0.3017)$.
* E7. Solve a combined case (tendon 6 N, tip force $(0.03, -0.05)$ N, distributed load $(0, -0.2)$ N/m) with both
  solvers. Report the largest difference in $m(s)$ and the run time of each. When would you prefer each solver?

### Checkpoint 4 — How wrong are CC and Euler–Bernoulli under contact? (C1–C2, ≈ 10 min)

Implement `catheter_tip` and `compare_models`.

* E8. With $\tau = 6$ N, push the tip back against its bend with $F_n$ = 0 … 0.10 N. Plot the three tip positions
  (Cosserat, CC, Euler–Bernoulli about the straight rod) and the two position errors against $F_n$.
* E9. Find the contact force at which the CC tip error reaches 2 mm (a typical targeting tolerance) and 5 mm.
  Why does the Euler–Bernoulli error *decrease* with load here? Is that good news?

### Checkpoint 5 — Tip-force estimation (D1–D5, ≈ 13 min)

Implement `compliance_matrix`, `estimate_force_linear`, `estimate_force_nl`, `estimate_log` and `force_errors`.

* E10. Compute the compliance matrix $C$ at $\tau = 0$ and $\tau = 6$ N. Compare the straight one with Euler–Bernoulli and
  the angle row of the bent one with the closed forms of Workshop P5 / Practice P5. Report the condition number of
  $WC$ with $W = \mathrm{diag}(1/\sigma_p, 1/\sigma_p, 1/\sigma_\theta)$.
* E11. Estimate the force for every row of `data/catheter_log.csv` with the four combinations {linear, nonlinear} ×
  {2-D force, contact model along `contact_direction`}. Report the RMS error and its normal and tangential parts
  against the load cell. Plot estimated vs reference normal force.
* E12. Explain in two sentences why the 2-D estimate fails and what the contact model assumes. Which error remains with
  the contact model, and where does it come from?

# Deliverables

1. Your module passing all 24 tests (`LAB_IMPL=<your module> pytest -q test_lab.py`).
2. A results table with: E4 errors; E6 tip; E7 solver difference and times; E9 forces; E10 condition number;
   E11 RMS errors (four methods).
3. Three plots: the rod shapes of E4 with the Euler–Bernoulli tips; the CC/EB error curves of E8; estimated vs
   reference normal force (E11, nonlinear contact-model and linear 2-D estimates).
4. Reflection (≤ 200 words): (a) one sentence each on when the CC model is exact, why Euler–Bernoulli overpredicts large
   deflections, and why the axial force is hard to observe; (b) what would you need to change to model a real
   catheter in the heart (3-D, friction, blood flow, sheath); (c) **AI-use note**: tool, what you asked, what you kept,
   what you checked yourself (or "no AI used").

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| `solve_ivp` returns shapes (6, 1) or `rod_rhs` fails in `solve_bvp` | `rod_rhs` must accept $z$ of shape (6,) and (6, k); build constant rows with `np.ones_like(theta)` |
| pure-moment test gives a rod bent the wrong way | sign of $m'$: it is $-(x' n_y - y' n_x)$; and the tendon moment is $+r\tau$ |
| shooting does not converge for large loads | start from the straight-rod guess `_rigid_guess`; use load continuation (solve at 10 %, 20 %, … of the load, re-using each answer) |
| `solve_bvp` reports "maximum number of mesh nodes exceeded" | give it a better initial guess (straight rod with the rigid-body internal wrench) and `max_nodes` ≥ 10000 |
| elastica test off by a few % | your load is not perpendicular to the undeformed rod, or you compared with $L$-normalised values in mm |
| compliance-matrix test fails at $10^{-4}$ | step $h$ too large or too small; central differences with $h = 10^{-3}$ N are fine |
| nonlinear estimate wanders to huge forces | ill-conditioned 2-D problem: scale the unknowns (`x_scale`), start from zero, or use the contact model |
| `estimate_log` takes minutes | your `catheter_tip` re-solves with three unknowns: with $f = 0$, $n \equiv F$ and only $m(0)$ is unknown (secant iteration) |

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab 1 / E1 | $\tau = 6$ N: $\kappa$, tip | 7.2 m⁻¹; (91.58, 34.47) mm, 0.72 rad (41.25°) |
| Pre-lab 3 | EB deflection, angle at 0.01 N | 3.333 mm, 0.050 rad |
| Pre-lab 5 | force uncertainty | 1.82 mN |
| E2 | gripper finger tip | (74.31, 22.88) mm, 0.80 rad |
| E4 | EB error at $\alpha$ = 0.1, 0.5, 1, 2 | +0.11 %, +2.79 %, +10.5 %, +35.1 % |
| E5 | arm: rigid guess, secant step, root (N m) | −1.962, −1.7366, −1.7357 |
| E6 | $\alpha = 1$ tip | $(x_L, y_L)/L = (0.9436, 0.3017)$, $\theta_L = 0.4614$ rad; base moment 9.436 mN m |
| E8 | Cosserat tip at $F_n$ = 0.05 / 0.10 N | (96.79, 20.73) mm, 0.494 rad / (99.21, 8.54) mm, 0.305 rad |
| | CC error at $F_n$ = 0.025, 0.05, 0.10 N | 7.60, 14.70, 27.03 mm |
| | EB error at $F_n$ = 0, 0.05, 0.10 N | 8.56, 4.23, 2.53 mm |
| E9 | $F_n$ for 2 mm / 5 mm CC error | 6.5 mN / 16.3 mN |
| E10 | $C$ at $\tau = 0$ | angle row (0, 5.00) rad/N, $y$ row (0, 0.333) m/N, $x$ row 0 |
| | $C$ at $\tau = 6$ N | [[0.0626, −0.1261], [−0.1261, 0.2622], [−2.278, 4.370]]; cond($WC$) ≈ 143 |
| E11 | RMS error (mN): linear 2-D / nonlinear 2-D | 897 / 92 (tangential 866 / 82) |
| | RMS error (mN): linear contact model / nonlinear contact model | 16.5 / 13.4 (normal 11.5 / 6.5; tangential 11.7 = the unmodelled friction component) |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, deliverables and reflection are the same; the auto-checks compare your results with the closed forms of the Python tests (CC arc, Euler–Bernoulli, the Workshop P5 compliances), with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: the exponential of a twist), with the elliptic-integral elastica and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations, documentation and sign conventions — do not change), `helpers.cpp` (given, do not edit: `load_catheter`, `load_csv`, `tip_pose`, `contact_direction`, `integrate_rod`, `rigid_guess`, `weights`, `free_tip_angle`, `linspace` and the three SciPy stand-ins `fsolve`, `least_squares`, `solve_bvp`), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers: the elastica tips and the Python results), `CMakeLists.txt`. The data files are the same `data/catheter.json` and `data/catheter_log.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 7 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="test_A*"                # only some tests (like pytest -k)
```

The C++ track has 22 test cases with the Python test names. With the untouched starter the program builds, 20 test cases fail with `TODO: <function>` and 2 pass (they check the data files read by the given code), exactly as the Python starter gives 22 failed, 2 passed. The Python track counts 24 tests because pytest counts each `alpha` of `test_B3_shooting_matches_the_elastica` (0.1, 1, 2) separately; the C++ test case loops over the three. A full run takes about 2 s. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 17 as in Python, same order): `cc_tip`, `tendon_curvature`, `cc_transform`, `pcc_tip`, `pneumatic_curvature` (Part A); `rod_rhs`, `shooting_residual`, `solve_rod_shooting`, `solve_rod_bvp`, `euler_bernoulli_tip` (Part B); `catheter_tip`, `compare_models` (Part C); `compliance_matrix`, `estimate_force_linear`, `estimate_force_nl`, `estimate_log`, `force_errors` (Part D). `integrate_rod` calls your `rod_rhs`, so every shooting test fails with `TODO: rod_rhs` until B1 is done.

**Checkpoints.** 1: `./build/lab_tests -tc="test_A*"` (4 test cases); 2: `-tc="test_B1*,test_B2*,test_B3*,test_B5*"` (7); 3: `-tc="test_B4*"` (1); 4: `-tc="test_C*"` (2); 5: `-tc="test_D*"` (6); finally `./build/lab_tests` must show 22 passed. The expected values of the table above (E1–E11) are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| catheter dict `cat["L"]`, `cat["EI"]`, `cat["r"]` | struct `lab::Catheter` with the same field names: `cat.L`, `cat.EI`, `cat.r`, `cat.sigma_p`, ... |
| log dict `log["tau"]`; `{k: v[:12] for k, v in log.items()}` | `lab::Log` (a `std::map` of `Eigen::VectorXd`): `log.at("tau")`; `v.head(12)` |
| rod-solution dict `sol["x"]`, `sol["m"]`, `sol["g"]`; result dicts of `compare_models` and `force_errors` | structs `lab::RodSolution` (`sol.x`, `sol.m`, `sol.g`), `lab::ModelComparison` (`out.cosserat`, `out.err_cc`, ...), `lab::ForceErrors` (`e.rms_normal`, ...) |
| `F=(0, P)`, `f=(0, -w)` keyword arguments | `Eigen::Vector2d(0, P)`, positional in the order of `lab.hpp`: `lab::solve_rod_shooting(L, EI, Eigen::Vector2d(0, 0), 0.0, Eigen::Vector2d(0, -w))` |
| `guess=None`, `direction=None`, `F_init=None` | `std::optional`: pass `std::nullopt` (the default) or a vector; inside, `if (direction) { Eigen::Vector2d d = *direction; ... }` |
| `rod_rhs` on a state of shape (6,) **or** (6, k) | one state (`lab::Vector6d`) per call; the solvers call it point by point (`test_B1` checks the same state at several $s$ instead of a tiled array) |
| `_integrate` with `solve_ivp` (RK45, `rtol=1e-10`) | given `lab::integrate_rod(z0, L, EI, f, kappa0, n_eval)`: classical RK4 with about 4000 fixed steps; returns a $6\times n_{eval}$ matrix (column = state at one $s$). It agrees with the Python solver to about $10^{-11}$ m |
| `scipy.optimize.fsolve(fun, g0)` | given `lab::fsolve(fun, g0)`: Newton with a finite-difference Jacobian; `fun` is a lambda `[&](const Eigen::VectorXd& g) -> Eigen::VectorXd { ... }`; check the residual afterwards |
| `scipy.integrate.solve_bvp(fun, bc, s, Z0)`, `res.success`, `res.message`, `res.sol(s)` | given `lab::solve_bvp(fun, bc, s, Z0)` with the same pattern and fields (`res.success`, `res.message`, `res.sol(s)`); `fun(s, z)` takes ONE state; same 4th-order Lobatto IIIA collocation as SciPy, but the mesh is refined by halving all intervals (at most 300 nodes) |
| `scipy.optimize.least_squares(fun, x0, x_scale=0.05)` | given `lab::least_squares(fun, x0, 0.05)` (Levenberg–Marquardt); for one unknown use an `Eigen::VectorXd` of size 1 |
| `np.linalg.pinv` | `mr::pinv` (from `<sbeg/mrcpp.hpp>`) |
| `mr.VecTose3`, `mr.MatrixExp6` | `mr::vec_to_se3`, `mr::matrix_exp6` |
| `raise RuntimeError(res.message)` | `throw std::runtime_error(res.message);` |
| elastica reference (`scipy.special` elliptic integrals) in `test_lab.py` | values computed in Python and stored in `cpp_cases.json` |

**Figures, timings and the results table.** The plots stay outside the auto-checks. Write the experiments (E1–E11: rod shapes, the shooting residual curve, the CC/EB error curves, the estimated vs reference forces) as a C++ program and save the data with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool; time the two solvers for E7 with `std::chrono::steady_clock`. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_catheter()` and `lab::load_csv("catheter_log.csv")` find the data from any build folder; for example `lab::RodSolution s = lab::solve_rod_shooting(L, EI, Eigen::Vector2d(0, P)); Eigen::MatrixXd M(s.s.size(), 3); M << s.s, s.x, s.y; sbeg::write_csv("shape_alpha1.csv", {"s", "x", "y"}, M);`. Deliverable 1 for the C++ track is `cpp/starter.cpp` passing all 22 test cases (add the functions to your library as `continuum.cpp` if you keep a C++ library); deliverable 3 is `experiments.cpp` plus your plotting script and the three plots. The results table and the reflection are unchanged; in the E7 answer, compare the C++ RK4 shooting and the C++ collocation solver.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../catheter.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| the shooting, BVP and catheter tests (B2–B4, C1–C2, D1–D4) fail with `TODO: rod_rhs` | the given `integrate_rod` (and your `solve_rod_bvp`) call your `rod_rhs` | write B1 first |
| `solve_bvp` fails: "Newton iteration ... did not converge" | a wrong sign in `rod_rhs` or `bc`, `bc` not returning 6 values, or an initial guess that is not one column per mesh node | check `rod_rhs` with `test_B1` first; `Z0` is $6\times n_{mesh}$ |
| `solve_bvp` fails: "maximum number of mesh nodes" | `n_mesh` far above the default 41, or the two solutions keep changing (wrong equations) | keep `n_mesh = 41`; check `test_B3_*` with shooting first |
| shooting returns NaN or a huge `m(0)` | the secant iteration divided by `rb - ra = 0`, or started from a poor guess | stop when `rb == ra`; start from `rigid_guess`; use load continuation for large loads |
| "Eigen: wrong matrix size or index out of range" | e.g. the `ModelComparison` matrices not sized (`out.cosserat.resize(k, 3);`) or `Vector6d(z)` built from a vector of the wrong length | check the sizes; the test program does not crash, the test simply fails |
| Results wrong only in some entries, no compile error | an Eigen expression stored in `auto` (`auto d = a - b;` keeps references to temporaries) | write the type: `Eigen::Vector3d d = a - b;` |
| Tests take much longer than a few seconds | a Debug build, or `catheter_tip` integrating 101 output points instead of the tip only | use the default build type; `n_eval = 2` in `catheter_tip` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
