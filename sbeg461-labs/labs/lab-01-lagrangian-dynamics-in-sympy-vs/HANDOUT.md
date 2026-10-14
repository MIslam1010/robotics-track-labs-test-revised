---
title: "Lab 1 — Lagrangian dynamics in SymPy vs. modern_robotics"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 1"
---

# Overview

| | |
|---|---|
| Session | Session 4, Computational Lab (50 min in the lab + up to 90 min to finish at home) |
| CLOs | 461-1 (derive equations of motion and explain $M$, $c$, $g$ and their properties); 461-2 (implement and verify dynamics code against a reference library) |
| Reading | MR §8.1, printed pp. 270–280: MR Eq. 8.3, 8.9–8.10, 8.14–8.19, Proposition 8.1, Eq. 8.21 |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks), `lab/data/arms.json` (robot models), `lab/data/states.csv` (test states) |
| Software | Python 3.11+, NumPy, SymPy, `modern_robotics` (dynamics reference); your SBEG 361 kinematics library `sbeg361_kinematics` (`fk.fk_space` from 361 Lab 5, `jacobians.jacobian_space` from 361 Lab 7), imported by `starter.py`; `modern_robotics` is only a fallback for students without a working library |
| Hardware | None (Tier 0, simulation only) |
| Assessment | Auto-checked notebook/code + short reflection (lab category) |

## Learning objectives

By the end of this lab you can:

1. Build the Lagrangian $\mathcal L = K - P$ of a planar $n$R chain symbolically and apply the Euler–Lagrange equations
   (MR Eq. 8.3) with SymPy.
2. Extract $M(\theta)$ as the Hessian of $K$ in $\dot\theta$ (MR Eq. 8.14), $g(\theta) = \partial P/\partial\theta$, the Christoffel
   symbols (MR Eq. 8.17) and the Coriolis matrix (MR Eq. 8.19).
3. Describe the same robot for `modern_robotics` (`Mlist`, `Glist`, `Slist`) and compare `MassMatrix`,
   `VelQuadraticForces`, `GravityForces` and `InverseDynamics` with your symbolic model to machine precision.
4. Check the structural properties numerically: $M = M^{\mathsf T} \succ 0$, $\dot M - 2C$ skew-symmetric, and the power
   balance $\frac{d}{dt}(K+P) = \dot\theta^{\mathsf T}\tau$.
5. Rebuild $M(\theta)$ from CoM Jacobians — the bridge from SBEG 361 kinematics to dynamics.

## The three robots (`data/arms.json`, illustrative values)

| Key | Robot | Use |
|---|---|---|
| `mr_2r_point` | MR Fig. 8.1 2R arm, point masses, $L_i = m_i = 1$ | Validate against MR Eq. 8.9–8.10 and Fig. 8.3 |
| `arm_exo_2r` | Shoulder–elbow arm + exoskeleton (Workshop Problem 3) | Biomedical case |
| `eod_3r` | Shoulder/elbow/wrist-pitch arm of a tracked EOD robot holding a 2 kg suspect package (Workshop Problem 4) | Defence (protective) case |

All chains move in the vertical $\hat x$–$\hat y$ plane, gravity along $-\hat y$, joint axes along $\hat z$, zero configuration
with all links along $+\hat x$. Link $i$: length `L[i]`, mass `m[i]`, CoM at `r[i]` from joint $i$, inertia `I[i]` about the CoM.

# Pre-lab check (complete before the session, 5 questions, ≈ 10 min)

1. For $K = \tfrac12(3 + 2\cos\theta_2)\dot\theta_1^2 + (1 + \cos\theta_2)\dot\theta_1\dot\theta_2 + \tfrac12\dot\theta_2^2$, what is $m_{12}$?
   *(Answer: $1 + \cos\theta_2$, the coefficient of $\dot\theta_1\dot\theta_2$, because the cross term appears twice in
   $\tfrac12\dot\theta^{\mathsf T}M\dot\theta$.)*
2. SymPy symbols are not functions of time. How do you compute $\frac{d}{dt}f(\theta,\dot\theta)$?
   *(Answer: chain rule, $\sum_i \frac{\partial f}{\partial\theta_i}\dot\theta_i + \frac{\partial f}{\partial\dot\theta_i}\ddot\theta_i$.)*
3. In `modern_robotics`, what does `Mlist[i]` contain, and in which frame is `Glist[i]` expressed?
   *(Answer: `Mlist[i]` is $M_{i-1,i}$, the pose of the link-$i$ CoM frame relative to the link-$(i-1)$ CoM frame at
   $\theta = 0$; the last entry is the end-effector frame. `Glist[i]` is the 6×6 spatial inertia in the link-$i$ CoM frame.)*
4. What is the screw axis of a revolute joint along $\hat z$ through $q = (0.6, 0, 0)$?
   *(Answer: $\mathcal S = (0,0,1,\ 0,-0.6,0)$, since $v = -\omega\times q$.)*
5. Which `modern_robotics` call returns $c(\theta,\dot\theta)$, and which returns $g(\theta)$? What gravity vector do we pass?
   *(Answer: `VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)` and
   `GravityForces(thetalist, g, Mlist, Glist, Slist)` with `g = [0, -9.81, 0]`.)*

# Setup

```bash
cd lab
# check the reference solution is NOT in your folder; work in starter.py
LAB_IMPL=starter python -m pytest -q test_lab.py     # all tests fail with NotImplementedError: expected
```

Run only the tests of the current checkpoint with `-k` and the test names listed under each checkpoint, e.g.
`LAB_IMPL=starter python -m pytest -q test_lab.py -k "euler_lagrange or time_derivative"`.
**Load your SBEG 361 library.** `starter.py` imports `fk_space` and `jacobian_space` from your package
`sbeg361_kinematics` (the provided import block at the top of the file). Make the package importable from the lab folder,
e.g. `pip install -e ~/sbeg361-repo` or `export PYTHONPATH=~/sbeg361-repo` (the folder that contains
`sbeg361_kinematics/`), then check:

```bash
python -c "import starter; print(starter.KIN_BACKEND)"   # expected: sbeg361_kinematics
```

If your library is missing or broken, the import block falls back to `mr.FKinSpace` / `mr.JacobianSpace` and prints
`modern_robotics (fallback)`; you can still finish the lab, but say so in `lab1_notes.md` and repair the library before
Lab 2, which builds `sbeg461_dynamics` on top of it. The test `test_361_library_matches_modern_robotics` checks your
library against `modern_robotics` (it is skipped on the fallback).

# Procedure and checkpoints

## Checkpoint 1 — Euler–Lagrange machinery (8 min)

Implement `make_symbols`, `time_derivative`, `euler_lagrange`.

* Test on the particle of MR Eq. 8.5–8.6: $\mathcal L = \tfrac12 m\dot x^2 - mgx$ must give $f = m\ddot x + mg$.
* Test on a pendulum: $\mathcal L = \tfrac12 ml^2\dot\theta^2 - mgl\sin\theta$.
* Tests: `-k "make_symbols or time_derivative or euler_lagrange"`.

## Checkpoint 2 — Energies, mass matrix, gravity, Christoffel symbols (12 min)

Implement `planar_chain_energy`, `mass_matrix`, `gravity_vector`, `christoffel`, `coriolis_matrix`.

* For `mr_2r_point` your $M$ must equal $\begin{bmatrix}3 + 2c_2 & 1 + c_2\\ 1 + c_2 & 1\end{bmatrix}$ (MR Eq. 8.10 with unit values)
  and $C\dot\theta$ must equal $c$ in MR Eq. 8.10.
* For `arm_exo_2r` compare your $\Gamma_{ijk}$ with the table of Workshop Problem 3b.
* For `eod_3r` check $\Gamma_{ijk} = \Gamma_{ikj}$ and $\Gamma_{ijk} + \Gamma_{jik} = \partial m_{ij}/\partial\theta_k$.
* Tests: `-k "energy or symbolic or christoffel or coriolis"`.

**Checkpoint question (write in your notes):** why is $\Gamma_{111} = 0$ for every planar $n$R chain?

## Checkpoint 3 — Numeric model and structural properties (10 min)

Implement `numeric_model` (lambdify $M$, $C$, $c$, $g$, $\dot M$, $\tau$, $K$, $P$), `skew_residual`, `power_residual`.

* Reproduce MR Fig. 8.3: $M(0, 90^\circ) = \begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix}$, $M(0, 150^\circ) \approx \begin{bmatrix}1.27 & 0.13\\ 0.13 & 1\end{bmatrix}$.
* For every state in `states.csv`: $M$ symmetric, smallest eigenvalue $> 0$, $\max|N + N^{\mathsf T}| < 10^{-10}$ for
  $N = \dot M - 2C$, power residual $< 10^{-9}$.
* Tests: `-k "fig_8_3 or closed_form or properties or power or workshop_p4"` (the last one checks the EOD arm against the hand numbers of Workshop Problem 4).

## Checkpoint 4 — The same robots in `modern_robotics` (10 min)

Implement `mr_chain`, `mr_terms` and `mass_matrix_from_jacobians`.

* `Mlist`: CoM frames at $x$ = (joint position + $r_i$), no rotation; the last entry goes from the last CoM to the tip.
  The product of all `Mlist` entries is the home pose of the tip, $\mathrm{Trans}(\sum L_i, 0, 0)$.
* `Glist[i] = diag(I_i, I_i, I_i, m_i, m_i, m_i)`: only $\mathcal I_{zz}$ matters for planar motion.
* `Slist` columns: $(0,0,1,\ 0,-x_i,0)$ with $x_i$ the joint position at $\theta = 0$.
* `mass_matrix_from_jacobians`: $M = \sum_i m_iJ_{v_i}^{\mathsf T}J_{v_i} + \mathcal I_iJ_{\omega_i}^{\mathsf T}J_{\omega_i}$, built with
  your 361 library: take the columns $j\le i$ of `jacobian_space(Slist, theta)`, $J_{\omega_i} = (\omega_j)$ and
  $J_{v_i} = (v_j + \omega_j\times p_{c_i})$, with $p_{c_i}$ from `fk_space` (MR Eq. 5.11 and $\dot p = v_s + \omega_s\times p$).
* Tests: `-k "mr_chain or mr_terms or from_jacobians or 361_library"`. The FK test checks your `Slist`/`Mlist` with `mr.FKinSpace`.

## Checkpoint 5 — SymPy vs. `modern_robotics` (10 min)

Implement `compare`. Run `python starter.py`: the printed maximum differences must all be below $10^{-9}$ for all three
robots. Then run the full test file. Finally answer in your notes:

1. Workshop Problem 4b in code: change `m[2]` of `eod_3r` from 3.5 to 1.5 kg and compute `g(0)` with both models. Do
   you get $(82.89, 21.09, 1.47)$ N·m?
2. Time the symbolic derivation for 2R and 3R (`time.perf_counter`). How would it scale to a 6R or 7R surgical arm?
   (MR §8.1, p. 270, and §8.1.4 explain why Newton–Euler is preferred for larger $n$; we implement it in Lab 2.)

All tests: `LAB_IMPL=starter python -m pytest -q test_lab.py` → **69 passed** with your `sbeg361_kinematics` library
(**66 passed, 3 skipped** on the `modern_robotics` fallback).

# Deliverables (by the start of Week 2 Lecture A: push to your Classroom 50 repo, or `python submit.py 1 --id <ID>` and upload the ZIP)

1. `starter.py` passing all tests (keep the file name or set `LAB_IMPL` to your module name).
2. `lab1_notes.md`: the kinematics backend you used (`KIN_BACKEND`), the checkpoint-2 answer, the two checkpoint-5 answers with numbers, and a table of maximum
   SymPy–`modern_robotics` differences for the three robots.
3. A one-paragraph reflection and AI-use note (below).

# Reflection (≤ 200 words, in `lab1_notes.md`)

* Where did your first attempt disagree with `modern_robotics`, and what was the cause (sign of gravity, `Mlist`
  offsets, inertia about the joint vs. about the CoM, …)?
* Exoskeleton vs. EOD arm: which term ($M\ddot\theta$, $c$, $g$) dominates the shoulder torque in each case at typical speeds,
  and what does that imply for motor sizing (Week 4)?
* **AI-use note** (required, course AI policy, Course syllabus §8 AI-use policy): which AI tool, if any, you used, for what (e.g. explaining
  `sp.lambdify`), what you kept, and confirmation that you can explain every line in an oral check.

# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NotImplementedError` in every test | You are running the untouched starter | Implement the checkpoint functions in order |
| `ModuleNotFoundError: starter` | Running pytest from another folder | `cd lab` first; `LAB_IMPL` is a module name, not a path |
| $g$ has the wrong sign | $P = m g y$ with $y$ up and gravity along $-\hat y$ | In `mr_chain` pass `[0, -g, 0]` (the gravity *vector*) |
| $M$ off by $m_ir_i^2$ | `I[i]` is about the CoM, not the joint | Use $\tfrac12 m\lVert v_c\rVert^2 + \tfrac12\mathcal I\omega^2$ |
| `modern_robotics` $M$ wrong only for link 2/3 | `Mlist[i]` is relative to the **previous CoM frame**, not to the base | Offsets are $x_{c,i} - x_{c,i-1}$ |
| `lambdify` returns nested lists / wrong shape | SymPy `Matrix` → NumPy | Wrap with `np.asarray(..., float).reshape(n, n)` |
| Derivation very slow for 3R | `sp.simplify` on large expressions | Simplify $K$ once; use `sp.expand`/`sp.trigsimp` sparingly |
| Skew test fails, $C\dot\theta = c$ passes | $C$ built by factoring $c$, not from MR Eq. 8.19 | Build $C$ from the Christoffel symbols |

# C++ track

You may do this lab in C++17 instead of Python. The robots, data files, checkpoints, questions and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`) and with the numbers of the Python reference (its SymPy model evaluated at the test states), so both tracks are marked against identical values.

**No SymPy in C++.** The C++ track derives the same model *numerically at a given state* instead of as formulas: $\mathcal L$, $\mathcal K$ and $\mathcal P$ are C++ functions of $(\theta, \dot\theta)$; $M(\theta)$ is read off $\mathcal K = \tfrac12\dot\theta^{\mathsf T}M\dot\theta$ exactly (with unit vectors, because $\mathcal K$ is a quadratic form in $\dot\theta$); $g = \partial\mathcal P/\partial\theta$, the Christoffel symbols, $\dot M$ and the Euler–Lagrange equations use given 5-point finite-difference helpers. The physics (MR Eqs. 8.3, 8.14, 8.17, 8.19) and the checks are the same as in Python.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`. The data files are the same `data/arms.json` and `data/states.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 1 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*time_derivative*,*euler_lagrange*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 20 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `time_derivative`, `euler_lagrange` (Checkpoint 1); `planar_chain_energy`, `mass_matrix`, `gravity_vector`, `christoffel`, `coriolis_matrix` (Checkpoint 2); `numeric_model`, `skew_residual`, `power_residual` (Checkpoint 3); `mr_chain`, `mr_terms`, `mass_matrix_from_jacobians` (Checkpoint 4); `compare` (Checkpoint 5). Given: `load_params`, the finite-difference helpers `partial`, `partial_q`, `partial_dq`, `partial_matrix`, and the kinematics backend `fk_space`, `jacobian_space`. `make_symbols` has no C++ counterpart. `mrcpp` may be used in Checkpoint 4 (as `modern_robotics` in Python).

**Kinematics backend.** In `starter.cpp`, `fk_space` and `jacobian_space` call `mr::fkin_space` / `mr::jacobian_space` (the equivalent of the Python `modern_robotics` fallback). If you did the SBEG 361 labs in C++, replace the two bodies by your own functions (361 Lab 5 FK, 361 Lab 7 space Jacobian); `test_361_library_matches_modern_robotics` checks them (and the screw axes of your `mr_chain`). In `lab1_notes.md` write which backend you used ("mrcpp fallback" or "own 361 C++ code").

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Euler–Lagrange machinery | `-tc="*time_derivative*,*euler_lagrange*"` | 3 |
| 2 — Energies, $M$, $g$, Christoffel, $C$ | `-tc="*energy*,*symbolic*,*christoffel*,*coriolis*"` | 6 |
| 3 — Numeric model | `-tc="*fig_8_3*,*closed_form*,*properties*,*power*,*workshop_p4*"` | 5 |
| 4 — `modern_robotics` model | `-tc="*mr_chain*,*mr_terms*,*from_jacobians*,*361_library*"` | 4 |
| 5 — Model vs. `modern_robotics` | `-tc="*compare*,*inverse_dynamics*"` | 2 |

All 20 test cases pass with `./build/lab_tests`. The Python track counts 69 tests because pytest counts each robot and each state of the `@pytest.mark.parametrize` tests separately (and has `test_make_symbols`); the C++ track checks the same robots and states inside one test case each. Test names are kept from Python (e.g. `test_gravity_symbolic`), although the C++ values are numerical.

**Checkpoint 5 questions in C++.** There is no `python starter.py` printout: write a small `experiments.cpp` with your own `main()` (build it next to `starter.cpp`, e.g. a second `add_executable` in a copy of the folder) that prints `lab::compare(...)` for the three robots (states from `data/states.csv`). Question 1: `lab::ChainParams p = lab::load_params("eod_3r"); p.m(2) = 1.5;` then `lab::numeric_model(p).g(Eigen::VectorXd::Zero(3))` and `lab::mr_terms(...)`. Question 2: time `numeric_model(p).tau(...)` for 2R and 3R with `std::chrono::steady_clock`, count how many evaluations of $\mathcal K$ one $\tau$ needs as a function of $n$, and compare with the SymPy derivation times (MR §8.1 explains why Newton–Euler is preferred for large $n$; Lab 2).

**Differences from the Python version.**

| Python | C++ |
|---|---|
| SymPy symbols and expressions (`make_symbols`, `sp.diff`, `lambdify`) | functions of the state: `lab::StateFn` $f(\theta,\dot\theta)$, `lab::ConfigFn` $f(\theta)$, `lab::MatrixFn` $F(\theta)$ (`std::function`; write lambdas); derivatives with the given helpers (5-point central differences, step `FD_STEP` = $10^{-3}$) |
| params dict `p["L"]`, … | `lab::ChainParams` from `lab::load_params(name)`: `p.L(i)`, `p.m(i)`, `p.r(i)`, `p.I(i)`, `p.g`, `p.n()` |
| `planar_chain_energy(params)` returns `(K, P, th, dth, ddth)` | `planar_chain_energy(params, theta, dtheta)` returns `lab::Energy` with fields `K`, `P` at that state |
| `mass_matrix(K, dq)`: symbolic Hessian | `mass_matrix(K, theta)`: exact from the quadratic form (hint in `starter.cpp`) |
| `christoffel(M, q)`: nested list `G[i][j][k]` | `christoffel(M, theta)`: `std::vector<Eigen::MatrixXd>`, `G[i](j, k)` |
| `numeric_model` returns a dict of functions | returns `lab::NumericModel` (`model.M(theta)`, `model.c(theta, dtheta)`, `model.tau(theta, dtheta, ddtheta)`, …); capture by value in the lambdas |
| `mr_chain`, `mr_terms` return tuples; `compare` a dict | `lab::MRChain{Mlist, Glist, Slist, gvec}`, `lab::MRTerms{M, c, g}`; `std::map<std::string, double>` |
| `sbeg361_kinematics` import block (fallback `modern_robotics`); `test_361_library_matches_modern_robotics` skipped on the fallback | given `lab::fk_space` / `lab::jacobian_space` (mrcpp; replace with your 361 C++ code if you have it); the test always runs |
| exact SymPy derivatives: tolerances $10^{-12}$ or exact equality | finite differences: $10^{-9}$ for $g$, $\Gamma_{ijk}$, $C\dot\theta$, the time derivative; $10^{-6}$ for the 1-DOF Euler–Lagrange checks; power residual $< 10^{-6}$ instead of $10^{-9}$ (the Euler–Lagrange equations take differences of differences). $M$, $\mathcal K$, $\mathcal P$ keep $10^{-12}$; `np.allclose` checks keep NumPy's relative tolerance $10^{-5}$ (this is what lets $\tau$, whose absolute error is about $10^{-9}$, pass the `atol` = $10^{-10}$ closed-form check, exactly as `np.allclose` does in Python) |
| `np.max(np.abs(A - B))`, `np.cross`, `Slist[:, :i+1]`, `theta[:i+1]` | `(A - B).cwiseAbs().maxCoeff()`, `a.cross(b)` (`Eigen::Vector3d`), `Slist.leftCols(i + 1)`, `theta.head(i + 1)` |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

A matrix-size mistake does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `bad_function_call` in a Checkpoint 3 test | a member of `NumericModel` was not assigned | assign all nine (`n`, `M`, `C`, `c`, `g`, `Mdot`, `tau`, `K`, `P`) |
| Checkpoint 3 crashes or gives random numbers | a lambda in `numeric_model` captured a local variable by reference (`[&]`) | capture by value: `[p]`, `[K]`, `[K, P]` |
| Diagonal of $M$ twice too small | used $m_{ii} = \mathcal K(e_i)$ | $m_{ii} = 2\mathcal K(e_i)$ |
| $\tau$ / power tests fail, $M$, $c$, $g$ pass | Lagrangian $\mathcal K + \mathcal P$, or the two Euler–Lagrange terms swapped | $\mathcal L = \mathcal K - \mathcal P$; $f_i = \frac{d}{dt}\partial\mathcal L/\partial\dot\theta_i - \partial\mathcal L/\partial\theta_i$ |
| "Eigen: wrong matrix size" in `mass_matrix_from_jacobians` | `Slist.leftCols(i + 1)` and `theta.head(i + 1)` of different lengths | use the same `i + 1` |
| Wrong results only after `auto J = A * B;` | `auto` keeps an Eigen expression, not a matrix | write the type: `Eigen::MatrixXd J = A * B;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
