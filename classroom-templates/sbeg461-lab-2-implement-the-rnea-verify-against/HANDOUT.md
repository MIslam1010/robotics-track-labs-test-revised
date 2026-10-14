---
title: "Lab 2 — Implement the RNEA; verify against modern_robotics and Pinocchio"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 2"
---

# Overview

| | |
|---|---|
| Session | 4 (computational lab), 50 min in class + up to 2 h to finish |
| CLOs | 461-2 (implement and verify recursive dynamics algorithms); 461-1 (structure of $M$, $c$, $g$) |
| Reading | MR Sections 8.2–8.3 (pp. 281–293); Featherstone §5.3 (pp. 92–97) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks), `lab/data/` (robots, states, trajectories) |
| Codebase | add your finished functions to your `sbeg461_dynamics` package (built on your SBEG 361 kinematics library) |
| Assessment | auto-checked (tests) + short reflection; oral check of any line of code on request |
| Hardware | none (Tier 0, simulation only) |

## Objectives

By the end of the lab you will be able to:

1. implement the single-rigid-body equations $\mathcal{G}_b$, $\mathcal{G}_a = [\mathrm{Ad}_{T_{ba}}]^{\mathsf T}\mathcal{G}_b[\mathrm{Ad}_{T_{ba}}]$ and $\mathcal{F}_b = \mathcal{G}_b\dot{\mathcal{V}}_b - [\mathrm{ad}_{\mathcal{V}_b}]^{\mathsf T}\mathcal{G}_b\mathcal{V}_b$ (MR Eqs. 8.32, 8.40, 8.42);
2. implement the forward and backward passes of the Newton–Euler inverse dynamics algorithm (MR Eqs. 8.50–8.54) for any open chain given in `modern_robotics` list format;
3. obtain $M(\theta)$, $c(\theta,\dot\theta)$, $g(\theta)$ and $J_b^{\mathsf T}\mathcal{F}_{tip}$ using only RNEA calls;
4. verify your implementation against `modern_robotics` and **Pinocchio**, handling Pinocchio's different conventions (linear-first spatial vectors, external forces *on* bodies in joint frames);
5. compute joint-torque profiles for a defence (EOD arm lifting a suspect package) and a biomedical (arm exoskeleton reaching) trajectory and interpret the peak torques.

## Robots in `data/robots.json`

| key | description |
|---|---|
| `mr_2r_point` | MR Section 8.1 / Figure 8.3 point-mass 2R arm ($L_1 = L_2 = m_1 = m_2 = 1$), gravity $-\hat y$ |
| `arm_exo_2r` | Lab 1 sagittal-plane arm + exoskeleton (same parameters as Week 1), illustrative values |
| `eod_3r` | Lab 1 vertical-plane arm of a tracked EOD robot; link 3 includes a 2 kg suspect package, illustrative values |
| `ur5` | nominal UR5 model used with the Modern Robotics software (links 1–3 as in the `modern_robotics.InverseDynamics` docstring), gravity $-\hat z$ |

Each entry stores `Mlist` $= [M_{01}, \ldots, M_{n,n+1}]$, `Glist` (CoM-frame spatial inertias), `Slist` (space screw axes as columns) and `g`.
`data/states.csv` holds test states (some with a tip wrench); `data/eod_lift.csv` and `data/exo_reach.csv` hold quintic joint trajectories
(columns `t, theta…, dtheta…, ddtheta…`). `data/make_data.py` regenerates all of them.

# Pre-lab check (complete before the session, 10 min)

1. In MR's RNEA, what are $\mathcal{V}_0$ and $\dot{\mathcal{V}}_0$ for a fixed base with gravity $\mathfrak{g} = (0,0,-9.81)$?
2. What is the size of $[\mathrm{Ad}_{T}]$, and how does it act on a twist and (transposed) on a wrench?
3. Write $[\mathrm{ad}_{\mathcal{V}}]$ for $\mathcal{V} = (\omega, v)$. Why does the RNEA use $[\mathrm{ad}_{\mathcal{V}}]^{\mathsf T}$ in the backward pass?
4. $\mathcal{F}_{tip}$ in `modern_robotics`: who applies it to whom, and in which frame is it expressed?
5. Pinocchio orders a spatial vector as (linear, angular). Write the permutation that converts an MR wrench $(m, f)$ to Pinocchio order.

*Answers (for self-check):* 1. $\mathcal{V}_0 = 0$, $\dot{\mathcal{V}}_0 = (0,0,0,\ 0,0,9.81)$. 2. $6\times6$; $\mathcal{V}_a = [\mathrm{Ad}_{T_{ab}}]\mathcal{V}_b$, $\mathcal{F}_b = [\mathrm{Ad}_{T_{ab}}]^{\mathsf T}\mathcal{F}_a$ (MR Eq. 3.97). 3. $\begin{bmatrix}[\omega]&0\\ [v]&[\omega]\end{bmatrix}$; wrenches are dual to twists, so the velocity-product term acting on momentum (a wrench-like quantity) uses the transpose (MR Eq. 8.39). 4. The wrench the end-effector applies **to the environment**, in $\{n+1\}$. 5. $(f, m)$: swap the two 3-blocks.

# Setup (5 min)

```bash
cd "<your Lab code folder>"     # the Lab 2 code folder (or your course repository copy)
source ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/activate   # or the course Docker image / JupyterHub
python -c "import numpy, modern_robotics, pinocchio; print(pinocchio.__version__)"
LAB_IMPL=starter pytest -q test_lab.py        # all tests fail with NotImplementedError -- expected
```

Work in `starter.py` (or copy it to `my_rnea.py` and run `LAB_IMPL=my_rnea pytest -q`). Run only one checkpoint's tests with
`pytest -q -k <name>`, e.g. `-k rigid_body`. You may use the `modern_robotics` helpers `Adjoint`, `ad`, `TransInv`, `MatrixExp6`,
`VecTose3`, `RpToTrans`, `TransToRp` (or your 361 library). Do **not** call `mr.InverseDynamics`, `mr.MassMatrix`, … or Pinocchio
inside checkpoints 1–4 and 6: they are the references.

# Procedure

## Checkpoint 1 — one rigid body (10 min)

Implement `spatial_inertia`, `transform_spatial_inertia` and `rigid_body_wrench`.
Tests: `-k "spatial_inertia or rigid_body"`. They check the blocks of $\mathcal{G}$, the prosthesis numbers of Workshop W1,
kinetic-energy invariance under a change of frame, agreement with Newton's and Euler's equations (MR Eqs. 8.22–8.23),
Quiz Q2, and the frame invariance of MR Eq. 8.43.

> **Checkpoint 1 question.** With $\dot{\mathcal{V}} = 0$ the wrench is not zero. Print `rigid_body_wrench` for the Quiz Q2 crate and name the physical force.

## Checkpoint 2 — forward iterations (10 min)

Implement `link_screw_axes` and `forward_iterations` (MR Eqs. 8.50–8.52). Use Python indices carefully: column 0 of `V`
and `Vdot` is the base; `T[i]` is $T_{i+1,i}$ in MR's 1-based numbering and `T[n]` $= M_{n+1,n}$.
Tests `-k "forward or link_screw"` check $T_{i,i-1}$ against `FKinSpace`, $\mathcal{V}_i$ against the body Jacobian of the
sub-chain ($\mathcal{V}_i = J_{b,i}(\theta)\dot\theta$), $\dot{\mathcal{V}}_i$ against a finite-difference time derivative of
$\mathcal{V}_i$ (with $\mathfrak{g} = 0$), and the Workshop W3 hand values.

> **Checkpoint 2 question.** For the W3 state, print `Vdot[:, 2]` with and without gravity. Which entries change, and why does the
> difference *not* depend on $\dot\theta$?

## Checkpoint 3 — backward iterations and full RNEA (10 min)

Implement `backward_iterations` (MR Eqs. 8.53–8.54) and `rnea`.
Tests `-k "backward or rnea"`: W3 hand values ($\tau = (3.7065, 5.124)$), the closed form MR Eq. 8.10 for `mr_2r_point`,
all states in `states.csv` (including tip wrenches and the 6-DOF UR5) against `mr.InverseDynamics`, and the docstring example of
`mr.InverseDynamics` ($\tau = (74.696, -33.068, -3.231)$).

## Checkpoint 4 — $M$, $c$, $g$ and $J^{\mathsf T}\mathcal{F}$ from RNEA calls (5 min)

Implement `dynamics_terms` and `tip_wrench_torques` using **only** your `rnea` (Featherstone p. 94):
$g = \mathrm{RNEA}(\theta,0,0,\mathfrak{g},0)$, $c = \mathrm{RNEA}(\theta,\dot\theta,0,0,0)$, column $j$ of $M$ = $\mathrm{RNEA}(\theta,0,e_j,0,0)$,
$J_b^{\mathsf T}\mathcal{F}_{tip} = \mathrm{RNEA}(\theta,0,0,0,\mathcal{F}_{tip})$.
Tests compare with `MassMatrix`, `VelQuadraticForces`, `GravityForces` and $J_b^{\mathsf T}\mathcal{F}$ (MR Eq. 5.26), and check
that $M$ is symmetric positive definite and that the four parts add up to `rnea`.

> **Checkpoint 4 question.** How many RNEA calls does `dynamics_terms` make for the UR5? What is the cost in $n$? (Week 3 introduces the CRBA, which is cheaper.)

## Checkpoint 5 — Pinocchio (10 min)

Read the provided `pinocchio_model(robot)`: it converts MR lists into a Pinocchio model whose joint frames have the
orientation of the MR link frames and origins on the joint axes. Then implement `pinocchio_inverse_dynamics`. The three
convention changes are:

1. Pinocchio's `fext[j]` is the wrench applied **on** body $j$; MR's $\mathcal{F}_{tip}$ is applied **by** the end-effector: use $-\mathcal{F}_{tip}$.
2. `fext[j]` is expressed in **joint frame $j$**: transform from $\{n+1\}$ with $[\mathrm{Ad}_{T_{ee,J_n}}]^{\mathsf T}$ (MR Eq. 3.98), $T_{ee,J_n} = T_{J_n,ee}^{-1}$ (`info["T_jn_ee"]`).
3. `pin.Force(linear, angular)` — **linear first**.

Tests `-k pinocchio` compare with `mr.InverseDynamics` on all robots, with and without tip wrenches.

## Checkpoint 6 — application: torque profiles (5 min)

Implement `torque_profile` and `peak_torques`, then run

```python
import starter as lab, numpy as np
for name, f in [("eod_3r", "eod_lift.csv"), ("arm_exo_2r", "exo_reach.csv")]:
    r = lab.load_robot(name); t, th, dth, ddth = lab.load_trajectory(f)
    tau = lab.torque_profile(r, th, dth, ddth)
    print(name, lab.peak_torques(tau), t[np.argmax(np.abs(tau), axis=0)])
```

Plot $\tau(t)$ for both trajectories together with the gravity-only torque $g(\theta(t))$ (from `dynamics_terms`).
Reference values: EOD lift peaks $\approx (107.5,\ 34.0,\ 3.8)$ N m; exoskeleton reach peaks $\approx (10.2,\ 4.3)$ N m.
Tests: `-k "torque_profile or peak"` (against `mr.InverseDynamicsTrajectory`).

## Optional — timing

`python starter.py` (after all checkpoints) prints the mean time per call of your RNEA, `modern_robotics` and Pinocchio (it uses
`time_rnea`). Typical results on a laptop: a few hundred µs for the Python implementations, under 1 µs for Pinocchio (C++).
Lab 4 repeats this comparison with your own C++/Eigen RNEA.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 2 into your cumulative library as `rnea.py`. The functions you write (12):

`spatial_inertia(I: np.ndarray, m: float)`, `transform_spatial_inertia(G_b: np.ndarray, T_ba: np.ndarray)`, `rigid_body_wrench(G: np.ndarray, V: np.ndarray, Vdot: np.ndarray)`, `link_screw_axes(Mlist, Slist)`, `forward_iterations(theta, dtheta, ddtheta, g, Mlist, Slist)`, `backward_iterations(fwd: dict, Ftip, Glist)`, `rnea(theta, dtheta, ddtheta, g, Ftip, Mlist, Glist, Slist)`, `dynamics_terms(theta, dtheta, g, Mlist, Glist, Slist)`, `tip_wrench_torques(theta, Ftip, Mlist, Glist, Slist)`, `pinocchio_inverse_dynamics(model, info: dict, theta, dtheta, ddtheta, Ftip=None)`, `torque_profile(robot: dict, theta, dtheta, ddtheta, Ftip=None)`, `peak_torques(tau: np.ndarray)`.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` (or your renamed module) passing all 40 tests: `LAB_IMPL=<module> pytest -q test_lab.py` — paste the summary line.
2. One figure: torque profiles of the EOD lift (3 joints) with the gravity-only torques dashed; axis labels with units; caption.
3. Answers to the four checkpoint questions (2–4 sentences each).
4. Reflection (≤ 300 words, below), including the AI-use note.

Submit through Classroom 50 by the start of the Week 3 lab.

# Reflection prompts

* For the EOD lift, what fraction of the peak shoulder torque is gravity, and what fraction is due to motion? What does this say about choosing a slower lift to protect the gearbox, and about holding a suspect package still?
* For the exoskeleton reach, the peak shoulder torque is *lower* than the peak gravity torque. Explain using the sign of $M\ddot\theta + c$ during deceleration. What does this imply for an assist-as-needed exoskeleton controller?
* Which convention difference between MR and Pinocchio cost you the most time, and how did the tests reveal it?
* **AI-use note (required, Course syllabus §8 AI-use policy):** state whether you used an AI tool, which one, the prompts you used, and what you kept or changed. You must be able to explain every line in the oral check. If you did not use AI, write “No AI tools used.”

# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| $\tau$ correct without gravity, wrong with it | sign of $\dot{\mathcal{V}}_0$ | $\dot{\mathcal{V}}_0 = (0, -\mathfrak{g})$; with $\mathfrak{g} = (0,0,-9.81)$ the linear part is $+9.81\hat z$ |
| Velocities match, accelerations off by a velocity-squared term | missing or wrong-sign $[\mathrm{ad}_{\mathcal{V}_i}]\mathcal{A}_i\dot\theta_i$ | use `mr.ad(V_i) @ A_i * dtheta_i` with the **new** $\mathcal{V}_i$ |
| Last joint wrong only when $\mathcal{F}_{tip} \ne 0$ | `T[n]` missing or not $M_{n+1,n} = M_{n,n+1}^{-1}$ | `T[n] = mr.TransInv(Mlist[n])` |
| Errors grow along the chain | using `Adjoint(T)` instead of `Adjoint(T).T` for wrenches | twists use $\mathrm{Ad}$; wrenches use $\mathrm{Ad}^{\mathsf T}$ of the *inverse* transform |
| Pinocchio differs by exactly $2J^{\mathsf T}\mathcal{F}_{tip}$ | force sign (on vs by) | use $-\mathcal{F}_{tip}$ |
| Pinocchio differs only for 3-D robots with a tip wrench | linear/angular swapped or wrong frame | `pin.Force(F_J[3:], F_J[:3])`, transform with $[\mathrm{Ad}_{T_{ee,J_n}}]^{\mathsf T}$ |
| `ImportError: pinocchio` | environment | use the course Docker image or JupyterHub; `pip install pin` |
| Shape errors in `torque_profile` | passing a 1-D state to `rnea` row by row incorrectly | iterate with `zip(theta, dtheta, ddtheta)` |

Demo link: MR Studio's Dynamics tab (“Compute τ(t) by RNEA”) shows the same computation for a URDF robot (see MR Studio `DEMO_GUIDE.md`).

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the numbers of the Python reference and with numbers computed by the real Pinocchio, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`, and given code you do not edit: `pinlite.hpp` / `pinlite.cpp` (a small C++ stand-in for the part of Pinocchio used in Checkpoint 5, with Pinocchio's conventions) and `torque_profiles.cpp` (the Checkpoint 6 program). The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). Pinocchio is not needed for the C++ track.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 2 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*spatial_inertia*,*rigid_body*"   # only some tests (like pytest -k)
./build/torque_profiles                        # Checkpoint 6: peaks + CSV files for the figure
./build/torque_profiles --time                 # optional timing
```

With the untouched starter, the program builds and all 18 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `spatial_inertia`, `transform_spatial_inertia`, `rigid_body_wrench` (Checkpoint 1); `link_screw_axes`, `forward_iterations` (2); `backward_iterations`, `rnea` (3); `dynamics_terms`, `tip_wrench_torques` (4); `pinocchio_inverse_dynamics` (5); `torque_profile`, `peak_torques` (6). The loaders `load_robot`, `load_states`, `load_trajectory` are given in `starter.cpp`; `pinocchio_model` and `pin_rnea` are given in `pinlite.cpp`. The same rules as in Python apply: you may use `mr::adjoint`, `mr::ad`, `mr::trans_inv`, `mr::matrix_exp6`, `mr::vec_to_se3`, `mr::rp_to_trans`, `mr::trans_to_rp`, but not `mr::inverse_dynamics`, `mr::mass_matrix`, … or `pin_rnea` inside Checkpoints 1–4 and 6.

**Checkpoints.** 1: `./build/lab_tests -tc="*spatial_inertia*,*rigid_body*"`; 2: `-tc="*forward*,*link_screw*"`; 3: `-tc="*backward*,*rnea*"`; 4: `-tc="*dynamics_terms*,*tip_wrench*"`; 5: `-tc="*pinocchio*"`; 6: `-tc="*torque_profile*,*peak*"`, then `./build/torque_profiles`; at the end all 18 test cases pass (`./build/lab_tests`). The Python track counts 40 tests because pytest counts each robot of a parametrised test separately; the C++ track checks the same robots in one test case.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| robot dict from `load_robot` | `lab::Robot` struct with the same fields: `r.Mlist`, `r.Glist`, `r.Slist`, `r.g`, `r.n` |
| `load_states` tuples, `load_trajectory` tuple | `lab::State{theta, dtheta, ddtheta, Ftip}`, `lab::Trajectory{t, theta, dtheta, ddtheta}` (trajectories: `Eigen::MatrixXd`, row $k$ = time sample $k$) |
| `forward_iterations` returns a dict | `lab::ForwardPass{A, T, V, Vdot}`: `out.V.col(i)` is Python `out["V"][:, i]`; `out.T` is a `std::vector` of $n+1$ matrices |
| `backward_iterations` returns `(tau, F)`; `dynamics_terms` returns `(M, c, grav)` | `std::pair`: `auto [tau, F] = ...`; `lab::DynamicsTerms{M, c, grav}` |
| `Ftip=None` | `std::optional<Vector6d>`: omitted or `std::nullopt` = None; `*Ftip` is the wrench when it is given |
| `mr.Adjoint`, `mr.ad`, `mr.TransInv`, `mr.MatrixExp6(mr.VecTose3(...))` | `mr::adjoint`, `mr::ad`, `mr::trans_inv`, `mr::matrix_exp6(mr::vec_to_se3(...))` |
| Pinocchio `pinocchio_model`, `pin.rnea`, `pin.Force(linear, angular)` | given `pinlite`: `lab::pinocchio_model`, `lab::pin_rnea(model, q, v, a, fext)`; `fext` is a `std::vector<Vector6d>` with `model.njoints` entries, each **linear first** $(f, m)$, on body $j$, in joint frame $J_j$ — the same three convention changes as in Python |
| plot $\tau(t)$ and $g(\theta(t))$ | `./build/torque_profiles` writes `eod_lift_torques.csv` and `exo_reach_torques.csv` (columns `t, tau1…, grav1…`, using your functions) in the folder you run it from; plot them with Python or any plotting tool |
| `python starter.py` (timing) | `./build/torque_profiles --time`: your RNEA, `mrcpp` and `pinlite`, all compiled C++ (expect a few µs, not hundreds) |

A matrix-size mistake (e.g. adding a 3-vector to a 6-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `Eigen: wrong matrix size` in `forward_iterations` | `V`, `Vdot` not sized $6\times(n+1)$, or `T` without the extra matrix `T[n]` | `MatrixXd::Zero(6, n + 1)`; `out.T` must have $n+1$ entries |
| Wrong values only through `auto` variables | an Eigen expression stored in `auto` | write the type (`Vector6d Vi = out.V.col(i + 1);`) |
| `pinocchio` test wrong for 3-D robots with a tip wrench | angular and linear parts not swapped for `fext` | `fext[n] << F_J.tail<3>(), F_J.head<3>();` (linear first) |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
