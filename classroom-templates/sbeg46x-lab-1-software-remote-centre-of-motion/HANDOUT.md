---
title: "Lab 1 — Software remote centre of motion: resolved-rate control of the course arm with a laparoscope"
subtitle: "SBEG 46X Medical Robotics — Week 1"
---

# Overview

A laparoscopic instrument enters the patient through a small incision (the **port** or trocar). Whatever the robot does
outside the body, the instrument shaft must keep passing through the port, or it tears the abdominal wall. Surgical arms
obtain this **remote centre of motion (RCM)** either mechanically (a double parallelogram, Workshop 1 Problem 1) or in
software, by constraining the inverse kinematics of a general-purpose arm. In this lab you implement the **software RCM**
on the course arm — the desktop UR-type 6R arm you used in SBEG 361 and SBEG 461 — holding a straight 0.12 m teaching
laparoscope, and you compare it with a naive controller that only tracks the tip.

Everything runs in **simulation** (no hardware this week). The lab also sets up your environment for the whole course:
you reload your SBEG 361 and SBEG 461 libraries and start the course library `sbeg46x_medrob`, whose first module,
`rcm.py`, is the code you write today. Later labs and the project (for example the field-telesurgery and needle-targeting
scenarios) reuse it.

**CLO:** 46X-1 (mechanical and software RCM constraints, port placement); first contact with 46X-2 (motion scaling).
**Time:** 50 min in class (Checkpoints 0–4) + about 2 h at home (Checkpoint 5 and the lab note).
**Reading:** *Modern Robotics* (MR) §5.1 (space Jacobian, MR Eq. 5.11, p. 178), the velocity of a body point
$\dot p = v_s + \omega_s\times p$ (from MR Eq. 3.73, p. 97), inverse velocity kinematics MR §6.3 (pp. 230–232), MR Eq. 6.7 (p. 231).

## Learning objectives

By the end of the lab you can:

1. load your own kinematics library and check it against `modern_robotics`;
2. compute the port geometry of a tool (insertion depth, closest shaft point, RCM error) from $T_{sb}$;
3. build the $5\times6$ task Jacobian of a software RCM (tip rows + two lateral port rows) from the space Jacobian;
4. implement constrained resolved-rate control with feedback, damped least squares and uniform rate limiting;
5. show quantitatively why tip-only control is unsafe in minimally invasive surgery;
6. compute the fulcrum ratio of a hand-held instrument and a motion-scaled tip command.

## The robot and the task (`course_models.py`, `data/lab1_task.json`, illustrative values)

* **Robot:** the course arm (PoE model of MR Example 4.5 form, $W_1 = 0.0634$, $W_2 = 0.0456$, $L_1 = 0.110$,
  $L_2 = 0.096$, $H_1 = 0.1312$, $H_2 = 0.0751$ m) with a straight tool of length $0.12$ m along $\hat z_b$. The tool frame
  $\{b\}$ is at the laparoscope **tip**, $\hat z_b$ along the shaft towards the tip. Robot data: "Course resources: Course
  robot models and hardware safety checklist"; the lab's own copy is `course_models.py` (numbers identical).
* **Limits:** software joint limits 10° inside the hardware limits (±155°, J6 ±165°) and joint speeds limited to 25 % of
  the rated 160°/s (40°/s), as the safety checklist requires for the first run of any new motion — so that the same code
  could later be run on the arm.
* **Port:** $p_\text{port} = (0.15,\ 0.08,\ 0.05)$ m. At the start configuration $\theta_0$ the shaft points straight down and
  the tip is 30 mm beyond the port.
* **Tip path** (port frame $\{p\}$: origin at the port, $\hat z_p$ along the initial shaft, so "depth" is $z_p$):
  insert to 50 mm (2 s) → move 10 mm sideways (1.5 s) → one 10 mm-radius inspection circle (6 s) → return and retract to
  30 mm (2.5 s). Total 12 s, simulated with $\Delta t = 5$ ms.

## The software-RCM controller

Let $p_\text{tip}$ and $z$ be the tip position and shaft direction (from $T_{sb}$). The shaft point closest to the port is
$$s = z\cdot(p_\text{tip} - p_\text{port}),\qquad p_c = p_\text{tip} - s\,z,\qquad e = p_\text{port} - p_c\ \ (\perp z).$$
With $J_s$ the space Jacobian and $J_p = [-[p]\ \ I]\,J_s$ the Jacobian of the body point currently at $p$, and $N = [n_1\ n_2]$
two unit vectors perpendicular to $z$:
$$\underbrace{\begin{bmatrix} J_{p_\text{tip}}\\ N^{\mathsf T}J_{p_c}\end{bmatrix}}_{A\ (5\times6)}\dot\theta =
\begin{bmatrix}\dot p_d + k_p\,(p_d - p_\text{tip})\\ k_\text{rcm}\,N^{\mathsf T}e\end{bmatrix},\qquad
\dot\theta = A^{\#}(\lambda)\,b,\quad A^{\#}(\lambda) = A^{\mathsf T}(AA^{\mathsf T} + \lambda^2I)^{-1}.$$
The first block tracks the tip (MR Eq. 6.7 with position feedback); the second asks the shaft point at the port to move
only **along** the shaft, and pulls it back if it drifts. Five equations, six joints: the minimum-norm solution resolves
the remaining DOF (mostly roll about the shaft). Gains: $k_p = 5$ s⁻¹, $k_\text{rcm} = 10$ s⁻¹, $\lambda = 0$.

# Pre-lab check (complete before the session, 5 questions, ≈ 10 min)

1. How many DOF does a rigid instrument keep when its shaft must pass through a fixed point? With a 3-D tip-position task,
   how many redundant DOF does a 6-DOF arm have?
2. Why is $\dot p = v_s + \omega_s\times p$ and not $\dot p = v_s$ for a point $p$ of the moving body?
3. The tip is at $(0.62, 0.05, 0.28)$, the shaft direction is $z = (0.6, 0, -0.8)$ and the port is at $(0.58, 0.06, 0.33)$.
   Compute $s$ and $|e|$.
4. Why do we constrain only the two **lateral** components of the port-point velocity, not all three?
5. A hand-held instrument of length 0.36 m is inserted 0.12 m. The hand moves 12 mm right. Where does the tip go?

*Answers (check after you try):* 1. $6 - 2 = 4$; $6 - (3 + 2) = 1$. 2. $v_s$ is the velocity of the body point at the
origin of $\{s\}$ (MR Eq. 3.73); a point at $p$ adds $\omega_s\times p$. 3. $s = 0.064$ m, $|e| = 10.2$ mm. 4. The third
component is sliding along the shaft, i.e. insertion, which must stay free. 5. 6 mm to the left (ratio $-0.5$).

# Setup (Checkpoint 0, ≈ 8 min)

1. **Environment.** Use the course Python environment (Python ≥ 3.11 with NumPy, SciPy, Matplotlib, `modern_robotics`,
   `pytest`; see the syllabus, Section 11). Open a terminal in **your Lab code folder** for Week 1.
2. **Reload your SBEG 361 and SBEG 461 libraries.** `starter.py` imports `fk_space` and `jacobian_space` from your package
   `sbeg361_kinematics` (`fk.py` from SBEG 361 Lab 5, `jacobians.py` from SBEG 361 Lab 7) and checks that
   `sbeg461_dynamics` is importable. Make both importable, e.g.

   ```bash
   pip install -e ~/sbeg361-repo      # the folder that contains sbeg361_kinematics/
   pip install -e ~/sbeg461-repo      # the folder that contains sbeg461_dynamics/
   python starter.py                  # prints versions and the active backends
   ```

   Expected: `kinematics sbeg361_kinematics` and `dynamics sbeg461_dynamics`. If your 361 library is missing or broken,
   the import block falls back to `modern_robotics` and prints `modern_robotics (fallback)`: you can still finish the lab,
   but say so in your lab note and repair the library before Lab 2. The test `test_361_library_matches_modern_robotics`
   compares your library with `modern_robotics` (it is skipped on the fallback). `sbeg461_dynamics` is not used today; keep
   it importable; you may reuse it in project extensions.
3. **Start the course library.** Create a Git repository `sbeg46x-repo` with a package folder `sbeg46x_medrob/`
   (an empty `__init__.py`). When all tests pass, you will copy your functions into `sbeg46x_medrob/rcm.py`.
4. **Run the tests once:**

   ```bash
   LAB_IMPL=starter python -m pytest -q test_lab.py      # all tests fail with NotImplementedError: expected
   ```

   Run only the tests of the current checkpoint with `-k`, e.g. `LAB_IMPL=starter python -m pytest -q test_lab.py -k geometry`.

# Procedure and checkpoints

## Checkpoint 1 — Tool and port geometry (≈ 8 min)

Implement `tool_pose`, `shaft`, `rcm_geometry` and `perp_basis`.

* `tool_pose` calls the backend `fk_space` (MR Eq. 4.14). `shaft` returns $p_\text{tip}$ (position column) and $z$ (third
  column of $R_{sb}$).
* `rcm_geometry` returns `depth` $s$, `p_c`, `e` and `dist` $= |e|$, exactly as in the box above.
* `perp_basis(z)` returns $N = [n_1\ n_2]$, orthonormal, perpendicular to $z$, with $\det[n_1\ n_2\ z] = +1$. Pick a helper
  vector that is not parallel to $z$.

Tests: `-k "tool_pose or start_pose or geometry or perp_basis"`. *Check:* at $\theta_0$ the depth is 30 mm and the RCM
error is below $10^{-6}$ m.

## Checkpoint 2 — The point Jacobian and the RCM task Jacobian (≈ 12 min)

Implement `point_jacobian(Js, p)` $= [-[p]\ \ I]\,J_s$ (use the provided `skew`), `rcm_jacobian` and `dls_inverse`.

* `rcm_jacobian` stacks the tip rows and $N^{\mathsf T}J_{p_c}$, and returns `(A, info)` with `info = dict(T=..., N=..., geometry=...)`.
* `dls_inverse(A, lam)`: `np.linalg.pinv(A)` for `lam == 0`, otherwise $A^{\mathsf T}(AA^{\mathsf T} + \lambda^2I)^{-1}$.

Tests: `-k "point_jacobian or rcm_jacobian or rcm_rows or dls"`. The point-Jacobian tests compare with
$R_{sb}J_b$'s linear rows (tip = origin of $\{b\}$) and with finite differences of the forward kinematics.
*Question for your note:* the test checks that $A$ has rank 5. What does the one-dimensional null space of $A$ do to the
instrument?

## Checkpoint 3 — Controllers and simulation (≈ 12 min)

Implement `rcm_rates`, `tip_only_rates`, `scale_rates` and `simulate`.

* `scale_rates` divides the whole vector by $\max_i|\dot\theta_i|/\dot\theta_{\max,i}$ when that ratio exceeds 1 (Practice
  Problem 5(d) explains why we do not clip joints separately).
* `simulate` loops over $k = 0,\dots,K$: evaluate the desired tip `traj(t)` (provided by `make_trajectory`), log
  `t, theta, p_tip, p_d, tip_err, rcm_err, depth`, then (except at the last step) compute the rates with the chosen
  controller and integrate $\theta \leftarrow \theta + \Delta t\,\text{scale\_rates}(\dot\theta)$. Return a dict of arrays.

Tests: `-k "rates or simulation"`. **Acceptance** (in the task file): with the software RCM the tip error stays below
0.5 mm and the RCM error below 0.5 mm for the whole path; joint limits and speed limits are respected. The reference
implementation reaches about 0.03 mm and 0.005 mm. With tip-only control the tip is tracked just as well, but the shaft
drifts by more than 5 mm at the port (about 13 mm in the reference run).

## Checkpoint 4 — Fulcrum effect and motion scaling (≈ 5 min)

Implement `fulcrum_ratio(L, depth)` $= -\,\text{depth}/(L - \text{depth})$ and `scaled_tip_command(d_leader, scale, R_sv)`
$= s\,R_{sv}\,d_\text{leader}$. Tests: `-k "fulcrum or scaled"`. These two ideas return in Lab 2
(teleoperation with scaling and clutching).

Run the whole file: `LAB_IMPL=starter python -m pytest -q test_lab.py` → all pass (1 test may be skipped on the fallback).

## Checkpoint 5 — Experiments for the lab note (at home)

Write a short script `lab1_experiments.py` (Matplotlib) that imports your module and produces:

1. **Figure 1:** tip error and RCM error (mm) versus time for the software-RCM and the tip-only controllers (two panels,
   logarithmic $y$ axis for the RCM error).
2. **Figure 2:** the six joint rates (°/s) of the software-RCM run with the 40°/s limit drawn as a dashed line.
3. **Table 1:** maximum tip error and maximum RCM error for $k_\text{rcm} \in \{0, 1, 10, 100\}$ s⁻¹ (all other gains
   unchanged). Explain the trend: what does the feedback term correct that the feed-forward constraint alone does not?
4. **Table 2:** maximum RCM error and maximum joint rate for $\lambda \in \{0, 0.01, 0.05\}$. What does damping trade?
5. **Port placement question:** move the port 30 mm in $\hat x_s$ (edit a copy of the task file and recompute $\theta_0$
   with `mr.IKinSpace` so that the shaft points straight down with the tip 30 mm below the new port). Does the path stay
   within the joint limits? Report the smallest singular value of $A$ along the path for both ports.

# Deliverables (by the start of the Week 2 lab: push to your Classroom 50 repo, or `python submit.py 1 --id <ID>` and upload the ZIP)

1. `rcm.py` in `sbeg46x_medrob/` (your completed functions) and the lab folder with your completed starter, passing all
   tests (`LAB_IMPL=starter python -m pytest -q test_lab.py`).
2. `lab1_experiments.py` and its figures and tables.
3. `lab1_notes.md` (≤ 1 page + figures): the answers to the questions in Checkpoints 2 and 5, the backend line printed
   by `python starter.py`, the reflection and the AI-use note below.

# Reflection (≤ 200 words, in `lab1_notes.md`)

* A surgeon asks: "If the software RCM keeps the error below 0.01 mm, why do you still prefer a mechanical RCM for my
  procedure?" Give two arguments for each side (think of software faults, power loss, the patient breathing, and
  flexibility of port placement).
* Where in your code would a failure tear the port? Suggest one runtime check (for example on `dist`) and what the robot
  should do when it fails.
* **AI-use note** (required, course AI-use policy, syllabus Section 8): which AI tool, if any, you used, for what (e.g.
  explaining an error message), what you kept or changed, and how you verified it (tests, comparison with
  `modern_robotics`). If none, write "No AI tools were used." You must be able to explain every line of your code in a
  2–3-minute oral check.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `python starter.py` prints `modern_robotics (fallback)` | Your 361 library is not importable: `pip install -e` the folder that **contains** `sbeg361_kinematics/`, or set `PYTHONPATH`. Check that `fk.py` defines `fk_space(M, Slist, theta)` and `jacobians.py` defines `jacobian_space(Slist, theta)`. |
| `test_361_library_matches_modern_robotics` fails | Your library's FK or Jacobian disagrees with `modern_robotics`: usually the order $(\omega, v)$, the PoE order, or a missing adjoint in $J_{si}$ (MR Eq. 5.11). Fix it in the library, not in this lab. |
| Point-Jacobian tests fail by a sign | $\dot p = v_s + \omega_s\times p = v_s - [p]\,\omega_s$; the matrix is $[-[p]\ \ I]$, applied to $J_s$ (angular rows first). |
| `perp_basis` fails for $z = \pm\hat x$ | Your helper vector is parallel to $z$; switch to another axis when $|z_x|$ is close to 1. |
| RCM error grows slowly during the circle | Missing feedback term $k_\text{rcm}N^{\mathsf T}e$, or $e$ has the wrong sign ($e = p_\text{port} - p_c$). |
| The simulation is slow | Compute $T_{sb}$ and $J_s$ once per step and pass them on; 2401 steps should take a few seconds. |
| Joint-rate test fails | You clipped joints separately or applied the limit before integrating only some joints; scale the whole vector. |
| `make_trajectory` error | Do not edit the provided helpers; check that your `shaft` returns the third column of $R_{sb}$ as $z$. |

*Safety note:* this lab is simulation only. If you later run an RCM motion on the course arm (project hardware sessions),
the course-arm checklist in "Course resources: Course robot models and hardware safety checklist" applies in full: 25 %
speed on the first run, keep-out zone, tested emergency stop and a supervisor present; a soft phantom, never a person, at
the port.

# C++ track

You may do this lab in C++17 instead of Python. The robot, task file, checkpoints, questions and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with finite differences and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers, including the random configurations that `test_lab.py` draws), `CMakeLists.txt`, and the given code `course_models.cpp` (C++ copy of `course_models.py`) and `helpers.cpp` (the provided helpers of `starter.py`: `load_task`, `environment_report`, `skew`, `min_jerk`, `port_frame`, `make_trajectory`, plus `save_log_csv`). The task file is the same `data/lab1_task.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 1 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*tool_pose*,*start_pose*,*geometry*,*perp_basis*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 17 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The whole test run, including both 2401-step simulations, takes well under a second.

**Functions to write** (in `starter.cpp`, same order as in Python, 13 functions): `tool_pose`, `shaft`, `rcm_geometry`, `perp_basis` (Checkpoint 1); `point_jacobian`, `rcm_jacobian`, `dls_inverse` (Checkpoint 2); `rcm_rates`, `tip_only_rates`, `scale_rates`, `simulate` (Checkpoint 3); `fulcrum_ratio`, `scaled_tip_command` (Checkpoint 4). Given: the robot model `course_arm_kinematics`, the helpers listed above, and the kinematics backend `fk_space`, `jacobian_space`, `kinematics_backend` (in `starter.cpp`).

**Kinematics backend (Checkpoint 0).** There is no `pip install -e` of your libraries in C++. In `starter.cpp`, `fk_space` and `jacobian_space` call `mr::fkin_space` / `mr::jacobian_space` (the equivalent of the Python `modern_robotics` fallback). If you did the SBEG 361 labs in C++, replace the two bodies by your own functions (361 Lab 5 FK, 361 Lab 7 space Jacobian) and change the name returned by `kinematics_backend()`; `test_361_library_matches_modern_robotics` checks them (in C++ this test always runs; in Python it is skipped on the fallback). `sbeg461_dynamics` is not needed in this lab. In `lab1_notes.md` write which backend you used ("mrcpp fallback" or "own 361 C++ code") instead of the line printed by `python starter.py`.

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 0 — Your SBEG 361 library | `-tc="*361_library*"` | 1 |
| 1 — Tool and port geometry | `-tc="*tool_pose*,*start_pose*,*geometry*,*perp_basis*"` | 4 |
| 2 — Point Jacobian and RCM task Jacobian | `-tc="*point_jacobian*,*rcm_jacobian*,*rcm_rows*,*dls*"` | 5 |
| 3 — Controllers and simulation | `-tc="*rates*,*simulation*"` | 5 |
| 4 — Fulcrum effect and motion scaling | `-tc="*fulcrum*,*scaled*"` | 2 |

All 17 test cases pass with `./build/lab_tests`. The Python track counts 39 tests because pytest counts each configuration, axis and parameter of the `@pytest.mark.parametrize` tests separately; the C++ track checks the same configurations (the five `THETAS`, stored in `cpp_cases.json`), axes and parameters inside one test case each.

**Checkpoint 5 in C++.** There is no `python starter.py` printout and no Matplotlib in C++: write a small `experiments.cpp` with your own `main()` (build it next to `starter.cpp`, e.g. a second `add_executable` in a copy of the folder) that runs `lab::simulate(robot, task, "rcm", task.dt, kp, k_rcm, lam)` and `"tip"` for the requested gains, prints the maximum errors (`log.tip_err.maxCoeff()`, `log.rcm_err.maxCoeff()`) and joint rates, and writes each run with `lab::save_log_csv("rcm.csv", log)`. Make Figures 1 and 2 from the CSV files with a short Python/Matplotlib script (or any plotting tool). For the port-placement question, edit a copy of the task file, load it with `lab::load_task(path)` and recompute $\theta_0$ with `mr::ikin_space` (from `<sbeg/mrcpp.hpp>`, as `mr.IKinSpace` in Python); the smallest singular value of $A$ is `Eigen::JacobiSVD<Eigen::MatrixXd>(A).singularValues().minCoeff()`.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `robot` dict (`robot["Slist"]`, `robot["M"]`, `robot["limits"]`, `robot["qdot_max"]`) | `lab::Robot` from `lab::course_arm_kinematics()`: `robot.Slist` (6 × 6, axes are the columns), `robot.M`, `robot.limits` (6 × 2), `robot.qdot_max` |
| `task` dict (`task["p_port"]`, `task["gains"]["kp"]`, `task["acceptance"]`) | `lab::Task` from `lab::load_task()`: `task.p_port`, `task.theta0`, `task.path`, `task.dt`, `task.gains.kp`, `task.gains.k_rcm`, `task.gains.lam`, `task.acceptance.max_tip_error_mm` |
| `shaft` returns a tuple; `rcm_jacobian` returns `(A, info)` | `std::pair`: `auto [p_tip, z] = lab::shaft(T);`, `auto [A, info] = lab::rcm_jacobian(...);` |
| `rcm_geometry` returns a dict; `info` dict | `lab::RcmGeometry{depth, p_c, e, dist}`; `lab::RcmInfo{T, N, geometry}` (`info.geometry.e`) |
| `perp_basis` returns a `(3, 2)` array | `lab::Matrix32d` (`Eigen::Matrix<double, 3, 2>`): `N << n1, n2;` |
| `simulate` returns a dict of arrays | `lab::SimLog{t, theta, p_tip, p_d, tip_err, rcm_err, depth}` (one row per step); `raise ValueError` → `throw std::invalid_argument` |
| `make_trajectory` returns `(traj, T_total)` with `traj(t) -> (p_d, pdot_d)` | `auto [traj, T_total] = lab::make_trajectory(task, p_tip0, z0);` then `auto [p_d, v_d] = traj(t);` |
| `scaled_tip_command(d, s, R_sv=None)` | `scaled_tip_command(d, s)` uses the default `R_sv = Eigen::Matrix3d::Identity()` |
| `np.linalg.pinv(A)`, `np.linalg.inv(G)`, `np.eye(n)` | `mr::pinv(A)` (same cut-off as NumPy), `G.inverse()`, `Eigen::MatrixXd::Identity(n, n)` |
| `np.vstack([P, Q])`, `T[:3, 3]`, `T[:3, 2]`, `a @ b`, `np.cross(a, b)` | `Eigen::MatrixXd A(5, 6); A << P, Q;`, `T.block<3, 1>(0, 3)`, `T.block<3, 1>(0, 2)`, `a.dot(b)`, `a.cross(b)` |
| `int(round(T_total / dt))`, `np.max(np.abs(q) / qmax)` | `static_cast<int>(std::lround(T_total / dt))`, `(q.cwiseAbs().array() / qmax.array()).maxCoeff()` |
| `sbeg361_kinematics` import block (fallback `modern_robotics`) | given `lab::fk_space` / `lab::jacobian_space` (mrcpp; replace with your 361 C++ code if you have it) |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

A matrix-size mistake does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../lab1_task.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size" in `rcm_jacobian` | the comma initialiser does not fill the 5 × 6 matrix exactly (3 + 2 rows) | stack `point_jacobian(Js, p_tip)` (3 × 6) and `info.N.transpose() * point_jacobian(Js, p_c)` (2 × 6) |
| Simulation tests fail with wrong shapes | the `SimLog` fields were not sized to $K+1$ rows before the loop | `log.theta.resize(K + 1, 6)` etc., or fill row by row after sizing |
| Wrong results only after `auto J = A * B;` | `auto` keeps an Eigen expression, not a matrix | write the type: `Eigen::MatrixXd J = A * B;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
