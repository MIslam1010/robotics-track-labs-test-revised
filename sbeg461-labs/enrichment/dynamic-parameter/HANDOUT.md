---
title: "Lab — Dynamic Parameter Identification from Hardware Logs"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Enrichment package (from Week 9, after Lab 6 and the midterm)"
---

# Overview

You identify the inertial, rotor and friction parameters of two robots from **simulated hardware logs**: the course
arm (desktop 6R) and the EOD arm 3R. The logs are in the format of your project package (`t, cmd_j, meas_j, cur_j`).
They were simulated with illustrative values, realistic encoder quantisation, current noise, servo current averaging,
friction with a Stribeck bump and clock jitter. You then validate your models on held-out logs and estimate the mass of
an inert training package held by the EOD arm.

Expected time: 8–12 hours over two to three weeks. Read the study guide §1–4 first.

# Pre-lab check (answer before you start)

1. Why is inverse dynamics linear in the inertial parameters, while forward dynamics is not? (guide §2.2)
2. Why do we use $h = mc$ instead of the CoM $c$ as a parameter? (guide §2.1)
3. Give one parameter of the course arm that can never be identified from joint torques, and one pair that can only be
   identified as a sum. (guide §3)
4. Why must the current be filtered with the same filter as the positions? (guide §5.2)
5. Which hardware rules limit an excitation trajectory on the course arm? (guide §4)

# Setup

Your Lab code folder contains:

| File | Content |
|---|---|
| `starter.py` | Your work: 16 functions marked `TODO` in Parts A–G. The provided functions must not be edited. |
| `test_lab.py` | Auto-checks. Run `pytest -q` in your Lab code folder. |
| `robots.py` | Course arm and EOD arm 3R, with the same numbers as Course resources: "Course robot models and hardware safety checklist" |
| `hwlog.py` | Log reader and resampler, the same as the project package |
| `data/` | `course_arm_excite.csv`, `course_arm_validate.csv`, `eod_3r_excite.csv`, `eod_3r_validate.csv`, `eod_3r_package.csv`, `meta.json` (torque constants, excitation coefficients, weighed package mass) |

Use the course Python environment (numpy, scipy, modern_robotics, matplotlib, pytest). Before you start, run
`pytest -q`: the data tests pass and every test of your code fails with `NotImplementedError`.

# Procedure

**Part A — Linear form (guide §2).** Implement `inertial_parameters`, `wrench_regressor` and `regressor`.
*Checkpoint:* `pytest -q -k "inertial or wrench or regressor_matches"` passes. Your $Y\pi$ equals
`mr.InverseDynamics` with general spatial inertias plus the rotor and friction terms.

**Part B — Base parameters (guide §3).** Implement `base_parameters`.
*Checkpoint:* ranks 17 (EOD arm) and 52 (course arm). Print the base names and explain two regroupings in your report.

**Part C — Logs (guide §5).** Implement `process_log` and `moving_rows`. Plot raw and filtered position, velocity and
torque for one joint, as in the filtering figure of the guide (§5.2). Try cut-offs of 2, 4 and 8 Hz and note the effect.

**Part D — Estimation (guide §6, §9).** Implement `ols`, `wls` and `validate`. Use `build_system` to get
$(\mathbf W, \mathbf y, \text{joint})$ from a processed log. Identify both robots on their `_excite` logs and validate on
the `_validate` logs. Compare against the CAD model, $\beta_{CAD} = E\,\pi_{CAD}$ (use `nominal_parameters`).
*Checkpoint:* held-out NRMS below 5 % on every EOD joint and on course-arm joints 2–3, and below 20 % on every
course-arm joint.

**Part E — Excitation (guide §4).** Implement `fourier_trajectory` and `excitation_cost`. Recompute the condition
number of the EOD excitation from the coefficients in `meta.json` and compare it with a single sine.

**Part F — Consistency (guide §7).** Implement `pseudo_inertia`, `is_consistent`, `log_cholesky_to_pi` and
`consistent_fit`. Check the minimum-norm completion $E^{+}\hat\beta$, then fit a consistent model. Prior: the CAD model
plus your own guesses for rotor inertia and friction. Compare the validation NRMS with WLS.

**Part G — Payload (guide §10, WE6).** Implement `payload_estimate` and estimate the training package's mass from
`eod_3r_package.csv`. Compare it with the weighed mass in `meta.json`.

**Expected values** (with the default settings; small differences in the last digit are fine):

| Quantity | EOD arm 3R | Course arm |
|---|---|---|
| Base parameters | 17 | 52 |
| Rows kept (excitation log) | 5300 of 8697 | 9999 of 14 394 |
| WLS held-out NRMS, joint 2 | 0.013 | 0.012 |
| Package mass | 1.221 kg | — |

# Deliverables

1. Your completed `starter.py` (all tests pass) and a short script that makes your report figures.
2. The identification report (rubric Part B), 4–6 pages.
3. A reflection paragraph: what limited the accuracy most, and what you would change for a real hardware run.
4. **AI-use note:** which tools you used, the prompts, and what you kept. You must be able to explain every line in the
   oral check.

# Troubleshooting

* *`regressor` test fails only for the course arm.* Check the order of the product $T_{ij} = T_{i,i-1}\cdots T_{j+1,j}$ and
  use the transpose of the adjoint for wrenches.
* *Rank 53 or more.* Scale the columns before the QR, and use a relative tolerance on $|R_{kk}|$.
* *Viscous friction far too high.* You kept the rows of slow joints (guide §5.4).
* *Huge velocities.* You differentiated before filtering.
* *`consistent_fit` runs for minutes.* Reduce the data term with a QR factorisation (see the docstring) and give
  `least_squares` the bounds $x_0\pm8$.
* *Payload mass far from the weighed mass, or negative.* Your arm model does not match the excitation fit (use $E\hat\pi = \hat\beta$),
  or you left in the rows at rest.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, expected values, report and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `inverse_dynamics` with general spatial inertias, `ad`, `gravity_forces`), with the rigid-body equation, finite differences, synthetic data, the weighed package mass and the numbers of the Python reference (including the random inputs of `test_lab.py` and SciPy's filter output for one log), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work: the 16 functions of Parts A–G), `lab.hpp` (declarations, data structures and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`, and the given code (do not edit): `provided.cpp` (the provided functions of `starter.py`: `load_meta`, `load_robot`, `n_params`, `param_names`, `inertia_matrix`, `spatial_inertia`, `nominal_parameters`, `link_frames_motion`, `stack_regressor`, `joint_index`, `stack_torque`, `build_system`, `pi_to_log_cholesky`, `random_states`), `robots.cpp` (= `robots.py`), `hwlog.cpp` (= `hwlog.py`: `load_log`, `resample`, `rms`) and `helpers.cpp` (the NumPy/SciPy calls of the Python lab, see the table below). The logs and `meta.json` are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). SciPy is not needed for the C++ track; Python with matplotlib is used only for the figures.

**Build and run** (from your Lab code folder):

```bash
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*inertial*,*wrench*,*regressor_matches*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 19 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The whole test run takes about one second once your code is complete (the Python tests take about 15 s).

**Functions to write** (in `starter.cpp`, same order as in Python): `inertial_parameters`, `wrench_regressor`, `regressor` (Part A); `base_parameters` (B); `process_log`, `moving_rows` (C); `ols`, `wls`, `validate` (D); `fourier_trajectory`, `excitation_cost` (E); `pseudo_inertia`, `is_consistent`, `log_cholesky_to_pi`, `consistent_fit` (F); `payload_estimate` (G). Everything else is given.

**Checkpoints.** A: `-tc="*inertial*,*wrench*,*regressor_matches*"`; B: `-tc="*base_parameters*"`; C: `-tc="*process_log*,*moving_rows*"`; D: `-tc="*ols*,*wls*,*validate*,*identified*"`; E: `-tc="*fourier*,*excitation*"`; F: `-tc="*pseudo_inertia*,*log_cholesky*,*consistent_fit*"`; G: `-tc="*payload*"`; at the end all 19 test cases pass. The two data tests (`test_meta_and_logs`, `test_robot_models_are_the_course_models`) also call `process_log` and `inertial_parameters` (C++ only), so they pass once Parts A and C are done. The Python track counts 22 tests: pytest counts the two robots of `test_regressor_matches_modern_robotics` and of `test_base_parameters` separately (the C++ track checks both robots in one test case each), and `test_starter_compiles` has no C++ counterpart (building is the check).

**Expected values.** The table of the procedure holds for C++ (17 and 52 base parameters, the same rows kept, held-out NRMS of joint 2 about 0.013 and 0.012, package mass 1.221 kg). Two numbers may differ from a Python run, because the base parameters are chosen differently (see the table below): which of the regrouped parameters are named in the base set, and the condition number of the weighted course-arm base regressor (about 490 in C++, about 420 in Python; the test accepts 200–5000). Predictions, NRMS and the payload mass do not depend on that choice.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| robot dict, log dict, result dicts (`process_log`, `base_parameters`, `ols`, `wls`, `validate`, `consistent_fit`, `payload_estimate`) | structs with the same field names: `lab::Robot`, `HwLog`, `ProcessedLog`, `BaseParams`, `OlsResult`, `WlsResult`, `Validation`, `ConsistentFit`, `Payload` |
| tuples `(q, dq, ddq)`, `(A, T, V, dV)`, `(Y, tau, joint)` | `lab::Trajectory`, `LinkFrames`, `LinearSystem` |
| `(N, n)` arrays, index arrays, row mask | `Eigen::MatrixXd` with one row per sample; `std::vector<int>` (0-based); `joint` is an `Eigen::VectorXi`; `std::vector<bool>` |
| `link=None`, `sigma=None`, `armature=None` | `link = -1`; an empty `Eigen::VectorXd()` |
| `np.linalg.lstsq`, `np.linalg.cond`, `np.sign`, `np.gradient` | `lab::lstsq` (minimum-norm, SVD), `lab::cond`, `lab::sign`, `lab::gradient` (given) |
| `signal.butter`, `signal.filtfilt` | `lab::butter`, `lab::filtfilt` (given; same coefficients and edge handling as SciPy, checked against it) |
| `scipy.linalg.qr(..., pivoting=True)` | `Eigen::ColPivHouseholderQR` (hint in `starter.cpp`): the same rank, but among dependent columns its pivoting may keep a different one as base parameter |
| `random_states` with NumPy's `default_rng` | `random_states` with `std::mt19937_64` (the same on every platform, not the same numbers as NumPy) |
| `scipy.optimize.least_squares(..., jac=, bounds=)` | `lab::least_squares(fun, jac, x0, lb, ub)` (given): bounded Gauss-Newton/Levenberg–Marquardt with SciPy's scaling near the bounds and its default tolerances; `cost` and `nfev` differ from SciPy, the fitted model passes the same test |
| `np.linalg.eigvalsh` | `Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>` |
| `mr.InverseDynamics`, `mr.ad`, `mr.Adjoint`, `mr.VecToso3`, `np.linalg.pinv` | `mr::inverse_dynamics`, `mr::ad`, `mr::adjoint`, `mr::vec_to_so3`, `mr::pinv` |
| figures with matplotlib | write the data with `sbeg::write_csv(path, header, matrix)` (in `<sbeg/labio.hpp>`) and plot them with `python plot_results.py <file.csv>` (given) or your own script |

A matrix-size mistake does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `regressor` wrong only for the course arm | order of the product $T_{ij}$, or `mr::adjoint(Tij)` not transposed | see the Python troubleshooting list |
| rank 53 or more | columns not scaled before the QR, or an absolute tolerance on $\lvert R_{kk}\rvert$ | divide each column by its norm; compare with `tol * std::abs(R(0, 0))` |
| `process_log` differs from the Python reference in `test_process_log_on_a_clean_sine` | grid not built like `np.arange` (number of points `ceil((t_end - t0) / (1.0 / fs))`), currents not filtered, or decimation before trimming | build the grid first, filter both, trim, then keep every `decimate`-th sample |
| `consistent_fit` slow | the data term uses all rows | reduce it with the QR factorisation (see `lab.hpp`) |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
