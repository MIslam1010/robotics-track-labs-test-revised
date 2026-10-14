---
title: "Lab 5 — Navigation pipeline: pivot calibration, paired-point registration and TRE (prediction vs Monte Carlo vs measurement)"
subtitle: "SBEG 46X Medical Robotics — Week 5"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class (course-arm digitising for groups with a hardware slot, simulation fallback for the others) + about 2 h of independent work |
| CLOs | **46X-4** (calibrate tools and tracking systems, perform point-based rigid registration for image-guided interventions, and predict and measure navigation accuracy: FLE, FRE, TRE); 46X-1 (review: course-arm kinematics, Week 1) |
| Reading | MR §3.3.1 (pp. 87–95: transformation matrices, MR Eqs. 3.62 and 3.64; Example 3.19, pp. 93–94); MR Eq. 4.14 (space-form PoE, review); Corke §14.3.9 (pp. 420–422: matching 3-D point sets with the SVD, ICP). Not in the course library (†): Arun, Huang & Blostein 1987 (SVD registration); Fitzpatrick, West & Maurer 1998 (TRE prediction) |
| Files (your Lab code folder) | `starter.py` (your code: 14 functions marked TODO A1 … D2), `test_lab.py` (auto-checks, 20 tests), `course_models.py` (the course arm, numbers identical to Course resources: Course robot models and hardware safety checklist), `data/phantom.json`, `data/pivot_joints.csv`, `data/fiducial_joints.csv`, `data/check_joints.csv` (simulated bench logs for the simulation fallback) |
| Library | The finished functions become module `registration` of your course library `sbeg46x_medrob` (after `rcm`, `teleop` and `haptics`) |
| Assessment | Auto-checked code (20 tests) + results table + three plots + reflection (one auto-checked lab in the Labs component; not a formal report) |
| AI use | Allowed for code **with disclosure** (AI-use note in the reflection); you must be able to explain every line in an oral check. Derivations are your own |

## Why this lab matters

Every image-guided robot — a neurosurgical biopsy arm, an orthopaedic drilling robot, a field-hospital system that
removes fragments under CT guidance — rests on the same three steps that you practised by hand in the workshop:
**calibrate** the tool (where is the tip?), **register** the patient to the image (where is the image in the robot's
frame?), and **know your error** (how far from the planned target can the tool really be?). Today you build that pipeline
for the course arm. The arm itself is your "tracker": its forward kinematics (SBEG 361 Lab 5, reused in Lab 1 of this course) gives the flange pose, a
3-D-printed pointer on the flange is calibrated by pivoting, the pointer touches six fiducial divots on a head phantom,
and the SVD registration maps the CT plan into the arm's base frame. You then predict the target registration error
with the Fitzpatrick–West–Maurer formula, check the prediction by Monte Carlo, and **measure** the error at two check
divots that were not used in the registration. The same functions are used in project scenarios 2 (image-guided needle
targeting), 4 (EOD keep-out region registered to the arm) and 5 (field telesurgery port and target registration).

## Frames, data and units (illustrative values)

| Item | Value | Source |
|---|---|---|
| Robot | course arm, flange frame $\{b\}$, base frame $\{s\}$; PoE model in `course_models.py` (metres) | Course resources (same numbers as SBEG 361/461) |
| Pointer | 3-D-printed, nominal length 70 mm along $\hat z_b$; true offset unknown (find it) | `phantom.json` |
| Encoders | 12 bit (0.088° per count); every logged angle is a whole number of counts | Course resources |
| Phantom | head phantom with six fiducial divots and two check divots; positions localised in a CT scan (voxel 0.5 × 0.5 × 0.6 mm) | `phantom.json` (image frame $\{I\}$, mm) |
| Targets | `T1_central`, `T2_lateral` (outside the fiducial cloud), `T3_deep` | `phantom.json` |
| Logs | joint angles (rad) recorded while the pointer tip sat in a divot: 12 pivot poses, 6 fiducials, 2 check points | `data/*.csv` |

All positions in the lab code are in **millimetres** (`flange_pose_mm` converts the PoE result). Notation follows MR:
$T_{ab}$ is frame $\{b\}$ seen from $\{a\}$. Pivot: $R_ip_\text{tip} + p_i = p_\text{piv}$. Registration: $y_k \approx Rx_k + t$ with
$x_k$ in $\{I\}$ and $y_k$ in $\{s\}$, so $T_{sI} = (R, t)$.

## Learning objectives

By the end of the lab you can:

1. calibrate a pointer by pivoting with a robot or a tracker, and explain how the tilt range conditions the result;
2. implement SVD point registration with the reflection fix and verify it against an independent method;
3. compute FRE, estimate FLE from it, and predict TRE anywhere with the Fitzpatrick–West–Maurer formula;
4. confirm the prediction by Monte-Carlo simulation and show that FRE does not predict TRE;
5. choose fiducial configurations, and measure TRE at check points with the complete pipeline.

# Pre-lab check (5 questions, submit before the session)

1. The camera sees the robot-base marker ($T_{CR}$) and the patient reference ($T_{CP}$); registration gives $T_{PI}$.
   Write $T_{RI}$.
2. A pointer is pivoted with rotations about its own shaft only. Which component of the tip offset is lost, and why?
3. Why can the plain SVD formula $R = VU^{\mathsf T}$ return a reflection, and how is it fixed?
4. Six fiducials with FLE = 0.5 mm: expected FRE? TRE at the centroid?
5. Your registration shows FRE = 0.1 mm, much lower than expected. Give two possible explanations, one harmless and
   one dangerous.

# Setup (5 min)

1. Copy your Lab code folder; work in `starter.py` or a copy with your own name (e.g. `registration.py`).
2. Run `LAB_IMPL=starter python -m pytest -q test_lab.py`: 3 data tests pass and 17 fail with `NotImplementedError`.
   Re-run after every function; the test names tell you which function is being checked.
3. **Hardware groups only:** read the course-arm section of Course resources: Course robot models and hardware safety
   checklist with the TA (rules 1–7). You digitise with the arm in **latched stop with torque disabled (free-move
   mode)** and guide the pointer by hand: this is the only situation in which a hand may be inside the keep-out zone
   (rule 6), and only with the safety officer's approval. Never enable torque while a hand is on the arm; the emergency stop is tested before
   the session (rule 7). The station script logs the encoder angles in the format of the `data/*.csv` files.

# Procedure

### Checkpoint 1 — Pointer calibration with the arm (A1–A4, ≈ 12 min)

Implement `flange_pose_mm`, `pivot_calibration`, `pivot_from_joint_log` and `tip_position_mm`.

* E1. Check `flange_pose_mm(np.zeros(6))` against the home pose $M$ of the course arm (in mm).
* E2. Calibrate the pointer with all 12 poses of `data/pivot_joints.csv`. Report $p_\text{tip}$, its length, $p_\text{piv}$ and the
  RMS residual. Is the pointer straight?
* E3. Repeat with the 15° poses only (rows 1–6) and with the 30° poses only (rows 7–12). Which has the smaller residual,
  and which result do you trust more? Explain with the lever-arm argument of Workshop Problem 2.
* **Hardware groups:** pivot your own pointer in the table divot (≥ 10 poses, tilts 15–30° in all directions, spin the
  pointer between poses) and log the angles. Compare with the simulated result.

### Checkpoint 2 — Paired-point registration (B1–B3, ≈ 10 min)

Implement `register_points`, `apply_T` and `fre`.

* E4. Digitise the six fiducials (`data/fiducial_joints.csv` with your $p_\text{tip}$) and register the CT fiducials to them.
  Report $T_{sI}$, the rotation angle (`mr.MatrixLog3`), the residual length of every fiducial and the FRE.
* E5. Remove the reflection fix (temporarily) and register a mirrored coplanar point set (test
  `test_B1_no_reflection_for_coplanar_points`). What does the plain formula return?
* **Hardware groups:** digitise the six divots of the printed phantom (arm in free-move mode) and register.

### Checkpoint 3 — Predicting and simulating the error (C1–C5, ≈ 15 min)

Implement `fre_expected`, `fle_from_fre`, `tre_predict`, `monte_carlo` and `best_subset`.

* E6. Estimate the FLE from your FRE. Predict the RMS TRE at the three targets.
* E7. Run `monte_carlo` (5000 trials, `seed=0`, FLE from E6, all three targets). Compare the RMS TRE with E6; give the 95th
  percentile; plot the FRE–TRE scatter for `T2_lateral` and give the correlation coefficient.
* E8. Plot a TRE map: predicted TRE on a grid in the plane $z_I = 10$ mm ($x_I$ from −60 to 180, $y_I$ from −60 to 140), with the
  fiducials overlaid. Where is it smallest, and how does it grow?
* E9. Fiducial configuration: for `T2_lateral`, find the best 3, 4 and 5 of the six fiducials. Which fiducials matter for a
  lateral target, and why?

### Checkpoint 4 — Frame chain and full pipeline (D1–D2, ≈ 8 min)

Implement `chain_robot_from_image` and `navigation_pipeline`.

* E10. Run the pipeline. Report the targets in the arm's base frame and the **measured** errors at the two check divots.
  Compare them with the prediction: the measured error also contains the check divot's own localisation errors (in CT
  and with the arm), so compare with $\sqrt{\mathrm{TRE}^2 + \mathrm{FLE}^2}$.
* E11. Use `chain_robot_from_image` for the optical-tracker version of the same set-up (numbers of Workshop Problem 1) and
  check the target position by hand.

# Deliverables

1. Your module passing all 20 tests (`LAB_IMPL=<your module> pytest -q test_lab.py`).
2. A results table with: E2 $p_\text{tip}$, length, $p_\text{piv}$, residual; E3 residuals and tip lengths; E4 FRE and residuals;
   E6 FLE estimate and TRE predictions; E7 Monte-Carlo RMS and 95th percentiles; E9 best subsets; E10 measured check-point
   errors and their prediction (and your hardware values if you had the slot).
3. Three plots: the FRE–TRE scatter (E7); the TRE map with the fiducials (E8); a 3-D plot of the digitised fiducials, the
   registered CT fiducials and the targets in the base frame (E10).
4. Reflection (≤ 200 words): (a) one sentence each on why a small FRE is not reassuring, why the target should sit inside
   the fiducial cloud, and which error the TRE model leaves out (hint: it assumes independent, zero-mean FLE — what
   about a joint-zero error of the arm?); (b) one result that surprised you; (c) **AI-use note**: tool, what you asked,
   what you kept, what you checked yourself (or "no AI used").

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| tip length ≈ 70 m or 0.07 mm | units: the PoE model is in metres; scale **only the position** of $T$ to mm |
| `pivot_calibration` returns nonsense with small residual | the sign of the right-hand side: $[R_i\ \ -I]\,[p_\text{tip};\,p_\text{piv}] = -p_i$ |
| registration rotates the wrong way | you built $H = \sum\tilde y\tilde x^{\mathsf T}$ (transposed) — this returns $R^{\mathsf T}$ |
| det R = −1 | missing reflection fix: $R = V\,\mathrm{diag}(1, 1, \det VU^{\mathsf T})U^{\mathsf T}$ |
| translation wrong by a few mm | $t = \bar y - R\bar x$ (not $\bar y - \bar x$) |
| `tre_predict` too large by $\sqrt3$ or $N$ | $f_k^2$ is the **mean** squared distance from axis $k$; the factor is $\tfrac13\sum d_k^2/f_k^2$ |
| Monte-Carlo TRE ≈ $\sqrt3$ × prediction | FLE is the RMS **magnitude**: use std FLE/$\sqrt3$ per axis |
| `best_subset` slow | there are only $\binom{6}{k} \le 20$ subsets; do not re-register, just call `tre_predict` |
| hardware: residual of several mm | pointer loose on the flange, worn divot, or the arm moved under gravity when torque was disabled — re-tighten and re-pivot |

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| E1 | home flange position | (206.0, 109.0, 56.1) mm |
| E2 | $p_\text{tip}$; length; $p_\text{piv}$; RMS residual | (1.21, −0.86, 70.48) mm; 70.50 mm; (159.99, 60.01, 39.87) mm; 0.20 mm |
| E3 | 15° poses: residual, length / 30° poses: residual, length | 0.13 mm, 70.29 mm / 0.25 mm, 70.49 mm |
| E4 | rotation angle of $T_{sI}$; $t$; FRE | 20.1°; (109.98, −30.16, 15.01) mm; 0.35 mm |
| E6 | FLE estimate; TRE at T1, T2, T3 | 0.43 mm; 0.18, 0.32, 0.19 mm |
| E7 | Monte-Carlo RMS TRE (T1, T2, T3); 95th percentiles; corr(FRE, TRE at T2) | 0.18, 0.32, 0.20 mm; 0.28, 0.53, 0.32 mm; ≈ 0 (−0.02) |
| E9 | best subsets for T2 (fiducials numbered 1–6) | 3: {2, 3, 4} 0.38 mm; 4: {1, 2, 3, 4} 0.35 mm; 5: {1, 2, 3, 4, 6} 0.34 mm; all six 0.32 mm |
| E10 | measured check-point errors; prediction $\sqrt{\mathrm{TRE}^2 + \mathrm{FLE}^2}$ | 0.43 and 0.23 mm; 0.47 and 0.48 mm |
| E11 | target of Workshop Problem 1 in $\{R\}$ | (190, −140, 60) mm |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, results table and deliverables are the same (deliverable 1 is `cpp/starter.cpp` instead of `starter.py`); the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the same independent references as `test_lab.py` (standard-DH kinematics, Horn's quaternion registration, the eigenvalue form of the TRE formula, brute force) and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided helpers: `load_phantom`, `load_joints`, `rot_to_T`, the random generator `lab::Rng` and `lab::combinations` — do not change), `course_models.hpp` / `course_models.cpp` (the course arm, numbers identical to `course_models.py` — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random inputs of `test_lab.py`), `plot_results.py` (the three plots), `CMakeLists.txt`. The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 5 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*A1*,*A2*"              # only some tests (like pytest -k)
python cpp/plot_results.py                     # plots of E7, E8 and E10 (after test_D2 passes)
```

With the untouched starter, the program builds and all 20 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python, 14 functions): `flange_pose_mm`, `pivot_calibration`, `pivot_from_joint_log`, `tip_position_mm` (A1–A4); `register_points`, `apply_T` (two overloads: one point, and N × 3 points), `fre` (B1–B3); `fre_expected`, `fle_from_fre`, `tre_predict`, `monte_carlo`, `best_subset` (C1–C5); `chain_robot_from_image`, `navigation_pipeline` (D1–D2). The helpers in `helpers.cpp` and the course arm in `course_models.hpp` are given, as the provided section of `starter.py` and `course_models.py`. You may call `mrcpp` (`mr::fkin_space`, `mr::trans_inv`, `mr::rp_to_trans`) where `starter.py` may call `modern_robotics`.

**Checkpoints.** 1: `./build/lab_tests -tc="*course_model*,*logged*,*A1*,*A2*,*A3*,*A4*"`; 2: `-tc="*data_files*,*B1*,*B2*,*B3*"`; 3: `-tc="*C1*,*C3*,*C4*,*C5*"`; 4: `-tc="*D1*,*D2*"`; finally all 20 test cases pass (`./build/lab_tests`), the same 20 tests as in Python. In C++ the three data tests also call one of your functions (`test_data_files_load` calls `apply_T`; `test_course_model_matches_dh_table` and `test_logged_angles_are_encoder_counts_inside_soft_limits` call `flange_pose_mm`), so they pass only once those functions are written; and three tests check one extra C++ case each (`std::invalid_argument` for unequal pose lists in `pivot_calibration`, fewer than 3 points in `register_points`, a 3 × N matrix in `apply_T`).

**Printing the values of the results table.** Add a temporary `TEST_CASE` at the end of `test_lab.cpp` (include `<iostream>`; remove it before you submit), for example:

```cpp
TEST_CASE("my_results") {
    const lab::PipelineResult out = lab::navigation_pipeline(5000, 0);         // E2, E4, E6, E10
    std::cout << "p_tip " << out.p_tip.transpose() << "  |p_tip| " << out.p_tip.norm()
              << "  rms " << out.pivot_rms << "\nFRE " << out.fre << "  FLE " << out.fle_est << "\n";
    for (const auto& kv : out.tre_pred) std::cout << kv.first << " TRE " << kv.second << "\n";
    std::cout << "check errors " << out.check_err.transpose() << "\n";
    const Eigen::MatrixXd Q = lab::load_joints("pivot_joints.csv").second;     // E3: the 15 deg poses
    std::cout << "15 deg: " << lab::pivot_from_joint_log(Q.topRows(6)).rms << "\n";
    const lab::Phantom ph = lab::load_phantom();                               // E9
    const auto best = lab::best_subset(ph.fiducials_image_mm, ph.targets_image_mm.at("T2_lateral"), 3, out.fle_est);
    for (int i : best.first) std::cout << i + 1 << " ";                        // fiducials numbered 1-6
    std::cout << " TRE " << best.second << "\n";
}
```

For the plots, `./build/lab_tests` writes `lab5_cpp_mc.csv` (E7: 5000 Monte-Carlo trials, `seed = 0`, FLE estimate of E6, all three targets), `lab5_cpp_tre_map.csv` (E8: the predicted TRE on the grid of E8) and `lab5_cpp_points.csv` (E10: digitised fiducials, registered CT fiducials and targets in $\{s\}$) into the folder you run it from, once `test_D2_navigation_pipeline` passes; `python cpp/plot_results.py` saves `lab5_E7_fre_tre_scatter.png`, `lab5_E8_tre_map.png` and `lab5_E10_pipeline_3d.png` and prints the RMS values, 95th percentiles and the correlation coefficient of E7.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| N × 3 point arrays; a point `(3,)` | `Eigen::MatrixXd` with one **row** per point; a point is an `Eigen::Vector3d` (`X.row(i).transpose()`) |
| `apply_T(T, P)` for `(3,)` or N × 3 | two overloads: `apply_T(T, Vector3d)` and `apply_T(T, MatrixXd)` (pass a named `Vector3d` or `MatrixXd`, not an expression such as `X.row(0)`) |
| tuples `(p_tip, p_piv, rms)`, `(fre_samples, tre_samples)`, `(indices, tre)` | `lab::PivotResult` (fields `p_tip, p_piv, rms`), `lab::MonteCarloResult` (`fre_samples, tre_samples`), `std::pair<std::vector<int>, double>` |
| `load_joints` → `(labels, Q)`; `load_phantom()` dict | `std::pair<std::vector<std::string>, Eigen::MatrixXd>` (`.second` is Q); `lab::Phantom` (fields named as the JSON keys; `targets_image_mm` is a `std::map`) |
| result dict of `navigation_pipeline` | `lab::PipelineResult` (same names as fields: `out.T_sI`, `out.tre_pred.at("T2_lateral")`, ...) |
| `course_models.course_arm_kinematics()` dict, `cm.Q_HOME` | `cm::course_arm_kinematics()` (fields `Slist, M, limits, qdot_max`), `cm::q_home()`; constants `cm::H1`, `cm::ENC_RES`, ... |
| `np.random.default_rng(seed)`, `rng.normal(0, s, X.shape)` | `lab::Rng rng(seed)`, `rng.normal(0.0, s, X.rows(), 3)` (filled row by row). The numbers differ from NumPy's, so your Monte-Carlo samples differ from the Python ones; `test_C4` and `test_D2` are statistical in both tracks and use the same tolerances. Expect E7 values that differ from the Python reference in the second decimal |
| `monte_carlo(X, targets)` with targets `(3,)` or M × 3 | targets always M × 3 (one target = a 1 × 3 matrix) |
| `itertools.combinations(range(N), k)` | `lab::combinations(N, k)` (same order) |
| `np.linalg.lstsq`, `np.linalg.svd`, `np.linalg.eigh` | `A.completeOrthogonalDecomposition().solve(b)`, `Eigen::JacobiSVD` (`svd.matrixV()` is $V$, not $V^{\mathsf T}$), `Eigen::SelfAdjointEigenSolver` (eigenvalues ascending, as `eigh`) |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| plots with Matplotlib | CSV files written by `lab_tests` + the given `plot_results.py` |

A matrix-size mistake (e.g. points stored as columns, 3 × N, instead of rows) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `call to 'apply_T' is ambiguous` | an Eigen expression (e.g. `X.row(i).transpose()`) passed to `apply_T` | store it first: `const Eigen::Vector3d p = X.row(i).transpose();` |
| `fre_expected` returns `fle_rms` | integer division: `2 / N == 0` in C++ | write `2.0 / N` |
| registration off, det R = −1, or $R^{\mathsf T}$ returned | `svd.matrixV()` treated as $V^{\mathsf T}$, or $H$ built as $\sum \tilde y\tilde x^{\mathsf T}$ | $R = V\,\mathrm{diag}(1, 1, d)\,U^{\mathsf T}$ with `V = svd.matrixV()` |
| `best_subset` returns a different subset with the same TRE | strict `<` without the `1e-12` margin, or own subset order | loop over `lab::combinations` and replace only if smaller by more than `1e-12` |
| `tre_mc` far from `tre_pred` | noise standard deviation `fle_rms` per axis | use `fle_rms / std::sqrt(3.0)` per axis |
| no `lab5_cpp_*.csv` files | `test_D2` does not pass yet, or you looked in another folder | they are written into the folder you run `lab_tests` from |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
