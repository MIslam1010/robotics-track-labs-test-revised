---
title: "Lab 3 — Exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation"
subtitle: "SBEG 361 Introduction to Robotics — Week 4"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2.5 h to finish (due before the Week 5 lab) |
| CLOs | 361-2 (represent and convert orientations and poses: SO(3), SE(3), exponential coordinates, quaternions); 361-3 (screw axes and screw motions); 361-7 (implement and validate against a reference library) |
| Files | `starter.py` (you edit), `test_lab.py` (auto-checks), `imu_knee.csv` and `screw_insertion.csv` (data), `make_data.py` (how the data were made) |
| Reference | `modern_robotics` (`VecToso3`, `so3ToVec`, `AxisAng3`, `MatrixExp3`, `MatrixLog3`, `VecTose3`, `MatrixExp6`, `MatrixLog6`, `Adjoint`), `scipy.spatial.transform.Rotation` |
| Reading | MR §3.2.3 (printed pp. 77–87), §3.3.2.2–3.3.3 (pp. 100–106), App. B.3 (pp. 583–584); Corke §3.2.3 *Inertial Navigation Systems* (printed pp. 53–55) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation and recorded data only) |

This lab teaches the exponential and the logarithm **once, for rotations and rigid motions side by side**: every SO(3) function you write has an SE(3) twin that reuses it ($e^{[\mathcal S]\theta}$ needs $e^{[\hat\omega]\theta}$ and $G(\theta)$; $\log T$ needs $\log R$ and $G^{-1}(\theta)$). The functions go into your `sbeg361_kinematics` library; the Week 5 lab (wrenches, force–torque sensing) and the product-of-exponentials labs from Week 6 build on them.

**Application context.**
*Biomedical:* gait laboratories estimate knee flexion from two IMUs on the thigh and shank; in navigated spine surgery the optical tracker follows the screwdriver while a pedicle screw is driven, and the screw motion it records tells the surgeon the actual screw axis and how far the screw has advanced.
*Defence (protective/logistic):* the same two-IMU method estimates the knee angle of a soldier load-carriage exoskeleton during field trials, and an EOD robot's controller needs the orientation error and the full pose error (a screw motion) between the current and the desired gripper pose. The data sets are **synthetic, with illustrative values**.

# Learning objectives

By the end of the lab you can:

1. Implement $[\omega]$, Rodrigues' formula (MR Eq. 3.51) and the SE(3) exponential with $G(\theta)$ (MR Prop. 3.25, Eqs. 3.87–3.89), and validate them against `modern_robotics`.
2. Implement the matrix logarithms of SO(3) (three cases, MR §3.2.3.3) and SE(3) (two cases, MR §3.3.3.2, Eqs. 3.91–3.92), including the special cases $R = I$ and $\theta = \pi$.
3. Find the screw axis $\{q, \hat s, h\}$ and distance $\theta$ that carry one pose to another, in the space and the body frame, and move along it.
4. Convert between rotation matrices and unit quaternions (MR Eqs. B.10–B.12) and compose quaternions (MR Eq. B.15).
5. Integrate body-frame gyroscope data with $R_{k+1} = R_k e^{[\omega_{b,k}]\Delta t}$, explain why the first-order update leaves $SO(3)$, and compute a joint angle and an orientation error with the matrix log.
6. Recover the lead of a tracked screw insertion from noisy poses, and explain why one logarithm of the whole motion cannot see full turns.

# Pre-lab check (complete before the session; 5 questions)

1. Write $[\omega]$ for $\omega = (3,\ 0,\ -1)$.
2. A rotation matrix has trace $2$. What is its rotation angle?
3. Write the screw axis $\mathcal S = (\omega, v)$ of a rotation about an axis along $\hat z$ through $q = (1, 0, 0)$ with pitch $h = 0$ (MR §3.3.2.2).
4. The matrix logarithm returns $\theta \in [0, \pi]$. What does $\log R$ return for a rotation of $370^\circ$ about $\hat z$?
5. Which of $\omega_s$ or $\omega_b$ does a gyroscope strapped to the shank measure? Which MR equation relates it to $\dot R$?

*Answers (for self-check):* 1. $\begin{bmatrix}0&1&0\\-1&0&-3\\0&3&0\end{bmatrix}$. 2. $\cos\theta = \tfrac12$, $\theta = 60^\circ$. 3. $\omega = (0,0,1)$, $v = -\omega\times q + h\omega = (0,-1,0)$, so $\mathcal S = (0,0,1,0,-1,0)$. 4. $[\hat z]\,\theta$ with $\theta = 10^\circ = 0.1745$ rad: full turns are invisible to the log. 5. $\omega_b$; MR Eq. (3.38), $R^{-1}\dot R = [\omega_b]$.

# Setup

```bash
cd "<your Lab code folder>"   # or your Classroom 50 repo
# activate the course Python environment (Docker image, JupyterHub or your own venv)
python -c "import numpy, scipy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py               # 30 failures (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Rules: do **not** call `modern_robotics`, `scipy.spatial.transform` or `scipy.linalg.expm/logm` inside `starter.py`; they are only for checking. The helpers at the top of `starter.py` (`rp_to_trans`, `trans_to_rp`, `trans_inv`, `vec_to_se3`, `se3_to_vec`, `twist_to_screw`) are given and complete; you met them in Labs 1–2. The new helper `load_pose_track` (reads a tracker log into a stack of poses; used in Step 7) is also given.

# Procedure

Run the tests after each step. A checkpoint is complete when all listed tests pass; ask a TA to sign it off.

## Step 1 — Skew-symmetric matrices (5 min)

Implement `vec_to_so3`, `so3_to_vec`, `axis_ang3` (copy `vec_to_so3` / `so3_to_vec` from your Lab 2 code).

**Checkpoint 1:** `test_vec_to_so3_matches_mr`, `test_skew_properties`, `test_axis_ang3`. Be ready to explain why `test_skew_properties` checks $R[a]R^{\mathsf T} = [Ra]$ (MR Prop. 3.8).

## Step 2 — Exponentials side by side: SO(3) and SE(3) (20 min)

a. Implement `matrix_exp3` with Rodrigues' formula (MR Eq. 3.51). Return $I$ when $\theta <$ `TOL`.
b. Implement `g_matrix` (MR Eq. 3.87) and `g_inv_matrix` (MR Eq. 3.92) for a unit axis $\hat\omega$ and $0 < \theta \le \pi$.
c. Implement `matrix_exp6` (MR Prop. 3.25): split $[\mathcal S]\theta$ into $\omega\theta$ and $v\theta$; if $\|\omega\theta\| <$ `TOL` it is a pure translation (Eq. 3.89); otherwise the rotation block is `matrix_exp3` of the top-left block and the translation is $G(\theta)v$ (Eq. 3.88).

Compare the two exponentials side by side in your notebook: the rotation block of $e^{[\mathcal S]\theta}$ *is* $e^{[\hat\omega]\theta}$; only the translation needs the new matrix $G(\theta)$.

**Checkpoint 2:** `test_matrix_exp3_mr_example_3_12` (reproduces MR Example 3.12), `test_matrix_exp3_matches_mr_and_is_rotation`, `test_g_matrix_and_inverse` ($G$ equals its power series, Eq. 3.86, and $G\,G^{-1} = I$), `test_matrix_exp6_matches_mr`, `test_exp6_screw_through_a_point` (axis $\hat z$ through $(1,0,0)$, $\theta = 90^\circ$, pitch $0$ and $0.1$ m/rad: the origin moves to $(1,-1,0)$, plus $0.157$ m along $\hat z$ for the second pitch).

## Step 3 — Logarithms side by side: SO(3) and SE(3) (20 min)

a. Implement `matrix_log3` with all three cases of the MR algorithm (§3.2.3.3). For case (b) try Eq. (3.58) first, then (3.59), then (3.60).
b. Implement `matrix_log6` (§3.3.3.2): call `matrix_log3(R)`; case (a) $R = I$ gives $[\,0\ \ p\,;\,0\ \ 0\,]$ (pure translation, $\theta = \|p\|$); case (b) uses $v = G^{-1}(\theta)p$ (Eq. 3.91). The $\theta = \pi$ case of SE(3) needs nothing new: it is handled by `matrix_log3`.

**Checkpoint 3:** `test_matrix_log3_matches_mr_random`, `test_matrix_log3_identity_and_pi_cases`, `test_exp_log_round_trip_small_and_large_angles`, `test_matrix_log6_matches_mr_and_inverts_exp`, `test_matrix_log6_special_cases`, `test_log6_mr_example_3_26` (MR Example 3.26: $\mathcal S = (0, 0, 1, 3.37, -3.37, 0)$, $\theta = \pi/6 = 0.524$ rad).
*Question for your report:* the round-trip test includes $\theta = 10^{-5}$ rad and $\theta = \pi - 10^{-4}$ rad. Which case of your SO(3) algorithm handles each, and how large is the round-trip error? (Print it.) Why does `matrix_log6` not need its own $\theta = \pi$ case?

## Step 4 — Screw motions (10 min)

Implement `screw_between(T1, T2, frame)`: the unit screw axis $\mathcal S$ (MR Def. 3.24) and distance $\theta$ with $T_2 = e^{[\mathcal S]\theta}T_1$ (`"space"`, log of $T_2T_1^{-1}$) or $T_2 = T_1e^{[\mathcal S]\theta}$ (`"body"`, log of $T_1^{-1}T_2$). Then implement `screw_path(T1, T2, s)` $= e^{[\mathcal S]s\theta}T_1$, the pose a fraction $s$ of the way along that screw motion.

**Checkpoint 4:** `test_screw_between_space_and_body` (it also checks $\mathcal S_b = [\mathrm{Ad}_{T_1^{-1}}]\mathcal S_s$, the Week 3 adjoint law), `test_screw_path_needle_guide` (half-way pose of a needle guide that turns $90^\circ$ while moving: origin at $(0.312,\ 0.171,\ 0.125)$ m).

## Step 5 — Quaternions (15 min)

Implement `rot_to_quat` (Eqs. B.10–B.11 when $q_0 > 0.5$, otherwise build $q$ from the log via B.9), `quat_to_rot` (B.12), `quat_mul` (B.15), `quat_conj`.
Remember that the printed Eq. (B.11) has a typo in its last entry: it must be $r_{21} - r_{12}$.

**Checkpoint 5:** `test_rot_to_quat_matches_scipy`, `test_quat_to_rot_round_trip`, `test_quat_mul_is_matrix_product`, `test_workshop_quaternion_example` ($q_x\otimes q_y = \tfrac12(1,1,1,1)$ for $\mathrm{Rot}(\hat x, 90^\circ)\mathrm{Rot}(\hat y, 90^\circ)$, a $120^\circ$ rotation about $(1,1,1)/\sqrt3$).
Note: SciPy stores quaternions scalar-**last** $(x, y, z, w)$; MR puts the scalar first.

## Step 6 — IMU orientation from gyroscope data (25 min)

Load the data:

```python
import numpy as np, starter as lab
d = np.genfromtxt("imu_knee.csv", delimiter=",", names=True)    # 501 samples, 100 Hz, 5 s
w_th = np.column_stack([d["thigh_wx"], d["thigh_wy"], d["thigh_wz"]])[:-1]
R0 = lab.quat_to_rot([d["thigh_q0"][0], d["thigh_q1"][0], d["thigh_q2"][0], d["thigh_q3"][0]])
```

a. Implement `integrate_gyro_exp` and `integrate_gyro_euler`. Integrate the ideal thigh and shank gyro signals from the true initial orientation.
b. Plot the rotation error $\|\log(\hat R_k^{\mathsf T}R_k)\|$ (in degrees) of the exponential integrator against time, and $\det\hat R_k$ of the Euler integrator against time. Expected: the exponential error stays below $10^{-8}$ rad (the ideal gyro is exact for a zero-order hold); the Euler determinant reaches about $1.16$ (thigh) after 5 s.
c. Implement `orientation_error`, `rotation_angle` and `knee_flexion`. Compute the knee flexion angle from the true quaternions and plot it; the peak is about $56.7^\circ$.
d. Repeat with the **noisy** gyro columns (`*_nw*`: constant bias + white noise). The shank orientation drifts by about $3.0^\circ$ in 5 s and the knee-angle error reaches about $4.0^\circ$. Plot the knee angle from the noisy gyros on top of the truth.

**Checkpoint 6:** `test_gyro_exponential_integration_reproduces_truth`, `test_euler_integration_drifts_off_SO3`, `test_knee_flexion_from_quaternions`, `test_noisy_gyro_drift_is_bounded_but_nonzero`.

## Step 7 — Tracked pedicle-screw insertion (20 min)

`screw_insertion.csv` is the optical-tracker log of the calibrated screwdriver **tip** frame $\{d\}$ (origin at the tip, $\hat z_d$ along the shaft) while a pedicle screw is driven: 401 samples at 50 Hz, quaternion (scalar first) and position in metres, ideal columns and noisy columns (`n*`). Load it with the given helper:

```python
t, Ts = lab.load_pose_track("screw_insertion.csv")              # Ts: (401, 4, 4)
t, Tn = lab.load_pose_track("screw_insertion.csv", noisy=True)
```

a. Use `screw_between(Ts[200], Ts[210], "space")` and the given `twist_to_screw` to find the screw axis while the surgeon turns at full speed. Expected: $\theta = 1.568$ rad, $\hat s = (0.300, -0.200, -0.933)$, pitch $h = 0.398$ mm/rad (lead $2\pi h = 2.5$ mm per turn), and the point of the axis closest to the tracker origin $q = (0.483, -0.302, 0.220)$ m. Check that the entry point $(0.120, -0.060, 1.350)$ m lies on this line. Repeat with `"body"`: why is $\mathcal S_b = (0, 0, 1, 0, 0, h)$ so simple?
b. Now try `screw_between(Ts[0], Ts[-1])` for the whole insertion. You get a **pure translation** of $12.5$ mm along $\hat s$. Explain why (pre-lab question 4).
c. Implement `screw_motion_profile(Ts)`: sum the step-by-step logarithms (signed angle about the axis of the largest step; advance $h_k\,\Delta\theta_k$). On the ideal data the total angle is $10\pi$ rad (5 turns) and the advance $12.5$ mm.
d. Run it on the noisy data, plot the advance $d$ against the angle $\theta$ for both data sets, and fit a straight line (`np.polyfit(theta, d, 1)`). The fitted lead is about $2.53$ mm per turn (1 % error) and the total angle about $5.005$ turns.

**Checkpoint 7:** `test_insertion_axis_from_two_poses`, `test_screw_motion_profile_ideal_data`, `test_screw_motion_profile_handles_reversal` (the screw is backed out for a moment: the angle must go down again), `test_screw_motion_profile_noisy_data`.

## Step 8 — Orientation and pose error for an EOD gripper (10 min)

An EOD robot's gripper is at $R_{cur} = \mathrm{Rot}(\hat z, 40^\circ)\mathrm{Rot}(\hat y, -20^\circ)$ and must reach $R_{des} = R_{cur}\mathrm{Rot}(\hat x, 35^\circ)$ to align with a package's handle.

a. Use `orientation_error` to find the required rotation in the gripper frame; you should get $(0.6109,\ 0,\ 0)$ rad, i.e. $35^\circ$ about the gripper's own $\hat x$-axis.
b. With the position included, $T_{cur} = (R_{cur}, (0.60, 0.10, 0.40))$ m, the desired pose is reached by turning $35^\circ$ about an axis parallel to the gripper $\hat x$-axis through $(0, 0, 0.05)$ m of the gripper frame while advancing $0.02$ m/rad along it. Recover this from $T_{cur}$ and $T_{des}$ with `screw_between(..., "body")` and `twist_to_screw`: $\mathcal S_b = (1, 0, 0, 0.02, 0.05, 0)$, $\theta = 0.6109$ rad, $q = (0, 0, 0.05)$, $h = 0.02$ m/rad.

**Checkpoint 8:** `test_orientation_error_eod_gripper`, `test_eod_gripper_screw_motion`. Finally run the whole suite: `LAB_IMPL=starter python -m pytest -q test_lab.py` must show **30 passed**.

# Deliverables (auto-checked notebook lab; not a formal report)

1. `starter.py` with all functions implemented; the full test suite passes.
2. A short notebook or PDF (≤ 4 pages) with: the two plots from Step 6b, the knee-angle plot from Steps 6c–6d, the advance-versus-angle plot with the fitted lead from Step 7d, the round-trip errors from Step 3, and answers to the questions in Steps 3, 7a and 7b.
3. The reflection below.

Marking (lab category): tests 50 %, plots and answers 35 %, reflection 15 %. Be prepared for a 2-minute oral check: a TA will point to any line of your code and ask you to explain it.

# Reflection (≤ 250 words)

- Why does the exponential update stay on $SO(3)$ while the first-order update does not? What would you do on a microcontroller that cannot afford `sin`/`cos` every sample?
- The bias in the noisy gyro is only about $0.01$ rad/s, yet the knee angle is $4^\circ$ wrong after 5 s. What does this imply for using IMUs to measure joint angles over a 10-minute gait trial or an exoskeleton route march, and how do commercial systems correct it?
- The logarithm of the whole screw insertion missed all five turns. What does this tell you about using one $\log$ to measure any motion that may rotate by more than $\pi$ (a surgical drill, a wrist joint, a wheel)?
- **AI-use note (required):** state which AI tools you used (or "none"), what you asked, and what you kept or changed. You must be able to explain every line you submit.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: starter` | Run pytest from inside your Lab code folder, or keep `test_lab.py` next to `starter.py`. |
| Every test fails with `NotImplementedError` | Expected until you implement the functions; set `LAB_IMPL=starter`, not `solution`. |
| `matrix_log3` returns `nan` | `arccos` received a value slightly outside $[-1, 1]$ because of rounding, or you divided by $\sin\theta \approx 0$; handle cases (a) and (b) before case (c). |
| π-case test fails for $\hat\omega = (1,0,0)$ | Equations (3.58) and (3.59) both divide by zero there; fall through to (3.60). |
| `matrix_exp6` translation wrong by a factor $\theta$ | $G(\theta)$ multiplies the **unit-axis** $v$ $(= v\theta/\theta)$, not $v\theta$. |
| `matrix_log6` gives `nan` or `inf` for $R = I$ | $G^{-1}$ divides by $\theta = 0$: return the pure-translation case (a) first. |
| Body and space screws swapped | Space: $\log(T_2T_1^{-1})$; body: $\log(T_1^{-1}T_2)$. |
| Quaternion test fails only by sign | You compared $q$ with $-q$; both are correct. Make $q_0 \ge 0$ in `rot_to_quat`. |
| Quaternions from SciPy look "shuffled" | SciPy order is $(x, y, z, w)$; MR order is $(q_0, q_1, q_2, q_3)$ with $q_0 = w$. |
| Gyro integration error is large | You pre-multiplied ($e^{[\omega]\Delta t}R_k$) a **body**-frame rate, or used all 501 rows (the last row of gyro data is padding; use `[:-1]`). |
| Knee angle has the wrong sign | Use $\log(R_{thigh}^{\mathsf T}R_{shank})$, not $\log(R_{shank}^{\mathsf T}R_{thigh})$. |
| Noisy insertion angle jumps or runs backwards | You took the reference axis from the first step; near standstill the noise dominates. Use the axis of the largest step. |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the numbers of the Python reference (including the `scipy.spatial.transform` quaternions and the noise-free data model, stored in `cpp_cases.json`) and, for the IMU ground truth, with Eigen's own quaternion class, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random test inputs of `test_lab.py`), `CMakeLists.txt`. The data files are the same `imu_knee.csv` and `screw_insertion.csv` (in your Lab code folder, next to the `cpp` folder). `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 3 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*vec_to_so3*,*skew*,*axis_ang3*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 30 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `vec_to_so3`, `so3_to_vec`, `axis_ang3` (Step 1); `matrix_exp3`, `g_matrix`, `g_inv_matrix`, `matrix_exp6` (Step 2); `matrix_log3`, `matrix_log6` (Step 3); `screw_between`, `screw_path` (Step 4); `screw_motion_profile` (Step 7); `rot_to_quat`, `quat_to_rot`, `quat_mul`, `quat_conj` (Step 5); `integrate_gyro_exp`, `integrate_gyro_euler`, `orientation_error`, `rotation_angle`, `knee_flexion` (Steps 6 and 8). Given and complete at the top of `starter.cpp`: `rp_to_trans`, `trans_to_rp`, `trans_inv`, `vec_to_se3`, `se3_to_vec`, `twist_to_screw`, `load_pose_track`, and the constant `TOL` (in `lab.hpp`). As in Python, do not call `mrcpp` (`mr::...`) or Eigen's `Quaterniond` / `AngleAxisd` inside `starter.cpp`; they are only for checking.

**Checkpoints** (the C++ test cases have the same names as the Python tests, one test case per Python test; 30 in total):

| Checkpoint | Command |
|---|---|
| 1 | `./build/lab_tests -tc="*vec_to_so3*,*skew*,*axis_ang3*"` |
| 2 | `./build/lab_tests -tc="*matrix_exp3*,*g_matrix*,*exp6*"` |
| 3 | `./build/lab_tests -tc="*matrix_log3*,*exp_log_round_trip*,*log6*"` |
| 4 | `./build/lab_tests -tc="*screw_between*,*screw_path*"` |
| 5 | `./build/lab_tests -tc="*rot_to_quat*,*quat_to_rot*,*quat_mul*,*workshop*"` |
| 6 | `./build/lab_tests -tc="*gyro*,*euler*,*knee*"` |
| 7 | `./build/lab_tests -tc="*insertion*,*screw_motion_profile*"` |
| 8 | `./build/lab_tests -tc="*eod*"`, then `./build/lab_tests`: 30 of 30 test cases pass |

For the plots of Steps 6 and 7, either use the Python track's code on your C++ results or write them as CSV with `sbeg::write_csv` (in `<sbeg/labio.hpp>`) and plot the CSV in Python; plotting and the line fit of Step 7d are not part of your C++ functions (the test fits the line itself).

**Differences from the Python version.**

| Python | C++ |
|---|---|
| NumPy 3-vector, 6-vector, `(3, 3)` and `(4, 4)` arrays | `Eigen::Vector3d`, `sbeg::Vector6d` (`lab::Vector6d`), `Eigen::Matrix3d`, `Eigen::Matrix4d` |
| quaternion `np.array([q0, q1, q2, q3])` | `Eigen::Vector4d q(q0, q1, q2, q3)`, scalar first as in MR (not `Eigen::Quaterniond`, which stores the scalar last) |
| `axis_ang3`, `screw_between`, `trans_to_rp` return tuples | return `std::pair`: `auto [S, theta] = lab::screw_between(T1, T2, "body");` |
| `screw_motion_profile` returns `(theta, d)` arrays | returns `std::pair<Eigen::VectorXd, Eigen::VectorXd>` |
| `twist_to_screw` returns a dict; `q` is `None` and `h` is `np.inf` for a pure translation | returns `lab::ScrewParams` with fields `S, thetadot, h, s_hat, q, has_q`; `has_q` is false and `h` is infinity for a pure translation |
| `load_pose_track` returns `(t, Ts)`, `Ts` of shape `(N, 4, 4)` | returns `lab::PoseTrack` with `t` (`Eigen::VectorXd`) and `Ts` (`std::vector<Eigen::Matrix4d>`) |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| gyro samples, array of shape `(N, 3)` | `Eigen::MatrixXd` with N rows and 3 columns; row k as a vector: `omega_b.row(k).transpose()` |
| result of shape `(N+1, 3, 3)` | `std::vector<Eigen::Matrix3d>` with N+1 entries (`push_back`) |
| `T[:3, :3]`, `T[:3, 3]`, `V[:3]`, `V[3:]` | `T.topLeftCorner<3, 3>()`, `T.topRightCorner<3, 1>()`, `V.head<3>()`, `V.tail<3>()` |
| `np.trace(R)`, `R.T`, `A @ B`, `np.linalg.norm(v)`, `np.cross(a, b)`, `a @ b` (vectors) | `R.trace()`, `R.transpose()`, `A * B`, `v.norm()`, `a.cross(b)`, `a.dot(b)` |
| `np.arccos`, `np.tan`, `np.pi` | `std::acos`, `std::tan`; a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

The tests read `imu_knee.csv` themselves (columns by name, the last gyro row dropped as in Python's `[:-1]`) and read `screw_insertion.csv` through the given `load_pose_track`. A matrix-size mistake does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../imu_knee.csv`, `.../screw_insertion.csv` or `.../cpp_cases.json` | Lab code folder moved after configuring, or a data file moved out of it | delete the `build` folder and configure again; keep the CSV files next to the `cpp` folder |
| `matrix_log3` gives `nan` | `std::acos` of a value slightly outside $[-1, 1]$, or division by $\sin\theta \approx 0$ | handle cases (a) and (b) before case (c) |
| `screw_motion_profile` tests fail with `TODO: quat_to_rot` | `load_pose_track` uses your `quat_to_rot` | finish Step 5 first |
| quaternion tests fail with `q` and the reference equal up to order | used `Eigen::Quaterniond` coefficients (`coeffs()` is $(x, y, z, w)$) | build the `Eigen::Vector4d` as $(q_0, q_1, q_2, q_3)$ yourself |
| `REQUIRE( Rs.size() == ... )` fails | the integrator did not return N+1 matrices | start with `out.push_back(R0)`, then one `push_back` per gyro row |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
