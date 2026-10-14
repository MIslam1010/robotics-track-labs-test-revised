---
title: "Lab 9 — IMU orientation fusion for an exoskeleton thigh: complementary filter vs extended Kalman filter"
subtitle: "SBEG 46X Medical Robotics — Week 10"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class (recorded dataset for everyone; live IMU module optional) + about 2 h of independent work |
| CLOs | **46X-7** (design state estimators — Kalman filter, EKF, IMU orientation fusion — for wearable and interventional medical robots; the camera and visual-servoing part of this CLO is Week 12 and Lab 11); 46X-6 (review: the exoskeleton joint of Lab 8 needs this angle for its controller) |
| Reading | Lecture notes Sections 4–10; Åström & Murray §7.1 (pp. 201–206: observability) and §7.4 (pp. 215–219: Kalman filter, Eqs. 7.18–7.22); Corke §3.2.3 (pp. 53–56: inertial sensors) and §6.1 (pp. 111–116: EKF); MR App. B.3 (pp. 583–584: unit quaternions, Eqs. B.9–B.15). Not in the course library (†): Mahony, Hamel & Pflimlin, "Nonlinear complementary filters on the special orthogonal group", *IEEE Trans. Autom. Control* 53(5), 2008 |
| Files (your Lab code folder) | `starter.py` (your code: 14 functions marked TODO A1 … C5), `test_lab.py` (auto-checks, 25 tests), `course_models.py` (the gait exoskeleton `gait_exo_2r` and the thigh-IMU pose, numbers identical to Course resources: Course robot models and hardware safety checklist), `data/thigh_imu.csv` (60 s recording), `data/imu_meta.json` (sample period, mounting, sensor specification), `data/expected.json` (reference results used by the tests) |
| Library | The finished functions become module `estimation` of your course library `sbeg46x_medrob` (after `rcm`, `teleop`, `haptics`, `registration`, `continuum` and `optctrl`) and replace the TODO `KalmanFilter` of the project starter |
| Assessment | Auto-checked code (25 tests) + results table + three plots + reflection (one auto-checked lab in the Labs component; not a formal report) |
| AI use | Allowed for code **with disclosure** (AI-use note in the reflection); you must be able to explain every line in an oral check. Derivations are your own |

## Why this lab matters

An exoskeleton cannot assist a step it cannot see. The hip-assist controller you designed in Lab 8 needs the thigh angle
and its rate every few milliseconds, and on a real wearer there is no encoder on the human thigh: the angle must come
from an inertial measurement unit (IMU) strapped to the cuff. The gyro is smooth but drifts; the accelerometer knows
where "down" is but is fooled by every swing of the leg. Today you fuse the two, first with the simple complementary
filter, then with an EKF that also estimates the gyro bias, and finally with a quaternion filter that works in 3-D and
shows what an accelerometer can never tell you (heading). The same estimator drives gait-phase detection in
rehabilitation exoskeletons and prostheses, and the soldier-worn IMUs of a load-carriage exoskeleton (project scenario 6);
scenario 3 uses it to estimate the motion of a user on the Hapkit rehabilitation device.

## The recording, frames and units (illustrative values)

* **Robot and segment.** Hip–knee gait exoskeleton `gait_exo_2r` (SBEG 461 project S2): swing leg on a test stand, hip
  axis fixed, thigh length 0.42 m. World {s}: $\hat x_s$ anterior, $\hat y_s$ up, $\hat z_s$ lateral (the hip axis),
  gravity $g_s = (0, -g, 0)$. $\theta$ = hip flexion, 0 with the leg hanging, positive forward.
* **IMU.** 6-axis MEMS IMU on the thigh cuff, $r = 0.20$ m below the hip. Sensor frame {b} with
  $R_{sb} = \mathrm{Rot}(\hat z, \theta)$: $\hat x_b$ anterior, $\hat y_b$ up the thigh, $\hat z_b$ = hip axis (lecture notes §7.2).
  The ideal readings are
  $$\omega_m = (0, 0, \dot\theta) + b + n_g, \qquad f = (r\ddot\theta + g\sin\theta,\ \ r\dot\theta^2 + g\cos\theta,\ \ 0) + n_a .$$
  `course_models.imu_pose` builds $T_{sb} = e^{[\mathcal S_1]\theta}M_{imu}$ (MR Eq. 4.14) and `imu_specific_force`
  gives $f$; the tests check both against modern_robotics.
* **Data.** `data/thigh_imu.csv`: 6001 samples at 100 Hz ($h = 0.01$ s), columns `t, gx, gy, gz` (rad/s),
  `ax, ay, az` (m/s²) and `theta_enc` (rad), the hip encoder of the test stand — **a reference for grading your
  estimate only, never an input of your filters**. Phases: 0–8 s rest (leg hanging), 8–36 s swing at a 1.0 Hz cadence
  (−10° to +30°), 36–50 s faster swing at 1.4 Hz (−8° to +38°), 50–60 s held at 30°.
  Each gyro sample is the mean rate over the preceding sample interval (the IMU's internal filter), so
  $\theta_k = \theta_{k-1} + h\,\omega_{m,k}$ is the right integration rule; each accelerometer sample is instantaneous.
* **Sensor faults in the data.** White noise (gyro 0.006 rad/s, accelerometer 0.06 m/s² per axis), quantisation, a gyro
  bias of about 0.02 rad/s (≈ 1.3°/s) on $z$ that **drifts** as the IMU warms up, and smaller biases on $x$ and $y$.
  This is a simulated bench recording made from the course model, not data of a person.

## Learning objectives

By the end of the lab you can:

1. compute the accelerometer tilt, integrate a gyro, and explain quantitatively why each fails alone;
2. implement and tune a complementary filter and predict its bias error ($\tau b$) and its crossover frequency;
3. implement a linear Kalman filter (predict/update) and an EKF with a gyro-bias state, and tune $R$ with the
   normalised innovation squared (NIS);
4. propagate a unit quaternion with gyro rates (MR App. B.3) and correct it with gravity (Mahony-type filter);
5. show from data which states are observable with an IMU (tilt and $z$-bias: yes; heading and the bias about the
   vertical: no).

# Pre-lab check (5 questions, submit before the session)

1. The leg hangs still and the accelerometer reads $f = (0.00, 9.81)$ m/s². Then the thigh is held at $\theta = 30°$. What
   does it read? Write $\theta$ from $(f_x, f_y)$.
2. A gyro with a constant bias of 1°/s is integrated for 60 s. What is the angle error? What is the steady error of a
   complementary filter with $\tau = 0.5$ s fed by the same gyro?
3. For the state $x = (\theta, b)$ with the gyro as input, write $A$ and $B$ for one sample $h$, and $W_o$ when the
   accelerometer tilt is measured. Is the bias observable?
4. Write the EKF measurement function $h(x)$ and Jacobian $H$ for the raw 2-axis accelerometer reading $(f_x, f_y)$.
5. For a 2-component innovation from a consistent filter, what is the expected value of the NIS $e^{\mathsf T}S^{-1}e$?
   What does a mean NIS of 70 tell you about $R$?

# Setup (5 min)

1. Copy your Lab code folder; work in `starter.py` or a copy with your own name (e.g. `estimation.py`).
2. Run `LAB_IMPL=starter python -m pytest -q test_lab.py`: 4 data tests pass and 21 fail with `NotImplementedError`.
   Re-run after every function; the test names tell you which function is being checked.
3. Load the data in a notebook or script: `rec = load_recording(); meta = load_meta()`. Plot `theta_enc`, `gz`, `ax`, `ay`
   against time and identify the four phases.
4. **Live IMU groups only (optional, unpenalised):** the IMU module is a low-voltage USB device with no actuator. Strap it
   to your own thigh with the elastic cuff so that its axes match {b} ($y$ up the thigh, $z$ to your right, the cable
   routed away from the knee). Stand on one leg next to the bench with a hand on it (or sit on the edge of a table) and
   swing the free leg gently; no jumping or running. The station logging script writes the column format of
   `data/thigh_imu.csv`, with `theta_enc` replaced by goniometer readings taken at three held poses (0°, 30°, 60°). Follow
   the TA's instructions and Course resources: Course robot models and hardware safety checklist for the lab-bench rules.

# Procedure

### Checkpoint 1 — The two sensors alone (A1, A2, A5, ≈ 10 min)

**E1.** Write `accel_tilt` (A1). Compute $\theta_{acc}$ for the whole recording and its RMS error against `theta_enc`
over the rest phase (0–8 s), the moving phases (8–50 s) and the whole record. Plot $\lVert f\rVert - g$ against time.
**E2.** Write `gyro_integrate` (A2) with $\theta_0 = \theta_{acc,0}$. Plot the error. Write `bias_at_rest` (A5), estimate
the $z$-gyro bias from the first 8 s, integrate again with it removed, and explain why the error still grows.

*Checkpoint question:* during the fast phase, where in the stride is the accelerometer tilt most wrong, and which term of
$f$ causes it (compare with Workshop Problem 3b)?

### Checkpoint 2 — Complementary filter (A3, A4, ≈ 10 min)

**E3.** Write `cf_alpha` (A3) and `complementary_filter` (A4). Run it with $\tau = 0.5$ s (raw gyro) and with the
rest-phase bias removed. **Sweep** $\tau \in \{0.05, 0.1, 0.2, 0.5, 1, 2, 5\}$ s (bias removed) and plot RMS error vs
$\tau$ on a log axis. Explain both ends of the curve: what dominates at small $\tau$, and at large $\tau$? Compute the
crossover frequency $1/(2\pi\tau)$ of your best $\tau$ and compare it with the cadence.

### Checkpoint 3 — Kalman filter and EKF with a bias state (B1–B4, ≈ 20 min)

**E4a.** Write `kf_predict` (B1) and `kf_update` (B2). The tests compare them with the information form and with the
Åström–Murray predictor form (Eq. 7.22), and rerun Workshop Problem 1.
**E4b.** Write `accel_model` (B3) and `ekf_tilt` (B4) (lecture notes §10.2). Run with $\sigma_g = 0.006$ rad/s,
$\sigma_b = 0.02$ rad/s² and $\sigma_{acc} = 0.06$ m/s² (the datasheet noise). Then sweep
$\sigma_{acc} \in \{0.06, 0.3, 1.0, 1.5, 2.5, 4.0\}$ m/s² and tabulate the RMS error **and the mean NIS**. Which
$\sigma_{acc}$ makes the filter consistent (mean NIS ≈ 2)? Is that also the most accurate setting? Explain why $R$ must be
far larger than the datasheet noise. Plot $\hat b(t)$ for $\sigma_{acc} = 1.5$ m/s² together with your rest-phase estimate,
and plot $\theta$ with the $\pm 2\sigma$ band from $P$.
**E4c (gating).** Run `ekf_tilt` with $\sigma_{acc} = 1.5$ m/s² and `gate=9.21`. Count the rejected updates, find where
they happen and compare the fast-phase RMS error with the ungated filter. Explain the **lock-out** you observe (Workshop
Problem 5c).

*Checkpoint question:* the bias estimate at 60 s differs from your rest-phase estimate. Which one is right, and how do you
know without the encoder?

### Checkpoint 4 — Quaternions and the 3-D view (C1–C5, ≈ 10 min in class, finish at home)

**E5.** Write `quat_mult` (C1, MR Eq. B.15), `quat_to_rot` (C2, Eq. B.12) and `quat_exp` (C3, Eq. B.9 with
$\phi = \lVert\omega\rVert h$). Check with Practice Problem 3 that $R_z(90°)R_x(90°)$ ↔ $(0.5, 0.5, 0.5, 0.5)$.
**E6.** Write `mahony_filter` (C4, $k_P = 1$, $k_I = 0.1$) and `thigh_angle_from_quat` (C5), and run them on all three
gyro and accelerometer axes. Compare the thigh-angle RMS error with E3 and E4. Plot the heading
(`heading_from_quat`) and the three bias estimates against time and explain which of them converge.
**E7 (optional extension).** Lever-arm compensation (lecture notes §10.4): subtract $(r\hat{\ddot\theta},\ r\omega_m^2)$
from $f$ before the update, with $\hat{\ddot\theta}$ from a smoothed difference of the gyro. How much does each filter
improve? **Live IMU groups:** run all filters on your own recording and report the error at the three goniometer poses.

# Deliverables

1. Your module passing all 25 tests (`LAB_IMPL=<your module> pytest -q test_lab.py`).
2. A results table: E1 RMS errors (rest, moving, all); E2 final drift and rest-phase bias; E3 RMS for $\tau = 0.5$ s and
   the sweep; E4 the $\sigma_{acc}$ sweep (RMS and mean NIS), the final bias estimate and $\sigma_\theta$, the gated
   run; E6 RMS, final heading and biases.
3. Three plots: (i) 5 s of the fast phase with the encoder, accelerometer tilt, complementary filter, EKF and Mahony
   estimates; (ii) RMS error vs $\tau$ (CF) and vs $\sigma_{acc}$ (EKF) on log axes; (iii) EKF bias estimate and
   Mahony heading vs time.
4. Reflection (≤ 200 words): (a) one sentence each on why the gyro and the accelerometer are complementary, why $R$ is a
   *model* parameter rather than a sensor specification, and what an exoskeleton controller would do with a heading that
   drifts by tens of degrees (does it matter for hip assist?); (b) one result that surprised you; (c) **AI-use note**:
   tool, what you asked, what you kept, what you checked yourself (or "no AI used").

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| tilt ≈ 90° − θ or wrong sign | `atan2(ay, ax)` instead of `atan2(ax, ay)` (frame {b}: $f = g(\sin\theta, \cos\theta)$ at rest) |
| gyro integration offset by about one sample | you used `gyro[k-1]`; this recording needs $\theta_k = \theta_{k-1} + h\,\omega_{m,k}$ |
| complementary filter follows the accelerometer noise | $\alpha$ and $1-\alpha$ swapped |
| EKF diverges or $P$ becomes negative | $H$ in degrees instead of per radian; wrong sign of $F_{12} = -h$; forgot to symmetrise $P$ |
| bias estimate never moves | $Q_{bb} = 0$ and $P_{bb}$ collapsed, or $P_0$ too small for the bias |
| NIS mean ≈ 1700 | $R$ = datasheet noise: the filter is told the accelerometer is perfect during swing |
| quaternion norm creeps above 1 | renormalise after every step (Practice Problem 3d) |
| thigh angle from the quaternion jumps by 180° | you used the wrong column of $R$; the thigh axis to the knee is $-R_{sb}\hat y_b$ |
| Mahony filter converges the wrong way | the error must be $e = v \times \hat v$ (measured × predicted), with $\hat v = \hat R^{\mathsf T}\hat y_s$ |

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| E1 | accelerometer-tilt RMS error: rest / moving (8–50 s) / all | 0.36° / 16.5° / 13.8° |
| E2 | gyro integration (raw): RMS / error at 60 s | 51.6° / 97° |
| E2 | rest-phase $z$-bias; RMS after removing it | 0.0220 rad/s (1.26°/s); 8.5° |
| E3 | CF, $\tau = 0.5$ s: raw gyro / bias removed | 3.49° / 3.53° |
| E3 | CF sweep $\tau$ = 0.05, 0.1, 0.2, 0.5, 1, 2, 5 s (bias removed) | 12.6, 10.6, 7.3, 3.5, 1.95, 1.14, 1.33° (optimum near 2 s) |
| E4b | EKF $\sigma_{acc}$ = 0.06, 0.3, 1.0, 1.5, 2.5, 4.0 m/s²: RMS | 4.67, 1.98, 1.08, 0.89, 0.72, 0.66° |
| E4b | same: mean NIS | 1688, 69.1, 6.23, 2.77, 1.00, 0.39 |
| E4b | $\sigma_{acc} = 1.5$: bias at 60 s (true ≈ 0.0377); $\sigma_\theta$ at 60 s; fast-phase RMS | 0.0358 rad/s; 0.63°; 1.35° |
| E4c | gated ($d^2 > 9.21$): rejected updates; fast-phase RMS | 619; 15.3° (lock-out) |
| E6 | Mahony ($k_P = 1$, $k_I = 0.1$): thigh RMS; heading at 60 s; $z$-bias at 60 s | 2.04°; −57°; 0.0331 rad/s |

Results within about ±5 % of these values (same parameters) are correct; the tests check your functions against
independent code and these reference runs.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, optional live-IMU steps and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`, used for the thigh-IMU pose and the rotation matrices), with the closed-form filter responses and independent formulas of `test_lab.py`, with `data/expected.json`, and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided functions — do not change), `course_models.cpp` (the C++ copy of `course_models.py`: `gait_exo_2r`, `imu_home`, `imu_pose`, `imu_specific_force`, numbers identical — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random test inputs of `test_lab.py`), `CMakeLists.txt`. The data are read from the same `data/thigh_imu.csv`, `data/imu_meta.json` and `data/expected.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 9 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*A4_*"                  # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 25 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The full run (three EKF runs and one quaternion filter over the 6001 samples) takes well under a second once the code is built.

**Functions to write** (in `starter.cpp`, the same 14 as in Python, same order): `accel_tilt`, `gyro_integrate`, `cf_alpha`, `complementary_filter`, `bias_at_rest` (A1–A5); `kf_predict`, `kf_update`, `accel_model`, `ekf_tilt` (B1–B4); `quat_mult`, `quat_to_rot`, `quat_exp`, `mahony_filter`, `thigh_angle_from_quat` (C1–C5). Given in `helpers.cpp` (complete, as in `starter.py`): `load_meta`, `load_recording`, `rms_deg` (with and without a mask), `quat_from_accel`, `heading_from_quat`, plus, C++ only, `accel_tilt(double ax, double ay)` for a single sample (it calls your vector version). Do not call `mrcpp` inside `starter.cpp` (the Python lab does not use `modern_robotics` in these functions either).

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — The two sensors alone (A1, A2, A5) | `-tc="*recording_format*,*recording_at_rest*,*A1_*,*A2_*,*A5_*"` | 6 |
| 2 — Complementary filter (A3, A4) | `-tc="*A3_*,*A4_*"` | 3 |
| 3 — Kalman filter and EKF (B1–B4) | `-tc="*B1_*,*B2_*,*B3_*,*B4_*"` | 8 |
| 4 — Quaternions and the 3-D view (C1–C5) | `-tc="*C1_*,*C2_*,*C3_*,*C4_*,*C5_*,*course_model_imu*,*provided_quat*"` | 8 |

All 25 test cases pass with `./build/lab_tests` (the Python track also counts 25 tests). One difference: in Python the 4 data and provided-code tests pass with the untouched starter; in C++ each of them also checks one of your functions (`test_recording_format` → `bias_at_rest`, `test_recording_at_rest_matches_model` → `accel_tilt`, `test_course_model_imu_pose` and `test_provided_quat_from_accel` → `quat_to_rot`), so they pass once those functions are written — this is why the last two belong to Checkpoint 4. A few checks are added against the Python reference numbers (the Kalman predict/update outputs, the Jacobian $H$ and the quaternion products, all to $10^{-10}$ or better). The random inputs of `test_lab.py` (seed 910) are stored in `cpp_cases.json`, so both tracks test the same angles, matrices and axes. The expected values above are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `rec = load_recording()` (dict of arrays), `rec["gz"]` | `lab::Recording rec = lab::load_recording();` with the same field names: `rec.t`, `rec.gz`, `rec.ax`, `rec.theta_enc` (`Eigen::VectorXd`) |
| `meta = load_meta()`, `meta["h"]`, `meta["imu_spec"]["acc_noise_m_s2"]` | `lab::ImuMeta meta = lab::load_meta();`, `meta.h`, `meta.acc_noise_m_s2`, `meta.phases[i].t0` |
| `accel_tilt` on a number or an array | `Eigen::VectorXd` version (yours) and a `double` version (given) |
| `theta0=None`, `x0=None`, `P0=None`, `gate=None`, `q0=None` | `std::optional`: leave the argument out or pass `std::nullopt`; inside your function `if (gate)` and `*gate` |
| `kf_predict(x, P, A, Q, B=None, u=None)`, `u` a number | `B` an n × m `Eigen::MatrixXd`, `u` an `Eigen::VectorXd`; empty (the defaults) = `None`; one input: `Eigen::VectorXd::Constant(1, u)` |
| `kf_update` returns `(x, P, nu, S)`; `accel_model` returns `(hx, H)` | struct `lab::KfUpdate` (fields `x`, `P`, `nu`, `S`); `std::pair<Eigen::Vector2d, Eigen::Matrix2d>` (`auto [hx, H] = lab::accel_model(x);`) |
| result dicts of `ekf_tilt` / `mahony_filter` | `lab::EkfResult` (`theta`, `bias`, `nis` as `Eigen::VectorXd`, `P` as `std::vector<Eigen::Matrix2d>`, `rejected` as `int`) and `lab::MahonyResult` (`q` N × 4, `bias` N × 3, one row per sample) |
| `gyro3`, `acc3` as `np.c_[rec["gx"], rec["gy"], rec["gz"]]` | `Eigen::MatrixXd gyro3(N, 3); gyro3 << rec.gx, rec.gy, rec.gz;` |
| quaternion `np.array([q0, q1, q2, q3])`, `np.r_[c, v]` | `Eigen::Vector4d` (`q(0)` = scalar part); `q << c, v;` |
| `np.cross`, `np.linalg.norm`, `np.linalg.solve(S, nu)`, `np.linalg.inv(S)` | `a.cross(b)`, `v.norm()`, `S.partialPivLu().solve(nu)`, `S.inverse()` |
| `np.nan`, `np.isnan`, `math.copysign`, `math.hypot` | `std::numeric_limits<double>::quiet_NaN()`, `std::isnan`, `std::copysign`, `std::hypot` |
| plots with matplotlib (E1–E6, deliverable 3) | write CSV files and plot them in Python (below) |

**Plots and experiments in C++.** Write a small `experiments.cpp` with your own `main()` (add it as a second executable, or build it by hand next to `starter.cpp`, `helpers.cpp` and `course_models.cpp`) for the sweeps of E3 and E4b, the gated run of E4c and E6. Save each curve with `sbeg::write_csv` from `<sbeg/labio.hpp>`, e.g. `Eigen::MatrixXd M(N, 3); M << rec.t, rec.theta_enc, ekf.theta; sbeg::write_csv("ekf_run.csv", {"t", "theta_enc", "theta_ekf"}, M);`, and plot it with a few lines of Python:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt("ekf_run.csv", delimiter=",", skiprows=1)
plt.plot(d[:, 0], np.degrees(d[:, 1]), d[:, 0], np.degrees(d[:, 2])); plt.xlabel("t (s)"); plt.ylabel("theta (deg)"); plt.show()
```

The results table, the optional live-IMU steps and the reflection do not change. For your cumulative library, keep your Lab 9 functions in an `estimation.cpp`/`estimation.hpp` pair.

A size or index mistake on an Eigen object (e.g. `out.theta(k)` before `out.theta.resize(N)`, or a 3-vector where a 4-vector is expected) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". Indexing a `std::vector` that is too short (e.g. `out.P[k]` without `out.P.resize(N)`) is not protected and can crash the program: size `out.P` first.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../thigh_imu.csv` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `ekf_tilt` test fails with "Eigen: wrong matrix size" | `x` or `P` built as `Eigen::VectorXd x;` without a size, or `B` passed as a row | `Eigen::VectorXd x(2);`, `B = Eigen::Vector2d(h, 0.0)` (a column) |
| program stops with a segmentation fault in `ekf_tilt` | `out.P[k]` written before `out.P.resize(N)` | size every output before the loop |
| `nis` check fails at k = 0 | `out.nis` initialised with zeros | `Eigen::VectorXd::Constant(N, std::numeric_limits<double>::quiet_NaN())` |
| `complementary_filter` ignores `theta0 = 0.0` | `if (*theta0)` tests the value, not whether it was given | `if (theta0)` |
| `quat_to_rot` / `mahony_filter` slightly off or drifting | quaternion not normalised (before building $R$, after every filter step) | `q / q.norm()` |
| `thigh_angle_from_quat` wrong sign near 0 | `std::abs` instead of `std::copysign(std::hypot(dx, dz), dx)` | see C5 in `lab.hpp` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
