---
title: "Lab 5 — Forward kinematics of the course arm: PoE, URDF, simulation and hardware"
subtitle: "SBEG 361 Introduction to Robotics — Week 6"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before the Week 7 lab) |
| CLOs | 361-4 (PoE forward kinematics, space and body forms); 361-7 (implement and validate against a reference library, a simulator and hardware) |
| Part A (auto-tested) | PoE forward kinematics and URDF → PoE conversion: `starter.py` (you edit), `test_lab.py` (32 auto-checks) |
| Part B (auto-tested + analysis) | Simulation versus hardware with a **recorded** log of the course arm and external tip measurements |
| Part C (optional demonstration) | An instructor or TA records a log on a course arm while you watch from outside the keep-out zone; you do **not** handle the arm this week |
| Data | `data/course_arm.urdf` (shared course-arm model), `ur5_mr.urdf` (MR §4.2), `servo_offsets.json`, `course_arm_log.csv`, `tip_checks.csv` (synthetic, illustrative values); `trajectory.py` gives the commanded motion `commanded(t)` of the recorded log |
| Reference | `modern_robotics` (`MatrixExp6`, `FKinSpace`, `FKinBody`, `Adjoint`); Pinocchio (URDF check); MR Studio (FK tab ①, *File → Import URDF…*) |
| Reading | MR §4.1–4.2 (printed pp. 138–156): Eqs. 4.14, 4.16, Examples 4.5 and 4.7, the UR5 URDF (pp. 153–156). Optional: MLS Ch. 3 §2 (pp. 85–95); Pardos-Gotor §3.3 (pp. 36–48; UR16e p. 45) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation and recorded data). The course arm appears only in the optional Part C demonstration, operated by staff; your first hands-on contact with the arm is the safety induction at the start of Week 8 Session 4 |

The functions you write today form the `fk.py` module of your `sbeg361_kinematics` library. Lab 6 compares them with a DH model (Robotics Toolbox) and a C++/Eigen implementation; Lab 7 builds the Jacobian on the same exponentials (its starter writes `screw_exp(S, theta)` as Lab 3's `matrix_exp6(vec_to_se3(S) * theta)`) and on `adjoint`; the project's hardware runs (H1, H2) use your course-arm model and the comparison code of Part B.

**The course arm.** A low-cost desktop 6R arm of the myCobot-280 class with the joint layout of the UR5 in MR Example 4.5, scaled down: $W_1 = 0.0634$, $W_2 = 0.0456$, $L_1 = 0.110$, $L_2 = 0.096$, $H_1 = 0.1312$, $H_2 = 0.0751$ m, reach about 0.28 m (Workshop 6, Problem 4). These numbers come from the shared course model and are the same in every lab and in the project. Rated speeds, servo offsets and all "measured" data in this lab are **illustrative values**.

**Application context.**
*Biomedical:* in project scenario S3 the course arm holds a 60 mm needle guide over a biopsy phantom. Before any IK or path planning, the team must know that the arm goes where the model says: a needle guide that is 5 mm off at the skin is 5 mm off at the target.
*Defence (protective use):* the same validation is done on an EOD robot's arm before a mission: the operator's display shows the gripper where FK puts it, so a zero-offset error on one joint makes the operator approach a suspect object from the wrong place. Today you find exactly such an error from measurements.

# Learning objectives

By the end of the lab you can:

1. Implement $e^{[\mathcal S]\theta}$ (MR Prop. 3.25) and the space- and body-form PoE (MR Eqs. 4.14, 4.16), and validate them against `modern_robotics` and the MR examples (UR5, WAM).
2. Derive the course arm's $M$ and screw axes from its geometry, and convert a URDF (MR §4.2) into the same PoE model.
3. Convert servo readings to model joint angles with a calibration file, and compare commanded and measured motion in joint space and in task space.
4. Use external tip measurements to detect and estimate a joint zero-offset error, and explain the difference between "the arm follows its commands" and "the arm is where the model says".

# Pre-lab check (complete before the session; 5 questions)

1. Write $\mathcal S$ for a revolute joint with axis $(0, 0, 1)$ through $q = (1, 2, 0)$.
2. In $T = e^{[\mathcal S_1]\theta_1}e^{[\mathcal S_2]\theta_2}M$, is $\mathcal S_2$ the axis of joint 2 at the current configuration or at the zero configuration?
3. Give $\mathcal B_i$ in terms of $M$ and $\mathcal S_i$.
4. A URDF joint has `<origin xyz="0 0 0.1312" rpy="0 0 0"/>` and `<axis xyz="0 1 0"/>`, and its parent frame is $\{s\}$. Write its screw axis.
5. The servo of joint 2 reads $55.6^\circ$ and the calibration file gives an offset of $90^\circ$ with $q_\text{model} = q_\text{servo} - \text{offset}$. What is the model angle in radians?

*Answers (for self-check):* 1. $v = -\hat z\times(1,2,0) = (2,-1,0)$, so $\mathcal S = (0,0,1,\ 2,-1,0)$. 2. At the zero configuration (MR p. 148). 3. $\mathcal B_i = [\mathrm{Ad}_{M^{-1}}]\mathcal S_i$ (MR Eq. 4.16). 4. $\omega = (0,1,0)$, $q = (0,0,0.1312)$, $\mathcal S = (0,1,0,\ -0.1312,0,0)$ (the course arm's $\mathcal S_2$). 5. $(55.6 - 90)^\circ = -34.4^\circ = -0.600$ rad.

# Setup

```bash
cd "<your Lab code folder>"   # or your Classroom 50 repo
# activate the course Python environment (course venv, Docker image or JupyterHub)
python -c "import numpy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py               # 31 failures (NotImplementedError) expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Rules: do **not** call `modern_robotics`, `scipy` or `np.linalg.inv` inside `starter.py`; they are only for checking. The helpers marked "complete helper" (`vec_to_so3`, `vec_to_se3`, `trans_inv`, `adjoint`, `rot_axis` (Lab 1's `rot_x`/`rot_y`/`rot_z` in one helper), `read_urdf_joints`, `urdf_chain`, `load_joint_log`, `load_offsets`) may be used. `course_arm_model.py` holds the reference numbers for the tests; do not import it in `starter.py`. One test (`test_course_arm_copy_is_identical_to_shared`) only checks the data files and passes from the start (it is skipped in a student copy of the lab).

# Part A — PoE forward kinematics (auto-tested)

Run the tests after each step. A checkpoint is complete when all listed tests pass; ask a TA to sign it off.

## Step 1 — Screw axes and the exponential (8 min)

Implement `screw_axis` (revolute: $\omega$ and $q$; prismatic: $v$; `ValueError` for a non-unit direction or missing data) and `screw_exp(S, theta)` $= e^{[\mathcal S]\theta}$ from MR Eq. 3.88 (rotation part by Rodrigues, Eq. 3.51; position part $G(\theta)v$ with $G$ from Eq. 3.87) and Eq. 3.89 for $\omega = 0$. It returns the same matrix as your Lab 4 `matrix_exp6(se3mat)` called as `matrix_exp6(vec_to_se3(S) * theta)`; here the axis and the angle are passed separately.

**Checkpoint 1:** `test_screw_axis_mr_example_4_1`, `test_screw_axis_prismatic_and_errors`, `test_screw_exp_matches_mr`, `test_mr_example_4_5_exponentials` (the two matrices printed in MR Example 4.5: $e^{-[\mathcal S_2]\pi/2}$ with entries $0.089$, and $e^{[\mathcal S_5]\pi/2}$ with $(0.708, 0.926)$).

## Step 2 — Space and body forms; the course arm (12 min)

Implement `fk_space` (Eq. 4.14), `fk_body` (Eq. 4.16), `body_screws(Slist, M)` and `course_arm_screws` (read the joint table in the docstring; it is the table of Workshop Problem 4).

**Checkpoint 2:** `test_fk_space_matches_mr` (six robots: MR Examples 4.1, 4.3, 4.4, the RRP robot of MR Exercise 5.7, the UR5 and the course arm), `test_ur5_mr_example_4_5` ($p = (0.095, 0.109, 0.988)$), `test_rrp_check_value`, `test_body_screws_mr_example_4_6`, `test_fk_body_equals_space`, `test_wam_mr_example_4_7` ($p = (0.3157, 0, 0.6571)$), `test_course_arm_screws`, `test_course_arm_mr_example_4_5_pose`.
Print `fk_space(M, S, [0, -pi/2, 0, 0, pi/2, 0])` for the course arm. You should get $p = (0.0751,\ 0.0634,\ 0.3828)$ m $= (H_2,\ W_1,\ H_1 + L_1 + L_2 + W_2)$. Sketch the pose and explain each coordinate.

## Step 3 — URDF to PoE (10 min)

Implement `rpy_to_rot` (URDF: $R = \mathrm{Rot}(\hat z,\text{yaw})\mathrm{Rot}(\hat y,\text{pitch})\mathrm{Rot}(\hat x,\text{roll})$) and `poe_from_urdf`: walk the chain from the base with all joints at zero, compose $T_{s,\text{child}} = T_{s,\text{parent}}\,(R(\text{rpy}),\ \text{xyz})$, and for each movable joint use $\omega = R_{s,\text{child}}\,\cdot$`axis` and the joint-frame origin as the point on the axis.

**Checkpoint 3:** `test_rpy_to_rot_matches_scipy`, `test_urdf_course_arm` (your PoE from `course_arm.urdf` equals the shared model; limits $\pm165^\circ$, $\pm175^\circ$), `test_urdf_ur5_from_mr_text` (the URDF printed in MR §4.2 reproduces the MR Example 4.5 table with $W_1 = 0.13585 - 0.1197 + 0.093 = 0.10915$), `test_urdf_matches_pinocchio`.

**Simulator check (2 min).** In MR Studio choose *File → Import URDF…* and load `data/course_arm.urdf`. In the FK tab ① compare the app's screw-axis table with your `Slist`, then set the joints to $(0, -90^\circ, 0, 0, 90^\circ, 0)$ and compare the displayed pose with Step 2. Screenshot for your notebook.

# Part B — Simulation versus hardware with recorded data

The files in `data/` are what a team recorded on arm "A3" (synthetic, illustrative). The log has the format of the project's `hwcompare.py`: `t, cmd_1..cmd_6, meas_1..meas_6, stop`, angles in **servo degrees**, 50 Hz, 20 s; the stop command is given at $t = 18$ s. The servo zero of this arm is the "candle" pose (arm straight up); `servo_offsets.json` gives the offsets $(0, 90, 0, 90, 0, 0)^\circ$ with $q_\text{model} = q_\text{servo} - \text{offset}$.

## Step 4 — Joint space and task space (10 min)

Implement `servo_to_model`, `tool_points` and `compare_log` (samples before the stop command only). The tool point is the tip of the 60 mm guide, $p_\text{tool} = (0, 0, 0.06)$ in $\{b\}$.

```python
import numpy as np, starter as lab
M, S = lab.course_arm_screws(0.0634, 0.0456, 0.110, 0.096, 0.1312, 0.0751)
off = lab.load_offsets("data/servo_offsets.json")
log = lab.load_joint_log("data/course_arm_log.csv")
res = lab.compare_log(log, M, S, off, p_tool=(0, 0, 0.06))
```

a. **Calibration sanity check.** `tool_points(M, S, servo_to_model(np.zeros((1, 6)), off))` must give the flange straight above joint 1 at $(0,\ 0.109,\ 0.4123)$ m $= (0,\ W_1+W_2,\ H_1+L_1+L_2+H_2)$. Explain why this proves the offset signs are right.
b. **Joint space (HW-2).** You should find 900 samples, joint RMS errors $(0.34, 0.54, 0.27, 0.22, 0.27, 0.47)^\circ$ and mean errors $(0.00, -0.50, -0.01, -0.00, 0.00, -0.01)^\circ$. Plot $e_j(t) = \text{meas}_j - \text{cmd}_j$ for all joints. Which joint has a **constant** offset, and what physical cause would you suspect (hint: which joint carries the most gravity torque)? Which joints show an error that follows the commanded **speed**, and what does that tell you about servo lag?
c. **Task space (HW-3).** Tool-point error between FK of the commanded and the measured joints: RMS $3.69$ mm, max $8.14$ mm. Plot it against time. Does the log pass HW-3 (RMS ≤ 5 mm, max ≤ 10 mm)?
d. **Speed (HW-5).** With `np.gradient(log["cmd"], log["t"], axis=0)` find the peak commanded joint speeds; the largest is $10.5^\circ$/s (joint 6). What fraction of the rated $160^\circ$/s is this, and is it within the 25 % rule of a first hardware session?

**Checkpoint 4:** `test_servo_to_model`, `test_servo_zero_is_candle_pose`, `test_tool_points_matches_mr`, `test_compare_log`.

## Step 5 — Is the arm where the model says? (10 min)

Joint logs only compare the servo with *itself*: if a servo's zero is mis-set, it still reports exactly what it was commanded. The team therefore measured the pointer tip at six poses with a printed grid and a height gauge on the bench (`tip_checks.csv`: servo angles and measured tip in mm, in $\{s\}$). Implement `tip_errors_mm` and `fit_joint_offset`.

a. Compute the FK prediction for each pose and the error to the measurement: about $(5.7, 5.0, 5.1, 4.6, 5.5, 3.9)$ mm, RMS $4.99$ mm. Does the arm pass the project's HW-1 test (≤ 10 mm at 3 poses)? Would you trust it for a 5 mm biopsy target?
b. The errors are all similar and much larger than the 0.4 mm probe uncertainty. Hypothesis: one joint's zero is wrong. For each joint $j$, `fit_joint_offset(M, S, Q, P, j, (0, 0, 0.06))` finds the offset $\delta_j$ that best explains the data. Tabulate $\delta_j$ and the RMS error after correction. You should find that joint 3 is the culprit: $\delta_3 = +1.52^\circ$ reduces the RMS to $0.48$ mm, while the best other joint leaves $2.4$ mm.
c. Why can joint 6 not be identified from these data at all? (Look at $\mathcal B_6$ and the direction of the pointer.)
d. Update the calibration (which entry of `servo_offsets.json` changes, and by how much, with $q_\text{model} = q_\text{servo} - \text{offset}$?). Explain why the log of Step 4 could never have revealed this error.

**Checkpoint 5:** `test_tip_checks_and_joint3_offset`. Finally run the whole suite: `LAB_IMPL=starter python -m pytest -q test_lab.py` must show **32 passed**, or 31 passed and 1 skipped in a student copy of the lab (the shared-model check is skipped there) or if Pinocchio is not installed (with both, 30 passed and 2 skipped).

# Part C (optional) — Logging demonstration on a course arm (instructor/TA-operated)

In Lab 5 **students do not handle the course arm.** If an arm is available, an instructor or TA runs the logging procedure below as a demonstration; you watch from outside the keep-out zone, and you may analyse the demonstration log with Steps 4–5. Your first hands-on contact with the arm is the course-arm hardware safety induction at the start of Week 8 Session 4 (hardware rotation 1); the Week 10 Lab 8 hardware station and the project hardware sessions H1 (Weeks 10–11) and H2 (Weeks 12–13) come after it and use this same procedure (project brief §4.1), so watching it now saves time later.

**Safety (values quoted from Course resources: Course robot models and hardware safety checklist; the demonstrator checks them before the arm is powered).**

| # | Rule | Value |
|---|---|---|
| 1 | Software joint limits inside the hardware limits | margin ≥ 10° on every joint (hardware limits ±165°, J6 ±175° → software limits ≤ ±155°, J6 ≤ ±165°) |
| 2 | Taped keep-out zone around the base | radius = reach + 0.3 m ≈ 0.28 + 0.30 = 0.58 m |
| 3 | Joint speed | ≤ 25 % of rated (≤ 40 °/s) on the first run of any new motion and throughout SBEG 361 session H1; ≤ 50 % of rated (≤ 80 °/s) otherwise |
| 4 | Payload | ≤ 50 % of the rated 0.25 kg, i.e. ≤ 0.125 kg, unless the department approves an exception |
| 5 | Induction | every person who touches the arm has completed the hardware safety induction before their first hands-on run |
| 6 | Keep-out discipline | nobody is inside the keep-out zone while the arm can move; a hand may enter for a measurement only with the arm in latched stop and torque disabled (or power removed), with the safety officer's approval |
| 7 | Before every session | emergency stop tested; supervisor present for the whole run |

In this demonstration: the payload is the pointer only; rule 5 means that only inducted staff touch the arm; the whole demonstration is a first run, so rule 3 gives ≤ 40 °/s (the recorded log of Part B peaks at 10.5 °/s).

**Procedure (carried out by the demonstrator; narrate each step to the class).**

1. *Identify and calibrate.* Record the arm ID and SDK/firmware version. Move the arm to its calibration (servo-zero) pose with the vendor tool, check it against the calibration jig or a square, and save `servo_offsets.json` for this arm (same format as the example). Students repeat Step 4a with these offsets.
2. *Dry run in simulation.* Generate the commanded trajectory (for example `np.rad2deg(trajectory.commanded(t))` + offsets, with `trajectory.py`, or the poses of `tip_checks.csv`), run `compare_log` on a log with `meas = cmd` and check limits and speeds before the arm is powered.
3. *Log.* Send the commanded angles and read the measured angles in one loop at a fixed rate ≥ 20 Hz with the vendor SDK (for a myCobot-type arm the Python SDK provides calls to send a list of joint angles with a speed and to read the current joint angles; check the exact names in the installed SDK version). Write every sample immediately as `t, cmd_1..cmd_6, meas_1..meas_6, stop` in servo degrees; set `stop = 1` from the instant the stop command is given near the end of the run (tests HW-6).
4. *Name and describe the files.* `log_<arm>_<yyyymmdd>_demo_<run>.csv` plus a short `.json` with arm ID, date, operator, SDK version, offsets file, sample rate, pointer length and any incident. Film the run with a ruler or grid in view.
5. *Tip checks.* Jog to 3–6 poses (≥ 10° from limits), measure the pointer tip on the grid and with the height gauge (with the arm in latched stop and torque disabled, rule 6; estimate the measurement uncertainty), and save them in the `tip_checks.csv` format.
6. *Analyse* (students, on their laptops) with Steps 4–5, and with the project tool: `python hwcompare.py <log.csv> --limits … --rated …` from the project starter code. Note: `hwcompare.py --model desktop6r` treats the logged angles as **model** angles; subtract the offsets first (write a converted log) before using its HW-3 result.
7. *Power down*, return the arm to its stow pose, leave the bench clear, and record any fault in the fault log.

If no arm is available or there is no demonstration, Part B is the complete lab.

# Deliverables (auto-checked notebook lab; not a formal report)

1. `starter.py` with all functions implemented; the full test suite passes.
2. A short notebook or PDF (≤ 3 pages): the sketch and numbers of Step 2, the MR Studio screenshot of Step 3, the error plots of Step 4b–c with answers to 4a–4d, the table of Step 5b with answers to 5a–5d, and (if a demonstration was run and you chose to) the same analysis of the demonstration log.
3. The reflection below.

Marking (lab category): tests 50 %, plots and answers 35 %, reflection 15 %. Be prepared for a 2-minute oral check: a TA will point to any line of your code and ask you to explain it.

# Reflection (≤ 200 words)

- Your FK passes 32 tests against `modern_robotics`, yet the arm was 5 mm off. Which of the two was wrong, and what does "validated" mean for a kinematic model?
- In a biopsy or an EOD task, who should be told when a calibration offset is changed, and how would you record it so that the next user does not undo it?
- Which source of error in this lab (lag, gravity droop, quantisation, zero offset, probe uncertainty) would you reduce first for project scenario S3, and how?
- **AI-use note (required):** state which AI tools you used (or "none"), what you asked, and what you kept or changed. You must be able to explain every line you submit.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: starter` | Run pytest from inside your Lab code folder, or keep `test_lab.py` next to `starter.py`. |
| Every test fails with `NotImplementedError` | Expected until you implement the functions; set `LAB_IMPL=starter`, not `solution`. |
| `screw_exp` correct for $\theta = \pi/2$ only | You used $p = v\theta$ for a revolute joint; the position part is $G(\theta)v$ (Eq. 3.87). |
| `fk_space` wrong except at $\theta = 0$ | $M$ multiplied on the left, or the exponentials multiplied in reverse order. |
| `fk_body` equals `fk_space` only for one joint | In the body form $M$ is on the **left** and $e^{[\mathcal B_1]\theta_1}$ is next to it. |
| Course-arm $\mathcal S_5$ test fails | Axis 5 points along $-\hat z_s$; check the sign of $\omega_5$ and $v_5 = (-W_1, L_1+L_2, 0)$. |
| URDF test fails from joint 2 on | You did not chain the origins ($T = T\,T_\text{origin}$), or you used the `axis` without rotating it by $R_{s,\text{child}}$. |
| `rpy_to_rot` test fails | Wrong order: URDF is $R_z(\text{yaw})R_y(\text{pitch})R_x(\text{roll})$. |
| Step 4a flange not straight up | Offsets added instead of subtracted ($q_\text{model} = q_\text{servo} - \text{offset}$), or degrees not converted to radians. |
| `compare_log` gives 1001 samples | You did not stop at the first `stop = 1` sample. |
| `fit_joint_offset` returns the search limit | The cost is flat (joint 6, see Step 5c) or the search interval is too small. |
| Part C demonstration: log rate drops below 20 Hz | Reading all six servos is slow on some arms; the demonstrator reads the angles once per loop, writes the file without flushing every line, and records the actual time stamps. |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same (deliverable 1 is `cpp/starter.cpp` instead of `starter.py`); the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`) and with the numbers of the Python reference (including the scipy and Pinocchio values), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the complete helpers: `vec_to_so3`, `vec_to_se3`, `trans_inv`, `adjoint`, `rot_axis`, `read_urdf_joints` with a small URDF/XML reader, `urdf_chain`, `load_joint_log`, `load_offsets` — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `plot_results.py` (plots for Steps 4–5), `CMakeLists.txt`. The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 5 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*screw_axis*,*screw_exp*,*4_5_exp*"   # only some tests (like pytest -k)
python cpp/plot_results.py                     # plots of Step 4b-c and the table of Step 5b
```

With the untouched starter, the program builds and all 21 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `screw_axis`, `screw_exp` (Step 1); `fk_space`, `fk_body`, `body_screws`, `course_arm_screws` (Step 2); `rpy_to_rot`, `poe_from_urdf` (Step 3); `servo_to_model` (two overloads: one joint vector, or one joint vector per row), `tool_points`, `compare_log` (Step 4); `tip_errors_mm`, `fit_joint_offset` (Step 5). The helpers in `helpers.cpp` are given, as the "complete helper" functions of `starter.py`. Do not call `mrcpp` or `.inverse()` in `starter.cpp`.

**Checkpoints.** 1: `./build/lab_tests -tc="*screw_axis*,*screw_exp*,*4_5_exponentials*"`; 2: `-tc="*fk_space*,*ur5_mr*,*rrp*,*body*,*wam*,*course_arm_screws*,*course_arm_mr*"`; 3: `-tc="*rpy*,*urdf*"`; 4: `-tc="*servo*,*tool_points*,*compare_log*"`; 5: `-tc="*tip_checks*"`; finally all 21 test cases pass (`./build/lab_tests`). The Python track counts 32 tests because pytest counts each of the six robots of `test_fk_space_matches_mr` and `test_fk_body_equals_space` separately (the C++ track checks the six robots in one test case each), and `test_course_arm_copy_is_identical_to_shared` (a data check that is skipped in your copy) has no C++ counterpart.

**Printing and plotting.** To print values (Step 2, 4a, 5a), add a temporary `TEST_CASE` at the end of `test_lab.cpp` with lines such as `std::cout << lab::fk_space(M, S, q) << "\n";` (include `<iostream>`; remove it before you submit). For the plots of Step 4b–c and the table of Step 5b, `./build/lab_tests` writes `lab5_cpp_log_errors.csv` and `lab5_cpp_tip_fits.csv` into the folder you run it from (when your `compare_log` and `fit_joint_offset` return results of the right size); `python cpp/plot_results.py` plots them (`lab5_step4_errors.png`) and prints the table. Step 4d uses only the data file: compute it with the NumPy line given in 4d, or with a finite difference of `log.cmd` in C++.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| screw axis, `Slist` / `Blist`, `theta` | `sbeg::Vector6d`; `Eigen::MatrixXd` 6 × n (one axis per **column**); `Eigen::VectorXd` |
| `screw_axis(omega, q)`, `screw_axis(v=v)`, `screw_axis()` | `screw_axis(omega, q)`, `screw_axis(std::nullopt, std::nullopt, v)`, `screw_axis()` — the arguments are `std::optional<Eigen::Vector3d>` (`std::nullopt` = `None`) |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `course_arm_screws(**COURSE_ARM_GEOMETRY)` returns `(M, Slist)` | the six numbers in order; returns `std::pair`: `auto [M, S] = lab::course_arm_screws(W1, W2, L1, L2, H1, H2);` |
| joint dicts of `read_urdf_joints`; `limit` may be `None` | `lab::UrdfJoint` (fields `name, type, parent, child, xyz, rpy, axis, limit`); `limit` is a `std::optional<std::pair<double, double>>` |
| `poe_from_urdf` returns `(M, Slist, names, limits)` | `lab::PoeModel` with the fields `M, Slist, names, limits` (limits n × 2, `std::numeric_limits<double>::infinity()` where none) |
| log dict; `log["stop"]` may be `None` | `lab::JointLog` (fields `t, cmd, meas, stop`); `stop.size() == 0` when there is no stop column |
| `compare_log` returns a dict | `lab::LogComparison` with the same field names |
| `(N, n)` array of joint vectors | `Eigen::MatrixXd` with one **row** per sample; `servo_to_model` has a `VectorXd` and a `MatrixXd` overload |
| `np.deg2rad`, `np.pi` | multiply by `PI / 180.0` with a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |
| `test_urdf_matches_pinocchio` calls Pinocchio | compares with Pinocchio's poses stored in `cpp_cases.json` (runs without Pinocchio) |
| plots with Matplotlib | CSV files written by `lab_tests` + the given `plot_results.py` |

A matrix-size mistake (e.g. reading column 6 of a 6 × 6 `Slist`) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `call to 'servo_to_model' is ambiguous` | an Eigen expression (e.g. `MatrixXd::Zero(1, 6)`) passed directly | store it in a `Eigen::MatrixXd` (or `Eigen::VectorXd`) variable first |
| `screw_exp` correct only for the prismatic case | the revolute position part must be $G(\theta)v$ (Eq. 3.87), not $v\theta$ | see the Python troubleshooting table above |
| `test_tip_checks_and_joint3_offset` never finishes | the golden-section loop does not shrink the interval | each iteration must move `a` or `b`; stop when `b - a <= tol_deg` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
