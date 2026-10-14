---
title: "Lab 6 — DH models in the Robotics Toolbox vs PoE; C++/Eigen FK"
subtitle: "SBEG 361 Introduction to Robotics — Week 7"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before Hardware rotation 1 in Week 8) |
| CLOs | 361-4 (FK with DH parameters and PoE; convert between them); 361-7 (implement and validate against a reference library, a second library and a compiled implementation) |
| Part A (auto-tested) | Python DH tools: `starter.py` (you edit), `test_lab.py` (23 auto-checks) |
| Part B (C++, guided) | C++17 + Eigen + CMake: `cpp/` — you complete `fk_starter.cpp`; the comparison with Python is checked by the last test (skipped automatically if no compiler or Eigen is installed) |
| Data | `data/q_samples.csv` (1000 course-arm joint vectors inside the joint limits); `course_arm_model.py` (course-arm PoE model and the RCM holder; this local lab copy is self-contained, with the same numbers as the course model); `make_data.py` shows how the data were made (read it **after** the lab) |
| Reference | `modern_robotics` (`FKinSpace`); Robotics Toolbox for Python (`DHRobot`, `RevoluteDH`, `RevoluteMDH`, `PrismaticDH`); Corke's printed examples (§7.1–7.2) |
| Reading | MR App. C (printed pp. 587–598), especially §C.3 (p. 592) and §C.5 (Eqs. C.6–C.15, pp. 595–597); Corke §7.1–7.2 (pp. 137–146) and §7.5.3 (modified DH, pp. 160–162) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation only); the same model is used on the course arms in Hardware rotation 1 (Week 8) |

The functions you write today join your `sbeg361_kinematics` library as module `dh.py`, next to Lab 5's `fk_space`.

**Application context.**
*Biomedical:* surgical and rehabilitation robots are often documented with DH tables (papers, controller files), while our analysis uses the PoE model. You will convert in both directions and model a **remote-centre-of-motion (RCM) laparoscope holder** (Workshop Problem 2), checking that the instrument shaft always passes through the trocar port.
*Defence (protective use):* EOD arms and field-hospital telesurgery arms run their kinematics in compiled controller code. Part B implements the course-arm FK in C++/Eigen and measures how much faster it is than Python, the kind of evidence an engineer needs before choosing where a teleoperation loop runs.
All robot data are **illustrative values** except the textbook examples.

# Learning objectives

By the end of the lab you can:

1. Implement the standard (Corke Eq. 7.2) and modified (MR §C.3) DH link transforms and FK from a DH table with revolute and prismatic joints, base and tool.
2. Write the DH table of the course arm in both conventions and validate it against the PoE model to $10^{-12}$ on 1000 configurations.
3. Convert DH → PoE (MR Eqs. C.11–C.13) and standard ↔ modified DH, and recognise the convention of a published table.
4. Build the same robot in the Robotics Toolbox for Python and confirm that three independent implementations agree.
5. Implement PoE and DH FK in C++17 with Eigen, compare the results with Python and measure the speed-up.

# Pre-lab check (complete before the session; 5 questions)

1. A standard DH row has $\theta = 90^\circ$, $d = 0.1$, $a = 0.2$, $\alpha = 0$. Where is the origin of frame $\{i\}$ in frame $\{i-1\}$?
2. A paper's DH table has columns $\alpha_{i-1}$, $a_{i-1}$, $d_i$, $\theta_i$. Which Robotics Toolbox class do you use for its revolute joints?
3. In a standard DH model, frame $\{i-1\}$ at $q = 0$ has $\hat z = (0,1,0)$ and origin $(0.1, 0, 0.2)$. Joint $i$ is revolute. Write $\mathcal S_i$.
4. Which DH parameter is the joint variable of a prismatic joint, and which parameter is then constant although it is a joint variable for a revolute joint?
5. What angle (in rad and degrees) is one count of a 12-bit joint encoder?

*Answers (for self-check):* 1. $(a\cos\theta, a\sin\theta, d) = (0, 0.2, 0.1)$. 2. `RevoluteMDH` (modified convention). 3. $\mathcal S_i = (0,1,0,\ -0.2, 0, 0.1)$ since $-\hat z\times o = -(0.2, 0, -0.1)$. 4. $d_i$ is the variable; $\theta_i$ is constant. 5. $2\pi/4096 = 1.534\times10^{-3}$ rad $= 0.088^\circ$.

# Setup

```bash
cd "<your Lab code folder>"   # or your Classroom 50 repo
source ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/activate                          # or the course Docker image / JupyterHub
python -c "import numpy, modern_robotics, roboticstoolbox; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py               # 22 failures expected (1 provided-helper check passes): 21 NotImplementedError + the C++ test ("fkPoE not implemented"; skipped without a compiler)
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Rules: do **not** call `modern_robotics`, `roboticstoolbox` (except inside `rtb_robot`) or `spatialmath` in `starter.py`; they are the references. The helpers already in `starter.py` (`screw_exp(S, theta)`, `fk_space(M, Slist, theta)` with the Lab 5 signature, `trans`, `rotz_h`, `rotx_h`, `_joint_values`) may be used.

**Table format.** A DH table is a dict with `convention` (`"standard"` or `"modified"`), `types` (e.g. `"RRPR"`), arrays `d`, `a`, `alpha`, `offset` (m, rad) and $4\times4$ `base` and `tool`. In the modified convention row $i$ stores $a_{i-1}$ and $\alpha_{i-1}$. Revolute: $\theta_i = \text{offset}_i + q_i$. Prismatic: $d_i \to d_i + q_i$, $\theta_i = \text{offset}_i$. The tables `course_arm_dh()` / `course_arm_mdh()` you write use this format; the course-arm numbers are in the lab copy `course_arm_model.py`.

# Part A — Procedure (auto-tested)

Run the tests after each step. A checkpoint is complete when all listed tests pass; ask a TA to sign it off.

## Step 1 — Link transforms (6 min)

Implement `dh_std` (Corke Eq. 7.2) and `dh_mod` (MR §C.3, the unnumbered matrix at the start of §C.3 on p. 592) entry by entry.

**Checkpoint 1:** `test_dh_std_corke_link_example` (Corke p. 140: `Link([0 0.1 0.2 pi/2]).A(0.5)` has translation $(0.1755, 0.0959, 0.1)$), `test_dh_std_is_rz_tz_tx_rx`, `test_dh_mod_is_rx_tx_tz_rz`.

## Step 2 — FK from a table (8 min)

Implement `link_transforms`, `fk_dh` (Corke Eq. 7.4) and `link_frames` (returns $[T_{s0}, T_{s1}, \dots, T_{sn}]$, base included, tool excluded).

**Checkpoint 2:** `test_fk_dh_corke_two_link` (Corke §7.2.1, p. 142: $q = (\pi/4, -\pi/4)$ puts the tip at $(1.7071, 0.7071, 0)$), `test_fk_dh_mr_example_c1_equals_example_4_1` (MR's modified table of Example C.1 reproduces the PoE of Example 4.1 exactly), `test_fk_dh_prismatic_mr_exercise_5_7` (RRP robot: $T(90^\circ, 90^\circ, 1)$ has $p = (0,0,6)$), `test_fk_dh_random_tables_both_conventions`.

## Step 3 — The course arm in DH form (10 min)

On paper, assign standard DH frames to the course arm (Workshop Problem 3 gives the table; derive it yourself first from the PoE screw table in `course_arm_model.py`: same zero pose and joint signs, base $= I$). Enter it in `course_arm_dh()`, then convert it by hand to the MR convention in `course_arm_mdh()`.

**Checkpoint 3:** `test_course_arm_dh_matches_poe` (all 1000 samples, $10^{-12}$), `test_course_arm_mdh_matches_poe`, `test_course_arm_ready_pose_checkpoint`. At `Q_HOME` $= (0, -0.6, 1.4, -2.4, -1.57, 0)$ the flange is at $(0.23407,\ 0.06344,\ 0.08106)$ m with $\hat z_b = (0.0292, 0.0008, -0.9996)$: the tool points almost straight down, the usual "ready" pose for a pick.
*Question A3:* which constant transform did you need, and could you have used a joint offset instead?

## Step 4 — Conversions (10 min)

Implement `poe_from_dh` (joint-$i$ axis = $\hat z$ of frame $\{i-1\}$ for standard, of frame $\{i\}$ for modified; revolute $\mathcal S_i = (\hat z, -\hat z\times o)$, prismatic $(0,\hat z)$; $M = T(0)$), `std_to_mod` and `mod_to_std` (regroup the factors as in Corke §7.5.3).

**Checkpoint 4:** `test_poe_from_dh_course_arm_both_conventions` (your $\mathcal S_i$ and $M$ equal the course-arm PoE model), `test_poe_from_dh_textbook_chains` (MR Examples C.1/4.1 and Exercise 5.7), `test_poe_from_dh_random_tables`, `test_std_mod_round_trip`.
*Question A4:* the PoE model of the course arm has 36 screw-axis numbers + $M$; the DH table has 24. Which one would you rather receive from a manufacturer, and why? (MR §C.6, pp. 597–598.)

## Step 5 — RCM holder and the Robotics Toolbox (8 min)

a. Implement `rcm_dh()` (Workshop Problem 2) and `rcm_distance(T, port)` (distance from the port to the shaft line). Verify that the distance stays below $10^{-12}$ m for 500 random configurations: the RCM is built into the kinematics.
b. Implement `rtb_robot(table)` with `RevoluteDH`/`PrismaticDH` (standard) or `RevoluteMDH`/`PrismaticMDH` (modified). Print `rtb_robot(course_arm_dh())`: the Toolbox shows your table. Compare `robot.fkine(q).A` with `mr.FKinSpace` for a few samples, and plot the arm with `robot.plot(Q_HOME)` (optional; needs a display).
c. Load the MR Example C.1 table *by mistake* as standard DH (`dict(table, convention="standard")`) and compare the end-effector positions at $q = 0$: they differ by $0.616$ m (Workshop Problem 5b). Write one sentence on how you will avoid this error with tables from papers.

**Checkpoint 5:** `test_rcm_dh_matches_poe_and_closed_form`, `test_rcm_property_and_distance`, `test_rtb_course_arm_both_conventions`, `test_rtb_prismatic_rcm_and_rrp`.

## Step 6 — Python side of the C++ comparison (8 min)

a. Implement `read_pose_csv`, `pose_errors` (max position error and max rotation angle of $R_a^{\mathsf T}R_b$, MR Eq. 3.54) and `time_fk` (best-of-`repeat` mean time per call with `time.perf_counter()`).
b. Time four Python FK implementations on 300 samples: your `fk_dh(course_arm_dh(), q)`, the provided `fk_space`, `mr.FKinSpace` and `rtb_robot(...).fkine`. Record the times in a table. (On the instructor's laptop: about 17, 66, 200 and 44 µs per call; your numbers will differ.) Why is the DH version faster than the PoE version in Python?
c. **Encoder resolution.** For all 1000 samples, compute the flange displacement when every joint is off by one 12-bit encoder count ($q + 2\pi/4096$). The worst case is $1.016$ mm and the mean $0.517$ mm. Compare with the 1.5 mm needle-tip budget of project scenario S3, which uses this arm with a 0.06 m guide.

**Checkpoint 6:** `test_read_pose_csv_and_pose_errors`, `test_time_fk_returns_seconds_per_call`, `test_encoder_quantisation_checkpoint`.

# Part B — C++/Eigen FK of the course arm (guided; finish at home)

The folder `cpp/` contains `fk.hpp` (declarations and course-arm data), `robot_data.cpp` (screw axes, $M$, DH table — the same numbers), `main.cpp` (driver: reads the samples, writes pose files, times both FKs), `fk_starter.cpp` (yours) and `CMakeLists.txt`; the reference `fk_solution.cpp` is released after the deadline.

**B1 — Build.** Install a C++17 compiler, CMake ≥ 3.16 and Eigen 3 (Ubuntu/WSL: `sudo apt install build-essential cmake libeigen3-dev`; macOS: `xcode-select --install` and `brew install cmake eigen`; the course Docker image has all three).

```bash
cd cpp                                                   # inside your Lab code folder
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release           # builds fk_starter.cpp
cmake --build build
./build/fk_compare ../data/q_samples.csv .               # "error: fkPoE not implemented" until B2 is done
```

**B2 — Implement** `dhStd`, `matrixExp6`, `fkPoE` and `fkDH` in `fk_starter.cpp`. Use the Eigen comma initialiser (`A << ...;`), `T.block<3,3>(0,0)` and `T.block<3,1>(0,3)`, and mirror your Python code line by line. Rebuild and run: the program prints the largest entry of $|T_\text{PoE} - T_\text{DH}|$ (it must be below $10^{-15}$) and the time per call of each method, and writes `poses_poe_cpp.csv`, `poses_dh_cpp.csv` and `timing_cpp.csv`.

**B3 — Compare with Python.**

```python
import numpy as np, modern_robotics as mr, starter as lab
from course_arm_model import SLIST, M_HOME
Q = np.loadtxt("data/q_samples.csv", delimiter=",", skiprows=1)
ref = np.array([mr.FKinSpace(M_HOME, SLIST, q) for q in Q])
for f in ("cpp/poses_poe_cpp.csv", "cpp/poses_dh_cpp.csv"):
    print(f, lab.pose_errors(ref, lab.read_pose_csv(f)))        # both below 1e-12 m and 1e-7 rad
```

The last test, `test_cpp_fk_matches_python`, compiles `main.cpp + robot_data.cpp + fk_<LAB_IMPL>.cpp` directly with your compiler and checks the same thing; it is skipped if no compiler or Eigen headers are found (set `EIGEN3_INCLUDE_DIR` if Eigen is in an unusual place).

**B4 — Speed-up.** Fill in the table and compute the speed-up factor of C++ over Python for each method. On the instructor's laptop C++ needs about 0.19 µs (PoE) and 0.12 µs (DH) per call, a factor of 100–1000 over Python. A teleoperation loop of an EOD arm or a surgical robot typically runs at 1 kHz with FK, Jacobian and IK inside: how much of the 1 ms budget would Python FK use? Does it matter for the 50 Hz hardware runs of the project?

| Method | Python (µs/call) | C++ (µs/call) | Speed-up |
|---|---|---|---|
| PoE (own) | | | |
| DH (own) | | | |
| `mr.FKinSpace` / `rtb fkine` | | — | — |

# Deliverables (auto-checked lab; not a formal report)

Submit to your Classroom 50 repository before Hardware rotation 1 (Week 8):

1. `starter.py` with all functions implemented; `LAB_IMPL=starter python -m pytest -q test_lab.py` → **23 passed** (22 passed and 1 skipped if you could not install a C++ compiler or Eigen; then submit B4 using the timing printed by a lab PC).
2. `cpp/fk_starter.cpp` completed, plus the generated `timing_cpp.csv`.
3. A one-page `lab6_answers.md`: your hand-derived course-arm table (standard and MR), answers to A3, A4, 5c, 6b, 6c and the B4 table with comments.
4. The reflection below.

# Reflection (≤ 200 words, in `lab6_answers.md`)

- Which representation (DH or PoE) would you use for your project robot, and what would you do with a DH table you find in a paper?
- What did the three-way agreement (your code, `modern_robotics`, Robotics Toolbox) and the C++ comparison give you that a single implementation would not?
- **AI-use note** (course AI policy, Course syllabus §8 AI-use policy): tool, what you asked, what you kept, and how you checked it. Be ready to explain any line of `starter.py` or `fk_starter.cpp` in an oral check.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| FK off by a rotation of $180^\circ$ about $\hat z_b$ only | Missing tool $\mathrm{Rot}(\hat z,\pi)$ on the course arm (Step 3) |
| Errors of exactly one link length in the wrong place | Mixed conventions: $a_i$ in row $i$ (standard) vs row $i+1$ (MR) |
| Joint 2 moves the wrong way | $\alpha_1$ sign: $\hat z_1$ must point along $+\mathcal S_2$ |
| `poe_from_dh` correct for standard, wrong for modified | Use frame $\{i\}$, not $\{i-1\}$, for the modified convention |
| Prismatic joint in the Toolbox does not move / moves twice | `PrismaticDH(theta=offset, a=..., alpha=..., offset=d)`: the Toolbox adds `offset` to $q$ |
| `std_to_mod` round trip fails only when $a_n$ or $\alpha_n \ne 0$ | The leftover $\mathrm{Trans}(\hat x, a_n)\mathrm{Rot}(\hat x, \alpha_n)$ must go into the tool |
| CMake: "Eigen3 not found" | Install `libeigen3-dev` / `brew install eigen`, or `-DEIGEN3_INCLUDE_DIR=/path/to/eigen3` |
| C++ times of 0 ns | Built in Debug or the loop was optimised away: build with `-DCMAKE_BUILD_TYPE=Release` and keep the `sink` variable in `main.cpp` |
| Windows: `fk_compare` not found | The executable is `build\Release\fk_compare.exe` with the Visual Studio generator |

# Project teams are formed this week

The course project (see the project brief, 15 % of the course mark) starts now (see also the "Project" section of this week's lecture notes). By the **end of Week 7** (exact deadline on the course site):

1. **Form a team of three.** Students without a team are placed by the instructor; if the class size is not a multiple of three, the instructor arranges the remaining teams.
2. **Read** `project_brief.md` §1–4 and §8–10 and the six scenario pages of `scenario_sheets.md` together.
3. **Choose one scenario** — three biomedical (S1 rehabilitation guide, S2 feeding/assistive fetch, S3 needle positioning with RCM) and three defence (S4 EOD arm, S5 casualty-extraction/litter loading, S6 field-hospital telesurgery with RCM). All six are equal choices; at most **two teams per scenario**, first come first served on the registration form. Defence scenarios are limited to protective, medical and humanitarian tasks (brief §8).
4. **Register** the team name, members and scenario on the course registration form, then accept the Classroom 50 assignment; your team repository is created from `starter/`. Post the repository link on the form.
5. **First commit:** each member clones the repository, runs `pytest -q` (38 passed, 6 skipped before you start) and `python baseline.py N` for your scenario, and commits a `TEAM.md` with names, IDs and draft roles (brief §10). Every member must be able to explain every part of the work.
6. **Plan ahead:** the hardware safety induction for all members is at the start of Week 8 Session 4 (your first hands-on contact with the course arm; nobody handles the arm before it); Weeks 8–9 model building; Week 10 proposal (`proposal_template.md`); the hardware booking sheet opens in Week 9 (brief §4.1).

*Link to today:* `poe_from_dh` and `rtb_robot` let you cross-check your scenario robot in a second library (S3 uses the course arm with a 0.06 m needle guide; S3 and S6 use the RCM idea of Step 5). Record such cross-checks in your repository: they are evidence for CLO 361-7 in the design review.

# C++ track

You may do Part A of this lab in C++17 instead of Python. The functions, data file, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the random inputs and reference numbers of the Python tests and with the Robotics Toolbox values computed in Python, so both tracks are marked against identical values. **Part B (`fk_compare`, `fk_starter.cpp`) is the same for both tracks** and is built by the same `CMakeLists.txt`.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations, the `DHTable` struct and the course-arm/RCM data of `course_arm_model.py` — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt` (builds `lab_tests` and the Part B program `fk_compare`); Part B files `fk.hpp`, `robot_data.cpp`, `main.cpp`, `fk_starter.cpp` as described above. The data file is the same `data/q_samples.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). No separate Eigen installation is needed: both programs use the Eigen copy in `cpp_support`.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 6 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build                            # builds lab_tests and fk_compare
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*dh_std*,*dh_mod*"      # only some tests (like pytest -k)
./build/fk_compare data/q_samples.csv cpp      # Part B (writes the pose and timing files into cpp/)
```

With the untouched starter, the program builds and 22 of the 23 test cases fail with `TODO: <function>`; `test_provided_helpers_use_library_signatures` passes (it checks the given helpers), exactly as in Python. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `dh_std`, `dh_mod` (Step 1); `link_transforms`, `fk_dh`, `link_frames` (Step 2); `course_arm_dh`, `course_arm_mdh` (Step 3); `poe_from_dh`, `std_to_mod`, `mod_to_std` (Step 4); `rcm_dh`, `rcm_distance` (Step 5); `read_pose_csv`, `pose_errors`, `time_fk` (Step 6). Given: `skew`, `screw_exp`, `fk_space`, `trans`, `rotz_h`, `rotx_h`, `joint_values` (in `starter.cpp`) and, in `lab.hpp`, `vec({...})` (a `VectorXd` from a list of numbers), the constants `H1, H2, L1, L2, W1, W2, PI, ENC_RES, RCM_TIP0` and the functions `course_arm_slist()`, `course_arm_m()`, `q_home()`, `rcm_port()`. Do not call `mr::...` in `starter.cpp`: `mrcpp` is the reference. `rtb_robot` is not written in C++ (see below).

**Checkpoints.** 1: `./build/lab_tests -tc="*dh_std*,*dh_mod*"`; 2: `-tc="*fk_dh*"`; 3: `-tc="test_course_arm_*"`; 4: `-tc="*poe_from_dh*,*std_mod*"`; 5: `-tc="*rcm*,*rtb*"`; 6: `-tc="*read_pose*,*time_fk*,*encoder*,*cpp_fk*"`. All 23 test cases pass (`./build/lab_tests`): the same 23 tests as `test_lab.py`, with the same names and tolerances.

- *Step 5b:* the C++ test cases `test_rtb_course_arm_both_conventions` and `test_rtb_prismatic_rcm_and_rrp` compare your `fk_dh` with the Robotics Toolbox `fkine` values of the same tables (computed in Python, stored in `cpp_cases.json`). Printing and plotting the Toolbox model is optional in the C++ track. *Step 5c:* copy the MR Example C.1 table, set `convention = "standard"` and compare the positions of `fk_dh` at $q = 0$.
- *Step 6b:* `test_time_fk_returns_seconds_per_call` prints the C++ time per call of your `fk_dh`, the given `fk_space` and `mr::fkin_space` (300 samples). For the Python column of the table, time the provided Python functions, e.g. `python -c "import timeit, numpy as np, modern_robotics as mr; from course_arm_model import SLIST, M_HOME; Q = np.loadtxt('data/q_samples.csv', delimiter=',', skiprows=1)[:300]; print(min(timeit.repeat(lambda: [mr.FKinSpace(M_HOME, SLIST, q) for q in Q], number=1, repeat=3)) / 300 * 1e6, 'us/call')"`.
- *Part B, B3:* in the C++ track `test_cpp_fk_matches_python` writes the poses of your `fk_dh` and of `fk_space` for all 1000 samples to pose files, reads them back with your `read_pose_csv` and checks them with your `pose_errors` against `mr::fkin_space` (below $10^{-12}$ m and $10^{-7}$ rad) and against the Python reference. For the `fk_compare` files, the program's own printout (largest entry of $|T_\text{PoE} - T_\text{DH}|$) replaces the Python snippet of B3.

**Deliverables.** As above, with `cpp/starter.cpp` instead of `starter.py` in item 1 (all 23 C++ test cases pass) and the C++ times of Step 6b in the B4 table.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| DH table = `dict` | `lab::DHTable` struct with the same field names (`convention`, `types`, `d`, `a`, `alpha`, `offset`, `base`, `tool`); `dict(table, convention="modified")` becomes a copy `lab::DHTable t = table; t.convention = "modified";` |
| `course_arm_model.py` (`SLIST`, `M_HOME`, `Q_HOME`, `ENC_RES`, `RCM_PORT`, `RCM_TIP0`, `H1` …) | given in `lab.hpp`: `course_arm_slist()`, `course_arm_m()`, `q_home()`, `ENC_RES`, `rcm_port()`, `RCM_TIP0`, `H1` … |
| `_joint_values(table, q)` | `joint_values(table, q)` (given) |
| list of $4\times4$ arrays, `(N, 4, 4)` array | `std::vector<Eigen::Matrix4d>` |
| `poe_from_dh` and `pose_errors` return tuples | `std::pair`: `const auto [S, M] = lab::poe_from_dh(tab);` |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `time_fk(fn, qs)` with `time.perf_counter()` | `fn` is a `std::function` (pass a lambda), `qs` an `Eigen::MatrixXd` (row = sample), `std::chrono::steady_clock` |
| `rtb_robot` (Robotics Toolbox for Python) | not available in C++: the `test_rtb_*` cases compare your `fk_dh` with stored Toolbox values |
| `test_cpp_fk_matches_python` compiles the Part B program | checks your C++ functions and the pose-file round trip (see B3 above) |
| `np.pi` | `PI` from `lab.hpp` (`M_PI` is not standard C++) |

Fill the vectors of a table with the given `lab::vec({...})`. A matrix-size mistake (e.g. filling `t.d << ...` before `t.d.resize(6)`) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". The `CMakeLists.txt` builds Part B against the Eigen copy in `cpp_support`, so the "Eigen3 not found" error of the Troubleshooting table no longer occurs with CMake; the Python test `test_cpp_fk_matches_python` still needs Eigen headers — if it is skipped, set `EIGEN3_INCLUDE_DIR` to the folder `third_party/eigen` inside `cpp_support`.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| Build stops in `fk_starter.cpp` | a compile error in Part B also stops `lab_tests` | fix it, or build only the tests: `cmake --build build --target lab_tests` |
| Eigen size error in a table test | a `VectorXd` field of `DHTable` filled with `<<` before `resize(n)` | use the given `vec`: `t.d = lab::vec({H1, 0, 0, W1, H2, W2});` (or `Eigen::VectorXd::Zero(n)`) |
| `lab_tests` stops with "terminating due to uncaught exception ... Too few coefficients" | a comma initialiser (`A << ...;`) with fewer numbers than the matrix has entries (16 for a `Matrix4d`) | count the entries; this error ends the whole test run instead of failing one test |
| `poe_from_dh` wrong only for modified tables | used `frames[i]` instead of `frames[i + 1]` | joint $i$ of a modified table is the $z$ axis of frame $\{i\}$ |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
