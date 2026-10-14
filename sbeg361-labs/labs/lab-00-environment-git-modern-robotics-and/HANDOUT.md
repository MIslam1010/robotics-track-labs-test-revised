---
title: "Lab 0 — Environment, Git, modern_robotics and MR Studio"
subtitle: "SBEG 361 Introduction to Robotics — Week 1"
---

# Overview

| | |
|---|---|
| Session | Session 4, Computational Lab (50 min in the lab + up to 60 min to finish at home) |
| CLOs | 361-1 (DOF and C-space), 361-7 (implement and test kinematic code against a reference library) |
| Reading | MR §2.1–2.3 (printed pp. 11–28): Table 2.1 (p. 17), Eq. 2.4 (p. 18), Table 2.2 (p. 26); the 2R arm of Lecture B (MR Eq. 6.1, p. 217) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks), `lab/data/mechanisms.json` |
| Hardware | None (Tier 0, simulation only). The robot-lab safety briefing (safety sheet) is part of this session; this is not the course-arm safety induction of checklist rule 5, which is in Week 8 (syllabus §7). |
| Assessment | Auto-checked; counts in the lab category only for completion this week |

## Learning objectives

By the end of this lab you can:

1. Run the course Python environment (venv, Docker image or browser fallback) and report its versions.
2. Use Git for your lab work (commit; push if your course uses Classroom 50) and submit a lab.
3. Implement Grübler's formula (MR Eq. 2.4) and the joint table (MR Table 2.1), and use them on biomedical and
   defence mechanisms.
4. Handle the topology of joint angles in code: wrap angles to $[-\pi, \pi)$, measure distances on the torus $T^n$,
   and print C-space topologies (MR §2.3.1).
5. Show numerically that latitude–longitude coordinates of the sphere are singular at the poles (MR §2.3.2).
6. Call `modern_robotics` for the first time and check your own code against it.
7. Find your way around MR Studio, the course simulator.

# Pre-lab check (complete before the session, 5 questions, ≈ 10 min)

1. In Grübler's formula $\text{dof} = m(N-1-J) + \sum f_i$, does $N$ include the ground link? What are $m$ for planar
   and spatial mechanisms?
2. How many constraints does a universal joint impose in a spatial mechanism? And a spherical joint?
3. Why can a revolute joint angle be described by the circle $S^1$ instead of the line $\mathbb{R}$? Give two angle
   values that describe the same joint configuration.
4. A point moves due east on the unit sphere at angular speed $\omega$ at latitude $\varphi$. Derive $\dot\lambda$ (the
   longitude rate) as a function of $\omega$ and $\varphi$.
5. What is the difference between `git commit` and `git push`?

*Answers for self-check:* (1) yes; $m = 3$, $m = 6$. (2) U: $6 - 2 = 4$; S: $6 - 3 = 3$. (3) $\theta$ and
$\theta + 2\pi$ give the same link position, so the angle "wraps around", e.g. $350^\circ$ and $-10^\circ$.
(4) $\dot\lambda = \omega/\cos\varphi$. (5) `commit` records a snapshot in your local repository; `push` uploads your
commits to a remote (e.g. Classroom 50), where the TA and the autograder can see them. Without a remote,
`commit` still gives you a history you can go back to.

# Robot-lab safety briefing (safety sheet; first 5 min, mandatory)

This is not the course-arm safety induction of checklist rule 5, which is in Week 8 (syllabus §7).

Even though Lab 0 is simulation only, everyone signs the robot-lab safety sheet today:

* Never enter the marked workspace of a powered arm; the arm can move faster than you can react.
* Know where the **emergency stop** is before any hardware session, and test it at the start.
* Run new code in simulation first (sim-first rule), then on hardware at reduced speed.
* One person per team is the "safety operator" with the hand on the E-stop during every hardware run.
* Report any collision, pinch or unexpected motion to the TA immediately, even without injury.

(Standards such as ISO 10218 and ISO/TS 15066 are covered in Week 13.)

# Setup

Choose **one** route. All three give the same packages (Python 3.11+, NumPy, SciPy, Matplotlib, `modern_robotics`,
pytest).

* **A. Lab PC / your laptop (venv):** `python3 -m venv .venv && source .venv/bin/activate && pip install -r
  requirements.txt` (Windows: use WSL2 Ubuntu 24.04).
* **B. Docker (macOS, Windows, Linux):** `docker run -it -v "$PWD":/work <course image>` (image name on the course
  site).
* **C. Browser fallback:** the course JupyterHub/Colab notebook (link on the course site). Use it if A and B fail;
  nobody should be blocked by installation.

MR Studio runs on the lab PCs (`MR_Studio.command` or `python run_app.py` in the MR-Studio folder).

# Procedure and checkpoints

Work in `starter.py`. After each checkpoint run the tests for it, e.g.
`LAB_IMPL=starter pytest -q test_lab.py -k grubler`. Show each ✅ checkpoint to the TA or record it in your notebook.
Timing: safety briefing 5 min + checkpoints 8 + 8 + 5 + 7 + 5 + 5 + 7 = 50 min. Finish any checkpoint you do not
complete in the session at home (up to 60 min).

## Checkpoint 1 — Environment and Git (8 min)

Your course uses one of two routes (the course site says which):

* **Route A — Classroom 50:** accept the Lab 0 assignment (accept link from the course site, or
  `gh student accept`) and clone your new repository (`git clone <your repo URL>`); the files are already in it.
  One-time: install the GitHub CLI and the student tool (`gh extension install foundation50/gh-student`, then
  `gh student login`).
* **Route B — course lab repository:** download `sbeg361-labs` (ZIP or `git clone`) from the course site, run its
  setup script once (`bash setup.sh`, or `setup.ps1` on Windows) and work in `labs/lab-00-…/`.

1. Configure Git once: `git config --global user.name "Your Name"` and
   `git config --global user.email <university email>`.
2. Implement `env_report()`. Run `pytest -q -k env` (with `LAB_IMPL=starter`; in Route B also `python run_lab.py 0`).
3. `git add . && git commit -m "Lab 0: environment check"`. Route A: also `git push` and check that the commit appears
   on GitHub. Pushing is not submitting: when the lab is finished, run `gh student submit` (or **Submit** in the web
   app); the autograder then scores that version and the result appears under **Releases** in your repository.

✅ *Checkpoint 1:* `test_env_report` passes and your first commit exists (Route A: on the remote).

## Checkpoint 2 — Joints and Grübler's formula (8 min)

1. Complete `JOINT_FREEDOMS` (MR Table 2.1 plus RP = 2, E = 3, F = 0) and implement `joint_freedoms`,
   `joint_constraints` and `grubler`.
2. Run `pytest -q -k "joint or grubler"`. The tests use MR Examples 2.3–2.8 (four-bar, slider-crank, five-bar,
   overlapping joints, parallelogram, Delta, Stewart–Gough).

✅ *Checkpoint 2:* all joint and Grübler tests pass. Explain to the TA why the parallelogram test expects 0 although
the mechanism moves.

## Checkpoint 3 — Mechanism table: hospital and field (5 min)

1. Implement `dof_table(path)` that reads `data/mechanisms.json` and returns `{name: dof}`.
2. Print the table sorted by application (`"app"` field). Which entries have dof ≤ 0, and what does that mean
   physically (locked arm, over-constrained exoskeleton)?

✅ *Checkpoint 3:* `test_dof_table` passes.

## Checkpoint 4 — C-space topology in code (7 min)

1. Implement `wrap_to_pi`, `torus_distance` and `cspace_topology`.
2. Experiment: for a 2R arm, compare `np.linalg.norm(q2 - q1)` with `torus_distance(q1, q2)` for
   $q_1 = (350^\circ, 0)$, $q_2 = (10^\circ, 0)$. Which one should a motion planner use, and why?
3. Optional plot: draw the square $[-\pi, \pi)^2$ and the shortest path between $q_1$ and $q_2$ crossing the edge (the
   edges are glued, MR Table 2.2).

✅ *Checkpoint 4:* topology tests pass; you can explain why $T^2 \ne S^2$.

## Checkpoint 5 — A singular representation (5 min)

1. Implement `latlong_to_xyz`, `xyz_to_latlong` and `longitude_rate`.
2. Plot `longitude_rate(lat, 0.2)` for latitudes from $0^\circ$ to $89.9^\circ$ (log scale on the $y$ axis). Then
   compute the Cartesian speed with finite differences and plot it on the same figure. One curve explodes, the other
   is flat: this is the difference between a singularity of the *representation* and the space itself.

✅ *Checkpoint 5:* representation tests pass; figure saved as `lab0_latlong.png`.

## Checkpoint 6 — First contact with modern_robotics (5 min)

1. Implement `fk_2r_trig` (trigonometry) and `fk_2r_mr`, which calls `mr.FKinSpace(M, Slist, theta)` with the screw
   axes given in the docstring. You do not need to understand screw axes yet (Weeks 3–6); treat the call as a black box.
   This is the planar 2R arm of Lecture B ($L_1 = 0.6$ m, $L_2 = 0.4$ m): at $\theta = (30^\circ, 60^\circ)$ both functions
   must give the tip $(0.5196, 0.7000)$ m (WE 18).
2. Run the full test file: `LAB_IMPL=starter pytest -q`. All 52 tests should pass.
3. Open `modern_robotics/core.py` (find it with `python -c "import modern_robotics; print(modern_robotics.__file__)"`)
   and read the docstring of `FKinSpace`. The book asks you to *read* the library, not only use it.

*Optional extra (2 min, not marked, no new tests):* in a Python shell, call `fk_2r_trig` at the two inverse-kinematics
answers of Lecture B, $(30^\circ, 60^\circ)$ and $(76.83^\circ, -60^\circ)$ (convert to radians with `np.radians`), and
confirm that both give the same tip. One tip, two configurations: the workspace is not the C-space. (C++ track: the
same check with `lab::fk_2r_trig` in a small `main`, or skip it.)

✅ *Checkpoint 6:* all tests pass; commit (Route A: and push).

## Checkpoint 7 — MR Studio tour (7 min, in pairs)

Start MR Studio and press **F1** to show the theory panel.

1. Tab ① *Forward Kinematics*: choose **Planar 2R arm**. Move the joint sliders through $\pm 180^\circ$ and watch the angles
   wrap. Compare the tip position with your `fk_2r_trig` at one configuration.
2. Switch to **UR5 (6R industrial)** and **Franka Panda (7R redundant)**. How many dof does each have? Which one could
   hold a fixed handle and still move its elbow (Workshop Problem 3)?
3. Tab ⑥ *Closed chains*: four-bar linkage, **▶ Rotate crank**. Then **Five-bar parallel robot (2 DOF)**. Check both
   dof against Grübler.
4. Tab ⑦ *Stewart–Gough*: **▶ Dance**. Grübler says 6 for UPS legs; how many leg lengths are being commanded?
5. Tab ⑨ *3×RPR & Delta*: select the Delta robot and compare what you see with MR Example 2.7 (15 by Grübler, 3 at the
   platform).
6. Tab ⑩ *Validation*: press **▶ Run all checks** and note how many checks pass. Why does a simulator need this?

✅ *Checkpoint 7:* six short answers in your notebook (one line each).

# Deliverables (by the start of Week 2 Lecture A: Route A `gh student submit`; Route B `python submit.py 0 --id <ID>` and upload the ZIP)

1. `starter.py` (renamed `lab0.py` is fine) passing all tests in `test_lab.py`.
2. `lab0_latlong.png` from Checkpoint 5.
3. `lab0_notes.md`: answers to the questions in Checkpoints 2, 3, 4 and 7, and the reflection below.

# Reflection (≤ 200 words, in `lab0_notes.md`)

* Which mechanism in the table surprised you, and why?
* Describe one situation in a hospital or a field (EOD, rescue, logistics) where getting the dof count wrong would
  cause a safety problem.
* **AI-use note** (required): did you use an AI tool? If yes: which tool, what you asked, what you kept, and what you
  changed. Lab code may be written with AI assistance if disclosed (course AI policy, Course syllabus §8 AI-use policy); you must be able to
  explain every line in an oral check.

# Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: modern_robotics` | Activate the venv (`source .venv/bin/activate`) or `pip install modern_robotics`. |
| `ModuleNotFoundError: starter` in pytest | Run pytest from the `lab/` folder, or set `LAB_IMPL` to your module name without `.py`. |
| Every test fails with `NotImplementedError` | Expected for the untouched starter: implement the functions one by one. |
| `test_dof_table` fails on one entry | Check that RP, E and F are in `JOINT_FREEDOMS`, and that you expand the `{type: count}` dict correctly. |
| `wrap_to_pi(np.pi)` returns `pi` | The interval is half-open, $[-\pi, \pi)$: use `(x + pi) % (2 pi) - pi`. |
| `git push` asks for a password | GitHub needs a personal access token or SSH key; see the course Git sheet. Never paste tokens into code. |
| MR Studio window is blank on a laptop | Use a lab PC, or start it with `python run_app.py` from the MR-Studio folder and check the terminal for errors. |
| Docker: `permission denied` on Linux | Add yourself to the `docker` group, log out and in again. |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data file, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`) and with the numbers of the Python reference, so both tracks are marked against identical values. Checkpoints 1 and 7 (Git, MR Studio) are the same for both tracks; Git tracks your `.cpp` files exactly like `.py` files.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`. The data file is the same `data/mechanisms.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). Install them during the Setup step of this lab; Python is still needed for `run_lab.py` and for the plot of Checkpoint 5.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 0 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*joint*,*grubler*"      # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 17 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `env_report` (Checkpoint 1); the table `JOINT_FREEDOMS`, `joint_freedoms`, `joint_constraints`, `grubler` (three overloads, see below) (Checkpoint 2); `dof_table` (Checkpoint 3); `wrap_to_pi` (scalar and vector), `torus_distance`, `cspace_topology` (Checkpoint 4); `latlong_to_xyz`, `xyz_to_latlong`, `longitude_rate` (Checkpoint 5); `fk_2r_trig`, `fk_2r_mr` (Checkpoint 6). `PLANAR_JOINTS`, `cpp_standard_value` and `compiler_name` are given.

**Checkpoints.** 1: `./build/lab_tests -tc="*env*"`; 2: `-tc="*joint*,*grubler*"`; 3: `-tc="*dof_table*"`; 4: `-tc="*wrap*,*torus*,*cspace*"`; 5: `-tc="*latlong*,*longitude*"`; 6: `-tc="*fk_2r*"`; at the end all 17 test cases pass (`./build/lab_tests`). The Python track counts 52 tests because pytest counts every parameter of a `parametrize` list separately (each joint type, each MR example, each chain length); the C++ track checks the same values in one test case per Python test function. In Checkpoint 6, step 3, read `mr::fkin_space` in `cpp_support/src/mrcpp.cpp` (declared in `cpp_support/include/sbeg/mrcpp.hpp`) instead of `modern_robotics/core.py`; the C++ code follows the Python library line by line.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `env_report()` returns the Python, NumPy and `modern_robotics` versions | returns `{"cpp_standard", "compiler", "eigen"}` (e.g. `"201703"`, `"clang 15.0.0"`, `"3.4.0"`) as a `std::map<std::string, std::string>`; the test requires C++17 or newer |
| `JOINT_FREEDOMS = {"R": 1, ...}` (dict) | `const std::map<std::string, int> JOINT_FREEDOMS = {{"R", 1}, ...};` in `starter.cpp` |
| `grubler(m, N, joints)` accepts a list of types, a list of $f_i$ or a dict | three overloads: `std::vector<std::string>`, `std::vector<int>`, `std::map<std::string, int>`; write the argument with its type, e.g. `std::vector<std::string> js = {"R", "R", "RP"}; lab::grubler(3, 3, js);` |
| `json.loads(...)` | `sbeg::load_json(path)`; `e.at("joints").get<std::map<std::string, int>>()` (hint in `starter.cpp`; use `.at("key")`, which throws for a missing key, not `["key"]`) |
| `wrap_to_pi` works on scalars and arrays | two overloads: `double` and `Eigen::VectorXd` |
| `(x + pi) % (2 pi) - pi` | `std::fmod` keeps the sign of `x` (Python's `%` keeps the sign of `2 pi`): add $2\pi$ when the remainder is negative |
| `xyz_to_latlong` returns a tuple | returns `std::pair<double, double>`: `auto [lat, lon] = lab::xyz_to_latlong(p);` |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `mr.FKinSpace(M, Slist, theta)` | `mr::fkin_space(M, Slist, theta)`; `Slist` is a 6×2 `Eigen::MatrixXd` with the screw axes as columns |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

The random test inputs of `test_latlong_roundtrip` and `test_fk_2r_against_modern_robotics` (NumPy generators) are stored in `cpp_cases.json`, so both tracks use the same 200 and 50 cases. The figure of Checkpoint 5 (`lab0_latlong.png`) is not auto-checked: make it with Python/Matplotlib as in the Python track (plotting stays in Python for the C++ track), using the same formula as your `longitude_rate`. Deliverable 1 for the C++ track is `cpp/starter.cpp` passing all 17 test cases. A matrix-size mistake (e.g. adding a 2-vector to a 3-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `call to 'grubler' is ambiguous` | a brace list such as `{"R", "R"}` fits all three overloads | write the type: `std::vector<std::string>{"R", "R"}` |
| `wrap_to_pi` wrong only for negative angles | `std::fmod` returned a negative remainder | add $2\pi$ when it is negative |
| `test_env_report` fails with `cpp_standard` 199711 | `__cplusplus` read directly on MSVC | use the given `cpp_standard_value()` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
