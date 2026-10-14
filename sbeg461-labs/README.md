# SBEG 461 Robot Kinematics and Dynamics — labs on your own computer

Everything you need to do the SBEG 461 Robot Kinematics and Dynamics labs on Windows, macOS or Linux: one folder per lab with its handout
(`HANDOUT.md`), your starter code (`starter*.py`, or `cpp/starter.cpp` for the C++ track — the only files you
edit), data and the auto-checks.

## 1. Set up (once, ≈ 10 min)

You need **Python 3.11** (python.org) and, if you like, VS Code. Download this repository (green **Code** button →
*Download ZIP*, or `git clone`), open a terminal in its folder and run:

| System | Command |
|---|---|
| macOS / Linux | `bash setup.sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy Bypass -File setup.ps1` |

Then activate the environment in every new terminal (`source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`)
and check it: `python check_env.py`.

**Windows, or if pip fails:** use conda instead — install Miniforge (conda-forge.org) and run
`conda env create -f environment.yml`, then `conda activate <name shown in the file>` (needed on Windows: Pinocchio, used in some labs, has no pip version for Windows).

**If nothing installs** on your computer: use the container — install Docker Desktop, open the folder in VS Code
and choose *Reopen in Container* (or see the commands in `Dockerfile`) — or use the lab PCs.

**ROS 2 parts** (labs whose handout has a ROS 2 section): use the ROS 2 container — VS Code *Reopen in Container* → `.devcontainer/ros2`, or build `docker/ros2/Dockerfile` — or the lab PCs. These parts are not auto-checked.


**MR Studio** (the course simulator used in demos and some lab steps): install it from the course site —
macOS (Apple Silicon): open `MR-Studio-1.0.0-macOS-arm64.dmg` and drag *MR Studio* to *Applications* (first start:
right-click → *Open*, because the app is not notarised); Windows 10/11: run `MR-Studio-1.0.0-Windows-x64-setup.exe` (Next, Next, Finish — no administrator rights needed;
if SmartScreen warns, choose *More info → Run anyway*), or unzip `MR-Studio-1.0.0-Windows-x64.zip` and run `MRStudio.exe`. The labs' auto-checks do not need MR Studio.

## 2. Do a lab

```bash
python run_lab.py            # list the labs
python run_lab.py 3          # run the checks of Lab 3 and show your score
```

Package versions are pinned in `requirements.txt` to the versions the labs were tested with; do not upgrade them
during the term.

Read the lab's `HANDOUT.md`, complete the `# TODO`s in its `starter*.py`, and run the checks again until they pass.
The handouts call the files `lab/…`; here they are inside the lab's folder.

## C++ track (optional, every lab)

Every lab can also be done in **C++17** (handout section *C++ track*): the same functions, data and checkpoints, marked
against the same reference numbers. Write your code in the lab's `cpp/starter.cpp` and run

```bash
python run_lab.py 3 --cpp            # configure + build cpp/ with CMake, run the C++ checks, show your score
python run_lab.py 3 --cpp -k name    # only the test cases whose name contains 'name'
```

You need a C++17 compiler (once):

| System | Compiler |
|---|---|
| Windows 10/11 | Visual Studio **Build Tools** (2022 or 2026) (free) → workload *Desktop development with C++* — `run_lab.py --cpp` finds it by itself |
| macOS | `xcode-select --install` (Apple clang) |
| Ubuntu / WSL | `sudo apt install build-essential` |

CMake and Ninja are installed by the setup script (`requirements.txt`); Eigen, doctest and the C++ port of the
textbook library (`mr::`) are in `cpp_support/` — nothing else to download. `python check_env.py` shows whether
the C++ tools are found. Details and troubleshooting: `cpp_support/README.md`. Submit with
`python submit.py 3 --id <ID> --cpp`.

## 3. Submit

Fill in the lab's `AI_USE_NOTE.md` (AI tools are allowed in lab code **with disclosure**; you must be able to
explain every line — syllabus §8), then

```bash
python submit.py 3 --id <your student ID>
```

and upload the ZIP from `submissions/` on the course site before the deadline.

## Labs

| Folder | Lab | Week |
|---|---|---|
| `enrichment/dynamic-parameter` | Enrichment: Dynamic Parameter Identification from Hardware Logs | 11–13 |
| `enrichment/safety-critical-control-with` | Enrichment: Safety-Critical Control with Control Barrier Functions | 11–13 |
| `labs/hw-rotation-1-joint-control-on-the` | Hardware rotation 1 — Joint control on the course arm: PID step response, gravity compensation, logging; safety induction | 8 |
| `labs/lab-01-lagrangian-dynamics-in-sympy-vs` | Lab 1 — Lagrangian dynamics in SymPy vs. modern_robotics | 1 |
| `labs/lab-10-grid-planner-for-a-2r-arm-among` | Lab 10 — Grid planner for a 2R arm among obstacles | 11 |
| `labs/lab-11-rrt-for-the-course-arm-path` | Lab 11 — RRT for the course arm; path smoothing and time scaling (Formal report 3) | 12 |
| `labs/lab-02-implement-the-rnea-verify-against` | Lab 2 — Implement the RNEA; verify against modern_robotics and Pinocchio | 2 |
| `labs/lab-03-forward-dynamics-simulator-and` | Lab 3 — Forward-dynamics simulator and integrator comparison (Formal report 1) | 3 |
| `labs/lab-04-urdf-dynamics-c-rnea-vs-python` | Lab 4 — URDF dynamics, C++ RNEA vs Python timing, and the constrained 2R arm | 4 |
| `labs/lab-05-joint-pid-in-simulation-a-ros-2` | Lab 5 — Joint PID in simulation; a ROS 2 controller node | 5 |
| `labs/lab-06-computed-torque-vs-pid-with-model` | Lab 6 — Computed torque vs PID with model error on the course arm | 6 |
| `labs/lab-07-lyapunov-and-lqr-energy-based` | Lab 7 — Lyapunov and LQR: energy-based stability check of PD + gravity compensation; LQR for a linearised 2R arm and the course arm | 7 |
| `labs/lab-08-impedance-control-on-a-hapkit-type` | Lab 8 — Impedance control on a Hapkit-type haptic device and a simulated rehabilitation robot (Formal report 2) | 9 |
| `labs/lab-09-trajectory-library-torque` | Lab 9 — Trajectory library, torque feasibility and time-optimal time scaling (TOPP) on the course arm | 10 |

Also here: `library/` — your cumulative `sbeg461_dynamics` package (copy each finished function into it; later labs and the
next course use it), and `project/` — the course-project starter for your team.

Do not edit the test files or the data: the course team checks your code with the original copies.
