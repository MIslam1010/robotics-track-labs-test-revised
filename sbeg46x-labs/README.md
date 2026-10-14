# SBEG 46X Medical Robotics — labs on your own computer

Everything you need to do the SBEG 46X Medical Robotics labs on Windows, macOS or Linux: one folder per lab with its handout
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
`conda env create -f environment.yml`, then `conda activate <name shown in the file>`.

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
| `labs/hw-rotation-1-project-safety` | Hardware Rotation 1 — Project Safety Induction and a Supervised Teleoperation Trial | 8 |
| `labs/lab-01-software-remote-centre-of-motion` | Lab 1 — Software remote centre of motion: resolved-rate control of the course arm with a laparoscope | 1 |
| `labs/lab-10-cbf-qp-safety-filter-versus` | Lab 10 — CBF-QP safety filter versus saturation on an exoskeleton joint and on the Hapkit | 11 |
| `labs/lab-11-image-based-visual-servoing-of-an` | Lab 11 — Image-based visual servoing of an endoscope-holding arm with a simulated pinhole camera (Formal report 3) | 12 |
| `labs/lab-02-teleoperation-with-a-hapkit-leader` | Lab 2 — Teleoperation with a Hapkit leader: scaling, clutch, tremor filter and two-port models | 2 |
| `labs/lab-03-delayed-bilateral-teleoperation` | Lab 3 — Delayed bilateral teleoperation: instability, wave variables and time-domain passivity (Formal report 1) | 3 |
| `labs/lab-04-haptic-rendering-on-the-hapkit` | Lab 4 — Haptic rendering on the Hapkit: virtual wall, Z-width, virtual fixtures and a needle-insertion profile | 4 |
| `labs/lab-05-navigation-pipeline-pivot` | Lab 5 — Navigation pipeline: pivot calibration, paired-point registration and TRE (prediction vs Monte Carlo vs measurement) | 5 |
| `labs/lab-06-two-segment-tendon-driven-catheter` | Lab 6 — Two-segment tendon-driven catheter: constant-curvature kinematics, Jacobian and resolved-rate tip control | 6 |
| `labs/lab-07-cosserat-rod-catheter-shooting-and` | Lab 7 — Cosserat-rod catheter: shooting and BVP solvers, CC vs Cosserat under tip loads, tip-force estimation | 7 |
| `labs/lab-08-pd-vs-lqr-vs-mpc-on-an-exoskeleton` | Lab 8 — PD vs LQR vs MPC on an exoskeleton joint with torque and range limits (Formal report 2) | 9 |
| `labs/lab-09-imu-orientation-fusion-for-an` | Lab 9 — IMU orientation fusion for an exoskeleton thigh: complementary filter vs extended Kalman filter | 10 |

Also here: `library/` — your cumulative `sbeg46x_medrob` package (copy each finished function into it; later labs and the
next course use it), and `project/` — the course-project starter for your team.

Do not edit the test files or the data: the course team checks your code with the original copies.
