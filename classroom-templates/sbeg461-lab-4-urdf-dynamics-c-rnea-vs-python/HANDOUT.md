---
title: "Lab 4 — URDF dynamics, C++ RNEA vs Python timing, and the constrained 2R arm"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 4"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before the Week 5 lab) |
| CLOs | 461-2 (implement and verify recursive dynamics; simulate with appropriate integration); 461-3 (task-space and constrained dynamics) |
| Files | `lab/starter.py` (you edit), `lab/test_lab.py` (auto-checks), `lab/arm6.urdf` and `lab/states_arm6.csv` (data), `lab/make_data.py` (how the data were made), `lab/cpp/` (C++: `rnea_starter.cpp` you edit; `se3.hpp`, `rnea.hpp`, `model_io.cpp`, `main.cpp`, `CMakeLists.txt` provided) |
| Reference | `modern_robotics` (`InverseDynamics`, `MassMatrix`, `FKinSpace`), Pinocchio (`buildModelFromUrdf`, `rnea`) |
| Reading | MR §8.3.2 (printed pp. 292–293), §8.6–8.8 (pp. 298–305); Featherstone §8.3 *Constraint Stabilization* (pp. 145–147); Gotor §8.6 *Inverse Dynamics Simulations* (pp. 247–256) |
| Assessment | Auto-checked (`test_lab.py`) + timing table + short reflection (not a formal report) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation only) |

The functions you write today go into your `sbeg461_dynamics` library: Lab 9 reuses the RNEA (a copy is provided); the constrained dynamics return in the Week 9 hybrid-control lecture and workshop (MR Eqs. 11.55–11.60).

**Application context.**
*Biomedical:* a rehabilitation or assistive arm is described by a URDF; its controller needs inverse dynamics at 1 kHz (MR p. 306), which is why the RNEA is written in C++ in real systems. A skin-marking or ultrasound robot whose tool is held on a line or surface is a constrained robot (MR §8.7).
*Defence (protective/logistic):* the 6-DOF arm `arm6.urdf` stands in for the manipulator of an EOD/IED-disposal robot; the same C++ dynamics code runs on its embedded computer. A gripper sliding a drawer or hatch along a rail is a constrained 2R/3R problem. `arm6.urdf` uses UR5-like geometry with **illustrative** inertias; it is not a model of any product.

# Learning objectives

By the end of the lab you can:

1. Translate the URDF `<joint>` and `<inertial>` elements into the MR quantities $M_{i-1,i}$, $\mathcal G_i$ and $S_i$ (MR §8.8) and validate them against Pinocchio.
2. Implement the RNEA (MR Eqs. 8.50–8.54) in Python and in C++ with Eigen, verify both against `modern_robotics`, and measure and explain their speed difference.
3. Compute task-space dynamics $\Lambda$, $\eta$ (MR Eqs. 8.89–8.91) and check them against joint-space dynamics.
4. Solve constrained forward and inverse dynamics with Lagrange multipliers (MR Eqs. 8.93–8.100), reproduce MR Example 8.5, and simulate the 2R arm in a channel with and without Baumgarte stabilisation.

# Pre-lab check (complete before the session; 5 questions)

1. In a URDF `<joint>`, which element gives the pose of the child-link frame in the parent-link frame, and in which frame is `<axis>` expressed?
2. In MR's RNEA, what is $\dot{\mathcal V}_0$ when gravity is $g = (0, 0, -9.81)$? Why is gravity put there?
3. For MR Example 8.5 ($\theta = (-60^\circ, 120^\circ)$, $\dot\theta = (1, 0)$, channel $x = 1$) write $A(\theta)$ and $\dot A(\theta,\dot\theta)$.
4. Why must the constraint forces $A^{\mathsf T}\lambda$ do no work on the robot? What does that imply for the kinetic energy when $\tau = 0$ and $g = 0$?
5. A rotor with $I_{\text{rotor}} = 1.25\times10^{-3}$ kg m² drives a joint through $G = 100$ (MR p. 312–313). What is its apparent inertia at the joint?

*Answers (for self-check):* 1. `<origin xyz rpy>` (with $\theta = 0$); `<axis>` is in the child-link (joint) frame. 2. $\dot{\mathcal V}_0 = (0, -g) = (0,0,0,\ 0,0,9.81)$: gravity is treated as an upward acceleration of the base (MR p. 292), so every link "feels" its weight without a separate gravity term. 3. $A = [-s_1 - s_{12},\ -s_{12}] = [0,\ -0.866]$, $\dot A = [-\dot\theta_1c_1 - (\dot\theta_1+\dot\theta_2)c_{12},\ -(\dot\theta_1+\dot\theta_2)c_{12}] = [-1,\ -0.5]$. 4. Workless means $\tau_{\text{con}}^{\mathsf T}\dot\theta = \lambda^{\mathsf T}A\dot\theta = 0$ (MR p. 299, frictionless constraint); with $\tau = 0$ and $g = 0$ nothing does work, so the kinetic energy is constant. 5. $100^2(1.25\times10^{-3}) = 12.5$ kg m².

# Setup

```bash
cd "<your Lab code folder>"   # the Lab 4 code folder (or your Classroom 50 repo)
python -c "import numpy, modern_robotics, pinocchio; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py   # 10 failed + 5 errors (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`.

**C++ toolchain** (needed for Step 3; already in the course Docker image):

| OS | Install |
|---|---|
| Ubuntu / WSL | `sudo apt install g++ cmake libeigen3-dev` |
| macOS | `xcode-select --install`, then `brew install cmake eigen` |
| Windows (native) | use the course Docker image or WSL |

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build                 # -> cpp/build/rnea_bench
```

Without CMake: `c++ -O3 -DNDEBUG -std=c++17 -I/usr/include/eigen3 cpp/main.cpp cpp/model_io.cpp cpp/rnea_starter.cpp -o cpp/build/rnea_bench`.

Rules: inside `starter.py` do **not** call `modern_robotics` or Pinocchio (they are for checking only); you may use the provided helpers at the top of the file (`skew`, `rpy_to_R`, `Adjoint`, `ad`, `MatrixExp6`, `read_urdf_chain`, …). The C++ RNEA must not call Pinocchio or any dynamics library; Eigen only.

# Procedure

Run the tests after each step. A checkpoint is complete when the listed tests pass; ask a TA to sign it off.

## Step 1 — URDF to MR parameters (15 min)

Implement `urdf_to_mr(path)`. `read_urdf_chain` already parses the XML and returns, for each joint from base to tip, `T_origin`, `axis`, `mass`, `T_com` and `I` (see its docstring). Walk the chain, accumulating the link-frame pose $T_{0,\text{link}}$:

- for a revolute joint: $\omega_i = R_{0,\text{link}}\,\text{axis}$, $q_i = p_{0,\text{link}}$, $S_i = (\omega_i,\ -\omega_i\times q_i)$; CoM frame $T_{0,i} = T_{0,\text{link}}T_{\text{com}}$; $M_{i-1,i} = T_{0,i-1}^{-1}T_{0,i}$; $\mathcal G_i = \mathrm{diag}(I_i,\ m_iI_3)$ (MR Eq. 8.32);
- for the final fixed joint(s): only update $T_{0,\text{link}}$; the last entry of `Mlist` is $M_{n,n+1} = T_{0,n}^{-1}T_{0,\text{tool}}$.

Note that the `forearm_link`, `wrist_1_link` and `wrist_3_link` inertial frames are rotated (`rpy` not zero) and the forearm tensor has products of inertia; $\mathcal G_i$ must be expressed in the *inertial* frame, which is exactly the frame the URDF gives $I$ in.

**Checkpoint 1:** `test_urdf_shapes`, `test_urdf_forward_kinematics_matches_pinocchio`, `test_urdf_dynamics_match_pinocchio`.

## Step 2 — Python RNEA and a timer (10 min)

Paste your Lab 2 RNEA into `rnea(...)` (same interface as `mr.InverseDynamics`) or rewrite it from MR Eqs. (8.50)–(8.54). Implement `time_per_call(fn, arg_tuples, repeats=3)` with `time.perf_counter()`: the median, over `repeats` runs, of the mean time per call.

**Checkpoint 2:** `test_rnea_matches_modern_robotics`, `test_rnea_2r_matches_closed_form` (needs Step 5's `twor_dynamics`; come back to it), `test_time_per_call_is_positive_and_scales`.

## Step 3 — C++ RNEA with Eigen (start in class, finish at home)

Translate your `rnea` into `cpp/rnea_starter.cpp` (`inverse_dynamics(model, theta, dtheta, ddtheta, g, Ftip)`); `se3.hpp` gives `skew`, `trans_inv`, `adjoint`, `ad`, `exp6`. Build, then run:

```bash
python -c "import starter as s; s.export_for_cpp('model.txt','states.txt', *s.urdf_to_mr('arm6.urdf'), s.load_states('states_arm6.csv'))"
cpp/build/rnea_bench model.txt states.txt tau_cpp.txt 2000
RNEA_CPP=cpp/build/rnea_bench LAB_IMPL=starter python -m pytest -q test_lab.py -k cpp
```

Then fill in the timing table with `starter.benchmark("arm6.urdf", "states_arm6.csv", "cpp/build/rnea_bench")` (microseconds per call, 200 states):

| Implementation | µs per call (yours) | indicative (instructor laptop) |
|---|---|---|
| Your Python `rnea` | | ≈ 260 |
| `modern_robotics.InverseDynamics` | | ≈ 660 |
| Pinocchio (C++ called from Python) | | ≈ 0.6 |
| Your C++ `inverse_dynamics` (Eigen, `-O3`) | | ≈ 0.8 |

**Checkpoint 3:** `test_cpp_rnea_agrees_with_python` passes (it is skipped when no binary is found) and the table is complete. Answer: is your C++ RNEA fast enough for a 1 kHz control loop on the EOD arm? For a 10 kHz loop? How many RNEA calls does a CRBA-free mass matrix (Lab 3: $n$ calls) need per control cycle, and does Python still fit?

## Step 4 — Task-space dynamics (5 min)

Implement `twor_dynamics` (MR Eqs. 8.9–8.10, point masses, gravity $g$ along $-\hat y$), `twor_jacobian` (tip $J$ and $\dot J$, MR Example 8.5) and `task_space_dynamics(M, h, J, Jdot, dtheta)` → $(\Lambda, \eta)$ (MR Eqs. 8.90–8.91).

**Checkpoint 4:** `test_twor_dynamics_matches_modern_robotics`, `test_task_space_dynamics_consistent_with_joint_space`, `test_task_space_mass_matrix_value` ($\Lambda = \mathrm{diag}(1, 2)$ at $\theta = (0, 90^\circ)$).

## Step 5 — Constrained 2R: MR Example 8.5 (10 min)

Implement `channel_constraint` ($A$, $\dot A$ for $x = $ const), `constrained_forward_dynamics` (MR Eqs. 8.96, 8.98), `constrained_inverse_dynamics` (MR Eq. 8.93) and `constraint_projector` (MR Eq. 8.100).

**Checkpoint 5:** `test_channel_constraint_is_gradient_of_x`, `test_mr_example_8_5_forward`, `test_projector_and_inverse_dynamics`. At $\tau = 0$ you should get $\lambda = -0.1667$ and $\ddot\theta = (0.2887,\ -1.1547)$; $P = \begin{bmatrix}1&0\\0.25&0\end{bmatrix}$.

## Step 6 — Simulation and constraint drift (finish at home)

Implement `simulate_channel_2r` with fourth-order Runge–Kutta (MR p. 298). With $\alpha = \beta = 0$ use MR Eq. (8.98) as is. With $\alpha, \beta > 0$ add Baumgarte stabilisation (Featherstone §8.3): require $\ddot e + 2\alpha\dot e + \beta^2e = 0$ for the constraint error $e = c_1 + c_{12} - 1$, i.e. add $2\alpha\dot e + \beta^2e$ inside the bracket of (8.98): $\lambda = (AM^{-1}A^{\mathsf T})^{-1}\big(AM^{-1}(\tau - h) + \dot A\dot\theta + 2\alpha\dot e + \beta^2e\big)$.

1. From the Example 8.5 state with $\tau = 0$, simulate 2 s ($\Delta t = 1$ ms). Plot the kinetic energy (it should stay at its initial value, 1.0 J), $e(t)$ and $\lambda(t)$.
2. Start with a small violation, $\theta(0) = (-60^\circ,\ 120^\circ + 0.57^\circ)$, and compare $\alpha = \beta = 0$ with $\alpha = \beta = 20$ s$^{-1}$. Plot $e(t)$ for both.
3. Try $\alpha = \beta = 2000$ s$^{-1}$ with $\Delta t = 1$ ms. What happens and why? (Relate to Featherstone's rule $\alpha = \beta = 1/T_{\text{stab}}$.)

**Checkpoint 6:** `test_simulation_conserves_energy_and_constraint`, `test_baumgarte_removes_drift`. Final run: `LAB_IMPL=starter python -m pytest -q test_lab.py` → all pass.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 4 into your cumulative library as `rnea.py` (Parts A–B: `urdf_to_mr`, `rnea`, `time_per_call`) and `constrained.py` (Parts C–D). The functions you write (11):

`urdf_to_mr(path)`, `rnea(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)`, `time_per_call(fn, arg_tuples, repeats=3)`, `task_space_dynamics(M, h, J, Jdot, dthetalist)`, `twor_dynamics(theta, dtheta, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=0.0)`, `twor_jacobian(theta, dtheta, L1=1.0, L2=1.0)`, `channel_constraint(theta, dtheta, L1=1.0, L2=1.0)`, `constrained_forward_dynamics(M, h, A, Adot, dtheta, tau)`, `constrained_inverse_dynamics(M, h, A, ddtheta, lam)`, `constraint_projector(M, A)`, `simulate_channel_2r(theta0, dtheta0, tau_fun, T, dt, x_c=1.0, alpha=0.0, beta=0.0, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=0.0)`.

Names shared with other labs:

* `rnea` has the same arguments, in the same order, as Lab 2's `rnea` (only the names differ: `thetalist` for `theta`, …). Keep one implementation in `rnea.py`; both must give the same torques.
* `task_space_dynamics(M, h, J, Jdot, dthetalist)` takes the joint-space terms. Lab 6 has a different function, `model_task_space_dynamics(model, q, dq)`, that computes them from a model first.
* The provided `load_states(csv_path)` takes a file path; Lab 2's provided `load_states(name)` takes a data-set name. Both are loaders, not library functions.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables (upload to the course site)

1. `starter.py` and `cpp/rnea_starter.cpp` (all tests pass; the C++ test is run by the TA if your machine cannot build it).
2. The timing table (Step 3) and your answers to the Checkpoint-3 questions (≤ 150 words).
3. Two plots from Step 6 (energy + $\lambda$; $e(t)$ with and without stabilisation), with axis labels and units.
4. Reflection (below), ≤ 200 words.

# Reflection (≤ 200 words)

- Where does the ≈ 300× speed-up of C++ over Python come from (interpreter overhead, small-matrix allocation, fixed-size Eigen types)? Which part of the RNEA loop would you optimise first?
- The URDF of a real assistive or EOD arm rarely lists motor rotor inertias. Using MR §8.9.2, explain how missing apparent inertia would bias your inverse-dynamics torques at $G = 100$.
- **AI-use note:** state whether you used an AI assistant, for which step, what you asked, and how you verified the answer (tests, hand check, comparison with Pinocchio). Code that you cannot explain in the lab practical (Week 13) will not earn credit.

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| FK matches Pinocchio but dynamics do not | $\mathcal G_i$ built in the link frame instead of the inertial frame, or `T_com` not applied to `Mlist` |
| `test_urdf_forward_kinematics…` fails by a constant offset | the fixed `tool_fixed` joint was skipped: the last `Mlist` entry must include it |
| Wrong sign of $S_i$ | used $\omega\times q$ instead of $v = -\omega\times q$ (MR Definition 3.24, p. 102, pitch $h = 0$) |
| Python RNEA correct but 10× slower than the table | `np.linalg.inv` inside the loop instead of `TransInv`; recomputing `Adjoint(TransInv(Mi))` twice |
| CMake: `Could not find Eigen3` | install `libeigen3-dev` / `brew install eigen`, or pass `-DEigen3_DIR=…/share/eigen3/cmake` |
| C++ torques all equal to the first state | reading the states file with the wrong $n$ (check line 1 of `model.txt`) |
| C++ timing shows 0 µs | built without using the result; keep the `checksum` line in `main.cpp` |
| Constrained simulation blows up | $\alpha$, $\beta$ too large for $\Delta t$ (stiff), or `Adot` sign error (check with `test_channel_constraint_is_gradient_of_x`) |
| Energy drifts slowly with $\alpha = \beta = 0$ | expected: RK4 error plus constraint drift; that is what Step 6.2 shows |

# C++ track

You may do the Python parts of this lab (Steps 1, 2, 4, 5 and 6) in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the random inputs and reference numbers of the Python tests, and with the Pinocchio values computed in Python, so both tracks are marked against identical values. **Step 3 (`rnea_bench`, `rnea_starter.cpp`) is the same for both tracks** and is built by the same `CMakeLists.txt`.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations with the docstrings of `starter.py`, and the small structs `UrdfJoint`, `MRModel`, `ChannelSim` — do not change), `helpers.cpp` (the provided helpers of `starter.py`, given), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `bench_main.cpp` (the given program `lab_bench` for the Step 3 table and the Step 6 files), `plot_results.py` (Step 6 plots), `CMakeLists.txt` (builds `lab_tests`, `lab_bench` and the Step 3 program `rnea_bench`); Step 3 files `se3.hpp`, `rnea.hpp`, `model_io.cpp`, `main.cpp`, `rnea_starter.cpp` as described above. The data files are the same `arm6.urdf` and `states_arm6.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`). No separate Eigen installation is needed: all three programs use the Eigen copy in `cpp_support`.

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 4 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build                            # builds lab_tests, lab_bench and rnea_bench
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*urdf*"                 # only some tests (like pytest -k)
./build/lab_bench .                            # Step 3 files + timing, Step 6 CSV files
./build/rnea_bench model.txt states.txt tau_cpp.txt 2000
python cpp/plot_results.py .                   # Step 6 plots
```

With the untouched starter, the program builds and all 15 test cases fail with `TODO: <function>`, as the Python tests fail with `NotImplementedError`. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order and arguments as in Python): `urdf_to_mr` (Step 1); `rnea`, `time_per_call` (Step 2); `twor_dynamics`, `twor_jacobian`, `task_space_dynamics` (Step 4); `channel_constraint`, `constrained_forward_dynamics`, `constrained_inverse_dynamics`, `constraint_projector` (Step 5); `simulate_channel_2r` (Step 6). Given in `helpers.cpp`: `skew`, `rpy_to_R`, `rp_to_trans`, `trans_inv`, `adjoint`, `ad`, `matrix_exp6`, `read_urdf_chain` (with a small XML reader), `load_states`, `export_for_cpp`, `benchmark`. As in Python, do not call `mr::...` in `starter.cpp`: `mrcpp` is the reference.

**Checkpoints.** 1: `./build/lab_tests -tc="*urdf*"`; 2: `-tc="*rnea_matches*,*rnea_2r*,*time_per_call*"`; 3: `-tc="*cpp_rnea*"` and the timing table; 4: `-tc="*twor_dynamics*,*task_space*"`; 5: `-tc="*channel*,*8_5*,*projector*"`; 6: `-tc="*simulation*,*baumgarte*"`. All 15 test cases pass (`./build/lab_tests`): the same 15 tests as `test_lab.py`, with the same names and tolerances.

- *Step 3 in the C++ track.* Your `rnea` is already C++, so `./build/lab_bench .` times it and `mr::inverse_dynamics` with your `time_per_call` (first 200 states) and writes `model.txt` and `states.txt` for `rnea_bench` (it replaces the `export_for_cpp` command of Step 3). Then translate your `rnea` into `rnea_starter.cpp` (fixed-size types, `se3.hpp` helpers) and run `rnea_bench` as in Step 3. For the timing table, enter `lab_bench`'s `cpp_own_rnea` in an extra row "Your C++ `rnea` (`lab::rnea`, dynamic sizes)", the `rnea_bench` time in the row "Your C++ `inverse_dynamics`", and take the Python, `modern_robotics` and Pinocchio rows from the indicative column (or time them in Python if you have the course environment). For timing, configure a separate optimised build: `cmake -S cpp -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release`.
- *Step 3 test.* `test_cpp_rnea_agrees_with_python` does not run the `rnea_bench` program (the Python test does). It checks the same chain with your C++ functions: `urdf_to_mr` → `export_for_cpp` (the files `rnea_bench` reads) → read back → your `rnea` on the first 50 states of `states_arm6.csv`, compared with `mrcpp` and with the Python reference torques; it also prints your time per call.
- *Step 6.* `lab_bench` runs the three simulations of Step 6 with your `simulate_channel_2r` and writes `step6_1_energy.csv`, `step6_2_plain.csv`, `step6_2_baumgarte.csv` and `step6_3_stiff.csv` (columns `t, theta1, theta2, dtheta1, dtheta2, lam, e, KE`); `python cpp/plot_results.py .` makes the two plots of the deliverables (`step6_energy_lambda.png`, `step6_drift.png`).

**Deliverables.** As above, with `cpp/starter.cpp` instead of `starter.py` in item 1 (all 15 C++ test cases pass), `cpp/rnea_starter.cpp` as for the Python track, and the C++ times of `lab_bench` and `rnea_bench` in the table.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `urdf_to_mr` returns the tuple `(Mlist, Glist, Slist)` | `lab::MRModel` struct with the same field names: `const auto [Mlist, Glist, Slist] = lab::urdf_to_mr(path);` |
| list of 4×4 / 6×6 arrays, `Slist` 6×n array | `std::vector<Eigen::Matrix4d>` (`TransformList`), `std::vector<Matrix6d>` (`InertiaList`), `Eigen::MatrixXd` (axes = columns) |
| `read_urdf_chain` returns a list of dicts | `std::vector<lab::UrdfJoint>` with the same keys as fields (`j.T_origin`, `j.axis`, `j.mass`, `j.T_com`, `j.I`, …) |
| `time_per_call(fn, arg_tuples, repeats=3)` with `time.perf_counter()` | `time_per_call(fn, n_calls, repeats = 3)`: `fn(k)` makes the k-th call (pass a lambda); `std::chrono::steady_clock`; the test times 2000 calls instead of 20 (one C++ call can be shorter than one clock tick) |
| `twor_dynamics`, `twor_jacobian`, `channel_constraint`, `constrained_forward_dynamics` return tuples | `std::pair`: `const auto [M, h] = lab::twor_dynamics(th, dth);` |
| `A`, `Adot` of shape (1, 2) | 1×2 `Eigen::MatrixXd` |
| `tau_fun(t, theta, dtheta)` | `lab::TorqueFunction` (a lambda returning `Eigen::Vector2d`) |
| `simulate_channel_2r` returns a dict | `lab::ChannelSim` with the same keys as fields (`t`, `theta`, `dtheta`, `lam`, `e`) |
| `benchmark(urdf, csv, cpp_exe)` (Python, `modern_robotics`, Pinocchio, C++ program) | `lab_bench`: your `lab::rnea` and `mr::inverse_dynamics`; Pinocchio is not available in C++ |
| Pinocchio checks of Step 1 | compare with the Pinocchio values stored in `cpp_cases.json` |
| Step 6 plots with matplotlib | `lab_bench` writes CSV files, `plot_results.py` plots them |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

The `CMakeLists.txt` builds `rnea_bench` against the Eigen copy in `cpp_support`, so the "`Could not find Eigen3`" row of the Troubleshooting table no longer occurs with CMake (the commands of Step 3 are unchanged). This holds for both tracks: the Step 3 CMake build needs `cpp_support` next to or above your Lab code folder even if you do the rest of the lab in Python (it also builds `lab_tests` and `lab_bench`, which you may ignore); the "Without CMake" command of the Setup still uses an installed Eigen. A matrix-size mistake in your code does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| Build stops in `rnea_starter.cpp` | a compile error in Step 3 also stops `lab_tests` | fix it, or build only the tests: `cmake --build build --target lab_tests` |
| `lab_tests` stops with "terminating due to uncaught exception ... Too few coefficients" | a comma initialiser (`S << w, v;`, `M << ...;`) with fewer numbers than the matrix has entries | count the entries; this error ends the whole test run instead of failing one test |
| `test_urdf_shapes` fails on `Slist.cols()` | `Slist` built with one row per joint | `Slist` is 6×n: `Slist.col(i) = S_i` |
| `time_per_call` test: `t_slow > 5 t_fast` fails | timing outside the loop, or dividing by the number of runs instead of the number of calls | time each run separately and return the median of (run time / `n_calls`) |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
