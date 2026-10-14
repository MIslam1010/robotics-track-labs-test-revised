---
title: "Lab 2 — Angular velocities and twists: space/body twists, the adjoint, and ROS 2 tf2 visualisation"
subtitle: "SBEG 361 Introduction to Robotics — Week 3"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before the Week 4 lab) |
| CLOs | 361-2 (describe the motion of rotating and moving frames with SO(3)/SE(3)); 361-3 (angular velocities, twists, screw axes, adjoint); 361-7 (implement and validate against a reference library and a middleware tool) |
| Part A (auto-tested) | Pure-Python angular-velocity and twist tools: `starter.py` (you edit), `test_lab.py` (14 auto-checks) |
| Part B (guided) | ROS 2 Jazzy tf2 + RViz: `ros2/nav_tf_broadcaster.py` publishes the navigation tree with the moving screwdriver; you inspect it with `tf2_echo`, `view_frames` and RViz |
| Data | `nav_tree.json`, `tool_track.csv`, `eod_tree.json`; `make_data.py` shows how they were made (read it **after** the lab) |
| Reference | `modern_robotics` (`VecToso3`, `VecTose3`, `Adjoint`, `TransInv`, `MatrixLog6`); ROS 2 `tf2_ros` |
| Reading | MR §3.2.2 (printed pp. 74–77): Eqs. 3.29–3.38, Prop. 3.8, Prop. 3.9; MR §3.3.2 (pp. 95–102): Eqs. 3.69–3.84, Def. 3.20, Prop. 3.21, Prop. 3.22, Example 3.23, Def. 3.24. Optional: MLS Ch. 2 §4 (pp. 51–58; MLS orders twists $(v,\omega)$) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation and recorded data only) |

Lab 1 gave you SE(3) and the transform tree: *where* a frame is. Today you compute *how fast* it moves. The SE(3) functions and the `TransformTree` of Lab 1 are already complete in `starter.py` (marked GIVEN; paste your own Lab 1 versions over them if you prefer). The functions you write today join your `sbeg361_kinematics` library; Lab 3 (exponential coordinates) and Lab 4 (force–torque sensor frame change) reuse `vec_to_se3` and `adjoint`.

**Application context.**
*Biomedical:* in navigated spine surgery an optical tracker follows marker arrays on the patient's vertebra and on the screwdriver. The navigation software keeps a transform tree (tracker → patient reference → CT image → planned trajectory) and shows the tool on the CT in real time. From the tracked screwdriver poses alone you will recover its **twist**, the screw's **thread lead**, the **velocity of the tool tip** and the **angular and positional deviation** of the actual screw axis from the surgical plan.
*Defence (protective use):* an explosive-ordnance-disposal (EOD) robot opens the lid of a suspect container by turning its gripper about the approach axis. The mast camera and the robot base describe that same motion with different six-number twists; the adjoint converts between them.
All data are **synthetic, with illustrative values**.

# Learning objectives

By the end of the lab you can:

1. Compute the space and body angular velocities $[\omega_s] = \dot RR^{-1}$, $[\omega_b] = R^{-1}\dot R$ (MR Eqs. 3.37–3.38) and check $\omega_s = R\omega_b$ (Eq. 3.35) and $R[\omega]R^{\mathsf T} = [R\omega]$ (Prop. 3.8).
2. Implement $[\mathcal V]$ (Eq. 3.71) and $[\mathrm{Ad}_T]$ (Def. 3.20), validate them against `modern_robotics`, and verify the adjoint law in both forms, $T[\mathcal V]T^{-1} = [\mathrm{Ad}_T\mathcal V]$ and $R[\omega]R^{\mathsf T}=[R\omega]$.
3. Extract body and spatial twists from $T$ and $\dot T$, or from a sampled pose trajectory ($[\mathcal V_b] = T^{-1}\dot T$, $[\mathcal V_s] = \dot TT^{-1}$), compute the velocity of any body point, and interpret a twist as a screw $\{q,\hat s,h\}$.
4. Publish and inspect a moving transform tree in ROS 2 (tf2, RViz) and relate tf2's conventions to MR notation.

# Pre-lab check (complete before the session; 5 questions)

1. A frame turns about the fixed $\hat z_s$-axis: $R(t) = \mathrm{Rot}(\hat z, 2t)$. Compute $\dot RR^{\mathsf T}$ and give $\omega_s$ and $\omega_b$.
2. Write the skew-symmetric matrix $[\omega]$ for $\omega = (1, 2, 3)$.
3. Write the $4\times4$ matrix $[\mathcal V]$ for $\mathcal V = (0, 0, 1, 2, 0, 0)$.
4. What is the pitch of $\mathcal V = (0, 0, 2, 0, 0, 1)$?
5. A body has spatial twist $\mathcal V_s = (0,0,1,\ 0,0,0)$ (rad/s, m/s). What is the velocity of the body point currently at $x_s = (2, 0, 0)$ m?

*Answers (for self-check):* 1. $\dot RR^{\mathsf T} = \begin{bmatrix}0&-2&0\\2&0&0\\0&0&0\end{bmatrix}$, so $\omega_s = (0,0,2)$ rad/s; the axis $\hat z$ is the same in both frames, so $\omega_b = R^{\mathsf T}\omega_s = (0,0,2)$ as well. 2. $\begin{bmatrix}0&-3&2\\3&0&-1\\-2&1&0\end{bmatrix}$. 3. $\begin{bmatrix}0&-1&0&2\\1&0&0&0\\0&0&0&0\\0&0&0&0\end{bmatrix}$. 4. $h = \omega^{\mathsf T}v/\|\omega\|^2 = 2/4 = 0.5$ m/rad. 5. $\omega\times x + v = (0,0,1)\times(2,0,0) = (0, 2, 0)$ m/s.

# Setup

```bash
cd "<your Lab code folder>"   # or your Classroom 50 repo
source ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/activate   # or the course Docker image / JupyterHub
python -c "import numpy, scipy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py               # 14 failures (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Rules: do **not** call `modern_robotics`, `scipy.spatial.transform` or `np.linalg.inv` inside `starter.py`; they are only for checking. Everything in the GIVEN section of `starter.py` (`vec_to_so3`, `so3_to_vec`, `rp_to_trans`, `trans_to_rp`, `trans_inv`, `is_se3`, `transform_point`, `quat_xyzw_to_rot`, `TransformTree`, the loaders, `axis_angle_between`, `point_line_distance`) may be used.

**Tree convention (same as tf2 and Lab 1).** An edge `parent → child` stores $T_{\text{parent},\text{child}}$. `tree.lookup(target, source)` returns $T_{\text{target},\text{source}}$, the pose of `source` seen from `target` — the argument order of tf2's `buffer.lookup_transform(target, source, time)` and of `ros2 run tf2_ros tf2_echo target source`. The tf2 files store quaternions $(x,y,z,w)$; quaternions are taught next week, so `quat_xyzw_to_rot` is given.

**Twist convention.** MR order: angular part first, $\mathcal V = (\omega, v)$. MLS and the ROS message `geometry_msgs/Twist` (field `linear` before `angular`) use the other order — always check.

# Part A — Procedure (auto-tested)

Run the tests after each step. A checkpoint is complete when all listed tests pass; ask a TA to sign it off.

## Step 1 — Angular velocity, space and body (8 min)

Implement `angular_velocities(R, Rdot)`, returning $(\omega_s, \omega_b)$ from $[\omega_s] = \dot RR^{\mathsf T}$ and $[\omega_b] = R^{\mathsf T}\dot R$ (MR Eqs. 3.37–3.38; $R^{-1} = R^{\mathsf T}$).

**Checkpoint 1:** `test_angular_velocities_space_and_body` (random $R$ and $\omega_b$ with $\dot R = R[\omega_b]$; checks $\omega_s = R\omega_b$, MR Eq. 3.35, and Prop. 3.8), `test_angular_velocity_turntable_example`.
The turntable example: $R(t) = \mathrm{Rot}(\hat z, at)\,\mathrm{Rot}(\hat x, bt)$ with $a = 0.5$, $b = 2$ rad/s, at $t = 1$ s — for example a radiotherapy couch turning about the vertical while a positioning insert rolls about its own long axis. Derive by the product rule that $\omega_s = a\hat z_s + b\,(R\hat x)$ and $\omega_b = a\,(R^{\mathsf T}\hat z_s) + b\hat x_b$, then check your code: $\omega_s = (1.7552,\ 0.9589,\ 0.5)$, $\omega_b = (2,\ 0.4546,\ -0.2081)$ rad/s. Both have the same length (2.0616 rad/s): why?

## Step 2 — se(3) and the adjoint (6 min)

Implement `vec_to_se3`, `se3_to_vec`, `adjoint`.

**Checkpoint 2:** `test_vec_to_se3_matches_mr`, `test_adjoint_matches_mr_and_properties` (checks MR Eqs. 3.77–3.78 and the adjoint law in its two forms: $T[\mathcal V]T^{-1} = [\mathrm{Ad}_T\mathcal V]$ and, in the top-left block, $R[\omega]R^{\mathsf T} = [R\omega]$). Explain in one sentence why the first form contains the second.

## Step 3 — Twists and the velocity of a point (10 min)

Implement `twists_from_T_Tdot` (MR Eqs. 3.81–3.82) and `point_velocity(V, x)` $= \omega\times x + v$: the velocity of the body point with coordinates $x$, where $\mathcal V$ and $x$ are expressed in the **same** frame. In $\{s\}$ this follows from $\dot x_s = \dot T x_b = \dot TT^{-1}x_s = [\mathcal V_s]x_s$ (homogeneous coordinates; MR Eqs. 3.72–3.74); in $\{b\}$ it gives $R^{\mathsf T}\dot x_s$. `x` may be one point or an $(N,3)$ array.

**Checkpoint 3:** `test_twists_from_T_Tdot_mr_example_3_23` (MR Example 3.23, the car: $\mathcal V_s = (0,0,2,-2,-4,0)$, $\mathcal V_b = (0,0,-2,2.8,4,0)$), `test_twists_and_angular_velocities_agree` (the angular parts of the twists are your Step 1 $\omega_s$, $\omega_b$; $v_b = R^{\mathsf T}\dot p$), `test_point_velocity`.
For the car, compute three point velocities with `point_velocity(V_s, ·)`: at $x_s = 0$ you get $v_s = (-2,-4,0)$ (MR Eq. 3.73: $v_s$ is the velocity of the body point at the $\{s\}$ origin, *not* of the car); at the car's origin $p = (4, 0.4, 0)$ you get $\dot p = (-2.8, 4, 0)$ m/s; at $(2,-1,0)$ you get zero. What is special about $(2,-1,0)$?

## Step 4 — Screws (8 min)

Implement `screw_to_twist` and `twist_to_screw` (MR §3.3.2.2, Def. 3.24). In `twist_to_screw` return the axis point closest to the origin, $q = \omega\times v/\|\omega\|^2$ (MLS Eq. 2.43, p. 47), $h = \infty$ and `q = None` for a pure translation, and raise `ValueError` for the zero twist.

**Checkpoint 4:** `test_screw_to_twist_and_back` (the car's screw axis passes through $q = (2,-1,0)$ with $h = 0$), `test_pure_translation_and_zero_twist`, `test_pitch_is_frame_invariant`.

## Step 5 — Navigated pedicle-screw insertion and the EOD gripper (18 min)

```python
import numpy as np, starter as lab
nav = lab.load_tree_json("nav_tree.json")        # theatre, tracker, patient_ref, ct_image, plan_entry, tool_tip
t, Ts = lab.load_track_csv("tool_track.csv")      # 401 poses T_tracker_tool_marker, 50 Hz, 8 s
nav.add("tracker", "tool_marker", Ts[0])          # update this edge for each sample k (Lab 1)
```

a. **Warm-up (Lab 1 tree).** `nav.lookup("ct_image", "tool_tip")[:3, 3]` at $t = 0$ is $(142.0,\ 84.03,\ 210.26)$ mm. Update the `tracker → tool_marker` edge with `Ts[-1]`: the tip in the plan frame is $(0,\ 0.826,\ 4.997)$ mm at $t = 8$ s, i.e. 5.0 mm deep along the plan.
b. **Twists from the tracker.** Implement `twists_from_samples` (central differences, then take the skew-symmetric part of the rotation block). Plot the six components of $\mathcal V_b$ and of $\mathcal V_s$ against time. $\mathcal V_b \approx (0,\ 0,\ -1.5705,\ 0,\ 0.07853,\ -0.000625)$ is constant, and $\mathcal V_s \approx (-0.1911,\ -0.7652,\ -1.3581,\ -1.2325,\ 1.4820,\ -0.6623)$ (tracker coordinates) is constant too. Explain both facts using MR Prop. 3.22 and the fact that the screwdriver performs a constant screw motion. Check one sample: $\mathcal V_s = [\mathrm{Ad}_{T_k}]\mathcal V_b$.
c. **Screw parameters.** Apply `twist_to_screw` to the mean $\mathcal V_b$: $\dot\theta \approx 1.5705$ rad/s, $\hat s = (0,0,-1)$ (the shaft, in the marker frame), $q_b = (0.05,\ 0,\ 0)$ m. Convert the pitch to a **thread lead** in mm per turn ($2\pi h$); you should get $2.500$ mm. The exact speed is $\pi/2 = 1.5708$ rad/s. Explain the $1.6\times10^{-4}$ relative error of the central difference.
d. **Velocity of the tip.** With `point_velocity`, the marker origin moves at $(0,\ 78.53,\ -0.625)$ mm/s (body coordinates) but the tip, $x_b = (0.05, 0, -0.25)$ m, only at $(0,\ 0,\ -0.625)$ mm/s. Why is the tip so slow? Compute the tip velocity at every sample from $\mathcal V_s$ and the tip position in tracker coordinates, compare it with the central difference of the tip track (they agree to $10^{-9}$ m/s), and express it in the plan frame at $t = 4$ s: $(0,\ -0.0218,\ 0.6246)$ mm/s. What does the small $y$-component tell the surgeon?
e. **Deviation from the plan.** Transform the mean spatial twist into the CT frame, $\mathcal V_{ct} = [\mathrm{Ad}_{T_{ct,\text{tracker}}}]\mathcal V_{\text{tracker}}$, find its screw axis, and compare it with the plan's $\hat z$-axis (`lookup("ct_image", "plan_entry")`): the angle between the axes is $2.00^\circ$ and the planned entry point is $1.00$ mm from the actual axis. Would you accept this screw? Typical clinical grading tolerates a few millimetres of cortical breach; state your criterion.
f. **EOD gripper (defence, protective use).** `eod = lab.load_tree_json("eod_tree.json")`. The gripper turns about its approach axis $\hat z_g$ at 0.5 rad/s: $\mathcal V_g = (0,0,0.5,0,0,0)$. Express it in the mast-camera frame, $\mathcal V_c = [\mathrm{Ad}_{T_{cg}}]\mathcal V_g = (0,0,0.5,\ 0.2,0,0)$, and in the base frame, $(0.5,0,0,\ 0,0.2,0)$. Show with `twist_to_screw` that both describe the same physical screw (pitch 0, axis through the gripper origin along $\hat z_g$). A jaw point at $r_g = (0, 0.04, 0.1)$ m moves at $(-0.02, 0, 0)$ m/s in gripper coordinates; check that `point_velocity` in camera coordinates gives $R_{cg}$ times that.

**Checkpoint 5:** `test_body_twist_from_tracking_data`, `test_screw_axis_and_lead_of_the_screwdriver`, `test_tool_tip_velocity`, `test_eod_gripper_twist_in_camera_frame`. Finally, once every function is implemented, run the whole suite: `LAB_IMPL=starter python -m pytest -q test_lab.py` must show **14 passed**.

# Part B — ROS 2 tf2 and RViz (guided, not auto-tested)

Use the course ROS 2 Jazzy Docker image or an Ubuntu 24.04 machine with ROS 2 Jazzy (Course syllabus §11 Software setup). If ROS 2 is unavailable, complete Part A and answer B3 and B6 on paper; a TA will run the demo once for the group.

**B1 — Publish the tree.**

```bash
source /opt/ros/jazzy/setup.bash
cd ros2 && python3 nav_tf_broadcaster.py        # static edges on /tf_static, tool replayed on /tf at 50 Hz
ros2 topic echo /tf_static --once                    # second terminal: inspect one message
```

Find the `patient_ref → ct_image` entry and check that its translation equals the one in `nav_tree.json`.

**B2 — RViz.** Run `rviz2`, set *Global Options → Fixed Frame* to `theatre`, *Add → TF*, and tick *Show Names*. You should see the tracker above the table and the tool marker turning about the screwdriver shaft. Switch the Fixed Frame to `ct_image`, then to `tool_marker`. What changes on the screen, and what does not? Relate this to $\mathcal V_s$ (depends on the fixed frame) and $\mathcal V_b$ (does not). (Screenshot for your report.)

**B3 — Conventions.** Read `nav_tf_broadcaster.py`. Answer the `EXERCISE B3` comment: why must `tool_marker`'s parent be `tracker`, and what would tf2 do if a second node published `patient_ref → tool_marker` as well? (Compare with the `ValueError` of `TransformTree.add`.) Why does the script convert the rotation to a quaternion in $(x,y,z,w)$ order?

**B4 — Compare tf2 with your tree and your twist.**

```bash
ros2 run tf2_ros tf2_echo patient_ref plan_entry     # static: must match nav.lookup to 1e-6 m
ros2 run tf2_ros tf2_echo plan_entry tool_tip        # dynamic: depth z grows 0 -> 5 mm, then the replay restarts
```

Compare the first output with `nav.lookup("patient_ref", "plan_entry")`. For the second, read the $z$-translation twice, a known time apart, and estimate the tip speed along the plan: it must be close to the $0.625$ mm/s of Step 5d.

**B5 — The tree as a picture.** Run `ros2 run tf2_tools view_frames`. It writes a PDF of the tree. Compare it with `nav.path("ct_image", "tool_tip")`.

**B6 — Add a frame from the command line.** A needle-guide tip 50 mm above the planned entry point:

```bash
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z -0.05 --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id plan_entry --child-frame-id guide_tip
```

Explain from the plan frame's definition (+$\hat z$ points into the bone) why $z = -0.05$ is *above* the entry point. Then add the same edge to your Python tree and check `lookup("ct_image", "guide_tip")` against `tf2_echo ct_image guide_tip`.

**B7 (optional) — A listener node.** Write a 15-line `rclpy` node with a `tf2_ros.Buffer` and `TransformListener` that prints the tip depth along the plan (`buffer.lookup_transform("plan_entry", "tool_tip", rclpy.time.Time())`) at 5 Hz, together with its rate of change (a finite-difference tip speed).

# Deliverables (auto-checked notebook lab; not a formal report)

1. `starter.py` with all functions implemented; the full test suite passes.
2. A short notebook or PDF (≤ 3 pages): the answers of Steps 1 and 3, the twist plots from Step 5b, the numbers and answers from Steps 5a–5f, and one RViz screenshot with answers to B2–B6 (or the paper answers to B3 and B6).
3. The reflection below.

Marking (lab category): tests 50 %, plots and answers 35 %, reflection 15 %. Be prepared for a 2-minute oral check: a TA will point to any line of your code and ask you to explain it.

# Reflection (≤ 200 words)

- $\mathcal V_s$ of the screwdriver changes if the tracker is moved; $\mathcal V_b$ does not. Which one would you log in a surgical-navigation record, and why?
- The twist from tracking data is only as good as the tracker. If each pose had 0.1 mm of noise, which of $\dot\theta$, $h$ and $q$ would you expect to suffer most from central differencing, and how could you reduce the effect?
- In the EOD case the operator watches the camera image. Why is it useful that the camera twist $\mathcal V_c$ and the gripper twist $\mathcal V_g$ describe the same screw?
- **AI-use note (required):** state which AI tools you used (or "none"), what you asked, and what you kept or changed. You must be able to explain every line you submit.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: starter` | Run pytest from inside your Lab code folder, or keep `test_lab.py` next to `starter.py`. |
| Every test fails with `NotImplementedError` | Expected until you implement the functions; set `LAB_IMPL=starter`, not `solution`. |
| $\omega_s$ and $\omega_b$ swapped | $\dot RR^{\mathsf T}$ (multiply by $R^{\mathsf T}$ on the **right**) is the space form; $R^{\mathsf T}\dot R$ is the body form. |
| `adjoint` matches MR except the bottom-left block | It is $[p]R$, not $R[p]$ (MR Def. 3.20). |
| `point_velocity` wrong in one frame only | $\mathcal V$ and $x$ were in different frames: use $\mathcal V_s$ with $x_s$, or $\mathcal V_b$ with $x_b$. |
| `point_velocity` has the wrong sign | It is $\omega\times x$, not $x\times\omega$. |
| `twists_from_samples` returns non-constant $\mathcal V_b$ | You used a forward difference with $T_k$ instead of the central difference about $T_k$, or you mixed up $T^{-1}\dot T$ ($\mathcal V_b$) and $\dot TT^{-1}$ ($\mathcal V_s$). |
| Lead comes out as 15.7 mm | You multiplied $h$ by $2\pi$ twice, or used $\|v\|$ instead of $\omega^{\mathsf T}v/\|\omega\|^2$. |
| RViz shows "No transform from [x] to [theatre]" | The broadcaster is not running, or the Fixed Frame name is misspelled; check with `ros2 run tf2_ros tf2_monitor`. |
| `tf2_echo` waits forever for the tool | The dynamic edge is only published while `nav_tf_broadcaster.py` runs; static edges are latched on `/tf_static`. |

# C++ track

You may do Part A of this lab in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the results of MR §3.2.2 and §3.3.2 and with the numbers of the Python reference, so both tracks are marked against identical values. The plots and written answers of Step 5 may be produced with the Python helpers (`load_tree_json`, `load_track_csv`) or from your C++ results; only the auto-checks differ.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the stored random inputs), `CMakeLists.txt`. The data files are the same `nav_tree.json`, `eod_tree.json` and `tool_track.csv` of the Lab code folder. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up). Optional for Part B: `ros2/cpp/` holds an `rclcpp` version of `nav_tf_broadcaster.py` (see below).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 2 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*angular*,*adjoint*"    # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 14 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `angular_velocities` (Step 1); `vec_to_se3`, `se3_to_vec`, `adjoint` (Step 2); `twists_from_T_Tdot`, `point_velocity`, `point_velocities` (Step 3); `screw_to_twist`, `twist_to_screw` (Step 4); `twists_from_samples` (Step 5b). Given (GIVEN section): `vec_to_so3`, `so3_to_vec`, `rp_to_trans`, `trans_to_rp`, `trans_inv`, `is_se3`, `transform_point`, `transform_points`, `quat_xyzw_to_rot`, the whole `TransformTree`, `transform_from_tf`, `load_tree_json`, `load_track_csv`, `axis_angle_between`, `point_line_distance`. Do not use `mrcpp` or Eigen's general `.inverse()` inside `starter.cpp`.

**Checkpoints.** 1: `./build/lab_tests -tc="*angular_velocit*"`; 2: `-tc="*vec_to_se3*,*adjoint*"`; 3: `-tc="*Tdot*,*agree*,*point_velocity*"`; 4: `-tc="*screw_to_twist*,*pure_translation*,*pitch*"`; 5: all 14 test cases pass (`./build/lab_tests`). The C++ test cases have the same names as the 14 Python tests.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| NumPy 3-vector, 6-vector twist, `(4, 4)` array | `Eigen::Vector3d`, `lab::Vector6d`, `Eigen::Matrix4d` (6×6 adjoint: `lab::Matrix6d`) |
| `angular_velocities`, `twists_from_T_Tdot` return a tuple | `std::pair`: `auto [w_s, w_b] = lab::angular_velocities(R, Rdot);` |
| `point_velocity(V, x)` with `x` of shape `(N, 3)` | a separate function `point_velocities(V, X)`, `X` an N×3 `Eigen::MatrixXd` (one point per row) |
| `transform_point(T, x)` with `x` of shape `(N, 3)` | `transform_points(T, X)` (given) |
| `(N, 4, 4)` pose array; `(N-2, 6)` twist arrays | `std::vector<Eigen::Matrix4d>`; `Eigen::MatrixXd` with one twist per row |
| `twist_to_screw` returns a dict | returns a `lab::Screw` struct with the same field names: `sc.S`, `sc.thetadot`, `sc.h`, `sc.s_hat`, `sc.q` |
| `q = None` (pure translation) | `sc.q` is a `std::optional<Eigen::Vector3d>`: empty for a pure translation (`sc.q.reset()`); read it as `*sc.q` |
| `np.inf` | `lab::INF` (`std::numeric_limits<double>::infinity()`); test with `std::isinf(h)` |
| `screw_to_twist(None, s, np.inf, 0.05)` | pass any `q`, e.g. `Eigen::Vector3d::Zero()`: it is ignored when `h` is infinite |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `np.linalg.inv` (forbidden) | `.inverse()` (equally forbidden) |

A matrix-size mistake (e.g. a 4×4 times a 3-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**Part B in C++ (optional, not auto-tested).** `ros2/cpp/` is a small `ament_cmake` package (`sbeg361_nav_tf`) whose node `nav_tf_broadcaster` publishes exactly what `nav_tf_broadcaster.py` publishes; the B3 question is marked in its source as well. It needs a JSON reader: the system package (`sudo apt install nlohmann-json3-dev`) or, if that is not installed, the copy in `cpp_support/third_party` (found automatically when `cpp_support` sits next to or above your Lab code folder, also through the symlink below). In the ROS 2 Jazzy environment:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src && ln -s "<your Lab code folder>/ros2/cpp" ~/ros2_ws/src/sbeg361_nav_tf
cd ~/ros2_ws && colcon build --packages-select sbeg361_nav_tf && source install/setup.bash
ros2 run sbeg361_nav_tf nav_tf_broadcaster --ros-args -p lab_dir:="<your Lab code folder>"
```

Then continue with B1–B6 as written above (B7 may be written with `rclcpp`, `tf2_ros::Buffer` and `tf2_ros::TransformListener`).

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../nav_tree.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `too many arguments provided to function-like macro` | a template argument list such as `T.topRightCorner<3, 1>()` inside `CHECK_...(...)` | store it in a variable first |
| Results are wrong or garbage after `auto M = A * B;` | `auto` keeps an Eigen expression, not a matrix | write the type: `Eigen::Matrix4d M = A * B;` |
| `bad optional access` | you read `*sc.q` for a pure translation, or `twist_to_screw` never set `sc.q` | set `sc.q = ...` whenever `w.norm() > TOL`; check `sc.q.has_value()` before `*sc.q` |
| `test_pure_translation_and_zero_twist` fails on `CHECK_THROWS_AS` | `twist_to_screw` threw `std::logic_error` / `std::runtime_error` instead of `std::invalid_argument` | throw `std::invalid_argument` |
| colcon: `nlohmann/json not found` | neither the system package nor `cpp_support` was found | `sudo apt install nlohmann-json3-dev`, or put `cpp_support` next to your Lab code folder |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
