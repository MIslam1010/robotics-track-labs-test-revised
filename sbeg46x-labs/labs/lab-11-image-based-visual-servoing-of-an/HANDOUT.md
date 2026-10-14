---
title: "Lab 11 — Image-based visual servoing of an endoscope-holding arm with a simulated pinhole camera (Formal report 3)"
subtitle: "SBEG 46X Medical Robotics — Week 12"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work for the report |
| CLOs | **46X-7** (estimate pose from image features and design vision-based feedback — position- and image-based visual servoing — for medical and assistive robots); **46X-1** (kinematics of an endoscope through a port: the remote-centre-of-motion constraint of Week 1); **46X-10** (technical report, honest disclosure of tool use) |
| Reading | Corke, *Robotics, Vision and Control* (1st ed.): §11.1 (pp. 251–261, Eqs. 11.2 and 11.7–11.10), §11.2.1–11.2.3 (pp. 262–266, Eqs. 11.15–11.16), §14.2.4 (p. 396, Eq. 14.12), §15.1 (pp. 456–459), §15.2 (pp. 459–472, Eqs. 15.1–15.11), §16.4 (pp. 488–489). *Modern Robotics* (MR): §3.3.2 (body twist, MR Eq. 3.70, p. 96), MR Eq. 3.88 (p. 103), §3.3.3.2 (matrix logarithm, p. 104), §5.1.2 (body Jacobian, MR Eq. 5.18, p. 183), §6.3 (inverse velocity kinematics, MR Eq. 6.7, pp. 230–231). Week 1 notes (4 DOF through a port, software RCM); this week's lecture notes and Workshop 11 |
| Files (your Lab code folder) | `starter.py` (your code), `test_lab.py` (auto-checks), `course_models.py` (the endoscope holder `ur5_laparoscope`: UR5 + 0.30 m laparoscope, copied from Course resources: Course robot models and hardware safety checklist), `data/scene.json` (cameras, target, start poses, endoscope scene), `data/calib_points.csv` (calibration-block markers and their measured pixels), `data/expected.json` |
| Hardware | **Simulation.** Optional, ungraded: run the IBVS loop of Part C on the course arm with a USB camera and a printed four-dot fiducial in a supervised hardware session (Safety note) |
| Assessment | Auto-checked code (tests) + **Formal report 3** (IEEE two-column, ≤ 6 pages, template and rubric below), due **before the Week 13 lab practical** |
| AI use | Allowed for code **with disclosure** (AI-use note in the report, appendix A); you must be able to explain every line in an oral check. Derivations, numbers, figures and conclusions in the report are your own and come from your runs |

## Why this lab matters

In laparoscopic surgery the surgeon's two hands hold the instruments, so somebody else — an assistant, or a robotic
endoscope holder — must point the camera. A good assistant keeps the instrument tip in the middle of the screen, does
not drift, does not shake, and never pulls on the patient's abdominal wall at the port. A robot can do this with
**visual servoing**: it measures where the instrument tip is in the image and moves the camera so that the tip goes
back to the centre. No model of the patient is needed — only the camera model, the **interaction matrix** that says how
image features move when the camera moves, and the arm's Jacobian. You will build the camera model and calibrate it,
derive and check the interaction matrix, run image-based (IBVS) and position-based (PBVS) visual servoing on a free
camera, find out what each one is sensitive to (depth, calibration, large rotations, the field of view), and finally
close the loop through a port on the UR5 endoscope holder, where gain, video latency and depth error decide between a
calm assistant and a shaking one. The same loop points the gripper camera of an EOD robot arm at a suspect object
from a safe stand-off, centres a needle guide on a fiducial, and aims an inspection camera into an aircraft bay
(Lecture B).

## Learning objectives

By the end of the lab you can:

1. build the intrinsic matrix, project points with a pinhole camera and calibrate a camera from known 3-D points by
   the direct linear transform (Corke §11.1–11.2);
2. derive the interaction matrix of a point feature in MR twist order, verify it numerically and explain its rank and
   null space (Corke §15.2.1);
3. implement IBVS and PBVS (pose from a planar target through a homography), and quantify their sensitivity to depth
   estimates, calibration errors, large rotations about the optical axis and the field of view;
4. restrict the camera twist to the 4 DOF allowed by a port (remote centre of motion), map it to joint rates with the
   body Jacobian (MR Eq. 6.7), and judge the effect of gain, latency and depth error on the closed loop;
5. report the results in an IEEE-style technical report with a justified recommendation for an endoscope holder.

# Pre-lab check (complete before the session, 5 questions)

1. The endoscope camera has $f = 2.4$ mm, square pixels of 4 µm, 640 × 480 pixels and its principal point at
   (320, 240). Write $K$, project the point $P_c = (0.01, -0.005, 0.08)$ m given in the camera frame, and give the
   horizontal field of view.
2. For a point at the principal point with depth $Z = 0.08$ m, write the $2\times6$ interaction matrix $L$ in MR order
   $(\omega_x, \omega_y, \omega_z, v_x, v_y, v_z)$. Give three camera twists that do not move the point in the image.
3. What is the rank of the stacked interaction matrix of 1, 2 and 3 point features (in general position)? Why does
   IBVS of a full 6-DOF camera need at least three points, and what goes wrong if they are collinear?
4. The scope tip is $\ell = 0.10$ m beyond the port. The holder pans with $\omega_x = 0.2$ rad/s and nothing else
   ($\omega_y = \omega_z = \dot d = 0$). Write the camera twist that keeps the port on the scope axis, and the image
   velocity of an instrument tip at the image centre, 0.08 m in front of the camera. How much of it comes from the
   rotation and how much from the translation?
5. A proportional image loop sampled at $h = 1/30$ s sees an image that is $d$ frames old:
   $e_{k+1} = e_k - \lambda h\, e_{k-d}$. It is stable for $0 < \lambda h < 2\sin\!\big(\pi/(2(2d+1))\big)$. What is
   the largest gain $\lambda$ for $d = 0$ and for $d = 4$ (133 ms of video latency)?

# The simulated cameras, target and endoscope holder

| Quantity | Value (all illustrative) |
|---|---|
| Inspection camera (Parts A–D; Corke's default camera) | $f = 8$ mm, 10 µm pixels, 1024 × 1024, principal point (512, 512): $f/\rho = 800$ px |
| Target (fiducial) | four markers on the corners of a 0.5 m square, centred at (0, 0, 3) m, facing the camera |
| Goal view | a 400 × 400 px square centred in the image (camera 1 m in front of the target, axes aligned) |
| Start poses | `corke`: camera at (1, 1, −3) m rotated 0.6 rad about $z$ (Corke §15.2.2); `fov`: camera at (−1.0, 1.7, 2.0) m looking at (0.3, 1.0, 3.0) m, rolled by −2.55 rad; roll tests: goal pose rotated about the optical axis by 0.5 … 2.5 rad |
| Servo settings | $\lambda = 1\ \text{s}^{-1}$, $h = 0.1$ s, stop when the rms image error ≤ 0.5 px, at most 400 steps |
| Endoscope camera (Part E, calibration data) | $f = 2.4$ mm, 4 µm pixels, 640 × 480, principal point (320, 240): $f/\rho = 600$ px |
| Endoscope holder | `ur5_laparoscope` (UR5 + 0.30 m scope, camera $\{c\}$ = $\{b\}$ at the scope tip, optical axis $z_b$), start $\theta_0 = (2.6, -2.5, 2.3, -1.4, -1.57, 0)$ rad, joint-speed limit 1 rad/s |
| Port and instrument | port 0.10 m behind the tip on the optical axis; instrument tip 0.08 m in front of the camera, 15 mm right of the centre at $t = 0$, still until $t = 3$ s, then a circle of radius 15 mm at 0.1 Hz while moving ±10 mm in depth |
| Endoscope loop | camera rate 30 Hz ($h = 1/30$ s), $\lambda = 2\ \text{s}^{-1}$, latency 0 frames, true depth (stereo endoscope), no zoom loop ($k_z = 0$), no port correction ($k_{\text{rcm}} = 0$); the run lasts 15 s |
| Calibration data | 32 markers on two orthogonal faces of a 4 × 4 cm calibration block, seen by the endoscope camera from about 9 cm, pixel noise $\sigma = 0.3$ px |

Conventions: pixel coordinates $(u, v)$ have their origin at the top-left corner, $u$ to the right, $v$ down; the
camera frame $\{c\}$ has $z_c$ along the optical axis, $x_c$ to the right and $y_c$ down. Camera poses are written
$T_{wc}$ (camera in the world, like $T_{sb}$ in MR). **Twists follow MR: $\mathcal V_c = (\omega, v)$, angular part
first, expressed in the camera frame (a body twist).** Corke writes the camera velocity as $(v, \omega)$; his
interaction matrix (Eq. 15.6) therefore has its two $2\times3$ blocks in the other order. Features are stacked
$(u_1, v_1, u_2, v_2, \dots)$.

# Setup

```bash
cd "<your Lab code folder>"
LAB_IMPL=starter python -m pytest -q test_lab.py     # at the start: 27 tests fail with NotImplementedError, 1 data test passes
```

Use the course Python environment (or the Docker image / JupyterHub). Work in `starter.py` and keep the function names
and signatures. The provided section of `starter.py` gives `load_scene`, `trans`, `rotz`, `pose_from_spec`,
`camera_K`, `target_points`, `load_calib`, `decompose_camera_matrix`, `endoscope_setup`, `instrument_tip`,
`simulate_endoscope`, `endoscope_metrics` and `servo_metrics`. You may call `modern_robotics` (`TransInv`,
`MatrixExp6`, `VecTose3`, `MatrixLog6`, `se3ToVec`, `JacobianBody`) where the TODO hints say so.

# Procedure and checkpoints

### Checkpoint A — Camera model and calibration (≈ 10 min)

1. `intrinsic_matrix` (Corke Eq. 11.7 with $f$ folded in) and `project_points` (Corke Eqs. 11.6–11.8): transform to
   the camera frame with $T_{wc}^{-1}$, divide by depth, apply $K$; refuse points behind the camera.
2. `camera_matrix_dlt` (Corke Eqs. 11.15–11.16, solved by the SVD instead of fixing $C_{34} = 1$) and
   `reprojection_rms`.
3. `python -m pytest -q test_lab.py -k "intrinsic or project or camera_matrix or reprojection"`. The tests reproduce Corke's
   numbers (pp. 255–258 and 460) and recover $K$ and the camera pose exactly from noise-free data.

**Checkpoint question:** your calibration of `calib_points.csv` gives a reprojection error of about 0.37 px with
$\sigma = 0.3$ px of pixel noise. Is that good? Compare with the reprojection error of the *true* camera on the same
data (Experiment A).

### Checkpoint B — Interaction matrix (≈ 10 min)

1. `interaction_matrix`: Corke Eq. 15.6 written for the MR twist order, stacked for $N$ points.
2. `python -m pytest -q test_lab.py -k interaction`. One test compares $L\mathcal V$ with central finite differences of
   the projection when the camera moves by $e^{[\mathcal V]s}$; one checks the null space of a single point (roll,
   motion along the ray, and the $v_x$/$\omega_y$ and $v_y$/$\omega_x$ pairs that Corke describes on pp. 462–464).

**Checkpoint question:** which columns of $L$ change when you halve the depth $Z$, and why does that make depth the
quantity IBVS is most often unsure of?

### Checkpoint C — IBVS and PBVS on a free camera (≈ 20 min)

1. `ibvs_velocity` (Corke Eq. 15.11), `camera_step` (MR Eq. 3.88), `run_ibvs` (the loop with its stopping rule and
   the condition number of $L$).
2. `homography_dlt` (Corke Eq. 14.12), `pose_from_homography` ($H \sim K[r_1\ r_2\ t]$, then the closest rotation by
   the SVD), `pbvs_velocity` ($\mathcal V_c = \lambda\log(T_{ct}T_{c^*t}^{-1})$, MR §3.3.3.2) and `run_pbvs`.
3. `python -m pytest -q test_lab.py -k "(ibvs and not rcm) or camera_step or homography or pbvs"`.

**Checkpoint question:** PBVS moves the camera on a screw path straight to the goal pose, IBVS moves the features on
straight lines in the image. Which one would you trust to keep an instrument in view, and which one to keep the
endoscope away from the liver?

### Checkpoint D — The endoscope through the port (rest of the session, then at home)

1. `rcm_twist_basis` ($\mathcal V_c = W(\ell)\,\xi$, $\xi = (\omega_x, \omega_y, \omega_z, \dot d)$),
   `rcm_ibvs_velocity` (pan and tilt only: roll fixed to keep the image horizon, insertion given by the zoom loop),
   `joint_rates` (MR Eq. 6.7 with one common scale factor for the joint-speed limit) and `rcm_errors`.
2. Run the whole suite: `python -m pytest -q test_lab.py` — all 28 tests must pass.
3. Do the experiments below. Save every figure with axis labels and units.

**Experiment A — Calibration of the endoscope camera.**

* A1. Calibrate with all 32 markers of `calib_points.csv`; split $C$ with `decompose_camera_matrix`. **Table 1:**
  estimated $f_u$, $f_v$, skew, $(u_0, v_0)$, camera position error (mm) and reprojection rms, next to the true
  values (`scene["endoscope_camera"]`, `scene["calib"]["T_wc"]`), and the reprojection rms of the true camera on the
  same data.
* A2. Repeat with six markers only (rows 0, 3, 12, 15, 19, 28) and evaluate on all 32. Then try six markers from one
  face only: what does the SVD tell you, and why does the DLT need non-coplanar points?

**Experiment B — The interaction matrix.**

* B1. Print $L$ for Corke's point (672, 672) at $Z = 5$ m and compare it, column by column, with Corke p. 462. Which
  columns move when you swap to Corke's order?
* B2. For the target seen from the `corke` start and from the goal, report the singular values and the condition
  number of the stacked $L$ (8 × 6). Relate the condition number to "perceptibility" (Corke p. 465).

**Experiment C — IBVS and the depth it needs.**

* C1. `run_ibvs` from the `corke` start with the true depth. **Figure 1:** feature paths in the image (start, goal,
  path), camera trajectory and the six components of $\mathcal V_c$ against the step. Report steps to converge,
  final pose error and the peak camera speed. Would a real arm follow that first step? (Corke p. 472 limits the norm
  of the commanded velocity.)
* C2. Repeat with constant depth estimates $\hat Z$ = 1 m (the goal depth), 0.3, 3 and 10 m. **Table 2:** steps,
  straightness of the image paths (`servo_metrics`), condition number at the start and at the end. **Figure 2:** the
  image paths for $\hat Z$ = 0.3 m and 10 m. Explain with the structure of $L$ why an underestimated depth slows the
  loop and an overestimated one makes the path jagged (Corke §15.2.3, p. 469).
* C3. Start at the goal rotated about the optical axis by 0.5, 1.0, 1.5, 2.0 and 2.5 rad, with IBVS and with PBVS.
  **Figure 3a:** IBVS camera retreat (`servo_metrics`, m) against the angle. Then try $\pi$ rad with IBVS and explain
  what happens (Corke p. 471, "camera retreat").

**Experiment D — PBVS, the field of view and calibration errors.**

* D1. `run_ibvs` and `run_pbvs` from the `fov` start. **Figure 3b:** both feature paths with the image border
  drawn. Which method loses features over the border (`min_margin` < 0), and what does the other pay instead (camera
  path length, retreat)? The simulator keeps projecting points outside the image so that you can see where they go;
  a real tracker would lose them.
* D2. Give both controllers a wrong camera model `K_hat` (focal length +10 %, $u_0$ + 20 px; the camera itself is
  unchanged). **Table 3:** steps, converged?, final image error and final camera position error of IBVS and PBVS.
  Explain why IBVS still ends exactly at the goal view and PBVS does not (where does each law use $K$?).

**Experiment E — The endoscope holder.**

* E1. `simulate_endoscope(scene)` with the default settings. **Figure 4:** distance of the instrument tip from the
  image centre, insertion depth $\ell$ and lateral port error against time; mark the end of the still phase
  ($t = 3$ s). Then sweep $\lambda$ = 1, 2, 4, 8 s⁻¹. **Table 4 (first rows):** rms and maximum image error for
  $t \ge 5$ s, error at $t = 3$ s, peak joint speed, saturated fraction, maximum port error. Compare the rms error with
  the steady lag of a proportional loop following a moving target, $\approx$ (image speed of the tip) / $\lambda$, with
  an image speed of about $600 \cdot 0.015 \cdot 2\pi \cdot 0.1 / 0.08 \approx 71$ px/s.
* E2. Add 4 frames of video latency (`delay=4`) for $\lambda$ = 4, 8 and 12 s⁻¹. **Figure 5:** image error against
  time for the three gains. Compare with pre-lab question 5. What happens at the port when the loop is unstable, and
  why is that a patient-safety issue and not only an image-quality issue?
* E3. Replace the stereo depth by a constant $\hat Z$ = 0.04, 0.16 and 0.32 m (true depth about 0.08 m), first with no
  latency and then with `delay=4`, $\lambda = 8$. **Table 4 (last rows).** Show that through a port the depth error acts
  like a gain error, $\lambda_{\text{eff}} \approx \lambda\,(1 + \ell/Z)/(1 + \ell/\hat Z)$ near the image centre, and
  use it to explain which wrong depth destabilises the delayed loop.
* E4. The port error grows slowly because the joints are integrated with finite steps (the twist keeps the port on the
  axis only to first order). Compare the maximum port error with `k_rcm=0` and `k_rcm=5` (a small lateral correction
  in the provided simulator), and run the zoom loop `k_z=1` (insertion follows the instrument depth): report the range
  of $\ell$. How far from the port would you allow the scope axis to drift before stopping the robot?

**Expected values (to check yourself; TAs use them for calibration).**

| Quantity | Expected |
|---|---|
| Goal view $p^*$ | (312, 312), (312, 712), (712, 712), (712, 312) px |
| A1, calibration with 32 markers: $f_u$, $f_v$, skew, $(u_0, v_0)$ | 599.67, 599.34, −0.47, (320.63, 238.31) px (true 600, 600, 0, (320, 240)) |
| A1, reprojection rms: estimated $C$ / true camera; camera position error | 0.367 px / 0.388 px; 0.17 mm |
| A2, six markers: $f_u$, $f_v$, $(u_0, v_0)$; rms on all 32; position error | 591.69, 593.26, (326.07, 234.54) px; 0.622 px; 1.28 mm |
| C1, IBVS true depth: steps, final position error, peak camera speed, cond($L$) start / end | 60 steps (6 s), 1.3 mm, 25.2 m/s, 942 / 34 |
| C2, IBVS with $\hat Z$ = 1 / 0.3 / 3 / 10 m: steps; straightness | 78 / 270 / 33 / 48 steps; 1.27 / 1.37 / 1.07 / 2.57 (true depth: 1.01) |
| C3, roll 0.5 / 1.0 / 1.5 / 2.0 / 2.5 rad: IBVS steps; retreat | 54 / 60 / 64 / 66 / 67 steps; 0.04 / 0.16 / 0.41 / 0.96 / 2.57 m (PBVS: 54–69 steps, no retreat) |
| C3, roll $\pi$ rad, IBVS | not converged after 400 steps; the camera retreats without bound (mean depth > $10^9$ m), image error 283 px |
| C, PBVS from the `corke` start: steps; straightness | 78 steps; 1.01 |
| D1, `fov` start, IBVS / PBVS: steps; minimum margin to the border; camera path | 65 / 77 steps; +78 px / −193 px; 40.1 m (retreat 7.6 m) / 2.7 m |
| D2, wrong `K_hat`, IBVS / PBVS: converged; final image error; final position error | yes (60 steps) / no (400 steps); 0.47 px / 34.6 px; 1.3 mm / 94 mm |
| E1, default ($\lambda = 2$): rms / max error ($t \ge 5$ s); error at 3 s; max port error; peak $\lvert\dot\theta\rvert$ | 34.0 / 38.2 px; 0.23 px; 0.28 mm; 0.33 rad/s |
| E1, $\lambda$ = 1 / 4 / 8: rms error; peak $\lvert\dot\theta\rvert$ | 60.7 / 17.6 / 8.8 px; 0.17 / 0.67 / 1.00 rad/s (saturated 0.4 % of the time at $\lambda = 8$) |
| E2, 4 frames latency, $\lambda$ = 4 / 8 / 12: rms error; max port error; saturated | 17.8 / 8.9 / 118 px; 0.52 / 1.30 / 32.7 mm; 0 / 1.1 / 64 % (unstable at 12) |
| E3, $\hat Z$ = 0.04 / 0.16 / 0.32 m (no latency): rms error | 49.7 / 25.0 / 20.3 px |
| E3, `delay=4`, $\lambda = 8$, $\hat Z$ = 0.04 / 0.32 m: rms error; max port error | 13.8 / 115 px; 0.69 / 32.8 mm (unstable with 0.32 m) |
| E4, `k_rcm=5`: max port error; `k_z=1`: range of $\ell$ | 0.030 mm (from 0.28 mm); 92.1–109.2 mm |

<!-- library-block:begin -->
# Your `sbeg46x_medrob` library

When the tests pass, add the functions you wrote in Lab 11 to a new module `vision.py` of your library. The functions
you write (16):

`intrinsic_matrix(f, rho_w, rho_h, u0, v0)`, `project_points(K, T_wc, P_w)`, `camera_matrix_dlt(P_w, p)`,
`reprojection_rms(C, P_w, p)`, `interaction_matrix(p, Z, K)`, `ibvs_velocity(p, p_star, Z_hat, K, lam)`,
`camera_step(T_wc, V_c, h)`, `run_ibvs(T_wc0, P_w, p_star, K, lam, h, Z_hat=None, n_max=400, tol=0.5, K_hat=None)`,
`homography_dlt(xy_t, p)`, `pose_from_homography(H, K)`, `pbvs_velocity(T_ct, T_cstar_t, lam)`,
`run_pbvs(T_wc0, P_w, xy_t, T_cstar_t, p_star, K, lam, h, n_max=400, tol=0.5, K_hat=None)`, `rcm_twist_basis(ell)`,
`rcm_ibvs_velocity(p, p_star, Z_hat, K, lam, ell, d_dot=0.0)`, `joint_rates(Blist, theta, V_c, qdot_max=None)`,
`rcm_errors(T_sc, p_port)`.

Also copy the provided `decompose_camera_matrix` and `servo_metrics` into `vision.py`. `rcm_twist_basis` is the
velocity form of the software RCM of Lab 1 (`rcm` module): keep both. Project teams with an endoscope-holder,
needle-guide or camera-pointing strand reuse this module. The full list of library functions of all labs is in the
Week 13 lab handout.
<!-- library-block:end -->

# Optional — ROS 2 pointers (ungraded)

* **Vision in ROS 2.** A camera driver publishes `sensor_msgs/Image` and `sensor_msgs/CameraInfo` (the $K$ of your
  calibration, in the `k` field, row by row); the `camera_calibration` package (`cameracalibrator` node) estimates $K$
  and the lens distortion from a printed checkerboard. A node that runs your `ibvs_velocity` subscribes to the detected
  feature points and publishes a twist to the arm's velocity controller. Write in your report (optional paragraph)
  which of your functions would sit in which node.
* **Mobile bases (project teams with the mobile-base scenario).** Nav2 packages wheel odometry (`/odom`), map-based
  localisation (AMCL, a particle filter), planners on costmaps and a path-tracking controller that outputs $(v, \omega)$
  on `/cmd_vel` (Lecture B, assistive mobile bases). In the course ROS 2 Docker image (or Ubuntu 24.04 with ROS 2
  Jazzy): `source /opt/ros/jazzy/setup.bash` and
  `ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False` (package and launch-file names follow the Nav2
  "Getting Started" page for your ROS 2 distribution; if a name has changed, use the one given there). Give the
  initial pose with **2D Pose Estimate**, send a **Nav2 Goal**, and `ros2 topic echo /cmd_vel` while it drives. Never
  run Nav2 on a real wheelchair with a person in it: a powered wheelchair is a medical device (Week 13).

# Deliverables

1. `starter.py` renamed `lab11_<ID>.py`, passing `LAB_IMPL=lab11_<ID> python -m pytest -q test_lab.py` (28 tests), and
   the 16 functions added to `vision.py` of your library.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 3** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–5 and
   Tables 1–4.
4. A reflection (≤ 200 words, appendix B) and an AI-use note (appendix A).

# Formal report 3 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Image-based visual
servoing of a robotic endoscope holder: camera calibration, interaction matrix, robustness and the remote centre of
motion*. Suggested length per section in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (pinhole model and DLT calibration, interaction matrix, IBVS and PBVS, RCM-constrained IBVS on a 6R holder), key numbers (calibration error, IBVS sensitivity to depth and calibration, tracking error, gain and latency limit, port error), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why an endoscope holder (or an EOD gripper camera, a needle guide) needs vision-based control; the laparoscopic scenario; objectives and contributions. |
| **II. Camera and motion model** (½–1 page) | Pinhole projection and $K$ (Corke Eqs. 11.2, 11.7–11.10), the DLT (Eqs. 11.15–11.16), the interaction matrix in MR twist order with its derivation from $\dot P_c = -v - \omega\times P_c$ (Corke Eqs. 15.1–15.6) and its rank and null space; the camera motion $T_{wc}e^{[\mathcal V]h}$ (MR Eq. 3.88). State that all values are illustrative. |
| **III. Controller design** (1 page) | IBVS (Corke Eq. 15.11), PBVS with the pose from a homography and the matrix logarithm (MR §3.3.3.2); the RCM twist basis $W(\ell)$, why roll is fixed and insertion is a separate loop, the joint rates by MR Eq. 6.7 with $J_b$ (MR Eq. 5.18) and the common scale factor; the stability limit with latency (pre-lab question 5). Cite Corke and MR with sections, equations and pages. |
| **IV. Verification** (¼ page) | Unit tests (what each checks), comparison with the expected values, and one independent check of your own (e.g. Corke's numbers on p. 462, a finite-difference $L$, or the lag estimate of E1). |
| **V. Results** (1½ pages) | Figures 1–5 and Tables 1–4. Every figure is referred to in the text and has a caption that says what to see. |
| **VI. Discussion** (1 page) | IBVS vs PBVS: what each controls directly and what it leaves free (image paths, camera path, field of view, retreat); robustness to depth and calibration errors and why; the endoscope loop: gain, latency and depth error as one stability budget, saturation, port forces, the 4 DOF of a port; what a clinical holder adds (voice or pedal enable, dead zone around the centre, speed limits, instrument-tracking failures, smoke and blood). One paragraph on a defence use (EOD camera arm or inspection camera) and how its constraints differ. Limits of the model (no lens distortion, perfect feature detection and correspondence, kinematic arm, rigid static anatomy). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences, with a recommendation (method, gain, maximum tolerated latency, depth source) for the endoscope holder. |
| **References** | Corke and MR (sections, equations, pages); IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "No AI tools were used."). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clinical (and, if relevant, defence) motivation with concrete needs (tip centred, no drift, no port forces, latency of the video chain); measurable objectives; key numbers in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Model and controller design | 20 | Correct projection, DLT and interaction matrix derived (not copied) in MR twist order, rank and null space explained; IBVS, PBVS and the RCM basis justified with their assumptions (depth, calibration, small steps, 4 DOF); stability limit with latency derived | Minor slips in derivation or notation | Equations listed without explanation | Wrong or missing |
| Verification evidence | 10 | All tests pass; expected values reproduced; at least one independent check of your own, quantified | Tests pass, independent check weak | Some tests fail or no check | No verification |
| Results (figures and tables) | 25 | Figures 1–5 and Tables 1–4 complete, labelled, with units, goal points and image border drawn, captions that say what to see; numbers consistent with the expected values | One figure or table weak | Several missing or unreadable | Missing |
| Discussion, safety and recommendation | 20 | IBVS vs PBVS trade-off explained from the equations; depth and calibration robustness explained; gain–latency–depth budget quantified; port safety and clinical safeguards with concrete mitigations; justified recommendation; model limits | Recommendation with partial justification | Descriptive only | Missing |
| Writing, format, references | 10 | IEEE format, ≤ 6 pages, concise, correct citations with pages | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every figure and table | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, that IBVS reaches the exact goal view with a 10 % focal-length error
  while PBVS stops 9 cm away, or that a depth guess that is "too large" destabilises the delayed loop), and why?
* If you were the surgeon, what would you want the endoscope holder to do — and to show you — when the instrument
  leaves the image, the video freezes for a quarter of a second, or the robot reaches its speed limit?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it
  was wrong (a test, a hand calculation, a plot)?

# Safety note

This lab is simulation only. If you run IBVS on the course arm with a USB camera (optional, supervised hardware
session): never with a person in the workspace or in front of the camera; run the exact command first in simulation;
limit the commanded twist (≤ 0.05 m/s and ≤ 0.3 rad/s for first runs) and the joint speeds in the arm's own controller
**in addition to** your `joint_rates` scaling; stop the arm when the features are lost, when the image is older than
a set limit (latency watchdog) or when the condition number of $L$ exceeds a set threshold; keep a spotter with the
emergency stop who is not operating the laptop. Follow Course resources: Course robot models and hardware safety
checklist. A robotic endoscope holder used on patients is a medical device (Week 13).

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Projection off by a large amount or mirrored | You used $T_{wc}$ instead of $T_{wc}^{-1}$ to bring points into the camera frame, or divided by the wrong coordinate. |
| `camera_matrix_dlt` correct up to sign or scale only | Normalise by $\lVert C_{3,1:3}\rVert$ and fix the sign so that the points have positive depth. |
| Corke's interaction-matrix test fails with the right numbers in the wrong places | Columns in Corke's order $(v, \omega)$: MR order puts the three angular columns first. |
| Finite-difference test fails by a sign | The image moves opposite to the camera: the translational columns are $-f/Z$ on the diagonal. |
| `run_ibvs` converges in a different number of steps | Error not stacked as $(u_1, v_1, u_2, \dots)$ (use `reshape(-1)` on the $N\times2$ array), wrong stopping test (≤ tol checked **before** computing a new velocity), or $\hat Z$ replaced by the true depth. |
| `run_ibvs` raises "point behind the camera" | A gain or step too large, or a roll test close to $\pi$: IBVS drives the camera backwards and the steps grow (Corke p. 471). |
| `pose_from_homography` returns the target behind the camera | Sign of the scale: choose it so that $t_z > 0$. |
| PBVS converges to the wrong pose | $\log(T_{c^*t}T_{ct}^{-1})$ instead of $\log(T_{ct}T_{c^*t}^{-1})$ (the twist is expressed in the current camera frame). |
| `rcm_twist_basis` test fails | $v = \omega \times (0, 0, \ell)$, i.e. $v_x = \ell\omega_y$, $v_y = -\ell\omega_x$; the insertion speed goes into $v_z$. |
| Endoscope test fails only for the unstable case | Saturation scales one joint instead of the whole vector. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside your Lab code folder, or set `LAB_IMPL` to your file name without `.py`. |

# Pre-lab answers (released after the session)

1. $K = \begin{bmatrix}600 & 0 & 320\\ 0 & 600 & 240\\ 0 & 0 & 1\end{bmatrix}$;
   $(u, v) = (320 + 600 \cdot 0.01/0.08,\ 240 - 600 \cdot 0.005/0.08) = (395,\ 202.5)$ px; horizontal field of view
   $2\arctan(320/600) = 56.1°$ (vertical $43.6°$).
2. At the centre $x = y = 0$: $L = \begin{bmatrix}0 & -600 & 0 & -7500 & 0 & 0\\ 600 & 0 & 0 & 0 & -7500 & 0\end{bmatrix}$.
   No image motion for: roll $(0, 0, 1, 0, 0, 0)$; motion along the ray $(0, 0, 0, 0, 0, 1)$; and the pairs
   $(0, 1, 0, -0.08, 0, 0)$ ($\omega_y$ with $v_x = -Z\omega_y$) and $(1, 0, 0, 0, 0.08, 0)$.
3. Ranks 2, 4 and 6: each point gives two equations; three points in general position fix all six twist components.
   With collinear points the rotation about the line through them does not move them, and $L$ loses rank (Corke p. 464).
4. $\mathcal V_c = (0.2, 0, 0,\ 0, -0.02, 0)$ ($v = \omega \times (0, 0, \ell)$: $v_y = -\ell\omega_x$).
   $\dot v = 600 \cdot 0.2 + (-600/0.08)(-0.02) = 120 + 150 = 270$ px/s, $\dot u = 0$: 120 px/s from the rotation and
   150 px/s from the translation — through a port, pan and translation add up, so the image moves $(1 + \ell/Z) = 2.25$
   times faster than for a pure rotation.
5. $d = 0$: $\lambda h < 2$, $\lambda < 60\ \text{s}^{-1}$; $d = 4$: $\lambda h < 2\sin(\pi/18) = 0.347$,
   $\lambda < 10.4\ \text{s}^{-1}$ (Experiment E2: $\lambda = 8$ is stable, $\lambda = 12$ is not).

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, expected values, report and deliverables are the same (deliverable 1 is `cpp/starter.cpp` instead of `starter.py`; deliverable 2 is your experiments code in C++ plus your plotting script); the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with Corke's worked numbers, with the closed forms and finite differences of `test_lab.py`, with `data/expected.json` and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided helpers: `load_scene`, `trans`, `rotz`, `camera_K`, `target_points`, `load_calib`, `decompose_camera_matrix`, an `interaction_matrix` overload for one common depth, `servo_metrics`, `endoscope_setup`, `instrument_tip`, `simulate_endoscope`, `endoscope_metrics` — do not change), `course_models.hpp` / `course_models.cpp` (the endoscope holder `ur5_laparoscope`, numbers identical to `course_models.py` — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random inputs of `test_lab.py`), `plot_results.py` (Figures 1, 2 and 4), `CMakeLists.txt`. The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 11 --cpp                     # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*intrinsic*,*project*"  # only some tests (like pytest -k)
python cpp/plot_results.py                     # Figures 1, 2 and 4 (after the tests of Parts C-E pass)
```

With the untouched starter, the program builds and all 27 test cases fail (26 with `TODO: <function>`, and `test_project_points_rejects_points_behind` because no `std::invalid_argument` is thrown): this is expected. Rebuild (`cmake --build build`) after every change. A full run takes about a quarter of a second.

**Functions to write** (in `starter.cpp`, same order as in Python, 16 functions): `intrinsic_matrix`, `project_points`, `camera_matrix_dlt`, `reprojection_rms` (A1–A4); `interaction_matrix` (B1); `ibvs_velocity`, `camera_step`, `run_ibvs` (C1–C3); `homography_dlt`, `pose_from_homography`, `pbvs_velocity`, `run_pbvs` (D1–D4); `rcm_twist_basis`, `rcm_ibvs_velocity`, `joint_rates`, `rcm_errors` (E1–E4). The helpers of `helpers.cpp` and the robot of `course_models.hpp` are given, as the provided section of `starter.py` and `course_models.py`; `simulate_endoscope` calls your functions exactly as the Python one does. As in Python, you may call `mrcpp` where the hints say so (`mr::trans_inv`, `mr::matrix_exp6`, `mr::vec_to_se3`, `mr::matrix_log6`, `mr::se3_to_vec`, `mr::jacobian_body`, `mr::pinv`).

**Checkpoints.** A: `./build/lab_tests -tc="*data_files*,*intrinsic*,*project_points*,*camera_matrix_dlt*,*reprojection*"` (8 test cases); B: `-tc="*interaction*"` (3); C and D: `-tc="test_ibvs*,*run_ibvs*,*camera_step*,*homography*,*pbvs*"` (10); E: `-tc="*rcm*,*joint_rates*,*endoscope*"` (6); finally all 27 test cases pass (`./build/lab_tests`). The Python track counts 28 tests because pytest counts the two parameters of `test_endoscope_closed_loop` separately; the C++ track checks both cases inside one test case with the same name. Difference: in C++ the data test also checks that the measured calibration pixels lie within 2 px of `project_points` with the true camera, so it passes only in Checkpoint A, not with the untouched starter.

**Running the experiments and printing the tables.** Add a temporary `TEST_CASE` at the end of `test_lab.cpp` (include `<iostream>`; remove it before you submit), for example:

```cpp
TEST_CASE("my_results") {
    const lab::Scene sc = lab::load_scene();
    const lab::Target tg = lab::target_points(sc);
    const Eigen::Matrix3d K = lab::camera_K(sc.inspection_camera);
    const Eigen::MatrixXd ps = lab::project_points(K, sc.ibvs.T_wc_star, tg.P_w).first;   // goal view
    for (double Zh : {1.0, 0.3, 3.0, 10.0}) {                                               // C2
        const lab::ServoRun r = lab::run_ibvs(sc.start_poses.corke, tg.P_w, ps, K, sc.ibvs.lam, sc.ibvs.h,
                                              Eigen::VectorXd::Constant(4, Zh), sc.ibvs.n_max, sc.ibvs.tol);
        std::cout << "Z_hat " << Zh << ": " << r.n_iter << " steps, straightness "
                  << lab::servo_metrics(r, sc.inspection_camera).straightness << "\n";
    }
    lab::EndoOptions o;                                                                     // E2
    o.lam = 12.0;
    o.delay = 4;
    const lab::EndoSim sim = lab::simulate_endoscope(sc, o);
    const lab::EndoMetrics m = lab::endoscope_metrics(sim);
    std::cout << "E2 lambda 12, delay 4: rms " << m.rms_err << " px, max port error " << 1e3 * m.max_e_lat << " mm\n";
    Eigen::MatrixXd f5(sim.t.size(), 2);                                                    // Figure 5 data
    f5 << sim.t, sim.err;
    sbeg::write_csv("lab11_e2_lam12.csv", {"t", "err"}, f5);
}
```

The reference solution prints 78, 270, 33 and 48 steps (straightness 1.27, 1.37, 1.07, 2.57) and an rms error of 118 px with a 32.7 mm port error — the values of the expected-values table. For the figures, `./build/lab_tests` writes `lab11_cpp_ibvs.csv` and `lab11_cpp_pbvs.csv` (Corke start, true depth: step, rms error, the four feature points, the camera position) and `lab11_cpp_endoscope.csv` (default endoscope run: time, tip pixel, error, insertion depth, port error) into the folder you run it from, each one when the functions it uses work; `python cpp/plot_results.py` saves `lab11_fig1_image_paths.png`, `lab11_fig2_error.png` and `lab11_fig4_endoscope.png`. Plot the other figures from your own CSV files (as `lab11_e2_lam12.csv` above) with a short Python script; plotting stays in Python.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| scene dict `scene["ibvs"]["lam"]`, poses as `{"p", "rz", "look_at"}` entries | `lab::Scene` with the same names: `sc.ibvs.lam`, `sc.start_poses.corke`, `sc.endoscope.theta0`; poses are already `Eigen::Matrix4d` |
| `(N, 2)` / `(N, 3)` arrays of points | `Eigen::MatrixXd` with one **row** per point |
| scalar or `(N,)` depth `Z`, `Z_hat` | `Eigen::VectorXd` with one depth per point (`Eigen::VectorXd::Constant(N, z)`); `interaction_matrix(p, z, K)` with a `double` is given |
| tuples `(p, Z)`, `(xi, V)`, `(ell, e)`, `(K, T_cw)` | `std::pair` (`.first`, `.second`) |
| result dict of `run_ibvs` / `run_pbvs` | `lab::ServoRun` (`T` and `p` are `std::vector`s of n + 1 entries; `Z`, `err`, `V` (n × 6, one twist per row), `cond`, `converged`, `n_iter`) |
| `Z_hat=None`, `K_hat=None`, `qdot_max=None` | `std::optional`: omit the argument or pass `std::nullopt` |
| `simulate_endoscope(scene, lam=..., delay=..., Z_hat=..., k_z=..., k_rcm=..., T_end=...)` | `lab::simulate_endoscope(sc, opt)` with `lab::EndoOptions opt; opt.lam = 8.0; opt.delay = 4;` (fields `lam, delay, Z_hat, k_z, k_rcm, T_end`); result `lab::EndoSim` with the same field names |
| `course_models.ur5_laparoscope()` dict | `cm::ur5_laparoscope()` (fields `Slist, Blist, M, q_home, qdot_max, L_tool`) |
| `np.linalg.svd(A)`: last row of `Vt` | `Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV)`: last **column** of `svd.matrixV()` |
| `np.linalg.pinv`, `np.linalg.cond` | `mr::pinv`; largest / smallest of `svd.singularValues()` |
| plots with Matplotlib | CSV files written by `lab_tests` + the given `plot_results.py` |

A size mistake between fixed-size types (e.g. returning a 4 × 6 matrix as `Matrix64d`) is a compile error (`YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES`). A size or index mistake with the dynamic-size types `Eigen::MatrixXd` / `Eigen::VectorXd` (e.g. an `interaction_matrix` with N rows instead of 2N) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `too many arguments provided to function-like macro` | a comma inside `CHECK_...` that is not inside parentheses (e.g. `M.block<3, 1>(0, 3)`) in your own test | put the argument in parentheses or copy it into a variable first |
| the program stops with `terminating` / `abort` right after a matrix is filled with `<<` | the comma initializer got too few or too many values (e.g. a 3 × 3 matrix with 8 numbers) | count the values: one per entry, row by row |
| homography or camera matrix wrong although the rows of $A$ are right | `svd.matrixV()` is $V$, not $V^{\mathsf T}$; the null vector is its last **column**, and only with `Eigen::ComputeFullV` | as in the hint |
| wrong values only after a few steps, or random-looking numbers | an Eigen expression stored in `auto` (`auto V = -lam * Lp * e;`) | write the type: `Vector6d V = ...;` or call `.eval()` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
