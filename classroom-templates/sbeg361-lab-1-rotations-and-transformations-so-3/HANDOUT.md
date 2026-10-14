---
title: "Lab 1 — Rotations and transformations: SO(3)/SE(3) property tests and transform trees"
subtitle: "SBEG 361 Introduction to Robotics — Week 2"
---

**Session 4 · 50 min in class (Parts A–D) + up to 2 h to finish (Part E), due before the Week 3 lab · Individual work, pairs may discuss · Tier 0 (simulation and recorded data only)**
CLOs: 361-2 (represent and compose orientations and poses with SO(3)/SE(3)), 361-7 (implement and validate against a reference library).
Reading: MR §3.1–3.2.1 (printed pp. 60–74) and §3.3.1 (printed pp. 87–95: Def. 3.13, Eqs. 3.62–3.67, Props. 3.15–3.18, Figs. 3.14–3.15, Example 3.19); MR App. B.1 (ZYX Euler angles, Eq. B.1) for one question of Part C. Library: `modern_robotics` (`RotInv`, `MatrixExp3`, `VecToso3`, `ProjectToSO3`, `TestIfSO3`, `RpToTrans`, `TransToRp`, `TransInv`).

This lab starts your own kinematics library (`sbeg361_kinematics`). It follows the lectures of this week: every SO(3) function you write has an SE(3) twin written right next to it (`rot_inv` / `trans_inv`, `is_rotation` / `is_se3`, `rotate_in_fixed` / `transform_in_fixed`, …). Later labs build on these functions: Lab 2 (Week 3) adds twists and the adjoint on top of `trans_inv` and publishes a transform tree like today's with ROS 2 tf2; Lab 3 (Week 4) writes the matrix exponential that `rot_axis_angle` anticipates and derives the quaternion formula that is given today as `quat_xyzw_to_rot`; Lab 4 transforms wrenches between the frames you learn to handle here. Some later starters give your functions back as complete helpers under the names of that lab (Lab 5: `rot_axis(axis, angle)` = your `rot_x`/`rot_y`/`rot_z`; Lab 9: `rot(axis, angle)` = your `rot_axis_angle`, normalising the axis instead of raising).

**Application context.**
*Biomedical:* an optical tracker in the operating theatre follows marker arrays on a laparoscopic instrument and, in navigated spine surgery, on the patient's vertebra and on the screwdriver. Its raw orientations must be checked before use (Part E1), and the navigation software keeps a transform tree (tracker → patient reference → CT image → planned trajectory) to show the tool tip on the CT in real time (Part E2).
*Defence (protective use):* an explosive-ordnance-disposal (EOD) robot pans its arm about a fixed vertical axis and rolls its wrist about the gripper's own axis (Part C), and its software keeps the same kind of tree (odometry → base → mast camera → suspect package; base → gripper) to tell the operator where the package is relative to the gripper (Part D).
All data are **synthetic, with illustrative values**.

## 1. Objectives

By the end of the lab you can:

1. build $\mathrm{Rot}(\hat x,\theta)$, $\mathrm{Rot}(\hat y,\theta)$, $\mathrm{Rot}(\hat z,\theta)$, $\mathrm{Rot}(\hat\omega,\theta)$ (MR p. 72) and $T = (R, p)$ (MR Eq. 3.62) in NumPy;
2. test numerically whether a matrix belongs to SO(3) ($R^TR = I$, $\det R = 1$; MR Definition 3.1) or to SE(3) (MR Definition 3.13), and explain the tolerance you chose;
3. invert $R$ and $T$ without a general matrix inverse ($R^{-1} = R^T$, MR Prop. 3.3; $T^{-1} = (R^T, -R^Tp)$, MR Eq. 3.64);
4. use $R$ and $T$ in their three roles, side by side: represent a frame, change the frame of a vector or point (MR Eqs. 3.22, 3.65), displace a frame about/along a fixed or a body axis (MR Eqs. 3.23–3.24 and 3.66–3.67), including the planar case $(P,p)$ and its SE(2) matrix (MR Eqs. 3.8–3.9, 3.63);
5. build a tf2-style transform tree and answer "where is frame $\{x\}$ seen from frame $\{y\}$?" for any two frames, reproducing MR Example 3.19;
6. apply both groups to real-looking data: clean surgical-tracker orientations (reject invalid samples, project near-rotations onto SO(3)) and follow a navigated screwdriver tip in CT and plan coordinates.

## 2. Pre-lab check (complete before the session, 5 questions)

1. How many independent numbers does a $3\times3$ rotation matrix contain, and how many constraints does $R^TR = I$ impose? How many independent numbers does a $4\times4$ transformation matrix $T\in SE(3)$ contain?
2. Why is $R^TR = I$ not enough to define SO(3)? Give a $3\times3$ example that satisfies it but is not a rotation.
3. $R_{sb}$ and $T_{sb}$ are known. Which product rotates $\{b\}$ by $R$ about an axis **fixed in $\{s\}$**: $RR_{sb}$ or $R_{sb}R$? Which product displaces $\{b\}$ by $T$ expressed in $\{s\}$: $TT_{sb}$ or $T_{sb}T$?
4. Compute $T^{-1}$ for $T = (\mathrm{Rot}(\hat z,90^\circ),\ (1,0,0))$.
5. A tf2 message has `header.frame_id = "tracker"` and `child_frame_id = "tool_marker"`. Which MR matrix does it carry: $T_{\text{tracker},\text{tool\_marker}}$ or its inverse?

*Answers (self-check):* 1) 3 independent numbers, 6 constraints; $T$ has 6 (3 for $R$, 3 for $p$). 2) It allows $\det R = -1$ (reflections), e.g. $\mathrm{diag}(1,1,-1)$. 3) $RR_{sb}$ and $TT_{sb}$: pre-multiplication = fixed frame, post-multiplication = body frame, for both groups. 4) $(\mathrm{Rot}(\hat z,-90^\circ),\ (0, 1, 0))$, i.e. $\begin{bmatrix}0&1&0&0\\-1&0&0&1\\0&0&1&0\\0&0&0&1\end{bmatrix}$. 5) $T_{\text{tracker},\text{tool\_marker}}$: the pose of the child in the parent.

## 3. Setup

```bash
cd "<your Lab code folder>"          # or your Classroom 50 repo
source ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/activate   # or the course Docker image / Colab
python -c "import numpy, scipy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q   # all 30 tests fail with NotImplementedError: this is expected
```

On Windows PowerShell use `$env:LAB_IMPL="starter"; python -m pytest -q`.

Files: `starter.py` (your work), `test_lab.py` (auto-checks), and in `data/`: `tracker_orientations.csv` (20 tracker samples, Part E1), `nav_tree.json` and `tool_track.csv` (surgical navigation tree and 401 tracked screwdriver poses, Part E2), `eod_tree.json` (EOD robot tree, Part D). `make_data.py` and `make_data_nav.py` show how the data were generated — read them only *after* the lab.

**Rules.** Do **not** call `modern_robotics`, `scipy.spatial.transform` or `np.linalg.inv` inside `starter.py`; they are only for checking. The functions marked *(Given.)* in `starter.py` (`rot_x`, `quat_xyzw_to_rot`, the tree helpers `frames`, `_ancestors`, `_common`, `path`, `transform_from_tf`, and the loaders) are complete and may be used.

## 4. Procedure

### Part A — Building R and T (10 min)

Implement `rot_y`, `rot_z` (copy the pattern of the given `rot_x`) and `rot_axis_angle` using the explicit formula of MR p. 72. Raise `ValueError` for a non-unit axis. Then implement `rp_to_trans` (MR Eq. 3.62) and `trans_to_rp`.

**Checkpoint A** — `LAB_IMPL=starter python -m pytest -q -k "elementary or axis_angle or rp_to_trans"` passes. Then, in a Python shell, verify by hand that `rot_axis_angle([0,0,1], th)` equals `rot_z(th)` and that $\mathrm{Rot}(\hat\omega,\theta) = \mathrm{Rot}(-\hat\omega,-\theta)$.

### Part B — Group property tests: SO(3) and SE(3) side by side (12 min)

Implement the SO(3) column first, then its SE(3) twin:

| SO(3) | SE(3) twin |
|---|---|
| `so3_errors`, `is_rotation` (MR Eqs. 3.19, 3.21) | `is_se3` (MR Def. 3.13: rotation block in SO(3) and last row $(0,0,0,1)$) |
| `rot_inv` — transpose, no `np.linalg.inv` (MR Prop. 3.3) | `trans_inv` — $(R^T, -R^Tp)$, no `np.linalg.inv` (MR Eq. 3.64) |
| `project_to_so3` (SVD; force $\det = +1$) | — (to repair a $T$, project its rotation block) |

**Checkpoint B** — `LAB_IMPL=starter python -m pytest -q -k "so3 or is_rotation or group or is_se3 or trans_inv"` passes. Answer in your notebook:
(i) what is `so3_errors(2*np.eye(3))` and why is the orthogonality error $\sqrt{27}$?
(ii) `test_group_properties` checks closure, inverse, non-commutativity and length preservation. Which MR propositions (3.3–3.6) does each assertion correspond to? Which SE(3) propositions (3.15–3.18) are their twins, and which of them do `test_is_se3`, `test_trans_inv_matches_mr` and `test_distance_preserved` check?
(iii) `is_rotation` uses the tolerance $10^{-3}$ and `is_se3` uses $10^{-6}$. Why is $10^{-3}$ right for tracker samples with 6 decimals and sensor noise (the `modern_robotics` function `TestIfSO3` also uses $10^{-3}$), and why would $10^{-12}$ reject every real sample?

### Part C — The three uses of R and T (12 min)

Implement `change_frame` and its SE(3) twin `transform_point` (MR Eq. 3.65: $Tx = Rx + p$, for one point or an $N\times3$ array), `rotate_in_fixed`, `rotate_in_body` and their twins `transform_in_fixed`, `transform_in_body` (MR Eqs. 3.66–3.67), and the planar pair `planar_to_se2` (MR Eq. 3.63) and `planar_compose` (MR Eqs. 3.8–3.9).

**Checkpoint C (biomedical, SO(3))** — reproduce MR Figure 3.7: with $R_b = \mathrm{Rot}(\hat z, 90^\circ)$ and $p_a = (1,1,0)$, obtain $p_b = (1,-1,0)$ and $p_c = (0,-1,-1)$, and show that $R_c$ is obtained **both** as `rotate_in_body(R_b, rot_y(-pi/2))` and as `rotate_in_fixed(R_b, rot_x(pi/2))`.

**Checkpoint C (SE(3))** — reproduce MR Figure 3.14 ($T_{bc} = T_{sb}^{-1}T_{sc}$, and $v_b = (0,0,1.5)\mapsto v_s = (1.5,-2,0)$) and MR Figure 3.15 (the same $T = (\mathrm{Rot}(\hat z, 90^\circ), (0,2,0))$ applied in the fixed and in the body frame gives two different frames). Check that the rotation block of `transform_in_fixed` / `transform_in_body` is exactly what `rotate_in_fixed` / `rotate_in_body` give: the SE(3) rule contains the SO(3) rule.

**Checkpoint C (defence/humanitarian)** — an EOD robot gripper has $R_{sb} = \mathrm{Rot}(\hat z, 30^\circ)\mathrm{Rot}(\hat y, 45^\circ)$. The arm's base joint pans the whole arm $90^\circ$ about the vertical $\hat z_s$ and the wrist rolls $60^\circ$ about the gripper's own $\hat x_b$. Compute both new orientations with your functions. Which column of $R_{sb}$ is unchanged by the wrist roll, and why? $R_{sb}$ itself is a product of body-frame rotations: which ZYX Euler angles $(\alpha,\beta,\gamma)$ (MR App. B.1, Eq. B.1) does it have? (`test_eod_base_pan_and_wrist`.)

**Checkpoint C (planar)** — compose $(P,p) = (\mathrm{Rot}(60^\circ), (2,1))$ with $(Q,q) = (\mathrm{Rot}(30^\circ), (1,0.5))$ once with `planar_compose` and once as the product of two $3\times3$ SE(2) matrices; the results must agree (`test_planar_compose`).

All of Part C: `LAB_IMPL=starter python -m pytest -q -k "figure_3_7 or eod_base or fig_3_1 or distance or planar"` passes.

### Part D — The transform tree (16 min)

**Tree convention (same as ROS 2 tf2).** An edge `parent → child` stores $T_{\text{parent},\text{child}}$. `tree.lookup(target, source)` returns $T_{\text{target},\text{source}}$, the pose of `source` seen from `target`. This is the argument order of tf2's `buffer.lookup_transform(target, source, time)`, which you will use in Lab 2.

Implement the methods `TransformTree.add`, `TransformTree._to_ancestor` and `TransformTree.lookup`:

- `add` must reject a second parent for the same child, a cycle, `parent == child`, and a matrix that is not in SE(3) (`ValueError`; use your `is_se3`). Re-adding the **same** edge updates it; this is how the tracked tool will move in Part E2.
- `_to_ancestor(frame, ancestor)` walks up from `frame` and multiplies on the **left**: $T_{\text{anc},f} = T_{\text{anc},\ldots}\cdots T_{\text{parent}(f),f}$ (subscript cancellation, MR Eq. 3.22 for $R$ and its SE(3) twin).
- `lookup(target, source)` $= T_{\text{common},\text{target}}^{-1}\,T_{\text{common},\text{source}}$, where `common = self._common(target, source)` is their lowest common ancestor (use your `trans_inv`).

The JSON files store rotations as tf2 quaternions $(x, y, z, w)$; the given `quat_xyzw_to_rot` converts them (you will derive this formula in Week 4).

**Checkpoint D** — `LAB_IMPL=starter python -m pytest -q -k "tree or quaternion"` passes: `test_tree_mr_example_3_19` (builds the tree $a\to d$, $d\to b$, $d\to e$, $b\to c$ of MR Example 3.19 and must return MR's $T_{ce}$, p. 95; path `c → b → d → e`), `test_tree_rejects_bad_edges`, `test_quaternion_order_and_json_loader`, `test_eod_tree_package_in_gripper`.
Write $T_{ce}$ of Example 3.19 as a product of the four stored edges and their inverses, and check that the subscripts cancel. For the EOD tree, also print `eod.lookup("odom", "package")[:3, 3]`; you should get $(6.262,\ 2.613,\ 0.3)$ m.

### Part E — Applications (after the session, up to 2 h)

**E1. Cleaning surgical-tracker data.** An optical tracker reports the orientation $R_{sm}$ of a marker on a laparoscopic instrument (`data/tracker_orientations.csv`, illustrative values). Some samples are corrupted.

1. Load the data with `load_tracker_csv`, implement `classify_samples`, and print `so3_errors` for each sample. Which samples are invalid, and for each one say **which** condition fails (orthogonality, unit norm, or determinant).
2. Project the valid samples with `project_to_so3` and report the largest orthogonality error before and after.
3. The marker-to-tip calibration is $R_{mt} = \mathrm{Rot}(\hat x, 30^\circ)$. Implement `tool_orientations` and compute $R_{st}$ for the valid samples. Explain your multiplication order with the subscript-cancellation rule.
4. Should a reflected sample ($\det = -1$) be projected and used? Discuss in one sentence.

**E2. Navigated pedicle-screw insertion.** In navigated spine surgery the tracker follows the patient reference array and the screwdriver; the navigation tree is in `data/nav_tree.json` (static edges) and the tracked screwdriver poses $T_{\text{tracker},\text{tool\_marker}}(t)$ in `data/tool_track.csv` (50 Hz, 8 s). Both files are in **metres**; multiply positions by 1000 to report them in mm as below.

```python
import numpy as np, starter as lab
nav = lab.load_tree_json("data/nav_tree.json")        # theatre, tracker, patient_ref, ct_image, plan_entry, tool_tip
t, Ts = lab.load_track_csv("data/tool_track.csv")      # 401 poses T_tracker_tool_marker
nav.add("tracker", "tool_marker", Ts[0])               # update this edge for each sample k
```

a. **Tool tip on the CT.** Compute `nav.lookup("ct_image", "tool_tip")` at $t = 0$. The tip is at $(142.0,\ 84.03,\ 210.26)$ mm in image coordinates. Express it in the plan frame (`lookup("plan_entry", "tool_tip")`): it is $1.00$ mm from the planned entry point. Write down the chain of frames that `path("ct_image", "tool_tip")` returns and explain why the theatre frame is not needed.
b. **Depth along the plan.** Loop over all samples, update the `tracker → tool_marker` edge and record the tip in the plan frame. Plot its $z$ (depth along the planned trajectory, $+\hat z$ points into the bone) and its lateral distance $\sqrt{x^2+y^2}$ against time. At $t = 8$ s you should read $(0,\ 0.826,\ 4.997)$ mm: 5.0 mm deep, and the lateral error has changed from 1.00 to 0.83 mm. Why does a straight insertion change the lateral error? What does that tell you about the direction of the actual screw axis compared with the plan? (In Lab 2 you will measure that direction from the same data with twists.)

**Checkpoint E** — `test_tracker_classification`, `test_tracker_projection_and_tool`, `test_tool_tip_in_ct_at_start`, `test_tip_depth_along_plan`. Finally run the whole suite: `LAB_IMPL=starter python -m pytest -q` must show **30 passed**.

## 5. Deliverables (upload to Classroom 50 by the start of next week's lab)

- `starter.py` completed (all 30 tests passing) — auto-checked.
- `lab1_notes.md` (or a notebook, ≤ 3 pages): answers to checkpoints A–E, the table of invalid tracker samples with the failing condition and the before/after errors from E1, the numbers and the two plots of E2.
- Reflection (≤ 200 words), including the AI-use note below.

Marking (lab category; not a formal report): tests 50 %, plots and answers 35 %, reflection 15 %. Be prepared for a 2-minute oral check: a TA will point to any line of your code and ask you to explain it.

**Reflection prompt** (answer two of the three).

- Where could an orientation error of the type found in E1 harm a patient in image-guided surgery, or put an operator at risk during EOD work? What test would you add to a medical-device software pipeline to catch it?
- Your tree stores one parent per frame, as tf2 does. What would go wrong in surgery if two different registrations of the same CT were both allowed into the tree?
- In the EOD tree, which edges come from measurement and which from calibration or design? Which one would you re-check first if the gripper missed the package by 5 cm?

**AI-use note (required).** AI tools are allowed for this lab code **with disclosure**: state the tool, the prompts you used and what you kept. You must be able to explain every line in the oral check. Write "No AI tools used" if that is the case.

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: starter` | pytest run from another folder | `cd` into your Lab code folder first |
| All tests pass but you changed nothing | `LAB_IMPL` not set, so the tests do not use your file | set `LAB_IMPL=starter` |
| `rot_y` fails only for some angles | sign of $\sin$ in the wrong corner | compare with MR p. 72 |
| `test_project_to_so3` fails for the reflection | you returned $UV^T$ with $\det = -1$ | multiply by $\mathrm{diag}(1,1,\det(UV^T))$ |
| `is_rotation` rejects every tracker sample | tolerance too tight | use `tol=1e-3` |
| Angles look wrong by a factor of ~57 | degrees passed to `np.cos` | convert with `np.deg2rad` |
| `ValueError` from `rot_axis_angle` | axis not normalised | divide by `np.linalg.norm(w)` before calling |
| `trans_inv` test fails only in the position column | you wrote $-p$ or $-Rp$ | MR Eq. 3.64 needs $-R^Tp$ |
| `transform_point` fails for the $(7,3)$ array | you wrote `R @ x + p` | for one point per row use `x @ R.T + p` |
| Fig. 3.15 test: fixed and body results swapped | pre- and post-multiplication exchanged | fixed frame: `T @ T_sb`; body frame: `T_sb @ T` |
| Example 3.19 lookup is the inverse of MR's answer | arguments swapped | `lookup(target, source)` returns $T_{\text{target},\text{source}}$, so $T_{ce}$ is `lookup("c", "e")` |
| `_to_ancestor` gives wrong rotations for chains of length ≥ 2 | you multiplied on the right | walking **up** the tree, each parent transform goes on the **left** |
| `load_tree_json` raises `ValueError ... not in SE(3)` | `is_se3` too strict or wrong last-row check | compare with `np.allclose(..., atol=tol)`; the last row must be `[0, 0, 0, 1]` |

## 7. C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the worked examples of MR Chapter 3 and with the numbers of the Python reference, so both tracks are marked against identical values. The plots and written answers of Part E may be produced with the Python helpers (`load_tree_json`, `load_track_csv`) or from your C++ results; only the auto-checks differ.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`. The data files are the same files in `data/` (`tracker_orientations.csv`, `nav_tree.json`, `eod_tree.json`, `tool_track.csv`). `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 1 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*elementary*,*axis_angle*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 25 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python): `rot_y`, `rot_z`, `rot_axis_angle`, `rp_to_trans`, `trans_to_rp` (Part A); `so3_errors`, `is_rotation`, `rot_inv`, `project_to_so3`, `is_se3`, `trans_inv` (Part B); `change_frame`, `transform_point`, `transform_points`, `rotate_in_fixed`, `rotate_in_body`, `transform_in_fixed`, `transform_in_body`, `planar_to_se2`, `planar_compose` (Part C); the `TransformTree` methods `add`, `_to_ancestor`, `lookup` (Part D); `classify_samples`, `tool_orientations` (Part E). Given: `rot_x`, `quat_xyzw_to_rot`, `TransformTree::frames`, `_ancestors`, `_common`, `path`, `transform_from_tf`, `load_tree_json`, `load_tracker_csv`, `load_track_csv`. Do not use `mrcpp` or Eigen's general `.inverse()` inside `starter.cpp`.

**Checkpoints.** A: `./build/lab_tests -tc="*elementary*,*axis_angle*,*rp_to_trans*"`; B: `-tc="*so3*,*is_rotation*,*group*,*is_se3*,*trans_inv*"`; C: `-tc="*figure_3_7*,*eod_base*,*fig_3_1*,*distance*,*planar*"`; D: `-tc="*tree*,*quaternion*"`; E: all 25 test cases pass (`./build/lab_tests`). The Python track counts 30 tests because pytest counts each angle of `test_elementary_rotations_match_mr` separately; the C++ track checks the same angles in one test case. The C++ test cases have the same names as the Python tests.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| NumPy `(3, 3)` / `(4, 4)` array, 3-vector | `Eigen::Matrix3d` / `Eigen::Matrix4d`, `Eigen::Vector3d` (planar: `Eigen::Matrix2d`, `Eigen::Vector2d`; SE(2): `Eigen::Matrix3d`) |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` |
| `KeyError` / `LookupError` (given `_common`) | `lab::KeyError` / `lab::LookupError` (declared in `lab.hpp`) |
| `so3_errors`, `trans_to_rp`, `planar_compose` return a tuple | `std::pair`: `auto [e, d] = lab::so3_errors(R);`, `auto [R, p] = lab::trans_to_rp(T);` |
| `transform_point(T, x)` with `x` of shape `(N, 3)` | a separate function `transform_points(T, X)`, `X` an N×3 `Eigen::MatrixXd` (one point per row) |
| `(N, 3, 3)` / `(N, 4, 4)` arrays, boolean array | `std::vector<Eigen::Matrix3d>` / `std::vector<Eigen::Matrix4d>`, `std::vector<bool>` |
| quaternion list `[x, y, z, w]` | `Eigen::Vector4d(x, y, z, w)` — still scalar last |
| `tree.frames` (property), `self.parent`, `self.T` (dicts) | `tree.frames()` (a `std::set`), `parent`, `T` (`std::map`; read with `.at(name)`) |
| `np.linalg.norm(A)` (Frobenius), `np.linalg.det`, `A.T`, `A @ B` | `A.norm()`, `A.determinant()`, `A.transpose()`, `A * B` |
| `U, S, Vt = np.linalg.svd(M)` | `Eigen::JacobiSVD<Eigen::Matrix3d>` (hint in `starter.cpp`); `svd.matrixU()` is $U$ and `svd.matrixV()` is $V$, **not** $V^T$ |
| `np.linalg.inv` (forbidden in `rot_inv`, `trans_inv`) | `.inverse()` (equally forbidden) |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

`so3_errors`, `is_rotation` and `is_se3` take an `Eigen::MatrixXd` (any size) so that a wrong size can be detected. A matrix-size mistake (e.g. multiplying a $4\times4$ by a 3-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../nav_tree.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `project_to_so3` wrong only for some samples | used `svd.matrixV()` as if it were $V^T$ | transpose it |
| `too many arguments provided to function-like macro` | a template argument list such as `T.topRightCorner<3, 1>()` inside `CHECK_...(...)` | store it in a variable first |
| Tree results are wrong or garbage after `auto M = A * B;` | `auto` keeps an Eigen expression, not a matrix | write the type: `Eigen::Matrix4d M = A * B;` |
| `test_tree_rejects_bad_edges` fails on a `CHECK_THROWS_AS` | `add` threw `std::logic_error` / `std::runtime_error` instead of `std::invalid_argument` | throw `std::invalid_argument` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
