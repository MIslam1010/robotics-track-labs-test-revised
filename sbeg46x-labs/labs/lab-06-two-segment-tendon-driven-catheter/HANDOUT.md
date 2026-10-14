---
title: "Lab 6 — Two-segment tendon-driven catheter: constant-curvature kinematics, Jacobian and resolved-rate tip control"
subtitle: "SBEG 46X Medical Robotics — Week 6"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class (simulation for everybody) + about 2 h of independent work |
| CLOs | **46X-5** (model continuum medical robots — constant-curvature kinematics and Jacobians, tendon-driven robots — and use the models for tip control); 46X-1 (review: resolved-rate control through a Jacobian, Week 1 / Lab 1) |
| Reading | MR §3.3.3 (pp. 102–106: exponential coordinates, $e^{[\mathcal S]\theta}$, MR Eq. 3.88); MR §5.1 (space Jacobian, as used in SBEG 361); MLS Ch. 6 §4 (pp. 293–300: tendon extension functions and the coupling matrix). Not in the course library (†): Webster & Jones, *IJRR* 29(13), 2010 (constant-curvature review) |
| Files (your Lab code folder) | `starter.py` (your code: 22 functions marked TODO A1 … C4), `test_lab.py` (auto-checks, 25 tests), `data/catheter.json` (catheter parameters), `data/ablation_ring.csv` (an 8 mm ablation circle to trace), `data/tendon_log.csv` (bench encoder readings for five poses) |
| Library | The finished functions become module `continuum` of your course library `sbeg46x_medrob` (after `rcm`, `teleop`, `haptics` and `registration`); Lab 7 adds the Cosserat-rod model to the same module |
| Assessment | Auto-checked code (25 tests) + results table + three plots + reflection (one auto-checked lab in the Labs component; not a formal report) |
| AI use | Allowed for code **with disclosure** (AI-use note in the reflection); you must be able to explain every line in an oral check. Derivations are your own |

## Why this lab matters

A steerable ablation catheter is pushed from the groin to the heart; inside the left atrium the electrophysiologist must
place its tip on a ring around a pulmonary-vein ostium and drag it around that ring, point by point, while the heart
beats. Robotic catheter systems do this by pulling tendons from motors at the handle. To command them, the robot needs
the two maps of Lecture A — **tendon lengths ↔ arc parameters ↔ tip pose** — and a **Jacobian** to turn a desired tip
velocity into tendon and insertion rates. Today you implement both maps for a two-segment catheter, check the closed-form
constant-curvature (CC) arc against the matrix exponential of *Modern Robotics*, derive the Jacobian from the exponential,
and use it for resolved-rate tracking of the ablation ring — the same controller you wrote for the laparoscope in Lab 1,
now on a robot with no joints. The identical code steers a tendon-driven continuum inspection probe for confined-space
CBRN and search-and-rescue inspection (Workshop Problem 3) — only `catheter.json` changes.

## The catheter (illustrative values)

| Quantity | Value | Notes |
|---|---|---|
| Segment lengths $\ell_1$, $\ell_2$ | 40 mm, 25 mm | proximal, distal steerable segments of an 8 Fr (2.7 mm) catheter |
| Tendon radius $d$ | 1.0 mm (both segments) | |
| Tendon angles $\sigma$ | segment 1: 0°, 120°, 240°; segment 2: 60°, 180°, 300° | segment-2 tendons run through segment 1 at the same angle |
| Bending stiffness $EI_1$, $EI_2$ | $1.0\times10^{-4}$, $0.6\times10^{-4}$ N m² | distal segment softer |
| Limits | bending angle $\lvert \boldsymbol\theta_j\rvert \le 150°$ per segment; insertion $0 \le z \le 30$ mm | |
| Base | sheath exit frame $T_\text{base}$ in the atrium frame {s} (in `catheter.json`) | 20 mm along $\hat x_s$, 15 mm along $\hat z_s$, sheath tilted −30° about $\hat x_s$ |

**Joint vector and model.** $q = (z, \theta_{1x}, \theta_{1y}, \theta_{2x}, \theta_{2y})$: insertion out of the sheath, then the bending-angle vector of each
segment, $\boldsymbol\theta_j = \theta_j(\cos\phi_j, \sin\phi_j) = \ell_j(\kappa_{jx}, \kappa_{jy})$ (the lecture notes' bending vector $(\kappa_x, \kappa_y)$ times the length). Each
segment is the exponential of a constant body twist, so
$$T_\text{tip}(q) = T_\text{base}\,\mathrm{Trans}(0,0,z)\,e^{[\mathcal V_1]}\,e^{[\mathcal V_2]},\qquad \mathcal V_j = (-\theta_{jy},\;\theta_{jx},\;0,\;0,\;0,\;\ell_j).$$
Tendon $i$ of segment 1 has length $\ell_1 - d\,s_i\cdot \boldsymbol\theta_1$; tendon $i$ of segment 2 has length $\ell_1 + \ell_2 - d\,s_i\cdot(\boldsymbol\theta_1 + \boldsymbol\theta_2)$,
with $s_i = (\cos\sigma_i, \sin\sigma_i)$. Using $\boldsymbol\theta$ instead of $(\kappa, \phi)$ avoids the singularity of $\phi$ when a segment is straight.

## Learning objectives

By the end of the lab you can:

1. compute the robot-independent CC map (arc parameters → $SE(3)$) in closed form and as an MR exponential, and show they agree;
2. compute the robot-specific map (bending ↔ tendon lengths) of a multi-segment catheter, including the coupling of distal
   tendons to the proximal segment, and invert it from encoder data;
3. compute non-negative tendon tensions that hold a pose (virtual work, MLS Ch. 6 §4);
4. derive the space Jacobian of a product of exponentials of **configuration-dependent** twists with the left Jacobian of
   the exponential, and verify it by finite differences;
5. implement damped-least-squares resolved-rate tip control with joint limits, track a clinical path, and explain the
   gain–sample-time limit $K\Delta t < 2$.

# Pre-lab check (5 questions, submit before the session)

1. A segment of length 40 mm bends 90° in the plane $\phi = 0$. Give the tip position and the tip $\hat z$-axis.
2. Write the body twist per unit arc length $\mathcal V$ of a CC segment with curvature $\kappa$ in the plane $\phi$, and the body
   twist $\mathcal V_j$ of the whole segment in terms of $\boldsymbol\theta_j$ and $\ell_j$.
3. Three tendons at 0°, 120°, 240°, $d = 1$ mm. The segment bends 60° towards $\phi = 180°$. Give the three length changes.
4. Why must the distal-segment tendons of a two-segment catheter be accounted for when computing the tension in the
   proximal tendons?
5. Lab 1 used $\dot q = J^\dagger(\dot p_d + K(p_d - p))$. What does $J$ map here, what are its dimensions, and why will you use
   damped least squares rather than the plain pseudoinverse?

# Setup (5 min)

1. Copy your Lab code folder; work in `starter.py` or a copy with your own name (e.g. `continuum.py`).
2. Run `LAB_IMPL=starter python -m pytest -q test_lab.py`: 2 data tests pass and 23 fail with `NotImplementedError`.
   Re-run after every function; the test names tell you which function is being checked.
3. Provided helpers (do not edit): `load_catheter`, `load_csv`, `vee`, `trans_z`, `split_q`, `catheter_backbone` (3-D
   plotting of the backbone once B1 works).

# Procedure

### Checkpoint 1 — One segment, two ways (A1–A3, ≈ 10 min)

Implement `arc_transform` (closed form, twist-free: $R = \mathrm{Rot}(\hat z,\phi)\mathrm{Rot}(\hat y,\theta)\mathrm{Rot}(\hat z,-\phi)$; return a
straight segment for $\kappa = 0$), `bend_to_arc`/`arc_to_bend`, `segment_twist` and `segment_transform_exp`
(`mr.MatrixExp6(mr.VecTose3(V))`).

* E1. Reproduce Workshop Problem 1: $p$ and $R$ for $\ell = 40$ mm, $\theta = 90°$, $\phi = 0$ and $\phi = 120°$.
* E2. For 1000 random $(\kappa, \phi, \ell)$ report the largest difference between `arc_transform` and `segment_transform_exp`.
  Plot the backbone of one segment for $\theta = 0, 30, \dots, 150°$ in the plane $\phi = 0$.

### Checkpoint 2 — Tendons: lengths, inversion, tensions (A4–A6, ≈ 10 min)

Implement `tendon_lengths`, `bend_from_tendons` (least squares; for three tendons $\boldsymbol\theta = -\tfrac{2}{3d}\sum\Delta l_i s_i$),
`tension_allocation` (minimum-norm tensions, then add the same constant so that the smallest is zero) and
`segment_tensions`.

* E3. Reproduce Workshop Problem 2 (a)–(d). Plot $\Delta l_i(\phi)$ for $\theta = 90°$ and $\phi \in [0, 360°)$.

### Checkpoint 3 — The two-segment catheter (B1–B3, ≈ 10 min)

Implement `catheter_fk`, `catheter_tendon_lengths`, `catheter_tendon_jacobian` (the constant $6\times5$ matrix $\partial l/\partial q$;
its bending part transposed is the MLS coupling matrix), `bend_from_catheter_tendons` and `catheter_tensions`
(segment 2 first; then segment 1 supplies only what the segment-2 tendons do not).

* E4. For the reference pose $q_\text{ref} = (10\text{ mm}, 0.4, -0.7, 1.1, 0.3)$ give the tip position in {s} and the six tensions.
  Draw the catheter with `catheter_backbone`.
* E5. `data/tendon_log.csv` holds the six measured length changes (5 µm encoder noise) and the insertion for five bench poses.
  Estimate $(\boldsymbol\theta_1, \boldsymbol\theta_2)$ and the tip position for each pose; report the residual $\lVert A\hat{\boldsymbol\theta} - \Delta l\rVert$.

### Checkpoint 4 — The Jacobian from the exponential (B4–B5, ≈ 10 min)

Each segment's twist depends on $q$, so the PoE Jacobian of SBEG 361 does not apply directly. Use
$$\frac{d}{dt}\big(e^{[\mathcal V]}\big)\,e^{-[\mathcal V]} = \Big[\,\textstyle\sum_{k\ge0}\frac{1}{(k+1)!}\,\mathrm{ad}_{\mathcal V}^k\;\dot{\mathcal V}\Big]\quad(\texttt{dexp\_left}, \text{ with } \texttt{mr.ad}),$$
so the columns of $J_s$ for segment $j$ are $\mathrm{Ad}_{T_{j-1}}\,\texttt{dexp\_left}(\mathcal V_j)\,\partial\mathcal V_j/\partial \boldsymbol\theta_j$, where $T_{j-1}$ is the
pose of the segment's base; the insertion column is $\mathrm{Ad}_{T_\text{base}}(0,0,0,0,0,1)$. Implement `dexp_left`,
`catheter_jacobian_space` and `tip_position_jacobian` ($\dot p = [-[p]\;\;I]\,\mathcal V_s$).

* E6. Compare $J_s(q_\text{ref})$ with central finite differences of `catheter_fk` for $h = 10^{-3}, 10^{-5}, 10^{-7}$ and explain the
  trend. Report the singular values of the position Jacobian at $q = (10\text{ mm}, 0, 0, 0, 0)$ and at $q_\text{ref}$, and
  interpret the largest one.

### Checkpoint 5 — Resolved-rate tip control (C1–C4, ≈ 10 min + homework)

Implement `clamp_q`, `dls_inverse`, `resolved_rate_step`, `go_to` and `track_path`.

* E7. From $q_0 = (10\text{ mm}, 0, 0.3, 0, 0.3)$ drive the tip to the first point of `ablation_ring.csv` with `go_to`
  ($\Delta t = 10$ ms, $K = 5$ s⁻¹). Plot the error on a log scale; how many steps to 0.1 mm?
* E8. Track the ring (100 Hz, $K = 20$ s⁻¹). Report the largest tip error, the largest bending angle of each segment, the
  insertion range and the largest tendon pull. Plot the commanded and achieved rings in 3-D with the catheter at four instants,
  and the six tendon displacements against time — this is the motor command a robotic catheter system would send.
* E9. Run the ring at 10 Hz (every 10th sample) with $K = 5, 20, 25$ s⁻¹. Explain the result with the discrete-time error
  dynamics $e_{k+1} = (1 - K\Delta t)\,e_k$.
* E10 (homework). Command a target 200 mm away along the sheath axis (out of reach). What do the clamp and the damping do?

# Deliverables

1. Your module passing all 25 tests (`LAB_IMPL=<your module> python -m pytest -q test_lab.py`).
2. A results table with: E1 values; E2 largest difference; E3 tensions; E4 tip and tensions; E5 estimates and residuals;
   E6 finite-difference errors and singular values; E7 steps; E8 and E9 errors.
3. Three plots: single-segment backbones (E2); catheter on the ablation ring at four instants (E8); tendon displacements
   against time (E8).
4. Reflection (≤ 200 words): (a) one sentence each on why the arc parameters are singular for a straight segment but $\boldsymbol\theta$ is
   not, why distal tendons must be compensated proximally, and why $K\Delta t$ is limited; (b) one limitation of the CC model
   for a real catheter in a beating heart (Lab 7 addresses loads with the Cosserat rod); (c) **AI-use note**: tool, what you
   asked, what you kept, what you checked yourself (or "no AI used").

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| `arc_transform` correct for $\phi = 0$ only | missing the final $\mathrm{Rot}(\hat z,-\phi)$, or $p$ radial part not rotated by $\phi$ |
| `nan` for a straight segment | dividing by $\kappa = 0$: handle $\kappa = 0$ separately (A1); `bend_to_arc` must return $\phi = 0$ for $\boldsymbol\theta = 0$ |
| tendon test fails by a sign | the tendon on the inside of the bend gets **shorter**: $l_i = \ell - d\,s_i\cdot \boldsymbol\theta$ |
| negative tensions | you returned the minimum-norm solution; add $-\min(t)$ to every tendon |
| segment-1 tensions wrong only when segment 2 is bent | the segment-2 tendons also load segment 1 — subtract their moment first (B3) |
| Jacobian columns 2–5 wrong, column 1 right | used $\mathrm{Ad}$ of the segment **tip** instead of its base, or forgot `dexp_left` (correct only for straight segments) |
| finite-difference check fails near a straight segment | `mr.MatrixExp6` switches formulas when $\lVert\omega\rVert < 10^{-6}$; use $h \ge 10^{-5}$ for differences around $\boldsymbol\theta = 0$ |
| `go_to` stalls far from the target | the target is out of reach, a limit is active, or `lam` is too large; print `clamp_q` activity |
| tracking oscillates or diverges | $K\Delta t \ge 2$ (E9): lower $K$ or raise the control rate |

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| E1 | $p$ at $\phi = 0$; at $\phi = 120°$ | (25.46, 0, 25.46) mm; (−12.73, 22.05, 25.46) mm |
| E2 | largest closed form – exponential difference | ≈ 1e−16 (machine precision) |
| E3 | $\Delta l$ for 90° at $\phi = 0$; tension holding it; tensions for 60° at $\phi = 60°$ | (−1.571, 0.785, 0.785) mm; 3.93 N; (2.62, 2.62, 0) N |
| E4 | tip in {s} at $q_\text{ref}$ | (46.55, 10.00, 75.38) mm |
| | tensions (seg-1 tendons, seg-2 tendons) | (0, 0.21, 3.07), (3.06, 0, 2.22) N |
| E5 | estimated $(\boldsymbol\theta_1, \boldsymbol\theta_2)$, poses 1–5 | (0.495, −0.013, 0.006, 0.017); (0.001, −0.004, 0.001, 1.199); (0.402, −0.603, 0.797, 0.304); (−0.999, 0.493, 1.500, −0.390); (1.603, 1.006, −0.898, 0.898) |
| | tip positions, poses 1–5 (mm) | (41.65, 34.88, 76.09); (20.04, 46.11, 68.60); (44.89, 18.79, 86.88); (−3.44, 39.73, 53.51); (52.32, 52.39, 33.29) |
| E6 | largest finite-difference error at $q_\text{ref}$, $h = 10^{-3}, 10^{-5}, 10^{-7}$ | ≈ 1.6e−7, 2e−11, 8e−10: truncation ∝ $h^2$ for large $h$, round-off ∝ $10^{-16}/h$ for small $h$; best near $h = 10^{-5}$ |
| | singular values of $J_p$ (mm per m of insertion, mm per rad of bending): straight; $q_\text{ref}$ | 1000, 46.7, 46.7; 1000.5, 37.9, 28.4 (the 1000 is insertion: 1 mm per mm = 1000 mm per m; the two numbers carry different units, so compare them only after scaling) |
| E7 | initial error; steps to 0.1 mm | 34.7 mm; 115 steps (1.15 s) |
| E8 | max tip error; max $\lvert \boldsymbol\theta_1\rvert$, $\lvert \boldsymbol\theta_2\rvert$; insertion range; max tendon pull | 0.063 mm; 77.9°, 27.1°; 5.6–28.4 mm; 1.59 mm |
| E9 | 10 Hz, $K$ = 5, 20, 25 s⁻¹ | 0.64 mm; 1.57 mm ($K\Delta t = 2$, marginal); diverges (≈ 72 mm, $K\Delta t = 2.5$) |
| E10 | out-of-reach target | error stays ≈ 121 mm; insertion saturates at 30 mm; no blow-up |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, results table and deliverables are the same (deliverable 1 is `cpp/starter.cpp` instead of your Python module); the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with the independent closed form and finite differences written in `test_lab.cpp`, and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided helpers: `load_catheter`, `load_csv`, `vee`, `trans_z`, `split_q`, `catheter_backbone`, and `gradient_rows` = `np.gradient(P, t, axis=0)` — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `plot_results.py` (plots for E2, E7 and E8), `CMakeLists.txt`. The data files are the same files in `data/`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 6 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*arc*,*segment_twist*"  # only some tests (like pytest -k)
python cpp/plot_results.py                     # plots for E2, E7 and E8
```

With the untouched starter, the program builds and all 25 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same order as in Python, 22 functions): `arc_transform`, `bend_to_arc`, `arc_to_bend`, `segment_twist`, `segment_transform_exp` (A1–A3); `tendon_lengths`, `bend_from_tendons`, `tension_allocation`, `segment_tensions` (A4–A6); `catheter_fk`, `catheter_tendon_lengths`, `catheter_tendon_jacobian`, `bend_from_catheter_tendons`, `catheter_tensions` (B1–B3); `dexp_left`, `catheter_jacobian_space`, `tip_position_jacobian` (B4–B5); `clamp_q`, `dls_inverse`, `resolved_rate_step`, `go_to`, `track_path` (C1–C4). The helpers of `helpers.cpp` are given, as the provided helpers of `starter.py`. As in Python, you may call `mrcpp` (`mr::rp_to_trans`, `mr::matrix_exp6`, `mr::vec_to_se3`, `mr::vec_to_so3`, `mr::adjoint`, `mr::ad`, `mr::pinv`).

**Checkpoints.** 1: `./build/lab_tests -tc="*arc*,*segment_twist*"` (5 test cases); 2: `-tc="*tendon_lengths_hand*,*bend_from_tendons*,*tension_allocation*,*segment_tensions*"` (4); 3: `-tc="*catheter*,*fk*,*data_files*"` (8); 4: `-tc="*dexp*,*jacobian*"` (4, plus the tendon Jacobian of Checkpoint 3); 5: `-tc="*clamp*,*dls*,*resolved*,*track*"` (4); finally all 25 test cases pass (`./build/lab_tests`). The C++ track has the same 25 tests as the Python track, with the same names. Difference: in C++ the two data tests also call your code (`test_catheter_json` checks `catheter_tendon_lengths` of the straight catheter, `test_data_files_load` checks `bend_from_catheter_tendons` on pose 2 of the tendon log), so they pass only in Checkpoint 3, not with the untouched starter.

**Printing the results table.** For the values of E1, E3–E6, E9 and E10, add a temporary `TEST_CASE` at the end of `test_lab.cpp` (include `<iostream>`; remove it before you submit), for example:

```cpp
TEST_CASE("my_e4") {
    const lab::Catheter cat = lab::load_catheter();
    lab::Vector5d q_ref;
    q_ref << 0.010, 0.4, -0.7, 1.1, 0.3;
    std::cout << "tip (mm): " << 1e3 * lab::catheter_fk(q_ref, cat).block<3, 1>(0, 3).transpose() << "\n";
    std::cout << "tensions (N): " << lab::catheter_tensions(q_ref, cat).transpose() << "\n";
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(lab::tip_position_jacobian(q_ref, cat));
    std::cout << "singular values of J_p: " << svd.singularValues().transpose() << "\n";
}
```

For the plots, `./build/lab_tests` writes `lab6_cpp_segment_backbones.csv` (E2), `lab6_cpp_goto.csv` (E7: `go_to` towards the first ring point), `lab6_cpp_ring.csv` and `lab6_cpp_ring_backbones.csv` (E8: commanded and achieved ring, $q$, tendon displacements, and the catheter at $t$ = 0, 2, 4, 6 s) into the folder you run it from, each one when the functions it uses work; `python cpp/plot_results.py` saves `lab6_e2_backbones.png`, `lab6_e7_goto.png`, `lab6_e8_ring.png` and `lab6_e8_tendons.png`. For E2 (1000 random arcs), E6, E9 and E10 write your own loops in the temporary test case.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| catheter dict `cat["ell"]`, `cat["sigma"]`, … | `lab::Catheter` with the same field names: `cat.ell(0)`, `cat.sigma.row(j)` (2 × 3: row = segment, column = tendon; radians), `cat.T_base`, `cat.theta_max` |
| `q` NumPy array of 5 | `lab::Vector5d`; `q.segment<2>(1)` is `q[1:3]`, `q.segment<2>(3)` is `q[3:5]` |
| `split_q(q)` returns a tuple; `bend_to_arc` returns a tuple | structs `SplitQ` (`z, theta_v1, theta_v2`) and `ArcParams` (`kappa, phi, theta`) |
| twist, 6 tendon lengths or tensions; $6\times5$, $3\times5$ Jacobians | `lab::Vector6d`; `lab::Matrix65d`, `lab::Matrix35d` |
| `sigma` array (one angle per tendon), tendon lengths of one segment | `Eigen::VectorXd` |
| result dicts of `go_to`, `track_path` | `lab::GoToResult` (`q`, `err`), `lab::TrackResult` (`q`, `p`, `err`, `tendon_dl`); one **row** per step |
| `W=None` | `W = Vector5d::Ones()` (default argument) |
| `load_csv(name)` returns a dict of columns | returns `sbeg::CsvTable`: `table.data.col(table.col("x"))` |
| `np.gradient(P, t, axis=0)` | `gradient_rows(P, t)` (given, same numbers) |
| `np.linalg.lstsq(A, b)`, `np.linalg.pinv(A)`, `np.linalg.inv(A)` | `A.colPivHouseholderQr().solve(b)`, `mr::pinv(A)`, `A.inverse()` |
| `math.factorial(k + 1)` | a running product in a `double` (integers overflow quickly) |
| plots with Matplotlib | CSV files written by `lab_tests` + the given `plot_results.py` |

A matrix-size mistake (e.g. a $3\times5$ Jacobian multiplied by a 6-vector) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range".

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../data/...` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `too many arguments provided to function-like macro` | a template argument list with a comma inside `CHECK_...` (e.g. `T.block<3, 1>(0, 3)`) in your own test | put the argument in parentheses |
| wrong values only for some poses, or random-looking numbers | an Eigen expression stored in `auto` (`auto J = A * B;`) | write the type: `lab::Matrix65d J = ...;` or call `.eval()` |
| `dexp_left` wrong or `inf` | the factorial kept in an `int`, or `1 / (k + 1)` integer division | keep $(k+1)!$ in a `double` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
