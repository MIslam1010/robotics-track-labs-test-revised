---
title: "Lab 11 — RRT for the course arm; path smoothing and time scaling (Formal report 3)"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 12"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work for the report |
| CLOs | **461-5** (formulate planning problems in C-space; implement and compare sampling-based planners); 461-4 (time scaling within joint limits); **461-9** (technical report) |
| Reading | MR §10.2.2 (pp. 364–366), §10.5 (pp. 380–388), §10.8 (p. 396), §9.2.2 (pp. 330–334); MR Exercises 10.15–10.17 (p. 403); LaValle §5.5 (pp. 228–237); Corke §5.2.5 (pp. 102–104) |
| Files | `lab/starter.py` (your code), `lab/test_lab.py` (auto-checks), `lab/data/course_arm.json`, `lab/data/scenes.json` (data made by the course team from the shared course models) |
| Assessment | Auto-checked code (33 tests) + **Formal report 3** (IEEE two-column, ≤ 6 pages, rubric below), due one week after the lab, before the Week 13 lab practical |
| AI use | Allowed for code **with disclosure** (AI-use note in the report appendix); you must be able to explain every line in an oral check |

## Why this lab matters

In Week 11 you planned for a 2R arm on a grid. The course arm has six joints: a 5° grid would have
$72^6 \approx 1.4\times10^{11}$ cells (Workshop 12, Problem 5), so we switch to **sampling**. You will build the planning
pipeline that the project (scenarios 3 and 5) uses: **collision model → RRT/RRT\* → smoothing → time scaling → torque
check**. Two scenes, both with **illustrative geometry**, sized for the desktop course arm (reach ≈ 0.28 m):

* **Biomedical — `bench_or`, a scaled bench-top operating-room rig.** The arm carries a 0.125 kg instrument carrier from
  the Week 10 ready pose A (the start pose of Lab 9) to a hand-over point B2 on its left. A sterile-drape dome and
  a lamp head clutter the way; the bench top ($z = 0$) and a screen ($x \le 0.30$ m) bound the space. This is the
  course case (Course syllabus §3 Weekly schedule) "planning an instrument path in a cluttered OR" at desktop scale.
* **Defence, protective use — `eod_rig`, a scaled EOD training rig.** The same arm reaches from A over the rim of an open
  crate (three spheres) to grasp an inert mock package at D2. A real EOD manipulator plans the same kind of motion so
  that the arm never touches the crate or the suspect object except with the gripper.

In both scenes the straight joint-space line from start to goal **collides** (you will verify this), so a planner is
needed.

## The course arm and its collision model

The robot is the course arm, provided in `lab/data/course_arm.json` and loaded by `load_course_arm()` in `lab/starter.py`
(numbers identical to the shared course model in Course resources: Course robot models and hardware safety checklist): a UR-type 6R with MR Example 4.5 layout, $W_1 = 0.0634$, $W_2 = 0.0456$, $L_1 = 0.110$,
$L_2 = 0.096$, $H_1 = 0.1312$, $H_2 = 0.0751$ m, limits ±165° (joint 6 ±175°). As in Lab 9 the planner uses the
**derated** speed $\dot q_\max = 0.5\times160$ °/s $= 1.396$ rad/s, $\ddot q_\max = 4$ rad/s² on every joint and 60 % of the rated
torques, with a 0.125 kg payload at the tool point (the hardware-run limit: ≤ 50 % of the rated 0.25 kg, safety
checklist rule 4). Inertias, ratings and limits are illustrative.

The collision model is a **skeleton of 7 points** (all on joint axes at home), base $p_0 = 0$, shoulder
$p_1 = (0,0,H_1)$, elbow $p_2 = (L_1,0,H_1)$, wrist 1 $p_3 = (L_1{+}L_2,0,H_1)$, wrist-1 offset $p_4 = (L_1{+}L_2,W_1,H_1)$,
wrist 2 $p_5 = (L_1{+}L_2,W_1,H_1{-}H_2)$ and tool point $p_6 = (L_1{+}L_2,W_1{+}W_2,H_1{-}H_2)$. Point $k$ is fixed in link
$k$, so at configuration $q$ it is at $T_k(q)\,p_k$ with $T_k = e^{[\mathcal S_1]q_1}\cdots e^{[\mathcal S_k]q_k}$ (MR Eq. 4.14 for
a truncated chain). Segments $p_1p_2, \dots, p_5p_6$ are **capsules** of radius 20 mm (MR §10.2.2 uses spheres; a
capsule is a "swept sphere"). The clearance is

$$d(q) = \min\Big(\min_{\text{capsule},\ \text{sphere}}\big(\mathrm{dist}(c, [a,b]) - r - R\big),\ \min_{\text{capsule},\ \text{plane}}\big(\min(n\cdot a, n\cdot b) - \delta - r\big)\Big)$$

for spheres (centre $c$, radius $R$) and planes $\{x: n\cdot x \ge \delta\}$. $q \in \mathcal C_\text{free}$ if it is inside the joint
limits and $d(q) > 0$. Self-collision is **not** modelled (discuss it in the report).

## Learning objectives

By the end of the lab you can:

1. build a conservative collision model of a 6R arm from its PoE kinematics and compute the clearance $d(q)$;
2. check straight joint-space motions by sampling, and explain when a check is guaranteed (MR p. 365);
3. implement RRT (MR Algorithm 10.3) and RRT\* (MR p. 385) and measure success rate, tree size and path cost;
4. smooth a planned path by subdivide-and-reconnect (MR §10.8) and time-scale it within joint limits (Lab 9);
5. compare planner and smoothing choices quantitatively and report them in an IEEE-style technical report.

# Pre-lab check (complete before the session, 5 questions)

1. In MR Algorithm 10.3, which line uses the local planner, and what does a straight-line local planner return if the
   sample is 0.8 rad from the nearest node and $d = 0.3$ rad?
2. Point $p_3$ (wrist 1) is fixed in link 3. Which partial product of exponentials moves it? Why may you also use the
   product up to joint 4?
3. An edge moves joints by $\Delta q = (0.12, -0.30, 0.05, 0, 0, 0.01)$ rad and is sampled every 0.02 rad in the largest
   joint. How many configurations are checked (ends included)?
4. What does RRT\* do after it has found its first path that RRT does not?
5. A via-point path is time-scaled rest-to-rest with a trapezoid on each segment. Why can a path with fewer via points be
   faster even if it is longer?

# Setup

```bash
cd "<course folder>/461/w12/lab"
PY=~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/python   # or the course Docker image / JupyterHub
LAB_IMPL=starter $PY -m pytest -q test_lab.py        # all 33 tests fail with NotImplementedError at the start
```

Work in `starter.py` (keep the function names and signatures). Provided, do not edit: `load_course_arm`,
`load_scene`, `exp6_batch` and `poe_frames` (batched PoE), `segment_point_distance`, `within_limits`, `motion_bound`,
the Lab 9 helpers `trapezoid_params`, `trapezoid_time_scaling`, `trapezoid_from_limits`, `check_limits`, the `rnea`
wrapper (plug in your Lab 2 RNEA if you like) and the pipeline `plan_and_scale`. Every configuration is a NumPy array
of 6 joint angles in radians; functions marked "batch" accept an $(N, 6)$ array and must not loop in Python over $N$.

# Procedure and checkpoints

### Checkpoint 1 — Collision model (≈ 10 min)

1. `link_points(arm, q)`: the 7 skeleton points, for one configuration (7 × 3) or a batch (N × 7 × 3).
2. `clearance(arm, scene, q)`: signed clearance (single or batch), `config_free(arm, scene, q, margin)`.
3. `pytest -q test_lab.py -k "link_points or clearance or start_goal or config_free"`. The tests compare your tool point
   with `mr.FKinSpace`, every point with truncated-chain FK, and the clearance with an independent loop implementation.

**Checkpoint question:** in each scene, which obstacle is closest to the arm at the start and at the goal?

### Checkpoint 2 — Edge checking (≈ 5 min)

1. `edge_free(arm, scene, qa, qb, step, margin, certified)`: sample the straight motion with
   $n = \max(1, \lceil\max_i\lvert\Delta q_i\rvert/\text{step}\rceil)$ intervals, **one batch** clearance call. With
   `certified=True` require $d > \text{margin} + D/2$, where $D$ = `motion_bound(arm, Δq/n)` bounds how far any skeleton point
   moves between two samples (the Week 11 idea, MR p. 365).
2. `pytest -k edge`. One test builds an edge whose end points are free but whose middle is not.

### Checkpoint 3 — RRT (≈ 15 min)

1. `sample_config` (draw `u = rng.random()` first; goal with probability `p_goal`, else uniform in the box), `nearest`,
   `steer`, `tree_path`, `path_length`.
2. `rrt(...)` exactly as its docstring: MR Algorithm 10.3, a new node within $d$ of the goal tries a direct free edge to
   the goal; the tree is stored in preallocated arrays `nodes` (max_nodes × 6) and `parent` (−1 for the root).
   The planners sample in the scene's **task box** `scene["sample_limits"]` (elbow up, wrist roughly down). It lies inside the
   joint limits except for joint 4, whose lower bound (−3.2 rad) is just beyond −165° (−2.88 rad); collision checks
   always use the true joint limits, so motions beyond them are rejected.
3. `pytest -k "sample or nearest or tree_path or rrt_finds or failure"`. Expect well under 0.1 s per plan.

### Checkpoint 4 — RRT\* (≈ 10 min, can be finished at home)

`rrt_star(...)`: same loop, plus (1) choose the parent in $X_\text{near} = \{x: \lVert x - x_\text{new}\rVert \le r_\text{near}\}$ that
minimises cost-to-come with a free edge, (2) rewire neighbours through $x_\text{new}$ and update the costs of their
descendants, (3) add the goal once and keep going until `max_nodes`. Use $r_\text{near} = 1.0$ rad (in 6-D a radius of
0.3–0.6 rad contains almost no neighbours; try it). `pytest -k rrt_star`.

### Checkpoint 5 — Smoothing, time scaling, torques (rest of the session)

1. `shortcut_greedy` (deterministic), `point_on_path`, `shortcut_random` (MR §10.8, random test-and-replace).
2. `time_scale_path(path, qdot_max, qddot_max, dt)`: fastest rest-to-rest trapezoid on every segment (Lab 9
   `trapezoid_from_limits`), stop at every via point; `torque_profile` with the RNEA.
3. Run all tests: `LAB_IMPL=starter pytest -q test_lab.py` (33 must pass).

## Experiments for the report

Use seeds 0–19 (`np.random.default_rng(seed)`) so that your results are reproducible; defaults $d = 0.3$ rad,
$p_\text{goal} = 0.1$, edge step 0.02 rad, `max_nodes` 3000 (RRT) or 1000 (RRT\*). Save every figure with axis labels and units.

* **A — Plan, smooth, time-scale (both scenes).** For 20 seeds: success, tree size, planning time, path length of the
  raw RRT path, after `shortcut_greedy`, and after `shortcut_random` (100 iterations, applied to the raw path), and the
  motion time of each after `time_scale_path`. **Table 1:** medians and ranges. **Figure 1:** 3-D view of the scene with
  the arm at start and goal and the tool-point path before and after smoothing. **Figure 2:** $q(t)$ and $\dot q(t)$ of the
  smoothed, time-scaled motion with the speed limits. **Figure 3:** the tree projected on $(\theta_1, \theta_2)$ with the path.
* **B — RRT versus RRT\*.** Six seeds per scene: best cost versus tree size (`best_cost_history`) for RRT\* and the RRT path
  length at the same seeds. **Figure 4.** Discuss the time cost of the extra collision checks.
* **C — Sampler design.** Repeat the RRT with the sampling box equal to the full joint limits (`limits=None`); report the
  success rate within 3000 nodes. Optionally vary $p_\text{goal} \in \{0.02, 0.1, 0.5\}$. Relate to MR §10.5.1.1.
* **D — Feasibility and safety.** For the greedy-smoothed motion: worst $\lvert\dot q\rvert/\dot q_\max$ and $\lvert\tau\rvert/\tau_\max$ per joint with
  the payload (`check_limits`). Check every segment of the smoothed path with `edge_free(..., certified=True,
  margin=0.01)`. Then re-plan with every sphere radius grown by 0.01 m (a copy of the scene: "grown" obstacles,
  MR p. 365) and compare minimum clearance, path length and motion time. One paragraph: what would you change before
  running the motion on the real arm?
* **E — Optional hardware run (ungraded, with a TA present).** Export the smoothed trajectory with your Lab 9
  `export_csv` and run it on the course arm at ≤ 25 % speed (first run of a new motion) with the mock obstacles in
  place, after calibrating the servo zero offsets. Follow the hardware safety checklist in Course resources: Course
  robot models and hardware safety checklist; in particular the payload is at most 0.125 kg (the value planned
  above) and every via point must lie inside the software joint limits (±155°, J6 ±165°). The run earns no extra
  marks; you may describe it in the Discussion.

**Expected values (reference solution; TAs use them for calibration — your random numbers may differ, the medians should
be close).**

| Quantity | `bench_or` | `eod_rig` |
|---|---|---|
| Clearance at start / goal (m) | 0.0459 / 0.0443 | 0.0576 / 0.0148 |
| Minimum clearance along the straight joint-space line (m) | −0.0463 (collides) | −0.0218 (collides) |
| Tool point at start A / goal (m) | (0.234, 0.063, 0.081) / (0.002, 0.244, 0.064) | (0.234, 0.063, 0.081) / (0.175, −0.155, 0.035) |
| Straight-line joint-space length (rad) | 1.31 | 1.09 |
| RRT in the task box, 20 seeds: success; median tree size | 20/20; ≈ 30 nodes | 20/20; ≈ 34 nodes |
| Median path length: raw / greedy / random shortcut (rad) | ≈ 2.57 / 1.68 / 1.46 | ≈ 1.81 / 1.34 / 1.14 |
| Median motion time: raw / greedy / random shortcut (s) | ≈ 4.0 / 1.63 / 3.3 | ≈ 2.9 / 1.45 / 2.9 |
| RRT with the full joint-limit box: success within 3000 nodes | ≈ 80 % | ≈ 40 % |
| RRT\* (1000 nodes, $r_\text{near} = 1$) vs RRT, 6 seeds: median cost (rad) | ≈ 1.61 vs 2.75 | ≈ 1.26 vs 1.71 |
| Greedy-smoothed motion: worst $\lvert\tau\rvert/\tau_\max$; worst $\lvert\dot q\rvert/\dot q_\max$ | ≈ 0.58; 1.00 (speed-limited) | ≈ 0.55; 1.00 |

Note the surprise in the time row: random shortcutting gives the *shortest* paths but, because it inserts many new via
points (≈ 13–14 instead of 3) and the robot stops at every via point, the *slowest* motions. This is Workshop 12 Problem 5 on
the real arm.

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 11 into your cumulative library as `planning_rrt.py`. The functions you write (16):

`link_points(arm: dict, q)`, `clearance(arm: dict, scene: dict, q)`, `config_free(arm: dict, scene: dict, q, margin: float=0.0)`, `edge_free(arm: dict, scene: dict, qa, qb, step: float=0.02, margin: float=0.0, certified: bool=False)`, `sample_config(rng: np.random.Generator, limits, q_goal, p_goal: float)`, `nearest(nodes, q)`, `steer(q_from, q_to, d: float)`, `tree_path(nodes, parent, i: int)`, `path_length(path)`, `rrt(arm: dict, scene: dict, q_start, q_goal, rng: np.random.Generator, d: float=0.3, p_goal: float=0.1, max_nodes: int=3000, step: float=0.02, limits=None)`, `rrt_star(arm: dict, scene: dict, q_start, q_goal, rng: np.random.Generator, d: float=0.3, p_goal: float=0.1, max_nodes: int=1000, r_near: float=0.6, step: float=0.02, limits=None)`, `shortcut_greedy(arm: dict, scene: dict, path, step: float=0.02)`, `point_on_path(path, s: float)`, `shortcut_random(arm: dict, scene: dict, path, rng: np.random.Generator, n_iter: int=100, step: float=0.02)`, `time_scale_path(path, qdot_max, qddot_max, dt: float=0.01)`, `torque_profile(arm: dict, Q, Qd, Qdd)`.

Names shared with other labs:

* `clearance`, `config_free`, `edge_free`, `motion_bound` and `path_length` have the same names as Lab 10's functions but different signatures (`clearance(arm, scene, q)` instead of Lab 10's `clearance(q, scene)`, …). This is why Lab 11 goes into its own module `planning_rrt.py`; do not merge it into Lab 10's `planning.py`.
* `torque_profile`, `check_limits` and the trapezoid functions have Lab 9's signatures (first argument: the arm dictionary of `load_course_arm`).

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab11_<ID>.py`, passing `LAB_IMPL=lab11_<ID> pytest -q test_lab.py` (auto-checked, 33 tests). Add
   the functions to your `sbeg461_dynamics` library as `planning_rrt.py` (keep the Lab 11 names; Lab 10's `planning.py`
   has different signatures for `clearance`, `config_free`, `edge_free`, `motion_bound` and `path_length`). The project
   starter's `sbeg461_project/planning.py` implements the same algorithms under different names and signatures (it takes a
   `Scenario`; it has `rrt`, `rrt_bidirectional` and a random `shortcut`, but no `rrt_star` or `shortcut_greedy`; time
   scaling is in `traj.py`), so port or wrap your Lab 11 functions when you reuse them there.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 3** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–4 and Table 1.
4. A reflection (≤ 200 words, report appendix B) and an AI-use note (appendix A).

# Formal report 3 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Sampling-based motion
planning, smoothing and time scaling for a desktop arm in a bench-top OR and an EOD training rig*. Suggested length in
brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (capsule collision model, RRT/RRT\*, shortcutting, trapezoidal time scaling), key numbers (success rates, path lengths, motion times, torque margin), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why collision-free planning matters for an arm working beside a patient and for an EOD manipulator; why a grid planner does not scale to 6 DOF ($72^6$ cells); objectives; contributions. |
| **II. Robot and scenes** (½–1 page) | Course-arm kinematics (PoE, screw table from `lab/data/course_arm.json`), skeleton and capsule model, the two scenes (figure; state that geometry and ratings are illustrative), joint and torque limits with derating. |
| **III. Methods** (1 page) | Clearance and edge checking (sampled and certified; MR §10.2.2), RRT (MR Algorithm 10.3) with sampler, metric and local planner choices, RRT\* (choose-parent and rewire), smoothing (MR §10.8), rest-to-rest trapezoidal time scaling (MR §9.2.2), torque check with the RNEA. |
| **IV. Verification** (½ page) | Unit tests vs `modern_robotics` (tool point, RNEA), independent clearance implementation, re-checking final paths at a finer step, tree-consistency checks for RRT\*. |
| **V. Results** (1½ pages) | Table 1 and Figures 1–4 (experiments A–C), feasibility numbers (D). Every figure is referred to in the text and has a caption that states what to see. |
| **VI. Discussion** (1 page) | Which planner and smoothing would you use for (a) the OR hand-over repeated many times a day, (b) a one-off EOD approach? Length versus time; sampler design (task box) and its risk (missing solutions outside the box); limits of the model (no self-collision, capsule radius, sampled checks, static obstacles); what must change before a hardware run (margin, certified check, speed). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | MR (with sections/pages), LaValle, Corke, `modern_robotics` software; IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "no AI tools used"). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Clear problem, both applications motivated, grid-vs-sampling argument quantified, key numbers in abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Robot, scenes and collision model | 15 | Correct PoE description, skeleton/capsule model justified (conservative?), scenes and limits tabulated, illustrative values flagged | Minor slips in notation or parameters | Model described without justification | Wrong or missing |
| Methods | 15 | RRT, RRT\*, smoothing and time scaling correctly stated with MR references; parameter choices ($d$, $p_\text{goal}$, $r_\text{near}$, step) justified | One method unclear or parameters unjustified | Pseudocode copied without explanation | Wrong or missing |
| Verification evidence | 15 | All tests pass; independent checks and fine re-checks quantified; reproducible seeds | Tests pass, verification partly quantified | Some tests fail or no independent check | No verification |
| Results (figures/tables) | 20 | Table 1 and Figures 1–4 complete, labelled, with units, captions, medians and spread over seeds | One figure weak or unlabelled | Several items missing or single-seed only | Missing |
| Discussion and safety | 15 | Planner choice justified per use case; length-vs-time insight explained; model limits and hardware precautions concrete | Choices stated with partial justification | Descriptive only | Missing |
| Writing, format, references | 5 | IEEE format, ≤ 6 pages, concise, correct citations | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every result | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, that the shortest path was not the fastest motion), and why?
* Randomised planners give different answers every run. How did you make your conclusions trustworthy?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct (for example an RRT\* rewiring
  that forgot the descendants), and how did you find out it was wrong?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Tool point correct but other skeleton points wrong | Point $k$ must use the partial product up to its own link, $T_k$, not the full FK and not $T_{k-1}$ (except where the point lies on axis $k$). |
| `clearance` single and batch results differ | Broadcasting: segments are `P[:, 1:6]` and `P[:, 2:7]` (shape N × 5 × 3); the radius array has shape (5,). |
| `edge_free` is slow | Build all samples first and call `clearance` once with the (n+1) × 6 batch; do not call `config_free` in a loop. |
| RRT never reaches the goal | Sampling the whole joint box (experiment C), `p_goal` = 0, or the goal edge not tried; check `config_free(goal)`. |
| RRT\* costs inconsistent (test fails) | After rewiring node $j$, update the cost of **every** descendant of $j$ (keep a children list); remove $j$ from its old parent's children. |
| RRT\* no better than RRT | $r_\text{near}$ too small for 6-D (almost no neighbours) or the loop stops at the first solution. |
| Greedy shortcut returns the original path | You test `edge_free(path[i], path[i+1])` first; test from the far end (`j = last` downwards). |
| Time-scaled velocity exceeds the limit | Use the largest joint change of the segment (`trapezoid_from_limits`), not the Euclidean length. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside `lab/`, or set `LAB_IMPL` to your file name without `.py`. |

# Pre-lab answers (released after the session)

1. Line 5. It returns $x_\text{new}$ at distance 0.3 rad from the nearest node on the straight line towards the sample.
2. $T_3 = e^{[\mathcal S_1]q_1}e^{[\mathcal S_2]q_2}e^{[\mathcal S_3]q_3}$. $p_3$ lies on the axis of joint 4, and a rotation about an axis does
   not move points on that axis, so $T_4\,p_3 = T_3\,p_3$.
3. $n = \lceil0.30/0.02\rceil = 15$ intervals, so 16 configurations.
4. It keeps sampling, chooses each new node's parent by total cost and rewires nearby nodes through new nodes, so the
   path to the goal keeps getting cheaper (asymptotic optimality).
5. Each rest-to-rest segment pays a fixed acceleration and deceleration time; fewer stops can save more time than a
   shorter distance.

# C++ track

You may do this lab in C++17 instead of Python. The robot, scenes, checkpoints, experiments, deliverables and formal report are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: `fkin_space` for the skeleton points, `inverse_dynamics` for the torques), with an independent clearance implementation inside `test_lab.cpp`, with the numbers of the Python reference, and with the defining properties of RRT/RRT\* trees, shortcutting and trapezoidal time scaling, exactly as the Python tests do.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided functions — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random test configurations of the Python tests), `CMakeLists.txt`, `plot_results.py` (draft Figures 1–3 from CSV files). The data files are the same `data/course_arm.json` and `data/scenes.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 11 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*link_points*,*clearance*"   # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 22 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The whole test program runs in well under a second once your code is complete.

**Functions to write** (in `starter.cpp`, same order as in Python): `link_points`, `link_points_batch`, `clearance`, `clearance_batch`, `config_free`, `edge_free` (Part A); `sample_config`, `nearest`, `steer`, `tree_path`, `path_length`, `rrt`, `rrt_star` (Part B); `shortcut_greedy`, `point_on_path`, `shortcut_random`, `time_scale_path`, `torque_profile` (Part C). These are the 16 Python functions; the Python "batch" behaviour of `link_points` and `clearance` is the two separate `_batch` functions. Given in `helpers.cpp`: `load_course_arm`, `load_scene`, `exp6`, `poe_frames`, `segment_point_distance`, `within_limits`, `within_limits_batch`, `motion_bound`, the Lab 9 helpers `trapezoid_params`, `trapezoid_time_scaling`, `trapezoid_from_limits`, `check_limits`, the `rnea` wrapper (calls `mr::inverse_dynamics`; paste your Lab 2 RNEA if you like), the pipeline `plan_and_scale` and `export_results` (CSV output). The random-number class `lab::Rng` is given in `lab.hpp`.

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Collision model | `-tc="*link_points*,*clearance*,*start_goal*,*config_free*"` | 6 |
| 2 — Edge checking | `-tc="*edge_free*"` | 2 |
| 3 — RRT | `-tc="*sample*,*nearest*,*tree_path*,*rrt_finds*,*failure*"` | 5 |
| 4 — RRT\* | `-tc="*rrt_star*"` | 2 |
| 5 — Smoothing, time scaling, torques | `-tc="*shortcut*,*point_on_path*,*time_scale*,*torque*,*pipeline*"` | 7 |

All 22 test cases pass with `./build/lab_tests`. The Python track counts 33 tests because pytest counts each scene and each seed of the `@pytest.mark.parametrize` tests separately; the C++ track checks the same scenes and seeds inside one test case each.

**Random numbers.** `lab::Rng rng(seed);` replaces `np.random.default_rng(seed)`: `rng.random()` is uniform in $[0, 1)$, `rng.uniform(lo, hi)` takes two numbers or two `Eigen::VectorXd` (one draw per joint). It is built on `std::mt19937_64`, so a seed gives the same numbers with every compiler, but **not** the same numbers as NumPy: your C++ trees differ from the Python ones. The tests only check properties of the plans (as in Python), and the medians of the experiments are close to the expected-values table above (checked with the C++ reference: e.g. `bench_or` 20/20 successes, median raw/greedy/random path length ≈ 2.2 / 1.68 / 1.49 rad against 2.57 / 1.68 / 1.46 in Python, greedy motion time ≈ 1.63 s; the raw RRT path length depends most on the random numbers, the smoothed results hardly at all). The random configurations that the Python tests draw (`rand_q`, the random edges, the `nearest` nodes) are stored in `cpp_cases.json`, so both tracks test the same configurations.

**Experiments and figures for the report.** The plots stay outside the auto-checks. Write the experiments program in C++ and plot with Python: add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

For example `lab::PipelineResult out = lab::plan_and_scale(lab::load_course_arm(), lab::load_scene("bench_or"), 0, "rrt", "greedy");` followed by `lab::export_results("bench_seed0", arm, out);` writes `bench_seed0_tree.csv`, `_path.csv`, `_smooth.csv`, `_traj.csv` (t, q, qdot, tau) and `_tool.csv` in the working folder; then `python ../cpp/plot_results.py bench_seed0 bench_or` (from the `build` folder) draws drafts of Figures 1–3. For Table 1 and Figure 4 write your own CSV files with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`); `RRTStarResult::best_cost_history` holds the (tree size, cost) pairs, and `std::chrono::steady_clock` measures planning times. Experiment D: copy the scene (`lab::Scene grown = sc; grown.spheres.col(3).array() += 0.01;`) and call `edge_free(arm, sc, qa, qb, 0.02, 0.01, true)` on every segment. Deliverable 2 is then `experiments.cpp` plus your plotting script; the formal report, rubric and reflection are unchanged.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `arm` dict of `load_course_arm`; `scene` dict of `load_scene` | `lab::Arm`, `lab::Scene` (same field names: `arm.Slist`, `arm.skeleton`, `arm.skeleton_link[k]`, `arm.link_radius(k)`, `scene.spheres`, `scene.q_start`, …) |
| configuration `q` (6,); batch `Q` (N, 6); via-point path (k, 6) | `Eigen::VectorXd` (6); `Eigen::MatrixXd` with one **row** per configuration (`Q.row(k).transpose()` is a `VectorXd`) |
| `link_points(arm, Q)` and `clearance(arm, scene, Q)` with a batch | `link_points_batch` (returns `std::vector<Eigen::MatrixXd>`, each 7 × 3) and `clearance_batch` (returns `VectorXd`); a loop over the rows is fine in C++ |
| `poe_frames(arm, Q)` (N, 7, 4, 4), `exp6_batch(S, theta)` | `poe_frames(arm, q)`: 7 `Eigen::Matrix4d` for one configuration; `exp6(S, theta)` for one angle |
| `np.random.Generator` argument | `lab::Rng&` (see "Random numbers") |
| result dicts of `rrt`, `rrt_star`, `plan_and_scale` | `lab::RRTResult`, `lab::RRTStarResult`, `lab::PipelineResult`; `parent` is `std::vector<int>`; `path = None` is a 0 × 6 matrix; `first_solution_nodes = None` is −1 |
| `limits=None` | empty matrix (the default argument): use `arm.limits` |
| `point_on_path` returns `(k, p)`; `time_scale_path` returns `(t, Q, Qd, Qdd, seg_T)` | `std::pair<int, Eigen::VectorXd>` (`auto [k, p] = lab::point_on_path(path, s);`); `lab::ScaledPath` with fields `t`, `Q`, `Qd`, `Qdd`, `seg_T` |
| `np.arange(dt, T, dt)` | compute `i * dt` for `i = 1, 2, …` while below `T` (adding `dt` repeatedly accumulates round-off) |
| `np.vstack` to splice a path | the comma initialiser: `out << path.topRows(k1 + 1), p1.transpose(), p2.transpose(), path.bottomRows(m);` |
| `**kw` of `plan_and_scale` | fixed defaults ($d = 0.3$, $p_\text{goal} = 0.1$, `max_nodes` 3000 / 1000, `r_near` 0.6, the scene's `sample_limits`); call `rrt` / `rrt_star` yourself for other values |
| figures with matplotlib | CSV files (`export_results`, `sbeg::write_csv`) + `plot_results.py` |

A matrix-size mistake does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". An index out of range in a `std::vector` (for example `parent[i]` with a wrong `i`) is **not** protected and may crash the program: use `parent.at(i)` while debugging.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../course_arm.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size" in `link_points` or `clearance` | used `P.row(k)` (1 × 3) where a `Vector3d` is needed | `Eigen::Vector3d a = P.row(k).transpose();` |
| `edge_free` checks only the two end points | `k / n` with two `int`s is 0 | `static_cast<double>(k) / n` |
| `rrt` very slow or `nearest` returns nodes that do not exist yet | searched all `max_nodes` rows of the preallocated matrix | `nearest(nodes.topRows(N), q)` |
| Test program crashes (segmentation fault) in an RRT test | `parent[i]` outside the vector, or a parent index ≥ the node count | use `parent.at(i)` while debugging; return only the first `N` entries of `parent` |
| RRT\* costs inconsistent | after rewiring, the descendants' costs were not updated, or the node was not removed from its old parent's `children` | as in Python: keep `std::vector<std::vector<int>> children`; erase `j` from the old list |
| Compiles on macOS/Linux but not on Windows (strange syntax errors in `rrt_star`) | a variable named `near` or `far`: Windows headers define both as macros | use another name, e.g. `X_near` |
| Wrong results only after `auto D = qb - qa;` | `auto` keeps an Eigen expression, not a vector | write the type: `Eigen::VectorXd D = qb - qa;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
