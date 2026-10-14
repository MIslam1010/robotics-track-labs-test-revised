---
title: "Lab 10 — Grid planner for a 2R arm among obstacles"
subtitle: "SBEG 461 Robot Kinematics and Dynamics — Week 11"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class + up to 2 h to finish (due before the Week 12 lab) |
| CLOs | 461-5 (formulate motion planning in C-space; implement and compare graph-search planners); 461-4 (the path is the input of your Week 10 time scaling) |
| Files | `lab/starter.py` (you edit), `lab/test_lab.py` (auto-checks), `lab/data/scenes.json` (the two scenes); a reference solution is released after the deadline |
| Robot | The course planar 2R, $L = (0.6, 0.4)$ m: `planar_2r` of the shared robot models (MR §4.1; MR Studio "Planar 2R arm"), with the links modelled as capsules |
| Reference | `modern_robotics.FKinSpace` (FK check); brute-force dense sampling (collision check); `scipy.sparse.csgraph.shortest_path` (search check) — used **only in the tests** |
| Reading | MR §10.2 (printed pp. 360–370) and §10.4 (pp. 371–375); MR Exercises 10.12–10.13 (pp. 402–403), which this lab implements; LaValle §2.2.2 (pp. 35–38) |
| Assessment | Auto-checked (`test_lab.py`) + experiment table + 3 plots + short reflection (not a formal report; Lab 11 in Week 12 is formal report 3 and reuses this code) |
| AI use | Allowed with disclosure (see Reflection) |
| Tier | 0 (simulation only) |

The functions you write today go into your `sbeg461_dynamics` library as the module `planning.py`. Lab 11 (RRT for the course arm, Week 12) reuses the capsule/clearance idea and `motion_bound`; its code (Week 12 `lab/starter.py`, submitted as `lab11_<ID>.py`) goes into the library as a separate module `planning_rrt.py`, because its `clearance`, `config_free`, `edge_free`, `motion_bound` and `path_length` have different signatures from today's (the first four take the arm as their first argument). The project starter (`sbeg461_project/planning.py`) uses the same capsule model.

**Application context.**
*Biomedical — scene `ward`:* a 2R patient-positioning arm beside a treatment couch, seen from above, must move a probe holder from its parking pose to a pose next to the patient's head. On the way it must not touch the head, an IV pole or a monitor cart. Its joints turn continuously, so the C-space is a torus.
*Defence (protective) — scene `eod`:* the 2R arm of an EOD (bomb-disposal) UGV, seen from the side, moves from its stowed pose to an inspection pose with the camera/gripper under a vehicle bumper. It must avoid the ground, the bumper and an overhanging pipe. Its joints have cable stops at $\pm180^\circ$, so the C-space is a square, not a torus.
Both scenes use **illustrative values**; neither models a specific product.

# Learning objectives

By the end of the lab you can:

1. Compute the clearance $d(q, \mathcal B)$ between a capsule-model arm and disc/half-plane obstacles (MR §10.2.2), and build the C-space occupancy grid of a 2R arm (MR Fig. 10.2, Exercise 10.12).
2. Explain why a sampled edge check can miss a collision, and implement a **certified** ("grown robot") edge check that cannot (MR pp. 365–366).
3. Implement A* exactly as MR Algorithm 10.1, plus Dijkstra and inflated A* as special cases, and a reopening option for inconsistent heuristics.
4. Choose grid connectivity and heuristics so that the heuristic is admissible and consistent, and measure the effect on the number of expanded nodes.
5. Plan collision-free joint paths for both scenes, and explain the role of C-space topology (wrap-around vs joint stops) and of grid resolution (resolution completeness, MR p. 372).

# Pre-lab check (complete before the session; 5 questions)

1. A disc obstacle of radius 0.10 m has its centre 0.30 m from the axis of a straight link modelled as a capsule of radius 0.03 m. What is the signed distance $d$? What does $d < 0$ mean?
2. On a $k\times k$ grid over $[-\pi, \pi)^2$, what is the grid step for $k = 72$ in degrees? What does "wrap-around" mean for the cells $i = 0$ and $i = k - 1$, and when is it physically correct?
3. With 8-connected moves (costs $\Delta$ and $\sqrt2\Delta$), is the Manhattan distance an admissible heuristic? Is the Euclidean distance?
4. In MR Algorithm 10.1, when is the goal test made: when the goal is put into OPEN, or when it is removed from OPEN? Why does it matter?
5. A straight joint-space step $\Delta q = (2^\circ, 1^\circ)$ of the 2R arm with $L = (0.6, 0.4)$ m: what is the bound $D = (L_1 + L_2)|\Delta\theta_1| + L_2|\Delta\theta_2|$ on how far any point of the arm moves?

*Answers (for self-check):* 1. $d = 0.30 - 0.03 - 0.10 = 0.17$ m; $d < 0$ is penetration (MR p. 364). 2. $360^\circ/72 = 5^\circ$. With wrap-around, cell $k-1$ ($\theta = 175^\circ$) is a neighbour of cell 0 ($-180^\circ$). This is correct only if the joint can turn continuously; a joint with cable stops has no such edge. 3. No: one diagonal step costs $\sqrt2\Delta$, but Manhattan says $2\Delta$, an overestimate. Euclidean is admissible (and consistent), and octile is tighter still. 4. When it is **removed** (MR line 6). Only then is its past cost guaranteed minimal; the first entry of G in OPEN can be via a longer path (Workshop 11, Problems 3 and 5). 5. $D = 1.0(0.0349) + 0.4(0.0175) = 0.0419$ m.

# Setup

```bash
cd "<your Lab code folder>"   # the Lab 10 code folder (or your Classroom 50 repo)
python -c "import numpy, scipy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py   # 21 failed + 10 errors (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`.

Rules: inside `starter.py` do **not** import `modern_robotics`, SciPy, `networkx` or any planning library; the tests use them as ground truth. Use `heapq` for OPEN. The provided helpers are `load_scene`, `wrap_angle`, `grid_angles` and `cell_to_q`. Run the tests after each step. A checkpoint is complete when its tests pass; ask a TA to sign it off.

# Procedure

## Step 1 — Geometry and collision checking (10 min)

Implement `fk_2r`, `point_segment_distance`, `capsule_circle_distance`, `clearance` and `config_free`. Each link is a capsule: the segment between two joint points, grown by `link_radius`. The clearance is the smallest signed distance from either capsule to any disc, and to the floor half-plane $y\le$ `floor` if the scene has one (for a capsule: $\min(a_y, b_y) - r - \text{floor}$). This is the distance $d(q, \mathcal B)$ of MR §10.2.2 with the exact geometry instead of spheres.

**Checkpoint 1:** `test_fk_matches_modern_robotics`, `test_point_segment_distance`, `test_capsule_circle_distance_sign`, `test_clearance_hand_values`, `test_clearance_matches_brute_force`, `test_config_free_joint_limits`, `test_start_and_goal_are_free`. Expected: `clearance((0, 0), ward) = 0.15` m (patient head) and `clearance((0, 0), eod) = -0.02` m (the straight arm hits the bumper).

## Step 2 — The C-space grid and edge checks (15 min)

1. Implement `q_to_cell` (nearest cell, with wrap) and `cspace_grid(scene, k)`. Plot both grids for $k = 72$ next to the workspace, with $\theta_1$ on the horizontal axis (as in MR Fig. 10.2):

    ```python
    import numpy as np, matplotlib.pyplot as plt, starter as s
    sc = s.load_scene("ward"); free = s.cspace_grid(sc, 72)
    plt.imshow(~free.T, origin="lower", extent=[-180, 180, -180, 180], cmap="Greys")
    plt.xlabel("theta1 (deg)"); plt.ylabel("theta2 (deg)"); plt.show()
    ```

    Expected free fractions: ward 0.868, eod 0.541. Identify each C-obstacle. Which ones are full-height bands, and why (Workshop 11, Problem 2)? In the ward grid, the IV pole gives a band at $\theta_1\approx118.6^\circ$.
2. Implement `joint_step`, `edge_free` (sampled) and `motion_bound`, then `edge_free_certified`. Every sample must have clearance $\ge D/2$, where $D$ is the motion bound of one sample spacing (Practice 11, Problem 3).

**Checkpoint 2:** `test_grid_angles_and_cells`, `test_cspace_grid_matches_brute_force`, `test_link1_obstacle_is_a_full_band`, `test_motion_bound_is_an_upper_bound`, `test_sampled_check_can_miss_but_certified_cannot`, `test_certified_true_implies_dense_free`, `test_edge_free_wraps_the_short_way`.

## Step 3 — A*, Dijkstra and the grid (15 min)

1. Implement `astar` from MR Algorithm 10.1 with `heapq` and lazy deletion, as described in its docstring. Implement `dijkstra` as a one-line call.
2. Implement `grid_neighbors` (4/8-connected, wrap or not), `grid_heuristic` (zero, Manhattan, Euclidean, octile) and `grid_astar`.
3. Reproduce Workshop 11: Problem 3 (A*: 4 expansions, Dijkstra: 7) and Problem 5 (no reopening: S–B–G, cost 6; reopening: S–A–B–G, cost 5).

**Checkpoint 3:** `test_astar_workshop_problem_3`, `test_inconsistent_heuristic_needs_reopening`, `test_astar_failure`, `test_grid_neighbors`, `test_heuristics_admissible_and_consistent`, `test_grid_astar_optimal_vs_scipy`, `test_inflated_astar_bound`, `test_topology_wrap_matters`, `test_enclosed_goal_reports_failure`.

## Step 4 — The planner (10 min in class, finish at home)

Implement `unwrap_path`, `path_length` and `plan_2r` (docstring steps 1–4). Plan both scenes with $k = 72$, 8-connected, Euclidean heuristic and certified edges. Plot each path on the C-space grid and as a sequence of arm poses in the workspace (every 4th waypoint). Expected grid costs: ward 5.195 rad, eod 4.301 rad. The ward path crosses $\theta_1 = \pm180^\circ$: explain why in your report.

**Checkpoint 4:** `test_plan_2r[ward]`, `test_plan_2r[eod]`, `test_unwrap_and_length`. Final run: `LAB_IMPL=starter python -m pytest -q test_lab.py` → 31 passed (about 10 s).

## Step 5 — Experiments (at home)

Fill in the table for $k = 72$ with `grid_astar` (no edge check). It uses the start and goal cells of each scene; record the cost (rad) and the number of expanded nodes:

| Scene | 4-conn, $h = 0$ | 4-conn, Manhattan | 8-conn, $h = 0$ | 8-conn, Euclidean | 8-conn, octile | 8-conn, Euclidean, $\eta = 2$ | 8-conn, Euclidean, $\eta = 5$ |
|---|---|---|---|---|---|---|---|
| ward | | | | | | | |
| eod | | | | | | | |

Indicative values (instructor's code; your expansion counts can differ slightly with your tie-breaking): ward 6.3705/3429, 6.3705/1040, 5.1436/3309, 5.1436/1221, 5.1436/939, 5.1436/227, 5.2159/105; eod 5.3233/1685, 5.3233/86, 4.3009/1565, 4.3009/564, 4.3009/411, 4.4455/242, 4.6623/282.

Then answer, in at most 2–3 sentences each:

1. Why do all admissible heuristics give the same cost? Why does the octile heuristic expand fewer nodes than the Euclidean one?
2. With $\eta = 2$ and $\eta = 5$, is the cost within $\eta\times$ the optimum? In the eod scene, $\eta = 5$ expands *more* nodes than $\eta = 2$. How can a greedier search expand more?
3. Run the ward scene with `wrap=False` (`dict(sc, wrap=False)`). What happens, and which C-obstacle is responsible? What does this mean for a patient-positioning arm whose cables do not allow continuous rotation?
4. Repeat the 8-connected Euclidean search for $k = 36$ and $k = 144$. Report cost, expansions and the time of `cspace_grid`. Indicative: ward 5.0563 / 5.1436 / 5.1074 rad. Why does the cost not decrease monotonically with $k$, and why can a coarse grid even be **cheaper**? (Hint: snapping, and what the grid cannot see between cells.)
5. Compare `plan_2r(..., certified=True)` with `certified=False` (ward: 5.1947 vs 5.1436 rad). Which one would you put on a robot next to a patient, and why?

## Extension (optional, for the project)

Time-scale the ward path with your Lab 9 trajectory library: use the waypoints as via points with a joint speed limit of 1 rad/s, and check the torques of the `planar_2r` model (which works in a vertical plane with gravity). Remove redundant waypoints first (straight-line shortcutting; MR §10.8 is in Week 12).

<!-- library-block:begin -->
# Your `sbeg461_dynamics` library

When the tests pass, copy the functions you wrote in Lab 10 into your cumulative library as `planning.py`. The functions you write (19):

`fk_2r(q, L=L_PLANAR_2R)`, `point_segment_distance(p, a, b)`, `capsule_circle_distance(a, b, r_link, c, r_obs)`, `clearance(q, scene)`, `config_free(q, scene, margin: float=0.0)`, `q_to_cell(q, k: int)`, `cspace_grid(scene, k: int)`, `joint_step(qa, qb, wrap: bool=True)`, `edge_free(qa, qb, scene, step: float=math.radians(1.0), wrap: bool=True)`, `motion_bound(dq, L=L_PLANAR_2R)`, `edge_free_certified(qa, qb, scene, step: float=math.radians(1.0), wrap: bool=True)`, `astar(start, goal, neighbors, h, eta: float=1.0, reopen: bool=False)`, `dijkstra(start, goal, neighbors)`, `grid_neighbors(cell, k: int, connectivity: int=8, wrap: bool=True, free=None)`, `grid_heuristic(cell, goal, k: int, kind: str='euclidean', wrap: bool=True)`, `grid_astar(free, start, goal, connectivity: int=8, heuristic: str='euclidean', eta: float=1.0, wrap: bool=True, edge_ok=None)`, `unwrap_path(qs)`, `path_length(qs)`, `plan_2r(scene, k: int=72, connectivity: int=8, heuristic: str='euclidean', eta: float=1.0, certified: bool=True)`.

Names shared with other labs:

* Lab 11 (Week 12) reuses the names `clearance`, `config_free`, `edge_free`, `motion_bound` and `path_length` with different signatures (the first four take the arm as their first argument). Lab 11's code therefore goes into a separate module, `planning_rrt.py`.

The provided helpers at the top of `starter.py` (loaders, plotting, simulators) do not have to go into the library. The full list of library functions of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables (upload to the course site)

1. `starter.py` with all tests passing.
2. The Step 5 table and your answers (≤ 400 words in total).
3. Three plots with labelled axes and units: (a) both C-space grids with start, goal and planned path; (b) the ward arm poses along the path in the workspace; (c) expansions vs heuristic (bar chart) for one scene. Use a colour-blind-safe palette, and make sure the plots are readable on a phone.
4. Reflection (below), ≤ 200 words.

# Reflection (≤ 200 words)

- Your grid planner is resolution complete and, with an admissible heuristic, optimal *on the grid*. Give one way in which the planned path can still be unsafe or poor for the real arm. Think of the snapping of start and goal, thin obstacles, the model error of the capsules, and the dynamics.
- Grid search is impractical for the 6-DOF course arm (Practice 11, Problem 4). Which parts of today's code will you reuse for the RRT of Lab 11, and which will you drop?
- **AI-use note:** state whether you used an AI assistant, for which step, what you asked, and how you verified the answer (tests, hand trace of Workshop 11, plots). Code that you cannot explain in the lab practical (Week 13) will not earn credit.

# Project link (Week 11)

Your team was formed in Week 7 (project brief, §3 and §5). Two things to do this week:

1. Check this week's milestones in the brief (§5) and the hardware-run rules (§5.1–5.3).
2. Decide which motions of your scenario need a planner. Scenario 3 (surgical arm) and Scenario 5 (EOD) teams will plan around obstacles in 6-D, where a grid is not an option (Practice 11, Problem 4).

Link to today's lab: the starter's `planning.py` checks collisions with the same capsule model (segments with a radius against spheres, boxes and a ground plane) and plans with an RRT, which you will study in Week 12.

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| `test_clearance_hand_values` gives 0.18 instead of 0.15 | link radius not subtracted (or obstacle radius subtracted twice) |
| Grid looks mirrored or transposed compared with MR Fig. 10.2 | `free[i1, i2]` has joint 1 as the row index; plot `free.T` with `origin="lower"` |
| A* cost larger than the scipy reference | goal test made when the goal is pushed instead of popped; or a stale heap entry was expanded (missing lazy-deletion check); or Manhattan heuristic used with 8-connectivity |
| A* cost correct but expansions ≫ Dijkstra's | heuristic not in the same units as the costs (cells vs radians: multiply by the grid step) |
| `test_inconsistent_heuristic_needs_reopening` fails only for `reopen=False` | you reopen CLOSED nodes by default; MR Algorithm 10.1 skips them |
| Ward planning fails with `wrap=True` | `grid_neighbors` does not take indices modulo $k$, or `joint_step` does not take the short way round |
| Planned ward path jumps by 360° in the plot | the path was not unwrapped (`unwrap_path`); plot `wrap_angle(q_path)` on the C-space grid instead |
| `plan_2r` very slow (> 30 s) | `cspace_grid` recomputed inside the search, or the certified check uses a tiny step; use step = grid step / 4 |

# C++ track

You may do this lab in C++17 instead of Python. The functions, scenes, checkpoints, experiments and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with brute-force dense sampling of the link capsules, and with the numbers of the Python reference (including the SciPy shortest-path costs), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers, including the random inputs that `test_lab.py` draws with NumPy and the SciPy cost-to-go tables), `CMakeLists.txt`. The scenes are read from the same `data/scenes.json`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 10 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*astar*"                # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 25 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The full run takes well under a second once the code is built.

**Functions to write** (in `starter.cpp`, the same 19 as in Python, same order): `fk_2r`, `point_segment_distance`, `capsule_circle_distance`, `clearance`, `config_free` (Step 1); `q_to_cell`, `cspace_grid`, `joint_step`, `edge_free`, `motion_bound`, `edge_free_certified` (Step 2); `astar`, `dijkstra`, `grid_neighbors`, `grid_heuristic`, `grid_astar` (Step 3); `unwrap_path`, `path_length`, `plan_2r` (Step 4). Given (complete, as in `starter.py`): `load_scene`, `wrap_angle` (a number or a joint vector), `grid_angles`, `cell_to_q`, the constants `L_PLANAR_2R`, `SLIST_PLANAR_2R`, `M_PLANAR_2R`, `PI`, `ONE_DEG`, and, C++ only, three small helpers at the top of `starter.cpp`: `py_mod` (Python's `%` for angles), `py_imod` (Python's `%` for indices, never negative) and `lin01(i, n)` (= `np.linspace(0, 1, n + 1)[i]`). Do not use `mrcpp` or any graph or planning library inside `starter.cpp` — the tests use them as ground truth. Use `std::priority_queue` for OPEN (hint in `starter.cpp`).

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Geometry and collision checking | `-tc="*fk_*,*point_segment*,*capsule*,*clearance*,*config_free*,*start_and_goal*"` | 7 |
| 2 — C-space grid and edge checks | `-tc="*grid_angles*,*cspace*,*link1*,*motion_bound*,*sampled*,*certified*,*wraps*"` | 7 |
| 3 — A*, Dijkstra and the grid | `-tc="*astar*,*inconsistent*,*grid_neighbors*,*heuristics*,*topology*,*enclosed*"` | 9 |
| 4 — The planner | `-tc="*plan_2r*,*unwrap*"` | 2 |

All 25 test cases pass with `./build/lab_tests`. The Python track counts 31 tests because pytest counts each parameter set separately (6 items of `test_grid_astar_optimal_vs_scipy`, 2 of `test_plan_2r`); the C++ track checks the same parameter sets in one test case each. The expected values in the procedure above (clearances, free fractions, the Workshop 11 searches, the grid costs ward 5.195 rad and eod 4.301 rad) are the same in both tracks. One check is added: `test_cspace_grid_matches_brute_force` also compares your whole $36\times36$ grids with the reference grids.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| scene `dict` from `load_scene`; `floor` is `None` or a number | `lab::Scene` with the same field names (`L`, `link_radius`, `circles` $m\times3$, `floor`, `wrap`, `q_start`, `q_goal`); `floor` is a `std::optional<double>` (`if (scene.floor) ... *scene.floor`). For a hand-made scene, set the fields of a `lab::Scene` |
| `fk_2r` returns `(p0, p1, p2)` | `lab::ArmPoints` with fields `p0`, `p1`, `p2` (`Eigen::Vector2d`) |
| grid cell `(i1, i2)`; `free`: $k\times k$ boolean array, `free[i1, i2]` | `lab::Cell` = `std::pair<int, int>` (`.first` = joint 1); `lab::BoolGrid` (`Eigen::Array<bool, Dynamic, Dynamic>`), `free(i1, i2)`, `free.count()`, `free.row(i).any()` |
| `astar` nodes: any hashable (letters, cells); `neighbors`, `h`: Python functions | nodes are non-negative `int`s (the tests use the character codes `'S'`, `'A'`, …; `grid_astar` numbers cell $(i, j)$ as $ik + j$); `lab::Neighbors` and `lab::Heuristic` are `std::function`s — pass a lambda |
| `heapq` with entries `(f, h, counter, node)` | `std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>` with `Entry = std::tuple<double, double, long long, int>`: the same order, so ties are broken exactly as in Python |
| result `dict` of `astar`; `path` `None` and `cost` `math.inf` on failure | `lab::SearchResult` (`success`, `path`, `cost`, `expanded`, `parent`, `past_cost`); empty `path`, `cost` = `std::numeric_limits<double>::infinity()`, `parent[start] = -1` |
| `grid_astar` returns the `astar` dict with cells | `lab::GridSearchResult` (`success`, `path` as cells, `cost`, `expanded`; no `parent`/`past_cost`) |
| `free=None`, `edge_ok=None` | leave the argument out (empty `BoolGrid`, empty `std::function`); test with `free.size() > 0`, and call `edge_ok` only when `if (edge_ok)` is true (the hint in `starter.cpp` shows the line) |
| `plan_2r` returns a dict; `q_path` $N\times2$ or `None` | `lab::PlanResult` with the same field names; `q_path` is an `Eigen::MatrixXd` with $N$ rows ($0\times2$ on failure) |
| `math.radians(1.0)`, `np.pi`, `np.rint`, `%` | `lab::ONE_DEG`, `lab::PI` (`M_PI` is not standard C++), `std::nearbyint` (rounds halves to even, like `np.rint`), `py_mod`/`py_imod` (C++'s `%` is negative for negative numbers and is not defined for `double`) |
| plots with matplotlib (Steps 2, 4 and 5) | write CSV files and plot them in Python (below) |

**Plots and experiments in C++.** Write a small `experiments.cpp` with your own `main()` (add it as a second executable or build it by hand next to `starter.cpp`). Save the grid as a 0/1 matrix and the path with `sbeg::write_csv` from `<sbeg/labio.hpp>`, e.g. `sbeg::write_csv("grid_ward.csv", {}, free.cast<double>().matrix());` and `sbeg::write_csv("path_ward.csv", {"theta1", "theta2"}, res.q_path);`. Then plot them with a few lines of Python:

```python
import numpy as np, matplotlib.pyplot as plt
free = np.loadtxt("grid_ward.csv", delimiter=",") > 0.5
q = np.degrees(np.loadtxt("path_ward.csv", delimiter=",", skiprows=1))
plt.imshow(~free.T, origin="lower", extent=[-180, 180, -180, 180], cmap="Greys")
plt.plot((q[:, 0] + 180) % 360 - 180, (q[:, 1] + 180) % 360 - 180, ".")
plt.xlabel("theta1 (deg)"); plt.ylabel("theta2 (deg)"); plt.show()
```

The Step 5 table, the `wrap = false` run (copy the scene and set `sc.wrap = false`) and the timings (`std::chrono::steady_clock`) are done in the same program; the expansion counts match the indicative values of Step 5 because the tie-breaking is the same. For your cumulative library, keep your Lab 10 functions in a `planning.cpp`/`planning.hpp` pair (Lab 11 uses the same names with other signatures, so keep its code in a separate file, as in Python). The report requirements do not change.

A size or index mistake (e.g. `free(k, 0)` on a $k\times k$ grid) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". An infinite loop in `astar` (e.g. a node that is never closed) does hang the program: stop it with Ctrl+C.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../scenes.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `grid_neighbors` with `wrap = true` returns cells like `(-1, 0)`, then "index out of range" | C++'s `-1 % k` is `-1` | use `py_imod(i, k)` |
| `q_to_cell` is off by one cell at some angles | `std::round` (halves away from zero) or a cast to `int` (truncation) instead of `std::nearbyint` | `static_cast<int>(std::nearbyint(x))` |
| A* finds the right cost but `expanded` differs from the Workshop 11 values (4 and 7) | `std::priority_queue` is a max-heap by default, or the entry order differs from `(f, h, counter, node)` | use `std::greater<Entry>` and the `Entry` tuple from the hint |
| `test_astar_workshop_problem_3` throws `map::at` | your `astar` asked for the neighbours or heuristic of a node that is not in the graph | only expand nodes that were pushed on OPEN |
| A* returns a too-long path | goal test when the goal is pushed instead of popped, or a stale entry expanded (missing lazy-deletion check) | see the Troubleshooting table of the Python track |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
