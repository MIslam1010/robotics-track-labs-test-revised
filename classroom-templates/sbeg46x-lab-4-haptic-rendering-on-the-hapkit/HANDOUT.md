---
title: "Lab 4 — Haptic rendering on the Hapkit: virtual wall, Z-width, virtual fixtures and a needle-insertion profile"
subtitle: "SBEG 46X Medical Robotics — Week 4"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class (hardware rotation for half the class, simulation for the other half) + about 2 h of independent work |
| CLOs | **46X-3** (implement haptic rendering and virtual fixtures on a haptic device, predict and measure the stable rendering range (Z-width), model tool–tissue and needle-insertion forces); 46X-2 (review: passivity, Week 3) |
| Reading | MR §11.7 (pp. 443–447: impedance, MR Eqs. 11.62–11.63; impedance-control algorithm, MR Eq. 11.65, and the warning on p. 446 about stiff environments, delay and quantisation); MR §11.8 (pp. 447–449: current control of a geared motor); Åström–Murray §9.5, pp. 287–288 (passivity, review). Not in the course library (†): Colgate & Brown 1994 (Z-width); Abbott, Marayong & Okamura 2007 (virtual fixtures); Okamura, Simone & O'Leary 2004 (needle forces) |
| Files (your Lab code folder) | `starter.py` (your code: 18 functions marked TODO A1 … D6), `test_lab.py` (auto-checks, 22 tests), `course_models.py` (the shared `hapkit` model, numbers identical to Course resources: Course robot models and hardware safety checklist), `data/device.json`, `data/needle_tissue.json`, `data/zwidth_trials.csv` and `data/needle_log.csv` (bench logs for the simulation fallback) |
| Library | The finished functions become module `haptics` of your course library `sbeg46x_medrob` (after `rcm` and `teleop`) |
| Assessment | Auto-checked code (22 tests) + results table + three plots + reflection (one auto-checked lab in the Labs component; not a formal report) |
| AI use | Allowed for code **with disclosure** (AI-use note in the reflection); you must be able to explain every line in an oral check. Derivations are your own |

## Why this lab matters

A haptic surgical simulator, a needle-insertion trainer or a forbidden-region fixture is only useful if what the
trainee feels is **stiff where it should be stiff and never buzzes**. In SBEG 461 Lab 8 you saw that a sampled spring
acts like a spring plus a negative damper, and estimated a stability limit for the device alone. Lecture A this week
sharpened that into a **passivity** limit that holds for any passive hand — Colgate's condition $b > KT/2 + \lvert B\rvert$ —
and added the quantisation limit $f_c > K\Delta/2$. Lecture B turned walls into **virtual fixtures** (forbidden regions,
guidance, proxies) and into **tissue models** (Kelvin–Voigt, Hunt–Crossley, the stiffness–friction–cutting needle
model). Today you implement all of them, predict the largest stiffness, and then **measure** it — on the Hapkit if your
group has the hardware slot, otherwise in simulation and on the bench log. The same functions are used in project
scenarios 1 (haptic needle-insertion teleoperation), 2 (image-guided targeting with a forbidden region), 4 (EOD keep-out
region) and 5 (field telesurgery with virtual fixtures).

## The device (illustrative values)

| Quantity | Value | Source |
|---|---|---|
| Handle mass $m_h$ | 0.0570 kg (65 % of it is the reflected rotor) | `course_models.hapkit()`, SBEG 461 Week 8 |
| Physical damping $b$ | 0.15 N s/m | SBEG 461 Lab 8 reference value (Lab 2 used a rounded 0.10 N s/m); CP3 identifies yours |
| Gravity stiffness $k_g$ | 1.21 N/m | `hapkit()` (sector CoM 10 mm below the pivot) |
| Coulomb friction $f_c$ | 0.0386 N at the handle | 0.2 mN m at the motor, $Gf/r_h$ with $G = 17.38$ |
| Loop, sensor | $T = 1$ ms, $\Delta = 20$ µm | `device.json` |
| Limits | force ≤ 2 N (saturated), range ±40 mm (watchdog) | Course resources: hardware safety checklist |

Sign conventions are in the header of `starter.py`: $F_\text{motor}$ is the force the device applies **to** the hand. The
needle depth is $z = x - x_\text{skin}$ with $x_\text{skin} = -20$ mm, and the tissue force $f_\text{tissue} > 0$ resists insertion
($F_\text{motor} = -f_\text{tissue}$).

## Learning objectives

By the end of the lab you can:

1. compute the passive stiffness limit of a sampled wall from $b$, $B$, $T$ (Colgate) and from $f_c$, $\Delta$ (quantisation),
   and explain from the frequency-by-frequency form why virtual damping lowers it;
2. show by exact energy bookkeeping that a zero-order-hold wall creates $KvTd$ per push, and recover Colgate's condition;
3. measure the stable stiffness limit with a bounce test (simulation and Hapkit) and use it to identify the device damping;
4. render forbidden-region fixtures with a proxy and explain pop-through; compute an admittance-type guidance velocity;
5. implement Kelvin–Voigt and Hunt–Crossley contact and a layered stiffness–friction–cutting needle model, check that it
   can be rendered passively, render it, and detect the puncture events in a force log.

# Pre-lab check (5 questions, submit before the session)

1. The Hapkit has $b = 0.15$ N s/m and $T = 1$ ms. What is the largest passive wall stiffness with $B = 0$, and with $B = 0.05$ N s/m?
2. A wall of 500 N/m is pushed 5 mm deep and back at 50 mm/s. How much energy does the zero-order hold create, and how
   much does the device damping dissipate?
3. With $\Delta = 20$ µm and $f_c = 0.0386$ N, what is the quantisation limit on $K$? Is it binding?
4. A thin forbidden slab is rendered by pushing toward the nearest surface. What happens when the hand passes the
   mid-plane, and how does a proxy prevent it?
5. A skin model $f = 100z + 2\times10^4z^2$ N punctures at 0.8 N. Find the puncture depth and the slope there; can the
   Hapkit render it passively?

# Setup (5 min)

1. Copy your Lab code folder; work in `starter.py` or a copy with your own name (e.g. `haptics.py`).
2. Run `LAB_IMPL=starter python -m pytest -q test_lab.py`: 3 data tests pass and 19 fail with `NotImplementedError`.
   Re-run after every function; the test names tell you which function is being checked.
3. **Hardware groups only:** read the haptic-device section of Course resources: Course robot models and hardware safety
   checklist, and complete the station bench sheet with the TA. The station firmware renders the same laws as your code
   (wall, slab with proxy, needle table) at 1 kHz with the 2 N saturation and the ±40 mm watchdog built in; you only set
   parameters from the station PC, and it logs `t, x_meas, f_tissue` in the format of `data/needle_log.csv`.
   **Never disable the saturation or the watchdog; start every new setting with your hand on the handle; stop at the
   first sustained buzz.**

# Procedure

### Checkpoint 1 — Passivity limits (A1–A3, ≈ 8 min)

Implement `colgate_K_max`, `required_damping` and `quantisation_K_max`.

* E1. Tabulate $K_\max$ for $B = 0, 0.05, 0.10, 0.15$ N s/m at 1 kHz and at 4 kHz.
* E2. Plot `required_damping(K, B, T, theta)` against $\theta = \omega T \in (0, \pi]$ for $K = 200$ N/m and $B = +0.05$, $0$ and
  $-0.05$ N s/m, with the line $b = 0.15$. Where is the maximum for each sign of $B$? Explain in one sentence why a damper that
  helps at low frequency hurts at the Nyquist frequency.
* E3. Quantisation limit for the device, and for $\Delta = 0.1$ mm and $0.5$ mm.

### Checkpoint 2 — Where the energy comes from (A4, ≈ 5 min)

Implement `sampled_wall_work`. For $K = 500$ N/m, $v = 0.05$ m/s, $d = 5$ mm compare with $KvTd$ and with the damping loss
$2bvd$. Plot the net energy $KvTd - 2bvd$ against $K$ from 100 to 600 N/m: where does it cross zero? (Workshop Problem 2.)

### Checkpoint 3 — Measure the stable stiffness limit (B1–B4, ≈ 12 min)

Implement `wall_force`, `bounce_restitution`, `measure_K_limit` and `threshold_from_trials`.

* E4. Plot the restitution $e(K)$ for $K$ = 100 … 500 N/m with $B = 0$ and $B = 0.05$ N s/m. Find the crossing $e = 1$ with
  `measure_K_limit` for both. Compare with Colgate (300 and 200 N/m) and with the SBEG 461 estimate $2(b + B)/T$. Why does a
  free bounce agree with the 461 estimate and not with Colgate when $B > 0$?
* E5. **Hardware groups:** render a wall at $x = 0$ with $B = 0$. Starting at $K = 150$ N/m, raise $K$ in steps of 25 N/m; at
  each step let the handle tap the wall three times from about 5 mm away and record whether the bounce grows (and whether
  the wall buzzes when you rest the handle lightly against it). Stop at the first sustained buzz. Log $K$ and your mean
  restitution from the station log. **Everyone:** analyse `data/zwidth_trials.csv` (a bench log from a station device)
  with `threshold_from_trials`; estimate the device damping as $b \approx K_\text{thr}T/2$ and compare it with 0.15 N s/m.
* E6 (optional). Repeat E4 with `quantize=True` and `friction=True`. Why does Coulomb friction raise the bounce threshold
  so much, and why is that **not** a reason to rely on it?

### Checkpoint 4 — Virtual fixtures (C1–C4, ≈ 10 min)

Implement `penalty_slab_force`, `ProxySlab.update`, `forbidden_disc_force` and `guidance_velocity`.

* E7. Sweep the handle from −5 mm to +9 mm and back through a 6 mm slab ($K = 250$ N/m) and plot the penalty and proxy
  forces on one figure. Mark the pop-through point.
* E8. Vessel fixture (Workshop Problem 4): reproduce the leader force at $p_f = (3, 2)$ mm (motion scaling $\alpha = 0.2$, $K_l = 100$ N/m, so call
  `forbidden_disc_force` with $K = K_l/\alpha$) and plot the force field on a grid around the vessel.
* E9. Guidance: plot the tool path for $f = (1.0, 0.5)$ N held for 2 s with $c_\tau$ = 1, 0.2 and 0.
* **Hardware groups:** feel the 6 mm slab with penalty rendering at $K = 100$ N/m, push past the mid-plane, then switch to the
  proxy. Describe the difference in one sentence.

### Checkpoint 5 — Tissue and needle models (D1–D6, ≈ 15 min)

Implement `kelvin_voigt_force`, `hunt_crossley_force`, `puncture_depth`, `NeedleTissue.force`, `detect_punctures` and
`max_profile_slope`.

* E10. Plot KV and HC force against penetration for a press-and-release at ±20 mm/s (Practice P2 numbers). Mark where KV pulls.
* E11. Plot `needle_quasistatic_profile` for the needle-decompression trainer (`data/needle_tissue.json`) from 0 to 35 mm.
  Report the puncture depths, the peak forces and `max_profile_slope` of the skin phase; compare the slope with E1.
* E12. Run `render_needle(dev, p, p["x_skin"], 0.01, 4.5, -0.025)` (a hand pushing at 10 mm/s through a 400 N/m grasp).
  Overlay the rendered force on the quasi-static profile, detect the punctures (`drop` = 0.3 N) and measure the forward
  lurch of the needle in the 0.1 s after skin puncture. Compare with the quasi-static estimate of Workshop Problem 5(e).
* E13. Detect the punctures in `data/needle_log.csv` (bench log at 12 mm/s, device friction on).
* **Hardware groups:** load the needle table into the firmware, insert slowly with the handle, then repeat at a brisk
  pace. Log both runs and analyse them with your `detect_punctures`.

# Deliverables

1. Your module passing all 22 tests (`LAB_IMPL=<your module> pytest -q test_lab.py`).
2. A results table with: E1 limits; E3 limits; E4 measured limits for $B = 0$ and $0.05$ N s/m; E5 threshold and identified $b$
   (bench log, and your hardware value if you had the slot); E11 puncture depths, peaks and skin slope; E12 lurch.
3. Three plots: the restitution curves of E4; penalty vs proxy (E7); rendered vs quasi-static needle profile with the
   detected punctures (E12).
4. Reflection (≤ 200 words): (a) one sentence each on why $B$ lowers the passive limit, why quantisation matters more when
   you compensate friction, and why a proxy needs memory; (b) which result surprised you on the hardware or in the bench
   log; (c) **AI-use note**: tool, what you asked, what you kept, what you checked yourself (or "no AI used").

# Troubleshooting

| Symptom | Likely cause |
|---|---|
| `colgate_K_max` returns negative values | clip at 0 when $\lvert B\rvert \ge b$ |
| `required_damping` gives `nan` | $\theta = 0$ divides by $1 - \cos\theta = 0$; the tests use $\theta \in (0, \pi]$ |
| `sampled_wall_work` off by $KvT \cdot vT$ | the sum must run over $2N$ intervals (in and out), with the force sampled at the **start** of each interval |
| restitution exactly 0 | the handle never left the wall: check the stop condition uses the **true** position and that the wall only pushes |
| `measure_K_limit` returns `K_hi` | $e(K_\text{hi}) < 1$: widen the bracket, or you rendered the wall with the wrong sign |
| proxy force jumps when leaving the slab | update the proxy **before** computing the force; the proxy becomes the handle position only once the handle is out on the side the proxy is on |
| needle force flickers between 0 and the friction value | you used the one-sample velocity: at 20 µm and 1 kHz it only takes the values 0, 20, 40 … mm/s (Workshop Problem 3c); `render_needle` provides a 20-sample difference |
| extra punctures detected | short dips when the needle briefly moves back flip the friction sign: require the drop to **hold** (`hold` samples) |
| Hapkit buzzes at a $K$ below your prediction | your device has less damping (or more delay) than 0.15 N s/m and 1 ms: that is what E5 measures. Lower $K$, do not raise $B$ |

# Expected values (reference solution; TAs use them for calibration)

| Experiment | Quantity | Value |
|---|---|---|
| Pre-lab / E1 | $K_\max$ at 1 kHz for $B$ = 0, 0.05, 0.10, 0.15 | 300, 200, 100, 0 N/m (×4 at 4 kHz) |
| E2 | max of `required_damping` for $K = 200$, $B = 0.05$ | 0.15 N s/m, at $\theta = \pi$ |
| E3 | quantisation limit at 20 µm, 0.1 mm, 0.5 mm | 3860, 772, 154 N/m |
| CP2 | created / dissipated energy ($K = 500$ N/m) | 125 µJ / 75 µJ; net zero at $K = 300$ N/m |
| E4 | restitution at $K$ = 200, 450 N/m ($B = 0$) | 0.974, 1.021 |
| | measured limit, $B = 0$ / $B = 0.05$ | ≈ 311 N/m / ≈ 421 N/m (Colgate 300 / 200; 461 estimate 300 / 400) |
| E5 | bench-log threshold; identified $b$ | 349.1 N/m; 0.175 N s/m (the station device has more damping than nominal) |
| E8 | vessel fixture force at (3, 2) mm, $K_l = 100$ N/m, $\alpha = 0.2$ | (0.580, 0.387) N |
| E11 | skin puncture depth, slope; pleura puncture depth; peaks | 4.30 mm, 272 N/m (grid slope 271.8); 28.0 mm; 0.80 N and 0.95 N |
| E12 | events (measured depths); lurch after skin puncture | 4.32 mm and 28.02 mm; ≈ 2.27 mm (quasi-static estimate 1.37 mm) |
| E13 | bench-log punctures | at 4.3 mm and 28.0 mm, peaks 0.80 N and 0.95 N |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, hardware steps and deliverables are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`, used for the hapkit model check), with the closed-form results and independent code of `test_lab.py`, and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (the provided functions — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers), `CMakeLists.txt`. The data are read from the same `data/device.json`, `data/needle_tissue.json`, `data/zwidth_trials.csv` and `data/needle_log.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 4 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*bounce*"               # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 22 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change. The full run takes well under a second once the code is built.

**Functions to write** (in `starter.cpp`, the same 18 as in Python, same order): `colgate_K_max`, `required_damping`, `quantisation_K_max`, `sampled_wall_work` (A1–A4); `wall_force`, `bounce_restitution`, `measure_K_limit`, `threshold_from_trials` (B1–B4); `penalty_slab_force`, `ProxySlab::update`, `forbidden_disc_force`, `guidance_velocity` (C1–C4); `kelvin_voigt_force`, `hunt_crossley_force`, `puncture_depth`, `NeedleTissue::force`, `detect_punctures`, `max_profile_slope` (D1–D6). Given in `helpers.cpp` (complete, as in `starter.py`): `load_device`, `load_tissue`, `load_csv`, `rk4_step`, `simulate_device`, `hand_ramp`, `needle_quasistatic_profile`, `render_needle`, the constructors of `ProxySlab` and `NeedleTissue`, `ProxySlab::force`, and `hapkit` (the C++ copy of `course_models.hapkit()`, numbers identical), plus, C++ only, `required_damping` for a single $\theta$ (it calls your vector version). Do not call `mrcpp` inside `starter.cpp`.

**Checkpoints.**

| Checkpoint | Command (`./build/lab_tests ...`) | C++ test cases |
|---|---|---|
| 1 — Passivity limits (A1–A3) | `-tc="*device_json*,*colgate_K*,*required_damping*,*quantisation*"` | 4 |
| 2 — Where the energy comes from (A4) | `-tc="*sampled_wall_work*,*energy_balance*"` | 2 |
| 3 — Stable stiffness limit (B1–B4) | `-tc="*data_files*,*simulator_free*,*wall_force*,*bounce*,*measure_K*,*threshold_from*"` | 6 |
| 4 — Virtual fixtures (C1–C4) | `-tc="*penalty*,*proxy*,*forbidden*,*guidance*"` | 4 |
| 5 — Tissue and needle models (D1–D6) | `-tc="*kelvin*,*puncture*,*needle_model*,*profile_slope*,*render_needle*"` | 6 |

All 22 test cases pass with `./build/lab_tests` (the Python track also counts 22 tests). One difference: in Python the 3 data and simulator tests pass with the untouched starter; in C++ each of them also checks one of your functions (`test_device_json_matches_the_course_hapkit_model` → `colgate_K_max`, `test_data_files_load` → `threshold_from_trials`, `test_simulator_free_motion_matches_analytic_response` → `wall_force` with a wall outside the motion), so they pass once those functions are written. A few checks are added against the Python reference numbers: the restitution values of the bounce test ($10^{-9}$), the whole proxy force sweep, the needle force sequence and the quasi-static profile ($10^{-12}$), and the puncture indices in the bench log. The expected values above are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| `dev = load_device()`, `p = load_tissue()` (dicts), `dev["T"]`, `p["skin"]["a1"]` | `lab::Device dev = lab::load_device();`, `lab::TissueParams p = lab::load_tissue();` with the same field names: `dev.T`, `p.skin.a1`, `p.pleura.z_pl`, `p.v_eps`, `p.x_skin` |
| `load_csv(name)` returns a dict of arrays | `lab::Columns` (`std::map<std::string, Eigen::VectorXd>`): `tr.at("K")`, `tr.at("e")` |
| `law`, `f_hand`, `stop` are Python functions; `None` | `lab::ForceLaw`, `lab::StopCondition` (`std::function`): pass a lambda, e.g. `[&](int k, double x, double v) { ... }`; an empty `lab::ForceLaw()` = `None` |
| `**kw` of `bounce_restitution`, `measure_K_limit`, `render_needle` | explicit arguments `quantize`, `friction` (same defaults) |
| result dicts of `simulate_device` / `render_needle` | `lab::DeviceSim` (`t`, `x`, `v`, `x_meas`, `v_est`, `F` as `Eigen::VectorXd`, `saturated` as `std::vector<bool>`) and `lab::NeedleSim` (the same plus `z`, `f_tissue`, `events`) |
| `self.p`, `self.events.append(("skin", z))` | the public fields `p`, `events` of the object: `events.emplace_back("skin", z)`; an event is a `std::pair` (`.first` = kind, `.second` = depth) |
| `required_damping(K, B, T, theta)` with a number or an array | `Eigen::VectorXd` version (yours) and a `double` version (given); complex numbers with `std::complex<double>` |
| `forbidden_disc_force` returns `(F, q)` | `std::pair<Eigen::VectorXd, Eigen::VectorXd>` (`.first` = F, `.second` = q) |
| `detect_punctures` returns a list of indices; `np.argmax`, `F[a:b].max()` | `std::vector<int>`; `F.segment(a, b - a).maxCoeff(&j)` (first maximum, as `np.argmax`) |
| `ValueError` | `throw std::invalid_argument("...")` |
| `round(x)` (halves to even), `math.copysign`, `x ** n` | `std::nearbyint(x)`, `std::copysign`, `std::pow(x, n)` |
| plots with matplotlib (E2, E4, E7, E10–E12) | write CSV files and plot them in Python (below) |

**Plots and experiments in C++.** Write a small `experiments.cpp` with your own `main()` (add it as a second executable or build it by hand next to `starter.cpp` and `helpers.cpp`). Save each curve with `sbeg::write_csv` from `<sbeg/labio.hpp>`, e.g. for E12 `sbeg::write_csv("needle_run.csv", {"z", "f_tissue"}, M);` with `Eigen::MatrixXd M(sim.z.size(), 2); M << sim.z, sim.f_tissue;`, and plot it with a few lines of Python:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt("needle_run.csv", delimiter=",", skiprows=1)
plt.plot(d[:, 0] * 1e3, d[:, 1]); plt.xlabel("depth z (mm)"); plt.ylabel("f_tissue (N)"); plt.show()
```

The results table, the hardware steps and the reflection do not change. For your cumulative library, keep your Lab 4 functions in a `haptics.cpp`/`haptics.hpp` pair.

A size or index mistake on an Eigen vector (e.g. `F.segment(i + window, hold)` past the end) does not crash the test program: it is reported as a failed test with the message "Eigen: wrong matrix size or index out of range". An endless loop (e.g. a bisection whose interval never shrinks, or a `while` in `detect_punctures` that does not advance `i`) does hang the program: stop it with Ctrl+C.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../device.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| `bounce_restitution` always returns 0 | `touched` captured by value (`[=]`) in the stop lambda, so the change is lost | capture by reference: `[&]` |
| `sampled_wall_work` slightly off | `N` truncated with a cast instead of rounded | `static_cast<int>(std::nearbyint(2 * depth / (speed * T)))` |
| `NeedleTissue::force` gives the wrong withdrawal force or `nan` at rest | `std::abs(v) / v` (divides by 0) or an integer sign | `std::copysign(1.0, v)` for $\lvert v\rvert \ge v_\varepsilon$, `v / v_eps` below |
| `detect_punctures` misses an event or returns an index one too late | slice bounds: NumPy `F[a:b]` is `F.segment(a, b - a)`; the peak index is `i + j` | see the hint in `starter.cpp` |
| `threshold_from_trials` test fails with "threw a DIFFERENT exception" | `std::runtime_error` or `std::logic_error` thrown | throw `std::invalid_argument` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
