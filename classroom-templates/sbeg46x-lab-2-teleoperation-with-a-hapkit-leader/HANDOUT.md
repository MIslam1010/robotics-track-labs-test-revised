---
title: "Lab 2 — Teleoperation with a Hapkit leader: scaling, clutch, tremor filter and two-port models"
subtitle: "SBEG 46X Medical Robotics — Week 2"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 (50 min): 10 min hardware refresher for everyone, then a rotation between the Hapkit stations (recording your own leader motion, TA present) and the simulation bench. Finish at home (1–3 h, due before the Week 3 lab). |
| CLOs | **46X-2** (design and analyse unilateral and bilateral teleoperation: motion scaling, clutching, transparency, stability–transparency trade-off) |
| Files (your Lab code folder) | `starter.py` (you edit; it becomes `sbeg46x_medrob/teleop.py`), `test_lab.py` (auto-checks), `course_models.py` (the lab's own copy of the Hapkit and EOD-arm models; numbers identical to Course resources: *Course robot models and hardware safety checklist*), `data/teleop_params.json`, `data/leader_log.csv` (a **synthetic** Hapkit leader recording, the simulation fallback) and `data/make_leader_log.py` (how it was made) |
| Reference | SciPy (`signal.butter`, `signal.lfilter`), `modern_robotics` (`JacobianBody`), closed-form results of Workshop 2 |
| Reading | MR §11.7, printed pp. 443–447 (impedance $Z(s)$ and admittance, Eqs. 11.62–11.66, the haptic interface of Fig. 11.23); Workshop 2; † Lawrence (1993) and † Hokayem & Spong (2006) on the course site |
| Assessment | Auto-checked (`test_lab.py`, 25 tests) + a short results table, three figures and the reflection. One auto-checked lab in the Labs component (Course syllabus §4); not a formal report (formal report 1 is Lab 3). |
| AI use | Allowed with disclosure for the code (see Reflection). You must be able to explain any line in a 2–3-minute oral check. |
| Tier | Tier 1 (Hapkit leader, motor output **off** for recording) for students who attend the refresher; Tier 0 (synthetic log, simulated follower) for everyone. Every part can be completed in Tier 0 without penalty. |

**Application context.**
*Biomedical:* teleoperated laparoscopy. The surgeon's handle motion is scaled down ($\alpha = 0.25$), tremor is
filtered out, and the clutch lets a small handle workspace drive a 40 mm instrument insertion. Force feedback lets the
surgeon feel tissue; you will see how much of the tissue's stiffness actually reaches the hand.
*Defence (protective, humanitarian):* the leader drives the approach axis of the UGV-mounted EOD arm
(`eod_6r_ugv`) towards a suspect object with the motion scaled **up** ($\alpha = 4$). The follower speed limit comes
from the arm's Jacobian and joint-speed rating. All device, tissue and robot values are **illustrative values**.

# Learning objectives

By the end of the lab you can:

1. Compute the effective mass of a Hapkit-type device at its handle and apply its two safety limits (2 N force
   saturation, ±40 mm watchdog) in software.
2. Implement a leader–follower mapping with motion scaling, clutching (indexing) without jumps, and a follower rate
   limit.
3. Design, implement and measure a tremor filter (first order and second-order Butterworth): attenuation in the
   8–12 Hz band versus delay of the voluntary motion.
4. Compute the hybrid matrix and the transmitted impedance of P–P and P–F architectures, and apply Llewellyn's
   absolute-stability test.
5. Simulate bilateral P–P and P–F teleoperation, measure transparency in free space and in contact, and find the
   largest stable force scale — and explain why the simulation with a one-sided wall becomes unstable earlier.
6. Derive a follower speed limit for an EOD arm from its Jacobian and map it to a hand-speed limit.

# Pre-lab check (complete before the session; 5 questions)

1. A Hapkit's handle is 0.090 m from the pivot. Its pivot inertia (sector, handle and reflected motor rotor) is
   $4.62\times10^{-4}$ kg m². What effective mass does the operator feel at the handle?
2. With $\alpha = 0.25$ and a 60 mm handle stroke, how many strokes does a 40 mm insertion need? What must the
   software do at each clutch release so that the instrument does not jump?
3. A first-order low-pass at $f_c = 2$ Hz: what fraction of a 9 Hz tremor passes, and about how much delay does it add
   to slow voluntary motion?
4. Write the hybrid matrix of P–F with scales $\alpha$, $\beta$ (Workshop 2, Problem 3). What does the operator feel in
   free space?
5. Give the two haptic-device safety limits and say what the watchdog does. May you disable the watchdog to emulate
   a communication delay?

*Answers (for self-check):* 1. $m = I/r_h^2 = 4.62\times10^{-4}/0.0081 = 0.057$ kg. 2. $40/(0.25\times60) = 2.67$, so
3 strokes. At each release, latch both offsets: $x_{s,d} = x_{s,\text{latch}} + \alpha(x_m - x_{m,\text{latch}})$ with
$x_{m,\text{latch}}$ = the handle position at the release and $x_{s,\text{latch}}$ = the held follower set-point.
3. $1/\sqrt{1 + 4.5^2} = 0.217$ (22 %); delay $\approx 1/(2\pi f_c) = 80$ ms. 4.
$H = \begin{bmatrix} Z_m & \beta\\ -\alpha C/(Z_s + C) & 1/(Z_s + C)\end{bmatrix}$; free space: $Z_t = h_{11} = Z_m$,
the leader alone. 5. Handle force saturated at 2 N (in the firmware) and a watchdog that sets the motor output to zero
if the handle leaves ±40 mm or the control loop overruns. **Never**: delay is always emulated in software, never by
disabling a watchdog or a force limit.

# Setup

```bash
cd "<your Lab code folder>"            # or your Classroom 50 repository
python -c "import numpy, scipy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py   # expected now: 25 failed (NotImplementedError)
```

On Windows (PowerShell): `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. The provided parts of
`starter.py` are `load_params`, `load_log`, `minjerk`, `reach_intent`, `simulate_teleop` (hand + leader + follower +
environment, 1 kHz controller with zero-order hold, Hapkit saturation and watchdog) and `jacobian_body`. You write the
19 functions marked TODO. When you finish, copy `starter.py` into your library as `sbeg46x_medrob/teleop.py`; Labs 3
and 4 import it.

# Session plan (50 min)

| Time | Everyone | Hapkit stations (TA present, one pair at a time, about 5 min per pair) |
|---|---|---|
| 0–10 | **Hardware refresher** (below), then sign the attendance list | TA demonstration: watchdog trip, force saturation, emergency unplug |
| 10–50 | Simulation bench: Parts A and B on the synthetic log, then C and D | Record your own 12 s leader log (motor output off); run Parts A–B on it |
| home | Parts C–E, figures, results table, reflection | — |

## Hardware refresher (first 10 minutes)

You completed the course-arm and haptic-device inductions in SBEG 461. Today's refresher covers the haptic-device
rules and how they apply to teleoperation. Only students who attended the SBEG 461 inductions **and** today's refresher
may touch the Hapkits; everyone else uses the synthetic log (no penalty). The course arm is not used today; its
checklist is in Course resources: *Course robot models and hardware safety checklist*.

| Rule | What it means today | Why |
|---|---|---|
| Force saturation | The commanded handle force is saturated at **2 N** in the firmware; your code saturates it too (`saturate_force`) | The firmware limit acts even if your code is wrong |
| Watchdog | Motor output set to zero if the handle leaves **±40 mm** or the control loop overruns; your simulation does the same (`watchdog_ok`) | An unstable loop drives the handle to the end of its range |
| Only the operator's hand | Only the operator's hand on the handle; nobody else touches the device while it is powered | A second hand changes the hand impedance and hides problems |
| Delay | Communication delay (Week 3) is **always emulated in software**, never by disabling a watchdog or a force limit | The limits are the last safety layer |
| Follower = leader limits | A simulated or second-Hapkit follower obeys the same 2 N and ±40 mm limits | A teleoperated follower is a haptic device too |
| Supervision and logging | A TA is present; every recording is logged (CSV) and faults are noted with time, device number and symptom | Faults that are not written down cannot be fixed |
| Stop | Know the stop: release the handle and unplug the USB/power lead; the motor output is then zero | |

## Recording your leader log (Hapkit stations)

With the motor output **disabled** (watchdog on), use the course Hapkit Python interface to record 12 s at 1 kHz in the
same three columns as `data/leader_log.csv`: `t` (s), `x_m` (m, handle displacement), `clutch` (1 while the clutch
input of your station is pressed). Perform three forward insertion strokes of about 60 mm, pressing the clutch while
you bring the handle back. Save the file as `data/my_leader_log.csv` and run Parts A and B on it with
`load_log(DATA / "my_leader_log.csv")`. If the station supports a paired (second-Hapkit) follower, the TA may let you
feel P–P and P–F coupling for a few seconds with the default gains ($k_c \le 200$ N/m); this is a demonstration only.

# Part A — Mapping, clutch and device limits (`hapkit_effective_mass`, `saturate_force`, `watchdog_ok`, `map_with_clutch`, `rate_limit`, `clutch_strokes`)

A1. `hapkit_effective_mass(model)`: $m_\text{eff} = (I_{zz,\text{com}} + m d^2 + I_\text{rotor}(r_s/r_p)^2)/r_h^2$, with
$d$ the distance of the centre of mass from the pivot axis. **Checkpoint:** 0.05705 kg (the value used for $m_m$ and
$m_s$ in `teleop_params.json`).

A2–A3. Force saturation and watchdog (scalars and arrays).

A4. `map_with_clutch(x_m, clutch, alpha, x_s0)`: while the clutch is pressed the follower set-point holds; at every
release re-latch $x_{m,\text{latch}}$ and $x_{s,\text{latch}}$. **Checkpoint:** with $\alpha = 0.25$ the synthetic log
ends at **37.79 mm** (37.5 mm of voluntary motion plus the tremor and noise captured at the latches). Plot $x_m$,
the clutch signal and $x_{s,d}$ (Figure 1).

A5. `rate_limit(x_d, v_max, dt)`: $|x_{k} - x_{k-1}|\le v_{\max}\,dt$.

A6. `clutch_strokes(travel, alpha, stroke)`. **Checkpoint:** 3 for the laparoscope (40 mm, 0.25, 60 mm) and 3 for the
EOD approach (0.60 m, 4, 70 mm).

# Part B — Tremor filtering (`lowpass1`, `butter2`, `iir_filter`, `band_rms`, `lag_estimate`)

B1. First-order low-pass $y_k = y_{k-1} + a(x_k - y_{k-1})$, $a = 1 - e^{-2\pi f_cT}$, starting at $y_0 = x_0$.

B2. Second-order Butterworth by the bilinear transform with pre-warping, $K = \tan(\pi f_cT)$:
$b = \frac{K^2}{1 + \sqrt2K + K^2}[1,\ 2,\ 1]$,
$a = [1,\ \frac{2(K^2 - 1)}{1 + \sqrt2K + K^2},\ \frac{1 - \sqrt2K + K^2}{1 + \sqrt2K + K^2}]$. The test compares it with
`scipy.signal.butter(2, fc, fs=1000)`.

B3. `iir_filter(b, a, x)`: the difference equation, starting at rest at $x_0$.

B4. `band_rms(x, dt, f_lo, f_hi)`: remove the straight line joining the first and last samples and the mean, then
$\text{rms}^2 = 2\sum_{\text{band}}|X_k|^2/N^2$ (one-sided FFT, Parseval). Without the detrending, the jump between
the end and the start of the record leaks into every band.

B5. `lag_estimate(x, y, dt, max_lag)`: the shift that minimises the mean squared difference between $y$ and the
shifted $x$.

**Checkpoints (synthetic log, 8–12 Hz band).** Raw tremor RMS **0.212 mm** (0.3 mm amplitude). Fraction passed and
lag:

| Filter | Tremor passed | Lag |
|---|---|---|
| First order, $f_c = 2$ Hz | 0.217 | 0.078 s |
| Butterworth, $f_c = 2$ Hz | 0.049 | 0.114 s |
| Butterworth, $f_c = 2.536$ Hz (Workshop 2, Problem 5 design) | 0.079 | 0.090 s |

Compare each row with the analytic magnitude at 9 Hz and the low-frequency delays $1/\omega_c$ and $\sqrt2/\omega_c$.
Figure 2: raw and filtered $x_m$ over a 1 s window, both filters. Choose a filter for the laparoscopy task and justify
it with the table.

# Part C — Two-port models (`hybrid_matrix`, `transmitted_impedance`, `llewellyn`)

C1. `hybrid_matrix(arch, s, p)` for `'pp'` and `'pf'` with scales (Workshop 2): it must work for a complex scalar and
for an array of $s$ values (shape `s.shape + (2, 2)`). **Checkpoint:** P–P, $\alpha = \beta = 1$, 2 Hz:
$h_{11} = 0.264 + j1.464$ N s/m.

C2. `transmitted_impedance(H, Ze)`. Plot $|Z_t(j\omega)|$ for 0.1–1000 rad/s for both architectures, in free space
and against the 300 N/m tissue (written as $Z_e = b_e + k_e/s$), together with $|Z_e|$ (Figure 3). Where is each
architecture transparent, and where not?

C3. `llewellyn(H)`: $\eta = 2\,\mathrm{Re}\,h_{11}\,\mathrm{Re}\,h_{22} - \mathrm{Re}(h_{12}h_{21}) - |h_{12}h_{21}|$.
**Checkpoint:** with the lab parameters ($\alpha = 0.5$, $\beta = 1$) P–F has $\min_\omega\eta = -0.55$ near 74 rad/s;
symmetric P–P ($\alpha = \beta = 1$) has $\eta\ge0$. Then compute the leader damping needed at each frequency,
$b_m(\omega) = (|G| - \mathrm{Re}\,G)/(2\,\mathrm{Re}\,h_{22})$ with $G = -h_{12}h_{21}$, and plot it (Practice
Problem 3).

# Part D — Bilateral control in time (`control_forces`, `closed_loop_matrix`, `max_stable_beta`)

D1. `control_forces(arch, xm, vm, xs, vs, fe, p)`:
$a_s = k_c(\alpha x_m - x_s) + b_c(\alpha v_m - v_s)$; P–P: $a_m = -\beta a_s$; P–F: $a_m = -\beta f_e$.
`simulate_teleop` calls it every 1 ms.

**Reach task.** The hand (spring–damper $k_h = 200$ N/m, $b_h = 2$ N s/m towards the intended position) moves the
handle from 0 to 20 mm in 1 s; the follower ($\alpha = 0.5$) meets a wall at 5 mm. **Checkpoints** (both
architectures, $\beta = 1$): steady contact force **0.649 N** on the 5000 N/m wall ($x_m = 16.75$ mm, $x_s = 5.13$ mm)
and **0.462 N** on the 300 N/m tissue. Check both against the static balance $k_h(x_\text{goal} - x_m) = \beta f_e$,
$f_e = \frac{k_ck_e}{k_c + k_e}(\alpha x_m - x_w)$. In free space the P–P leader motor force peaks at only
**4.4 mN** (the follower is a light Hapkit); explain how this would change with a 2 kg surgical arm as follower.

D2. `closed_loop_matrix(arch, p, ke, be)`: the $4\times4$ matrix $A$ of the linear in-contact model, state
$(x_m, v_m, x_s, v_s)$, wall at 0:
$(m_m + m_h)\dot v_m = -k_hx_m - (b_h + b_m)v_m + a_m$, $m_s\dot v_s = a_s - b_sv_s - (k_ex_s + b_ev_s)$.

D3. `max_stable_beta(arch, p, ke, be)`: the largest $\beta$ with $\max\mathrm{Re}\,\lambda(A) < 0$ (bisection or
`brentq`); return `math.inf` if it is stable up to `beta_hi`. **Checkpoints:** P–F: $\beta_{\max} = $ **8.155** on the
wall and **6.144** on the tissue (the limit does not simply grow or shrink with $k_e$, so always check it); P–P: no
finite limit in this model.

**Stability experiment.** Simulate P–F on the wall with a slower approach (`minjerk(t, 0, 2, 0, 0.020)`, 3 s), with
`saturate=False, watchdog=False`, for $\beta = 0.5, 0.7, 0.8, 1.0, 1.2\times\beta_{\max}$, and plot $f_e(t)$
(Figure 4). Find by bisection the smallest $\beta$ at which the contact force still oscillates at the end
(peak-to-peak > 1 mN in the last 0.5 s). **Checkpoint:** about **6.4**, i.e. **0.78** $\beta_{\max}$. The linear model
assumes the follower never leaves the wall; in the simulation the lightly damped contact mode makes it bounce off the
one-sided wall, and switching between contact and free motion destabilises the loop earlier. Repeat at
$1.2\beta_{\max}$ **with** saturation and watchdog: the 2 N limit holds, but the operator feels a sustained buzz (chatter).
This is the stability–transparency trade-off: a larger $\beta$ gives a firmer, more transparent feel and less
margin.

# Part E — EOD follower speed limit (`eod_approach_rates`, `max_approach_speed`)

E1. `eod_approach_rates(model, q)`: $\dot\theta = J_b(\theta)^{-1}(0, 0, 0, 1, 0, 0)$ at the ready pose (use the
provided `jacobian_body` or your own 361 function; the test compares with `modern_robotics`). **Checkpoint:**
$(0, -1.310, 2.019, 0, -0.709, 0)$ rad/s.

E2. `max_approach_speed(model, q, speed_fraction)`. **Checkpoint:** 0.259 m/s at 100 % and **0.130 m/s** at 50 % of
the 30°/s rating. With $\alpha = 4$, set the rate limiter of Part A on the follower set-point and report the
corresponding hand-speed limit (32 mm/s) and the approach time for 0.60 m.

# Deliverables (before the Week 3 lab)

1. `starter.py` passing all tests: `LAB_IMPL=starter python -m pytest -q test_lab.py` → **25 passed**; commit it to your
   repository as `sbeg46x_medrob/teleop.py` with your tag `lab2`.
2. A results table with every checkpoint value from your own code (and from your own Hapkit log, if you recorded one:
   final follower position, tremor RMS, tremor passed by your chosen filter).
3. Figures 1–4 (mapping with clutch; tremor filters; $|Z_t|$ versus frequency; stability experiment), each with axis
   labels, units and a one-sentence caption that states what the figure shows.
4. Reflection (below), at most 250 words.

# Reflection (with AI-use note)

Answer briefly: (1) Which architecture would you choose for a palpation task with a light follower, and which for an
EOD arm with a heavy follower? Use your Figure 3 and the free-space numbers. (2) Your filter choice for laparoscopy:
what did you give up for the tremor attenuation? (3) Why did the simulated wall chatter below the linear-model
$\beta_{\max}$, and what would you do on the real device? (4) **AI-use note:** tools used, what you asked, what you
kept or changed, and how you verified it (tests, checkpoints). If you used none, write "No AI tools were used."

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `NotImplementedError` in a test | That function is still a TODO; the test name tells you which part |
| `test_A4_clutch_mapping_small_example` fails at the release sample | You re-latch only $x_m$ or update the latch while the clutch is still pressed; latch both, at the first unclutched sample |
| `band_rms` of a pure sine is wrong | Remove the end-to-end line and the mean, use $2\sum|X_k|^2/N^2$ and exclude DC |
| `butter2` differs from SciPy | Use $K = \tan(\pi f_cT)$ (pre-warped), not $\pi f_cT$; normalise so that $a_0 = 1$ |
| Your filters start with a large transient | Start at rest at $x_0$ (past inputs and outputs equal to $x_0$), not at zero |
| `hybrid_matrix` has the wrong shape for arrays | Build `H = np.zeros(s.shape + (2, 2), complex)` and fill `H[..., i, j]` |
| $\beta_{\max}$ is 0 or `inf` for P–F | Sign of the reflected force: $a_m = -\beta f_e$ pushes the hand back; in $A$ the leader row gets $-\beta k_e/(m_m + m_h)$ in the $x_s$ column |
| Steady contact force is twice the checkpoint | You applied $\beta$ twice, or used $\alpha$ on the follower's own position |
| The watchdog trips in your Hapkit recording | You moved beyond ±40 mm: keep strokes within ±30 mm and use the clutch |
| Hapkit not found on the serial port | Close other programs using the port; replug; ask the TA. Use the synthetic log meanwhile (no penalty) |

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, deliverables and reflection are the same; the auto-checks compare your results with the closed forms of the Python tests, with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: body Jacobian of the EOD arm) and with the numbers of the Python reference (including the `scipy.signal` filter outputs), so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `helpers.cpp` (given, do not edit: `load_params`, `load_log`, `minjerk`, `reach_intent`, `simulate_teleop`, `jacobian_body` and the two course models `hapkit()` and `eod_6r_ugv()` of `course_models.py`), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers: the random signals drawn by `test_lab.py`, the `scipy.signal` results, the Python reference values), `CMakeLists.txt`. The data files are the same `data/teleop_params.json` and `data/leader_log.csv` (or your own `data/my_leader_log.csv`: `lab::load_log(sbeg::lab_path("../data/my_leader_log.csv"))`). `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 2 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*A1*,*A2*"              # only some tests (like pytest -k)
```

With the untouched starter the program builds and all 23 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 19 as in Python, same order): `hapkit_effective_mass`, `saturate_force`, `watchdog_ok`, `map_with_clutch`, `rate_limit`, `clutch_strokes` (Part A); `lowpass1`, `butter2`, `iir_filter`, `band_rms`, `lag_estimate` (Part B); `hybrid_matrix`, `transmitted_impedance`, `llewellyn` (Part C); `control_forces`, `closed_loop_matrix`, `max_stable_beta` (Part D); `eod_approach_rates`, `max_approach_speed` (Part E). Given in `helpers.cpp` besides the provided Python helpers: the scalar `saturate_force(double, f_max)` and `hybrid_matrix(arch, Eigen::VectorXcd, p)` for several values of $s$; both call your functions.

**Checkpoints.** A: `./build/lab_tests -tc="test_A*"` (6 test cases); B: `-tc="test_B*"` (6); C: `-tc="test_C*"` (3); D: `-tc="test_D*"` (6); E: `-tc="test_E*"` (2). Finally `./build/lab_tests` must show 23 passed. The Python track counts 25 tests because pytest counts each architecture of `test_C1_hybrid_matrix` and `test_D2_closed_loop_matrix` separately; the C++ test case loops over `"pp"` and `"pf"`. The checkpoint values of Parts A–E are the same in both tracks.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| parameter `dict` `p`; `p["raw"]["environments"]["wall"]`, `p["raw"]["reach_task"]` | `lab::Params p = lab::load_params();` with the same field names (`p.kc`, `p.beta`, ...); `p.environments.at("wall")` (a `lab::Environment` with `k`, `b`, `x_wall`), `p.reach_task`; also `p.laparoscopy`, `p.tremor`, `p.eod` |
| `dict(p, beta=b)` | a copy: `lab::Params q = p; q.beta = b;` |
| log `dict`, model `dict`s, result `dict` of `simulate_teleop` | structs `lab::LeaderLog` (`t`, `x_m`, `clutch`), `lab::HapkitModel`, `lab::EodModel`, `lab::TeleopSim` (`r.f_e`, `r.tripped`, ...) with the same field names; `shoulder=(a, h)` of `eod_6r_ugv` is two arguments |
| 1-D arrays; boolean arrays | `Eigen::VectorXd`; `std::vector<bool>` |
| `saturate_force` on scalars and arrays | you write the `Eigen::VectorXd` version; the `double` version is given |
| complex `s`, `H[..., i, j]`, "`s` may be an array" | `std::complex<double>`, `Eigen::Matrix2cd` with `H(i, j)`; you write the scalar-`s` version, the version for a vector of $s$ (returns `std::vector<Eigen::Matrix2cd>`) is given |
| tuples `(b, a)` of `butter2`, `(a_m, a_s)` of `control_forces` | `std::pair`: `auto [b, a] = lab::butter2(2.0, 0.001);` |
| keyword arguments of `simulate_teleop` (`saturate=False, watchdog=False, control=...`) | positional, in the order of `lab.hpp`: `lab::simulate_teleop("pf", q, env, xi, 0.0, 0, false, false, lab::control_forces)` (`dt = 0`, `substeps = 0` mean the values of `p`) |
| `np.fft.rfft` in `band_rms` | no FFT in C++: compute only the DFT bins inside the band (hint in `starter.cpp`); the result is the same Parseval sum |
| `np.linalg.eigvals`, `brentq` in `max_stable_beta`; `math.inf` | `Eigen::EigenSolver<Eigen::Matrix4d>`; bisection to $10^{-10}$ (hint in `starter.cpp`); `std::numeric_limits<double>::infinity()` |
| `np.linalg.solve(Jb, v)` | `Jb.partialPivLu().solve(v)` |
| `scipy.signal` (`lfilter`, `butter`) and `np.random.default_rng(0)`, `(1)` in the tests | values computed in Python and stored in `cpp_cases.json` |
| `raise ValueError` | `throw std::invalid_argument("...")` (`hybrid_matrix` and `control_forces` with an unknown `arch`) |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

**Figures and results table.** The plots stay outside the auto-checks. Write the experiments (Figure 1: mapping with clutch; Figure 2: tremor filters; Figure 3: $|Z_t|$ versus frequency; Figure 4: stability experiment, and the bisection for the smallest oscillating $\beta$) as a C++ program and save the data with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab(...)`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp helpers.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::load_params()` and `lab::load_log()` find the data from any build folder. Deliverable 1 for the C++ track is `cpp/starter.cpp` passing all 23 test cases (add the functions to your library as `teleop.cpp` if you keep a C++ library; Labs 3 and 4 have a C++ track too); deliverable 3 is `experiments.cpp` plus your plotting script and the four figures. The results table and the reflection are unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../teleop_params.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| every Part D simulation test fails with `TODO: control_forces` | `simulate_teleop` calls your `control_forces` every millisecond | write `control_forces` first |
| `band_rms` slightly wrong for a long record, or very slow | angle computed as `2 * PI * k * n / N` in floating point, or all $N/2$ bins computed | use the integer remainder `(k * n) % N`, and only the bins inside the band |
| `max_stable_beta` never returns | the bisection loop has no iteration limit | stop when `hi - lo < 1e-10` **or** after 200 iterations |
| "Eigen: wrong matrix size or index out of range" | e.g. a result vector not sized (`Eigen::VectorXd y(x.size());`), or `x(N)` instead of `x(N - 1)` | check the sizes; the test program does not crash, the test simply fails |
| Results wrong only in some entries, no compile error | an Eigen expression stored in `auto` (`auto d = x - y;` keeps references to temporaries) | write the type: `Eigen::VectorXd d = x - y;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
