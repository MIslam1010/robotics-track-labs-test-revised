---
title: "Lab 3 — Delayed bilateral teleoperation: instability, wave variables and time-domain passivity (Formal report 1)"
subtitle: "SBEG 46X Medical Robotics — Week 3"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4 computational lab, 50 min in class + about 3 h of independent work for the report |
| CLOs | **46X-2** (bilateral teleoperation; stability under communication delay with passivity, wave-variable and time-domain-passivity methods); **46X-10** (technical report, honest disclosure of tool use) |
| Reading | Åström & Murray, *Feedback Systems*: Table 8.1 (p. 236), §9.3 delay margin (p. 281), §9.5 small gain and passivity (pp. 287–288); open references † Anderson & Spong (1989), † Niemeyer & Slotine (1991), † Hannaford & Ryu (2002); this week's lecture notes, the Week 2 notes (P–P and P–F architectures) and the SBEG 461 Week 7 notes on passivity (recapped in Lecture A) |
| Files (your Lab code folder) | `starter.py` (your code), `test_lab.py` (auto-checks), `course_models.py` (the Hapkit model, copied from Course resources: Course robot models and hardware safety checklist), `data/teleop.json` (parameters), `data/ref_*.csv` (reference runs), `data/expected.json` |
| Hardware | **Simulation only.** Two simulated Hapkit-type devices (leader and follower) with the Lab 2 hand model. Communication delay is always emulated in software |
| Assessment | Auto-checked code (tests) + **Formal report 1** (IEEE two-column, ≤ 6 pages, template and rubric below), due one week after the lab, before the Week 4 lab session |
| AI use | Allowed for code **with disclosure** (AI-use note in the report, appendix A); you must be able to explain every line in an oral check. Derivations, numbers, figures and conclusions in the report are your own and come from your runs |

## Why this lab matters

In Lab 2 the leader and follower were connected by a wire. In this lab they are connected by a **network**, and the
network delays every packet. Two cases drive the lab:

* **Biomedical — long-distance telesurgery.** A surgeon in a city hospital operates on a patient in a district hospital
  through a fibre link with a one-way delay of tens of milliseconds. The instrument presses on soft tissue (300 N/m).
* **Defence — EOD teleoperation and field telesurgery.** The operator of an explosive-ordnance-disposal (EOD) robot
  touches the casing of a suspect object (a stiff 5000 N/m contact) over a radio link with about 100 ms one-way delay;
  a surgeon supports a field hospital over a geostationary satellite link with about 300 ms. Both uses are protective
  or medical.

You will see that force reflection which is perfectly stable in Lab 2 starts to **bounce on the tissue after a few
milliseconds of delay**, predict that limit with the delay margin, find where the energy comes from, and fix it in two
ways — wave variables and time-domain passivity — and measure what each fix costs the operator.

## Learning objectives

By the end of the lab you can:

1. predict the delay margin of a force-reflecting teleoperator in contact from a linear model (characteristic equation
   $ms^2 + bs + k + k_pe^{-s\tau} = 0$) and compare it with simulation;
2. show with an energy balance that a delayed force/velocity channel is active (generates energy);
3. implement the wave transformation at both ends of a delayed channel and explain why it is passive for any constant
   delay, and what it costs (heavier free motion, softer contact, drift);
4. implement passivity observers and passivity controllers for the two energy flows of a delayed channel
   (time-domain passivity) and evaluate them under jitter and packet loss;
5. report the comparison in a short IEEE-style technical report.

# Pre-lab check (complete before the session, 5 questions)

1. A pure delay of 20 ms: what is its gain and its phase lag (in degrees) at 10 Hz?
2. In the position–position (P–P) architecture of Week 2, what force does the leader receive when the link delays every
   packet by $T$ in each direction? Write it as a function of time.
3. With wave impedance $b = 5$ N·s/m, a resting handle ($\dot x = 0$) pushed with $F = 1$ N: compute $u$ and $v$.
4. Why must the follower's passivity observer use the leader's input energy **delayed** by $N$ samples, and not the
   current value?
5. The control loop runs at 1 kHz. How many samples is a one-way delay of 100 ms, and how many packets are "in flight"
   in each direction at any time?

# The model

Both devices are Hapkit-type handles with the effective mass from `course_models.hapkit()` (as in Lab 2). The hand is a
spring–damper that pulls the leader towards the intended position $x_i(t)$ (a 20 mm minimum-jerk reach in 1 s, then
hold — so the operator keeps pushing 15 mm "into" a wall at 5 mm); its mass is lumped with the handle.

$$\begin{aligned}
(m_m + m_h)\ddot x_m &= k_h(x_i - x_m) + b_h(\dot x_i - \dot x_m) - b_m\dot x_m - F_m &&\text{(leader + hand)}\\
m_s\ddot x_s &= F_c - b_s\dot x_s - f_e,\qquad F_c = K_p(x_d - x_s) - K_d\dot x_s &&\text{(follower + PD)}\\
f_e &= \max\big(0,\ k_e(x_s - x_w) + b_e\dot x_s\big)\ \text{if } x_s > x_w,\ \text{else } 0 &&\text{(contact)}
\end{aligned}$$

| Quantity | Value (illustrative) | Quantity | Value |
|---|---|---|---|
| $m_m = m_s$ (Hapkit handle) | 0.0570 kg | $K_p$, $K_d$ (follower PD) | 200 N/m, 2.0 N·s/m |
| $b_m$, $b_s$ | 0.15 N·s/m, 0.10 N·s/m | tissue $k_e$, $b_e$ | 300 N/m, 1.0 N·s/m at $x_w = 5$ mm |
| hand $m_h$, $b_h$, $k_h$ | 0.10 kg, 2.0 N·s/m, 200 N/m | stiff object $k_e$, $b_e$ | 5000 N/m, 1.0 N·s/m at $x_w = 5$ mm |
| control period, mechanics | 1 ms, 10 sub-steps | force limit, watchdog | 2 N, ±40 mm (as on the hardware) |

$F_m$ is the force of the teleoperator on the leader (positive = resisting the operator; Lab 2's leader motor force is
$a_m = -F_m$). The follower damping acts on $\dot x_s$ only, so that the PD force of a sample does not depend on that
sample's set-point step (this avoids an algebraic loop in the wave and passivity controllers). Power into the channel
at the leader is $P_m = F_m\dot x_m$; power out of the channel at the follower is $P_s = F_c\dot x_d$.

The provided `simulate(arch, T, env=...)` runs four architectures with a one-way delay $T$ (at least one sample):

| `arch` | Forward ($x_d$) | Backward ($F_m$) |
|---|---|---|
| `"pp"` | $x_d(t) = x_m(t - T)$ | $F_m(t) = F_c(t - T)$ (follower controller force) |
| `"pf"` | $x_d(t) = x_m(t - T)$ | $F_m(t) = f_e(t - T)$ (follower force sensor) |
| `"wave"` | $u_s(t) = u_m(t - T)$, follower integrates $\dot x_d$ | $v_m(t) = v_s(t - T)$, low-pass filtered (passive) |
| `"tdpa"` | as `"pp"`, set-point step scaled by $r_s$ | as `"pp"`, force scaled by $r_m$ |

It calls **your** functions `follower_pd`, `wave_leader`, `wave_follower`, `tdpa_leader` and `tdpa_follower`, so it
works only when they are written. It returns arrays of every signal plus the channel energy $E_{ch}$, the observer
values $W_m$, $W_s$ and the controller ratios $r_m$, $r_s$.

# Setup

```bash
cd "<your Lab code folder>"
LAB_IMPL=starter python -m pytest -q test_lab.py     # at the start: 22 tests fail with NotImplementedError, 3 data tests pass
```

Use the course Python environment (or the Docker image / JupyterHub). Work in `starter.py` and keep the function names
and signatures.

# Procedure and checkpoints

### Checkpoint 1 — Why force reflection becomes unstable (≈ 12 min)

1. `contact_stiffness(Kp, k_e)` (series springs) and `delay_margin(m, b, k, kp)`: solve the magnitude condition as a
   quadratic in $\omega^2$, keep the real positive roots, and for each one find the smallest $\tau > 0$ with
   $e^{-j\omega\tau} = -(k - m\omega^2 + jb\omega)/k_p$ (use `np.angle(...)` and `% (2*np.pi)`). Workshop Problem 2 is
   the same calculation by hand.
2. `python -m pytest -q test_lab.py -k "margin or stiffness"` (5 tests). One test checks the integrator loop
   $s + k_pe^{-s\tau}$, whose margin $\pi/(2k_p)$ you know from the workshop; one checks your answer against an
   independent frequency sweep.
3. With $m = m_m + m_h$, $b = b_m + b_h$, $k = k_h$, compute the one-way margin $T^* = \tau^*/2$ for tissue and for the
   stiff object.

**Checkpoint question:** which contact gives the smaller margin, and why?

### Checkpoint 2 — Wave variables (≈ 12 min)

1. `wave_encode`, `wave_leader`, `wave_follower`: solve the definitions of $u$ and $v$ for the unknown power variable
   at each end (leader: given $\dot x_m$ and the arriving $v_m$, find $F_m$ and $u_m$; follower: given the arriving
   $u_s$ and $F_c$, find $\dot x_d$ and $v_s$).
2. `wave_apparent_impedance(b, T, omega, termination)`: $b\coth(j\omega T)$ (rigid) and $b\tanh(j\omega T)$ (free).
3. `python -m pytest -q test_lab.py -k wave` — the static-contact test checks that a resting wave channel transmits
   force exactly.

### Checkpoint 3 — Energy bookkeeping and time-domain passivity (≈ 15 min)

1. `follower_pd`, `port_energies` (cumulative input and output energies of a port), `passivity_ratio`
   ($r = \max(\text{budget}, 0)/\text{requested}$ when the request exceeds the budget, else 1).
2. `tdpa_leader` (impedance-type port: scales $F_m$) and `tdpa_follower` (admittance-type port: scales the set-point
   step $\Delta x_d$). Each updates its own input/output energies and compares its output with the **other** side's
   input energy received through the delayed channel. Workshop Problem 4 is this calculation by hand.
3. `python -m pytest -q test_lab.py -k "tdpa or energies or ratio or pd"`.

### Checkpoint 4 — Simulations, metrics and the network (rest of the session, then at home)

1. `performance_metrics(sim, t_from=2.5)` (means after 2.5 s, drift $x_d - x_m$, rms tracking error, standard deviation
   of $x_m$ as the chatter measure, minimum channel energy) and `latency_budget`.
2. Run the whole suite: `python -m pytest -q test_lab.py` — all 25 tests must pass. The simulation tests compare your
   runs with `data/ref_*.csv`.
3. Do the experiments below for the report. Save every figure with axis labels and units.

**Experiment A — the delay limit of force reflection (biomedical and defence).**

* A1. Sweep the one-way delay $T \in \{2, 4, \dots, 20\}$ ms for `"pp"` and `"pf"`, with `env="tissue"` and
  `env="wall"`. **Figure 1:** chatter amplitude (`x_std`) versus $T$, four curves, with your predicted $T^*$ as vertical
  lines. Where does each curve leave the floor? Explain the difference between the prediction and the simulation
  (sampling and zero-order hold add delay; the follower is not quasi-static; `pf` reflects the measured contact force,
  including its damping term).
* A2. **Figure 2:** $x_m$, $x_s$ and $F_m$ versus time for `"pp"`, tissue, $T = 5$ ms and $T = 20$ ms, and the channel
  energy $E_{ch}(t)$ for both. Relate the sign of $E_{ch}$ to what you see.
* A3 *(optional, advanced)*. Run `"pp"`, tissue, for $T = 50$, 70 and 100 ms. Explain the result with the two crossing
  frequencies of `delay_margin` (Workshop Problem 2e), and say why you would never rely on it.

**Experiment B — wave variables at EOD-radio and satellite delays.**

* B1. `"wave"` with $T = 100$ ms (EOD radio) and $T = 300$ ms (satellite), tissue. **Figure 3:** positions and forces.
  Is the contact stable? How large are the force felt by the operator and the position drift?
* B2. **Table 1:** for $T = 100$ ms and $b \in \{2, 5, 20\}$ N·s/m: $\bar F_m$, drift, chatter, and the predicted
  contact stiffness $b/T$ and free-space mass $bT$. Which $b$ would you choose for touching a suspect object, and which
  for free motion? Why can't one $b$ be best for both?

**Experiment C — time-domain passivity.**

* C1. `"tdpa"` with $T = 20$ ms (terrestrial telesurgery), tissue; compare with `"pp"` and `"wave"` at the same delay.
  **Figure 4:** $E_{ch}(t)$ for the three architectures on one plot. **Figure 5:** $r_m$, $r_s$ (how often and how
  strongly each passivity controller acts) and the drift $x_d - x_m$.
* C2. Repeat at $T = 100$ ms. Explain, with the energy budgets, why the follower drifts so much: in free motion the
  follower needs energy *now*, but the leader's input energy reaches it only $T$ later, and the leader feels the
  follower's force $2T$ late.

**Experiment D — jitter, packet loss and the latency budget.**

* D1. **Table 2:** `"pp"`, `"wave"`, `"tdpa"` with `env="wall"`, $T = 100$ ms, and (jitter, loss) = (0, 0),
  (20 ms, 0), (0, 0.1), (20 ms, 0.1): minimum $E_{ch}$, chatter, $\bar F_m$, drift, watchdog trip. Which architecture
  keeps $E_{ch} \ge 0$ in every case, and why is that guaranteed for one of them and only observed for another?
* D2. Write a one-way latency budget (at least five components) for a field-telesurgery link of your choice and compare
  it with $T^*$ of Checkpoint 1 (use `latency_budget`). One paragraph: which architecture would you deploy?

**Expected values (to check yourself; TAs use them for calibration).**

| Quantity | Expected |
|---|---|
| Leader model $m$, $b$, $k$ | 0.157 kg, 2.15 N·s/m, 200 N/m |
| One-way delay margin $T^*$ (crossing $\omega_2$): tissue / stiff object | 10.15 ms (42.1 rad/s) / 5.90 ms (48.0 rad/s) |
| Ideal static force at the leader (tissue, no delay) | 1.125 N |
| `pp` tissue, $T = 5$ ms: $\bar F_m$, chatter | 1.125 N, 0.0004 mm |
| Chatter (`x_std`) at $T = 20$ ms: `pp` tissue / `pf` tissue / `pp` wall | 8.31 mm / 10.2 mm / 9.65 mm |
| Minimum channel energy at $T = 20$ ms: `pp` tissue / `pf` tissue / `pp` wall | −0.404 J / −0.823 J / −0.857 J |
| Sweep A1: `pp` tissue chatter < 0.5 mm at 10 ms, > 3 mm at 15 ms; `pf` tissue > 1 mm already at 6 ms | |
| `wave` tissue, $T = 100$ ms: $\bar F_m$, drift, chatter | 0.442 N, −9.09 mm, 0.11 mm |
| `wave` tissue, $T = 300$ ms: $\bar F_m$, drift, chatter | 0.203 N, −12.3 mm, 0.72 mm |
| `wave` tissue, $T = 100$ ms, $\bar F_m$ for $b$ = 2 / 5 / 20 N·s/m | 0.232 / 0.442 / 0.819 N |
| `tdpa` tissue, $T = 20$ ms: $\bar F_m$, drift, chatter | 0.805 N, −4.25 mm, 0.07 mm |
| `tdpa` tissue, $T = 100$ ms: $\bar F_m$, drift, chatter | 0.100 N, −13.7 mm, 0.01 mm |
| $E_{ch}$, $W_m$, $W_s$ for `wave` and `tdpa` | never below 0 (to round-off) |

<!-- library-block:begin -->
# Your `sbeg46x_medrob` library

When the tests pass, add the functions you wrote in Lab 3 to the module `teleop.py` of your library (the module you
started in Lab 2). The functions you write (13):

`contact_stiffness(Kp, k_e)`, `delay_margin(m, b, k, kp)`, `wave_encode(F, xdot, b)`, `wave_leader(xdot_m, v_m, b)`,
`wave_follower(u_s, F_c, b)`, `wave_apparent_impedance(b, T, omega, termination)`,
`follower_pd(x_d, x_s, xdot_s, Kp, Kd)`, `port_energies(P, h)`, `passivity_ratio(out_trial, budget)`,
`tdpa_leader(F_m, xdot_m, h, E_in_m, E_out_m, E_in_s_recv)`, `tdpa_follower(F_c, dx_d, E_in_s, E_out_s, E_in_m_recv)`,
`performance_metrics(sim, t_from=2.5)`, `latency_budget(components_ms, limit_ms)`.

None of these names is used by Lab 2. The provided helpers at the top of `starter.py` (`load_params`, `minjerk`,
`reach_intent`, `DelayLine`, `network_delays`, `contact_force`, `simulate`) do not have to go into the library; the
project scenarios 1, 4 and 5 (delayed teleoperation) reuse your library functions. The full list of library functions
of all labs is in the Week 13 lab handout.
<!-- library-block:end -->

# Deliverables

1. `starter.py` renamed `lab3_<ID>.py`, passing `LAB_IMPL=lab3_<ID> python -m pytest -q test_lab.py` (25 tests), and
   the 13 functions added to `teleop.py` of your library.
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report.
3. **Formal report 1** (PDF, IEEE two-column template, ≤ 6 pages excluding appendices) with Figures 1–5 and
   Tables 1–2.
4. A reflection (≤ 200 words, appendix B) and an AI-use note (appendix A).

# Formal report 1 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran` or the Word version). Title: *Stability and transparency
of delayed bilateral teleoperation for telesurgery and EOD: force reflection, wave variables and time-domain
passivity*. Suggested length per section in brackets.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (delay-margin analysis, energy bookkeeping, wave variables, TDPA), key numbers ($T^*$ predicted vs observed, force felt and drift for each fix at 100 ms), one-sentence conclusion. |
| **I. Introduction** (½ page) | Why delay matters in long-distance telesurgery and in EOD and field-telesurgery links; typical one-way delays (fibre, radio, satellite) with your latency budget; objectives and contributions. |
| **II. System model** (½–1 page) | Leader, hand, follower, contact and network models with a parameter table (state that the values are illustrative); the four architectures; sign conventions for power. |
| **III. Analysis** (1 page) | Delay margin of the delayed reflected spring (derivation, both crossing frequencies, $T^*$ for tissue and stiff contact); energy generated by a delayed channel; wave transformation and why its stored energy is non-negative; the two passivity observers and controllers. Cite Åström & Murray and the open references. |
| **IV. Verification** (¼ page) | Unit tests (what each checks), reference runs, and one independent check of your own (e.g. a hand calculation of $T^*$ or of one wave value). |
| **V. Results** (1½ pages) | Figures 1–5 and Tables 1–2. Every figure is referred to in the text and has a caption that says what to see. |
| **VI. Discussion** (1 page) | Prediction vs simulation; transparency cost of each fix (force felt, drift, heaviness, chatter); behaviour under jitter and loss; which architecture you would deploy for (a) a 20 ms terrestrial telesurgery link and (b) a 100–300 ms EOD radio or satellite link, and when you would switch to supervisory or shared control; limits of the model (1-DOF, linear contact, no video delay, no operator adaptation); safety implication (force limit and watchdog on the hardware). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences. |
| **References** | Åström & Murray (sections and pages), Anderson & Spong 1989, Niemeyer & Slotine 1991, Hannaford & Ryu 2002, Lab 2 notes; IEEE citation style. |
| **Appendix A — AI-use note** | Tool, what you asked, what you kept or changed, how you verified it (or "No AI tools were used."). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |

# Formal report rubric (100 points)

| Criterion | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction | 10 | Problem and both applications motivated with numbers (latency budget); measurable objectives; key results in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Model and analysis | 20 | Correct model and sign conventions; delay-margin derivation with both crossings and $T^*$ for both contacts; energy argument for the delayed channel; wave and TDPA equations explained, not just copied | Minor slips in derivation or notation | Equations listed without explanation | Wrong or missing |
| Verification evidence | 10 | All tests pass; reference comparison; at least one independent check of your own, quantified | Tests pass, independent check weak | Some tests fail or no check | No verification |
| Results (figures and tables) | 25 | Figures 1–5 and Tables 1–2 complete, labelled, with units and captions; numbers consistent with the expected values; prediction marked on Figure 1 | One figure or table weak | Several missing or unreadable | Missing |
| Discussion, safety and recommendation | 20 | Prediction vs simulation explained; transparency cost of each fix quantified; jitter/loss discussed (guaranteed vs observed passivity); justified recommendation for both links incl. supervisory/shared control; model limits and hardware safety | Recommendation with partial justification | Descriptive only | Missing |
| Writing, format, references | 10 | IEEE format, ≤ 6 pages, concise, correct citations with pages | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every figure and table | One item weak | Items superficial | Missing |

Oral check: a TA may ask you to explain any line of your code or any figure (2–3 minutes); failure to explain caps the
code mark at 50 %.

# Reflection prompts (appendix B, ≤ 200 words)

* Which result surprised you most (for example, how few milliseconds break force reflection, or how much the follower
  drifts under time-domain passivity), and why?
* If you were the operator, which of the three teleoperators would you trust at 100 ms, and what would you want the
  display to tell you?
* **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it
  was wrong (a test, a hand calculation, a plot)?

# Safety note

This lab is simulation only. In the hardware rotation (Week 8) and in the project, delayed teleoperation on the Hapkit
pair and the course arm follows Course resources: Course robot models and hardware safety checklist: handle force
saturated at 2 N, watchdog at ±40 mm, delay **emulated in software and never by disabling a watchdog or a force
limit**, the exact command first run in simulation, and the emergency stop held by a team member who is not operating
the leader.

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `delay_margin` returns a value about $\pi/\omega$ too large or too small | Phase of $(k - m\omega^2 + jb\omega)/k_p$ instead of $-(\cdot)/k_p$, or a negative angle not wrapped with `% (2*np.pi)`. |
| `delay_margin` returns the 111 ms crossing for tissue | You kept only one root of the quadratic; take the smallest $\tau$ over **all** real positive roots. |
| Wave tests: `wave_encode(F_m, xdot_m, b)` does not return $(u_m, v_m)$ | Sign of $F$ in $v$: $v = (b\dot x - F)/\sqrt{2b}$; then $F_m = b\dot x_m - \sqrt{2b}\,v_m$. |
| `"wave"` run explodes or $E_{ch} < 0$ | `wave_follower` returns $v_s$ with the wrong sign, or $\dot x_d$ divided by $\sqrt{2b}$ instead of $b$. |
| TDPA tests fail by a factor of 2 | You compared the request with $E_{in}$ of your **own** port, or used the current instead of the received (delayed) energy. |
| $W_s$ slightly negative ($-10^{-17}$) | Round-off; the tests allow $10^{-12}$. Larger negative values mean the ratio is not clipped at 0. |
| Reference comparison fails only for `"tdpa"` after 1–2 s | Tie-breaking: use `out_trial <= budget` → $r = 1$, and update $E_{out}$ with the **scaled** output. |
| `ModuleNotFoundError: starter` | Run `pytest` from inside your Lab code folder, or set `LAB_IMPL` to your file name without `.py`. |
| Everything is slow | One 4 s run takes about 0.1 s; the full sweep for Figure 1 (40 runs) takes about 5 s. Do not re-run simulations inside plotting loops. |

# Pre-lab answers (released after the session)

1. Gain 1 (a delay never changes the amplitude); phase lag $\omega\tau = 2\pi(10)(0.02) = 1.257$ rad $= 72^\circ$.
2. $F_m(t) = F_c(t - T)$ with $F_c(t) = K_p\big(x_m(t - T) - x_s(t)\big) - K_d\dot x_s(t)$: the leader feels a spring
   acting on its own position of $2T$ earlier.
3. $\sqrt{2b} = \sqrt{10} = 3.162$: $u = 1/3.162 = 0.316$, $v = -1/3.162 = -0.316$ (and $\tfrac12(u^2 - v^2) = 0 = F\dot x$).
4. The follower can only spend energy that has already entered the channel at the leader **and travelled** to it; the
   value it knows at sample $n$ is the one sent at $n - N$. Using the current value would assume knowledge that has
   not arrived and can let the channel output energy that does not exist yet.
5. $N = 100$ samples; 100 packets are in flight in each direction at any time.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, experiments, report and rubric are the same; the auto-checks compare your results with the closed-form results and the independent frequency sweep of the Python tests, with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`: the Hapkit handle mass), with the reference runs `data/ref_*.csv` and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work; the provided parameters, hand intent, `DelayLine`, `network_delays`, `contact_force` and `simulate` are at its top), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers: the random inputs drawn by `test_lab.py`, the packet delays of the jitter/loss run, the Python results), `CMakeLists.txt`. The data files are the same `data/teleop.json`, `data/expected.json` and `data/ref_*.csv`. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 3 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*delay_margin*"         # only some tests (like pytest -k)
```

The C++ track has 24 test cases with the Python test names. With the untouched starter the program builds, 21 test cases fail with `TODO: <function>` and 3 pass (they check the data and the given code: the parameters against the Hapkit model, `DelayLine`, the reference files), exactly as the Python starter gives 22 failed, 3 passed. The Python track counts 25 tests because pytest counts each contact of `test_delay_margin_course_teleoperator` (tissue, wall) separately; the C++ test case loops over both. A full run takes well under a second. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, the same 13 as in Python, same order): `contact_stiffness`, `delay_margin` (Checkpoint 1); `wave_encode`, `wave_leader`, `wave_follower`, `wave_apparent_impedance` (Checkpoint 2); `follower_pd`, `port_energies`, `passivity_ratio`, `tdpa_leader`, `tdpa_follower` (Checkpoint 3); `performance_metrics`, `latency_budget` (Checkpoint 4). Given and complete: `load_params`, `minjerk`, `reach_intent`, `DelayLine`, `network_delays`, `contact_force`, `simulate` (it calls your `follower_pd`, `wave_leader`, `wave_follower`, `tdpa_leader` and `tdpa_follower`, so every simulation test fails with `TODO: follower_pd` until Checkpoint 3 is done).

**Checkpoints.** 1: `./build/lab_tests -tc="*stiffness*,*delay_margin*"` (4 test cases); 2: `-tc="*wave*" -tce="*long_delay*"` (4; the fifth `wave` test, `test_wave_long_delay_is_passive_and_stable`, is a simulation and also needs Checkpoint 3); 3: `-tc="*tdpa*,*energies*,*ratio*,*pd*"` (7); 4: `./build/lab_tests` must show 24 passed.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| parameter dict `p = load_params()`; `p["controller"]["Kp"]`, `p["environments"]["wall"]["k"]` | struct `lab::Params` with the same field names: `p.controller.Kp`, `p.environments.at("wall").k` |
| `simulate("tdpa", 0.1, env="wall", jitter=0.02, loss=0.1, b_wave=2.0)` | keyword arguments in a `lab::SimOptions`: `lab::SimOptions o; o.env = "wall"; o.jitter = 0.02; o.loss = 0.1; o.b_wave = 2.0;` then `lab::simulate("tdpa", 0.1, o)` (or `lab::simulate(arch, T, p, o)` with your own `Params p`); an environment dict is `o.env_custom = lab::Environment{k, b, x_wall};` |
| result dict `r["x_m"]`, `r["E_ch"]`, `r["tripped"]` | struct `lab::SimResult`: `r.x_m`, `r.E_ch` (`Eigen::VectorXd`, one value per control sample), `r.tripped`, `r.N` |
| dict of `performance_metrics`: `m["x_std"]` | struct `lab::Metrics`: `m.x_std` (population standard deviation, as `np.std`) |
| tuples `(tau, w)`, `(u, v)`, `(F_m, u_m)`, `(E_in, E_out)`, `(total, margin)` | `std::pair`: `auto [tau, w] = lab::delay_margin(m, b, k, kp);` |
| 4-tuples of `tdpa_leader` / `tdpa_follower` | structs `lab::TdpaLeaderResult {F_m, E_in_m, E_out_m, r}` and `lab::TdpaFollowerResult {dx_d, E_in_s, E_out_s, r}`: `return {F_new, E_in_m, E_out_m, r};` |
| `np.roots` (quadratic in $\omega^2$) | the quadratic formula (negative discriminant = complex roots = no crossing) |
| `1j`, `np.angle(z)`, `np.tanh(z)`, `(-phase) % (2*np.pi)` | `std::complex<double>`, `std::arg(z)`, `std::tanh(z)`; `std::fmod(-phase, 2 * PI)` and add `2 * PI` if negative (Python's `%` is never negative) |
| `math.inf`, `math.nan` | `std::numeric_limits<double>::infinity()`, `quiet_NaN()` (test with `std::isinf`, `std::isnan`) |
| `latency_budget({"encode": 5.0, ...}, 100.0)` | `lab::latency_budget({{"encode", 5.0}, {"radio", 60.0}, {"decode", 5.0}}, 100.0)` (a list of name/value pairs, kept in order) |
| `network_delays` with jitter or loss (`np.random.default_rng(seed)`) | `std::mt19937(seed)`: the same statistics but **different packets** than Python with the same seed, so Table 2 numbers can differ slightly from a Python run. The jitter/loss test uses the Python packet delays from `cpp_cases.json` (`o.fwd_delays`, `o.bwd_delays`), so it runs exactly the Python case |
| random inputs of `test_lab.py` (`np.random.default_rng(463)`) | stored in `cpp_cases.json` |
| `np.pi` | a local `constexpr double PI = 3.14159265358979323846;` (`M_PI` is not standard C++) |

The `"pp"`, `"pf"` and `"wave"` runs of the C++ `simulate` reproduce `data/ref_*.csv` to round-off ($10^{-14}$ m). The `"tdpa"` run is not compared sample by sample in either track (the passivity controller switches on thresholds, so round-off can move a switching instant); its passivity and stability checks are the same.

**Figures and tables for the report.** The plots stay outside the auto-checks. Write the experiments A–D as a C++ program and save the time series and tables with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab()`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `lab::simulate(...)` and `lab::load_params()` find `data/teleop.json` from any build folder; for example `Eigen::MatrixXd M(r.t.size(), 3); M << r.t, r.x_m, r.E_ch; sbeg::write_csv("pp_T020.csv", {"t", "x_m", "E_ch"}, M);`. Deliverable 1 for the C++ track is `cpp/starter.cpp` passing all 24 test cases (add the 13 functions to your library as `teleop.cpp` if you keep a C++ library); Deliverable 2 is `experiments.cpp` plus your plotting script; the report, reflection and AI-use note are unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../teleop.json` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| every simulation test fails with `TODO: follower_pd` | the given `simulate` calls your Part B and C functions | finish Checkpoints 2 and 3 first |
| `delay_margin` gives a negative or too small `tau` | `std::fmod` keeps the sign of its argument, unlike Python's `%` | add `2 * PI` when the result is negative |
| `delay_margin` returns NaN instead of a crossing | `std::sqrt` of a negative discriminant or of a negative root | skip complex roots and roots $\omega^2 \le 0$ |
| "Eigen: wrong matrix size or index out of range" | e.g. `E_in` / `E_out` of `port_energies` not sized (`Eigen::VectorXd E_in(P.size());`) | check the sizes; the test program does not crash, the test simply fails |
| Results wrong only in some entries, no compile error | an Eigen expression stored in `auto` (`auto d = a - b;` keeps references to temporaries) | write the type: `Eigen::VectorXd d = ...;` |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
