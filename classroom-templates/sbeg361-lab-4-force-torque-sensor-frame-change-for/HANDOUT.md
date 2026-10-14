---
title: "Lab 4 — Force–torque sensor frame change for a surgical tool and an EOD gripper (Formal report 1)"
subtitle: "SBEG 361 Introduction to Robotics — Week 5"
---

# Overview

| Item | Detail |
|---|---|
| Session | Session 4, 50 min in class (aim to complete Steps 1–5, Checkpoints 1a–3b; the step times below add up to about 55 min, so Steps 6–8 may be finished afterwards) + up to 3 h to finish the code and the analysis |
| CLOs | 361-2 (exponential coordinates on SE(3), re-used from Lab 3); 361-3 (wrenches, adjoint map, F/T sensor frame change); 361-7 (implement and validate against a reference library); 361-9 (technical report, honest disclosure of tool use) |
| Code (auto-tested) | `starter.py` (you edit, 11 functions), `test_lab.py` (20 auto-checks against `modern_robotics` and MR Examples 3.26 and 3.28) |
| Data | `lap_calib.csv`, `lap_palpation.csv`, `eod_lift.json`; `make_data.py` and `robots.py` show how they were made (read them **after** the lab) |
| Assessment | Auto-checked code (tests) + **Formal report 1** (IEEE two-column, ≤ 6 pages, template and 100-point rubric at the end of this handout), due one week after the lab, before the Week 6 lab session |
| Reference | `modern_robotics` (`MatrixExp6`, `MatrixLog6`, `Adjoint`, `TransInv`) |
| Reading | MR §3.4 (printed pp. 106–109: Eqs. 3.93–3.98, Prop. 3.27, Example 3.28); revise §3.3.3 (pp. 102–106: Prop. 3.25, Eqs. 3.86–3.92, Example 3.26) from Week 4. Optional: MLS Ch. 2 §5.1–5.2 (pp. 61–66; MLS orders wrenches $(f,\tau)$) |
| AI use | Allowed with disclosure (report Appendix A) |
| Tier | 0 (simulation and recorded data only) |

The functions you write today join your `sbeg361_kinematics` library. `matrix_exp6` and `matrix_log6` are the ones you wrote in Lab 3 (Week 4): you paste them in and re-validate them in Steps 1–2, because they are the core of the product-of-exponentials forward kinematics (Lab 5, Week 6) and of numerical inverse kinematics (Lab 9); the wrench frame change returns in the statics week (Lab 8: `body_to_space_wrench(F_b, T_sb)` is your `change_wrench_frame` with $T_{bs}$, next to $\tau = J^{\mathsf T}\mathcal F$, MR Eq. 5.26), and the power pairing behind `power` is the reason $\tau = J^{\mathsf T}\mathcal F$ holds.

**Application context.**
*Biomedical:* in robot-assisted laparoscopy the surgeon cannot feel the tissue. A 6-axis force–torque (F/T) sensor between the robot flange and the instrument can restore that information, but it measures the tool's own weight as well as the tissue force, it measures them at the flange instead of at the tip, and it cannot tell a tip force from a force applied by the abdominal-wall port (trocar). You will identify the tool's mass and centre of mass, compensate gravity, move the wrench to the tip, and detect when the port is pushing on the shaft.
*Defence (protective use):* an explosive-ordnance-disposal (EOD) robot lifts a suspect package. The wrist F/T sensor tells the operator the package's weight and centre of mass (to decide how to grip and carry it) and, after a frame change to the turret base and to the front edge of the tracks, how much the package adds to the tipping moment of the vehicle.
Robots: the UR5 with the 0.30 m, 0.6 kg laparoscope (`ur5_laparoscope`, SBEG 461 project scenario 3) and the UGV-mounted EOD 6R arm (`eod_6r_ugv`, SBEG 361 project scenario S4), both provided in your lab folder as the local module `robots.py` (a self-contained copy of the course robot models, numbers unchanged). All data are **synthetic, with illustrative values**.

# Learning objectives

By the end of the lab you can:

1. Implement the matrix exponential and logarithm on SE(3) (MR Prop. 3.25 and the algorithm of §3.3.3.2) and validate them against `modern_robotics`, including the special cases $R = I$ and $\theta = \pi$.
2. Find the screw motion that carries one pose to another, in the space and body frames, and relate the two with the adjoint map.
3. Represent forces as wrenches $\mathcal F = (m, f)$, change their frame with $[\mathrm{Ad}_T]^{\mathsf T}$ (MR Eq. 3.95), and check power invariance (Eq. 3.94).
4. Identify a payload's mass, centre of mass and the sensor bias by linear least squares, and use them for gravity compensation of an F/T sensor.
5. Report the analysis in a formal IEEE-style technical report.

# Pre-lab check (complete before the session; 5 questions)

1. Write the screw axis $\mathcal S$ of a pure rotation about the $\hat z$-axis through $q = (0.1, 0, 0)$.
2. What is $e^{[\mathcal S]\theta}$ for $\mathcal S = (0, 0, 0, 1, 0, 0)$ and $\theta = 0.2$?
3. A force $f = (0, 0, -10)$ N acts at $r = (0.2, 0, 0)$ m in frame $\{a\}$. Write $\mathcal F_a$ in MR order.
4. MR writes $\mathcal V_a = [\mathrm{Ad}_{T_{ab}}]\mathcal V_b$. Complete: $\mathcal F_b = [\ \cdot\ ]^{\mathsf T}\mathcal F_a$.
5. An F/T sensor is held still in two orientations with no contact. Why do the two readings differ?

*Answers (for self-check):* 1. $(0, 0, 1,\ 0, -0.1, 0)$. 2. $(I,\ (0.2, 0, 0))$. 3. $(0,\ 2,\ 0,\ \ 0,\ 0,\ -10)$ (N m, N). 4. $[\mathrm{Ad}_{T_{ab}}]^{\mathsf T}$, the **same** $T_{ab}$. 5. Gravity on the tool is fixed in $\{s\}$ but the sensor axes rotate, so $g_f = R_{sf}^{\mathsf T}g_s$ (and the moment $r_c\times f$) changes; the bias stays the same.

# Setup

```bash
cd "<your Lab code folder>"    # or your Classroom 50 repo
source ~/Documents/Edu/Lectures/Robotics/MR-Studio/.venv/bin/activate   # or the course Docker image / JupyterHub
python -c "import numpy, modern_robotics; print('ok')"
LAB_IMPL=starter python -m pytest -q test_lab.py                # 20 failures (NotImplementedError) is expected
```

On Windows (PowerShell) use `$env:LAB_IMPL="starter"; python -m pytest -q test_lab.py`. Rules: do **not** call `modern_robotics` or SciPy inside `starter.py`. The helpers from Labs 1–3 (`vec_to_so3` … `adjoint`), the loaders and `tip_forces` are complete and may be used; `np.linalg.lstsq` is allowed in `estimate_payload`.

**F/T sensor convention (MR Example 3.28).** The sensor frame $\{f\}$ reports the total wrench of all loads acting on the bodies beyond it (tool weight, tissue, port, package), expressed in $\{f\}$, plus a constant electrical offset $b$ ("bias"). Wrench vectors and CSV columns are ordered $(m_x, m_y, m_z, f_x, f_y, f_z)$ in N m and N. Each CSV row also holds $T_{sf}$ (rotation $R_{11}\ldots R_{33}$ row by row, then $p$); you do not need forward kinematics, which is next week's topic.

# Procedure (auto-tested)

Run the tests after each step. A checkpoint is complete when the listed tests pass; ask a TA to sign it off.

## Step 1 — Matrix exponential on SE(3), from Lab 3 (4 min)

Paste your Lab 3 `matrix_exp6` (MR Eqs. 3.87–3.89) into `starter.py` (or write it again; reuse `matrix_exp3` for the rotation block) and run the tests.

**Checkpoint 1a:** `test_matrix_exp6_matches_mr`, `test_exp6_screw_through_a_point` (the screw along $\hat z$ through $q = (1, 0, 0)$, $90^\circ$, pitch 0 and 0.1 m/rad, as in Week 4).

## Step 2 — Matrix logarithm and screw motions (8 min)

Paste your Lab 3 `matrix_log6` (MR §3.3.3.2), `screw_between` and `screw_path` (Lab 3 Step 4; or write them again from the notes below).

- In `matrix_log6` use `matrix_log3`; it already handles $\theta = \pi$. Return the $4\times4$ matrix $[\mathcal S]\theta$.
- `screw_between(T1, T2, "space")` returns the unit screw $\mathcal S$ and $\theta$ with $T_2 = e^{[\mathcal S]\theta}T_1$; with `"body"`, $T_2 = T_1e^{[\mathcal S]\theta}$.
- `screw_path(T1, T2, s)` $= e^{[\mathcal S]s\theta}T_1$: the constant-screw path, a straight line in exponential coordinates.

**Checkpoint 1b:** `test_matrix_log6_matches_mr_and_inverts_exp`, `test_matrix_log6_special_cases`, `test_log6_mr_example_3_26` ($\mathcal S = (0, 0, 1, 3.366, -3.366, 0)$, $\theta = 0.5236$), `test_screw_between_space_and_body` (checks $\mathcal S_b = [\mathrm{Ad}_{T_1^{-1}}]\mathcal S_s$), `test_screw_path_needle_guide` (half-way point $(0.3121, 0.1707, 0.125)$ of the needle-guide screw from $T_1 = (I, (0.3, 0, 0.1))$ to $T_2 = (\mathrm{Rot}(\hat z, 90^\circ), (0.2, 0.3, 0.15))$).

## Step 3 — Wrenches (8 min)

Implement `point_force_wrench`, `change_wrench_frame` ($\mathcal F_b = [\mathrm{Ad}_{T_{ab}}]^{\mathsf T}\mathcal F_a$), `power` and `wrench_to_screw` (pitch, line of action, direction, magnitude; MLS Eqs. 2.69–2.71).

**Checkpoint 2:** `test_point_force_wrench_and_power`, `test_change_wrench_frame_mr_example_3_28` (the hand gives $(0, 0, -0.5, 0, -5, 0)$, the apple $(0, 0, -0.25, 0, -1, 0)$, the sum $(0, 0, -0.75, 0, -6, 0)$), `test_power_is_frame_independent`, `test_wrench_to_screw`.

## Step 4 — Identify the laparoscope and the sensor bias (8 min)

Implement `gravity_wrench`, `estimate_payload` and `compensate`. The payload model is linear in $x = (m,\ c = m\,r_c,\ b_m,\ b_f)$ (10 unknowns): with $\gamma_k = R_{sf,k}^{\mathsf T}g_s$,
$$f_k = b_f + m\,\gamma_k,\qquad m_k = b_m - [\gamma_k]\,c .$$
```python
import numpy as np, starter as lab
_, Ts, Fs = lab.load_pose_wrench_csv("lap_calib.csv")        # 12 static poses, no contact
m, r, b = lab.estimate_payload([T[:3, :3] for T in Ts], Fs)
```
You should get $m = 0.600$ kg, $r_c = (0.000, 0.000, 0.100)$ m and $b = (0.012, -0.008, 0.004,\ 0.354, -0.222, 0.604)$. The first raw reading is $(-0.006, -0.008, 0.006,\ 0.368, -0.081, 6.483)$: almost all of it is the tool's weight. Why must the 12 poses include several different orientations (what happens to the least-squares matrix if all $R_k$ are equal)?

**Checkpoint 3a:** `test_gravity_wrench_workshop_3`, `test_estimate_payload_exact_data`, `test_laparoscope_calibration`, `test_compensate_removes_gravity_and_bias`.

## Step 5 — Tip force during palpation (8 min)

`lap_palpation.csv` holds 10 s at 50 Hz while the surgeon palpates tissue three times (presses at 1–3 s, 4–6 s, 7–9 s); the robot tilts the tool slowly by up to 10°. Use `tip_forces(Ts, Fs, m, r, b, T_ft)` with $T_{ft} = (I, (0, 0, 0.30))$ to get the tip wrench $\mathcal F_t$ for every sample.

a. Plot the three components of $f_t$ and $\|f_t\|$ against time. At $t = 2$ s you should read $\mathcal F_t = (-0.012, 0.006, 0.001,\ 0.283, -0.138, -1.999)$: 2.0 N of axial push-back plus a small lateral (friction) force.
b. Repeat **without** gravity compensation (subtract only the bias). At $t = 2$ s you get $f_t = (1.123, -0.312, 3.822)$ N. Explain the error.
c. The peak $\|f_t\|$ in presses 1–2 is about 2.06 N; the RMS error against the true force (presses 1–2, from `make_data.palpation_truth`) is about 0.035 N. Where does this error come from?

**Checkpoint 3b:** `test_tip_force_during_palpation`.

## Step 6 — Is the port pushing on the shaft? (6 min)

A single point force at the tip produces **no moment about the tip**, so $m_t$ is a built-in consistency check.

a. Plot $\|m_t\|$ against time. During the peaks of presses 1 and 2 its mean is about 0.007 N m (noise); during press 3 it is about 0.055 N m. At $t = 8$ s, $\mathcal F_t = (0.066, 0.001, 0.001,\ 0.298, 0.418, -1.999)$.
b. Assume the extra load is a lateral port force $f_p = (f_{px}, f_{py}, 0)$ acting where the shaft passes through the port, at $(0, 0, z_p)$ in $\{f\}$ with $z_p = 0.18$ m (known from the port placement). Its moment about the tip is $m_t = (z_p - 0.30)\,\hat z\times f_p$. Solve for $f_p$ at $t = 8$ s and correct the tip force. You should get $f_p \approx (-0.009,\ 0.548,\ 0)$ N and a corrected tip force $\approx (0.307,\ -0.130,\ -1.999)$ N.
c. Count equations and unknowns: why can you **not** also find $z_p$ from the six sensor numbers? What would you tell the surgeon about the tip-force display during press 3?

**Checkpoint 3c:** `test_port_force_detected_by_tip_moment`.

## Step 7 — EOD robot: weigh the package (6 min)

`eod_lift.json` holds six static readings with the **empty** gripper and six in the same poses **holding** the package (wrist sensor $\{f\}$, gripper frame 0.25 m ahead along $\hat x_f$).

```python
d = lab.load_json("eod_lift.json")
Rs = [np.array(x["T_sf"])[:3, :3] for x in d["empty"]]
mg, rg, bg = lab.estimate_payload(Rs, [x["F"] for x in d["empty"]])
mt, rt, bt = lab.estimate_payload(Rs, [x["F"] for x in d["loaded"]])
```

The empty gripper: $m_g = 2.999$ kg, $r_g = (0.120, 0, 0)$ m. Gripper + package: $m = 4.994$ kg, $r = (0.196, 0.006, -0.016)$ m. Use the composite centre of mass $m\,r = m_g r_g + m_p r_p$ to get the package alone: $m_p = 1.995$ kg, $r_p = (0.311, 0.016, -0.040)$ m. Why is the package's centre of mass 6 cm beyond the gripper frame a concern for the grip?

**Checkpoint 4a:** `test_eod_gripper_and_package_identification`.

## Step 8 — EOD robot: tipping moment and power (6 min)

At the lift pose (`d["lift"]`, sensor at $(1.578, 0, 0.289)$ m in $\{s\}$, axes parallel to $\{s\}$):

a. Compensate the reading with the **gripper** parameters and the loaded bias to get the package wrench in $\{f\}$: $(-0.298, 6.076, 0.006,\ -0.105, -0.201, -19.706)$.
b. Move it to the turret base, $\mathcal F_s = [\mathrm{Ad}_{T_{fs}}]^{\mathsf T}\mathcal F_f$: $m_{s,y} = 37.1$ N m. Then to the front track edge $\{e\}$ (`front_edge["T_se"]`): $m_{e,y} = 26.3$ N m. Compare with the restoring moment of the UGV's own weight, $m g d = 220\times9.81\times0.35 = 755$ N m (illustrative), and discuss what else contributes (the arm itself, slopes, dynamic loads during motion).
c. The arm raises the package by turning joint 2 at $-0.2$ rad/s; `d["lift"]["V_f"]` is the body twist of $\{f\}$. Compute the power of the package wrench in $\{f\}$ and, after transforming both, in $\{s\}$: both are $-7.04$ W. Interpret the sign.

**Checkpoint 4b:** `test_eod_wrench_at_turret_and_front_edge`, `test_eod_power_duality`. Finally: `LAB_IMPL=starter python -m pytest -q test_lab.py` must show **20 passed**.

# Deliverables

1. `starter.py` renamed `lab4_<ID>.py`, passing `LAB_IMPL=lab4_<ID> python -m pytest -q test_lab.py` (20 tests).
2. A script or notebook `experiments.py` / `.ipynb` that regenerates every figure and table of the report from the data files.
3. **Formal report 1** (PDF, IEEE two-column, ≤ 6 pages excluding appendices) with at least Figures 1–4 and Table 1 listed in the template below.
4. Reflection (≤ 200 words, report Appendix B) and AI-use note (Appendix A).

Marking: Lab 4 counts as one of the three formal reports of the lab category (Course syllabus §4 Assessment): code and tests 30 %, report 70 % (rubric below). Be prepared for a 2–3 minute oral check: a TA may point to any line of your code or any figure and ask you to explain it; failure to explain caps the code mark at 50 %.

# Formal report 1 — template

Use the IEEE conference two-column template (LaTeX `IEEEtran`; a ready-to-fill skeleton is `report_template.tex`, or use the Word version of the IEEE template). Title: *Gravity-compensated force sensing at the tip of a laparoscopic instrument and at an EOD gripper*. Suggested lengths in brackets; total ≤ 6 pages for the abstract to the conclusion.

| Section | What to write |
|---|---|
| **Abstract** (≤ 150 words) | Problem, method (exponential coordinates, wrench frame change, least-squares payload identification), key numbers (identified mass and centre of mass, tip-force RMS error, port-force detection, tipping moment), one-sentence conclusion. |
| **Index terms** | wrench, adjoint map, force–torque sensor, gravity compensation, `<your application>`. |
| **I. Introduction** (½ page) | Why tip forces matter in laparoscopy (haptic feedback, tissue safety) and why the package's weight and centre of mass matter for an EOD robot (grip, carriage, stability). Objectives in 3 bullets. |
| **II. Theory** (1 page) | Exponential coordinates and log on SE(3) (MR Prop. 3.25, Eqs. 3.88, 3.91–3.92; Week 4); wrenches (Eq. 3.93), frame change (Eq. 3.95) derived from power invariance (Eq. 3.94); Poinsot screw of a wrench (MLS Eqs. 2.69–2.70); MR notation throughout. Keep derivations short and cite. |
| **III. Method** (1 page) | Sensor convention (MR Example 3.28); the linear payload model and its least-squares solution (show the $6\times10$ block of $A$); why several orientations are needed; compensation and transfer to the tip; the $m_t = 0$ consistency test and the threshold you chose; composite centre of mass of gripper + package; frames $\{s\}$, $\{f\}$, $\{t\}$, $\{e\}$ in a figure. |
| **IV. Verification** (½ page) | Unit tests against `modern_robotics` (`MatrixExp6`, `MatrixLog6`, `Adjoint`) and MR Examples 3.26 and 3.28; one hand-calculated check (e.g. Workshop 3 or 4). |
| **V. Results** (1½ pages) | **Table 1:** identified parameters (laparoscope, gripper, package) with true values and errors. **Fig. 1:** tip force components and $\|f_t\|$ versus time, with and without gravity compensation. **Fig. 2:** $\|m_t\|$ versus time with your detection threshold; port-force estimate for press 3. **Fig. 3:** frames of the EOD robot at the lift pose with the package wrench's line of action. **Fig. 4:** your choice (e.g. calibration residuals, effect of the number of poses, or the needle-guide screw path of Step 2). Tipping moment and power results. Every figure is referred to in the text, has labelled axes with units and a caption that says what to see. |
| **VI. Discussion** (1 page) | Sources of error (noise, too few or too similar orientations, dynamic forces during motion, sensor temperature drift); what one flange sensor can and cannot measure (tip versus port forces); safety implications for the patient (force limits in tissue handling) and for the EOD operator (grip, tipping); what a better design would do (distal sensor, port model, re-zeroing policy). |
| **VII. Conclusion** (¼ page) | Answer the objectives in 3–5 sentences with numbers. |
| **References** | MR (with sections and pages), MLS (with pages), the `modern_robotics` software; IEEE citation style. |
| **Appendix A — AI-use note** | Tool and version, what you asked, what you kept or changed, how you verified it (or "no AI tools used"). |
| **Appendix B — Reflection** | ≤ 200 words, prompts below. |
| **Appendix C — Reproduction** | Commands that regenerate every number and figure (e.g. `python experiments.py`), expected run time. |

# Formal report rubric (100 points)

| Criterion (CLO) | Points | Excellent (90–100 %) | Proficient (70–89 %) | Developing (40–69 %) | Missing / incorrect (0–39 %) |
|---|---|---|---|---|---|
| Abstract and introduction (361-9) | 10 | Clear problem, both applications motivated, measurable objectives, key numbers in the abstract | Clear but generic; few numbers | Vague objectives | Missing |
| Theory (361-2, 361-3) | 15 | Correct MR equations with numbers and notation (angular/moment first); frame-change rule derived from power invariance; subscripts consistent | Minor notation slips | Equations copied without explanation, or $[\mathrm{Ad}]$ used where $[\mathrm{Ad}]^{\mathsf T}$ is needed | Wrong or missing |
| Method (361-3, 361-7) | 15 | Payload model and least squares explained (matrix $A$, why ≥ 3 distinct orientations); compensation, tip transfer, $m_t$ test with a justified threshold, composite centre of mass | One step unclear | Steps listed without justification | Missing |
| Verification evidence (361-7) | 10 | All tests pass; comparison with `modern_robotics` and MR Examples 3.26/3.28; one hand check | Tests pass, comparison partly reported | Some tests fail or no reference | No verification |
| Results — figures and tables (361-3, 361-7) | 20 | Table 1 and Figs. 1–4 complete, labelled with units, captions state what to see, values correct (mass ±1 %, tip RMS error, port detection, tipping moment, power) | One figure weak or unlabelled | Several figures missing or unreadable | Missing |
| Discussion, safety and limits (361-3, 361-8) | 15 | Error sources quantified; tip-versus-port ambiguity explained; patient and operator safety implications; concrete design improvement | Discussion correct but partly qualitative | Descriptive only | Missing |
| Writing, format, references (361-9) | 10 | IEEE two-column, ≤ 6 pages, concise, correct citations with sections/pages, colour-blind-safe figures | Minor format issues | Over length or poor structure | Not in format |
| AI-use note, reflection, reproducibility (361-9) | 5 | Honest AI note, thoughtful reflection, `experiments.py` regenerates every result | One item weak | Items superficial | Missing |

TAs calibrate on two shared sample reports before marking (Course syllabus §4 Assessment). Late or over-length reports follow the course policy.

# Reflection prompts (Appendix B, ≤ 200 words)

- Which result surprised you most (for example, how large the tool's own weight is compared with the tissue force), and why?
- Where did a test catch a bug in your code? What would have happened in the report without it?
- An F/T sensor is often zeroed ("tared") once at the start of a procedure. After this lab, when is that safe and when is it not?
- **AI-use note:** if you used an AI assistant, which suggestion did you reject or correct, and how did you find out it was wrong?

# Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: starter` | Run pytest from inside your Lab code folder, or keep `test_lab.py` next to your file and set `LAB_IMPL` to its name without `.py`. |
| Every test fails with `NotImplementedError` | Expected until you implement the functions. |
| `matrix_exp6` correct for rotations but wrong position | You used $v\theta$ instead of $G(\theta)v$, or passed $v\theta$ to $G$ without dividing by $\theta$. |
| `matrix_log6` wrong only for large angles | $\cot(\theta/2)$: use `1 / math.tan(theta / 2)`, with the **half** angle. |
| `matrix_log6` returns `nan` for a pure translation | Test $\|\omega\theta\|$ against `TOL` **before** dividing by $\theta$ (case (a) of the algorithm). |
| Example 3.28 gives $(0, 0, +0.75, \ldots)$ or a wrong force | You used $[\mathrm{Ad}_{T_{fh}}]^{\mathsf T}$ instead of $[\mathrm{Ad}_{T_{hf}}]^{\mathsf T}$, or $[\mathrm{Ad}]$ without the transpose. |
| Mass comes out negative or the centre of mass is mirrored | Sign of the moment rows: $m_k = b_m + c\times\gamma_k = b_m - [\gamma_k]c$. |
| Mass correct but the bias absorbs the weight | Gravity was written in $\{f\}$ as $R\,g_s$ instead of $R^{\mathsf T}g_s$. |
| `estimate_payload` warns about rank | All orientations are (nearly) the same; you need at least three different orientations. |
| Tip force is 5–6 N even with no contact | Gravity not compensated, or the CoM put at the tip instead of 0.10 m. |
| Tipping moment at $\{s\}$ equals the moment at $\{f\}$ | You rotated the moment but did not move it: the $p\times f$ term of $[\mathrm{Ad}]^{\mathsf T}$ is missing. |
| Power differs between $\{f\}$ and $\{s\}$ | You transformed only the wrench or only the twist, or used $[\mathrm{Ad}]$ for both. |

# Pre-lab answers

See the self-check answers under the pre-lab questions above.

# C++ track

You may do this lab in C++17 instead of Python. The functions, data files, checkpoints, deliverables and report are the same; the auto-checks compare your results with `mrcpp` (the C++ port of `modern_robotics` in the course folder `cpp_support`), with MR Examples 3.26 and 3.28, and with the numbers of the Python reference, so both tracks are marked against identical values.

**Files** (in the `cpp` sub-folder of your Lab code folder): `starter.cpp` (your work), `lab.hpp` (declarations and documentation — do not change), `test_lab.cpp` (auto-checks, doctest), `cpp_cases.json` (reference numbers and the random poses and twists of `test_lab.py`), `CMakeLists.txt`. The data files are the same `lap_calib.csv`, `lap_palpation.csv` and `eod_lift.json` in your Lab code folder. `cpp_support` (Course resources) must sit next to your lab folders or in a parent folder of them (the build finds it automatically up to 6 levels up).

**Requirements.** A C++17 compiler — Windows: Visual Studio Build Tools (2022 or 2026), workload "Desktop development with C++"; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`. CMake and Ninja come with `pip install cmake ninja` (both are in the course `requirements.txt`).

**Build and run** (from your Lab code folder):

```bash
python run_lab.py 4 --cpp                      # configure, build, run, print Score: k/n
# or directly:
cmake -S cpp -B build -G Ninja
cmake --build build
./build/lab_tests                              # Windows: build\lab_tests.exe
./build/lab_tests -tc="*exp6*,*log6*"          # only some tests (like pytest -k)
```

With the untouched starter, the program builds and all 20 test cases fail with `TODO: <function>`: this is expected. Rebuild (`cmake --build build`) after every change.

**Functions to write** (in `starter.cpp`, same 11 functions and order as in Python): `matrix_exp6`, `matrix_log6`, `screw_between`, `screw_path` (Part A; paste your Lab 3 C++ versions if you did Lab 3 in C++); `point_force_wrench`, `change_wrench_frame`, `power`, `wrench_to_screw`, `gravity_wrench`, `estimate_payload`, `compensate` (Part B). Given and complete: the helpers from Labs 1–3 (`vec_to_so3` … `adjoint`), `load_pose_wrench_csv`, `load_json` and `tip_forces`. Do not call `mrcpp` inside the TODO functions (as `modern_robotics` in Python); an Eigen least-squares solve is allowed in `estimate_payload`.

**Checkpoints** (20 C++ test cases, one per Python test, same names). 1a: `./build/lab_tests -tc="*exp6*"`; 1b: `-tc="*log6*,*screw_between*,*screw_path*"`; 2: `-tc="*point_force*,*change_wrench*,*frame_independent*,*wrench_to_screw*"`; 3a: `-tc="*gravity_wrench*,*estimate_payload*,*laparoscope*,*compensate*"`; 3b: `-tc="*tip_force*"`; 3c: `-tc="*port_force*"`; 4a and 4b: `-tc="*eod*"`. Finally `./build/lab_tests` must show 20 passed.

**Differences from the Python version.**

| Python | C++ |
|---|---|
| 6-vector (twist, wrench), $4\times4$ and $6\times6$ arrays | `sbeg::Vector6d` (written `Vector6d` in `lab.hpp`), `Eigen::Matrix4d`, `sbeg::Matrix6d` |
| `raise ValueError(...)` | `throw std::invalid_argument("...")` (unknown frame in `screw_between`, zero wrench in `wrench_to_screw`) |
| `screw_between` returns `(S, theta)` | `std::pair<Vector6d, double>`: `auto [S, theta] = lab::screw_between(T1, T2, "body");` |
| `wrench_to_screw` returns `(h, q, s_hat, magnitude)`; `q is None`, `h = math.inf` for a pure moment | struct `WrenchScrew{h, q, s_hat, magnitude}`; `q` is a `std::optional<Eigen::Vector3d>` (empty for a pure moment), `h = std::numeric_limits<double>::infinity()` |
| `estimate_payload` returns `(mass, r_com, bias)`; `F_list` is a list of 6-vectors | struct `PayloadEstimate{mass, r_com, bias}` (`auto [m, r, b] = ...`); `F_list` is an `Eigen::MatrixXd` with one wrench per **row** ($K\times6$), `R_list` a `std::vector<Eigen::Matrix3d>` |
| `load_pose_wrench_csv` returns `(t or None, Ts, Fs)` | struct `PoseWrenchData{t, Ts, Fs}`; `t` has size 0 if the file has no time column |
| `np.linalg.lstsq(A, y)` | `A.completeOrthogonalDecomposition().solve(y)` (same minimum-norm least-squares solution) |
| `np.r_[m, f]`, `F[:3]`, `F[3:]`, `np.cross`, `np.dot` | `Vector6d F; F << m, f;`, `F.head<3>()`, `F.tail<3>()`, `a.cross(b)`, `a.dot(b)` |
| `T[:3, :3]`, `T[:3, 3]` | `T.topLeftCorner<3, 3>()`, `T.topRightCorner<3, 1>()` |
| `make_data.press_profile` and the noise-free truth used by the tests | the same formula is written in `test_lab.cpp`; the true masses, centres of mass and bias are in `cpp_cases.json` |

**Figures and tables for the report.** The plots stay outside the auto-checks. Write the analysis program in C++ and save its results with `sbeg::write_csv(path, header, matrix)` (from `<sbeg/labio.hpp>`), then plot the CSV files with Python/matplotlib or any other tool. To build such a program, add these lines at the end of `cpp/CMakeLists.txt` (below `sbeg_lab()`) and write `cpp/experiments.cpp` with a `main()`:

```cmake
add_executable(experiments experiments.cpp starter.cpp)
target_link_libraries(experiments PRIVATE sbeg_mrcpp)
target_compile_definitions(experiments PRIVATE "SBEG_LAB_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
```

Inside it, `sbeg::lab_path("../lap_palpation.csv")` finds the data from any build folder. Deliverable 1 is then your `cpp` folder with `starter.cpp` (keep the file name: the build expects it) passing all 20 test cases, and deliverable 2 is `experiments.cpp` plus your plotting script; the formal report, rubric and reflection are unchanged.

**C++ troubleshooting.**

| Symptom | Likely cause | Fix |
|---|---|---|
| `cpp_support not found` | the course folder `cpp_support` is not next to / above your Lab code folder | copy it there, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | no C++ compiler installed | install it (see Requirements); on Windows run from the "x64 Native Tools Command Prompt" if Ninja cannot find `cl` |
| `cannot open file: .../cpp_cases.json` or `.../lap_calib.csv` | Lab code folder moved after configuring | delete the `build` folder and configure again |
| "Eigen: wrong matrix size or index out of range" | e.g. a 3-vector put into a 6-vector, or `block<3, 3>` at the wrong row/column in `estimate_payload` | check the sizes; the test program does not crash, the test simply fails |
| `matrix_exp6` or `matrix_log6` returns `nan` | divided by $\theta$ before testing $\|\omega\theta\| <$ `TOL` | handle the pure-translation case first |
| Results wrong only in some entries, no compile error | an Eigen product stored in `auto` (`auto G = A * B;` keeps references to temporaries) | write the type: `Eigen::Matrix3d G = A * B;` |
| `F_list` has the wrong shape in `estimate_payload` | wrenches stored as columns | one wrench per **row**: `F_list.row(k).transpose()` is the $k$-th wrench |
| Tests still show your old result | program not rebuilt | `cmake --build build` before `./build/lab_tests` |
