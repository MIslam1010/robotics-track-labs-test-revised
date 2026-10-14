# Lab 9 — IMU orientation fusion for an exoskeleton thigh: complementary filter vs extended Kalman filter

**SBEG 46X Medical Robotics — Week 10.** This repository is your copy of the lab. The full instructions are in
[`HANDOUT.md`](HANDOUT.md) (also on the course site as the Week 10 lab handout).

> The handout refers to files as `lab/…`: in this repository they are at the top level.

## Work locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q                                     # runs the auto-checks on your starter code
```

The tests fail with `NotImplementedError` until you complete the `# TODO`s. Commit and push as often as you like.
When you are ready, **submit**: `gh student submit` (or **Submit** in the Classroom 50 web app, classroom50.org). Each
submission is graded on GitHub with the same checks; the score appears under **Releases** in this repository. Plain
pushes are not graded, so check your work locally first.

## C++ track (optional)

You may do this lab in C++17 instead of Python (handout section *C++ track*). Write your code in `cpp/starter.cpp`,
then

```bash
python run_lab.py --cpp      # configure + build cpp/ with CMake, run the C++ checks, print your score
```

To have your **C++** code graded instead of Python, change the first line of the file `LANGUAGE` from `python` to
`cpp`, commit, push and submit. You need a C++17 compiler (Windows: Visual Studio Build Tools (2022 or 2026), workload *Desktop
development with C++*; macOS: `xcode-select --install`; Ubuntu: `sudo apt install build-essential`); CMake and Ninja
are installed by `requirements.txt`. The libraries (Eigen, doctest, JSON) are in `cpp_support/` — nothing else to
install. See `cpp_support/README.md`.

## Rules

- Lab code: AI tools are **allowed with disclosure** — add a short AI-use note to your submission (tool, prompts,
  what you kept) as the handout asks; you must be able to explain every line in the oral check (syllabus §8).
- Do not edit the test files or the data; the auto-grader uses the original copies.
