# cpp_support — shared C++ support for the SBEG 361 / 461 / 46X labs

Every lab of SBEG 361, SBEG 461 and SBEG 46X (and the 461 enrichment packages) can be done in Python (default) **or** in C++17. This folder holds everything the
C++ labs share. Keep it next to your lab folders (or in a folder above them): each lab's `CMakeLists.txt` finds it
automatically by looking for `cpp_support` or `0 Course resources/cpp_support` in its own folder and up to 6 folders
above. You can also point to it explicitly with `-DSBEG_CPP_SUPPORT=<path>` or the environment variable
`SBEG_CPP_SUPPORT`.

| Path | What it is |
|---|---|
| `include/sbeg/mrcpp.hpp`, `src/mrcpp.cpp` | `mr::` — C++ port of the `modern_robotics` Python library (all 47 functions, snake_case names) |
| `include/sbeg/labio.hpp` | `sbeg::` — load JSON / CSV, JSON ↔ Eigen, `CHECK_MAT_NEAR` test macro |
| `include/sbeg/eigen.hpp` | includes Eigen with the course settings (size errors throw instead of aborting) |
| `cmake/SBEGLab.cmake` | `sbeg_lab()`: builds the `lab_tests` program of a lab |
| `src/doctest_main.cpp` | the `main()` of every `lab_tests` program |
| `tools/sbeg_cpp_run.py` | configure + build + run one lab's C++ checks and print the score (used by `run_lab.py --cpp`, the GitHub Classroom autograder and the course CI) |
| `tests/` | self-test: every `mr::` function against the Python library |
| `third_party/` | Eigen 3.4.0, doctest 2.4.11, nlohmann/json 3.11.3 (unmodified, with licences) |

## 1. Install a compiler, CMake and Ninja (once)

| System | C++17 compiler | CMake + Ninja |
|---|---|---|
| Windows 10/11 | Visual Studio Build Tools (2022 or 2026) (free), workload **Desktop development with C++**. Build from the **x64 Native Tools Command Prompt for VS** (or use the Visual Studio generator, see below). | `pip install cmake ninja` |
| macOS | `xcode-select --install` (Apple clang) | `pip install cmake ninja` (or `brew install cmake ninja`) |
| Ubuntu / WSL | `sudo apt install build-essential` (GCC) | `pip install cmake ninja` (or `sudo apt install cmake ninja-build`) |

CMake ≥ 3.16 is required. Check with `c++ --version` (Windows: `cl`), `cmake --version`, `ninja --version`.
Nothing is downloaded during a build: all libraries are in `third_party/`.

## 2. Build and run a lab

In the **student lab repository** (from its top folder): `python run_lab.py <n> --cpp`. In a **GitHub Classroom**
lab repository: `python run_lab.py --cpp`. Anywhere else (e.g. the Lab code folder from the course site, with
`cpp_support` next to or above it), from the lab's folder:

```bash
python <path>/cpp_support/tools/sbeg_cpp_run.py cpp   # configure + build + run, prints Score: k/n
# or by hand:
cmake -S cpp -B build -G Ninja         # once (add -DCMAKE_BUILD_TYPE=Debug to debug with a debugger)
cmake --build build                    # after every change
./build/lab_tests                      # Windows: build\lab_tests.exe
./build/lab_tests -tc="*name*"         # only the test cases whose name matches (like pytest -k)
./build/lab_tests -s                   # also show successful checks
```

Without Ninja, leave out `-G Ninja` (CMake then uses Makefiles, or Visual Studio on Windows; with Visual Studio
build with `cmake --build build --config Release`; the program is still `build\lab_tests.exe`).

You write your code in `cpp/starter.cpp`. Until a function is written it throws `std::logic_error("TODO: <name>")`,
which the tests report as a failure — the program never crashes because of it. A matrix-size mistake (for example
`Matrix3d * Vector2d` at run time, or `A(3, 0)` on a 3×3) is also reported as a failed test ("Eigen: wrong matrix
size or index out of range") instead of stopping the program.

## 3. What you may use

* **Eigen** (`#include <sbeg/eigen.hpp>` — it includes `<Eigen/Dense>`): matrices and vectors. Always include
  `sbeg/eigen.hpp` (or `sbeg/mrcpp.hpp`, `sbeg/labio.hpp`) instead of `<Eigen/...>` directly.
* **mrcpp** (`#include <sbeg/mrcpp.hpp>`, namespace `mr`): exactly where the Python lab allows `modern_robotics`,
  never to replace a function the lab asks you to write. Name mapping: `MatrixExp6` → `mr::matrix_exp6`,
  `VecToso3` → `mr::vec_to_so3`, `so3ToVec` → `mr::so3_to_vec`, `FKinSpace` → `mr::fkin_space`,
  `IKinBody` → `mr::ikin_body` (returns `{thetalist, success}`), `JacobianSpace` → `mr::jacobian_space`, … .
  Screw-axis lists are 6 × n matrices (axes are columns), `Mlist`/`Glist` are `std::vector`s of 4×4 / 6×6
  matrices, trajectory matrices are N × n (row = time step), exactly as in Python.
* **labio** (`#include <sbeg/labio.hpp>`, namespace `sbeg`): `sbeg::read_csv`, `sbeg::write_csv` (write results,
  then plot them with the lab's Python plotting script), `sbeg::load_json`, `sbeg::as<Eigen::Matrix3d>(json)`.

## 4. Self-test of this folder (instructors; optional for students)

```bash
python tests/make_mrcpp_cases.py       # optional: regenerate tests/mrcpp_cases.json from modern_robotics
cmake -S tests -B build-tests -G Ninja
cmake --build build-tests
./build-tests/mrcpp_tests              # 14 test cases, 424 checks, all must pass
```

`make_mrcpp_cases.py` draws random inputs with a fixed seed (re-running reproduces the file byte for byte) for all
47 public functions of `modern_robotics.core` (plus the `pinv` helper) and stores the Python outputs. The C++
results must agree to **1e-9** (absolute, every entry). Documented exceptions: a numerical IK that *fails* is checked
only for `success == false` (after 20 diverging Newton steps the iterates are not reproducible to round-off);
`forward_dynamics_trajectory` and `simulate_control` use 1e-8 (round-off accumulates through Euler integration).
Differences from Python by design: `mr::simulate_control` does not plot; functions that return a tuple in Python
return a `std::pair` (or `mr::IKResult`); the C++ functions check list sizes and throw `std::invalid_argument`.

## 5. Licences

The course code in `include/`, `src/`, `cmake/` and `tests/` is course material of SBEG 361/461/46X. `mrcpp` follows the
algorithms of the `modern_robotics` library by Lynch, Park et al. (MIT licence). Third-party libraries in
`third_party/` are unmodified and keep their licences (see `third_party/README.md`):

| Library | Version | Licence file |
|---|---|---|
| Eigen | 3.4.0 | `third_party/eigen/COPYING.MPL2` (MPL-2.0; the build defines `EIGEN_MPL2_ONLY`) |
| doctest | 2.4.11 | `third_party/doctest/LICENSE.txt` (MIT) |
| nlohmann/json | 3.11.3 | `third_party/nlohmann/LICENSE.MIT` (MIT) |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `cpp_support not found` | put this folder next to (or above) your lab folders, or configure with `-DSBEG_CPP_SUPPORT=<path>` |
| `No CMAKE_CXX_COMPILER could be found` | install the compiler (section 1); on Windows open the x64 Native Tools Command Prompt |
| `CMake was unable to find a build program corresponding to "Ninja"` | `pip install ninja`, or leave out `-G Ninja` |
| `cannot open file: …/cpp_cases.json` | the folder was moved after configuring: delete `build/` and configure again |
| `LAB_IMPL must be 'starter' or 'solution'` | configure without `-DLAB_IMPL` (default `starter`) |
| Old results after an edit | run `cmake --build build` before `./build/lab_tests` |
