#!/usr/bin/env python3
"""Configure, build and run the C++ auto-checks of one lab, and report one result per doctest test case.

    python sbeg_cpp_run.py <lab cpp folder> [--impl starter|solution] [--build-dir DIR] [-tc PATTERN] [--clean]

Used by run_lab.py --cpp (student lab repositories), by the GitHub Classroom C++ autograder (test_cpp_track.py) and
by the course CI.  Works on Windows (MSVC), macOS (Apple clang) and Linux (GCC/clang):

* CMake and Ninja are taken from PATH or from the Python environment (pip install cmake ninja).
* macOS / Linux: Ninja (or Makefiles) with the system compiler.  Windows: Ninja + MSVC when `cl` is on PATH (the
  "x64 Native Tools Command Prompt"); otherwise the Visual Studio generator (CMake finds Visual Studio / Build Tools
  2022 by itself); without Visual Studio, Ninja + MinGW g++ with a statically linked runtime.
* The test program (lab_tests, student_tests or repair_tests) is run with doctest's JUnit reporter, so every test case
  is scored separately.  If the program crashes, every test case is re-run on its own so that one crash costs one case.
"""
import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

PROGRAMS = ("lab_tests", "student_tests", "repair_tests")
WIN = sys.platform == "win32"
TIMEOUT_BUILD, TIMEOUT_ALL, TIMEOUT_ONE = 900, 600, 120


def _tool(name):
    """Executable from PATH, else from this Python's scripts folder (pip install cmake ninja)."""
    found = shutil.which(name)
    if found:
        return found
    for d in {sysconfig.get_path("scripts"), os.path.dirname(sys.executable)}:
        for ext in ((".exe", "") if WIN else ("",)):
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


class Result:
    def __init__(self):
        self.build_ok = False
        self.cases = []          # list of (name, passed: bool, message)
        self.log = ""

    @property
    def passed(self):
        return sum(1 for _, ok, _ in self.cases if ok)

    @property
    def total(self):
        return len(self.cases)


def _run(cmd, cwd=None, timeout=None, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return None, out + f"\n[timed out after {timeout} s]"
    except OSError as e:
        return -999, str(e)


def _plan(ninja):
    """Generator plans to try, best first: (name, extra configure args, multi-config)."""
    if not WIN:
        return [("ninja", ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"], False)] \
            if ninja else [("default", ["-DCMAKE_BUILD_TYPE=RelWithDebInfo"], False)]
    plans = []
    if ninja and shutil.which("cl"):  # "x64 Native Tools Command Prompt": MSVC on PATH
        # Release: RelWithDebInfo (/Zi) makes parallel MSVC builds queue on one shared PDB server (mspdbsrv)
        plans.append(("ninja-msvc", ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=Release",
                                     "-DCMAKE_CXX_COMPILER=cl"], False))
    # a normal terminal: CMake finds Visual Studio / Build Tools 2022 by itself
    plans.append(("visual-studio", ["-A", "x64"], True))
    gxx = os.environ.get("CXX") or shutil.which("g++")
    if ninja and gxx:  # MinGW only (no Visual Studio): link the runtime statically so no DLL mismatch can occur
        plans.append(("ninja-mingw", ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                                      f"-DCMAKE_CXX_COMPILER={gxx}", "-DCMAKE_EXE_LINKER_FLAGS=-static"], False))
    return plans


def build(cpp_dir, impl="starter", build_dir=None, clean=False, support=None):
    """Configure + build. Returns (ok, build_dir, multi_config, log)."""
    cpp_dir = Path(cpp_dir).resolve()
    build_dir = Path(build_dir) if build_dir else cpp_dir.parent / "build" / f"cpp-{impl}"
    build_dir = build_dir.resolve()
    cmake = _tool("cmake")
    if not cmake:
        return False, build_dir, False, "CMake not found: run  python -m pip install cmake ninja  (or see the handout)."
    if clean and build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    log, out = "", ""
    stamp = build_dir / ".sbeg_plan"
    for name, args, multi in _plan(_tool("ninja")):
        # reuse the folder only if it was configured with the same plan
        if build_dir.exists() and (not stamp.exists() or stamp.read_text().strip() != name):
            shutil.rmtree(build_dir, ignore_errors=True)
        cfg = [cmake, "-S", str(cpp_dir), "-B", str(build_dir), f"-DLAB_IMPL={impl}"] + args
        if support:
            cfg.append(f"-DSBEG_CPP_SUPPORT={support}")
        log += f"$ {' '.join(cfg)}\n"
        rc, out = _run(cfg, timeout=TIMEOUT_BUILD)
        log += out
        if rc == 0:
            stamp.write_text(name)
            break
        if "CMakeLists.txt" in out and "Error in" in out and "compiler" not in out.lower():
            break  # an error in the lab's own CMake files: another generator will not help
        shutil.rmtree(build_dir, ignore_errors=True)
    else:
        hint = ("\nNo working C++ compiler found. Windows: install Visual Studio Build Tools (2022 or 2026) (workload 'Desktop "
                "development with C++'); macOS: xcode-select --install; Ubuntu: sudo apt install build-essential.")
        return False, build_dir, False, log + hint
    if rc != 0:
        return False, build_dir, False, log
    bld = [cmake, "--build", str(build_dir), "--parallel"] + (["--config", "Release"] if multi else [])
    log += f"$ {' '.join(bld)}\n"
    rc, out = _run(bld, timeout=TIMEOUT_BUILD)
    log += out
    return rc == 0, build_dir, multi, log


def _program(build_dir):
    for name in PROGRAMS:
        for p in (build_dir / (name + (".exe" if WIN else "")), build_dir / "Release" / (name + ".exe")):
            if p.is_file():
                return p
    return None


def _list_cases(prog, filt):
    cmd = [str(prog), "--list-test-cases", "--no-colors"] + ([f"-tc={filt}"] if filt else [])
    rc, out = _run(cmd, timeout=60)
    lines, sep = out.splitlines(), 0
    names = []
    for ln in lines:
        if ln.startswith("====="):
            sep += 1
            continue
        if sep == 2 and ln.strip():
            names.append(ln.rstrip())
    return names


def _esc(name):
    return name.replace("\\", "\\\\").replace(",", "\\,")


def _parse_junit(path):
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        bad = [x for x in tc if x.tag in ("failure", "error")]
        if tc.get("status") == "skipped" or any(x.tag == "skipped" for x in tc):
            continue
        msg = "\n".join(filter(None, ((b.get("message") or "") + "\n" + (b.text or "") for b in bad))).strip()
        cases.append((tc.get("name", "?"), not bad, msg))
    # doctest writes one <testcase> per (test case, subcase); merge entries with the same name
    merged = {}
    for name, ok, msg in cases:
        if name in merged:
            ok0, msg0 = merged[name]
            merged[name] = (ok0 and ok, "\n".join(filter(None, (msg0, msg))))
        else:
            merged[name] = (ok, msg)
    return [(n, ok, m) for n, (ok, m) in merged.items()]


def run_tests(build_dir, filt=None, cwd=None):
    """Run the test program; returns (cases, log)."""
    prog = _program(Path(build_dir))
    if prog is None:
        return [], "No test program (lab_tests / student_tests / repair_tests) was built."
    names = _list_cases(prog, filt)
    diag = f"[test program: {prog}; {len(names)} test case(s) listed]\n"
    xml = Path(build_dir) / "results.xml"
    if xml.exists():
        xml.unlink()
    cmd = [str(prog), "--reporters=junit", f"--out={xml}", "--no-colors"] + ([f"-tc={filt}"] if filt else [])
    rc, out = _run(cmd, cwd=cwd or prog.parent, timeout=TIMEOUT_ALL)
    out = diag + f"[exit code {rc}; report {'found' if xml.exists() else 'missing'}]\n" + out
    cases = []
    crashed = rc is None or rc not in (0, 1)
    if not crashed:
        try:
            cases = _parse_junit(xml)
        except (ET.ParseError, OSError):
            crashed = True
    if crashed or (names and len(cases) < len(names)):
        # a crash, a hang or an incomplete report: run each test case on its own
        cases, out = [], out + "\n[the test program stopped early: running each test case separately]\n"
        for n in names:
            one = Path(build_dir) / "one.xml"
            if one.exists():
                one.unlink()
            r1, o1 = _run([str(prog), "--reporters=junit", f"--out={one}", "--no-colors", f"-tc={_esc(n)}"],
                          cwd=cwd or prog.parent, timeout=TIMEOUT_ONE)
            got = []
            if r1 in (0, 1):
                try:
                    got = _parse_junit(one)
                except (ET.ParseError, OSError):
                    got = []
            if got:
                cases += got
            else:
                why = "timed out" if r1 is None else f"crashed (exit code {r1})"
                cases.append((n, False, f"the test program {why} in this test case\n{o1[-1500:]}"))
    return cases, out


def run_lab(cpp_dir, impl="starter", build_dir=None, filt=None, clean=False, support=None):
    res = Result()
    ok, bdir, _, log = build(cpp_dir, impl, build_dir, clean, support)
    res.log = log
    if not ok:
        return res
    res.build_ok = True
    res.cases, out = run_tests(bdir, filt)
    res.log += out
    return res


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cpp_dir")
    ap.add_argument("--impl", default=os.environ.get("LAB_IMPL", "starter"), choices=("starter", "solution"))
    ap.add_argument("--build-dir")
    ap.add_argument("-tc", "--test-case", dest="filt", help="only test cases whose name matches (doctest -tc)")
    ap.add_argument("--clean", action="store_true", help="delete the build folder first")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    res = run_lab(a.cpp_dir, a.impl, a.build_dir, a.filt, a.clean)
    if not res.build_ok:
        print(res.log[-6000:])
        print("Score: 0 — the C++ code did not build (see the compiler messages above).")
        return 2
    if not a.quiet:
        print("\n".join(l for l in res.log.splitlines() if "warning" in l.lower())[-3000:])
    for name, ok, msg in res.cases:
        print(("PASS  " if ok else "FAIL  ") + name)
        if not ok and msg and not a.quiet:
            print("      " + "\n      ".join(msg.strip().splitlines()[:6]))
    if res.total:
        print(f"Score: {res.passed}/{res.total} C++ checks passed = {round(100 * res.passed / res.total)} %")
        if res.passed < res.total and "TODO" in res.log:
            print("(Checks fail with 'TODO: <function>' until you write the functions in starter.cpp.)")
    else:
        print(res.log[-6000:])
        print("No C++ checks ran — see the messages above.")
    return 0 if res.total and res.passed == res.total else 1


if __name__ == "__main__":
    sys.exit(main())
