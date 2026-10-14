#!/usr/bin/env python3
"""Run the auto-checks of one lab and print your score.

    python run_lab.py            list the labs
    python run_lab.py 3          run Lab 3 (or give the folder name, e.g. labs/lab-03-...)
    python run_lab.py 3 -k name  run only the tests whose name contains 'name'
    python run_lab.py 3 --cpp    C++ track: configure + build the lab's cpp/ folder with CMake, run its C++ checks
"""
import glob, os, re, subprocess, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
labs = sorted(glob.glob(os.path.join(HERE, "labs", "*", "")) + glob.glob(os.path.join(HERE, "enrichment", "*", "")))

def find(arg):
    if os.path.isdir(arg): return os.path.abspath(arg)
    for d in labs:
        name = os.path.basename(d.rstrip(os.sep))
        m = re.match(r"(lab|hw)-(\d+)", name)
        if (m and arg.isdigit() and int(m.group(2)) == int(arg) and m.group(1) == "lab") or arg.lower() == name.lower():
            return d
    if arg.lower().startswith("hw"):
        for d in labs:
            if os.path.basename(d.rstrip(os.sep)).startswith("hw-"): return d
    sys.exit(f"No lab '{arg}'. Run 'python run_lab.py' to see the list.")

CPP = "--cpp" in sys.argv
if CPP:
    sys.argv.remove("--cpp")
if len(sys.argv) < 2:
    print("Labs in this repository:")
    for d in labs: print("  ", os.path.relpath(d, HERE))
    print("\nRun one with:  python run_lab.py <number or folder>")
    sys.exit(0)
d = find(sys.argv[1])
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
env.setdefault("LAB_IMPL", "starter")
if CPP:
    cpp = os.path.join(d, "cpp")
    if not os.path.isfile(os.path.join(cpp, "CMakeLists.txt")):
        sys.exit(f"{os.path.relpath(d, HERE)} has no C++ track (no cpp/CMakeLists.txt).")
    rest = sys.argv[2:]
    if "-k" in rest:
        i = rest.index("-k"); rest[i:i + 2] = ["-tc", f"*{rest[i + 1]}*"]
    print(f"Building and running the C++ checks of {os.path.relpath(d, HERE)} ({env['LAB_IMPL']}.cpp) ...\n", flush=True)
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "cpp_support", "tools", "sbeg_cpp_run.py"), cpp,
                              "--impl", env["LAB_IMPL"]] + rest, env=env))
lib = os.path.join(HERE, "library")
env["PYTHONPATH"] = os.pathsep.join(p for p in (lib, env.get("PYTHONPATH", "")) if p)
cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rN"] + sys.argv[2:]
print(f"Running the checks of {os.path.relpath(d, HERE)} ...\n")
p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
out = p.stdout + p.stderr
print(out[-4000:])
m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", out)}
passed = m.get("passed", 0); failed = m.get("failed", 0) + m.get("error", 0) + m.get("errors", 0)
total = passed + failed
if total:
    print(f"Score: {passed}/{total} checks passed = {round(100 * passed / total)} %")
    if failed and "NotImplementedError" in out:
        print("(Checks fail with NotImplementedError until you complete the # TODOs.)")
else:
    print("No checks ran — see the messages above.")
sys.exit(0 if failed == 0 else 1)
