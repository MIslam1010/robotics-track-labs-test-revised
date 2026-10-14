#!/usr/bin/env python3
"""Run this lab's auto-checks and print your score.

    python run_lab.py            Python checks (pytest)
    python run_lab.py --cpp      C++ checks: configure + build cpp/ with CMake, run the doctest cases
    (a lab number before the option, as in the handout, is accepted and ignored)
"""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not re.fullmatch(r"\d+|hw", a)]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
env.setdefault("LAB_IMPL", "starter")
if "--cpp" in args:
    args.remove("--cpp")
    V = env.get("LAB_VERSION", "")
    cpp = f"cpp_{V}" if V and os.path.isdir(os.path.join(HERE, f"cpp_{V}")) else "cpp"
    if "-k" in args:
        i = args.index("-k"); args[i:i + 2] = ["-tc", f"*{args[i + 1]}*"]
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "cpp_support", "tools", "sbeg_cpp_run.py"),
                              os.path.join(HERE, cpp), "--impl", env["LAB_IMPL"]] + args, env=env))
env["LAB_LANGUAGE"] = "python"
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + args, cwd=HERE, env=env))
