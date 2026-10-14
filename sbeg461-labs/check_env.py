#!/usr/bin/env python3
"""Check that your computer is ready for the labs (run once in Week 1)."""
import importlib, platform, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ok = True
print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
if sys.version_info < (3, 10):
    print("  ✗ Python 3.10 or newer is needed (3.11 recommended)."); ok = False
NEEDED = {'numpy': 'arrays', 'scipy': 'numerical tools', 'modern_robotics': "the textbook's library", 'pytest': 'the auto-checks', 'matplotlib': 'plots', 'pinocchio': 'fast dynamics (Pinocchio)', 'sympy': 'symbolic maths', 'toppra': 'time-optimal time scaling'}
OPTIONAL = {'rclpy': 'ROS 2 parts — use the course Docker image or lab PC', 'pymycobot': 'real-arm driver — lab PCs only'}
for mod, why in NEEDED.items():
    try:
        m = importlib.import_module(mod); print(f"  ✓ {mod:18s} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  ✗ {mod:18s} missing — {why}. Run the setup script again."); ok = False
for mod, why in OPTIONAL.items():
    try:
        importlib.import_module(mod); print(f"  ✓ {mod:18s} (optional)")
    except Exception:
        print(f"  – {mod:18s} not installed (optional: {why})")
import os, shutil, subprocess, sysconfig
def _tool(n):
    p = shutil.which(n)
    if p: return p
    for d in (sysconfig.get_path("scripts"), os.path.dirname(sys.executable)):
        for e in (".exe", ""):
            if os.path.isfile(os.path.join(d, n + e)): return os.path.join(d, n + e)
print("C++ track (optional — only if you do labs in C++):")
for n in ("cmake", "ninja"):
    p = _tool(n)
    print(f"  ✓ {n:18s} {subprocess.run([p, '--version'], capture_output=True, text=True).stdout.splitlines()[0] if p else ''}" if p
          else f"  – {n:18s} not found (pip install -r requirements.txt installs it)")
cxx = [c for c in (("cl",) if sys.platform == "win32" else ("c++", "g++", "clang++")) if shutil.which(c)]
vs = sys.platform == "win32" and os.path.exists(os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"))
if cxx:
    print(f"  ✓ C++ compiler       {cxx[0]}")
elif vs:
    print("  ✓ C++ compiler       Visual Studio found (used automatically by run_lab.py --cpp)")
elif sys.platform == "win32" and shutil.which("g++"):
    print("  ✓ C++ compiler       MinGW g++ (Visual Studio Build Tools recommended)")
else:
    print("  – C++ compiler       not found (Windows: Visual Studio Build Tools (2022 or 2026), 'Desktop development with C++';"
          " macOS: xcode-select --install; Ubuntu: sudo apt install build-essential)")
print("\nAll set — you can start the labs." if ok else "\nSomething is missing: see the lines marked ✗ (or use the Docker option in README.md).")
sys.exit(0 if ok else 1)
