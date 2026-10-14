#!/usr/bin/env python3
"""Package your work on one lab for upload to the course site (Moodle/LMS).

    python submit.py 3 --id 1234567
Creates submissions/<id>_<lab>.zip with your code (starter*.py and any files you added), your AI-use note and your
reflection. Tests and data are not included (the course team has the originals).
"""
import argparse, glob, os, re, subprocess, sys, zipfile
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("lab"); ap.add_argument("--id", required=True, help="your student ID")
ap.add_argument("--cpp", action="store_true", help="you did the lab in C++ (score from the C++ checks)")
a = ap.parse_args()
sys.argv = ["run_lab.py", a.lab]
labs = sorted(glob.glob(os.path.join(HERE, "labs", "*", "")) + glob.glob(os.path.join(HERE, "enrichment", "*", "")))
d = None
for x in labs:
    name = os.path.basename(x.rstrip(os.sep)); m = re.match(r"lab-(\d+)", name)
    if (m and a.lab.isdigit() and int(m.group(1)) == int(a.lab)) or a.lab.lower() in (name.lower(), "hw") and (name.startswith("hw-") or a.lab.lower() == name.lower()):
        d = x; break
if d is None: sys.exit(f"No lab '{a.lab}'.")
name = os.path.basename(d.rstrip(os.sep))
note = os.path.join(d, "AI_USE_NOTE.md")
if not os.path.exists(note) or open(note, encoding="utf8").read().strip().startswith("<!-- TEMPLATE"):
    sys.exit(f"Fill in {os.path.relpath(note, HERE)} first (tool, prompts, what you kept — or 'No AI tools used').")
r = subprocess.run([sys.executable, os.path.join(HERE, "run_lab.py"), a.lab] + (["--cpp", "--quiet"] if a.cpp else []),
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
score = (re.findall(r"Score: .*", r.stdout) or ["Score: not available"])[-1]
os.makedirs(os.path.join(HERE, "submissions"), exist_ok=True)
out = os.path.join(HERE, "submissions", f"{a.id}_{name}.zip")
skip = ("test_", "conftest")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("data", "__pycache__", ".pytest_cache", "figures", "build")]
        for f in files:
            if f.endswith((".py", ".md", ".txt", ".ipynb", ".cpp", ".hpp", ".h", ".csv")) and not f.startswith(skip) \
               and f not in ("HANDOUT.md", "README.md"):
                p = os.path.join(root, f); z.write(p, os.path.relpath(p, d))
    z.writestr("SCORE.txt", score + ("  (C++ track)" if a.cpp else "  (Python track)") + "\n")
print(f"Created {os.path.relpath(out, HERE)}  ({score}). Upload it on the course site.")
