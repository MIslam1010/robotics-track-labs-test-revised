"""Run every lab of one course twice: untouched starter (must fail cleanly) and with the solution (must pass).
Usage: python ci/run_all.py sbeg361-labs"""
import glob, os, re, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP = "--cpp" in sys.argv
repo = [a for a in sys.argv[1:] if not a.startswith("--")][0]
src = os.path.join(ROOT, repo)
bad = []
def run(cwd, env):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **env)
    p = subprocess.run([sys.executable, "run_lab.py", lab_rel] + (["--cpp", "--quiet"] if CPP else []), cwd=cwd, env=e,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    out = p.stdout + p.stderr
    m = re.findall(r"Score: (\d+)/(\d+)", out)
    return (m[-1] if m else None), out
labs = sorted(os.path.relpath(d, src) for d in glob.glob(os.path.join(src, "labs", "*", "")) + glob.glob(os.path.join(src, "enrichment", "*", "")))
for lab_rel in labs:
    lab_rel = lab_rel.replace(os.sep, "/").rstrip("/")
    if CPP and not os.path.isfile(os.path.join(src, lab_rel, "cpp", "CMakeLists.txt")):
        bad.append(f"NO C++ TRACK {lab_rel}"); continue
    sc, out = run(src, {"LAB_IMPL": "starter"})
    if CPP and (sc is None or "did not build" in out):
        bad.append(f"STARTER (C++) {lab_rel}: {out[-3000:]}")
    elif sc is None or ("ModuleNotFoundError" in out and "No module named 'solution" not in out) or "ImportError" in out:
        bad.append(f"STARTER {lab_rel}: {out[-600:]}")
    tmp = tempfile.mkdtemp(); shutil.copytree(src, tmp, dirs_exist_ok=True)
    shutil.copytree(os.path.join(ROOT, "solutions", repo, lab_rel), os.path.join(tmp, lab_rel), dirs_exist_ok=True)
    sc2, out2 = run(tmp, {"LAB_IMPL": "solution"})
    ok = sc2 is not None and sc2[0] == sc2[1]
    if not ok:
        names = ""
        if not CPP:  # failing test names only (no tracebacks): re-run pytest with the summary on
            e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LAB_IMPL="solution",
                     PYTHONPATH=os.pathsep.join([os.path.join(tmp, "library"), os.environ.get("PYTHONPATH", "")]))
            q = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no", "-rfE", "-p", "no:cacheprovider"],
                               cwd=os.path.join(tmp, lab_rel), env=e, capture_output=True, text=True, timeout=1800)
            names = "\n".join(l for l in (q.stdout + q.stderr).splitlines() if l.startswith(("FAILED ", "ERROR ")))
        bad.append(f"SOLUTION {lab_rel}: {sc2} {out2[-3000:]}\n{names}")
    print(f"{lab_rel:60s} starter {sc}  solution {sc2}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
if CPP:
    sys.path.insert(0, os.path.join(ROOT, "cpp_support", "tools"))
    import sbeg_cpp_run
    code = {"sbeg361-labs": "361", "sbeg461-labs": "461", "sbeg46x-labs": "46X"}[repo]
    for V in "AB":
        d = os.path.join(ROOT, "practicals", code, f"cpp_{V}")
        if not os.path.isdir(d): continue  # the public test repository has no exam material
        r1 = sbeg_cpp_run.run_lab(d, "starter", build_dir=os.path.join(tempfile.mkdtemp(), "b"))
        r2 = sbeg_cpp_run.run_lab(d, "solution", build_dir=os.path.join(tempfile.mkdtemp(), "b"))
        print(f"practical {code} {V}: starter build {r1.build_ok} {r1.passed}/{r1.total}  solution {r2.passed}/{r2.total}", flush=True)
        if not r1.build_ok: bad.append(f"PRACTICAL {code} {V} starter does not build: {r1.log[-3000:]}")
        if not (r2.build_ok and r2.total and r2.passed == r2.total):
            bad.append(f"PRACTICAL {code} {V} solution: {r2.passed}/{r2.total} {r2.log[-3000:]} {[c for c in r2.cases if not c[1]][:5]}")
for extra in (() if CPP else ("project", "library")):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=os.path.join(src, extra), capture_output=True, text=True)
    print(f"{extra}: {(p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ''}")
    if p.returncode: bad.append(f"{extra}: {(p.stdout + p.stderr)[-800:]}")
if os.environ.get("PUBLIC_LOGS"):  # public repository: never print test output (it could quote solution code);
    # keep only the names of failing tests and the exception type
    def _names(b):
        keep = [l.split(" - ")[0] + (" - " + l.split(" - ")[1].split(":")[0] if " - " in l else "")
                for l in b.splitlines() if l.startswith(("FAILED ", "ERROR ")) or l.startswith(("FAIL  ",))]
        return b.split(":")[0] + ("".join("\n    " + k for k in keep[:10]))
    bad = [_names(b) for b in bad]
print(f"\n{len(bad)} problem(s)"); print("\n\n".join(bad))
sys.exit(1 if bad else 0)
