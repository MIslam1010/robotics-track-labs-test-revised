"""Grade one assignment repository exactly as Classroom 50 does: its runner.py executes our declarative tests.

    python c50_check.py <runner.py> <tests.json> <mode-label> <expect: full|partial|ran>
Prints the per-test outcome and the total; exit 1 if the total is not what was expected.
"""
import importlib.util, json, pathlib, sys, tempfile
runner_path, tests_path, label, expect = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("c50runner", runner_path)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
tests = json.load(open(tests_path))
fixtures = pathlib.Path(tempfile.mkdtemp())
total = maxp = 0
for t in tests:
    out = runner.execute_test(t, cwd=pathlib.Path.cwd(), fixtures_dir=fixtures)
    total += out["score"]; maxp += t["points"]
    print(f"  {label:22s} {t['name'][:40]:40s} {'PASS' if out['passed'] else 'FAIL'} {out['score']:>3}/{t['points']:<3} {out.get('detail', '')[:90]}")
    if not out["passed"] and t["type"] == "python":  # names + exception type only (public log: no tracebacks)
        for line in (out.get("capture", {}).get("output", "") or "").splitlines():
            if line.startswith(("FAILED ", "ERROR ")):
                print("      " + line.split(" - ")[0] + (" - " + line.split(" - ")[1].split(":")[0] if " - " in line else ""))
ok = (total == maxp) if expect == "full" else (total < maxp) if expect == "partial" else True
print(f"{label}: {total}/{maxp}  ({'as expected' if ok else 'NOT as expected: ' + expect})")
sys.exit(0 if ok else 1)
