"""C++ track auto-checks: builds the cpp folder with CMake and runs its doctest test cases (one pytest test each).
Only collected when the first line of LANGUAGE is cpp (see conftest.py)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cpp_support", "tools"))
import sbeg_cpp_run  # noqa: E402

V = os.environ.get("LAB_VERSION", "")
CPP = os.path.join(HERE, f"cpp_{V}") if V and os.path.isdir(os.path.join(HERE, f"cpp_{V}")) else os.path.join(HERE, "cpp")
RES = sbeg_cpp_run.run_lab(CPP, impl=os.environ.get("LAB_IMPL", "starter"))
if not RES.build_ok:
    CASES = [("cpp_build", False, "The C++ code did not build:\n" + RES.log[-6000:])]
else:
    CASES = RES.cases or [("cpp_tests_ran", False, RES.log[-3000:])]


@pytest.mark.parametrize("name,ok,msg", CASES, ids=[c[0] for c in CASES])
def test_cpp(name, ok, msg):
    assert ok, msg
