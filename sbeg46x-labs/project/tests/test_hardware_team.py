"""YOUR hardware acceptance tests HWk-1 ... HWk-4 (scenario_sheets.md), run on the raw logs that you
commit to results/hardware/ after each supervised session. Everything here is SKIPPED until the first
log is committed.

Name every log <date>_<session>_<test>.csv, for example 2026-11-25_session2_HW3-2.csv (session1,
session2 or rebooked), in the format of hardware.write_log. Never edit a log by hand: the raw CSV is the
evidence that the session took place, and the markers re-run these tests on it.

The generic checks below run on every log. Add one test function per hardware test of your scenario
underneath (template at the end) and compute its numbers with the helpers of hardware.py and hwtasks.py.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from sbeg46x_project import hardware, hwtasks

ROOT = Path(__file__).resolve().parents[1]
LOGS = sorted((ROOT / "results" / "hardware").glob("*.csv"))
NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_(session1|session2|rebooked)_HW[1-6]-[1-4](_[A-Za-z0-9-]+)?\.csv$")

pytestmark = pytest.mark.skipif(not LOGS, reason="no hardware logs committed yet (results/hardware/)")


@pytest.mark.parametrize("path", LOGS, ids=[p.name for p in LOGS])
def test_log_is_named_and_readable(path):
    assert NAME.match(path.name), f"{path.name}: use <date>_<session>_HWk-n[_tag].csv"
    log = hardware.load_log(path)
    assert len(log["t"]) > 10 and np.all(np.diff(log["t"]) > 0)


@pytest.mark.parametrize("path", LOGS, ids=[p.name for p in LOGS])
def test_haptic_force_never_above_the_firmware_limit(path):
    """For haptic-device logs the force column is the force applied by the motor (after saturation):
    it can never exceed 2 N. A value above it means the log or the firmware is wrong."""
    log = hardware.load_log(path)
    if "force" in log:
        assert np.max(np.abs(log["force"])) <= hwtasks.HAPKIT_FORCE_MAX + 1e-9


# --------------------------------------------------------------------------- template (edit)
def test_HWk_1_template():
    """Replace by your HWk-1 test: load the session-2 log(s), compute the measured value(s) and assert the
    provisional threshold of hwtasks.HW[k] (or the confirmed value published in Week 9)."""
    pytest.skip("write your HWk-1 ... HWk-4 tests here (see the docstring)")
