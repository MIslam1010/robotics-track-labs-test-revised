"""Run the BASELINE of each project scenario through its acceptance tests and print the results.

    python baseline.py            # all scenarios (about 30 s)
    python baseline.py 3 5        # selected scenarios

The baselines are deliberately simple (scenario_sheets.md, "Baseline"). Your design must pass every
acceptance test and you must explain physically why the baseline does not.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from sbeg46x_project import baselines, tasks

warnings.simplefilter("ignore")

RUNS = {1: (tasks.s1_acceptance, baselines.s1_architecture), 2: (tasks.s2_acceptance, baselines.S2Baseline()),
        3: (tasks.s3_acceptance, baselines.S3Baseline()), 4: (tasks.s4_acceptance, baselines.S4Baseline()),
        5: (tasks.s5_acceptance, baselines.S5Baseline()), 6: (tasks.s6_acceptance, baselines.S6Baseline())}


def _fmt(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or list(RUNS):
        fn, design = RUNS[k]
        print(f"\n=== Scenario {k} baseline")
        for at, ok, vals in fn(design):
            print(f"{at}  {'PASS' if ok else 'fail'}  {_fmt(vals)}")
