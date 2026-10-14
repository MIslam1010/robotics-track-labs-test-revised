"""Print every scenario model as Markdown tables (screw axes, home configuration, link inertias,
limits). The tables in scenario_sheets.md were generated with this script.

    python print_models.py            # all scenarios
    python print_models.py 5          # one scenario
"""
from __future__ import annotations

import sys

import numpy as np

from sbeg461_project import dyn
from sbeg461_project.models import load_scenario, spatial_inertia_to_mci


def f(x, d=4):
    x = float(x)
    if abs(x) < 0.5 * 10**-d:
        x = 0.0
    return f"{x:.{d}g}"


def vec(v, d=4):
    return "(" + ", ".join(f(x, d) for x in v) + ")"


def table(k):
    s = load_scenario(k)
    out = [f"**Scenario {k} model** ({s.n} joints, total moving mass {s.total_mass():.3g} kg, "
           f"gravity g = {vec(s.g, 3)} m/s^2 in {{s}})", ""]
    out += ["| Joint | $\\omega_i$ | $v_i$ | limits (rad) | $\\tau_{\\max}$ (N m) | "
            "$\\dot\\theta_{\\max}$ (rad/s) |", "|---|---|---|---|---|---|"]
    for i in range(s.n):
        S = s.Slist[:, i]
        out.append(f"| {i + 1} | {vec(S[:3])} | {vec(S[3:])} | [{f(s.q_limits[i, 0], 3)}, "
                   f"{f(s.q_limits[i, 1], 3)}] | {f(s.tau_max[i])} | {f(s.qdot_max[i])} |")
    R, p = s.M[:3, :3], s.M[:3, 3]
    out += ["", f"Home configuration: $M$ has $p$ = {vec(p)} m and rotation rows "
            + ", ".join(vec(r, 3) for r in R) + ".", ""]
    out += ["| Link | mass (kg) | CoM at home in {s} (m) | $I_{xx}, I_{yy}, I_{zz}$ about CoM (kg m$^2$) |",
            "|---|---|---|---|"]
    T = np.eye(4)
    for i in range(s.n):
        T = T @ s.Mlist[i]
        m, c, Ic = spatial_inertia_to_mci(s.Glist[i])
        pc = T[:3, :3] @ c + T[:3, 3]
        out.append(f"| {i + 1} | {f(m)} | {vec(pc)} | {vec(np.diag(Ic))} |")
    g0 = dyn.gravity(s, s.q_home)
    out += ["", f"Check values: $\\theta_{{home}}$ = {vec(s.q_home)}; "
            f"$g(\\theta_{{home}})$ = {vec(g0)} N m; "
            f"$M(\\theta_{{home}})$ diagonal = {vec(np.diag(dyn.mass_matrix(s, s.q_home)))} kg m$^2$."]
    return "\n".join(out)


if __name__ == "__main__":
    keys = [int(a) for a in sys.argv[1:]] or range(1, 7)
    for k in keys:
        print(table(k))
        print()
