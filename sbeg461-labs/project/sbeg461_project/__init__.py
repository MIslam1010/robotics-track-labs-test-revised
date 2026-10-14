"""SBEG 461 course project starter package.

Six project scenarios (three biomedical, three defence: protective / medical / logistic uses only),
each described the *Modern Robotics* way: home configuration M, space screw axes Slist,
link-frame transforms Mlist and spatial inertias Glist (MR Sec. 4.1, 8.2.2, 8.3.2).

Modules
-------
models    -- scenario robot models and parameters (``load_scenario(k)``)
dyn       -- dynamics helpers built on ``modern_robotics`` and a fixed-step simulator
traj      -- reference trajectories and torque-feasibility checks
planning  -- collision checking and a baseline joint-space RRT
control   -- baseline controllers + TODO stubs for the team's controllers
metrics   -- error, torque and energy metrics used by the acceptance tests
hardware  -- hardware-log reading and sim-vs-hardware comparison (required hardware run)

All numerical parameters are ILLUSTRATIVE VALUES for teaching; they are not data of any
real product.
"""
from .models import load_scenario, SCENARIOS  # noqa: F401

__all__ = ["load_scenario", "SCENARIOS"]
