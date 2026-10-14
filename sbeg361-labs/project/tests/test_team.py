"""Checks the team's TEAM acceptance tests once implemented in team_design.py."""
import pytest
import importlib
import os

td = importlib.import_module(os.environ.get("TEAM_IMPL", "team_design"))


@pytest.mark.parametrize("sid", sorted(td.EVALUATORS))
def test_team_acceptance(sid):
    try:
        res = td.EVALUATORS[sid]()
    except NotImplementedError:
        pytest.skip(f"S{sid} not implemented by this team")
    for tid in td.TEAM_TESTS[sid]:
        assert tid in res, f"missing result for {tid}"
        value, passed = res[tid]
        assert passed, f"{tid} failed (value = {value})"
