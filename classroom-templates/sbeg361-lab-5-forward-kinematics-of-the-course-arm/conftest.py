"""Selects the Python or the C++ auto-checks from the first line of the file LANGUAGE (python | cpp)."""
import os
from pathlib import Path


def _language(config):
    v = os.environ.get("LAB_LANGUAGE", "")
    f = Path(str(config.rootpath)) / "LANGUAGE"
    if not v and f.exists():
        words = f.read_text(encoding="utf-8").split()
        v = words[0] if words else ""
    return "cpp" if v.strip().lower() in ("cpp", "c++") else "python"


def pytest_configure(config):
    config.addinivalue_line("markers", "local_only: needs files outside a single-lab repository; not graded")


def pytest_collection_modifyitems(config, items):
    """When graded (SBEG_GRADING=1), leave out tests that can only run in the full course tree."""
    if not os.environ.get("SBEG_GRADING"):
        return
    drop = [i for i in items if i.get_closest_marker("local_only")]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [i for i in items if not i.get_closest_marker("local_only")]


def pytest_ignore_collect(collection_path, config):
    p = Path(str(collection_path))
    if p.is_dir():
        return True if p.name in ("cpp_support", "build") else None
    if not (p.name.startswith("test_") and p.suffix == ".py"):
        return None
    cpp_test = p.name == "test_cpp_track.py"
    return True if (_language(config) == "cpp") != cpp_test else None
