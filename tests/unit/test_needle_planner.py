import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datamesh import qa
from datamesh.core import planner_needle


def catalogue_names():
    path = os.path.join(ROOT, "training", "needle", "tools.json")
    with open(path, encoding="utf-8") as f:
        return {tool["name"] for tool in json.load(f)}


def check_tool_spec_covers_catalogue():
    names = catalogue_names()
    mapped = set(planner_needle.TOOL_SPEC)
    assert names == mapped, "tools.json and TOOL_SPEC disagree: {} vs {}".format(
        sorted(names), sorted(mapped)
    )


def check_structural_guard():
    ok = qa.needle_spec_ok
    assert ok({"op": "distinct_count", "entity": None}, "how many vendors are there")
    assert ok({"op": "sum", "entity": "Acme Ltd"}, "total spend with Acme Ltd")
    assert ok({"op": "sum", "entity": None}, "total spend across all vendors")
    assert ok({"op": "lookup", "entity": "Acme Ltd"}, "what city is Acme Ltd in")
    assert not ok({"op": "lookup", "entity": None}, "how many vendors")
    assert not ok({"op": "count", "entity": None}, "how many transactions")
    assert not ok(None, "anything")


if __name__ == "__main__":
    check_tool_spec_covers_catalogue()
    check_structural_guard()
    print("needle planner checks passed")
