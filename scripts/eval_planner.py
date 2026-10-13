import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datamesh import qa
from datamesh.core import planner_needle

RESULTS = os.path.join(ROOT, "research", "results")


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def expected(case):
    tool = case.get("tool")
    if not tool or tool == "none":
        return None
    mapped = planner_needle.TOOL_SPEC.get(tool)
    if mapped is None:
        return None
    op, column = mapped
    return {"op": op, "column": column, "entity": case.get("vendor")}


def key_of(spec, vendors):
    if spec is None:
        return None
    entity = spec.get("entity")
    vendor = qa.find_vendor(entity, vendors) if entity else None
    return (spec.get("op"), spec.get("column"), vendor["id"] if vendor else None)


def score(cases, vendors, fn, label):
    rows = []
    for case in cases:
        want = key_of(expected(case), vendors)
        got = key_of(fn(case["query"]), vendors)
        rows.append(
            {
                "query": case["query"],
                "category": case.get("category", "unknown"),
                "want": want,
                "got": got,
                "correct": int(want == got),
            }
        )
    n = len(rows) or 1
    by_cat = {}
    for r in rows:
        b = by_cat.setdefault(r["category"], {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += r["correct"]
    return {
        "label": label,
        "examples": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "by_category": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4),
            }
            for k, v in sorted(by_cat.items())
        },
        "failures": [r for r in rows if not r["correct"]][:60],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=os.path.join(ROOT, "training", "needle", "planner_holdout.jsonl")
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    vendors, _ = qa.load_canonical(os.path.join(ROOT, "data", "canonical"))
    cases = load_cases(args.data)

    model = score(cases, vendors, planner_needle.plan, "model only")
    integrated = score(
        cases,
        vendors,
        lambda q: qa.plan(q, vendors, use_llm=False, use_needle=True)[0],
        "integrated planner chain",
    )

    report = {
        "weights": planner_needle.WEIGHTS_PATH,
        "data": os.path.relpath(args.data, ROOT),
        "cases": len(cases),
        "model_only": model,
        "integrated": integrated,
    }
    out = args.out or os.path.join(RESULTS, "planner_eval.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for arm in (model, integrated):
        print(
            "{}: {}/{} = {:.1%}".format(
                arm["label"], arm["correct"], arm["examples"], arm["accuracy"]
            )
        )
        for cat, b in arm["by_category"].items():
            print("    {:<16} {:>3}/{:<3}  {:.0%}".format(cat, b["correct"], b["total"], b["accuracy"]))
        print()
    print("model-only failures:")
    for f in model["failures"][:15]:
        print("    want {} got {}  | {}".format(f["want"], f["got"], f["query"][:60]))
    print()
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
