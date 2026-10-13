import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training", "needle")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_arguments(arguments):
    if not isinstance(arguments, dict):
        return {}
    return {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in sorted(arguments.items())}


def same_call(predicted, expected):
    if not isinstance(predicted, dict):
        return False
    if predicted.get("name") != expected.get("name"):
        return False
    return norm_arguments(predicted.get("arguments")) == norm_arguments(expected.get("arguments"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(TRAIN_DIR, "tuned.cact"))
    parser.add_argument("--data", default=os.path.join(TRAIN_DIR, "val.jsonl"))
    parser.add_argument("--tools", default=os.path.join(TRAIN_DIR, "tools.json"))
    parser.add_argument("--out", default=os.path.join(TRAIN_DIR, "eval.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import needle

    with open(args.tools, encoding="utf-8") as f:
        tools_text = json.dumps(json.load(f), separators=(",", ":"), ensure_ascii=False)
    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    agent = needle.Needle(tools=tools_text, weights=args.weights, auto_date=False, stateless=True)

    exact = 0
    tool_ok = 0
    empty_ok = 0
    empty_total = 0
    per_tool = {}
    failures = []
    started = time.time()

    for i, row in enumerate(rows, start=1):
        query = row["query"]
        expected = row["answers"]
        try:
            response = agent.complete(query)
            predicted = response.get("function_calls") or []
        except Exception as exc:
            predicted = []
            failures.append({"query": query, "error": str(exc)})

        row_exact = len(expected) == len(predicted) and all(
            same_call(p, e) for p, e in zip(predicted, expected)
        )
        if row_exact:
            exact += 1
            if not expected:
                empty_ok += 1
        else:
            name_ok = len(expected) == len(predicted) and all(
                p.get("name") == e.get("name") for p, e in zip(predicted, expected)
            )
            if name_ok:
                tool_ok += 1
            failures.append({"query": query, "expected": expected, "predicted": predicted})
        if not expected:
            empty_total += 1

        for answer in expected:
            bucket = per_tool.setdefault(answer["name"], {"total": 0, "exact": 0})
            bucket["total"] += 1
            if row_exact:
                bucket["exact"] += 1

        if i % 25 == 0:
            print("  {}/{} exact so far {}".format(i, len(rows), exact), flush=True)

    elapsed = time.time() - started
    n = max(len(rows), 1)
    report = {
        "weights": args.weights,
        "examples": len(rows),
        "exact_matches": exact,
        "exact_accuracy": round(exact / n, 4),
        "tool_name_accuracy": round((exact + tool_ok) / n, 4),
        "empty_answers_total": empty_total,
        "empty_answers_correct": empty_ok,
        "per_tool": per_tool,
        "seconds": round(elapsed, 1),
        "ms_per_query": round(1000.0 * elapsed / n, 1),
        "failures": failures[:40],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("examples          {}".format(len(rows)))
    print("exact match       {}/{}  ({:.1%})".format(exact, len(rows), exact / n))
    print("tool name match   {}/{}  ({:.1%})".format(exact + tool_ok, len(rows), (exact + tool_ok) / n))
    print("off-topic right   {}/{}".format(empty_ok, empty_total))
    print("speed             {} ms/query".format(report["ms_per_query"]))
    print("per tool          {}".format(json.dumps(per_tool)))
    print()
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
