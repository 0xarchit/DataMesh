import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datamesh import qa
from datamesh.core import planner_needle

CANON = os.path.join(ROOT, "data", "canonical")
GOLD = os.path.join(ROOT, "data", "gold")
RESULTS = os.path.join(ROOT, "research", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=os.path.join(GOLD, "questions_bench.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS, "needle_benchmark.json"))
    args = parser.parse_args()

    vendors, rows = qa.load_canonical(CANON)
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    per_question = []
    sources = {}
    correct = 0
    seconds = 0.0
    for item in questions:
        started = time.time()
        spec, source = qa.plan(item["q"], vendors, use_llm=False, use_needle=True)
        answer = qa.normalize(qa.execute(spec, vendors, rows))
        elapsed = time.time() - started
        gold = qa.normalize(item["answer"])
        ok = int(answer == gold)
        correct += ok
        seconds += elapsed
        sources[source] = sources.get(source, 0) + 1
        per_question.append(
            {
                "q": item["q"],
                "type": item.get("type", "unknown"),
                "gold": item["answer"],
                "answer": answer,
                "plan": spec,
                "source": source,
                "correct": ok,
                "seconds": round(elapsed, 4),
            }
        )

    total = len(questions) or 1
    report = {
        "weights": os.path.relpath(planner_needle.WEIGHTS_PATH, ROOT),
        "questions": os.path.relpath(args.questions, ROOT),
        "correct": correct,
        "total": len(questions),
        "accuracy": round(correct / total, 4),
        "tokens": 0,
        "seconds": round(seconds, 2),
        "ms_per_question": round(1000 * seconds / total, 1),
        "sources": sources,
        "per_question": per_question,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("weights:", report["weights"])
    print(
        "needle planner: {}/{} = {:.1%}   0 tokens   {} ms/question".format(
            correct, len(questions), report["accuracy"], report["ms_per_question"]
        )
    )
    print("plan sources:", sources)
    for row in per_question:
        if not row["correct"]:
            print(
                "    FAIL {:<34} gold {:<10} got {}".format(
                    row["q"][:34], str(row["gold"])[:10], row["answer"]
                )
            )
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
