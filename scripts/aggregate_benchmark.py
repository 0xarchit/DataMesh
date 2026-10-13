import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "research", "results")


def pooled_by_type(runs):
    out = {}
    for run in runs:
        for arm, buckets in run.get("by_type", {}).items():
            for qtype, bucket in buckets.items():
                cell = out.setdefault(arm, {}).setdefault(qtype, {"correct": 0, "questions": 0})
                cell["correct"] += bucket["correct"]
                cell["questions"] += bucket["questions"]
    for arm in out:
        for qtype in out[arm]:
            cell = out[arm][qtype]
            cell["accuracy"] = round(cell["correct"] / cell["questions"], 4)
    return out


def main(paths):
    if not paths:
        print("no benchmark_*.json found in research/results")
        return
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))

    arms = list(runs[0]["arms"])
    report = {
        "runs": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "questions_per_run": runs[0]["arms"][arms[0]]["questions"],
        "model": runs[0].get("model"),
        "arms": {},
        "by_type": pooled_by_type(runs),
    }

    for arm in arms:
        accuracy = [r["arms"][arm]["accuracy"] for r in runs]
        tokens = [r["arms"][arm]["total_tokens"] for r in runs]
        seconds = [r["arms"][arm]["total_seconds"] for r in runs]
        report["arms"][arm] = {
            "accuracy_mean": round(statistics.mean(accuracy), 4),
            "accuracy_stdev": round(statistics.pstdev(accuracy), 4),
            "accuracy_min": min(accuracy),
            "accuracy_max": max(accuracy),
            "tokens_mean": round(statistics.mean(tokens), 1),
            "seconds_mean": round(statistics.mean(seconds), 2),
            "per_seed_accuracy": accuracy,
        }

    with open(os.path.join(RESULTS, "benchmark_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("seeds:", report["seeds"], " questions/run:", report["questions_per_run"])
    print()
    print("{:<34} {:>18} {:>10} {:>10}".format("arm", "accuracy mean+-sd", "tokens", "seconds"))
    for arm, s in report["arms"].items():
        print(
            "{:<34} {:>18} {:>10} {:>10}".format(
                arm,
                "{:.3f} +- {:.3f}".format(s["accuracy_mean"], s["accuracy_stdev"]),
                s["tokens_mean"],
                s["seconds_mean"],
            )
        )
    print()
    print("pooled accuracy by question type")
    types = sorted({t for arm in report["by_type"].values() for t in arm})
    print("{:<24} {:>10} {:>10} {:>10}".format("type", "ours", "rag_raw", "rag_canon"))
    for t in types:
        cells = []
        for arm in arms:
            cell = report["by_type"].get(arm, {}).get(t, {"correct": 0, "questions": 0})
            cells.append("{}/{}".format(cell["correct"], cell["questions"]))
        print("{:<24} {:>10} {:>10} {:>10}".format(t, cells[0], cells[1], cells[2]))
    print()
    print("wrote research/results/benchmark_sweep.json")


if __name__ == "__main__":
    main(sorted(glob.glob(os.path.join(RESULTS, "benchmark_[0-9]*.json"))))
