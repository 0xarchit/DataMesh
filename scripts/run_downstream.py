import json
import os

from datamesh import downstream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "data", "canonical")
INTER = os.path.join(ROOT, "data", "intermediate")
RESULTS = os.path.join(ROOT, "research", "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    raw = downstream.raw_table(INTER, CANON)
    canon = downstream.canonical_table(CANON)
    model, model_name = downstream.build_model()
    result = {
        "model": model_name,
        "task": "classify whether a vendor is high-value from its aggregated transaction features",
        "raw_table": downstream.evaluate(raw, model),
        "consolidated_table": downstream.evaluate(canon, model),
    }
    with open(os.path.join(RESULTS, "downstream.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
