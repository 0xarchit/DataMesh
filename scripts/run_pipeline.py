import json
import os

from datamesh.pipeline import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    report = run(
        raw_dir=os.path.join(ROOT, "data", "raw"),
        canonical_dir=os.path.join(ROOT, "data", "canonical"),
        intermediate_dir=os.path.join(ROOT, "data", "intermediate"),
        use_llm=True,
    )
    print(json.dumps(report, indent=2))
