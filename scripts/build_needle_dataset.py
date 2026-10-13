import glob
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training", "needle")
SHARDS = os.path.join(TRAIN_DIR, "shards")
TOOLS_FILE = os.path.join(TRAIN_DIR, "tools.json")

LEGAL_TOOLS = {"total_spend", "count_transactions", "count_vendors", "vendor_city", "vendor_gstin", "vendor_name"}
LEGACY_DETAIL = "vendor_detail"
DETAIL_FIELDS = {"city", "gstin", "name"}
FIELD_TOOL = {"city": "vendor_city", "gstin": "vendor_gstin", "name": "vendor_name"}
MAX_CHARS = 3000
VAL_FRACTION = 0.1
SEED = 7


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def remap(line):
    for answer in line.get("answers", []):
        if answer.get("name") == LEGACY_DETAIL:
            field = (answer.get("arguments") or {}).pop("field", None)
            answer["name"] = FIELD_TOOL.get(field, "vendor_name")
    return line


def check(line):
    if not isinstance(line, dict):
        return "not an object"
    query = line.get("query")
    tools = line.get("tools")
    answers = line.get("answers")
    reasoning = line.get("reasoning", "")
    if not isinstance(query, str) or not query.strip():
        return "missing query"
    if not isinstance(tools, list) or not tools:
        return "missing tools"
    if not isinstance(answers, list):
        return "answers is not a list"
    if len(json.dumps(line)) > MAX_CHARS:
        return "too long"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return "missing reasoning"
    if len(query.split()) > 30:
        return "query too long"
    q_tokens = tokens(query)
    for answer in answers:
        if not isinstance(answer, dict):
            return "answer is not an object"
        name = answer.get("name")
        arguments = answer.get("arguments")
        if name not in LEGAL_TOOLS and name != LEGACY_DETAIL:
            return "unknown tool"
        if not isinstance(arguments, dict):
            return "arguments is not an object"
        if name == "count_vendors":
            if arguments:
                return "count_vendors must take no arguments"
            continue
        if name == "total_spend" and not arguments:
            continue
        vendor = arguments.get("vendor")
        if not isinstance(vendor, str) or not vendor.strip():
            return "missing vendor argument"
        if not tokens(vendor).issubset(q_tokens):
            return "vendor not grounded in the query"
        if name == LEGACY_DETAIL:
            if arguments.get("field") not in DETAIL_FIELDS:
                return "vendor_detail bad field"
            if set(arguments) - {"vendor", "field"}:
                return "vendor_detail has extra arguments"
        elif set(arguments) - {"vendor"}:
            return "{} has extra arguments".format(name)
    return None


def main():
    with open(TOOLS_FILE, encoding="utf-8") as f:
        catalogue = json.load(f)

    seen = set()
    kept = []
    reasons = {}
    files = sorted(glob.glob(os.path.join(SHARDS, "*.jsonl")))
    if not files:
        print("no shards found in", os.path.relpath(SHARDS, ROOT))
        return

    total = 0
    tool_counts = {}
    off_topic = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                total += 1
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    reasons["unparseable"] = reasons.get("unparseable", 0) + 1
                    continue
                problem = check(line)
                if problem:
                    reasons[problem] = reasons.get(problem, 0) + 1
                    continue
                key = " ".join(sorted(tokens(line["query"])))
                if key in seen:
                    reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                line = remap(line)
                line["tools"] = catalogue
                kept.append(line)
                if not line["answers"]:
                    off_topic += 1
                for answer in line["answers"]:
                    tool_counts[answer["name"]] = tool_counts.get(answer["name"], 0) + 1

    rng = random.Random(SEED)
    rng.shuffle(kept)
    split = int(len(kept) * (1 - VAL_FRACTION))
    train, val = kept[:split], kept[split:]
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(TRAIN_DIR, name + ".jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("shards read: {}".format(len(files)))
    print("raw lines: {}".format(total))
    print("kept: {}  (train {} / val {})".format(len(kept), len(train), len(val)))
    print("empty answers (off-topic): {}  ({:.1f}%)".format(off_topic, 100.0 * off_topic / max(len(kept), 1)))
    print("tool calls: {}".format(tool_counts))
    print("dropped: {}".format(reasons or "none"))
    print()
    print("wrote training/needle/train.jsonl and training/needle/val.jsonl")


if __name__ == "__main__":
    main()
