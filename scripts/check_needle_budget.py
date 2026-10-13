import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training", "needle")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(TRAIN_DIR, "train.jsonl"))
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    from needle.model.finetune import render_example
    from needle.model.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    rows = load(args.data)
    if not rows:
        print("no examples in", args.data)
        return 1

    prompts, targets = [], []
    worst = 0
    truncated = []
    tools_tokens = None

    for index, row in enumerate(rows):
        prompt, target = render_example(row)
        prompt_len = len(tokenizer.encode(prompt))
        target_len = len(tokenizer.encode(target))
        total = 1 + prompt_len + target_len + 1
        prompts.append(prompt_len)
        targets.append(target_len)
        worst = max(worst, total)
        if total > args.max_len:
            truncated.append((index, total, row.get("query", "")))
        if tools_tokens is None:
            tools_tokens = len(
                tokenizer.encode(json.dumps(row["tools"], separators=(",", ":"), ensure_ascii=False))
            )

    print("examples          {}".format(len(rows)))
    print("tools json tokens {}".format(tools_tokens))
    print("prompt tokens     min {}  median {}  max {}".format(min(prompts), int(statistics.median(prompts)), max(prompts)))
    print("target tokens     min {}  median {}  max {}".format(min(targets), int(statistics.median(targets)), max(targets)))
    print("budget            {}".format(args.max_len))
    print("longest example   {}".format(worst))
    print()
    print("answer fully fits {}/{}  ({:.1%})".format(len(rows) - len(truncated), len(rows), 1 - len(truncated) / len(rows)))

    if truncated:
        print()
        print("FAIL {} example(s) lose tokens to truncation at --max-len {}".format(len(truncated), args.max_len))
        print("     raise --max-len to at least {} and retrain".format(worst))
        for index, total, query in truncated[:5]:
            print("     line {}  {} tokens  {!r}".format(index + 1, total, query[:60]))
        return 1

    print("OK every example keeps its full reasoning and tool call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
