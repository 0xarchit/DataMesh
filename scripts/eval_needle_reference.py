import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training", "needle")


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "needle3.safetensors"))
    parser.add_argument("--adapter", default=os.path.join(TRAIN_DIR, "adapter_v5.safetensors"))
    parser.add_argument("--data", default=os.path.join(TRAIN_DIR, "val.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(TRAIN_DIR, "eval_reference.json"))
    args = parser.parse_args()

    import jax.numpy as jnp

    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.finetune import _calls_of, merge_lora, read_adapter, render_example
    from needle.model.quantize import WEIGHT_BITS, cq_ste_params
    from needle.model.run import generate, load_checkpoint
    from needle.model.tokenizer import TOOL_CALL_END, TOOL_CALL_START, get_tokenizer

    params, config = load_checkpoint(args.checkpoint)
    tokenizer = get_tokenizer(config.vocab_size)
    model = SimpleAttentionNetwork(config)

    adapter = read_adapter(args.adapter)
    lora = {tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
            for key, v in adapter["lora"].items()}
    merged = cq_ste_params(merge_lora(params, lora, adapter["scale"]), WEIGHT_BITS)

    rows = load_jsonl(args.data, args.limit)
    exact = 0
    per_tool = {}
    failures = []
    started = time.time()

    for row in rows:
        prompt, _ = render_example({**row, "answers": []})
        text = generate(model, merged, tokenizer, prompt, max_new_tokens=96,
                        temperature=0.0, stream=False)
        want = _calls_of(TOOL_CALL_START + json.dumps(
            row.get("answers", []), separators=(",", ":")) + TOOL_CALL_END)
        got = _calls_of(text)
        ok = got == want
        if ok:
            exact += 1
        else:
            failures.append({"query": row["query"], "expected": want, "predicted": got,
                             "raw": text[:300]})
        for name, _ in (want or []):
            bucket = per_tool.setdefault(name, {"total": 0, "exact": 0})
            bucket["total"] += 1
            if ok:
                bucket["exact"] += 1

    elapsed = time.time() - started
    n = max(len(rows), 1)
    report = {
        "path": "reference render_example plus python generate",
        "adapter": args.adapter,
        "data": args.data,
        "examples": len(rows),
        "exact_matches": exact,
        "exact_accuracy": round(exact / n, 4),
        "per_tool": per_tool,
        "seconds": round(elapsed, 1),
        "ms_per_query": round(1000.0 * elapsed / n, 1),
        "failures": failures[:40],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("examples        {}".format(len(rows)))
    print("exact match     {}/{}  ({:.1%})".format(exact, len(rows), exact / n))
    print("speed           {} ms/query".format(report["ms_per_query"]))
    print("per tool        {}".format(json.dumps(per_tool)))
    print("wrote", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
