import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datamesh import qa
from datamesh.core import llm, planner_needle
from datamesh.pipeline import run
from datamesh.rag import baseline as rag
from generate_data import build

RAW = os.path.join(ROOT, "data", "raw")
CANON = os.path.join(ROOT, "data", "canonical")
INTER = os.path.join(ROOT, "data", "intermediate")
GOLD = os.path.join(ROOT, "data", "gold")
STORE = os.path.join(ROOT, "data", "chroma")
CANON_STORE = os.path.join(ROOT, "data", "chroma_canonical")
RESULTS = os.path.join(ROOT, "research", "results")


def counted(tokens, prompt_text, completion_text):
    if not tokens.get("estimated"):
        return tokens
    prompt = max(1, len(prompt_text) // 4)
    completion = max(1, len(completion_text) // 4)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "estimated": True,
    }


def measured_plan(question, vendors):
    text, tokens, elapsed = llm.chat_measured(question, system=qa.PLAN_SYSTEM, max_tokens=220)
    spec = llm.parse_json(text)
    if not spec or spec.get("op") not in ("sum", "count", "distinct_count", "lookup"):
        return qa.plan_rules(question, vendors), counted(tokens, qa.PLAN_SYSTEM + question, text), elapsed, "keyword rules"
    return spec, counted(tokens, qa.PLAN_SYSTEM + question, text), elapsed, "local model"


def measured_needle(question, vendors):
    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": False}
    started = time.time()
    spec, source = qa.plan(question, vendors, use_llm=False, use_needle=True)
    return spec, zero, time.time() - started, source


def measured_rag(question, retriever, k=None):
    context = retriever.retrieve(question, k=k or retriever.k)
    if not context:
        context = retriever.chunks[: retriever.k]
    system = rag.answer_system(context)
    text, tokens, elapsed = llm.chat_measured(question, system=system, max_tokens=64)
    return text, counted(tokens, system + question, text), elapsed, len(context)


def summarize(rows):
    n = len(rows) or 1
    return {
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "prompt_tokens": sum(r["tokens"]["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["tokens"]["completion_tokens"] for r in rows),
        "total_tokens": sum(r["tokens"]["total_tokens"] for r in rows),
        "mean_tokens_per_question": round(sum(r["tokens"]["total_tokens"] for r in rows) / n, 1),
        "total_seconds": round(sum(r["seconds"] for r in rows), 2),
        "mean_seconds_per_question": round(sum(r["seconds"] for r in rows) / n, 3),
    }


def by_type(rows):
    out = {}
    for r in rows:
        bucket = out.setdefault(r.get("type", "unknown"), {"questions": 0, "correct": 0})
        bucket["questions"] += 1
        bucket["correct"] += r["correct"]
    for bucket in out.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["questions"], 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--questions", default=os.path.join(GOLD, "questions_bench.json"))
    args = parser.parse_args()

    if not llm.available():
        print("no LLM server on", llm.BASE_URL)
        return
    print("model:", llm.model_name())

    if args.seed is not None:
        build(emit=lambda s: None, seed=args.seed)
        args.reuse = False

    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    if args.reuse and os.path.exists(os.path.join(CANON, "vendors.json")):
        with open(os.path.join(CANON, "reconciliation_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        consolidation_seconds = None
    else:
        report = run(RAW, CANON, INTER, use_llm=True)
        consolidation_seconds = round(time.time() - started, 2)
    print("consolidation seconds:", consolidation_seconds)

    vendors, rows = qa.load_canonical(CANON)
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    print("questions:", len(questions))
    use_needle = planner_needle.available()
    if not use_needle:
        print("needle weights not found, skipping the local planner arm")

    index_started = time.time()
    raw_chunks, raw_retriever = rag.load_store(RAW, STORE, use_llm=True, rebuild=args.rebuild, k=args.k or rag.DEFAULT_K)
    canon_chunks, canon_retriever = rag.canonical_store(CANON, CANON_STORE, k=args.k or rag.DEFAULT_K)
    rag_index_seconds = round(time.time() - index_started, 2)
    print(
        "retrieval: k={} over {} raw chunks, {} consolidated chunks, index {:.1f}s".format(
            raw_retriever.k, len(raw_chunks), len(canon_chunks), rag_index_seconds
        )
    )

    ours, needle, rag_raw, rag_canon, table = [], [], [], [], []
    needle_sources = {}
    for item in questions:
        question = item["q"]
        gold = qa.normalize(item["answer"])

        spec, plan_tokens, plan_seconds, source = measured_plan(question, vendors)
        compute_started = time.time()
        ours_answer = qa.normalize(qa.execute(spec, vendors, rows))
        ours_seconds = plan_seconds + (time.time() - compute_started)
        ours_correct = int(ours_answer == gold)
        qtype = item.get("type", "unknown")
        ours.append({"correct": ours_correct, "tokens": plan_tokens, "seconds": ours_seconds, "type": qtype})

        spec_n, needle_tokens, needle_seconds, needle_source = measured_needle(question, vendors)
        compute_started = time.time()
        needle_answer = qa.normalize(qa.execute(spec_n, vendors, rows))
        needle_seconds += time.time() - compute_started
        needle_correct = int(needle_answer == gold)
        needle.append(
            {"correct": needle_correct, "tokens": needle_tokens, "seconds": needle_seconds, "type": qtype}
        )
        needle_sources[needle_source] = needle_sources.get(needle_source, 0) + 1

        text, raw_tokens, seconds, seen = measured_rag(question, raw_retriever, k=args.k)
        raw_answer = rag.normalize_answer(text)
        raw_correct = int(raw_answer == gold)
        rag_raw.append({"correct": raw_correct, "tokens": raw_tokens, "seconds": seconds, "type": qtype})

        text_c, canon_tokens, seconds_c, seen_c = measured_rag(question, canon_retriever, k=args.k)
        canon_answer = rag.normalize_answer(text_c)
        canon_correct = int(canon_answer == gold)
        rag_canon.append({"correct": canon_correct, "tokens": canon_tokens, "seconds": seconds_c, "type": qtype})

        table.append(
            {
                "question": question,
                "gold": item["answer"],
                "ours": {
                    "answer": ours_answer,
                    "correct": ours_correct,
                    "plan_source": source,
                    **plan_tokens,
                    "seconds": round(ours_seconds, 3),
                },
                "ours_needle_planner": {
                    "answer": needle_answer,
                    "correct": needle_correct,
                    "plan_source": needle_source,
                    **needle_tokens,
                    "seconds": round(needle_seconds, 3),
                },
                "rag_raw": {
                    "answer": raw_answer,
                    "correct": raw_correct,
                    "chunks": seen,
                    **raw_tokens,
                    "seconds": round(seconds, 3),
                },
                "rag_canonical": {
                    "answer": canon_answer,
                    "correct": canon_correct,
                    "chunks": seen_c,
                    **canon_tokens,
                    "seconds": round(seconds_c, 3),
                },
            }
        )
        print(
            "{:<46} gold {:<9} ours {:<9} rag {:<9} ragc {:<9}".format(
                question[:44],
                str(item["answer"])[:8],
                str(ours_answer)[:8],
                str(raw_answer)[:8],
                str(canon_answer)[:8],
            )
        )

    arms = {
        "ours_consolidated_engine": summarize(ours),
        "rag_over_raw_records": summarize(rag_raw),
        "rag_over_consolidated_table": summarize(rag_canon),
    }
    types = {
        "ours_consolidated_engine": by_type(ours),
        "rag_over_raw_records": by_type(rag_raw),
        "rag_over_consolidated_table": by_type(rag_canon),
    }
    if use_needle:
        arms["ours_needle_planner"] = summarize(needle)
        types["ours_needle_planner"] = by_type(needle)
        arms["ours_needle_planner"]["plan_sources"] = needle_sources

    result = {
        "model": llm.model_name(),
        "seed": args.seed,
        "questions_file": os.path.basename(args.questions),
        "retrieval_k": raw_retriever.k,
        "corpus_chunks": {"raw": len(raw_chunks), "consolidated": len(canon_chunks)},
        "consolidation_seconds": consolidation_seconds,
        "rag_index_seconds": rag_index_seconds,
        "reconciliation": report,
        "arms": arms,
        "by_type": types,
        "per_question": table,
    }
    total_questions = len(questions)
    if consolidation_seconds:
        for arm in result["arms"].values():
            arm["consolidation_seconds_amortized"] = round(consolidation_seconds / total_questions, 3)
            arm["total_seconds_including_consolidation"] = round(arm["total_seconds"] + consolidation_seconds, 2)

    out_name = "benchmark_{}.json".format(args.seed) if args.seed is not None else "benchmark.json"
    out_path = os.path.join(RESULTS, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("{:<34} {:>10} {:>12} {:>12} {:>10}".format("arm", "accuracy", "total_tokens", "mean_tokens", "total_s"))
    for name, arm in result["arms"].items():
        print(
            "{:<34} {:>4}/{:<5} {:>12} {:>12} {:>10}".format(
                name,
                arm["correct"],
                arm["questions"],
                arm["total_tokens"],
                arm["mean_tokens_per_question"],
                arm["total_seconds"],
            )
        )
    print()
    print("accuracy by question type")
    arm_names = tuple(result["arms"])
    type_names = sorted({t for arm in result["by_type"].values() for t in arm})
    header = "{:<24}" + " {:>12}" * len(arm_names)
    print(header.format("type", *[a.replace("ours_", "")[:11] for a in arm_names]))
    for t in type_names:
        cells = []
        for a in arm_names:
            bucket = result["by_type"][a].get(t, {"correct": 0, "questions": 0})
            cells.append("{}/{}".format(bucket["correct"], bucket["questions"]))
        print(header.format(t, *cells))
    print()
    print("wrote", os.path.relpath(out_path, ROOT))


if __name__ == "__main__":
    main()
