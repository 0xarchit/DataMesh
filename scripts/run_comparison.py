import json
import os
from collections import Counter

from datamesh import qa
from datamesh.core import llm
from datamesh.pipeline import run
from datamesh.rag import baseline as rag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CANON = os.path.join(ROOT, "data", "canonical")
INTER = os.path.join(ROOT, "data", "intermediate")
GOLD = os.path.join(ROOT, "data", "gold")
RESULTS = os.path.join(ROOT, "research", "results")

LOAD_ORDER = {"csv": 0, "xlsx": 1, "pdf": 2, "receipt": 3}


def score_conflicts():
    gold = {c["invoice_id"]: c for c in json.load(open(os.path.join(GOLD, "conflicts.json"), encoding="utf-8"))}
    decided = {
        c["invoice_id"]: c for c in json.load(open(os.path.join(CANON, "conflicts.json"), encoding="utf-8"))
    }

    records = json.load(open(os.path.join(INTER, "records.json"), encoding="utf-8"))
    first_seen = {}
    for r in sorted(records, key=lambda r: LOAD_ORDER.get(r["source_type"], 9)):
        if r["invoice_id"] and r["amount"]:
            first_seen.setdefault(r["invoice_id"], r)

    baseline_wrong = [
        i for i, g in gold.items() if i in first_seen and first_seen[i]["amount"] != g["true_amount"]
    ]
    ours_wrong = [i for i, g in gold.items() if i in decided and decided[i]["chosen_amount"] != g["true_amount"]]
    return {
        "planted": len(gold),
        "detected": len(set(gold) & set(decided)),
        "missed": sorted(set(gold) - set(decided)),
        "spurious": sorted(set(decided) - set(gold)),
        "resolved": sum(1 for c in decided.values() if c["decision"] == "resolved"),
        "for_review": sum(1 for c in decided.values() if c["decision"] == "review"),
        "load_order_baseline_wrong": len(baseline_wrong),
        "ours_wrong": len(ours_wrong),
        "ours_wrong_invoices": sorted(ours_wrong),
    }


def baseline_engine():
    if llm.available():
        return "local llm: " + llm.model_name()
    return "keyword-rules fallback - no LLM server reachable, so this arm is NOT a RAG run"


def full_context_scores(questions, retriever, canon_retriever, chunks, canon_chunks, use_llm=True):
    scores = {}
    for name, retriever_, corpus in (("raw", retriever, chunks), ("canonical", canon_retriever, canon_chunks)):
        correct = 0
        for item in questions:
            gold = qa.normalize(item["answer"])
            answer = rag.rag_answer(item["q"], retriever_, chunks=corpus, use_llm=use_llm)
            correct += int(rag.normalize_answer(answer) == gold)
        scores[name] = round(correct / len(questions), 4)
    return scores



def main(use_llm=True):
    os.makedirs(RESULTS, exist_ok=True)
    report = run(RAW, CANON, INTER, use_llm=use_llm)
    vendors, rows = qa.load_canonical(CANON)
    with open(os.path.join(GOLD, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)

    store = os.path.join(ROOT, "data", "chroma")
    chunks, retriever = rag.load_store(RAW, store, use_llm=use_llm, rebuild=True)
    canon_chunks, canon_retriever = rag.canonical_store(CANON, os.path.join(ROOT, "data", "chroma_canonical"))

    table = []
    ours_ok = 0
    rag_ok = 0
    rag_canon_ok = 0
    plan_sources = Counter()
    for item in questions:
        gold = qa.normalize(item["answer"])
        spec, plan_source = qa.plan(item["q"], vendors, use_llm=use_llm)
        ours = qa.normalize(qa.execute(spec, vendors, rows))
        plan_sources[plan_source] += 1
        rag_ans = rag.rag_answer(item["q"], retriever, use_llm=use_llm)
        rag_norm = rag.normalize_answer(rag_ans)
        canon_ans = rag.rag_answer(item["q"], canon_retriever, use_llm=use_llm)
        canon_norm = rag.normalize_answer(canon_ans)
        o = ours == gold
        r = rag_norm == gold
        rc = canon_norm == gold
        ours_ok += int(o)
        rag_ok += int(r)
        rag_canon_ok += int(rc)
        table.append(
            {
                "question": item["q"],
                "gold": item["answer"],
                "ours": ours,
                "rag": rag_norm,
                "rag_canonical": canon_norm,
                "rag_raw": rag_ans,
                "rag_canonical_raw": canon_ans,
                "ours_correct": o,
                "rag_correct": r,
                "rag_canonical_correct": rc,
            }
        )

    n = len(questions)
    conflicts = score_conflicts()
    engine = baseline_engine()
    full = full_context_scores(questions, retriever, canon_retriever, chunks, canon_chunks, use_llm=use_llm)
    out = {
        "questions": n,
        "ours_accuracy": round(ours_ok / n, 4),
        "rag_accuracy": round(rag_ok / n, 4),
        "rag_canonical_accuracy": round(rag_canon_ok / n, 4),
        "rag_full_context": full,
        "rag_engine": engine,
        "qa_plan_source": dict(plan_sources),
        "rag_k": retriever.k,
        "rag_corpus_chunks": len(chunks),
        "rag_canonical_chunks": len(canon_chunks),
        "conflicts": conflicts,
        "reconciliation": report,
        "table": table,
    }
    with open(os.path.join(RESULTS, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("{:<50} {:>11} {:>11} {:>11} {:>11}".format("question", "gold", "ours", "rag-raw", "rag-consol"))
    for row in table:
        print(
            "{:<50} {:>11} {:>11} {:>11} {:>11}".format(
                row["question"][:48],
                str(row["gold"])[:11],
                str(row["ours"])[:11],
                str(row["rag"])[:11],
                str(row["rag_canonical"])[:11],
            )
        )
    print()
    print("rag baseline ran as: {}".format(engine))
    print("our query plans came from: {}".format(dict(plan_sources)))
    print(
        "retrieval: k={} over {} raw chunks and {} consolidated chunks".format(
            retriever.k, len(chunks), len(canon_chunks)
        )
    )
    print()
    print("cross-source conflicts planted: {}".format(conflicts["planted"]))
    print("  detected: {}".format(conflicts["detected"]))
    print("  resolved: {}   flagged for review: {}".format(conflicts["resolved"], conflicts["for_review"]))
    print("  load-order baseline got {} wrong".format(conflicts["load_order_baseline_wrong"]))
    print("  DataMesh got {} wrong".format(conflicts["ours_wrong"]))
    print()
    print("ours accuracy:            {} / {}".format(ours_ok, n))
    print("rag over raw corpus:      {} / {}   (top-{} retrieval)".format(rag_ok, n, retriever.k))
    print("rag over consolidated:    {} / {}   (top-{} retrieval)".format(rag_canon_ok, n, retriever.k))
    print()
    print("full-corpus ablation, no retrieval truncation - isolates the corpus alone:")
    print("  rag over raw records:        {:.0%}".format(full["raw"]))
    print("  rag over consolidated table: {:.0%}".format(full["canonical"]))
    return out


if __name__ == "__main__":
    main()
