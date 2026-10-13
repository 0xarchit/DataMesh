import json
import os
import sys
import time

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RAW = os.path.join(ROOT, "data", "raw")
GOLD = os.path.join(ROOT, "data", "gold")
CANON = os.path.join(ROOT, "data", "canonical")
INTER = os.path.join(ROOT, "data", "intermediate")
STORE = os.path.join(ROOT, "data", "chroma")
CANON_STORE = os.path.join(ROOT, "data", "chroma_canonical")
RESULTS = os.path.join(ROOT, "research", "results")

from datamesh import downstream, qa
from datamesh.core import llm
from datamesh.ingestion import loaders
from datamesh.pipeline import run as run_pipeline
from datamesh.rag import baseline as rag
from generate_data import build
from run_comparison import baseline_engine, score_conflicts
from run_demo import write_summary

st.set_page_config(page_title="DataMesh AI", layout="wide")


def raw_ready():
    return os.path.exists(os.path.join(RAW, "customers.csv"))


def canon_ready():
    return os.path.exists(os.path.join(CANON, "vendors.json"))


class Live:
    def __init__(self, log_box, tok_box, key="live"):
        self.log_box = log_box
        self.tok_box = tok_box
        self.key = key
        self.lines = st.session_state.get(key + "_log", [])
        self.tokens = st.session_state.get(key + "_tok", "")
        self.render()

    def reset(self):
        self.lines = []
        self.tokens = ""
        self.store()
        self.render()

    def store(self):
        st.session_state[self.key + "_log"] = self.lines
        st.session_state[self.key + "_tok"] = self.tokens

    def say(self, line):
        line = str(line)
        if line.startswith("LLM "):
            self.tokens = ""
        self.lines.append(line)
        self.store()
        self.render()

    def tok(self, text):
        self.tokens += text
        self.store()
        self.log_box.code("\n".join(self.lines[-300:]), language="text")
        self.tok_box.code(self.tokens if self.tokens else "waiting for the model ...", language="text")

    def render(self):
        self.log_box.code("\n".join(self.lines[-300:]), language="text")
        self.tok_box.code(self.tokens if self.tokens else "no token stream yet", language="text")


def live_panel(key):
    st.markdown("**Step log**")
    log_box = st.empty()
    st.markdown("**Model stream (tokens arrive live)**")
    tok_box = st.empty()
    return Live(log_box, tok_box, key=key)


def gold_questions():
    path = os.path.join(GOLD, "questions.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


with st.sidebar:
    st.markdown("## DataMesh AI")
    st.caption("Evidence aware multimodal consolidation")
    online = llm.available(timeout=3)
    if online:
        st.success("Local model online")
        st.caption(llm.model_name())
    else:
        st.error("Local model offline")
        st.caption(llm.BASE_URL)
    use_llm = st.checkbox("Use local LLM", value=online, disabled=not online)
    st.caption("Thinking is disabled, calls answer in about 1 to 3 seconds")
    st.divider()
    store_meta = rag.store_meta(STORE)
    st.caption("raw ready: {}".format("yes" if raw_ready() else "no"))
    st.caption("canonical ready: {}".format("yes" if canon_ready() else "no"))
    st.caption(
        "vector store: {}".format(
            "{} vectors, {}".format(store_meta.get("count", 0), store_meta.get("embedding", "not built"))
            if store_meta
            else "not built"
        )
    )

tab_gen, tab_raw, tab_pipe, tab_qa, tab_demo, tab_down, tab_res = st.tabs(
    [
        "1. Generate data",
        "2. Explore raw data",
        "3. Consolidate (our pipeline)",
        "4. Ask questions",
        "5. Demo: ours vs RAG",
        "6. Downstream model",
        "7. Results",
    ]
)

with tab_gen:
    st.subheader("Generate the multi format raw dataset")
    st.write(
        "Writes customers.csv, transactions.xlsx, invoice PDFs and receipt images into data/raw, "
        "plus the ground truth answers into data/gold. Local only, no model call."
    )
    if raw_ready():
        st.success("Raw data already present in data/raw")
    live = live_panel("gen")
    if st.button("Generate raw data", type="primary"):
        live.reset()
        t0 = time.time()
        vendors, invoices, rows = build(emit=live.say)
        st.success("Done in {:.1f}s: {} vendors, {} invoices, {} raw rows".format(time.time() - t0, vendors, invoices, rows))

    with st.expander("Ground truth questions used for scoring"):
        questions = gold_questions()
        if questions:
            st.table([{"question": q["q"], "answer": str(q["answer"])} for q in questions])
        else:
            st.info("Not generated yet")

with tab_raw:
    st.subheader("Explore the raw sources")
    if not raw_ready():
        st.warning("No raw data yet, use tab 1.")
    else:
        choice = st.radio(
            "Source",
            ["customers.csv", "transactions.xlsx", "invoices (PDF)", "receipts (images)"],
            horizontal=True,
        )
        if choice == "customers.csv":
            df = pd.read_csv(os.path.join(RAW, "customers.csv"))
            st.metric("rows", len(df))
            st.dataframe(df, width="stretch")
            st.caption("Same vendor appears here under a different spelling than in the ledger or invoices.")
        elif choice == "transactions.xlsx":
            df = pd.read_excel(os.path.join(RAW, "transactions.xlsx"))
            st.metric("rows", len(df))
            st.dataframe(df, width="stretch")
            st.caption("Duplicate invoice ids appear here and again in the PDFs.")
        elif choice == "invoices (PDF)":
            files = sorted(os.listdir(os.path.join(RAW, "invoices")))
            st.metric("files", len(files))
            pick = st.selectbox("invoice", files)
            rec = loaders.load_pdf(os.path.join(RAW, "invoices", pick))[0]
            st.code(
                "invoice_id: {}\nvendor: {}\namount: {}\ndate: {}\ncity: {}\ntax_id: {}".format(
                    rec.invoice_id, rec.vendor, rec.amount, rec.date, rec.city, rec.tax_id
                ),
                language="text",
            )
            st.caption("Parsed with pypdf, no model call, confidence 0.95")
        else:
            files = sorted(os.listdir(os.path.join(RAW, "receipts")))
            st.metric("files", len(files))
            pick = st.selectbox("receipt", files)
            path = os.path.join(RAW, "receipts", pick)
            st.image(path, width=420)
            live = live_panel("vis")
            if st.button("Read this receipt with the local model", type="primary"):
                live.reset()
                live.say("LLM vision {} ...".format(pick))
                data = llm.vision_extract(path, on_token=live.tok)
                live.say("vision done")
                st.json(data)
            st.caption("Image goes to the local model as base64, confidence 0.8")

with tab_pipe:
    st.subheader("Raw sources to one canonical table")
    st.write(
        "Step 1 ingests every format. Step 2 scores evidence (name, tax id, city, provenance) and clusters "
        "identity variants. Step 3 drops duplicate transactions. Step 4 writes the canonical tables."
    )
    live = live_panel("pipe")
    col_run, _ = st.columns([1, 3])
    with col_run:
        run_it = st.button("Run consolidation", type="primary", disabled=not raw_ready())
    if run_it:
        live.reset()
        t0 = time.time()
        report = run_pipeline(RAW, CANON, INTER, use_llm=use_llm, emit=live.say, on_token=live.tok)
        st.session_state["report"] = report
        st.success("Consolidation finished in {:.1f}s".format(time.time() - t0))

    report = st.session_state.get("report")
    if report is None and canon_ready():
        with open(os.path.join(CANON, "reconciliation_report.json"), encoding="utf-8") as f:
            report = json.load(f)
    if report:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Raw records", report["raw_records"])
        c2.metric("Vendor clusters", report["vendor_clusters"])
        c3.metric("Unique invoices", report["unique_transactions"])
        c4.metric("Duplicates removed", report["duplicates_removed"])
        c5.metric("Total spend INR", "{:,}".format(report["total_spend_inr"]))
        st.json(report.get("source_counts", {}))

    if canon_ready():
        st.markdown("### Canonical output")
        sub1, sub2, sub3, sub4 = st.tabs(["vendors.json", "transactions.csv", "provenance.jsonl", "report"])
        with sub1:
            st.dataframe(pd.read_json(os.path.join(CANON, "vendors.json")), width="stretch", height=320)
        with sub2:
            tx = pd.read_csv(os.path.join(CANON, "transactions.csv"))
            st.metric("rows", len(tx))
            st.dataframe(tx, width="stretch", height=320)
        with sub3:
            prov = pd.read_json(os.path.join(CANON, "provenance.jsonl"), lines=True)
            st.dataframe(prov, width="stretch", height=320)
        with sub4:
            with open(os.path.join(CANON, "reconciliation_report.json"), encoding="utf-8") as f:
                st.json(json.load(f))
    else:
        st.info("Run consolidation first to see the canonical tables.")

with tab_qa:
    st.subheader("Ask a question")
    if not canon_ready():
        st.warning("Run tab 3 first.")
    else:
        vendors, rows = qa.load_canonical(CANON)
        gold_map = {q["q"]: q["answer"] for q in gold_questions()}
        picked = st.selectbox("Question", ["Custom question"] + list(gold_map))
        if picked == "Custom question":
            text = st.text_input("Type your question", value="What is the total spend with Acme Ltd in INR?")
            st.caption(
                "Try a real one, then try one the data cannot answer, such as "
                "'What is the total spend with FooBar Inc in INR?' or 'What is the weather in Mumbai?'"
            )
        else:
            text = picked
        ask_rag = st.checkbox("Ask RAG the same question and compare", value=True)
        live = live_panel("qa")
        if st.button("Answer", type="primary"):
            live.reset()
            result = qa.outcome(text, vendors, rows, use_llm=use_llm, on_token=live.tok)
            if result["plan"]:
                live.say("our query plan ({}): {}".format(result["source"], json.dumps(result["plan"], sort_keys=True)))
            live.say("our status: {}".format(result["status"]))

            st.markdown("### Our pipeline")
            if result["status"] == "answered":
                st.code(str(result["value"]), language="text")
                if text in gold_map:
                    if result["value"] == qa.normalize(gold_map[text]):
                        st.success("Matches ground truth")
                    else:
                        st.error("Ground truth is {}".format(qa.normalize(gold_map[text])))
            elif result["status"] == "entity_not_found":
                st.warning("Not answered. " + result["message"])
                st.caption("The engine refuses rather than compute a number for an entity that is not in the table.")
            else:
                st.info("Not answered. " + result["message"])
                st.caption("Both the model planner and the keyword planner found nothing to compute, so nothing is returned.")

            if result["vendor"]:
                match = [r for r in rows if r["vendor_cluster"] == result["vendor"]["id"]]
                st.markdown("### Evidence rows used")
                c1, c2, c3 = st.columns(3)
                c1.metric("Vendor", result["vendor"]["name"])
                c2.metric("Cluster", result["vendor"]["id"])
                c3.metric("Transactions", len(match))
                st.dataframe(pd.DataFrame(match), width="stretch")

            if ask_rag:
                st.markdown("### RAG over the raw records, same question")
                try:
                    _, retriever = rag.load_store(RAW, STORE, use_llm=use_llm, emit=live.say, on_token=live.tok)
                    rag_text = rag.rag_answer(text, retriever, use_llm=use_llm, on_token=live.tok, emit=live.say)
                    st.code(str(rag_text).strip() or "(empty)", language="text")
                    if result["status"] == "answered":
                        if rag.normalize_answer(rag_text) == result["value"]:
                            st.caption("RAG agrees with the value computed over the canonical table.")
                        else:
                            st.caption("RAG differs from the value computed over the canonical table.")
                except Exception as exc:
                    st.error("RAG arm unavailable: {}".format(exc))
        st.caption(
            "The model writes a query plan as JSON and the executor computes it over the canonical table. "
            "Nothing is retrieved and no answer is prewritten, so the value is always calculated from the rows. "
            "When a question cannot be computed, the engine says so instead of returning a number."
        )

with tab_demo:
    st.subheader("Full cycle: consolidation against a local RAG vector database")
    st.write(
        "Both sides see the same raw files. Our side reconciles first, then the model writes a query plan "
        "that is executed over the canonical table. RAG embeds the raw corpus into ChromaDB with MiniLM and "
        "retrieves the closest chunks before the model answers."
    )
    regen = st.checkbox("Regenerate raw data and rebuild the vector store", value=False)
    k = st.slider("RAG retrieval depth (chunks out of the corpus)", 5, 40, rag.DEFAULT_K)
    live = live_panel("demo")
    prog = st.progress(0.0, text="idle")
    table_box = st.empty()
    if st.button("Run full demo", type="primary"):
        live.reset()
        if regen:
            prog.progress(0.05, text="generating raw data")
            build(emit=live.say)
        fresh = canon_ready() and rag.store_is_fresh(RAW, STORE)
        if fresh:
            live.say("reuse on: consolidated tables and vector store match raw fingerprint {}".format(rag.fingerprint(RAW)))
            with open(os.path.join(CANON, "reconciliation_report.json"), encoding="utf-8") as f:
                report = json.load(f)
            vendors, rows = qa.load_canonical(CANON)
            prog.progress(0.2, text="loading the prebuilt vector store")
            chunks, retriever = rag.load_store(RAW, STORE, use_llm=use_llm, emit=live.say, on_token=live.tok, k=k)
        else:
            prog.progress(0.15, text="running consolidation pipeline")
            report = run_pipeline(RAW, CANON, INTER, use_llm=use_llm, emit=live.say, on_token=live.tok)
            st.session_state["report"] = report
            vendors, rows = qa.load_canonical(CANON)
            prog.progress(0.3, text="embedding the raw corpus into chromadb")
            chunks, retriever = rag.load_store(RAW, STORE, use_llm=use_llm, emit=live.say, on_token=live.tok, k=k)
        live.say("retrieval engine: {}, depth {}".format(retriever.engine(), retriever.k))
        canon_chunks, canon_retriever = rag.canonical_store(CANON, CANON_STORE, k=k, emit=live.say)
        live.say(
            "control arm: {} consolidated chunks, same embedding model and depth {}".format(
                len(canon_chunks), canon_retriever.k
            )
        )
        questions = gold_questions()
        n = len(questions)
        view = []
        ours_ok = 0
        rag_ok = 0
        rag_canon_ok = 0
        plan_sources = {}
        for i, item in enumerate(questions, start=1):
            live.say("question {}/{}: {}".format(i, n, item["q"]))
            gold = qa.normalize(item["answer"])
            spec, source = qa.plan(item["q"], vendors, use_llm=use_llm, on_token=live.tok)
            live.say("LLM our query plan ({}): {}".format(source, json.dumps(spec, sort_keys=True)))
            plan_sources[source] = plan_sources.get(source, 0) + 1
            ours = qa.normalize(qa.execute(spec, vendors, rows))
            live.say("our table answer: {}".format(ours))
            rag_text = rag.rag_answer(item["q"], retriever, use_llm=use_llm, on_token=live.tok, emit=live.say)
            rag_norm = rag.normalize_answer(rag_text)
            live.say("rag over raw records: {}".format(rag_norm))
            canon_text = rag.rag_answer(item["q"], canon_retriever, use_llm=use_llm, on_token=live.tok, emit=live.say)
            canon_norm = rag.normalize_answer(canon_text)
            live.say("rag over consolidated table: {}".format(canon_norm))
            o = ours == gold
            r = rag_norm == gold
            rc = canon_norm == gold
            ours_ok += int(o)
            rag_ok += int(r)
            rag_canon_ok += int(rc)
            view.append(
                {
                    "question": item["q"],
                    "gold": gold,
                    "ours": ours,
                    "rag": rag_norm,
                    "rag_consolidated": canon_norm,
                    "ours_ok": "yes" if o else "no",
                    "rag_ok": "yes" if r else "no",
                    "rag_consolidated_ok": "yes" if rc else "no",
                }
            )
            table_box.table([{k: str(v) for k, v in row.items()} for row in view])
            prog.progress(0.3 + 0.6 * i / n, text="question {} of {}".format(i, n))
        prog.progress(0.95, text="training the downstream model")
        raw = downstream.raw_table(INTER, CANON)
        canon = downstream.canonical_table(CANON)
        model, model_name = downstream.build_model()
        down = {
            "model": model_name,
            "task": "classify whether a vendor is high-value from its aggregated transaction features",
            "raw_table": downstream.evaluate(raw, model),
            "consolidated_table": downstream.evaluate(canon, model),
        }
        os.makedirs(RESULTS, exist_ok=True)
        comparison = {
            "questions": n,
            "ours_accuracy": round(ours_ok / n, 4),
            "rag_accuracy": round(rag_ok / n, 4),
            "rag_canonical_accuracy": round(rag_canon_ok / n, 4),
            "rag_engine": baseline_engine(),
            "qa_plan_source": plan_sources,
            "conflicts": score_conflicts(),
            "reconciliation": report,
            "table": [
                {
                    "question": r["question"],
                    "gold": r["gold"],
                    "ours": r["ours"],
                    "rag": r["rag"],
                    "ours_correct": r["ours_ok"] == "yes",
                    "rag_correct": r["rag_ok"] == "yes",
                }
                for r in view
            ],
        }
        with open(os.path.join(RESULTS, "comparison.json"), "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        with open(os.path.join(RESULTS, "downstream.json"), "w", encoding="utf-8") as f:
            json.dump(down, f, indent=2)
        write_summary(comparison, down)
        st.session_state["demo"] = comparison
        st.session_state["down"] = down
        prog.progress(1.0, text="finished")
        st.balloons()

    demo = st.session_state.get("demo")
    if demo:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Our accuracy", "{}/{}".format(round(demo["ours_accuracy"] * demo["questions"]), demo["questions"]))
        c2.metric("RAG raw", "{}/{}".format(round(demo["rag_accuracy"] * demo["questions"]), demo["questions"]))
        c3.metric(
            "RAG consolidated",
            "{}/{}".format(
                round(demo.get("rag_canonical_accuracy", 0.0) * demo["questions"]), demo["questions"]
            ),
        )
        c4.metric("Duplicates removed", demo["reconciliation"]["duplicates_removed"])
        st.dataframe(pd.DataFrame(demo["table"]).astype(str), width="stretch")

with tab_down:
    st.subheader("Does consolidation help a downstream predictor")
    st.write(
        "Same model, same features, two tables. The raw table groups by the vendor string as written, "
        "so one real vendor splits into several noisy rows."
    )
    live = live_panel("down")
    if st.button("Run benchmark", type="primary", disabled=not canon_ready()):
        live.reset()
        live.say("building raw table (grouped by raw vendor string)")
        raw = downstream.raw_table(INTER, CANON)
        live.say("raw table rows: {}".format(len(raw)))
        live.say("building consolidated table (grouped by vendor cluster)")
        canon = downstream.canonical_table(CANON)
        live.say("consolidated table rows: {}".format(len(canon)))
        model, model_name = downstream.build_model()
        live.say("model: {}".format(model_name))
        live.say("leave one out cross validation on raw table")
        r1 = downstream.evaluate(raw, model)
        live.say("raw accuracy {}".format(r1["accuracy"]))
        live.say("leave one out cross validation on consolidated table")
        r2 = downstream.evaluate(canon, model)
        live.say("consolidated accuracy {}".format(r2["accuracy"]))
        st.session_state["down"] = {
            "model": model_name,
            "task": "classify whether a vendor is high-value from its aggregated transaction features",
            "raw_table": r1,
            "consolidated_table": r2,
        }
        st.success("Benchmark finished")
    down = st.session_state.get("down")
    if down:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw table accuracy", down["raw_table"]["accuracy"])
        c2.metric("Consolidated accuracy", down["consolidated_table"]["accuracy"])
        c3.metric("Model", down["model"])
        st.dataframe(
            [
                {"table": "raw", **down["raw_table"]},
                {"table": "consolidated", **down["consolidated_table"]},
            ],
            width="stretch",
        )
        st.markdown("### Feature tables")
        s1, s2 = st.tabs(["raw table", "consolidated table"])
        with s1:
            st.dataframe(downstream.raw_table(INTER, CANON), width="stretch", height=320)
        with s2:
            st.dataframe(downstream.canonical_table(CANON), width="stretch", height=320)

with tab_res:
    st.subheader("Saved results")
    summary = os.path.join(RESULTS, "SUMMARY.md")
    png = os.path.join(RESULTS, "summary.png")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("No summary yet, run the full demo in tab 5.")
    if os.path.exists(png):
        st.image(png)
    for name in ("comparison.json", "downstream.json"):
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            with st.expander(name):
                with open(path, encoding="utf-8") as f:
                    st.json(json.load(f))
