# DataMesh AI: Evidence-Aware Multimodal Enterprise Data Consolidation

> **B.Tech (Honors) Semester Project** — GLA University, Mathura
> **Authors:** Archit Jain (2415800016) & Deepti Yadav (2415800029)

---

## 1. Project Overview

Business-critical information is spread across PDFs, scanned invoices, Excel
ledgers, CSVs and images. Modern multimodal models can *extract* text from all
of them, but **extraction is not integration**. Two problems survive extraction:

1. **Entity resolution** — the same vendor appears as `Acme Ltd`, `ACME` and
   `Acme Limited` across sources, so any aggregate over the raw records is wrong.
2. **Conflict resolution** — the same invoice can be written into two sources
   with two different amounts, and a pipeline that just loads files in order
   silently keeps whichever copy it read first.

DataMesh AI addresses both with an **evidence-aware reconciliation and decision
layer**:

* A **seven-signal evidence vector** `E = [E1..E7]` — semantic name similarity,
  tax-id agreement, city agreement, coreference, provenance confidence,
  temporal freshness and cross-source consensus.
* **Cross-source conflict resolution** that detects disagreeing copies of one
  invoice, resolves them from cross-copy majority and the vendor's own amount
  profile, and flags what it cannot resolve instead of guessing.
* **Cell-level provenance** for every consolidated value, so any number in the
  output can be traced back to the file it came from.

Decision vocabulary per invoice: merge / resolved / **flagged for review**. The
review branch is the abstain path — an unresolved conflict is surfaced, never
silently merged.

---

## 2. Repository layout

```text
DataMesh/
├── academic/
│   ├── literature/           # verified journal papers + the reference-set notes
│   ├── synopsis/             # synopsis.tex
│   ├── reports/              # (empty - reports go here)
│   └── presentations/        # (empty - slides go here)
│
├── data/
│   ├── raw/                  # generated inputs: customers.csv, transactions.xlsx,
│   │                         # invoices/*.pdf, receipts/*.png
│   ├── intermediate/         # records.json - extracted records, pre-consolidation
│   ├── canonical/            # vendors.json, transactions.csv, provenance.jsonl,
│   │                         # conflicts.json, reconciliation_report.json
│   ├── gold/                 # ground truth: invoice_truth, conflicts, receipts, questions
│   └── chroma/               # the RAG baseline's vector store
│
├── research/results/         # SUMMARY.md, summary.png, comparison.json, downstream.json
│
├── src/datamesh/
│   ├── core/                 # schema.py (Record, name/city normalization), llm.py
│   ├── ingestion/            # loaders.py - CSV, XLSX, PDF, and VLM image adapters
│   ├── evidence/             # signals.py - the E1..E7 vector and its weights
│   ├── reconciliation/       # resolver.py - vendor clustering
│   ├── fusion/               # consensus.py - cross-source conflict detection
│   ├── rag/                  # baseline.py - the traditional-RAG arm
│   ├── pipeline.py           # the four-step consolidation run
│   ├── qa.py                 # question answering over the canonical table
│   └── downstream.py         # downstream tabular prediction
│
├── scripts/                  # generate_data, run_pipeline, run_comparison,
│                             # run_downstream, run_demo, mock_llm_server
├── tests/test_gold.py        # asserts canonical output against ground truth
└── app.py                    # Streamlit demo UI
```

`apps/` exists in the intended architecture but is empty; the Streamlit app in
`app.py` is the demo surface.

---

## 3. Literature

Seven peer-reviewed journal papers (2023–2026) back this project. Full notes,
per-paper caveats and honesty framing are in
[`academic/literature/`](academic/literature/):

| # | Paper | Journal | Backs |
|---|---|---|---|
| 1 | Zhang et al., SMUTF: Schema Matching Using Generative Tags and Hybrid Features | Information Systems, 2025 | Multi-signal evidence vector |
| 2 | Christen et al., Graph Metrics-driven Record Cluster Repair | ACM J. Data & Inf. Quality, 2025 | False-merge failure mode |
| 3 | Robach et al., False Discovery Estimation in Record Linkage | Statistics in Medicine, 2025 | False-merge rate |
| 4 | Silva Filho et al., Classifier Calibration: A Survey | Machine Learning, 2023 | Calibration (ECE/Brier) |
| 5 | Ji et al., Table Integration in Data Lakes Unleashed | The VLDB Journal, 2025 | Conflict resolution |
| 6 | Gregori et al., PROLIT: LLM-guided Multi-granular Data Provenance | Journal of Big Data, 2025 | Cell-level provenance |
| 7 | Hollmann et al., Accurate Predictions on Small Data with a Tabular Foundation Model | Nature, 2025 | Downstream tabular ML |

Kasem et al. (ACM Computing Surveys, 2024) is cited in prose for table
detection only. Hendrickx et al. (Machine Learning, 2024) is cited in prose for
the reject-option mechanism.

**Not yet implemented, and claimed nowhere in the code:** a *learned* combiner
for the evidence signals (the weights are hand-set), and calibrated
probabilities over the decision (there is no ECE/Brier number yet). Those are
the pillars from papers 1 and 4 and are the next step.

---

## 4. What runs today

One command, with a local OpenAI-compatible model serving on port 8080:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -e .
uv run --python .venv python scripts/run_demo.py
```

On the bundled synthetic dataset (327 records over CSV, Excel, PDF and receipt
images):

| | Result |
|---|---|
| Vendor clusters | **20**, ground truth 20 |
| Duplicate invoices removed | 108 of 307 raw rows |
| Total spend | 6,185,400 INR — matches ground truth exactly |
| Cross-source conflicts | 20 planted, **20 detected, 0 resolved incorrectly** (load order gets 7 wrong) |
| Question answering, top-k retrieval | **10/10** consolidated vs **1/10** RAG over raw records vs **4/10** RAG over the consolidated table |
| Question answering, full corpus in context | RAG over raw **20%**, RAG over consolidated **60%**, pipeline **100%** |
| Downstream prediction | balanced accuracy **0.899** vs **0.749** on the raw table |

Three arms share one model, one prompt and one question set; the two RAG arms
share one vector store, one embedding model and one retrieval depth, so the
corpus is the only thing that differs between them. The full-corpus ablation
removes retrieval truncation as a confound: even with every consolidated row in
context, RAG only reaches 60%, because it asks the language model to sum 199
rows. The pipeline plans a query and executes it instead.

Full numbers and caveats: [`research/results/SUMMARY.md`](research/results/SUMMARY.md).
How the demo is framed and what it deliberately does *not* claim: [`DEMO.md`](DEMO.md).

Verify the canonical output against ground truth at any time:

```bash
uv run --python .venv python tests/test_gold.py
```
