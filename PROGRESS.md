# DataMesh AI — Progress Report

**Project:** *Evidence-Aware Multimodal Data Integration for Reliable Enterprise Data Consolidation*
**Product:** DataMesh AI
**Authors:** Archit Jain (2415800016), Deepti Yadav (2415800029) — B.Tech (Honours) CSE, GLA University, Mathura
**Status date:** October 2026 — working end-to-end system with measured results
**Purpose of this file:** hand-off to the paper-writing teammate. Everything here is either implemented and re-runnable, or is marked as planned. Section 3 maps each synopsis reference to the part of the system it informed.

---

## 1. What the system does

Enterprises hold the same business facts in incompatible containers: a customer master CSV, an Excel ledger, generated/scanned PDF invoices, and photographed receipts. Extraction from each format is now routine; **integration between them is not**. Two failures survive extraction:

1. **Entity fragmentation** — one vendor appears as `Acme Ltd`, `ACME`, `Acme Limited` across sources, so every aggregate computed over the union is wrong.
2. **Silent conflict** — the same invoice number can appear in two sources with two different amounts. A pipeline that simply loads files in order keeps whichever copy it read first, with no record that a disagreement ever existed.

DataMesh AI is an **evidence-aware reconciliation and decision layer** that fixes both before any downstream use, and keeps a provenance trail for every value it emits.

---

## 2. Architecture and what is implemented

### Stage 1 — Multimodal ingestion
Format-specific adapters produce a uniform `Record(source_type, source_file, invoice_id, vendor, amount, date, city, tax_id, confidence)`.

| Source | Adapter | Records |
|---|---|---|
| Customer master CSV | deterministic column map | 20 |
| Excel ledger | `openpyxl`, header-mapped | 100 |
| PDF invoices | `pypdf` text extraction + labelled-field parsing | 199 |
| Receipt images (PNG) | **local vision-language model** via an OpenAI-compatible endpoint, JSON-mode extraction | 8 |

**Total: 327 extracted records over four formats.**

### Stage 2 — Evidence-aware identity resolution
A **seven-signal evidence vector** is computed for every (record, candidate cluster) pair and fused by a weighted combiner:

| Signal | Meaning |
|---|---|
| E1 semantic name | normalised string/token similarity |
| E2 tax-id match | GSTIN agreement |
| E3 city agreement | agreement after city-alias normalisation (Bombay→Mumbai, Bangalore→Bengaluru, …) |
| E4 coreference | is one name a whole-word subset or acronym of the other |
| E5 provenance confidence | source reliability × extraction confidence |
| E6 temporal freshness | distance between record date and the newest date in the cluster |
| E7 cross-source consensus | how many distinct source types already agree on this identity |

Clusters advance their own state (`last_date`, `source_types`) as they grow, so E6 and E7 are computed against real cluster history rather than a static snapshot.

**Result: 20 vendor clusters — exactly matching the 20 ground-truth vendors.**

### Stage 3 — Cross-source conflict resolution
When one invoice number appears in more than one source with a disagreeing amount, the fusion layer resolves it through a defined escalation:

1. **Agreement across copies** → nothing to decide (`agreed`).
2. **Strict majority of copies** → majority wins (`resolved`).
3. **No majority** → the vendor's own **amount profile** breaks the tie: a real amount recurs across that vendor's other invoices, a digit-shift misread does not. Source reliability is consulted only when it separates the candidates by a meaningful margin.
4. **Still undecided** → the row is **flagged for human review** (`review`) rather than silently merged. This is the abstain branch.

Every decision, its candidate values, the competing sources and the rule that settled it are written to `data/canonical/conflicts.json`.

**Result: 20 conflicts planted, 20 detected, 20 resolved correctly, 0 wrong.** The load-order baseline (keep the first copy read) gets **7 of them wrong**.

### Stage 4 — Canonical output with provenance
`transactions.csv` (199 unique invoices), `vendors.json`, `provenance.jsonl` (per-value lineage back to the originating file), `conflicts.json`, `reconciliation_report.json`.

### Stage 5 — Consumption layer
* **Question answering** — the model converts a natural-language question into a **query plan** (`op`, `entity`, `column`); a deterministic executor runs it against the canonical table. The language model plans; it never computes the numbers.
* **Downstream tabular prediction** — vendor-level features feed a tabular classifier, trained on the raw table vs the consolidated table.
* **Baseline RAG arm and control arm** — for the comparative evaluation in §4.

---

## 3. How each synopsis reference was used

This is the section the paper's Related Work and Methodology can be built from directly.

| # | Reference | What we took from it | Where it lives in the system | Our extension |
|---|---|---|---|---|
| 1 | **SMUTF** — Zhang et al., *Information Systems* 133:102570, 2025 | The core idea that heterogeneous, weakly-correlated signals should be **fused into one combiner** rather than thresholded individually; fusing lexical, value-distribution and embedding-level evidence | `src/datamesh/evidence/signals.py` — the E1–E7 vector and `combine()` | SMUTF fuses signals for **schema (column) matching**. We apply the same fusion discipline to **row-level merge-or-not decisions** across documents. |
| 2 | **Christen et al.** — *ACM J. Data & Information Quality* 17(2):7, 2025 | The failure mode: transitive closure (A~B, B~C) creates false merges, and repairing them after the fact is expensive | Motivated the **append-only, threshold-gated clustering** in `reconciliation/resolver.py` and the abstain branch | They **repair** bad clusters post hoc with graph metrics. We **prevent** the merge up front by refusing to merge under-evidenced pairs. |
| 3 | **Robach et al.** — *Statistics in Medicine* 44:e70292, 2025 | The evaluation protocol for **how often accepted links are wrong** — a linker should be measured on precision of accepted merges, not only recall | `scripts/run_comparison.py::score_conflicts()` and `tests/test_gold.py` — every planted conflict is scored against generator ground truth | We score **per-conflict decision correctness** against ground truth. Full false-merge-rate estimation with decoy records is planned (§5). |
| 4 | **Silva Filho et al.** — *Machine Learning* 112, 2023 | The calibration vocabulary (ECE, Brier, reliability diagrams) and the principle that a decision score should behave like a probability before it drives a merge | Shaped the decision taxonomy — the outcome set is designed around a score that *should* be calibrated | Calibration itself is **not yet applied**: `combine()` currently returns an uncalibrated score. This is the clearest next-phase item (§5). |
| 5 | **Ji et al.** — *The VLDB Journal* 34:36, 2025 | The architectural template: integration as a staged pipeline ending in **multi-tuple conflict resolution** rather than a single similarity threshold | `src/datamesh/fusion/consensus.py` — the conflict stage | They **assume schema alignment is already done** and always emit one merged tuple. We run end-to-end from raw multi-format sources and can **abstain** (`review`) instead of emitting a forced merge. |
| 6 | **Gregori et al. (PROLIT)** — *Journal of Big Data* 12:187, 2025 | Multi-granular provenance: every emitted value should be traceable to its source at cell level | `data/canonical/provenance.jsonl` — per-value lineage with source file, source type and match score | PROLIT **records** provenance. We additionally **consume** it as a decision input: E5 (source reliability × extraction confidence) is a weighted term in the merge function. |
| 7 | **Hollmann et al. (TabPFN v2)** — *Nature* 637:319, 2025 | In-context tabular foundation models are competitive on **small** tabular datasets without task-specific training — which is precisely our regime (20–200 rows) | `src/datamesh/downstream.py::build_model()` — wires `TabFMClassifier` (Google Research's sklearn-compatible tabular FM) with an automatic fallback | Extension: we evaluate the same model on the **raw vs consolidated** table, turning consolidation quality into a downstream accuracy difference. |

**Cited in prose, not among the seven:**
* **Kasem et al.**, *ACM Computing Surveys* 56(12), 2024 — table detection and structure recognition; scopes the extraction stage honestly as using off-the-shelf methods.
* **Hendrickx et al.**, *Machine Learning* 113(5), 2024 — the reject-option / abstention mechanism; the direct basis for the `review` outcome in the conflict resolver, which **is implemented**.

---

## 4. Measured results

All figures below are produced by re-running the system; the raw JSON is in `research/results/`.

### 4.1 Consolidation

| Metric | Value |
|---|---|
| Extracted records | 327 (CSV 20, XLSX 100, PDF 199, receipt images 8) |
| Vendor clusters | **20** (ground truth: 20) |
| Raw transaction rows → unique invoices | 307 → **199** (108 duplicates removed) |
| Total spend | **₹6,185,400** — matches ground truth exactly |

### 4.2 Cross-source conflict resolution

| | |
|---|---|
| Conflicts planted in the corpus | 20 |
| Detected | **20** (0 missed, 0 spurious) |
| Resolved correctly | **20** |
| Flagged for review | 0 |
| **Load-order baseline: wrong** | **7** |
| **DataMesh: wrong** | **0** |

### 4.3 Question answering — three arms, one model, one question set

The two RAG arms share one vector store (ChromaDB), one embedding model (all-MiniLM-L6-v2, 384-dim), and one retrieval depth (k=15). **Only the corpus differs**, which is what makes this a controlled comparison rather than two unrelated runs.

| Arm | top-k retrieval | full corpus in context |
|---|---|---|
| **DataMesh (query plan + deterministic execution)** | **10/10** | **100%** |
| RAG over the consolidated table | 4/10 | **60%** |
| RAG over the raw extracted records | 1/10 | **20%** |

This decomposes the result into three levels, which is the strongest form of the argument:

1. **20%** — no consolidation: duplicate rows inflate totals and counts, name variants split one vendor into several, city aliases are never normalised.
2. **60%** — consolidated corpus, but the language model is still asked to **aggregate in context** over 199 rows, which it cannot do reliably.
3. **100%** — consolidation **plus** query planning: the model plans, the table computes.

The control arm and the full-context ablation exist specifically to close off the two obvious objections — "your RAG was misconfigured" and "the gap is just retrieval truncation".

**Provenance of the arms themselves.** Both the RAG engine and the QA planner are recorded in `research/results/comparison.json`, so the run is auditable rather than merely asserted:

```json
"rag_engine": "local llm: qwen-3.8-27B",
"qa_plan_source": {"query plan from local model": 10}
```

Both subsystems carry a deterministic, hand-written **fallback** (`rag.keyword_rules_answer`, `qa.plan_rules`) so the demo and the offline test still run with no model serving. Each fallback announces itself when it fires, and neither contributed to the numbers above — all ten query plans came from the model. The offline gold test runs with `use_llm=False`, which deliberately exercises the fallback planner; it therefore validates the **executor**, not the model's planning.

### 4.4 Downstream tabular prediction

Balanced accuracy and macro F1 are reported because the two arms do not have the same row count (fragmentation is the effect under study) and both metrics are invariant to class prevalence.

| Table | Rows | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|---|
| Raw (grouped by vendor string as written) | 41 | 0.7561 | 0.7488 | 0.7488 |
| **Consolidated** | 20 | **0.9000** | **0.8990** | **0.8990** |

### 4.5 Verification

`tests/test_gold.py` asserts the canonical output against generator ground truth on four axes — invoice set, total spend, cluster count, per-conflict correctness, and the receipt image set. It runs in seconds and is the regression guard for every change to the resolution or fusion layer.

---

## 5. Next phase (planned, not yet implemented)

Stated as design intent so the paper's "Future Work" and "Limitations" sections can be written without overclaiming:

1. **Calibrated decisions** — replace the hand-set combiner weights with a trained combiner, and calibrate the merge score so it behaves as a probability (ECE, Brier, reliability diagrams — paper 4).
2. **Learned signal fusion** — the current weights are hand-tuned and dataset-specific. Paper 1's approach (a learned combiner over heterogeneous features) is the target.
3. **False-merge-rate estimation** — apply paper 3's decoy-record method to estimate how often an accepted merge is wrong, instead of only scoring against ground truth.
4. **Entity-matching baselines** — string/lexical, semantic-only (Sentence-BERT cosine), and LLM-only matching, so the pipeline is compared against established matchers and not only against itself.
5. **Tabular foundation model run** — install the tabular FM backend and report its numbers alongside the current gradient-boosting fallback.
6. **Temporal forecasting task** — extend beyond static prediction to a genuine **next-N-period expenditure forecast**: build a daily spend series, derive lag and calendar features, and predict the following 10-day total. Compared as naive baseline vs gradient boosting vs tabular FM, and — crucially — run on **both** the raw and the consolidated series, so consolidation is shown to help a *predictive* task and not only question answering.
7. **Scale and robustness** — larger synthetic corpora, multi-seed runs with confidence intervals, and an evaluation harness in the style of SMBench.

---

## 6. How to run the demo

**Prerequisite:** a local OpenAI-compatible endpoint on port 8080 (llama.cpp `llama-server` with the Qwen model). It serves both the vision receipt extraction and the RAG arms.

```powershell
cd D:\B.Tech\ProjeX\DataMesh
.venv\Scripts\python.exe scripts\run_demo.py
```

From a clean checkout:

```powershell
cd D:\B.Tech\ProjeX\DataMesh
uv venv .venv --python 3.11
uv pip install --python .venv -e .
uv run --python .venv python scripts\run_demo.py
```

Verify the output against ground truth at any time:

```powershell
.venv\Scripts\python.exe tests\test_gold.py
```

**Pre-demo checklist**

1. Model server running and reachable:
   `.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');from datamesh.core import llm;print(llm.available(), llm.model_name())"`
   — must print `True` and the model name. If it prints `False`, receipts are skipped and the RAG arms fall back to keyword rules.
2. Run `tests\test_gold.py` → must print `all gold checks passed`.
3. Run the full demo once and **time it**. It makes ~40 model calls (10 questions × 3 arms, plus 10 query plans); on a local model this takes minutes. If it is too slow for the slot, the full-context ablation can be skipped.
4. Open `research/results/SUMMARY.md` and `research/results/summary.png` — these are the demo outputs.
5. Streamlit UI, if you want the interactive version: `.venv\Scripts\streamlit.exe run app.py`.

**Outputs produced**

| Path | Contents |
|---|---|
| `data/raw/` | the generated four-format corpus |
| `data/intermediate/records.json` | extracted records before consolidation |
| `data/canonical/` | canonical tables, provenance ledger, conflict report |
| `data/gold/` | ground truth: invoices, conflicts, receipts, questions |
| `research/results/` | `SUMMARY.md`, `summary.png`, `comparison.json`, `downstream.json` |

---

## 7. Note for the paper writer

Every number in §4 is reproducible from `research/results/*.json` by anyone who runs the demo, and the repository is self-verifying (`tests/test_gold.py`). **Use those figures, and only those figures.** Where the system has a gap, §5 states it plainly — a reviewer who finds an unstated gap costs far more than a limitation the authors declared themselves. Concretely: do not attribute ECE/Brier numbers, false-merge-rate estimates, learned-combiner results, entity-matching baseline comparisons, or a tabular-foundation-model result to this system yet — none of them have been run.

The honest framing is also the stronger one: the three-level decomposition in §4.3 (20% → 60% → 100%) is a more interesting result than a bare "our pipeline beats RAG" claim, and it survives questioning.
