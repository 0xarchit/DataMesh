# DataMesh AI demo

This is the runnable slice of the project. It shows one thing concretely:

**A consolidated, reconciled dataset answers questions about scattered
multi-format business data correctly, while traditional RAG over the same raw
data does not, because RAG never deduplicates records, resolves entity names,
or notices that two sources disagree about the same invoice.**

## What it does

1. Generates synthetic enterprise data across four formats: a CSV customer
   master, an Excel transaction ledger, PDF invoices, and PNG receipt images.
   The same vendor appears under several name variants, invoices are duplicated
   across formats, cities use old and new spellings, and **20 invoices are
   written into two or three sources with a different amount** — a digit-shift
   error, the kind a scanner produces.
2. Runs the consolidation pipeline: ingest all formats, cluster vendor
   identities with the E1..E7 evidence vector, **detect and resolve the
   cross-source conflicts**, deduplicate, and write canonical tables plus a
   provenance log and a conflict report.
3. Answers a gold question set from the canonical table.
4. Answers the same questions with a traditional RAG baseline over the raw text.
5. Trains a downstream tabular model on the raw table vs the consolidated table.

## Run it

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -e .
uv run --python .venv python scripts/run_demo.py
```

Outputs:

- `data/canonical/` — vendors, transactions, provenance, **conflicts**, report
- `data/intermediate/records.json` — raw extracted records
- `research/results/SUMMARY.md` — results summary
- `research/results/summary.png` — three-panel comparison chart
- `research/results/comparison.json`, `research/results/downstream.json`

Check the output against ground truth:

```bash
uv run --python .venv python tests/test_gold.py
```

## Local model required

The demo expects an OpenAI-compatible endpoint at `http://localhost:8080/v1`
(llama.cpp's `llama-server` works). It is used for two things:

- **Receipt images** — they need a vision model to be read at all. Without a
  server the receipt records are simply skipped, and the pipeline runs on three
  formats instead of four.
- **The RAG arm** — without a server, `rag_answer` degrades to hand-written
  keyword rules. That fallback prints `this arm is NOT a RAG run`, and
  `comparison.json` records which engine actually produced the number under
  `rag_engine`. Do not report a keyword-rules number as a RAG result.

Tuning knobs:

- `DATAMESH_LLM_BASE` (default `http://localhost:8080/v1`)
- `DATAMESH_LLM_MODEL` (default: auto-detected from `/v1/models`)
- `DATAMESH_LLM_THINKING` set to `1` to keep thinking mode (default off)
- `DATAMESH_LLM_MAX_TOKENS` (default 1536), `DATAMESH_LLM_TIMEOUT` (default 300)

`scripts/mock_llm_server.py` is a stub that speaks the same API. **It is a
wiring test only** — it returns a fixed fake receipt for every image, so the
injected records do not match the generated dataset and the reconciliation
totals will no longer match ground truth. Use it to check that the image path
is connected, then re-run against the real model:

```bash
uv run --python .venv python scripts/mock_llm_server.py &
DATAMESH_LLM_BASE=http://localhost:8099/v1 \
  uv run --python .venv python scripts/run_pipeline.py
```

## Individual steps

```bash
uv run --python .venv python scripts/generate_data.py
uv run --python .venv python scripts/run_pipeline.py
uv run --python .venv python scripts/run_comparison.py
uv run --python .venv python scripts/run_downstream.py
```

Two modules carry a self-check that needs no data and no model:

```bash
uv run --python .venv python src/datamesh/fusion/consensus.py
uv run --python .venv python src/datamesh/evidence/signals.py
```

## Honest scope

- This is a controlled demo on synthetic data, not a benchmark.
- **The RAG comparison uses three arms with one model, one prompt and one
  question set.** The two RAG arms share the same vector store, the same
  embedding model (`all-MiniLM-L6-v2`) and the same retrieval depth (k=15);
  only the corpus differs — raw extracted records vs the consolidated table.
  That is what makes it a control rather than two unrelated runs.
- **Retrieval truncation is a confound, so it is measured separately.** With
  k=15 of 327 raw chunks the model cannot see enough rows to total anything, and
  would fail for a reason unrelated to consolidation. Both RAG arms are
  therefore re-run with the whole corpus in context: raw 20%, consolidated 60%,
  against the pipeline's 100%. The residual gap is not retrieval and not
  consolidation — it is that RAG asks the language model to do the arithmetic
  over 199 rows, while the pipeline plans a query and executes it.
- **The RAG arms are deterministic-rules free when a model is reachable**:
  persistent Chroma collection, 384-dim MiniLM embeddings, real vector
  similarity retrieval, real generation over the retrieved context. Only the
  no-server fallback is rule-based, and it says so.
- **Vendor resolution uses a hand-weighted evidence vector**, not a trained
  calibrated classifier. The weights in `evidence/signals.py` were tuned
  against this dataset, and the merge threshold in
  `reconciliation/resolver.py` is tuned to match them. Changing one without
  re-checking the other will split or over-merge clusters — `tests/test_gold.py`
  catches that.
- **The conflict resolver knows which conflicts it cannot decide.** Those are
  written to `canonical/conflicts.json` with `"decision": "review"` instead of a
  guessed number. That branch is exercised by the unit self-check in
  `fusion/consensus.py`; the generated dataset happens to contain only
  resolvable conflicts, so the demo run reports 0 for review.
- **The downstream comparison reports balanced accuracy and macro F1**, not raw
  accuracy, because the two arms have different row counts (fragmentation is
  the effect being measured) and both metrics are invariant to class
  prevalence. The raw arm is also handed the ground-truth vendor list to assign
  labels, which is generous to it. Both arms are ~20–40 rows under
  leave-one-out CV, so treat the gap as indicative, not significant.
- The downstream model is a scikit-learn gradient boosting classifier. Install
  `tabfm[pytorch]` and `downstream.build_model()` will use `TabFMClassifier`
  instead, with the same fit/predict interface.
