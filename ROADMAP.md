# DataMesh AI — How to Get From Here to a Finished, Published Project

**For:** Archit & Deepti. **Where you are:** ~1 month in, 7 papers selected, nothing built yet.
**Don't panic.** One month of "lost" time is normal. The path below is concrete; follow it week by week.

Your faculty gave three deliverables. This guide turns them into steps:
1. **Read papers + have backing** → Phases A–B below.
2. **Build the product** → Phases C–D.
3. **Publish a conference paper** → Phase E (but you start writing in Phase A).

---

## 0. Reality check (read this once, then move on)

- **You have roughly 4 months of usable time.** That is enough for a *small, honest, complete* result. It is NOT enough for the full system the synopsis describes. **You will scope down** — that is not failure, it is how real research works.
- **"Publish a conference paper" — be realistic about the venue.** Two undergraduates in one semester will not land NeurIPS/VLDB. Realistic targets: a student research symposium, an IEEE/Springer regional or applied conference, an indexed workshop, or at minimum an arXiv preprint plus your college's requirement. **Ask your faculty which specific venue/deadline they expect** — that date drives everything. Write it at the top of your calendar.
- **The single most important rule: build the smallest thing that runs end-to-end first, then make it bigger.** A tiny pipeline that takes one receipt image plus one CSV, extracts and matches the records, and prints an F1 score is worth more in month 2 than a perfect PDF parser built from scratch. Multimodal stays in from day one — you just start with *one example of each format*, not all of them.

---

## 1. Your "vs RAG + prediction" plan — you're right; frame it as two levels

You don't mean RAG as your entity-matching baseline. You mean the *common way people answer questions and predict from data with an LLM*, and you want to show your consolidation pipeline makes a **large, measurable difference on financial data** for (a) answering questions over multimodal data and (b) predicting values with TabFM-like models. That is a strong, concrete story. Keep it clean by splitting it into **two levels of comparison**, because they answer different questions and need different baselines.

### Level 1 — the research core: consolidation quality (this is your novelty)

Here you prove the pipeline *reconciles* better than alternatives. RAG has no place here — the baselines are matching methods. This is where the ablation lives.

| Baseline | What it is | Why it's in the study |
|---|---|---|
| **B1. String/lexical** | Match on edit distance / exact match of names | The classic floor |
| **B2. Semantic-only** | Match on embedding cosine similarity (Sentence-BERT) | The "obvious modern" approach |
| **B3. LLM-only** | Prompt an LLM to decide the merge directly | The trendy approach; H1 must beat it |
| **B4. (optional) an existing tool** | e.g. a Magellan/DITTO-style matcher | A real research baseline |
| **YOURS** | 7-signal evidence vector -> calibrated MERGE/REVIEW/REJECT/ABSTAIN | The thing you're proving |

### Level 2 — the value demonstration on financial data (why consolidation matters)

**This is where "vs RAG" belongs.** Two downstream tasks, each run **two ways** — on the raw multimodal data vs. on your consolidated canonical dataset:

- **(a) Question answering.** Ask financial questions (e.g. *"total spend with vendor X across all these invoices?"*).
  - *RAG-style baseline:* retrieve from the raw documents and let an LLM answer.
  - *Yours:* consolidate first (dedupe entities, resolve conflicts, keep provenance), then answer over the clean canonical table.
  - *Expected win:* the RAG-style answer double-counts duplicate entities and trips on conflicting values; yours is correct and traceable. **This is the "large difference" you want to show.**
- **(b) Prediction with TabFM.** Predict a financial target (e.g. a risk flag, or a numeric value).
  - *Baseline:* TabFM on the raw / naively-merged data.
  - *Yours:* TabFM on the consolidated dataset.
  - *Expected win:* cleaner data -> better prediction. This is exactly your **H3**.

**Two honest cautions for Level 2 — get these right or the demo won't count as evidence:**
1. **You need gold answers.** To *measure* the QA win (not just demo it), build a small set of ~20–50 financial questions with known-correct answers over a small document set. Without ground truth it is a demo, not a result.
2. **TabFM predicts; it does not forecast time-series natively.** TabPFN v2 does tabular classification/regression — predict a target *column* from feature *columns*. If "future values" means genuine time-series (next month from a sequence), that is a different, harder setup. The clean, honest framing is **regression on the consolidated table** (predict an attribute/target), which TabPFN does well. Say "prediction"; only say "forecasting" if you actually build a time-ordered target and use a forecaster.

**"Other similar products" (Julius AI etc.)** are closed commercial tools — compare them **qualitatively** in a product-landscape paragraph (what they do vs. what you add: reconciliation, provenance, abstention). Keep them out of the numbers table.

---

## 2. How to actually read a research paper (you said 0 idea — here's the method)

Use the **three-pass method** (this is the standard technique; look up "Keshav, How to Read a Paper"):

- **Pass 1 (5–10 min):** Read title, abstract, section headings, figures, and conclusion. Answer: *What problem? What's their method in one sentence? What did they show?* Stop.
- **Pass 2 (~1 hr):** Read the whole thing but skip heavy math proofs. Understand the method and the experiments. Note anything you don't understand.
- **Pass 3 (only for your 2–3 most important papers):** Re-implement the idea in your head; understand every step.

**For each of your 7 papers, fill one row** (a shared Google Sheet or a `notes/` markdown file):

| Field | What to write |
|---|---|
| Problem | What they were solving |
| Method | Their approach, 2–3 lines |
| What I reuse | The idea/method/dataset you'll borrow |
| The gap | What they *don't* do that you will |
| My one-sentence delta | "They did A; we extend to B" |

**Shortcut: this is already done for you.** Open `academic/literature/references.json` — every paper has `one_liner`, `relevance`, `how_to_cite`, and `caveat` fields. Read the actual PDF to confirm, but the notes are a head start. The `caveat` line for each is the thing that keeps you honest in the viva.

**Budget:** 2 papers/day for about 4 days = all 7 read in Week 1. Split them: each of you reads all 7 at Pass-1, then each takes 3–4 for Pass-2 and teaches the other.

---

## 3. The build — scope it down HARD (this is where projects die)

The synopsis lists PDF + scans + XLSX + CSV ingestion, 7 evidence signals, a trained calibrated classifier, fusion, provenance, a review-queue UI, a dashboard, and downstream ML. **You cannot build all of that.** Here is the order that guarantees you always have *something working*:

### Build the smallest end-to-end slice first — but keep it multimodal

Multimodal IS your thesis: receipts, bills, scanned images and PDFs must become structured data. So the first runnable slice includes **one example of each format**, not CSV alone. You keep it small by using **off-the-shelf extractors** — do NOT build your own OCR/parser. Building a parser is a huge project on its own and is *not your contribution*; your contribution is the evidence + decision layer that comes after extraction.

```
1 receipt image + 1 PDF bill + 1 CSV  ->  off-the-shelf extractor  ->  structured records
     ->  candidate pairs  ->  a few evidence signals  ->  a decision  ->  an F1 score
```

Concretely, the **minimum viable pipeline**:
1. **Extraction (core, not optional).** Push one receipt image and one PDF bill through an off-the-shelf multimodal extractor — a vision-language model (Qwen-VL via API, or a hosted OCR) or a document tool such as Docling/Tesseract — to get `field -> value` rows with a per-field confidence. Load a CSV directly. All three land in the **same intermediate record format** so the rest of the pipeline doesn't care where a row came from.
2. Generate candidate record pairs (start naive: compare within blocks by first letter — this is "blocking").
3. Compute **3 evidence signals** to start (not all 7): E1 semantic (embedding cosine), E3 value-pattern (edit distance / type match), E5 source-provenance / extraction-confidence.
4. Feed the signal vector to a classifier (LightGBM or logistic regression) -> probability.
5. Threshold into MERGE / REVIEW / REJECT / ABSTAIN.
6. Compute F1, precision, recall, and False-Merge Rate against gold labels.

**When one image + one PDF + one CSV flow through to a decision and a number, your multimodal claim is real and demonstrable.** Everything after is widening.

### Then widen, in this priority order

1. Add more source formats and more documents (XLSX, additional receipt/bill layouts, scans) through the *same* extractor interface, so "multi-format" is genuinely broad, not a single example.
2. Add the remaining evidence signals (E2 structural, E4 coreference, E6 temporal, E7 cross-source) — this gives you the **ablation study** (your core result).
3. Add **calibration** (PyCalib: temperature/isotonic scaling) + **ECE/Brier** reporting -> backs H2.
4. Add the **downstream ML check** (TabPFN v2 on raw vs. consolidated table) -> backs H3.
5. **Last:** a thin product wrapper (a FastAPI endpoint + a simple page, or a Streamlit app) so it *looks* like a product for the demo.

> **De-risking rule (multimodal stays in):** the risk was never "multimodal" — it is *building your own extractor*. Neutralize it by treating extraction as a black box you call (a VLM/OCR API or Docling), so a bad OCR result never blocks the evidence-and-decision layer that is your actual research contribution. Get one image and one PDF working in the very first slice; if one specific format misbehaves later you still have the others, but the multimodal claim is already proven.

### Tech stack (all free, standard, no exotic dependencies)

| Need | Use |
|---|---|
| Language | Python 3.11 |
| Data | `pandas` |
| **Multimodal extraction (core)** | **Qwen-VL (API) or a hosted OCR for receipts/bills/scans; `Docling` or `Tesseract` for PDFs — call as a black box, don't build one** |
| Embeddings (E1) | `sentence-transformers` |
| String signals (E3) | `rapidfuzz` (edit distance, Jaro-Winkler) |
| Classifier | `lightgbm` or `scikit-learn` (LogisticRegression) |
| Calibration + ECE/Brier | `PyCalib` (from the Silva Filho paper) + `scikit-learn` |
| Downstream ML (H3) | `tabpfn` package + `xgboost` baseline |
| Benchmark data | SMBench datasets (you hold the paper) + public ER datasets (Abt-Buy, Amazon-Google, DBLP-ACM — all free) |
| Product wrapper (last) | `fastapi` + one HTML page, or `streamlit` |

Your `src/datamesh/` folders already match the pipeline — build into them:
`ingestion/` (CSV loader first) -> `evidence/` (the signals) -> `reconciliation/` (classifier + 4-way policy) -> `fusion/` (conflict resolution) -> `core/` (schemas, config).

---

## 4. Experiments & results (this is the "backing")

Two results tables and two plots are enough for a solid paper:

- **Table 1 — Ablation.** Rows = evidence configs (E1 only; E1+E3; E1+E3+E7; all signals). Columns = Precision, Recall, F1, False-Merge Rate. *This proves H1: more signals -> better.*
- **Table 2 — Baselines vs. yours.** Rows = B1 string, B2 semantic, B3 LLM-only, YOURS. Same columns. *This is the head-to-head.*
- **Plot 1 — Risk-coverage curve.** X = coverage (fraction auto-decided), Y = error rate. Shows abstention buys reliability. *Proves H2.*
- **Plot 2 — Calibration reliability diagram** (from PyCalib) + report ECE.
- **Table 3 — Downstream prediction.** TabPFN accuracy/F1/RMSE predicting a financial target on raw/naively-merged vs. consolidated data. *Proves H3.*
- **Table 4 — Downstream question-answering.** Answer accuracy on ~20–50 financial questions with known answers: RAG-style over raw documents vs. answering over your consolidated table. *Shows the consolidation payoff for the common LLM-QA use-case — the "big difference" you want to demonstrate.*

Run everything on the **same** datasets under the **same** blocking, so comparisons are fair. Log every run (a CSV of results) so you never lose a number.

---

## 5. Writing the paper (start early, finish in Phase E)

You already have a huge head start: `synopsis.tex` -> the paper's **Introduction, Related Work, and Method** are ~70% written. A conference paper is:

1. **Abstract** (write last)
2. **Introduction** — from synopsis Ch1
3. **Related Work** — from synopsis Ch2 (your 7 papers, already written honestly)
4. **Method** — the pipeline + the 7 evidence signals + the decision policy
5. **Experiments** — datasets, baselines, metrics (Section 4 above)
6. **Results** — your tables/plots + discussion
7. **Conclusion & Future Work** — from synopsis Ch3

Use the venue's template (IEEE `IEEEtran` or ACM/Springer LNCS — the venue tells you). Keep it 6–8 pages.

**Rule for honesty (this sank your first draft — don't repeat it):** never write "paper X shows Y" unless X literally shows Y. Use the `caveat` fields in `references.json`. If a result is negative ("more signals didn't help on dataset Z"), *report it* — negative results are legitimate and examiners respect them.

---

## 6. Week-by-week calendar (~16 weeks; adjust to your real deadline)

| Weeks | Phase | Deliverable at the end |
|---|---|---|
| **1–2** | **A. Read + freeze scope** | All 7 papers read & noted; one-page scope doc: exact datasets, exact baselines, exact metrics. Paper skeleton (`.tex`) created with Intro + Related Work pasted from synopsis. |
| **3–4** | **B. Data + harness** | Benchmark datasets downloaded; blocking + gold-label loader working; the 3 baselines (string, semantic, LLM-only) run and print F1. **This alone is a milestone — show faculty.** |
| **5–7** | **C. Core pipeline (multimodal slice)** | End-to-end on 1 image + 1 PDF + 1 CSV: off-the-shelf extractor -> structured records -> evidence vector -> classifier -> 4-way decision -> metrics. |
| **8–9** | **C. Widen** | More formats/documents through the same extractor; all 7 signals implemented; calibration + ECE/Brier working. |
| **10–11** | **D. Experiments** | Ablation table, baseline table, risk-coverage plot, calibration plot — all generated from scripts. |
| **12** | **D. Downstream** | TabPFN raw-vs-consolidated result; broaden the document/format coverage. |
| **13** | **E. Product wrapper** | Minimal FastAPI/Streamlit demo for the viva. |
| **14–15** | **E. Write** | Full paper draft; faculty review; revise. |
| **16** | **E. Submit + final report** | Submit to the venue; finalize the semester report and demo. |

**Writing runs in parallel** — add to the paper every week, don't leave it to Week 14.

---

## 7. Who does what (two people, split cleanly)

Split by pipeline half so you rarely edit the same file:

- **Person 1 — "data & evidence":** ingestion, blocking, evidence signals (E1–E7), datasets, baselines B1/B2.
- **Person 2 — "decision & eval":** classifier, calibration, 4-way policy, metrics, plots, LLM-only baseline B3.
- **Both:** read all 7 papers; co-write the paper; Person 1 owns Method + Experiments prose, Person 2 owns Results + Related Work.

Use **git from day one** (see section 8). Agree: `main` always runs; each person works and commits their half daily.

---

## 8. Habits so you don't lose another month

- **`git init` today.** The repo is not under version control — that's the first fix. Commit every working day. (`.gitignore` the PDFs and data if they're large.)
- **Weekly 30-min check-in with each other**, and a short update to faculty every 2 weeks. Momentum comes from small visible wins.
- **"Always green":** never leave the code in a broken state overnight. If it runs and prints a number, commit it.
- **One results log** (`research/experiments/results.csv`): every experiment appends a row. You will thank yourself when writing.
- **When stuck for >2 hours:** stop, write down exactly what's blocking, and ask (faculty, a forum, or here). Don't grind silently.

---

## 9. Your very next actions (this week)

1. **Ask faculty the target venue + deadline.** Everything scales to that date.
2. `git init` and commit the current repo.
3. Read papers 1–4 (Pass 1 + Pass 2), fill the notes table.
4. Two Week-1 "hello worlds": (a) load one benchmark dataset (Abt-Buy or a SMBench dataset) in a Python script and print the rows + gold labels; (b) push **one receipt image and one PDF** through an off-the-shelf extractor (Qwen-VL API or Docling) and print the `field -> value` rows it returns. Those two prove the structured path and the multimodal path both work.
5. Create `paper/main.tex` from the synopsis Intro + Related Work.

Do those five things and you are no longer behind — you're moving. Everything else in this guide is just "keep going."

---

*Reference files already in your repo: `academic/literature/references.json` (per-paper notes), `academic/synopsis/synopsis.tex` (paper starting point), `results.md` (why these 7 papers, how to cite each). The `src/datamesh/` folders already match the pipeline stages — build into them.*
