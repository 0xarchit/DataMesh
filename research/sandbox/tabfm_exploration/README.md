# TabFM Exploration: Downstream Tabular Intelligence

This sandbox module evaluates **Google Research's TabFM** (Tabular Foundation Model) as the primary downstream intelligence and verification component for **DataMesh AI**.

---

## 1. Role in DataMesh AI

> [!IMPORTANT]
> **Separation of Research Contribution:**
> TabFM is **not** the data integration algorithm, and its pre-trained weights are not the research novelty of this B.Tech capstone project. 
> * **Multimodal Front-End (Qwen / OCR):** Ingests and parses heterogeneous enterprise files.
> * **Our Core Novelty:** The evidence-aware reconciliation engine ($E_1$ through $E_7$) and the calibrated risk-sensitive decision layer (`MERGE`, `REVIEW`, `REJECT`, `ABSTAIN`).
> * **TabFM's Role:** Acts as an objective downstream benchmark to prove **Hypothesis $H_3$**—that evidence-backed data consolidation directly yields measurable accuracy improvements in downstream machine learning tasks compared with uncurated, dirty, or naive LLM-merged data.

---

## 2. What is TabFM?

**TabFM** is a foundation model developed by Google Research designed for zero-shot classification and regression on tabular data without task-specific gradient fine-tuning:
* **Zero-Shot Generalization:** Evaluated across the **TabArena** benchmark spanning 38 classification and 13 regression datasets, demonstrating superior performance over standard tree-based models (XGBoost, CatBoost) under zero-shot and few-shot regimes.
* **Invariant In-Context Learning:** Treats table columns and cell values as structured token sequences, learning cross-column relationships and numerical distributions directly in-context.
* **Sensitivity to Noise:** Like all tabular models, TabFM's predictive performance degrades sharply when fed tables containing false entity merges, contradictory duplicate rows, or unnormalized categorical values.

---

## 3. Downstream Evaluation Protocol (Testing $H_3$)

To rigorously validate the real-world business value of DataMesh AI, we test TabFM predictions across three comparative variants of the same enterprise data:

```text
  Raw Uncurated Sources          Naive Direct LLM Merge          DataMesh AI Canonical Asset
(Disparate PDFs, CSVs, XLSX)    (Prompt-based auto-join)      (Evidence-Aware Reconciled)
            │                              │                               │
            ▼                              ▼                               ▼
     TabFM Prediction               TabFM Prediction                TabFM Prediction
  (High Noise / Errors)         (Silent False Merges)           (Calibrated / Provenance)
            │                              │                               │
            └──────────────────────────────┼───────────────────────────────┘
                                           ▼
                               Comparative Evaluation:
                       Accuracy, F1-Score, RMSE, Calibration
```

### Planned Benchmark Tasks
1. **Financial Risk & Loan Default:** Consolidating borrower history across annual PDF credit reports, legacy CSV transaction logs, and monthly Excel statements.
2. **Customer Churn & Lifetime Value (LTV):** Integrating fragmented CRM records and billing systems across differing customer IDs.
3. **Inventory & Supply Chain Discrepancies:** Unifying product catalogue codes and supplier invoices.

---

## 4. Experimental Artifacts & Next Steps

* [ ] Implement test runner converting DataMesh canonical SQLite/Parquet exports into TabFM-compatible inference tensors.
* [ ] Benchmark zero-shot ROC-AUC / F1 on clean vs. noisy integrated datasets.
* [ ] Measure feature attribution stability to assess whether cell-level provenance aids model explainability.
