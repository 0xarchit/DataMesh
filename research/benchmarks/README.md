# Integration Benchmarks & Ground Truth Datasets

This directory organizes the public academic benchmarks and gold-standard evaluation harnesses used to test **DataMesh AI** across all pipeline stages.

---

## 1. Primary Benchmark Suite

| Benchmark | Academic Origin | Target Task | Primary Evaluated Metrics |
|---|---|---|---|
| **MaDI-Bench** | End-to-end Data Integration Suite (2025/2026) | Schema Matching, Value Normalization, Entity Resolution, Data Fusion | Precision, Recall, Macro-F1, Fusion Accuracy |
| **SMBench** | Astappiev et al., *Information Systems* (2026) | Supervised & Deep Learning Entity Matching | Blocking Reduction Ratio, Pair Completeness, Matching F1 |
| **ExStrucTiny & VAREX** | Document & Variable-Schema Extraction (ACL / EMNLP) | Layout-aware document extraction from PDFs/images | Cell Coordinate IOU, Header Precision, Value Extraction F1 |
| **TabArena** | Google Research Tabular Suite | Downstream Tabular Machine Learning | Zero-shot Classification ROC-AUC, Regression RMSE |

---

## 2. Benchmark Deep-Dives

### 2.1 MaDI-Bench (Primary End-to-End Suite)
* **Scope:** 20 comprehensive integration tasks across five distinct real-world domains (E-commerce, Healthcare, Financial Services, Demographics, and Corporate Registries).
* **Ground Truth Annotations:** Provides triple gold-standard labels:
  1. *Schema Match:* Target column alignment pairs.
  2. *Entity Match:* Ground truth coreference cluster IDs.
  3. *Data Fusion:* Verified consensus value for every multi-source attribute conflict.
* **Role in DataMesh AI:** Serves as the primary end-to-end benchmark to measure overall canonical dataset reconstruction accuracy.

### 2.2 SMBench (Entity Matching Robustness)
* **Scope:** Evaluates modern learning-based entity matching frameworks across structured, semi-structured, and dirty tabular inputs.
* **Role in DataMesh AI:** Used specifically in **Phase 5** to test our selective decision layer (`MERGE`, `REVIEW`, `REJECT`, `ABSTAIN`) under simulated domain shift, proving that our calibrated boundaries prevent the catastrophic accuracy collapse observed in existing binary matchers.

### 2.3 VAREX & ExStrucTiny (Multimodal Ingestion)
* **Scope:** Benchmarks variable-schema structured extraction from heterogeneous PDFs, scanned invoices, and receipt images.
* **Role in DataMesh AI:** Validates the front-end extraction adapters (Qwen / OCR), ensuring that visual coordinates, table cell boundaries, and extraction confidences are faithfully captured.

---

## 3. Directory Layout

```text
research/benchmarks/
├── README.md               # This document
├── madi_bench/             # Downloaded MaDI-Bench evaluation harnesses & gold labels
├── smbench/                # SMBench datasets (structured, dirty, semi-structured)
├── varex_docs/             # Sample heterogeneous enterprise documents & ground truths
└── downstream_tabarena/    # Selected TabArena classification/regression tasks
```
