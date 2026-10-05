# DataMesh AI: Core Architecture & Module Map

This directory contains the core Python package for **DataMesh AI**, structured around the four architectural phases defined in [`projectContext.md`](../../projectContext.md) and [`synopsis.tex`](../../synopsis.tex).

---

## Architecture Flow

```text
┌───────────────────────────┐
│     1. Ingestion          │  Adapters for PDF, scans, XLSX, CSV;
│   (datamesh.ingestion)    │  extracts cells, bounding boxes, and OCR confidences.
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│      2. Evidence          │  Computes the 7-dimensional evidence vector (E1...E7)
│    (datamesh.evidence)    │  spanning semantic, structural, and temporal features.
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│    3. Reconciliation      │  Calibrated selective classifier enforcing 4-way policy:
│ (datamesh.reconciliation) │  MERGE, REVIEW, REJECT, and ABSTAIN.
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│     4. Data Fusion        │  Truth discovery and conflict resolution for multi-source
│     (datamesh.fusion)     │  contradictions; generates cell-level provenance trails.
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│         5. Core           │  Shared configuration, canonical schemas, SQLite/Parquet
│      (datamesh.core)      │  data models, and logging infrastructure.
└───────────────────────────┘
```

---

## Module Descriptions

### 1. `datamesh.ingestion`
* **Purpose:** Handles document and table ingestion from heterogeneous raw formats.
* **Grounded in:** Kasem et al. (*ACM Computing Surveys*, 2024) for table detection and structure recognition.
* **Key Responsibilities:**
  * PDF and image parsing via multimodal vision-language models and OCR engines.
  * Extracting tabular matrices and key-value records while preserving pixel bounding boxes and extraction confidence scores.

### 2. `datamesh.evidence`
* **Purpose:** Extracts and normalizes multi-signal evidence vectors between candidate schema attributes and entity record pairs.
* **Grounded in:** Zeakis et al. (*VLDB J.*, 2025) and Freire et al. (*IEEE Data Eng. Bull.*, 2025).
* **Key Signals:**
  * $E_1$: Semantic embedding cosine distance (Sentence-BERT / RoBERTa).
  * $E_2$: Structural layout overlap and column alignment.
  * $E_3$: Regular expression pattern and value-type consistency.
  * $E_4$: Entity coreference over shared business identifiers.
  * $E_5$: Upstream source reliability and extraction confidence.
  * $E_6$: Temporal freshness and decay weighting.
  * $E_7$: Cross-source consensus voting.

### 3. `datamesh.reconciliation`
* **Purpose:** Evaluates evidence vectors to output risk-sensitive integration determinations.
* **Grounded in:** Hendrickx et al. (*Machine Learning*, Springer, 2024) and Astappiev et al. (*Information Systems*, 2026).
* **Key Responsibilities:**
  * Probabilistic confidence calibration (Platt scaling / Isotonic regression).
  * Enforcing the four-way selective decision policy:
    * `MERGE`: High confidence above operating threshold ($\ge \tau_{merge}$).
    * `REVIEW`: Ambiguous match routed to human-in-the-loop review queue.
    * `REJECT`: High confidence non-match ($\le \tau_{reject}$).
    * `ABSTAIN`: Insufficient evidence or conflicting signals.

### 4. `datamesh.fusion`
* **Purpose:** Resolves conflicting attribute values when records are merged.
* **Grounded in:** Wang et al. (*IEEE Transactions on Big Data*, 2025).
* **Key Responsibilities:**
  * Dynamic provider credibility estimation.
  * Confidence-weighted voting and temporal recency decay.
  * Emitting complete cell-level provenance records and conflict audit trails.

### 5. `datamesh.core`
* **Purpose:** Common data schemas, configuration loaders, serialization utilities, and database abstractions for canonical SQLite and Parquet outputs.
