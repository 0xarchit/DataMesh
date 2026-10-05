# DataMesh Tiered Storage Architecture

This directory defines the three-tiered data progression pipeline for **DataMesh AI**, ensuring strict lineage, auditability, and immutability from raw ingest to canonical structured assets.

---

## 1. Directory Structure & Lifecycle

```text
data/
├── raw/            # Tier 1: Immutable original source artifacts
├── intermediate/   # Tier 2: Evidence-bearing structured extractions (JSON/Parquet)
└── canonical/      # Tier 3: Consolidated, provenance-backed canonical assets (SQLite/Parquet)
```

```text
  [ raw/ ]                   [ intermediate/ ]                [ canonical/ ]
┌──────────────┐           ┌───────────────────┐           ┌───────────────────┐
│ PDF Invoices │           │ Bounding Boxes    │           │ Unified Entities  │
│ Scanned Imgs │ ────────► │ OCR Confidences   │ ────────► │ Resolved Conflicts│
│ Spreadsheets │  Extract  │ Extracted Records │ Reconcile │ Cell Provenance   │
│ CSV Exports  │           │ Metadata & Hashes │           │ Review Queue Logs │
└──────────────┘           └───────────────────┘           └───────────────────┘
```

---

## 2. Tier Specifications

### Tier 1: `data/raw/` (Immutable Source Ingestion)
* **Description:** Preserves the untouched original enterprise files exactly as uploaded.
* **Integrity Guarantee:** Never modified by any pipeline stage. Every file receives an immutable SHA-256 content hash upon ingestion.
* **Supported Formats:** `.pdf`, `.png`, `.jpg`, `.tiff`, `.xlsx`, `.xls`, `.csv`, `.docx`.

### Tier 2: `data/intermediate/` (Evidence-Bearing Extractions)
* **Description:** Structured representations produced by the multimodal extraction adapters (Qwen / OCR / Tabular Parsers).
* **Key Fields Preserved:**
  * `source_file_id`: Foreign key reference to the raw file hash.
  * `bounding_box`: Normalised pixel coordinates `[ymin, xmin, ymax, xmax]` on the source document page.
  * `extraction_confidence`: Numerical model confidence score $c \in [0.0, 1.0]$.
  * `raw_text`: Extracted verbatim token string prior to normalization.
  * `inferred_type`: Preliminary column / attribute data type.

### Tier 3: `data/canonical/` (Auditable Consolidated Assets)
* **Description:** Final unified tables produced by the evidence reconciliation and conflict resolution engine.
* **Key Components:**
  * **Canonical Tables:** Reconciled schema tables formatted for SQL analytics and downstream ML pipelines (Parquet / SQLite).
  * **Provenance Ledger:** Detailed cell-by-cell mapping indicating which source document, page, and bounding box contributed to each specific attribute value.
  * **Conflict Resolution Log:** Audit trail recording cases where multiple sources disagreed, documenting the applied resolution rule (e.g., Wang et al. source reliability weighting vs. temporal freshness).
  * **Human Review Queue Export:** Serialized candidate pairs categorized as `REVIEW` or `ABSTAIN` for auditor inspection.
