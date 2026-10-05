Below is the version I would actually use for the synopsis. I have deliberately separated the **research contribution** from the use of Qwen/TabFM and from the final product.

# 1. Project Title

### **Evidence-Aware Multimodal Data Integration for Reliable Enterprise Data Consolidation**

**Proposed product:** **DataMesh AI** — a software system that converts heterogeneous business data such as PDFs, Excel files, CSVs, images and documents into a unified, structured, provenance-backed dataset, while detecting ambiguous mappings and conflicting information.

---

# 2. Introduction & Problem Statement

Organizations rarely store information in one clean format. The same business entity or attribute may appear differently across Graphs, images, Excel sheets, CSV exports, PDF reports, invoices, scanned documents and legacy systems.

For example:

```text
customers.xlsx
Customer ID = 1023
Name = ACME Ltd.

annual_report.pdf
Client Number = A-1023
Organization = Acme Limited

legacy.csv
account_ref = 1023
company_name = ACME
```

A modern multimodal model can extract these fields, but **extraction is not the complete problem**. The system must decide whether different fields represent the same attribute/entity, normalize their values, resolve conflicting records, and decide when there is insufficient evidence to merge them.

This is precisely the broader data-integration problem addressed by the newly released **MaDI-Bench**, which evaluates schema matching, value normalization, entity matching and data fusion end-to-end. ([GitHub][1])

Existing research such as **Matchmaker** demonstrates that schema matching itself can be approached with language-model-based methods, while recent document benchmarks such as **ExStrucTiny** and **VAREX** show that flexible structured extraction from heterogeneous documents remains challenging. ([Proceedings of Machine Learning Research][2])

### Problem Statement

> **How can heterogeneous information extracted from multiple enterprise sources be reliably reconciled into a single canonical structured representation while minimizing incorrect entity merges and unsupported data-fusion decisions?**

The project will specifically investigate whether **multiple independent evidence sources**—document semantics, table structure, value patterns, provenance, temporal information and cross-source agreement—can improve integration decisions compared with semantic similarity or LLM-only approaches.

---

# 3. Research Innovation & Technical Objectives

## Research Innovation

The central research contribution is **not Qwen, TabFM, PDF extraction or visualization**.

The research contribution is an:

### **Evidence-Aware Reconciliation and Decision Layer**

For every possible schema/entity/value mapping, the system aggregates several evidence types:

```text
Semantic evidence
        +
Structural evidence
        +
Value-pattern evidence
        +
Entity evidence
        +
Source provenance
        +
Temporal consistency
        +
Cross-source agreement
        ↓
Integration decision
```

The decision is not limited to:

```text
MATCH / NOT MATCH
```

It can be:

```text
MERGE
REVIEW
ABSTAIN
CONFLICT
```

This creates a **risk-sensitive integration problem**: incorrectly merging two different entities may be much more damaging than leaving an uncertain record unresolved.

### Technical Objectives

1. Develop a multimodal extraction pipeline capable of processing heterogeneous enterprise sources.

2. Develop an evidence aggregation mechanism for schema and entity reconciliation.

3. Investigate which evidence types contribute most to integration accuracy.

4. Develop confidence estimation for integration decisions.

5. Introduce selective abstention for ambiguous mappings rather than forcing an automatic merge.

6. Develop conflict-resolution strategies using source provenance, temporal information and cross-source agreement.

7. Evaluate whether improved data integration produces measurable improvements in downstream tabular ML tasks.

8. Provide source-level provenance for every integrated value.

---

# 4. Proposed Systematic Architecture & Methodology

```text
           HETEROGENEOUS INPUT
                   │
      ┌────────────┼─────────────┐
      │            │             │
     PDF          XLSX          CSV
      │            │             │
   Images        Tables       Structured
      │            │             │
      └────────────┼─────────────┘
                   ↓
        Multimodal Extraction
             Qwen3.8-27B
                   ↓
       Extracted facts + tables
                   ↓
      ┌─────────────────────────┐
      │  Evidence Construction  │
      │                         │
      │  • semantics            │
      │  • structure            │
      │  • value patterns       │
      │  • provenance           │
      │  • temporal evidence    │
      │  • cross-source support │
      └────────────┬────────────┘
                   ↓
        Evidence Aggregation
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     MERGE       REVIEW      ABSTAIN
       │           │
       └──────┬────┘
              ↓
       Conflict Resolution
              ↓
       Canonical Structured
            Dataset
              ↓
     ┌────────┴──────────┐
     ↓                   ↓
 Analytics          TabFM prediction
     ↓                   ↓
 Visualizations     Classification/
                    Regression
```

### Role of Qwen3.8-27B

Qwen is the **multimodal extraction/reasoning component**, not the research novelty.

It can process text/image inputs and provide candidate structures, entities and extracted values.

### Role of TabFM

TabFM is a **downstream tabular intelligence component**, not the integration algorithm. Google describes TabFM as a zero-shot foundation model for classification and regression on previously unseen tables. It was evaluated on TabArena across 38 classification and 13 regression datasets. ([Google Research][3])

Thus:

```text
Qwen → understand heterogeneous sources

YOUR RESEARCH → reconcile and integrate them

TabFM → test usefulness of resulting structured data
```

This separation is important for the paper.

---

# 5. Implementation Plan

### Phase 1 — Dataset & Baselines

Implement/reproduce:

* rule/string-based schema matching
* embedding-based matching
* LLM-based matching
* existing/best-of-breed integration baseline

Use **MaDI-Bench**, which provides 20 integration tasks across five domains and gold annotations for schema matching, entity matching and data fusion. ([GitHub][1])

### Phase 2 — Multimodal Extraction

Build adapters for:

```text
PDF
DOCX
XLSX
CSV
images/scanned documents
```

Use Qwen3.8-27B for document/table understanding.

For document extraction evaluation, VAREX and ExStrucTiny provide useful recent benchmark references for variable-schema structured extraction. ([ACL Anthology][4])

### Phase 3 — Evidence Engine

Implement individual evidence modules:

```text
E1 = semantic similarity
E2 = schema/structural compatibility
E3 = value-format compatibility
E4 = entity similarity
E5 = source provenance
E6 = temporal compatibility
E7 = cross-source agreement
```

Train/test a lightweight fusion model:

[
P(correct\ integration | E_1,\ldots,E_7)
]

### Phase 4 — Selective Integration

Instead of forcing decisions:

```text
confidence > threshold₁  → MERGE
threshold₂ < confidence ≤ threshold₁ → REVIEW
confidence ≤ threshold₂ → ABSTAIN
```

Optimize the system for **high integration precision while controlling false merges**.

### Phase 5 — Conflict Resolution

Example:

```text
Source A = ₹12.5 lakh
Source B = ₹12.5 lakh
Source C = ₹11.8 lakh
```

The system determines whether the difference is likely due to:

* reporting period
* unit/currency
* stale source
* duplicate entity
* genuinely conflicting information

### Phase 6 — Downstream Validation

Feed:

```text
baseline integrated dataset
vs
our integrated dataset
```

into TabFM and compare downstream prediction performance.

### Phase 7 — Product

Develop:

**DataMesh AI**

Features:

* multi-format upload
* automatic schema discovery
* entity reconciliation
* unified database/table generation
* conflict detection
* confidence scores
* provenance tracing
* manual review queue
* downstream analytics/prediction
* export to CSV/JSON/SQL/Parquet

---

# 6. Research Papers Which Will Be Studied

The primary literature will be centered around **data integration**, not around Qwen or TabFM.

### 1. MaDI-Bench — Primary Benchmark

**Steiner, Peeters & Bizer, 2026 — “MaDI-Bench: An End-to-End Data Integration Benchmark.”**

This is the main benchmark/reference because it evaluates the full chain:

> Schema Matching → Value Normalization → Entity Matching → Data Fusion

and provides ground truth for each stage and the final integrated result. ([GitHub][1])

### 2. Matchmaker — Schema Matching Baseline

**Seedat & Van Der Schaar, ICML 2025 — “Bootstrapping Self-Improvement of Language Model Programs for Zero-Shot Schema Matching.”**

Matchmaker uses candidate generation, refinement and confidence scoring for schema matching and demonstrates strong performance on real-world schema-matching benchmarks. ([Proceedings of Machine Learning Research][2])

This gives you a serious baseline for the **schema-matching portion**.

### 3. ExStrucTiny — Document Extraction

**Sibue et al., EACL 2026 — “ExStrucTiny: A Benchmark for Schema-Variable Structured Information Extraction from Document Images.”**

It focuses on structured extraction from diverse enterprise-style documents with flexible schemas. ([ACL Anthology][4])

### 4. VAREX — Multimodal Extraction

**Barzelay et al., CVPR Workshops 2026 — “VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents.”**

VAREX contains 1,700+ documents with highly varied schemas and evaluates multiple input modalities. It specifically studies models at small parameter scales and finds structured-output compliance to be a major bottleneck. ([CVPR Open Access][5])

### 5. TabFM — Downstream Validation

**Google Research, 2026 — “Introducing TabFM: A zero-shot foundation model for tabular data.”**

TabFM provides the downstream classification/regression component of the proposed system. ([Google Research][3])

---

# 7. Comparison of Existing Researches and What We Will Do

| Area                      | Existing research                                             | Limitation / gap                                                         | Proposed work                                                              |
| ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **Document extraction**   | VLMs extract structured information from individual documents | Extraction does not guarantee correct cross-source integration           | Use multimodal extraction as the first stage                               |
| **Schema matching**       | Matchmaker performs zero-shot semantic schema matching        | Mainly focuses on identifying attribute correspondence                   | Add structural/value/provenance/cross-source evidence                      |
| **Entity matching**       | Traditional and ML entity matching systems compare records    | Difficult when identifiers/names differ and evidence conflicts           | Evidence-aware entity reconciliation                                       |
| **Data fusion**           | MaDI-Bench demonstrates fusion remains difficult              | Incorrect merges and conflicting values remain major failure modes       | Risk-sensitive fusion with confidence and abstention                       |
| **LLM-based integration** | MaDI-Bench provides an LLM-based reference pipeline           | Strong at some stages but weaker on difficult entity/fusion cases        | Investigate whether explicit evidence aggregation improves difficult cases |
| **Conflict handling**     | Existing systems generally make a final prediction            | They may not adequately expose uncertainty                               | MERGE / REVIEW / ABSTAIN decision policy                                   |
| **Explainability**        | Extraction systems often provide extracted values/locations   | Doesn't necessarily explain *why two sources were considered equivalent* | Evidence trail for every integration decision                              |
| **Downstream analytics**  | TabFM performs prediction directly on tables                  | Assumes the input table is already trustworthy/structured                | Test whether better integration improves downstream prediction             |

MaDI-Bench is particularly important here because its reference evaluation already shows that **schema matching can be much easier than later entity matching and data fusion**, giving the proposed research a concrete target rather than an invented weakness. ([GitHub][1])

---

# Competitor / Existing Product Landscape

This section should be in the **product/business context**, not presented as research baselines.

### Julius AI

Julius already accepts CSV, Excel, JSON, PDF, images and other files, allows natural-language analysis, generates visualizations and can connect to databases/warehouses. ([Julius AI | Excel, Slides, Tasks with AI][6])

**Difference:**

Julius primarily positions itself as an **AI data-analysis assistant**:

```text
Upload data
    ↓
Ask question
    ↓
Analyze / visualize
```

Your system:

```text
Multiple heterogeneous sources
    ↓
reconcile entities/schemas
    ↓
resolve conflicts
    ↓
canonical structured dataset
    ↓
provenance + confidence
    ↓
analytics / prediction
```

So **Julius is a competitor at the product level**, but not the primary scientific baseline.

Other practical competitors/categories include:

* enterprise ETL/data integration platforms
* document extraction platforms
* AI spreadsheet/data-analysis tools
* data-cleaning/entity-resolution systems

The product therefore should **not** be positioned as “Julius but open-source.”

It should be positioned as:

> **A trustworthy heterogeneous-data consolidation and reconciliation layer that turns scattered business sources into a canonical, auditable data asset.**

---

# Core Experimental Variables

These are the variables I would explicitly put into the synopsis because they make the research concrete.

### Independent variable

**Evidence configuration**

```text
Semantic only
Semantic + structural
Semantic + value
Semantic + provenance
Semantic + cross-source
All evidence
```

### Dependent variables

**Schema Matching F1**

**Entity Matching F1**

**Value Normalization Accuracy**

**Data Fusion Accuracy**

**False Merge Rate**

**False Split Rate**

**Abstention Precision / Coverage**

**Confidence Calibration**

**End-to-End Integration Score**

### Downstream variable

**TabFM performance on the resulting integrated dataset**

For example:

```text
Accuracy / F1 / AUROC
Regression RMSE / MAE
```

### System variables

```text
Latency
Memory
Model size
LLM/API calls
Cost per integrated document/table
```

---

# The actual research hypothesis

I would put this in the synopsis almost verbatim:

> **H₁:** A data-integration system that combines semantic, structural, value-level, provenance and cross-source evidence will achieve higher entity-matching and data-fusion reliability, particularly under conflicting and heterogeneous source conditions, than semantic-similarity and LLM-only baselines.

And a second hypothesis:

> **H₂:** Introducing confidence-aware selective abstention will significantly reduce false entity merges while maintaining acceptable integration coverage.

And a third, very useful one:

> **H₃:** Improvements in data-integration reliability will lead to measurable improvements in downstream tabular prediction compared with LLM-only data consolidation.

That gives you **three actual research questions**, rather than a pile of technologies.

---

## The one-sentence project definition

> **We are not researching how to convert PDFs to JSON; we are researching how an AI system can determine whether information extracted from different heterogeneous sources can be safely unified, how conflicts should be resolved, and when the system should refuse to merge uncertain information.**

That is the version I would submit.

And I would **not put “Qwen3.8-27B + TabFM” in the project title**. Those are implementation components. Putting them in the title makes the project look like a model-combination exercise rather than a research contribution.

[1]: https://github.com/wbsg-uni-mannheim/MaDI-Bench?utm_source=chatgpt.com "GitHub - wbsg-uni-mannheim/MaDI-Bench: The Mannheim Data Integration Benchmark (MaDI-Bench) is an end-to-end benchmark for tabular data integration. It provides integration tasks across five domains, covering schema matching, value normalization, entity matching, and data fusion. It supports difficulty variants and measuring step-wise as well as end-to-end performance. · GitHub"
[2]: https://proceedings.mlr.press/v267/seedat25a.html?utm_source=chatgpt.com "Bootstrapping Self-Improvement of Language Model Programs for Zero-Shot Schema Matching"
[3]: https://www.research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/?utm_source=chatgpt.com "Introducing TabFM: A zero-shot foundation model for tabular data"
[4]: https://aclanthology.org/2026.eacl-long.265/?utm_source=chatgpt.com "ExStrucTiny: A Benchmark for Schema-Variable Structured Information Extraction from Document Images - ACL Anthology"
[5]: https://openaccess.thecvf.com/content/CVPR2026W/MMFM5/html/Barzelay_VAREX_A_Benchmark_for_Multi-Modal_Structured_Extraction_from_Documents_CVPRW_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[6]: https://julius.ai/docs/get-started/quickstart?utm_source=chatgpt.com "Getting Started - Julius"
