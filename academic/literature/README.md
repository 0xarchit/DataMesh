# Academic Literature: Significance & Architecture Mapping

This directory contains the six peer-reviewed journal papers (published between 2024 and 2026) that establish the scientific foundation, algorithmic models, and baseline evaluation protocols for **DataMesh AI** (*Evidence-Aware Multimodal Data Integration for Reliable Enterprise Data Consolidation*).

---

## Architecture Alignment Overview

```text
               HETEROGENEOUS ENTERPRISE INPUTS
         (PDF Reports, Scanned Documents, XLSX, CSVs)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Multimodal Document Parsing & Ingestion                  │
│    Paper: Kasem et al. (ACM CSUR, 2024)                     │
│    Role:  Table detection (TD) & table structure            │
│           recognition (TSR); extracts cell bounding-boxes,   │
│           headers, and visual confidences.                  │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Schema Discovery & LLM Integration Decoupling            │
│    Paper: Freire et al. (IEEE Data Eng. Bull., 2025)        │
│    Role:  Analyzes LLM failure modes in data integration    │
│           (hallucinations, token cost, lack of provenance); │
│           justifies a structured evidence aggregation layer.│
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Semantic & Learning-Based Entity Resolution              │
│    Papers: Zeakis et al. (VLDB J., 2025 / PVLDB, 2023)      │
│            Astappiev et al. (Information Systems, 2026)     │
│    Role:   Evaluates pre-trained embeddings & ML matchers;   │
│            proves that cosine similarity on surface text    │
│            fails on subtle numeric differences without      │
│            multi-signal verification and calibrated bounds. │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Risk-Sensitive Decision Layer & Reject Option            │
│    Paper: Hendrickx et al. (Machine Learning, 2024)         │
│    Role:  Mathematical theory of classification with a      │
│           reject option; formalizes the four-way policy:    │
│           MERGE, REVIEW, REJECT, and ABSTAIN.               │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Truth Discovery & Multi-Source Conflict Resolution       │
│    Paper: Wang et al. (IEEE Trans. Big Data, 2025)          │
│    Role:  Weight-updating algorithms for source credibility,│
│           temporal recency, and consensus voting on         │
│           contradictory enterprise fields.                  │
└─────────────────────────────┬───────────────────────────────┘
                              ▼
               CANONICAL STRUCTURED ASSET
        (Cell-Level Provenance & Downstream Tabular ML)
```

---

## Detailed Paper Breakdown

### 1. Document Parsing & Table Extraction
* **File:** [`kasem_2024_table_detection_survey.pdf`](kasem_2024_table_detection_survey.pdf)
* **Citation:** Mahmoud Kasem et al., *"Deep Learning for Table Detection and Structure Recognition: A Survey"*, **ACM Computing Surveys (CSUR)**, Vol. 56, No. 12, pp. 1–41, Oct 2024. DOI: `10.1145/3657281`.
* **What It Already Solves:** Provides an exhaustive survey of vision-language architectures, layout transformers, and deep CNN/GNN models for Table Detection (TD) and Table Structure Recognition (TSR) across complex, unstructured document images and PDF files.
* **Limitation Relevant to DataMesh AI:** Current table extractors operate exclusively in isolation on single documents. They discard cell-level bounding-box confidence scores upon emitting plain text and offer zero mechanism to reconcile extracted tables across multiple disparate files.
* **What DataMesh AI Changes:** DataMesh AI preserves extraction bounding boxes, cell coordinates, and visual OCR/VLM extraction confidences as upstream evidence features ($E_2$ structural evidence and $E_5$ provenance signals) for cross-document reconciliation.
* **Viva Defense Response:** *"We cite Kasem et al. to ground our multimodal ingestion layer in state-of-the-art layout and table extraction methodologies, while addressing the literature gap where single-document extractors fail to propagate uncertainty to multi-file consolidation."*

---

### 2. Schema Discovery & LLM Integration
* **File:** [`freire_2025_llm_data_integration.pdf`](freire_2025_llm_data_integration.pdf)
* **Citation:** Juliana Freire et al., *"Large Language Models for Data Discovery and Integration: Challenges and Opportunities"*, **IEEE Data Engineering Bulletin**, Vol. 48, No. 1, pp. 3–31, March 2025.
* **What It Already Solves:** Systematically investigates the capabilities and failure modes of Large Language Models (LLMs) when applied to enterprise table discovery, column-type annotation, schema matching, and entity resolution.
* **Limitation Relevant to DataMesh AI:** Confirms that unguided LLMs suffer from high hallucination rates on complex numerical tables, quadratic token consumption ($O(N^2)$ candidate pairs), prompt brittleness, and a total absence of verifiable data lineage.
* **What DataMesh AI Changes:** Rather than prompting an LLM to blindly merge tables end-to-end, DataMesh AI decouples extraction from reconciliation. It uses lightweight embedding models for candidate blocking and feeds multi-signal features into a calibrated decision engine with cell-level provenance trails.
* **Viva Defense Response:** *"Freire et al. is our primary justification against naive LLM data integration. It proves why an evidence-aware decision layer with auditable provenance is scientifically superior to asking a generative model to merge enterprise databases."*

---

### 3. Pre-Trained Language Embeddings for Entity Resolution
* **File:** [`zeakis_2025_entity_resolution_embeddings.pdf`](zeakis_2025_entity_resolution_embeddings.pdf)
* **Citation:** Alexandros Zeakis, George Papadakis, Dimitrios Skoutas, and Manolis Koubarakis, *"An in-depth analysis of pre-trained embeddings for entity resolution"*, **The VLDB Journal**, Vol. 34, No. 1, Art. 5, 2025 (extended from **Proc. VLDB Endow.**, Vol. 16, No. 9, pp. 2225–2238). DOI: `10.1007/s00778-024-00879-4`.
* **What It Already Solves:** Conducts a large-scale experimental benchmark of 12 pre-trained language model representations (BERT, RoBERTa, Sentence-BERT, FastText) across 17 entity resolution benchmarks for blocking and matching.
* **Limitation Relevant to DataMesh AI:** Demonstrates that while semantic embeddings capture typographical variations well, relying solely on surface textual semantics induces severe false positives when attributes exhibit subtle discrepancies (e.g., conflicting model codes or numerical IDs).
* **What DataMesh AI Changes:** DataMesh AI combines semantic embeddings ($E_1$) with six additional orthogonal signals: table layout structure ($E_2$), value patterns ($E_3$), entity coreference ($E_4$), source provenance ($E_5$), temporal freshness ($E_6$), and cross-source consensus ($E_7$).
* **Viva Defense Response:** *"Zeakis et al. shows that semantic embeddings alone cannot reliably resolve entities when attribute values conflict. We use their experimental insights to build our composite multi-signal evidence representation."*

---

### 4. Benchmarking Learning-Based Entity Matching
* **File:** [`astappiev_2026_smbench.pdf`](astappiev_2026_smbench.pdf)
* **Citation:** Oleh Astappiev, Franziska Neuhof, Marco Fisichella, and George Papadakis, *"SMBench: No-code benchmarking of learning-based entity matching"*, **Information Systems**, Vol. 139, Art. 102711, pp. 1–17, 2026. DOI: `10.1016/j.is.2026.102711`.
* **What It Already Solves:** Provides a standardized benchmark evaluating supervised and deep-learning entity matchers (e.g., Magellan, Ditto, DeepMatcher) across structured and semi-structured tabular domains.
* **Limitation Relevant to DataMesh AI:** Shows that learning-based entity matchers experience severe performance degradation under domain shift and noisy inputs, primarily because they enforce forced binary decisions (match vs. non-match) without uncertainty quantification.
* **What DataMesh AI Changes:** Incorporates SMBench's evaluation benchmarks while replacing forced binary classification with calibrated confidence boundaries and a selective review queue, preventing catastrophic false merges.
* **Viva Defense Response:** *"Astappiev et al. (2026) gives us our baseline evaluation protocol. It empirically proves that existing matchers degrade under noisy extractions, demonstrating why DataMesh AI's calibrated thresholding is necessary."*

---

### 5. Truth Discovery & Conflict Resolution
* **File:** [`wang_2024_truth_discovery_survey.pdf`](wang_2024_truth_discovery_survey.pdf)
* **Citation:** Shuang Wang et al., *"A Survey on Truth Discovery: Concepts, Methods, Applications, and Opportunities"*, **IEEE Transactions on Big Data**, Vol. 11, No. 2, pp. 314–332, March/April 2025 (online 2024). DOI: `10.1109/TBDATA.2024.3423677`.
* **What It Already Solves:** Comprehensive categorization of truth discovery algorithms, iterative source credibility estimation, probabilistic graphical models, and temporal consensus methods for resolving conflicting information from multiple providers.
* **Limitation Relevant to DataMesh AI:** Almost all existing truth discovery algorithms assume pre-existing, clean relational database tables. They cannot handle extraction uncertainties or multimodal noise arising from PDF and scanned inputs.
* **What DataMesh AI Changes:** Bridges truth discovery with multimodal extraction: visual OCR/VLM extraction confidence is explicitly factored into source credibility weights, enabling evidence-based conflict resolution across contradictory enterprise fields.
* **Viva Defense Response:** *"Wang et al. provides the mathematical machinery for our data fusion module. We extend their relational algorithms by incorporating multimodal extraction confidence into provider credibility weighting."*

---

### 6. Selective Classification & Reject Option
* **File:** [`hendrickx_2024_reject_option_survey.pdf`](hendrickx_2024_reject_option_survey.pdf)
* **Citation:** Kilian Hendrickx, Lorenzo Perini, Dries Van der Plas, Wannes Meert, and Jesse Davis, *"Machine Learning with a Reject Option: A survey"*, **Machine Learning**, Springer, Vol. 113, No. 5, pp. 3073–3110, May 2024. DOI: `10.1007/s10994-024-06534-x`.
* **What It Already Solves:** Formal mathematical foundation for learning with a reject option (selective classification). Formalizes the distinction between *ambiguity rejection* (near decision boundary) and *novelty rejection* (out-of-distribution), defining risk-coverage curves.
* **Limitation Relevant to DataMesh AI:** Theoretical formulations in the survey focus on standard single-step prediction tasks (e.g., medical image diagnosis) and have not been adapted to multi-stage data integration pipelines.
* **What DataMesh AI Changes:** Operationalizes selective classification within enterprise consolidation: computes calibrated probabilities over composite evidence vectors and maps decisions to **`MERGE`** (high confidence), **`REVIEW`** (ambiguous match, human-in-the-loop), **`REJECT`** (confident non-match), and **`ABSTAIN`** (insufficient evidence).
* **Viva Defense Response:** *"Hendrickx et al. gives DataMesh AI its theoretical rigor. In business data consolidation, a false merge is catastrophic; we use their reject-option framework to mathematically bound selective risk at target coverage."*

---

## Summary Comparison Matrix

| Studied Journal Paper | Target Pipeline Stage | Core Innovation Introduced by DataMesh AI | Evaluated Metrics |
|---|---|---|---|
| **Kasem et al. (2024)**<br>*ACM CSUR* | Ingestion & Document Parsing | Propagates bounding-box coordinates & OCR confidence as upstream evidence | Extraction Precision/Recall, Coordinate Accuracy |
| **Freire et al. (2025)**<br>*IEEE Data Eng. Bull.* | Schema Matching & Discovery | Decouples extraction from reconciliation; avoids prompt hallucination & quadratic cost | Schema F1, Execution Latency, Token Cost |
| **Zeakis et al. (2025)**<br>*VLDB J. / PVLDB* | Semantic Entity Resolution | Augments semantic embeddings with structural & value-pattern signals | Entity Precision, Recall, F1 |
| **Astappiev et al. (2026)**<br>*Information Systems* | Entity Matching Benchmarks | Replaces forced binary matching with calibrated risk boundaries | False-Merge Rate, False-Split Rate |
| **Wang et al. (2025)**<br>*IEEE Trans. Big Data* | Conflict Resolution & Fusion | Integrates multimodal extraction confidence into provider credibility weights | Data Fusion Accuracy, Truth Error Rate |
| **Hendrickx et al. (2024)**<br>*Machine Learning* | Risk-Sensitive Decision Layer | Formalizes 4-way selective policy (`MERGE`, `REVIEW`, `REJECT`, `ABSTAIN`) | Risk-Coverage AUC, ECE, Brier Score |
