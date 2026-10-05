# Experimental Design, Hypotheses & Ablation Matrix

This directory manages the experimental protocols, hypothesis evaluation runs, ablation logs, and performance metrics for **DataMesh AI**.

---

## 1. Research Hypotheses Under Investigation

* **Hypothesis 1 ($H_1$ — Multi-Signal Superiority):**
  A multi-signal reconciliation layer combining semantic, structural, value-level, provenance, temporal, and cross-source consensus signals achieves statistically significant improvements in entity resolution and data fusion accuracy over semantic-only (cosine similarity) and unguided LLM-only baselines.
* **Hypothesis 2 ($H_2$ — Risk-Sensitive Selective Abstention):**
  Enforcing a calibrated four-way decision policy (`MERGE`, `REVIEW`, `REJECT`, `ABSTAIN`) with a formal reject option mathematically bounds the False-Merge Rate (FMR) below a predefined operational threshold ($\epsilon$) while maintaining high automated consolidation coverage ($C$).
* **Hypothesis 3 ($H_3$ — Downstream Tabular Value):**
  Canonical enterprise datasets generated via evidence-aware consolidation demonstrate measurable accuracy gains and reduced variance on downstream tabular machine learning tasks (evaluated via TabFM and tree-based baselines) compared with datasets merged naively or left uncurated.

---

## 2. The 7-Dimensional Evidence Vector ($\mathbf{E}$)

For every candidate mapping or record pair $(r_i, r_j)$, DataMesh AI extracts an orthogonal 7-dimensional feature representation:

$$\mathbf{E}(r_i, r_j) = \big[ E_1, E_2, E_3, E_4, E_5, E_6, E_7 \big]$$

| Feature | Evidence Dimension | Computation / Representation |
|---|---|---|
| **$E_1$** | **Semantic Similarity** | Dense embedding cosine similarity over attribute names and values (Sentence-BERT / RoBERTa). |
| **$E_2$** | **Structural Compatibility** | Normalized table layout overlap, row/column alignment score, and relational hierarchy match. |
| **$E_3$** | **Value-Pattern Regularity** | Regex, data-type consistency, string edit distances (Jaro-Winkler, Levenshtein), and numeric range compatibility. |
| **$E_4$** | **Entity Coreference** | Contextual entity-linking score over shared identifiers (Tax IDs, email domains, phone stems). |
| **$E_5$** | **Source Provenance** | Historical reliability rating and extraction confidence score of the source file ($c_{\text{ocr}}, c_{\text{vlm}}$). |
| **$E_6$** | **Temporal Consistency** | Timestamp decay factor: $\exp(-\lambda \cdot |\Delta t|)$ modeling freshness and reporting date alignment. |
| **$E_7$** | **Cross-Source Agreement** | Fraction of independent enterprise sources corroborating the candidate mapping / fused value. |

---

## 3. Systematic Ablation Study Configurations

To isolate the marginal value of each evidence signal, experiments run across 6 standardized configurations:

```text
Config 0 (Baseline A): Lexical String Matching (Jaro-Winkler / Levenshtein only)
Config 1 (Baseline B): Pure Semantic Similarity (E1 only via Sentence-BERT)
Config 2 (Baseline C): Unguided Generative LLM (Prompt-based zero-shot matching)
Config 3:              E1 + E2 (Semantics + Layout Structure)
Config 4:              E1 + E2 + E3 (Semantics + Structure + Value Patterns)
Config 5:              E1 + E2 + E3 + E4 (Semantics + Structure + Patterns + Identifiers)
Config 6 (Full Engine): E1 through E7 (Complete Multi-Signal Evidence Vector)
```

---

## 4. Evaluated Quantitative Metrics

1. **Integration Correctness:**
   * Precision, Recall, and Macro-F1 for Schema and Entity Matching.
   * **False-Merge Rate (FMR):** $\frac{\text{False Positive Merges}}{\text{Total Predicated Merges}}$ (Critical enterprise penalty metric).
   * **False-Split Rate (FSR):** $\frac{\text{False Negative Splits}}{\text{Total Ground-Truth Entities}}$.
2. **Selective Classification Quality (Hendrickx et al., 2024):**
   * **Coverage ($C$):** Fraction of candidate pairs classified automatically (`MERGE` or `REJECT`) without routing to human review.
   * **Selective Risk ($R$):** Error rate restricted to the accepted/automated subset.
   * **Risk-Coverage AUC:** Area under the risk-coverage curve across varying confidence thresholds $\tau$.
3. **Probability Calibration:**
   * **Expected Calibration Error (ECE):** Bin-wise discrepancy between predicted confidence and empirical accuracy.
   * **Brier Score:** Mean squared difference between predicted probabilities and ground truth binary outcomes.
4. **Downstream Tabular ML Utility ($H_3$):**
   * Classification ROC-AUC and F1-score on TabFM / XGBoost models trained on the resulting canonical tables.
