# Literature for DataMesh AI — Final Reference Set

**Project:** *Evidence-Aware Multimodal Data Integration for Reliable Enterprise Data Consolidation* (DataMesh AI)
**Authors:** Archit Jain (2415800016), Deepti Yadav (2415800029) — B.Tech (Honours) CSE, GLA University, Mathura
**Rule applied:** recent (2023–2026), peer-reviewed **journal**, published — no conference, workshop, bulletin, or preprint.
**Date:** September 2026

Every paper below was venue-verified against Crossref (real journal name, not a conference/bulletin) **and read in full** to confirm it genuinely supports its role.

---

## The 7 papers to cite in `synopsis.tex`

| # | Citation | Journal | Year | DOI | Backs | File |
|---|---|---|---|---|---|---|
| 1 | Zhang et al., **SMUTF: Schema Matching Using Generative Tags and Hybrid Features** | Information Systems 133:102570 | 2025 | `10.1016/j.is.2025.102570` | Schema matching + the multi-signal evidence-vector idea | `smutf_schema_matching_infosystems_2025.pdf` |
| 2 | Christen et al., **Graph Metrics-driven Record Cluster Repair meets LLM-based Active Learning** | ACM J. Data & Information Quality 17(2):7 | 2025 | `10.1145/3735511` | False-merge failure mode + active learning | `christen_cluster_repair_jdiq_2025.pdf` |
| 3 | Robach, Hof & van de Wiel, **False Discovery Estimation in Record Linkage** | Statistics in Medicine 44:e70292 | 2025 | `10.1002/sim.70292` | False-merge-rate estimation (H2) | `robach_false_discovery_record_linkage_statmed_2025.pdf` |
| 4 | Silva Filho et al., **Classifier Calibration: A Survey** | Machine Learning 112 | 2023 | `10.1007/s10994-023-06336-7` | Calibration methods (ECE/Brier) | `silvafilho_calibration_survey_ml_2023.pdf` |
| 5 | Ji et al., **Table Integration in Data Lakes Unleashed** | The VLDB Journal 34:36 | 2025 | `10.1007/s00778-025-00917-9` | End-to-end integration + conflict resolution | `ji_table_integration_vldbj_2025.pdf` |
| 6 | Gregori et al., **PROLIT: An LLM-guided Platform for Multi-granular Data Provenance** | Journal of Big Data 12:187 | 2025 | `10.1186/s40537-025-01209-3` | Cell/column-level provenance | `gregori_prolit_provenance_jbd_2025.pdf` |
| 7 | Hollmann et al., **Accurate Predictions on Small Data with a Tabular Foundation Model** (TabPFN v2) | Nature 637:319 | 2025 | `10.1038/s41586-024-08328-6` | Downstream tabular ML (H3) | `hollmann_tabpfn_v2_nature_2025.pdf` |

These cover schema matching, entity resolution, data fusion, calibration, end-to-end integration, provenance, human-in-the-loop, and downstream ML — and back all three hypotheses (H1, H2, H3).

---

## How to cite each one honestly

Each note is the defensible framing, plus the one caveat an examiner could catch if you overstate it.

**1. SMUTF** — Fuses lexical, data-type, value-distribution, PLM-embedding, and LLM-generated-tag signals into one XGBoost classifier. It is the closest published ancestor of your 7-signal evidence vector.
- *Caveat:* it does **schema (column) matching only**, not entity resolution. Say: "SMUTF fused multiple signals for schema matching; we extend the learned multi-signal combiner to row-level merge decisions."

**2. Christen et al.** — Directly about false merges caused by transitive closure (A~B, B~C ⇒ wrong cluster), which it repairs **after the fact** with graph metrics.
- *Caveat:* its active-learning oracle is an **LLM, not a human** — cite it for "active learning over uncertain links," not "human review." Your delta: you *prevent* the merge up front via calibrated abstention; it *repairs* after.

**3. Robach et al.** — Estimates the false-discovery proportion among accepted links using synthetic "decoy" records; works with any linkage method.
- *Caveat:* it is a **biomedical** paper resting on a "links happen at random" assumption that can break on enterprise data with strong IDs. Cite it as the *method you borrow* for your False-Merge Rate and state you re-validate on your own data. It *estimates*, it does not guarantee a bound.

**4. Silva Filho et al.** — Defines ECE, Brier score, reliability diagrams, and temperature/Platt/isotonic/Dirichlet calibration for binary and multiclass; ships the PyCalib library.
- *Caveat:* it covers calibrating probabilities, **not the abstain rule itself**. Pair it with Hendrickx (below) for the MERGE/REVIEW/REJECT/ABSTAIN mechanism.

**5. Ji et al.** — A real 3-stage integration pipeline: pairwise integrability judgment → integrable-set discovery → multi-tuple conflict resolution. Genuinely does data fusion.
- *Verified delta:* it **assumes schema alignment is already done** and **always emits one merged tuple** (no abstention). Say: "Ji et al. integrate already-aligned tables and always merge; we go end-to-end from raw multi-format sources with a calibrated merge-or-abstain at every stage."

**6. Gregori et al. (PROLIT)** — Multi-granular provenance (cell/entity, column, pipeline) with lineage queries; LLM-guided.
- *Delta:* PROLIT only **records** provenance. Using provenance as a decision input (your evidence signal E5 that feeds the merge) is your extension.

**7. Hollmann et al. (TabPFN v2)** — Runnable in-context tabular foundation model (code + API), classification and regression. This replaces the "TabFM" blog post you were going to cite.
- *Caveat:* bounded to **≤10,000 rows / 500 features**. If your canonical table is larger, also report an **XGBoost/CatBoost** baseline (the paper names exactly these).

---

## The one gap and how to fill it

**P1 — multi-format / document extraction — is not among the 7.** Cover it with a paper you already hold: **Kasem et al., "Deep Learning for Table Detection and Structure Recognition: A Survey," ACM Computing Surveys 56(12), 2024** (`10.1145/3657281`). Two conditions:
1. Download the **CSUR journal version** — the file you currently hold is the 2022 arXiv preprint.
2. Cite it honestly for *table detection / structure recognition* only. It says nothing about multimodal/VLM extraction, cross-document integration, provenance, or uncertainty — do not attribute those to it.

Because your extraction stage uses off-the-shelf VLM/OCR (not a novel method), P1 can be a supporting citation rather than a headline research slot — so this is either an 8th reference or a prose citation.

---

## Keep in prose (not counted in the 7)

- **Hendrickx et al., "Machine Learning with a Reject Option: A Survey," Machine Learning 113(5), 2024** — the reject-option/abstention mechanism the calibration survey does not cover. *Fix:* it is strictly binary and contains **no ECE or Brier** — do not attribute those metrics to it.
- **SMBench (Astappiev et al., Information Systems 139, 2026)** — you hold this; use it in the experimental-setup section as your entity-matching evaluation harness. It is a benchmark, not a baseline, so it complements Christen.

---

## If you need fewer than 7

- **6:** drop PROLIT → cite it in prose for provenance.
- **5:** SMUTF + Christen + Ji + calibration survey + TabPFN (covers schema/entity/fusion/calibration/integration/downstream + all hypotheses); add Kasem for P1 → 6.
- **Never drop** SMUTF, Ji, TabPFN, or Christen — each is the only anchor for a pillar or hypothesis. If cutting, drop Robach first (calibration survey already carries P5, and Robach's biomedical transfer is the weakest link).

---

## What to remove from your current synopsis

The original six references had specific problems — fix these before submission:

| Original citation | Problem | Action |
|---|---|---|
| **Freire** (IEEE Data Engineering Bulletin) | Not a peer-reviewed journal — it is an unrefereed bulletin (no Crossref journal record). You call it a "landmark study." | Drop it; SMUTF replaces its schema role. |
| **Zeakis** (cited as VLDB Journal 2025) | You hold the PVLDB 2023 **conference** version, a different document. | Drop or re-label; Christen/SMUTF cover entity/schema. |
| **Kasem** (cited as CSUR 2024) | You hold the 2022 preprint, not the journal version. | Keep for P1 but download the CSUR version and re-scope the claim. |
| **Hendrickx** metrics claim | README/synopsis attribute ECE + Brier to it; the paper has neither. | Move to prose; correct the metrics claim. |
| "6 peer-reviewed journal papers" | Stated in ~4 places; was not true of all six. | With this set it becomes true — keep the phrase only once the bibliography is updated. |

Also: the stale `projectContext.md` describes a **different** project (MaDI-Bench, Qwen3.8-27B, a CONFLICT decision instead of REJECT). Delete it or move it out of the tree so a supervisor sees one coherent project.

---

## Trust note

All 7 DOIs resolve on Crossref as genuine `journal-article` records with real journal names and valid ISSNs, dated 2023–2026 — no PVLDB, bulletin, or conference slipped through. Each PDF was read end-to-end; the caveats above come from the actual text, not from abstracts.
