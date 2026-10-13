# Challenges, Fixes, and Next Steps

Engineering notes for the project write-up. Every number here comes from an actual run of the pipeline in this repo. Where a number depends on whether the local model was running, that is stated.

## 1. The same invoice arrives from many formats and they disagree

Problem: one invoice can appear in the Excel ledger, as a PDF, and as a scanned receipt. The copies are not always identical. A scanner can shift a digit (24800 read as 248000) or drop one. If the pipeline simply keeps the first copy it reads, the winner is decided by load order, which is an accident.

Fix: `datamesh.fusion.consensus` groups every copy by invoice id and decides on purpose. Three rules, in order. If all copies agree, nothing to decide. If a strict majority of copies agree, take the majority. Otherwise rank the candidates by the vendor's own amount profile and by source reliability. If the two candidates cannot be separated by that evidence, because the reliability gap is smaller than the margin, the row is flagged for review instead of guessed.

Result: 20 conflicts were planted with a x10 digit error. All 20 were detected. Keeping the first copy in load order gets 7 wrong. The consensus layer gets 0 wrong and flags 0 for review on this dataset.

Note: the review branch is real but this dataset does not trigger it, because the planted conflicts stay separable. The branch is covered by a unit self-check in `datamesh/fusion/consensus.py`.

## 2. One vendor appears under many spellings

Problem: the same company is written as "Acme Ltd", "ACME", "Acme Limited", and "Acme" across sources. Treating each string as a separate vendor splits one real vendor into many rows and breaks every aggregate.

Fix: `datamesh.reconciliation.resolver` builds clusters by scoring a seven-signal evidence vector between a record and each existing cluster, then comparing the weighted score to a merge threshold. The signals are name similarity, tax id match, city agreement, coreference, provenance confidence, temporal freshness, and cross-source consensus. City aliases (Bombay to Mumbai, Bangalore to Bengaluru) are normalized before comparison.

Result: 20 clean vendor clusters from the mixed-spelling input.

## 3. Entity fragmentation hurts the downstream model

Problem: if the same vendor is split across spellings, a downstream model sees one real vendor as several small noisy rows.

Fix and evidence: build the same features two ways and compare. The raw table groups by the vendor string as written (38 rows). The consolidated table groups by the resolved vendor cluster (20 rows). Same model, same features, same leave-one-out cross validation.

- Raw table: balanced accuracy 0.77, macro F1 0.76.
- Consolidated table: balanced accuracy 0.90, macro F1 0.90.

Balanced accuracy is reported because the two arms do not have the same row count (that difference is the effect being measured), and it is invariant to class proportions.

## 4. Retrieval truncation confounds the comparison

Problem: in a top-k retrieval setup, the answering model only sees k chunks out of the whole corpus. With k=15 out of roughly 320 chunks it cannot see enough rows to total anything, so it would fail for a reason that has nothing to do with consolidation.

Fix: run both retrieval arms again with the entire corpus in context. Now the corpus is the only difference between the two arms.

- Full corpus, raw records: about 20 percent.
- Full corpus, consolidated table: about 60 percent.

The consolidated arm recovers the entity questions but still fails the totals, because it is asking a language model to sum roughly 200 rows in context. The remaining gap is not retrieval and not consolidation. It is that retrieval makes the model do the arithmetic, while the pipeline plans a query and executes it against the table.

These figures are from a run with the local model online. With the model offline both arms collapse to keyword rules and the comparison is not meaningful, which the code itself warns about.

## 5. Model choice for the evidence combiner

Six models were compared on the ground-truth-labeled candidate pairs under 5-fold stratified cross validation: logistic regression, random forest, gradient boosting, XGBoost, LightGBM, and a small neural network.

On the synthetic data the numbers were: logistic regression and the neural network led on macro F1 at 0.952, random forest and gradient boosting followed at 0.951, then XGBoost at 0.934 and LightGBM at 0.928. AUC was 0.999 or above for every model.

The spread is small because the task is close to separable. That is the honest result, and it decides the design: when every model ties, the simplest one wins. Logistic regression is smaller, faster, has no compiled dependency, trains in seconds, and its coefficients read directly as signal weights. So the combiner is logistic regression.

What the downstream experiment compares is not one model against another. It is raw input against consolidated input, the same model both times, so the data is the only variable.

## 6. Next: TabFM and a custom model

TabFM (tabular foundation model). The code already tries it first. `datamesh.downstream.build_model` attempts `from tabfm import TabFMClassifier` and falls back to GradientBoosting when the package is not installed. To turn it on:

```
uv pip install --python .venv "tabfm[pytorch]"
```

It is zero-shot and in-context, so it needs no training, which fits tables this small better than a tree ensemble.

One correction to keep the write-up accurate: TabFM does tabular classification and regression, it does not forecast a time series natively. Predicting a target column from feature columns is regression, and that is what it does. Real forecasting, meaning the next value from a time-ordered sequence, needs a time-ordered target and a forecaster, which is a different setup. In the write-up, say prediction, and say forecasting only if a time-ordered target is actually built.

A custom deep model. The combiner is already a learned model, a logistic regression over the signals, fitted on ground-truth-labeled pairs. The next step is a deeper combiner that can learn interactions between signals, such as a small multilayer network, evaluated on harder data where a linear model separates less cleanly.

## 7. Real public benchmark data

The synthetic data proves the pipeline works end to end, but it is synthetic. To check the matching core on real records, two public entity resolution benchmarks were used: Abt-Buy (9,575 labeled pairs, 1,028 true matches) and Amazon-Google (11,460 pairs, 1,167 matches), the standard datasets from the DeepMatcher and Magellan line of work.

These benchmarks hold product records, not invoices, so only the fields that exist could be used. Four signals were computed: name similarity, coreference, text similarity over the description or manufacturer, and price agreement. Tax id, city, date, and cross-source consensus have no counterpart and were dropped. This is worth stating plainly: it is a reduced version of the seven-signal vector, not the same one.

Results under 5-fold cross validation, positive class is a true match.

Abt-Buy

- name similarity alone: F1 0.422, AUC 0.755
- description similarity alone: F1 0.194, AUC 0.497
- learned combiner, logistic regression: F1 0.393, AUC 0.786
- learned combiner, random forest: F1 0.413, AUC 0.783

Amazon-Google

- name similarity alone: F1 0.524, AUC 0.852
- manufacturer similarity alone: F1 0.200, AUC 0.553
- learned combiner, logistic regression: F1 0.481, AUC 0.900
- learned combiner, random forest: F1 0.564, AUC 0.897

The honest reading: the multi-signal combiner gives the best ranking quality on both datasets (AUC 0.786 and 0.900, against 0.755 and 0.852 for the best single signal), and the best F1 on Amazon-Google. On Abt-Buy the name-only baseline edges it on F1 at the fixed 0.5 cutoff even though it ranks worse overall. It is not a clean sweep, and reporting it as one would be wrong.

Two points worth stating in the paper. First, real entity resolution is hard: F1 sits at 0.4 to 0.6, far below the synthetic 1.0, which is exactly why a synthetic-only evaluation flatters any method. Second, the strongest single signal is name similarity, which is the baseline a multi-signal approach has to beat to justify itself, and on ranking quality it does.

## 8. Honest limitations to state in the paper

- The synthetic data is seeded and generated, not hand-written, and the gold answers are computed from it, but it is not real enterprise data. Real benchmark results are reported separately in section 7.
- The evidence weights are learned, by logistic regression on ground-truth-labeled candidate pairs, and stored as a model artifact. That combiner is the only trained component in the consolidation path; clustering, conflict resolution, and deduplication are rule-based.
- The review and abstain branch exists but this dataset does not exercise it.
- Calibration (ECE, Brier) is not implemented yet. The merge threshold is a plain cutoff, not a calibrated probability.
- The local LLM is used for two things only: writing the query plan on our side, and answering as the retrieval reader on the other side. The consolidation itself runs with no model at all.
