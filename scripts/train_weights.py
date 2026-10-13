import json
import os
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datamesh.evidence.signals import WEIGHTS, evidence_vector, name_similarity
from datamesh.ingestion import loaders
from datamesh.pipeline import PRIORITY
from datamesh.reconciliation.resolver import VendorResolver

RAW = os.path.join(ROOT, "data", "raw")
GOLD = os.path.join(ROOT, "data", "gold")
OUT = os.path.join(ROOT, "src", "datamesh", "evidence", "learned_weights.json")
KEYS = list(WEIGHTS)


def models():
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "xgboost": XGBClassifier(n_estimators=300, random_state=0, eval_metric="logloss"),
        "lightgbm": LGBMClassifier(n_estimators=300, random_state=0, verbose=-1),
        "neural_mlp": MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=0),
    }


def build_clusters(records):
    resolver = VendorResolver()
    identities = [r for r in records if r.vendor and not (r.invoice_id and r.amount)]
    for r in sorted(identities, key=lambda x: PRIORITY.get(x.source_type, 9)):
        resolver.assign(r)
    for r in sorted(records, key=lambda x: PRIORITY.get(x.source_type, 9)):
        if r.vendor and (r.invoice_id or r.amount):
            resolver.assign(r)
    return resolver.finalize()


def gold_lookup():
    vendors = json.load(open(os.path.join(GOLD, "canonical_vendors.json"), encoding="utf-8"))
    truth = {
        t["invoice_id"]: t["vendor_id"]
        for t in json.load(open(os.path.join(GOLD, "invoice_truth.json"), encoding="utf-8"))
    }
    by_tax = {v["tax_id"]: v["vendor_id"] for v in vendors}
    return vendors, truth, by_tax


def record_truth(records, truth, by_tax):
    out = []
    for r in records:
        vid = None
        if r.invoice_id and r.invoice_id in truth:
            vid = truth[r.invoice_id]
        elif r.tax_id and r.tax_id in by_tax:
            vid = by_tax[r.tax_id]
        out.append(vid)
    return out


def cluster_truth(clusters, vendors):
    by_tax = {v["tax_id"]: v["vendor_id"] for v in vendors}
    out = {}
    for c in clusters:
        if c.get("tax_id") and c["tax_id"] in by_tax:
            out[c["id"]] = by_tax[c["tax_id"]]
            continue
        best, score = None, 0.0
        for v in vendors:
            s = name_similarity(c["name"], v["name"])
            if s > score:
                best, score = v["vendor_id"], s
        out[c["id"]] = best
    return out


def build_pairs(records, clusters, rtruth, ctruth):
    X, y = [], []
    for r, tv in zip(records, rtruth):
        if not r.vendor or tv is None:
            continue
        for c in clusters:
            cv = ctruth.get(c["id"])
            if cv is None:
                continue
            ev = evidence_vector(r, c)
            X.append([ev[k] for k in KEYS])
            y.append(1 if tv == cv else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def evaluate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def signal_weights(model, X, y):
    model.fit(X, y)
    if hasattr(model, "coef_"):
        raw = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    else:
        raw = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=0).importances_mean
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones_like(raw)
    raw = raw / raw.sum()
    return {k: round(float(v), 4) for k, v in zip(KEYS, raw)}


def main():
    records = loaders.load_all(RAW, use_llm=False)
    clusters = build_clusters(records)
    vendors, truth, by_tax = gold_lookup()
    rtruth = record_truth(records, truth, by_tax)
    ctruth = cluster_truth(clusters, vendors)
    X, y = build_pairs(records, clusters, rtruth, ctruth)

    comparison = {}
    for name, model in models().items():
        comparison[name] = evaluate(model, X, y)
        print("{:<20} auc {}  acc {}  f1 {}".format(name, comparison[name]["auc"], comparison[name]["accuracy"], comparison[name]["f1"]))

    best = max(comparison, key=lambda n: (comparison[n]["auc"], comparison[n]["f1"]))
    best_model = models()[best]
    result = {
        "source": "model comparison on ground-truth-labeled candidate pairs, 5-fold stratified CV",
        "records": len(records),
        "clusters": len(clusters),
        "pairs": int(len(y)),
        "positives": int(y.sum()),
        "comparison": comparison,
        "selected_model": best,
        "selected_metrics": comparison[best],
        "handset_baseline_auc": round(
            float(roc_auc_score(y, X @ np.array([WEIGHTS[k] for k in KEYS]))), 4
        ),
        "weights": signal_weights(best_model, X, y),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print()
    print("selected:", best, comparison[best])
    print("handset baseline auc:", result["handset_baseline_auc"])
    print("learned weights:", result["weights"])


if __name__ == "__main__":
    main()
