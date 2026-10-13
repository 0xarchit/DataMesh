import csv
import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from datamesh.evidence.signals import coreference, name_similarity

EXTERNAL = os.path.join(ROOT, "data", "external")
RESULTS = os.path.join(ROOT, "research", "results")

DATASETS = {
    "abt_buy": {"name_a": "A_name", "name_b": "B_name", "text_a": "A_description", "text_b": "B_description"},
    "amazon_google": {"name_a": "A_title", "name_b": "B_title", "text_a": "A_manufacturer", "text_b": "B_manufacturer"},
}

FEATURES = ["name_similarity", "coreference", "text_similarity", "price_agreement"]


def price_value(text):
    if text is None:
        return None
    found = re.findall(r"\d+\.?\d*", str(text).replace(",", ""))
    if not found:
        return None
    try:
        return float(found[0])
    except ValueError:
        return None


def price_agreement(a, b):
    x, y = price_value(a), price_value(b)
    if not x or not y:
        return 0.5
    return round(min(x, y) / max(x, y), 4)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def text_similarity(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 4)


def load(dataset):
    cfg = DATASETS[dataset]
    X, y = [], []
    with open(os.path.join(EXTERNAL, dataset + ".csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            na, nb = row[cfg["name_a"]], row[cfg["name_b"]]
            X.append(
                [
                    name_similarity(na, nb),
                    coreference(na, nb),
                    text_similarity(row[cfg["text_a"]], row[cfg["text_b"]]),
                    price_agreement(row["A_price"], row["B_price"]),
                ]
            )
            y.append(int(row["label"]))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def best_threshold(scores, y):
    best, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(scores, 3)):
        score = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if score > best_f1:
            best, best_f1 = float(t), score
    return best


def metrics(y, pred, scores=None):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out = {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "false_merge_rate": round(fp / (fp + tp), 4) if (fp + tp) else 0.0,
        "predicted_matches": tp + fp,
    }
    if scores is not None:
        out["auc"] = round(float(roc_auc_score(y, scores)), 4)
    return out


def cross_validate(X, y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    linear = np.zeros(len(y))
    forest = np.zeros(len(y))
    name_only = np.zeros(len(y))
    text_only = np.zeros(len(y))
    thr_name, thr_text = 0.5, 0.5
    for train, test in skf.split(X, y):
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X[train], y[train])
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0).fit(X[train], y[train])
        linear[test] = lr.predict_proba(X[test])[:, 1]
        forest[test] = rf.predict_proba(X[test])[:, 1]
        name_only[test] = X[test][:, 0]
        text_only[test] = X[test][:, 2]
        thr_name = best_threshold(X[train][:, 0], y[train])
        thr_text = best_threshold(X[train][:, 2], y[train])
    return {
        "name_similarity_only": metrics(y, (name_only >= thr_name).astype(int), name_only),
        "text_similarity_only": metrics(y, (text_only >= thr_text).astype(int), text_only),
        "learned_linear_multi_signal": metrics(y, (linear >= 0.5).astype(int), linear),
        "learned_forest_multi_signal": metrics(y, (forest >= 0.5).astype(int), forest),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    report = {"source": "public entity resolution benchmarks", "features": FEATURES, "datasets": {}}
    for dataset in DATASETS:
        path = os.path.join(EXTERNAL, dataset + ".csv")
        if not os.path.exists(path):
            continue
        X, y = load(dataset)
        result = cross_validate(X, y)
        result["pairs"] = int(len(y))
        result["positives"] = int(y.sum())
        report["datasets"][dataset] = result
        print("== {} ==  pairs {}  matches {}".format(dataset, len(y), int(y.sum())))
        for name, m in result.items():
            if isinstance(m, dict):
                print(
                    "  {:<30} f1 {}  prec {}  rec {}  false_merge {}  auc {}".format(
                        name, m["f1"], m["precision"], m["recall"], m["false_merge_rate"], m.get("auc")
                    )
                )
    with open(os.path.join(RESULTS, "er_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print()
    print("wrote research/results/er_benchmark.json")


if __name__ == "__main__":
    main()
