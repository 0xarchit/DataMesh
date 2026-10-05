import csv
import json
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from datamesh.evidence.signals import name_similarity


def _month(date):
    try:
        return int(str(date).split("-")[1])
    except Exception:
        return 0


def _labels(true_totals):
    values = sorted(true_totals.values())
    median = values[len(values) // 2]
    return {vid: int(total > median) for vid, total in true_totals.items()}


def true_totals(canonical_dir):
    totals = {}
    with open(os.path.join(canonical_dir, "transactions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            totals[row["vendor_cluster"]] = totals.get(row["vendor_cluster"], 0) + int(row["amount_inr"])
    return totals


def canonical_table(canonical_dir):
    vendors = {v["id"]: v for v in json.load(open(os.path.join(canonical_dir, "vendors.json"), encoding="utf-8"))}
    labels = _labels(true_totals(canonical_dir))
    groups = {}
    with open(os.path.join(canonical_dir, "transactions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["vendor_cluster"]
            g = groups.setdefault(vid, {"vendor": vid, "amounts": [], "cities": set(), "sources": set(), "months": set()})
            g["amounts"].append(int(row["amount_inr"]))
            g["cities"].add(vendors[vid]["city"])
            g["sources"].add(row["source"])
            g["months"].add(_month(row["date"]))
    rows = []
    for vid, g in groups.items():
        rows.append(
            {
                "txn_count": len(g["amounts"]),
                "total_amount": sum(g["amounts"]),
                "mean_amount": round(sum(g["amounts"]) / len(g["amounts"])),
                "n_sources": len(g["sources"]),
                "n_cities": len(g["cities"]),
                "n_months": len(g["months"]),
                "high_value": labels.get(vid, 0),
            }
        )
    return pd.DataFrame(rows)


def raw_table(intermediate_dir, canonical_dir):
    records = json.load(open(os.path.join(intermediate_dir, "records.json"), encoding="utf-8"))
    vendors = json.load(open(os.path.join(canonical_dir, "vendors.json"), encoding="utf-8"))
    labels = _labels(true_totals(canonical_dir))
    mapping = {}
    for v in vendors:
        mapping[v["id"]] = v["name"]
    groups = {}
    for r in records:
        if not (r["invoice_id"] and r["amount"]):
            continue
        best, best_score = None, 0.0
        for v in vendors:
            score = name_similarity(r["vendor"], v["name"])
            if score > best_score:
                best, best_score = v, score
        rid = best["id"] if best else ""
        key = r["vendor"]
        g = groups.setdefault(key, {"amounts": [], "cities": set(), "sources": set(), "months": set(), "rid": rid})
        g["amounts"].append(int(r["amount"]))
        g["cities"].add(r["city"] or "unknown")
        g["sources"].add(r["source_type"])
        g["months"].add(_month(r["date"]))
    rows = []
    for key, g in groups.items():
        rows.append(
            {
                "txn_count": len(g["amounts"]),
                "total_amount": sum(g["amounts"]),
                "mean_amount": round(sum(g["amounts"]) / len(g["amounts"])),
                "n_sources": len(g["sources"]),
                "n_cities": len(g["cities"]),
                "n_months": len(g["months"]),
                "high_value": labels.get(g["rid"], 0),
            }
        )
    return pd.DataFrame(rows)


def build_model():
    try:
        from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        return TabFMClassifier(model=tabfm_v1_0_0.load(model_type="classification")), "TabFM"
    except Exception:
        return GradientBoostingClassifier(random_state=0), "GradientBoostingClassifier"


def evaluate(df, model=None):
    feature_cols = ["txn_count", "total_amount", "mean_amount", "n_sources", "n_cities", "n_months"]
    X = df[feature_cols]
    y = df["high_value"]
    if model is None:
        model, _ = build_model()
    pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    # The two arms do not have the same row count - fragmentation is the thing
    # being measured - so balanced accuracy and macro F1 are reported alongside
    # raw accuracy. Both are invariant to how the classes are proportioned.
    return {
        "rows": len(df),
        "high_value_share": round(float(y.mean()), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
    }
