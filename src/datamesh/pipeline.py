import csv
import json
import os
from types import SimpleNamespace

from datamesh.evidence.signals import evidence_vector
from datamesh.fusion.consensus import reconcile
from datamesh.ingestion import loaders
from datamesh.reconciliation.resolver import VendorResolver, dedupe_transactions

PRIORITY = {"csv": 0, "xlsx": 1, "pdf": 2, "receipt": 3}


def _load(raw_dir, use_llm, emit, on_token):
    return loaders.load_all(raw_dir, use_llm=use_llm, emit=emit, on_token=on_token)


def run(raw_dir, canonical_dir, intermediate_dir, use_llm=True, emit=None, on_token=None):
    say = emit or (lambda s: None)
    os.makedirs(canonical_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    say("step 1 of 4: ingest raw sources")
    records = _load(raw_dir, use_llm, say, on_token)
    with open(os.path.join(intermediate_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
    say("step 1 done: {} records written to intermediate/records.json".format(len(records)))

    say("step 2 of 4: evidence fusion, clustering identities")
    resolver = VendorResolver()
    identities = [r for r in records if r.vendor and not (r.invoice_id and r.amount)]
    say("identity anchors first: {} registry rows".format(len(identities)))
    for r in sorted(identities, key=lambda x: PRIORITY.get(x.source_type, 9)):
        resolver.assign(r)
    for r in sorted(records, key=lambda x: PRIORITY.get(x.source_type, 9)):
        if r.vendor and (r.invoice_id or r.amount):
            resolver.assign(r)
    clusters = resolver.finalize()
    say("step 2 done: {} vendor clusters".format(len(clusters)))

    transactions = []
    provenance = []
    for r in records:
        if not (r.invoice_id and r.amount):
            continue
        cluster, score = resolver.best_cluster_for_name(r.vendor)
        ev = evidence_vector(r, cluster) if cluster else {}
        txn = SimpleNamespace(
            invoice_id=r.invoice_id,
            vendor_cluster=cluster["id"] if cluster else "",
            vendor_raw=r.vendor,
            amount=r.amount,
            date=r.date,
            source=r.source_type,
            source_file=r.source_file,
            confidence=r.confidence,
            match_score=round(score, 4),
        )
        transactions.append(txn)
        provenance.append(
            {
                "invoice_id": r.invoice_id,
                "value_amount": r.amount,
                "source_file": r.source_file,
                "source_type": r.source_type,
                "vendor_cluster": txn.vendor_cluster,
                "match_score": txn.match_score,
                "evidence": ev,
            }
        )

    say("step 3 of 4: reconcile {} rows, resolve conflicts, drop duplicates".format(len(transactions)))
    naive_unique, _ = dedupe_transactions(transactions)
    unique, conflicts = reconcile(transactions)
    naive_amounts = {t.invoice_id: t.amount for t in naive_unique}
    naive_wrong = sum(1 for t in unique if naive_amounts.get(t.invoice_id) != t.amount)
    say(
        "step 3 done: {} unique invoices, {} cross-source conflicts ({} resolved, {} for review), "
        "load-order baseline disagrees on {} of them".format(
            len(unique),
            len(conflicts),
            sum(1 for c in conflicts if c["decision"] == "resolved"),
            sum(1 for c in conflicts if c["decision"] == "review"),
            naive_wrong,
        )
    )

    say("step 4 of 4: write canonical outputs")
    with open(os.path.join(canonical_dir, "vendors.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)

    with open(os.path.join(canonical_dir, "transactions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["invoice_id", "vendor_cluster", "amount_inr", "date", "source"])
        for t in unique:
            w.writerow([t.invoice_id, t.vendor_cluster, t.amount, t.date, t.source])

    with open(os.path.join(canonical_dir, "provenance.jsonl"), "w", encoding="utf-8") as f:
        for p in provenance:
            f.write(json.dumps(p) + "\n")

    with open(os.path.join(canonical_dir, "conflicts.json"), "w", encoding="utf-8") as f:
        json.dump(conflicts, f, indent=2)

    source_counts = {}
    for r in records:
        source_counts[r.source_type] = source_counts.get(r.source_type, 0) + 1

    report = {
        "raw_records": len(records),
        "source_counts": source_counts,
        "receipts_parsed": source_counts.get("receipt", 0),
        "vendor_clusters": len(clusters),
        "raw_transactions": len(transactions),
        "unique_transactions": len(unique),
        "duplicates_removed": len(transactions) - len(unique),
        "conflicts_detected": len(conflicts),
        "conflicts_resolved": sum(1 for c in conflicts if c["decision"] == "resolved"),
        "conflicts_for_review": sum(1 for c in conflicts if c["decision"] == "review"),
        "load_order_baseline_wrong": naive_wrong,
        "total_spend_inr": sum(t.amount for t in unique),
        "generated_from": os.path.basename(raw_dir),
    }
    with open(os.path.join(canonical_dir, "reconciliation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    say("step 4 done: canonical/vendors.json, transactions.csv, provenance.jsonl, conflicts.json, reconciliation_report.json")
    say("total spend {} INR from {} unique invoices".format(report["total_spend_inr"], report["unique_transactions"]))
    return report
