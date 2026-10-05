"""Gold check for the consolidation pipeline.

Run after scripts/run_pipeline.py. Every assertion compares the pipeline's
canonical output against the generator's ground truth, so a change that breaks
reconciliation, deduplication or conflict resolution fails here instead of
quietly changing the demo numbers.

    .venv/Scripts/python.exe tests/test_gold.py
"""

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

GOLD = os.path.join(ROOT, "data", "gold")
CANON = os.path.join(ROOT, "data", "canonical")


def _load(name, folder=GOLD):
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        return json.load(f)


def _canonical_rows():
    with open(os.path.join(CANON, "transactions.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_reconciliation():
    truth = {t["invoice_id"]: t for t in _load("invoice_truth.json")}
    rows = _canonical_rows()

    assert len(rows) == len(truth), "expected {} invoices, canonical has {}".format(len(truth), len(rows))
    assert {r["invoice_id"] for r in rows} == set(truth), "canonical invoice set differs from ground truth"

    spend = sum(int(r["amount_inr"]) for r in rows)
    expected = sum(t["amount"] for t in truth.values())
    assert spend == expected, "total spend {}, ground truth {}".format(spend, expected)

    vendors = _load("vendors.json", CANON)
    expected_clusters = len({t["vendor_id"] for t in truth.values()})
    assert len(vendors) == expected_clusters, "{} clusters, expected {}".format(len(vendors), expected_clusters)
    print("reconciliation ok: {} invoices, {} clusters, {} INR".format(len(rows), len(vendors), spend))


def check_conflicts():
    planted = {c["invoice_id"]: c for c in _load("conflicts.json")}
    decided = {c["invoice_id"]: c for c in _load("conflicts.json", CANON)}

    missed = set(planted) - set(decided)
    assert not missed, "conflicts not detected: {}".format(sorted(missed))

    wrong = [i for i, c in decided.items() if c["chosen_amount"] != planted[i]["true_amount"]]
    assert not wrong, "conflicts resolved to the wrong amount: {}".format(sorted(wrong))

    for c in decided.values():
        assert c["decision"] in ("resolved", "review"), "unknown decision {}".format(c["decision"])
    print("conflicts ok: {} planted, all detected, 0 resolved incorrectly".format(len(planted)))


def check_questions():
    from datamesh import qa

    vendors, rows = qa.load_canonical(CANON)
    questions = _load("questions.json")
    failures = []
    for item in questions:
        got = qa.normalize(qa.answer(item["q"], vendors, rows, use_llm=False))
        if got != qa.normalize(item["answer"]):
            failures.append((item["q"], item["answer"], got))
    assert not failures, "gold questions answered wrong: {}".format(failures)
    print("questions ok: {} / {} answered from the consolidated table".format(len(questions), len(questions)))


def check_receipts():
    expected = {r["invoice_id"] for r in _load("receipts.json")}
    on_disk = {
        os.path.splitext(n)[0]
        for n in os.listdir(os.path.join(ROOT, "data", "raw", "receipts"))
        if n.endswith(".png")
    }
    assert on_disk == expected, "receipt images {} do not match ground truth {}".format(
        sorted(on_disk), sorted(expected)
    )
    print("receipts ok: {} images, filenames match ground truth".format(len(on_disk)))


def main():
    check_reconciliation()
    check_conflicts()
    check_questions()
    check_receipts()
    print("\nall gold checks passed")


if __name__ == "__main__":
    main()
