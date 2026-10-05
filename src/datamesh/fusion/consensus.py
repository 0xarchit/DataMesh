"""Cross-source conflict detection for the same business key.

Two sources describing the same invoice can disagree about the amount. Loading
order decides the winner by accident; this module decides on purpose, and
records what it saw either way.

Three outcomes per invoice:
  AGREED    every copy carried the same amount, nothing to decide
  RESOLVED  copies disagreed, evidence picked one
  REVIEW    copies disagreed and evidence could not separate them - the row is
            flagged rather than silently merged (the abstain branch)
"""

from collections import Counter, defaultdict

SOURCE_PRIORITY = {"csv": 0, "xlsx": 1, "pdf": 2, "receipt": 3}
SOURCE_RELIABILITY = {"csv": 1.0, "xlsx": 1.0, "pdf": 0.95, "receipt": 0.8}

# xlsx and pdf reliability differ by less than this, so that gap on its own is
# not evidence and the invoice is flagged instead of guessed.
RELIABILITY_MARGIN = 0.1

AGREED = "agreed"
RESOLVED = "resolved"
REVIEW = "review"


def group_by_invoice(transactions):
    groups = defaultdict(list)
    for t in transactions:
        groups[t.invoice_id].append(t)
    return groups


def vendor_profiles(transactions):
    """Count how often each amount occurs per vendor cluster.

    A vendor bills from a small set of recurring values, so a digit-shift
    misread appears once while a real amount repeats across other invoices.
    """
    profile = defaultdict(Counter)
    for t in transactions:
        profile[t.vendor_cluster][t.amount] += 1
    return profile


def _by_priority(rows):
    return sorted(rows, key=lambda r: SOURCE_PRIORITY.get(r.source, 9))


def resolve_group(rows, profile):
    """Decide one amount for one invoice. Returns (chosen_row, decision, detail)."""
    ordered = _by_priority(rows)
    amounts = [r.amount for r in ordered]
    counts = Counter(amounts)

    if len(counts) == 1:
        return ordered[0], AGREED, {"candidates": amounts, "rule": "all copies agree"}

    top, top_n = counts.most_common(1)[0]
    if top_n * 2 > len(amounts):
        chosen = next(r for r in ordered if r.amount == top)
        return chosen, RESOLVED, {"candidates": amounts, "rule": "majority of the copies"}

    prof = Counter(profile.get(ordered[0].vendor_cluster, Counter()))
    # evidence has to come from the vendor's OTHER invoices: this invoice's own
    # copies would inflate every candidate by one and cancel the difference out
    prof.subtract(amounts)

    def strength(amount):
        best = min(
            (r.source for r in ordered if r.amount == amount),
            key=lambda s: SOURCE_PRIORITY.get(s, 9),
        )
        return (max(prof.get(amount, 0), 0), SOURCE_RELIABILITY.get(best, 0.5))

    ranked = sorted(counts, key=strength, reverse=True)
    first, second = strength(ranked[0]), strength(ranked[1])
    separable = first[0] > second[0] or (first[1] - second[1]) >= RELIABILITY_MARGIN
    if separable:
        chosen = next(r for r in ordered if r.amount == ranked[0])
        return chosen, RESOLVED, {"candidates": amounts, "rule": "vendor amount profile, then source reliability"}

    chosen = ordered[0]
    return chosen, REVIEW, {
        "candidates": amounts,
        "rule": "no evidence separates the candidates, flagged for human review",
    }


def reconcile(transactions):
    """Collapse duplicate invoices, resolving cross-source disagreements.

    Returns (unique_transactions, conflict_report).
    """
    profile = vendor_profiles(transactions)
    unique, conflicts = [], []
    for invoice_id, rows in group_by_invoice(transactions).items():
        chosen, decision, detail = resolve_group(rows, profile)
        unique.append(chosen)
        if decision != AGREED:
            conflicts.append(
                {
                    "invoice_id": invoice_id,
                    "vendor_cluster": chosen.vendor_cluster,
                    "decision": decision,
                    "chosen_amount": chosen.amount,
                    "chosen_source": chosen.source,
                    "candidate_amounts": detail["candidates"],
                    "candidate_sources": [r.source for r in _by_priority(rows)],
                    "rule": detail["rule"],
                }
            )
    unique.sort(key=lambda t: t.invoice_id)
    conflicts.sort(key=lambda c: c["invoice_id"])
    return unique, conflicts


def demo():
    """Self-check: three copies resolve by majority, two copies by vendor profile."""
    from types import SimpleNamespace

    def row(source, amount, cluster="C001"):
        return SimpleNamespace(invoice_id="INV-0001", vendor_cluster=cluster, amount=amount, source=source)

    profile = defaultdict(Counter, {"C001": Counter({24800: 3, 96000: 2})})

    # same invoice in three sources, the xlsx copy is a digit-shift misread
    chosen, decision, _ = resolve_group([row("xlsx", 248000), row("pdf", 24800), row("receipt", 24800)], profile)
    assert (chosen.amount, decision) == (24800, RESOLVED), (chosen.amount, decision)

    # two copies, no majority - the vendor amount profile breaks the tie
    chosen, decision, _ = resolve_group([row("xlsx", 248000), row("pdf", 24800)], profile)
    assert (chosen.amount, decision) == (24800, RESOLVED), (chosen.amount, decision)

    # two copies, nothing to separate them - flagged, not guessed
    chosen, decision, _ = resolve_group([row("xlsx", 111), row("pdf", 222)], defaultdict(Counter))
    assert decision == REVIEW, decision

    # agreement is not a conflict
    chosen, decision, _ = resolve_group([row("xlsx", 500), row("pdf", 500)], profile)
    assert decision == AGREED, decision

    print("consensus self-check ok")


if __name__ == "__main__":
    demo()
