from collections import Counter, defaultdict

SOURCE_PRIORITY = {"csv": 0, "xlsx": 1, "pdf": 2, "receipt": 3}
SOURCE_RELIABILITY = {"csv": 1.0, "xlsx": 1.0, "pdf": 0.95, "receipt": 0.8}

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
    profile = defaultdict(Counter)
    for t in transactions:
        profile[t.vendor_cluster][t.amount] += 1
    return profile


def _by_priority(rows):
    return sorted(rows, key=lambda r: SOURCE_PRIORITY.get(r.source, 9))


def resolve_group(rows, profile):
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
    from types import SimpleNamespace

    def row(source, amount, cluster="C001"):
        return SimpleNamespace(invoice_id="INV-0001", vendor_cluster=cluster, amount=amount, source=source)

    profile = defaultdict(Counter, {"C001": Counter({24800: 3, 96000: 2})})

    chosen, decision, _ = resolve_group([row("xlsx", 248000), row("pdf", 24800), row("receipt", 24800)], profile)
    assert (chosen.amount, decision) == (24800, RESOLVED), (chosen.amount, decision)

    chosen, decision, _ = resolve_group([row("xlsx", 248000), row("pdf", 24800)], profile)
    assert (chosen.amount, decision) == (24800, RESOLVED), (chosen.amount, decision)

    chosen, decision, _ = resolve_group([row("xlsx", 111), row("pdf", 222)], defaultdict(Counter))
    assert decision == REVIEW, decision

    chosen, decision, _ = resolve_group([row("xlsx", 500), row("pdf", 500)], profile)
    assert decision == AGREED, decision

    print("consensus self-check ok")


if __name__ == "__main__":
    demo()
