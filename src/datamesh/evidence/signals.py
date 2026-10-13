import json
import os
from datetime import date
from difflib import SequenceMatcher

from datamesh.core.schema import canon_city, normalize_name

SOURCE_RELIABILITY = {"csv": 1.0, "xlsx": 1.0, "pdf": 0.95, "receipt": 0.8}
KNOWN_SOURCE_TYPES = ("csv", "xlsx", "pdf", "receipt")

NEUTRAL = 0.5


def name_similarity(a, b):
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    tokens_a, tokens_b = set(na.split()), set(nb.split())
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    ratio = SequenceMatcher(None, na, nb).ratio()
    return round(max(overlap * 0.6 + 0.4 * ratio if overlap else 0.0, ratio * 0.5), 4)


def coreference(a, b):
    ta, tb = set(normalize_name(a).split()), set(normalize_name(b).split())
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:
        return 1.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) == 1:
        word = next(iter(short))
        if "".join(w[0] for w in sorted(long_)) == word:
            return 0.8
    return 0.0


def _as_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def freshness(record_date, cluster_date):
    a, b = _as_date(record_date), _as_date(cluster_date)
    if a is None or b is None:
        return NEUTRAL
    return round(max(0.0, 1.0 - abs((a - b).days) / 365.0), 4)


def cross_source_consensus(source_type, seen_types):
    seen = set(seen_types or ())
    seen.add(source_type)
    return round(len(seen & set(KNOWN_SOURCE_TYPES)) / float(len(KNOWN_SOURCE_TYPES)), 4)


def evidence_vector(record, cluster):
    tax_match = 1.0 if record.tax_id and cluster.get("tax_id") and record.tax_id == cluster["tax_id"] else 0.0
    city_match = 1.0 if canon_city(record.city) and canon_city(record.city) == cluster.get("city") else 0.0
    return {
        "E1_semantic_name": name_similarity(record.vendor, cluster.get("name", "")),
        "E2_tax_id_match": tax_match,
        "E3_city_agreement": city_match,
        "E4_coreference": coreference(record.vendor, cluster.get("name", "")),
        "E5_provenance_confidence": SOURCE_RELIABILITY.get(record.source_type, 0.5) * record.confidence,
        "E6_temporal_freshness": freshness(record.date, cluster.get("last_date")),
        "E7_cross_source_consensus": cross_source_consensus(record.source_type, cluster.get("source_types")),
    }


WEIGHTS = {
    "E1_semantic_name": 0.60,
    "E2_tax_id_match": 0.15,
    "E4_coreference": 0.10,
    "E3_city_agreement": 0.05,
    "E5_provenance_confidence": 0.04,
    "E6_temporal_freshness": 0.03,
    "E7_cross_source_consensus": 0.03,
}


_LEARNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_weights.json")
_active_weights = None


def active_weights():
    global _active_weights
    if _active_weights is None:
        try:
            with open(_LEARNED_PATH, encoding="utf-8") as f:
                learned = {k: float(v) for k, v in json.load(f)["weights"].items()}
            _active_weights = learned if abs(sum(learned.values()) - 1.0) < 1e-6 else WEIGHTS
        except Exception:
            _active_weights = WEIGHTS
    return _active_weights


def combine(ev):
    weights = active_weights()
    return round(sum(weights.get(k, 0.0) * ev.get(k, 0.0) for k in WEIGHTS), 4)


def demo():
    assert name_similarity("Acme Ltd", "ACME") == 1.0
    assert coreference("Hooli", "Hooli Services") == 1.0
    assert coreference("Acme Ltd", "Globex Corporation") == 0.0
    assert freshness("2025-03-01", "2025-03-11") > freshness("2025-03-01", "2025-11-01")
    assert freshness("", "") == NEUTRAL
    assert cross_source_consensus("pdf", {"csv", "xlsx"}) == 0.75
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"
    print("signals self-check ok")


if __name__ == "__main__":
    demo()
