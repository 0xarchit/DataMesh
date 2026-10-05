from collections import Counter

from datamesh.core.schema import canon_city, normalize_name
from datamesh.evidence.signals import combine, evidence_vector, name_similarity

MERGE_THRESHOLD = 0.68


class VendorResolver:
    def __init__(self):
        self.clusters = []

    def _new_cluster(self, record):
        cluster = {
            "id": "C{:03d}".format(len(self.clusters) + 1),
            "name": record.vendor,
            "tax_id": record.tax_id,
            "city": canon_city(record.city),
            "last_date": record.date,
            "source_types": [record.source_type],
            "city_votes": Counter({canon_city(record.city): 1}) if record.city else Counter(),
            "sources": [record.source_file],
            "members": [record.source_file],
        }
        self.clusters.append(cluster)
        return cluster

    def assign(self, record):
        best, best_score, best_ev = None, 0.0, None
        for cluster in self.clusters:
            ev = evidence_vector(record, cluster)
            score = combine(ev)
            if score > best_score:
                best, best_score, best_ev = cluster, score, ev
        if best is None or best_score < MERGE_THRESHOLD:
            cluster = self._new_cluster(record)
            return cluster, 1.0, evidence_vector(record, cluster)
        if record.tax_id and not best["tax_id"]:
            best["tax_id"] = record.tax_id
        if record.city:
            best["city_votes"][canon_city(record.city)] += 1
        # E6 and E7 read these back, so they have to advance as the cluster grows
        if record.date > best["last_date"]:
            best["last_date"] = record.date
        if record.source_type not in best["source_types"]:
            best["source_types"].append(record.source_type)
        best["sources"].append(record.source_file)
        best["members"].append(record.source_file)
        return best, best_score, best_ev

    def finalize(self):
        for cluster in self.clusters:
            if cluster["city_votes"]:
                cluster["city"] = cluster["city_votes"].most_common(1)[0][0]
            cluster["sources"] = sorted(set(cluster["sources"]))
            cluster["source_types"] = sorted(set(cluster["source_types"]))
            cluster.pop("city_votes", None)
        return self.clusters

    def best_cluster_for_name(self, name):
        best, best_score = None, 0.0
        for cluster in self.clusters:
            score = name_similarity(name, cluster["name"])
            if score > best_score:
                best, best_score = cluster, score
        return best, best_score


def dedupe_transactions(transactions):
    unique = {}
    duplicates = []
    for t in transactions:
        key = (t.vendor_cluster, t.amount, t.date)
        if t.invoice_id:
            key = ("invoice", t.invoice_id)
        if key in unique:
            duplicates.append(t)
        else:
            unique[key] = t
    return list(unique.values()), duplicates
