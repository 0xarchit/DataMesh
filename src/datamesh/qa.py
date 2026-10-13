import csv
import json
import os
import re

from datamesh.core import llm
from datamesh.core.schema import normalize_name

PLAN_SYSTEM = (
    "You turn a question into a query plan over a vendor transaction table. "
    "Reply with strict JSON only, no explanation, no markdown.\n"
    "Keys:\n"
    'op: "sum" | "count" | "distinct_count" | "lookup"\n'
    "entity: the vendor the question is about, or null when it spans all vendors\n"
    "column: for op lookup use city, tax_id, name or id, otherwise use amount_inr, invoice_id or vendor_cluster\n"
    "op count returns how many transaction rows match, op sum adds amount_inr, "
    "op distinct_count returns how many distinct vendors exist, op lookup returns one vendor attribute.\n"
    "Examples:\n"
    'Question: How many distinct vendors appear in the data?\n'
    '{"op": "distinct_count", "entity": null, "column": "vendor_cluster"}\n'
    'Question: What is the total spend with Acme Ltd in INR?\n'
    '{"op": "sum", "entity": "Acme Ltd", "column": "amount_inr"}\n'
    'Question: What city is Wayne Enterprises registered in?\n'
    '{"op": "lookup", "entity": "Wayne Enterprises", "column": "city"}'
)


def load_canonical(canonical_dir):
    with open(os.path.join(canonical_dir, "vendors.json"), encoding="utf-8") as f:
        vendors = json.load(f)
    rows = []
    with open(os.path.join(canonical_dir, "transactions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return vendors, rows


def _words(text):
    return " " + re.sub(r"[^a-z0-9]+", " ", normalize_name(text)).strip() + " "


def find_vendor(text, vendors):
    if not text:
        return None
    q = _words(text)
    best, best_len = None, 0
    for v in vendors:
        name = _words(v["name"]).strip()
        if len(name) < 4:
            continue
        if " " + name + " " in q and len(name) > best_len:
            best, best_len = v, len(name)
    return best


def plan_llm(question, on_token=None):
    if not llm.available():
        return None
    try:
        raw = llm.chat(question, system=PLAN_SYSTEM, max_tokens=220, on_token=on_token)
    except Exception:
        return None
    spec = llm.parse_json(raw)
    if not spec or spec.get("op") not in ("sum", "count", "distinct_count", "lookup"):
        return None
    return spec


GLOBAL_PHRASES = ("across all", "all vendors", "distinct vendors", "how many vendors", "in the data", "overall")

WITH_ENTITY = re.compile(r"\b(?:with|of|for)\s+([A-Za-z][A-Za-z0-9&.,'\- ]{2,40}?)\s*(?:in inr|\?|$|,|\.)", re.I)


def looks_global(question):
    q = question.lower()
    return any(phrase in q for phrase in GLOBAL_PHRASES)


def mentioned_entity(question):
    m = WITH_ENTITY.search(question)
    return m.group(1).strip(" ,.") if m else None


def plan_rules(question, vendors):
    q = question.lower()
    match = find_vendor(question, vendors)
    entity = match["name"] if match else None
    if "how many distinct vendors" in q or "how many vendors" in q:
        return {"op": "distinct_count", "entity": None, "column": "vendor_cluster"}
    if "how many transactions" in q:
        if entity is None:
            return None
        return {"op": "count", "entity": entity, "column": "invoice_id"}
    if "total spend" in q or "total amount" in q:
        if entity is None and not looks_global(question):
            return None
        return {"op": "sum", "entity": entity, "column": "amount_inr"}
    if "gstin" in q or "tax" in q:
        if entity is None:
            return None
        return {"op": "lookup", "entity": entity, "column": "tax_id"}
    if "city" in q:
        if entity is None:
            return None
        return {"op": "lookup", "entity": entity, "column": "city"}
    return None


def select_rows(vendors, rows, entity):
    if not entity:
        return rows
    v = find_vendor(entity, vendors)
    if v is None:
        return []
    return [r for r in rows if r["vendor_cluster"] == v["id"]]


def execute(spec, vendors, rows):
    if not spec:
        return None
    op = spec.get("op")
    column = spec.get("column") or "amount_inr"
    entity = spec.get("entity")
    if op == "distinct_count":
        return len(vendors)
    if op == "count":
        return len(select_rows(vendors, rows, entity))
    if op == "sum":
        return sum(int(r["amount_inr"]) for r in select_rows(vendors, rows, entity))
    if op == "lookup":
        v = find_vendor(entity, vendors)
        if v is None:
            return None
        if column in ("name", "id", "city", "tax_id"):
            return v.get(column)
        matched = select_rows(vendors, rows, entity)
        if not matched:
            return None
        return matched[0].get(column)
    return None


def plan(question, vendors, use_llm=True, on_token=None):
    if use_llm:
        spec = plan_llm(question, on_token=on_token)
        if spec is not None:
            return spec, "query plan from local model"
    spec = plan_rules(question, vendors)
    if spec is None:
        return None, "out of scope"
    return spec, "query plan from keyword rules"


def outcome(question, vendors, rows, use_llm=True, on_token=None):
    spec, source = plan(question, vendors, use_llm=use_llm, on_token=on_token)
    guess = mentioned_entity(question)
    if spec is None:
        if guess and find_vendor(guess, vendors) is None:
            return {
                "status": "entity_not_found",
                "value": None,
                "plan": None,
                "source": source,
                "vendor": None,
                "message": "No vendor matching {!r} exists in the consolidated table.".format(guess),
            }
        return {
            "status": "out_of_scope",
            "value": None,
            "plan": None,
            "source": source,
            "vendor": None,
            "message": "This question is not about the vendors or transactions in this workspace, so nothing is computed.",
        }
    entity = spec.get("entity")
    vendor = find_vendor(entity, vendors) if entity else None
    if entity and vendor is None:
        return {
            "status": "entity_not_found",
            "value": None,
            "plan": spec,
            "source": source,
            "vendor": None,
            "message": "No vendor matching {!r} exists in the consolidated table.".format(entity),
        }
    if vendor is None and guess and find_vendor(guess, vendors) is None and not looks_global(question):
        return {
            "status": "entity_not_found",
            "value": None,
            "plan": spec,
            "source": source,
            "vendor": None,
            "message": "No vendor matching {!r} exists in the consolidated table.".format(guess),
        }
    return {
        "status": "answered",
        "value": normalize(execute(spec, vendors, rows)),
        "plan": spec,
        "source": source,
        "vendor": vendor,
        "message": "",
    }


def answer(question, vendors, rows, use_llm=True, on_token=None, emit=None):
    spec, source = plan(question, vendors, use_llm=use_llm, on_token=on_token)
    if emit:
        emit("query plan via {}: {}".format(source, json.dumps(spec, sort_keys=True)))
    return execute(spec, vendors, rows)


def normalize(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    digits = re.findall(r"\d[\d,]*", text)
    if digits and text.replace(",", "").replace(".", "").isdigit():
        return int(text.replace(",", ""))
    return text.lower()
