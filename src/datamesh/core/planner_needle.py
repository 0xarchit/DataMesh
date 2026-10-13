import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TOOLS_PATH = os.path.join(ROOT, "training", "needle", "tools.json")
WEIGHTS_PATH = os.environ.get(
    "DATAMESH_NEEDLE_WEIGHTS", os.path.join(ROOT, "training", "needle", "tuned_v10.cact")
)

TOOL_SPEC = {
    "total_spend": ("sum", "amount_inr"),
    "count_transactions": ("count", "invoice_id"),
    "count_vendors": ("distinct_count", "vendor_cluster"),
    "vendor_city": ("lookup", "city"),
    "vendor_gstin": ("lookup", "tax_id"),
    "vendor_name": ("lookup", "name"),
}

_agent = None
_broken = False


def available():
    return os.path.exists(WEIGHTS_PATH) and os.path.exists(TOOLS_PATH)


def load():
    global _agent, _broken
    if _agent is not None or _broken:
        return _agent
    try:
        import needle

        with open(TOOLS_PATH, encoding="utf-8") as f:
            tools = json.dumps(json.load(f), separators=(",", ":"), ensure_ascii=False)
        _agent = needle.Needle(tools=tools, weights=WEIGHTS_PATH, auto_date=False, stateless=True)
    except Exception:
        _broken = True
        _agent = None
    return _agent


def plan(question):
    agent = load()
    if agent is None:
        return None
    try:
        response = agent.complete(question)
    except Exception:
        return None
    calls = response.get("function_calls") or []
    if not calls:
        return None
    mapped = TOOL_SPEC.get(calls[0].get("name"))
    if mapped is None:
        return None
    op, column = mapped
    entity = (calls[0].get("arguments") or {}).get("vendor")
    return {"op": op, "entity": (entity or "").strip() or None, "column": column}
