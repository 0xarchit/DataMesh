import csv
import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datamesh.core import llm
from datamesh.ingestion import loaders

try:
    import chromadb
except Exception:
    chromadb = None

STORE_COLLECTION = "records"
CANON_COLLECTION = "canonical_records"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 15


def fingerprint(raw_dir):
    total = 0
    files = 0
    for root, _, names in os.walk(raw_dir):
        for name in names:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            files += 1
            total += st.st_size + int(st.st_mtime)
    return "{}-{}".format(files, total)


def record_to_text(r):
    parts = []
    if r.vendor:
        parts.append("vendor " + r.vendor)
    if r.invoice_id:
        parts.append("invoice " + r.invoice_id)
    if r.amount:
        parts.append("amount " + str(r.amount))
    if r.date:
        parts.append("date " + r.date)
    if r.city:
        parts.append("city " + r.city)
    if r.tax_id:
        parts.append("gstin " + r.tax_id)
    return (r.source_type + " " + r.source_file + " | " + " | ".join(parts)).strip()


def build_corpus(raw_dir, use_llm=True, emit=None, on_token=None):
    records = loaders.load_all(raw_dir, use_llm=use_llm, emit=emit, on_token=on_token)
    chunks = [record_to_text(r) for r in records]
    say = emit or (lambda s: None)
    say("corpus built: {} raw text chunks".format(len(chunks)))
    return chunks, records


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]


class Retriever:
    def __init__(self, chunks, collection=None, k=DEFAULT_K):
        self.chunks = chunks
        self.collection = collection
        self.k = k
        self.vectorizer = None
        self.matrix = None
        if collection is None:
            self.vectorizer = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(chunks)

    def engine(self):
        if self.collection is not None:
            return "chromadb+{}".format(EMBEDDING_MODEL)
        return "tfidf-fallback"

    def retrieve(self, query, k=None):
        depth = k or self.k
        if self.collection is not None:
            res = self.collection.query(query_texts=[query], n_results=depth)
            return [d for d in res["documents"][0] if d]
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:depth]
        return [self.chunks[i] for i in order if scores[i] > 0]


def _client(store_dir):
    if chromadb is None:
        raise RuntimeError("chromadb is not installed")
    os.makedirs(store_dir, exist_ok=True)
    return chromadb.PersistentClient(path=store_dir)


def store_meta(store_dir):
    path = os.path.join(store_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def store_is_fresh(raw_dir, store_dir):
    meta = store_meta(store_dir)
    return bool(meta) and meta.get("fingerprint") == fingerprint(raw_dir)


def load_store(raw_dir, store_dir, use_llm=True, emit=None, on_token=None, rebuild=False, k=DEFAULT_K):
    say = emit or (lambda s: None)
    fp = fingerprint(raw_dir)
    if not rebuild and store_is_fresh(raw_dir, store_dir):
        try:
            col = _client(store_dir).get_or_create_collection(STORE_COLLECTION)
            if col.count():
                docs = col.get()["documents"]
                say("vector store reused: {} vectors, raw fingerprint {} unchanged".format(len(docs), fp))
                return docs, Retriever(docs, col, k=k)
        except Exception as exc:
            say("vector store reuse failed, rebuilding: {}".format(exc))
    chunks, records = build_corpus(raw_dir, use_llm=use_llm, emit=emit, on_token=on_token)
    say("embedding {} chunks with {} into chromadb".format(len(chunks), EMBEDDING_MODEL))
    client = _client(store_dir)
    try:
        client.delete_collection(STORE_COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(STORE_COLLECTION)
    col.add(ids=[str(i) for i in range(len(chunks))], documents=chunks)
    with open(os.path.join(store_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"fingerprint": fp, "collection": STORE_COLLECTION, "count": len(chunks), "embedding": EMBEDDING_MODEL, "k": k},
            f,
            indent=2,
        )
    say("vector store ready: {} vectors in collection '{}'".format(col.count(), STORE_COLLECTION))
    return chunks, Retriever(chunks, col, k=k)


def canonical_chunks(canonical_dir):
    with open(os.path.join(canonical_dir, "vendors.json"), encoding="utf-8") as f:
        vendors = json.load(f)
    by_id = {v["id"]: v for v in vendors}
    chunks = [
        "vendor registry {} | name {} | city {} | gstin {}".format(v["id"], v["name"], v["city"], v["tax_id"])
        for v in vendors
    ]
    with open(os.path.join(canonical_dir, "transactions.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vendor = by_id.get(row["vendor_cluster"], {})
            chunks.append(
                "canonical transaction {} | vendor {} | vendor_id {} | amount {} | date {} | source {}".format(
                    row["invoice_id"],
                    vendor.get("name", ""),
                    row["vendor_cluster"],
                    row["amount_inr"],
                    row["date"],
                    row["source"],
                )
            )
    return chunks


def canonical_store(canonical_dir, store_dir, k=DEFAULT_K, emit=None):
    say = emit or (lambda s: None)
    chunks = canonical_chunks(canonical_dir)
    client = _client(store_dir)
    try:
        client.delete_collection(CANON_COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(CANON_COLLECTION)
    col.add(ids=[str(i) for i in range(len(chunks))], documents=chunks)
    say("control vector store ready: {} vectors in '{}'".format(col.count(), CANON_COLLECTION))
    return chunks, Retriever(chunks, col, k=k)


STOPWORDS = {
    "how", "many", "distinct", "vendors", "appear", "in", "the", "data", "what", "is",
    "total", "spend", "across", "all", "with", "city", "registered", "gstin", "of",
    "transactions", "involve", "and", "for", "to", "a", "an",
}


def _extract_number(text):
    if text is None:
        return None
    digits = re.findall(r"\d[\d,]*", str(text).replace(".0", ""))
    if not digits:
        return None
    try:
        return int(digits[0].replace(",", ""))
    except ValueError:
        return None


def _field(chunk, label):
    for piece in chunk.split("|"):
        piece = piece.strip()
        if piece.startswith(label + " "):
            return piece[len(label) + 1 :].strip()
    return ""


def _keywords(question):
    return {w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOPWORDS}


def keyword_rules_answer(question, chunks):
    q = question.lower()
    keywords = _keywords(question)
    if "how many distinct vendors" in q:
        return len({_field(c, "vendor") for c in chunks if _field(c, "vendor")})
    if "how many transactions" in q:
        matching = [c for c in chunks if any(w in _field(c, "vendor").lower() for w in keywords)]
        if not keywords:
            matching = chunks
        return sum(1 for c in matching if _field(c, "invoice"))
    matching = [c for c in chunks if any(w in _field(c, "vendor").lower() for w in keywords)]
    if "total spend" in q:
        if "across all" in q or "all vendors" in q:
            matching = chunks
        return sum(int(_field(c, "amount") or 0) for c in matching)
    if "city" in q:
        for c in matching:
            if _field(c, "city"):
                return _field(c, "city")
    if "gstin" in q:
        for c in matching:
            if _field(c, "gstin"):
                return _field(c, "gstin")
    return None


_warned_no_llm = False


def _warn_no_llm(say):
    global _warned_no_llm
    if not _warned_no_llm:
        _warned_no_llm = True
        say("no LLM server reachable - falling back to keyword rules; this arm is NOT a RAG run")


def answer_system(context):
    return (
        "You answer questions about raw business records using only these records. "
        "Output ONLY the answer value on a single line. No words, no commas, no explanation. "
        "Amounts and counts must be a plain integer.\n\nRecords:\n"
        + "\n".join(context)
    )


def rag_answer(question, retriever, chunks=None, use_llm=True, k=None, on_token=None, emit=None):
    say = emit or (lambda s: None)
    all_chunks = retriever.chunks if chunks is None else chunks
    if use_llm and llm.available():
        depth = k or retriever.k
        context = retriever.retrieve(question, k=depth) if chunks is None else chunks
        if not context:
            context = all_chunks
        system = answer_system(context)
        say("LLM rag retrieved {}/{} chunks by vector similarity, answering".format(len(context), len(all_chunks)))
        try:
            return llm.chat(question, system=system, max_tokens=64, on_token=on_token)
        except Exception as exc:
            say("llm failed, falling back to rules: {}".format(exc))
    _warn_no_llm(say)
    return keyword_rules_answer(question, all_chunks)


def normalize_answer(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text and text.replace(",", "").replace(".", "").isdigit():
        return int(text.replace(",", ""))
    return text.lower()
