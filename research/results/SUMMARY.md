# DataMesh AI demo results

Synthetic multi-format data: CSV master, Excel ledger, PDF invoices and
scanned receipt images.

## Consolidation

- raw records: 327
- sources: {'csv': 20, 'xlsx': 100, 'pdf': 199, 'receipt': 8}
- vendor clusters: 20 (ground truth 20)
- raw transactions: 307, unique: 199, duplicates removed: 108
- total spend: 6185400 INR (matches gold)

## Cross-source conflict resolution

20 invoices were written into more than one source with a different amount.

- detected: 20, missed: none, spurious: none
- resolved: 20, flagged for review: 0
- keeping the first copy in load order gets 7 of them wrong
- DataMesh gets 0 wrong

## Question answering (10 gold questions)

Three arms, same local model, same prompt, same questions. The two RAG arms
use the same vector store, the same embedding model and the same retrieval
depth; only the corpus differs.

- DataMesh consolidated table: **10 correct**
- RAG over the raw record corpus: 1 correct
- RAG over the consolidated table: 4 correct
- RAG arm ran as: local llm: .\weights\qwen-3.8-27B.gguf

RAG errors over the raw corpus come from no entity reconciliation: duplicate
rows inflate totals and counts, name variants split one vendor into many, and
city aliases are never normalized.

### Full-corpus ablation

Top-k truncation is a confound of its own: with 15 of 327 raw chunks retrieved,
the model cannot see enough rows to total anything, and would fail for a reason
that has nothing to do with consolidation. So both RAG arms are run again with the
**entire corpus in context** instead of the top-k. Now the corpus is the only
difference between them.

- RAG over the full raw corpus: **20%**
- RAG over the full consolidated corpus: **60%**

The consolidated arm recovers the entity questions but still fails the
aggregations, because it is asking a language model to sum 199 rows in context.
The remaining gap is not retrieval and not consolidation - it is that RAG makes
the model do the arithmetic, while the pipeline plans a query and executes it
against the table.

## Downstream prediction (H3)

Balanced accuracy and macro F1 are reported because the two arms do not have the
same row count - fragmentation is the effect being measured - and both are invariant
to class prevalence. The raw arm is also handed the ground-truth vendor list to
assign labels, which is generous to it.

- Raw table (41 rows, 0.4146 high-value): acc 0.7561, balanced acc 0.7488, macro F1 0.7488
- Consolidated table (20 rows, 0.45 high-value): acc 0.9, balanced acc 0.899, macro F1 0.899

## Reproduce

```
uv venv .venv --python 3.11
uv pip install --python .venv -e .
uv run --python .venv python scripts/run_demo.py
```

Start a local OpenAI-compatible model on port 8080 first, or the RAG arm degrades
to keyword rules and the vision receipts are skipped.
