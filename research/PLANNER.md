# The question planner

This note says exactly what the fine-tuned Needle model decides, what the deterministic
engine does, and what was deliberately kept out so the system is not propped up by word lists.

## The split

The project's rule is: the model plans, the engine computes. Nothing else.

| Decision | Made by | How |
|---|---|---|
| Which of the six tools the question needs | **model** | one forward pass, no LLM call |
| Which vendor the question names | **model** | copied out of the question text |
| Whether the question should be refused | **model** | emits an empty call list |
| Which operation and column that tool means | code | fixed table, `TOOL_SPEC` |
| Whether a plan is structurally usable | code | schema shape only |
| Resolving the vendor string to a table row | code | string match against `vendors.json` |
| The actual number | code | `execute()` over `data/canonical` |

The model never sees the data and never produces a number, so a fabricated value is not
representable. It answers only "which tool, which vendor, or refuse".

## What the engine does, and why each step is not a word list

- **`TOOL_SPEC`** maps the six tool names to `(op, column)`. It is a fixed rename between the
  model's vocabulary and the engine's, not a heuristic. `tests/unit/test_needle_planner.py`
  asserts it stays in step with `training/needle/tools.json`.
- **`needle_spec_ok`** checks shape only: a `lookup` or `count` must name a vendor, because
  those two tools declare `vendor` as required. It reads no words from the question.
- **`find_vendor`** matches the model's vendor string against the real vendor table by
  normalised tokens. It is driven by the data, so it works for any vendor name, including ones
  invented after training.
- **`execute`** computes the answer from the canonical table.

## What was deliberately not done

- No regex over the question to guess intent. Earlier drafts used a `GLOBAL_PHRASES` word list
  to decide whether a spend question spanned all vendors; that decision is now the model's, and
  the word list only survives inside the keyword-rules fallback, which is rule-based by design.
- No per-vendor special cases anywhere.
- No post-hoc rewriting of the model's answer beyond resolving the vendor name against the table.

## Training data

Built only from data, never from the demo's vendor names.

| Batch | What it adds | Examples |
|---|---|---|
| v4 | six tools, varied phrasing, no seeded vendor list and no seeded sentences | 1889 |
| v7 | `count_vendors` rebalanced, which was 5x rarer than every other tool | +290 |
| v8 | in-domain refusals, confusable entity spans, unseen name shapes, negation | 750 |
| v9 | global sum with the vendor omitted, which had zero examples | 165 |
| v10 | capitalised and single-word vendor names, which were absent entirely | 131 |
| v11 | negation refusals, `count_vendors` phrased without the word `total`, more capitalised names | 500 |
| v12 | no new data. Same 3838, trained from v10's weights instead of the base | 0 |

Final set: 3838 examples, 3454 train and 384 held out. 21 percent are refusals.

v11 and v12 share that set exactly, including the validation split, because the split comes from
`--seed 0` over the same file. Their validation losses are therefore directly comparable, and the
comparison is what identified warm-starting as the thing that worked.

Two label rules are enforced by the builder rather than left to the generators:

- a vendor-taking tool must carry a vendor that really occurs in the question;
- `count_vendors` takes no arguments, and `total_spend` may omit the vendor only for a
  genuine all-vendor question.

No training line carries one of the thirty demo vendor names. The generated batches were
filtered for this, and a check confirms the two vendor sets do not intersect, so a score on
the evaluation set cannot come from having seen the names.

## Why the result is a generalisation claim

The evaluation set is written around the thirty real vendors in `data/canonical`. Training uses
1591 invented vendor names, none of which appears in that set, and the two query sets share only
a couple of generic questions such as "how many vendors are there". So the measured accuracy is
on vendor names and phrasings the model has never seen.

The model is at its weakest on vendor-name shapes that training did not contain. The first
version had no all-capitals and no single-word vendor names at all, which is why it truncated
`CYBERDYNE SYSTEMS` to `CYBERDYNE`. That is a data gap, and it was closed with data.

## Against the other ways of answering the same questions

`scripts/run_benchmark.py` runs every arm over one consolidated table on the project's 42
questions, so the arms differ only in how the plan is produced.

| Arm | Correct | Total tokens | Total seconds |
|---|---|---|---|
| Local model writes the plan | 42/42 | **0** | **3.87** |
| Local 27B model writes the plan | 42/42 | 11,901 | 137.37 |
| Retrieval over raw records | 9/42 | 41,940 | 187.0 |
| Retrieval over the consolidated table | 17/42 | 30,003 | 163.77 |

The small model matches the 27B planner on accuracy while removing the token cost entirely and
cutting wall clock by 35 times, because the plan is one local forward pass instead of a
generation. Retrieval does well only where the answer is copied verbatim rather than computed:
it scores 8/8 on tax ids and 8/8 on city with the consolidated corpus, and near zero everywhere a
sum or a count has to be produced.

## Measured accuracy

`scripts/eval_planner.py` reports the raw model and the integrated planner chain separately so a
fallback can never be credited to the model. They are identical for both candidate models, which
is the point: the chain no longer improves on the model, and once that was true the
entity-resolution fallback was deleted as dead weight.

The v11 batch was built to fix the two named weaknesses below. It fixed them and it is still not
the shipped model, because it bought them with a new failure the benchmark could see. The v12 run
took a different route to the same gains and is the one that ships.

| Measure | v10 | v11 | **v12, shipped** |
|---|---|---|---|
| Frozen holdout, real vendors, 180 cases | 163/180, 90.6 percent | 166/180, 92.2 percent | **172/180, 95.6 percent** |
| Positive questions | 129/137, 94 percent | 127/137, 93 percent | **132/137, 96 percent** |
| Negation refusals | 11/14, 79 percent | 13/14, 93 percent | **13/14, 93 percent** |
| Missing-argument refusals | 3/4, 75 percent | 3/4, 75 percent | 3/4, 75 percent |
| Off-topic refusals | 11/13, 85 percent | 11/13, 85 percent | **12/13, 92 percent** |
| `count_vendors` | 8/11, 73 percent | 11/11, 100 percent | **11/11, 100 percent** |
| Project benchmark, 42 questions | 42/42, 100 percent | 41/42, 97.6 percent | **42/42, 100 percent** |
| Speed | about 100 ms per question, CPU, no GPU | same | 105 ms |
| Model calls | 0 LLM API calls, all local | same | same |

### Warm-starting is what made v12 work

v11 and v12 train on the same 3838 examples. The only difference is the starting point: v11
starts from the published base, v12 starts from v10's weights merged back into that base, so its
LoRA learns only a correction.

| | v11, from the base | v12, from v10 |
|---|---|---|
| Validation loss, final | 0.2977 | **0.2806** |
| Validation loss, epoch 1 | 0.4989 | **0.2877** |
| Holdout | 166/180 | **172/180** |
| Benchmark | 41/42 | **42/42** |

v12's first epoch is already better than v11's last, so this is a better minimum rather than the
same one reached twice. Validation loss was still falling at v12's final epoch, which means three
epochs was not enough and a longer warm run has room left.

The improvement over v10 is real by a paired test: on the cases where they disagree v12 wins 11
and loses 2, giving McNemar chi-square 4.92 and p 0.027. The v11 comparison was p 0.66, which is
why v11 was not shipped and v12 is.

v11's two regressions are both gone in v12. The entity-drop on `X ka total spend kitna hai?` falls
from two cases to one, and the confusable-pair over-refusal that cost the benchmark question does
not occur.

Nothing about this is a code path. `scripts/make_warmstart.py` merges an adapter into a base
checkpoint, `scripts/train_needle.py --checkpoint` trains from it, and `needle build` merges the
result onto the same warm base. The model is the same architecture and the same six tools.

## Known weaknesses

- `count_vendors` was the weakest tool at 73 percent, because the spend tool needs the word
  `total` as its main cue and `total` had to be kept out of the count training data to stop the
  two colliding. Questions like "total kitne suppliers hain" therefore routed to spend. The v11
  batch took the documented fix, more count data phrased without `total`, and took the tool to
  11/11. It holds in the shipped v12.
- Single-word capitals such as `INGEN` and `OSCORP` are where most of the remaining failures are:
  four of the eight v12 holdout failures are a lookup refused or mis-tooled on `INGEN`, `Nova
  Retail` and `Nova Retailers`. Capital coverage went from 2.3 to 8.2 percent of vendor mentions
  in the v10 batch, then up again in v11, and the real vendors are one third capitals, so more
  would still help.
- The `X ka total spend kitna hai?` phrasing can still drop the vendor and return the all-vendor
  sum. The v9 global-sum batch taught the model that `total spend` may stand alone, and on this
  one phrasing that outranks the entity. It is down to a single holdout case, `OSCORP pe kitna
  paisa kharch hua hai total?`, from two in v11.
- Local fine-tuning leaves the confidence head untrained, so `confidence` is `None` and the
  engine cannot be routed on it. Platform fine-tuning keeps the head.

## Tool triggers are available and deliberately unused

The engine honours a `triggers` list of case-insensitive regular expressions on each tool
schema, and a match narrows decoding to that tool. Measured directly, a trigger on
`count_vendors` for `\btotal spend\b` forces the wrong tool, so triggers do work. They are not
used, because the routing here is the model's contribution and a regex layer would replace it.
If a deployment needs deterministic routing over a fixed phrasing set, this is the lever.

