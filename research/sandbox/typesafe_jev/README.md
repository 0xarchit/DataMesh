# TypeSafe AI Jev Exploration: System-One Decision Modeling

This sandbox directory investigates **Jev**, a proprietary "System One" decision engine released by **TypeSafe AI** (September 2026), and evaluates its viability for high-speed, risk-calibrated entity reconciliation decisions.

---

## 1. What is TypeSafe AI Jev?

Founded by former OpenAI researcher and ChatGPT co-inventor Diogo Almeida, **TypeSafe AI** developed **Jev** as a specialized model designed explicitly for software automation rather than conversational text generation:

* **Non-Generative Decision Paradigm:** Instead of autoregressively generating free-form text or Markdown tokens, Jev executes a single-pass evaluation over structured options, functioning effectively as an *"AI-powered `if` statement"*.
* **Extreme Throughput & Cost Reduction:** Claims 20×–200× lower latency and 40×–400× lower token overhead compared to standard large generative models (e.g., GPT-4o, Claude 3.5 Sonnet, LLaMA 3) by completely eliminating the sequential token generation loop.
* **Three Deterministic Output Primitives:**
  1. `Choice`: Categorical selection from a fixed, predefined list of discrete options.
  2. `Score`: Quantitative ordinal or continuous rating against defined rubric levels.
  3. `Noul`: Calibrated binary probability assessment (Yes / No).
* **Calibrated Confidence Numbers:** Every decision output is paired with a mathematically calibrated confidence score, explicitly designed to enable automated escalation to human review queues.

---

## 2. Relevance to DataMesh AI

DataMesh AI's core innovation is the **Evidence-Aware Reconciliation and Decision Layer**, which must evaluate a composite 7-dimensional evidence vector for every candidate record pair and make a risk-sensitive determination:

```text
Evidence Vector [E1...E7] ──► Decision Engine ──► [ MERGE | REVIEW | REJECT | ABSTAIN ]
```

### Potential Applications in DataMesh:
1. **Low-Latency Blocking & Candidate Filtering:** Serving as an ultra-fast first-pass filter over millions of pairwise candidate entity comparisons.
2. **Four-Way Decision Mapping:** Utilizing Jev's `Choice` primitive over the candidate set `["MERGE", "REVIEW", "REJECT", "ABSTAIN"]`.
3. **Calibrated Confidence Routing:** Leveraging Jev's native confidence score to dynamically trigger human-in-the-loop review queues when confidence falls below operational thresholds ($\tau_{review}$).

---

## 3. Why Jev is Kept in `research/sandbox/`

> [!WARNING]
> **Academic Integrity & Reproducibility Notice:**
> * **Proprietary & Emerging:** Jev was released in September 2026 under a closed-access / waitlist API. Relying solely on a closed API poses reproducibility risks for an academic capstone evaluation.
> * **Scientific Grounding:** The primary theoretical baseline of DataMesh AI is rigorously grounded in peer-reviewed literature (**Hendrickx et al., *Machine Learning*, 2024** on selective classification with a reject option, evaluated with Brier scores, Expected Calibration Error [ECE], and risk-coverage AUC curves using open, reproducible classifiers like LightGBM, Logistic Regression, and Conformal Predictors).
> * **Sandbox Purpose:** This directory isolates experimental probes, prompt schemas, and mock interfaces for Jev so we can compare its decision speed and calibration against our open, reproducible classifiers without creating a project dependency.

---

## 4. Exploration Roadmap

```text
typesafe_jev/
├── README.md               # This specification
├── mock_jev_client.py      # Local mock simulator replicating single-pass option scoring
├── prompts/                # Structured decision prompts for MERGE/REVIEW/REJECT
└── evaluation/             # Benchmarks comparing Jev vs. LightGBM on MaDI-Bench
```

* [ ] Build a local Python simulator (`mock_jev_client.py`) that replicates single-pass softmax scoring over defined options.
* [ ] Format candidate pair evidence vectors into compact Jev prompts.
* [ ] Benchmark calibration reliability (ECE) of Jev's native confidence scores against isotonic regression and Platt scaling.
