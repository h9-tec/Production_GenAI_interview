[← Back to Main](../README.md) | [← Previous: Fine-tuning](./10-fine-tuning.md) | [Next: Guardrails & Security →](./12-guardrails-security.md)

---

# Section 11: Evaluation & Metrics

> **Key Stat**: Only 30% of RAG deployments in 2024 included systematic evaluation from day 1. By 2025, this reached 60%.

### Intermediate Level

#### Q11.1: What are the core RAG evaluation metrics and what do they measure?

**Expected Answer:**

**Retrieval Metrics:**

| Metric | Measures | Formula |
|--------|----------|---------|
| **Recall@k** | Are relevant docs in top-k? | (Relevant in top-k) / (Total relevant) |
| **Precision@k** | How many top-k are relevant? | (Relevant in top-k) / k |
| **MRR** | Rank of first relevant doc | Mean of 1/rank |
| **nDCG** | Graded relevance with position | Complex, handles degrees of relevance |

**Generation Metrics:**

| Metric | Measures |
|--------|----------|
| **Faithfulness** | Is answer grounded in context? |
| **Answer Relevance** | Does answer address the question? |
| **Context Relevance** | Is retrieved context useful? |
| **Completeness** | Does answer cover all aspects? |

**End-to-End Metrics:**

| Metric | Measures |
|--------|----------|
| **Answer Correctness** | Is the final answer right? |
| **Hallucination Rate** | % of unsupported claims |
| **Citation Accuracy** | Do citations support claims? |

**RAGAS Framework Metrics:**
1. Faithfulness: Answer supported by context?
2. Answer Relevance: Answer addresses question?
3. Context Precision: Relevant items ranked higher?
4. Context Recall: All relevant info retrieved?

---

#### Q11.2: What are the limitations of LLM-as-Judge evaluation?

**Expected Answer:**

**The Appeal:**
- Scales to thousands of evaluations
- Cheaper than human evaluation
- Consistent criteria application
- Fast iteration

**The Limitations:**

**1. Position Bias**
- LLMs prefer first or last options
- Affects pairwise comparisons
- Mitigation: Randomize order, average

**2. Verbosity Bias**
- Longer responses often rated higher
- Regardless of actual quality
- Mitigation: Normalize for length

**3. Self-Enhancement Bias**
- GPT-4 rates GPT-4 outputs higher
- Same for Claude rating Claude
- Mitigation: Use different model as judge

**4. Prompt Sensitivity**
- Small prompt changes → Different judgments
- Criteria interpretation varies
- Mitigation: Test prompt stability

**5. Lacks Domain Expertise**
- Can't verify factual accuracy
- Medical, legal, scientific domains problematic
- Mitigation: Combine with domain expert review

**6. Black Box Reasoning**
- Hard to know WHY a rating was given
- Explanations may be post-hoc rationalizations
- Mitigation: Require reasoning, check consistency

**Best Practices:**
- Use for directional signal, not absolute truth
- Always validate with human eval subset
- Use for relative ranking, not absolute scores
- Combine multiple judges/prompts
- Include calibration examples

**Key insight:** LLM-as-Judge is a tool for scaling, not a replacement for human judgment on critical decisions.

---

### Advanced Level

#### Q11.3: How do you detect hallucinations in RAG responses?

**Expected Answer:**

**Types of Hallucinations in RAG:**

1. **Intrinsic**: Contradicts retrieved context
2. **Extrinsic**: Claims not in context (but might be true)
3. **Fabricated**: Made-up facts, citations, numbers

**Detection Methods:**

**1. Entailment-Based**
- Check if context entails each claim
- Use NLI model or LLM
- Flag claims not supported

**2. Claim Decomposition**
- Break response into atomic claims
- Verify each claim against context
- More granular than whole-response check

**3. Citation Verification**
- Extract citations from response
- Check if cited source supports claim
- Detect fabricated references

**4. Confidence Scoring**
- LLM self-reports confidence
- Low confidence = Higher hallucination risk
- Calibrate threshold empirically

**5. Cross-Reference Checking**
- Generate multiple responses
- Compare for consistency
- Inconsistency suggests hallucination

**Implementation Approach:**

**Step 1: Extract Claims**
- Use LLM to break response into claims
- "Response contains 5 factual claims"

**Step 2: Match to Sources**
- For each claim, find supporting context
- Use semantic similarity

**Step 3: Verify Support**
- For each claim-context pair
- Does context support claim?
- Yes/No/Partial

**Step 4: Score and Flag**
- Calculate % supported claims
- Flag response if below threshold
- Surface unsupported claims

**Key insight:** Hallucination detection is an ongoing process, not a one-time check. Build it into your pipeline.

---

### Fresh Level

#### Q11.4: What is the difference between offline and online evaluation for LLM applications?

**Expected Answer:**

**Offline Evaluation (Before Deployment):**
- Run against a fixed test set (golden dataset)
- Automated metrics (RAGAS, BLEU, ROUGE)
- LLM-as-Judge scoring
- Regression testing against previous versions
- Repeatable, consistent, cheap

**Online Evaluation (In Production):**
- Real user interactions
- User feedback (thumbs up/down, ratings)
- Implicit signals (follow-up questions, task completion)
- A/B testing between model versions
- Real-world distribution of queries

**Comparison:**

| Aspect | Offline | Online |
|--------|---------|--------|
| Speed | Fast (minutes to hours) | Slow (days to weeks) |
| Cost | Low | Higher (production traffic) |
| Realism | Limited by test set | Real user behavior |
| Coverage | Your test set only | Full query distribution |
| Risk | None | Potential user impact |

**Why You Need Both:**
- Offline catches known regressions quickly
- Online reveals unknown failure modes
- Offline doesn't capture query distribution shifts
- Online can't test rarely-seen edge cases

**Recommended Workflow:**
1. Offline eval → Gate for deployment
2. Shadow deployment → Compare against production
3. Canary deployment → 5% of traffic
4. Full deployment → Monitor online metrics
5. Continuous offline eval → Catch regressions

---

### Intermediate Level

#### Q11.5: How do you build a golden dataset for RAG evaluation? What makes a good one?

**Expected Answer:**

**What is a Golden Dataset:**
A curated set of (question, expected_answer, relevant_documents) triples used to evaluate RAG systems consistently.

**Building Process:**

**Step 1: Query Collection (200-500+ queries)**
- Production logs (most representative)
- Domain expert questions
- Edge cases and failure modes
- Adversarial queries

**Step 2: Answer Annotation**
- Domain experts write expected answers
- Include: Full answer, key facts, acceptable variations
- Multiple annotators for quality (inter-annotator agreement)
- Document which source documents support each answer

**Step 3: Relevance Labeling**
- For each query, label which documents are relevant
- Use graded relevance: Not relevant, Partially, Highly
- Enables both retrieval and generation evaluation

**What Makes a Good Golden Dataset:**

| Property | Why It Matters |
|----------|---------------|
| Representative | Matches real query distribution |
| Diverse | Covers all difficulty levels and topics |
| Fresh | Updated as knowledge base changes |
| Size | 200+ for statistical significance |
| Multi-annotator | Reduces individual bias |
| Versioned | Track changes over time |

**Common Mistakes:**
- Too small (<50 queries) — not statistically meaningful
- Not updated when knowledge base changes — stale evaluations
- Only "happy path" queries — misses failure modes
- Single annotator — introduces personal bias
- No edge cases — overestimates production quality

**Maintenance:**
- Review quarterly
- Add queries from production failures
- Remove queries for deleted content
- Track metric trends across versions

**Key insight:** A golden dataset is a living artifact, not a one-time creation. Budget ongoing effort for maintenance.

---

### Expert Level

#### Q11.6: Design an automated evaluation pipeline that runs on every deployment.

**Expected Answer:**

**Pipeline Architecture:**

**Stage 1: Pre-Deployment Gates**
```
Code merge → Trigger eval pipeline
                    ↓
            Run golden dataset (500 queries)
                    ↓
            Calculate metrics:
            - Faithfulness (target: >0.9)
            - Answer relevance (target: >0.85)
            - Context precision (target: >0.8)
            - Hallucination rate (target: <5%)
                    ↓
            Compare against baseline (previous version)
                    ↓
            Pass/Fail gate (regression threshold: 2%)
```

**Stage 2: Regression Detection**

| Metric | Baseline | Current | Status |
|--------|----------|---------|--------|
| Faithfulness | 0.91 | 0.93 | PASS |
| Relevance | 0.87 | 0.85 | WARN (-2.3%) |
| Hallucination | 3.2% | 2.8% | PASS |
| Latency P99 | 2.8s | 4.1s | FAIL (+46%) |

**Stage 3: Slice Analysis**
- Break down metrics by query category
- Identify if specific topics regressed
- Separate retrieval vs generation failures
- Flag new failure patterns

**Stage 4: Human-in-the-Loop**
- Auto-sample 20 responses with lowest scores
- Route to domain experts for review
- Block deployment if critical failures found

**Stage 5: Canary Deployment**
- Deploy to 5% of production traffic
- Compare live metrics vs control
- Auto-rollback if quality drops >3%
- Graduate to 25% → 50% → 100%

**Tools:**
- RAGAS / DeepEval for metrics
- GitHub Actions / Jenkins for CI/CD
- Langfuse / LangSmith for production tracing
- Custom dashboards for trend tracking

**Key insight:** Treat LLM evaluation like software testing — automate, gate deployments, and never ship without measuring.

---

---

[← Previous: Fine-tuning](./10-fine-tuning.md) | [← Back to Main](../README.md) | [Next: Guardrails & Security →](./12-guardrails-security.md)
