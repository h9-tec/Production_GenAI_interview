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

---

[← Previous: Fine-tuning](./10-fine-tuning.md) | [← Back to Main](../README.md) | [Next: Guardrails & Security →](./12-guardrails-security.md)
