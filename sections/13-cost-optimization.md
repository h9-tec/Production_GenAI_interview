[← Back to Main](../README.md) | [← Previous: Guardrails & Security](./12-guardrails-security.md) | [Next: Observability & Monitoring →](./14-observability-monitoring.md)

---

# Section 13: Cost Optimization

> **Key Stat**: Teams using model routing + caching report 60-80% cost reduction compared to using frontier models for everything.

### Advanced Level

#### Q13.1: Design a model routing strategy that balances cost and quality.

**Expected Answer:**

**The Principle:**
Use expensive models only when necessary.

**Routing Tiers:**

| Tier | Model | Use For | Cost |
|------|-------|---------|------|
| 1 | GPT-4 mini / Haiku | Classification, simple Q&A | $$ |
| 2 | GPT-4o / Sonnet | Standard analysis | $$$ |
| 3 | GPT-4 / Opus | Complex reasoning | $$$$ |

**Routing Logic:**

**1. Query Classification**
- Classify query complexity
- Use lightweight model for classification
- Categories: Simple, Standard, Complex

**2. Confidence-Based Escalation**
- Start with cheapest model
- If confidence low, escalate to next tier
- Stop when confidence threshold met

**3. Task-Type Routing**
- Classification/tagging → Tier 1
- Summarization → Tier 2
- Reasoning/analysis → Tier 3

**Implementation:**
```
1. Classify query (Tier 1 model)
2. If simple:
   - Answer with Tier 1
   - Return if confidence > 0.9
3. If standard or low confidence:
   - Answer with Tier 2
   - Return if confidence > 0.85
4. If complex or low confidence:
   - Answer with Tier 3
```

**Expected Savings:**
- 70% of queries handled by Tier 1
- 25% by Tier 2
- 5% by Tier 3
- Overall cost: 60-70% less than Tier 3 for all

---

#### Q13.2: How did semantic caching reduce a client's costs from $52K to $4.8K monthly?

**Expected Answer:**

**The Scenario:**
- Enterprise RAG system
- 50K+ daily queries
- Using GPT-4 for all responses
- $52K/month in API costs

**Analysis:**
- 70% of queries were variations of same questions
- "What's the refund policy?" in 50 different phrasings
- Each paid full LLM cost despite same answer

**Solution: Multi-Layer Caching**

**Layer 1: Exact Match Cache**
- Hash of query → response
- Hit rate: ~15%
- Latency: <10ms

**Layer 2: Semantic Cache**
- Query embedding → similar cached query
- Threshold: 0.92 similarity
- Hit rate: ~55%
- Latency: ~50ms

**Layer 3: Retrieval Cache**
- Cache retrieval results separately
- Same docs? Reuse context
- Hit rate: ~20%

**Results:**
- Total cache hit: ~70% of queries
- 70% queries: $0 (cached)
- 30% queries: Full cost

**Cost Calculation:**
- Before: 50K queries × $1.04 avg = $52K
- After: 15K queries × $1.04 = $15.6K
- Additional savings from model routing: $10.8K reduction
- Final: ~$4.8K/month

**Additional Optimizations:**
- Model routing for non-cached queries
- Prompt optimization (fewer tokens)
- Response length limits

**Key insight:** Caching addresses the fact that real-world query distributions are highly skewed. A small number of question types dominate.

---

---

[← Previous: Guardrails & Security](./12-guardrails-security.md) | [← Back to Main](../README.md) | [Next: Observability & Monitoring →](./14-observability-monitoring.md)
