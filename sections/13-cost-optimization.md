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

### Intermediate Level

#### Q13.3: What are the main cost components of an LLM application and how do you optimize each?

**Expected Answer:**

**Cost Breakdown of a Typical LLM Application:**

| Component | % of Total Cost | Key Optimization |
|-----------|----------------|------------------|
| Input tokens | 40-60% | Prompt compression, caching |
| Output tokens | 20-30% | Response length control, structured output |
| Embedding costs | 10-15% | Batch calls, open-source models |
| Infrastructure | 10-20% | Right-sizing, auto-scaling, spot instances |
| Retrieval costs | 5-10% | Cache retrieval results |

**Optimizing Each Component:**

**1. Input Tokens (Largest Cost Driver):**
- **Prompt compression:** LLMLingua can compress prompts by up to 20x with minimal quality loss
- **Shorter system prompts:** remove verbose instructions, use concise formatting
- **Remove redundant context:** only include relevant retrieved documents, not everything
- **Prompt caching:** Anthropic prompt caching provides 90% savings on cached prefixes; OpenAI offers similar features
- **Dynamic context selection:** use a relevance threshold to include only high-scoring retrieved chunks

**2. Output Tokens:**
- **Constrain response length:** set `max_tokens` appropriately for each use case
- **Structured output:** JSON output is typically 30-50% fewer tokens than verbose natural language
- **Stop tokens:** configure stop sequences to prevent unnecessary generation
- **Instruction tuning:** instruct the model to be concise ("Answer in 2-3 sentences")

**3. Embedding Costs:**
- **Batch embedding calls:** embed multiple texts in a single API call instead of one at a time
- **Cache embeddings:** store computed embeddings — never re-embed the same text
- **Open-source models:** use models like `all-MiniLM-L6-v2` or `bge-small-en` locally — saves 100% of embedding API costs
- **Dimensionality reduction:** use Matryoshka embeddings to reduce vector size without losing much quality

**4. Infrastructure:**
- **Right-size GPU instances:** don't use an A100 when an L4 suffices for inference
- **Spot instances:** use for batch processing jobs (up to 90% savings)
- **Auto-scaling:** scale down during off-peak hours, scale up during peak
- **Serverless inference:** pay only for actual compute used (e.g., AWS Lambda, Modal)

**5. Retrieval Costs:**
- **Cache retrieval results:** if the same query returns the same documents, cache the results
- **Optimize index:** use HNSW parameters that balance recall with search cost
- **Reduce search scope:** use metadata filters to narrow the search space

**The Cost Equation:**
```
Total Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)
           + Infrastructure + Embedding Costs + Retrieval Costs
```

**Production Cost Example:**
- A typical RAG query: 2000 input tokens + 500 output tokens
- On GPT-4o: ~$0.02 per query
- At 100K queries/day = $2,000/day = **$60,000/month**
- With optimization (caching + routing + compression): $12,000-$18,000/month

> **Key insight:** Most teams overspend by 3-5x because they don't measure cost per query. The first step to optimization is instrumentation — you cannot optimize what you cannot measure.

---

### Advanced Level

#### Q13.4: What is prompt caching and how does it reduce costs in production?

**Expected Answer:**

**What is Prompt Caching?**
- Reuse computation from identical prompt prefixes across requests
- Instead of reprocessing the same system prompt + few-shot examples for every request, the provider caches the KV-cache for that prefix
- Subsequent requests with the same prefix skip the cached computation

**How It Works:**
```
Request 1:
[System Prompt (2000 tokens)] + [User Query A (100 tokens)]
→ Full computation: 2100 tokens processed
→ System prompt KV-cache stored

Request 2:
[System Prompt (2000 tokens)] + [User Query B (120 tokens)]
→ Cache hit on system prompt
→ Only 120 new tokens processed
→ 90% cost reduction on the cached prefix
```

**Provider Implementations:**

| Provider | Feature | Cost Reduction | Latency Reduction |
|----------|---------|---------------|-------------------|
| Anthropic | Prompt Caching | 90% on cached prefix | 85% on cached prefix |
| OpenAI | Cached Context | 50% on cached prefix | Variable |
| Google | Context Caching | 75% on cached prefix | Significant |

**When Prompt Caching Helps Most:**
- Long system prompts (>1000 tokens)
- Few-shot examples included in every request
- RAG applications with repeated context patterns
- Multi-turn conversations where context accumulates

**Implementation Best Practice — Structure Prompts for Maximum Caching:**
```
┌─────────────────────────────┐
│ System Prompt (STATIC)      │  ← Cacheable
│ + Few-shot Examples (STATIC)│  ← Cacheable
├─────────────────────────────┤
│ Retrieved Context (VARIES)  │  ← Sometimes cacheable
├─────────────────────────────┤
│ User Query (UNIQUE)         │  ← Never cached
└─────────────────────────────┘

Key: Put static content FIRST, dynamic content LAST
```

**The Cache Opportunity is Massive:**
- 31% of LLM queries in production exhibit semantic similarity — meaning nearly a third of all queries could benefit from some form of caching
- Combined with exact prefix caching, the savings compound significantly

**Prompt Caching vs Semantic Caching:**

| Aspect | Prompt Caching | Semantic Caching |
|--------|---------------|-----------------|
| Match type | Exact prefix match | Similar meaning match |
| Where it happens | Provider-side (automatic) | Application-side (you build it) |
| What's cached | KV-cache computation | Full response |
| Cost saving | On input token processing | Eliminates LLM call entirely |
| Latency saving | Reduces processing time | Eliminates LLM latency entirely |

**Combined Strategy for Maximum Savings:**
1. **Prompt caching** for prefix reuse — reduces cost of every LLM call
2. **Semantic caching** for similar queries — eliminates LLM calls entirely for repeated questions
3. Together: prompt caching handles the calls that must go to the LLM, semantic caching prevents calls that don't need to

> **Key insight:** Prompt caching is free performance. You get cost and latency reduction simply by structuring your prompts to maximize prefix reuse — static content first, dynamic content last.

---

### Expert Level

#### Q13.5: Design a cost monitoring and optimization system for an enterprise LLM deployment.

**Expected Answer:**

**1. Cost Attribution:**
- Track cost per team, per use case, per model, per query type
- Tag every request with: team ID, project ID, use case, model used, environment (dev/staging/prod)
- Enable granular analysis: "Which team spends the most? Which use case is least efficient?"

**2. Real-Time Dashboards:**

| Dashboard | Metrics | Audience |
|-----------|---------|----------|
| Operational | Cost/query, tokens/query, queries/minute | Engineering |
| Financial | Daily/weekly/monthly spend, cost by model, cost by team | Finance/Management |
| Efficiency | Cache hit rate, routing distribution, cost per successful query | Platform team |
| Business | Cost per conversation, cost per resolved ticket, ROI | Business stakeholders |

**3. Alerting System:**

| Alert Level | Condition | Action |
|-------------|-----------|--------|
| Critical | Daily cost > 3x rolling average | Page on-call, auto-throttle |
| Warning | Weekly cost > 1.5x budget | Notify team lead |
| Info | New use case exceeds $500/day | Review and optimize |
| Quota | Team approaching monthly budget | Notify team, suggest optimization |

- Anomaly detection: ML-based detection of sudden cost spikes (not just static thresholds)
- Per-team quotas: hard limits or soft warnings when teams approach budget

**4. Optimization Levers (In Order of Impact):**

| Priority | Lever | Expected Savings | Effort |
|----------|-------|------------------|--------|
| 1 | Caching (semantic + prompt) | 40-70% | Medium |
| 2 | Model routing (cheap models first) | 30-60% | Medium |
| 3 | Prompt optimization (compression) | 10-30% | Low |
| 4 | Response length control | 5-15% | Low |
| 5 | Batch processing for non-urgent | 10-20% | Medium |

**5. Automated Optimization:**
- **Smart routing model:** ML classifier that learns which queries need expensive models vs cheap ones, trained on historical quality scores
- **Auto-compression:** automatically apply prompt compression when cost exceeds thresholds
- **Adaptive caching:** dynamically adjust semantic cache similarity threshold based on quality feedback
- **Off-peak batching:** queue non-time-sensitive requests for off-peak processing at lower rates

**6. Chargeback Model:**
- Allocate costs to business units based on actual usage
- Monthly cost reports per team with breakdown by model and use case
- Incentivize efficiency: teams that reduce cost per query get more budget for experimentation
- Show cost-per-value metrics: cost per resolved support ticket, cost per generated report

**7. Capacity Planning:**
- Forecast costs based on growth projections and new use cases
- Model cost scenarios: "What if query volume doubles?" "What if we switch from GPT-4o to Claude Sonnet?"
- Negotiate volume discounts with providers based on committed usage
- Plan for provider diversification to avoid single-vendor pricing risk

**8. Governance:**
- Approval process for new LLM use cases (estimate cost before launching)
- Cost review for high-volume deployments before production rollout
- Regular optimization reviews: monthly for active projects, quarterly for the portfolio
- Sunset policy: decommission use cases where cost exceeds value

**Architecture:**
```
┌──────────────────────────────────────────────┐
│              LLM Application                 │
│  (Each request tagged with cost metadata)    │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│          Cost Tracking Middleware             │
│  (Capture: tokens, model, latency, team)     │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│           Cost Analytics Pipeline            │
│  (Aggregate, attribute, trend analysis)      │
└──────────┬──────────────┬────────────────────┘
           ↓              ↓
┌────────────────┐ ┌─────────────────┐
│   Dashboards   │ │   Alert Engine   │
│  (Grafana,     │ │  (PagerDuty,     │
│   Metabase)    │ │   Slack)         │
└────────────────┘ └─────────────────┘
```

**Expected Results:**
- Teams typically achieve **60-80% cost reduction** by implementing caching + routing + prompt optimization
- Payback period: 2-4 weeks for the monitoring system to identify savings that exceed its cost
- Cultural shift: teams become cost-aware and self-optimize when they see their spending

> **Key insight:** Cost optimization is an ongoing process, not a one-time project. The biggest savings come from model routing — 70% of queries can be handled by models that cost 10-20x less. But you cannot route intelligently without first measuring what you have.

---

---

[← Previous: Guardrails & Security](./12-guardrails-security.md) | [← Back to Main](../README.md) | [Next: Observability & Monitoring →](./14-observability-monitoring.md)
