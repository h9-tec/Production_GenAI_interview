[← Back to Main](../README.md) | [← Previous: Cost Optimization](./13-cost-optimization.md) | [Next: LLM Reasoning Failures →](./15-llm-reasoning-failures.md)

---

# Section 14: Observability & Monitoring

> **Key Stat**: Teams with comprehensive LLM observability resolve production issues 5x faster than those without.

### Intermediate Level

#### Q14.1: What are the key metrics to monitor in an LLM application?

**Expected Answer:**

**Latency Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| TTFT | Time to First Token | <500ms |
| Total latency | Full response time | <3s |
| P50/P99 | Percentile latencies | Track trends |

**Quality Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Grounded in context | >0.9 |
| Answer relevance | Addresses question | >0.85 |
| Hallucination rate | % unsupported claims | <5% |
| User satisfaction | Thumbs up/down | >80% |

**Cost Metrics:**
| Metric | Description | Track |
|--------|-------------|-------|
| Tokens/query | Input + output tokens | Average, P99 |
| Cost/query | Dollar cost per query | Average |
| Daily/monthly spend | Total cost | Budget alerts |

**Operational Metrics:**
| Metric | Description | Alert On |
|--------|-------------|----------|
| Error rate | Failed requests | >1% |
| Timeout rate | Requests exceeding limit | >0.5% |
| Guardrail triggers | Security violations | Trend increase |
| Cache hit rate | Efficiency indicator | <50% |

**Retrieval-Specific:**
| Metric | Description |
|--------|-------------|
| Retrieval latency | Time to get docs |
| Docs retrieved | Number per query |
| Context utilization | % of context used |

---

#### Q14.2: Compare Langfuse, LangSmith, and Arize Phoenix for LLM observability.

**Expected Answer:**

| Aspect | Langfuse | LangSmith | Arize Phoenix |
|--------|----------|-----------|---------------|
| **Type** | Open-source | Commercial | Open-source |
| **Self-hosting** | Yes | No | Yes |
| **Tracing** | Full | Full | Full |
| **Evaluation** | Built-in | Built-in | Built-in |
| **Pricing** | Free/Cloud | Free tier + paid | Free |
| **Best for** | Self-hosted, privacy | LangChain users | Quick setup |

**Langfuse:**
- Pros: Open-source, self-hostable, good privacy
- Cons: Need to manage infrastructure
- Best for: Privacy-sensitive, want control

**LangSmith:**
- Pros: Deep LangChain integration, polished UI
- Cons: Cloud-only, LangChain ecosystem lock-in
- Best for: LangChain users, want managed service

**Arize Phoenix:**
- Pros: Single Docker container, fast setup
- Cons: Less mature, smaller community
- Best for: Quick prototyping, learning

**Recommendation:**
- Starting out: Arize Phoenix (simplest)
- Production with LangChain: LangSmith
- Production with privacy needs: Langfuse
- Enterprise: Datadog LLM Observability (if already using Datadog)

---

### Advanced Level

#### Q14.3: How do you implement distributed tracing for an LLM application?

**Expected Answer:**

**Why Tracing Matters:**
- LLM applications have many components: embedding, retrieval, reranking, generation, tool calls, output validation
- Without tracing, debugging is guesswork — you cannot tell which component caused a slow response or failure
- Tracing gives you a complete picture of every request's journey through the system

**Trace Structure:**
- Each request = 1 **trace** (end-to-end journey)
- Each component = 1 **span** (individual operation)
- Spans have parent-child relationships forming a tree
- The root span represents the entire request; child spans represent sub-operations

**What to Capture Per Span:**

| Field | Description | Example |
|-------|-------------|---------|
| Span name | Component identifier | `embed_query`, `vector_search`, `llm_generate` |
| Input | What went into the component | Query text, retrieved docs |
| Output | What came out | Embedding vector, LLM response |
| Latency | Time taken | 1500ms |
| Tokens used | Input + output tokens | 2000 + 500 |
| Model name | Which model was called | `gpt-4o`, `claude-sonnet-4-20250514` |
| Cost | Dollar cost of this span | $0.015 |
| Status | Success/failure | `success`, `error: timeout` |
| Metadata | Additional context | Cache hit/miss, retrieval count |

**OpenTelemetry Integration:**
- OpenTelemetry (OTel) is the standard protocol for distributed tracing
- As of March 2025, both Langfuse and LangSmith support OpenTelemetry natively
- This means you can instrument once with OTel and send traces to any compatible backend
- Benefit: no vendor lock-in, switch observability providers without re-instrumenting

**Trace Example for a RAG Query:**
```
Trace: "What is our refund policy?"
│
├── [Embed Query]         20ms    │ Model: text-embedding-3-small
│                                 │ Tokens: 8, Cost: $0.000001
│
├── [Vector Search]       50ms    │ Index: product-docs
│                                 │ Results: 5 chunks, Score: 0.89-0.95
│
├── [Rerank]             200ms    │ Model: cohere-rerank-v3
│                                 │ Input: 5 chunks, Output: 3 chunks
│
├── [Context Assembly]     5ms    │ Total context: 1200 tokens
│
├── [LLM Generation]   1500ms    │ Model: gpt-4o
│                                 │ Input: 1500 tokens, Output: 300 tokens
│                                 │ Cost: $0.012
│
├── [Output Validation]   10ms    │ PII check: clean
│                                 │ Guardrail: passed
│
Total:                  1785ms    │ Total cost: $0.012
```

**Key Dimensions to Trace:**
- **Latency breakdown:** which component is the bottleneck? (Usually LLM generation)
- **Token usage per component:** are you sending too much context? Are responses too long?
- **Error rates per component:** which component fails most often?
- **Cache hit/miss:** is caching working? What is the hit rate per component?

**Correlation Across Signals:**
- Link traces to **user sessions:** see all queries in a conversation
- Link traces to **feedback signals:** when a user gives thumbs-down, find the trace and diagnose why
- Link traces to **evaluation scores:** correlate low faithfulness scores with specific retrieval patterns

**Alerting on Trace Anomalies:**
- Unusual latency spikes: P99 latency exceeds 2x the rolling average
- Missing spans: a component in the pipeline did not execute (component failure)
- Token explosions: suddenly processing 10x more tokens than normal (possible injection or loop)
- Error cascades: one component failure causing downstream failures

> **Key insight:** Without tracing, debugging production LLM issues is like debugging without stack traces. It is the first observability capability to set up — before dashboards, before alerts, before everything else.

---

### Expert Level

#### Q14.4: Design an alerting strategy for a production LLM system. What alerts matter?

**Expected Answer:**

**Alert Tiers:**

**Tier 1 — Page Immediately (Wake someone up):**

| Alert | Threshold | Why It Matters |
|-------|-----------|----------------|
| Error rate | >5% of requests failing | System is broken for users |
| Latency P99 | >10 seconds | Unacceptable user experience |
| Cost spike | >3x normal daily rate | Runaway costs, possible abuse |
| Guardrail bypass | Any confirmed bypass | Security incident |
| Data breach indicators | PII detected in logs/responses | Legal and compliance risk |

**Tier 2 — Alert Within 1 Hour (Needs attention today):**

| Alert | Threshold | Why It Matters |
|-------|-----------|----------------|
| Faithfulness score | Drops below 0.8 average | Quality degradation, hallucinations increasing |
| Hallucination rate | >10% of responses | Users receiving incorrect information |
| Cache hit rate | Drops below 30% | Cost efficiency dropping, possible cache failure |
| Token usage spike | >2x average per query | Prompt bloat or context explosion |
| Model provider errors | >2% of calls failing | Provider issues, need fallback |

**Tier 3 — Daily Review (Track trends):**

| Alert | What to Review | Action |
|-------|---------------|--------|
| Gradual quality degradation | Faithfulness/relevance trending down over days | Investigate data drift, model updates |
| Cost trends | Week-over-week cost increase | Optimize or budget adjustment |
| User satisfaction | Thumbs-up ratio declining | Review recent changes, investigate queries |
| New query patterns | Queries outside training distribution | May need new data or guardrails |
| Cache efficiency | Semantic cache hit rate changes | Tune similarity thresholds |

**Anti-Patterns to Avoid:**

| Anti-Pattern | Problem | Solution |
|-------------|---------|----------|
| Alert on every metric | Alert fatigue — team ignores all alerts | Prioritize user-facing metrics only |
| No severity levels | Everything treated as equal urgency | Implement the 3-tier system above |
| No runbooks | Alert fires, nobody knows what to do | Write runbook for every Tier 1 and Tier 2 alert |
| Raw values instead of trends | Noisy alerts on normal variation | Alert on deviations from rolling averages |
| No deduplication | Same alert fires 100 times in 5 minutes | Group related alerts, suppress duplicates |

**Runbook Structure (Every Alert Should Have One):**
```
Alert: Faithfulness Score Below 0.8
Severity: Tier 2
Owner: ML Platform Team

1. CHECK: Is this affecting all queries or specific query types?
   → Run: query faithfulness by category in dashboard

2. TRIAGE: When did the drop start?
   → Check: recent deployments, model updates, data changes

3. DIAGNOSE: Common causes:
   - Retrieval quality degraded (check retrieval scores)
   - Context window overflow (check token counts)
   - Model provider changed behavior (check model version)
   - New data ingested with quality issues

4. MITIGATE:
   - If retrieval: rollback recent index changes
   - If model: switch to fallback model
   - If data: quarantine recent ingestion batch

5. RESOLVE: Fix root cause, verify scores recover, write post-mortem
```

**Dashboard Design:**

**Real-Time Operational View:**
- Requests/minute, error rate, latency (P50, P95, P99)
- Active alerts count by severity
- Model provider status (up/degraded/down)
- Cache hit rates (exact, semantic)

**Quality View:**
- Faithfulness, relevance, and hallucination rate trends (hourly, daily)
- User feedback distribution (thumbs up/down ratio)
- Guardrail trigger rate and types
- Top failing query categories

**Business View:**
- Queries per day, unique users
- Cost per conversation, cost per resolved issue
- User satisfaction score (weekly trend)
- ROI metrics: time saved, tickets deflected

**Integration:**
- **PagerDuty / OpsGenie:** for Tier 1 pages with escalation policies
- **Slack:** for Tier 2 alerts to team channels
- **Email / Weekly reports:** for Tier 3 trends to leadership
- **Incident management:** auto-create incidents for Tier 1 alerts

> **Key insight:** The goal is not to catch every issue — it is to catch issues that affect users before users notice. Focus on user-facing metrics first (latency, errors, quality), infrastructure metrics second. An alert that nobody acts on is worse than no alert at all.

---

---

[← Previous: Cost Optimization](./13-cost-optimization.md) | [← Back to Main](../README.md) | [Next: LLM Reasoning Failures →](./15-llm-reasoning-failures.md)
