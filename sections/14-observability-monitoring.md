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

---

[← Previous: Cost Optimization](./13-cost-optimization.md) | [← Back to Main](../README.md) | [Next: LLM Reasoning Failures →](./15-llm-reasoning-failures.md)
