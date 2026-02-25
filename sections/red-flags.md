[← Back to Main](../README.md) | [← Previous: System Design Questions](./16-system-design-questions.md) | [Next: Resources →](./resources.md)

---

# Red Flags Section

### Common Red Flags by Topic

**RAG:**
- ❌ Thinks retrieval is the easy part, focuses only on generation
- ❌ No mention of evaluation
- ❌ Suggests vector-only search without hybrid
- ❌ Ignores chunking as "just split by tokens"

**Multi-Agent:**
- ❌ More agents = better performance (ignores coordination tax)
- ❌ No mention of state management
- ❌ Doesn't consider error propagation
- ❌ No fallback or degradation strategy

**Deployment:**
- ❌ Thinks quantization always hurts quality significantly
- ❌ Ignores memory bandwidth bottlenecks
- ❌ No caching strategy
- ❌ Doesn't mention latency vs throughput trade-offs

**Evaluation:**
- ❌ Only knows BLEU/ROUGE
- ❌ Trusts LLM-as-Judge without caveats
- ❌ No mention of human evaluation
- ❌ Doesn't evaluate retrieval separately from generation

**Security:**
- ❌ Thinks prompt injection is easily solved
- ❌ No layered defense strategy
- ❌ Trusts LLM output without validation
- ❌ No mention of OWASP LLM Top 10

---

---

[← Previous: System Design Questions](./16-system-design-questions.md) | [← Back to Main](../README.md) | [Next: Resources →](./resources.md)
