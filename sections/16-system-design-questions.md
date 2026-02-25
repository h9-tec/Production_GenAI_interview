[← Back to Main](../README.md) | [← Previous: LLM Reasoning Failures](./15-llm-reasoning-failures.md) | [Next: Red Flags →](./red-flags.md)

---

# Section 16: System Design Questions

> These questions test end-to-end thinking. Expect 30-45 minutes per question in interviews.

### Expert Level

#### Q16.1: Design a RAG system for a legal firm with 1 million documents and strict accuracy requirements.

**Key Points to Cover:**

1. **Requirements Gathering**
   - Latency: <5s for most queries
   - Accuracy: >95% faithfulness required
   - Scale: 100 concurrent users
   - Security: Client confidentiality critical

2. **Document Processing**
   - Multi-format ingestion (PDF, Word, scans)
   - OCR for older documents
   - Structure extraction (sections, clauses)
   - Hierarchical chunking respecting legal structure

3. **Retrieval Architecture**
   - Hybrid search (BM25 for legal terms + vector)
   - Legal-domain fine-tuned embeddings
   - Multi-stage: Broad → Precise → Rerank
   - Citation extraction and verification

4. **Quality Assurance**
   - Human review for high-stakes queries
   - Confidence scoring with thresholds
   - Hallucination detection on every response
   - Audit trail for all interactions

5. **Security**
   - Client isolation (multi-tenancy)
   - Access control per document
   - PII detection and handling
   - Encryption at rest and in transit

---

#### Q16.2: Design a multi-agent customer support system that handles 10K queries per day.

**Key Points to Cover:**

1. **Agent Design**
   - Router agent: Classifies and routes queries
   - FAQ agent: Handles common questions (cached)
   - Technical agent: Product-specific issues
   - Escalation agent: Detects need for human
   - Human handoff interface

2. **Orchestration**
   - Orchestrator-worker pattern
   - Parallel processing where possible
   - Session state management
   - Graceful degradation

3. **Cost Optimization**
   - Tiered model usage
   - Semantic caching (expect 60%+ hit rate)
   - Fast path for simple queries
   - Batch processing for non-urgent

4. **Quality & Safety**
   - Guardrails on all responses
   - Sentiment detection
   - Escalation triggers
   - Customer satisfaction tracking

5. **Scalability**
   - Horizontal scaling of agents
   - Queue management for peak loads
   - Rate limiting per customer
   - Circuit breakers for downstream services

---

#### Q16.3: Design semantic caching for a government system with 50K concurrent users during peak events.

**Key Points to Cover:**

1. **Challenge Analysis**
   - Synchronized traffic (e.g., prayer times, announcements)
   - Dialect variations (regional Arabic)
   - Same questions in many forms
   - Latency critical during peaks

2. **Architecture**
   - Distributed cache (Redis cluster)
   - Semantic clustering layer
   - Dialect normalization pipeline
   - Pre-warming based on patterns

3. **Cache Strategy**
   - Exact match: First check
   - Semantic similarity: Second check
   - Cluster-based: Novel query to cluster
   - TTL + event-based invalidation

4. **Peak Handling**
   - Pre-compute likely queries
   - Analyze yesterday's patterns
   - Pre-warm cache before predictable peaks
   - Graceful degradation (stale OK briefly)

5. **Monitoring**
   - Hit rate by time of day
   - Cache size and eviction rate
   - Latency at each layer
   - False positive rate for semantic match

---

---

[← Previous: LLM Reasoning Failures](./15-llm-reasoning-failures.md) | [← Back to Main](../README.md) | [Next: Red Flags →](./red-flags.md)
