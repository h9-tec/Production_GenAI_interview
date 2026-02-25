[← Back to Main](../README.md) | [← Previous: Hybrid Search & Reranking](./04-hybrid-search-reranking.md) | [Next: Multi-Agent Systems →](./06-multi-agent-systems.md)

---

# Section 5: Semantic Caching

> **Key Stat**: Semantic caching can reduce LLM costs by 50-70% and improve latency by 60% for repetitive query patterns.

![Semantic Caching Architecture](../diagrams/semantic-caching.svg)

### Intermediate Level

#### Q5.1: What's the difference between exact-match caching and semantic caching?

**Expected Answer:**

**Exact-Match Caching:**
- Cache key = exact query string
- "What is RAG?" hits cache
- "What's RAG?" misses cache (different string)
- Simple to implement
- Low hit rate for natural language

**Semantic Caching:**
- Cache key = query embedding
- Find similar cached queries by vector similarity
- "What is RAG?" and "Explain RAG" may share cache
- Higher hit rate
- More complex implementation

**Comparison:**

| Aspect | Exact-Match | Semantic |
|--------|-------------|----------|
| Implementation | Simple hash lookup | Vector similarity search |
| Hit rate | Low (5-15%) | High (40-70%) |
| False positives | None | Possible |
| Storage | Query → Response | Query + Embedding → Response |
| Latency overhead | Negligible | 5-20ms for similarity search |

**When to use which:**
- Exact-match: API responses, deterministic queries
- Semantic: Natural language, chatbots, search queries
- Both: Exact-match first, then semantic if miss

---

#### Q5.2: How do you choose the right similarity threshold for semantic cache?

**Expected Answer:**

**The Trade-off:**
- High threshold (0.95+): Few hits, high accuracy
- Low threshold (0.80): Many hits, risk of wrong answers

**Factors Affecting Threshold:**

1. **Risk Tolerance**
   - Customer support: Can afford some errors → 0.85
   - Medical/Legal: Must be accurate → 0.95+
   - Internal tools: Moderate → 0.90

2. **Query Diversity**
   - Similar queries only: Lower threshold OK
   - Diverse queries: Need higher threshold

3. **Answer Sensitivity**
   - Generic answers: Lower threshold OK
   - Specific answers: Higher threshold needed

**How to Determine Threshold:**

1. **Collect Query Pairs**
   - Get 1000+ query pairs from logs
   - Label: "Same intent" or "Different intent"

2. **Compute Similarities**
   - Embed all queries
   - Calculate pairwise similarities

3. **Find Optimal Threshold**
   - Plot precision/recall at different thresholds
   - Choose threshold that maximizes F1
   - Or: Set minimum precision, maximize recall

**Typical Ranges:**
- Conservative: 0.92-0.95
- Balanced: 0.88-0.92
- Aggressive: 0.82-0.88

**Key insight:** Start conservative (0.92), monitor false positives, lower gradually.

---

### Advanced Level

#### Q5.3: How would you implement semantic caching for a system with dialect variations (e.g., Arabic)?

**Expected Answer:**

**The Challenge:**
- "إجازة رمضان" vs "عطلة الصيام" = Same intent
- Saudi vs Egyptian phrasing differs
- Standard similarity may not capture this

**Architecture: Semantic Clustering Cache**

**Component 1: Dialect Normalizer**
- Pre-process queries before caching
- Map dialect variations to canonical form
- Use lightweight dialect detection model
- Normalize common variations

**Component 2: Query Embedding**
- Use Arabic-optimized embedding model
- Options: AraBERT, CAMeLBERT, multilingual-e5
- May need domain fine-tuning

**Component 3: Cluster Management**
- Group similar queries into clusters
- Store cluster centroid as cache key
- New query → Find nearest cluster

**Component 4: Cache Logic**
```
1. Normalize query (dialect → standard)
2. Embed normalized query
3. Find nearest cluster centroid
4. If similarity > threshold:
   - Return cached response
5. Else:
   - Execute full pipeline
   - Update/create cluster
```

**Optimization: Pre-warming**
- Analyze yesterday's query patterns
- Pre-generate embeddings for likely queries
- Especially for predictable spikes (prayer times, Ramadan)

**Results from MENA deployment:**
- 94% cache hit rate during peak
- 70% GPU reduction
- <2s response time (from 8s)

---

#### Q5.4: How do you handle cache invalidation when source documents change?

**Expected Answer:**

**The Problem:**
Cached answers become stale when underlying data changes.

**Invalidation Strategies:**

**1. Time-Based Expiration (TTL)**
- Set maximum cache lifetime
- Simple but wasteful (valid caches expire too)
- Good for: Frequently changing data

**2. Document-Triggered Invalidation**
- Track which documents each cached answer used
- When document updates → Invalidate related caches
- Requires: Document → Cache mapping

**3. Selective Invalidation**
- Not all document changes matter
- Track specific entities/facts in cache
- Only invalidate if relevant facts changed
- Requires: Entity extraction from answers

**4. Versioned Caching**
- Include document version in cache key
- New version = automatic cache miss
- Simple but increases storage

**Implementation Approach:**

**For RAG systems:**
```
Cache entry:
- Query embedding
- Response
- Document IDs used
- Document versions used
- Timestamp

Invalidation:
- Document update triggers
- Find caches using that document
- Mark as invalid or delete
```

**Hybrid Strategy (Recommended):**
- TTL: 24 hours maximum
- Document-triggered: Immediate for changed docs
- Soft expiration: Serve stale + async refresh

**Key insight:** Perfect invalidation is often not worth the complexity. Bounded staleness (TTL) with critical-path invalidation covers 95% of needs.

---

---

[← Previous: Hybrid Search & Reranking](./04-hybrid-search-reranking.md) | [← Back to Main](../README.md) | [Next: Multi-Agent Systems →](./06-multi-agent-systems.md)
