[← Back to Main](../README.md) | [← Previous: Embeddings & Vector Databases](./03-embeddings-vector-databases.md) | [Next: Semantic Caching →](./05-semantic-caching.md)

---

# Section 4: Hybrid Search & Reranking

> **Key Stat**: Hybrid search (BM25 + vector) improves precision by 15-30% over vector-only search across enterprise deployments.

![Hybrid Search Pipeline](../diagrams/hybrid-search.svg)

### Intermediate Level

#### Q4.1: Why does pure vector search fail in production?

**Expected Answer:**

**Failure Modes of Vector-Only Search:**

1. **Exact Match Blindness**
   - User searches for "Error Code E-5023"
   - Vector search finds semantically similar errors
   - Misses the EXACT code they need

2. **Rare Term Ignorance**
   - Unique product names, IDs, acronyms
   - Not well represented in embedding space
   - BM25 handles these perfectly

3. **Negation Confusion**
   - "Not working" vs "Working"
   - Embeddings often ignore negation
   - Retrieves opposite of what's needed

4. **Length Bias**
   - Embeddings favor certain text lengths
   - Very short queries embed poorly
   - Very long documents compress too much

5. **Out-of-Distribution Queries**
   - Domain-specific jargon not in training
   - New terminology post-training cutoff
   - Embeddings are essentially random

**Evidence:** In a 2024 enterprise RAG benchmark:
- Vector only: 67% accuracy
- BM25 only: 61% accuracy
- Hybrid (vector + BM25): 82% accuracy

**Key insight:** BM25 and vector search fail on DIFFERENT queries. Combining them covers each other's weaknesses.

---

#### Q4.2: Explain Reciprocal Rank Fusion (RRF) and how to tune it.

**Expected Answer:**

**What is RRF:**
A method to combine rankings from multiple retrieval systems into a single ranked list.

**The Formula:**
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```
Where:
- k = constant (typically 60)
- rank_i = document's rank in system i

**Why it works:**
- Doesn't need score normalization
- Robust to different score scales
- Top-ranked documents get exponentially more weight
- Penalizes documents that appear in only one system

**Example:**
- Doc A: Rank 1 in vector, Rank 5 in BM25
- Doc B: Rank 3 in vector, Rank 2 in BM25
- With k=60:
  - RRF(A) = 1/61 + 1/65 = 0.0164 + 0.0154 = 0.0318
  - RRF(B) = 1/63 + 1/62 = 0.0159 + 0.0161 = 0.0320
- Doc B wins (appears high in both)

**Tuning Parameters:**

1. **k constant:**
   - Higher k: More equal weighting
   - Lower k: Top ranks dominate more
   - Start with k=60, tune based on evaluation

2. **Number of candidates:**
   - Retrieve top-N from each system
   - Typical: top-100 from each
   - More candidates = better recall, more compute

3. **System weights:**
   - Optional: Weight systems differently
   - If vector is more reliable: weight it higher
   - Determine through A/B testing

---

#### Q4.3: When does reranking hurt more than it helps?

**Expected Answer:**

**Reranking Costs:**
- Additional latency (100-500ms typically)
- Additional compute/API costs
- Complexity in the pipeline

**When Reranking Hurts:**

1. **Initial Retrieval is Already Poor**
   - Reranking can only reorder, not add new docs
   - If correct doc isn't in top-100, reranking won't find it
   - Garbage in → Garbage out

2. **Latency-Critical Applications**
   - Real-time autocomplete (need <50ms)
   - High-frequency queries (cost multiplies)
   - User patience is low

3. **Simple/Factoid Queries**
   - "What is the capital of France?"
   - First result is usually correct
   - Reranking adds cost without benefit

4. **Highly Curated Datasets**
   - Small, domain-specific corpus
   - Documents already highly relevant
   - Reranking may introduce noise

5. **Resource Constraints**
   - No GPU available for cross-encoder
   - API costs exceed budget
   - Simpler system preferred for maintenance

**When to Skip Reranking:**
- Retrieval evaluation shows >95% precision@10
- Latency budget is <200ms total
- Query volume is >100K/day with tight budget

**Key insight:** Reranking is high ROI when retrieval precision is 60-80%. Below 60%, fix retrieval first. Above 90%, reranking may not be needed.

---

### Advanced Level

#### Q4.4: Design a multi-stage retrieval pipeline. What goes in each stage?

**Expected Answer:**

**The Principle:**
Start fast and broad, progressively narrow and refine.

**Stage 1: Candidate Generation (Broad, Fast)**
- Goal: High recall, get ALL potentially relevant docs
- Methods: BM25, approximate nearest neighbor (ANN)
- Output: Top 500-1000 candidates
- Latency: <50ms

**Stage 2: First-Pass Ranking (Balance)**
- Goal: Remove obvious non-relevant, rough ordering
- Methods: Lightweight bi-encoder, fast cross-encoder
- Output: Top 50-100 candidates
- Latency: 50-100ms

**Stage 3: Reranking (Precise, Slow)**
- Goal: Optimal ordering of top candidates
- Methods: Heavy cross-encoder, LLM-based reranker
- Output: Top 10-20 candidates
- Latency: 100-300ms

**Stage 4: Final Selection (Domain Logic)**
- Goal: Apply business rules, diversity
- Methods: Deduplication, source diversity, recency boost
- Output: Top 3-5 for LLM context
- Latency: <10ms

**Example Configuration:**

| Stage | Method | Input | Output | Latency |
|-------|--------|-------|--------|---------|
| 1a | BM25 | Query | Top 500 | 20ms |
| 1b | HNSW vector | Query embedding | Top 500 | 30ms |
| 2 | RRF fusion + bi-encoder | 1000 candidates | Top 100 | 80ms |
| 3 | Cross-encoder | Top 100 | Top 20 | 200ms |
| 4 | Business rules | Top 20 | Top 5 | 5ms |

**Total latency: ~335ms**

---

---

[← Previous: Embeddings & Vector Databases](./03-embeddings-vector-databases.md) | [← Back to Main](../README.md) | [Next: Semantic Caching →](./05-semantic-caching.md)
