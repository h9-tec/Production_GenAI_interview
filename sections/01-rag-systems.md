[← Back to Main](../README.md) | [Next: Chunking Strategies →](./02-chunking-strategies.md)

---

# Section 1: RAG Systems

> **Key Stat**: 90% of agentic RAG projects failed in production in 2024 — not because the technology was broken, but because engineers underestimated compounding failures.

![RAG Architecture](../diagrams/rag-architecture.svg)

### Fresh Level

#### Q1.1: What is RAG and why do we need it?

**Expected Answer:**

RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. Instead of relying solely on what an LLM learned during training, RAG fetches relevant documents from an external knowledge base and includes them in the prompt.

**Why RAG matters:**
- LLMs have knowledge cutoff dates — they don't know recent events
- LLMs hallucinate when they don't know something
- Fine-tuning is expensive and doesn't scale for frequently changing data
- RAG provides citations and traceability

**The basic flow:**
1. User asks a question
2. System converts question to embedding
3. Vector database finds similar documents
4. Retrieved documents + question go to LLM
5. LLM generates answer grounded in the documents

---

#### Q1.2: When should you use RAG vs Fine-tuning?

**Expected Answer:**

| Scenario | RAG | Fine-tuning |
|----------|-----|-------------|
| Data changes frequently | ✅ | ❌ |
| Need citations/sources | ✅ | ❌ |
| Domain-specific terminology | ✅ | ✅ |
| Specific output format/style | ❌ | ✅ |
| Limited training data (<1000 examples) | ✅ | ❌ |
| Latency-critical applications | ❌ | ✅ |
| Budget constraints | ✅ | ❌ |

**Rule of thumb:** Start with RAG. Only fine-tune when RAG + prompt engineering isn't enough.

---

#### Q1.3: What are the main components of a RAG pipeline?

**Expected Answer:**

1. **Document Ingestion** — Loading and parsing documents (PDF, HTML, etc.)
2. **Chunking** — Splitting documents into smaller pieces
3. **Embedding** — Converting chunks to vector representations
4. **Indexing** — Storing vectors in a vector database
5. **Retrieval** — Finding relevant chunks for a query
6. **Reranking** — Reordering results by relevance (optional but recommended)
7. **Context Assembly** — Combining retrieved chunks into a prompt
8. **Generation** — LLM produces the final answer
9. **Post-processing** — Validation, formatting, citation extraction

---

### Intermediate Level

#### Q1.4: Explain Self-RAG, CRAG, and Corrective RAG. When would you use each?

**Expected Answer:**

**Self-RAG (Self-Reflective RAG):**
- LLM decides whether retrieval is needed for each query
- After retrieval, LLM evaluates if retrieved docs are relevant
- Can regenerate if initial response is unsatisfactory
- Use when: Query types vary widely (some need retrieval, some don't)

**CRAG (Corrective RAG):**
- Adds a "retrieval evaluator" that scores document relevance
- If relevance is low, triggers web search or alternative retrieval
- Includes knowledge refinement step before generation
- Use when: Knowledge base may be incomplete

**Corrective RAG:**
- Focuses on detecting and correcting errors in generated responses
- Uses fact-checking against retrieved documents
- May iterate multiple times until response is verified
- Use when: Accuracy is critical (legal, medical, financial)

**Key insight:** These aren't mutually exclusive — production systems often combine elements from all three.

---

#### Q1.5: What is Agentic RAG and how does it differ from naive RAG?

**Expected Answer:**

**Naive RAG:**
- Single retrieval → Single generation
- No iteration or self-correction
- Query goes in, answer comes out

**Agentic RAG:**
- LLM acts as an agent that can:
  - Decide IF retrieval is needed
  - Choose WHICH retrieval source to use
  - Determine if MORE retrieval is needed
  - REFORMULATE queries if results are poor
  - VERIFY answers against sources

**Key differences:**

| Aspect | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| Retrieval decisions | Fixed | Dynamic |
| Query reformulation | None | Automatic |
| Multi-hop reasoning | Limited | Supported |
| Source selection | Single | Multiple |
| Self-correction | None | Built-in |

**When to use Agentic RAG:**
- Complex questions requiring multiple sources
- Questions that may need clarification
- When retrieval quality varies significantly

---

#### Q1.6: How do you handle multi-hop questions in RAG?

**Expected Answer:**

Multi-hop questions require information from multiple documents that must be combined. Example: "What is the GDP per capita of the country where the Eiffel Tower is located?"

**Strategies:**

1. **Query Decomposition**
   - Break complex query into sub-queries
   - "Where is the Eiffel Tower?" → France
   - "What is France's GDP per capita?"
   - Combine answers

2. **Iterative Retrieval**
   - First retrieval gets partial information
   - Use partial answer to form new query
   - Continue until answer is complete

3. **Graph-based RAG (GraphRAG)**
   - Build knowledge graph from documents
   - Traverse graph to connect related entities
   - Better for relationship-heavy queries

4. **Chain-of-Thought Retrieval**
   - LLM generates reasoning steps
   - Each step triggers targeted retrieval
   - Final answer synthesizes all retrievals

**Red Flag:** Candidate suggests single retrieval can handle multi-hop questions.

---

### Advanced Level

#### Q1.7: Why do 80% of RAG failures trace back to chunking decisions?

**Expected Answer:**

Chunking is the foundation of RAG quality. Poor chunking creates problems that no amount of retrieval optimization can fix.

**How chunking causes failures:**

1. **Context Fragmentation**
   - Fixed-size chunks split sentences mid-thought
   - Related information ends up in different chunks
   - Legal documents lose critical cross-references

2. **Semantic Boundary Violations**
   - A paragraph about Topic A gets merged with Topic B
   - Retrieval returns irrelevant content alongside relevant
   - LLM gets confused by mixed contexts

3. **Information Density Mismatch**
   - Dense technical content needs smaller chunks
   - Narrative content needs larger chunks
   - One-size-fits-all fails for diverse corpora

4. **Metadata Loss**
   - Chunks lose information about source, section, date
   - Can't filter or boost by document attributes
   - No way to trace answers back to sources

**Evidence:** A 2025 CDC policy RAG study found:
- Naive (fixed-size) chunking: Faithfulness score 0.47–0.51
- Optimized semantic chunking: Faithfulness score 0.79–0.82

**Key insight:** Invest 80% of optimization effort in chunking and retrieval, 20% in generation.

---

#### Q1.8: Explain the "compounding error problem" in RAG. How do you mitigate it?

**Expected Answer:**

**The Problem:**

RAG pipelines have multiple components, each with its own accuracy. Errors don't add — they multiply.

Example with 5 components at 95% accuracy each:
- Overall reliability: 0.95 × 0.95 × 0.95 × 0.95 × 0.95 = 77%

**Where errors compound:**

1. **Chunking** → Poor chunks mean relevant info is missing
2. **Embedding** → Wrong vectors mean wrong documents retrieved
3. **Retrieval** → Irrelevant docs pollute the context
4. **Reranking** → Bad reranking pushes good docs down
5. **Generation** → LLM hallucinates based on bad context

**Mitigation Strategies:**

1. **Error Isolation**
   - Each component should fail gracefully
   - Fallback paths for each failure mode
   - Don't pass obviously bad results downstream

2. **Validation Gates**
   - Check retrieval relevance before generation
   - Verify generation is grounded in context
   - Confidence scoring at each stage

3. **Redundancy**
   - Multiple retrieval strategies (hybrid search)
   - Multiple embedding models (ensemble)
   - Multiple rerankers with voting

4. **Observability**
   - Log inputs/outputs at every stage
   - Track accuracy metrics per component
   - Identify which stage degrades first

**Key insight:** The gap between demo and production is about making failures not compound, not making each component perfect.

---

### Expert Level

#### Q1.9: Design a RAG system for 1 million legal documents with 40 years of regulatory history. What are your key architecture decisions?

**Expected Answer:**

**Challenge Analysis:**
- Volume: 1M docs × ~50 pages avg = 50M pages
- Heterogeneity: OCR quality varies (1980s scans vs modern PDFs)
- Terminology: Legal language evolved over 40 years
- Precision: Legal domain requires near-perfect accuracy
- Latency: Lawyers expect sub-3-second responses

**Architecture Decisions:**

**1. Document Processing Pipeline**
- Multi-tier OCR: Modern → Legacy → Manual review queue
- Document classification: Era, jurisdiction, document type
- Quality scoring: Flag low-confidence documents

**2. Hierarchical Chunking**
- Level 1: Document-level embeddings for broad retrieval
- Level 2: Section-level (by legal headings)
- Level 3: Paragraph-level for precise retrieval
- Preserve parent-child relationships

**3. Hybrid Retrieval**
- Dense: Legal-domain fine-tuned embeddings
- Sparse: BM25 for exact legal terminology
- Fusion: Reciprocal Rank Fusion (RRF) with tuned weights
- Reranking: Cross-encoder trained on legal relevance

**4. Temporal Awareness**
- Metadata: Effective dates, amendment history
- Query understanding: "Current law" vs "Law as of 2010"
- Superseded document handling

**5. Evaluation Framework**
- Golden dataset: 500+ annotated query-answer pairs
- Metrics: Faithfulness, legal accuracy, citation correctness
- Human-in-the-loop for edge cases

**Key Trade-offs:**
- Latency vs Accuracy: Accept 3-5s for complex queries
- Storage vs Precision: Store multiple chunk sizes
- Cost vs Quality: Use expensive reranker for top-50 only

---

#### Q1.10: How would you debug a RAG system where users report "the AI knows the answer is in the documents but can't find it"?

**Expected Answer:**

This is the classic "retrieval failure with good generation" problem.

**Systematic Debugging Process:**

**Step 1: Reproduce and Isolate**
- Get specific failing queries
- Manually verify the answer IS in the documents
- Identify which documents contain the answer

**Step 2: Check Retrieval**
- What documents were actually retrieved?
- Was the correct document retrieved but ranked low?
- Was it not retrieved at all?

**Step 3: Diagnosis by Failure Mode**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Correct doc retrieved, wrong section | Chunk boundaries | Increase overlap, semantic chunking |
| Correct doc not in top-k | Embedding mismatch | Hybrid search, query expansion |
| Similar but wrong docs retrieved | Semantic ambiguity | Better metadata filtering |
| Nothing relevant retrieved | Query-document vocabulary gap | Query reformulation, synonyms |

**Step 4: Deep Dive Checks**

- **Embedding similarity**: What's the cosine similarity between query and correct doc?
- **Chunk inspection**: Is the answer split across chunks?
- **Metadata mismatch**: Is the doc filtered out by date/type?
- **Index freshness**: Was the doc indexed after the last update?

**Step 5: Solutions by Root Cause**

1. **Vocabulary gap**: Add query expansion (HyDE, multi-query)
2. **Chunk splitting**: Implement parent-child retrieval
3. **Semantic mismatch**: Add BM25 for exact term matching
4. **Ranking issue**: Add or improve reranking

**Key insight:** 90% of these issues are retrieval problems, not generation problems.

---

---

[← Back to Main](../README.md) | [Next: Chunking Strategies →](./02-chunking-strategies.md)
