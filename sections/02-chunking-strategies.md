[← Back to Main](../README.md) | [← Previous: RAG Systems](./01-rag-systems.md) | [Next: Embeddings & Vector Databases →](./03-embeddings-vector-databases.md)

---

# Section 2: Chunking Strategies

> **Key Stat**: A 2025 study found semantic chunking improved faithfulness scores from 0.47 to 0.79 — a 68% improvement from chunking alone.

![Chunking Strategies](../diagrams/chunking-strategies.svg)

### Fresh Level

#### Q2.1: What is chunking and why is it necessary?

**Expected Answer:**

Chunking is the process of splitting documents into smaller pieces for RAG systems.

**Why it's necessary:**
- LLMs have context limits (can't fit entire documents)
- Embeddings work better on focused text
- Retrieval is more precise with smaller units
- Irrelevant content dilutes relevant context

**Basic chunking approaches:**
- **Fixed-size**: Split every N characters/tokens
- **Sentence-based**: Split at sentence boundaries
- **Paragraph-based**: Split at paragraph breaks
- **Semantic**: Split at topic/meaning changes

---

#### Q2.2: What are the trade-offs between small and large chunk sizes?

**Expected Answer:**

| Small Chunks (100-300 tokens) | Large Chunks (500-1000+ tokens) |
|-------------------------------|--------------------------------|
| ✅ More precise retrieval | ✅ Better context preservation |
| ✅ Less noise in results | ✅ Fewer chunks to process |
| ❌ May lose context | ❌ May include irrelevant content |
| ❌ More chunks to store/search | ❌ Less precise matching |

**Rule of thumb starting points:**
- Q&A systems: 256-512 tokens
- Document summarization: 512-1024 tokens
- Code understanding: Function/class boundaries
- Legal/medical: Semantic boundaries (sections)

**The real answer:** There's no universal optimal size. Test on YOUR data with YOUR queries.

---

### Intermediate Level

#### Q2.3: Compare Fixed, Recursive, and Semantic chunking. When would you use each?

**Expected Answer:**

**Fixed-size Chunking:**
- Split every N characters/tokens
- Add overlap (10-20%) to preserve context
- Pros: Simple, predictable, fast
- Cons: Ignores document structure, splits sentences
- Use when: Quick prototyping, uniform documents

**Recursive Chunking:**
- Try to split on paragraphs first
- If too large, try sentences
- If still too large, try words
- Pros: Respects natural boundaries
- Cons: Variable chunk sizes, more complex
- Use when: Documents have clear structure

**Semantic Chunking:**
- Use embeddings to detect topic changes
- Split where semantic similarity drops
- Pros: Preserves meaning units
- Cons: Expensive, slower, needs tuning
- Use when: Accuracy is critical, diverse content

**Production reality:** Most systems use recursive chunking with overlap as default, add semantic chunking for high-value documents.

---

#### Q2.4: What is overlap in chunking and how do you choose the right amount?

**Expected Answer:**

Overlap means adjacent chunks share some content at their boundaries.

**Why overlap matters:**
- Prevents information loss at chunk boundaries
- Allows retrieval of context that spans chunks
- Helps with sentence-level coherence

**Typical overlap ranges:**
- Minimal: 10% of chunk size
- Standard: 15-20% of chunk size
- High: 25-50% for critical documents

**Trade-offs:**

| More Overlap | Less Overlap |
|--------------|--------------|
| ✅ Better boundary handling | ✅ Less storage |
| ✅ Higher recall | ✅ Faster indexing |
| ❌ Duplicate content in results | ❌ May miss boundary info |
| ❌ Increased storage/compute | ❌ Context discontinuity |

**How to choose:**
1. Start with 50 tokens or 10% overlap
2. Test with queries that span topics
3. Increase if you see boundary-related failures
4. Deduplicate results if overlap causes redundancy

---

### Advanced Level

#### Q2.5: How does poor chunking cause 35% context loss in legal documents?

**Expected Answer:**

Legal documents have specific structural requirements that naive chunking destroys.

**Legal Document Characteristics:**
- Cross-references: "As stated in Section 3.2(a)..."
- Definitions: Terms defined once, used throughout
- Nested clauses: Complex sentence structures
- Numbered lists: Items that belong together
- Amendments: References to superseded sections

**How naive chunking fails:**

1. **Cross-reference Severing**
   - Chunk 1: "Subject to Section 5.1..."
   - Chunk 2: Contains Section 5.1
   - Retrieval gets one without the other

2. **Definition Separation**
   - Definitions section in one chunk
   - Usage of defined terms in another
   - LLM doesn't know "Affiliate" means what the contract defines

3. **List Fragmentation**
   - "The following are prohibited: (a) action one..."
   - Chunk boundary after item (c)
   - Items (d)-(f) orphaned from context

4. **Clause Nesting Breaks**
   - Complex sentence split mid-thought
   - Conditions separated from their effects

**The 35% figure:** Studies show naive chunking loses 35% of semantic relationships in structured documents compared to hierarchy-aware chunking.

**Solutions:**
- Parse document structure first (headings, lists, sections)
- Chunk within structural boundaries
- Include parent context in child chunks
- Maintain definition lookup tables

---

#### Q2.6: Explain parent-child chunking and when it provides significant benefit.

**Expected Answer:**

**What is Parent-Child Chunking:**
- Create two levels of chunks from the same content
- Parent: Larger chunks (e.g., full page or section)
- Child: Smaller chunks within each parent
- Link children to their parent

**How it works in retrieval:**
1. Search against child chunks (precise matching)
2. When child matches, retrieve parent (full context)
3. LLM sees the broader context, not just the matching snippet

**When it provides significant benefit:**

| Scenario | Benefit Level | Why |
|----------|---------------|-----|
| Technical documentation | High | Context needed to understand specifics |
| Legal contracts | Very High | Cross-references within sections |
| Q&A from manuals | High | Answers need surrounding context |
| News articles | Low | Usually self-contained paragraphs |
| Chat logs | Low | Context is the conversation |

**Implementation considerations:**
- Storage: 2x the embeddings (parent + child)
- Retrieval: More complex logic
- Deduplication: Same parent from multiple children

**Key insight:** Parent-child is most valuable when the MATCHING text is different from the CONTEXT needed to answer.

---

### Expert Level

#### Q2.7: Design a chunking strategy for a corpus with mixed content: technical manuals, contracts, and support tickets.

**Expected Answer:**

**Challenge:** Different content types need different chunking strategies.

**Step 1: Document Classification**
- Classify each document by type
- Use metadata if available, otherwise ML classifier
- Route to type-specific pipeline

**Step 2: Type-Specific Strategies**

**Technical Manuals:**
- Parse structure (chapters, sections, procedures)
- Chunk by procedure/section
- Include hierarchy path in metadata
- Parent-child for detailed procedures
- Preserve code blocks intact

**Contracts:**
- Identify sections, clauses, schedules
- Never split numbered clauses
- Include definitions as retrievable context
- Link amendments to original sections
- Semantic chunking within sections

**Support Tickets:**
- Treat each ticket as atomic unit
- Include: subject, body, resolution
- Add structured metadata (category, product, date)
- Don't chunk short tickets
- For long threads, chunk by conversation turn

**Step 3: Unified Index with Metadata**
- All chunks go to same vector DB
- Rich metadata enables filtering
- Type-specific retrieval strategies

**Step 4: Type-Aware Retrieval**
- Query classifier determines likely content type
- Apply type-specific reranking
- Merge results with type-aware weighting

**Key Trade-off:** Complexity vs Accuracy. This approach is 3x more complex but can improve accuracy by 40% on mixed corpora.

---

### Intermediate Level

#### Q2.8: What is Late Chunking and how does it differ from traditional approaches?

**Expected Answer:**

**Traditional Chunking Flow:**
```
Document → Split into chunks → Embed each chunk independently
```
Each chunk is embedded WITHOUT awareness of surrounding content. Context is lost at boundaries.

**Late Chunking Flow:**
```
Document → Process ENTIRE document through embedding model → Apply chunk boundaries → Create chunk embeddings
```
Each chunk embedding retains awareness of the full document context.

**Why It Matters:**
- Traditional: chunk about "the policy" doesn't know WHICH policy (context was in previous chunk)
- Late chunking: chunk retains understanding of document context even though it's split

**Comparison with Related Approaches:**

| Approach | How It Works | Quality | Cost |
|----------|-------------|---------|------|
| Traditional chunking | Chunk → Embed independently | Baseline | Low |
| Contextual retrieval (Anthropic) | Add context summaries to chunks before embedding | Better | Medium (LLM calls) |
| Late chunking | Embed full document → then split | Best context retention | High (long document processing) |

**When to Use Late Chunking:**
- High-value documents where context preservation is critical
- Documents with many internal cross-references
- Technical or legal content where terms are defined once and used throughout

**When NOT to Use:**
- Simple, self-contained documents (paragraphs stand alone)
- Very large documents exceeding model context window
- High-volume processing where compute cost matters

**Trade-off:** Late chunking requires processing full documents through the embedding model — significantly more compute per document, but better retrieval quality.

**Key insight:** Late chunking addresses the fundamental problem that independent chunk embeddings lose document context. It's one of the most impactful recent advances in RAG retrieval quality.

---

### Advanced Level

#### Q2.9: What is LLM-based chunking and when is it worth the cost?

**Expected Answer:**

**What It Is:**
Using a language model to analyze document structure and decide where to split, instead of following fixed rules or embedding similarity.

**How It Works:**
1. Feed document (or sections) to LLM
2. LLM analyzes content structure, topic boundaries, and semantic coherence
3. LLM outputs chunk boundaries and optionally chunk summaries
4. Application splits document at LLM-recommended boundaries

**Advantages:**
- Best semantic coherence — chunks are truly self-contained topics
- Handles complex documents (mixed formats, nested structures, tables within text)
- Can generate chunk summaries for better retrieval
- Understands nuances that rule-based approaches miss

**Disadvantages:**
- Expensive: $0.01-$0.10 per 10K-word document
- Slow: multiple LLM calls per document
- Non-deterministic: same document may chunk differently each time
- Not scalable for large corpora

**Cost Analysis:**

| Corpus Size | Cost at $0.05/doc | Time (parallel) |
|-------------|-------------------|-----------------|
| 1K documents | $50 | Minutes |
| 10K documents | $500 | Hours |
| 100K documents | $5,000 | Days |
| 1M documents | $50,000 | Weeks |

**When Worth It:**
- High-value documents (legal contracts, medical records)
- Small corpus (<10K docs) where accuracy is paramount
- Complex mixed-format documents
- Initial setup where you want the best possible chunking

**When NOT Worth It:**
- Large corpus (>100K docs) — cost prohibitive
- Frequently changing documents — must re-chunk on every change
- Budget-constrained projects
- Simple, well-structured documents

**Practical Alternative:**
Use LLM-based chunking on a representative sample (100-500 docs) to LEARN patterns, then create rules that can be applied at scale with traditional methods.

**Key insight:** LLM-based chunking is not for production-scale processing. Use it strategically on high-value content or as a learning tool to create better rules.

---

### Expert Level

#### Q2.10: How do you evaluate chunking quality? What metrics actually matter?

**Expected Answer:**

**The Key Principle:**
Chunking quality should be measured by DOWNSTREAM impact, not by how "nice" the chunks look.

**Direct Chunking Metrics (Less Useful):**

| Metric | What It Measures | Limitation |
|--------|-----------------|------------|
| Semantic coherence | Does each chunk cover one topic? | Doesn't guarantee retrieval quality |
| Boundary quality | Splits at natural boundaries? | Subjective, hard to measure |
| Size distribution | Consistent chunk sizes? | Not directly related to quality |
| Information completeness | No critical info split? | Hard to measure automatically |

**Downstream Metrics (Most Important):**

| Metric | What It Measures | How to Evaluate |
|--------|-----------------|-----------------|
| Retrieval Recall@5 | Are relevant chunks in top 5? | Golden dataset with labeled chunks |
| Context Relevance | Is retrieved context useful? | RAGAS metric on test queries |
| Faithfulness | Is answer grounded in context? | RAGAS metric on test queries |
| Chunk Utilization | How much retrieved context is actually used? | Compare context vs answer |
| Answer Correctness | Is the final answer right? | Golden dataset comparison |

**Evaluation Methodology:**

**Step 1:** Create golden Q&A pairs (200+ queries with expected answers and source documents)

**Step 2:** Test multiple chunking strategies on SAME corpus:
- Fixed-size (256, 512, 1024 tokens)
- Recursive with overlap
- Semantic chunking
- Domain-specific (if applicable)

**Step 3:** For each strategy, measure end-to-end RAG metrics:
- Retrieval recall@5/10
- Faithfulness
- Answer correctness

**Step 4:** Choose strategy with best downstream performance

**Step 5:** A/B test in production:
- Deploy two strategies
- Measure user satisfaction and quality metrics on live traffic
- Choose winner based on real-world performance

**Common Finding:** The "best" chunking strategy varies by:
- Document type (structured vs unstructured)
- Query type (specific vs exploratory)
- Domain (legal vs support tickets vs technical)

**Key insight:** The best chunking strategy is the one that produces the best end-to-end RAG results on YOUR data, not the one that scores highest on theoretical metrics.

---

---

[← Previous: RAG Systems](./01-rag-systems.md) | [← Back to Main](../README.md) | [Next: Embeddings & Vector Databases →](./03-embeddings-vector-databases.md)
