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

---

[← Previous: RAG Systems](./01-rag-systems.md) | [← Back to Main](../README.md) | [Next: Embeddings & Vector Databases →](./03-embeddings-vector-databases.md)
