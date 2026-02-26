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

#### Q16.4: Design a real-time AI-powered content moderation system for a social media platform with 1M posts per day.

**Key Points to Cover:**

1. **Requirements Analysis**
   - Latency: <100ms per post for real-time moderation
   - Multi-modal: text + images (and potentially video thumbnails)
   - Multi-language: English + Arabic (with dialect awareness)
   - Categories: hate speech, violence, misinformation, NSFW content
   - False positive rate <1% — over-moderation drives users away
   - Scale: ~12 posts/second average, 50+ posts/second at peak

2. **Architecture**
   ```
   Post Ingestion (Kafka/Kinesis)
       ↓
   Parallel Processing Pipeline:
       ├── Text Classifier (fine-tuned BERT/RoBERTa)
       ├── Image Classifier (CLIP + safety models)
       └── Metadata Analysis (user history, post patterns)
       ↓
   Decision Aggregator
       ├── Clear violation → Auto-remove (high-confidence ML decisions)
       ├── Clear safe → Auto-approve
       └── Ambiguous → LLM Reasoning Pipeline
           ↓
       LLM Analysis (structured output: decision + explanation)
           ↓
       Action (remove / flag / approve)
   ```

3. **Tiered Approach**
   - Fast ML models handle 90% of content (clear violations and clear safe)
   - LLM reasoning for the ambiguous 10% where classifiers disagree or have low confidence
   - This keeps costs manageable — LLM calls only for edge cases

4. **Multi-Modal Pipeline**
   - **Text classification**: fine-tuned BERT/RoBERTa for each violation category, separate models per language
   - **Image classification**: CLIP embeddings + fine-tuned safety classifier, NSFW detection model
   - **Combined reasoning**: for posts where text is benign but image is concerning (or vice versa), use VLM to understand the combined context

5. **LLM Integration for Ambiguous Cases**
   - Trigger: ML models disagree or confidence < threshold
   - Structured output: decision (remove/flag/approve) + category + explanation
   - Prompt includes platform policy guidelines as context
   - Response used for both decision and audit trail

6. **Arabic/Multilingual Considerations**
   - Dialect-aware processing: Gulf Arabic vs. Egyptian Arabic vs. MSA have different slang and expressions
   - Code-switching detection: Arabic + English mixed in same post is common and changes meaning
   - Cultural context awareness: expressions that are offensive in one dialect may be neutral in another
   - Right-to-left text handling in image text detection (OCR)

7. **Feedback Loop**
   - Human reviewers for user appeals — every appealed decision gets human review
   - Use reviewed decisions to improve ML models (active learning)
   - Track false positive/negative rates by category, language, and content type
   - Weekly model retraining with new labeled data from reviews

8. **Scaling**
   - Horizontal scaling of classifier workers (stateless, auto-scale on queue depth)
   - Perceptual hashing and caching for repeated/viral content (same image posted many times)
   - Batch processing for non-time-sensitive review queues
   - Circuit breakers: if LLM pipeline is slow, fall back to ML-only decisions with lower confidence threshold

---

#### Q16.5: Design a multi-lingual enterprise knowledge management system using RAG for a company with 50K employees across 10 countries.

**Key Points to Cover:**

1. **Requirements**
   - Support 5+ languages (English, Arabic, French, German, Mandarin, etc.)
   - Unified search across all languages — user can query in any language and find relevant docs in any language
   - Role-based access control: departments, seniority levels, project-specific access
   - Real-time updates when documents change
   - Handle 100K+ documents across departments (HR policies, technical docs, legal, finance)
   - Sub-3-second response time for queries

2. **Document Processing Pipeline**
   ```
   Document Ingestion
       ↓
   Language Detection (fastText / lingua)
       ↓
   Format Processing:
       ├── PDF → Text extraction + OCR for scanned docs
       ├── Word/PPT → Structure extraction
       ├── Email → Thread parsing
       └── Wiki pages → HTML parsing
       ↓
   Metadata Enrichment:
       ├── Department classification
       ├── Language tag
       ├── Confidentiality level (public / internal / restricted / confidential)
       ├── Document type (policy / procedure / report / reference)
       └── Entity extraction (people, projects, dates)
       ↓
   Chunking (language-aware: different strategies for CJK vs. Latin scripts)
       ↓
   Embedding + Indexing
   ```

3. **Embedding Strategy**
   - Use multilingual embedding model: multilingual-e5-large or Cohere multilingual embeddings
   - Cross-lingual retrieval: query in English, retrieve relevant Arabic or German documents
   - Single embedding space for all languages enables unified search
   - Benchmark embedding quality per language pair — some language pairs work better than others

4. **Retrieval Architecture**
   - **Hybrid search per language**: BM25 (language-specific tokenizers) + vector search
   - **Cross-lingual retrieval**: vector search across all languages simultaneously
   - **Language-aware reranking**: cross-encoder that handles multilingual input
   - **Department-level filtering**: metadata filters reduce search space before vector search
   - **Access control enforcement at retrieval time**: user permissions checked before any document is returned

5. **Multi-Tenancy and Access Control**
   ```
   Query → User Authentication → Permission Resolution
       ↓
   Retrieval with ACL Filter:
       - Department: user's department + public docs
       - Role: documents at or below user's clearance
       - Project: project-specific docs user is assigned to
       ↓
   Results (only documents user is authorized to see)
   ```
   - Document-level and chunk-level access control
   - Query routing based on user profile and language preference
   - Audit logging for all document access (compliance requirement)

6. **Response Generation**
   - Respond in user's preferred language (detected from query or profile setting)
   - Cite sources with language metadata: "Source: HR Policy v3.2 (German), Section 4"
   - Translate relevant passages if the source document is in a different language than the query
   - Use multilingual LLM (GPT-4, Claude) that handles all target languages well

7. **Scalability**
   - CDN for static content (cached responses to frequent queries)
   - Read replicas for vector database across regions
   - Regional deployments for latency: EU cluster, MENA cluster, APAC cluster
   - Incremental indexing: new/updated documents indexed within minutes, not hours
   - Document change detection: webhook or polling for source system changes

8. **Evaluation Framework**
   - Per-language quality metrics: retrieval accuracy, answer faithfulness, citation correctness
   - Cross-lingual retrieval accuracy: can a German query find the right English document?
   - User satisfaction tracking by region and language
   - A/B testing for embedding model and reranker improvements
   - Golden test set with queries in each supported language

---

---

[← Previous: LLM Reasoning Failures](./15-llm-reasoning-failures.md) | [← Back to Main](../README.md) | [Next: Red Flags →](./red-flags.md)
