[← Back to Main](../README.md) | [← Previous: Arabic NLP Challenges](./08-arabic-nlp-challenges.md) | [Next: Fine-tuning →](./10-fine-tuning.md)

---

# Section 9: LLM Deployment & Inference

> **Key Stat**: vLLM with PagedAttention achieves 24x higher throughput than naive HuggingFace inference.

### Intermediate Level

#### Q9.1: Compare vLLM, TGI, and Ollama. When would you use each?

**Expected Answer:**

**vLLM:**
- High-performance inference server
- PagedAttention for efficient memory
- Continuous batching
- Best for: Production deployments, high throughput

**TGI (Text Generation Inference by HuggingFace):**
- Production-ready, battle-tested
- Good HuggingFace ecosystem integration
- Built-in safety features
- Best for: HuggingFace models, enterprise deployments

**Ollama:**
- Simple, user-friendly
- Easy model management
- Runs on CPU and GPU
- Best for: Local development, experimentation, edge

**Comparison:**

| Aspect | vLLM | TGI | Ollama |
|--------|------|-----|--------|
| Throughput | Highest | High | Moderate |
| Ease of setup | Medium | Medium | Easy |
| Memory efficiency | Excellent | Good | Good |
| Model support | Wide | HuggingFace | GGUF/GGML |
| Production ready | Yes | Yes | Limited |
| API compatibility | OpenAI | OpenAI | OpenAI |

**Decision Guide:**
- Need max performance → vLLM
- Using HuggingFace, need stability → TGI
- Local dev, quick testing → Ollama
- Edge/laptop deployment → Ollama

---

#### Q9.2: What is continuous batching and why does it matter?

**Expected Answer:**

**Traditional (Static) Batching:**
- Collect requests until batch is full OR timeout
- Process entire batch together
- Wait for ALL requests to complete
- Return results together

**Problem:**
- Short requests wait for long requests
- Wasted compute on padding
- High latency for fast queries

**Continuous Batching:**
- Process requests as they arrive
- Dynamically add new requests mid-generation
- Remove completed requests immediately
- No waiting for batch to fill

**Why It Matters:**

1. **Latency**
   - Requests processed immediately
   - No waiting for batch formation
   - Short queries return quickly

2. **Throughput**
   - GPU always working
   - No idle time between batches
   - Better hardware utilization

3. **Efficiency**
   - No padding waste
   - Memory used only for active requests
   - Scales better under load

**Impact:**
- 2-4x throughput improvement
- 50% latency reduction for short queries
- Better user experience under load

---

### Advanced Level

#### Q9.3: Compare GPTQ, AWQ, and GGUF quantization. What are the trade-offs?

**Expected Answer:**

**GPTQ (GPT Quantization):**
- Post-training quantization
- Uses calibration data
- Primarily 4-bit
- Good quality, widely supported
- Best for: GPU deployment, balanced quality/size

**AWQ (Activation-aware Weight Quantization):**
- Preserves important weights at higher precision
- Better quality than GPTQ at same bits
- Slightly more complex
- Best for: When quality is critical

**GGUF (GPT-Generated Unified Format):**
- CPU-optimized format (llama.cpp)
- Multiple quantization levels (Q2 to Q8)
- Runs without GPU
- Best for: CPU deployment, edge devices

**Comparison:**

| Aspect | GPTQ | AWQ | GGUF |
|--------|------|-----|------|
| Target hardware | GPU | GPU | CPU/GPU |
| Quality (4-bit) | Good | Better | Good |
| Speed | Fast | Fast | Moderate |
| Memory savings | 4x | 4x | 4-8x |
| Ecosystem | Wide | Growing | llama.cpp |

**Quality vs Compression:**

| Method | Bits | Quality Loss | Size Reduction |
|--------|------|--------------|----------------|
| FP16 | 16 | None | 2x vs FP32 |
| INT8 | 8 | ~1% | 4x vs FP32 |
| GPTQ-4bit | 4 | ~3-5% | 8x vs FP32 |
| AWQ-4bit | 4 | ~2-4% | 8x vs FP32 |
| GGUF-Q4 | 4 | ~3-5% | 8x vs FP32 |
| GGUF-Q2 | 2 | ~10-15% | 16x vs FP32 |

**When to Use What:**
- Cloud GPU, max quality → FP16 or AWQ
- Cloud GPU, cost-sensitive → GPTQ
- Local/edge deployment → GGUF
- Mobile/embedded → GGUF Q4 or lower

---

#### Q9.4: What are memory bandwidth bottlenecks in LLM inference? How do you address them?

**Expected Answer:**

**The Problem:**
LLM inference is memory-bandwidth bound, not compute bound.

**Why This Happens:**
- LLMs have billions of parameters
- Each token generation reads ALL parameters
- GPU compute is fast, memory is slow
- GPU sits idle waiting for memory

**The Math:**
- Llama-70B FP16: 140GB weights
- A100 memory bandwidth: 2TB/s
- Time to read weights: 140GB / 2TB/s = 70ms
- That's just reading, not computing
- Sets floor on per-token latency

**Manifestations:**
- Low GPU utilization despite high load
- Latency doesn't improve with faster GPU
- Batching helps throughput but not latency

**Solutions:**

**1. Quantization**
- 4-bit = 4x less data to move
- Directly reduces bandwidth needs
- Most impactful optimization

**2. KV-Cache Optimization**
- Cache key-value pairs from previous tokens
- Don't recompute, just read from cache
- PagedAttention (vLLM) manages this efficiently

**3. Speculative Decoding**
- Small model generates draft tokens
- Large model verifies in parallel
- Reduces number of large model passes

**4. Tensor Parallelism**
- Split model across GPUs
- Each GPU has less to read
- But adds communication overhead

**5. Flash Attention**
- Optimizes attention memory access
- Reduces intermediate storage
- 2-4x faster attention

**Key insight:** Before adding more GPUs, optimize memory access. Quantization + KV-cache optimization often 4x throughput on same hardware.

---

### Fresh Level

#### Q9.5: What is the difference between batch inference and real-time inference? When do you use each?

**Expected Answer:**

**Real-Time Inference:**
- Process one request at a time (or small batch)
- Response in milliseconds to seconds
- User is waiting for the response
- Examples: chatbot, search, real-time translation

**Batch Inference:**
- Process many requests at once (hundreds to millions)
- Response in minutes to hours
- No user waiting — results stored for later use
- Examples: document classification, embedding generation, content moderation backlog

**Comparison:**

| Aspect | Real-Time | Batch |
|--------|-----------|-------|
| Latency | <3 seconds | Minutes to hours |
| Throughput | Lower | Much higher |
| Cost per query | Higher | 50-90% cheaper |
| GPU utilization | Variable (idle between requests) | Very high (continuous) |
| User experience | Interactive | Background processing |
| Scaling | Auto-scale with traffic | Fixed capacity, scheduled |

**Cost Savings with Batch:**
- OpenAI Batch API: 50% discount
- Self-hosted: 3-5x higher throughput (full GPU utilization)
- No wasted compute on idle time

**When to Use Batch:**
- Embedding generation for new documents
- Bulk classification or tagging
- Evaluation runs
- Data enrichment pipelines
- Report generation

**When to Use Real-Time:**
- User-facing chat/Q&A
- Search and retrieval
- Real-time content moderation
- Interactive applications

**Hybrid Approach:**
Many production systems use both:
- Real-time for user interactions
- Batch for background processing (nightly re-embedding, weekly evaluation)

**Key insight:** If the user isn't waiting, use batch. It's 2-5x cheaper and lets you fully utilize your GPU.

---

### Intermediate Level

#### Q9.6: What is speculative decoding and how does it speed up LLM inference?

**Expected Answer:**

**The Problem:**
LLM inference is sequential — each token depends on the previous one. Can't parallelize the generation itself.

**How Speculative Decoding Works:**

1. **Draft Phase:** Small, fast model (draft model) generates K candidate tokens quickly
2. **Verification Phase:** Large, accurate model (target model) verifies ALL K tokens in a single forward pass
3. **Accept/Reject:** Accept tokens where draft matches target, reject and resample from target where they differ
4. **Result:** Multiple tokens per large model forward pass instead of one

**Why It Works:**
- Large model verification of K tokens takes almost the same time as generating 1 token
- For "easy" tokens (articles, common words), draft model matches target 70-90% of the time
- Only "hard" tokens (factual, reasoning) require target model generation
- Net result: 2-3x speedup with zero quality loss

**Key Property:**
Speculative decoding is **mathematically equivalent** to standard decoding — the output distribution is identical. It's not an approximation — it's a speed optimization with no quality trade-off.

**Requirements:**
- Draft model must be much smaller (3-10x) than target model
- Draft model should be from same family or fine-tuned on similar data
- Example: Llama-7B as draft for Llama-70B

**Performance:**

| Scenario | Speedup | Acceptance Rate |
|----------|---------|-----------------|
| Code generation | 2-3x | 70-80% |
| Creative writing | 1.5-2x | 60-70% |
| Technical Q&A | 2-2.5x | 65-75% |

**Limitations:**
- Requires running two models simultaneously (memory for both)
- Speedup depends on acceptance rate (domain-dependent)
- Not all inference engines support it yet

**Key insight:** Speculative decoding gives you free speed with no quality loss. If you're running self-hosted inference, this should be one of your first optimizations.

---

### Advanced Level

#### Q9.7: What is SGLang and how does it compare to vLLM for structured output workloads?

**Expected Answer:**

**SGLang Overview:**
SGLang (Structured Generation Language) is an inference framework optimized for structured LLM outputs (JSON, code, constrained generation).

**Key Innovation: RadixAttention**
- Reuses KV-cache across requests that share common prefixes
- Particularly powerful for structured generation where multiple outputs share the same prompt/schema
- Achieves up to 6.4x higher throughput on structured workloads

**Comparison with vLLM:**

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Primary focus | General-purpose high throughput | Structured output optimization |
| KV-cache | PagedAttention | RadixAttention (prefix reuse) |
| Best for | High-concurrency general chat | JSON/structured output, batch |
| Throughput (general) | Excellent | Good |
| Throughput (structured) | Good | Up to 6.4x better |
| Latency (structured) | Good | Up to 3.7x lower |
| Maturity | Very mature | Growing rapidly |
| Community | Large | Medium |

**When to Use SGLang:**
- Heavy structured output workloads (JSON APIs, data extraction)
- Workloads with prefix reuse (same system prompt, many queries)
- Batch processing with shared prompt templates
- Constrained generation (regex, grammar-guided)

**When to Use vLLM:**
- General-purpose chat/completions
- High-concurrency interactive workloads
- Maximum ecosystem compatibility
- When stability and maturity matter most

**Production Consideration:**
You can use both — SGLang for batch/structured workloads, vLLM for real-time interactive workloads. They serve different optimization profiles.

**Key insight:** SGLang shines when you have structured output patterns. If your primary workload is JSON generation or template-based completions, SGLang can give you 3-6x better performance than general-purpose frameworks.

---

### Expert Level

#### Q9.8: How do you right-size GPU infrastructure for LLM inference? Walk through the capacity planning.

**Expected Answer:**

**Step 1: Model Memory Requirements**

```
Model memory = Parameters × Bytes per parameter
              + KV-cache per request × Max concurrent requests
              + Activation memory overhead (~10-20%)
```

**Example: Llama-70B in FP16**
- Model weights: 70B × 2 bytes = 140GB
- KV-cache per request: ~1-2GB (varies by sequence length)
- For 32 concurrent requests: 140GB + 48GB = 188GB
- Needs: 3× A100-80GB or 2× H100-80GB

**Step 2: Throughput Calculation**

| Quantization | Memory | Tokens/sec (A100) | Quality |
|-------------|--------|-------------------|---------|
| FP16 | 140GB | 30-50 t/s | Best |
| INT8 | 70GB | 50-80 t/s | ~99% of FP16 |
| INT4 (GPTQ) | 35GB | 80-120 t/s | ~95-97% of FP16 |

**Step 3: Requests per Second**

```
Requests/sec = (Tokens/sec × Batch size) / Average output tokens per request

Example:
- INT4 on A100: 100 tokens/sec with batch=16
- Average response: 200 tokens
- Throughput: (100 × 16) / 200 = 8 requests/sec per GPU
```

**Step 4: Match to Traffic**

```
Required GPUs = Peak requests/sec / Throughput per GPU × Safety factor

Example:
- Peak: 50 requests/sec
- Throughput: 8 req/s per GPU
- Safety factor: 1.5x (for spikes)
- Required: 50 / 8 × 1.5 = ~10 GPUs
```

**Step 5: Cost Optimization**

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Quantization (INT4) | 2-4x less GPUs | 3-5% quality loss |
| Spot instances | 60-70% cost savings | Interruption risk |
| Auto-scaling | Only pay for what you use | Cold start latency |
| Batch processing off-peak | Better utilization | Delayed results |
| Speculative decoding | 2-3x throughput | Need draft model memory |

**GPU Selection Guide:**

| GPU | VRAM | Best For | Cloud Cost/hr |
|-----|------|----------|---------------|
| A10G | 24GB | 7B models, INT4 13B | $1.00 |
| A100 40GB | 40GB | 13B FP16, 70B INT4 | $3.50 |
| A100 80GB | 80GB | 70B FP16 (with TP) | $5.00 |
| H100 80GB | 80GB | 70B FP16, fastest | $8.00 |

**Key insight:** Start with quantized models on the smallest GPU that fits. Quantization is the single biggest lever — it can reduce GPU requirements by 4x with minimal quality loss. Only scale up when quality evaluation shows you need it.

---

---

[← Previous: Arabic NLP Challenges](./08-arabic-nlp-challenges.md) | [← Back to Main](../README.md) | [Next: Fine-tuning →](./10-fine-tuning.md)
