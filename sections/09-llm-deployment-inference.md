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

---

[← Previous: Arabic NLP Challenges](./08-arabic-nlp-challenges.md) | [← Back to Main](../README.md) | [Next: Fine-tuning →](./10-fine-tuning.md)
