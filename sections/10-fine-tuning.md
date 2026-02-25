[← Back to Main](../README.md) | [← Previous: LLM Deployment & Inference](./09-llm-deployment-inference.md) | [Next: Evaluation & Metrics →](./11-evaluation-metrics.md)

---

# Section 10: Fine-tuning

> **Key Stat**: LoRA achieves 90-95% of full fine-tuning quality with only 1-2% of trainable parameters.

### Intermediate Level

#### Q10.1: When should you fine-tune vs rely on prompting + RAG?

**Expected Answer:**

**Fine-tune When:**
- Need specific output format/style consistently
- Domain has unique patterns not in general models
- Latency is critical (can't afford RAG retrieval)
- Have 1000+ high-quality examples
- Need to reduce prompt length (cost savings)

**Use Prompting + RAG When:**
- Knowledge changes frequently
- Need citations/traceability
- Limited training data (<1000 examples)
- Don't want to manage model versions
- Rapid iteration needed

**Decision Matrix:**

| Scenario | Recommendation |
|----------|----------------|
| Customer support + KB | RAG |
| Code style matching | Fine-tune |
| Legal document Q&A | RAG |
| Specific JSON output | Fine-tune |
| Medical diagnosis | RAG + fine-tune |
| Brand voice | Fine-tune |

**Hybrid Approach:**
- Fine-tune for style/format
- RAG for knowledge
- Best of both worlds
- Example: Fine-tuned model + retrieval

---

#### Q10.2: Explain LoRA and QLoRA. What are the key hyperparameters?

**Expected Answer:**

**LoRA (Low-Rank Adaptation):**
- Freeze original model weights
- Add small trainable "adapter" matrices
- Train only adapters (0.1-1% of parameters)
- Merge adapters back after training

**QLoRA:**
- LoRA + 4-bit quantization
- Load base model in 4-bit
- Train LoRA adapters in higher precision
- Even more memory efficient

**Key Hyperparameters:**

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **r (rank)** | Dimension of adapter matrices | 8, 16, 32, 64 |
| **alpha** | Scaling factor | 16, 32 (usually 2×r) |
| **target_modules** | Which layers to adapt | q_proj, v_proj, etc. |
| **dropout** | Regularization | 0.05-0.1 |

**Rank (r) Trade-offs:**
- Higher r = More capacity, more memory, slower
- Lower r = Less capacity, faster, may underfit
- Start with r=16, adjust based on results

**Target Modules:**
- Attention layers: q_proj, k_proj, v_proj, o_proj
- FFN layers: gate_proj, up_proj, down_proj
- More modules = Better quality, more memory

**Memory Comparison (7B model):**

| Method | GPU Memory |
|--------|------------|
| Full fine-tune FP16 | 28GB |
| LoRA FP16 | 16GB |
| QLoRA 4-bit | 6GB |

---

### Advanced Level

#### Q10.3: What is catastrophic forgetting and how do you prevent it?

**Expected Answer:**

**The Problem:**
When fine-tuning on new data, model forgets what it previously knew.

**Examples:**
- Fine-tune on legal docs → Loses general knowledge
- Train for French → Gets worse at English
- Learn new task → Forgets old tasks

**Why It Happens:**
- Neural networks overwrite weights
- New task gradients change general weights
- No explicit memory of old tasks

**Prevention Strategies:**

**1. Use LoRA/PEFT**
- Freeze original weights
- Only train adapters
- Original knowledge preserved
- Most effective for preventing forgetting

**2. Mix Training Data**
- Include samples from original distribution
- Typically 10-20% general data
- Maintains balance

**3. Lower Learning Rate**
- Small changes = Less forgetting
- Trade-off with adaptation speed
- Start with 1e-5, go lower if forgetting

**4. Regularization**
- L2 regularization toward original weights
- Elastic Weight Consolidation (EWC)
- Penalize changes to important weights

**5. Continual Learning Techniques**
- Progressive neural networks
- Memory replay
- More complex to implement

**Detection:**
- Evaluate on held-out general benchmarks
- Track performance on original tasks
- Monitor during training, not just after

**Key insight:** If using LoRA, catastrophic forgetting is largely solved. For full fine-tuning, data mixing is the most practical solution.

---

---

[← Previous: LLM Deployment & Inference](./09-llm-deployment-inference.md) | [← Back to Main](../README.md) | [Next: Evaluation & Metrics →](./11-evaluation-metrics.md)
