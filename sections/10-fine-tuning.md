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

### Fresh Level

#### Q10.4: What is fine-tuning and how does it differ from pre-training?

**Expected Answer:**

**Pre-training:**
- Training a model from scratch on massive data (trillions of tokens)
- Learns general language understanding
- Costs millions of dollars
- Takes weeks/months on thousands of GPUs
- Done by model providers (OpenAI, Meta, Google)

**Fine-tuning:**
- Adapting a pre-trained model to a specific task or domain
- Starts from existing knowledge
- Costs $10-$10,000 depending on method and scale
- Takes hours to days on 1-8 GPUs
- Done by application developers

**Types of Fine-tuning:**

| Type | What Changes | Cost | Quality |
|------|-------------|------|---------|
| Full fine-tuning | All parameters | $$$ | Best |
| LoRA/PEFT | Small adapter layers | $ | 90-95% of full |
| Prompt tuning | Soft prompt embeddings | $ | Good for narrow tasks |
| RLHF/DPO | Alignment with preferences | $$ | Best for behavior |

**When Fine-tuning Happens in the Stack:**
```
Pre-training → Instruction tuning → Fine-tuning → RLHF/DPO → Production
(Model provider)   (Model provider)   (You)          (You)       (You)
```

---

### Intermediate Level

#### Q10.5: What is DPO (Direct Preference Optimization) and how does it compare to RLHF?

**Expected Answer:**

**RLHF (Reinforcement Learning from Human Feedback):**
1. Collect human preference data (A is better than B)
2. Train a separate reward model on preferences
3. Use PPO to optimize LLM against reward model
4. Complex, unstable, expensive

**DPO (Direct Preference Optimization):**
1. Collect same human preference data
2. Directly optimize the LLM using preference pairs
3. No separate reward model needed
4. Simpler, more stable, cheaper

**Comparison:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| Reward model | Required | Not needed |
| Training stability | Tricky (PPO instability) | More stable |
| Compute cost | 2-3x more | Baseline |
| Memory | Needs reward model + policy | Just the policy model |
| Quality | Gold standard | Comparable (90-95% of RLHF) |
| Complexity | High (3-stage pipeline) | Low (single training run) |

**When to Use Each:**
- **RLHF**: When budget allows and marginal quality matters (frontier model training)
- **DPO**: When you want preference alignment without RLHF complexity (most production cases)
- **Neither**: When supervised fine-tuning on good examples is sufficient

**Newer Alternatives (2025):**
- **ORPO**: Combines SFT and preference alignment in one step
- **KTO**: Only needs "good" or "bad" labels, not paired preferences
- **SimPO**: Simplifies DPO further with reference-free optimization

**Key insight:** For most production use cases, DPO on a few thousand preference pairs gives 90% of RLHF benefit at 30% of the cost.

---

### Expert Level

#### Q10.6: How do you evaluate fine-tuning quality and detect when fine-tuning has gone wrong?

**Expected Answer:**

**Evaluation Dimensions:**

| Dimension | Metric | What It Catches |
|-----------|--------|-----------------|
| Task performance | Accuracy on test set | Did it learn the task? |
| General capability | MMLU, HellaSwag scores | Did it lose general knowledge? |
| Instruction following | MT-Bench, AlpacaEval | Can it still follow instructions? |
| Safety | ToxiGen, red-team tests | Did it become unsafe? |
| Domain quality | Domain-specific test set | Does it know the domain? |

**Signs Fine-tuning Has Gone Wrong:**

**1. Catastrophic Forgetting**
- General benchmarks drop >5%
- Model can't do basic tasks anymore
- Fix: Lower learning rate, add general data to mix

**2. Overfitting**
- Perfect on training data, poor on test data
- Training loss still decreasing, validation loss increasing
- Fix: More data, regularization, early stopping

**3. Mode Collapse**
- Model gives same/similar response to different inputs
- Loss of diversity in outputs
- Fix: Reduce learning rate, fewer epochs, temperature sampling

**4. Safety Degradation**
- Model becomes more willing to produce harmful content
- Guardrail bypass becomes easier
- Fix: Include safety examples in fine-tuning data

**5. Format Corruption**
- Model loses ability to output structured formats
- JSON/code output becomes malformed
- Fix: Include format examples in training data

**Evaluation Framework:**
```
Before fine-tuning:
1. Run all benchmarks on base model (baseline)
2. Define acceptable regression thresholds

After fine-tuning:
3. Run same benchmarks on fine-tuned model
4. Compare: Task improvement vs general regression
5. If regression > threshold → Reject, adjust hyperparameters
6. Human evaluation on 100+ samples
7. A/B test in production with small traffic %
```

**Key insight:** Always maintain a comprehensive eval suite. The question isn't "did the model improve on my task?" but "did the model improve on my task WITHOUT degrading elsewhere?"

---

---

[← Previous: LLM Deployment & Inference](./09-llm-deployment-inference.md) | [← Back to Main](../README.md) | [Next: Evaluation & Metrics →](./11-evaluation-metrics.md)
