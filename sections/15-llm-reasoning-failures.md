[← Back to Main](../README.md) | [← Previous: Observability & Monitoring](./14-observability-monitoring.md) | [Next: System Design Questions →](./16-system-design-questions.md)

---

# Section 15: LLM Reasoning Failures

> **Key Stat**: Even GPT-4 gets basic logic wrong quietly. Hallucination rate floor is ~0.5% even with best techniques.

### Advanced Level

#### Q15.1: What is "hallucination snowballing" and why is it dangerous?

**Expected Answer:**

**The Phenomenon:**
When an LLM makes an early mistake, it doesn't just persist — it gets elaborated on, explained, and defended.

**How It Works:**
1. LLM makes initial error (e.g., wrong fact)
2. Subsequent text builds on that error
3. LLM generates supporting details (fabricated)
4. Error becomes deeply embedded in response
5. Model confidently defends the mistake

**Example:**
```
Query: "When did Einstein discover relativity?"

Error chain:
1. Initial error: "Einstein published relativity in 1902"
2. Elaboration: "This was during his time at the Swiss Patent Office"
3. Fabrication: "The paper was titled 'On the Electrodynamics of Moving Bodies in Light'"
4. Confidence: "This is well documented in his collected works"

Reality: It was 1905, not 1902
```

**Why It's Dangerous:**

1. **Plausibility**
   - Supporting details make error believable
   - Harder to spot than isolated errors

2. **Confidence**
   - Model sounds certain
   - Users trust confident responses

3. **Compounding in Agents**
   - Error affects next action
   - Next action based on wrong premise
   - Cascade of wrong decisions

**Mitigation:**
- Verify facts at each reasoning step
- Use retrieval to ground claims
- Confidence calibration
- Self-consistency checking

---

#### Q15.2: What are the fundamental cognitive limitations of LLMs that cause reasoning failures?

**Expected Answer:**

**1. Limited Working Memory**
- Context window is finite
- Can't truly "hold" long-term dependencies
- Forgets earlier context in long conversations
- Impact: Loses track in multi-step reasoning

**2. No True State Tracking**
- Each token prediction is stateless
- "Memory" is just context window
- Can't maintain dynamic state
- Impact: Fails at tasks requiring state updates

**3. Pattern Matching, Not Reasoning**
- Learned statistical patterns
- Can mimic reasoning without understanding
- Breaks on novel problems outside training
- Impact: Brittle on edge cases

**4. Lack of Self-Model**
- Doesn't know what it knows
- Can't reliably assess confidence
- Hallucinates when uncertain
- Impact: Confident wrong answers

**5. No Causal Understanding**
- Knows correlations, not causation
- Can describe but not reason about causes
- Impact: Fails on causal reasoning tasks

**6. Inability to Verify**
- Can't check its own work
- No external feedback loop
- Propagates errors confidently
- Impact: Self-consistency ≠ correctness

**Implications for Production:**
- Don't trust LLMs for calculations → Use tools
- Don't trust for facts → Use retrieval
- Don't trust confidence → Calibrate externally
- Don't trust complex reasoning → Verify steps

**Key insight:** These aren't bugs to fix — they're fundamental properties of current architectures. Build systems that work DESPITE these limitations.

---

---

[← Previous: Observability & Monitoring](./14-observability-monitoring.md) | [← Back to Main](../README.md) | [Next: System Design Questions →](./16-system-design-questions.md)
