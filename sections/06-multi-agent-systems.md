[← Back to Main](../README.md) | [← Previous: Semantic Caching](./05-semantic-caching.md) | [Next: Function Calling & Tool Use →](./07-function-calling-tool-use.md)

---

# Section 6: Multi-Agent Systems

> **Key Stat**: Google DeepMind research found that adding agents beyond 4 often DECREASES performance due to "Coordination Tax."

![Multi-Agent Architecture](../diagrams/multi-agent-architecture.svg)

### Intermediate Level

#### Q6.1: When should you use multi-agent vs single-agent systems?

**Expected Answer:**

**Use Single Agent When:**
- Task is straightforward and linear
- One domain of expertise needed
- Latency is critical
- Simplicity is valued
- Budget is limited

**Use Multi-Agent When:**
- Task requires multiple specialized skills
- Different perspectives improve output
- Task has clear parallel subtasks
- Need checks and balances (review/approval)
- Complex reasoning with multiple steps

**Decision Framework:**

| Factor | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Task complexity | Simple to moderate | Complex, multi-faceted |
| Domain breadth | Narrow | Wide |
| Quality needs | Good enough | Best possible |
| Latency budget | <5 seconds | 10+ seconds OK |
| Error tolerance | Some OK | Needs verification |
| Cost sensitivity | High | Lower |

**Key insight:** Start with single agent. Only add agents when you've identified specific bottlenecks that parallelization or specialization can solve.

---

#### Q6.2: Explain the ReAct pattern. What are its strengths and limitations?

**Expected Answer:**

**ReAct = Reasoning + Acting**

**The Pattern:**
1. **Thought**: LLM reasons about what to do
2. **Action**: LLM decides on an action (tool call)
3. **Observation**: Get result from action
4. **Repeat**: Continue until task complete

**Example:**
```
User: "What's the weather in the city where Apple is headquartered?"

Thought: I need to find where Apple is headquartered
Action: search("Apple headquarters location")
Observation: Apple is headquartered in Cupertino, California

Thought: Now I need weather for Cupertino
Action: get_weather("Cupertino, CA")
Observation: 72°F, sunny

Thought: I have the answer
Action: respond("The weather in Cupertino (Apple's HQ) is 72°F and sunny")
```

**Strengths:**
- Interpretable reasoning chain
- Handles multi-step problems
- Self-correcting (can retry on failure)
- Flexible tool use

**Limitations:**
- Sequential (slow for parallel tasks)
- Token-expensive (reasoning visible in context)
- Can get stuck in loops
- Requires careful prompt engineering
- Limited working memory

**When to use:**
- Tasks requiring 2-5 tool calls
- Debugging/transparency needed
- Moderate latency acceptable

---

### Advanced Level

#### Q6.3: Why doesn't adding more agents always improve performance? Explain the "Coordination Tax."

**Expected Answer:**

**The Intuition:**
Adding engineers to a late project makes it later (Brooks' Law). Same applies to agents.

**The Coordination Tax:**
Every additional agent adds:
- Communication overhead
- Potential for miscommunication
- State synchronization needs
- Conflict resolution requirements
- Increased failure modes

**Research Evidence (Google DeepMind, 2025):**
- 1-4 agents: Performance generally improves
- 4+ agents: Performance often plateaus or decreases
- Optimal number is task-dependent

**Why This Happens:**

1. **Communication Overhead**
   - N agents = N(N-1)/2 potential communication pairs
   - 4 agents = 6 pairs
   - 8 agents = 28 pairs
   - Most time spent coordinating, not working

2. **Error Propagation**
   - One agent's mistake cascades
   - More agents = more failure points
   - 5 agents at 95% each = 77% system reliability

3. **Context Dilution**
   - Each agent has limited context window
   - Sharing context between agents loses information
   - State becomes inconsistent

4. **Latency Multiplication**
   - Sequential agents: latency adds
   - Even parallel: synchronization adds overhead
   - More agents rarely means faster completion

**Optimal Topology Matters:**
- Independent: Each agent works alone → Scales well
- Centralized: One coordinator → Bottleneck
- Decentralized: All-to-all communication → Chaos

**Key insight:** The question isn't "how many agents" but "what topology minimizes coordination while enabling necessary collaboration."

---

#### Q6.4: How do you handle state corruption in multi-agent systems?

**Expected Answer:**

**The Problem:**
Multiple agents updating shared state simultaneously can cause:
- Race conditions
- Inconsistent views
- Lost updates
- Cascading errors

**Common Failure Modes:**

1. **Race Conditions**
   - Agent A reads balance = $100
   - Agent B reads balance = $100
   - Agent A subtracts $30, writes $70
   - Agent B subtracts $20, writes $80
   - Final balance: $80 (should be $50)

2. **Stale Reads**
   - Agent A makes decision based on old data
   - Data changed between read and action
   - Action is now invalid

3. **Circular Dependencies**
   - Agent A waits for Agent B
   - Agent B waits for Agent A
   - Deadlock

**Prevention Strategies:**

**1. Immutable State**
- Never modify state, create new versions
- Each agent works on specific version
- Merge at defined synchronization points
- Used by: LangGraph

**2. Single Writer**
- Only one agent can modify each state element
- Clear ownership boundaries
- Others read, one writes

**3. Transactional Updates**
- Batch related changes
- Apply atomically or rollback
- Detect conflicts before commit

**4. Event Sourcing**
- State = sequence of events
- Agents emit events, don't modify state
- Single reducer applies events to state

**Detection and Recovery:**

1. **Validation Gates**
   - Validate state after each agent action
   - Detect corruption early

2. **Checkpointing**
   - Save state at known-good points
   - Rollback on corruption detected

3. **Reconciliation**
   - Periodically verify state consistency
   - Heal inconsistencies automatically

**Key insight:** The best strategy is preventing corruption through architecture, not detecting and fixing it.

---

#### Q6.5: Compare Orchestrator-Worker vs Choreography patterns. When would you use each?

**Expected Answer:**

**Orchestrator-Worker (Centralized):**
- One "boss" agent coordinates
- Workers execute specific tasks
- Clear hierarchy and control flow

**Choreography (Decentralized):**
- Agents react to events
- No central coordinator
- Self-organizing behavior

**Detailed Comparison:**

| Aspect | Orchestrator-Worker | Choreography |
|--------|---------------------|--------------|
| Control | Centralized | Distributed |
| Complexity | Simpler to reason about | Complex emergent behavior |
| Bottleneck | Orchestrator | None (or many) |
| Failure mode | Single point of failure | Cascade failures |
| Scaling | Limited by orchestrator | Better horizontal scaling |
| Debugging | Easier (clear flow) | Harder (trace events) |
| Flexibility | Changes require orchestrator update | Agents adapt independently |

**When to Use Orchestrator-Worker:**
- Clear task decomposition
- Need predictable execution order
- Debugging/audit trail important
- Team less experienced with distributed systems
- <10 agents

**When to Use Choreography:**
- Highly dynamic task requirements
- Agents have equal importance
- Need to scale to many agents
- Events naturally define workflow
- Resilience to individual failures

**Hybrid Approach (Common in Production):**
- Orchestrator for high-level task coordination
- Choreography within agent teams
- Example: Manager orchestrates teams, teams self-coordinate

**Key insight:** Orchestrator-Worker is easier to build and debug. Use it unless you have specific reasons requiring choreography.

---

---

[← Previous: Semantic Caching](./05-semantic-caching.md) | [← Back to Main](../README.md) | [Next: Function Calling & Tool Use →](./07-function-calling-tool-use.md)
