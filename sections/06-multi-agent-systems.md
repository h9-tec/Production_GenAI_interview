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

### Fresh Level

#### Q6.6: What is an AI agent and how does it differ from a simple LLM call?

**Expected Answer:**

**Simple LLM Call:**
- Input → LLM → Output
- Single pass, no iteration
- No access to external tools
- No memory between calls
- Stateless

**AI Agent:**
- LLM that can plan, use tools, and iterate
- Decides what to do next based on observations
- Has access to tools (search, code execution, APIs)
- Maintains state across multiple steps
- Can self-correct and retry

**Key Differences:**

| Aspect | Simple LLM Call | AI Agent |
|--------|----------------|----------|
| Steps | Single | Multiple |
| Tools | None | Yes |
| Planning | None | Yes |
| Memory | None | Session state |
| Self-correction | None | Can retry/adjust |
| Latency | Fast (1-3s) | Slow (10-60s+) |
| Cost | Low | High (multiple calls) |
| Reliability | Predictable | Variable |

**When to Use Each:**
- Simple call: classification, summarization, Q&A, structured extraction
- Agent: research tasks, multi-step workflows, tasks requiring tool use, dynamic problem-solving

**Red Flag:** Candidate describes every LLM application as an "agent" — conflating a chat endpoint with agentic behavior.

**Key insight:** Most production LLM applications should NOT be agents. Use agents only when the task genuinely requires planning, tool use, and iteration.

---

### Intermediate Level

#### Q6.7: Compare CrewAI, LangGraph, and AutoGen. When would you use each?

**Expected Answer:**

**CrewAI:**
- Role-based multi-agent framework
- Define agents with roles, goals, and backstories
- Agents collaborate in crews to complete tasks
- Strengths: intuitive API, good for structured workflows
- Weaknesses: less control over execution flow
- Best for: teams wanting quick multi-agent setup, role-based task decomposition
- Adoption: raised $18M funding, used by 60% of Fortune 500 (2025)

**LangGraph:**
- Graph-based agent workflow framework by LangChain
- Define workflows as directed graphs (nodes = steps, edges = transitions)
- Fine-grained control over execution flow
- Strengths: precise flow control, stateful workflows, cycles supported
- Weaknesses: steeper learning curve, LangChain ecosystem dependency
- Best for: complex stateful workflows, production deployments needing precise control
- Adoption: in production at LinkedIn, Uber, 400+ companies (2025)

**AutoGen (now Microsoft Agent Framework):**
- Conversation-based multi-agent framework
- Agents communicate through messages (chat protocol)
- Strengths: flexible, supports human-in-the-loop naturally
- Weaknesses: harder to control, conversations can diverge
- Best for: research, exploratory tasks, human-AI collaboration
- Note: Microsoft merged AutoGen with Semantic Kernel into unified framework (Oct 2025), GA in Q1 2026

**Comparison:**

| Aspect | CrewAI | LangGraph | AutoGen |
|--------|--------|-----------|---------|
| Mental model | Teams with roles | Graphs with nodes | Conversations |
| Control level | Medium | High | Low |
| Learning curve | Easy | Medium | Medium |
| Production-ready | Yes | Yes | Transitioning |
| Best for | Structured tasks | Complex workflows | Research/exploration |

**Decision Guide:**
- Need quick prototype → CrewAI
- Need precise control in production → LangGraph
- Need human-in-the-loop collaboration → AutoGen
- Already in LangChain ecosystem → LangGraph
- Enterprise/Azure environment → AutoGen/Microsoft Agent Framework

---

### Advanced Level

#### Q6.8: How do you prevent infinite loops and runaway costs in agent systems?

**Expected Answer:**

**The Problem:**
Agents can get stuck in loops, repeatedly calling tools without making progress, or escalating costs unboundedly. A single runaway agent conversation can cost $50-$500+ in API calls.

**Prevention Strategies:**

**1. Hard Limits (Non-negotiable)**

| Limit | Typical Value | Purpose |
|-------|-------------|---------|
| Max steps per task | 10-25 | Prevent infinite loops |
| Max tokens per conversation | 50K-100K | Prevent unbounded generation |
| Max cost per task | $1-$10 | Prevent cost explosion |
| Max time per task | 5-30 minutes | Prevent hanging tasks |
| Max consecutive failures | 3 | Prevent retry loops |

**2. Loop Detection**
- Track action-observation patterns
- If same action repeated 3+ times → break and escalate
- If same tool called with same parameters → break
- Monitor "reasoning progress" — is the agent making forward progress?

**3. Cost Tracking (Real-time)**
- Count tokens at every step
- Calculate running cost
- Alert at 50% of budget, hard-stop at 100%
- Log cost breakdown per agent per step

**4. Circuit Breakers**
- If error rate exceeds threshold → stop and alert
- If latency exceeds threshold → timeout and fallback
- If cost exceeds threshold → graceful shutdown

**5. Fallback Strategies**
- When agent exceeds limits → hand off to simpler system
- When agent fails repeatedly → escalate to human
- When agent is stuck → provide a default/safe response

**Anti-Patterns:**
- No limits on agent execution (the most common and dangerous)
- Only time-based limits (agent can burn $100 in 2 minutes)
- Retrying failed agent tasks without investigating root cause
- Trust that "the prompt says stop after 5 steps" (LLMs ignore instructions)

**Key insight:** Never trust the LLM to self-limit. Implement hard limits in APPLICATION code, not in prompts. Prompts are suggestions; code limits are enforced.

---

### Expert Level

#### Q6.9: Design a multi-agent system for automated code review that handles 500 PRs per day.

**Expected Answer:**

**Requirements:**
- 500 PRs/day average, peak 100/hour
- Multi-language: Python, TypeScript, Java
- Review types: bugs, security, performance, style
- SLA: review within 30 minutes of PR creation
- False positive rate: <10% (developers ignore noisy reviewers)

**Agent Architecture:**

**Agent 1: Triage Agent (fast, cheap)**
- Classifies PR: size (S/M/L), risk level, languages used
- Routes to appropriate specialist agents
- Small PRs (<50 lines) get fast-path review
- Model: GPT-4o mini or similar (fast, cheap)

**Agent 2: Security Agent**
- Specialized in vulnerability detection
- Checks: SQL injection, XSS, secrets in code, auth issues
- Uses SAST tools + LLM reasoning
- Model: GPT-4o (needs strong reasoning)

**Agent 3: Bug Detection Agent**
- Looks for logic errors, race conditions, null handling
- Analyzes diff in context of surrounding code
- Cross-references with test coverage
- Model: Claude Sonnet or GPT-4o

**Agent 4: Style/Quality Agent**
- Enforces team conventions and best practices
- Checks naming, structure, documentation
- Lower stakes, faster model OK
- Model: GPT-4o mini

**Orchestrator:**
- Receives PR webhook
- Launches triage agent
- Based on triage, launches relevant specialist agents in parallel
- Collects findings from all agents
- Deduplicates and prioritizes
- Posts consolidated review comment on PR

**Cost Control:**

| PR Size | Agents Used | Estimated Cost |
|---------|------------|---------------|
| Small (<50 lines) | Triage + Style | $0.05 |
| Medium (50-200 lines) | All 4 agents | $0.30 |
| Large (200+ lines) | All 4 + extra context | $0.80 |

**At 500 PRs/day:** ~$100/day = $3,000/month

**Quality Metrics:**
- Track: accepted suggestions vs rejected (signal-to-noise ratio)
- Developer feedback: thumbs up/down on each finding
- False positive rate by agent and category
- Use feedback to improve prompts and thresholds

**Key insight:** The orchestrator-worker pattern works perfectly here. Keep specialist agents focused and fast. The biggest risk is false positives — developers stop reading reviews that cry wolf.

---

---

[← Previous: Semantic Caching](./05-semantic-caching.md) | [← Back to Main](../README.md) | [Next: Function Calling & Tool Use →](./07-function-calling-tool-use.md)
