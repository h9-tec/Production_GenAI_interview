[← Back to Main](../README.md) | [← Previous: Multi-Agent Systems](./06-multi-agent-systems.md) | [Next: Arabic NLP Challenges →](./08-arabic-nlp-challenges.md)

---

# Section 7: Function Calling & Tool Use

> **Key Stat**: Tool calling fails 3-15% of the time in production. Robust error handling is not optional.

### Intermediate Level

#### Q7.1: What are the key principles of designing good tool schemas?

**Expected Answer:**

**Principle 1: Clear, Specific Descriptions**
- Bad: "Search function"
- Good: "Search the product database for items matching the query. Returns up to 10 results with product name, price, and availability."

**Principle 2: Constrained Parameters**
- Use enums when options are limited
- Set min/max for numbers
- Provide defaults for optional parameters
- Validate formats (email, date, etc.)

**Principle 3: Atomic Operations**
- One tool = one action
- Bad: "search_and_order" (does two things)
- Good: "search_products" + "place_order" (separate)

**Principle 4: Predictable Return Format**
- Consistent structure across calls
- Always include status/error field
- Document expected output shape

**Principle 5: Error Information**
- Tools should explain failures
- Bad: Returns null on error
- Good: Returns {error: "No products match query 'xyz'"}

**Principle 6: Minimal Required Parameters**
- Fewer required params = easier for LLM
- Use sensible defaults
- Only require what's truly necessary

**Common Mistakes:**
- Overloading tools with multiple functions
- Vague descriptions LLM can't interpret
- Missing error handling
- Inconsistent parameter naming

---

#### Q7.2: How do you handle tool calling failures gracefully?

**Expected Answer:**

**Types of Failures:**

1. **Tool Execution Failure**
   - API timeout, network error
   - Invalid API response
   - Rate limiting

2. **LLM Selection Failure**
   - Wrong tool chosen
   - Invalid parameters provided
   - Tool called when not needed

3. **Result Interpretation Failure**
   - LLM misunderstands tool output
   - Hallucinates based on partial result

**Handling Strategies:**

**For Execution Failures:**
- Implement retries with exponential backoff
- Set reasonable timeouts (don't wait forever)
- Have fallback tools or responses
- Return structured errors LLM can understand

**For Selection Failures:**
- Validate parameters before execution
- Return helpful error messages
- Allow LLM to self-correct

**For Interpretation Failures:**
- Structure tool outputs clearly
- Include context with results
- Validate LLM's interpretation

**Error Response Template:**
```
{
  "success": false,
  "error_type": "api_timeout",
  "error_message": "Product search timed out after 5s",
  "suggestion": "Try a more specific search query",
  "partial_results": null
}
```

**Key insight:** The LLM should be able to recover from tool errors. Design error messages for LLM consumption, not just human debugging.

---

### Advanced Level

#### Q7.3: What are the limitations of MCP (Model Context Protocol)? When does it fall short?

**Expected Answer:**

**What MCP Solves:**
- Standardizes tool interface (no more M×N integrations)
- Universal protocol for any LLM to use any tool
- Simplifies tool discovery and schema sharing

**MCP Limitations:**

1. **Doesn't Solve Reliability**
   - LLMs still hallucinate tool calls
   - Wrong tool selection is still common
   - MCP standardizes the interface, not the quality

2. **Cognitive Load Remains**
   - More tools = more confusion for LLM
   - LLMs struggle with unbounded action spaces
   - MCP makes adding tools easier, not using them better

3. **Latency Overhead**
   - HTTP/SSE transport adds latency
   - Tool discovery phase adds startup time
   - Each tool call is a network round-trip

4. **Security Concerns**
   - Remote tool servers = attack surface
   - Tool calls may leak sensitive data
   - Authentication complexity

5. **Debugging Complexity**
   - Distributed system debugging
   - Hard to reproduce issues
   - Logging across tool servers

**Research Finding (Salesforce MCP-Universe, 2025):**
"Interface standardization is a secondary bottleneck compared to cognitive reliability."

**When MCP Falls Short:**
- High-frequency tool calling (latency adds up)
- Security-critical applications (trust issues)
- Simple applications (overhead not justified)
- Offline/edge deployment (needs network)

**Key insight:** MCP solves the integration problem, not the reliability problem. You still need robust tool selection logic, error handling, and validation.

---

### Fresh Level

#### Q7.4: What is function calling in LLMs and how does it work?

**Expected Answer:**

Function calling allows LLMs to interact with external systems by generating structured data describing API calls, rather than executing them directly.

**How It Works:**
1. Developer defines available tools with JSON Schema descriptions
2. User sends a query to the LLM
3. LLM decides if a tool call is needed
4. LLM generates a structured JSON object (function name + parameters)
5. Application code executes the actual function call
6. Result is returned to the LLM for final response

**Key Distinction:**
- The LLM does NOT execute functions — it generates a data structure describing the call
- The application is responsible for execution, validation, and error handling
- This separation is critical for security and reliability

**Common Use Cases:**
- Database queries
- API integrations (weather, booking, search)
- Calculator/math operations
- File operations
- Multi-step workflows

**Example Flow:**
```
User: "What's the weather in Dubai?"
LLM Output: {"function": "get_weather", "parameters": {"city": "Dubai"}}
App: Calls weather API → Returns 42°C
LLM: "The current temperature in Dubai is 42°C"
```

**Red Flag:** Candidate thinks the LLM directly executes functions or API calls.

---

### Intermediate Level

#### Q7.5: How does function calling differ from MCP (Model Context Protocol)? When would you use each?

**Expected Answer:**

**Function Calling:**
- Defined per-application by the developer
- Tools are hardcoded in the application code
- Each LLM provider has slightly different API format
- Developer manages tool registration and execution
- Works offline, no network dependency for tool discovery

**MCP (Model Context Protocol):**
- Standardized protocol proposed by Anthropic (2024)
- Tools are hosted on remote MCP servers
- Universal interface — any LLM can use any MCP tool
- Dynamic tool discovery at runtime
- Requires network connectivity

**The Relationship:**
MCP and function calling are not in conflict — they're complementary:
- Function calling = HOW the LLM decides what to do
- MCP = HOW tools are discovered, described, and executed

**When to Use Function Calling Only:**
- Simple applications with 2-5 tools
- Offline/edge deployments
- Custom internal tools
- Latency-critical paths

**When to Use MCP:**
- Need standardized tool ecosystem
- Want plug-and-play tool integration
- Building platform for multiple tool providers
- Team builds tools consumed by multiple applications

**Production Consideration:**
In July 2025, a Replit AI agent deleted a production database via MCP despite explicit safety instructions — highlighting that MCP solves the integration problem, not the safety problem.

**Key insight:** MCP standardizes the plumbing. You still need validation, rate limiting, and permission controls on top.

---

### Expert Level

#### Q7.6: Design a production-safe tool execution framework. What guardrails do you need?

**Expected Answer:**

**The Problem:**
LLMs make tool calling errors 3-15% of the time. In production, this means destructive actions, data leaks, or infinite loops without proper safeguards.

**Framework Architecture:**

**Layer 1: Tool Registry & Permissions**
- Categorize tools by risk level: read-only, write, destructive
- Define per-user permission boundaries
- Rate limits per tool and per user
- Allowlists for parameter values where possible

**Layer 2: Pre-Execution Validation**

| Check | Purpose | Example |
|-------|---------|---------|
| Schema validation | Parameters match spec | Reject missing required fields |
| Value bounds | Parameters within range | Amount < $10,000 |
| Semantic validation | Tool makes sense for context | Don't delete in a read query |
| Confirmation gate | Human approval for high-risk | "Are you sure you want to delete?" |

**Layer 3: Execution Sandbox**
- Timeouts on all tool executions (5-30s)
- Resource limits (memory, CPU)
- Network isolation where possible
- Idempotency tokens for write operations

**Layer 4: Post-Execution Verification**
- Validate return format
- Check for error indicators
- Log all executions with full context
- Audit trail for destructive operations

**Layer 5: Circuit Breakers**
- Max tool calls per conversation (prevent loops)
- Max consecutive failures before human escalation
- Cost limits per session
- Anomaly detection on tool usage patterns

**Anti-Patterns to Avoid:**
- Trusting LLM parameter values without validation
- No timeout on tool execution
- Allowing destructive tools without confirmation
- No logging or audit trail
- Unlimited tool call loops

**Key insight:** Design tool execution as if the LLM is an untrusted junior developer — validate everything, limit blast radius, and always have a rollback plan.

---

---

[← Previous: Multi-Agent Systems](./06-multi-agent-systems.md) | [← Back to Main](../README.md) | [Next: Arabic NLP Challenges →](./08-arabic-nlp-challenges.md)
