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

---

[← Previous: Multi-Agent Systems](./06-multi-agent-systems.md) | [← Back to Main](../README.md) | [Next: Arabic NLP Challenges →](./08-arabic-nlp-challenges.md)
