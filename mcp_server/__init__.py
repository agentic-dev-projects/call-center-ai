"""
MCP Server — Milestone 20

LEARNING: WHAT IS MCP?
════════════════════════

MCP (Model Context Protocol) is an open standard by Anthropic for
connecting LLMs to external tools and data sources.

Think of it as "USB-C for AI" — a universal plug that any LLM client
(Claude Desktop, LangChain, custom agents) can use to connect to any
data source or tool, without custom integration code per tool.

Before MCP:
  Each AI app wires its own tool integrations manually.
  Claude Desktop → custom Slack integration
  LangChain agent → custom Slack integration
  Your app → custom Slack integration
  (three separate integrations, all different)

After MCP:
  Build ONE MCP server. Any MCP-compatible client connects to it.

─────────────────────────────────────────────────────────────────────
HOW MCP WORKS
─────────────────────────────────────────────────────────────────────

Protocol layer: JSON-RPC 2.0 over stdio (subprocess) or HTTP/SSE.

Three primitives:

  TOOLS      — functions the LLM can call (like OpenAI function calling)
               "search_call_history", "get_call_stats", etc.

  RESOURCES  — read-only data the LLM can access (like file system, DB)
               "call://abc123/transcript", "call://abc123/summary"

  PROMPTS    — pre-built prompt templates (not used here)

Transport in this server: STDIO
  The MCP client (e.g. Claude Desktop) launches this script as a
  subprocess. It writes JSON-RPC requests to our stdin, we write
  JSON-RPC responses to stdout. Simple and reliable.

─────────────────────────────────────────────────────────────────────
OUR MCP SERVER EXPOSES
─────────────────────────────────────────────────────────────────────

Tools:
  search_call_history(keyword)   — full-text search across all calls
  get_call_details(call_id)      — full record for one call
  get_call_stats()               — aggregate metrics
  check_outage(area)             — existing outage lookup tool

Resources:
  call://{call_id}/summary       — summary text for a stored call
  call://{call_id}/transcript    — raw transcript for a stored call
"""
