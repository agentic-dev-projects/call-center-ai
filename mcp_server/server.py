"""
Call Center AI — MCP Server

Run this server standalone:
    python mcp_server/server.py

Connect from Claude Desktop by adding to claude_desktop_config.json:
    {
      "mcpServers": {
        "call-center-ai": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/call_center_ai/mcp_server/server.py"]
        }
      }
    }
"""

import json
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
# This script may be launched as a subprocess by Claude Desktop from any cwd.
# We always need the project root on the path so our modules are importable.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import FastMCP

from db.sqlite_store import get_all_calls, get_call, get_call_count
from tools.outage_tool import check_outage as _check_outage
from utils.logger import logger


# ── Create the MCP server ─────────────────────────────────────────────────────
# LEARNING: FastMCP is the high-level decorator-based API.
# The name "call-center-ai" is what MCP clients display as the server name.
mcp = FastMCP("call-center-ai")


# ═════════════════════════════════════════════════════════════════════════════
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────
# LEARNING: @mcp.tool() works like @app.get() in FastAPI.
# The function's docstring becomes the tool description the LLM reads.
# Type annotations are converted to a JSON Schema the LLM uses to
# validate arguments before calling the tool.
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_call_history(keyword: str) -> str:
    """
    Search stored call records by keyword.

    Searches across call summaries and transcripts. Returns matching calls
    with their ID, status, QA score, and a snippet of the summary.

    Use this to find calls about a specific topic (e.g. "billing", "outage",
    "refund") or to look up a customer complaint.

    Args:
        keyword: word or phrase to search for (case-insensitive)
    """
    calls = get_all_calls(limit=200)
    keyword_lower = keyword.lower()

    matches = []
    for c in calls:
        summary    = (c.get("summary") or "").lower()
        transcript = (c.get("raw_transcript") or "").lower()

        if keyword_lower in summary or keyword_lower in transcript:
            qa = c.get("qa_scores") or {}
            matches.append({
                "call_id":   c.get("call_id", "")[:12],
                "status":    c.get("status", ""),
                "qa_score":  qa.get("overall_score"),
                "summary_snippet": (c.get("summary") or "")[:120] + "...",
                "saved_at":  (c.get("updated_at") or "")[:19],
            })

    if not matches:
        return f"No calls found matching '{keyword}'."

    lines = [f"Found {len(matches)} call(s) matching '{keyword}':\n"]
    for m in matches:
        score = f"{m['qa_score']:.1f}/5.0" if m["qa_score"] else "N/A"
        lines.append(
            f"• [{m['call_id']}] {m['status'].upper()} | QA: {score} | {m['saved_at']}\n"
            f"  {m['summary_snippet']}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def get_call_details(call_id: str) -> str:
    """
    Retrieve full details for a specific call by its ID.

    Returns the complete summary, key points, action items, QA scores,
    and any guardrail violations for the call.

    Args:
        call_id: the call ID (first 12 characters are sufficient)
    """
    # Support prefix matching — user can pass first 12 chars
    all_calls = get_all_calls(limit=200)
    record = None
    for c in all_calls:
        if c["call_id"].startswith(call_id):
            record = c
            break

    if not record:
        return f"No call found with ID starting with '{call_id}'."

    qa = record.get("qa_scores") or {}
    violations = record.get("guardrail_violations") or []

    lines = [
        f"Call ID:  {record['call_id']}",
        f"Status:   {(record.get('status') or '').upper()}",
        f"Saved:    {(record.get('updated_at') or '')[:19]}",
        f"Cache:    {'Yes' if record.get('from_cache') else 'No'}",
        "",
        "SUMMARY:",
        record.get("summary") or "No summary available.",
        "",
        "KEY POINTS:",
    ]
    for kp in (record.get("key_points") or []):
        lines.append(f"  • {kp}")

    lines += ["", "ACTION ITEMS:"]
    for ai in (record.get("action_items") or []):
        lines.append(f"  • {ai}")

    if qa:
        lines += ["", "QA SCORES:"]
        for dim, score in qa.items():
            if isinstance(score, (int, float)):
                lines.append(f"  {dim}: {score:.1f}")

    if violations:
        lines += ["", f"GUARDRAIL VIOLATIONS ({len(violations)}):"]
        for v in violations:
            lines.append(f"  ⚠ {v}")

    if record.get("error"):
        lines += ["", f"ERROR: {record['error']}"]

    return "\n".join(lines)


@mcp.tool()
def get_call_stats() -> str:
    """
    Return aggregate statistics across all stored calls.

    Includes total call count, average QA score, cache hit rate,
    number of blocked calls, and score breakdown by QA dimension.

    Use this for a quick health check of the call center pipeline.
    """
    calls = get_all_calls(limit=1000)
    total = len(calls)

    if total == 0:
        return "No calls stored yet."

    scored = [c for c in calls if c.get("qa_scores") and c["qa_scores"].get("overall_score")]
    cache_hits = sum(1 for c in calls if c.get("from_cache"))
    blocked    = sum(1 for c in calls if c.get("guardrail_blocked"))
    with_violations = sum(1 for c in calls if c.get("guardrail_violations"))
    failed     = sum(1 for c in calls if c.get("status") == "failed")

    avg_overall = (
        sum(c["qa_scores"]["overall_score"] for c in scored) / len(scored)
        if scored else None
    )

    # Per-dimension averages
    dims = ["empathy", "resolution", "tone", "professionalism"]
    dim_avgs = {}
    for dim in dims:
        vals = [c["qa_scores"][dim] for c in scored if c.get("qa_scores", {}).get(dim)]
        dim_avgs[dim] = sum(vals) / len(vals) if vals else None

    lines = [
        "CALL CENTER STATS",
        "═" * 30,
        f"Total calls stored:    {total}",
        f"Successfully scored:   {len(scored)}",
        f"Failed / blocked:      {failed} / {blocked}",
        f"Cache hits:            {cache_hits} ({100*cache_hits//total}%)",
        f"Guardrail warnings:    {with_violations}",
        "",
        "QA SCORES:",
        f"  Overall avg:         {avg_overall:.2f}/5.0" if avg_overall else "  Overall avg: N/A",
    ]
    for dim, avg in dim_avgs.items():
        lines.append(f"  {dim.title():<20} {avg:.2f}" if avg else f"  {dim.title():<20} N/A")

    return "\n".join(lines)


@mcp.tool()
def check_outage(area: str) -> str:
    """
    Check whether there is a known service outage in a given area.

    Use this when a customer reports connectivity or service issues
    to determine whether it is a known infrastructure problem.

    Args:
        area: geographic area to check (e.g. "california", "new york")
    """
    return _check_outage(area)


# ═════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
# LEARNING: Resources are read-only data the LLM can access by URI.
# They're like files on a filesystem — the LLM requests them by path
# and gets back the content. Good for structured data the LLM should
# be able to "read" without calling a tool.
#
# URI pattern: call://{call_id}/{field}
# ═════════════════════════════════════════════════════════════════════════════

@mcp.resource("call://{call_id}/summary")
def get_summary_resource(call_id: str) -> str:
    """
    Read the summary for a stored call.

    URI: call://{call_id}/summary
    """
    all_calls = get_all_calls(limit=200)
    for c in all_calls:
        if c["call_id"].startswith(call_id):
            return c.get("summary") or "No summary available."
    return f"Call '{call_id}' not found."


@mcp.resource("call://{call_id}/transcript")
def get_transcript_resource(call_id: str) -> str:
    """
    Read the raw transcript for a stored call.

    URI: call://{call_id}/transcript
    """
    all_calls = get_all_calls(limit=200)
    for c in all_calls:
        if c["call_id"].startswith(call_id):
            return c.get("raw_transcript") or "No transcript available."
    return f"Call '{call_id}' not found."


# ── Entry point ───────────────────────────────────────────────────────────────
# LEARNING: mcp.run() starts the server with stdio transport.
# It reads JSON-RPC from stdin and writes to stdout — designed to be
# launched as a subprocess by an MCP client like Claude Desktop.
if __name__ == "__main__":
    logger.info("MCP server starting — call-center-ai")
    mcp.run()
