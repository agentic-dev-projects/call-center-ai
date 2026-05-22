"""
QA Review Crew — three specialised agents that produce a call center report.

Agents:
  1. Call Analyst    — retrieves and organises call data using tools
  2. QA Reviewer     — evaluates quality, identifies patterns and issues
  3. Report Writer   — formats findings into an executive-ready report

Process: sequential — each agent's output becomes the next agent's context.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from config.settings import settings

# Import the same functions used by MCP and SupervisorAgent
from mcp_server.server import (
    search_call_history as _search,
    get_call_details    as _details,
    get_call_stats      as _stats,
    check_outage        as _outage,
)


# ── CrewAI tools ──────────────────────────────────────────────────────────────
# LEARNING: CrewAI tools use the @tool decorator.
# The function name becomes the tool name.
# The docstring becomes the description the LLM reads to decide when to use it.
# Type annotations define the argument schema.

@tool("Search Call History")
def search_call_history(keyword: str) -> str:
    """
    Search stored call records by keyword across summaries and transcripts.
    Use to find calls about billing, outages, refunds, escalations, etc.
    """
    return _search(keyword)


@tool("Get Call Details")
def get_call_details(call_id: str) -> str:
    """
    Get full details for a specific call: summary, key points,
    action items, QA scores, and guardrail violations.
    """
    return _details(call_id)


@tool("Get Call Statistics")
def get_call_stats() -> str:
    """
    Return aggregate statistics: total calls, average QA scores by dimension,
    cache hit rate, blocked call count, guardrail warning count.
    Use for overall performance overview.
    """
    return _stats()


@tool("Check Service Outage")
def check_outage(area: str) -> str:
    """
    Check whether there is a known service outage in a geographic area.
    Use when calls mention connectivity or service disruption issues.
    """
    return _outage(area)


# ── LLM configuration ─────────────────────────────────────────────────────────
# LEARNING: CrewAI uses its own LLM wrapper so you can swap models per-agent.
# Here all agents share the same model for cost efficiency.
# In production you'd use a cheaper model for the analyst (data retrieval)
# and a more capable model for the reviewer and writer (reasoning-heavy).

_llm = LLM(
    model=f"openai/{settings.QA_MODEL}",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.3,
)


# ═════════════════════════════════════════════════════════════════════════════
# AGENTS
# ═════════════════════════════════════════════════════════════════════════════

analyst_agent = Agent(
    role="Call Center Data Analyst",
    goal=(
        "Retrieve complete and accurate call center data. "
        "Use all available tools to gather call statistics, search for specific "
        "call types, and pull detailed records. Be thorough — the QA team depends "
        "on complete data."
    ),
    backstory=(
        "You are a senior data analyst with 8 years of call center analytics experience. "
        "You know exactly which data to pull and how to structure it for QA review. "
        "You always verify data completeness before handing off to the review team."
    ),
    tools=[search_call_history, get_call_details, get_call_stats, check_outage],
    llm=_llm,
    verbose=True,
)

qa_reviewer_agent = Agent(
    role="Quality Assurance Specialist",
    goal=(
        "Evaluate call quality rigorously. Identify patterns of excellence and "
        "areas needing improvement. Flag calls that show poor empathy, unresolved "
        "issues, or compliance concerns. Be specific — vague feedback helps no one."
    ),
    backstory=(
        "You are a QA specialist with a background in customer experience and "
        "compliance. You've reviewed thousands of calls and have a sharp eye for "
        "agent behaviour patterns. You score objectively and always back up "
        "observations with specific evidence from the call data."
    ),
    tools=[],   # reviewer works from context passed by the analyst
    llm=_llm,
    verbose=True,
)

report_writer_agent = Agent(
    role="Executive Report Writer",
    goal=(
        "Transform analytical findings into a clear, actionable report suitable "
        "for call center management. Use structured sections, bullet points, and "
        "concrete recommendations. The report must be readable by non-technical managers."
    ),
    backstory=(
        "You are a communications specialist who has written hundreds of operational "
        "reports for Fortune 500 companies. You excel at distilling complex data into "
        "clear narratives with specific, prioritised action items."
    ),
    tools=[],   # writer works from context passed by analyst + reviewer
    llm=_llm,
    verbose=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# TASKS
# ═════════════════════════════════════════════════════════════════════════════
# LEARNING: expected_output is critical — it tells the LLM what format
# its output should take. Precise expected_output = consistent results.

def make_tasks(focus: str) -> tuple:
    """
    Create tasks parameterised by the review focus topic.

    LEARNING: Tasks are created fresh per crew run so the `focus` topic
    is baked into the description. If tasks were module-level constants,
    they'd always use the same hardcoded topic.
    """

    analyst_task = Task(
        description=(
            f"Retrieve all relevant call center data related to: '{focus}'.\n\n"
            "Steps:\n"
            "1. Get overall call statistics (get_call_stats)\n"
            "2. Search for calls matching the focus topic\n"
            "3. For each matching call found, retrieve its full details\n"
            "4. Compile all data into a structured summary\n\n"
            "Be thorough. If no calls match, say so and provide overall stats only."
        ),
        expected_output=(
            "A structured data summary containing:\n"
            "- Overall call center statistics (totals, averages, cache rate)\n"
            "- List of calls matching the focus topic with their IDs and summaries\n"
            "- Detailed breakdown of QA scores for matching calls\n"
            "- Any guardrail violations or errors noted"
        ),
        agent=analyst_agent,
    )

    reviewer_task = Task(
        description=(
            f"Review the call data provided by the analyst for quality issues "
            f"related to: '{focus}'.\n\n"
            "Evaluate:\n"
            "- Agent performance across QA dimensions (empathy, resolution, tone, professionalism)\n"
            "- Patterns of success or failure\n"
            "- Specific calls that stand out (positively or negatively)\n"
            "- Root causes of any quality issues\n"
            "- Compliance or guardrail concerns\n\n"
            "Be specific — reference call IDs and actual scores."
        ),
        expected_output=(
            "A quality assessment containing:\n"
            "- Overall quality verdict (Good / Needs Improvement / Critical)\n"
            "- Top 3 strengths observed\n"
            "- Top 3 issues or concerns with specific call references\n"
            "- Root cause analysis for any recurring issues\n"
            "- Risk flags (guardrail violations, escalations, low scores)"
        ),
        agent=qa_reviewer_agent,
        context=[analyst_task],   # receives analyst's output as context
    )

    writer_task = Task(
        description=(
            "Write an executive QA report based on the analyst's data and "
            "the reviewer's assessment. The report is for call center management.\n\n"
            "The report must include:\n"
            "- Executive Summary (3-4 sentences)\n"
            "- Performance Overview (key metrics)\n"
            "- Quality Findings (strengths and issues)\n"
            "- Specific Call Highlights (notable calls)\n"
            "- Recommended Actions (prioritised, concrete)\n\n"
            "Keep it concise, professional, and actionable."
        ),
        expected_output=(
            "A complete executive QA report in markdown format with clearly "
            "labelled sections, bullet points for findings, and numbered "
            "recommendations ordered by priority."
        ),
        agent=report_writer_agent,
        context=[analyst_task, reviewer_task],   # receives both previous outputs
    )

    return analyst_task, reviewer_task, writer_task


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def run_qa_crew(focus: str = "overall performance") -> str:
    """
    Run the QA Review Crew and return the final markdown report.

    Args:
        focus: topic to review (e.g. "billing complaints", "escalated calls",
               "calls with low empathy scores")

    Returns:
        Markdown-formatted executive QA report as a string.

    LEARNING: crew.kickoff() runs all tasks sequentially.
    Each task's output is automatically passed as context to subsequent
    tasks via the `context` parameter. The final task's output is returned.
    """
    analyst_task, reviewer_task, writer_task = make_tasks(focus)

    crew = Crew(
        agents=[analyst_agent, qa_reviewer_agent, report_writer_agent],
        tasks=[analyst_task, reviewer_task, writer_task],
        process=Process.sequential,
        verbose=False,   # suppress per-step chatter; we show final output only
    )

    result = crew.kickoff()

    # CrewAI returns a CrewOutput object; .raw gives the final task's text
    return str(result.raw) if hasattr(result, "raw") else str(result)
