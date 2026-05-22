"""
CrewAI — Milestone 22

LEARNING: CREWAI vs SINGLE-AGENT (A2A)
════════════════════════════════════════

M21 (A2A): ONE agent, tool-calling loop
  - Single LLM decides when to call tools and when to stop
  - Good for: ad-hoc queries, open-ended questions
  - Weakness: one agent tries to do everything — analyst + reviewer + writer

M22 (CrewAI): MULTIPLE specialised agents, structured pipeline
  - Each agent has a ROLE, GOAL, and BACKSTORY
  - Tasks are explicit: what to do, what output is expected
  - Agents hand off context to each other in sequence
  - Good for: structured workflows, quality pipelines, report generation

ANALOGY:
  A2A Supervisor = a smart generalist doing research
  CrewAI         = a team: analyst gathers data → reviewer critiques → writer publishes

─────────────────────────────────────────────────────────────────────
CREWAI CORE CONCEPTS
─────────────────────────────────────────────────────────────────────

AGENT
  An LLM with a role, goal, backstory, and optional tools.
  The role/goal/backstory prime the LLM's persona — they go into
  the system prompt to shape how it reasons and what it focuses on.

  agent = Agent(
      role="QA Specialist",
      goal="Identify quality issues in agent-customer interactions",
      backstory="10 years experience in call center quality auditing...",
      tools=[...],
      llm=...,
  )

TASK
  A unit of work assigned to one agent.
  description    → what to do (detailed instructions)
  expected_output → what the result should look like
  agent          → who does this task

  task = Task(
      description="Analyse the retrieved call data for quality issues...",
      expected_output="A bullet-point list of quality issues found...",
      agent=qa_agent,
  )

CREW
  Combines agents + tasks with a process.
  Process.sequential → tasks run in order, each sees previous outputs
  Process.hierarchical → a manager agent delegates to workers

  crew = Crew(
      agents=[analyst, reviewer, writer],
      tasks=[t1, t2, t3],
      process=Process.sequential,
  )

  result = crew.kickoff()   # runs the full pipeline

OUTPUT FLOW (sequential):
  Task 1 result → appended to Task 2's context → Task 2 result → ...
"""
