"""
LangSmith Tracing Setup — Milestone 16

LEARNING: HOW LANGSMITH TRACING WORKS
══════════════════════════════════════

LangSmith is Anthropic/LangChain's observability platform for LLM applications.
It captures every LLM call, chain, and agent run as a structured "trace".

Architecture:
                                                ┌─────────────────┐
  Your Code                                     │   LangSmith UI  │
  ──────────                                    │  app.smith.ai   │
  pipeline.invoke()                             └────────┬────────┘
       │                                                 │
       ▼                                                 │ HTTPS
  LangGraph StateGraph                                   │
       │                                                 │
       ▼                                         ┌───────┴────────┐
  LangChain callback system  ──────────────────► │ LangSmith API  │
  (auto-hooked via env vars)                     └────────────────┘

How it hooks in:
  LangChain/LangGraph check for LANGCHAIN_TRACING_V2=true on startup.
  If set, they register a LangSmithCallbackHandler automatically — zero
  code changes needed in the pipeline itself. Every LLM call and graph
  step is wrapped and sent to the LangSmith API in a background thread.

What you see in the LangSmith UI:
  ┌─ Pipeline Run ─────────────────────────────────────────────────┐
  │  intake_node          2ms                                       │
  │  router_node          1ms                                       │
  │  summarization_node   1.2s                                      │
  │    └─ ChatOpenAI      1.1s  tokens_in:512  tokens_out:128  $0.0003 │
  │  router_node          1ms                                       │
  │  qa_node              0.8s                                      │
  │    └─ ChatOpenAI      0.7s  tokens_in:380  tokens_out:96   $0.0002 │
  └────────────────────────────────────────────────────────────────┘

IMPORTANT: LangSmith only traces LangChain/LangGraph operations.
Direct openai.Client() calls (like in SummarizationAgent) are NOT
auto-traced — we add metadata manually via run_metadata below.
"""

import os
from utils.logger import logger
from config.settings import settings


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing via environment variables.

    LangChain reads these specific env var names on import — setting them
    here (before any LangChain import) enables tracing for the whole process.

    Returns True if tracing was enabled, False if skipped (no API key).
    """
    if not settings.LANGCHAIN_API_KEY:
        logger.info("LangSmith: no LANGCHAIN_API_KEY set — tracing disabled")
        return False

    if settings.LANGCHAIN_TRACING_V2.lower() != "true":
        logger.info("LangSmith: LANGCHAIN_TRACING_V2 != true — tracing disabled")
        return False

    # LangChain reads these specific env var names — must be set before import
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = settings.LANGCHAIN_PROJECT

    logger.info(f"LangSmith: tracing enabled → project '{settings.LANGCHAIN_PROJECT}'")
    return True
