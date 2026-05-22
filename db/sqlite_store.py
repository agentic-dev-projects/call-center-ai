"""
SQLite Persistence — Milestone 19

LEARNING: WHY SQLITE?
══════════════════════

SQLite is a file-based relational database built into Python's stdlib.
No server, no install, no connection pool — just a .db file on disk.

Perfect for:
  - Local development and demos
  - Single-process apps (our Streamlit app is single-process)
  - Apps that need SQL queries but not distributed writes

Not suitable for:
  - Multiple concurrent writers (SQLite has file-level locking)
  - Horizontal scaling / cloud ephemeral filesystems (use Postgres/Supabase)

LEARNING: THE REPOSITORY PATTERN
══════════════════════════════════

Instead of scattering SQL queries across agents and UI code, we isolate
all DB logic in one module. The pipeline and UI call functions like
save_call() and get_all_calls() — they never touch SQL directly.

Benefits:
  - Swap SQLite → Postgres by rewriting ONLY this file
  - Easy to test (mock this module in unit tests)
  - All schema changes in one place

LEARNING: sqlite3 — PYTHON'S BUILT-IN DB
══════════════════════════════════════════

Key concepts:
  connection = sqlite3.connect(path)       # open/create the .db file
  cursor = connection.cursor()             # execute SQL statements
  cursor.execute(sql, params)              # parameterised query (safe from injection)
  connection.commit()                      # persist writes
  connection.close()                       # release file lock

  row_factory = sqlite3.Row               # makes rows behave like dicts
                                          # so row["call_id"] works

SCHEMA
══════
One table: `calls`
  - call_id is the PRIMARY KEY (SHA256 hash from intake agent)
  - key_points, action_items, qa_scores stored as JSON strings
  - INSERT OR REPLACE handles re-processing the same call
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from agents.schemas import CallRecord
from config.settings import settings
from utils.logger import logger


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """
    Context manager that opens a connection, yields it, then commits + closes.

    LEARNING: CONTEXT MANAGERS FOR RESOURCE MANAGEMENT
    The `with _get_conn() as conn:` pattern guarantees the connection is
    always closed — even if an exception is raised inside the block.
    This prevents file-lock leaks that would make the DB inaccessible.

    We also set row_factory = sqlite3.Row so callers can do:
        row["call_id"]  instead of  row[0]
    """
    Path(settings.SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    call_id             TEXT PRIMARY KEY,
    input_type          TEXT,
    status              TEXT,
    agent_name          TEXT,
    customer_id         TEXT,
    duration_seconds    REAL,
    raw_transcript      TEXT,
    summary             TEXT,
    key_points          TEXT,
    action_items        TEXT,
    qa_scores           TEXT,
    error               TEXT,
    from_cache          INTEGER DEFAULT 0,
    guardrail_violations TEXT,
    guardrail_blocked   INTEGER DEFAULT 0,
    created_at          TEXT,
    updated_at          TEXT
)
"""


def init_db() -> None:
    """
    Create the calls table if it doesn't exist yet.

    LEARNING: CREATE TABLE IF NOT EXISTS is idempotent — safe to call on
    every app startup. It's a no-op if the table already exists, so you
    never need to guard it with a version check.

    Call this once at module import time (bottom of this file) so the
    table is ready before any agent tries to write to it.
    """
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
    logger.info(f"SQLite: DB ready at {settings.SQLITE_DB_PATH}")


# ── Write operations ──────────────────────────────────────────────────────────

def save_call(record: CallRecord) -> None:
    """
    Insert or update a CallRecord in the database.

    LEARNING: INSERT OR REPLACE
    If a row with the same call_id already exists, SQLite replaces it
    entirely. This means re-processing the same call overwrites the old
    result — which is the behaviour we want (idempotent saves).

    LEARNING: PARAMETERISED QUERIES (? placeholders)
    Never build SQL with f-strings: f"INSERT ... VALUES ('{record.call_id}')"
    That's SQL injection waiting to happen. Always use ? placeholders and
    pass values as a tuple — sqlite3 escapes them safely.
    """
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO calls (
                call_id, input_type, status, agent_name, customer_id,
                duration_seconds, raw_transcript, summary,
                key_points, action_items, qa_scores, error,
                from_cache, guardrail_violations, guardrail_blocked,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                COALESCE(
                    (SELECT created_at FROM calls WHERE call_id = ?),
                    ?
                ),
                ?
            )
            """,
            (
                record.call_id,
                record.input_type.value if record.input_type else None,
                record.status.value if record.status else None,
                record.agent_name,
                record.customer_id,
                record.duration_seconds,
                record.raw_transcript,
                record.summary,
                json.dumps(record.key_points) if record.key_points else None,
                json.dumps(record.action_items) if record.action_items else None,
                json.dumps(record.qa_scores) if record.qa_scores else None,
                record.error,
                int(record.from_cache),
                json.dumps(record.guardrail_violations) if record.guardrail_violations else None,
                int(record.guardrail_blocked),
                record.call_id,   # for the COALESCE subquery
                now,              # created_at on first insert
                now,              # updated_at always
            ),
        )
    logger.info(f"SQLite: saved call {record.call_id}")


# ── Read operations ───────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert a sqlite3.Row to a plain dict, deserialising JSON fields.

    LEARNING: sqlite3 stores everything as text/int/real. JSON columns
    (key_points, action_items, qa_scores) need to be decoded back to
    Python objects before the rest of the app can use them.
    """
    d = dict(row)
    for json_field in ("key_points", "action_items", "qa_scores", "guardrail_violations"):
        if d.get(json_field):
            try:
                d[json_field] = json.loads(d[json_field])
            except (json.JSONDecodeError, TypeError):
                d[json_field] = []
    d["from_cache"] = bool(d.get("from_cache", 0))
    d["guardrail_blocked"] = bool(d.get("guardrail_blocked", 0))
    return d


def get_call(call_id: str) -> Optional[dict]:
    """Fetch a single call by ID. Returns None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_all_calls(limit: int = 100) -> List[dict]:
    """
    Fetch all calls, newest first.

    LEARNING: ORDER BY updated_at DESC puts the most recent call at the top
    of the history list. LIMIT prevents the query from returning thousands
    of rows to the UI at once.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_call_count() -> int:
    """Return total number of stored calls."""
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]


def delete_call(call_id: str) -> bool:
    """Delete a call by ID. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM calls WHERE call_id = ?", (call_id,))
    return cursor.rowcount > 0


# ── Initialise on import ──────────────────────────────────────────────────────
# LEARNING: Running init_db() at module level means the table is created
# the first time anything imports this module. No separate migration step needed.
init_db()
