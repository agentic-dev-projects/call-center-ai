"""
Chunker: Splits transcript into meaningful chunks

LEARNING: WHY CHUNKING MATTERS
════════════════════════════════

RAG retrieval quality is highly sensitive to chunk boundaries. Two problems
with the original M5 chunker:

  1. Hard sentence splits lose cross-sentence context.
     "The customer was double-charged. The agent confirmed the error."
     Split into separate chunks → the second chunk is meaningless without the first.

  2. Speaker turns are ignored.
     A chunk mixing Agent speech and Customer speech confuses the retriever
     because the two have different intent (instruction vs. complaint).

M18 fixes:

  OVERLAP CHUNKING
  ─────────────────
  Each chunk includes the last `overlap` sentences of the previous chunk.
  Ensures that context at chunk boundaries is preserved on both sides.

  Example (chunk_size=3, overlap=1):
    Chunk 1: [S1, S2, S3]
    Chunk 2: [S3, S4, S5]   ← S3 repeated for context
    Chunk 3: [S5, S6, S7]

  SPEAKER-TURN CHUNKING
  ──────────────────────
  Detect "Agent:" / "Customer:" speaker prefixes and split there.
  Each turn becomes its own unit before further splitting by size.
  This keeps one speaker's complete thought in the same chunk.
"""

import re
from typing import List
from config.settings import settings


# ── Overlap chunker (default) ─────────────────────────────────────────────────

def chunk_transcript(
    transcript: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split a transcript into overlapping chunks.

    LEARNING: OVERLAP IS A SLIDING WINDOW
    Each chunk advances by (chunk_size - overlap) sentences, so context
    bleeds into the next chunk. The trade-off: more chunks = more storage
    and retrieval cost, but better recall at boundaries.

    Args:
        transcript: raw text
        chunk_size: max sentences per chunk (default: settings.CHUNK_MAX_LINES)
        overlap:    sentences shared between adjacent chunks (default: settings.CHUNK_OVERLAP)
    """
    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_MAX_LINES
    overlap    = overlap    if overlap    is not None else settings.CHUNK_OVERLAP

    # Split on sentence boundaries
    sentences = _split_sentences(transcript)
    if not sentences:
        return [transcript]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)   # how far to advance each window

    i = 0
    while i < len(sentences):
        window = sentences[i : i + chunk_size]
        chunks.append(" ".join(window))
        i += step

    return chunks


def chunk_by_speaker(
    transcript: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split transcript by speaker turns, then apply overlap chunking within
    each turn.

    LEARNING: SPEAKER-TURN AWARENESS
    Call center transcripts have a predictable structure:
      Agent: <instruction / response>
      Customer: <complaint / question>
    Mixing these in one chunk teaches the retriever to conflate agent-speak
    with customer-speak. Splitting by turn keeps intent clean.

    Pattern matched: lines starting with "Agent:", "Customer:", "Rep:", etc.
    """
    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_MAX_LINES
    overlap    = overlap    if overlap    is not None else settings.CHUNK_OVERLAP

    # Detect speaker-prefixed lines
    speaker_re = re.compile(
        r"^(Agent|Customer|Rep|Representative|Caller|Support|Supervisor)\s*:",
        re.IGNORECASE | re.MULTILINE,
    )

    # Check if the transcript uses speaker labels at all
    if not speaker_re.search(transcript):
        # Fall back to plain overlap chunking
        return chunk_transcript(transcript, chunk_size, overlap)

    # Split into turns
    turns = _split_turns(transcript, speaker_re)
    chunks: List[str] = []

    for turn_text in turns:
        # Each turn is chunked independently
        turn_chunks = chunk_transcript(turn_text, chunk_size, overlap)
        chunks.extend(turn_chunks)

    return chunks if chunks else [transcript]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?'  keeping punctuation."""
    # Simple but effective for call-center transcripts
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_turns(transcript: str, speaker_re: re.Pattern) -> List[str]:
    """
    Split the transcript into per-speaker turns.

    LEARNING: re.split() with a capturing group keeps the delimiter in the
    result list. We zip adjacent pairs to reconstruct "Speaker: text" chunks.
    """
    parts = speaker_re.split(transcript)
    # parts = ['', 'Agent', ' Hello...', 'Customer', ' I have...', ...]
    # Skip leading empty string, then pair (speaker, text)
    turns: List[str] = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        body    = parts[i + 1].strip()
        if body:
            turns.append(f"{speaker}: {body}")
        i += 2
    return turns
