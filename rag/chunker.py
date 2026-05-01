"""
Chunker: Splits transcript into meaningful chunks
"""

from typing import List
from config.settings import settings


def chunk_transcript(transcript: str, max_lines: int = None) -> List[str]:
    max_lines = max_lines if max_lines is not None else settings.CHUNK_MAX_LINES

    lines = transcript.split(". ")

    chunks = []
    current_chunk = []

    for line in lines:
        current_chunk.append(line)

        if len(current_chunk) >= max_lines:
            chunks.append(". ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks
