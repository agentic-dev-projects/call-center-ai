"""
Chunker: Splits transcript into meaningful chunks
"""

from typing import List


def chunk_transcript(transcript: str, max_lines: int = 4) -> List[str]:
    """
    Splits transcript into chunks based on lines (simple version)

    Future improvement:
    - Speaker-based chunking
    - Semantic chunking
    """

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