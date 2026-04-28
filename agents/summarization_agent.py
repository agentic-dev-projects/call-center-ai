"""
Summarization Agent

Responsibilities:
- Generate structured insights from transcript
"""

import json
from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, CallStatus
from config.settings import settings
from openai import OpenAI

from rag.chunker import chunk_transcript
from rag.vector_store import store_chunks
from rag.retriever import retrieve

from utils.logger import logger

from cache.semantic_cache import get_from_cache, store_in_cache


class SummarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SummarizationAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        if not record.raw_transcript:
            raise ValueError("Transcript missing")


        # Check cache first
        query = " ".join(record.raw_transcript.strip().lower().split())
        cached = get_from_cache(query)

        if cached:
            logger.info("CACHE HIT - skipping LLM call")
            record.summary = cached.get("summary")
            record.key_points = cached.get("key_points")
            record.action_items = cached.get("action_items")
            record.status = CallStatus.SUMMARIZED
            return record
        
        # ----------------------------
        # RAG PIPELINE
        # ----------------------------

        # Step 1: chunk transcript
        chunks = chunk_transcript(record.raw_transcript)

        # Step 2: store chunks in vector DB
        store_chunks(record.call_id, chunks)

        # Step 3: retrieve relevant chunks
        relevant_chunks = retrieve(record.raw_transcript)

        # Combine retrieved chunks
        if not relevant_chunks:
            logger.warning("No relevant chunks found, using full transcript")
            context = record.raw_transcript
        else:
            context = "\n".join(relevant_chunks)

        # ADD DEBUG PRINT HERE
        logger.info(f"Retrieved context: {context}")

        # Load prompt
        with open("config/prompts/summarization_v1.txt", "r") as f:
            prompt_template = f.read()
        
        formatted_template = prompt_template.format(
            transcript=record.raw_transcript
        )
        
        prompt = f"""
        Context:
        {context}

        Full Transcript:
        {record.raw_transcript}

        Instructions:
        {formatted_template}
        """

        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only response generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  
        )

        content = response.choices[0].message.content

        # Try parsing JSON output
        try:
            parsed = json.loads(content)
        except Exception:
            raise ValueError("Failed to parse LLM output as JSON")

        # Update record
        record.summary = parsed.get("summary")
        record.key_points = parsed.get("key_points")
        record.action_items = parsed.get("action_items")

        record.status = CallStatus.SUMMARIZED

        # Store result in cache
        logger.info("Storing response in cache")
        store_in_cache(query, {
            "summary": record.summary,
            "key_points": record.key_points,
            "action_items": record.action_items
        })

        return record