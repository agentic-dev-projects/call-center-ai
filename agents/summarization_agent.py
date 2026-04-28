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


class SummarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SummarizationAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        if not record.raw_transcript:
            raise ValueError("Transcript missing")


        # ----------------------------
        # RAG PIPELINE
        # ----------------------------

        # Step 1: chunk transcript
        chunks = chunk_transcript(record.raw_transcript)

        # Step 2: store chunks in vector DB
        store_chunks(record.call_id, chunks)

        # Step 3: retrieve relevant chunks
        relevant_chunks = retrieve("customer issue and resolution")

        # Combine retrieved chunks
        context = "\n".join(relevant_chunks)

        # ADD DEBUG PRINT HERE
        logger.info(f"Retrieved context: {context}")

        # Load prompt
        with open("config/prompts/summarization_v1.txt", "r") as f:
            prompt_template = f.read()
        
        prompt = f"""
        Context:
        {context}

        Full Transcript:
        {record.raw_transcript}

        Instructions:
        {prompt_template}
        """

        prompt = prompt_template.format(transcript=record.raw_transcript)

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

        return record