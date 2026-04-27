# AI Call Center Assistant

> A production-grade multi-agent AI pipeline that ingests call center audio or transcripts and produces structured summaries, QA scores, and actionable insights — built as a capstone for the *Applied Agentic AI for SWEs* curriculum.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [System architecture](#2-system-architecture)
3. [Folder structure](#3-folder-structure)
4. [Agent descriptions](#4-agent-descriptions)
5. [Tech stack](#5-tech-stack)
6. [Setup and installation](#6-setup-and-installation)
7. [Configuration](#7-configuration)
8. [Running the pipeline](#8-running-the-pipeline)
9. [Running the UI](#9-running-the-ui)
10. [RAG layer](#10-rag-layer)
11. [Semantic cache](#11-semantic-cache)
12. [Guardrails](#12-guardrails)
13. [Evaluation](#13-evaluation)
14. [LLMOps and AgentOps](#14-llmops-and-agentops)
15. [A2A communication](#15-a2a-communication)
16. [MCP control plane](#16-mcp-control-plane)
17. [Docker deployment](#17-docker-deployment)
18. [Testing](#18-testing)
19. [Prompt versioning](#19-prompt-versioning)
20. [Contributing and roadmap](#20-contributing-and-roadmap)

---

## 1. Project overview

Call centers generate thousands of hours of conversation data daily. Insights from these conversations — customer pain points, agent performance, compliance issues, resolution rates — are locked inside unstructured audio recordings and raw transcripts.

This project builds an end-to-end AI pipeline that:

- Accepts a call recording (`.mp3`, `.wav`, `.m4a`) or a pre-transcribed JSON file
- Validates, normalizes, and routes the input through a five-agent LangGraph pipeline
- Produces a structured `CallRecord` containing: full transcript, speaker-labeled segments, abstractive summary, key points, action items, and a rubric-based QA score
- Exposes everything through a Streamlit UI with upload, review, and export capabilities
- Observes every agent run with LangSmith tracing and AgentOps dashboards

### What you learn by building this

| Concept | Implementation |
|---|---|
| Multi-agent orchestration | LangGraph StateGraph + CrewAI |
| Agent-to-agent communication (A2A) | Pydantic message contracts |
| Model Context Protocol (MCP) | LiteLLM router + `mcp.yaml` |
| RAG pipeline | ChromaDB + LangChain retriever |
| Semantic caching | GPTCache + Redis |
| Guardrails | Guardrails AI + custom Pydantic validators |
| Evaluation | F1, ROUGE-L, BERTScore, RAGAS |
| LLMOps | LangSmith, AgentOps, cost tracking |

---

## 2. System architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│        Upload · Transcript · Summary · QA · Tags            │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │  MCP Control    │   mcp.yaml — routes to
                │  Plane          │   Claude / GPT-4 / fallback
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │ Semantic Cache  │   GPTCache + Redis
                │                 │   cache hit ──────────────┐
                └────────┬────────┘                           │
                         │                                    │
┌────────────────────────▼──────────────────────────────┐     │
│              LangGraph StateGraph                     │     │
│                                                       │     │
│  [Intake] → [Transcription] → [Summarization] → [QA] │     │
│                     ↕              ↕                  │     │
│              [Routing Agent — conditional edges]      │     │
│                                                       │     │
│  PipelineState (TypedDict) flows through every node  │     │
└────────────┬──────────────────────────────────────────┘     │
             │                                                │
    ┌────────┼────────────────────────────────┐              │
    │        │  Supporting layers             │              │
    │  ┌─────▼──────┐ ┌──────────┐ ┌───────┐ │              │
    │  │ RAG layer  │ │Guardrails│ │Memory │ │              │
    │  │ ChromaDB   │ │PII · val │ │Redis  │ │              │
    │  └────────────┘ └──────────┘ └───────┘ │              │
    └────────────────────────────────────────┘              │
             │                                              │
    ┌────────▼──────────────────────────────────────────────▼┐
    │            CallRecord (complete Pydantic model)         │
    │  summary · qa_scores · segments · action_items · tags   │
    └─────────────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────┐
    │         LLMOps — LangSmith · AgentOps · cost tracker    │
    │    traces every call · latency per agent · token cost   │
    └─────────────────────────────────────────────────────────┘
```

### Data flow

```
Input (audio / JSON)
  → IntakeAgent        — validates format, extracts metadata → CallRecord
  → TranscriptionAgent — Whisper API, speaker segments       → CallRecord.raw_transcript
  → SummarizationAgent — RAG retrieval + LangChain prompt    → CallRecord.summary
  → QAScoringAgent     — function calling + rubric           → CallRecord.qa_scores
  → RoutingAgent       — conditional edges, retry / escalate
  → Output             — complete CallRecord written to DB and UI
```

---

## 3. Folder structure

```
call_center_ai/
│
├── agents/                        # One file per agent
│   ├── base_agent.py              # Abstract base: run(), handle_error()
│   ├── schemas.py                 # All Pydantic models (CallRecord, etc.)
│   ├── intake_agent.py            # Validates input, detects type
│   ├── transcription_agent.py     # Whisper API, audio chunking
│   ├── summarization_agent.py     # LangChain + RAG context
│   ├── qa_scoring_agent.py        # Function calling + rubric
│   ├── routing_agent.py           # LangGraph conditional edges
│   └── __init__.py
│
├── pipeline/                      # LangGraph orchestration
│   ├── graph.py                   # StateGraph definition, node wiring
│   ├── state.py                   # PipelineState TypedDict
│   └── orchestrator.py            # run_pipeline() entry point
│
├── rag/                           # Retrieval-Augmented Generation
│   ├── chunker.py                 # Speaker-turn chunking strategy
│   ├── embedder.py                # OpenAI / sentence-transformer embeddings
│   ├── vector_store.py            # ChromaDB wrapper (local + Pinecone)
│   └── retriever.py               # Query-time retrieval, reranking
│
├── utils/                         # Shared utilities
│   ├── audio_preprocessor.py      # ffmpeg: 16kHz mono, chunking
│   ├── cache.py                   # GPTCache setup, semantic lookup
│   └── logger.py                  # Structured JSON logging
│
├── guardrails/                    # Input and output safety
│   ├── input_guard.py             # PII detection, format validation
│   └── output_guard.py            # Schema validation, hallucination check
│
├── evaluation/                    # Eval harness
│   ├── metrics.py                 # F1, ROUGE-L, BERTScore
│   ├── ragas_eval.py              # RAGAS: faithfulness, relevance
│   └── run_evals.py               # Entry point: runs all evals on a dataset
│
├── ops/                           # LLMOps and observability
│   ├── langsmith_setup.py         # LangSmith client, project config
│   ├── agentops_setup.py          # AgentOps init, session tagging
│   └── cost_tracker.py            # Per-agent token cost accumulator
│
├── ui/                            # Streamlit application
│   ├── app.py                     # Entry point: st.set_page_config, routing
│   ├── pages/
│   │   ├── 01_upload.py           # File upload page
│   │   ├── 02_review.py           # Transcript + summary review
│   │   └── 03_analytics.py        # QA scores dashboard, trends
│   └── components/
│       ├── transcript_viewer.py   # Speaker-labeled transcript component
│       ├── score_card.py          # QA score visualization
│       └── summary_panel.py       # Summary + action items display
│
├── config/
│   ├── settings.py                # Pydantic BaseSettings (reads .env)
│   ├── mcp.yaml                   # Model routing rules
│   └── prompts/
│       ├── summarization_v1.txt   # Versioned prompt templates
│       ├── summarization_v2.txt
│       └── qa_rubric_v1.txt
│
├── data/
│   ├── sample_transcripts/        # 10–15 JSON test transcripts
│   ├── sample_audio/              # Short audio files for testing
│   └── eval_references/           # Human reference summaries for eval
│
├── tests/
│   ├── unit/
│   │   ├── test_intake_agent.py
│   │   ├── test_transcription_agent.py
│   │   ├── test_summarization_agent.py
│   │   └── test_qa_scoring_agent.py
│   ├── integration/
│   │   └── test_full_pipeline.py
│   └── fixtures/
│       └── sample_call_record.py
│
├── .env.example                   # Environment variable template
├── pyproject.toml                 # Dependencies (uv / pip)
├── docker-compose.yml             # Services: app, redis, chromadb
├── Makefile                       # Common dev commands
└── README.md                      # This file
```

---

## 4. Agent descriptions

### Call Intake Agent (`agents/intake_agent.py`)

**Responsibility:** First contact point for all inputs. Detects input type (audio vs. JSON), validates format and size constraints, extracts available metadata, and produces an initial `CallRecord` with `status=PENDING`.

**Key behaviors:**
- Rejects audio files over 25MB with a descriptive error
- Supports `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`
- Generates a deterministic `call_id` via SHA-256 hash of the input
- For JSON inputs, validates the presence of required fields (`transcript`)
- All validation errors raise `IntakeValidationError` — never silently continue

### Transcription Agent (`agents/transcription_agent.py`)

**Responsibility:** Converts audio `CallRecord`s to text. JSON transcripts skip this agent entirely via LangGraph routing.

**Key behaviors:**
- Preprocesses audio to 16kHz mono WAV via ffmpeg before API call
- Chunks audio longer than 10 minutes into overlapping segments
- Calls `whisper-1` with `verbose_json` to capture per-segment timestamps
- Applies heuristic speaker diarization (gap-based speaker flipping)
- For production: integrate `pyannote-audio` or AssemblyAI's `speaker_labels`

### Summarization Agent (`agents/summarization_agent.py`)

**Responsibility:** Generates structured summaries grounded in retrieved context from the vector store.

**Key behaviors:**
- Retrieves the top-5 most relevant chunks from ChromaDB using the transcript as query
- Constructs a LangChain prompt with retrieved context + raw transcript
- Produces structured output: `summary`, `key_points`, `action_items`, `call_category`
- All output is validated against a Pydantic schema before writing to state

### QA Scoring Agent (`agents/qa_scoring_agent.py`)

**Responsibility:** Evaluates agent performance using a structured rubric via LLM function calling.

**Rubric dimensions (each scored 1–5):**
- `empathy` — did the agent acknowledge the customer's feelings?
- `resolution` — was the customer's issue resolved?
- `tone` — was the agent's tone professional and calm?
- `professionalism` — were policies followed correctly?

**Key behaviors:**
- Uses OpenAI function calling to enforce structured JSON output
- Computes `overall_score` as a weighted average
- Flags calls where any dimension scores below 3 for supervisor review

### Routing Agent (`agents/routing_agent.py`)

**Responsibility:** LangGraph conditional edge logic — decides what happens after QA scoring.

**Routing logic:**
- `overall_score >= 4.0` → write to output, mark `SCORED`
- `overall_score < 2.0` → escalate flag + retry summarization once
- `status == FAILED` → log error, route to dead-letter queue
- `retries > 2` → force-complete with error flag, never infinite loop

---

## 5. Tech stack

| Layer | Primary | Alternatives |
|---|---|---|
| Orchestration | LangGraph | CrewAI (agent roles) |
| Language models | Claude Sonnet 4.5, GPT-4 | Gemini, Mistral |
| Model routing | LiteLLM + `mcp.yaml` | LangSmith router |
| Transcription | OpenAI Whisper API | Deepgram, AssemblyAI |
| RAG | LangChain + ChromaDB | Pinecone (production) |
| Embeddings | `text-embedding-3-small` | sentence-transformers |
| Semantic cache | GPTCache + Redis | Redis alone |
| Guardrails | Guardrails AI + Pydantic | NeMo Guardrails |
| Evaluation | RAGAS, `evaluate` (HF) | Ragas cloud |
| LLMOps | LangSmith | Helicone, Phoenix |
| AgentOps | AgentOps SDK | Custom logging |
| UI | Streamlit | Gradio, Flask |
| Deployment | Docker Compose | Streamlit Cloud |
| Language | Python 3.11+ | — |

---

## 6. Setup and installation

### Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on `PATH`
- Docker + Docker Compose (for Redis and ChromaDB services)
- API keys: OpenAI (Whisper + GPT-4), Anthropic (Claude), LangSmith, AgentOps

### Install

```bash
git clone https://github.com/your-org/call-center-ai.git
cd call-center-ai

# Using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -e ".[dev]"
```

### Verify ffmpeg

```bash
ffmpeg -version
ffprobe -version
```

### Start backing services

```bash
docker-compose up -d redis chromadb
```

---

## 7. Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=call-center-ai

AGENTOPS_API_KEY=...

REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8000

LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Model routing (`config/mcp.yaml`)

```yaml
# config/mcp.yaml
default_model: claude-sonnet-4-5
fallback_model: gpt-4o-mini
budget_limit_usd_per_call: 0.10

routes:
  intake:
    model: claude-haiku-4-5      # fast, cheap — no generation needed
    max_tokens: 256
  transcription:
    model: whisper-1             # fixed, no routing
  summarization:
    model: claude-sonnet-4-5
    max_tokens: 1024
    temperature: 0.3
  qa_scoring:
    model: gpt-4o                # function calling reliability
    max_tokens: 512
    temperature: 0.0
  routing:
    model: claude-haiku-4-5
    max_tokens: 128
```

---

## 8. Running the pipeline

### Single call (CLI)

```python
from pipeline.orchestrator import run_pipeline

# From an audio file
result = run_pipeline("data/sample_audio/call_001.mp3")

# From a JSON transcript
result = run_pipeline({
    "transcript": "Agent: Thank you for calling. Customer: My bill is wrong.",
    "agent_name": "Sarah",
    "customer_id": "C-4892",
    "duration_seconds": 183.0,
})

print(result["summary"])
print(result["qa_scores"])
```

### Batch processing

```bash
python pipeline/orchestrator.py --batch data/sample_transcripts/ --output results/
```

### Makefile shortcuts

```bash
make run-single FILE=data/sample_audio/call_001.mp3
make run-batch DIR=data/sample_transcripts/
make run-ui
make run-evals
make test
make lint
```

---

## 9. Running the UI

```bash
streamlit run ui/app.py
```

The UI has three pages:

**Upload** — drag-and-drop audio or JSON, shows real-time pipeline progress per agent.

**Review** — side-by-side view of the transcript (speaker-labeled) and the generated summary, key points, and action items. Allows human correction.

**Analytics** — QA score breakdown per dimension (bar chart), score trends over time, flagged calls list, category distribution.

---

## 10. RAG layer

The RAG layer ensures summaries are grounded in actual transcript content rather than relying on the LLM's tendency to hallucinate details.

### How it works

1. After transcription, the `Chunker` splits the transcript into speaker-turn chunks (each chunk = one complete speaker turn, max 512 tokens).
2. The `Embedder` converts each chunk to a vector using `text-embedding-3-small`.
3. Chunks are stored in ChromaDB with metadata: `call_id`, `speaker`, `timestamp_start`, `timestamp_end`.
4. At summarization time, the `Retriever` queries ChromaDB with the full transcript as the query and returns the top-5 most relevant chunks.
5. These chunks are injected into the summarization prompt as `<context>` blocks.

### Populating the vector store

```bash
python rag/vector_store.py --ingest data/sample_transcripts/
```

### ChromaDB collection schema

```python
{
    "id": "call_001_chunk_03",
    "document": "Customer: My internet has been down for three days...",
    "metadata": {
        "call_id": "a3f8c2d1",
        "speaker": "Customer",
        "chunk_index": 3,
        "timestamp_start": 45.2,
        "timestamp_end": 58.7,
    }
}
```

---

## 11. Semantic cache

The semantic cache sits between the MCP control plane and the LLM. When a call comes in that is semantically similar to a previously processed call (cosine similarity > 0.92), the cached `CallRecord` is returned immediately — no LLM calls made.

### Cache key strategy

The cache key is the embedding of the first 500 tokens of the transcript. Two calls about "billing dispute on internet service" from different customers will share a cache entry for the summarization step if their transcripts are similar enough.

### Configuration

```python
# utils/cache.py
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation import SearchDistanceEvaluation

cache.init(
    embedding_func=Onnx().to_embeddings,
    data_manager=get_data_manager(
        CacheBase("sqlite"),
        VectorBase("redis", host="localhost", port=6379)
    ),
    similarity_evaluation=SearchDistanceEvaluation(),
    similarity_threshold=0.92,
)
```

---

## 12. Guardrails

### Input guardrails (`guardrails/input_guard.py`)

Run before the transcript reaches any LLM:

- **PII detection** — scans for phone numbers, SSNs, credit card numbers, email addresses using regex + `presidio-analyzer`. Redacts or flags before storage.
- **Format validation** — confirms transcript is valid UTF-8, not empty, not excessively short (< 50 words).
- **Language check** — flags non-English transcripts for human review (configurable).

### Output guardrails (`guardrails/output_guard.py`)

Run after the Summarization Agent before writing to `CallRecord`:

- **Schema validation** — Pydantic models enforce types and required fields; LLM cannot produce a malformed output.
- **Hallucination check** — RAGAS `faithfulness` score checks whether every claim in the summary is supported by the source transcript. Summaries with faithfulness < 0.85 are flagged.
- **Length sanity** — summary must be between 50 and 500 words.

---

## 13. Evaluation

### Running evals

```bash
python evaluation/run_evals.py --dataset data/eval_references/ --output results/eval_report.json
```

### Metrics

| Metric | What it measures | Target |
|---|---|---|
| ROUGE-L | N-gram overlap between generated and reference summaries | > 0.45 |
| BERTScore F1 | Semantic similarity to reference summaries | > 0.88 |
| RAGAS Faithfulness | Are summary claims supported by the transcript? | > 0.85 |
| RAGAS Answer Relevancy | Is the summary relevant to the call topic? | > 0.80 |
| QA F1 | Precision/Recall on QA rubric vs. human labels | > 0.75 |

### QA scoring F1 calculation

The QA Scoring Agent produces binary flags per rubric dimension (pass/fail at threshold 3.0). Human annotators label the same calls. F1 is calculated per dimension:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

Where TP = agent and human both flag, FP = agent flags, human does not, FN = human flags, agent does not.

---

## 14. LLMOps and AgentOps

### LangSmith

Every LangGraph node automatically emits a trace to LangSmith when `LANGCHAIN_TRACING_V2=true`. Each trace shows:

- Input and output for every agent
- Token usage per LLM call
- Latency per node
- Which prompt template version was used
- Full chain of agent handoffs

View traces at [smith.langchain.com](https://smith.langchain.com).

### AgentOps

AgentOps provides agent-level dashboards beyond what LangSmith covers:

```python
# ops/agentops_setup.py
import agentops

agentops.init(
    api_key=os.getenv("AGENTOPS_API_KEY"),
    tags=["call-center", "production"],
)
```

Key AgentOps dashboards for this project:
- Routing Agent fallback rate (target < 5%)
- Transcription Agent average latency (target < 30s per minute of audio)
- Cache hit rate (target > 30% in steady state)
- Cost per call (target < $0.05)

### Cost tracking

```python
# ops/cost_tracker.py
# Accumulates token costs per agent per call_id
# Alerts if total cost per call exceeds mcp.yaml budget_limit_usd_per_call
```

---

## 15. A2A communication

Every agent communicates exclusively through typed Pydantic messages. No agent reads or writes raw dicts. The `CallRecord` is the message envelope — it flows through every agent as the shared state object.

Agent outputs are validated before being written back to `PipelineState`. If an agent produces an invalid output, the `RoutingAgent` catches the `ValidationError` and routes to fallback rather than crashing the pipeline.

```python
# Example A2A contract: SummarizationAgent output
class SummaryPayload(BaseModel):
    summary: str = Field(..., min_length=50, max_length=2000)
    key_points: list[str] = Field(..., min_items=1, max_items=10)
    action_items: list[str] = Field(default_factory=list)
    call_category: Literal["billing", "technical", "complaint", "inquiry", "other"]
    confidence: float = Field(..., ge=0.0, le=1.0)
```

---

## 16. MCP control plane

The MCP (Model Context Protocol) control plane is implemented as a LiteLLM router configured by `config/mcp.yaml`. It provides:

- **Per-agent model assignment** — intake uses a cheap fast model, summarization uses a capable model
- **Fallback chains** — if Claude is unavailable, route to GPT-4; if GPT-4 is unavailable, route to GPT-4o-mini
- **Budget enforcement** — abort and flag calls that exceed `budget_limit_usd_per_call`
- **Rate limit handling** — exponential backoff with jitter, automatic retry

```python
# config/settings.py — reads mcp.yaml at startup
from litellm import Router

def build_router(config: dict) -> Router:
    return Router(
        model_list=[
            {"model_name": "summarizer", "litellm_params": {"model": config["routes"]["summarization"]["model"]}},
            {"model_name": "summarizer-fallback", "litellm_params": {"model": config["default_model"]}},
        ],
        fallbacks=[{"summarizer": ["summarizer-fallback"]}],
        num_retries=3,
        retry_after=5,
    )
```

---

## 17. Docker deployment

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [redis, chromadb]
    command: streamlit run ui/app.py --server.port 8501

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: ["chroma_data:/chroma/.chroma"]

volumes:
  redis_data:
  chroma_data:
```

```bash
# Build and run everything
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f app
```

---

## 18. Testing

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running Redis + ChromaDB)
pytest tests/integration/ -v

# With coverage
pytest --cov=agents --cov=pipeline --cov-report=html
```

### Writing agent tests

Each agent test follows the same pattern: construct a `CallRecord` fixture, call `agent.run(record)`, assert on the output fields.

```python
# tests/unit/test_intake_agent.py
from agents.intake_agent import CallIntakeAgent, IntakeValidationError
from agents.schemas import InputType

def test_valid_json_transcript():
    agent = CallIntakeAgent()
    result = agent.run({"transcript": "Agent: Hello. Customer: My bill is wrong."})
    assert result.input_type == InputType.JSON_TRANSCRIPT
    assert result.call_id is not None
    assert result.raw_transcript is not None

def test_rejects_empty_transcript():
    agent = CallIntakeAgent()
    with pytest.raises(IntakeValidationError):
        agent.run({"transcript": ""})
```

---

## 19. Prompt versioning

All LLM prompts live in `config/prompts/` as plain text files. The filename encodes the version: `summarization_v1.txt`, `summarization_v2.txt`.

The active version is set in `config/settings.py`:

```python
SUMMARIZATION_PROMPT_VERSION = "v2"
```

LangSmith automatically records which prompt version was active for each trace, enabling before/after comparison when you update a prompt.

To A/B test two prompt versions on the same dataset:

```bash
python evaluation/run_evals.py --prompt-version v1 --dataset data/eval_references/
python evaluation/run_evals.py --prompt-version v2 --dataset data/eval_references/
# Compare ROUGE-L and BERTScore between the two runs
```

---

## 20. Contributing and roadmap

### Current state (Week 1 complete)

- [x] `CallRecord` Pydantic schema
- [x] `CallIntakeAgent` with full validation
- [x] `TranscriptionAgent` with Whisper API + chunking
- [x] Audio preprocessor (ffmpeg)
- [x] Sample transcripts in `data/`

### Week 2 targets

- [ ] LangGraph `StateGraph` wiring all five agents
- [ ] `SummarizationAgent` with RAG context
- [ ] `QAScoringAgent` with function calling
- [ ] `RoutingAgent` with conditional edges and fallback
- [ ] ChromaDB vector store + retriever
- [ ] GPTCache + Redis semantic cache
- [ ] Guardrails AI integration (PII + output validation)
- [ ] Streamlit UI — all three pages
- [ ] LangSmith tracing + AgentOps setup
- [ ] Evaluation harness (ROUGE, BERTScore, RAGAS, F1)
- [ ] Docker Compose deployment

### Future enhancements

- Real-time streaming transcription via WebSocket
- pyannote-audio speaker diarization (replace heuristic)
- Pinecone migration for production vector store
- Slack / webhook integration for flagged call alerts
- Batch processing API endpoint (FastAPI)
- Multi-language support with automatic language detection routing

---

## License

MIT License — see `LICENSE` for details.

---

*Built as part of the Interview Kickstart Applied Agentic AI for SWEs curriculum.*
