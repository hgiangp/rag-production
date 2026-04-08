# CLAUDE.md — Production RAG Chatbot System

> This file is the single source of truth for AI-assisted development.
> Every AI agent (Claude, Copilot, Cursor) MUST read this before writing code.
> Rules here override any default conventions from training data.

---

## 1. Project Purpose

A **production-ready RAG (Retrieval-Augmented Generation) chatbot** combining:
- **Agentic RAG pipeline** (LangGraph orchestration, hierarchical indexing, self-correction)
- **LlamaIndex complement** (sentence-window, auto-merging, reranking — exposed as LangGraph tools)
- **FastAPI backend** (JWT auth, streaming, rate limiting, OpenAI-compatible API)
- **Full observability** (Langfuse LLM traces + Prometheus metrics + Grafana dashboards)
- **RAG Triad evaluation** (Context Relevance + Groundedness + Answer Relevance, auto-scored post-response)
- **Long-term memory** (mem0ai + pgvector, per-user cross-session)
- **Streamlit UI** (MVP) with Next.js path (professional)

**Goal**: Enable GitHub Copilot, Claude, and other AI tools to autonomously implement, test, evaluate, and improve the system with minimal human intervention.

---

## 2. Architecture Overview

```
UI (Streamlit / Next.js)
        │ HTTP / SSE
        ▼
API Layer (FastAPI)
  ├── POST /api/v1/chat          → streaming chat
  ├── POST /api/v1/chat/invoke   → non-streaming chat
  ├── POST /api/v1/documents     → ingest documents
  ├── GET  /api/v1/eval/results  → evaluation results
  └── POST /api/v1/auth/*        → login / refresh
        │
        ▼
LangGraph Main Graph
  START
  ├── load_user_memory     (mem0ai: inject user context)
  ├── summarize_history    (LLM: compress past turns)
  ├── rewrite_query        (LLM: clarify + decompose → N sub-questions)
  ├── [interrupt] request_clarification  (human-in-the-loop)
  ├── [parallel] Agent Subgraph × N      (map over sub-questions)
  │     ├── orchestrator   (LLM with tools)
  │     ├── tools          (RAG search tools)
  │     ├── should_compress_context
  │     └── compress_context / fallback_response
  ├── aggregate_answers    (LLM: merge N answers)
  ├── save_user_memory     (mem0ai: store new facts)
  └── END
        │ (async, non-blocking)
        ▼
Evaluation Pipeline
  ├── context_relevance    (LLM-as-judge)
  ├── groundedness         (LLM-as-judge)
  └── answer_relevance     (LLM-as-judge)
  → scores logged to Langfuse + Prometheus

RAG Tools (called by Agent Subgraph)
  ├── Tool A: search_child_chunks      (Qdrant: precision search)
  ├── Tool B: fetch_parent_chunks      (Qdrant: context retrieval)
  ├── Tool C: llamaindex_query         (LlamaIndex: sentence-window + reranking)
  └── Tool D: web_search               (DuckDuckGo, MCP-extensible)

Persistence Layer
  ├── Qdrant          → document vectors + embeddings
  ├── PostgreSQL      → users, sessions, LangGraph checkpoints
  └── pgvector        → mem0ai long-term memory vectors

Observability
  ├── Langfuse        → LLM traces, tool calls, eval scores
  ├── Prometheus      → latency, token counts, eval scores (gauges)
  └── Grafana         → dashboards
```

---

## 3. Complete Module Map

Every file in the project, what it does, and its key exports.

### Infrastructure
| File | Purpose | Key exports |
|------|---------|-------------|
| `pyproject.toml` | uv dependencies + ruff config | — |
| `docker-compose.yml` | Full stack: postgres, qdrant, app, prometheus, grafana | — |
| `Makefile` | Dev commands: dev, test, eval, lint, docker-up | — |
| `.env.example` | All required env vars with descriptions | — |
| `schema.sql` | PostgreSQL schema: users, sessions | — |

### `app/` — FastAPI Application
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/main.py` | FastAPI app factory, lifespan, middleware mount | `app` |
| `app/core/config.py` | Pydantic Settings, env-based config | `settings` |
| `app/core/logging.py` | structlog setup, JSON + console renderer | `logger`, `get_logger()` |
| `app/core/metrics.py` | Prometheus metric definitions | `REQUEST_LATENCY`, `RAG_TRIAD_SCORE`, `LLM_TOKENS`, `RETRIEVAL_LATENCY` |
| `app/core/middleware.py` | Correlation ID injection, latency recording | `CorrelationIdMiddleware`, `LatencyMiddleware` |
| `app/core/auth.py` | JWT creation/verification, password hashing | `create_access_token()`, `get_current_session()` |
| `app/core/limiter.py` | slowapi rate limiter instance + key functions | `limiter`, `get_remote_address()` |

### `app/schemas/` — Pydantic Request/Response Models
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/schemas/auth.py` | LoginRequest, TokenResponse, RegisterRequest | — |
| `app/schemas/chat.py` | ChatRequest, ChatResponse, StreamChunk, Message, Citation | — |
| `app/schemas/ingest.py` | IngestRequest, IngestResponse, DocumentStatus | — |
| `app/schemas/eval.py` | EvalRequest, TriadScores, EvalResponse | — |
| `app/schemas/graph.py` | GraphState TypedDict (mirrors `app/graph/state.py`) | `GraphState` |

### `app/models/` — SQLModel Database Models
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/models/user.py` | User ORM model | `User` |
| `app/models/session.py` | Session ORM model | `Session` |

### `app/api/v1/` — API Endpoints
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/api/v1/auth.py` | POST /login, POST /register, POST /refresh | `router` |
| `app/api/v1/chat.py` | POST /chat (stream), POST /chat/invoke | `router` |
| `app/api/v1/ingest.py` | POST /documents, DELETE /documents/{id} | `router` |
| `app/api/v1/eval.py` | POST /eval/run, GET /eval/results | `router` |

### `app/graph/` — LangGraph Orchestration
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/graph/state.py` | GraphState TypedDict + AgentState TypedDict | `GraphState`, `AgentState` |
| `app/graph/main_graph.py` | Full pipeline graph factory | `create_main_graph()` |
| `app/graph/agent_subgraph.py` | Agent reasoning subgraph factory | `create_agent_subgraph()` |
| `app/graph/edges.py` | All routing/conditional edge functions | `route_after_rewrite()`, `route_after_orchestrator()` |
| `app/graph/nodes/memory.py` | load_user_memory, save_user_memory nodes | — |
| `app/graph/nodes/query.py` | summarize_history, rewrite_query, request_clarification | — |
| `app/graph/nodes/retrieval.py` | Tool dispatch nodes, should_compress_context | — |
| `app/graph/nodes/generation.py` | orchestrator, aggregate_answers, fallback_response, collect_answer | — |

### `app/rag/` — RAG Engine
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/rag/langchain/indexer.py` | Hierarchical chunking: parent (header-based) + child (fixed-size) → Qdrant | `HierarchicalIndexer` |
| `app/rag/langchain/retriever.py` | Child similarity search → parent fetch | `HierarchicalRetriever` |
| `app/rag/langchain/tools.py` | LangGraph @tool wrappers: search_child_chunks, fetch_parent_chunks | `get_langchain_tools()` |
| `app/rag/llamaindex/indexer.py` | LlamaIndex sentence-window + auto-merging indexing pipeline | `LlamaIndexer` |
| `app/rag/llamaindex/query_engine.py` | Query engine with reranking (Cohere/BGE) + metadata filter | `LlamaQueryEngine` |
| `app/rag/llamaindex/tools.py` | LangGraph @tool wrapper: llamaindex_query | `get_llamaindex_tools()` |

### `app/evaluation/` — RAG Triad Evaluation
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/evaluation/schemas.py` | TriadScores, EvalResult, EvalReport dataclasses | — |
| `app/evaluation/triad.py` | LLM-as-judge: context_relevance, groundedness, answer_relevance | `RAGTriadEvaluator` |
| `app/evaluation/runner.py` | Async post-response eval runner (non-blocking fire-and-forget) | `EvalRunner` |

### `app/memory/` — Memory Management
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/memory/longterm.py` | mem0ai AsyncMemory wrapper: add, search, delete per user | `LongTermMemory` |
| `app/memory/shortterm.py` | AsyncPostgresSaver factory for LangGraph checkpointing | `get_checkpointer()` |

### `app/services/` — External Service Clients
| File | Purpose | Key exports |
|------|---------|-------------|
| `app/services/llm.py` | LLM factory: OpenAI/Anthropic/Ollama with retry + Langfuse callback | `llm_service`, `get_llm()` |
| `app/services/embedding.py` | Embedding factory: OpenAI/HuggingFace/Ollama | `embedding_service`, `get_embedding()` |
| `app/services/vector_store.py` | Qdrant client + collection management | `vector_store_service` |

### `evals/` — Batch Evaluation
| File | Purpose |
|------|---------|
| `evals/golden_dataset.json` | Test Q&A pairs with expected citations |
| `evals/run_eval.py` | CLI: run batch eval, generate report |
| `evals/metrics/context_relevance.md` | LLM judge prompt for context relevance |
| `evals/metrics/groundedness.md` | LLM judge prompt for groundedness |
| `evals/metrics/answer_relevance.md` | LLM judge prompt for answer relevance |

### `ui/` — User Interfaces
| File | Purpose |
|------|---------|
| `ui/streamlit/app.py` | Streamlit chat UI: upload docs, chat, show citations + triad scores |

---

## 4. Critical Rules (AI MUST follow)

### R1 — Async First
```python
# CORRECT
async def get_user(user_id: str, db: AsyncSession) -> User: ...

# WRONG — blocks event loop
def get_user(user_id: str, db: Session) -> User: ...
```

### R2 — All Imports at File Top
```python
# CORRECT — top of file
from app.core.config import settings
from app.services.llm import get_llm

def some_function():
    llm = get_llm()

# WRONG — import inside function
def some_function():
    from app.core.config import settings  # ← NEVER
```

### R3 — Structured Logging (structlog), Never f-strings in Events
```python
# CORRECT
logger.info("chat_request_received", user_id=user_id, session_id=session_id, message_count=len(messages))

# WRONG
logger.info(f"Chat request received for user {user_id}")  # ← NEVER use f-string as event
```

### R4 — LangGraph Nodes are Pure Functions
```python
# CORRECT — node receives state, returns state delta
async def summarize_history(state: GraphState, llm: BaseChatModel) -> dict:
    # ... logic
    return {"conversation_summary": summary}

# WRONG — node has side effects on external state
async def summarize_history(state: GraphState):
    global_cache["summary"] = summary  # ← NEVER
```

### R5 — LlamaIndex is ONLY Exposed via LangGraph Tool
```python
# CORRECT — LlamaIndex called through tool
@tool("llamaindex_query")
async def llamaindex_query(query: str) -> str:
    return await llama_engine.query(query)

# WRONG — calling LlamaIndex directly from API endpoint
@router.post("/chat")
async def chat(request: ChatRequest):
    result = await llama_engine.query(request.message)  # ← NEVER bypass LangGraph
```

### R6 — Every LLM Call Has Langfuse Callback
```python
# CORRECT
config = {"callbacks": [langfuse_handler], "run_name": "rewrite_query"}
response = await llm.ainvoke(messages, config=config)

# WRONG — no tracing
response = await llm.ainvoke(messages)  # ← always missing trace
```

### R7 — RAG Evaluation is Fire-and-Forget (Never Blocks Response)
```python
# CORRECT — run eval async, don't await in response path
async def chat_endpoint(request):
    response = await graph.ainvoke(state)
    asyncio.create_task(eval_runner.run(query, contexts, response))  # ← non-blocking
    return response

# WRONG — eval blocks the API response
async def chat_endpoint(request):
    response = await graph.ainvoke(state)
    await eval_runner.run(query, contexts, response)  # ← blocks user!
    return response
```

### R8 — Retry with Tenacity for All LLM Calls
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_llm(messages): ...
```

### R9 — Rate Limiting on Every Route
```python
@router.post("/chat")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"])
async def chat(request: Request, ...): ...
```

### R10 — Source Citations are Mandatory
Every RAG response must include citations. The `ChatResponse.citations` field is NEVER empty if documents were retrieved.

### R11 — Config via Settings, Never os.getenv() in App Code
```python
# CORRECT
from app.core.config import settings
model = settings.DEFAULT_LLM_MODEL

# WRONG
import os
model = os.getenv("DEFAULT_LLM_MODEL")  # ← only in config.py itself
```

### R12 — Self-Correction Max 3 Iterations
The agent subgraph must stop re-querying after `settings.MAX_SELF_CORRECTION_ITERATIONS` (default: 3). Exceed → `fallback_response` node.

### R13 — Prometheus Metric Names (NEVER rename these)
```python
REQUEST_LATENCY     = "rag_request_duration_seconds"
RETRIEVAL_LATENCY   = "rag_retrieval_duration_seconds"
LLM_TOKENS_TOTAL    = "rag_llm_tokens_total"
RAG_TRIAD_SCORE     = "rag_eval_triad_score"        # labels: metric=[context_relevance|groundedness|answer_relevance]
ACTIVE_SESSIONS     = "rag_active_sessions_total"
```

### R14 — GraphState Fields (NEVER add fields without updating this file)
See `app/graph/state.py`. Current fields:
- `messages`: list[BaseMessage] — conversation history
- `conversation_summary`: str — compressed history
- `original_query`: str — user's raw question
- `rewritten_questions`: list[str] — decomposed sub-questions (1 or more)
- `question_is_clear`: bool — whether query needed clarification
- `agent_answers`: list[str] — answers from each agent subgraph
- `final_answer`: str — aggregated final response
- `retrieved_contexts`: list[str] — all context chunks used
- `source_citations`: list[Citation] — document citations for response
- `user_id`: str — current user
- `session_id`: str — current session
- `correlation_id`: str — request correlation ID
- `self_correction_count`: int — iterations of self-correction

`AgentState` additional fields (agent subgraph only):
- `question`: str — the sub-question this agent is answering
- `context_summary`: str — compressed context from prior tool calls
- `answer`: str — this agent's final answer
- `correlation_id`: str — propagated from GraphState for Langfuse span linking inside tools

### R15 — Evaluation Uses a Separate LLM (settings.EVALUATION_LLM)
The inference LLM and evaluation LLM are always different instances to avoid self-serving bias.

---

## 5. LangGraph Patterns

### Adding a New Graph Node
```python
# 1. Define pure async function in app/graph/nodes/<category>.py
async def my_new_node(state: GraphState, llm: BaseChatModel) -> dict:
    """One-line docstring: what this node does."""
    logger.info("my_new_node_executing", session_id=state["session_id"])
    # ... logic
    return {"some_field": result}  # only return changed fields

# 2. Register in app/graph/main_graph.py
graph_builder.add_node("my_new_node", partial(my_new_node, llm=llm))
graph_builder.add_edge("previous_node", "my_new_node")

# 3. Add Prometheus counter
NODE_EXECUTION_COUNT.labels(node="my_new_node").inc()
```

### Adding a New RAG Tool
```python
# 1. Implement in app/rag/langchain/tools.py or app/rag/llamaindex/tools.py
@tool("my_new_search_tool")
async def my_new_search_tool(query: str, collection: str = "default") -> str:
    """Search description — this is used by the LLM to decide when to call the tool."""
    with RETRIEVAL_LATENCY.labels(tool="my_new_search_tool").time():
        results = await vector_store.search(query, collection)
    logger.info("tool_executed", tool="my_new_search_tool", results_count=len(results))
    return format_results(results)

# 2. Add to tools list in app/graph/agent_subgraph.py
all_tools = get_langchain_tools() + get_llamaindex_tools() + [my_new_search_tool]
```

### Checkpointing (Short-term Memory)
```python
# Each session has isolated state via thread_id
config = {
    "configurable": {"thread_id": f"{user_id}:{session_id}"},
    "callbacks": [langfuse_handler],
}
result = await graph.ainvoke(state, config=config)
```

---

## 6. RAG Patterns

### Hierarchical Indexing Strategy
```
Document (PDF/DOCX/TXT/MD)
    ↓ HeaderTextSplitter
Parent Chunks (H1/H2/H3 boundaries, ~1500 tokens)
    ↓ RecursiveCharacterTextSplitter
Child Chunks (~200 tokens, stored in Qdrant with parent_id metadata)

Query flow:
1. search_child_chunks(query) → top K child chunks by cosine similarity
2. fetch_parent_chunks(parent_ids) → full parent context for each matched child
3. LLM generates answer from parent context (not child)
```

### LlamaIndex Tool Contract
```python
# LlamaIndex is ONLY called through this tool
# Returns: formatted string with "Source: {filename}\nContent: {text}\n---"
# Never returns raw LlamaIndex objects to the graph state
@tool("llamaindex_query")
async def llamaindex_query(query: str, mode: Literal["sentence_window", "auto_merging"] = "sentence_window") -> str:
    ...
```

### Document Namespace (Multi-tenancy)
```python
# Each user gets their own Qdrant collection namespace
collection_name = f"docs_{user_id}"  # e.g., "docs_usr_abc123"
# Or project-level: f"project_{project_id}"
```

---

## 7. Evaluation Patterns

### RAG Triad — Scoring Scale
All three metrics return a float in [0.0, 1.0].
- **Context Relevance**: "Does the retrieved context contain information needed to answer the query?"
- **Groundedness**: "Is every claim in the answer supported by the retrieved context?"
- **Answer Relevance**: "Does the answer directly address the original user query?"

### Auto-Eval Flow (per response)
```
API Response sent to user
    ↓ asyncio.create_task (non-blocking)
EvalRunner.run(query, contexts, answer, trace_id)
    ↓ parallel asyncio.gather
[context_relevance, groundedness, answer_relevance]
    ↓
Langfuse: langfuse.score(trace_id, name, value)
Prometheus: RAG_TRIAD_SCORE.labels(metric=name).set(value)
```

### Batch Eval (CI/CD)
```bash
make eval              # interactive mode
make eval-batch        # runs against evals/golden_dataset.json
```

### Eval LLM Judge Prompt Format
Prompts in `evals/metrics/*.md`. Format:
```
You are evaluating a RAG system. Score the following on a scale of 0.0 to 1.0.

QUERY: {query}
CONTEXT: {context}
ANSWER: {answer}

Criteria: ...
Score (float 0.0-1.0):
```

---

## 8. Observability Patterns

### Langfuse Trace Structure
```
Trace: {correlation_id}
  ├── Span: "load_user_memory"
  ├── Span: "rewrite_query"
  │     └── Generation: LLM call
  ├── Span: "agent_subgraph_{i}"
  │     ├── Tool: "search_child_chunks"
  │     ├── Tool: "fetch_parent_chunks"
  │     └── Generation: LLM call
  ├── Span: "aggregate_answers"
  │     └── Generation: LLM call
  └── Score: context_relevance = 0.87
      Score: groundedness = 0.91
      Score: answer_relevance = 0.84
```

### Correlation ID Propagation
The `X-Correlation-ID` header (or generated UUID) flows through:
1. HTTP middleware → injected into `GraphState.correlation_id`
2. `GraphState.correlation_id` → Langfuse trace ID
3. Langfuse trace ID → eval score attachment

### Log Context Binding
```python
# Bind context at request start, all subsequent logs inherit it
log = logger.bind(correlation_id=correlation_id, user_id=user_id, session_id=session_id)
log.info("graph_execution_started")
```

---

## 9. Security Patterns

### Authentication Flow
```
POST /api/v1/auth/login → JWT access token (30 days)
All protected routes: Authorization: Bearer <token>
FastAPI dependency: get_current_session() → validates JWT + loads Session
```

### Input Validation
- All inputs validated via Pydantic models before entering graph
- Query max length: `settings.MAX_QUERY_LENGTH` (default: 2000 chars)
- File upload max size: `settings.MAX_FILE_SIZE_MB` (default: 50 MB)
- Allowed file types: PDF, DOCX, TXT, MD

### Secrets
- Never in code, always via `.env.*` files
- Never log secrets (even partial)
- `settings.JWT_SECRET_KEY` must be >= 32 chars in production

---

## 10. Testing Patterns

### Unit Test: Graph Node
```python
# tests/graph/test_query_nodes.py
async def test_rewrite_query_clear_question():
    state = GraphState(messages=[HumanMessage("What is RAG?")], ...)
    mock_llm = AsyncMock()
    mock_llm.with_config().with_structured_output().invoke.return_value = QueryAnalysis(
        questions=["What is RAG?"], is_clear=True
    )
    result = await rewrite_query(state, llm=mock_llm)
    assert result["question_is_clear"] is True
    assert len(result["rewritten_questions"]) == 1
```

### Integration Test: RAG Tools
```python
# tests/rag/test_retrieval.py — uses testcontainers (real Qdrant)
@pytest.mark.integration
async def test_search_child_chunks_returns_results():
    # Uses real Qdrant via testcontainers, real embeddings
    ...
```

### Eval Test
```python
# tests/evaluation/test_triad.py
async def test_groundedness_high_score():
    evaluator = RAGTriadEvaluator(llm=get_llm(settings.EVALUATION_LLM))
    score = await evaluator.groundedness(
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France and a major European city."]
    )
    assert score >= 0.8
```

---

## 11. Environment Variables Reference

All env vars are in `.env.example`. Key groups:

```bash
# App
APP_ENV=development          # development|staging|production
PROJECT_NAME=rag-production
DEBUG=true

# LLM (inference)
LLM_PROVIDER=openai          # openai|anthropic|ollama
DEFAULT_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434

# LLM (evaluation — use stronger model)
EVALUATION_LLM=gpt-4o
EVALUATION_API_KEY=sk-...

# Embedding
EMBEDDING_PROVIDER=huggingface   # huggingface|openai|ollama
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_production
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=                  # empty for local

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Auth
JWT_SECRET_KEY=change-this-in-production-min-32-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_DAYS=30

# Long-term Memory
MEM0_LLM_MODEL=gpt-4o-mini
MEM0_EMBEDDER_MODEL=text-embedding-3-small
LONG_TERM_MEMORY_COLLECTION=ltm_vectors

# RAG
QDRANT_CHILD_COLLECTION=child_chunks
QDRANT_PARENT_COLLECTION=parent_store
CHILD_CHUNK_SIZE=200
CHILD_CHUNK_OVERLAP=20
PARENT_CHUNK_SIZE=1500
MAX_RETRIEVAL_K=5
RETRIEVAL_SCORE_THRESHOLD=0.7
MAX_SELF_CORRECTION_ITERATIONS=3
BASE_TOKEN_THRESHOLD=8000

# Reranking (optional)
COHERE_API_KEY=
RERANKER_MODEL=rerank-multilingual-v3.0
RERANKER_TOP_N=3
```

---

## 12. Key Dependencies

```toml
# Orchestration
langgraph = ">=1.0.0"
langgraph-checkpoint-postgres = ">=3.0.0"
langchain = ">=1.0.0"
langchain-openai = ">=1.0.0"
langchain-anthropic = ">=0.3.0"
langchain-qdrant = ">=0.2.0"

# LlamaIndex
llama-index = ">=0.12.0"
llama-index-vector-stores-qdrant = ">=0.4.0"
llama-index-postprocessor-cohere-rerank = ">=0.3.0"

# API
fastapi = ">=0.121.0"
uvicorn = {extras=["standard"]}
uvloop = ">=0.22.0"

# Database
sqlmodel = ">=0.0.24"
psycopg = {extras=["binary"]}
qdrant-client = ">=1.13.0"

# Memory
mem0ai = ">=1.0.0"

# Observability
langfuse = ">=3.9.0"
prometheus-client = ">=0.19.0"
structlog = ">=25.2.0"

# Utils
tenacity = ">=9.1.0"
slowapi = ">=0.1.9"
pydantic-settings = ">=2.8.0"
```

---

## 13. Common Pitfalls to Avoid

| Pitfall | Impact | Prevention |
|---------|--------|-----------|
| Sync DB call in async handler | Event loop blocking | Always use `await db.execute(...)` |
| f-string in structlog event | Loses structured query-ability | Use kwargs: `logger.info("event", key=val)` |
| Import inside function | Circular imports, perf overhead | All imports at file top |
| Missing Langfuse callback | Silent LLM calls, no traces | `get_llm()` always attaches handler |
| Eval blocking response | +2-5s user latency | `asyncio.create_task()` always |
| Hardcoded collection name | Multi-tenancy breaks | Use `f"docs_{user_id}"` |
| Missing rate limit decorator | DOS vulnerability | `@limiter.limit(...)` on all routes |
| LlamaIndex called outside tool | Untraceable calls | LlamaIndex only via `@tool` |
| Graph node mutates state directly | State corruption | Nodes return dict delta only |
| Missing source citations | Hallucination undetectable | `source_citations` always populated |

---

## 14. Makefile Reference

```bash
make dev              # start FastAPI dev server (hot reload)
make test             # run pytest
make test-integration # run integration tests (requires Docker services)
make eval             # interactive eval CLI
make eval-batch       # batch eval against golden_dataset.json
make lint             # ruff check
make format           # ruff format
make docker-up        # start all services (postgres, qdrant, prometheus, grafana)
make docker-down      # stop all services
make ui               # start Streamlit UI
make ingest-sample    # ingest sample documents for testing
make seed-db          # seed database with test users
```

---

## 15. How AI Should Work in This Codebase

When implementing a new feature:
1. Read CLAUDE.md first (this file) to understand rules and conventions
2. Check the module map (Section 3) to know exactly which file to modify
3. Follow the patterns in Section 5-10 specific to the feature type
4. Add Prometheus metric + Langfuse span for every new node or tool
5. Add unit test in `tests/` mirroring the module structure
6. Update the golden dataset if the feature affects answer quality
7. Never skip rate limiting, auth, or eval instrumentation

When asked to "add a new RAG strategy":
→ Implement in `app/rag/langchain/` or `app/rag/llamaindex/`
→ Expose via `@tool` in the corresponding `tools.py`
→ Add to `all_tools` list in `app/graph/agent_subgraph.py`
→ Add unit test + integration test
→ Update `evals/golden_dataset.json` with relevant test cases

When asked to "add a new evaluation metric":
→ Add LLM judge prompt in `evals/metrics/<name>.md`
→ Add async method to `app/evaluation/triad.py`
→ Add Prometheus gauge in `app/core/metrics.py` (follow naming convention R13)
→ Log score to Langfuse in `app/evaluation/runner.py`
