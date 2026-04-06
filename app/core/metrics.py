"""Prometheus metric definitions.

IMPORTANT: Never rename these metrics — dashboards depend on exact names.
See CLAUDE.md Rule R13.

Metric naming: rag_{domain}_{measurement}_{unit}
"""

from prometheus_client import Counter, Gauge, Histogram

# ─── Request Metrics ──────────────────────────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end HTTP request duration",
    ["method", "endpoint", "status_code"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

ACTIVE_SESSIONS = Gauge(
    "rag_active_sessions_total",
    "Number of active chat sessions",
)

# ─── LLM Metrics ─────────────────────────────────────────────────────────────
LLM_TOKENS_TOTAL = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],  # direction: input | output
)

LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds",
    "LLM call duration",
    ["model", "node"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "LLM call errors",
    ["model", "error_type"],
)

# ─── Retrieval Metrics ────────────────────────────────────────────────────────
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Vector store retrieval duration",
    ["tool", "collection"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results_count",
    "Number of chunks retrieved per query",
    ["tool"],
    buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
)

SELF_CORRECTION_COUNT = Counter(
    "rag_self_corrections_total",
    "Number of self-correction iterations",
)

# ─── Evaluation Metrics (RAG Triad) ───────────────────────────────────────────
RAG_TRIAD_SCORE = Gauge(
    "rag_eval_triad_score",
    "RAG triad evaluation score (rolling average)",
    ["metric"],  # context_relevance | groundedness | answer_relevance
)

EVAL_LATENCY = Histogram(
    "rag_eval_duration_seconds",
    "Evaluation pipeline duration",
    ["metric"],
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0],
)

# ─── Document Ingestion Metrics ───────────────────────────────────────────────
INGEST_COUNT = Counter(
    "rag_ingest_documents_total",
    "Total documents ingested",
    ["file_type", "status"],  # status: success | failed
)

INGEST_CHUNKS = Histogram(
    "rag_ingest_chunks_count",
    "Number of chunks created per document",
    buckets=[10, 50, 100, 200, 500, 1000],
)

INGEST_LATENCY = Histogram(
    "rag_ingest_duration_seconds",
    "Document ingestion duration",
    ["file_type"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ─── Graph Metrics ────────────────────────────────────────────────────────────
GRAPH_NODE_LATENCY = Histogram(
    "rag_graph_node_duration_seconds",
    "LangGraph node execution duration",
    ["node"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
)

GRAPH_NODE_COUNT = Counter(
    "rag_graph_node_executions_total",
    "Total LangGraph node executions",
    ["node"],
)
