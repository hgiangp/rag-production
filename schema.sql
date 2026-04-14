-- Production RAG Chatbot — PostgreSQL Schema
-- Run: psql -U postgres -d rag_production -f schema.sql

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ─── Users ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS "user" (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email       TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_email ON "user" (email);

-- ─── Sessions ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    name        TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_user_id ON session (user_id);
CREATE INDEX IF NOT EXISTS idx_session_created_at ON session (created_at DESC);

-- ─── Documents ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_size   BIGINT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | indexed | failed
    collection  TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_user_id ON document (user_id);
CREATE INDEX IF NOT EXISTS idx_document_status ON document (status);

-- ─── Eval Results ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_result (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id            TEXT NOT NULL,
    session_id          UUID REFERENCES session(id) ON DELETE SET NULL,
    user_id             UUID REFERENCES "user"(id) ON DELETE SET NULL,
    query               TEXT NOT NULL,
    answer              TEXT NOT NULL,
    context_relevance   FLOAT,
    groundedness        FLOAT,
    answer_relevance    FLOAT,
    latency_ms          INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_trace_id ON eval_result (trace_id);
CREATE INDEX IF NOT EXISTS idx_eval_user_id ON eval_result (user_id);
CREATE INDEX IF NOT EXISTS idx_eval_created_at ON eval_result (created_at DESC);

-- ─── Chat Messages ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_message (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id  UUID NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    citations   JSONB NOT NULL DEFAULT '[]',
    eval_scores JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_created ON chat_message (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_message_user_id ON chat_message (user_id);

-- Auto-bump session.updated_at whenever a message is inserted
CREATE OR REPLACE FUNCTION update_session_updated_at_on_message()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE session SET updated_at = NOW() WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_session_updated_at_on_message
    AFTER INSERT ON chat_message
    FOR EACH ROW
    EXECUTE FUNCTION update_session_updated_at_on_message();

-- ─── LangGraph Checkpointer Tables (created by langgraph-checkpoint-postgres) ──
-- These are created automatically by AsyncPostgresSaver.setup()
-- Shown here for reference:
-- CREATE TABLE IF NOT EXISTS checkpoints (...)
-- CREATE TABLE IF NOT EXISTS checkpoint_blobs (...)
-- CREATE TABLE IF NOT EXISTS checkpoint_writes (...)

-- ─── Trigger: auto-update updated_at ──────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_user_updated_at
    BEFORE UPDATE ON "user"
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_session_updated_at
    BEFORE UPDATE ON session
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_document_updated_at
    BEFORE UPDATE ON document
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
