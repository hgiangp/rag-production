-- Migration 002: Add workflow_steps to chat_message
-- Run: psql -U postgres -d rag_production -f migrations/002_add_workflow_steps.sql
--
-- Adds a JSONB column that stores the ordered list of WorkflowStep records
-- captured during the SSE streaming pipeline.  Existing rows get an empty
-- array automatically (NOT NULL DEFAULT '[]'), so no back-fill is needed.

ALTER TABLE chat_message
    ADD COLUMN IF NOT EXISTS workflow_steps JSONB NOT NULL DEFAULT '[]';

-- Optional GIN index enables future JSONB path queries
-- (e.g. find messages where a specific tool was called).
CREATE INDEX IF NOT EXISTS idx_chat_message_workflow_steps
    ON chat_message USING gin (workflow_steps);

COMMENT ON COLUMN chat_message.workflow_steps IS
    'Ordered list of WorkflowStep JSON objects emitted during streaming. '
    'Empty array for messages created before this migration or via /chat/invoke.';
