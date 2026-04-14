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
