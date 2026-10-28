# -*- coding: utf-8 -*-
"""本地持久化：SQLite 存用户会话 + 文档登记簿；可选 mem0 长期记忆（USE_MEM0=true）。

多用户/跨会话设计：
- 每个用户用 user_id（UUID，写入浏览器长效 Cookie）区分；
- sessions 表按 user_id 关联，记录会话标题与最近更新时间，支持"查看所有历史会话"；
- messages 表记录全部历史（UI 展示用），并保存随消息发送的附件解析结果；
- 发给大模型的只有最近 3 轮（见 llm.trim_messages）；
- 关闭网页后重新打开：on_load 依据 user_id 恢复最近一次会话，可继续追问；
- mem0 为可选：pip install mem0ai 并 USE_MEM0=true 启用，否则退化为仅 SQLite。
"""
import os
import sqlite3
import threading
import time
import uuid

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", encoding="utf-8")

_DB = os.path.abspath(os.getenv("HISTORY_DB", "./data/history.db"))
USE_MEM0 = os.getenv("USE_MEM0", "false").lower() == "true"

os.makedirs(os.path.dirname(_DB), exist_ok=True)
_lock = threading.RLock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents(
                doc_name   TEXT PRIMARY KEY,
                src_type   TEXT NOT NULL,
                n_chunks   INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL DEFAULT 'anonymous',
                title      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                attachments TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """
        )
        # 兼容旧库自动迁移：补齐缺失列
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "user_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'")
        if "title" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        if "updated_at" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN updated_at REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC)")

        msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "attachments" not in msg_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")


# ---------- 文档登记簿 ----------
def register_document(doc_name: str, src_type: str, n_chunks: int):
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents(doc_name, src_type, n_chunks, created_at) VALUES(?,?,?,?)",
            (doc_name, src_type, n_chunks, time.time()),
        )


def list_documents() -> list:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT doc_name, src_type, n_chunks, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def unregister_document(doc_name: str):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM documents WHERE doc_name=?", (doc_name,))


# ---------- 会话与历史 ----------
def new_session(user_id: str = "anonymous", title: str = None) -> str:
    sid = uuid.uuid4().hex
    now = time.time()
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, user_id, title, created_at, updated_at) VALUES(?,?,?,?,?)",
            (sid, user_id, title, now, now),
        )
    return sid


def list_user_sessions(user_id: str, limit: int = 200) -> list:
    """列出该用户全部会话（按最近活动倒序），用于侧边栏历史列表。"""
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT s.session_id, s.title, s.created_at, s.updated_at, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS message_count "
            "FROM sessions s "
            "WHERE s.user_id=? ORDER BY COALESCE(s.updated_at, s.created_at) DESC, s.created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str, user_id: str) -> dict | None:
    """读取单个会话（校验归属），不存在或不属于该用户时返回 None。"""
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT s.session_id, s.title, s.created_at, s.updated_at, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS message_count "
            "FROM sessions s WHERE s.session_id=? AND s.user_id=?",
            (session_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str, user_id: str) -> bool:
    """删除该用户名下的某个会话及其全部消息。"""
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id=? AND user_id=?", (session_id, user_id))
    return True


def set_session_title(session_id: str, title: str):
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE sessions SET title=? WHERE session_id=? AND (title IS NULL OR title='')",
            (title, session_id),
        )


def _touch(session_id: str):
    with _lock, _conn() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (time.time(), session_id))


def save_message(session_id: str, role: str, content: str, sources: str = None, attachments: str = None):
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, sources, attachments, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (session_id, role, content, sources, attachments, time.time()),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (time.time(), session_id))


def load_session_messages(session_id: str, limit: int = 200) -> list:
    """读取单个会话的历史（按时间升序），供恢复 UI 用。

    返回的 sources / attachments 为数据库中的原始文本（新格式为 JSON），
    由上层负责解析，避免存储层与展示格式耦合。
    """
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, attachments, created_at FROM messages "
            "WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "sources": r["sources"],
            "attachments": r["attachments"],
            "created_at": r["created_at"],
        }
        for r in reversed(rows)
    ]


# ---------- 可选 mem0 长期记忆 ----------
_mem0 = None
_mem0_failed = False


def _get_mem0():
    global _mem0, _mem0_failed
    if not USE_MEM0 or _mem0_failed:
        return None
    if _mem0 is not None:
        return _mem0
    try:
        from mem0 import Memory  # noqa: PLC0415

        url = os.getenv("ALI_URL")
        key = os.getenv("ALI_TOKEN")
        _mem0 = Memory.from_config({
            "llm": {"provider": "openai", "config": {
                "model": os.getenv("LLM_MODEL", "qwen3.8-flash"),
                "openai_base_url": url, "api_key": key,
            }},
            "embedder": {"provider": "openai", "config": {
                # mem0 走 OpenAI 兼容的纯文本 embedding，不复用多模态模型
                "model": os.getenv("MEM0_EMBED_MODEL", "qwen3.7-text-embedding-flash"),
                "embedding_dims": int(os.getenv("EMBED_DIM", "1024")),
                "openai_base_url": url, "api_key": key,
            }},
            "vector_store": {"provider": "chroma", "config": {
                "collection_name": "mem0", "path": "./data/mem0_chroma",
            }},
        })
        return _mem0
    except Exception:  # noqa: BLE001
        _mem0_failed = True
        return None


def mem0_add(session_id: str, user_msg: str, assistant_msg: str):
    m = _get_mem0()
    if m is None:
        return
    try:
        m.add(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": assistant_msg}],
            user_id=session_id,
        )
    except Exception:  # noqa: BLE001
        pass


init_db()