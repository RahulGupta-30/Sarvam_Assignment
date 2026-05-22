import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any


DB_PATH = "research_sessions.db"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def count_messages(session_id: str) -> int:
    conn = connect_db()

    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,)
    ).fetchone()

    conn.close()

    return row["count"] if row else 0


def get_all_messages(session_id: str) -> list[dict]:
    conn = connect_db()

    rows = conn.execute(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,)
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_summary_message_count(session_id: str) -> int:
    conn = connect_db()

    row = conn.execute(
        """
        SELECT summary_message_count
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,)
    ).fetchone()

    conn.close()

    if not row:
        return 0

    return row["summary_message_count"] or 0

def column_exists(conn, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)

def connect_db():
    """
    Opens a SQLite connection.

    check_same_thread=False helps Streamlit because Streamlit can rerun scripts
    and use the connection across different execution contexts.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # WAL improves read/write behavior for small production-ish apps.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


def init_db():
    """
    Creates all required tables if they do not already exist.
    """

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT,
        rolling_summary TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    if not column_exists(conn, "sessions", "summary_message_count"):
        cur.execute("""
        ALTER TABLE sessions
        ADD COLUMN summary_message_count INTEGER DEFAULT 0
        """)    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        search_results_json TEXT,
        selected_urls_json TEXT,
        fetched_pages_json TEXT,
        used_sources_json TEXT,
        final_answer TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()


def create_session(title: Optional[str] = None) -> str:
    """
    Creates a new research session and returns session_id.
    """

    session_id = str(uuid.uuid4())
    timestamp = now_iso()

    conn = connect_db()
    conn.execute(
        """
        INSERT INTO sessions (session_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, title or "New Research Session", timestamp, timestamp)
    )

    conn.commit()
    conn.close()

    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = connect_db()

    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,)
    ).fetchone()

    conn.close()

    if not row:
        return None

    return dict(row)


def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = connect_db()

    rows = conn.execute(
        """
        SELECT session_id, title, created_at, updated_at
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def touch_session(session_id: str):
    conn = connect_db()

    conn.execute(
        """
        UPDATE sessions
        SET updated_at = ?
        WHERE session_id = ?
        """,
        (now_iso(), session_id)
    )

    conn.commit()
    conn.close()


def update_session_title(session_id: str, title: str):
    conn = connect_db()

    conn.execute(
        """
        UPDATE sessions
        SET title = ?, updated_at = ?
        WHERE session_id = ?
        """,
        (title, now_iso(), session_id)
    )

    conn.commit()
    conn.close()


def add_message(session_id: str, role: str, content: str):
    """
    role should be 'user' or 'assistant'.
    """

    conn = connect_db()

    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, role, content, now_iso())
    )

    conn.execute(
        """
        UPDATE sessions
        SET updated_at = ?
        WHERE session_id = ?
        """,
        (now_iso(), session_id)
    )

    conn.commit()
    conn.close()


def get_messages(session_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Returns recent messages in chronological order.
    """

    conn = connect_db()

    rows = conn.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit)
    ).fetchall()

    conn.close()

    messages = [dict(row) for row in rows]

    return list(reversed(messages))


def add_turn(
    session_id: str,
    query: str,
    search_results: List[Dict[str, Any]],
    selected_urls: List[Dict[str, Any]],
    fetched_pages: List[Dict[str, Any]],
    used_sources: List[Dict[str, Any]],
    final_answer: str,
):
    """
    Stores one complete research turn.
    """

    # Avoid storing huge full page text forever.
    # For production, store only metadata + preview, not the entire raw page text.
    fetched_pages_light = []

    for page in fetched_pages:
        fetched_pages_light.append({
            "title": page.get("title", ""),
            "url": page.get("url", ""),
            "domain": page.get("domain", ""),
            "retrieved_at": page.get("retrieved_at", ""),
            "text_preview": page.get("text", "")[:1000],
        })

    conn = connect_db()

    conn.execute(
        """
        INSERT INTO turns (
            session_id,
            query,
            search_results_json,
            selected_urls_json,
            fetched_pages_json,
            used_sources_json,
            final_answer,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            query,
            json.dumps(search_results, ensure_ascii=False),
            json.dumps(selected_urls, ensure_ascii=False),
            json.dumps(fetched_pages_light, ensure_ascii=False),
            json.dumps(used_sources, ensure_ascii=False),
            final_answer,
            now_iso()
        )
    )

    conn.execute(
        """
        UPDATE sessions
        SET updated_at = ?
        WHERE session_id = ?
        """,
        (now_iso(), session_id)
    )

    conn.commit()
    conn.close()


def get_turns(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = connect_db()

    rows = conn.execute(
        """
        SELECT *
        FROM turns
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit)
    ).fetchall()

    conn.close()

    turns = []

    for row in rows:
        item = dict(row)

        item["search_results"] = json.loads(item.pop("search_results_json") or "[]")
        item["selected_urls"] = json.loads(item.pop("selected_urls_json") or "[]")
        item["fetched_pages"] = json.loads(item.pop("fetched_pages_json") or "[]")
        item["used_sources"] = json.loads(item.pop("used_sources_json") or "[]")

        turns.append(item)

    return list(reversed(turns))


def get_rolling_summary(session_id: str) -> str:
    conn = connect_db()

    row = conn.execute(
        """
        SELECT rolling_summary
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,)
    ).fetchone()

    conn.close()

    if not row:
        return ""

    return row["rolling_summary"] or ""


def update_rolling_summary(
    session_id: str,
    summary: str,
    summary_message_count: int | None = None
):
    conn = connect_db()

    if summary_message_count is None:
        conn.execute(
            """
            UPDATE sessions
            SET rolling_summary = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (summary, now_iso(), session_id)
        )
    else:
        conn.execute(
            """
            UPDATE sessions
            SET rolling_summary = ?,
                summary_message_count = ?,
                updated_at = ?
            WHERE session_id = ?
            """,
            (summary, summary_message_count, now_iso(), session_id)
        )

    conn.commit()
    conn.close()


def build_history_context(session_id: str, max_messages: int = 8, max_turns: int = 3) -> str:
    """
    Builds a compact history block to pass into Gemini.

    This is NOT web context.
    This is conversation/session context.
    """

    rolling_summary = get_rolling_summary(session_id)
    messages = get_messages(session_id, limit=max_messages)
    turns = get_turns(session_id, limit=max_turns)

    parts = []

    if rolling_summary:
        parts.append(f"""
                    [ROLLING SUMMARY]
                    {rolling_summary}
                    """.strip())

    if messages:
        message_lines = []

        for msg in messages:
            message_lines.append(
                f"{msg['role'].upper()} at {msg['created_at']}:\n{msg['content']}"
            )

        parts.append("""
                [RECENT CONVERSATION]
                {}
                """.format("\n\n".join(message_lines)).strip())

    if turns:
        turn_lines = []

        for turn in turns:
            turn_lines.append(f"""
                Previous Query:
                {turn['query']}

                Previous Final Answer:
                {turn['final_answer'][:1500]}
                """.strip())

            parts.append("""
                [RELEVANT PRIOR TURNS]
                {}
                """.format("\n\n---\n\n".join(turn_lines)).strip())

    return "\n\n====================\n\n".join(parts)


def delete_session(session_id: str):
    conn = connect_db()

    conn.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,)
    )

    conn.commit()
    conn.close()