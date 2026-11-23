"""
src/memory.py - Dual-Tier Memory Architecture for Autonomous AI Research Assistant
Implements Short-Term Working Memory (Context Retention & Sliding Window)
and Long-Term Persistent Memory (SQLite Session Store & Knowledge Offload).
"""

import os
import json
import sqlite3
import datetime
from typing import List, Dict, Optional, Any

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "memory.db")


class ShortTermMemory:
    """
    Short-Term Working Memory (Lives inside LLM context window).
    Manages current conversation turns with a sliding-window token/turn budget
    to retain context across multi-turn drill-down Q&A sessions.
    """
    def __init__(self, window_size: int = 6):
        """
        :param window_size: Maximum number of dialogue turns (User + Assistant pairs) to retain.
        """
        self.window_size = window_size
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        """Appends a dialogue turn and maintains sliding-window budget."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        # Keep only the most recent N turns (sliding window)
        max_turns = self.window_size * 2
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the active working memory turns."""
        return list(self.history)

    def format_for_prompt(self) -> str:
        """Formats the sliding window memory into an LLM-digestible context block."""
        if not self.history:
            return "No prior conversational turns in working memory."
        
        lines = []
        for turn in self.history:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clears active working memory."""
        self.history = []

    def load_from_list(self, history_list: List[Dict[str, str]]) -> None:
        """Hydrates working memory from a saved session."""
        self.history = list(history_list)


class LongTermMemory:
    """
    Long-Term Memory Store (Persists across sessions in SQLite database).
    Stores past research dossiers, query plans, retrieved source indices,
    and dialogue history. Enables on-demand recall and cross-session search.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates the SQLite schema for persistent research storage."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                session_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                topic_title TEXT,
                model_name TEXT,
                created_at TEXT NOT NULL,
                plan_json TEXT,
                sources_json TEXT,
                final_report TEXT,
                chat_history_json TEXT,
                reflection_json TEXT
            );
            """)
            try:
                conn.execute("ALTER TABLE research_sessions ADD COLUMN reflection_json TEXT;")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def save_session(
        self,
        session_id: str,
        objective: str,
        topic_title: str,
        model_name: str,
        plan: Any,
        sources: List[Dict],
        final_report: str,
        chat_history: Optional[List[Dict]] = None,
        reflection_history: Optional[List[Dict]] = None
    ) -> None:
        """Persists or updates a complete research session into long-term memory."""
        plan_dict = plan.dict() if hasattr(plan, "dict") else (plan if isinstance(plan, dict) else {})
        chat_list = chat_history or []
        reflection_list = reflection_history or []

        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO research_sessions (
                session_id, objective, topic_title, model_name, created_at,
                plan_json, sources_json, final_report, chat_history_json, reflection_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                objective=excluded.objective,
                topic_title=excluded.topic_title,
                model_name=excluded.model_name,
                plan_json=excluded.plan_json,
                sources_json=excluded.sources_json,
                final_report=excluded.final_report,
                chat_history_json=excluded.chat_history_json,
                reflection_json=excluded.reflection_json;
            """, (
                session_id,
                objective,
                topic_title,
                model_name,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(plan_dict),
                json.dumps(sources),
                final_report,
                json.dumps(chat_list),
                json.dumps(reflection_list)
            ))
            conn.commit()

    def update_chat_history(self, session_id: str, chat_history: List[Dict]) -> None:
        """Updates the conversation turns for an existing research session."""
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE research_sessions
            SET chat_history_json = ?
            WHERE session_id = ?;
            """, (json.dumps(chat_history), session_id))
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single research session by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE session_id = ?;", (session_id,)
            ).fetchone()
            if not row:
                return None
            reflection_val = []
            try:
                if "reflection_json" in row.keys() and row["reflection_json"]:
                    reflection_val = json.loads(row["reflection_json"])
            except Exception:
                pass

            return {
                "session_id": row["session_id"],
                "objective": row["objective"],
                "topic_title": row["topic_title"],
                "model_name": row["model_name"],
                "created_at": row["created_at"],
                "plan": json.loads(row["plan_json"] or "{}"),
                "sources": json.loads(row["sources_json"] or "[]"),
                "final_report": row["final_report"],
                "chat_history": json.loads(row["chat_history_json"] or "[]"),
                "reflection_history": reflection_val
            }

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Lists recent research sessions with metadata for sidebar history."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT session_id, objective, topic_title, model_name, created_at,
                   length(final_report) as report_len
            FROM research_sessions
            ORDER BY created_at DESC
            LIMIT ?;
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        """Searches past research sessions across objectives and reports (Long-term recall)."""
        search_pattern = f"%{query}%"
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT session_id, objective, topic_title, model_name, created_at
            FROM research_sessions
            WHERE objective LIKE ? OR final_report LIKE ? OR topic_title LIKE ?
            ORDER BY created_at DESC
            LIMIT 10;
            """, (search_pattern, search_pattern, search_pattern)).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        """Deletes a session from long-term memory."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM research_sessions WHERE session_id = ?;", (session_id,))
            conn.commit()
