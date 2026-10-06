import os
import sqlite3
import time
from contextlib import closing


class SessionMemory:
    """SQLite-backed chat history (swap for Redis in multi-instance deployments)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        with closing(self._conn()) as c, c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
                "role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id)")

    def _conn(self):
        return sqlite3.connect(self.path)

    def add(self, session_id: str, role: str, content: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?,?,?,?)",
                (session_id, role, content, time.time()),
            )

    def history(self, session_id: str, turns: int) -> list[dict]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, turns * 2),
            ).fetchall()
        return [{"role": r, "content": t} for r, t in reversed(rows)]


    def clear(self, session_id: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
