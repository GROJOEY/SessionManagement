"""SQLite storage: one global conversation, topics, and the segments that link them."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    role        TEXT NOT NULL,          -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    topic_ids   TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    topic_id    INTEGER NOT NULL REFERENCES topics(id),
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,
    corrected   INTEGER NOT NULL DEFAULT 0,  -- 1 = user moved it by hand
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_topic ON segments(topic_id);
"""


@dataclass
class Topic:
    id: int
    name: str
    summary: str
    updated_at: float
    segment_count: int = 0


@dataclass
class Segment:
    id: int
    message_id: int
    topic_id: int
    role: str
    text: str
    embedding: np.ndarray | None
    corrected: bool
    created_at: float


class DB:
    def __init__(self, path: str | Path = "data/onethread.db"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- topics ----------
    def create_topic(self, name: str, summary: str = "") -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO topics(name, summary, created_at, updated_at) VALUES (?,?,?,?)",
            (name.strip(), summary, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_topics(self) -> list[Topic]:
        rows = self.conn.execute(
            """SELECT t.*, COUNT(s.id) AS n FROM topics t
               LEFT JOIN segments s ON s.topic_id = t.id
               GROUP BY t.id HAVING n > 0 OR t.summary != ''
               ORDER BY t.updated_at DESC"""
        ).fetchall()
        return [Topic(r["id"], r["name"], r["summary"], r["updated_at"], r["n"]) for r in rows]

    def get_topic(self, topic_id: int) -> Topic | None:
        r = self.conn.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        return Topic(r["id"], r["name"], r["summary"], r["updated_at"]) if r else None

    def find_topic_by_name(self, name: str) -> Topic | None:
        r = self.conn.execute(
            "SELECT * FROM topics WHERE lower(name)=lower(?)", (name.strip(),)
        ).fetchone()
        return Topic(r["id"], r["name"], r["summary"], r["updated_at"]) if r else None

    def update_topic(self, topic_id: int, *, summary: str | None = None, name: str | None = None):
        if summary is not None:
            self.conn.execute("UPDATE topics SET summary=? WHERE id=?", (summary, topic_id))
        if name is not None:
            self.conn.execute("UPDATE topics SET name=? WHERE id=?", (name.strip(), topic_id))
        self.touch_topic(topic_id)

    def touch_topic(self, topic_id: int):
        self.conn.execute("UPDATE topics SET updated_at=? WHERE id=?", (time.time(), topic_id))
        self.conn.commit()

    # ---------- messages ----------
    def add_message(self, role: str, content: str, topic_ids: list[int] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(role, content, topic_ids, created_at) VALUES (?,?,?,?)",
            (role, content, json.dumps(topic_ids or []), time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_message_topics(self, message_id: int, topic_ids: list[int]):
        self.conn.execute(
            "UPDATE messages SET topic_ids=? WHERE id=?", (json.dumps(topic_ids), message_id)
        )
        self.conn.commit()

    def recent_messages(self, limit: int = 6) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return list(reversed(rows))

    def all_messages(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM messages ORDER BY id").fetchall()

    # ---------- segments ----------
    def add_segment(self, message_id: int, topic_id: int, role: str, text: str,
                    embedding: np.ndarray | None) -> int:
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        cur = self.conn.execute(
            """INSERT INTO segments(message_id, topic_id, role, text, embedding, created_at)
               VALUES (?,?,?,?,?,?)""",
            (message_id, topic_id, role, text, blob, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def _seg(self, r: sqlite3.Row) -> Segment:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        return Segment(r["id"], r["message_id"], r["topic_id"], r["role"], r["text"], emb,
                       bool(r["corrected"]), r["created_at"])

    def topic_segments(self, topic_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE topic_id=? ORDER BY id", (topic_id,)
        ).fetchall()
        return [self._seg(r) for r in rows]

    def message_segments(self, message_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE message_id=? ORDER BY id", (message_id,)
        ).fetchall()
        return [self._seg(r) for r in rows]

    def get_segment(self, segment_id: int) -> Segment | None:
        r = self.conn.execute("SELECT * FROM segments WHERE id=?", (segment_id,)).fetchone()
        return self._seg(r) if r else None

    def move_segment(self, segment_id: int, topic_id: int):
        self.conn.execute(
            "UPDATE segments SET topic_id=?, corrected=1 WHERE id=?", (topic_id, segment_id)
        )
        self.conn.commit()
