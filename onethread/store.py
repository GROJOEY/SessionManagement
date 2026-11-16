"""SQLite store: topics, each with a running summary, and the notes filed under them."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import cosine

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id                   INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL UNIQUE COLLATE NOCASE,
    summary              TEXT NOT NULL DEFAULT '',
    notes_since_summary  INTEGER NOT NULL DEFAULT 0,
    created_at           REAL NOT NULL,
    updated_at           REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY,
    topic_id    INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   BLOB,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic_id);
"""


@dataclass
class Topic:
    id: int
    name: str
    summary: str
    notes_since_summary: int
    updated_at: float
    note_count: int = 0


@dataclass
class Note:
    id: int
    topic_id: int
    content: str
    embedding: np.ndarray | None
    created_at: float


class Store:
    def __init__(self, path: str | Path, embedder):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.embedder = embedder

    # ---------------------------------------------------------------- topics
    def _topic(self, r) -> Topic:
        keys = r.keys()
        return Topic(r["id"], r["name"], r["summary"], r["notes_since_summary"], r["updated_at"],
                     r["n"] if "n" in keys else 0)

    def list_topics(self) -> list[Topic]:
        rows = self.conn.execute(
            """SELECT t.*, COUNT(n.id) AS n FROM topics t LEFT JOIN notes n ON n.topic_id = t.id
               GROUP BY t.id ORDER BY t.updated_at DESC""").fetchall()
        return [self._topic(r) for r in rows]

    def get_topic(self, name: str) -> Topic | None:
        r = self.conn.execute(
            """SELECT t.*, COUNT(n.id) AS n FROM topics t LEFT JOIN notes n ON n.topic_id = t.id
               WHERE t.name = ? COLLATE NOCASE GROUP BY t.id""", (name.strip(),)).fetchone()
        return self._topic(r) if r else None

    def get_or_create_topic(self, name: str) -> tuple[Topic, bool]:
        t = self.get_topic(name)
        if t:
            return t, False
        now = time.time()
        self.conn.execute("INSERT INTO topics(name, created_at, updated_at) VALUES (?,?,?)",
                          (name.strip(), now, now))
        self.conn.commit()
        return self.get_topic(name), True

    def set_summary(self, name: str, summary: str) -> Topic | None:
        t = self.get_topic(name)
        if not t:
            return None
        self.conn.execute(
            "UPDATE topics SET summary=?, notes_since_summary=0, updated_at=? WHERE id=?",
            (summary.strip(), time.time(), t.id))
        self.conn.commit()
        return self.get_topic(name)

    def rename_topic(self, old: str, new: str) -> Topic | None:
        t = self.get_topic(old)
        if not t:
            return None
        self.conn.execute("UPDATE topics SET name=?, updated_at=? WHERE id=?", (new.strip(), time.time(), t.id))
        self.conn.commit()
        return self.get_topic(new)

    def merge_topics(self, source: str, target: str) -> Topic | None:
        s, t = self.get_topic(source), self.get_topic(target)
        if not s or not t or s.id == t.id:
            return None
        self.conn.execute("UPDATE notes SET topic_id=? WHERE topic_id=?", (t.id, s.id))
        self.conn.execute("UPDATE topics SET notes_since_summary=notes_since_summary+?, updated_at=? WHERE id=?",
                          (s.note_count, time.time(), t.id))
        self.conn.execute("DELETE FROM topics WHERE id=?", (s.id,))
        self.conn.commit()
        return self.get_topic(target)

    # ----------------------------------------------------------------- notes
    def _note(self, r) -> Note:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        return Note(r["id"], r["topic_id"], r["content"], emb, r["created_at"])

    def add_note(self, topic_name: str, content: str) -> tuple[Topic, Note, bool]:
        topic, created = self.get_or_create_topic(topic_name)
        emb = self.embedder.embed(f"{topic.name}：{content}")
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO notes(topic_id, content, embedding, created_at) VALUES (?,?,?,?)",
            (topic.id, content.strip(), emb.astype(np.float32).tobytes(), now))
        self.conn.execute(
            "UPDATE topics SET notes_since_summary=notes_since_summary+1, updated_at=? WHERE id=?",
            (now, topic.id))
        self.conn.commit()
        note = self._note(self.conn.execute("SELECT * FROM notes WHERE id=?", (cur.lastrowid,)).fetchone())
        return self.get_topic(topic.name), note, created

    def topic_notes(self, topic_id: int) -> list[Note]:
        rows = self.conn.execute("SELECT * FROM notes WHERE topic_id=? ORDER BY id", (topic_id,)).fetchall()
        return [self._note(r) for r in rows]

    def get_note(self, note_id: int) -> Note | None:
        r = self.conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return self._note(r) if r else None

    def move_note(self, note_id: int, topic_name: str) -> Topic | None:
        note = self.get_note(note_id)
        if not note:
            return None
        topic, _ = self.get_or_create_topic(topic_name)
        now = time.time()
        self.conn.execute("UPDATE notes SET topic_id=? WHERE id=?", (topic.id, note_id))
        self.conn.execute("UPDATE topics SET notes_since_summary=notes_since_summary+1, updated_at=? WHERE id IN (?,?)",
                          (now, topic.id, note.topic_id))
        self.conn.commit()
        return self.get_topic(topic.name)

    def delete_note(self, note_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------------------- recall
    def recall(self, message: str, k_topics: int = 3, k_notes: int = 3, floor: float | None = None):
        """Rank topics by their best-matching note, return the top ones with summary and notes."""
        floor = getattr(self.embedder, "recall_floor", 0.2) if floor is None else floor
        q = self.embedder.embed(message)
        scored = []
        for t in self.list_topics():
            notes = self.topic_notes(t.id)
            if not notes and not t.summary:
                continue
            sims = sorted(((cosine(q, n.embedding), n) for n in notes), key=lambda x: x[0], reverse=True)
            name_sim = cosine(q, self.embedder.embed(t.name))
            best = max([s for s, _ in sims[:1]] + [name_sim])
            scored.append((best, t, sims, notes))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for best, t, sims, notes in scored[:k_topics]:
            if best < floor:
                continue
            picked = {n.id: n for _, n in sims[:k_notes]}
            if notes:
                picked.setdefault(notes[-1].id, notes[-1])  # always the latest note
            hits.append({"topic": t, "score": round(best, 3),
                         "notes": sorted(picked.values(), key=lambda n: n.id)})
        return hits
