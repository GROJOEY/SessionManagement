"""FastAPI app: one chat box, topics filed automatically.

Run:  ANTHROPIC_API_KEY=... uvicorn onethread.server:app --reload
      ONETHREAD_MOCK=1 uvicorn onethread.server:app   (no key, scripted replies)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import DB
from .embed import HashEmbedder, make_embedder
from .engine import Engine
from .llm import ClaudeLLM, MockLLM

MOCK = os.environ.get("ONETHREAD_MOCK") == "1"
db = DB(os.environ.get("ONETHREAD_DB", "data/onethread.db"))
engine = Engine(db, MockLLM() if MOCK else ClaudeLLM(),
                HashEmbedder() if MOCK else make_embedder())

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="OneThread")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    text: str


class MoveIn(BaseModel):
    topic_id: int | None = None
    new_topic: str | None = None


class TopicPatch(BaseModel):
    name: str | None = None
    summary: str | None = None


def topic_dict(t):
    return {"id": t.id, "name": t.name, "summary": t.summary,
            "updated_at": t.updated_at, "segment_count": t.segment_count}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
def chat(body: ChatIn, background: BackgroundTasks):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    result = engine.handle(text)
    background.add_task(engine.update_summaries, result)
    return {
        "user_message_id": result.user_message_id,
        "assistant_message_id": result.assistant_message_id,
        "segments": [s.__dict__ for s in result.segments],
        "related": result.related,
        "reply": result.reply,
    }


@app.get("/api/messages")
def messages():
    out = []
    for m in db.all_messages():
        segs = db.message_segments(m["id"]) if m["role"] == "user" else []
        out.append({
            "id": m["id"], "role": m["role"], "content": m["content"],
            "topic_ids": json.loads(m["topic_ids"]),
            "segments": [{"id": s.id, "text": s.text, "topic_id": s.topic_id,
                          "topic_name": db.get_topic(s.topic_id).name,
                          "corrected": s.corrected} for s in segs],
        })
    return out


@app.get("/api/topics")
def topics():
    return [topic_dict(t) for t in db.list_topics()]


@app.get("/api/topics/{topic_id}")
def topic(topic_id: int):
    t = db.get_topic(topic_id)
    if not t:
        raise HTTPException(404, "no such topic")
    segs = [{"id": s.id, "role": s.role, "text": s.text, "created_at": s.created_at}
            for s in db.topic_segments(topic_id)]
    return {**topic_dict(t), "segments": segs}


@app.patch("/api/topics/{topic_id}")
def patch_topic(topic_id: int, body: TopicPatch):
    if not db.get_topic(topic_id):
        raise HTTPException(404, "no such topic")
    db.update_topic(topic_id, name=body.name, summary=body.summary)
    return topic_dict(db.get_topic(topic_id))


@app.post("/api/segments/{segment_id}/move")
def move(segment_id: int, body: MoveIn):
    try:
        tid = engine.move_segment(segment_id, topic_id=body.topic_id, new_topic=body.new_topic)
    except KeyError:
        raise HTTPException(404, "no such segment")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"topic_id": tid, "topic_name": db.get_topic(tid).name}
