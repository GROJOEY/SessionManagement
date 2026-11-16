"""OneThread: a lightweight MCP memory server.

You keep one conversation with Claude and talk about anything. Claude files what matters
under topics it picks itself, and pulls the right topic back in the next time it comes up.
The server only stores and searches; all the judgment (which topic, what to keep, how to
summarize) is done by the Claude you are already talking to, so no API key is needed.

Run:
  python -m onethread.server              # remote, http://localhost:8000/mcp
  python -m onethread.server --stdio      # local, for Claude Desktop / Claude Code
Env:
  ONETHREAD_DB   SQLite file (default ./data/onethread.db)
  EMBED_MODEL    'hash' (default, no download) or a sentence-transformers model
  AUTH_TOKEN     if set, the remote URL must carry ?token=<value>
  PORT           default 8000
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from starlette.responses import JSONResponse

from .embed import make_embedder
from .store import Store

SUMMARY_EVERY = 5     # ask Claude to refresh a topic summary after this many new notes
INDEX_WITH_GIST = 40  # recall's topic index: this many get a one-line summary, the rest names only

store = Store(os.environ.get("ONETHREAD_DB", "./data/onethread.db"), make_embedder())

mcp = MCPServer(
    name="onethread",
    version="0.2.0",
    instructions=(
        "OneThread is the user's long-term memory, organized by topic. The user talks about "
        "everything in one conversation and never says which topic they mean; you keep track.\n"
        "1. At the start of EVERY user message, before answering, call `recall` with the message. "
        "Use what comes back naturally, like a friend who remembers; don't announce that you looked it up.\n"
        "2. After answering, if the message contained something worth remembering (a fact, decision, "
        "plan, preference, progress, open question), call `remember` once per topic it touches. "
        "Pick an existing topic when one fits; otherwise create a short, recognizable one (2–8 words). "
        "Store the gist in one or two sentences, not the raw message. Skip small talk.\n"
        "3. When `remember` reports the summary is stale, call `update_summary` with a rewritten summary "
        "(facts, decisions, plans, open questions; newest wins on conflicts; under 120 words).\n"
        "4. Never ask the user which topic something belongs to. If they say something was filed wrong, "
        "fix it with `move_note` or `merge_topics`."
    ),
)


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


@mcp.tool(title="Recall memory", annotations=ToolAnnotations(readOnlyHint=True))
def recall(message: str) -> str:
    """Call at the start of every user message. Returns (1) the topics whose notes best match the
    message, with summary and notes, and (2) an index of every other topic with a one-line summary.
    The matching is rough: read the index yourself, and if a topic there is relevant, call
    get_topic for it before answering.

    Args:
        message: the user's latest message, verbatim
    """
    hits = store.recall(message)
    lines = []
    if hits:
        for h in hits:
            t = h["topic"]
            lines.append(f"## {t.name}  (match {h['score']})")
            lines.append(f"Summary: {t.summary or '(none yet)'}")
            for n in h["notes"]:
                lines.append(f"- [{n.id}] {_when(n.created_at)}: {n.content}")
            lines.append("")
    shown = {h["topic"].id for h in hits}
    others = [t for t in store.list_topics() if t.id not in shown]
    if others:
        lines.append("## Other topics (call get_topic if one is relevant)")
        # the most recently active get a one-line gist; the rest are listed by name only,
        # which costs a few characters each, so Claude still knows every topic exists
        for t in others[:INDEX_WITH_GIST]:
            gist = (t.summary or (store.topic_notes(t.id)[-1].content if t.note_count else ""))[:80]
            lines.append(f"- {t.name}: {gist}")
        older = others[INDEX_WITH_GIST:]
        if older:
            lines.append("Older topics: " + ", ".join(t.name for t in older))
    if not lines:
        return "No memory yet. Anything worth keeping from this message starts a new topic."
    return "\n".join(lines).strip()


@mcp.tool(title="Remember")
def remember(topic: str, content: str) -> str:
    """File one piece of information under a topic. The topic is created if it doesn't exist
    (names match case-insensitively). Call once per topic a message touches.

    Args:
        topic: an existing topic name, or a short new one (2-8 words)
        content: the gist in one or two sentences, written so it makes sense later on its own
    """
    t, note, created = store.add_note(topic, content)
    msg = f"Saved note [{note.id}] under '{t.name}'" + (" (new topic)" if created else "") + "."
    if t.notes_since_summary >= SUMMARY_EVERY or (created and not t.summary):
        msg += f" The summary of '{t.name}' is stale: call update_summary for it."
    return msg


@mcp.tool(title="Update topic summary")
def update_summary(topic: str, summary: str) -> str:
    """Replace a topic's running summary. Write it from the old summary plus the newer notes
    (use get_topic if you need them): facts, decisions, plans, open questions; newest wins.

    Args:
        topic: topic name
        summary: the new summary, under 120 words
    """
    t = store.set_summary(topic, summary)
    return f"Updated summary of '{t.name}'." if t else f"No topic named '{topic}'."


@mcp.tool(title="List topics", annotations=ToolAnnotations(readOnlyHint=True))
def list_topics() -> str:
    """List every topic with its note count and summary, most recently active first."""
    ts = store.list_topics()
    if not ts:
        return "No topics yet."
    return "\n".join(f"- {t.name} ({t.note_count} notes, active {_when(t.updated_at)}): "
                     f"{t.summary or '(no summary yet)'}" for t in ts)


@mcp.tool(title="Get topic", annotations=ToolAnnotations(readOnlyHint=True))
def get_topic(topic: str) -> str:
    """Show one topic in full: summary and every note with its id.

    Args:
        topic: topic name
    """
    t = store.get_topic(topic)
    if not t:
        return f"No topic named '{topic}'. Topics: " + ", ".join(x.name for x in store.list_topics())
    notes = "\n".join(f"- [{n.id}] {_when(n.created_at)}: {n.content}" for n in store.topic_notes(t.id))
    return f"# {t.name}\nSummary: {t.summary or '(none yet)'}\n\n{notes or '(no notes)'}"


@mcp.tool(title="Move note")
def move_note(note_id: int, topic: str) -> str:
    """Move a note that was filed under the wrong topic. Creates the topic if needed.

    Args:
        note_id: the number in square brackets
        topic: the topic it belongs to
    """
    t = store.move_note(note_id, topic)
    return f"Moved note [{note_id}] to '{t.name}'." if t else f"No note [{note_id}]."


@mcp.tool(title="Merge topics")
def merge_topics(source: str, target: str) -> str:
    """Fold one topic into another when they turn out to be the same thing. Afterwards,
    refresh the target's summary with update_summary.

    Args:
        source: topic to remove
        target: topic that keeps all the notes
    """
    t = store.merge_topics(source, target)
    return (f"Merged '{source}' into '{t.name}'. Refresh its summary." if t
            else "Both topics must exist and be different.")


@mcp.tool(title="Forget note", annotations=ToolAnnotations(destructiveHint=True))
def forget(note_id: int) -> str:
    """Delete a note. Only when the user asks you to forget something.

    Args:
        note_id: the number in square brackets
    """
    return f"Forgot note [{note_id}]." if store.delete_note(note_id) else f"No note [{note_id}]."


def http_app():
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        return app

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            qs = dict(kv.split("=", 1) for kv in scope.get("query_string", b"").decode().split("&") if "=" in kv)
            if qs.get("token") != token:
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)

    return guarded


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run("stdio")
    else:
        import uvicorn
        uvicorn.run(http_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
