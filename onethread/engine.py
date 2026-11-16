"""The memory pipeline. Every user message goes through:

    segment + route  ->  recall  ->  answer  ->  (later) update topic summaries

Topics are never chosen by the user: the model files each piece of a message under an
existing topic or opens a new one, and the next time something similar comes up the
right topic's memory is pulled back in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .db import DB, Segment
from .embed import cosine
from .llm import extract_json

ROUTE_SYSTEM = """你是一个记忆整理器。用户只有一个对话入口，想到什么说什么。
你的工作是把用户这条消息切成若干片段，并把每个片段归到一个话题下。

规则：
- 一条消息通常只属于一个话题；只有明显在说两件不同的事时才切分。
- 优先归到已有话题。只有在跟所有已有话题都无关时才新建话题。
- 新话题的名字要短（2 到 8 个字），是用户会认得的说法，比如"减肥"、"跳槽"、"解谜游戏创业"。
- 寒暄、感谢、"好的"这类没有内容的话，归到最近活跃的话题；如果没有话题，就新建"闲聊"。
- 片段文本保留用户原话，不要改写。

只输出 JSON，格式：
{"segments": [{"text": "...", "topic_id": 3 或 null, "new_topic": null 或 "新话题名"}]}"""

ANSWER_SYSTEM = """你是用户的长期对话伙伴。用户只用一个对话框跟你聊所有事，不会告诉你现在在聊哪个话题。
下面是系统根据用户这条消息自动找回的记忆。像一个记性好的朋友那样自然地用上它们：
- 不要说"根据记忆"、"我查到"之类的话，直接接着聊。
- 如果记忆里有和当前问题相关的事实、决定或进展，用上它，别让用户重复交代。
- 如果"相关联想"里的内容跟当前话题确实有关，可以顺带提一句，但别硬凑。
- 用用户使用的语言回答，简洁、具体。

<memory>
{memory}
</memory>"""

SUMMARY_SYSTEM = """你维护一个话题的摘要，供以后的对话快速回忆。
根据旧摘要和新的对话，输出更新后的摘要：
- 只保留长期有用的信息：事实、决定、计划、偏好、进展、未解决的问题。
- 新信息和旧信息冲突时，以新的为准，并简短注明变化（例如"原计划用 Java，后改为 Python"）。
- 不超过 150 字，用要点式短句，不要寒暄。
只输出摘要本身。"""


@dataclass
class RoutedSegment:
    id: int
    text: str
    topic_id: int
    topic_name: str
    is_new_topic: bool


@dataclass
class TurnResult:
    user_message_id: int
    assistant_message_id: int
    segments: list[RoutedSegment]
    related: list[dict] = field(default_factory=list)  # cross-topic associations
    reply: str = ""


class Engine:
    def __init__(self, db: DB, llm, embedder, *, max_route_topics: int = 15,
                 recall_k: int = 4, related_threshold: float | None = None):
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.max_route_topics = max_route_topics
        self.recall_k = recall_k
        env = os.environ.get("ONETHREAD_RELATED_THRESHOLD")
        self.related_threshold = (related_threshold if related_threshold is not None
                                  else float(env) if env
                                  else getattr(embedder, "related_threshold", 0.55))

    # ------------------------------------------------------------------ routing
    def _topic_centroid(self, topic_id: int) -> np.ndarray | None:
        embs = [s.embedding for s in self.db.topic_segments(topic_id) if s.embedding is not None]
        if not embs:
            return None
        c = np.mean(embs, axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c

    def _candidate_topics(self, msg_emb: np.ndarray):
        topics = self.db.list_topics()
        if len(topics) <= self.max_route_topics:
            return topics
        # keep the prompt short: most similar topics plus the most recently active ones
        scored = sorted(topics, key=lambda t: cosine(msg_emb, self._topic_centroid(t.id)), reverse=True)
        keep = {t.id: t for t in scored[: self.max_route_topics - 3]}
        for t in topics[:3]:  # list_topics is ordered by recency
            keep.setdefault(t.id, t)
        return list(keep.values())

    def route(self, text: str, msg_emb: np.ndarray) -> list[dict]:
        topics = self._candidate_topics(msg_emb)
        topic_xml = "\n".join(
            f'<topic id="{t.id}" name="{t.name}">{t.summary[:200]}</topic>' for t in topics
        ) or "（还没有任何话题）"
        recent = self.db.recent_messages(4)[:-1]  # context before this message
        recent_txt = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent) or "（无）"
        prompt = (f"<topics>\n{topic_xml}\n</topics>\n\n<recent>\n{recent_txt}\n</recent>\n\n"
                  f"<message>\n{text}\n</message>")
        raw = self.llm.complete("route", ROUTE_SYSTEM, [{"role": "user", "content": prompt}], 800)
        try:
            segs = extract_json(raw).get("segments") or []
        except Exception:
            segs = []
        valid_ids = {t.id for t in self.db.list_topics()}
        out = []
        for s in segs:
            seg_text = (s.get("text") or "").strip()
            if not seg_text:
                continue
            tid = s.get("topic_id")
            out.append({"text": seg_text,
                        "topic_id": tid if tid in valid_ids else None,
                        "new_topic": (s.get("new_topic") or "").strip() or None})
        if not out:  # model reply unusable: keep the whole message, open a topic for it
            out = [{"text": text, "topic_id": None, "new_topic": "未分类"}]
        return out

    def _resolve_topic(self, seg: dict) -> tuple[int, bool]:
        if seg["topic_id"]:
            return seg["topic_id"], False
        name = seg["new_topic"] or "未分类"
        existing = self.db.find_topic_by_name(name)
        if existing:
            return existing.id, False
        return self.db.create_topic(name), True

    # ------------------------------------------------------------------- recall
    def _recall_topic(self, topic_id: int, query_emb: np.ndarray, exclude_msg: int) -> list[Segment]:
        segs = [s for s in self.db.topic_segments(topic_id) if s.message_id != exclude_msg]
        if not segs:
            return []
        by_sim = sorted(segs, key=lambda s: cosine(query_emb, s.embedding), reverse=True)
        picked = {s.id: s for s in by_sim[: self.recall_k]}
        for s in segs[-2:]:  # always include the latest exchange in this topic
            picked.setdefault(s.id, s)
        return sorted(picked.values(), key=lambda s: s.id)

    def _related(self, query_emb: np.ndarray, own_topics: set[int], exclude_msg: int) -> list[dict]:
        """Human-like association: strong matches that live in *other* topics."""
        best: dict[int, tuple[float, Segment]] = {}
        for t in self.db.list_topics():
            if t.id in own_topics:
                continue
            for s in self.db.topic_segments(t.id):
                if s.message_id == exclude_msg or s.role != "user":
                    continue
                sim = cosine(query_emb, s.embedding)
                if sim >= self.related_threshold and sim > best.get(t.id, (0, None))[0]:
                    best[t.id] = (sim, s)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:2]
        return [{"topic_id": tid, "topic_name": self.db.get_topic(tid).name,
                 "text": seg.text, "score": round(sim, 3)} for tid, (sim, seg) in ranked]

    def _memory_block(self, topic_ids: list[int], query_emb: np.ndarray, exclude_msg: int,
                      related: list[dict]) -> str:
        parts = []
        for tid in topic_ids:
            t = self.db.get_topic(tid)
            recalled = self._recall_topic(tid, query_emb, exclude_msg)
            lines = "\n".join(f"- [{s.role}] {s.text[:300]}" for s in recalled) or "- （这是新话题，还没有历史）"
            parts.append(f'<topic name="{t.name}">\n摘要：{t.summary or "（暂无）"}\n相关片段：\n{lines}\n</topic>')
        if related:
            rel = "\n".join(f'- 「{r["topic_name"]}」里提到过：{r["text"][:200]}' for r in related)
            parts.append(f"<related>\n相关联想：\n{rel}\n</related>")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------- a turn
    def handle(self, text: str) -> TurnResult:
        msg_id = self.db.add_message("user", text)
        msg_emb = self.embedder.embed(text)

        routed: list[RoutedSegment] = []
        for seg in self.route(text, msg_emb):
            tid, is_new = self._resolve_topic(seg)
            sid = self.db.add_segment(msg_id, tid, "user", seg["text"], self.embedder.embed(seg["text"]))
            self.db.touch_topic(tid)
            routed.append(RoutedSegment(sid, seg["text"], tid, self.db.get_topic(tid).name, is_new))

        topic_ids = list(dict.fromkeys(r.topic_id for r in routed))
        self.db.set_message_topics(msg_id, topic_ids)

        related = self._related(msg_emb, set(topic_ids), msg_id)
        memory = self._memory_block(topic_ids, msg_emb, msg_id, related)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in self.db.recent_messages(7)[:-1]]
        while history and history[0]["role"] != "user":  # API needs to start with user
            history.pop(0)
        reply = self.llm.complete("answer", ANSWER_SYSTEM.replace("{memory}", memory),
                                  history + [{"role": "user", "content": text}], 1500)

        reply_id = self.db.add_message("assistant", reply, topic_ids)
        reply_emb = self.embedder.embed(reply)
        for tid in topic_ids:
            self.db.add_segment(reply_id, tid, "assistant", reply, reply_emb)

        return TurnResult(msg_id, reply_id, routed, related, reply)

    # ---------------------------------------------------------------- summaries
    def update_summaries(self, result: TurnResult):
        """Fold the new exchange into each touched topic's summary. Run after replying."""
        for tid in dict.fromkeys(r.topic_id for r in result.segments):
            t = self.db.get_topic(tid)
            said = "\n".join(r.text for r in result.segments if r.topic_id == tid)
            new = f"用户：{said}\n助手：{result.reply[:800]}"
            self._summarize(tid, t.summary, new)

    def rebuild_summary(self, topic_id: int):
        """Re-derive a summary from the raw segments (after a correction, or to stop drift)."""
        segs = self.db.topic_segments(topic_id)[-30:]
        if not segs:
            self.db.update_topic(topic_id, summary="")
            return
        new = "\n".join(f"{'用户' if s.role == 'user' else '助手'}：{s.text[:400]}" for s in segs)
        self._summarize(topic_id, "", new)

    def _summarize(self, topic_id: int, old: str, new: str):
        t = self.db.get_topic(topic_id)
        prompt = f"话题：{t.name}\n\n<old>\n{old or '（无）'}\n</old>\n\n<new>\n{new}\n</new>"
        summary = self.llm.complete("summarize", SUMMARY_SYSTEM,
                                    [{"role": "user", "content": prompt}], 400).strip()
        if summary:
            self.db.update_topic(topic_id, summary=summary)

    # --------------------------------------------------------------- correction
    def move_segment(self, segment_id: int, *, topic_id: int | None = None,
                     new_topic: str | None = None) -> int:
        """The user says 'this belongs elsewhere'. Returns the target topic id."""
        seg = self.db.get_segment(segment_id)
        if seg is None:
            raise KeyError(segment_id)
        if topic_id is None:
            if not new_topic:
                raise ValueError("need topic_id or new_topic")
            found = self.db.find_topic_by_name(new_topic)
            topic_id = found.id if found else self.db.create_topic(new_topic)
        old_tid = seg.topic_id
        if old_tid == topic_id:
            return topic_id
        self.db.move_segment(segment_id, topic_id)
        # keep the message's topic list and the assistant copy of the reply in step
        user_tids = list(dict.fromkeys(s.topic_id for s in self.db.message_segments(seg.message_id)))
        self.db.set_message_topics(seg.message_id, user_tids)
        reply = self.db.conn.execute(
            "SELECT id FROM messages WHERE id > ? AND role='assistant' ORDER BY id LIMIT 1",
            (seg.message_id,)).fetchone()
        if reply:
            reply_segs = self.db.message_segments(reply["id"])
            have = {s.topic_id for s in reply_segs}
            for s in reply_segs:
                if s.topic_id == old_tid and old_tid not in user_tids:
                    if topic_id in have:
                        self.db.conn.execute("DELETE FROM segments WHERE id=?", (s.id,))
                        self.db.conn.commit()
                    else:
                        self.db.move_segment(s.id, topic_id)
            self.db.set_message_topics(reply["id"], user_tids)
        for tid in (old_tid, topic_id):
            self.rebuild_summary(tid)
        return topic_id
