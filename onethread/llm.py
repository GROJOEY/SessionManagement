"""Thin wrapper around the Claude API, plus a scripted mock for offline tests."""

from __future__ import annotations

import json
import os
import re


class ClaudeLLM:
    def __init__(self, model: str | None = None):
        import anthropic  # lazy so tests don't need a key

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("ONETHREAD_MODEL", "claude-sonnet-5-5")
        # a cheaper model is fine for the bookkeeping calls (routing, summaries)
        self.fast_model = os.environ.get("ONETHREAD_FAST_MODEL", "claude-haiku-5-5")

    def complete(self, task: str, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        model = self.model if task == "answer" else self.fast_model
        resp = self.client.messages.create(
            model=model, system=system, messages=messages, max_tokens=max_tokens
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class MockLLM:
    """Routes by keyword so the whole pipeline can be exercised without a key."""

    KEYWORDS = {
        "减肥": ["减肥", "体重", "健身", "卡路里", "跑步"],
        "创业": ["创业", "解谜", "游戏", "产品", "用户"],
        "跳槽": ["跳槽", "面试", "刷题", "offer", "谷歌"],
    }

    def complete(self, task: str, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        last = messages[-1]["content"]
        if task == "route":
            msg = re.search(r"<message>(.*?)</message>", last, re.S).group(1).strip()
            existing = re.findall(r'<topic id="(\d+)" name="([^"]+)"', last)
            parts = [p.strip() for p in re.split(r"[。；;]|另外|还有", msg) if p.strip()]
            segs = []
            for p in parts:
                name = next((k for k, ws in self.KEYWORDS.items() if any(w in p for w in ws)), "闲聊")
                hit = next((tid for tid, n in existing if n == name), None)
                segs.append({"text": p, "topic_id": int(hit) if hit else None,
                             "new_topic": None if hit else name})
            return json.dumps({"segments": segs}, ensure_ascii=False)
        if task == "summarize":
            new = re.search(r"<new>(.*?)</new>", last, re.S).group(1).strip()
            return ("要点：" + new.replace("\n", " "))[:120]
        if task == "answer":
            topics = re.findall(r'<topic name="([^"]+)"', system)
            return f"（mock 回答）我接上了你的：{'、'.join(topics) or '新话题'}。"
        raise ValueError(task)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(m.group(0))
