"""End-to-end over HTTP, the way Claude would call the tools.

Start the server first (ONETHREAD_DB=:memory: python -m onethread.server), then:
    python tests/test_mcp.py [url]
"""

import sys

import anyio
from mcp.client.client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/mcp"


async def call(c, name, args):
    r = await c.call_tool(name, args)
    text = "\n".join(x.text for x in r.content if hasattr(x, "text"))
    print(f"\n>>> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})\n{text}")
    return text


async def main():
    async with Client(URL) as c:
        tools = [t.name for t in (await c.list_tools()).tools]
        print("tools:", tools)

        # day 1: user talks about weight loss
        out = await call(c, "recall", {"message": "最近在减肥，体重卡在80公斤不动了"})
        assert "No memory yet" in out
        await call(c, "remember", {"topic": "减肥", "content": "体重卡在 80 公斤，进入平台期"})

        # then, in the same chat, about the startup
        await call(c, "recall", {"message": "我想做一个网页解谜游戏，30分钟一局"})
        await call(c, "remember", {"topic": "解谜游戏创业", "content": "想做网页解谜游戏，每局约 30 分钟，免费，靠直播引流"})

        # one message, two topics
        await call(c, "recall", {"message": "今天跑步5公里，另外游戏的第一关剧情写完了"})
        await call(c, "remember", {"topic": "减肥", "content": "今天跑步 5 公里"})
        await call(c, "remember", {"topic": "解谜游戏创业", "content": "第一关剧情写完了"})

        # later: no topic named, the right memory comes back on its own
        out = await call(c, "recall", {"message": "体重终于掉到79了"})
        assert out.splitlines()[0].startswith("## 减肥")
        out = await call(c, "recall", {"message": "第二关的谜题怎么设计比较好"})
        assert out.splitlines()[0].startswith("## 解谜游戏创业")

        # summaries go stale and Claude refreshes them
        for i in range(3):
            out = await call(c, "remember", {"topic": "减肥", "content": f"第 {i + 2} 周坚持控制饮食"})
        assert "stale" in out
        await call(c, "update_summary", {"topic": "减肥",
                                         "summary": "目标减重；曾在 80 公斤平台期，现 79 公斤；在跑步和控制饮食。"})

        # corrections
        await call(c, "remember", {"topic": "游戏", "content": "考虑用 Phaser 做前端"})
        await call(c, "merge_topics", {"source": "游戏", "target": "解谜游戏创业"})
        await call(c, "list_topics", {})
        await call(c, "get_topic", {"topic": "减肥"})
        print("\nALL CHECKS PASSED")


anyio.run(main)
