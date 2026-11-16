# OneThread

A lightweight MCP memory server. Keep one conversation with Claude and talk about anything, in any order. Claude files what matters under topics it picks itself, and pulls the right topic back in the next time it comes up. You never say "now we're talking about my diet."

The server only stores and searches. All the judgment (which topic, what's worth keeping, how to summarize) is done by the Claude you're already talking to. No API key, no extra model, one SQLite file.

## How a turn goes

```
you: "ran 5k today, also finished the story for level 1"
 │
 ├─ recall(message)            → best-matching topics with summary + notes,
 │                               plus an index of every other topic
 ├─ Claude answers, already knowing your weight plateau and your game plan
 │
 ├─ remember("Weight loss", "Ran 5 km")
 └─ remember("Puzzle game startup", "Level 1 story done")
       └─ "summary is stale" → update_summary(...)
```

## Tools

| Tool | What it does |
| --- | --- |
| `recall(message)` | Called at the start of every message. Returns related topics and the topic index |
| `remember(topic, content)` | Files a one-line gist under a topic, creating it if needed |
| `update_summary(topic, summary)` | Rewrites a topic's running summary (prompted every 5 new notes) |
| `list_topics()` / `get_topic(topic)` | Browse what's stored |
| `move_note` / `merge_topics` / `forget` | Fix mistakes when you point them out |

The server's instructions tell Claude to call `recall` first on every message and never ask you which topic you mean. MCP can't force a call on every turn, so this works most of the time, not every time.

## Run it

```bash
pip install -r requirements.txt
```

**Claude Desktop or Claude Code (local, stdio).** Add to your MCP config:

```json
{
  "mcpServers": {
    "onethread": {
      "command": "/path/to/SessionManagement/.venv/bin/python",
      "args": ["-m", "onethread.server", "--stdio"],
      "env": {
        "PYTHONPATH": "/path/to/SessionManagement",
        "ONETHREAD_DB": "/path/to/SessionManagement/data/onethread.db"
      }
    }
  }
}
```

Use absolute paths: the app doesn't start the server from the repo folder.

For Claude Code: `claude mcp add onethread -- python3 -m onethread.server --stdio` from the repo folder.

**claude.ai custom connector (remote).** Claude connects from the cloud, so the server must be reachable from the internet:

```bash
AUTH_TOKEN=pick-a-secret python3 -m onethread.server     # http://localhost:8000/mcp
cloudflared tunnel --url http://localhost:8000             # quick public URL for testing
```

Then in Claude: Customize → Connectors → Add custom connector → `https://<your-url>/mcp?token=pick-a-secret`.

| Variable | Default | |
| --- | --- | --- |
| `ONETHREAD_DB` | `./data/onethread.db` | SQLite file |
| `EMBED_MODEL` | `auto` | uses `intfloat/multilingual-e5-small` if `sentence-transformers` is installed, else a built-in embedder with no download. Notes are re-embedded automatically when the model changes |
| `AUTH_TOKEN` | unset | require `?token=` on the remote URL |
| `PORT` | `8000` | |

## Test

```bash
ONETHREAD_DB=/tmp/test.db PORT=8099 python3 -m onethread.server &
python3 tests/test_mcp.py http://localhost:8099/mcp
```

The test plays Claude's side of a short conversation that mixes two topics and checks that each later message lands back in the right one.

## Not done yet

- OAuth instead of a URL token
- Decay: notes that never come back up slowly drop out of recall
- A small page to browse and edit topics
