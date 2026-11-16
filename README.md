# OneThread

One chat box. You talk about anything, in any order. It files what you say into topics on its own, and the next time you bring up something related it already has that topic's memory in hand.

You don't open a "diet" chat or a "startup" chat. You just talk, the way you would with a friend who remembers.

## How it works

Every message goes through four steps:

1. **Segment and route.** The model splits the message if it covers two things ("skipped the gym today; also the puzzle game playtest is done") and files each piece under an existing topic, or opens a new one.
2. **Recall.** For each matched topic it pulls the running summary, the most similar past snippets (embeddings), and the latest exchange. Strong matches in *other* topics come back as associations ("this connects to your startup topic").
3. **Answer.** Claude replies with that memory in context, without you restating anything.
4. **Update.** After replying, each touched topic's summary is rewritten in the background.

Each message shows which topic it was filed under. If it's wrong, click the tag and move it; both topics' summaries are rebuilt.

```
message ─▶ segment + route ─▶ recall (summary + similar snippets + associations) ─▶ answer
                                                                  └─▶ update topic summaries
```

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn onethread.server:app --reload
# open http://localhost:8000
```

Try it without a key (scripted replies, keyword routing):

```bash
ONETHREAD_MOCK=1 uvicorn onethread.server:app
```

Better recall: `pip install sentence-transformers`. On first run it downloads `intfloat/multilingual-e5-small`. Without it, a built-in character n-gram embedder is used.

| Variable | Default | What it does |
| --- | --- | --- |
| `ONETHREAD_MODEL` | `claude-sonnet-5-5` | model that writes replies |
| `ONETHREAD_FAST_MODEL` | `claude-haiku-5-5` | model for routing and summaries |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | `hash` forces the offline embedder |
| `ONETHREAD_RELATED_THRESHOLD` | per embedder | how similar another topic must be to count as an association |
| `ONETHREAD_DB` | `data/onethread.db` | SQLite file |

## Layout

```
onethread/engine.py   the pipeline: route, recall, answer, summarize, correct
onethread/db.py       SQLite: topics, messages, segments
onethread/embed.py    embeddings (local model or offline fallback)
onethread/llm.py      Claude API wrapper + mock for tests
onethread/server.py   FastAPI endpoints
static/index.html     the chat UI
tests/                pytest, runs offline with the mock
```

## Cost

Each message makes three Claude calls: routing (fast model), the reply, and a summary update per touched topic (fast model). Embeddings run locally.

## Not done yet

- Merging and splitting topics automatically as they drift
- Forgetting: decay for memories that never come up again
- Importing existing chat history to bootstrap topics
- Measuring routing accuracy on a labeled set of real conversations
