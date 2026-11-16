from onethread.db import DB
from onethread.embed import HashEmbedder
from onethread.engine import Engine
from onethread.llm import MockLLM


def make():
    db = DB(":memory:")
    return db, Engine(db, MockLLM(), HashEmbedder())


def test_routes_and_reconnects():
    db, eng = make()
    r1 = eng.handle("最近在减肥，体重卡在 80 公斤")
    eng.update_summaries(r1)
    assert [s.topic_name for s in r1.segments] == ["减肥"]
    assert r1.segments[0].is_new_topic

    r2 = eng.handle("想做一个解谜游戏的创业项目")
    eng.update_summaries(r2)
    assert r2.segments[0].topic_name == "创业"

    # later, a related message lands back in the existing topic, no new topic made
    r3 = eng.handle("今天跑步了，体重掉了一点")
    assert r3.segments[0].topic_name == "减肥"
    assert not r3.segments[0].is_new_topic
    assert "减肥" in r3.reply  # mock answer echoes the recalled topics
    assert len(db.list_topics()) == 2


def test_one_message_two_topics():
    db, eng = make()
    r = eng.handle("今天没去健身。另外解谜游戏的用户测试做完了")
    names = [s.topic_name for s in r.segments]
    assert names == ["减肥", "创业"]
    assert len(db.list_topics()) == 2


def test_summary_updates():
    db, eng = make()
    r = eng.handle("减肥目标是减到 72 公斤")
    eng.update_summaries(r)
    t = db.find_topic_by_name("减肥")
    assert "72" in t.summary


def test_correction_moves_segment_and_reply():
    db, eng = make()
    r = eng.handle("今天想了想产品的事")  # mock files this under 创业
    seg = r.segments[0]
    target = eng.move_segment(seg.id, new_topic="跳槽")
    assert db.get_topic(target).name == "跳槽"
    assert db.get_segment(seg.id).topic_id == target
    # the assistant's reply followed the user's segment
    reply_segs = db.message_segments(r.assistant_message_id)
    assert {s.topic_id for s in reply_segs} == {target}


def test_bad_route_json_falls_back():
    class Broken(MockLLM):
        def complete(self, task, system, messages, max_tokens=1024):
            if task == "route":
                return "sorry, no json"
            return super().complete(task, system, messages, max_tokens)

    db = DB(":memory:")
    eng = Engine(db, Broken(), HashEmbedder())
    r = eng.handle("随便说点什么")
    assert r.segments[0].topic_name == "未分类"
