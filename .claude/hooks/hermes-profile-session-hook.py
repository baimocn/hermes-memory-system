#!/usr/bin/env python3
"""
Hermes Session Hook — SessionStart
注入三块内容到会话上下文：
  1. 用户画像摘要
  2. 近期记忆（最近 7 天）
  3. 高置信度教训
5-second hard timeout.
"""
import sys
import json
import signal
import time


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _alarm_handler)


def _build_session_output(db_path: str) -> dict:
    """Query profile, recent memories, and lessons."""
    sys.path.insert(0, "C:/Users/35955/.claude/mcp-servers")
    from vmem.store import MemoryVectorStore

    store = MemoryVectorStore(db_path)
    conn = store._get_conn()
    now = int(time.time())

    sections = []

    # --- 1. 用户画像 ---
    profile = store.profile_summary()
    sections.append(f"## User Profile\n\n{profile}")

    # --- 2. 近期记忆（最近 7 天，按置信度降序，取 top 10）---
    week_ago = now - 7 * 86400
    recent = conn.execute(
        """SELECT key, value, source, confidence
           FROM memory_vectors
           WHERE updated_at > ? AND importance != 'archived'
           ORDER BY confidence DESC
           LIMIT 10""",
        (week_ago,),
    ).fetchall()

    if recent:
        lines = ["## Recent Memories (7 days)"]
        for key, value, source, conf in recent:
            conf_pct = f"{conf:.0%}" if conf else "50%"
            preview = value[:100].replace("\n", " ")
            lines.append(f"- [{key}] ({source}, {conf_pct}) {preview}")
        sections.append("\n".join(lines))

    # --- 3. 高置信度教训（confidence >= 0.7）---
    lessons = conn.execute(
        """SELECT key, value, confidence
           FROM memory_vectors
           WHERE (source LIKE '%lesson%' OR tags LIKE '%lesson%')
             AND confidence >= 0.7
           ORDER BY confidence DESC
           LIMIT 5"""
    ).fetchall()

    if lessons:
        lines = ["## Important Lessons"]
        for key, value, conf in lessons:
            preview = value[:150].replace("\n", " ")
            lines.append(f"- [{key}] ({conf:.0%}) {preview}")
        sections.append("\n".join(lines))

    # --- 4. 活跃因果链摘要 ---
    causal_count = conn.execute("SELECT COUNT(*) FROM causal_links").fetchone()[0]
    contradiction_count = conn.execute(
        "SELECT COUNT(*) FROM memory_links WHERE link_type = 'contradicts'"
    ).fetchone()[0]

    if causal_count > 0 or contradiction_count > 0:
        sections.append(
            f"## System Status\n\n"
            f"- Causal chains: {causal_count}\n"
            f"- Unresolved contradictions: {contradiction_count}"
        )

    store.close()

    context = "\n\n".join(sections) if sections else "(no memory data yet)"

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context
        }
    }


def main():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(5)
    try:
        raw = sys.stdin.read()
        output = _build_session_output("C:/Users/35955/.claude/memory/vmem.db")
        print(json.dumps(output, ensure_ascii=False))
    except (_Timeout, Exception):
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
