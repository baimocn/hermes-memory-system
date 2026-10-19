#!/usr/bin/env python3
"""
Hermes Recall Hook — UserPromptSubmit
每次用户发消息时，自动搜索相关记忆并注入上下文。
让 Claude 在回答前就能看到相关历史，而非事后才想起来。

5-second hard timeout.
通过 _helper 统一初始化，复用 HookWriter 安全访问 DB。
"""
import sys
import json
import signal
import os

# 统一初始化：确保 vmem 包可导入、获取正确 DB 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helper import init_hook_env, get_hook_db_path

init_hook_env()


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _alarm_handler)


def _build_recall_output(prompt: str, cwd: str = "") -> dict:
    """Search for related memories and format as hook output."""
    if not prompt or len(prompt) < 5 or len(prompt) > 2000:
        return {}

    from _hook_writer import HookWriter
    from _helper import derive_project
    from vmem.store import EmbeddingEngine, search_fusion

    db_path = get_hook_db_path()
    engine = EmbeddingEngine()
    project = derive_project(cwd)

    # 通过 HookWriter 复用连接池，保证多进程安全
    with HookWriter(db_path, timeout_sec=3.0) as writer:
        results = search_fusion(
            store=writer.store,
            encoder=engine,
            query=prompt,
            top_k=3,
            project_filter=project if project else None,
        )

    if not results:
        return {}

    filtered = [r for r in results if r.get("score", 0) > 0.3]
    if not filtered:
        return {}

    lines = ["## Related Memories (auto-recalled)"]
    for r in filtered:
        key = r.get("key", "")
        value = r.get("value", "")
        source = r.get("source", "")
        score = round(r.get("score", 0), 2)
        preview = value[:200].replace("\n", " ")
        lines.append(f"- [{key}] ({source}, score:{score}) {preview}")

    context = "\n".join(lines)

    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context
        }
    }


def main():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(5)
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", "")
        cwd = data.get("cwd", "")

        if not prompt:
            return

        output = _build_recall_output(prompt, cwd=cwd)

        if output:
            print(json.dumps(output, ensure_ascii=False))

    except (_Timeout, Exception):
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
