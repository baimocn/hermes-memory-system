#!/usr/bin/env python3
"""
Hermes Introspect Hook — SessionEnd
Runs lightweight self-reflection: GC stale memories, check health.
Reads stdin JSON, outputs health summary to stdout.
5-second hard timeout.
Uses HookWriter for multi-process safe writes.
"""
import sys
import json
import signal
import os

# ---------------------------------------------------------------------------
# 统一初始化
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helper import init_hook_env, get_hook_db_path

init_hook_env()


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _alarm_handler)


def _run_auto_dedupe(writer, total_memories: int) -> dict:
    """
    当记忆总数超过 500 条时，自动运行去重合并。
    使用 0.95 的高阈值，只合并高度重复的记忆。
    返回 {merged_count: int, pairs_found: int}
    """
    AUTO_DEDUPE_THRESHOLD = 500
    DEDUPE_THRESHOLD = 0.95
    MAX_MERGES_PER_SESSION = 20  # 限制单次合并数量，避免 hook 超时

    result = {"pairs_found": 0, "merged_count": 0}

    if total_memories < AUTO_DEDUPE_THRESHOLD:
        return result

    try:
        duplicates = writer.find_duplicates(threshold=DEDUPE_THRESHOLD)
        result["pairs_found"] = len(duplicates)

        if not duplicates:
            return result

        # 只合并相似度最高的若干对，避免超时
        merged_keys = set()
        for pair in duplicates[:MAX_MERGES_PER_SESSION]:
            key_a = pair["key_a"]
            key_b = pair["key_b"]
            # 跳过已被合并的 key
            if key_a in merged_keys or key_b in merged_keys:
                continue
            merge_result = writer.merge_memories(
                key_a, key_b, strategy="keep_richer",
            )
            if merge_result and "survivor_key" in merge_result:
                result["merged_count"] += 1
                merged_keys.add(merge_result["survivor_key"])
                merged_keys.add(merge_result["merged_key"])
    except Exception:
        pass  # 去重失败不影响主流程

    return result


def _build_introspect_output() -> dict:
    """Run introspection and format as hook output.
    Uses HookWriter for multi-process safe DB access.
    """
    from _hook_writer import HookWriter

    db_path = get_hook_db_path()

    with HookWriter(db_path, timeout_sec=3.0) as writer:
        # Run GC
        archived = writer.introspect_gc()

        # Get health
        health = writer.introspect_health()

        # Auto-dedupe (only when memory count > 500)
        total = health.get("total_memories", 0)
        dedupe_result = _run_auto_dedupe(writer, total)

        # Re-fetch health after dedupe
        if dedupe_result.get("merged_count", 0) > 0:
            health = writer.introspect_health()

    summary_parts = []
    if archived:
        summary_parts.append(f"Archived {len(archived)} stale memories")
    if health.get("contradictions", 0) > 0:
        summary_parts.append(f"{health['contradictions']} unresolved contradictions")
    if health.get("gc_candidates", 0) > 0:
        summary_parts.append(f"{health['gc_candidates']} more GC candidates")
    if dedupe_result.get("merged_count", 0) > 0:
        summary_parts.append(
            f"Merged {dedupe_result['merged_count']} duplicate pairs"
        )
    if not summary_parts:
        summary_parts.append("System healthy")

    context = f"## Memory Health\n\nTotal: {health.get('total_memories', 0)} memories, "
    context += f"Avg confidence: {health.get('avg_confidence', 0):.1%}\n"
    context += "; ".join(summary_parts)

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionEnd",
            "additionalContext": context
        }
    }


def main():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(5)
    try:
        raw = sys.stdin.read()
        output = _build_introspect_output()
        print(json.dumps(output, ensure_ascii=False))
    except (_Timeout, Exception):
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

if __name__ == "__main__":
    main()
