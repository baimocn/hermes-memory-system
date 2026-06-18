#!/usr/bin/env python3
"""
Hermes Introspect Hook — SessionEnd
Runs lightweight self-reflection: GC stale memories, check health.
Reads stdin JSON, outputs health summary to stdout.
5-second hard timeout.
"""
import sys
import json
import signal


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _alarm_handler)


def _build_introspect_output(db_path: str) -> dict:
    """Run introspection and format as hook output."""
    sys.path.insert(0, "C:/Users/35955/.claude/mcp-servers")
    from vmem.store import MemoryVectorStore

    store = MemoryVectorStore(db_path)

    # Run GC
    archived = store.introspect_gc()

    # Get health
    health = store.introspect_health()

    store.close()

    summary_parts = []
    if archived:
        summary_parts.append(f"Archived {len(archived)} stale memories")
    if health.get("contradictions", 0) > 0:
        summary_parts.append(f"{health['contradictions']} unresolved contradictions")
    if health.get("gc_candidates", 0) > 0:
        summary_parts.append(f"{health['gc_candidates']} more GC candidates")
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
        output = _build_introspect_output("C:/Users/35955/.claude/memory/vmem.db")
        print(json.dumps(output, ensure_ascii=False))
    except (_Timeout, Exception):
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
