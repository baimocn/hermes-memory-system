#!/usr/bin/env python3
"""
Hermes Profile Session Hook — SessionStart
Reads stdin JSON ({"session_id": "..."}), queries user profile summary,
outputs JSON with additionalContext for Claude Code to inject.
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


def _build_session_output(db_path: str) -> dict:
    """Query profile summary and format as hook output JSON."""
    sys.path.insert(0, "C:/Users/35955/.claude/mcp-servers")
    from vmem.store import MemoryVectorStore

    store = MemoryVectorStore(db_path)
    summary = store.profile_summary()
    store.close()

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"## User Profile\n\n{summary}"
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
