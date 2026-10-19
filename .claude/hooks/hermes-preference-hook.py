#!/usr/bin/env python3
"""
Hermes Preference Hook
----------------------
Scans user input for preference signals and stores them as structured
profile attributes in the profile_attributes table.
Reads JSON from stdin: {"prompt": "...", "session_id": "..."}
Outputs nothing to stdout on success.
5-second hard timeout.
Uses HookWriter for multi-process safe writes.
"""

import sys
import json
import re
import signal
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 统一初始化
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helper import init_hook_env, get_hook_db_path

init_hook_env()

# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _Timeout()

# Install alarm where available
if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _alarm_handler)

# ---------------------------------------------------------------------------
# Preference patterns
# ---------------------------------------------------------------------------

# Each entry: (tag_name, compiled_regex)
_PATTERNS = [
    # Tool preference (must be before explicit — "用python代替bash" matches both)
    ("tool", re.compile(
        r"(?:用\s*(?:python|bash|git|node|npm|yarn|pnpm|docker|vim|nvim|vscode)\s*(?:不(?:用|要)|代替|替换|而非))",
        re.IGNORECASE,
    )),
    # Avoidance (must be before explicit — "不要用bash" matches both)
    ("avoid", re.compile(
        r"(?:不要|不喜欢|别用|别再|禁止|免了|算了|讨厌|受不了)",
        re.IGNORECASE,
    )),
    # Language preference (must be before explicit — "用中文回复" matches both)
    ("language", re.compile(
        r"(?:用中文|中文回复|用英文|英文回复|speak\s+(?:chinese|english)|reply\s+in\s+(?:chinese|english))",
        re.IGNORECASE,
    )),
    # Style preference
    ("style", re.compile(
        r"(?:极简|暗色|深色|亮色|彩色|好看|简约|扁平|拟物|毛玻璃|neumorphism|minimal(?:ist)?|dark\s*mode|light\s*mode)",
        re.IGNORECASE,
    )),
    # Explicit preference (broad — check last among overlapping patterns)
    ("explicit", re.compile(
        r"(?:我(?:喜欢|想要|偏好|希望|习惯|觉得应该)|(?:请|帮忙)?(?:用|使用|换成|改用|选择)[\w一-鿿]+)",
        re.IGNORECASE,
    )),
    # Efficiency preference
    ("efficiency", re.compile(
        r"(?:快点|不要废话|直接(?:做|给)|少说多做|别解释|简短|简洁|精简|concise|brief|just\s+do\s+it)",
        re.IGNORECASE,
    )),
]

# ---------------------------------------------------------------------------
# Tag → profile dimension mapping
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "explicit":   ("preferences", "user_statement"),
    "avoid":      ("preferences", "avoidance"),
    "tool":       ("tools",       "tool_preference"),
    "style":      ("preferences", "style"),
    "language":   ("identity",    "language"),
    "efficiency": ("efficiency",  "work_style"),
}

_CATEGORY_CONFIDENCE = {
    "explicit": 0.4, "avoid": 0.4, "tool": 0.3,
    "style": 0.3, "language": 0.5, "efficiency": 0.4,
}

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_preferences(prompt: str):
    """Yield (pref_type, matched_text) for each detected preference signal."""
    if not prompt or not isinstance(prompt, str):
        return

    # Skip very long inputs (likely pasted code / logs, not conversational)
    if len(prompt) > 2000:
        return

    for tag, pattern in _PATTERNS:
        for match in pattern.finditer(prompt):
            text = match.group(0).strip()
            if text:
                yield tag, text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Start alarm (5 s)
    if hasattr(signal, "SIGALRM"):
        signal.alarm(5)

    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", "")

        if not prompt or len(prompt) > 2000:
            return

        from _hook_writer import HookWriter

        db_path = get_hook_db_path()

        seen = set()
        extracted = []
        for pref_type, pref_text in extract_preferences(prompt):
            key_text = f"{pref_type}:{pref_text}"
            if key_text in seen:
                continue
            seen.add(key_text)
            extracted.append((pref_type, pref_text))

        if not extracted:
            return

        # Use HookWriter for multi-process safe writes
        with HookWriter(db_path, timeout_sec=3.0) as writer:
            for pref_type, pref_text in extracted:
                cat, attr_key = _CATEGORY_MAP[pref_type]
                conf = _CATEGORY_CONFIDENCE[pref_type]

                # Check if attribute already exists; merge values if different
                existing = writer.profile_get(attr_key=attr_key)
                if existing:
                    old_val = existing[0]["attr_value"]
                    if pref_text in old_val:
                        continue  # duplicate
                    new_val = f"{old_val}; {pref_text}"
                else:
                    new_val = pref_text

                writer.profile_set(attr_key, new_val, cat, conf, "auto-preference")

    except (_Timeout, Exception):
        # Silently swallow all errors — hook must never break the host
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # Cancel alarm


if __name__ == "__main__":
    main()
