#!/usr/bin/env python3
"""
SessionStart hook: ensure statusLine is present in settings.json.
Claude Code strips unknown fields on exit; this re-injects on next start.
"""
import json, os, sys

SETTINGS = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
STATUSLINE = {
    "type": "command",
    "command": "python C:/Users/35955/.claude/statusline-command.py"
}

try:
    with open(SETTINGS, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("statusLine") != STATUSLINE:
        data["statusLine"] = STATUSLINE
        with open(SETTINGS, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(json.dumps({"hookSpecificOutput": {"statusLineEnsured": True}}))
except Exception as e:
    print(json.dumps({"hookSpecificOutput": {"statusLineEnsured": False, "error": str(e)}}))
