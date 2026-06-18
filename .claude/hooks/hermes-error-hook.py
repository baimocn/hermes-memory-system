#!/usr/bin/env python3
"""
hermes-error-hook.py — 工具调用失败时自动分析错误模式并存储教训到 vmem

从 stdin 读取 JSON：
  { tool_name, tool_input, error, session_id }

功能：
  1. 对 error 字符串做模式匹配
  2. 如果命中已知模式，生成因果链（问题→原因→解法）
  3. 存储到 vmem（优先带向量，超时则无向量留待 SessionEnd 补）
  4. 5 秒总超时，所有异常静默吞掉
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VMEM_DIR = Path(r"C:\Users\35955\.claude\mcp-servers")
DB_PATH = r"C:\Users\35955\.claude\memory\vmem.db"
TIMEOUT_SEC = 4.5

# ---------------------------------------------------------------------------
# 错误模式定义（含因果链扩展）
# ---------------------------------------------------------------------------

PATTERNS = [
    {
        "name": "command_not_found",
        "keywords": ["command not found", "不是内部命令", "CommandNotFoundException", "is not recognized"],
        "lesson": "命令 {cmd} 未安装或不在 PATH 中",
        "cause": "系统找不到可执行文件 {cmd}",
        "resolution": "检查 PATH 或安装对应工具",
    },
    {
        "name": "encoding_error",
        "keywords": ["encoding", "codec", "UnicodeEncodeError", "UnicodeDecodeError", "utf-8", "charmap"],
        "lesson": "{tool} 操作 {file} 时遇到编码问题",
        "cause": "文件编码与预期不符",
        "resolution": "指定正确编码或使用 errors='replace'",
    },
    {
        "name": "json_error",
        "keywords": ["json", "parse", "SyntaxError", "JSONDecodeError", "Expecting value"],
        "lesson": "{tool} 操作的文件 JSON 格式有误",
        "cause": "JSON 文件格式错误或为空",
        "resolution": "验证 JSON 格式，检查文件是否完整",
    },
    {
        "name": "permission_denied",
        "keywords": ["permission denied", "access denied", "拒绝访问", "PermissionError"],
        "lesson": "{tool} 操作 {file} 时权限不足",
        "cause": "当前用户无权访问目标文件或目录",
        "resolution": "以管理员运行或修改文件权限",
    },
    {
        "name": "file_not_found",
        "keywords": ["no such file", "not found", "找不到", "FileNotFoundError", "No such file or directory"],
        "lesson": "{tool} 操作的文件 {file} 不存在",
        "cause": "文件路径错误或文件已被删除",
        "resolution": "检查文件路径是否正确",
    },
    {
        "name": "path_backslash",
        "keywords": ["bash", "\\", "path", "No such file or directory"],
        "lesson": "{tool} 操作 {file} 时路径格式错误：必须用正斜杠 /",
        "cause": "Windows 反斜杠路径在 bash 中不被识别",
        "resolution": "统一使用正斜杠 / 作为路径分隔符",
    },
]


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------


def match_pattern(tool: str, error: str, file_path: str, command: str) -> dict | None:
    """对 error 做模式匹配，返回匹配的模式 dict 或 None。"""
    error_lower = error.lower()
    for pat in PATTERNS:
        for kw in pat["keywords"]:
            if kw.lower() in error_lower:
                return pat
    return None


def _stable_hash(text: str) -> str:
    """生成稳定短哈希，用于 key 去重。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _format_field(template: str, tool: str, file_path: str, command: str) -> str:
    """Format a template string with error context fields."""
    return template.format(
        tool=tool,
        file=file_path or "(unknown)",
        cmd=command or tool,
    )


def store_without_embedding(key: str, value: str, source: str, level: str, tags: str) -> bool:
    """直接写 SQLite，跳过 embedding 编码。"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")

        now = int(time.time())
        row = conn.execute(
            "SELECT created_at FROM memory_vectors WHERE key = ?", (key,)
        ).fetchone()
        created_at = row[0] if row else now

        tags_json = json.dumps([t.strip() for t in tags.split(",") if t.strip()], ensure_ascii=False) if tags else "[]"

        conn.execute(
            """
            INSERT OR REPLACE INTO memory_vectors
                (key, value, source, importance, tags, related, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, '[]', NULL, ?, ?)
            """,
            (key, value, source, level, tags_json, created_at, now),
        )

        conn.execute("DELETE FROM memory_fts WHERE key = ?", (key,))
        conn.execute(
            "INSERT INTO memory_fts (key, value, source, tags) VALUES (?, ?, ?, ?)",
            (key, value, source, tags),
        )

        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def store_with_embedding(key: str, value: str, source: str, level: str, tags: str) -> bool:
    """用 MemoryVectorStore 存储（含向量编码）。"""
    try:
        sys.path.insert(0, str(VMEM_DIR))
        from vmem.store import MemoryVectorStore

        with MemoryVectorStore(DB_PATH) as store:
            store.store(key=key, value=value, source=source, level=level, tags=tags)
        return True
    except Exception:
        return store_without_embedding(key, value, source, level, tags)


def store_causal_chain(tool: str, error: str, file_path: str, command: str, pattern: dict) -> bool:
    """Create a causal chain: problem → cause → resolution."""
    try:
        lesson = _format_field(pattern["lesson"], tool, file_path, command)
        cause = _format_field(pattern["cause"], tool, file_path, command)
        resolution = pattern.get("resolution", "")
        hash_val = _stable_hash(lesson)

        problem_key = f"err:problem:{hash_val}"
        cause_key = f"err:cause:{hash_val}"
        resolution_key = f"err:resolution:{hash_val}"

        # Store problem and cause entries
        store_with_embedding(problem_key, lesson, "error", "short", f"error,{tool.lower()},problem")
        store_with_embedding(cause_key, cause, "error", "short", f"error,{tool.lower()},cause")

        # Create causal link: problem ← caused_by ← cause
        sys.path.insert(0, str(VMEM_DIR))
        from vmem.store import MemoryVectorStore

        with MemoryVectorStore(DB_PATH) as store:
            store.causal_add(problem_key, cause_key, "caused_by")

            # If resolution exists, add it
            if resolution:
                store_with_embedding(resolution_key, resolution, "error", "short", f"error,{tool.lower()},resolution")
                store.causal_add(cause_key, resolution_key, "resolved")

        return True

    except Exception:
        return False


def _run_with_timeout(func, args=(), timeout_sec: float = TIMEOUT_SEC) -> bool:
    """用线程执行函数，超时则中止并返回 False。"""
    result = [False]

    def _target():
        try:
            result[0] = func(*args)
        except Exception:
            result[0] = False

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return False
    return result[0]


def main() -> None:
    """入口：从 stdin 读取数据，匹配模式，存储因果链。"""
    try:
        raw = sys.stdin.read()
        if not raw:
            return

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return

        tool = data.get("tool_name", "")
        inp = data.get("tool_input", {}) or {}
        error = str(data.get("error", "") or "")

        if not error or error.strip() == "":
            return

        file_path = inp.get("file_path", "") or inp.get("path", "") or ""
        command = inp.get("command", "") or ""

        # 模式匹配
        pattern = match_pattern(tool, error, file_path, command)
        if not pattern:
            return

        # 优先创建因果链（带超时）
        ok = _run_with_timeout(
            store_causal_chain,
            args=(tool, error, file_path, command, pattern),
            timeout_sec=TIMEOUT_SEC,
        )

        if not ok:
            # 降级：存储扁平教训（向后兼容）
            lesson = _format_field(pattern["lesson"], tool, file_path, command)
            key = f"auto-err:{tool}:{_stable_hash(lesson)}"
            store_without_embedding(key, lesson, "auto-lesson", "short", f"error,{tool.lower()},auto")

    except Exception:
        pass


if __name__ == "__main__":
    main()
