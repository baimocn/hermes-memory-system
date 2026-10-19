"""
Hermes Hook Helper — 路径自动检测与统一初始化
所有 hooks import 此模块获取路径和初始化函数，无需各自硬编码 sys.path。
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

_HOME = Path.home()
_MCP_DIR = _HOME / ".claude" / "mcp-servers"
_HOOKS_DIR = Path(__file__).parent
_DB_PATH = str(_HOME / ".claude" / "memory" / "vmem.db")


def init_hook_env():
    """统一初始化 hook 运行环境。

    1. 将 hooks 目录加入 sys.path（使 _hook_writer 等模块可导入）
    2. 将 vmem 包目录加入 sys.path（使 vmem.store 等模块可导入）

    所有 hook 入口文件应在顶部调用此函数。
    """
    # hooks 目录（_hook_writer.py 等模块所在目录）
    hooks_dir = str(_HOOKS_DIR)
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)

    # vmem 包目录
    mcp_dir = str(_MCP_DIR)
    if mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)


def get_hook_db_path() -> str:
    """获取 vmem.db 路径（统一入口）。

    优先级：HMS_DB_PATH 环境变量 > vmem.config 模块 > 默认路径。
    """
    env = os.environ.get("HMS_DB_PATH")
    if env:
        return env

    # 尝试使用 vmem.config（支持 config.json 覆盖）
    try:
        from vmem.config import DB_PATH
        return DB_PATH
    except Exception:
        return _DB_PATH


# ---------------------------------------------------------------------------
# 向后兼容：保留旧接口
# ---------------------------------------------------------------------------

# 确保 vmem 可导入（向后兼容，新代码应使用 init_hook_env()）
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))


def get_db_path() -> str:
    """获取 vmem.db 路径。（向后兼容别名，新代码请用 get_hook_db_path()）"""
    return get_hook_db_path()


def get_store():
    """获取 MemoryVectorStore 实例。"""
    from vmem.store import MemoryVectorStore
    return MemoryVectorStore(get_hook_db_path())


def derive_project(cwd: str = "") -> str:
    """从工作目录推断项目标识符。

    规则：
    - 空路径或用户主目录 → 返回空字符串（全局记忆）
    - 否则取路径的最后一段文件夹名作为 project 标识
    """
    if not cwd:
        return ""
    p = Path(cwd).resolve()
    # 用户主目录本身不算项目
    if p == _HOME:
        return ""
    return p.name
