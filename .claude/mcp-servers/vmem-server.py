"""
vmem MCP Server - 向量记忆服务
使用 FastMCP (mcp Python SDK) 提供向量存储和检索能力。
通过 stdio 方式被 Claude Code 调用。
"""

import sys
import os
import json
from datetime import datetime

# 确保项目根目录在 sys.path 中，以便导入 vmem 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

# 配置常量
MODEL_PATH = "C:/Users/35955/bge-small-zh-v1.5"
DB_PATH = "C:/Users/35955/.claude/memory/vmem.db"

# 延迟导入，便于调试
_store_instance = None
_encoder_instance = None


def get_encoder():
    """延迟初始化 EmbeddingEngine"""
    global _encoder_instance
    if _encoder_instance is None:
        from vmem.store import EmbeddingEngine
        _encoder_instance = EmbeddingEngine(model_path=MODEL_PATH)
    return _encoder_instance


def get_store():
    """延迟初始化 MemoryVectorStore"""
    global _store_instance
    if _store_instance is None:
        from vmem.store import MemoryVectorStore
        encoder = get_encoder()
        _store_instance = MemoryVectorStore(db_path=DB_PATH, embedding_engine=encoder)
    return _store_instance


mcp = FastMCP(
    "vmem",
    description="向量记忆服务 - 提供向量存储、语义搜索、教训检索和统计功能",
)


@mcp.tool()
def vmem_store(
    key: str,
    value: str,
    source: str = "user",
    level: str = "normal",
    tags: list[str] | None = None,
    related: list[str] | None = None,
) -> str:
    """存储一条记忆到向量数据库。

    自动调用 EmbeddingEngine.encode(value) 生成向量嵌入。

    Args:
        key: 记忆的唯一标识键
        value: 记忆内容文本
        source: 来源标识，如 user/lesson/system 等
        level: 重要等级，如 normal/important/critical
        tags: 标签列表
        related: 相关记忆的 key 列表
    """
    store = get_store()
    encoder = get_encoder()

    # 生成向量
    embedding = encoder.encode(value)
    dim = len(embedding) if embedding is not None else 0

    # tags 传入为 list，转为 JSON 数组存储（D-07）
    tags_str = json.dumps(tags, ensure_ascii=False) if tags else "[]"

    # 存储
    store.store(
        key=key,
        value=value,
        source=source,
        level=level,
        tags=tags_str,
        related=related or [],
        embedding=embedding,
    )

    return json.dumps({"ok": True, "key": key, "dim": dim}, ensure_ascii=False)


@mcp.tool()
def vmem_search(
    query: str,
    top_k: int = 10,
    source_filter: str | None = None,
    level_filter: str | None = None,
    tags_filter: list[str] | None = None,
) -> str:
    """语义搜索记忆条目。

    使用向量相似度 + 关键词融合搜索策略。

    Args:
        query: 搜索查询文本
        top_k: 返回的最大结果数
        source_filter: 按来源过滤
        level_filter: 按等级过滤
        tags_filter: 按标签过滤（任一匹配）
    """
    from vmem.store import search_fusion

    store = get_store()
    encoder = get_encoder()

    results = search_fusion(
        store=store,
        encoder=encoder,
        query=query,
        top_k=top_k,
        source_filter=source_filter,
        level_filter=level_filter,
        tags_filter=tags_filter,
    )

    # 格式化输出
    formatted = []
    for r in results:
        entry = {
            "key": r.get("key", ""),
            "value": r.get("value", ""),
            "source": r.get("source", ""),
            "importance": r.get("importance", ""),
            "tags": r.get("tags", []),
            "score": round(r.get("score", 0), 4),
            "updated_at": r.get("updated_at", ""),
        }
        formatted.append(entry)

    return json.dumps(formatted, ensure_ascii=False, indent=2)


@mcp.tool()
def vmem_recall_lessons(query: str = "", top_k: int = 5) -> str:
    """检索教训和经验。

    搜索 source="lesson" 或 tags 含 "lesson" 的条目。
    如果 query 为空，默认搜索 "教训 经验 注意事项"。

    Args:
        query: 搜索查询，为空时使用默认关键词
        top_k: 返回的最大结果数
    """
    from vmem.store import search_fusion

    store = get_store()
    encoder = get_encoder()

    effective_query = query if query.strip() else "教训 经验 注意事项"

    results = search_fusion(
        store=store,
        encoder=encoder,
        query=effective_query,
        top_k=top_k * 3,  # 多取一些，再过滤
        source_filter=None,
        level_filter=None,
        tags_filter=None,
    )

    # 过滤出 lesson 相关条目
    lesson_results = []
    for r in results:
        is_lesson = r.get("source", "") == "lesson"
        if not is_lesson:
            tags_str = r.get("tags", "")
            tags_set = set(t.strip() for t in tags_str.split(",") if t.strip()) if isinstance(tags_str, str) else set(tags_str)
            is_lesson = "lesson" in tags_set
        if is_lesson:
            lesson_results.append(r)
        if len(lesson_results) >= top_k:
            break

    # 格式化为教训列表
    if not lesson_results:
        return "未找到相关教训记录。"

    lines = ["## 教训与经验检索结果\n"]
    for i, r in enumerate(lesson_results, 1):
        score = round(r.get("score", 0), 4)
        key = r.get("key", "未知")
        value = r.get("value", "")
        tags = r.get("tags", [])
        updated = r.get("updated_at", "")
        tag_str = tags if tags else "无"
        lines.append(f"### {i}. [{key}] (相关度: {score})")
        lines.append(f"- **标签:** {tag_str}")
        lines.append(f"- **更新时间:** {updated}")
        lines.append(f"- **内容:** {value}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def vmem_stats() -> str:
    """返回向量记忆数据库的统计信息。"""
    store = get_store()

    stats = store.get_stats()

    return json.dumps(stats, ensure_ascii=False, indent=2)


@mcp.tool()
def vmem_profile_get(key: str = "", category: str = "") -> str:
    """查询用户画像属性。

    Args:
        key: 属性键名（精确查询）
        category: 画像维度类别（返回该类别所有属性）
    """
    store = get_store()
    results = store.profile_get(
        attr_key=key if key else None,
        category=category if category else None,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def vmem_profile_set(
    key: str,
    value: str,
    category: str = "preferences",
    confidence: float = 0.5,
) -> str:
    """设置用户画像属性。

    Args:
        key: 属性键名
        value: 属性值
        category: 画像维度 (identity/skills/preferences/tools/efficiency/projects)
        confidence: 置信度 (0.0-1.0)
    """
    store = get_store()
    ok = store.profile_set(
        attr_key=key,
        attr_value=value,
        category=category,
        confidence=confidence,
        source="user-explicit",
    )
    return json.dumps({"ok": ok, "key": key, "category": category})


@mcp.tool()
def vmem_causal_chain(key: str, max_depth: int = 5) -> str:
    """查询记忆的因果链。

    返回指定记忆的所有祖先节点（因果链上游），按深度排序。

    Args:
        key: 记忆键名
        max_depth: 最大查询深度（默认5）
    """
    store = get_store()
    results = store.causal_chain(key, max_depth)
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def vmem_causal_add(
    from_key: str,
    to_key: str,
    relation: str,
    strength: float = 1.0,
) -> str:
    """添加因果关系链接。

    Args:
        from_key: 原因/来源记忆键
        to_key: 结果/目标记忆键
        relation: 关系类型 (caused_by/led_to/resolved/contradicts/supersedes)
        strength: 关系强度 (0.0-1.0)
    """
    store = get_store()
    ok = store.causal_add(from_key, to_key, relation, strength)
    return json.dumps({"ok": ok, "from": from_key, "to": to_key, "relation": relation})


@mcp.tool()
def vmem_causal_get(key: str) -> str:
    """获取记忆的直接因果链接。

    返回所有 from_id 或 to_id 为指定 key 的因果链接。

    Args:
        key: 记忆键名
    """
    store = get_store()
    results = store.causal_get(key)
    return json.dumps(results, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
