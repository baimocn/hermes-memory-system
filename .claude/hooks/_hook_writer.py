"""
HookWriter — 多进程安全的 vmem 写入包装器
供 hooks 使用，提供 context manager 接口和便捷写入方法。
"""

import os
import sys

# 确保 vmem 包可导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mcp-servers'))


class HookWriter:
    """
    轻量包装 MemoryVectorStore，提供：
    - context manager（with 语句）
    - store_memory() 便捷写入（自动编码 embedding）
    - causal_add() 因果链接
    - .store 属性直接访问底层 store

    Args:
        db_path: SQLite 数据库路径。
        timeout_sec: 超时秒数（记录用，不阻塞）。
    """

    def __init__(self, db_path: str, timeout_sec: float = 5.0):
        from vmem.store import MemoryVectorStore
        self._db_path = db_path
        self._timeout = timeout_sec
        self._store = None
        self._encoder = None

    def __enter__(self):
        from vmem.store import MemoryVectorStore, EmbeddingEngine
        self._store = MemoryVectorStore(self._db_path)
        try:
            self._encoder = EmbeddingEngine()
        except Exception:
            self._encoder = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
        return False

    @property
    def store(self):
        """直接访问底层 MemoryVectorStore 实例。"""
        return self._store

    def store_memory(
        self,
        key: str,
        value: str,
        source: str = "",
        level: str = "",
        tags: str = "",
        project: str = "",
    ) -> bool:
        """写入一条记忆，自动编码 embedding（超时则无向量降级）。"""
        if not self._store:
            return False
        try:
            embedding = None
            if self._encoder:
                try:
                    embedding = self._encoder.encode([value])[0]
                except Exception:
                    pass  # 降级：无向量写入
            self._store.store(
                key=key, value=value, source=source,
                level=level, tags=tags, embedding=embedding, project=project,
            )
            return True
        except Exception:
            return False

    def causal_add(self, from_key: str, to_key: str, relation: str, strength: float = 1.0) -> bool:
        """添加因果关系链接。"""
        if not self._store:
            return False
        try:
            return self._store.causal_add(from_key, to_key, relation, strength)
        except Exception:
            return False
