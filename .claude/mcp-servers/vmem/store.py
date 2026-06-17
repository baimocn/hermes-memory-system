"""
向量记忆存储层 — MemoryVectorStore
SQLite + FTS5 + numpy 余弦相似度 + sentence-transformers embedding
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# EmbeddingEngine — 懒加载 bge-small-zh-v1.5
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """基于 sentence-transformers 的嵌入引擎，输出 float32[512] 归一化向量。"""

    def __init__(self, model_path: str = r"C:\Users\35955\bge-small-zh-v1.5"):
        self._model_path = model_path
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_path)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为 float32[512] 归一化向量。
        返回 shape = (len(texts), 512) 的 numpy 数组。
        """
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # 确保 shape 正确
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# MemoryVectorStore
# ---------------------------------------------------------------------------

class MemoryVectorStore:
    """
    单 SQLite 文件的向量记忆存储。
    - memory_vectors: 结构化数据 + float32[512] embedding
    - memory_fts:     FTS5 trigram 全文索引 (key, value, source, tags)
    - 融合搜索: 0.60*向量 + 0.25*全文 + 0.15*时间衰减
    """

    DB_PATH = r"C:\Users\35955\.claude\memory\vmem.db"
    EMBED_DIM = 512

    def __init__(self, db_path: Optional[str] = None, embedding_engine: Optional[EmbeddingEngine] = None):
        self._db_path = db_path or self.DB_PATH
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._engine = embedding_engine or EmbeddingEngine()
        self._conn: Optional[sqlite3.Connection] = None

        self._init_db()

    # ------------------------------------------------------------------
    # 数据库初始化
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_vectors (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                source      TEXT DEFAULT '',
                importance  TEXT DEFAULT '',
                tags        TEXT DEFAULT '',
                related     TEXT DEFAULT '[]',
                embedding   BLOB,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                key, value, source, tags,
                tokenize='trigram'
            );
        """)
        conn.commit()

        # Run schema migrations (D-09, D-10)
        from vmem.migrations import run_migrations
        run_migrations(conn)

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def store(
        self,
        key: str,
        value: str,
        source: str = "",
        level: str = "",
        tags: str = "",
        related: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """
        INSERT OR REPLACE 一条记忆。
        如果未提供 embedding，自动用 EmbeddingEngine 编码 value。
        注意：level 参数名保持不变（向后兼容），写入 importance 列。
        """
        try:
            now = int(time.time())

            if embedding is None:
                embedding = self._engine.encode([value])[0]
            embedding = embedding.astype(np.float32).tobytes()

            related_json = json.dumps(related or [], ensure_ascii=False)

            # Convert tags to JSON array (D-07)
            if isinstance(tags, str) and tags:
                tags_json = json.dumps([t.strip() for t in tags.split(",") if t.strip()], ensure_ascii=False)
            elif isinstance(tags, list):
                tags_json = json.dumps(tags, ensure_ascii=False)
            else:
                tags_json = "[]"

            conn = self._get_conn()
            # 查旧记录的 created_at（D-05: access_count 已移除）
            row = conn.execute(
                "SELECT created_at FROM memory_vectors WHERE key = ?",
                (key,),
            ).fetchone()
            created_at = row[0] if row else now

            conn.execute(
                """
                INSERT OR REPLACE INTO memory_vectors
                    (key, value, source, importance, tags, related, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key, value, source, level, tags_json, related_json, embedding, created_at, now),
            )

            # 同步 FTS（FTS5 用自己的列名，不受 rename 影响）
            conn.execute("DELETE FROM memory_fts WHERE key = ?", (key,))
            conn.execute(
                "INSERT INTO memory_fts (key, value, source, tags) VALUES (?, ?, ?, ?)",
                (key, value, source, tags if isinstance(tags, str) else ",".join(tags) if isinstance(tags, list) else ""),
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"[vmem] store error: {e}")
            return False

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        向量余弦相似度搜索。
        query_emb: shape (512,) 或 (1, 512) 的 float32 向量（已归一化）。
        返回 top_k 条结果，按相似度降序。
        """
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT key, value, source, importance, tags, related, embedding, created_at, updated_at "
                "FROM memory_vectors WHERE embedding IS NOT NULL"
            ).fetchall()

            if not rows:
                return []

            query_emb = np.asarray(query_emb, dtype=np.float32).flatten()

            results = []
            for row in rows:
                db_emb = np.frombuffer(row[6], dtype=np.float32)
                if db_emb.shape[0] != self.EMBED_DIM:
                    continue
                score = float(np.dot(query_emb, db_emb))
                results.append({
                    "key": row[0],
                    "value": row[1],
                    "source": row[2],
                    "importance": row[3],
                    "tags": row[4],
                    "related": json.loads(row[5]) if row[5] else [],
                    "created_at": row[7],
                    "updated_at": row[8],
                    "score": score,
                })

            results.sort(key=lambda x: x["score"], reverse=True)

            return results[:top_k]

        except Exception as e:
            print(f"[vmem] search error: {e}")
            return []

    def search_fts(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        FTS5 全文搜索，按 rank 排序（rank 值越小越相关，这里取负值方便与其他分数融合）。
        """
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT m.key, m.value, m.source, m.importance, m.tags, m.related,
                       m.created_at, m.updated_at,
                       fts.rank
                FROM memory_fts AS fts
                JOIN memory_vectors AS m ON m.key = fts.key
                WHERE memory_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (query, top_k),
            ).fetchall()

            results = []
            for row in rows:
                results.append({
                    "key": row[0],
                    "value": row[1],
                    "source": row[2],
                    "importance": row[3],
                    "tags": row[4],
                    "related": json.loads(row[5]) if row[5] else [],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "score": -row[8],  # rank 越小越好，转为正分
                })

            return results

        except Exception as e:
            print(f"[vmem] search_fts error: {e}")
            return []

    def search_fusion(
        self,
        query: str,
        query_emb: np.ndarray,
        top_k: int = 10,
        w_vec: float = 0.60,
        w_fts: float = 0.25,
        w_time: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        融合搜索：w_vec*向量 + w_fts*全文 + w_time*时间衰减。
        各分数归一化到 [0, 1] 后加权求和。
        """
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT key, value, source, importance, tags, related, embedding,
                       created_at, updated_at
                FROM memory_vectors WHERE embedding IS NOT NULL
                """
            ).fetchall()

            if not rows:
                return []

            query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
            now = int(time.time())
            half_life = 7 * 24 * 3600  # 7 天半衰期

            # --- 向量得分 ---
            vec_scores = {}
            for row in rows:
                db_emb = np.frombuffer(row[6], dtype=np.float32)
                if db_emb.shape[0] != self.EMBED_DIM:
                    continue
                vec_scores[row[0]] = float(np.dot(query_emb, db_emb))

            # --- FTS 得分（从候选集中做 trigram 匹配）---
            fts_raw: Dict[str, float] = {}
            try:
                fts_rows = conn.execute(
                    """
                    SELECT key, rank FROM memory_fts
                    WHERE memory_fts MATCH ?
                    ORDER BY rank LIMIT 200
                    """,
                    (query,),
                ).fetchall()
                for key, rank in fts_rows:
                    fts_raw[key] = -rank  # rank 越小越好
            except Exception:
                pass  # trigram 可能匹配失败，忽略

            # --- 时间衰减得分 ---
            time_scores: Dict[str, float] = {}
            for row in rows:
                updated_at = row[8] or row[7]
                elapsed = max(now - updated_at, 0)
                time_scores[row[0]] = 2 ** (-(elapsed / half_life))

            # --- 归一化函数 ---
            def _normalize(d: Dict[str, float]) -> Dict[str, float]:
                if not d:
                    return {}
                vals = list(d.values())
                mn, mx = min(vals), max(vals)
                if mx - mn < 1e-9:
                    return {k: 0.5 for k in d}
                return {k: (v - mn) / (mx - mn) for k, v in d.items()}

            vec_norm = _normalize(vec_scores)
            fts_norm = _normalize(fts_raw)
            time_norm = _normalize(time_scores)

            # --- 融合 ---
            all_keys = set()
            for d in (vec_norm, fts_norm, time_norm):
                all_keys.update(d.keys())

            fused = []
            for key in all_keys:
                s = (
                    w_vec * vec_norm.get(key, 0.0)
                    + w_fts * fts_norm.get(key, 0.0)
                    + w_time * time_norm.get(key, 0.0)
                )
                fused.append((key, s))

            fused.sort(key=lambda x: x[1], reverse=True)
            top_keys = [k for k, _ in fused[:top_k]]

            # 取回完整记录
            placeholders = ",".join("?" for _ in top_keys)
            full_rows = conn.execute(
                f"""
                SELECT key, value, source, importance, tags, related,
                       created_at, updated_at
                FROM memory_vectors WHERE key IN ({placeholders})
                """,
                top_keys,
            ).fetchall()

            row_map = {r[0]: r for r in full_rows}
            score_map = dict(fused)

            results = []
            for key in top_keys:
                r = row_map.get(key)
                if r is None:
                    continue
                results.append({
                    "key": r[0],
                    "value": r[1],
                    "source": r[2],
                    "importance": r[3],
                    "tags": r[4],
                    "related": json.loads(r[5]) if r[5] else [],
                    "created_at": r[6],
                    "updated_at": r[7],
                    "score": score_map.get(key, 0.0),
                })

            return results

        except Exception as e:
            print(f"[vmem] search_fusion error: {e}")
            return []

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """精确获取一条记忆。"""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT key, value, source, importance, tags, related, created_at, updated_at "
                "FROM memory_vectors WHERE key = ?",
                (key,),
            ).fetchone()

            if row is None:
                return None

            return {
                "key": row[0],
                "value": row[1],
                "source": row[2],
                "importance": row[3],
                "tags": row[4],
                "related": json.loads(row[5]) if row[5] else [],
                "created_at": row[6],
                "updated_at": row[7],
            }

        except Exception as e:
            print(f"[vmem] get error: {e}")
            return None

    def delete(self, key: str) -> bool:
        """删除一条记忆（同时删除 FTS 索引）。"""
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM memory_vectors WHERE key = ?", (key,))
            conn.execute("DELETE FROM memory_fts WHERE key = ?", (key,))
            conn.commit()
            return True
        except Exception as e:
            print(f"[vmem] delete error: {e}")
            return False

    def list_all(self, limit: int = 100, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出记忆，按更新时间降序。"""
        try:
            conn = self._get_conn()
            if source_filter:
                rows = conn.execute(
                    "SELECT key, value, source, importance, tags, related, created_at, updated_at "
                    "FROM memory_vectors WHERE source = ? ORDER BY updated_at DESC LIMIT ?",
                    (source_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, value, source, importance, tags, related, created_at, updated_at "
                    "FROM memory_vectors ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            return [
                {
                    "key": r[0],
                    "value": r[1],
                    "source": r[2],
                    "importance": r[3],
                    "tags": r[4],
                    "related": json.loads(r[5]) if r[5] else [],
                    "created_at": r[6],
                    "updated_at": r[7],
                }
                for r in rows
            ]

        except Exception as e:
            print(f"[vmem] list_all error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """返回存储统计信息。"""
        try:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
            sources = conn.execute(
                "SELECT source, COUNT(*) FROM memory_vectors GROUP BY source ORDER BY COUNT(*) DESC"
            ).fetchall()
            importance = conn.execute(
                "SELECT importance, COUNT(*) FROM memory_vectors GROUP BY importance ORDER BY COUNT(*) DESC"
            ).fetchall()

            oldest = conn.execute("SELECT MIN(created_at) FROM memory_vectors").fetchone()[0]
            newest = conn.execute("SELECT MAX(updated_at) FROM memory_vectors").fetchone()[0]

            return {
                "total": total,
                "sources": {s[0] or "(empty)": s[1] for s in sources},
                "importance": {l[0] or "(empty)": l[1] for l in importance},
                "oldest_created_at": oldest,
                "newest_updated_at": newest,
                "db_path": self._db_path,
            }

        except Exception as e:
            print(f"[vmem] get_stats error: {e}")
            return {"total": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# 便捷函数：供 MCP server 调用
# ---------------------------------------------------------------------------

def search_fusion(
    store: MemoryVectorStore,
    encoder: EmbeddingEngine,
    query: str,
    top_k: int = 10,
    source_filter: Optional[str] = None,
    level_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    融合搜索的便捷包装函数。
    自动用 encoder 编码 query，调用 store.search_fusion()，
    再根据 source/level/tags 过滤。
    注意：level_filter 参数名保持不变（向后兼容），过滤 importance 字段。
    """
    query_emb = encoder.encode([query])[0]
    results = store.search_fusion(query=query, query_emb=query_emb, top_k=top_k * 3)

    # 后置过滤
    filtered = []
    for r in results:
        if source_filter and r.get("source", "") != source_filter:
            continue
        if level_filter and r.get("importance", "") != level_filter:
            continue
        if tags_filter:
            entry_tags = r.get("tags", "")
            # Tags are now JSON arrays (D-07), with fallback for legacy comma-separated
            if isinstance(entry_tags, str):
                try:
                    entry_tags = json.loads(entry_tags)
                except (json.JSONDecodeError, TypeError):
                    entry_tags = [t.strip() for t in entry_tags.split(",") if t.strip()]
            entry_tags_set = set(entry_tags) if isinstance(entry_tags, list) else set()
            if not any(t in entry_tags_set for t in tags_filter):
                continue
        filtered.append(r)
        if len(filtered) >= top_k:
            break

    return filtered
