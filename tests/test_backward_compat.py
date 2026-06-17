"""Integration tests for backward compatibility (DATA-01, DATA-10).

These tests verify:
- All 25+ existing records survive migration
- Search (vector, FTS, fusion) works after column changes
- Store+retrieve cycle works with new schema
- FTS5 index is functional after migration
- WAL mode is active
"""
import json
import sqlite3
import pytest
import numpy as np


def test_existing_data_preserved(migrated_store):
    """DATA-10: 25+ existing records survive migration without data loss."""
    conn = migrated_store._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
    assert count >= 25, f"Expected >= 25 records, got {count}"

    # Verify specific known records exist
    known_keys = [
        "wechat-scan", "kb-core", "claude-code-path-format",
        "evo:test_key", "evo:statusline-config",
    ]
    for key in known_keys:
        row = conn.execute("SELECT key FROM memory_vectors WHERE key = ?", (key,)).fetchone()
        assert row is not None, f"Known record {key} missing after migration"


def test_importance_values_preserved(migrated_store):
    """D-06: level values preserved as importance after rename."""
    conn = migrated_store._get_conn()
    rows = conn.execute("SELECT key, importance FROM memory_vectors WHERE importance != ''").fetchall()
    assert len(rows) > 0, "No importance values found (level rename may have failed)"

    # Known level values: "long", "working", "permanent"
    values = {r[1] for r in rows}
    assert "long" in values, f"Expected 'long' in importance values, got {values}"


def test_tags_valid_json(migrated_store):
    """D-07: All tags are valid JSON arrays after migration."""
    conn = migrated_store._get_conn()
    rows = conn.execute("SELECT key, tags FROM memory_vectors WHERE tags != '' AND tags IS NOT NULL").fetchall()
    assert len(rows) > 0, "No tags found"
    for key, tags_str in rows:
        parsed = json.loads(tags_str)
        assert isinstance(parsed, list), f"Tags for {key} should be list, got {type(parsed)}"


def test_vmem_store_after_migration(migrated_store):
    """vmem_store works: write a record and retrieve it."""
    ok = migrated_store.store(
        key="test:phase1:integration",
        value="This is a Phase 1 integration test record",
        source="test",
        level="normal",
        tags="test,integration",
    )
    assert ok is True, "store() returned False"

    record = migrated_store.get("test:phase1:integration")
    assert record is not None, "get() returned None for just-stored record"
    assert record["value"] == "This is a Phase 1 integration test record"
    assert record["importance"] == "normal"
    assert json.loads(record["tags"]) == ["test", "integration"]

    # Cleanup
    migrated_store.delete("test:phase1:integration")


def test_vmem_search_after_migration(migrated_store):
    """vmem_search returns results after migration (vector search)."""
    # Encode a query similar to an existing record
    from vmem.store import EmbeddingEngine
    engine = EmbeddingEngine()
    query_emb = engine.encode(["微信聊天数据知识库"])[0]

    results = migrated_store.search(query_emb, top_k=5)
    assert len(results) > 0, "search() returned 0 results after migration"
    assert "importance" in results[0], f"Result missing 'importance' key, got keys: {list(results[0].keys())}"
    assert "access_count" not in results[0], "access_count should not be in search results"


def test_fts_search_after_migration(migrated_store):
    """FTS5 search returns results after migration."""
    results = migrated_store.search_fts("project", top_k=5)
    assert len(results) > 0, "search_fts() returned 0 results for 'project'"
    assert "importance" in results[0]


def test_fusion_search_after_migration(migrated_store):
    """Fusion search returns results after migration."""
    from vmem.store import EmbeddingEngine, search_fusion
    engine = EmbeddingEngine()
    results = search_fusion(
        store=migrated_store,
        encoder=engine,
        query="错误 教训",
        top_k=5,
    )
    assert len(results) > 0, "search_fusion() returned 0 results"
    assert "importance" in results[0]


def test_wal_mode_active(migrated_store):
    """WAL mode active on the database."""
    conn = migrated_store._get_conn()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"


def test_store_with_json_tags(migrated_store):
    """Store accepts tags as comma-separated string, stores as JSON."""
    ok = migrated_store.store(
        key="test:json:tags",
        value="testing json tag storage",
        tags="alpha,beta,gamma",
    )
    assert ok is True

    conn = migrated_store._get_conn()
    row = conn.execute("SELECT tags FROM memory_vectors WHERE key = 'test:json:tags'").fetchone()
    assert row is not None
    parsed = json.loads(row[0])
    assert parsed == ["alpha", "beta", "gamma"], f"Expected ['alpha','beta','gamma'], got {parsed}"

    # Cleanup
    migrated_store.delete("test:json:tags")
