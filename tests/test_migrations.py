"""Tests for schema migrations (DATA-05, DATA-10 column cleanup)."""
import json
import sqlite3
import pytest


def test_column_rename_level_to_importance(migrated_store):
    """D-06: level column renamed to importance."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(memory_vectors)").fetchall()]
    assert "importance" in columns, f"importance column missing, got: {columns}"
    assert "level" not in columns, f"level column should not exist, got: {columns}"


def test_access_count_dropped(migrated_store):
    """D-05: access_count column dropped."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(memory_vectors)").fetchall()]
    assert "access_count" not in columns, f"access_count should be dropped, got: {columns}"


def test_tags_json_array(migrated_store):
    """D-07: tags converted from comma-separated to JSON array."""
    conn = migrated_store._get_conn()
    rows = conn.execute("SELECT key, tags FROM memory_vectors WHERE tags != '' AND tags IS NOT NULL").fetchall()
    for key, tags_str in rows:
        parsed = json.loads(tags_str)
        assert isinstance(parsed, list), f"tags for {key} should be JSON array, got: {type(parsed)}"


def test_schema_migrations_tracked(migrated_store):
    """All 13 migrations recorded in schema_migrations."""
    conn = migrated_store._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert count == 13, f"Expected 13 migrations, got {count}"
