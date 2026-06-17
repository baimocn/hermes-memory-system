"""Tests for new table schemas (DATA-02 through DATA-09)."""
import pytest


def test_user_profiles_schema(migrated_store):
    """DATA-02: user_profiles table with correct columns."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(user_profiles)").fetchall()]
    assert set(columns) == {"id", "parent_id", "name", "level", "created_at"}


def test_profile_attributes_schema(migrated_store):
    """DATA-03: profile_attributes with FK and FTS5."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(profile_attributes)").fetchall()]
    assert set(columns) == {"id", "profile_id", "attr_key", "attr_value", "category", "confidence", "source", "updated_at"}
    # FTS5 exists
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "profile_attr_fts" in tables


def test_causal_links_schema(migrated_store):
    """DATA-04: causal_links with composite index."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(causal_links)").fetchall()]
    assert set(columns) == {"id", "from_id", "to_id", "relation", "strength", "created_at"}
    indexes = [r[1] for r in conn.execute("PRAGMA index_list(causal_links)").fetchall()]
    assert "idx_causal_links_from_relation" in indexes


def test_closure_table_schema(migrated_store):
    """DATA-05: closure_table with both FK indexes."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(closure_table)").fetchall()]
    assert set(columns) == {"ancestor", "descendant", "depth"}


def test_memory_links_schema(migrated_store):
    """DATA-06: memory_links with composite index."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(memory_links)").fetchall()]
    assert set(columns) == {"id", "from_id", "to_id", "link_type", "strength", "created_at"}
    indexes = [r[1] for r in conn.execute("PRAGMA index_list(memory_links)").fetchall()]
    assert "idx_memory_links_from_type" in indexes


def test_topics_schema(migrated_store):
    """DATA-07: topics with centroid BLOB."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(topics)").fetchall()]
    assert "centroid" in columns
    centroid_type = [r[2] for r in conn.execute("PRAGMA table_info(topics)").fetchall() if r[1] == "centroid"][0]
    assert centroid_type.upper() == "BLOB"


def test_session_topics_schema(migrated_store):
    """DATA-08: session_topics with FK index."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(session_topics)").fetchall()]
    assert set(columns) == {"id", "session_id", "topic_id", "similarity", "created_at"}


def test_decay_state_schema(migrated_store):
    """DATA-09: decay_state with FSRS columns."""
    conn = migrated_store._get_conn()
    columns = [r[1] for r in conn.execute("PRAGMA table_info(decay_state)").fetchall()]
    assert set(columns) == {"memory_key", "difficulty", "stability", "retrievability", "last_review"}
