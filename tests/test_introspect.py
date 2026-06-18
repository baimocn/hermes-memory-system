import pytest
import time
import importlib
import sys
import os

# Add hooks directory
hooks_dir = os.path.join(os.path.dirname(__file__), '..', '.claude', 'hooks')
sys.path.insert(0, hooks_dir)

# Import session hook
for mod_name in list(sys.modules.keys()):
    if 'hermes-introspect' in mod_name:
        del sys.modules[mod_name]
_introspect_hook = importlib.import_module("hermes-introspect-hook")
_build_introspect_output = _introspect_hook._build_introspect_output


class TestIntrospect:

    def _insert_key(self, store, key, value="test", source="user", confidence=0.5, importance="normal", updated_ago_days=0):
        """Helper: insert a memory_vectors entry."""
        conn = store._get_conn()
        now = int(time.time())
        updated_at = now - updated_ago_days * 86400
        conn.execute(
            """INSERT OR IGNORE INTO memory_vectors
               (key, value, source, importance, tags, related, embedding, created_at, updated_at, confidence)
               VALUES (?, ?, ?, ?, '[]', '[]', NULL, ?, ?, ?)""",
            (key, value, source, importance, now, updated_at, confidence),
        )
        conn.commit()

    def test_introspect_gc_marks_archived(self, migrated_store):
        """Low confidence + old → marked as archived."""
        self._insert_key(migrated_store, "GC1", confidence=0.1, updated_ago_days=31)
        archived = migrated_store.introspect_gc()
        assert "GC1" in archived
        # Verify marked
        conn = migrated_store._get_conn()
        row = conn.execute("SELECT importance FROM memory_vectors WHERE key = 'GC1'").fetchone()
        assert row[0] == "archived"

    def test_introspect_gc_skips_recent(self, migrated_store):
        """Low confidence but recent → NOT archived."""
        self._insert_key(migrated_store, "GC2", confidence=0.1, updated_ago_days=5)
        archived = migrated_store.introspect_gc()
        assert "GC2" not in archived

    def test_introspect_gc_skips_high_conf(self, migrated_store):
        """High confidence + old → NOT archived."""
        self._insert_key(migrated_store, "GC3", confidence=0.8, updated_ago_days=31)
        archived = migrated_store.introspect_gc()
        assert "GC3" not in archived

    def test_introspect_gc_skips_already_archived(self, migrated_store):
        """Already archived memories not returned again."""
        self._insert_key(migrated_store, "GC4", confidence=0.1, updated_ago_days=31, importance="archived")
        archived = migrated_store.introspect_gc()
        assert "GC4" not in archived

    def test_introspect_contradictions(self, migrated_store):
        """Contradiction links are returned."""
        self._insert_key(migrated_store, "CTR1", value="python is best")
        self._insert_key(migrated_store, "CTR2", value="rust is best")
        migrated_store.link_add("CTR1", "CTR2", "contradicts", 0.9)
        contradictions = migrated_store.introspect_contradictions()
        assert len(contradictions) >= 1
        assert contradictions[0]["from_key"] in ("CTR1", "CTR2")

    def test_introspect_contradictions_empty(self, migrated_store):
        """No contradictions → empty list."""
        contradictions = migrated_store.introspect_contradictions()
        assert isinstance(contradictions, list)

    def test_introspect_health_keys(self, migrated_store):
        """Health dict has all expected keys."""
        health = migrated_store.introspect_health()
        expected_keys = {
            "total_memories", "total_topics", "avg_confidence",
            "gc_candidates", "contradictions", "causal_chains",
            "profile_attributes", "archived",
        }
        assert expected_keys.issubset(set(health.keys()))

    def test_introspect_health_counts(self, migrated_store):
        """Health counts are accurate."""
        self._insert_key(migrated_store, "HC1", confidence=0.6)
        self._insert_key(migrated_store, "HC2", confidence=0.8)
        health = migrated_store.introspect_health()
        assert health["total_memories"] >= 2
        assert health["avg_confidence"] > 0

    def test_session_hook_output_format(self, tmp_vmem_db):
        """SessionEnd hook output has correct structure."""
        output = _build_introspect_output(tmp_vmem_db)
        assert "hookSpecificOutput" in output
        ho = output["hookSpecificOutput"]
        assert ho["hookEventName"] == "SessionEnd"
        assert isinstance(ho["additionalContext"], str)
        assert "Memory Health" in ho["additionalContext"]
