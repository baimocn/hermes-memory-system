import pytest
import time


class TestConfidence:

    def _insert_key(self, store, key, value="test value", source="test"):
        """Helper: insert a memory_vectors entry."""
        conn = store._get_conn()
        now = int(time.time())
        conn.execute(
            "INSERT OR IGNORE INTO memory_vectors (key, value, source, importance, tags, related, embedding, created_at, updated_at) VALUES (?, ?, ?, 'normal', '[]', '[]', NULL, ?, ?)",
            (key, value, source, now, now),
        )
        conn.commit()

    def test_initial_confidence_user(self, migrated_store):
        """store with source='user' → confidence ≈ 0.6."""
        migrated_store.store(key="CUSER", value="test", source="user", level="normal")
        conf = migrated_store.confidence_get("CUSER")
        assert conf == 0.6

    def test_initial_confidence_lesson(self, migrated_store):
        """store with source='lesson' → confidence ≈ 0.7."""
        migrated_store.store(key="CLESS", value="test", source="lesson", level="normal")
        conf = migrated_store.confidence_get("CLESS")
        assert conf == 0.7

    def test_initial_confidence_auto(self, migrated_store):
        """store with source='auto' → confidence ≈ 0.4."""
        migrated_store.store(key="CAUTO", value="test", source="auto", level="normal")
        conf = migrated_store.confidence_get("CAUTO")
        assert conf == 0.4

    def test_initial_confidence_default(self, migrated_store):
        """Unknown source → confidence 0.5."""
        migrated_store.store(key="CDEF", value="test", source="unknown_source", level="normal")
        conf = migrated_store.confidence_get("CDEF")
        assert conf == 0.5

    def test_confidence_update_positive(self, migrated_store):
        """confidence_update with +0.15 increases confidence."""
        self._insert_key(migrated_store, "CUP", source="user")
        migrated_store.store(key="CUP", value="test", source="user", level="normal")
        old_conf = migrated_store.confidence_get("CUP")
        ok = migrated_store.confidence_update("CUP", 0.15, "confirm")
        assert ok is True
        new_conf = migrated_store.confidence_get("CUP")
        assert new_conf == pytest.approx(old_conf + 0.15, abs=0.01)

    def test_confidence_update_negative(self, migrated_store):
        """confidence_update with -0.20 decreases confidence."""
        migrated_store.store(key="CDN", value="test", source="user", level="normal")
        old_conf = migrated_store.confidence_get("CDN")
        ok = migrated_store.confidence_update("CDN", -0.20, "contradict")
        assert ok is True
        new_conf = migrated_store.confidence_get("CDN")
        assert new_conf == pytest.approx(old_conf - 0.20, abs=0.01)

    def test_confidence_clamp_0(self, migrated_store):
        """Update below 0 → clamped to 0.0."""
        migrated_store.store(key="CL0", value="test", source="auto", level="normal")
        migrated_store.confidence_update("CL0", -1.0, "test_clamp")
        conf = migrated_store.confidence_get("CL0")
        assert conf == 0.0

    def test_confidence_clamp_1(self, migrated_store):
        """Update above 1 → clamped to 1.0."""
        migrated_store.store(key="CL1", value="test", source="user", level="normal")
        migrated_store.confidence_update("CL1", 1.0, "test_clamp")
        conf = migrated_store.confidence_get("CL1")
        assert conf == 1.0

    def test_confidence_log_recorded(self, migrated_store):
        """After update, confidence_log has entry with correct delta and reason."""
        migrated_store.store(key="CLOG", value="test", source="user", level="normal")
        migrated_store.confidence_update("CLOG", 0.15, "evidence_found")
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT delta, reason FROM confidence_log WHERE memory_key = 'CLOG'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.15, abs=0.01)
        assert row[1] == "evidence_found"

    def test_confidence_get_default(self, migrated_store):
        """confidence_get for nonexistent key returns 0.5."""
        conf = migrated_store.confidence_get("nonexistent_key_xyz")
        assert conf == 0.5

    def test_store_sets_initial_confidence(self, migrated_store):
        """store() with source='user' sets initial confidence to 0.6."""
        migrated_store.store(key="CSIC", value="test", source="user", level="normal")
        conf = migrated_store.confidence_get("CSIC")
        assert conf == 0.6

    def test_confidence_update_preserves_on_restore(self, migrated_store):
        """Updating a memory doesn't reset its confidence."""
        migrated_store.store(key="CPRS", value="v1", source="user", level="normal")
        migrated_store.confidence_update("CPRS", 0.20, "confirmed")
        conf_after_update = migrated_store.confidence_get("CPRS")
        # Re-store same key
        migrated_store.store(key="CPRS", value="v2", source="user", level="normal")
        conf_after_rerstore = migrated_store.confidence_get("CPRS")
        # Confidence should NOT reset to 0.6 (initial) because _set_initial_confidence
        # only sets when confidence = 0.5 or IS NULL
        assert conf_after_rerstore == conf_after_update
