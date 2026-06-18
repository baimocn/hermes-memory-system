import pytest
import time


class TestFSDSDecay:

    def _insert_key(self, store, key, value="test", source="user", importance="normal"):
        """Helper: insert a memory_vectors entry."""
        conn = store._get_conn()
        now = int(time.time())
        conn.execute(
            "INSERT OR IGNORE INTO memory_vectors (key, value, source, importance, tags, related, embedding, created_at, updated_at) VALUES (?, ?, ?, ?, '[]', '[]', NULL, ?, ?)",
            (key, value, source, importance, now, now),
        )
        conn.commit()

    def test_base_stability_error(self, migrated_store):
        """Error/lesson memories have 90-day base stability."""
        s = migrated_store._base_stability("auto-lesson", "normal")
        assert s == 90.0

    def test_base_stability_preference(self, migrated_store):
        """Preference memories have 180-day base stability."""
        s = migrated_store._base_stability("auto-preference", "normal")
        assert s == 180.0

    def test_base_stability_session(self, migrated_store):
        """Session memories have 7-day base stability."""
        s = migrated_store._base_stability("session", "session")
        assert s == 7.0

    def test_base_stability_default(self, migrated_store):
        """Unknown type has 30-day base stability."""
        s = migrated_store._base_stability("user", "normal")
        assert s == 30.0

    def test_decay_init_creates_entry(self, migrated_store):
        """decay_init creates a decay_state entry."""
        self._insert_key(migrated_store, "DI1", source="user")
        ok = migrated_store.decay_init("DI1", "user")
        assert ok is True
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT stability, retrievability FROM decay_state WHERE memory_key = 'DI1'"
        ).fetchone()
        assert row is not None
        assert row[0] == 30.0  # user → 30 days
        assert row[1] == 1.0   # initial retrievability

    def test_decay_init_idempotent(self, migrated_store):
        """Calling decay_init twice doesn't error."""
        self._insert_key(migrated_store, "DI2", source="user")
        migrated_store.decay_init("DI2")
        migrated_store.decay_init("DI2")  # should not error
        conn = migrated_store._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM decay_state WHERE memory_key = 'DI2'"
        ).fetchone()[0]
        assert count == 1

    def test_decay_recall_increases_stability(self, migrated_store):
        """Recall increases stability by 10%."""
        self._insert_key(migrated_store, "DR1", source="user")
        migrated_store.decay_init("DR1")
        conn = migrated_store._get_conn()
        old_s = conn.execute(
            "SELECT stability FROM decay_state WHERE memory_key = 'DR1'"
        ).fetchone()[0]
        ok = migrated_store.decay_recall("DR1")
        assert ok is True
        new_s = conn.execute(
            "SELECT stability FROM decay_state WHERE memory_key = 'DR1'"
        ).fetchone()[0]
        assert new_s == pytest.approx(old_s * 1.10, rel=0.01)

    def test_decay_recall_compounds(self, migrated_store):
        """Multiple recalls compound stability increase."""
        self._insert_key(migrated_store, "DR2", source="user")
        migrated_store.decay_init("DR2")
        for _ in range(5):
            migrated_store.decay_recall("DR2")
        conn = migrated_store._get_conn()
        s = conn.execute(
            "SELECT stability FROM decay_state WHERE memory_key = 'DR2'"
        ).fetchone()[0]
        # 30 * 1.10^5 ≈ 48.3
        assert s == pytest.approx(30.0 * (1.10 ** 5), rel=0.01)

    def test_decay_recall_auto_inits(self, migrated_store):
        """decay_recall auto-initializes if no decay_state exists."""
        self._insert_key(migrated_store, "DR3", source="user")
        ok = migrated_store.decay_recall("DR3")
        assert ok is True
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT stability FROM decay_state WHERE memory_key = 'DR3'"
        ).fetchone()
        assert row is not None
        # Should be 30 * 1.10 (init then recall)
        assert row[0] == pytest.approx(30.0 * 1.10, rel=0.01)

    def test_decay_retrievability_initial(self, migrated_store):
        """Fresh memory has retrievability ≈ 1.0."""
        self._insert_key(migrated_store, "DR4", source="user")
        migrated_store.decay_init("DR4")
        r = migrated_store.decay_retrievability("DR4")
        assert r == pytest.approx(1.0, abs=0.01)

    def test_decay_retrievability_decays(self, migrated_store):
        """Retrievability decreases over time."""
        self._insert_key(migrated_store, "DR5", source="user")
        conn = migrated_store._get_conn()
        # Set last_review to 30 days ago
        now = int(time.time())
        conn.execute(
            """INSERT INTO decay_state (memory_key, difficulty, stability, retrievability, last_review)
               VALUES ('DR5', 0.0, 30.0, 1.0, ?)""",
            (now - 30 * 86400,),
        )
        conn.commit()
        r = migrated_store.decay_retrievability("DR5")
        # R = (1 + 30/(9*30))^(-1) = (1 + 1/9)^(-1) ≈ 0.9
        assert r == pytest.approx(0.9, abs=0.05)
        assert r < 1.0

    def test_decay_retrievability_no_entry(self, migrated_store):
        """No decay_state entry returns 1.0."""
        r = migrated_store.decay_retrievability("nonexistent_key")
        assert r == 1.0

    def test_decay_does_not_delete(self, migrated_store):
        """Decay never deletes memory entries."""
        self._insert_key(migrated_store, "DND", source="user")
        migrated_store.decay_init("DND")
        # Even with very old last_review, the memory should still exist
        conn = migrated_store._get_conn()
        conn.execute(
            "UPDATE decay_state SET last_review = 0 WHERE memory_key = 'DND'"
        )
        conn.commit()
        r = migrated_store.decay_retrievability("DND")
        assert r > 0  # never zero
        # Memory still exists
        row = conn.execute(
            "SELECT key FROM memory_vectors WHERE key = 'DND'"
        ).fetchone()
        assert row is not None

    def test_store_initializes_decay(self, migrated_store):
        """store() automatically initializes decay state."""
        migrated_store.store(key="SID", value="test", source="user", level="normal")
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT stability FROM decay_state WHERE memory_key = 'SID'"
        ).fetchone()
        assert row is not None
        assert row[0] == 30.0  # user → 30 days
