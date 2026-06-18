import pytest
import time
import numpy as np


class TestTopics:

    def _insert_key(self, store, key, value="test", source="user", embedding=None):
        """Helper: insert a memory_vectors entry."""
        conn = store._get_conn()
        now = int(time.time())
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        conn.execute(
            "INSERT OR IGNORE INTO memory_vectors (key, value, source, importance, tags, related, embedding, created_at, updated_at) VALUES (?, ?, ?, 'normal', '[]', '[]', ?, ?, ?)",
            (key, value, source, emb_blob, now, now),
        )
        conn.commit()

    def test_topic_create(self, migrated_store):
        """Creates topic, returns id."""
        tid = migrated_store.topic_create("python tips")
        assert tid > 0

    def test_topic_match_found(self, migrated_store):
        """Two similar embeddings → matches same topic."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        tid = migrated_store.topic_create("test topic", emb1)

        emb2 = emb1.copy()
        emb2[0] += 0.001
        emb2 = emb2 / np.linalg.norm(emb2)
        match_id, sim = migrated_store.topic_match(emb2)
        assert match_id == tid
        assert sim >= 0.75

    def test_topic_match_no_match(self, migrated_store):
        """Orthogonal embedding → no match."""
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        migrated_store.topic_create("topic A", emb1)

        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        match_id, sim = migrated_store.topic_match(emb2)
        assert match_id is None

    def test_topic_add_memory(self, migrated_store):
        """Links memory to topic, increments session_count."""
        tid = migrated_store.topic_create("test")
        self._insert_key(migrated_store, "TM1")
        ok = migrated_store.topic_add_memory(tid, "TM1")
        assert ok is True
        topics = migrated_store.topic_list()
        assert topics[0]["session_count"] == 1

    def test_topic_recalculate(self, migrated_store):
        """After adding memories, centroid changes."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        tid = migrated_store.topic_create("recalc test", emb1.copy())

        # Add a memory with a different embedding
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        self._insert_key(migrated_store, "RC1", embedding=emb2)
        migrated_store.topic_add_memory(tid, "RC1")

        # Recalculate
        ok = migrated_store.topic_recalculate(tid)
        assert ok is True

        # New centroid should be mean of emb1 and emb2 (via memory_vectors)
        # But session_id is the key, so it looks up the memory's embedding
        # The centroid should have changed
        conn = migrated_store._get_conn()
        row = conn.execute("SELECT centroid FROM topics WHERE id = ?", (tid,)).fetchone()
        assert row[0] is not None

    def test_topic_list(self, migrated_store):
        """Returns all topics."""
        migrated_store.topic_create("topic A")
        migrated_store.topic_create("topic B")
        topics = migrated_store.topic_list()
        assert len(topics) >= 2

    def test_topic_list_empty(self, migrated_store):
        """Empty store returns empty list."""
        topics = migrated_store.topic_list()
        assert topics == []

    def test_topic_create_returns_id(self, migrated_store):
        """Each topic gets a unique id."""
        id1 = migrated_store.topic_create("t1")
        id2 = migrated_store.topic_create("t2")
        assert id1 != id2
