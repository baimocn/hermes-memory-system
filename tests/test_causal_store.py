import pytest


class TestCausalStore:

    def _insert_key(self, store, key, value="test value"):
        """Helper: insert a memory_vectors entry so FK constraints pass."""
        conn = store._get_conn()
        import time
        now = int(time.time())
        conn.execute(
            "INSERT OR IGNORE INTO memory_vectors (key, value, source, importance, tags, related, embedding, created_at, updated_at) VALUES (?, ?, 'test', 'normal', '[]', '[]', NULL, ?, ?)",
            (key, value, now, now),
        )
        conn.commit()

    def test_causal_add_basic(self, migrated_store):
        """Add A→B with 'caused_by'; verify in DB."""
        self._insert_key(migrated_store, "A", "problem")
        self._insert_key(migrated_store, "B", "cause")
        ok = migrated_store.causal_add("A", "B", "caused_by")
        assert ok is True
        links = migrated_store.causal_get("A")
        assert len(links) == 1
        assert links[0]["from_id"] == "A"
        assert links[0]["to_id"] == "B"
        assert links[0]["relation"] == "caused_by"

    def test_causal_add_all_relations(self, migrated_store):
        """All 5 relation types are accepted."""
        keys = ["K1", "K2", "K3", "K4", "K5", "K6"]
        for k in keys:
            self._insert_key(migrated_store, k, f"value_{k}")
        relations = ["caused_by", "led_to", "resolved", "contradicts", "supersedes"]
        for i, rel in enumerate(relations):
            ok = migrated_store.causal_add(keys[i], keys[i + 1], rel)
            assert ok is True, f"Failed for relation: {rel}"

    def test_causal_add_chain(self, migrated_store):
        """Add A→B→C; chain from C returns ancestors."""
        for k in ["A", "B", "C"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("A", "B", "caused_by")
        migrated_store.causal_add("B", "C", "led_to")
        chain = migrated_store.causal_chain("C")
        chain_keys = [r["key"] for r in chain]
        assert "B" in chain_keys
        assert "A" in chain_keys

    def test_causal_get_direct(self, migrated_store):
        """causal_get returns all links involving a key."""
        for k in ["X", "Y", "Z"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("X", "Y", "caused_by")
        migrated_store.causal_add("Z", "Y", "resolved")
        links = migrated_store.causal_get("Y")
        assert len(links) == 2
        from_ids = {l["from_id"] for l in links}
        assert from_ids == {"X", "Z"}

    def test_causal_delete(self, migrated_store):
        """Delete a link and verify it's gone."""
        for k in ["D1", "D2"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("D1", "D2", "caused_by")
        links = migrated_store.causal_get("D1")
        link_id = links[0]["id"]
        ok = migrated_store.causal_delete(link_id)
        assert ok is True
        assert len(migrated_store.causal_get("D1")) == 0

    def test_closure_table_self_ref(self, migrated_store):
        """After causal_add, closure_table has self-references."""
        for k in ["S1", "S2"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("S1", "S2", "led_to")
        conn = migrated_store._get_conn()
        rows = conn.execute(
            "SELECT ancestor, descendant, depth FROM closure_table WHERE depth = 0 AND ancestor IN ('S1', 'S2')"
        ).fetchall()
        pairs = {(r[0], r[1]) for r in rows}
        assert ("S1", "S1") in pairs
        assert ("S2", "S2") in pairs

    def test_closure_table_transitive(self, migrated_store):
        """A→B, B→C creates transitive A→C (depth=2)."""
        for k in ["T1", "T2", "T3"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("T1", "T2", "caused_by")
        migrated_store.causal_add("T2", "T3", "led_to")
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT depth FROM closure_table WHERE ancestor = 'T1' AND descendant = 'T3'"
        ).fetchone()
        assert row is not None
        assert row[0] == 2

    def test_causal_chain_multi_hop(self, migrated_store):
        """A→B→C→D; chain from D returns all ancestors with correct depths."""
        for k in ["M1", "M2", "M3", "M4"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("M1", "M2", "caused_by")
        migrated_store.causal_add("M2", "M3", "led_to")
        migrated_store.causal_add("M3", "M4", "resolved")
        chain = migrated_store.causal_chain("M4")
        chain_keys = [r["key"] for r in chain]
        assert "M3" in chain_keys
        assert "M2" in chain_keys
        assert "M1" in chain_keys
        # Depths should be 1, 2, 3
        depth_map = {r["key"]: r["depth"] for r in chain}
        assert depth_map["M3"] == 1
        assert depth_map["M2"] == 2
        assert depth_map["M1"] == 3

    def test_invalid_relation(self, migrated_store):
        """causal_add with invalid relation returns False."""
        self._insert_key(migrated_store, "I1")
        self._insert_key(migrated_store, "I2")
        ok = migrated_store.causal_add("I1", "I2", "invalid_relation")
        assert ok is False

    def test_causal_chain_empty(self, migrated_store):
        """Chain for nonexistent key returns empty list."""
        chain = migrated_store.causal_chain("nonexistent_key_xyz")
        assert chain == []

    def test_causal_delete_rebuilds_closure(self, migrated_store):
        """After deleting middle link in A→B→C, transitive A→C should be gone."""
        for k in ["E1", "E2", "E3"]:
            self._insert_key(migrated_store, k, f"value_{k}")
        migrated_store.causal_add("E1", "E2", "caused_by")
        migrated_store.causal_add("E2", "E3", "led_to")
        # Verify transitive exists
        conn = migrated_store._get_conn()
        row = conn.execute(
            "SELECT depth FROM closure_table WHERE ancestor = 'E1' AND descendant = 'E3'"
        ).fetchone()
        assert row is not None

        # Delete E2→E3 link
        links = migrated_store.causal_get("E2")
        led_to_link = [l for l in links if l["relation"] == "led_to"][0]
        migrated_store.causal_delete(led_to_link["id"])

        # Transitive should be gone
        row = conn.execute(
            "SELECT depth FROM closure_table WHERE ancestor = 'E1' AND descendant = 'E3'"
        ).fetchone()
        assert row is None
