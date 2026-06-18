import pytest
import time


class TestProfileStore:

    def test_ensure_default_profile(self, migrated_store):
        """Creates a root profile with parent_id IS NULL; calling twice returns same id."""
        pid1 = migrated_store._ensure_default_profile()
        pid2 = migrated_store._ensure_default_profile()
        assert pid1 == pid2
        assert isinstance(pid1, int)
        # Verify in DB
        row = migrated_store._get_conn().execute(
            "SELECT parent_id, name, level FROM user_profiles WHERE id = ?", (pid1,)
        ).fetchone()
        assert row is not None
        assert row[0] is None  # parent_id IS NULL
        assert row[1] == "default"
        assert row[2] == "root"

    def test_profile_set_and_get(self, migrated_store):
        """Set attr_key='lang' with category='identity'; get by attr_key returns correct dict."""
        ok = migrated_store.profile_set("lang", "python", "identity", 0.5, "auto-preference")
        assert ok is True
        results = migrated_store.profile_get(attr_key="lang")
        assert len(results) == 1
        r = results[0]
        assert r["attr_key"] == "lang"
        assert r["attr_value"] == "python"
        assert r["category"] == "identity"
        assert r["confidence"] == 0.5
        assert r["source"] == "auto-preference"
        assert "updated_at" in r

    def test_profile_set_upsert_merges_confidence(self, migrated_store):
        """Set same key twice with conf 0.3 then 0.5; result confidence = max(0.3, 0.5) = 0.5."""
        migrated_store.profile_set("lang", "python", "identity", 0.3, "test")
        migrated_store.profile_set("lang", "rust", "identity", 0.5, "test")
        results = migrated_store.profile_get(attr_key="lang")
        assert len(results) == 1
        assert results[0]["attr_value"] == "rust"  # updated value
        assert results[0]["confidence"] == 0.5  # max(0.3, 0.5)

    def test_profile_set_upsert_merges_confidence_reverse(self, migrated_store):
        """Set same key with conf 0.8 then 0.3; confidence stays 0.8."""
        migrated_store.profile_set("tool", "vim", "tools", 0.8, "test")
        migrated_store.profile_set("tool", "nvim", "tools", 0.3, "test")
        results = migrated_store.profile_get(attr_key="tool")
        assert results[0]["confidence"] == 0.8  # max(0.8, 0.3)

    def test_profile_get_by_category(self, migrated_store):
        """Set 3 attrs in different categories; get(category='tools') returns only tools sorted by confidence DESC."""
        migrated_store.profile_set("lang", "python", "identity", 0.6, "test")
        migrated_store.profile_set("editor", "vim", "tools", 0.7, "test")
        migrated_store.profile_set("shell", "fish", "tools", 0.4, "test")
        results = migrated_store.profile_get(category="tools")
        assert len(results) == 2
        assert results[0]["attr_key"] == "editor"  # higher confidence first
        assert results[1]["attr_key"] == "shell"

    def test_profile_get_all(self, migrated_store):
        """Set attrs in 3 categories; get() returns all, ordered by category then confidence DESC."""
        migrated_store.profile_set("lang", "python", "identity", 0.6, "test")
        migrated_store.profile_set("editor", "vim", "tools", 0.8, "test")
        migrated_store.profile_set("shell", "fish", "tools", 0.4, "test")
        migrated_store.profile_set("style", "minimal", "preferences", 0.5, "test")
        results = migrated_store.profile_get()
        assert len(results) == 4
        # First category alphabetically: identity
        assert results[0]["category"] == "identity"
        # Then preferences
        assert results[1]["category"] == "preferences"
        # Then tools (editor 0.8 before shell 0.4)
        assert results[2]["category"] == "tools"
        assert results[2]["confidence"] >= results[3]["confidence"]

    def test_profile_dimensions(self, migrated_store):
        """Set one attr per dimension (6 categories); get() returns 6 results."""
        categories = ["identity", "skills", "preferences", "tools", "efficiency", "projects"]
        for i, cat in enumerate(categories):
            migrated_store.profile_set(f"dim_{cat}", f"value_{i}", cat, 0.5, "test")
        results = migrated_store.profile_get()
        assert len(results) == 6
        result_cats = {r["category"] for r in results}
        assert result_cats == set(categories)

    def test_confidence_and_source(self, migrated_store):
        """Set with confidence=0.7, source='user-explicit'; get returns those values."""
        migrated_store.profile_set("reply_lang", "chinese", "identity", 0.7, "user-explicit")
        results = migrated_store.profile_get(attr_key="reply_lang")
        assert results[0]["confidence"] == 0.7
        assert results[0]["source"] == "user-explicit"

    def test_profile_summary(self, migrated_store):
        """Set 2 attrs in identity; summary contains '### identity' and formatted bullets."""
        migrated_store.profile_set("lang", "python", "identity", 0.5, "test")
        migrated_store.profile_set("os", "linux", "identity", 0.6, "test")
        summary = migrated_store.profile_summary()
        assert "### identity" in summary
        assert "- lang: python (50%)" in summary
        assert "- os: linux (60%)" in summary
        assert "(no profile data)" not in summary

    def test_profile_summary_empty(self, migrated_store):
        """Summary on empty store returns '(no profile data)'."""
        summary = migrated_store.profile_summary()
        assert summary == "(no profile data)"

    def test_profile_summary_high_confidence_no_annotation(self, migrated_store):
        """Set attr with confidence=0.9; summary omits '(90%)' annotation."""
        migrated_store.profile_set("lang", "python", "identity", 0.9, "test")
        summary = migrated_store.profile_summary()
        assert "- lang: python" in summary
        assert "(90%)" not in summary

    def test_profile_fts_sync(self, migrated_store):
        """Set attr; FTS5 search finds the newly set attribute."""
        migrated_store.profile_set("preferred_editor", "vscode", "tools", 0.6, "test")
        conn = migrated_store._get_conn()
        rows = conn.execute(
            "SELECT rowid FROM profile_attr_fts WHERE profile_attr_fts MATCH ?",
            ("vscode",),
        ).fetchall()
        assert len(rows) >= 1
        # Verify rowid matches profile_attributes id
        attr_row = conn.execute(
            "SELECT id FROM profile_attributes WHERE attr_key = 'preferred_editor'"
        ).fetchone()
        assert attr_row is not None
        assert rows[0][0] == attr_row[0]
