import pytest
import sys
import os
import json
import importlib

# Add hooks directory to path
hooks_dir = os.path.join(os.path.dirname(__file__), '..', '.claude', 'hooks')
sys.path.insert(0, hooks_dir)

# Force fresh import
for mod_name in list(sys.modules.keys()):
    if 'hermes' in mod_name or 'hermes-' in mod_name:
        del sys.modules[mod_name]
# Ensure hooks dir is at front of path
if hooks_dir not in sys.path:
    sys.path.insert(0, hooks_dir)

_pref_hook = importlib.import_module("hermes-preference-hook")
_session_hook = importlib.import_module("hermes-profile-session-hook")

extract_preferences = _pref_hook.extract_preferences
_CATEGORY_MAP = _pref_hook._CATEGORY_MAP
_CATEGORY_CONFIDENCE = _pref_hook._CATEGORY_CONFIDENCE
_build_session_output = _session_hook._build_session_output


class TestExtractPreferences:

    def test_extract_preferences_explicit(self):
        results = list(extract_preferences("我喜欢用python"))
        assert len(results) >= 1
        assert results[0][0] == "explicit"

    def test_extract_preferences_avoid(self):
        results = list(extract_preferences("不喜欢这个工具"))
        assert len(results) >= 1
        assert results[0][0] == "avoid"

    def test_extract_preferences_tool(self):
        # Tool pattern: "用X代替Y" — tool is checked before explicit
        results = list(extract_preferences("用python代替bash"))
        assert len(results) >= 1
        assert results[0][0] == "tool"

    def test_extract_preferences_language(self):
        results = list(extract_preferences("用中文回复"))
        assert len(results) >= 1
        assert results[0][0] == "language"

    def test_extract_preferences_efficiency(self):
        results = list(extract_preferences("直接做"))
        assert len(results) >= 1
        assert results[0][0] == "efficiency"

    def test_extract_preferences_no_match(self):
        results = list(extract_preferences("今天天气怎么样"))
        assert len(results) == 0

    def test_extract_preferences_long_input_skipped(self):
        results = list(extract_preferences("a" * 2001))
        assert len(results) == 0

    def test_category_map_coverage(self):
        """All 6 _CATEGORY_MAP keys have corresponding _CATEGORY_CONFIDENCE entries
        and map to valid profile dimensions."""
        valid_dims = {"identity", "skills", "preferences", "tools", "efficiency", "projects"}
        for key in _CATEGORY_MAP:
            assert key in _CATEGORY_CONFIDENCE, f"Missing confidence for tag: {key}"
            cat, attr_key = _CATEGORY_MAP[key]
            assert cat in valid_dims, f"Invalid category '{cat}' for tag '{key}'"
            assert isinstance(attr_key, str) and len(attr_key) > 0


class TestSessionHook:

    def test_session_hook_output_format(self, tmp_vmem_db):
        """Session hook output has correct structure."""
        output = _build_session_output(tmp_vmem_db)
        assert "hookSpecificOutput" in output
        ho = output["hookSpecificOutput"]
        assert ho["hookEventName"] == "SessionStart"
        assert isinstance(ho["additionalContext"], str)
        assert ho["additionalContext"].startswith("## User Profile")

    def test_session_hook_output_with_profile_data(self, tmp_vmem_db):
        """Session hook output includes profile data when present."""
        from vmem.store import MemoryVectorStore
        store = MemoryVectorStore(db_path=tmp_vmem_db)
        store.profile_set("lang", "python", "identity", 0.6, "test")
        store.close()

        output = _build_session_output(tmp_vmem_db)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert "python" in ctx
        assert "### identity" in ctx

    def test_session_hook_output_empty_profile(self, tmp_vmem_db):
        """Session hook output shows '(no profile data)' on empty store."""
        output = _build_session_output(tmp_vmem_db)
        ctx = output["hookSpecificOutput"]["additionalContext"]
        assert "(no profile data)" in ctx
