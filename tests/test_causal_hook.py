import pytest
import sys
import os
import json
import importlib
import sqlite3

# Add hooks directory to path
hooks_dir = os.path.join(os.path.dirname(__file__), '..', '.claude', 'hooks')
sys.path.insert(0, hooks_dir)

# Force fresh import
for mod_name in list(sys.modules.keys()):
    if 'hermes' in mod_name:
        del sys.modules[mod_name]

_err_hook = importlib.import_module("hermes-error-hook")

match_pattern = _err_hook.match_pattern
_stable_hash = _err_hook._stable_hash


class TestErrorHookMatch:

    def test_match_command_not_found(self):
        pat = match_pattern("Bash", "command not found: xyz", "", "xyz")
        assert pat is not None
        assert pat["name"] == "command_not_found"

    def test_match_file_not_found(self):
        pat = match_pattern("Read", "No such file or directory", "/tmp/test.py", "")
        assert pat is not None
        assert pat["name"] == "file_not_found"

    def test_match_permission_denied(self):
        pat = match_pattern("Write", "Permission denied", "/etc/passwd", "")
        assert pat is not None
        assert pat["name"] == "permission_denied"

    def test_match_encoding_error(self):
        pat = match_pattern("Write", "UnicodeEncodeError: 'charmap' codec", "test.py", "")
        assert pat is not None
        assert pat["name"] == "encoding_error"

    def test_match_json_error(self):
        pat = match_pattern("Read", "JSONDecodeError: Expecting value", "config.json", "")
        assert pat is not None
        assert pat["name"] == "json_error"

    def test_no_match_random_error(self):
        pat = match_pattern("Bash", "something went wrong", "", "")
        assert pat is None

    def test_pattern_has_causal_fields(self):
        """All patterns have cause and resolution fields for causal chains."""
        for pat in _err_hook.PATTERNS:
            assert "cause" in pat, f"Pattern {pat['name']} missing 'cause'"
            assert "resolution" in pat, f"Pattern {pat['name']} missing 'resolution'"
            assert isinstance(pat["cause"], str) and len(pat["cause"]) > 0
            assert isinstance(pat["resolution"], str) and len(pat["resolution"]) > 0

    def test_stable_hash_consistency(self):
        """Same input always produces same hash."""
        h1 = _stable_hash("test error message")
        h2 = _stable_hash("test error message")
        assert h1 == h2
        assert len(h1) == 8

    def test_format_field_substitution(self):
        """Pattern lesson/cause templates are properly formatted."""
        pat = match_pattern("Bash", "command not found", "", "npm")
        assert pat is not None
        lesson = pat["lesson"].format(tool="Bash", file="(unknown)", cmd="npm")
        assert "npm" in lesson
        cause = pat["cause"].format(tool="Bash", file="(unknown)", cmd="npm")
        assert "npm" in cause
