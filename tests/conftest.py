import pytest
import shutil
import tempfile
import os
import sys

# Add vmem to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.claude', 'mcp-servers'))

@pytest.fixture
def tmp_vmem_db(tmp_path):
    """Copy real vmem.db to temp directory for safe migration testing."""
    real_db = r"C:\Users\35955\.claude\memory\vmem.db"
    tmp_db = str(tmp_path / "vmem.db")
    shutil.copy2(real_db, tmp_db)
    return tmp_db

@pytest.fixture
def migrated_store(tmp_vmem_db):
    """MemoryVectorStore pointing at temp DB (migrations run on init)."""
    from vmem.store import MemoryVectorStore
    store = MemoryVectorStore(db_path=tmp_vmem_db)
    yield store
    store.close()
