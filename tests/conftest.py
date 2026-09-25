import pytest

from agent_reach import main as M
from agent_reach.config import Settings
from agent_reach.pipeline import clusterer as C
from tests.fakes import FakeOllama, MockAsyncClient


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(C.SemanticClusterer, "_get_client", lambda self: fake)
    return fake


@pytest.fixture
def mock_http(monkeypatch):
    monkeypatch.setattr(M.httpx, "AsyncClient", MockAsyncClient)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        db_path=tmp_path / "test.db", http_backoff_base_s=0.05, reddit_request_spacing_s=0.1, tiktok_request_spacing_s=0.1,
        reddit_subreddits=["popular", "news"], max_items_for_llm=40, min_items_per_source_for_llm=0,
    )
