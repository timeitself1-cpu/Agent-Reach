"""Labelling prompt stays compact: central members only, short de-duplicated excerpts."""
import asyncio
import json

from agent_reach.config import Settings
from agent_reach.models import CleanedTrendItem, SourceName
from agent_reach.pipeline.clusterer import DraftCluster, SemanticClusterer, _excerpt
from agent_reach.pipeline.density import _cosine_gate


class _RecordingOllama:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def chat(self, model, messages, format, options, keep_alive):
        self.prompts.append(messages[1]["content"])
        return {"message": {"content": json.dumps({"groups": [], "assignments": []})}}


def _item(i, title, context=None):
    return CleanedTrendItem(item_id=i, title=title, normalized_title=title, source=SourceName.HACKERNEWS,
                            heuristic_score=0.7, context=context)


def test_cosine_gate_orders_members_most_central_first():
    vecs = {1: [0.6, 0.8], 2: [1.0, 0.0], 3: [0.8, 0.6]}
    kept, rejected = _cosine_gate([1, 2, 3], vecs, min_cos=0.0)
    assert rejected == []
    assert kept == [3, 1, 2]  # 3 sits nearest the centroid, 2 farthest


def test_excerpt_cuts_at_word_boundary():
    assert _excerpt("short", 60) == "short"
    out = _excerpt("Samsung pushed a faulty firmware update to smart fridges", 30)
    assert out == "Samsung pushed a faulty..." and len(out) <= 33


def test_prompt_shows_central_members_and_short_context():
    title = "Samsung fridge firmware update bricks units"
    long_ctx = "Owners reported spoiled food after the update disabled thousands of fridges. " * 5
    items = [_item(i, f"{title} report {i}", context=f"{title} report {i}. {long_ctx}") for i in range(1, 8)]
    by_id = {it.item_id: it for it in items}
    fake = _RecordingOllama()
    c = SemanticClusterer(Settings(llm_items_per_group=4, llm_context_chars=100))
    c._client = fake

    asyncio.run(c._relabel([DraftCluster([it.item_id for it in items], "", "", needs_label=True)], [], by_id))

    prompt = fake.prompts[0]
    member_lines = [ln for ln in prompt.splitlines() if ln.startswith("  - ")]
    assert len(member_lines) == 4
    assert all(f"report {i} |" in member_lines[i - 1] for i in range(1, 5))  # first four = most central
    assert "(+3 more similar signals)" in prompt
    for ln in member_lines:
        ctx = ln.rsplit(" | ", 1)[1]
        assert len(ctx) <= 103
        assert not ctx.lower().startswith("samsung fridge firmware")  # repeated page title stripped
