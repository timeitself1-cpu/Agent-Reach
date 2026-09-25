"""Entity isolation: fragments link only through co-occurrence evidence found in the run."""
import asyncio
import json

from agent_reach.config import Settings
from agent_reach.models import CleanedTrendItem, SourceName
from agent_reach.pipeline.clusterer import DraftCluster, LinkIndex, SemanticClusterer

_n = [0]


def _it(title, src):
    _n[0] += 1
    return CleanedTrendItem(item_id=_n[0], title=title, normalized_title=title, source=SourceName(src), heuristic_score=0.7)


def _fixture():
    P, F, JL = _it("Packers", "x_trends24"), _it("Falcons", "x_trends24"), _it("Jordan Love", "x_trends24")
    SA = _it("Owners mourn spoiled food after firmware update bricks Samsung fridges", "hackernews")
    FD = _it("F-Droid 2.0: A new chapter for Android freedom", "hackernews")
    NEWS = _it("Packers vs. Falcons: Jordan Love outduels Penix", "google_news")
    return P, F, JL, SA, FD, NEWS


def test_cooccurrence_links_fragments_and_splits_unrelated():
    P, F, JL, SA, FD, NEWS = _fixture()
    batch = [P, F, JL, SA, FD]
    c = SemanticClusterer(Settings())
    drafts = [DraftCluster([P.item_id, F.item_id, JL.item_id], "NFL", "Sports", ["Packers", "Falcons", "Jordan Love"]),
              DraftCluster([SA.item_id, FD.item_id], "Samsung", "Tech", ["Samsung", "F-Droid 2.0"])]
    out, orphans = c._enforce_coherence(drafts, LinkIndex(batch, corpus=batch + [NEWS]))
    assert [sorted(d.item_ids) for d in out] == [sorted([P.item_id, F.item_id, JL.item_id])]
    assert sorted(orphans) == sorted([SA.item_id, FD.item_id])


def test_no_evidence_no_group():
    P, F, JL, *_ = _fixture()
    c = SemanticClusterer(Settings())
    out, orphans = c._enforce_coherence([DraftCluster([P.item_id, F.item_id, JL.item_id], "NFL", "Sports", ["Packers", "Falcons", "Jordan Love"])],
                                        LinkIndex([P, F, JL]))
    assert out == [] and len(orphans) == 3


class _AssigningOllama:
    """Labels group 1 as the Samsung story and assigns BOTH unassigned signals to it."""

    async def chat(self, model, messages, format, options, keep_alive):
        group = dict(group_id=1, headline="Samsung Fridge Firmware Update Bricks Units", category="Tech",
                     primary_entities=["Samsung", "F-Droid"],  # 'F-Droid' is the LLM's word only
                     summary="A Samsung firmware update bricked smart fridges. Owners reported spoiled food.", relevance_score=6)
        out = {"groups": [group], "assignments": [{"item_id": 1, "group_id": 1}, {"item_id": 2, "group_id": 1}]}
        return {"message": {"content": json.dumps(out)}}


def test_llm_assignment_needs_evidence():
    SA = _it("Owners mourn spoiled food after firmware update bricks Samsung fridges", "hackernews")
    SA2 = _it("Samsung smart fridges bricked by firmware update", "google_news")
    FD = _it("F-Droid 2.0: A new chapter for Android freedom", "hackernews")
    FIX = _it("Samsung promises firmware fix for bricked fridges", "google_news")
    by_id = {it.item_id: it for it in (SA, SA2, FD, FIX)}
    c = SemanticClusterer(Settings())
    c._client = _AssigningOllama()
    draft = DraftCluster([SA.item_id, SA2.item_id], "", "", needs_label=True)

    drafts, orphans = asyncio.run(c._relabel([draft], [FD.item_id, FIX.item_id], by_id, LinkIndex(list(by_id.values()))))

    assert sorted(drafts[0].item_ids) == sorted([SA.item_id, SA2.item_id, FIX.item_id])  # shares distinctive tokens
    assert orphans == [FD.item_id]  # LLM said so, nothing in the run supports it


def test_insufficient_detection():
    assert SemanticClusterer.is_insufficient("[INSUFFICIENT_DATA]")
    assert SemanticClusterer.is_insufficient("X is trending. However, no specific information is available.")
    assert not SemanticClusterer.is_insufficient("The Packers beat the Falcons 27-20. Jordan Love threw three touchdowns.")
