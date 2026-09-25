"""Offline fakes: mock HTTP for all sources + article pages, and a fake Ollama (chat + embed).

The fake embedding maps topic keywords to fixed dimensions so related stories are close and
unrelated ones are far apart - enough to exercise the real HDBSCAN code path deterministically.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import httpx

RealAsyncClient = httpx.AsyncClient  # captured before any test patches httpx.AsyncClient
NOW = int(time.time())


def article(title: str, para: str) -> str:
    return (
        f'<html><head><title>{title}</title><meta name="description" content="{para[:140]}"></head>'
        f"<body><nav>Subscribe to our newsletter</nav><article><h1>{title}</h1><p>{para}</p>"
        "<p>Analysts said the development drew wide attention across the industry and among consumers "
        "this week, with more reporting expected.</p></article><footer>All rights reserved</footer></body></html>"
    )


PAGES = {
    "samsung.example": article("Samsung fridge firmware update bricks units", "A faulty firmware update pushed by Samsung disabled thousands of Family Hub smart fridges, leaving owners with spoiled food."),
    "f-droid.org": article("F-Droid 2.0: a new chapter for Android freedom", "The F-Droid project released version 2.0 of its open-source Android app store with a redesigned client and faster sync."),
    "crypto.example": article("Two-tier encryption for messaging", "Researchers proposed a two-tier end-to-end encryption scheme that separates message keys from device keys."),
    "openai.example": article("OpenAI releases GPT-6", "OpenAI released GPT-6 on Thursday with native agent tooling, and developers began testing it immediately."),
    "nfl.example": article("Packers beat Falcons on Thursday Night Football", "Jordan Love threw three touchdowns as the Green Bay Packers beat the Atlanta Falcons 27-20 on Thursday Night Football."),
}


PAPERS_MD = """# AI Papers of the Week - 2026

## Top AI Papers of the Week (September 14 - September 20) - 2026
| **Paper**  | **Links** |
| ------------- | ------------- |
| 1) **Agentic Reasoning Scaling** - A study of how agentic reasoning scales with model size and tool access. <br>● Detail: bigger tool budgets help more than bigger models. | [Paper](https://academy.dair.ai/papers/scaling-agentic-reasoning-in-large-language-models-2609.00001), [Tweet](https://x.com/dair_ai/status/1) |
| 2) **GAUGE** - Amazon audits LLM user simulators against verifiable rewards across 25 agents. <br>● Satisfaction does not track success. | [Paper](https://example.org/gauge), [Tweet](https://x.com/dair_ai/status/2) |

---

## Top AI Papers of the Week (September 7 - September 13) - 2026
| **Paper**  | **Links** |
| ------------- | ------------- |
| 1) **Older Paper** - Last week's pick. | [Paper](https://arxiv.org/abs/2609.00002) |
"""


def _rss(items: list[str]) -> bytes:
    return ('<?xml version="1.0"?><rss xmlns:ht="https://trends.google.com/trending/rss"><channel>' + "".join(items) + "</channel></rss>").encode()


def handler(req: httpx.Request) -> httpx.Response:
    h, path = req.url.host, req.url.path
    if h in PAGES:
        return httpx.Response(200, text=PAGES[h], headers={"content-type": "text/html; charset=utf-8"})
    if "trends24" in h:
        names = ["Packers", "Falcons", "Jordan Love", "#fallvibes", "Cleveland", "#TSTLOASTheEncore", "Bijan"]
        lis = "".join(f'<li><a class="trend-link" href="#">{n}</a><span class="tweet-count" data-count="{(9 - i) * 1000}"></span></li>' for i, n in enumerate(names))
        return httpx.Response(200, text=f'<ol class="trend-card__list">{lis}</ol>')
    if "reddit" in h:
        return httpx.Response(403, text="Blocked") if path.endswith(".json") else httpx.Response(429)
    if "tiktok" in h:
        return httpx.Response(403)
    if "trends.google" in h:
        it = ('<item><title>packers</title><ht:approx_traffic>500K+</ht:approx_traffic><ht:news_item><ht:news_item_title>Packers vs. Falcons: Jordan Love leads win</ht:news_item_title>'
              '<ht:news_item_url>https://nfl.example/tnf</ht:news_item_url></ht:news_item></item>'
              '<item><title>gpt-6</title><ht:approx_traffic>100K+</ht:approx_traffic><ht:news_item><ht:news_item_title>OpenAI unveils GPT-6</ht:news_item_title>'
              '<ht:news_item_url>https://openai.example/gpt6</ht:news_item_url></ht:news_item></item>')
        return httpx.Response(200, content=_rss([it]))
    if "news.google" in h:
        its = "".join(f'<item><title>{t} - Pub</title><link>https://news.google.com/rss/articles/{i}</link><source url="x">Pub</source></item>'
                      for i, t in enumerate(["Senate passes stopgap bill to avert shutdown", "Lizzie Borden museum reopens"]))
        return httpx.Response(200, content=_rss([its]))
    if h == "wikimedia.org":
        return httpx.Response(200, json={"items": [{"articles": [{"article": a, "views": v} for a, v in [("Main_Page", 9e6), ("Jordan_Love", 3e5), ("Lizzie_Borden", 1e5)]]}]})
    if h == "en.wikipedia.org" and "/page/summary/" in path:
        ext = {"Jordan_Love": ("American football quarterback", "Jordan Love is an American football quarterback for the Green Bay Packers of the National Football League."),
               "Lizzie_Borden": ("American woman", "Lizzie Borden was an American woman tried and acquitted of the 1892 axe murders of her father and stepmother.")}
        d, e = ext[path.rsplit("/", 1)[-1]]
        return httpx.Response(200, json={"description": d, "extract": e})
    if h == "raw.githubusercontent.com" and "/AI-Papers-of-the-Week/" in path:
        return httpx.Response(200, text=PAPERS_MD)
    if "arxiv" in h:
        return httpx.Response(200, content=b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1</id><title>Scaling Agentic Reasoning in Large Language Models</title><summary>We study how agentic reasoning scales with model size and tool access across benchmarks.</summary><published>2026-09-24T00:00:00Z</published></entry></feed>')
    if "algolia" in h:
        rows = [("Owners mourn spoiled food after firmware update bricks Samsung fridges", "https://samsung.example/a", 900),
                ("F-Droid 2.0: A new chapter for Android freedom", "https://f-droid.org/2", 700),
                ("Two-tier encryption for end-to-end messaging", "https://crypto.example/e2e", 300),
                ("OpenAI releases GPT-6", "https://openai.example/gpt6", 1500),
                ("Cursed Fonts", "https://fonts.example/x", 400)]
        return httpx.Response(200, json={"hits": [{"objectID": str(i), "title": t, "url": u, "points": p, "num_comments": 50, "created_at_i": NOW} for i, (t, u, p) in enumerate(rows)]})
    if h == "github.com" and path == "/trending":
        return httpx.Response(200, text='<article class="Box-row"><h2><a href="/openai/gpt6-agents">x</a></h2><p>Agents SDK for GPT-6</p><span>2,104 stars today</span></article>')
    if "producthunt" in h:
        return httpx.Response(200, content=b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Agentify</title><link href="https://www.producthunt.com/p/1"/><content type="html">Build GPT-6 agents without code</content></entry></feed>')
    return httpx.Response(404)


class MockAsyncClient(RealAsyncClient):
    def __init__(self, *a, **k):
        k["transport"] = httpx.MockTransport(handler)
        super().__init__(*a, **k)


TOPICS = {0: ["packers", "falcons", "jordan love", "bijan", "thursday night", "quarterback", "nfl"], 1: ["gpt-6", "openai", "agentify", "agents sdk"],
          2: ["samsung", "fridge"], 3: ["f-droid", "android app store"], 4: ["encryption"], 5: ["senate", "shutdown"],
          6: ["lizzie borden", "axe murders"], 7: ["cleveland"], 8: ["tstloas"], 9: ["cursed fonts"], 10: ["agentic reasoning"]}


def fake_vec(text: str) -> list[float]:
    t = text.lower()
    v = [0.0] * 16
    for d, kws in TOPICS.items():
        v[d] += sum(3.0 for k in kws if k in t)
    for w in re.findall(r"[a-z]+", t):
        v[11 + int(hashlib.md5(w.encode()).hexdigest(), 16) % 5] += 0.15
    return v


class FakeOllama:
    """Chat labels groups like a sloppy 8B model (wrong category, URL in summary, a filler group,
    an [INSUFFICIENT_DATA] group); embed returns topic vectors."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = {"embed": 0, "chat": 0}

    async def list(self):
        class Mdl:
            model = "llama3.1:8b"

        class L:
            models = [Mdl()]

        return L()

    async def embed(self, model, input, keep_alive=None):
        self.calls["embed"] += 1
        return {"embeddings": [fake_vec(x) for x in input]}

    async def chat(self, model, messages, format, options, keep_alive):
        self.calls["chat"] += 1
        user = messages[1]["content"]
        self.prompts.append(user)
        out = []
        for gid, body in re.findall(r"^Group (\d+):\n((?:  - .*\n?)+)", user, re.M):
            b = body.lower()
            if "packers" in b or "jordan love" in b:
                out.append(dict(group_id=gid, headline="Packers beat Falcons on Thursday Night Football", category="Tech",
                                primary_entities=["Packers", "Falcons", "Jordan Love"],
                                summary="Jordan Love threw three touchdowns as the Packers beat the Falcons 27-20. The Thursday night game dominated search and social chatter.",
                                relevance_score=9))
            elif "gpt-6" in b:
                out.append(dict(group_id=gid, headline="OpenAI releases GPT-6 with native agents", category="AI", primary_entities=["OpenAI", "GPT-6"],
                                summary="OpenAI released GPT-6 with native agent tooling. Developers immediately began building on it https://openai.example.",
                                relevance_score=10))
            elif "lizzie" in b:
                out.append(dict(group_id=gid, headline="Lizzie Borden", category="Entertainment", primary_entities=["Lizzie Borden"],
                                summary="[INSUFFICIENT_DATA]", relevance_score=1))
            elif "cleveland" in b:
                out.append(dict(group_id=gid, headline="Cleveland", category="News", primary_entities=["Cleveland"],
                                summary="Cleveland is trending. However, no specific information is available.", relevance_score=5))
            else:
                ents = re.findall(r"\| ([A-Z][\w-]+)", body)[:2]
                out.append(dict(group_id=gid, headline=body.split("|")[2].strip()[:60], category="Tech", primary_entities=ents,
                                summary="A thing happened in tech. It matters for developers.", relevance_score=6))
        return {"message": {"content": json.dumps({"groups": out, "assignments": []})}, "eval_count": 300, "eval_duration": 30_000_000_000}
