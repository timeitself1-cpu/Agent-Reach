"""Curated research ingester: DAIR.AI "AI Papers of the Week" (GitHub).

The repository keeps one Markdown file per year (``years/<year>.md``), newest week first.
Each week is a heading followed by a table whose rows read::

    | 1) **Paper Name** - What it does and why it matters. <br>● detail ... | [Paper](url), [Tweet](url) |

Only the newest week is ingested. Paper links usually end in an arXiv id, which becomes the
item URL (``https://arxiv.org/abs/<id>``) so the same paper from the arXiv feed shares a URL.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from agent_reach.ingestion.base import BaseIngester, IngestionError
from agent_reach.models import CategoryEnum, RawTrendItem, SourceName

WEEK_HEADING_RX = re.compile(r"^##\s+Top AI Papers of the Week\s*\(([^)]*)\)(?:\s*-\s*(\d{4}))?", re.MULTILINE)
SECTION_END_RX = re.compile(r"^(?:##\s|---\s*$)", re.MULTILINE)
ROW_RX = re.compile(r"^\|\s*(\d+)\)\s*\*\*(.+?)\*\*\s*[-:]?\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$", re.MULTILINE)
PAPER_LINK_RX = re.compile(r"\[Paper\]\((https?://[^)\s]+)\)", re.IGNORECASE)
ANY_LINK_RX = re.compile(r"\((https?://[^)\s]+)\)")
ARXIV_ID_RX = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$")
MD_MARKUP_RX = re.compile(r"\*\*|__|`|<[^>]+>")


def _week_end(label: str, year: int) -> datetime:
    """'September 14 - September 20' -> 2026-09-20 UTC; unparseable -> now."""
    end = label.split("-")[-1].strip()
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(f"{end} {year}", fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _paper_url(links: str) -> str | None:
    m = PAPER_LINK_RX.search(links) or ANY_LINK_RX.search(links)
    if not m:
        return None
    url = m.group(1)
    arxiv = ARXIV_ID_RX.search(url)
    return f"https://arxiv.org/abs/{arxiv.group(1)}" if arxiv else url


def parse_latest_week(markdown: str, year: int) -> tuple[str, datetime, list[dict]]:
    """Return (week label, week end, rows) for the first (newest) week in ``markdown``."""
    head = WEEK_HEADING_RX.search(markdown)
    if not head:
        return "", datetime.now(timezone.utc), []
    label = head.group(1).strip()
    week_year = int(head.group(2)) if head.group(2) else year
    body = markdown[head.end():]
    stop = SECTION_END_RX.search(body)
    body = body[: stop.start()] if stop else body
    rows = []
    for rank, name, desc, links in ROW_RX.findall(body):
        lead = MD_MARKUP_RX.sub(" ", desc.split("<br>")[0])
        rows.append({
            "rank": int(rank),
            "title": re.sub(r"\s+", " ", MD_MARKUP_RX.sub(" ", name)).strip(),
            "description": re.sub(r"\s+", " ", lead).strip(),
            "url": _paper_url(links),
        })
    return label, _week_end(label, week_year), rows


class AIPapersOfTheWeekIngester(BaseIngester):
    """Newest week of DAIR.AI's curated 'AI Papers of the Week' list."""

    source = SourceName.AI_PAPERS
    use_bot_user_agent = True

    def _year_url(self, year: int) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.settings.ai_papers_repo}/"
            f"{self.settings.ai_papers_branch}/years/{year}.md"
        )

    async def fetch(self) -> list[RawTrendItem]:
        this_year = datetime.now(timezone.utc).year
        label, week_end, rows = "", datetime.now(timezone.utc), []
        errors: list[str] = []
        # the new year's file may not exist (or be empty) during the first week of January
        for year in (this_year, this_year - 1):
            try:
                text = await self.get_text(self._year_url(year), retry_statuses={403})
            except IngestionError as exc:
                errors.append(str(exc)[:120])
                continue
            label, week_end, rows = parse_latest_week(text, year)
            if rows:
                break
        if not rows:
            raise IngestionError("no weekly paper table found" + (f" ({'; '.join(errors)})" if errors else ""))

        n = len(rows)
        out: list[RawTrendItem] = []
        for row in rows:
            item = self.make_item(
                title=row["title"],
                raw_score=float(n - row["rank"] + 1),
                url=row["url"],
                timestamp=week_end,
                category_hint=CategoryEnum.SCIENCE_AI,
                description=row["description"][:500] or None,
                metadata={"week": label, "rank": row["rank"]},
            )
            if item:
                out.append(item)
        return out
