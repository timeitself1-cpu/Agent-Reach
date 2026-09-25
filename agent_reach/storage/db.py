"""SQLite persistence: runs, raw items, macro clusters and entity snapshots (WAL mode).

All methods are synchronous and thread-safe (single connection guarded by a lock);
async callers wrap them with ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_reach.models import CleanedTrendItem, MacroCluster, PipelineReport, RawTrendItem

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    ingested      INTEGER DEFAULT 0,
    filtered      INTEGER DEFAULT 0,
    cluster_count INTEGER DEFAULT 0,
    llm_mode      TEXT,
    exec_seconds  REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS raw_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL,
    normalized_title TEXT,
    category_hint    TEXT,
    raw_score        REAL,
    comment_count    INTEGER,
    url              TEXT,
    item_ts          REAL,
    captured_at      REAL NOT NULL,
    content_hash     TEXT NOT NULL,
    kept             INTEGER NOT NULL DEFAULT 0,
    heuristic_score  REAL,
    duplicate_count  INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_raw_captured ON raw_items(captured_at);
CREATE INDEX IF NOT EXISTS idx_raw_run_kept ON raw_items(run_id, kept);
CREATE INDEX IF NOT EXISTS idx_raw_hash ON raw_items(content_hash);

CREATE TABLE IF NOT EXISTS clusters (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id      TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    headline        TEXT NOT NULL,
    category        TEXT NOT NULL,
    relevance_score INTEGER NOT NULL,
    velocity_score  REAL NOT NULL,
    combined_score  REAL NOT NULL,
    momentum        TEXT,
    velocity_basis  TEXT,
    summary         TEXT,
    entities        TEXT,
    source_urls     TEXT,
    sources         TEXT,
    raw_item_count  INTEGER NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clusters_created ON clusters(created_at);
CREATE INDEX IF NOT EXISTS idx_clusters_cid ON clusters(cluster_id, created_at);

CREATE TABLE IF NOT EXISTS entity_snapshots (
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    entity_key   TEXT NOT NULL,
    entity_label TEXT NOT NULL,
    frequency    INTEGER NOT NULL,
    rate         REAL NOT NULL,
    captured_at  REAL NOT NULL,
    PRIMARY KEY (run_id, entity_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_key_time ON entity_snapshots(entity_key, captured_at);
"""


class TrendDatabase:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), timeout=30, check_same_thread=False, isolation_level=None  # explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    # ------------------------------------------------------------ setup
    def _configure(self) -> None:
        with self._lock:
            mode = self._conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
            if str(mode).lower() != "wal":
                log.warning("SQLite did not enter WAL mode (got %s)", mode)
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")

    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)  # idempotent DDL (executescript manages its own transaction)
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                yield cur
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA optimize;")
            except sqlite3.Error:
                pass
            self._conn.close()

    @property
    def journal_mode(self) -> str:
        with self._lock:
            return str(self._conn.execute("PRAGMA journal_mode;").fetchone()[0])

    # ------------------------------------------------------------ writes
    def begin_run(self, run_id: str, started_at: float) -> None:
        with self._tx() as cur:
            cur.execute("INSERT INTO runs(run_id, started_at) VALUES(?, ?)", (run_id, started_at))

    def save_items(
        self,
        run_id: str,
        raw_items: Iterable[RawTrendItem],
        cleaned: Iterable[CleanedTrendItem],
        captured_at: float,
    ) -> int:
        """Persist every ingested item; surviving (de-duplicated) items are flagged kept=1."""
        kept_by_hash = {c.content_hash: c for c in cleaned}
        rows: list[tuple[Any, ...]] = []
        for it in raw_items:
            h = it.content_hash
            c = kept_by_hash.pop(h, None)
            rows.append(
                (
                    run_id,
                    it.source.value,
                    it.title,
                    c.normalized_title if c else None,
                    it.category_hint.value if it.category_hint else None,
                    it.raw_score,
                    it.comment_count,
                    it.url,
                    it.timestamp.timestamp(),
                    captured_at,
                    h,
                    1 if c else 0,
                    c.heuristic_score if c else None,
                    c.duplicate_count if c else 1,
                )
            )
        with self._tx() as cur:
            cur.executemany(
                """INSERT INTO raw_items(run_id, source, title, normalized_title, category_hint, raw_score,
                   comment_count, url, item_ts, captured_at, content_hash, kept, heuristic_score, duplicate_count)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def save_clusters(self, run_id: str, clusters: Iterable[MacroCluster]) -> None:
        rows = [
            (
                c.cluster_id,
                run_id,
                c.headline,
                c.category.value,
                c.relevance_score,
                c.velocity_score,
                c.combined_score,
                c.momentum,
                c.velocity_basis,
                c.summary,
                json.dumps(c.primary_entities),
                json.dumps(c.source_urls),
                json.dumps(c.sources),
                c.raw_item_count,
                c.created_at.timestamp(),
            )
            for c in clusters
        ]
        with self._tx() as cur:
            cur.executemany(
                """INSERT INTO clusters(cluster_id, run_id, headline, category, relevance_score, velocity_score,
                   combined_score, momentum, velocity_basis, summary, entities, source_urls, sources,
                   raw_item_count, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def save_entity_snapshots(
        self, run_id: str, rows: Iterable[tuple[str, str, int, float]], captured_at: float
    ) -> None:
        data = [(run_id, key, label, freq, rate, captured_at) for key, label, freq, rate in rows]
        with self._tx() as cur:
            cur.executemany(
                """INSERT INTO entity_snapshots(run_id, entity_key, entity_label, frequency, rate, captured_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(run_id, entity_key) DO UPDATE SET
                     frequency=MAX(frequency, excluded.frequency), rate=MAX(rate, excluded.rate)""",
                data,
            )

    def finish_run(self, report: PipelineReport, finished_at: float) -> None:
        with self._tx() as cur:
            cur.execute(
                """UPDATE runs SET finished_at=?, ingested=?, filtered=?, cluster_count=?, llm_mode=?, exec_seconds=?
                   WHERE run_id=?""",
                (
                    finished_at,
                    report.ingested_count,
                    report.filtered_count,
                    report.cluster_count,
                    report.llm_mode,
                    report.execution_time,
                    report.run_id,
                ),
            )

    def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self._tx() as cur:
            cur.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
            n = cur.rowcount
        if n:
            log.info("purged %d runs older than %d days", n, days)
        return n

    # ------------------------------------------------------------ reads
    def find_reference_run(
        self, target_ts: float, lo_ts: float, hi_ts: float, exclude_run_id: str
    ) -> sqlite3.Row | None:
        """Completed run whose start time is closest to ``target_ts`` within [lo_ts, hi_ts]."""
        with self._lock:
            return self._conn.execute(
                """SELECT r.run_id, r.started_at,
                          (SELECT COUNT(*) FROM raw_items i WHERE i.run_id = r.run_id AND i.kept = 1) AS kept_count
                   FROM runs r
                   WHERE r.finished_at IS NOT NULL AND r.run_id != ? AND r.started_at BETWEEN ? AND ?
                   ORDER BY ABS(r.started_at - ?) ASC LIMIT 1""",
                (exclude_run_id, lo_ts, hi_ts, target_ts),
            ).fetchone()

    def load_kept_titles(self, run_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT normalized_title, duplicate_count FROM raw_items WHERE run_id=? AND kept=1", (run_id,)
            ).fetchall()
        out: list[str] = []
        for r in rows:
            out.extend([r["normalized_title"] or ""] * max(1, int(r["duplicate_count"] or 1)))
        return out

    def count_runs(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL").fetchone()[0])

    def cluster_history(self, cluster_id: str, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM clusters WHERE cluster_id=? ORDER BY created_at DESC LIMIT ?", (cluster_id, limit)
            ).fetchall()
