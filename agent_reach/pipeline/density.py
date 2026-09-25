"""Stage 3a: density-based grouping (embeddings + HDBSCAN).

Items are embedded with a local Ollama embedding model (``nomic-embed-text`` by default) from
their title plus enriched context, L2-normalised (so Euclidean distance is monotonic in cosine
distance), and grouped with HDBSCAN. HDBSCAN never forces a point into a cluster: anything in
a low-density region is labelled noise (-1) and is DROPPED, which is what keeps unrelated
singletons (a firmware story, an app-store release, a crypto paper) from being glued together.

After HDBSCAN every cluster must also pass a cosine gate: each member's similarity to the
cluster centroid must be >= ``density_member_min_cosine``; failing members become noise.

When scikit-learn is unavailable the module falls back to strict cosine-threshold grouping
(complete-linkage style: a point joins a group only if it is similar to EVERY member).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from agent_reach.config import Settings
from agent_reach.models import CleanedTrendItem

log = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

try:
    from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3
except Exception:  # noqa: BLE001
    HDBSCAN = None  # type: ignore[assignment,misc]


class EmbeddingUnavailable(Exception):
    pass


@dataclass
class DensityResult:
    groups: list[list[int]] = field(default_factory=list)  # item_ids per cluster
    noise: list[int] = field(default_factory=list)  # item_ids labelled outlier / failed the gate
    method: str = ""


def embedding_text(it: CleanedTrendItem, prefix: str = "") -> str:
    ctx = (it.context or "")[:300]
    return f"{prefix}{it.normalized_title}" + (f". {ctx}" if ctx else "")


def _normalise(vectors: list[list[float]]) -> list[list[float]]:
    out = []
    for v in vectors:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def embed_items(client, settings: Settings, items: list[CleanedTrendItem]) -> list[list[float]]:
    """Embed items through Ollama's /api/embed. Raises EmbeddingUnavailable on any failure."""
    texts = [embedding_text(it, settings.embed_prefix) for it in items]
    vectors: list[list[float]] = []
    try:
        for k in range(0, len(texts), settings.embed_batch_size):
            chunk = texts[k : k + settings.embed_batch_size]
            resp = await client.embed(model=settings.embed_model, input=chunk, keep_alive=settings.ollama_keep_alive)
            got = getattr(resp, "embeddings", None)
            if got is None and isinstance(resp, dict):
                got = resp.get("embeddings")
            if not got or len(got) != len(chunk):
                raise EmbeddingUnavailable(f"embed returned {0 if not got else len(got)} vectors for {len(chunk)} inputs")
            vectors.extend([list(map(float, v)) for v in got])
    except EmbeddingUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - model missing, server down, timeout...
        raise EmbeddingUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return _normalise(vectors)


def _cosine_gate(ids: list[int], vecs: dict[int, list[float]], min_cos: float) -> tuple[list[int], list[int]]:
    """Iteratively drop the member least similar to the centroid until all pass the gate.

    Kept members are returned most central first, so the labeller can show only the top few.
    """
    members = list(ids)
    rejected: list[int] = []
    sims: dict[int, float] = {}
    while len(members) >= 2:
        dim = len(vecs[members[0]])
        centroid = [sum(vecs[m][d] for m in members) / len(members) for d in range(dim)]
        norm = math.sqrt(sum(x * x for x in centroid)) or 1.0
        centroid = [x / norm for x in centroid]
        sims = {m: _dot(vecs[m], centroid) for m in members}
        worst = min(members, key=lambda m: sims[m])
        if sims[worst] >= min_cos:
            break
        members.remove(worst)
        rejected.append(worst)
    if len(members) < 2:
        rejected.extend(members)
        members = []
    return sorted(members, key=lambda m: -sims[m]), rejected


def _threshold_groups(ids: list[int], vecs: dict[int, list[float]], threshold: float) -> tuple[list[list[int]], list[int]]:
    """Complete-linkage threshold grouping (fallback when sklearn is missing)."""
    order = list(ids)
    groups: list[list[int]] = []
    for i in order:
        best, best_min = None, -1.0
        for g in groups:
            m = min(_dot(vecs[i], vecs[j]) for j in g)
            if m >= threshold and m > best_min:
                best, best_min = g, m
        if best is None:
            groups.append([i])
        else:
            best.append(i)
    clusters = [g for g in groups if len(g) >= 2]
    noise = [g[0] for g in groups if len(g) == 1]
    return clusters, noise


def density_cluster(items: list[CleanedTrendItem], vectors: list[list[float]], settings: Settings) -> DensityResult:
    ids = [it.item_id for it in items]
    vecs = dict(zip(ids, vectors))
    result = DensityResult()
    if len(ids) < 2:
        result.noise = ids
        result.method = "none (too few items)"
        return result

    raw_groups: list[list[int]] = []
    noise: list[int] = []
    if HDBSCAN is not None and np is not None and len(ids) >= settings.hdbscan_min_cluster_size:
        X = np.asarray(vectors, dtype=float)
        model = HDBSCAN(
            min_cluster_size=settings.hdbscan_min_cluster_size,
            min_samples=settings.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=settings.hdbscan_selection,
            allow_single_cluster=False,
            copy=True,
        )
        labels = model.fit_predict(X)
        by_label: dict[int, list[int]] = {}
        for iid, lab in zip(ids, labels):
            if int(lab) < 0:
                noise.append(iid)
            else:
                by_label.setdefault(int(lab), []).append(iid)
        raw_groups = list(by_label.values())
        result.method = f"hdbscan(min_cluster_size={settings.hdbscan_min_cluster_size}, selection={settings.hdbscan_selection})"
    else:
        raw_groups, noise = _threshold_groups(ids, vecs, settings.density_fallback_cosine)
        result.method = f"cosine-threshold({settings.density_fallback_cosine})"

    gate_rejects = 0
    for g in raw_groups:
        kept, rejected = _cosine_gate(g, vecs, settings.density_member_min_cosine)
        noise.extend(rejected)
        gate_rejects += len(rejected)
        if kept:
            result.groups.append(kept)
    result.noise = noise
    log.info(
        "stage 3a density: %s -> %d groups covering %d items, %d noise (%d removed by cosine gate >= %.2f)",
        result.method, len(result.groups), sum(len(g) for g in result.groups), len(noise), gate_rejects,
        settings.density_member_min_cosine,
    )
    return result
