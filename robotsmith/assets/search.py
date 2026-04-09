"""Tag-based asset search engine. MVP: exact tag matching with scoring."""

from __future__ import annotations

import re

from robotsmith.assets.schema import Asset

TAG_ALIASES: dict[str, list[str]] = {
    "杯": ["mug", "cup"],
    "水杯": ["mug", "cup"],
    "碗": ["bowl"],
    "盘": ["plate", "dish"],
    "盘子": ["plate", "dish"],
    "叉": ["fork"],
    "叉子": ["fork"],
    "勺": ["spoon"],
    "勺子": ["spoon"],
    "瓶": ["bottle"],
    "瓶子": ["bottle"],
    "积木": ["block", "cube"],
    "罐": ["can"],
    "罐子": ["can"],
    "抽屉": ["drawer"],
    "门把手": ["handle"],
    "桌": ["table"],
    "桌子": ["table"],
    "机器人": ["robot"],
    "地面": ["plane", "ground"],
}


def _normalize_query(query: str) -> set[str]:
    """Extract search tags from a query string (Chinese + English)."""
    tags: set[str] = set()
    q_lower = query.lower().strip()

    for zh, en_tags in TAG_ALIASES.items():
        if zh in q_lower:
            tags.update(en_tags)

    tokens = re.split(r"[\s,;\-_，；]+", q_lower)
    tags.update(t for t in tokens if t)

    return tags


def search_assets(
    query: str,
    assets: list[Asset],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[Asset, float]]:
    """Search assets by tag overlap. Returns (asset, score) sorted by score desc."""
    query_tags = _normalize_query(query)
    if not query_tags:
        return []

    scored: list[tuple[Asset, float]] = []
    for asset in assets:
        asset_tags = {t.lower() for t in asset.tags}
        overlap = query_tags & asset_tags
        if overlap:
            score = len(overlap) / max(len(query_tags), 1)
            if score > threshold:
                scored.append((asset, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
