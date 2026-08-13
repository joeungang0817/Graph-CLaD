"""Deterministic sample capping for controlled Phase 3 experiments.

The Colab ``balanced-v3`` run used :func:`category_aware_cap_v1`.  That
function is preserved exactly for result provenance, including its greedy
category-quota traversal.  New experiments should use
:func:`category_aware_episode_round_robin_cap`, which advances episodes fairly
while satisfying the same category quotas.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


DEFAULT_TARGET_CATEGORIES = (
    "holding_changed",
    "future_holding_positive",
    "hard_negative",
    "background",
)


def category_counts(samples: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count multi-label sampling categories across selected samples."""

    return Counter(
        str(category)
        for sample in samples
        for category in sample.get("target_categories", [])
    )


def episode_round_robin_cap(
    samples: Sequence[dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    """Select up to ``cap`` rows while cycling through episodes evenly."""

    if cap < 0:
        raise ValueError("cap must be non-negative")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_episode[str(sample["episode_id"])].append(sample)
    episodes = sorted(by_episode)
    cursors = {episode: 0 for episode in episodes}
    selected: list[dict[str, Any]] = []
    while len(selected) < cap:
        progressed = False
        for episode in episodes:
            rows = by_episode[episode]
            cursor = cursors[episode]
            if cursor < len(rows):
                selected.append(rows[cursor])
                cursors[episode] += 1
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
    return selected


def category_aware_cap_v1(
    samples: Sequence[dict[str, Any]],
    cap: int,
    categories: Sequence[str] = DEFAULT_TARGET_CATEGORIES,
    category_quota: int = 120,
) -> list[dict[str, Any]]:
    """Reproduce the category-aware sampler used by the Colab balanced-v3 run.

    The quota stage restarts from the first episode after every accepted row.
    It therefore guarantees category support but is not a true episode
    round-robin.  Keep this function for reproducing the existing artifact;
    use ``category_aware_episode_round_robin_cap`` for new experiments.
    """

    if cap < 0 or category_quota < 0:
        raise ValueError("cap and category_quota must be non-negative")
    indexed = list(enumerate(samples))
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, sample in indexed:
        by_episode[str(sample["episode_id"])].append(index)
    episodes = sorted(by_episode)
    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    quota = min(category_quota, cap // max(len(categories), 1))
    category_cursors = {
        category: {episode: 0 for episode in episodes}
        for category in categories
    }

    def add_index(index: int) -> None:
        selected_indices.add(index)
        row = samples[index]
        selected.append(row)
        counts.update(str(value) for value in row.get("target_categories", []))

    for category in categories:
        while len(selected) < cap and counts[category] < quota:
            progressed = False
            for episode in episodes:
                positions = by_episode[episode]
                cursor = category_cursors[category][episode]
                while cursor < len(positions):
                    index = positions[cursor]
                    cursor += 1
                    if index in selected_indices:
                        continue
                    row_categories = {
                        str(value)
                        for value in samples[index].get("target_categories", [])
                    }
                    if category not in row_categories:
                        continue
                    add_index(index)
                    progressed = True
                    break
                category_cursors[category][episode] = cursor
                if progressed or counts[category] >= quota or len(selected) >= cap:
                    break
            if not progressed:
                break

    return _round_robin_fill(samples, by_episode, selected, selected_indices, cap)


def category_aware_episode_round_robin_cap(
    samples: Sequence[dict[str, Any]],
    cap: int,
    categories: Sequence[str] = DEFAULT_TARGET_CATEGORIES,
    category_quota: int = 120,
) -> list[dict[str, Any]]:
    """Select unique rows with category quotas and fair episode traversal."""

    if cap < 0 or category_quota < 0:
        raise ValueError("cap and category_quota must be non-negative")
    by_episode: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_episode[str(sample["episode_id"])].append(index)
    episodes = sorted(by_episode)
    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    quota = min(category_quota, cap // max(len(categories), 1))

    def add_index(index: int) -> None:
        selected_indices.add(index)
        row = samples[index]
        selected.append(row)
        counts.update(str(value) for value in row.get("target_categories", []))

    for category in categories:
        cursors = {episode: 0 for episode in episodes}
        while len(selected) < cap and counts[category] < quota:
            progressed = False
            for episode in episodes:
                positions = by_episode[episode]
                cursor = cursors[episode]
                while cursor < len(positions):
                    index = positions[cursor]
                    cursor += 1
                    if index in selected_indices:
                        continue
                    row_categories = {
                        str(value)
                        for value in samples[index].get("target_categories", [])
                    }
                    if category in row_categories:
                        add_index(index)
                        progressed = True
                        break
                cursors[episode] = cursor
                if counts[category] >= quota or len(selected) >= cap:
                    break
            if not progressed:
                break

    return _round_robin_fill(samples, by_episode, selected, selected_indices, cap)


def _round_robin_fill(
    samples: Sequence[dict[str, Any]],
    by_episode: dict[str, list[int]],
    selected: list[dict[str, Any]],
    selected_indices: set[int],
    cap: int,
) -> list[dict[str, Any]]:
    episodes = sorted(by_episode)
    cursors = {episode: 0 for episode in episodes}
    while len(selected) < cap:
        progressed = False
        for episode in episodes:
            positions = by_episode[episode]
            cursor = cursors[episode]
            while cursor < len(positions) and positions[cursor] in selected_indices:
                cursor += 1
            if cursor < len(positions):
                index = positions[cursor]
                selected_indices.add(index)
                selected.append(samples[index])
                cursor += 1
                progressed = True
            cursors[episode] = cursor
            if len(selected) >= cap:
                break
        if not progressed:
            break
    return selected


def select_with_config(
    samples: Sequence[dict[str, Any]],
    cap: int,
    sampling: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Dispatch a sampler from the experiment configuration."""

    sampling = dict(sampling or {})
    method = str(sampling.get("method", "episode_round_robin"))
    categories = tuple(sampling.get("categories", DEFAULT_TARGET_CATEGORIES))
    quota = int(sampling.get("category_quota", 120))
    if method == "episode_round_robin":
        return episode_round_robin_cap(samples, cap)
    if method == "category_aware_v1":
        return category_aware_cap_v1(samples, cap, categories, quota)
    if method == "category_aware_episode_round_robin_v2":
        return category_aware_episode_round_robin_cap(samples, cap, categories, quota)
    raise ValueError(f"unknown sampling method: {method}")
