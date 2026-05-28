"""Edge node with three cache-replacement variants.

The same EdgeNode object services every policy. Insertion is routed
through ``add_lru`` / ``add_lfu`` / ``add_ice_style`` so cache-management
differences between policies are isolated from routing differences.
``add_content`` is retained for the initial DCCC-hash seed population
that runs before any policy starts evaluating requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EdgeNode:
    node_id: int
    region: int
    cache_capacity_mb: int
    cache: Dict[int, int] = field(default_factory=dict)
    used_mb: int = 0
    load: float = 0.0
    frequency: Dict[int, int] = field(default_factory=dict)
    last_access: Dict[int, int] = field(default_factory=dict)
    reward_score: Dict[int, float] = field(default_factory=dict)

    def has_content(self, content_id: int) -> bool:
        return content_id in self.cache

    def remaining_capacity(self) -> int:
        return self.cache_capacity_mb - self.used_mb

    def _drop(self, victim: int) -> None:
        self.used_mb -= self.cache[victim]
        self.cache.pop(victim, None)
        self.frequency.pop(victim, None)
        self.last_access.pop(victim, None)
        self.reward_score.pop(victim, None)

    def add_content(self, content_id: int, size_mb: int) -> bool:
        """Seed insertion used only for the initial DCCC-hash distribution.

        Does not evict; returns False if there is no room.
        """
        if content_id in self.cache:
            return True
        if self.used_mb + size_mb <= self.cache_capacity_mb:
            self.cache[content_id] = size_mb
            self.used_mb += size_mb
            self.frequency.setdefault(content_id, 0)
            return True
        return False

    def add_lru(self, content, time_step: int) -> int:
        cid = content.content_id
        size = content.size_mb
        if cid in self.cache:
            self.last_access[cid] = time_step
            return 0
        evictions = 0
        while self.used_mb + size > self.cache_capacity_mb and self.cache:
            victim = min(self.cache.keys(), key=lambda x: self.last_access.get(x, 0))
            self._drop(victim)
            evictions += 1
        if size <= self.cache_capacity_mb:
            self.cache[cid] = size
            self.used_mb += size
            self.last_access[cid] = time_step
        return evictions

    def add_lfu(self, content, time_step: int) -> int:
        cid = content.content_id
        size = content.size_mb
        if cid in self.cache:
            self.frequency[cid] = self.frequency.get(cid, 0) + 1
            return 0
        evictions = 0
        while self.used_mb + size > self.cache_capacity_mb and self.cache:
            victim = min(self.cache.keys(), key=lambda x: self.frequency.get(x, 0))
            self._drop(victim)
            evictions += 1
        if size <= self.cache_capacity_mb:
            self.cache[cid] = size
            self.used_mb += size
            self.frequency[cid] = 1
        return evictions

    def add_ice_style(self, content, time_step: int, popularity: float) -> int:
        """ICE-style insertion: reward = popularity * log1p(size_mb).

        Evicts the lowest-reward victim first. Refreshes reward on
        re-access so cold-but-large items eventually drop out.
        """
        cid = content.content_id
        size = content.size_mb
        if cid in self.cache:
            self.last_access[cid] = time_step
            self.frequency[cid] = self.frequency.get(cid, 0) + 1
            self.reward_score[cid] = popularity * math.log1p(size)
            return 0
        evictions = 0
        while self.used_mb + size > self.cache_capacity_mb and self.cache:
            victim = min(self.cache.keys(), key=lambda x: self.reward_score.get(x, 0.0))
            self._drop(victim)
            evictions += 1
        if size <= self.cache_capacity_mb:
            self.cache[cid] = size
            self.used_mb += size
            self.frequency[cid] = self.frequency.get(cid, 0) + 1
            self.last_access[cid] = time_step
            self.reward_score[cid] = popularity * math.log1p(size)
        return evictions

    def __str__(self) -> str:
        return (
            f"EdgeNode(id={self.node_id}, region={self.region}, "
            f"used={self.used_mb}/{self.cache_capacity_mb}MB, "
            f"items={len(self.cache)}, load={self.load:.2f})"
        )
