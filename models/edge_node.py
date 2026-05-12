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

    def has_content(self, content_id: int) -> bool:
        return content_id in self.cache

    def remaining_capacity(self) -> int:
        return self.cache_capacity_mb - self.used_mb

    def add_content(self, content_id: int, size_mb: int) -> bool:
        if content_id in self.cache:
            return True

        if self.used_mb + size_mb <= self.cache_capacity_mb:
            self.cache[content_id] = size_mb
            self.used_mb += size_mb
            return True

        return False

    def __str__(self):
        return (
            f"EdgeNode("
            f"id={self.node_id}, "
            f"region={self.region}, "
            f"used={self.used_mb}/{self.cache_capacity_mb}MB, "
            f"items={len(self.cache)}, "
            f"load={self.load:.2f})"
        )