from dataclasses import dataclass
from typing import List


@dataclass
class Content:
    content_id: int
    size_mb: int
    base_popularity: float
    region_bias: List[float]
    content_type: str

    def __str__(self):
        return (
            f"Content("
            f"id={self.content_id}, "
            f"size={self.size_mb}MB, "
            f"popularity={self.base_popularity:.3f}, "
            f"type={self.content_type})"
        )