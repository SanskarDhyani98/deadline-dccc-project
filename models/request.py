from dataclasses import dataclass


@dataclass
class Request:
    request_id: int
    user_region: int
    content_id: int
    deadline_ms: int

    def __str__(self):
        return (
            f"Request("
            f"id={self.request_id}, "
            f"region={self.user_region}, "
            f"content={self.content_id}, "
            f"deadline={self.deadline_ms}ms)"
        )